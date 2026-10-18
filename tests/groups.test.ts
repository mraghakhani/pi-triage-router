/** Groups feature tests: expansion (refs/wildcard/exclusion), validation
 * problems, the active-group filter with its fallback chain, override/sticky
 * interaction, and the dashboard line. All hermetic: no network, no Laya. */
import assert from "node:assert/strict";
import test from "node:test";
import * as router from "../triage-router.ts";
import { expandGroupRefs, filterByComposites } from "../triage-router-lib/triage-router-groups.ts";
import { formatDashboard, groupLabel } from "../triage-router-lib/triage-router-dashboard.ts";
import {
	ALL_KEYS,
	makeCtx,
	registryFor,
	routeRequest,
	testConfig,
	userMsg,
	mockClassifier,
} from "./fixtures.ts";

const CATALOG_KEYS = [
	"litellm/deepseek/deepseek-v4-flash",
	"litellm/z-ai/glm-5.3-flash",
	"litellm/z-ai/glm-5.3",
	"litellm/deepseek/deepseek-v4-pro",
	"litellm/qwen/qwen3.8-max",
	"litellm/xiaomi/mimo-v2.5",
];

// ---------------------------------------------------------------------------
// expandGroupRefs: pure ref algebra
// ---------------------------------------------------------------------------

test("expandGroupRefs: explicit refs resolve in catalog order and dedupe", () => {
	const problems: string[] = [];
	const out = expandGroupRefs(
		["litellm/z-ai/glm-5.3", "litellm/z-ai/glm-5.3", "litellm/qwen/qwen3.8-max"],
		{},
		CATALOG_KEYS,
		problems,
	);
	assert.deepEqual(out, ["litellm/z-ai/glm-5.3", "litellm/qwen/qwen3.8-max"]);
	assert.equal(problems.length, 0);
});

test('expandGroupRefs: "*" is every catalog model, exclusions apply last', () => {
	const problems: string[] = [];
	const out = expandGroupRefs(["*", "-litellm/qwen/qwen3.8-max"], {}, CATALOG_KEYS, problems);
	assert.equal(out.length, CATALOG_KEYS.length - 1);
	assert.ok(!out.includes("litellm/qwen/qwen3.8-max"));
	assert.equal(problems.length, 0);
});

test('expandGroupRefs: exclusion of a never-included model is a no-op', () => {
	const problems: string[] = [];
	const out = expandGroupRefs(["litellm/z-ai/glm-5.3", "-litellm/qwen/qwen3.8-max"], {}, CATALOG_KEYS, problems);
	assert.deepEqual(out, ["litellm/z-ai/glm-5.3"]);
	assert.equal(problems.length, 0, "excluding an absent model is not a problem");
});

test('expandGroupRefs: "@group" includes another group\'s members', () => {
	const groups: Record<string, string[]> = {
		cheap: ["litellm/deepseek/deepseek-v4-flash"],
		composed: ["@cheap", "litellm/z-ai/glm-5.3"],
	};
	const problems: string[] = [];
	const out = expandGroupRefs(groups.composed, groups, CATALOG_KEYS, problems);
	assert.deepEqual(out, ["litellm/deepseek/deepseek-v4-flash", "litellm/z-ai/glm-5.3"]);
	assert.equal(problems.length, 0);
});

test("expandGroupRefs: unknown model and unknown group are problems", () => {
	const problems: string[] = [];
	const out = expandGroupRefs(
		["litellm/nope/nothing", "@ghost", "litellm/z-ai/glm-5.3"],
		{},
		CATALOG_KEYS,
		problems,
	);
	assert.deepEqual(out, ["litellm/z-ai/glm-5.3"], "valid refs still resolve");
	assert.ok(problems.some((p) => p.includes("unknown model")), "unknown model problem");
	assert.ok(problems.some((p) => p.includes("unknown group")), "unknown group problem");
});

test("expandGroupRefs: inherited-prototype names are unknown groups, not crashes", () => {
	// "constructor"/"toString" resolve to Object.prototype members (functions,
	// truthy) under a naive truthiness check; functions are not iterable, so
	// expansion would crash. Array.isArray makes them plain unknown groups.
	for (const name of ["constructor", "toString", "hasOwnProperty", "valueOf"]) {
		const problems: string[] = [];
		const out = expandGroupRefs([`@${name}`], {}, CATALOG_KEYS, problems);
		assert.deepEqual(out, []);
		assert.equal(problems.length, 1, `"@${name}" reported as unknown, no throw`);
	}
});

test("validateConfig: prototype-chain group names cannot crash load or runtime", () => {
	// Regression: "@constructor" crashed validateConfig ("list is not iterable")
	// and activeGroup "constructor" passed validation then crashed every route.
	// JSON.parse creates "__proto__" as an OWN property (unlike an object
	// literal, which would set the prototype) - this mirrors real config loads.
	const parsedGroups = JSON.parse(
		'{"g": ["@constructor", "litellm/z-ai/glm-5.3"], "__proto__": ["litellm/z-ai/glm-5.3"]}',
	) as Record<string, string[]>;
	const config = testConfig({
		groups: parsedGroups,
		activeGroup: "constructor",
		defaultGroup: "toString",
	});
	assert.ok(config.problems.some((p) => p.includes('unknown group "@constructor"')));
	assert.ok(config.problems.some((p) => p.includes('"__proto__" is a reserved group name')));
	assert.equal(config.activeGroup, null, "inherited name is not a defined group");
	assert.equal(config.defaultGroup, null);
	// Runtime expansion must stay safe for configs that bypass validation:
	const handBuilt = testConfig({ activeGroup: "constructor" as unknown as string });
	assert.doesNotThrow(() => router.eligibleModels(handBuilt, registryFor(ALL_KEYS)));
});

test("expandGroupRefs: circular @group references are problems, not hangs", () => {
	const groups: Record<string, string[]> = {
		a: ["@b"],
		b: ["@a"],
	};
	const problems: string[] = [];
	const out = expandGroupRefs(groups.a, groups, CATALOG_KEYS, problems);
	assert.equal(out.length, 0);
	assert.ok(problems.some((p) => p.includes("circular reference")), "cycle reported");
});

test("filterByComposites filters a model list to group members in order", () => {
	const config = testConfig({
		groups: { economy: ["litellm/deepseek/deepseek-v4-flash", "litellm/z-ai/glm-5.3-flash"] },
	});
	const members = filterByComposites(config.models, [
		"litellm/z-ai/glm-5.3-flash",
		"litellm/deepseek/deepseek-v4-flash",
	]);
	assert.deepEqual(
		members.map((m) => `${m.provider}/${m.id}`),
		["litellm/deepseek/deepseek-v4-flash", "litellm/z-ai/glm-5.3-flash"],
		"catalog order preserved regardless of group ref order",
	);
});

// ---------------------------------------------------------------------------
// validateConfig: groups parsing
// ---------------------------------------------------------------------------

test("validateConfig: no groups section = empty groups, no filter", () => {
	const config = testConfig();
	assert.equal(config.activeGroup, null);
	assert.equal(config.defaultGroup, null);
	assert.deepEqual(config.groups, {});
});

test("validateConfig: groups parse; activeGroup/defaultGroup accepted", () => {
	const config = testConfig({
		groups: {
			economy: ["litellm/deepseek/deepseek-v4-flash"],
			composed: ["@economy", "litellm/z-ai/glm-5.3"],
			all: ["*"],
			allExceptQwen: ["*", "-litellm/qwen/qwen3.8-max"],
		},
		activeGroup: "economy",
		defaultGroup: "allExceptQwen",
	});
	assert.equal(config.activeGroup, "economy");
	assert.equal(config.defaultGroup, "allExceptQwen");
	assert.equal(config.problems.length, 0, "no problems from valid groups");
});

test("validateConfig: bad group shapes and unknown refs are problems", () => {
	const config = testConfig({
		groups: {
			bad: "not-an-array",
			ghostRef: ["litellm/nope/nothing"],
			ghostGroup: ["@ghost"],
		},
		activeGroup: 42,
		defaultGroup: "nowhere",
	});
	assert.equal(config.activeGroup, null, "non-string activeGroup ignored");
	assert.equal(config.defaultGroup, null, "undefined group name ignored");
	assert.ok(config.problems.some((p) => p.includes('"bad" must be an array')));
	assert.ok(config.problems.some((p) => p.includes('unknown model "litellm/nope/nothing"')));
	assert.ok(config.problems.some((p) => p.includes('unknown group "@ghost"')));
	assert.ok(config.problems.some((p) => p.includes('"activeGroup" must be a non-empty string or null')));
	assert.ok(config.problems.some((p) => p.includes('"defaultGroup" names no defined group')));
});

test("validateConfig: unknown group keys in groups object are reported", () => {
	const config = testConfig({
		groups: { a: ["litellm/z-ai/glm-5.3"], b: ["@a"] } as Record<string, unknown> as never,
	});
	assert.deepEqual(config.groups.a, ["litellm/z-ai/glm-5.3"]);
	assert.equal(config.problems.filter((p) => p.includes("unknown")).length, 0, "valid nested refs are clean");
});

// ---------------------------------------------------------------------------
// eligibleModels: the active-group filter + fallback chain
// ---------------------------------------------------------------------------

test("eligibleModels: no activeGroup = full catalog (today's behavior)", () => {
	const config = testConfig();
	const eligible = router.eligibleModels(config, registryFor(ALL_KEYS));
	assert.equal(eligible.length, config.models.length);
});

test("eligibleModels: activeGroup filters to the group's models", () => {
	const config = testConfig({
		groups: { economy: ["litellm/deepseek/deepseek-v4-flash", "litellm/z-ai/glm-5.3-flash"] },
		activeGroup: "economy",
	});
	const eligible = router.eligibleModels(config, registryFor(ALL_KEYS));
	assert.deepEqual(
		eligible.map((e) => `${e.provider}/${e.id}`),
		["litellm/deepseek/deepseek-v4-flash", "litellm/z-ai/glm-5.3-flash"],
	);
});

test("eligibleModels: disabled or registry-absent group members drop out", () => {
	const config = testConfig({
		groups: {
			economy: ["litellm/deepseek/deepseek-v4-flash", "litellm/z-ai/glm-5.3-flash", "litellm/ghost/model"],
		},
		activeGroup: "economy",
		models: [
			{ provider: "litellm", id: "deepseek/deepseek-v4-flash", description: "x", priceIn: 0.01 },
			{ provider: "litellm", id: "z-ai/glm-5.3-flash", description: "y", priceIn: 0.15, enabled: false },
			{ provider: "litellm", id: "z-ai/glm-5.3", description: "z", priceIn: 0.04 },
		],
	});
	const eligible = router.eligibleModels(config, registryFor(["litellm/deepseek/deepseek-v4-flash", "litellm/z-ai/glm-5.3"]));
	assert.deepEqual(
		eligible.map((e) => `${e.provider}/${e.id}`),
		["litellm/deepseek/deepseek-v4-flash"],
		"absent + disabled members excluded",
	);
});

test("eligibleModels: empty active group falls back to defaultGroup, then default", () => {
	const config = testConfig({
		groups: {
			dead: ["litellm/ghost/model"],
			backup: ["litellm/z-ai/glm-5.3"],
			alsoDead: ["litellm/ghost/other"],
		},
		activeGroup: "dead",
		defaultGroup: "alsoDead",
		default: "litellm/z-ai/glm-5.3-flash",
	});
	// defaultGroup exhausted too -> `default` model is used.
	const eligible = router.eligibleModels(config, registryFor(ALL_KEYS));
	assert.deepEqual(
		eligible.map((e) => `${e.provider}/${e.id}`),
		["litellm/z-ai/glm-5.3-flash"],
		"default model rescued the empty chain",
	);
});

test("eligibleModels: empty active group with no rescue yields [] (route errors)", () => {
	const config = testConfig({
		groups: { dead: ["litellm/ghost/model"] },
		activeGroup: "dead",
	});
	const eligible = router.eligibleModels(config, registryFor(ALL_KEYS));
	assert.equal(eligible.length, 0);
});

test("eligibleModels: unknown activeGroup degrades to the full catalog", () => {
	const config = testConfig({ activeGroup: "ghost" as unknown as string });
	// Bypasses validation: runtime must warn + ignore rather than break.
	const eligible = router.eligibleModels(config, registryFor(ALL_KEYS));
	assert.equal(eligible.length, config.models.length, "full catalog still routable");
});

// ---------------------------------------------------------------------------
// route(): override and sticky outrank the group filter
// ---------------------------------------------------------------------------

test("route(): manual override wins even outside the active group", async () => {
	const config = testConfig({
		groups: { economy: ["litellm/deepseek/deepseek-v4-flash"] },
		activeGroup: "economy",
		override: "litellm/qwen/qwen3.8-max",
	});
	router.setLayaForTests(
		mockClassifier({
			model: { "litellm/deepseek/deepseek-v4-flash": 0.9 },
			complexity: { standard: 0.9 },
			category: { question: 0.9 },
		}),
	);
	// Use the exported pipeline directly: eligible is group-filtered, but the
	// override lookup is not.
	const route = await router.handleRequest(routeRequest([userMsg("hello")], undefined) as never, makeCtx() as never, config, "local" as never);
	assert.equal(route.model.id, "qwen/qwen3.8-max", "override pins the model despite the group");
});

test("route(): sticky keeps a model outside the active group until the next user turn", async () => {
	const config = testConfig({
		groups: { economy: ["litellm/deepseek/deepseek-v4-flash"] },
		activeGroup: "economy",
	});
	router.setLayaForTests(
		mockClassifier({
			model: { "litellm/deepseek/deepseek-v4-flash": 0.9 },
			complexity: { standard: 0.9 },
			category: { question: 0.9 },
		}),
	);
	const route = await router.handleRequest(
		routeRequest([userMsg("continue")], { model: "litellm/z-ai/glm-5.3", summary: "" }, "continuation") as never,
		makeCtx() as never,
		config,
		"local" as never,
	);
	assert.equal(route.model.id, "z-ai/glm-5.3", "sticky survives a group switch on a continuation");
});

test("route(): a user turn after a group switch re-classifies within the group", async () => {
	const config = testConfig({
		groups: { economy: ["litellm/deepseek/deepseek-v4-flash"] },
		activeGroup: "economy",
	});
	router.setLayaForTests(
		mockClassifier({
			model: { "litellm/deepseek/deepseek-v4-flash": 0.9 },
			complexity: { standard: 0.9 },
			category: { question: 0.9 },
		}),
	);
	const route = await router.handleRequest(
		routeRequest([userMsg("new topic")], { model: "litellm/z-ai/glm-5.3", summary: "" }, "user") as never,
		makeCtx() as never,
		config,
		"local" as never,
	);
	assert.equal(route.model.id, "deepseek/deepseek-v4-flash", "new user turn routes within the group");
});

// ---------------------------------------------------------------------------
// direct requests (/compact and other out-of-loop calls)
// ---------------------------------------------------------------------------

test("direct picks the cheapest model of defaultGroup when configured", async () => {
	router.setLayaForTests(mockClassifier({ model: {}, complexity: {}, category: {} }));
	const config = testConfig({
		defaultGroup: "economy",
		groups: {
			economy: [
				"litellm/deepseek/deepseek-v4-flash",
				"litellm/qwen/qwen3.8-max", // expensive decoy in the same group
				"litellm/z-ai/glm-5.3", // cheaper than qwen but out of group
			],
		},
	});
	router.setConfigForTests(config);
	const route = await router.handleRequest(
		routeRequest([userMsg("compaction summary")], undefined, "direct") as never,
		makeCtx(null) as never,
		config,
		"local" as never,
	);
	assert.equal(route.model.id, "deepseek/deepseek-v4-flash", "cheapest INSIDE defaultGroup");
});

test("direct without defaultGroup still prefers the configured default model", async () => {
	router.setLayaForTests(mockClassifier({ model: {}, complexity: {}, category: {} }));
	const config = testConfig({ default: "litellm/z-ai/glm-5.3" }); // not the cheapest model
	router.setConfigForTests(config);
	const route = await router.handleRequest(
		routeRequest([userMsg("compaction summary")], undefined, "direct") as never,
		makeCtx(null) as never,
		config,
		"local" as never,
	);
	assert.equal(route.model.id, "z-ai/glm-5.3", "default wins when defaultGroup is absent");
});

test("direct degrades gracefully when defaultGroup has no eligible models", async () => {
	router.setLayaForTests(mockClassifier({ model: {}, complexity: {}, category: {} }));
	const config = testConfig({
		defaultGroup: "dead",
		groups: { dead: ["litellm/ghost/model"] },
		default: "litellm/deepseek/deepseek-v4-flash",
	});
	router.setConfigForTests(config);
	const route = await router.handleRequest(
		routeRequest([userMsg("compaction summary")], undefined, "direct") as never,
		makeCtx(null) as never,
		config,
		"local" as never,
	);
	assert.equal(route.model.id, "deepseek/deepseek-v4-flash", "dead defaultGroup falls back to default");
});

test("direct ignores activeGroup - it uses defaultGroup even when activeGroup excludes it", async () => {
	// Regression: directModels received the activeGroup-filtered set, so an
	// economy defaultGroup collapsed to empty and /compact picked the
	// cheapest of premium instead of the cheapest of economy.
	router.setLayaForTests(mockClassifier({ model: {}, complexity: {}, category: {} }));
	const config = testConfig({
		groups: {
			premium: ["litellm/qwen/qwen3.8-max", "litellm/z-ai/glm-5.3"],
			economy: ["litellm/deepseek/deepseek-v4-flash", "litellm/z-ai/glm-5.3-flash"],
		},
		activeGroup: "premium",
		defaultGroup: "economy",
	});
	router.setConfigForTests(config);
	const route = await router.handleRequest(
		routeRequest([userMsg("compaction summary")], undefined, "direct") as never,
		makeCtx(null) as never,
		config,
		"local" as never,
	);
	assert.equal(
		route.model.id,
		"deepseek/deepseek-v4-flash",
		"compact uses the cheapest of defaultGroup, never the active group's",
	);
});

// ---------------------------------------------------------------------------
// dashboard
// ---------------------------------------------------------------------------

test("dashboard: Config section shows the active group with ref count, or none", () => {
	const withGroup = testConfig({
		groups: { economy: ["litellm/deepseek/deepseek-v4-flash", "litellm/z-ai/glm-5.3-flash"] },
		activeGroup: "economy",
	});
	const report = formatDashboard(withGroup);
	assert.ok(
		report.includes('- Groups: active "economy" (2 refs) · default none'),
		"merged group line present with active count",
	);

	const without = testConfig();
	const plain = formatDashboard(without);
	assert.ok(
		plain.includes("- Groups: active none · default none"),
		"unset merged line stays short (no wrap on 80-col terminals)",
	);
});

test("dashboard: Config section shows the default group (/compact pool), or none", () => {
	const withDefault = testConfig({
		groups: { economy: ["litellm/deepseek/deepseek-v4-flash"] },
		defaultGroup: "economy",
	});
	const report = formatDashboard(withDefault);
	assert.ok(
		report.includes('- Groups: active none · default "economy" (1 ref)'),
		"merged group line present with default count",
	);

	const both = testConfig({
		groups: { economy: ["litellm/deepseek/deepseek-v4-flash", "litellm/z-ai/glm-5.3-flash"] },
		activeGroup: "economy",
		defaultGroup: "economy",
	});
	assert.ok(
		formatDashboard(both).includes('- Groups: active "economy" (2 refs) · default "economy" (2 refs)'),
		"both set renders both on one line",
	);
});

test("groupLabel: label is correct for both fields, unset defaults to short none", () => {
	const config = testConfig({
		groups: { economy: ["litellm/deepseek/deepseek-v4-flash", "litellm/z-ai/glm-5.3-flash"] },
	});
	assert.equal(groupLabel(config, "activeGroup"), "none");
	assert.equal(groupLabel(config, "defaultGroup"), "none");
	config.activeGroup = "economy";
	assert.equal(groupLabel(config, "activeGroup"), '"economy" (2 refs)');
	// Explicit override for specialized labels (menu Active row).
	assert.equal(groupLabel(config, "defaultGroup", "nope"), "nope");
});