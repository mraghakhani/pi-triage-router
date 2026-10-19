/**
 * Unit tests for triage-router. Deterministic: no Laya load, no network, no
 * paid API calls. The classifier is mocked; every test controls its output.
 *
 * Run: node --test tests/triage-router.test.ts
 */
import { test, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import * as router from "../triage-router.ts";
import { piAgentDir } from "../triage-router-lib/triage-router-openrouter.ts";
import type { CapturedVirtualModel } from "./capture.ts";
import { captureRoute } from "./capture.ts";
import {
	CATALOG,
	ALL_KEYS,
	registryFor,
	testConfig,
	userMsg,
	asstMsg,
	makeCtx,
	routeRequest,
	mockClassifier,
} from "./fixtures.ts";
import type { FakeModel } from "./fixtures.ts";

beforeEach(() => router.setConfigForTests(testConfig()));
afterEach(() => {
	router.resetLayaForTests();
	router.setConfigForTests(null);
});

// ---------------------------------------------------------------------------
// 1-2. Context building: history reaches Laya, current request wins
// ---------------------------------------------------------------------------

test("routing input includes prior conversation context for follow-up requests", () => {
	const config = testConfig();
	const messages = [
		userMsg("We are debugging a PostgreSQL connectivity problem in our Kubernetes cluster."),
		asstMsg("I checked the pg service endpoints."),
		userMsg("Can you fix the same issue in the other service?"),
	];
	const input = router.buildRoutingInput(messages, "", "Debugging PostgreSQL in Kubernetes", config);
	assert.ok(input.prompt.includes("PostgreSQL"), "prior context with PostgreSQL reaches Laya");
	assert.ok(input.prompt.includes("Kubernetes"));
	// The current request is preserved verbatim and is the last section.
	assert.ok(input.prompt.includes("Can you fix the same issue in the other service?"));
	assert.ok(
		input.prompt.lastIndexOf("Can you fix the same issue in the other service?") > input.prompt.lastIndexOf("PostgreSQL"),
		"current request must come after (win over) older context",
	);
	assert.equal(input.currentRequest, "Can you fix the same issue in the other service?");
});

test("rolling state summary is preferred; Pi's session summary seeds empty state", () => {
	const config = testConfig();
	const messages = [userMsg("Fix the typo in README")];
	const fromState = router.buildRoutingInput(messages, "state summary kept", null, config);
	assert.ok(fromState.prompt.includes("state summary kept"));
	const fromSession = router.buildRoutingInput(messages, "", "session compaction summary", config);
	assert.ok(fromSession.prompt.includes("session compaction summary"));
});

test("budget bounds context; the current request is never silently shortened", () => {
	const config = testConfig({ contextBudgetChars: 400 });
	const longHistory = Array.from(
		{ length: 20 },
		(_, i) => userMsg(`Turn ${i} about migrating the Postgres schema and the billing service, part ${i}.`),
	);
	const messages = [...longHistory, userMsg("Now do the same for the analytics service.")];
	const input = router.buildRoutingInput(messages, "", null, config);
	assert.ok(input.prompt.length < 3600, `assembled prompt should stay small, got ${input.prompt.length}`);
	assert.equal(input.currentRequest, "Now do the same for the analytics service.");
	assert.ok(input.prompt.includes("Now do the same for the analytics service."));
});

test("a very long current request is truncated with an explicit marker, never silently", () => {
	const config = testConfig();
	const long = "Refactor the billing module. ".repeat(400);
	const input = router.buildRoutingInput([userMsg(long)], "", null, config);
	assert.ok(input.currentRequest.length <= 6000);
	assert.ok(input.currentRequest.endsWith("[current request truncated]"));
	assert.equal(router.lastUserText([userMsg(long)]), long, "lastUserText itself never truncates");
});

test("unrelated new task after a long discussion still carries the full new request", () => {
	const config = testConfig();
	const messages = [
		userMsg("Long discussion about the PostgreSQL connection pool."),
		asstMsg("Pool size 100 seems wrong."),
		userMsg("Completely unrelated: write a haiku about UDP packets."),
	];
	const input = router.buildRoutingInput(messages, "Old summary about Postgres pool debugging", null, config);
	assert.ok(input.prompt.includes("write a haiku about UDP packets."));
});

test("refreshSummary rolls forward, bounds size, and drops the oldest content first", () => {
	const config = testConfig({ recentTurns: 2 });
	// Simulate a growing conversation: each new request rolls one turn out of
	// the recent window and into the summary, like real route() calls do.
	const turns = [
		"First task about nginx ingress annotations.",
		...Array.from({ length: 10 }, (_, i) => `Bulk task ${i} ${"x".repeat(200)}`),
	];
	const current = "CURRENT: continue the migration.";
	let summary = "";
	for (let i = 1; i <= turns.length; i++) {
		const messages = [...turns.slice(0, i), current].map((t) => userMsg(t));
		summary = router.refreshSummary(summary, messages, config);
	}
	assert.ok(summary.length <= 640, `summary must stay bounded, got ${summary.length}`);
	assert.ok(!summary.includes("nginx"), "oldest content must eventually be dropped");
	assert.ok(summary.includes("Bulk task 7"), "newest rolled-out turn survives; Bulk task 8/9 stay in the recent window");
});

test("recent turns are not duplicated into the rolling summary", () => {
	const config = testConfig();
	const messages = [
		userMsg("Old task about the TLS cipher suites."),
		userMsg("Latest task about the WebSocket gateway."),
	];
	const summary = router.refreshSummary("prior context: TLS cipher suites", messages, config);
	// The newest turn is in the recent window, so the summary only carries older context.
	assert.ok(!summary.includes("WebSocket"));
});

// ---------------------------------------------------------------------------
// 3. Config validation
// ---------------------------------------------------------------------------

test("config validation distinguishes missing prices from zero, rejects invalid entries", () => {
	const config = router.validateConfig({
		models: [
			{ provider: "litellm", id: "a/a", description: "ok", priceIn: 0 },
			{ provider: "litellm", id: "b/b", description: "no price" },
			{ provider: "litellm", id: "c/c" },
			{ provider: "litellm", id: "d/d", description: "bad price", priceIn: -5 },
			{ provider: "litellm", id: "a/a", description: "dupe" },
		],
	});
	assert.equal(config.models.length, 2, "valid + free-priced entries survive");
	assert.equal(config.models[0].priceIn, 0, "zero price is legitimate");
	assert.equal(config.models[1].priceIn, undefined, "missing price stays missing");
	assert.ok(config.problems.some((p) => p.includes("description")));
	assert.ok(config.problems.some((p) => p.includes("priceIn")));
	assert.ok(config.problems.some((p) => p.includes("duplicate")));
});

test("config validation reports malformed fallback/default/override keys", () => {
	const config = router.validateConfig({
		models: [{ provider: "p", id: "m", description: "d" }],
		fallback: "nonsense",
		default: "no-slash",
		override: 42,
		unknownKey: true,
	});
	assert.equal(config.fallback, "cheapest");
	assert.equal(config.defaultModel, undefined);
	assert.equal(config.override, undefined);
	assert.ok(config.problems.some((p) => p.includes("unknown config key")));
	assert.ok(config.problems.some((p) => p.includes('"default"')));
});

test("missing or empty models falls back to defaults with a problem logged", () => {
	const config = router.validateConfig({});
	assert.equal(config.models, router.DEFAULT_MODELS);
	assert.ok(config.problems.some((p) => p.includes("models")));
});

// ---------------------------------------------------------------------------
// 4. Eligibility
// ---------------------------------------------------------------------------

test("disabled and registry-missing models are never eligible", () => {
	const config = testConfig();
	config.models[5].enabled = false; // mimo-v2.5 disabled
	config.models[4].id = "qwen/qwen3.8-max-0902"; // not in the fake registry
	const eligible = router.eligibleModels(config, registryFor(ALL_KEYS));
	const ids = eligible.map((m) => m.id);
	assert.equal(ids.length, 4);
	assert.ok(!ids.includes("xiaomi/mimo-v2.5"));
	assert.ok(!ids.includes("qwen/qwen3.8-max-0902"));
});

// ---------------------------------------------------------------------------
// 5-7. Decision validation, rules, fallback policy
// ---------------------------------------------------------------------------

test("Laya picks outside the eligible set are rejected deterministically", () => {
	const config = testConfig();
	const eligible = router.eligibleModels(config, registryFor(ALL_KEYS));
	const decision = router.decide(
		{
			model: { "some/unknown-model": 0.99 },
			complexity: { standard: 1 },
			category: { question: 1 },
		},
		config,
		eligible,
	);
	assert.equal(decision.rule, "fallback");
	assert.equal(decision.entry.id, "deepseek/deepseek-v4-flash");
	assert.ok(decision.notes.some((n) => n.includes("rejected")));
	assert.equal(decision.confidence, null);
});

test("complex demanding work is escalated off the weakest tier", () => {
	const config = testConfig({ fallback: "strongest" });
	const eligible = router.eligibleModels(config, registryFor(ALL_KEYS));
	// Laya picks the cheapest but flags complex debugging work.
	const decision = router.decide(
		{
			model: { "litellm/deepseek/deepseek-v4-flash": 0.9 },
			complexity: { complex: 0.9 },
			category: { debugging: 0.9 },
		},
		config,
		eligible,
	);
	assert.equal(decision.rule, "escalated");
	assert.equal(decision.entry.id, "qwen/qwen3.8-max", "strongest eligible wins the escalation");
});

test("complex work on a non-weakest model is left alone", () => {
	const config = testConfig();
	const eligible = router.eligibleModels(config, registryFor(ALL_KEYS));
	const decision = router.decide(
		{
			model: { "litellm/deepseek/deepseek-v4-pro": 0.9 },
			complexity: { complex: 0.9 },
			category: { reasoning: 0.9 },
		},
		config,
		eligible,
	);
	assert.equal(decision.rule, "primary");
	assert.equal(decision.entry.id, "deepseek/deepseek-v4-pro");
});

test("budget-tagged work is downgraded to the cheapest eligible model", () => {
	const config = testConfig();
	const eligible = router.eligibleModels(config, registryFor(ALL_KEYS));
	const decision = router.decide(
		{
			model: { "litellm/qwen/qwen3.8-max": 0.9 },
			complexity: { standard: 0.9 },
			category: { budget: 0.9 },
		},
		config,
		eligible,
	);
	assert.equal(decision.rule, "downgraded");
	assert.equal(decision.entry.id, "deepseek/deepseek-v4-flash");
});

test("fallback policy prefers a valid default, then the configured direction", () => {
	const eligible = router.eligibleModels(testConfig(), registryFor(ALL_KEYS));
	const withDefault = testConfig({ default: "litellm/deepseek/deepseek-v4-pro", fallback: "strongest" });
	assert.equal(
		router.decide({ model: {}, complexity: {}, category: {} }, withDefault, eligible).entry.id,
		"deepseek/deepseek-v4-pro",
	);
	const strongest = testConfig({ fallback: "strongest" });
	assert.equal(router.decide({ model: {}, complexity: {}, category: {} }, strongest, eligible).entry.id, "qwen/qwen3.8-max");
	const cheapest = testConfig({ fallback: "cheapest" });
	assert.equal(
		router.decide({ model: {}, complexity: {}, category: {} }, cheapest, eligible).entry.id,
		"deepseek/deepseek-v4-flash",
	);
	const badDefault = testConfig({ default: "litellm/no/such-model" });
	assert.equal(
		router.decide({ model: {}, complexity: {}, category: {} }, badDefault, eligible).entry.id,
		"deepseek/deepseek-v4-flash",
	);
});

test("empty classifier probabilities default to standard/coding, not question", () => {
	const eligible = router.eligibleModels(testConfig(), registryFor(ALL_KEYS));
	const decision = router.decide({ model: {}, complexity: {}, category: {} }, testConfig(), eligible);
	assert.equal(decision.complexity, "standard");
	assert.equal(decision.category, "coding");
});

test("fallback throws (observably) when nothing is eligible", () => {
	const config = testConfig();
	config.models[0].enabled = false;
	assert.throws(() => router.decide({ model: {}, complexity: {}, category: {} }, config, []), /no eligible models/);
});

// ---------------------------------------------------------------------------
// 8. Session peek + cost estimation
// ---------------------------------------------------------------------------

test("peekSession finds the latest summary and latest usage entry", () => {
	const ctx = makeCtx(null, [
		{ type: "compaction", summary: "Old summary about Terraform state" },
		{ type: "message", message: {} },
		{ type: "compaction", summary: "New summary about Kubernetes Operators" },
		{ type: "usage", provider: "litellm", model: "deepseek/deepseek-v4-flash", usage: { input: 100, output: 50, totalTokens: 150, cost: { total: 0.01 } } },
		{ type: "usage", provider: "litellm", model: "z-ai/glm-5.3", usage: { input: 900, output: 100, totalTokens: 1000, cost: { total: 0.2 } } },
	]);
	const peek = router.peekSession(ctx);
	assert.equal(peek.summary, "New summary about Kubernetes Operators");
	assert.equal(peek.lastUsage?.model, "z-ai/glm-5.3");
	assert.equal(peek.lastUsage?.cost, 0.2);
});

test("peekSession never fabricates context when the session manager fails", () => {
	const ctx = { sessionManager: { getEntries: () => { throw new Error("boom"); } } } as never;
	const peek = router.peekSession(ctx);
	assert.equal(peek.summary, null);
	assert.equal(peek.lastUsage, null);
});

test("cost estimation prefers actual usage data, then context tokens, then chars/4", () => {
	const entry = { ...router.DEFAULT_MODELS[0], priceIn: 0.01 };
	const registryModel = CATALOG["litellm/deepseek/deepseek-v4-flash"] as never;
	const fromUsage = router.estimateCost(entry, registryModel, 4000, 5000, {
		provider: "litellm",
		model: "deepseek/deepseek-v4-flash",
		inputTokens: 12_000,
		outputTokens: 0,
		cost: 0.12,
	});
	assert.equal(fromUsage.source, "usage");
	assert.equal(fromUsage.inputTokens, 12_000);
	assert.equal(fromUsage.estCostIn, 0.00012, "12k tokens at 0.01 USD per 1M tokens");
	const fromContext = router.estimateCost(entry, registryModel, 4000, 5000, null);
	assert.equal(fromContext.source, "context");
	assert.equal(fromContext.inputTokens, 5000);
	const fromChars = router.estimateCost(entry, registryModel, 4000, null, null);
	assert.equal(fromChars.source, "chars");
	assert.equal(fromChars.inputTokens, 1000);
	const nothing = router.estimateCost({ ...entry, priceIn: undefined }, undefined, 0, null, null);
	assert.equal(nothing.estCostIn, null, "no price + no tokens: never fabricate a cost");
});

// ---------------------------------------------------------------------------
// 9-14. classifyAndDecide with a mocked classifier
// ---------------------------------------------------------------------------

test("classifyAndDecide routes a Kubernetes debugging task with history and estimates cost", async () => {
	router.setLayaForTests(
		mockClassifier({
			model: { "litellm/z-ai/glm-5.3": 0.8 },
			complexity: { complex: 0.7 },
			category: { infra: 0.9 },
		}),
	);
	const config = testConfig({ recentTurns: 1 });
	const ctx = makeCtx({ tokens: 42_000, contextWindow: 1_000_000 });
	const messages = [
		userMsg("Our Kubernetes pods crash-loop with 503 from the ingress."),
		asstMsg("The readiness probe fails on port 8080."),
		userMsg("Fix the deployment so the gateway stops returning 503."),
	];
	const { decision, input, summary } = await router.classifyAndDecide(
		routeRequest(messages, undefined),
		ctx,
		config,
		router.eligibleModels(config, ctx.modelRegistry),
	);
	assert.equal(decision.entry.id, "z-ai/glm-5.3");
	assert.equal(decision.rule, "primary");
	assert.equal(decision.category, "infra");
	assert.equal(decision.complexity, "complex");
	assert.equal(decision.estimate.source, "context");
	assert.equal(decision.estimate.inputTokens, 42_000);
	assert.ok(input.prompt.includes("readiness probe"));
	assert.ok(summary.includes("503"), "summary rolls prior turns forward in state");
});

test("classifyAndDecide escalates when Laya picks the weakest tier for complex debugging", async () => {
	router.setLayaForTests(
		mockClassifier({
			model: { "litellm/deepseek/deepseek-v4-flash": 0.9 },
			complexity: { complex: 0.9 },
			category: { debugging: 0.9 },
		}),
	);
	const config = testConfig({ fallback: "strongest" });
	const ctx = makeCtx(null);
	const { decision } = await router.classifyAndDecide(
		routeRequest([userMsg("Root-cause the flaky e2e suite across 40 services.")], undefined),
		ctx,
		config,
		router.eligibleModels(config, ctx.modelRegistry),
	);
	assert.equal(decision.rule, "escalated");
	assert.equal(decision.entry.id, "qwen/qwen3.8-max");
});

test("classifyAndDecide downgrades budget-tagged work off expensive picks", async () => {
	router.setLayaForTests(
		mockClassifier({
			model: { "litellm/qwen/qwen3.8-max": 0.9 },
			complexity: { standard: 0.9 },
			category: { budget: 0.9 },
		}),
	);
	const config = testConfig();
	const ctx = makeCtx(null);
	const { decision } = await router.classifyAndDecide(
		routeRequest([userMsg("Cheap as possible: update the docs table.")], undefined),
		ctx,
		config,
		router.eligibleModels(config, ctx.modelRegistry),
	);
	assert.equal(decision.rule, "downgraded");
	assert.equal(decision.entry.id, "deepseek/deepseek-v4-flash");
});

test("a rejecting or malformed Laya response falls back deterministically without throwing", async () => {
	const config = testConfig();
	const ctx = makeCtx(null);
	const eligible = router.eligibleModels(config, ctx.modelRegistry);
	const rejects = { systemOne: async () => {
		throw new Error("inference failed");
	} } as unknown as router.Classifier;
	router.setLayaForTests(rejects);
	const { decision } = await router.classifyAndDecide(
		routeRequest([userMsg("Anything.")], undefined),
		ctx,
		config,
		eligible,
	);
	assert.equal(decision.rule, "fallback");
	assert.equal(decision.entry.id, "deepseek/deepseek-v4-flash");
	const garbage = { systemOne: async () => ({ answers: { model: { probabilities: null } } }) } as unknown as router.Classifier;
	router.setLayaForTests(garbage);
	const { decision: d2 } = await router.classifyAndDecide(
		routeRequest([userMsg("Anything.")], undefined),
		ctx,
		config,
		eligible,
	);
	assert.equal(d2.rule, "fallback");
});

test("Laya unavailable (load failed) falls back deterministically", async () => {
	router.setLayaForTests(null);
	const config = testConfig();
	const ctx = makeCtx(null);
	const { decision, layaMs } = await router.classifyAndDecide(
		routeRequest([userMsg("Anything.")], undefined),
		ctx,
		config,
		router.eligibleModels(config, ctx.modelRegistry),
	);
	assert.equal(decision.rule, "fallback");
	assert.equal(layaMs, null);
});

test("slow Laya inference times out and falls back", async () => {
	const slow: router.Classifier = { systemOne: () => new Promise(() => {}) };
	router.setLayaForTests(slow);
	const config = testConfig({ layaTimeoutMs: 150 }); // validation floor is 100
	const ctx = makeCtx(null);
	const { decision } = await router.classifyAndDecide(
		routeRequest([userMsg("Anything.")], undefined),
		ctx,
		config,
		router.eligibleModels(config, ctx.modelRegistry),
	);
	assert.equal(decision.rule, "fallback");
	assert.equal(decision.entry.id, "deepseek/deepseek-v4-flash");
});

// ---------------------------------------------------------------------------
// 15-17. Full route() flow with a captured virtual model
// ---------------------------------------------------------------------------

test("direct requests route to the cheapest eligible model", async () => {
	router.setLayaForTests(mockClassifier({ model: {}, complexity: {}, category: {} }));
	const vm = captureRoute();
	const route = await vm.route(
		routeRequest([userMsg("compaction summary")], undefined, "direct"),
		makeCtx(null) as never,
	);
	assert.equal(route.model.id, "deepseek/deepseek-v4-flash");
});

test("override and direct routes report the model that actually handled the turn", async () => {
	router.setConfigForTests(testConfig({ override: "litellm/deepseek/deepseek-v4-pro" }));
	const prior: router.LayaState = { model: "litellm/z-ai/glm-5.3", summary: "kept" };
	const ctx = makeCtx(null) as never;

	// Override wins and the persisted state must point at the override model,
	// so a later continuation sticks to the model that served the turn.
	const vm = captureRoute();
	const overrideRoute = await vm.route(routeRequest([userMsg("any request")], prior, "user"), ctx);
	assert.equal(overrideRoute.model.id, "deepseek/deepseek-v4-pro");
	const state = overrideRoute.state as router.LayaState;
	assert.equal(state.model, "litellm/deepseek/deepseek-v4-pro");
	assert.equal(state.summary, "kept", "summary preserved");
});

test("continuations stay sticky on the session model, retry included", async () => {
	const vm = captureRoute();
	const ctx = makeCtx(null) as never;
	const state: router.LayaState = { model: "litellm/z-ai/glm-5.3", summary: "was debugging" };
	for (const reason of ["continuation", "retry"] as const) {
		const route = await vm.route(routeRequest([userMsg("tool result follows")], state, reason), ctx);
		assert.equal(route.model.id, "z-ai/glm-5.3", `${reason} keeps the session model`);
	}
});

test("continuation whose state model disappeared re-classifies instead of crashing", async () => {
	router.setLayaForTests(
		mockClassifier({
			model: { "litellm/deepseek/deepseek-v4-pro": 0.9 },
			complexity: { moderate: 0.9 },
			category: { coding: 0.9 },
		}),
	);
	const vm = captureRoute();
	const state: router.LayaState = { model: "litellm/no/such-model", summary: "" };
	const route = await vm.route(
		routeRequest([userMsg("continue the refactor")], state, "continuation"),
		makeCtx(null) as never,
	);
	assert.equal(route.model.id, "deepseek/deepseek-v4-pro");
});

test("user messages classify with context and the choice lands on a real model", async () => {
	router.setLayaForTests(
		mockClassifier({
			model: { "litellm/deepseek/deepseek-v4-pro": 0.75 },
			complexity: { moderate: 0.8 },
			category: { coding: 0.8 },
		}),
	);
	const vm = captureRoute();
	const ctx = makeCtx({ tokens: 1000, contextWindow: 1_000_000 }) as never;
	const route = await vm.route(
		routeRequest(
			[userMsg("Implement the retry policy in the payments client."), asstMsg("On it."), userMsg("Also add the tests.")],
			undefined,
			"user",
		),
		ctx,
	);
	assert.equal(route.model.id, "deepseek/deepseek-v4-pro");
	assert.equal((route as { state: router.LayaState }).state.model, "litellm/deepseek/deepseek-v4-pro");
});

test("routing throws observably when no eligible model exists", async () => {
	router.setLayaForTests(mockClassifier({ model: {}, complexity: {}, category: {} }));
	const vm = captureRoute();
	// A registry that answers nothing for the shipped config's providers.
	const emptyCtx = {
		modelRegistry: registryFor([]),
		getContextUsage: () => undefined,
		sessionManager: { getEntries: () => [] },
	} as never;
	await assert.rejects(
		() => vm.route(routeRequest([userMsg("Anything.")], undefined), emptyCtx),
		/no eligible models/,
	);
});
// ---------------------------------------------------------------------------
// Audit 1. Effective context limits: the prompt must fit Laya's state room
// ---------------------------------------------------------------------------

test("a long current request is bounded to Laya's state room with an explicit marker", () => {
	const config = testConfig();
	const long = "Refactor the billing module. ".repeat(400); // ~11k chars
	const input = router.buildRoutingInput([userMsg(long)], "", null, config);
	// Laya's state room is ~316 tokens; conservative allocation ~876 chars.
	// The request may be truncated, but only explicitly, never silently.
	assert.ok(input.currentRequest.length <= 800, `request must fit the room, got ${input.currentRequest.length}`);
	assert.ok(input.currentRequest.endsWith("[current request truncated]"), "truncation must be explicit");
	assert.equal(router.lastUserText([userMsg(long)]), long, "lastUserText itself never truncates");
	// Context is dropped before the request is cut further.
	assert.equal(input.recent, "", "no history: recent section empty");
	assert.equal(input.summary, "");
	assert.equal(input.stateTokens, null, "no tokenizer: stateTokens unknown");
	assert.equal(input.roomTokens, router.LAYA_STATE_ROOM_TOKENS);
});

test("huge history never squeezes the current request; context sections stay bounded", () => {
	const config = testConfig();
	const history = Array.from(
		{ length: 30 },
		(_, i) => userMsg(`Turn ${i}: migrating the Postgres schema and the billing service, part ${i}.`),
	);
	const messages = [...history, userMsg("Now do the same for the analytics service.")];
	const input = router.buildRoutingInput(messages, "", null, config);
	assert.equal(input.currentRequest, "Now do the same for the analytics service.", "request is never shortened for context");
	// Room-derived bounds: recent + summary bounded, total prompt bounded.
	const overhead = input.prompt.length - input.recent.length - input.summary.length - input.currentRequest.length;
	assert.ok(input.prompt.length <= 900 + overhead, `prompt must stay near the room, got ${input.prompt.length}`);
	assert.ok(input.recent.length <= 700, `recent must stay bounded, got ${input.recent.length}`);
	assert.ok(input.summary.length <= 600);
});

test("summary section in the prompt is recency-biased (newest content kept, oldest dropped)", () => {
	const config = testConfig();
	const oldLines = Array.from({ length: 30 }, (_, i) => `OLD-${i} PostgreSQL tuning note about the shared buffer cache.`);
	const summary = `${oldLines.join(" ")} NEWEST: switch the gateway to gRPC.`;
	const messages = [userMsg("Fix the gateway timeout.")];
	const input = router.buildRoutingInput(messages, summary, null, config);
	assert.ok(input.prompt.includes("NEWEST: switch the gateway to gRPC."), "newest summary content must survive");
	assert.ok(!input.prompt.includes("OLD-0 "), "oldest summary content must drop first");
	assert.ok(!input.prompt.includes("OLD-9 "), "old summary content must drop before the newest");
});

test("a classifier with a tokenizer reports state tokens and shrinks the prompt to the room", () => {
	const config = testConfig();
	const messages = [
		userMsg("Turn 1 about nginx ingress annotations in the cluster."),
		userMsg("Also explain the mTLS certificate rotation."),
	];
	// Dense tokenizer: every 2 chars = 1 token, so the room is quickly exceeded.
	const dense: router.Classifier = {
		systemOne: async () => ({ answers: {} }),
		encode: (text: string) => new Array(Math.ceil(text.length / 2)).fill(1),
		config: { max_len: 512, head_max_len: 192 },
	};
	const input = router.buildRoutingInput(messages, "Some rolling summary.", null, config, dense);
	assert.equal(input.roomTokens, 512 - 192 - 4, "room derives from the classifier's own config");
	assert.ok(input.stateTokens !== null, "tokenizer path reports state tokens");
	assert.ok(input.stateTokens <= input.roomTokens, `state must fit the room, got ${input.stateTokens}/${input.roomTokens}`);
});

test("no-tokenizer allocation is conservative: total prompt stays within the room in chars", () => {
	const config = testConfig({ contextBudgetChars: 5000 });
	const history = Array.from({ length: 20 }, (_, i) => userMsg(`Historical turn ${i} about the legacy SOAP gateway and its quirks.`));
	const messages = [...history, userMsg("A reasonably long request. ".repeat(60))];
	const input = router.buildRoutingInput(messages, "Rolling summary text. ".repeat(40), null, config);
	// contextBudgetChars is an upper bound; Laya's room is the binding limit.
	const overhead = 160;
	assert.ok(
		input.prompt.length <= router.LAYA_STATE_ROOM_TOKENS * 3 - router.STATE_ROLE_MARGIN_TOKENS * 3 + overhead,
		`conservative allocation must bound the whole prompt, got ${input.prompt.length}`,
	);
});

// ---------------------------------------------------------------------------
// Audit 2. Task changes vs continuations: when sticky, when re-classified
// ---------------------------------------------------------------------------

test("a substantially more complex user task re-classifies instead of staying sticky", async () => {
	const vm = captureRoute();
	const ctx = makeCtx(null) as never;
	// Simple task first: cheap tier nominated.
	router.setLayaForTests(
		mockClassifier({
			model: { "litellm/deepseek/deepseek-v4-flash": 0.9 },
			complexity: { standard: 0.9 },
			category: { coding: 0.9 },
		}),
	);
	const first = await vm.route(routeRequest([userMsg("Fix the typo in README.")], undefined, "user"), ctx);
	assert.equal(first.model.id, "deepseek/deepseek-v4-flash");
	// Same session, new complex user message: classification must run again.
	router.setLayaForTests(
		mockClassifier({
			model: { "litellm/deepseek/deepseek-v4-flash": 0.9 },
			complexity: { complex: 0.9 },
			category: { debugging: 0.9 },
		}),
	);
	const second = await vm.route(
		routeRequest([userMsg("Fix the typo in README."), asstMsg("Fixed."), userMsg("Root-cause the distributed 503 storm across the gateway.")], { model: "litellm/deepseek/deepseek-v4-flash", summary: "" }, "user"),
		ctx,
	);
	assert.notEqual(second.model.id, first.model.id, "complex escalation must move off the cheap tier");
	assert.equal(second.model.id, "qwen/qwen3.8-max");
});

test("a contextual follow-up keeps the model its context points at, and sees that context", async () => {
	const ctx = makeCtx(null);
	const config = testConfig();
	const messages = [
		userMsg("We're fixing the PostgreSQL connection pool exhaustion in the orders service."),
		asstMsg("Found it: the pool leak is in checkout.js."),
		userMsg("Apply the same fix to the refunds service."),
	];
	router.setLayaForTests(
		mockClassifier({
			model: { "litellm/z-ai/glm-5.3": 0.9 },
			complexity: { moderate: 0.9 },
			category: { coding: 0.9 },
		}),
	);
	const first = await router.classifyAndDecide(
		routeRequest(messages, undefined),
		ctx as never,
		config,
		router.eligibleModels(config, ctx.modelRegistry),
	);
	const second = await router.classifyAndDecide(
		routeRequest(messages, undefined),
		ctx as never,
		config,
		router.eligibleModels(config, ctx.modelRegistry),
	);
	assert.equal(first.decision.entry.id, second.decision.entry.id, "same context + request: same model");
	assert.ok(first.input.prompt.includes("pool leak"), "follow-up context must reach the classifier");
	assert.ok(first.input.prompt.includes("PostgreSQL"));
});

test("an unrelated new task carries the full new request and bounded stale context", () => {
	const config = testConfig();
	const stale = "Stale notes about the PostgreSQL pool. ".repeat(30);
	const messages = [
		userMsg("Long discussion about PostgreSQL connection pool sizes."),
		asstMsg("Pool size 100 seems wrong."),
		userMsg("Completely unrelated: write a haiku about UDP packets."),
	];
	const input = router.buildRoutingInput(messages, stale, null, config);
	assert.ok(input.prompt.includes("write a haiku about UDP packets."), "the new request must be complete");
	assert.ok(input.currentRequest.length < 60);
	assert.ok(input.summary.length <= 600, "stale summary is bounded");
	// The current request is the last section: it wins.
	assert.ok(
		input.prompt.lastIndexOf("write a haiku about UDP packets.") > input.prompt.lastIndexOf("PostgreSQL"),
	);
});

// ---------------------------------------------------------------------------
// Audit 3. Cost accounting: estimates vs actuals, never confused
// ---------------------------------------------------------------------------

test("usage from another provider with the same model id is never attributed", () => {
	const entry = { ...router.DEFAULT_MODELS[0], priceIn: 0.01 };
	const registryModel = CATALOG["litellm/deepseek/deepseek-v4-flash"] as never;
	const crossProvider = router.estimateCost(
		entry as never,
		registryModel,
		4000,
		5000,
		{ provider: "openrouter", model: "deepseek/deepseek-v4-flash", inputTokens: 99_000, outputTokens: 10, cost: 9 },
	);
	assert.equal(crossProvider.source, "context", "same id, different provider: fall through to context");
	assert.equal(crossProvider.inputTokens, 5000);
	const sameModel = router.estimateCost(
		entry as never,
		registryModel,
		4000,
		5000,
		{ provider: "litellm", model: "deepseek/deepseek-v4-flash", inputTokens: 12_000, outputTokens: 500, cost: 0.12 },
	);
	assert.equal(sameModel.source, "usage", "same provider+model: actual usage is the basis");
	assert.equal(sameModel.estCostIn, 0.00012, "12k tokens at 0.01 USD per 1M tokens");
	assert.equal(sameModel.outputTokens, 500);
	assert.equal(sameModel.estCostOut, 0.00064, "500 output tokens at 1.28 USD per 1M tokens");
});

test("zero and missing prices stay distinct; nothing is fabricated", () => {
	const freeEntry = { ...router.DEFAULT_MODELS[0], priceIn: 0, priceOut: 0 };
	const registryModel = CATALOG["litellm/deepseek/deepseek-v4-flash"] as never;
	const zero = router.estimateCost(freeEntry as never, undefined, 400, null, null);
	assert.equal(zero.estCostIn, 0, "a zero price estimates zero, not null");
	assert.equal(zero.estCostOut, null, "no usage: no output estimate is invented");
	const missing = router.estimateCost(
		{ ...router.DEFAULT_MODELS[0], priceIn: undefined, priceOut: undefined } as never,
		undefined,
		400,
		null,
		null,
	);
	assert.equal(missing.estCostIn, null, "no price: no cost is invented");
	assert.equal(missing.estCostOut, null, "no output price: no output cost is invented");
	// Actual usage with zero tokens is still an estimate basis of zero.
	const zeroTokens = router.estimateCost(
		freeEntry as never,
		registryModel,
		400,
		null,
		{ provider: "litellm", model: "deepseek/deepseek-v4-flash", inputTokens: 0, outputTokens: 0, cost: 0 },
	);
	assert.equal(zeroTokens.inputTokens, 0);
	assert.equal(zeroTokens.estCostIn, 0);
});

test("estimate uses the latest of several usage entries and the registry price", () => {
	const entry = { ...router.DEFAULT_MODELS[0], priceIn: 0.01 };
	const ctx = makeCtx(null, [
		{ type: "usage", provider: "litellm", model: "z-ai/glm-5.3", usage: { input: 900, output: 100, cost: { total: 0.2 } } },
		{ type: "usage", provider: "litellm", model: "deepseek/deepseek-v4-flash", usage: { input: 8_000, output: 1_000, cost: { total: 0.08 } } },
	]);
	const peek = router.peekSession(ctx as never);
	assert.equal(peek.lastUsage?.model, "deepseek/deepseek-v4-flash", "latest usage entry wins");
	const est = router.estimateCost(entry as never, CATALOG["litellm/deepseek/deepseek-v4-flash"] as never, 100, null, peek.lastUsage);
	assert.equal(est.inputTokens, 8_000, "previous request's usage, labeled estimate");
	assert.equal(est.source, "usage");
	assert.equal(est.estCostIn, 8_000 / 1e6 * 0.01);
});

test("the decision log labels estimates and the actual previous-request cost", async () => {
	const vm = captureRoute();
	const ctx = makeCtx(
		{ tokens: 42_000, contextWindow: 1_000_000 },
		[{ type: "usage", provider: "litellm", model: "deepseek/deepseek-v4-flash", usage: { input: 12_000, output: 500, cost: { total: 0.12 } } }],
	) as never;
	router.setLayaForTests(
		mockClassifier({
			model: { "litellm/deepseek/deepseek-v4-pro": 0.9 },
			complexity: { moderate: 0.9 },
			category: { coding: 0.9 },
		}),
	);
	const lines: string[] = [];
	const orig = console.error;
	console.error = (...args: unknown[]) => {
		lines.push(args.join(" "));
	};
	try {
		await vm.route(routeRequest([userMsg("Implement the retry policy in the payments client.")], undefined, "user"), ctx);
	} finally {
		console.error = orig;
	}
	const line = lines.find((l) => l.includes("[pi-triage-router] route="));
	assert.ok(line, "a decision log line must be emitted");
	assert.match(line, /est_cost_in=[\d.]+/, "estimated input cost is labeled est_cost_in");
	assert.match(line, /est_cost_out=[\d.?]+/, "estimated output cost is labeled est_cost_out");
	assert.match(line, /last_cost=0\.12\(actual, previous request\)/, "actual previous cost is labeled actual");
	assert.match(line, /state_tokens=-\//, "no tokenizer available: state tokens unknown");
	assert.match(line, /est_in=\d+\(context\)/, "estimate source is visible");
});

// ---------------------------------------------------------------------------
// Audit 4. Context-window eligibility
// ---------------------------------------------------------------------------

test("models whose context window cannot hold the conversation are excluded", () => {
	const config = testConfig();
	const required = 130_000; // v4-pro and mimo (128K) cannot hold this
	const eligible = router.eligibleModels(config, registryFor(ALL_KEYS), required);
	const ids = eligible.map((m) => m.id);
	assert.ok(!ids.includes("deepseek/deepseek-v4-pro"), "128K window is too small");
	assert.ok(!ids.includes("xiaomi/mimo-v2.5"), "128K window is too small");
	assert.ok(ids.includes("deepseek/deepseek-v4-flash"), "1M window fits");
	assert.ok(ids.includes("qwen/qwen3.8-max"));
});

test("unknown windows never exclude; an all-too-small set keeps every model", () => {
	const config = testConfig();
	const noWindows = {
		find: () => ({ cost: { input: 0.01 }, contextWindow: undefined }),
	} as never;
	assert.equal(
		router.eligibleModels(config, noWindows, 1_000_000).length,
		6,
		"a missing window is never treated as a fabricated limit",
	);
	const tiny = {
		find: () => ({ cost: { input: 0.01 }, contextWindow: 100 }),
	} as never;
	const lines: string[] = [];
	const orig = console.error;
	console.error = (...args: unknown[]) => lines.push(args.join(" "));
	try {
		const kept = router.eligibleModels(config, tiny, 1_000_000);
		assert.equal(kept.length, 6, "when nothing fits, route anyway (observable)");
	} finally {
		console.error = orig;
	}
	assert.ok(lines.some((l) => l.includes("every eligible model is below")), "the fallback-to-all is logged");
});

test("route() excludes window-ineligible models and re-classifies when sticky no longer fits", async () => {
	const vm = captureRoute();
	// 120K conversation: v4-pro (128K) cannot hold it once reserve is added.
	const ctx = makeCtx({ tokens: 120_000, contextWindow: 1_000_000 }) as never;
	router.setLayaForTests(
		mockClassifier({
			model: { "litellm/z-ai/glm-5.3": 0.9 },
			complexity: { moderate: 0.9 },
			category: { coding: 0.9 },
		}),
	);
	const route = await vm.route(
		routeRequest([userMsg("Continue the refactor of the payments service.")], { model: "litellm/deepseek/deepseek-v4-pro", summary: "" }, "continuation"),
		ctx,
	);
	assert.equal(route.model.id, "z-ai/glm-5.3", "sticky v4-pro is window-ineligible: classify picks a fitting model");
});

test("a routed turn logs the context-window exclusion once, not twice", async () => {
	// Regression: handleRequest ran the window filter twice per turn - once
	// inside eligibleModels (logged) and once for the present-but-not-group
	// set (logged again) - producing two identical stacked
	// "context window: excluding ..." console lines on every oversized turn.
	const vm = captureRoute();
	const ctx = makeCtx({ tokens: 120_000, contextWindow: 1_000_000 }) as never;
	router.setLayaForTests(
		mockClassifier({
			model: { "litellm/z-ai/glm-5.3": 0.9 },
			complexity: { moderate: 0.9 },
			category: { coding: 0.9 },
		}),
	);
	const lines: string[] = [];
	const orig = console.error;
	console.error = (...args: unknown[]) => lines.push(args.join(" "));
	try {
		await vm.route(
			routeRequest(
				[userMsg("Continue the refactor of the payments service.")],
				{ model: "litellm/deepseek/deepseek-v4-pro", summary: "" },
				"continuation",
			),
			ctx,
		);
	} finally {
		console.error = orig;
	}
	const windowLines = lines.filter((l) => l.includes("context window:"));
	assert.equal(
		windowLines.length,
		1,
		`exactly one context-window diagnostic per turn (got ${windowLines.length}: ${JSON.stringify(windowLines)})`,
	);
	assert.match(
		windowLines[0],
		/excluding \d+ models? below ~\d+ tokens:/,
		"the surviving line keeps the exclusion diagnostic shape",
	);
	assert.match(windowLines[0], /deepseek-v4-pro/, "the excluded sticky model is named");
});

// 47-49. Criterion construction: descriptions only, no appended pricing
// ---------------------------------------------------------------------------

test("describe() returns the description verbatim with no pricing text", () => {
	const config = testConfig();
	for (const entry of config.models) {
		const criterion = router.describe(entry);
		assert.equal(criterion, entry.description);
		assert.ok(!criterion.includes("Pricing"), "no auto-appended price line");
		assert.ok(!/price|priced|budget model/i.test(criterion), "no price language in criteria");
	}
});

test("the model question's criteria are the short descriptions, never price lines", async () => {
	const seen: { model: { criteria?: Record<string, string> } }[] = [];
	const capturing = {
		systemOne: async (_state: unknown, questions: unknown) => {
			seen.push(questions as never);
			return {
				answers: {
					model: { probabilities: { "litellm/z-ai/glm-5.3": 0.9 } },
					complexity: { probabilities: { moderate: 0.8 } },
					category: { probabilities: { coding: 0.9 } },
				},
			};
		},
	} as never as router.Classifier;
	const config = testConfig();
	const ctx = makeCtx();
	const eligible = router.eligibleModels(config, ctx.modelRegistry);
	router.setLayaForTests(capturing);
	await router.classifyAndDecide(routeRequest([userMsg("Refactor the auth middleware.")], undefined), ctx, config, eligible);
	assert.ok(seen.length >= 1, "classifier was asked");
	const criteria = seen[0]?.model.criteria ?? {};
	const byDescription = new Map(config.models.map((m) => [`${m.provider}/${m.id}`, m.description]));
	for (const [key, text] of Object.entries(criteria)) {
		assert.equal(text, byDescription.get(key), `criterion for ${key} is exactly its description`);
		assert.ok(!text.includes("Pricing"), "no price line inside Laya's head");
	}
});

test("oversized descriptions are flagged so criteria stay inside Laya's head budget", async () => {
	const long = `${"word ".repeat(45)}end.`; // > 200 chars
	const flagged = router.validateConfig({
		models: [{ provider: "litellm", id: "deepseek/deepseek-v4-flash", description: long, priceIn: 0.01 }],
	});
	assert.ok(
		flagged.problems.some((p) => p.includes("one short sentence") && p.includes("chars")),
		"length guard reported",
	);
	assert.equal(flagged.models.length, 1, "problem is non-fatal: the entry is still used");
	const { readFileSync } = await import("node:fs");
	const shipped = router.validateConfig(JSON.parse(readFileSync(new URL("../triage-router.json", import.meta.url), "utf8")));
	assert.equal(shipped.problems.filter((p) => p.includes("short sentence")).length, 0, "shipped descriptions pass the guard");
});

test("shipped configs' default/override are provider/id composites of configured models", () => {
	// "default" and "override" are matched against composite(entry) = "provider/id".
	// A bare model id (e.g. "deepseek/deepseek-v4-flash") never matches, silently
	// downgrades every fallback to cheapest, and warns on each compaction.
	for (const file of ["triage-router.json", "triage-router.sample.json"]) {
		const cfg = router.validateConfig(
			JSON.parse(readFileSync(new URL(`../${file}`, import.meta.url), "utf8")),
		);
		for (const [field, value] of [
			["default", cfg.defaultModel],
			["override", cfg.override],
		] as const) {
			if (value == null) continue;
			const known = cfg.models.some((m) => `${m.provider}/${m.id}` === value);
			assert.ok(known, `${file} ${field} "${value}" matches provider/id of a configured model`);
		}
	}
});

test("benchmark cache and config paths live in the pi agent dir, PI_CODING_AGENT_DIR-aware", () => {
	const shipped = router.validateConfig(JSON.parse(readFileSync(new URL("../triage-router.json", import.meta.url), "utf8")));
	const pkgDir = dirname(fileURLToPath(new URL("../triage-router.json", import.meta.url)));
	assert.ok(
		!shipped.benchmarks.cachePath.startsWith(pkgDir),
		"default benchmark cache is outside the package dir (pi update wipes package contents)",
	);
	assert.ok(
		shipped.benchmarks.seedPath !== undefined && shipped.benchmarks.seedPath.startsWith(pkgDir),
		"bundled snapshot is the read fallback until the first refresh",
	);
	const saved = process.env.PI_CODING_AGENT_DIR;
	try {
		process.env.PI_CODING_AGENT_DIR = join(tmpdir(), "pi-agent-override-test");
		const overridden = router.validateConfig(
			JSON.parse(readFileSync(new URL("../triage-router.sample.json", import.meta.url), "utf8")),
		);
		assert.equal(
			overridden.benchmarks.cachePath,
			join(process.env.PI_CODING_AGENT_DIR, "triage-router.benchmarks.json"),
		);
		assert.ok(
			router.configCandidates().includes(join(process.env.PI_CODING_AGENT_DIR, "triage-router.json")),
			"user config candidate honors PI_CODING_AGENT_DIR",
		);
		assert.equal(piAgentDir(), process.env.PI_CODING_AGENT_DIR);
	} finally {
		if (saved === undefined) delete process.env.PI_CODING_AGENT_DIR;
		else process.env.PI_CODING_AGENT_DIR = saved;
	}
});

test("stuck local load (first-run download): request is bounded, deterministic fallback, TUI status explains", async () => {
	// A first laya/auto run downloads ~2 GB of ONNX weights from Hugging Face.
	// If that hangs (cold cache, blocked network), the request must NOT hang
	// with it: wait a bounded time, then route deterministically.
	router.setLayaForTests(new Promise<never>(() => {}) as never, Date.now() - 30_000);
	const config = testConfig();
	const ctx = makeCtx();
	const status: string[] = [];
	ctx.mode = "tui";
	ctx.hasUI = true;
	ctx.ui = { setStatus: (_key: string, text: string) => void status.push(text) };
	const eligible = router.eligibleModels(config, ctx.modelRegistry);
	const started = Date.now();
	const { decision, clf } = await router.classifyAndDecide(
		routeRequest([{ role: "user", content: "Inspect the failing deployment." }], undefined, "user"),
		ctx,
		config,
		eligible,
	);
	const elapsed = Date.now() - started;
	assert.ok(elapsed >= 1_500 && elapsed < 8_000, `bounded wait, took ${elapsed}ms`);
	assert.equal(clf.provider, "local");
	assert.equal(clf.ms, null, "no inference happened");
	assert.ok(decision.entry?.id, "deterministic decision still produced");
	assert.ok(
		status.some((s) => s.includes("laya:")),
		`TUI status explains the wait, got ${JSON.stringify(status)}`,
	);
});

test("stuck local load with failurePolicy=error: rejects instead of hanging", async () => {
	router.setLayaForTests(new Promise<never>(() => {}) as never, Date.now() - 60_000);
	const config = testConfig({ classifier: { failurePolicy: "error" } });
	const eligible = router.eligibleModels(config, makeCtx().modelRegistry);
	await assert.rejects(
		() =>
			router.classifyAndDecide(
				routeRequest([{ role: "user", content: "Inspect the failing deployment." }], undefined, "user"),
				makeCtx(),
				config,
				eligible,
			),
		/failurePolicy=error/,
	);
});
