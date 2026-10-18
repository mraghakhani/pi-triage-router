/**
 * Two-entry design tests: the jev section, backend isolation between the
 * laya/auto and jev/auto virtual models, endpoint/proxy/credential
 * resolution, and sample-config validity.
 *
 * Everything is mocked: no network, no paid API call, no key required for
 * the standard run.
 */
import { test, after, afterEach } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import * as router from "../triage-router.ts";
import {
	classifierEndpoint,
	DECISIONS_URL,
	isLoopbackUrl,
	resolveApiKey,
	resolveBaseUrl,
	resolveProxyUrl,
	hasApiKey,
	setFetchForTests,
	setRuntimePathsForTests,
} from "../triage-router-lib/triage-router-openrouter.ts";
import { captureModels } from "./capture.ts";
import { testConfig, makeCtx, routeRequest, mockClassifier } from "./fixtures.ts";

const TMP = mkdtempSync(join(tmpdir(), "laya-jev-test-"));
const NO_AUTH = join(TMP, "missing-auth.json");
const AUTH_FILE = join(TMP, "auth.json");
writeFileSync(
	AUTH_FILE,
	JSON.stringify({
		litellm: { key: "tmp-litellm-key", baseUrl: "https://litellm.example.com/v1" },
		openrouter: { key: "tmp-or-key" },
	}),
);

function withSettings(httpProxy: string | null): string {
	const path = join(TMP, `settings-${Math.random().toString(36).slice(2)}.json`);
	writeFileSync(path, JSON.stringify(httpProxy === null ? {} : { httpProxy }));
	setRuntimePathsForTests({ settingsPath: path });
	return path;
}

// resolveProxyUrl reads all four of these; clear every one so no ambient
// proxy (always set on machines that route through one) leaks into an
// assertion that expects "no proxy configured".
const PROXY_ENV_KEYS = ["https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY"] as const;
function clearProxyEnv(): void {
	for (const key of PROXY_ENV_KEYS) delete process.env[key];
}

after(() => rmSync(TMP, { recursive: true, force: true }));

afterEach(() => {
	setRuntimePathsForTests({ settingsPath: null, authFile: null });
	setFetchForTests(null);
	router.resetLayaForTests();
	router.setConfigForTests(null);
	delete process.env.OPENROUTER_API_KEY;
	delete process.env.LITELLM_API_KEY;
	clearProxyEnv();
});

// ---------------------------------------------------------------------------
// Endpoint composition
// ---------------------------------------------------------------------------

test("classifierEndpoint: defaults per protocol, base URL appends the protocol path", () => {
	assert.equal(classifierEndpoint("decisions", null), DECISIONS_URL);
	assert.equal(classifierEndpoint("systemone", null), "https://openrouter.ai/api/v1/systemone");
	assert.equal(
		classifierEndpoint("systemone", "https://litellm.example.com/v1"),
		"https://litellm.example.com/v1/systemone",
	);
	assert.equal(
		classifierEndpoint("decisions", "https://litellm.example/base/"),
		"https://litellm.example/base/api/alpha/decisions",
		"trailing slashes are stripped before the path is appended",
	);
});

test("proxy resolution: settings.json httpProxy is used; \"direct\" opts out", () => {
	withSettings("http://127.0.0.1:10808");
	assert.equal(resolveProxyUrl(null), "http://127.0.0.1:10808");
	assert.equal(resolveProxyUrl("direct"), null, "direct = never proxy, even with settings set");
	assert.equal(resolveProxyUrl("http://explicit:1"), "http://explicit:1", "explicit config wins");
	withSettings(null);
	clearProxyEnv();
	assert.equal(resolveProxyUrl(null), null);
});

test("loopback targets are recognized for proxy bypass", () => {
	assert.ok(isLoopbackUrl("http://localhost:4000/v1"));
	assert.ok(isLoopbackUrl("http://127.0.0.1:8080/x"));
	assert.ok(!isLoopbackUrl("https://litellm.example.com/v1"));
});

// ---------------------------------------------------------------------------
// Credential resolution (env first, Pi auth file second)
// ---------------------------------------------------------------------------

test("resolveApiKey: env wins, auth file fills in, missing stays null", () => {
	setRuntimePathsForTests({ authFile: AUTH_FILE });
	assert.equal(resolveApiKey("LITELLM_API_KEY", "litellm", NO_AUTH), null);
	assert.equal(resolveApiKey("LITELLM_API_KEY", "litellm", AUTH_FILE), "tmp-litellm-key");
	process.env.LITELLM_API_KEY = "env-key";
	assert.equal(resolveApiKey("LITELLM_API_KEY", "litellm", AUTH_FILE), "env-key", "env beats auth file");
	delete process.env.LITELLM_API_KEY;
	assert.ok(hasApiKey("LITELLM_API_KEY", "litellm", AUTH_FILE));
	assert.ok(!hasApiKey("LITELLM_API_KEY", "litellm", NO_AUTH));
	// Regression: a litellm section must not pick up OPENROUTER_API_KEY and
	// leak that credential to the LiteLLM proxy. Only the env var named by
	// apiKeyEnv (the litellm default here) is consulted.
	process.env.OPENROUTER_API_KEY = "openrouter-key-not-a-secret";
	assert.equal(
		resolveApiKey("LITELLM_API_KEY", "litellm", AUTH_FILE),
		"tmp-litellm-key",
		"OPENROUTER_API_KEY in the environment must not shadow the litellm auth entry",
	);
	assert.equal(resolveApiKey("LITELLM_API_KEY", "litellm", NO_AUTH), null);
	delete process.env.OPENROUTER_API_KEY;
});

test("resolveBaseUrl: entry.baseUrl wins, env.<NAME>_BASE_URL falls back, missing stays null", () => {
	const FILE = join(TMP, "baseurl-auth.json");
	writeFileSync(
		FILE,
		JSON.stringify({
			litellm: { key: "k", baseUrl: "https://explicit.example.com/v1" },
			openrouter: { key: "k", env: { LITELLM_BASE_URL: "https://envvar.example.com" } },
			openai: { key: "k" },
		}),
	);
	assert.equal(resolveBaseUrl("litellm", FILE), "https://explicit.example.com/v1");
	assert.equal(resolveBaseUrl("openrouter", FILE), "https://envvar.example.com", "env-shaped entries are honored");
	assert.equal(resolveBaseUrl("openai", FILE), null, "no base URL = OpenRouter defaults");
	assert.equal(resolveBaseUrl("nonexistent", FILE), null);
	assert.equal(resolveBaseUrl("litellm", NO_AUTH), null, "missing auth file = OpenRouter defaults");
});

// ---------------------------------------------------------------------------
// jev section validation
// ---------------------------------------------------------------------------

test("jev section: absent = null (no virtual model), present = remote defaults", () => {
	const without = router.validateConfig({ models: [] });
	assert.equal(without.jev, null, "existing configs keep their behavior: no jev entry");
	const withDefault = router.validateConfig({ models: [], jev: {} });
	assert.ok(withDefault.jev, "an empty jev object is enough to register the entry");
	assert.equal(withDefault.jev?.provider, "openrouter");
	assert.equal(withDefault.jev?.protocol, "decisions");
	assert.equal(withDefault.jev?.model, "typesafe/jev-1.13");
});

test("jev section: litellm accepted, baseUrl is not a config key; provider \"local\" rejected; unknown keys reported", () => {
	const config = router.validateConfig({
		models: [],
		jev: { provider: "litellm", model: "typesafe/jev-1.13", protocol: "systemone", baseUrl: "https://l/v1", bogus: 1 },
	});
	assert.equal(config.jev?.provider, "litellm");
	assert.ok(config.problems.some((p) => p.includes('"jev": unknown key "baseUrl"')), "base URLs live in the auth file, never in config");
	assert.ok(config.problems.some((p) => p.includes('"jev": unknown key "bogus"')));

	const local = router.validateConfig({ models: [], jev: { provider: "local" } });
	assert.ok(
		local.problems.some((p) => p.includes('"jev.provider"')),
		"the jev entry is the remote backend; local stays with laya/auto",
	);
	assert.equal(local.jev?.provider, "openrouter", "kept on the remote default, never switched to local");
});

test("jev section without credentials: explicit non-fatal startup problem mentioning jev", () => {
	setRuntimePathsForTests({ authFile: NO_AUTH });
	const config = router.validateConfig({ models: [], jev: { provider: "litellm" } });
	assert.ok(
		config.problems.some((p) => p.startsWith("jev provider") && p.includes("still starts")),
		`expected a jev credentials problem, got: ${JSON.stringify(config.problems)}`,
	);
	// With the auth file present, no credentials problem for jev.
	setRuntimePathsForTests({ authFile: AUTH_FILE });
	const okConfig = router.validateConfig({ models: [], jev: { provider: "litellm" } });
	assert.ok(!okConfig.problems.some((p) => p.startsWith("jev provider")));
});

test("jev credential env follows the provider; explicit apiKeyEnv wins", () => {
	const litellm = router.validateConfig({ models: [], jev: { provider: "litellm" } });
	assert.equal(litellm.jev?.apiKeyEnv, "LITELLM_API_KEY", "a litellm jev section never inherits OPENROUTER_API_KEY");
	const explicit = router.validateConfig({ models: [], jev: { provider: "litellm", apiKeyEnv: "MY_JEV_KEY" } });
	assert.equal(explicit.jev?.apiKeyEnv, "MY_JEV_KEY", "an explicit apiKeyEnv always wins");
	const openrouter = router.validateConfig({ models: [], jev: {} });
	assert.equal(openrouter.jev?.apiKeyEnv, "OPENROUTER_API_KEY", "openrouter default unchanged");
});

// ---------------------------------------------------------------------------
// Backend plumbing: getClassifier + classifyAndDecide + handleRequest isolation
// ---------------------------------------------------------------------------

const JEV_BODY = {
	answers: {
		model: {
			type: "choice",
			probabilities: { "litellm/qwen/qwen3.8-max": 0.85, "litellm/deepseek/deepseek-v4-flash": 0.15 },
		},
		complexity: { type: "choice", probabilities: { complex: 0.9, standard: 0.1 } },
		category: { type: "choice", probabilities: { coding: 0.8, question: 0.2 } },
	},
	usage: { input_tokens: 300, output_tokens: 25, cost: 0.00004 },
};

function jevConfig(overrides: Record<string, unknown> = {}) {
	return testConfig({
		jev: {
			provider: "litellm",
			model: "typesafe/jev-1.13",
			protocol: "systemone",
			apiKeyEnv: "LITELLM_API_KEY",
			...overrides,
		},
	});
}

test("getClassifier(config, \"jev\"): litellm handle, systemone endpoint, auth-file key, cost captured", async () => {
	setRuntimePathsForTests({ authFile: AUTH_FILE });
	const config = jevConfig();
	const handle = await router.getClassifier(config, "jev");
	assert.equal(handle.kind, "litellm");
	assert.equal(handle.model, "typesafe/jev-1.13");

	const seen: { url: string; auth: string | undefined }[] = [];
	setFetchForTests(async (url, init) => {
		seen.push({ url: String(url), auth: (init as { headers?: Record<string, string> })?.headers?.Authorization });
		return { ok: true, status: 200, json: async () => JEV_BODY } as never;
	});
	await handle.classifier?.systemOne({ role: "r", prompt: "p" }, {});
	assert.equal(seen[0].url, "https://litellm.example.com/v1/systemone", "endpoint resolved from the auth file's litellm entry");
	assert.equal(seen[0].auth, "Bearer tmp-litellm-key", "auth file key, never logged");
	assert.equal(handle.lastCall?.cost, 0.00004, "provider-reported actual cost captured");
});

test("classifyAndDecide with backend \"jev\" routes on remote answers; clf stats say litellm", async () => {
	setRuntimePathsForTests({ authFile: AUTH_FILE });
	setFetchForTests(async () => ({ ok: true, status: 200, json: async () => JEV_BODY }) as never);
	const config = jevConfig();
	const ctx = makeCtx();
	const eligible = router.eligibleModels(config, ctx.modelRegistry);
	const { decision, clf } = await router.classifyAndDecide(
		routeRequest([{ role: "user", content: "Implement the feature." }], undefined, "user") as never,
		ctx as never,
		config as never,
		eligible,
		"jev",
	);
	assert.equal(clf.provider, "litellm");
	assert.equal(clf.costKind, "actual");
	assert.equal(clf.inputTokens, 300);
	assert.equal(decision.entry.provider + "/" + decision.entry.id, "litellm/qwen/qwen3.8-max");
});

test("backend isolation: laya/auto never fetches; jev/auto never loads local Laya", async () => {
	setRuntimePathsForTests({ authFile: AUTH_FILE });
	let fetchCount = 0;
	setFetchForTests(async () => {
		fetchCount++;
		return { ok: true, status: 200, json: async () => JEV_BODY } as never;
	});
	let layaCalls = 0;
	const base = mockClassifier({
		model: { "litellm/deepseek/deepseek-v4-flash": 1 },
		complexity: { standard: 1 },
		category: { coding: 1 },
	});
	router.setLayaForTests({
		config: base.config,
		systemOne: async (input, questions) => {
			layaCalls++;
			return base.systemOne(input, questions);
		},
	});
	const config = jevConfig();
	const ctx = makeCtx();
	const request = routeRequest([{ role: "user", content: "Fix the flaky test." }], undefined, "user");
	const eligible = router.eligibleModels(config, ctx.modelRegistry);

	const local = await router.classifyAndDecide(request as never, ctx as never, config as never, eligible);
	assert.equal(local.clf.provider, "local");
	assert.equal(fetchCount, 0, "the laya/auto entry never touches the network");
	assert.equal(layaCalls, 1);

	const remote = await router.classifyAndDecide(
		request as never,
		ctx as never,
		config as never,
		eligible,
		"jev",
	);
	assert.equal(remote.clf.provider, "litellm");
	assert.equal(fetchCount, 1, "the jev/auto entry uses exactly one remote call");
	assert.equal(layaCalls, 1, "the local classifier was not consulted for the jev entry");
	assert.notEqual(remote.decision.entry.id, local.decision.entry.id, "different classifiers, independent decisions");
});

test("virtual model registration: jev/auto appears only with a jev section", () => {
	router.setConfigForTests(testConfig());
	assert.deepEqual(
		captureModels().map((m) => `${m.provider}/${m.id}`),
		["laya/auto"],
	);
	router.setConfigForTests(jevConfig());
	assert.deepEqual(
		captureModels().map((m) => `${m.provider}/${m.id}`),
		["laya/auto", "jev/auto"],
	);
});

test("handleRequest: both entries share the pipeline; sticky/override rules work for jev too", async () => {
	// No classifier call may happen on the override path: any fetch is a bug.
	setFetchForTests(async () => {
		throw new Error("override path must not classify");
	});
	const config = jevConfig();
	const ctx = makeCtx();
	const request = routeRequest([{ role: "user", content: "Anything." }], undefined, "user");
	const override = router.validateConfig({
		...(config as unknown as Record<string, unknown>),
		models: config.models.map((m) => ({ ...m })),
		override: "deepseek/deepseek-v4-flash",
	});
	const route = await router.handleRequest(request as never, ctx as never, override as never, "jev");
	assert.equal(route.model.id, "deepseek/deepseek-v4-flash", "override applies to the jev entry as well (no classifier call)");
});

// ---------------------------------------------------------------------------
// Sample config is loadable and clean
// ---------------------------------------------------------------------------

test("triage-router.sample.json validates with zero problems and both entries", async () => {
	const { readFileSync } = await import("node:fs");
	const raw = JSON.parse(readFileSync(new URL("../triage-router.sample.json", import.meta.url), "utf8"));
	const config = router.validateConfig(raw);
	assert.deepEqual(config.problems, [], "the sample must load cleanly");
	assert.equal(config.classifier.provider, "local");
	assert.ok(config.jev, "the sample documents the jev entry");
});

// TUI presentation: the selection goes to the status bar; stderr stays clean
// (console spam corrupts the input line). TRIAGE_ROUTER_LOG=1 forces console.
test("TUI: selection shown via ctx.ui.setStatus, stderr clean; TRIAGE_ROUTER_LOG restores console", async () => {
	setRuntimePathsForTests({ authFile: AUTH_FILE });
	setFetchForTests(async () => ({ ok: true, status: 200, json: async () => JEV_BODY }) as never);
	const config = jevConfig();
	const status: Array<{ key: string; text: string }> = [];
	const ctx = makeCtx() as unknown as Record<string, unknown>;
	ctx.mode = "tui";
	ctx.hasUI = true;
	ctx.ui = { setStatus: (key: string, text: string) => void status.push({ key, text }) };
	const eligible = router.eligibleModels(config, ctx.modelRegistry as never);
	const request = routeRequest(
		[{ role: "user", content: "Implement the feature." }],
		undefined,
		"user",
	);

	const origError = console.error;
	const lines: string[] = [];
	console.error = (...args: unknown[]) => void lines.push(args.map(String).join(" "));
	delete process.env.TRIAGE_ROUTER_LOG;
	try {
		await router.handleRequest(request as never, ctx as never, config as never, "jev");
	} finally {
		console.error = origError;
	}
	assert.equal(lines.length, 0, `TUI stderr should stay clean, got: ${lines.join(" | ")}`);
	const selection = status.find(
		(s) => s.key === "laya-route" && s.text.includes("qwen/qwen3.8-max"),
	);
	assert.ok(selection, `status bar should show the routed model, got ${JSON.stringify(status)}`);
	assert.ok(selection.text.startsWith("→ "), "status text is a visible selection line");

	process.env.TRIAGE_ROUTER_LOG = "1";
	const forced: string[] = [];
	console.error = (...args: unknown[]) => void forced.push(args.map(String).join(" "));
	try {
		await router.handleRequest(request as never, ctx as never, config as never, "jev");
	} finally {
		console.error = origError;
		delete process.env.TRIAGE_ROUTER_LOG;
	}
	assert.ok(
		forced.some((l) => l.includes("clf=litellm")),
		`TRIAGE_ROUTER_LOG=1 should force console output, got: ${forced.slice(0, 3).join(" | ")}`,
	);
});
