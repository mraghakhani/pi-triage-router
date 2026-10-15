/**
 * Remote classifier backend tests (OpenRouter Decisions API / Jev).
 *
 * Everything is mocked: no network, no paid API call, no key required for
 * the standard run. Coverage: startup validation and backward compatibility,
 * structured-output validation, bounded retries, timeouts, credential errors,
 * failure policies (deterministic vs error), no silent provider switching,
 * credential-free logs, and separate classifier cost accounting.
 */
import { test, afterEach } from "node:test";
import assert from "node:assert/strict";
import * as router from "../triage-router.ts";
import {
	callDecisions,
	parseAnswers,
	parseUsage,
	RemoteClassifierError,
	setFetchForTests,
	setRuntimePathsForTests,
	DECISIONS_URL,
	type FetchLike,
} from "../triage-router-lib/triage-router-openrouter.ts";
import { testConfig, makeCtx, routeRequest, mockClassifier } from "./fixtures.ts";

// Credentials are looked up env-first, then Pi's auth file. These tests must
// observe the NO-CREDENTIALS state regardless of the developer machine, so
// both lookups are pointed at nothing.
const NO_AUTH = "/nonexistent/triage-router-auth.json";
function isolateCredentials(): () => void {
	const prevEnv = process.env.OPENROUTER_API_KEY;
	delete process.env.OPENROUTER_API_KEY;
	setRuntimePathsForTests({ authFile: NO_AUTH });
	return () => {
		setRuntimePathsForTests({ authFile: null });
		if (prevEnv !== undefined) process.env.OPENROUTER_API_KEY = prevEnv;
	};
}

afterEach(() => {
	setFetchForTests(null);
	router.resetLayaForTests();
	delete process.env.OPENROUTER_API_KEY;
	delete process.env.LAYA_TEST_KEY;
});

/** Capture console.error lines during `fn`, restoring the original sink after. */
async function captureErrors(fn: () => Promise<void>): Promise<string[]> {
	const lines: string[] = [];
	const original = console.error;
	console.error = (...args: unknown[]) => {
		lines.push(args.map((a) => (a instanceof Error ? `${a.name}: ${a.message}` : String(a))).join(" "));
	};
	try {
		await fn();
	} finally {
		console.error = original;
	}
	return lines;
}

interface MockCall {
	url: string;
	init?: RequestInit;
}

function responder(
	handler: (init: RequestInit | undefined, callIndex: number) => { ok: boolean; status: number; body: unknown } | Promise<{ ok: boolean; status: number; body: unknown }>,
	calls: MockCall[],
): FetchLike {
	return async (url, init) => {
		calls.push({ url, init });
		const result = await handler(init, calls.length - 1);
		return {
			ok: result.ok,
			status: result.status,
			json: async () => result.body,
			text: async () => JSON.stringify(result.body),
		};
	};
}

const ANSWERS = {
	model: { type: "choice", probabilities: { "litellm/z-ai/glm-5.3": 0.8, "litellm/deepseek/deepseek-v4-flash": 0.2 } },
	complexity: { type: "choice", probabilities: { moderate: 0.9, standard: 0.1, complex: 0 } },
	category: { type: "choice", probabilities: { coding: 0.9, question: 0.1, infra: 0, debugging: 0, reasoning: 0, research: 0, long_horizon: 0, budget: 0 } },
};

function decisionBody(usage = { input_tokens: 476, output_tokens: 70, cost: 0.000019992 }): unknown {
	return { model: "typesafe/jev-1.13-20260917", answers: ANSWERS, usage };
}

// ---------------------------------------------------------------------------
// Startup validation and backward compatibility
// ---------------------------------------------------------------------------

test("config without a classifier section keeps local Laya defaults (backward compatible)", () => {
	const config = router.validateConfig({
		models: [{ provider: "litellm", id: "deepseek/deepseek-v4-flash", description: "Simple work" }],
	});
	assert.equal(config.classifier.provider, "local");
	assert.equal(config.classifier.model, "typesafe/jev-1.13");
	assert.equal(config.classifier.failurePolicy, "deterministic");
	assert.equal(config.scoring.enabled, false, "scoring stays off unless explicitly enabled");
	assert.equal(config.benchmarks.enabled, true);
	assert.equal(config.problems.length, 0, `no problems expected: ${JSON.stringify(config.problems)}`);
});

test("legacy layaTimeoutMs feeds classifier.timeoutMs when the classifier section omits it", () => {
	const config = router.validateConfig({ models: [], layaTimeoutMs: 150 });
	assert.equal(config.classifier.timeoutMs, 150);
	const explicit = router.validateConfig({ models: [], layaTimeoutMs: 150, classifier: { timeoutMs: 900 } });
	assert.equal(explicit.classifier.timeoutMs, 900, "explicit classifier.timeoutMs wins");
});

test("unsupported classifier provider is a clear problem and falls back to local at load", () => {
	const config = router.validateConfig({ models: [], classifier: { provider: "anthropic" } });
	assert.ok(config.problems.some((p) => p.includes('"classifier.provider"')), "problem names the bad key");
	assert.equal(config.classifier.provider, "local");
});

test("openrouter provider without credentials: explicit non-fatal startup problem, provider unchanged", () => {
	const restore = isolateCredentials();
	try {
		const config = router.validateConfig({ models: [], classifier: { provider: "openrouter" } });
		assert.equal(config.classifier.provider, "openrouter", "a request never silently changes provider; load keeps the choice");
		assert.ok(
			config.problems.some((p) => p.includes("OPENROUTER_API_KEY") && p.includes("still starts")),
			"missing key is reported at startup, without blocking the extension",
		);
	} finally {
		restore();
	}
});

test("openrouter provider with credentials validates clean", () => {
	process.env.LAYA_TEST_KEY = "test-key-not-a-secret";
	const config = router.validateConfig({
		models: [],
		classifier: { provider: "openrouter", model: "typesafe/jev-1.13", apiKeyEnv: "LAYA_TEST_KEY", maxInputTokens: 2048 },
	});
	assert.equal(config.classifier.provider, "openrouter");
	assert.equal(config.classifier.maxInputTokens, 2048);
	assert.equal(config.problems.filter((p) => p.includes("apiKeyEnv") || p.includes("OPENROUTER")).length, 0);
});

test("classifier section validates every field with actionable problems", () => {
	const config = router.validateConfig({
		models: [],
		classifier: {
			provider: "local",
			timeoutMs: 10,
			maxInputTokens: 4,
			failurePolicy: "explode",
			proxyUrl: 42,
			localPrompt: "",
			apiKeyEnv: "",
			bogus: 1,
		},
	});
	for (const needle of ["timeoutMs", "maxInputTokens", "failurePolicy", "proxyUrl", "localPrompt", "apiKeyEnv", "bogus"]) {
		assert.ok(config.problems.some((p) => p.includes(needle)), `problem mentions ${needle}`);
	}
	assert.equal(config.classifier.timeoutMs, 5000, "invalid values fall back to defaults");
	assert.equal(config.classifier.failurePolicy, "deterministic");
});

test("classifier maxAttempts and retryDelayMs validate with defaults", () => {
	const bad = router.validateConfig({
		models: [],
		classifier: { provider: "local", maxAttempts: 0, retryDelayMs: -1 },
	});
	for (const needle of ["maxAttempts", "retryDelayMs"]) {
		assert.ok(bad.problems.some((p) => p.includes(needle)), `problem mentions ${needle}`);
	}
	assert.equal(bad.classifier.maxAttempts, 1, "invalid maxAttempts falls back to default");
	assert.equal(bad.classifier.retryDelayMs, 0, "invalid retryDelayMs falls back to default");

	const good = router.validateConfig({
		models: [],
		classifier: { provider: "local", maxAttempts: 4, retryDelayMs: 250 },
	});
	assert.equal(good.classifier.maxAttempts, 4);
	assert.equal(good.classifier.retryDelayMs, 250);
	assert.ok(!good.problems.some((p) => p.includes("maxAttempts") || p.includes("retryDelayMs")));
});

test("classifier maxAttempts defaults to 1 (no retries)", () => {
	const config = router.validateConfig({ models: [], classifier: { provider: "local" } });
	assert.equal(config.classifier.maxAttempts, 1, "default is 1");
});

// ---------------------------------------------------------------------------
// Transport: structured output, retries, timeouts, credentials
// ---------------------------------------------------------------------------

test("callDecisions returns validated probabilities and provider-reported usage", async () => {
	const calls: MockCall[] = [];
	const result = await callDecisions({
		model: "typesafe/jev-1.13",
		role: "router",
		prompt: "request text",
		questions: { model: { type: "choice", criteria: {} } },
		timeoutMs: 1000,
		apiKey: "sk-test",
		fetchImpl: responder(() => ({ ok: true, status: 200, body: decisionBody() }), calls),
	});
	assert.equal(calls.length, 1);
	assert.equal(calls[0].url, DECISIONS_URL);
	assert.equal(result.usage.inputTokens, 476);
	assert.equal(result.usage.cost, 0.000019992);
	assert.equal(result.servedModel, "typesafe/jev-1.13-20260917");
	assert.ok(result.answers.model?.probabilities?.["litellm/z-ai/glm-5.3"] === 0.8);
	assert.ok(typeof result.latencyMs === "number");
	const auth = (calls[0].init?.headers as Record<string, string>).Authorization;
	assert.equal(auth, "Bearer sk-test", "key sent in the header, never in the payload");
});

test("malformed payloads are rejected, never trusted", () => {
	assert.throws(() => parseAnswers({ no_answers: true }), RemoteClassifierError);
	assert.throws(() => parseAnswers({ answers: [] }), RemoteClassifierError);
	assert.throws(() => parseAnswers({ answers: {} }), RemoteClassifierError);
	const bad = parseAnswers({ answers: { model: { probabilities: { a: "high", b: Number.NaN, c: 42, d: 0.5 } } } });
	assert.deepEqual(bad.model?.probabilities, { d: 0.5 }, "non-numeric/out-of-range values are dropped");
	assert.deepEqual(parseUsage({}), { inputTokens: null, outputTokens: null, cost: null });
	assert.deepEqual(parseUsage({ input_tokens: -3, cost: "lots" }), { inputTokens: null, outputTokens: null, cost: null });
});

test("HTTP errors: 401 not retried and credential-free; 429 retried exactly once", async () => {
	const calls: MockCall[] = [];
	await assert.rejects(
		callDecisions({
			model: "m", role: "r", prompt: "p", questions: {}, timeoutMs: 500,
			apiKey: "test-key-not-a-secret",
			fetchImpl: responder(() => ({ ok: false, status: 401, body: { error: "bad key" } }), calls),
		}),
		(err: Error) => {
			assert.ok(err.message.includes("credentials"), "clear credential error");
			assert.ok(!err.message.includes("sk-or"), "key never appears in the error");
			return true;
		},
	);
	assert.equal(calls.length, 1, "401 is not retried");

	const retryCalls: MockCall[] = [];
	const result = await callDecisions({
		model: "m", role: "r", prompt: "p", questions: {}, timeoutMs: 500, apiKey: "k", maxAttempts: 2,
		fetchImpl: responder((_init, i) => (
			i === 0 ? { ok: false, status: 429, body: {} } : { ok: true, status: 200, body: decisionBody() }
		), retryCalls),
	});
	assert.equal(retryCalls.length, 2, "one bounded retry on rate limit");
	assert.ok(result.answers.model);
});

test("attempts are bounded: repeated 5xx stops after maxAttempts", async () => {
	const calls: MockCall[] = [];
	await assert.rejects(
		callDecisions({
			model: "m", role: "r", prompt: "p", questions: {}, timeoutMs: 500, apiKey: "k", maxAttempts: 2,
			fetchImpl: responder(() => ({ ok: false, status: 503, body: {} }), calls),
		}),
		/HTTP 503/,
	);
	assert.equal(calls.length, 2);
});

test("retryDelayMs waits between attempts, 0 keeps the old instant retry", async () => {
	const calls: MockCall[] = [];
	const stamps: number[] = [];
	const result = await callDecisions({
		model: "m", role: "r", prompt: "p", questions: {}, timeoutMs: 500, apiKey: "k",
		maxAttempts: 2,
		retryDelayMs: 60,
		fetchImpl: responder((_init, i) => {
			stamps.push(Date.now());
			return i === 0 ? { ok: false, status: 429, body: {} } : { ok: true, status: 200, body: decisionBody() };
		}, calls),
	});
	assert.equal(calls.length, 2, "one bounded retry on rate limit");
	assert.ok(stamps[1] - stamps[0] >= 60, "delay applied between attempts");
	assert.ok(result.answers.model);

	const instant: MockCall[] = [];
	const instantStamps: number[] = [];
	await callDecisions({
		model: "m", role: "r", prompt: "p", questions: {}, timeoutMs: 500, apiKey: "k", maxAttempts: 2,
		fetchImpl: responder((_init, i) => {
			instantStamps.push(Date.now());
			return i === 0 ? { ok: false, status: 429, body: {} } : { ok: true, status: 200, body: decisionBody() };
		}, instant),
	});
	assert.ok(instantStamps[1] - instantStamps[0] < 60, "default 0ms keeps instant retry");
});

test("timeouts abort the request and report the budget", async () => {
	const calls: MockCall[] = [];
	const hanging: FetchLike = (_url, init) => {
		calls.push({ url: _url, init });
		return new Promise((_resolve, reject) => {
			init?.signal?.addEventListener("abort", () => {
				const err = new Error("This operation was aborted");
				err.name = "AbortError";
				reject(err);
			});
		});
	};
	await assert.rejects(
		callDecisions({ model: "m", role: "r", prompt: "p", questions: {}, timeoutMs: 40, apiKey: "k", maxAttempts: 2, fetchImpl: hanging }),
		/timed out after 40ms/,
	);
	assert.equal(calls.length, 2, "timeout retried once within the attempt cap");
});

// ---------------------------------------------------------------------------
// End-to-end: classifyAndDecide over the remote backend
// ---------------------------------------------------------------------------

function remoteConfig(overrides: Record<string, unknown> = {}) {
	return testConfig({ classifier: { provider: "openrouter", proxyUrl: null, ...overrides } });
}

test("classifyAndDecide over the remote backend validates answers and reports actual classifier cost", async () => {
	process.env.OPENROUTER_API_KEY = "test-key-not-a-secret";
	setFetchForTests(responder(() => ({ ok: true, status: 200, body: decisionBody() }), []));
	const config = remoteConfig();
	const ctx = makeCtx();
	const eligible = router.eligibleModels(config, ctx.modelRegistry);
	const logs = await captureErrors(async () => {
		const { decision, clf } = await router.classifyAndDecide(
			routeRequest([{ role: "user", content: "Refactor the billing service into two files." }], undefined) as never,
			ctx as never,
			config as never,
			eligible,
		);
		assert.equal(decision.entry.id, "z-ai/glm-5.3", "remote probabilities flow through the shared decide()");
		assert.equal(decision.category, "coding");
		assert.equal(clf.provider, "openrouter");
		assert.equal(clf.model, "typesafe/jev-1.13");
		assert.equal(clf.cost, 0.000019992, "provider-reported actual cost");
		assert.equal(clf.costKind, "actual");
		assert.equal(clf.inputTokens, 476);
		assert.ok((clf.ms ?? -1) >= 0);
	});
	assert.equal(logs.filter((l) => l.includes("test-key")).length, 0, "key never logged");
});

test("remote failure follows the deterministic policy: no provider switch, no silent local classification", async () => {
	process.env.OPENROUTER_API_KEY = "test-key-not-a-secret";
	setFetchForTests(responder(() => {
		throw new Error("tunnel down");
	}, []));
	const config = remoteConfig();
	const ctx = makeCtx();
	const eligible = router.eligibleModels(config, ctx.modelRegistry);
	const logs = await captureErrors(async () => {
		const { decision, clf } = await router.classifyAndDecide(
			routeRequest([{ role: "user", content: "Fix the flaky test." }], undefined) as never,
			ctx as never,
			config as never,
			eligible,
		);
		assert.equal(decision.rule, "fallback", "deterministic policy decides without classification");
		assert.equal(decision.entry.id, "deepseek/deepseek-v4-flash", "fallback = cheapest (shipped policy)");
		assert.equal(clf.provider, "openrouter", "backend identity is still the configured one");
		assert.equal(clf.cost, null, "failed call reports no cost");
		assert.equal(clf.costKind, null);
	});
	assert.ok(logs.some((l) => l.includes("classifier openrouter/typesafe/jev-1.13 failed")), "failure is loud and specific");
	assert.ok(logs.some((l) => l.includes("deterministic policy decides")));
	assert.equal(logs.filter((l) => l.includes("test-key")).length, 0, "key never logged");
});

test("failurePolicy=error surfaces the routing error instead of degrading", async () => {
	process.env.OPENROUTER_API_KEY = "test-key-not-a-secret";
	setFetchForTests(responder(() => {
		throw new Error("tunnel down");
	}, []));
	const config = remoteConfig({ failurePolicy: "error" });
	const ctx = makeCtx();
	const eligible = router.eligibleModels(config, ctx.modelRegistry);
	await assert.rejects(
		router.classifyAndDecide(
			routeRequest([{ role: "user", content: "Fix the flaky test." }], undefined) as never,
			ctx as never,
			config as never,
			eligible,
		),
		/failurePolicy=error/,
	);
});

test("missing credentials follow the failure policy with a clear message", async () => {
	const restore = isolateCredentials();
	try {
		delete process.env.OPENROUTER_API_KEY;
	setFetchForTests(responder(() => ({ ok: true, status: 200, body: decisionBody() }), []));
	const config = remoteConfig({ failurePolicy: "error" });
	const ctx = makeCtx();
	const eligible = router.eligibleModels(config, ctx.modelRegistry);
	await assert.rejects(
		router.classifyAndDecide(
			routeRequest([{ role: "user", content: "Fix the flaky test." }], undefined) as never,
			ctx as never,
			config as never,
			eligible,
		),
		/OPENROUTER_API_KEY/,
	);
	// fetch must never be reached without a key
	setFetchForTests(responder(() => {
		throw new Error("must not be called");
	}, []));
	const config2 = remoteConfig(); // deterministic policy by default
	const { decision, clf } = await router.classifyAndDecide(
		routeRequest([{ role: "user", content: "Fix the flaky test." }], undefined) as never,
		ctx as never,
		config2 as never,
		eligible,
	);
	assert.equal(decision.rule, "fallback");
	assert.equal(clf.costKind, null);
	} finally {
		restore();
	}
});

test("remote state is bounded by classifier.maxInputTokens", async () => {
	const config = remoteConfig({ maxInputTokens: 256 });
	const handle = await router.getClassifier(config);
	assert.equal(handle.kind, "openrouter");
	assert.ok(handle.classifier);
	assert.equal(router.stateRoomTokens(handle.classifier), 256, "room = configured state budget");
	const big = "x".repeat(20_000);
	const input = router.buildRoutingInput(
		[{ role: "user", content: big }] as never,
		"",
		null,
		config,
		handle.classifier,
		config.classifier.remotePrompt,
	);
	// Conservative 3 chars/token allocation minus role margin: 232 * 3 = 696 chars.
	assert.ok(input.prompt.length <= 760, `remote prompt must fit the room (got ${input.prompt.length})`);
	assert.ok(input.prompt.includes("[current request truncated]"), "explicit truncation marker");
});

test("a Jev-sized room keeps conversation context alongside a long request", async () => {
	// Regression: jev.maxInputTokens shipped at 512 (Laya's scale), but Jev's
	// real limit is 32k for state+question. A long request at 512 tokens eats
	// the whole room and starves the conversation summary out of the state.
	const big = "REQMARKER database connection pool exhausted during failover.\n".repeat(120); // ~7.4k chars
	const messages = [
		{ role: "assistant", content: "Prior turn: we rolled back the k8s deployment." },
		{ role: "user", content: big },
	] as never;
	const summary = "CTXMARKER billing service moved to the new queue.";

	// The old 512-token budget: long request consumes the room, no context left.
	const small = remoteConfig({ maxInputTokens: 512 });
	const smallHandle = await router.getClassifier(small);
	const smallInput = router.buildRoutingInput(
		messages, "", summary, small, smallHandle.classifier, small.classifier.remotePrompt,
	);
	assert.equal(smallInput.summary, "", "512-token room: a long request evicts all context");

	// The shipped jev room (28672 = Jev's 32k state+question minus question headroom).
	const large = remoteConfig({ maxInputTokens: 28672 });
	const largeHandle = await router.getClassifier(large);
	const largeInput = router.buildRoutingInput(
		messages, "", summary, large, largeHandle.classifier, large.classifier.remotePrompt,
	);
	assert.ok(largeInput.summary.includes("CTXMARKER"), "summary retained");
	assert.ok(largeInput.prompt.includes("CTXMARKER"), "summary reaches the state");
	assert.ok(
		largeInput.currentRequest.length > 4000,
		`most of the request kept too (got ${largeInput.currentRequest.length})`,
	);
});

test("local backend stays the default path: getClassifier returns the Laya handle", async () => {
	const config = testConfig();
	const fake = mockClassifier({ model: {}, complexity: {}, category: {} });
	router.setLayaForTests(fake);
	const handle = await router.getClassifier(config);
	assert.equal(handle.kind, "local");
	assert.equal(handle.model, "laya");
	assert.equal(handle.classifier, fake, "local path uses the Laya singleton, never the remote client");
	assert.equal(handle.lastCall, null, "local runs report no remote usage");
});
