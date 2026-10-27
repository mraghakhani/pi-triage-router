/**
 * Remote classifier validation (jev backend): credentials + model-existence
 * checks with deny/warn levels, endpoint construction, list caching, and the
 * endpoint probe that settles ids the model list omits (LiteLLM team filters,
 * OpenRouter decisions models). Stubs distinguish GET (list) from POST
 * (probe) by method.
 */
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { setFetchForTests, setRuntimePathsForTests, type FetchLike } from "../triage-router-lib/triage-router-openrouter.ts";
import { checkRemoteClassifier, modelsUrlFor, remoteProxyUrl, resetValidationCacheForTests } from "../triage-router-lib/triage-router-validation.ts";

const authDir = mkdtempSync(join(tmpdir(), "triage-validation-"));
const authPath = join(authDir, "auth.json");
// apiKeyEnv deliberately points at a name that never exists in this process:
// key resolution must come from the temp auth file below.
const apiKeyEnv = "TRIAGE_JEV_TEST_KEY_ENV";
const section = {
	provider: "openrouter",
	model: "typesafe/jev-1.13",
	apiKeyEnv,
	authFile: authPath,
	proxyUrl: null as string | null,
};

function jsonResponse(payload: unknown, status = 200): ReturnType<FetchLike> {
	return Promise.resolve({
		ok: status >= 200 && status < 300,
		status,
		json: () => Promise.resolve(payload),
		text: () => Promise.resolve(JSON.stringify(payload)),
	});
}

/** Valid minimal scoring payload the endpoint probe accepts. */
const probeAnswers = {
	answers: {
		model: { probabilities: { any: 1 } },
		complexity: { probabilities: { any: 1 } },
		category: { probabilities: { any: 1 } },
	},
	usage: { input_tokens: 1, output_tokens: 1, cost: 0 },
};

/** Stub that answers the model-list GET and the scoring-endpoint POST
 * (probe) differently. listIds without the probed id triggers a probe. */
function listAndProbeStub(listIds: string[], probe: { status: number; payload: unknown } = { status: 200, payload: probeAnswers }) {
	return async (url: string, init?: { method?: string }) => {
		if (init?.method === "POST") return jsonResponse(probe.payload, probe.status);
		return jsonResponse({ data: listIds.map((id) => ({ id })) });
	};
}

test.afterEach(() => {
	setFetchForTests(null);
	resetValidationCacheForTests();
	// The proxy-resolution test points settings.json at a temp file; restore
	// the ambient lookup so later tests are unaffected.
	setRuntimePathsForTests({ settingsPath: null });
});

test.after(() => {
	rmSync(authDir, { recursive: true, force: true });
});

test("modelsUrlFor builds provider model-list endpoints", () => {
	assert.equal(modelsUrlFor("openrouter", null), "https://openrouter.ai/api/v1/models");
	assert.equal(modelsUrlFor("openrouter", "https://example.com"), "https://openrouter.ai/api/v1/models", "openrouter ignores a config base");
	assert.equal(modelsUrlFor("litellm", null), null, "litellm without a baseUrl cannot be listed");
	assert.equal(modelsUrlFor("litellm", "http://127.0.0.1:4000"), "http://127.0.0.1:4000/v1/models");
	assert.equal(modelsUrlFor("litellm", "http://127.0.0.1:4000/"), "http://127.0.0.1:4000/v1/models", "trailing slash trimmed");
	assert.equal(modelsUrlFor("litellm", "http://127.0.0.1:4000/v1"), "http://127.0.0.1:4000/v1/models", "no double /v1");
	assert.equal(modelsUrlFor("openai", "https://x"), null, "unknown provider has no list endpoint");
});

test("missing credentials deny the edit without any network call", async () => {
	// Auth file has a litellm entry but not openrouter; env var unset.
	writeFileSync(authPath, JSON.stringify({ litellm: { key: "k" } }));
	let calls = 0;
	setFetchForTests(async () => {
		calls++;
		return jsonResponse({});
	});
	const result = await checkRemoteClassifier(section, "jev");
	assert.equal(result.level, "problem");
	assert.match(result.message, /no openrouter credentials found/);
	assert.match(result.message, new RegExp(apiKeyEnv));
	assert.match(result.message, /auth file/);
	assert.equal(calls, 0, "credentials are checked before any fetch");
});

test("known model on openrouter verifies (ok)", async () => {
	writeFileSync(authPath, JSON.stringify({ openrouter: { key: "k" } }));
	const urls: string[] = [];
	setFetchForTests(async (url) => {
		urls.push(url);
		return jsonResponse({ data: [{ id: "typesafe/jev-1.13" }, { id: "other/model" }] });
	});
	const result = await checkRemoteClassifier(section, "jev");
	assert.equal(result.level, "ok");
	assert.match(result.message, /openrouter\/typesafe\/jev-1\.13 verified/);
	assert.deepEqual(urls, ["https://openrouter.ai/api/v1/models"]);
});

test("unknown model on openrouter denies the edit (probe says does not exist)", async () => {
	writeFileSync(authPath, JSON.stringify({ openrouter: { key: "k" } }));
	setFetchForTests(
		listAndProbeStub(["other/model"], {
			status: 400,
			payload: { error: { message: "Model typesafe/jev-1.14 does not exist", code: 400 } },
		}),
	);
	const result = await checkRemoteClassifier({ ...section, model: "typesafe/jev-1.14" }, "jev");
	assert.equal(result.level, "problem");
	assert.match(result.message, /model "typesafe\/jev-1\.14" not found on openrouter/);
});

test("model absent from the list but served by the endpoint verifies (probe ok)", async () => {
	// OpenRouter's public catalog omits decision models entirely; the probe
	// settles them. A tiny real inference call - acceptable at menu/startup.
	writeFileSync(authPath, JSON.stringify({ openrouter: { key: "k" } }));
	const urls: string[] = [];
	setFetchForTests(async (url, init) => {
		urls.push(`${init?.method ?? "GET"} ${url}`);
		return listAndProbeStub(["other/model"], { status: 200, payload: probeAnswers })(url, init);
	});
	const result = await checkRemoteClassifier({ ...section, model: "typesafe/jev-1.14" }, "jev");
	assert.equal(result.level, "ok");
	assert.match(result.message, /openrouter\/typesafe\/jev-1\.14 verified/);
	assert.deepEqual(urls, [
		"GET https://openrouter.ai/api/v1/models",
		"POST https://openrouter.ai/api/v1/systemone",
	]);
});

test("litellm team-restricted model is a problem with the truthful reason", async () => {
	// Real-world case: the proxy's /v1/models is team-filtered, so a real model
	// the key cannot access is absent from the list; the probe surfaces the
	// permission rejection instead of a misleading "check the model id".
	writeFileSync(authPath, JSON.stringify({ litellm: { key: "k", baseUrl: "http://127.0.0.1:4000" } }));
	setFetchForTests(
		listAndProbeStub(["typesafe/jev-1.13"], {
			status: 400,
			payload: { error: { message: "team not allowed to access model. This team can only access models=['qwen/qwen3.8-max']" } },
		}),
	);
	const result = await checkRemoteClassifier(
		{ ...section, provider: "litellm", model: "typesafe/jev-1.14" },
		"jev",
	);
	assert.equal(result.level, "problem");
	assert.match(result.message, /rejected model "typesafe\/jev-1\.14" for your credentials/);
	assert.match(result.message, /routing would fail/);
	assert.match(result.message, /team not allowed/, "the provider's rejection reason is quoted");
});

test("scoring endpoint unreachable during the probe is a warning, not a denial", async () => {
	writeFileSync(authPath, JSON.stringify({ openrouter: { key: "k" } }));
	setFetchForTests(async (url: string, init?: { method?: string }) => {
		if (init?.method === "POST") throw new Error("connect ECONNREFUSED");
		return jsonResponse({ data: [{ id: "other/model" }] });
	});
	const result = await checkRemoteClassifier({ ...section, model: "typesafe/jev-1.14" }, "jev");
	assert.equal(result.level, "warn");
	assert.match(result.message, /could not reach the openrouter scoring endpoint/);
	assert.match(result.message, /unverified/);
});

test("litellm lists models against the auth-file baseUrl with the bearer key", async () => {
	writeFileSync(authPath, JSON.stringify({ litellm: { key: "secret", baseUrl: "http://127.0.0.1:4000" } }));
	let seenUrl = "";
	let seenAuth = "";
	setFetchForTests(async (url, init) => {
		seenUrl = url;
		seenAuth = String((init?.headers as Record<string, string>)?.Authorization ?? "");
		return jsonResponse({ data: [{ id: "typesafe/jev-1.13" }] });
	});
	const result = await checkRemoteClassifier({ ...section, provider: "litellm" }, "jev");
	assert.equal(result.level, "ok");
	assert.equal(seenUrl, "http://127.0.0.1:4000/v1/models");
	assert.equal(seenAuth, "Bearer secret");
});

test("the probe honors the section protocol (decisions endpoint)", async () => {
	// The menu validates the endpoint routing will use: a protocol "decisions"
	// section must probe /api/alpha/decisions, not the systemone default.
	writeFileSync(authPath, JSON.stringify({ openrouter: { key: "k" } }));
	const urls: string[] = [];
	setFetchForTests(async (url, init) => {
		urls.push(`${init?.method ?? "GET"} ${url}`);
		if (init?.method === "POST") return jsonResponse(probeAnswers);
		return jsonResponse({ data: [{ id: "other/model" }] });
	});
	const result = await checkRemoteClassifier(
		{ ...section, model: "typesafe/jev-1.14", protocol: "decisions" },
		"jev",
	);
	assert.equal(result.level, "ok");
	assert.deepEqual(urls, [
		"GET https://openrouter.ai/api/v1/models",
		"POST https://openrouter.ai/api/alpha/decisions",
	]);
});

test("validation proxy resolution matches routing: explicit, then settings/env fallback", () => {
	// Neutralize ambient settings.json / env proxies so the chain is deterministic
	// (this machine's own settings.json carries an httpProxy).
	setRuntimePathsForTests({ settingsPath: join(authDir, "settings.json") });
	const proxyVars = ["https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY"] as const;
	const saved = Object.fromEntries(proxyVars.map((n) => [n, process.env[n]]));
	try {
		for (const n of proxyVars) delete process.env[n];
		assert.equal(remoteProxyUrl({ ...section, proxyUrl: null }), null, "no proxy when nothing is configured");
		process.env.https_proxy = "http://env-only:7";
		assert.equal(remoteProxyUrl({ ...section, proxyUrl: null }), "http://env-only:7", "env fallback applies when the config value is unset");
		assert.equal(remoteProxyUrl({ ...section, proxyUrl: "direct" }), null, "\"direct\" means never proxy");
		assert.equal(remoteProxyUrl({ ...section, proxyUrl: "http://cfg:9" }), "http://cfg:9", "explicit config wins");
	} finally {
		for (const n of proxyVars) {
			if (saved[n] === undefined) delete process.env[n];
			else process.env[n] = saved[n];
		}
	}
});

test("unreachable provider is a warning, not a denial", async () => {
	writeFileSync(authPath, JSON.stringify({ openrouter: { key: "k" } }));
	setFetchForTests(async () => {
		throw new Error("connect ECONNREFUSED");
	});
	const result = await checkRemoteClassifier(section, "jev");
	assert.equal(result.level, "warn");
	assert.match(result.message, /could not connect to openrouter/);
	assert.match(result.message, /unverified/);
});

test("litellm without a baseUrl in the auth file is an unverified warning", async () => {
	writeFileSync(authPath, JSON.stringify({ litellm: { key: "k" } }));
	let calls = 0;
	setFetchForTests(async () => {
		calls++;
		return jsonResponse({});
	});
	const result = await checkRemoteClassifier({ ...section, provider: "litellm" }, "jev");
	assert.equal(result.level, "warn");
	assert.match(result.message, /no baseUrl for litellm/);
	assert.equal(calls, 0, "no endpoint to call: the message comes from endpoint construction");
});

test("model lists and probe verdicts are cached within the TTL", async () => {
	writeFileSync(authPath, JSON.stringify({ openrouter: { key: "k" } }));
	let calls = 0;
	const stub = listAndProbeStub(["typesafe/jev-1.13"]);
	setFetchForTests(async (url, init) => {
		calls++;
		return stub(url, init);
	});
	const first = await checkRemoteClassifier(section, "jev");
	assert.equal(first.level, "ok");
	assert.equal(calls, 1, "list hit: one fetch, no probe");
	const second = await checkRemoteClassifier({ ...section, model: "typesafe/jev-1.14" }, "jev");
	assert.equal(second.level, "ok", "probe settles the id the list omits");
	assert.equal(calls, 2, "list served from cache; probe adds one POST");
	const third = await checkRemoteClassifier({ ...section, model: "typesafe/jev-1.14" }, "jev");
	assert.equal(third.level, "ok");
	assert.equal(calls, 2, "probe verdict is cached, no further calls");
});
