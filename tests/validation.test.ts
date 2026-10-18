/**
 * Remote classifier validation (jev backend): credentials + model-existence
 * checks with deny/warn levels, endpoint construction, and list caching.
 */
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { setFetchForTests, type FetchLike } from "../triage-router-lib/triage-router-openrouter.ts";
import { checkRemoteClassifier, modelsUrlFor, resetValidationCacheForTests } from "../triage-router-lib/triage-router-validation.ts";

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

test.afterEach(() => {
	setFetchForTests(null);
	resetValidationCacheForTests();
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

test("unknown model on openrouter denies the edit", async () => {
	writeFileSync(authPath, JSON.stringify({ openrouter: { key: "k" } }));
	setFetchForTests(async () => jsonResponse({ data: [{ id: "other/model" }] }));
	const result = await checkRemoteClassifier({ ...section, model: "typesafe/jev-1.14" }, "jev");
	assert.equal(result.level, "problem");
	assert.match(result.message, /model "typesafe\/jev-1\.14" not found on openrouter/);
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

test("model lists are cached per provider within the TTL", async () => {
	writeFileSync(authPath, JSON.stringify({ openrouter: { key: "k" } }));
	let calls = 0;
	setFetchForTests(async () => {
		calls++;
		return jsonResponse({ data: [{ id: "typesafe/jev-1.13" }] });
	});
	const first = await checkRemoteClassifier(section, "jev");
	const second = await checkRemoteClassifier({ ...section, model: "typesafe/jev-1.14" }, "jev");
	assert.equal(first.level, "ok");
	assert.equal(second.level, "problem", "cached list is reused, so the absent id is a definite answer");
	assert.equal(calls, 1, "second check served from the cache");
});
