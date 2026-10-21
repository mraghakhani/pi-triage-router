/**
 * Tests for the manual "Refresh prices & benchmarks" action in the /triage
 * dashboard menu (refreshDataMenu): both dataset parts refresh on demand,
 * results are notified per part, and failures keep the last known good
 * cache while surfacing the error.
 *
 * Network is faked via the same deps injection the background refresh uses;
 * cachePath values point at a temp dir so no real user cache is touched.
 */
import { test, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import * as router from "../triage-router.ts";
import { refreshDataMenu } from "../triage-router.ts";

let dir: string;
let notifications: { message: string; level: string }[];

function makeCtx() {
	return {
		ui: {
			notify: (message: string, level: string) => {
				notifications.push({ message, level });
			},
		},
	} as never;
}

/** Seed a minimal config: one enabled model, temp cache paths, no network. */
function seedConfig(extra: Record<string, unknown> = {}) {
	const configPath = join(dir, "triage-router.json");
	writeFileSync(configPath, JSON.stringify({
		models: [{ provider: "litellm", id: "z-ai/glm-5.3", description: "x" }],
		prices: { cachePath: join(dir, "prices.json") },
		benchmarks: { cachePath: join(dir, "benchmarks.json"), source: "openrouter-models" },
		...extra,
	}));
	return () => router.loadConfig(configPath);
}

beforeEach(() => {
	dir = mkdtempSync(join(tmpdir(), "triage-refresh-menu-"));
	notifications = [];
	router.setConfigForTests(null);
});

afterEach(() => {
	router.setConfigForTests(null);
	rmSync(dir, { recursive: true, force: true });
});

const catalogFetch = (body: unknown) =>
	((async (_url: string) => ({ ok: true, status: 200, json: async () => body })) as never);

test("refreshDataMenu refreshes both datasets and notifies per part", async () => {
	const getConfig = seedConfig();
	const priceFetch = catalogFetch({
		data: [{ id: "z-ai/glm-5.3", pricing: { prompt: "0.00000004", completion: "0.00000008" } }],
	});
	const benchFetch = catalogFetch({
		data: [{ id: "z-ai/glm-5.3", benchmarks: { artificial_analysis: { coding_index: 71.5 } } }],
	});
	await router.refreshDataMenu(makeCtx(), getConfig, { fetchImpl: priceFetch }, { fetchImpl: benchFetch });
	const infos = notifications.filter((n) => n.level === "info").map((n) => n.message);
	assert.ok(infos.some((m) => m.includes("Refreshing prices & benchmarks")), "upfront progress notice");
	assert.ok(infos.some((m) => m.includes("prices refreshed (1 models)")), `prices notified: ${JSON.stringify(infos)}`);
	assert.ok(infos.some((m) => m.includes("benchmarks refreshed (1 models)")), "benchmarks notified");
	// Both caches were written with the fetched data.
	const prices = JSON.parse(readFileSync(join(dir, "prices.json"), "utf8"));
	assert.equal(prices.prices["z-ai/glm-5.3"].input, 0.04);
	const bench = JSON.parse(readFileSync(join(dir, "benchmarks.json"), "utf8"));
	assert.equal(bench.models["z-ai/glm-5.3"].coding, 71.5);
});

test("refreshDataMenu surfaces failures without touching the last known good cache", async () => {
	const getConfig = seedConfig();
	const good = catalogFetch({
		data: [{ id: "z-ai/glm-5.3", pricing: { prompt: "0.00000004", completion: "0.00000008" } }],
	});
	await router.refreshDataMenu(makeCtx(), getConfig, { fetchImpl: good }, { fetchImpl: catalogFetch({ data: [] }) });
	// Prices written; benchmarks (openrouter-models source) wrote the empty-set dataset.
	// Now fail both parts: the written caches must survive untouched.
	const failing = (async () => ({ ok: false, status: 500, json: async () => ({}) })) as never;
	await router.refreshDataMenu(makeCtx(), getConfig, { fetchImpl: failing }, { fetchImpl: failing });
	const warns = notifications.filter((n) => n.level === "warning").map((n) => n.message);
	assert.equal(warns.length, 2, `both parts warn: ${JSON.stringify(warns)}`);
	assert.ok(warns[0].startsWith("price refresh failed:"), "prices failure surfaced");
	assert.ok(warns[1].startsWith("benchmark refresh failed:"), "benchmarks failure surfaced");
	const prices = JSON.parse(readFileSync(join(dir, "prices.json"), "utf8"));
	assert.equal(prices.prices["z-ai/glm-5.3"].input, 0.04, "last known good prices kept");
});

test("disabled models are not fetched; enabled flags are respected", async () => {
	const getConfig = seedConfig({
		models: [
			{ provider: "litellm", id: "z-ai/glm-5.3", description: "x" },
			{ provider: "litellm", id: "a/b", description: "x", enabled: false },
		],
	});
	const seen: string[] = [];
	const fetchImpl = (async (url: string) => {
		seen.push(url);
		return { ok: true, status: 200, json: async () => ({ data: [] }) };
	}) as never;
	await router.refreshDataMenu(makeCtx(), getConfig, { fetchImpl }, { fetchImpl });
	// Both parts fetched; the empty catalog means zero models in each dataset.
	// Both parts fetched. Prices only keep catalog matches (glm absent → 0);
	// benchmarks create an entry per requested id, disabled "a/b" excluded → 1.
	const infos = notifications.filter((n) => n.level === "info").map((n) => n.message);
	assert.ok(infos.some((m) => m.includes("prices refreshed (0 models)")), "price dataset reflects catalog matches");
	assert.ok(infos.some((m) => m.includes("benchmarks refreshed (1 models)")), "disabled model not requested");
});