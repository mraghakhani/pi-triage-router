/**
 * Price cache tests: normalize/parse/write/read, refresh failure keeping
 * last-known-good, refresh interval + enabled switch, and config validation.
 * No network: fetch is injected; catalog rows are synthetic fixtures.
 */
import { test, afterEach, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, utimesSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
	MODELS_URL,
	normalizePrices,
	parsePriceDataset,
	readPriceDataset,
	refreshPrices,
	resetPriceCacheForTests,
	resetPriceRefreshStateForTests,
	schedulePriceRefresh,
	writePriceDataset,
	type PriceConfig,
} from "../triage-router-lib/triage-router-prices.ts";
import * as router from "../triage-router.ts";

const IDS = ["deepseek/deepseek-v4-flash", "z-ai/glm-5.3"];

function catalog(): Array<{ id: string; pricing: { prompt: string; completion: string } }> {
	return [
		{ id: "deepseek/deepseek-v4-flash", pricing: { prompt: "0.00000001", completion: "0.00000003" } },
		{ id: "z-ai/glm-5.3", pricing: { prompt: "0.00000004", completion: "0.00000008" } },
	];
}

function priceConfig(cachePath: string): PriceConfig {
	return {
		enabled: true,
		cachePath,
		refreshIntervalMs: 86_400_000,
		timeoutMs: 100,
		proxyUrl: null,
	};
}

let dir = "";
beforeEach(() => {
	dir = mkdtempSync(join(tmpdir(), "laya-prices-"));
});
afterEach(() => {
	rmSync(dir, { recursive: true, force: true });
	resetPriceCacheForTests();
	resetPriceRefreshStateForTests();
});

test("normalizePrices converts USD/token to USD/M, keeps only configured ids", () => {
	const dataset = normalizePrices(
		[
			...catalog(),
			{ id: "other/model", pricing: { prompt: "0.5", completion: "0.5" } },
			{ id: "broken/model", pricing: { prompt: "lots", completion: "-" } },
		],
		"2026-10-15T00:00:00.000Z",
		IDS,
	);
	assert.equal(dataset.source, "openrouter-models");
	assert.deepEqual(Object.keys(dataset.prices).sort(), IDS);
	assert.deepEqual(dataset.prices["deepseek/deepseek-v4-flash"], { input: 0.01, output: 0.03 });
});

test("parsePriceDataset rejects corrupt caches, sanitizes entries", () => {
	assert.equal(parsePriceDataset(null), null);
	assert.equal(parsePriceDataset({ fetchedAt: "soon" }), null);
	assert.equal(parsePriceDataset({ fetchedAt: "2026-10-15T00:00:00.000Z", source: "bogus", prices: {} }), null);
	const good = parsePriceDataset({
		fetchedAt: "2026-10-15T00:00:00.000Z",
		source: "openrouter-models",
		prices: {
			"a/b": { input: 1, output: 2 },
			"bad/input": { input: "cheap" },
			"neg": { input: -1, output: 0 },
		},
	});
	assert.deepEqual(good?.prices, { "a/b": { input: 1, output: 2 } });
});

test("write/read roundtrip survives a reread", () => {
	const path = join(dir, "prices.json");
	const dataset = normalizePrices(catalog(), "2026-10-15T00:00:00.000Z", IDS);
	writePriceDataset(path, dataset);
	assert.deepEqual(readPriceDataset(path), dataset);
});

test("bundled seed serves prices until the agent-dir cache exists", () => {
	const seedPath = join(dir, "seed.json");
	const dataset = normalizePrices(catalog(), "2026-10-15T00:00:00.000Z", IDS);
	writePriceDataset(seedPath, dataset);
	// No cache file yet: the package snapshot is served.
	assert.deepEqual(readPriceDataset(join(dir, "absent-cache.json"), seedPath), dataset);
	// Once the cache exists, it wins over the seed.
	const fresh = normalizePrices([{ id: "z-ai/glm-5.3", pricing: { prompt: "0.00000002", completion: "0.00000002" } }], "2026-10-16T00:00:00.000Z", IDS);
	const cachePath = join(dir, "cache.json");
	writePriceDataset(cachePath, fresh);
	assert.deepEqual(readPriceDataset(cachePath, seedPath), fresh);
});

test("refreshPrices fetches, persists, and keeps last known good on failure", async () => {
	const path = join(dir, "prices.json");
	const cfg = priceConfig(path);
	const fetchImpl = (async (url: string) => ({
		ok: true,
		status: 200,
		json: async () => ({ data: catalog() }),
	})) as never;
	const ok = await refreshPrices(cfg, IDS, { fetchImpl });
	assert.equal(ok.ok, true);
	assert.equal(ok.dataset?.prices["z-ai/glm-5.3"].input, 0.04);
	assert.deepEqual(readPriceDataset(path)?.prices["z-ai/glm-5.3"], { input: 0.04, output: 0.08 });

	const bad = await refreshPrices(cfg, IDS, {
		fetchImpl: (async () => ({ ok: false, status: 500, json: async () => ({}) })) as never,
	});
	assert.equal(bad.ok, false);
	assert.ok(bad.error);
	// Last known good survives the failed refresh.
	assert.deepEqual(readPriceDataset(path)?.prices["z-ai/glm-5.3"], { input: 0.04, output: 0.08 });
});

test("schedulePriceRefresh respects age, enabled, and single flight", async () => {
	const path = join(dir, "prices.json");
	const cfg = priceConfig(path);
	let calls = 0;
	const deps = {
		now: () => Date.now(),
		fetchImpl: (async () => {
			calls += 1;
			return { ok: true, status: 200, json: async () => ({ data: catalog() }) };
		}) as never,
	};
	// Disabled: nothing scheduled.
	assert.equal(schedulePriceRefresh({ ...cfg, enabled: false }, IDS, deps), null);
	// Fresh cache: nothing scheduled.
	writePriceDataset(path, normalizePrices(catalog(), new Date().toISOString(), IDS));
	assert.equal(schedulePriceRefresh(cfg, IDS, deps), null);
	assert.equal(calls, 0);
	// Stale cache (mtime 2 days old): refresh, single flight.
	const old = new Date(Date.now() - 2 * 86_400_000);
	utimesSync(path, old, old);
	const first = schedulePriceRefresh(cfg, IDS, deps);
	assert.ok(first);
	const second = schedulePriceRefresh(cfg, IDS, deps);
	assert.equal(second, first, "in-flight refresh is shared");
	await first;
	assert.equal(calls, 1);
	// Cache rewritten: not stale anymore.
	assert.equal(schedulePriceRefresh(cfg, IDS, deps), null);
	assert.equal(readPriceDataset(path)?.source, "openrouter-models");
});

test("config prices section validates and defaults", () => {
	const config = router.validateConfig({ models: [] });
	assert.equal(config.prices.enabled, true);
	assert.equal(config.prices.refreshIntervalMs, 86_400_000);

	const bad = router.validateConfig({
		models: [],
		prices: { enabled: "yes", refreshIntervalMs: 1, timeoutMs: 1, cachePath: "", bogus: 1 },
	});
	for (const needle of ["prices.enabled", "prices.refreshIntervalMs", "prices.timeoutMs", "prices.cachePath", "bogus"]) {
		assert.ok(bad.problems.some((p) => p.includes(needle)), `problem mentions ${needle}`);
	}
	assert.equal(bad.prices.refreshIntervalMs, 86_400_000, "invalid falls back to default");

	const good = router.validateConfig({
		models: [],
		prices: { enabled: false, refreshIntervalMs: 3_600_000, timeoutMs: 5000 },
	});
	assert.equal(good.prices.enabled, false);
	assert.equal(good.prices.refreshIntervalMs, 3_600_000);
	assert.equal(good.prices.timeoutMs, 5000);
	assert.ok(!good.problems.some((p) => p.includes("prices")));
});

test("estimateCost: cached price fills in when registry and config are silent", () => {
	const entry = { provider: "litellm", id: "a/b", description: "", priceIn: undefined, priceOut: undefined };
	// Config price still wins over cache? No: cache outranks config, registry outranks cache.
	const fromCache = router.estimateCost(entry as never, undefined, 4000, 1000, null, { input: 2, output: 4 });
	assert.equal(fromCache.estCostIn, 0.002);
	assert.equal(fromCache.estCostOut, null, "no usage: output tokens unknown");
	const fromRegistry = router.estimateCost(entry as never, { cost: { input: 9 } }, 4000, 1000, null, { input: 2, output: 4 });
	assert.equal(fromRegistry.estCostIn, 0.009);
	const fromConfig = router.estimateCost({ ...entry, priceIn: 5 } as never, undefined, 4000, 1000, null, undefined);
	assert.equal(fromConfig.estCostIn, 0.005);
});

test("MODELS_URL points at the public OpenRouter catalog", () => {
	assert.equal(MODELS_URL, "https://openrouter.ai/api/v1/models");
});

test("price cache is keyed by model id without provider prefix loss", () => {
	const dataset = normalizePrices(
		[{ id: "deepseek/deepseek-v4-flash", pricing: { prompt: "0.00000001", completion: "0.00000003" } }],
		"2026-10-15T00:00:00.000Z",
		["deepseek/deepseek-v4-flash"],
	);
	// Config model ids keep their vendor-slug shape (litellm/deepseek/... -> id deepseek/...).
	assert.ok(dataset.prices["deepseek/deepseek-v4-flash"]);
});