/**
 * Benchmark ingestion, caching, and scoring tests.
 *
 * No network: fetch is injected. Unit rows are clearly synthetic fixtures;
 * the REAL OpenRouter dataset fetched for evaluation lives in
 * tests/fixtures/benchmarks.json (source + as_of preserved) and is exercised
 * by tests/strategy-eval.test.ts.
 *
 * Covered: slug resolution (exact/dated, never family-prefix), variant
 * averaging, cache write/read, refresh failure keeping last-known-good,
 * keyless fallback source, refresh interval + single flight + explicit
 * refresh, freshness/coverage observability, and the deterministic scoring
 * policy (normalization, missing/stale policies, complexity weights,
 * capabilities overrides, tie-breaks, no-data fallback).
 */
import { test, afterEach, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, utimesSync, existsSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
	datasetAgeMs,
	datasetStatus,
	normalizeModelsCatalog,
	normalizeUnified,
	parseDataset,
	benchmarkProxyUrl,
	readDataset,
	refreshBenchmarks,
	refreshNow,
	resetDatasetReadCacheForTests,
	resetRefreshStateForTests,
	resolveSlug,
	scoreEligible,
	writeDataset,
	type BenchmarkConfig,
	type BenchmarkDataset,
} from "../triage-router-lib/triage-router-benchmarks.ts";
import { setFetchForTests, setRuntimePathsForTests, type FetchLike } from "../triage-router-lib/triage-router-openrouter.ts";
import { testConfig } from "./fixtures.ts";
import { decide, loadConfig, validateConfig, type ModelEntry } from "../triage-router.ts";

let dir: string;

beforeEach(() => {
	dir = mkdtempSync(join(tmpdir(), "laya-bench-"));
});

afterEach(() => {
	resetRefreshStateForTests();
	resetDatasetReadCacheForTests();
	setFetchForTests(null);
	setRuntimePathsForTests({ authFile: null });
	delete process.env.OPENROUTER_API_KEY;
	try {
		rmSync(dir, { recursive: true, force: true });
	} catch {
		// best effort
	}
});

function benchConfig(overrides: Partial<BenchmarkConfig> = {}): BenchmarkConfig {
	return {
		enabled: true,
		source: "openrouter",
		cachePath: join(dir, "bench.json"),
		refreshIntervalMs: 60_000,
		staleAfterMs: 2_592_000_000,
		timeoutMs: 5_000,
		apiKeyEnv: "OPENROUTER_API_KEY",
		proxyUrl: null,
		...overrides,
	};
}

function fetchReturning(body: unknown, status = 200): FetchLike {
	return async () => ({
		ok: status >= 200 && status < 300,
		status,
		json: async () => {
			if (body instanceof Error) throw body;
			return body;
		},
		text: async () => JSON.stringify(body),
	});
}

/** Synthetic unified-endpoint rows exercising every resolution rule. */
const ROWS: Array<Record<string, unknown>> = [
	// exact slug, two display variants (must be averaged)
	{ source: "artificial-analysis", model_permaslug: "z-ai/glm-5.3", intelligence_index: 40, coding_index: 70, agentic_index: 50 },
	{ source: "artificial-analysis", model_permaslug: "z-ai/glm-5.3", intelligence_index: 50, coding_index: 80, agentic_index: 60 },
	// family trap: must NEVER bleed into z-ai/glm-5.3
	{ source: "artificial-analysis", model_permaslug: "z-ai/glm-5.3-flash-20260826", intelligence_index: 5, coding_index: 5, agentic_index: 5 },
	// two dated releases: newest wins, older ignored
	{ source: "artificial-analysis", model_permaslug: "qwen/qwen3.8-max-20260803", intelligence_index: 30, coding_index: 30, agentic_index: 30 },
	{ source: "artificial-analysis", model_permaslug: "qwen/qwen3.8-max-20260902", intelligence_index: 60, coding_index: 90, agentic_index: 70 },
	// 4-digit MMDD release interpreted with the dataset year
	{ source: "artificial-analysis", model_permaslug: "deepseek/deepseek-v4-flash-0731", intelligence_index: 34.3, coding_index: 69.1, agentic_index: 41 },
	// design arena: code category counts, non-code does not
	{ source: "design-arena", model_permaslug: "z-ai/glm-5.3", category: "codecategories", elo: 1500 },
	{ source: "design-arena", model_permaslug: "z-ai/glm-5.3", category: "3d", elo: 9999 },
	// openrouter evals
	{ source: "openrouter", model_permaslug: "z-ai/glm-5.3", benchmark_type: "gpqa_diamond", accuracy: 0.82, last_run_timestamp: "2026-10-01T00:00:00Z" },
	{ source: "openrouter", model_permaslug: "z-ai/glm-5.3", benchmark_type: "tau_bench_verified_airline", accuracy: 0.76, avg_cost_per_task: 0.016, last_run_timestamp: "2026-10-01T00:00:00Z" },
	{ source: "openrouter", model_permaslug: "z-ai/glm-5.3", benchmark_type: "search_hle", accuracy: 0.5, last_run_timestamp: "2026-09-01T00:00:00Z" },
	{ source: "openrouter", model_permaslug: "z-ai/glm-5.3", benchmark_type: "search_browsecomp", accuracy: 0.3, last_run_timestamp: "2026-09-15T00:00:00Z" },
];

const META = { as_of: "2026-10-09T12:00:00Z" };
const IDS = ["z-ai/glm-5.3", "qwen/qwen3.8-max", "deepseek/deepseek-v4-flash", "z-ai/glm-5.3-flash", "no/data-model"];

// ---------------------------------------------------------------------------
// Slug resolution + normalization
// ---------------------------------------------------------------------------

test("resolveSlug: exact, dated, and never family-prefix", () => {
	assert.equal(resolveSlug("z-ai/glm-5.3", "z-ai/glm-5.3", 2026), "exact");
	assert.equal(resolveSlug("qwen/qwen3.8-max", "qwen/qwen3.8-max-20260902", 2026), "dated");
	assert.equal(resolveSlug("deepseek/deepseek-v4-flash", "deepseek/deepseek-v4-flash-0731", 2026), "dated");
	assert.equal(resolveSlug("z-ai/glm-5.3", "z-ai/glm-5.3-flash-20260826", 2026), null, "flash is a different model");
	assert.equal(resolveSlug("z-ai/glm-5.3", "z-ai/glm-5.3-flash", 2026), null);
	assert.equal(resolveSlug("qwen/qwen3.8-max", "qwen/qwen3.5", 2026), null);
	assert.equal(resolveSlug("gpt-5", "gpt-5-2025", 2026), null, "4-digit suffix that is not MMDD (a year) must not match");
	assert.equal(resolveSlug("gpt-5", "gpt-5-1234", 2026), null, "month 12/day 34 is not a calendar date");
	assert.equal(resolveSlug("gpt-5", "gpt-5-0831", 2026), "dated", "real MMDD still matches");
});

test("normalizeUnified resolves variants, winners, and every metric group", () => {
	const dataset = normalizeUnified(ROWS as never, META, "2026-10-10T00:00:00Z", "https://openrouter.ai/api/v1/benchmarks", IDS);
	const glm = dataset.models["z-ai/glm-5.3"];
	assert.equal(glm.match, "exact");
	assert.equal(glm.variants, 2, "two display variants averaged");
	assert.equal(glm.coding, 75, "mean of 70 and 80");
	assert.equal(glm.reasoning, 45);
	assert.equal(glm.arena, 1500, "codecategories only; 3d row excluded");
	assert.equal(glm.gpqa, 0.82);
	assert.equal(glm.taskSuccess, 0.76);
	assert.equal(glm.costPerTask, 0.016);
	assert.equal(glm.research, 0.4, "mean of search evals");
	assert.equal(glm.updatedAt, META.as_of, "AA rows carry no timestamp: dataset as_of anchors freshness");
	assert.ok(glm.sources.includes("artificial-analysis") && glm.sources.includes("design-arena") && glm.sources.includes("openrouter"));

	const qwen = dataset.models["qwen/qwen3.8-max"];
	assert.equal(qwen.match, "dated");
	assert.equal(qwen.coding, 90, "newest dated release wins, older release ignored");
	assert.equal(qwen.reasoning, 60);

	const flash = dataset.models["deepseek/deepseek-v4-flash"];
	assert.equal(flash.match, "dated");
	assert.equal(flash.coding, 69.1, "MMDD slug resolved with dataset year");

	const glmFlash = dataset.models["z-ai/glm-5.3-flash"];
	assert.equal(glmFlash.coding, 5, "flash resolves its OWN dated slug");
	assert.equal(glm.coding, 75, "glm-5.3 is never contaminated by flash rows");

	const missing = dataset.models["no/data-model"];
	assert.equal(missing.coding, null, "no data stays null - never invented");
	assert.equal(missing.match, "exact");
	assert.equal(dataset.asOf, META.as_of);
	assert.equal(dataset.sourceUrl, "https://openrouter.ai/api/v1/benchmarks");
});

test("normalizeModelsCatalog: keyless source carries AA indices only", () => {
	const dataset = normalizeModelsCatalog(
		[
			{ id: "z-ai/glm-5.3", benchmarks: { artificial_analysis: { intelligence_index: 44.8, coding_index: 74.8, agentic_index: 53.1 } } },
			{ id: "other/model", benchmarks: { artificial_analysis: { intelligence_index: 10, coding_index: null, agentic_index: null } } },
		],
		"2026-10-10T00:00:00Z",
		"https://openrouter.ai/api/v1/models",
		["z-ai/glm-5.3", "absent/model"],
	);
	assert.equal(dataset.models["z-ai/glm-5.3"].coding, 74.8);
	assert.equal(dataset.models["z-ai/glm-5.3"].gpqa, null, "not available from this source");
	assert.equal(dataset.models["absent/model"].coding, null);
	assert.equal(dataset.source, "openrouter-models");
});

test("parseDataset rejects malformed caches", () => {
	assert.equal(parseDataset(null), null);
	assert.equal(parseDataset({}), null);
	assert.equal(parseDataset({ fetchedAt: 5, models: {} }), null);
	assert.equal(parseDataset("junk"), null);
	const ok = parseDataset({ fetchedAt: "2026-10-09T00:00:00Z", source: "openrouter", models: {} });
	assert.ok(ok);
});

// ---------------------------------------------------------------------------
// Refresh: success, failure, last-known-good, keyless, interval, explicit
// ---------------------------------------------------------------------------

const UNIFIED_PAYLOAD = { data: ROWS, meta: META };

test("refresh success writes an atomic cache the router can read", async () => {
	process.env.OPENROUTER_API_KEY = "test-key";
	setFetchForTests(fetchReturning(UNIFIED_PAYLOAD));
	const result = await refreshBenchmarks(benchConfig(), IDS);
	assert.equal(result.ok, true);
	assert.ok(existsSync(join(dir, "bench.json")), "cache written");
	assert.ok(!existsSync(join(dir, "bench.json.tmp")), "no tmp file left behind");
	const dataset = readDataset(join(dir, "bench.json"));
	assert.ok(dataset);
	assert.equal(dataset.models["z-ai/glm-5.3"].coding, 75);
	assert.equal(dataset.source, "openrouter");
});

test("readDataset serves the bundled seed while the agent-dir cache does not exist yet", async () => {
	const main = join(dir, "absent.json");
	const seed = join(dir, "seed.json");
	process.env.OPENROUTER_API_KEY = "test-key";
	setFetchForTests(fetchReturning(UNIFIED_PAYLOAD));
	await refreshBenchmarks(benchConfig({ cachePath: seed }), IDS);
	assert.ok(existsSync(seed));
	// No seed configured: missing cache stays missing.
	assert.equal(readDataset(main), null);
	// Seed configured: first reads get data before the first refresh lands.
	const seeded = readDataset(main, seed);
	assert.ok(seeded);
	assert.equal(seeded.models["z-ai/glm-5.3"].coding, 75);
	// Seed equal to the main path never recurses into itself.
	assert.equal(readDataset(main, main), null);
});

test("refresh failure keeps the last known good cache byte-for-byte", async () => {
	process.env.OPENROUTER_API_KEY = "test-key";
	setFetchForTests(fetchReturning(UNIFIED_PAYLOAD));
	await refreshBenchmarks(benchConfig(), IDS);
	const before = readFileSync(join(dir, "bench.json"), "utf8");

	for (const [name, fetchImpl, expected] of [
		["rate limited", fetchReturning({}, 429), /HTTP 429/],
		["server error", fetchReturning({}, 500), /HTTP 500/],
		["malformed body", fetchReturning({ not: "benchmarks" }), /missing `data`/],
		["transport failure", null, /boom|timed out|fetch/i],
	] as const) {
		if (fetchImpl) setFetchForTests(fetchImpl);
		else
			setFetchForTests(async () => {
				throw new Error("boom");
			});
		const result = await refreshBenchmarks(benchConfig(), IDS);
		assert.equal(result.ok, false, `${name}: refresh reports failure`);
		assert.ok(expected.test(result.error ?? ""), `${name}: error message (${result.error})`);
		assert.ok(result.dataset, `${name}: last known good still returned`);
	}
	const after = readFileSync(join(dir, "bench.json"), "utf8");
	assert.equal(after, before, "cache untouched by failed refreshes");
});

test("unified source without a key fails clearly; openrouter-models works keyless", async () => {
	delete process.env.OPENROUTER_API_KEY;
	setRuntimePathsForTests({ authFile: join(dir, "no-auth-file.json") });
	setFetchForTests(fetchReturning(UNIFIED_PAYLOAD));
	const noKey = await refreshBenchmarks(benchConfig(), IDS);
	assert.equal(noKey.ok, false);
	assert.ok((noKey.error ?? "").includes("OPENROUTER_API_KEY"));

	// keyless public source
	setFetchForTests(fetchReturning({ data: [{ id: "z-ai/glm-5.3", benchmarks: { artificial_analysis: { coding_index: 74.8 } } }] }));
	const keyless = await refreshBenchmarks(benchConfig({ source: "openrouter-models" }), IDS);
	assert.equal(keyless.ok, true);
	assert.equal(readDataset(join(dir, "bench.json"))?.models["z-ai/glm-5.3"].coding, 74.8);
});

test("unified source: missing env key falls back to the auth file's openrouter entry", async () => {
	delete process.env.OPENROUTER_API_KEY;
	writeFileSync(join(dir, "auth.json"), JSON.stringify({ openrouter: { type: "api", key: "auth-file-key" } }));
	setRuntimePathsForTests({ authFile: join(dir, "auth.json") });
	setFetchForTests(fetchReturning(UNIFIED_PAYLOAD));
	const result = await refreshBenchmarks(benchConfig(), IDS);
	assert.equal(result.ok, true);
	assert.equal(readDataset(join(dir, "bench.json"))?.models["z-ai/glm-5.3"].coding, 75);
});

test("scheduleRefresh: absent/stale cache triggers background refresh; fresh cache does not; single flight", async () => {
	process.env.OPENROUTER_API_KEY = "test-key";
	let fetches = 0;
	setFetchForTests(async (...args) => {
		fetches++;
		return fetchReturning(UNIFIED_PAYLOAD)(...args);
	});
	const config = benchConfig({ refreshIntervalMs: 60_000 });

	// No cache: refresh scheduled (non-blocking promise returned).
	const first = refreshNow(config, IDS);
	assert.ok(first, "explicit refresh returns a promise");
	await first;
	assert.equal(fetches, 1);

	// Fresh cache: scheduleRefresh declines.
	assert.equal(refreshBenchmarks.length > 0, true);
	const before = fetches;
	const { scheduleRefresh } = await import("../triage-router-lib/triage-router-benchmarks.ts");
	assert.equal(scheduleRefresh(config, IDS), null, "fresh cache: no background refresh");

	// Stale cache: scheduleRefresh triggers exactly one in-flight refresh.
	const old = new Date(Date.now() - 3_600_000);
	utimesSync(config.cachePath, old, old);
	const p1 = scheduleRefresh(config, IDS);
	assert.ok(p1, "stale cache triggers refresh");
	const p2 = scheduleRefresh(config, IDS);
	assert.ok(p2 === p1, "single flight while in progress");
	await p1;
	assert.equal(fetches, before + 1, "exactly one fetch for the stale refresh");
});

test("datasetStatus and datasetAgeMs expose freshness and coverage", async () => {
	process.env.OPENROUTER_API_KEY = "test-key";
	setFetchForTests(fetchReturning(UNIFIED_PAYLOAD));
	await refreshBenchmarks(benchConfig(), IDS);
	resetDatasetReadCacheForTests();
	const config = benchConfig();
	const status = datasetStatus(config, IDS);
	assert.ok(/bench=\d+\/\d+/.test(status), `coverage visible: ${status}`);
	assert.ok(status.includes("age=0h"), `age visible: ${status}`);
	assert.ok(!status.includes("stale"), "fresh dataset not flagged stale");
	const age = datasetAgeMs(config.cachePath);
	assert.ok(age !== null && age < 60_000);

	// Stale flag
	const dataset = JSON.parse(readFileSync(config.cachePath, "utf8")) as BenchmarkDataset;
	dataset.fetchedAt = new Date(Date.now() - 40 * 24 * 3_600_000).toISOString();
	writeFileSync(config.cachePath, JSON.stringify(dataset));
	resetDatasetReadCacheForTests();
	assert.ok(datasetStatus(config, IDS).includes("stale"), "old dataset flagged stale");

	// Absent cache
	assert.equal(datasetStatus(benchConfig({ cachePath: join(dir, "absent.json") }), IDS), "bench=absent");
});

// ---------------------------------------------------------------------------
// Scoring policy
// ---------------------------------------------------------------------------

function metricFixture(overrides: Record<string, Record<string, unknown>> = {}): BenchmarkDataset {
	const base = (m: Record<string, unknown>) => ({
		coding: null, reasoning: null, agentic: null, arena: null, gpqa: null,
		taskSuccess: null, costPerTask: null, research: null, infra: null,
		updatedAt: "2026-10-01T00:00:00Z", match: "exact" as const, variants: 1, sources: ["test"],
		...(m as Partial<Record<string, number | null>>),
	});
	return {
		fetchedAt: "2026-10-09T00:00:00Z",
		source: "test-fixture",
		sourceUrl: "tests (synthetic)",
		asOf: "2026-10-09T00:00:00Z",
		models: {
			"model-a": base(overrides["model-a"] ?? { coding: 70, reasoning: 50 }),
			"model-b": base(overrides["model-b"] ?? { coding: 30, reasoning: 60 }),
			"model-c": base(overrides["model-c"] ?? { coding: 50, reasoning: 55 }),
		},
	};
}

const ENTRIES: ModelEntry[] = [
	{ provider: "litellm", id: "model-a", description: "a", priceIn: 0.5, priceOut: 1 },
	{ provider: "litellm", id: "model-b", description: "b", priceIn: 0.01, priceOut: 0.02 },
	{ provider: "litellm", id: "model-c", description: "c", priceIn: 0.15, priceOut: 0.3 },
];

function scoringConfig(overrides: Record<string, unknown> = {}) {
	return testConfig({ scoring: { enabled: true, ...overrides } });
}

function est(entry: ModelEntry): number | null {
	if (entry.priceIn === undefined || entry.priceOut === undefined) return null;
	return (4000 / 1e6) * entry.priceIn + (1000 / 1e6) * entry.priceOut;
}

test("scoring picks quality for coding and cost for questions, with documented weights", () => {
	const dataset = metricFixture();
	const coding = scoreEligible({
		eligible: ENTRIES, category: "coding", complexity: "moderate",
		config: scoringConfig(), dataset, estimatedCost: est,
	});
	assert.equal(coding.ranked[0].entry.id, "model-a", "coding: quality (0.65) outweighs cost - model-a wins");

	const question = scoreEligible({
		eligible: ENTRIES, category: "question", complexity: "standard",
		config: scoringConfig(), dataset, estimatedCost: est,
	});
	assert.equal(question.ranked[0].entry.id, "model-b", "question: cost (0.7 after standard delta) wins for the cheapest");
});

test("missing quality is neutral (mean), never zero; penalize policy ranks it below covered models", () => {
	const dataset = metricFixture({
		"model-c": { coding: null, reasoning: null }, // no data
	});
	const neutral = scoreEligible({
		eligible: ENTRIES, category: "coding", complexity: "moderate",
		config: scoringConfig(), dataset, estimatedCost: est,
	});
	const c = neutral.ranked.find((r) => r.entry.id === "model-c")!;
	assert.ok(c.quality > 0, `missing data scored neutrally, not zero (got ${c.quality})`);
	assert.ok(c.notes.some((n) => n.includes("no quality data")), "missing data is visible in notes");

	const penalized = scoreEligible({
		eligible: ENTRIES, category: "coding", complexity: "moderate",
		config: scoringConfig({ missingPolicy: "penalize", missingPenalty: 0.5 }), dataset, estimatedCost: est,
	});
	const pc = penalized.ranked.find((r) => r.entry.id === "model-c")!;
	assert.ok(pc.quality < c.quality, "penalize ranks below neutral treatment");
	assert.ok(pc.notes.some((n) => n.includes("penalized")), "penalty is visible in notes");
});

test("no quality data for any model returns an empty ranking (caller falls back)", () => {
	const empty: BenchmarkDataset = {
		fetchedAt: "2026-10-09T00:00:00Z", source: "test", sourceUrl: "x", asOf: null,
		models: {
			"model-a": { coding: null, reasoning: null, agentic: null, arena: null, gpqa: null, taskSuccess: null, costPerTask: null, research: null, infra: null, updatedAt: null, match: "exact", variants: 0, sources: [] },
			"model-b": { coding: null, reasoning: null, agentic: null, arena: null, gpqa: null, taskSuccess: null, costPerTask: null, research: null, infra: null, updatedAt: null, match: "exact", variants: 0, sources: [] },
			"model-c": { coding: null, reasoning: null, agentic: null, arena: null, gpqa: null, taskSuccess: null, costPerTask: null, research: null, infra: null, updatedAt: null, match: "exact", variants: 0, sources: [] },
		},
	};
	const result = scoreEligible({
		eligible: ENTRIES, category: "coding", complexity: "moderate",
		config: scoringConfig(), dataset: empty, estimatedCost: est,
	});
	assert.equal(result.ranked.length, 0, "no measured quality -> no scored decision");
	assert.equal(result.missingUsed, false);
});

test("stale policy: 'missing' drops stale rows, 'use' keeps them flagged", () => {
	const oldStamp = "2025-01-01T00:00:00Z"; // beyond 30d staleAfterMs
	const dataset = metricFixture({ "model-a": { coding: 99, reasoning: 99, updatedAt: oldStamp } });

	const strict = scoreEligible({
		eligible: ENTRIES, category: "coding", complexity: "moderate",
		config: scoringConfig({ stalePolicy: "missing" }), dataset, estimatedCost: est, now: Date.parse("2026-10-10T00:00:00Z"),
	});
	const strictA = strict.ranked.find((r) => r.entry.id === "model-a")!;
	assert.ok(strictA.notes.length > 0, "stale-as-missing model flagged (neutral quality)");

	const lenient = scoreEligible({
		eligible: ENTRIES, category: "coding", complexity: "moderate",
		config: scoringConfig({ stalePolicy: "use" }), dataset, estimatedCost: est, now: Date.parse("2026-10-10T00:00:00Z"),
	});
	const lenientA = lenient.ranked.find((r) => r.entry.id === "model-a")!;
	assert.ok(lenientA.notes.some((n) => n.includes("stale")), "stale-but-used is flagged");
	assert.equal(lenient.ranked[0].entry.id, "model-a", "stale data still ranks when policy=use");
});

test("config capabilities (internal results) override cached data for infra", () => {
	const dataset = metricFixture();
	const withCaps = ENTRIES.map((e) => {
		if (e.id === "model-b") return { ...e, capabilities: { infra: 95 } };
		if (e.id === "model-c") return { ...e, capabilities: { infra: 60 } };
		return e;
	});
	const caps = (e: ModelEntry) => e.capabilities;
	const result = scoreEligible({
		eligible: withCaps, category: "infra", complexity: "moderate",
		config: scoringConfig(), dataset, estimatedCost: est, capabilities: caps,
	});
	assert.equal(result.ranked[0].entry.id, "model-b", "internal infra evaluation decides infra tasks");
	const without = scoreEligible({
		eligible: ENTRIES, category: "infra", complexity: "moderate",
		config: scoringConfig(), dataset, estimatedCost: est, capabilities: caps,
	});
	assert.notEqual(without.ranked[0].entry.id, "model-b", "without internal data, infra falls back to general metrics");
});

test("missing price is neutral cost, not free; a pricier model can still win on quality", () => {
	const noPrice = ENTRIES.map((e) => (e.id === "model-a" ? { ...e, priceIn: undefined, priceOut: undefined } : e));
	const dataset = metricFixture();
	const result = scoreEligible({
		eligible: noPrice, category: "coding", complexity: "moderate",
		config: scoringConfig(),
		dataset,
		estimatedCost: est,
	});
	const a = result.ranked.find((r) => r.entry.id === "model-a")!;
	assert.ok(a.notes.some((n) => n.includes("missing price")), "missing price visible");
	assert.equal(result.ranked[0].entry.id, "model-a", "quality still decides when cost data is missing for the best model");
});

test("complexity shifts the quality/cost balance deterministically", () => {
	const dataset = metricFixture();
	const complex = scoreEligible({
		eligible: ENTRIES, category: "coding", complexity: "complex",
		config: scoringConfig(), dataset, estimatedCost: est,
	});
	const standard = scoreEligible({
		eligible: ENTRIES, category: "coding", complexity: "standard",
		config: scoringConfig(), dataset, estimatedCost: est,
	});
	assert.equal(complex.ranked[0].entry.id, "model-a", "coding work lands on the best model under complex");
	const aComplex = complex.ranked.find((r) => r.entry.id === "model-a")!;
	const aStandard = standard.ranked.find((r) => r.entry.id === "model-a")!;
	assert.ok(aComplex.score > aStandard.score, "complexity raises the quality-driven score of the best model");
	assert.ok(aComplex.quality === aStandard.quality, "quality itself is complexity-independent; only the weight changes");
});

test("degenerate metrics normalize to 0.5 (documented neutral, not zero)", () => {
	const dataset = metricFixture({
		"model-a": { coding: 50 },
		"model-b": { coding: 50 },
		"model-c": { coding: 50 },
	});
	const result = scoreEligible({
		eligible: ENTRIES, category: "coding", complexity: "moderate",
		config: scoringConfig(), dataset, estimatedCost: est,
	});
	for (const row of result.ranked) {
		assert.ok(Math.abs(row.quality - 0.5) < 1e-9, `all-equal quality maps to 0.5 (got ${row.quality})`);
	}
});

test("ranking is fully deterministic: score, quality, price, config order", () => {
	const dataset = metricFixture({
		"model-a": { coding: 50, reasoning: 50 },
		"model-b": { coding: 50, reasoning: 50 },
		"model-c": { coding: 50, reasoning: 50 },
	});
	const run = () =>
		scoreEligible({
			eligible: ENTRIES, category: "coding", complexity: "moderate",
			config: scoringConfig(), dataset, estimatedCost: est,
		}).ranked.map((r) => r.entry.id);
	const first = run();
	assert.deepEqual(run(), first, "same inputs, same order");
	assert.deepEqual(first, ["model-b", "model-c", "model-a"], "identical scores: cheaper price first");
});

test("scoring disabled returns nothing extra (behavior preserved)", () => {
	const off = testConfig(); // scoring.enabled false by default
	assert.equal(off.scoring.enabled, false);
	// scoreEligible itself is only called when enabled; guard here that the
	// shipped default config never enters the scored path.
	assert.equal(off.scoring.weights.coding.quality, 0.65, "default weights documented");
	assert.equal(off.scoring.missingPolicy, "neutral");
	assert.equal(off.scoring.stalePolicy, "use");
});

test("classifierWeight blends the classifier's model probability into the score", () => {
	const dataset = metricFixture();
	// model-a is the quality winner, model-b the cost winner. The classifier
	// strongly prefers model-b and assigns model-a nothing.
	const support: Record<string, number> = { "model-a": 0, "model-b": 1, "model-c": 0.5 };
	const run = (classifierWeight: number) =>
		scoreEligible({
			eligible: ENTRIES,
			category: "coding",
			complexity: "moderate",
			config: scoringConfig({ classifierWeight }),
			dataset,
			estimatedCost: est,
			classifierProbability: (entry) => support[entry.id] ?? 0,
		});

	// Legacy default (0): benchmark-only, quality wins.
	const legacy = run(0);
	assert.equal(legacy.ranked[0].entry.id, "model-a", "classifierWeight 0 keeps benchmark-only ranking");
	assert.equal(legacy.ranked[0].support, 0, "no classifier signal contributes no support");

	// A confident classifier pick overturns the benchmark-only winner.
	const blended = run(0.5);
	assert.equal(blended.ranked[0].entry.id, "model-b", "classifier preference wins once blended");

	const byId = new Map(blended.ranked.map((r) => [r.entry.id, r]));
	assert.equal(byId.get("model-a")!.support, 0, "zero probability gets zero support");
	assert.equal(byId.get("model-b")!.support, 1, "eligible argmax normalizes to 1");
	assert.equal(byId.get("model-c")!.support, 0.5, "relative preference is preserved");
});

test("classifierWeight rescales without a signal; scores stay in 0..1 and order holds", () => {
	const dataset = metricFixture();
	const scoresFor = (classifierWeight: number) =>
		scoreEligible({
			eligible: ENTRIES,
			category: "coding",
			complexity: "moderate",
			config: scoringConfig({ classifierWeight }),
			dataset,
			estimatedCost: est,
		});
	const ids = (w: number) => scoresFor(w).ranked.map((r) => r.entry.id);
	const scores = (w: number) => scoresFor(w).ranked.map((r) => r.score);

	for (const w of [0, 0.35, 0.9]) {
		for (const s of scores(w)) assert.ok(s >= 0 && s <= 1, `score ${s} in 0..1 at classifierWeight=${w}`);
	}
	// No classifierProbability supplied: the weights only rescale, so ordering
	// must be identical to the legacy benchmark-only ranking.
	assert.deepEqual(ids(0.9), ids(0), "scaling alone never reorders");
});

test("the reported Jev case: a zero-probability model no longer wins on benchmarks alone", () => {
	const cfg = validateConfig(JSON.parse(readFileSync(new URL("../triage-router.json", import.meta.url), "utf8")));
	assert.deepEqual(cfg.problems, [], "shipped config validates");
	assert.ok(cfg.scoring.classifierWeight > 0, "shipped config opts into the blend");
	const eligible = cfg.models.filter((m) => m.enabled !== false);
	const dataset = readDataset(cfg.benchmarks.cachePath, cfg.benchmarks.seedPath);
	assert.ok(dataset, "benchmark cache present");

	// Exact answers from a real typesafe/jev-1.13 /systemone response.
	const model: Record<string, number> = {
		"litellm/deepseek/deepseek-v4-flash": 0.54,
		"litellm/z-ai/glm-5.3-flash": 0.44,
		"litellm/deepseek/deepseek-v4-pro": 0.01,
		"litellm/deepseek/deepseek-v4.1-flash": 0.01,
	};
	const decision = decide(
		{
			model,
			category: { debugging: 0.81, infra: 0.18, coding: 0.01 },
			complexity: { standard: 0.94, moderate: 0.04, complex: 0.02 },
		} as any,
		cfg,
		eligible,
		{
			dataset,
			estimatedCost: est,
			capabilities: (entry: ModelEntry) => entry.capabilities,
		} as any,
	);

	assert.equal(decision.rule, "scored");
	assert.notEqual(
		decision.entry?.id,
		"xiaomi/mimo-v2.6-pro",
		"regression: Jev gave mimo-v2.6-pro 0.0, so benchmark quality alone must not route to it",
	);
	// confidence stays null for scored picks: the classifier did not make this
	// decision, so its probability must not masquerade as decision confidence.
	assert.equal(decision.confidence, null);
	assert.ok(
		decision.notes.some((n) => n.startsWith("scored pick score=")),
		"the routed pick's score is reported via notes",
	);
	assert.ok(
		model[`${decision.entry?.provider}/${decision.entry?.id}`] > 0,
		"the routed model is one the classifier actually endorsed",
	);
});

test("scored pick reports confidence null: the classifier did not make the pick", () => {
	const cfg = scoringConfig();
	const dataset = metricFixture();
	const decision = decide(
		{
			model: { "litellm/model-a": 0.9, "litellm/model-b": 0.05, "litellm/model-c": 0.05 },
			category: { coding: 1 },
			complexity: { moderate: 1 },
		} as any,
		cfg,
		ENTRIES,
		{
			dataset,
			estimatedCost: est,
			capabilities: (entry: ModelEntry) => entry.capabilities,
		} as any,
	);
	assert.equal(decision.rule, "scored");
	assert.equal(decision.confidence, null, "classifier probability must not pose as scored-pick confidence");
	assert.ok(decision.notes.some((n) => n.startsWith("scored pick score=")), "the actual score is in the notes");
});

test("benchmarkProxyUrl: \"direct\" means never proxy even when settings/env proxies are set", () => {
	const env = (name: string) => (name === "https_proxy" ? "http://env-proxy:3128" : undefined);
	assert.equal(benchmarkProxyUrl("direct", env), null);
	assert.equal(benchmarkProxyUrl("", env), null);
	assert.equal(benchmarkProxyUrl("http://explicit:8080", env), "http://explicit:8080");
});

test("parseDataset sanitizes corrupt metric rows so undefined/NaN never reach scoring", () => {
	const dataset = parseDataset({
		fetchedAt: "2026-01-01T00:00:00.000Z",
		source: "openrouter",
		sourceUrl: "https://openrouter.ai",
		asOf: null,
		models: {
			"litellm/a": {
				coding: 10,
				reasoning: "not-a-number",
				agentic: undefined,
				arena: 20,
				gpqa: Number.NaN,
				taskSuccess: null,
				costPerTask: 1.5,
				research: null,
				infra: null,
				updatedAt: "garbage",
				match: "exact",
				variants: 1,
				sources: ["artificial-analysis"],
			},
			"litellm/b": "not-an-object", // dropped
			"litellm/c": null, // dropped
		},
	});
	assert.ok(dataset, "dataset is structurally valid and parsed");
	const a = dataset.models["litellm/a"];
	assert.equal(a.coding, 10);
	assert.equal(a.reasoning, null);
	assert.equal(a.agentic, null);
	assert.equal(a.arena, 20);
	assert.equal(a.gpqa, null);
	assert.equal(a.costPerTask, 1.5);
	assert.equal(a.updatedAt, null);
	assert.equal(a.variants, 1);
	assert.equal(a.sources.length, 1);
	assert.equal(dataset.models["litellm/b"], undefined);
	assert.equal(dataset.models["litellm/c"], undefined);
});

test("parseDataset rejects a cache whose fetchedAt is not a parseable date", () => {
	assert.equal(parseDataset({ fetchedAt: "not-a-date", source: "openrouter", models: {} }), null);
});

test("free (zero-price) models score 1.0 on the cost axis, not the missing-price neutral bucket", () => {
	const mk = () => ({
		coding: 60, reasoning: 60, agentic: null, arena: null, gpqa: null,
		taskSuccess: null, costPerTask: null, research: null, infra: null,
		updatedAt: "2026-10-01T00:00:00Z", match: "exact" as const, variants: 1, sources: ["test"],
	});
	const dataset: BenchmarkDataset = {
		fetchedAt: "2026-10-09T00:00:00Z",
		source: "test-fixture",
		sourceUrl: "tests (synthetic)",
		asOf: "2026-10-09T00:00:00Z",
		models: { free: mk(), cheap: mk(), pricey: mk() },
	};
	const entries: ModelEntry[] = [
		{ provider: "litellm", id: "free", description: "free", priceIn: 0, priceOut: 0 },
		{ provider: "litellm", id: "cheap", description: "cheap", priceIn: 0.01, priceOut: 0.02 },
		{ provider: "litellm", id: "pricey", description: "pricey", priceIn: 0.5, priceOut: 1 },
	];
	const result = scoreEligible({
		eligible: entries, category: "question", complexity: "standard",
		config: scoringConfig(), dataset, estimatedCost: est,
	});
	const costOf = Object.fromEntries(result.ranked.map((r) => [r.entry.id, r.cost]));
	assert.equal(costOf["free"], 1, "free is the cheapest tier");
	assert.equal(costOf["cheap"], 1, "cheapest paid model ties at the axis cap");
	assert.ok(costOf["pricey"] < 1, "pricier model scores below the cheapest");
});

test("loadConfig caches per path, not mtime alone", () => {
	const dir2 = mkdtempSync(join(tmpdir(), "laya-cfg-"));
	try {
		const a = join(dir2, "a.json");
		const b = join(dir2, "b.json");
		writeFileSync(a, JSON.stringify({ models: [{ provider: "p", id: "m-a", description: "a" }] }));
		writeFileSync(b, JSON.stringify({ models: [{ provider: "p", id: "m-b", description: "b" }] }));
		const t = new Date("2026-01-01T00:00:00Z").getTime();
		utimesSync(a, t, t);
		utimesSync(b, t, t);
		assert.equal(loadConfig(a).models[0].id, "m-a");
		assert.equal(loadConfig(b).models[0].id, "m-b", "same mtime but different path must not collide");
	} finally {
		rmSync(dir2, { recursive: true, force: true });
	}
});

test("research metric is an equal-weight mean, independent of row order", () => {
	const types = ["search_hle", "search_browsecomp", "search_qa"];
	const rowsFor = (values: [number, number, number]) =>
		types.map((benchmark_type, i) => ({
			source: "openrouter",
			model_permaslug: "order/model",
			benchmark_type,
			accuracy: values[i],
			last_run_timestamp: "2026-09-01T00:00:00Z",
		}));
	const a = normalizeUnified(rowsFor([0.9, 0.5, 0.1]), META, "2026-10-10T00:00:00Z", "https://openrouter.ai/api/v1/benchmarks", ["order/model"]);
	const b = normalizeUnified(rowsFor([0.1, 0.5, 0.9]), META, "2026-10-10T00:00:00Z", "https://openrouter.ai/api/v1/benchmarks", ["order/model"]);
	assert.equal(a.models["order/model"].research, 0.5, "equal weight per search_* type");
	assert.equal(a.models["order/model"].research, b.models["order/model"].research, "row order must not change the mean");
});
