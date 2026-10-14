/**
 * Benchmark ingestion, caching, and deterministic scoring for the triage-router
 * extension.
 *
 * Data sources (verified 2026-10-09, documented OpenRouter API surfaces):
 *  - `openrouter` source: GET https://openrouter.ai/api/v1/benchmarks
 *    (auth: any OpenRouter API key; 30 req/min, 500 req/day). Unified rows
 *    from Artificial Analysis (intelligence/coding/agentic indices),
 *    Design Arena (category ELOs incl. `codecategories`), and OpenRouter's
 *    own evals (gpqa_diamond, tau_bench_verified_airline with accuracy AND
 *    avg_cost_per_task, search_* retrieval evals).
 *  - `openrouter-models` source: GET https://openrouter.ai/api/v1/models
 *    (public, no key). Carries per-model Artificial Analysis indices,
 *    pricing, and context_length only - a keyless fallback with narrower
 *    coverage.
 *
 * Rules honored here:
 *  - No network per routing decision: routing reads the local cache; a
 *    refresh is scheduled in the background and never blocks a route.
 *  - Last known good dataset survives refresh failures (timeout, rate limit,
 *    malformed body, unavailability).
 *  - Missing values stay null. Scores are never invented or copied across
 *    model families (`glm-5.3-flash` can never lend its score to `glm-5.3`).
 *  - Slug resolution: exact permaslug, else a trailing release date
 *    (`-YYYYMMDD` / `-MMDD`, aligned to the dataset year) is stripped and
 *    matched; the newest dated release wins; same-release variant rows are
 *    averaged and counted. Provenance (`match`, `variants`) stays attached.
 *
 * This module type-imports shared types from triage-router.ts but imports no
 * runtime values from it (the router imports values from here).
 */
import { readFileSync, renameSync, statSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { dirname } from "node:path";
import type { Category, ModelEntry, RouterConfig } from "../triage-router.ts";
import { makeFetch, normalizeProxy, resolveApiKey, resolveProxyUrl, RemoteClassifierError } from "./triage-router-openrouter.ts";

export type FetchLike = Parameters<typeof makeFetch>[1];

// ---------------------------------------------------------------------------
// Configuration types
// ---------------------------------------------------------------------------

export interface BenchmarkConfig {
	/** Master switch. Default true (reading the cache is free; refresh is background). */
	enabled: boolean;
	/** Data source: "openrouter" (unified, key required) | "openrouter-models" (public). */
	source: "openrouter" | "openrouter-models";
	/** Cache file path. Default: `triage-router.benchmarks.json` in the Pi agent dir (PI_CODING_AGENT_DIR-aware), outside the package so `pi update` never wipes it. */
	cachePath: string;
	/** Internal: bundled snapshot read as a fallback until the cache file exists. Never set via config JSON. */
	seedPath?: string;
	/** Minimum age before a background refresh is attempted. Default 24h. */
	refreshIntervalMs: number;
	/** Datasets older than this are stale: handled per scoring.stalePolicy. Default 30d. */
	staleAfterMs: number;
	/** Per-fetch timeout. Default 15000. */
	timeoutMs: number;
	/** Environment variable holding the OpenRouter key (unified source). Default OPENROUTER_API_KEY. */
	apiKeyEnv: string;
	/** Optional HTTP(S) proxy for refresh fetches; falls back to env https_proxy. */
	proxyUrl: string | null;
}

export type MissingPolicy = "neutral" | "penalize";
export type StalePolicy = "use" | "missing";

export interface CategoryWeights {
	/** Weight of (normalized) task-relevant quality; cost gets 1 - quality. */
	quality: number;
	cost: number;
}

export interface ScoringConfig {
	/** Master switch. Default false: existing classifier-pick behavior is unchanged. */
	enabled: boolean;
	/** How a model with no usable quality data is scored. Default "neutral". */
	missingPolicy: MissingPolicy;
	/** With "penalize": quality *= 1 - penalty * (1 - coverage), coverage in [0,1]. Default 0.15. */
	missingPenalty: number;
	/** How stale rows are treated. Default "use" (used, flagged). */
	stalePolicy: StalePolicy;
	/** Expected input tokens for the estimated task cost. Default 4000. */
	expectedInputTokens: number;
	/** Expected output tokens for the estimated task cost. Default 1000. */
	expectedOutputTokens: number;
	/** Quality-weight shift for complexity: complex +delta, standard -delta. Default 0.15. */
	complexityQualityDelta: number;
	/** Per-task-category {quality, cost} weights. Quality+cost is renormalized to 1. */
	weights: Record<Category, CategoryWeights>;
	/** Share of the score given to the classifier's model probability (0..1).
	 *  Quality and cost are scaled by (1 - classifierWeight) so the three weights
	 *  still sum to 1 and the score stays in 0..1. 0 = benchmark-only selection,
	 *  the legacy behavior. Default 0. */
	classifierWeight: number;
}

/** Per-entry benchmark override from the config (measured internal results win over the cache). */
export interface CapabilityOverrides {
	coding?: number | null;
	reasoning?: number | null;
	agentic?: number | null;
	/** Internal infrastructure evaluation, 0..100. No public benchmark maps to infra. */
	infra?: number | null;
	taskSuccess?: number | null;
	research?: number | null;
}

/** Default task-specific weights: quality vs estimated cost per task. */
export const DEFAULT_SCORING_WEIGHTS: Record<Category, CategoryWeights> = {
	// Routine coding: relevant coding benchmarks dominate, cost matters.
	coding: { quality: 0.65, cost: 0.35 },
	// Hard debugging: quality dominates.
	debugging: { quality: 0.75, cost: 0.25 },
	// Infrastructure: technical competence with real cost awareness.
	infra: { quality: 0.6, cost: 0.4 },
	// Complex reasoning: quality dominates.
	reasoning: { quality: 0.8, cost: 0.2 },
	// Long agentic work: success-per-cost focus.
	long_horizon: { quality: 0.7, cost: 0.3 },
	// Research/retrieval: quality moderate, cost moderate.
	research: { quality: 0.6, cost: 0.4 },
	// Simple questions: adequate quality at low total cost.
	question: { quality: 0.3, cost: 0.7 },
	// Budget-tagged work: cost dominates.
	budget: { quality: 0.2, cost: 0.8 },
};

/**
 * Quality plan per category: which measured metrics blend into "quality" and
 * with what relative weight. Weights are renormalized over the metrics a
 * model actually has (partial coverage is explicit, never zero-filled).
 * Metrics within a plan come from different methodologies (AA index vs arena
 * ELO vs task accuracy) and are normalized across the eligible set before
 * blending - raw percentages and ELOs are never mixed directly.
 */
export const QUALITY_PLAN: Record<Category, Partial<Record<MetricKey, number>>> = {
	coding: { coding: 0.7, arena: 0.3 },
	debugging: { coding: 0.6, reasoning: 0.4 },
	infra: { infra: 0.6, coding: 0.4 },
	reasoning: { gpqa: 0.6, reasoning: 0.4 },
	long_horizon: { agentic: 0.5, taskSuccess: 0.3, coding: 0.2 },
	research: { research: 1 },
	question: { reasoning: 1 },
	budget: { reasoning: 1 },
};

// ---------------------------------------------------------------------------
// Dataset shape
// ---------------------------------------------------------------------------

export type MetricKey = "coding" | "reasoning" | "agentic" | "arena" | "gpqa" | "taskSuccess" | "research" | "infra";

export interface ModelBenchmarkMetrics {
	/** Artificial Analysis coding index (0..100). */
	coding: number | null;
	/** Artificial Analysis intelligence index (0..100). */
	reasoning: number | null;
	/** Artificial Analysis agentic index (0..100). */
	agentic: number | null;
	/** Design Arena `codecategories` ELO. */
	arena: number | null;
	/** OpenRouter gpqa_diamond accuracy (0..1). */
	gpqa: number | null;
	/** OpenRouter tau_bench_verified_airline accuracy (0..1) - task success. */
	taskSuccess: number | null;
	/** tau_bench avg cost per completed task, USD - real cost-per-success where measured. */
	costPerTask: number | null;
	/** Mean accuracy across OpenRouter search_* retrieval evals (0..1). */
	research: number | null;
	/** Internal-infra metric: never sourced from OpenRouter (config capabilities only). */
	infra: number | null;
	/** Newest per-row timestamp (ISO), or the dataset as_of for timestampless sources. */
	updatedAt: string | null;
	/** How the config id resolved to benchmark rows. */
	match: "exact" | "dated";
	/** Rows averaged behind this entry (same release, display variants). */
	variants: number;
	/** Sources that contributed at least one metric. */
	sources: string[];
}

export interface BenchmarkDataset {
	/** When this dataset was fetched (ISO). */
	fetchedAt: string;
	/** Source identifier (which endpoint produced it). */
	source: string;
	/** Documented endpoint the data came from. */
	sourceUrl: string;
	/** The source's own "as of" timestamp, when reported. */
	asOf: string | null;
	/** Model id -> metrics. */
	models: Record<string, ModelBenchmarkMetrics>;
}

function emptyMetrics(match: ModelBenchmarkMetrics["match"]): ModelBenchmarkMetrics {
	return {
		coding: null,
		reasoning: null,
		agentic: null,
		arena: null,
		gpqa: null,
		taskSuccess: null,
		costPerTask: null,
		research: null,
		infra: null,
		updatedAt: null,
		match,
		variants: 0,
		sources: [],
	};
}

// ---------------------------------------------------------------------------
// Slug resolution (exact, else dated; never family-prefix)
// ---------------------------------------------------------------------------

const DATED_SLUG_RE = /^(.*)-(\d{8}|\d{4})$/;

/** `-MMDD` suffixes must be real calendar dates, not arbitrary 4-digit tails
 * like `-2025` (a year) or `-1234` that would otherwise false-match. */
function isValidMMDD(digits: string): boolean {
	const month = Number(digits.slice(0, 2));
	const day = Number(digits.slice(2, 4));
	return month >= 1 && month <= 12 && day >= 1 && day <= 31;
}

/**
 * Resolve a config model id to a benchmark permaslug: exact match, else a
 * trailing release date is stripped (`-YYYYMMDD`, or `-MMDD` interpreted with
 * the dataset's year when known). Never a family prefix match, so
 * `z-ai/glm-5.3` can never be served `z-ai/glm-5.3-flash` rows.
 */
export function resolveSlug(id: string, slug: string, datasetYear?: number): "exact" | "dated" | null {
	if (slug === id) return "exact";
	const m = DATED_SLUG_RE.exec(slug);
	if (!m || m[1] !== id) return null;
	if (m[2].length === 8) return "dated";
	if (m[2].length === 4 && datasetYear && isValidMMDD(m[2])) return "dated";
	return null;
}

/** Sort key for dated slugs: bigger = newer. Exact (undated) is treated as newest-canonical. */
function slugDateRank(slug: string, datasetYear: number): number {
	const m = DATED_SLUG_RE.exec(slug);
	if (!m) return Number.MAX_SAFE_INTEGER; // exact/undated slug wins
	if (m[2].length === 8) return Number(m[2]);
	return datasetYear * 10000 + Number(m[2]); // -MMDD interpreted in dataset year
}

// ---------------------------------------------------------------------------
// Normalization: raw API rows -> dataset
// ---------------------------------------------------------------------------

interface RawRow {
	source?: string;
	model_permaslug?: string;
	display_name?: string;
	intelligence_index?: number | null;
	coding_index?: number | null;
	agentic_index?: number | null;
	elo?: number | null;
	category?: string;
	benchmark_type?: string;
	accuracy?: number | null;
	avg_cost_per_task?: number | null;
	last_run_timestamp?: string;
	pricing?: { prompt?: string; completion?: string };
	[key: string]: unknown;
}

const AA_FIELDS: ["coding" | "reasoning" | "agentic", string][] = [
	["coding", "coding_index"],
	["reasoning", "intelligence_index"],
	["agentic", "agentic_index"],
];

const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const iso = (v: unknown): string | null => (typeof v === "string" && /^\d{4}-\d{2}-\d{2}/.test(v) ? v : null);

/** Dataset year used to interpret `-MMDD` slugs (from as_of, else the newest dated slug). */
function datasetYear(rows: RawRow[], asOf: string | null): number {
	if (asOf) return Number(asOf.slice(0, 4));
	let year = 0;
	for (const row of rows) {
		const m = DATED_SLUG_RE.exec(row.model_permaslug ?? "");
		if (m && m[2].length === 8) year = Math.max(year, Number(m[2].slice(0, 4)));
	}
	return year || new Date().getUTCFullYear();
}

/**
 * Group rows per (config id, group key): keep only rows resolving to the id,
 * then per group pick the winning slug (exact > newest dated) and average
 * same-release variant rows. Returns id -> groupKey -> winning rows.
 */
function collect(
	rows: RawRow[],
	ids: string[],
	year: number,
	groupOf: (row: RawRow) => string,
): Map<string, Map<string, RawRow[]>> {
	const out = new Map<string, Map<string, RawRow[]>>();
	for (const id of ids) {
		for (const row of rows) {
			const slug = row.model_permaslug;
			if (!slug || !resolveSlug(id, slug, year)) continue;
			const groups = out.get(id) ?? new Map<string, RawRow[]>();
			const key = groupOf(row);
			const bucket = groups.get(key) ?? [];
			bucket.push(row);
			groups.set(key, bucket);
			out.set(id, groups);
		}
	}
	for (const [, groups] of out) {
		for (const [key, bucket] of groups) {
			const bySlug = new Map<string, RawRow[]>();
			for (const row of bucket) {
				const list = bySlug.get(row.model_permaslug!) ?? [];
				list.push(row);
				bySlug.set(row.model_permaslug!, list);
			}
			let bestSlug = "";
			let bestRank = -1;
			for (const slug of bySlug.keys()) {
				const rank = slug === undefined ? -1 : slugDateRank(slug, year);
				if (rank > bestRank) {
					bestRank = rank;
					bestSlug = slug;
				}
			}
			groups.set(key, bySlug.get(bestSlug) ?? []);
		}
	}
	return out;
}

/** Build a dataset for the given model ids from unified-endpoint rows. */
export function normalizeUnified(
	rows: RawRow[],
	meta: { as_of?: string | null } | null,
	fetchedAt: string,
	sourceUrl: string,
	ids: string[],
): BenchmarkDataset {
	const asOf = typeof meta?.as_of === "string" ? meta.as_of : null;
	const year = datasetYear(rows, asOf);
	const models: Record<string, ModelBenchmarkMetrics> = {};
	for (const id of ids) models[id] = emptyMetrics("exact");

	// Winner-slug groups per (id, group).
	const groupRows = collect(
		rows.filter((r) => r.source === "artificial-analysis" && typeof r.model_permaslug === "string"),
		ids,
		year,
		() => "aa",
	);
	mergeAA(groupRows, models, asOf);

	const arenaRows = collect(
		rows.filter((r) => r.source === "design-arena" && r.category === "codecategories"),
		ids,
		year,
		() => "code",
	);
	mergeArena(arenaRows, models, asOf);

	const orRows = collect(
		rows.filter((r) => r.source === "openrouter"),
		ids,
		year,
		(r) => r.benchmark_type ?? "unknown",
	);
	mergeOpenrouter(orRows, models, asOf);

	return { fetchedAt, source: "openrouter", sourceUrl, asOf, models };
}

function noteSource(m: ModelBenchmarkMetrics, source: string): void {
	if (!m.sources.includes(source)) m.sources.push(source);
}

function mergeAA(
	groups: Map<string, Map<string, RawRow[]>>,
	models: Record<string, ModelBenchmarkMetrics>,
	asOf: string | null,
): void {
	for (const [id, byGroup] of groups) {
		const rows = byGroup.get("aa") ?? [];
		if (!rows.length) continue;
		const target = models[id];
		target.match = DATED_SLUG_RE.test(rows[0].model_permaslug ?? "") ? "dated" : "exact";
		target.variants = Math.max(target.variants, rows.length);
		for (const [field, key] of AA_FIELDS) {
			const values = rows.map((r) => num(r[key])).filter((v): v is number => v !== null);
			if (values.length) target[field] = values.reduce((a, b) => a + b, 0) / values.length;
		}
		const stamps = rows.map((r) => iso(r.last_run_timestamp)).filter((s): s is string => s !== null);
		if (stamps.length) target.updatedAt = stamps.sort().at(-1) ?? target.updatedAt;
		else if (asOf) target.updatedAt = target.updatedAt ?? asOf;
		noteSource(target, "artificial-analysis");
	}
}

function mergeArena(
	groups: Map<string, Map<string, RawRow[]>>,
	models: Record<string, ModelBenchmarkMetrics>,
	asOf: string | null,
): void {
	for (const [id, byGroup] of groups) {
		const rows = byGroup.get("code") ?? [];
		if (!rows.length) continue;
		const target = models[id];
		target.match = target.sources.length ? target.match : DATED_SLUG_RE.test(rows[0].model_permaslug ?? "") ? "dated" : "exact";
		target.variants = Math.max(target.variants, rows.length);
		const elos = rows.map((r) => num(r.elo)).filter((v): v is number => v !== null);
		if (elos.length) target.arena = elos.reduce((a, b) => a + b, 0) / elos.length;
		if (asOf) target.updatedAt = target.updatedAt ?? asOf;
		noteSource(target, "design-arena");
	}
}

function mergeOpenrouter(
	groups: Map<string, Map<string, RawRow[]>>,
	models: Record<string, ModelBenchmarkMetrics>,
	asOf: string | null,
): void {
	for (const [id, byGroup] of groups) {
		const researchMeans: number[] = [];
		for (const [benchmarkType, rows] of byGroup) {
			if (!rows.length) continue;
			const target = models[id];
			const accuracies = rows.map((r) => num(r.accuracy)).filter((v): v is number => v !== null);
			const costs = rows.map((r) => num(r.avg_cost_per_task)).filter((v): v is number => v !== null);
			const stamps = rows.map((r) => iso(r.last_run_timestamp)).filter((s): s is string => s !== null);
			const stamp = stamps.sort().at(-1) ?? asOf;
			if (benchmarkType === "gpqa_diamond") {
				if (accuracies.length) target.gpqa = accuracies.at(-1)!;
				target.match = target.sources.length ? target.match : DATED_SLUG_RE.test(rows[0].model_permaslug ?? "") ? "dated" : "exact";
				if (stamp) target.updatedAt = target.updatedAt && target.updatedAt > stamp ? target.updatedAt : stamp;
				noteSource(target, "openrouter");
			} else if (benchmarkType === "tau_bench_verified_airline") {
				if (accuracies.length) target.taskSuccess = accuracies.at(-1)!;
				if (costs.length) target.costPerTask = costs.at(-1)!;
				target.match = target.sources.length ? target.match : DATED_SLUG_RE.test(rows[0].model_permaslug ?? "") ? "dated" : "exact";
				if (stamp) target.updatedAt = target.updatedAt && target.updatedAt > stamp ? target.updatedAt : stamp;
				noteSource(target, "openrouter");
			} else if (benchmarkType.startsWith("search_")) {
				if (accuracies.length) {
					researchMeans.push(accuracies.reduce((a, b) => a + b, 0) / accuracies.length);
				}
				target.match = target.sources.length ? target.match : DATED_SLUG_RE.test(rows[0].model_permaslug ?? "") ? "dated" : "exact";
				if (stamp) target.updatedAt = target.updatedAt && target.updatedAt > stamp ? target.updatedAt : stamp;
				noteSource(target, "openrouter");
			}
		}
		// Equal weight per search_* benchmark type, independent of row order.
		if (researchMeans.length) {
			models[id].research = researchMeans.reduce((a, b) => a + b, 0) / researchMeans.length;
		}
	}
}

/** Build a dataset from the public models catalog (keyless fallback: AA indices only). */
export function normalizeModelsCatalog(
	catalogRows: Array<{ id?: string; benchmarks?: { artificial_analysis?: Record<string, unknown> } }>,
	fetchedAt: string,
	sourceUrl: string,
	ids: string[],
): BenchmarkDataset {
	const models: Record<string, ModelBenchmarkMetrics> = {};
	for (const id of ids) {
		const target = emptyMetrics("exact");
		const row = catalogRows.find((r) => r.id === id);
		const aa = row?.benchmarks?.artificial_analysis;
		if (row && aa) {
			target.coding = num(aa.coding_index);
			target.reasoning = num(aa.intelligence_index);
			target.agentic = num(aa.agentic_index);
			if (target.coding !== null || target.reasoning !== null || target.agentic !== null) {
				target.sources.push("artificial-analysis");
			}
		}
		models[id] = target;
	}
	return { fetchedAt, source: "openrouter-models", sourceUrl, asOf: null, models };
}

// ---------------------------------------------------------------------------
// Cache I/O + refresh
// ---------------------------------------------------------------------------

/** Minimal dataset validation: a corrupt cache file is discarded, not trusted. */
export function parseDataset(raw: unknown): BenchmarkDataset | null {
	if (typeof raw !== "object" || raw === null) return null;
	const d = raw as Partial<BenchmarkDataset>;
	if (typeof d.fetchedAt !== "string" || typeof d.source !== "string" || typeof d.models !== "object" || !d.models) {
		return null;
	}
	if (!Number.isFinite(Date.parse(d.fetchedAt))) return null;
	// Sanitize per-model rows: a corrupt or hand-edited cache must never feed
	// non-numbers into scoring (undefined/NaN would poison normalize() into NaN
	// scores). Invalid numeric fields become null; invalid rows are dropped.
	const cleanModels: Record<string, ModelBenchmarkMetrics> = {};
	for (const [id, entry] of Object.entries(d.models)) {
		if (typeof entry !== "object" || entry === null) continue;
		const src = entry as unknown as Record<string, unknown>;
		cleanModels[id] = {
			coding: num(src.coding),
			reasoning: num(src.reasoning),
			agentic: num(src.agentic),
			arena: num(src.arena),
			gpqa: num(src.gpqa),
			taskSuccess: num(src.taskSuccess),
			costPerTask: num(src.costPerTask),
			research: num(src.research),
			infra: num(src.infra),
			updatedAt:
				typeof src.updatedAt === "string" && Number.isFinite(Date.parse(src.updatedAt)) ? src.updatedAt : null,
			match: src.match === "dated" ? "dated" : "exact",
			variants: typeof src.variants === "number" && Number.isFinite(src.variants) ? src.variants : 0,
			sources: Array.isArray(src.sources) ? src.sources.filter((s): s is string => typeof s === "string") : [],
		};
	}
	return { ...d, models: cleanModels } as BenchmarkDataset;
}

// mtime-cached reads: routing may consult the dataset twice per request.
let readCache: { path: string; mtimeMs: number; dataset: BenchmarkDataset | null } | null = null;

export function readDataset(cachePath: string, seedPath?: string): BenchmarkDataset | null {
	const dataset = readOne(cachePath);
	if (dataset) return dataset;
	// First reads on a packaged install: the agent-dir cache does not exist yet,
	// so serve the snapshot bundled with the package instead of nothing.
	if (seedPath && seedPath !== cachePath) return readOne(seedPath);
	return null;
}

function readOne(cachePath: string): BenchmarkDataset | null {
	try {
		if (!existsSync(cachePath)) return null;
		const mtimeMs = statSync(cachePath).mtimeMs;
		if (readCache && readCache.path === cachePath && readCache.mtimeMs === mtimeMs) return readCache.dataset;
		const dataset = parseDataset(JSON.parse(readFileSync(cachePath, "utf8")));
		readCache = { path: cachePath, mtimeMs, dataset };
		return dataset;
	} catch {
		return null;
	}
}

export function resetDatasetReadCacheForTests(): void {
	readCache = null;
}

/** Write atomically (tmp + rename) so a crash never leaves a half-written cache. */
export function writeDataset(cachePath: string, dataset: BenchmarkDataset): void {
	mkdirSync(dirname(cachePath), { recursive: true });
	const tmp = `${cachePath}.tmp`;
	writeFileSync(tmp, JSON.stringify(dataset, null, "\t"));
	renameSync(tmp, cachePath);
}

export interface RefreshDeps {
	fetchImpl?: FetchLike;
	now?: () => number;
	/** Overridable for tests: reads the environment. */
	env?: (name: string) => string | undefined;
}

export interface RefreshResult {
	ok: boolean;
	dataset: BenchmarkDataset | null;
	error: string | null;
}

const BENCH_BASE = "https://openrouter.ai";

async function httpGetJson(
	url: string,
	opts: { timeoutMs: number; proxyUrl: string | null; apiKey?: string; fetchImpl?: FetchLike; maxAttempts?: number },
): Promise<unknown> {
	const doFetch = await makeFetch(opts.proxyUrl, opts.fetchImpl);
	const maxAttempts = Math.max(1, opts.maxAttempts ?? 2);
	let lastMessage = "fetch failed";
	for (let attempt = 1; attempt <= maxAttempts; attempt++) {
		const controller = new AbortController();
		const timer = setTimeout(() => controller.abort(), opts.timeoutMs);
		try {
			const headers: Record<string, string> = { Accept: "application/json" };
			if (opts.apiKey) headers.Authorization = `Bearer ${opts.apiKey}`;
			const response = await doFetch(url, { headers, signal: controller.signal });
			if (!response.ok) {
				lastMessage = `HTTP ${response.status}`;
				const retryable = response.status === 429 || response.status >= 500;
				if (retryable && attempt < maxAttempts) continue;
				throw new Error(lastMessage);
			}
			return await response.json();
		} catch (err) {
			lastMessage = err instanceof Error ? err.message : String(err);
			const aborted = err instanceof Error && err.name === "AbortError";
			if (aborted) lastMessage = `timeout after ${opts.timeoutMs}ms`;
			const networkLevel = !(err instanceof Error && /^HTTP \d+$/.test(err.message));
			if (networkLevel && attempt < maxAttempts) continue;
			throw new Error(lastMessage);
		} finally {
			clearTimeout(timer);
		}
	}
	throw new Error(lastMessage);
}

/**
 * Fetch + normalize + persist. On any failure the previous cache is kept and
 * the error is returned (never throws): callers log and continue routing.
 */
export function benchmarkProxyUrl(
	proxyUrl: string | null,
	env: (name: string) => string | undefined = (name) => process.env[name],
): string | null {
	return proxyUrl != null ? normalizeProxy(proxyUrl) : resolveProxyUrl(null, env);
}

export async function refreshBenchmarks(
	config: BenchmarkConfig,
	ids: string[],
	deps: RefreshDeps = {},
): Promise<RefreshResult> {
	const env = deps.env ?? ((name: string) => process.env[name]);
	// Proxy: explicit "direct" means never; otherwise settings.json httpProxy,
	// then env proxies (shared resolution with the classifier backends).
	const proxyUrl = benchmarkProxyUrl(config.proxyUrl, env);
	try {
		let dataset: BenchmarkDataset;
		if (config.source === "openrouter-models") {
			const url = `${BENCH_BASE}/api/v1/models`;
			const payload = (await httpGetJson(url, {
				timeoutMs: config.timeoutMs,
				proxyUrl,
				fetchImpl: deps.fetchImpl,
			})) as { data?: unknown[] };
			if (!Array.isArray(payload.data)) throw new Error("models response missing `data` array");
			dataset = normalizeModelsCatalog(
				payload.data as never,
				new Date().toISOString(),
				url,
				ids,
			);
		} else {
			// Unified endpoint is OpenRouter's: env var first, then the Pi auth
			// file's "openrouter" entry (same resolution as the classifier).
			const key = env(config.apiKeyEnv) ?? resolveApiKey(config.apiKeyEnv, "openrouter");
			if (!key) {
				throw new RemoteClassifierError(
					`benchmark refresh requires ${config.apiKeyEnv} or an "openrouter" key in the Pi auth file (~/.pi/agent/auth.json) for the unified OpenRouter benchmark endpoint`,
				);
			}
			const url = `${BENCH_BASE}/api/v1/benchmarks`;
			const payload = (await httpGetJson(url, {
				timeoutMs: config.timeoutMs,
				proxyUrl,
				apiKey: key,
				fetchImpl: deps.fetchImpl,
			})) as { data?: unknown[]; meta?: { as_of?: string } };
			if (!Array.isArray(payload.data)) throw new Error("benchmarks response missing `data` array");
			dataset = normalizeUnified(
				payload.data as RawRow[],
				payload.meta ?? null,
				new Date().toISOString(),
				url,
				ids,
			);
		}
		writeDataset(config.cachePath, dataset);
		return { ok: true, dataset, error: null };
	} catch (err) {
		return { ok: false, dataset: readDataset(config.cachePath, config.seedPath), error: err instanceof Error ? err.message : String(err) };
	}
}

// Module-level refresh bookkeeping (single-flight, hot-reload safe).
let refreshInFlight: Promise<void> | null = null;

export function resetRefreshStateForTests(): void {
	refreshInFlight = null;
}

/**
 * Non-blocking refresh trigger: schedules a background refresh when the cache
 * is older than the interval (or absent). Never awaited by routing; failures
 * log and keep the last known good dataset.
 */
export function scheduleRefresh(
	config: BenchmarkConfig,
	ids: string[],
	deps: RefreshDeps = {},
): Promise<void> | null {
	if (!config.enabled) return null;
	if (refreshInFlight) return refreshInFlight;
	const now = deps.now ?? Date.now;
	try {
		const stat = statSync(config.cachePath);
		if (now() - stat.mtimeMs < config.refreshIntervalMs) return null;
	} catch {
		// No cache yet: refresh now.
	}
	refreshInFlight = refreshBenchmarks(config, ids, deps)
		.then((result) => {
			if (result.ok) {
				console.error(
					`[pi-triage-router] benchmarks refreshed source=${config.source} models=${Object.keys(result.dataset?.models ?? {}).length} as_of=${result.dataset?.asOf ?? "-"}`,
				);
			} else {
				console.error(
					`[pi-triage-router] benchmarks refresh failed (${result.error}); keeping last known good cache`,
				);
			}
		})
		.finally(() => {
			refreshInFlight = null;
		});
	return refreshInFlight;
}

/** Explicit refresh for diagnostics/tests. Bypasses the interval, still single-flight. */
export async function refreshNow(
	config: BenchmarkConfig,
	ids: string[],
	deps: RefreshDeps = {},
): Promise<RefreshResult> {
	while (refreshInFlight) await refreshInFlight;
	return refreshBenchmarks(config, ids, deps);
}

/** Dataset age in ms (null when no readable cache). */
export function datasetAgeMs(cachePath: string, now = Date.now): number | null {
	try {
		const dataset = readDataset(cachePath);
		if (!dataset) return null;
		const fetched = Date.parse(dataset.fetchedAt);
		return Number.isFinite(fetched) ? Math.max(0, now() - fetched) : null;
	} catch {
		return null;
	}
}

/** Freshness/coverage summary for diagnostics logging. */
export function datasetStatus(config: BenchmarkConfig, ids: string[]): string {
	const dataset = readDataset(config.cachePath, config.seedPath);
	if (!dataset) return "bench=absent";
	const ageMs = Math.max(0, Date.now() - Date.parse(dataset.fetchedAt));
	const covered = ids.filter((id) => {
		const m = dataset.models[id];
		return !!m && ["coding", "reasoning", "agentic", "gpqa", "taskSuccess", "research", "arena"].some((k) => m[k as MetricKey] !== null);
	}).length;
	const stale = ageMs > config.staleAfterMs ? " stale" : "";
	return `bench=${covered}/${ids.length} age=${Math.round(ageMs / 3_600_000)}h${stale} as_of=${dataset.asOf ?? dataset.fetchedAt.slice(0, 10)}`;
}

// ---------------------------------------------------------------------------
// Scoring
// ---------------------------------------------------------------------------

export interface ScoreInput {
	/** Eligible models (already filtered by enabled/registry/window rules). */
	eligible: ModelEntry[];
	category: Category;
	complexity: "standard" | "moderate" | "complex";
	config: RouterConfig;
	dataset: BenchmarkDataset | null;
	/** Estimated total task cost in USD for an entry (null = unknown price). */
	estimatedCost: (entry: ModelEntry) => number | null;
	/** Per-entry config capabilities (measured internal results). */
	capabilities?: (entry: ModelEntry) => CapabilityOverrides | undefined;
	/** Raw classifier probability for an entry. Normalized against the eligible
	 *  argmax before use, so only relative preference matters. Absent = no
	 *  classifier signal and ranking is benchmark-only. */
	classifierProbability?: (entry: ModelEntry) => number;
	now?: number;
}

export interface ScoredEntry {
	entry: ModelEntry;
	score: number;
	quality: number;
	cost: number;
	/** Classifier support in 0..1: this entry's probability relative to the eligible argmax. */
	support: number;
	coverage: number;
	stale: boolean;
	notes: string[];
}

export interface ScoreResult {
	/** Ranked best-first. Empty when quality data is unusable for every model. */
	ranked: ScoredEntry[];
	missingUsed: boolean;
}

/**
 * Min-max normalize a metric across the eligible set. Degenerate (all equal
 * or all missing) maps to 0.5 for everyone - a documented neutral, never 0.
 */
function normalize(values: Map<string, number>): Map<string, number> {
	const out = new Map<string, number>();
	const list = [...values.values()];
	if (list.length === 0) return out;
	const min = Math.min(...list);
	const max = Math.max(...list);
	if (min === max) {
		for (const key of values.keys()) out.set(key, 0.5);
		return out;
	}
	for (const [key, value] of values) out.set(key, (value - min) / (max - min));
	return out;
}

/**
 * Deterministic scoring policy: blend normalized, task-relevant metrics into
 * quality, blend estimated task cost into cost, apply per-category weights
 * (complexity-adjusted), handle missing/stale data per policy, and rank.
 * Pure and fully deterministic - the only randomness is none.
 */
export function scoreEligible(input: ScoreInput): ScoreResult {
	const { eligible, category, complexity, config } = input;
	const scoring = config.scoring;
	const now = input.now ?? Date.now();
	const plan = QUALITY_PLAN[category];
	let missingUsed = false;

	// 1. Per-metric raw values across eligible models (config capabilities win).
	const rawByMetric = new Map<MetricKey, Map<string, number>>();
	const metricValue = (entry: ModelEntry, metric: MetricKey): number | null => {
		const caps = input.capabilities?.(entry) as Record<string, number | null | undefined> | undefined;
		if (caps && caps[metric] !== undefined) return caps[metric]; // explicit override (may be null)
		const m = input.dataset?.models[entry.id];
		if (!m) return null;
		const stale =
			scoring.stalePolicy === "missing" &&
			m.updatedAt !== null &&
			now - Date.parse(m.updatedAt) > config.benchmarks.staleAfterMs;
		if (stale) return null;
		return m[metric];
	};
	for (const metric of Object.keys(plan) as MetricKey[]) {
		const values = new Map<string, number>();
		for (const entry of eligible) {
			const raw = metricValue(entry, metric);
			if (raw !== null) values.set(entry.id, raw);
		}
		rawByMetric.set(metric, values);
	}
	const normalized = new Map<MetricKey, Map<string, number>>();
	for (const [metric, values] of rawByMetric) normalized.set(metric, normalize(values));

	// 2. Per-model quality: weighted blend over available metrics (weights
	// renormalized to the coverage the model actually has).
	const qualities = new Map<string, { quality: number; coverage: number; stale: boolean }>();
	for (const entry of eligible) {
		const key = entry.id;
		let weightSum = 0;
		let blend = 0;
		let have = 0;
		let total = 0;
		let stale = false;
		const m = input.dataset?.models[key];
		if (m && scoring.stalePolicy === "use" && m.updatedAt !== null && now - Date.parse(m.updatedAt) > config.benchmarks.staleAfterMs) {
			stale = true;
		}
		for (const [metric, weight] of Object.entries(plan) as [MetricKey, number][]) {
			if (!weight) continue;
			total += weight;
			const n = normalized.get(metric)?.get(key);
			if (n === undefined) continue;
			have += weight;
			weightSum += weight;
			blend += n * weight;
		}
		if (weightSum > 0) {
			const coverage = total > 0 ? have / total : 0;
			let quality = blend / weightSum;
			if (scoring.missingPolicy === "penalize" && coverage < 1) {
				quality *= 1 - scoring.missingPenalty * (1 - coverage);
			}
			qualities.set(key, { quality, coverage, stale });
		}
	}
	if (qualities.size === 0) {
		// No measured quality for any eligible model: scoring cannot decide.
		return { ranked: [], missingUsed: false };
	}
	const neutralQuality =
		[...qualities.values()].reduce((sum, q) => sum + q.quality, 0) / Math.max(1, qualities.size);

	// 3. Per-model estimated cost, normalized so the cheapest eligible model
	// scores 1.0 (ratio, not min-max: cost differences are meaningful even
	// when models cluster). Missing price = missing data, per policy.
	const costs = new Map<string, number>();
	const rawCost = new Map<string, number>();
	const estByKey = new Map<string, number | null>();
	for (const entry of eligible) {
		const key = entry.id;
		const est = input.estimatedCost(entry);
		estByKey.set(key, est);
		if (est !== null && est > 0) {
			rawCost.set(key, est);
		}
	}
	const minCost = rawCost.size ? Math.min(...rawCost.values()) : null;
	for (const [key, est] of rawCost) costs.set(key, minCost! / est);
	// Free models (est === 0) are the cheapest tier: score 1.0 on the cost axis
	// instead of being dropped into the "missing price" neutral bucket.
	for (const [key, est] of estByKey) {
		if (est === 0) costs.set(key, 1);
	}
	const neutralCost = costs.size ? [...costs.values()].reduce((a, b) => a + b, 0) / costs.size : 0.5;
	if (costs.size < eligible.length) missingUsed = true;

	// 4. Classifier support (optional): the entry's probability relative to the
	// eligible argmax, so a confident classifier pick can move the ranking.
	// Absent or all-zero support contributes 0 to every entry, which leaves the
	// benchmark-only ranking exactly as it was.
	const probability = input.classifierProbability;
	let maxProbability = 0;
	if (probability) {
		for (const entry of eligible) maxProbability = Math.max(maxProbability, probability(entry));
	}
	const supportOf = (entry: ModelEntry): number =>
		probability && maxProbability > 0 ? probability(entry) / maxProbability : 0;

	// 5. Complexity-adjusted weights, then score. The classifier share scales
	// quality and cost down by (1 - classifierWeight), so the three weights sum
	// to 1 and the score stays in 0..1.
	const base = scoring.weights[category];
	const delta = scoring.complexityQualityDelta;
	const qualityWeight = Math.min(
		0.95,
		Math.max(
			0.05,
			base.quality + (complexity === "complex" ? delta : complexity === "standard" ? -delta : 0),
		),
	);
	const wClassifier = Math.min(0.95, Math.max(0, scoring.classifierWeight));
	const rest = 1 - wClassifier;
	const wQuality = qualityWeight * rest;
	const wCost = (1 - qualityWeight) * rest;

	const ranked: ScoredEntry[] = [];
	for (const entry of eligible) {
		const key = entry.id;
		const q = qualities.get(key);
		let quality = q ? q.quality : neutralQuality;
		const cost = costs.get(key) ?? neutralCost;
		if (!q) {
			missingUsed = true;
			if (scoring.missingPolicy === "penalize") quality *= 1 - scoring.missingPenalty;
		}
		const support = supportOf(entry);
		const score = wQuality * quality + wCost * cost + wClassifier * support;
		const entryNotes: string[] = [];
		if (!q) entryNotes.push(scoring.missingPolicy === "penalize" ? "no quality data (penalized)" : "no quality data: neutral quality");
		else if (q.stale) entryNotes.push("stale benchmark data (used, flagged)");
		else if (q.coverage < 1 && scoring.missingPolicy === "neutral") {
			entryNotes.push(`partial coverage ${(q.coverage * 100).toFixed(0)}% (neutral)`);
		} else if (q.coverage < 1) entryNotes.push(`partial coverage ${(q.coverage * 100).toFixed(0)}% (penalized)`);
		if (!costs.has(key)) entryNotes.push("missing price: neutral cost");
		ranked.push({
			entry,
			score: round6(score),
			quality: round6(quality),
			cost: round6(cost),
			support: round6(support),
			coverage: q ? q.coverage : 0,
			stale: q?.stale ?? false,
			notes: entryNotes,
		});
	}
	// Deterministic order: score desc, quality desc, price asc, config order.
	ranked.sort((a, b) => {
		if (b.score !== a.score) return b.score - a.score;
		if (b.quality !== a.quality) return b.quality - a.quality;
		const pa = a.entry.priceIn ?? Number.POSITIVE_INFINITY;
		const pb = b.entry.priceIn ?? Number.POSITIVE_INFINITY;
		if (pa !== pb) return pa - pb;
		return eligible.indexOf(a.entry) - eligible.indexOf(b.entry);
	});
	return { ranked, missingUsed };
}

function round6(v: number): number {
	return Number(v.toFixed(6));
}
