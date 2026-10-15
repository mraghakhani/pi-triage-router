/**
 * Price ingestion and caching for the triage-router extension.
 *
 * Source (verified OpenRouter API surface): GET https://openrouter.ai/api/v1/models
 * (public, no key). Each row carries `pricing: { prompt, completion }` as USD
 * per token; stored here as USD per 1M tokens, the unit the router scores with.
 *
 * Rules honored here (same contract as the benchmarks cache):
 *  - No network per routing decision: routing reads the local cache; a
 *    refresh is scheduled in the background and never blocks a route.
 *  - Last known good dataset survives refresh failures (timeout, rate
 *    limit, malformed body, unavailability).
 *  - Missing prices stay absent. Prices are never invented or copied across
 *    models; a model absent from the catalog simply has no cached price and
 *    falls back to the config `priceIn`/`priceOut`.
 *  - Precedence at lookup time (router): live registry cost → cached price →
 *    config `priceIn`/`priceOut`.
 *
 * This module imports shared helpers from triage-router-benchmarks.ts and
 * imports nothing from triage-router.ts (the router imports values from here).
 */
import { existsSync, mkdirSync, readFileSync, renameSync, statSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { benchmarkProxyUrl, httpGetJson, type FetchLike } from "./triage-router-benchmarks.ts";

/** USD per 1M tokens. */
export interface ModelPrice {
	input: number;
	output: number;
}

export interface PriceConfig {
	/** Master switch. Default true (reading the cache is free; refresh is background). */
	enabled: boolean;
	/** Cache file path. Default: `triage-router.prices.json` in the Pi agent dir, outside the package so `pi update` never wipes it. */
	cachePath: string;
	/** Internal: bundled snapshot read as a fallback until the cache file exists. Never set via config JSON. */
	seedPath?: string;
	/** Minimum age before a background refresh is attempted. Default 24h. */
	refreshIntervalMs: number;
	/** Per-fetch timeout. Default 15000. */
	timeoutMs: number;
	/** Optional HTTP(S) proxy for refresh fetches; falls back to env https_proxy. */
	proxyUrl: string | null;
}

export interface PriceDataset {
	fetchedAt: string;
	source: "openrouter-models";
	sourceUrl: string;
	/** USD per 1M tokens, keyed by model id (OpenRouter slug, e.g. "deepseek/deepseek-v4-flash"). */
	prices: Record<string, ModelPrice>;
}

export const MODELS_URL = "https://openrouter.ai/api/v1/models";

function usdPerMillion(perToken: unknown): number | null {
	const v = typeof perToken === "string" ? Number(perToken) : typeof perToken === "number" ? perToken : Number.NaN;
	if (!Number.isFinite(v) || v < 0) return null;
	return v * 1e6;
}

/** Build a dataset from the public models catalog, keeping only configured ids. */
export function normalizePrices(
	catalogRows: Array<{ id?: string; pricing?: { prompt?: unknown; completion?: unknown } }>,
	fetchedAt: string,
	ids: string[],
): PriceDataset {
	const prices: Record<string, ModelPrice> = {};
	for (const id of ids) {
		const row = catalogRows.find((r) => r.id === id);
		const input = usdPerMillion(row?.pricing?.prompt);
		const output = usdPerMillion(row?.pricing?.completion);
		if (input !== null && output !== null) prices[id] = { input, output };
	}
	return { fetchedAt, source: "openrouter-models", sourceUrl: MODELS_URL, prices };
}

/** Minimal dataset validation: a corrupt cache file is discarded, not trusted. */
export function parsePriceDataset(raw: unknown): PriceDataset | null {
	if (typeof raw !== "object" || raw === null) return null;
	const d = raw as Partial<PriceDataset>;
	if (typeof d.fetchedAt !== "string" || d.source !== "openrouter-models" || typeof d.prices !== "object" || !d.prices) {
		return null;
	}
	if (!Number.isFinite(Date.parse(d.fetchedAt))) return null;
	const prices: Record<string, ModelPrice> = {};
	for (const [id, price] of Object.entries(d.prices)) {
		if (typeof price !== "object" || price === null) continue;
		const p = price as Partial<ModelPrice>;
		if (typeof p.input !== "number" || typeof p.output !== "number") continue;
		if (!Number.isFinite(p.input) || !Number.isFinite(p.output) || p.input < 0 || p.output < 0) continue;
		prices[id] = { input: p.input, output: p.output };
	}
	return { fetchedAt: d.fetchedAt, source: "openrouter-models", sourceUrl: typeof d.sourceUrl === "string" ? d.sourceUrl : MODELS_URL, prices };
}

// Mtime-memoized reads: the cache is read per route, so the hot path must
// not re-parse JSON when the file is unchanged.
let readCache: { path: string; mtimeMs: number; dataset: PriceDataset | null } | null = null;

function readOne(cachePath: string): PriceDataset | null {
	try {
		if (!existsSync(cachePath)) return null;
		const mtimeMs = statSync(cachePath).mtimeMs;
		if (readCache && readCache.path === cachePath && readCache.mtimeMs === mtimeMs) return readCache.dataset;
		const dataset = parsePriceDataset(JSON.parse(readFileSync(cachePath, "utf8")));
		readCache = { path: cachePath, mtimeMs, dataset };
		return dataset;
	} catch {
		return null;
	}
}

export function readPriceDataset(cachePath: string, seedPath?: string): PriceDataset | null {
	const dataset = readOne(cachePath);
	if (dataset) return dataset;
	// First reads on a packaged install: the agent-dir cache does not exist yet,
	// so serve the snapshot bundled with the package instead of nothing.
	if (seedPath && seedPath !== cachePath) return readOne(seedPath);
	return null;
}

export function resetPriceCacheForTests(): void {
	readCache = null;
}

/** Write atomically (tmp + rename) so a crash never leaves a half-written cache. */
export function writePriceDataset(cachePath: string, dataset: PriceDataset): void {
	mkdirSync(dirname(cachePath), { recursive: true });
	const tmp = `${cachePath}.tmp`;
	writeFileSync(tmp, `${JSON.stringify(dataset, null, "\t")}\n`);
	renameSync(tmp, cachePath);
	readCache = null;
}

export interface PriceRefreshDeps {
	fetchImpl?: FetchLike;
	now?: () => number;
	env?: (name: string) => string | undefined;
}

export interface PriceRefreshResult {
	ok: boolean;
	dataset: PriceDataset | null;
	error: string | null;
}

/** Fetch + normalize + persist. On any failure the previous cache is kept and
 * the error is returned (never throws): callers log and continue routing. */
export async function refreshPrices(
	config: PriceConfig,
	ids: string[],
	deps: PriceRefreshDeps = {},
): Promise<PriceRefreshResult> {
	const env = deps.env ?? ((name: string) => process.env[name]);
	const proxyUrl = benchmarkProxyUrl(config.proxyUrl, env);
	try {
		const payload = (await httpGetJson(MODELS_URL, {
			timeoutMs: config.timeoutMs,
			proxyUrl,
			fetchImpl: deps.fetchImpl,
		})) as { data?: unknown[] };
		if (!Array.isArray(payload.data)) throw new Error("models response missing `data` array");
		const dataset = normalizePrices(payload.data as never, new Date().toISOString(), ids);
		writePriceDataset(config.cachePath, dataset);
		return { ok: true, dataset, error: null };
	} catch (err) {
		return { ok: false, dataset: readPriceDataset(config.cachePath, config.seedPath), error: err instanceof Error ? err.message : String(err) };
	}
}

// Module-level refresh bookkeeping (single-flight, hot-reload safe).
let refreshInFlight: Promise<void> | null = null;

export function resetPriceRefreshStateForTests(): void {
	refreshInFlight = null;
}

/**
 * Non-blocking refresh trigger: schedules a background refresh when the cache
 * is older than the interval (or absent). Never awaited by routing; failures
 * log and keep the last known good dataset.
 */
export function schedulePriceRefresh(
	config: PriceConfig,
	ids: string[],
	deps: PriceRefreshDeps = {},
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
	refreshInFlight = refreshPrices(config, ids, deps)
		.then((result) => {
			if (result.ok) {
				console.error(
					`[pi-triage-router] prices refreshed models=${Object.keys(result.dataset?.prices ?? {}).length}`,
				);
			} else {
				console.error(
					`[pi-triage-router] price refresh failed (${result.error}); keeping last known good cache`,
				);
			}
		})
		.finally(() => {
			refreshInFlight = null;
		});
	return refreshInFlight;
}