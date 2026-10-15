/**
 * Remote classifier client for the triage-router extension.
 *
 * Speaks the structured-question protocol shared by Jev (TypeSafe System One,
 * `typesafe/jev-1.13`) with two verified transports:
 * - `decisions`: POST /api/alpha/decisions on OpenRouter
 *   (docs: https://openrouter.ai/docs/guides/community/jev-tutorial)
 * - `systemone`: POST /systemone - TypeSafe's native protocol, served by
 *   OpenRouter (api/v1/systemone) and by LiteLLM proxies that front TypeSafe
 *   (verified: <litellm>/systemone and <litellm>/v1/systemone both answer).
 *
 * Both transports return EXACTLY the same answer shape as local Laya's
 * `systemOne` (`answers.<name>.probabilities`), so the router's questions,
 * validation, and deterministic decision rules are shared unchanged - this
 * module only owns transport, timeouts, bounded retries, structured-output
 * validation, and usage/cost capture.
 *
 * This module deliberately imports NOTHING from triage-router.ts (the router
 * imports values from here), so there is no runtime module cycle.
 *
 * Privacy: only the bounded routing input (current request + compact context)
 * is sent, and only when the user has explicitly selected the openrouter
 * classifier. Prompts/summaries and API keys are never logged.
 */

/** Per-call usage reported by the Decisions API (`usage` in the response). */
export interface RemoteUsage {
	inputTokens: number | null;
	outputTokens: number | null;
	/** Actual cost in USD when the provider reports it; null = unknown (never fabricated). */
	cost: number | null;
}

/** Result of one remote classification call. */
export interface RemoteCallResult {
	/** `answers.<question>.probabilities` maps, Laya-style. */
	answers: Record<string, { probabilities?: Record<string, number> } | undefined>;
	usage: RemoteUsage;
	/** Wall-clock latency of the call, ms. */
	latencyMs: number;
	/** Dated snapshot that served the request, when the response names one. */
	servedModel: string | null;
}

export type FetchLike = (url: string, init?: RequestInit) => Promise<{
	ok: boolean;
	status: number;
	json(): Promise<unknown>;
	text(): Promise<string>;
}>;

// Test seam: when set, all transport goes through this instead of
// global/undici fetch. Never set in production paths.
let fetchOverride: FetchLike | null = null;
export function setFetchForTests(fetchImpl: FetchLike | null): void {
	fetchOverride = fetchImpl;
}

export interface DecisionCallOptions {
	/** Remote decision model id, e.g. "typesafe/jev-1.13". */
	model: string;
	/** Role/prompt state (the bounded classification prompt). */
	role: string;
	prompt: string;
	/** The shared questions object (type/instructions/criteria per question). */
	questions: Record<string, unknown>;
	/** Per-attempt timeout in ms (each attempt aborts independently at this bound). */
	timeoutMs: number;
	/** OpenRouter API key. Never logged or embedded in error messages. */
	apiKey: string;
	/** Optional HTTP(S) proxy, e.g. "http://localhost:10808". */
	proxyUrl?: string | null;
	/** Full endpoint URL; defaults to the OpenRouter decisions URL. */
	endpoint?: string | null;
	/** Provider name used in error messages ("OpenRouter", "LiteLLM", ...). */
	providerLabel?: string;
	/** Injection seam for tests. Defaults to global fetch. */
	fetchImpl?: FetchLike;
	/** Total attempt cap (default 2: first attempt + one bounded retry). */
	maxAttempts?: number;
	/** Delay in ms between attempts (default 0). Applies only when retrying. */
	retryDelayMs?: number;
}

/** Thrown for every remote-classification failure; message never contains credentials. */
export class RemoteClassifierError extends Error {
	readonly status: number | null;
	constructor(message: string, status: number | null = null) {
		super(message);
		this.name = "RemoteClassifierError";
		this.status = status;
	}
}

export const DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions";
/** Default systemone transport on OpenRouter (pi-ai uses the same path). */
export const SYSTEMONE_URL = "https://openrouter.ai/api/v1/systemone";

export type DecisionProtocol = "decisions" | "systemone";

/**
 * Compose the classifier endpoint. With an explicit base URL (e.g. a LiteLLM
 * proxy), the protocol path is appended; without one, the OpenRouter default
 * for the protocol is used.
 */
export function classifierEndpoint(protocol: DecisionProtocol, baseUrl?: string | null): string {
	if (baseUrl && baseUrl.trim() !== "") {
		const base = baseUrl.trim().replace(/\/+$/u, "");
		return `${base}/${protocol === "decisions" ? "api/alpha/decisions" : "systemone"}`;
	}
	return protocol === "decisions" ? DECISIONS_URL : SYSTEMONE_URL;
}

/**
 * Build the fetch implementation for this process: plain global fetch when no
 * proxy is configured, undici fetch with a ProxyAgent dispatcher when one is.
 * Proxy support uses the `undici` dependency declared in this extension's
 * package.json (env-var proxying is a startup-time setting in Node and cannot
 * be opted into from inside a running process).
 */
/**
 * Build the fetch implementation for this process: plain global fetch when no
 * proxy is configured, undici fetch with a ProxyAgent dispatcher when one is.
 * Proxy support uses the `undici` dependency declared in this extension's
 * package.json (env-var proxying is a startup-time setting in Node and cannot
 * be opted into from inside a running process). Loopback targets (local
 * LiteLLM, llama.cpp, ...) NEVER go through the proxy.
 */
const LOOPBACK_HOSTS = new Set(["localhost", "127.0.0.1", "::1", "[::1]"]);

export function isLoopbackUrl(url: string): boolean {
	try {
		return LOOPBACK_HOSTS.has(new URL(url).hostname);
	} catch {
		return false;
	}
}

// ---------------------------------------------------------------------------
// Proxy + credentials resolution (shared by classifier and benchmark paths)
// ---------------------------------------------------------------------------

import { existsSync, readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

let settingsPathForTests: string | null = null;
let authFileForTests: string | null = null;
/** Test hook: point settings.json / auth.json lookups at temp files. */
export function setRuntimePathsForTests(paths: { settingsPath?: string | null; authFile?: string | null }): void {
	if ("settingsPath" in paths) settingsPathForTests = paths.settingsPath ?? null;
	if ("authFile" in paths) authFileForTests = paths.authFile ?? null;
}

export function defaultSettingsPath(): string {
	return settingsPathForTests ?? join(piAgentDir(), "settings.json");
}

export function defaultAuthFile(): string {
	return authFileForTests ?? join(piAgentDir(), "auth.json");
}

/**
 * Pi's agent dir. Default `~/.pi/agent`, overridable with the documented
 * `PI_CODING_AGENT_DIR` environment variable (pi docs: environment-variables).
 */
export function piAgentDir(): string {
	return process.env.PI_CODING_AGENT_DIR || join(homedir(), ".pi", "agent");
}

function readJsonFile(path: string): Record<string, unknown> | null {
	try {
		const parsed: unknown = JSON.parse(readFileSync(path, "utf8"));
		return typeof parsed === "object" && parsed !== null ? (parsed as Record<string, unknown>) : null;
	} catch {
		return null;
	}
}

/** Pi's global `httpProxy` from ~/.pi/agent/settings.json, when present. */
export function settingsHttpProxy(): string | null {
	const settings = readJsonFile(defaultSettingsPath());
	const value = settings?.httpProxy;
	return typeof value === "string" && value.trim() !== "" ? value.trim() : null;
}

/** "direct" means "never proxy"; null means "unspecified". */
export function normalizeProxy(value: string | null | undefined): string | null {
	if (value == null) return null;
	const trimmed = value.trim();
	if (trimmed === "" || trimmed === "direct") return null;
	return trimmed;
}

/**
 * Proxy for a remote call: explicit config value ("direct" = never) first,
 * then Pi's settings.json httpProxy, then env proxies.
 */
export function resolveProxyUrl(
	explicit: string | null | undefined,
	env: (name: string) => string | undefined = (name) => process.env[name],
): string | null {
	// An explicit value short-circuits: "direct" (or empty) means NEVER proxy.
	if (explicit != null) return normalizeProxy(explicit);
	return (
		settingsHttpProxy() ??
		normalizeProxy(env("https_proxy")) ??
		normalizeProxy(env("HTTPS_PROXY")) ??
		normalizeProxy(env("http_proxy")) ??
		normalizeProxy(env("HTTP_PROXY")) ??
		null
	);
}

/**
 * API key for a remote classifier backend: env var first (apiKeyEnv), then
 * Pi's auth file (keyed by provider name, e.g. auth.json["litellm"]). Keys
 * are never logged or embedded in error messages.
 */
export function resolveApiKey(apiKeyEnv: string, provider: string, authFile?: string | null): string | null {
	const fromEnv = process.env[apiKeyEnv];
	if (fromEnv && fromEnv.trim() !== "") return fromEnv.trim();
	const auth = readJsonFile(authFile ?? defaultAuthFile());
	const entry = auth?.[provider];
	if (typeof entry === "object" && entry !== null) {
		const key = (entry as Record<string, unknown>).key;
		if (typeof key === "string" && key.trim() !== "") return key.trim();
	}
	return null;
}

/**
 * Base URL for a remote classifier backend, read from Pi's auth file entry for
 * the provider - never from plugin config, so private endpoints stay out of
 * shared config files. Accepted shapes: `entry.baseUrl` (string) or any
 * `entry.env.<NAME>_BASE_URL` string (e.g. "LITELLM_BASE_URL").
 * Returns null when the entry has none; the caller then uses the OpenRouter
 * default for the protocol.
 */
export function resolveBaseUrl(provider: string, authFile?: string | null): string | null {
	const auth = readJsonFile(authFile ?? defaultAuthFile());
	const entry = auth?.[provider];
	if (typeof entry !== "object" || entry === null) return null;
	const e = entry as Record<string, unknown>;
	if (typeof e.baseUrl === "string" && e.baseUrl.trim() !== "") return e.baseUrl.trim();
	const env = e.env;
	if (typeof env === "object" && env !== null) {
		for (const [name, value] of Object.entries(env as Record<string, unknown>)) {
			if (/_BASE_URL$/iu.test(name) && typeof value === "string" && value.trim() !== "") {
				return value.trim();
			}
		}
	}
	return null;
}

/** Whether credentials exist without revealing them (startup validation). */
export function hasApiKey(apiKeyEnv: string, provider: string, authFile?: string | null): boolean {
	if (process.env[apiKeyEnv]?.trim()) return true;
	const path = authFile ?? defaultAuthFile();
	if (!existsSync(path)) return false;
	return resolveApiKey(apiKeyEnv, provider, path) !== null;
}

export async function makeFetch(proxyUrl?: string | null, provided?: FetchLike): Promise<FetchLike> {
	if (fetchOverride) return fetchOverride;
	if (provided) return provided;
	if (!proxyUrl) return globalThis.fetch as unknown as FetchLike;
	const { ProxyAgent, fetch: undiciFetch } = await import("undici");
	const agent = new ProxyAgent(proxyUrl);
	const bound: FetchLike = (url, init) =>
		isLoopbackUrl(String(url))
			? (globalThis.fetch as unknown as FetchLike)(url, init)
			: (undiciFetch(url, { ...(init as object), dispatcher: agent } as never) as never);
	return bound;
}

function sleep(ms: number): Promise<void> {
	return ms > 0 ? new Promise((resolve) => setTimeout(resolve, ms)) : Promise.resolve();
}

function isRetryable(status: number | null, err: unknown): boolean {
	if (status === 429) return true;
	if (status !== null && status >= 500) return true;
	// Network-level failures (reset, tunnel, DNS) are retryable; AbortError from
	// our own timeout budget is NOT retried beyond the attempt cap anyway.
	return status === null && !(err instanceof RemoteClassifierError);
}

function describeHttpError(status: number, body: string, label = "OpenRouter"): RemoteClassifierError {
	const snippet = body.slice(0, 200);
	switch (status) {
		case 401:
			return new RemoteClassifierError(
				`${label} rejected the credentials (401): check the configured API key (env var or auth file)`,
				401,
			);
		case 402:
			return new RemoteClassifierError(`${label} account has insufficient credits (402)`, 402);
		case 403:
			return new RemoteClassifierError(`${label} denied access (403): spend limit, disabled key, or blocked`, 403);
		case 404:
			return new RemoteClassifierError(`${label} could not find the classifier model or endpoint (404): ${snippet}`, 404);
		case 429:
			return new RemoteClassifierError(`${label} rate-limited the classifier (429)`, 429);
		default:
			return new RemoteClassifierError(`${label} classifier HTTP ${status}: ${snippet}`, status);
	}
}

/** Numbers only, finite, inside [0, 1]: drop anything else. Never invents values. */
function sanitizeProbabilities(raw: unknown): Record<string, number> {
	if (typeof raw !== "object" || raw === null) return {};
	const out: Record<string, number> = {};
	for (const [key, value] of Object.entries(raw as Record<string, unknown>)) {
		if (typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1) out[key] = value;
	}
	return out;
}

/** Parse + validate; separate function so tests can exercise it directly. */
export function parseAnswers(raw: unknown): Record<string, { probabilities?: Record<string, number> } | undefined> {
	if (typeof raw !== "object" || raw === null) {
		throw new RemoteClassifierError("classifier response was not a JSON object");
	}
	const answers = (raw as { answers?: unknown }).answers;
	if (typeof answers !== "object" || answers === null || Array.isArray(answers)) {
		throw new RemoteClassifierError("classifier response is missing a structured `answers` object");
	}
	const out: Record<string, { probabilities?: Record<string, number> } | undefined> = {};
	for (const [name, value] of Object.entries(answers as Record<string, unknown>)) {
		if (typeof value !== "object" || value === null) continue;
		const probabilities = (value as { probabilities?: unknown }).probabilities;
		out[name] = probabilities === undefined ? undefined : { probabilities: sanitizeProbabilities(probabilities) };
	}
	if (Object.keys(out).length === 0) {
		throw new RemoteClassifierError("classifier returned no usable answers");
	}
	return out;
}

/** Parse usage: real numbers only; missing pieces stay null (never fabricated). */
export function parseUsage(raw: unknown): RemoteUsage {
	const usage = typeof raw === "object" && raw !== null ? (raw as Record<string, unknown>) : {};
	const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) && v >= 0 ? v : null);
	return {
		inputTokens: num(usage.input_tokens ?? usage.inputTokens),
		outputTokens: num(usage.output_tokens ?? usage.outputTokens),
		cost: num(usage.cost),
	};
}

/**
 * One bounded classification call. Retries at most `maxAttempts` times, only
 * on rate limits, server errors, and network failures - and never switches
 * provider or model on failure. Every error message is credential-free.
 * Between attempts it waits `retryDelayMs` (default 0).
 */
export async function callDecisions(opts: DecisionCallOptions): Promise<RemoteCallResult> {
	const doFetch = await makeFetch(opts.proxyUrl, opts.fetchImpl);
	const url = opts.endpoint ?? DECISIONS_URL;
	const label = opts.providerLabel ?? "OpenRouter";
	const maxAttempts = Math.max(1, opts.maxAttempts ?? 1);
	const retryDelayMs = Math.max(0, opts.retryDelayMs ?? 0);
	const body = JSON.stringify({
		model: opts.model,
		state: { role: opts.role, prompt: opts.prompt },
		questions: opts.questions,
	});
	const started = Date.now();
	let lastError: RemoteClassifierError | null = null;

	for (let attempt = 1; attempt <= maxAttempts; attempt++) {
		const controller = new AbortController();
		const timer = setTimeout(() => controller.abort(), opts.timeoutMs);
		let status: number | null = null;
		try {
			const response = await doFetch(url, {
				method: "POST",
				headers: {
					Authorization: `Bearer ${opts.apiKey}`,
					"Content-Type": "application/json",
				},
				body,
				signal: controller.signal,
			});
			status = response.status;
			if (!response.ok) {
				const text = await response.text().catch(() => "");
				lastError = describeHttpError(status, text, label);
				if (attempt < maxAttempts && isRetryable(status, null)) {
					await sleep(retryDelayMs);
					continue;
				}
				throw lastError;
			}
			const payload = (await response.json()) as Record<string, unknown>;
			const answers = parseAnswers(payload);
			const served =
				typeof payload.model === "string" && payload.model ? (payload.model as string) : null;
			return {
				answers: answers as RemoteCallResult["answers"],
				usage: parseUsage(payload.usage),
				latencyMs: Date.now() - started,
				servedModel: served ?? opts.model,
			};
		} catch (err) {
			if (err instanceof RemoteClassifierError) {
				lastError = err;
				if (attempt < maxAttempts && isRetryable(err.status, err)) {
					await sleep(retryDelayMs);
					continue;
				}
				throw err;
			}
			// Timeout or transport failure (fetch rejects with DOMException/TypeError).
			const aborted = err instanceof Error && err.name === "AbortError";
			lastError = new RemoteClassifierError(
				aborted
					? `classifier request timed out after ${opts.timeoutMs}ms`
					: `classifier request failed: ${err instanceof Error ? err.message : String(err)}`,
			);
			if (attempt < maxAttempts && isRetryable(null, err)) {
				await sleep(retryDelayMs);
				continue;
			}
			throw lastError;
		} finally {
			clearTimeout(timer);
		}
	}
	throw lastError ?? new RemoteClassifierError("classifier request failed");
}
