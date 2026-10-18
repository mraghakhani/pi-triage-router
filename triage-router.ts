/**
 * Laya router - a virtual model that routes each user request to an execution
 * model from a configurable catalog, using Laya, the open-source, local
 * System 1 decision model (https://huggingface.co/convaiinnovations/laya).
 *
 * Registers `laya/auto`. See README.md for full documentation.
 *
 * Design:
 * - Classification is ALWAYS local and free: Laya runs in-process via ONNX.
 *   No cloud model is ever used for classification or summarization.
 * - Laya receives the current request (verbatim) plus compact conversation
 *   context: a rolling summary (router state) seeded from Pi's own compaction
 *   summaries when available, and the most recent short turns.
 * - Every decision is validated against the config and the live model
 *   registry, then refined by deterministic, documented rules.
 * - Fallback is a deterministic policy; a routing failure never silently
 *   selects a paid model Laya did not choose.
 *
 * Usage: pi --model laya/auto
 */

import { mkdirSync, readFileSync, renameSync, statSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import type { Message, Usage } from "@earendil-works/pi-ai";
import type {
	ExtensionAPI,
	ExtensionContext,
	ModelRoute,
	ModelRouteRequest,
} from "@earendil-works/pi-coding-agent";
import {
	DEFAULT_SCORING_WEIGHTS,
	datasetStatus,
	readDataset,
	scheduleRefresh,
	scoreEligible,
	type BenchmarkConfig,
	type BenchmarkDataset,
	type CapabilityOverrides,
	type CategoryWeights,
	type ScoringConfig,
} from "./triage-router-lib/triage-router-benchmarks.ts";
import { readPriceDataset, schedulePriceRefresh, type ModelPrice, type PriceConfig } from "./triage-router-lib/triage-router-prices.ts";
import { checkRemoteClassifier, type RemoteSectionLike } from "./triage-router-lib/triage-router-validation.ts";
import {
	filterByComposites,
	groupComposites,
	validateGroupsSection,
	warnOnce,
} from "./triage-router-lib/triage-router-groups.ts";
import {
	costSummary,
	cacheSummary,
	activeGroupLabel,
	groupLabel,
	configSummary,
	formatDashboard,
	historyOptions,
	modelDetail,
	modelOptions,
	recordCost,
	recordRoute,
	routeHistory,
	sessionCost,
	usdLabel,
} from "./triage-router-lib/triage-router-dashboard.ts";
import {
	callDecisions,
	classifierEndpoint,
	hasApiKey,
	piAgentDir,
	resolveApiKey,
	resolveBaseUrl,
	resolveProxyUrl,
	RemoteClassifierError,
} from "./triage-router-lib/triage-router-openrouter.ts";

const HERE = dirname(fileURLToPath(import.meta.url));
const LOG_PREFIX = "[pi-triage-router]";

/**
 * Config locations, most specific first.
 *
 * The bundled file lives inside whatever directory pi installed the package
 * into, and `pi update` replaces that whole tree - so a user's edits must
 * never live there. `~/.pi/agent/triage-router.json` is the user-owned slot
 * and wins whenever it exists; `TRIAGE_ROUTER_CONFIG` overrides everything.
 * The extensions-dir slot covers installs that predate packaging.
 */
export function configCandidates(): string[] {
	const home = piAgentDir();
	const list = [
		process.env.TRIAGE_ROUTER_CONFIG,
		join(home, "triage-router.json"),
		join(home, "extensions", "triage-router.json"),
		join(HERE, "triage-router.json"),
	].filter((p): p is string => Boolean(p));
	return [...new Set(list)];
}

/** First existing config path, else the bundled one (loadConfig then uses defaults). */
export const CONFIG_PATH = (() => {
	const candidates = configCandidates();
	return candidates.find((p) => {
		try {
			return statSync(p).isFile();
		} catch {
			return false;
		}
	}) ?? candidates[candidates.length - 1];
})();

// ---------------------------------------------------------------------------
// Laya sequence budget (measured on the shipped weights)
// ---------------------------------------------------------------------------
// Laya builds one sequence per question: CLS + question head + SEP + state +
// SEP, capped at config.max_len (512). The head is CLS + instructions + SEP +
// option ids + SEP ~= config.head_max_len (192) + 3 tokens for any question
// with >= 2 options, so the state - the whole JSON-dumped {role, prompt}
// object - gets max_len - head_max_len - 4 tokens. Measured: 512 - 192 - 4 = 316.
/** Default state room in tokens; refined per instance from Laya's own config. */
export const LAYA_STATE_ROOM_TOKENS = 316;
/** Tokens reserved inside the state for the role text and JSON escaping. */
export const STATE_ROLE_MARGIN_TOKENS = 24;
/** Conservative chars/token for code/symbol-heavy text when no tokenizer is available. */
export const CONSERVATIVE_CHARS_PER_TOKEN = 3;
/** Reserve for the response and safety margin when checking a candidate's context window. */
export const OUTPUT_RESERVE_TOKENS = 16_384;
/** Role text sent as part of the state; the classification prompt follows it. */
export const ROLE_TEXT = "You are routing a software engineering request to a coding agent.";
const HEADER_RECENT = "[Recent conversation]";
const HEADER_SUMMARY = "[Conversation summary]";
const HEADER_REQUEST = "[Current request - base the decision primarily on this]";

/** Fallback when the config file is missing or has no valid entries. */
export const DEFAULT_MODELS: ModelEntry[] = [
	{ provider: "litellm", id: "deepseek/deepseek-v4-flash", description: "Any request" },
];

/** Role text for the remote classifier's state (remote budgets allow detail). */
export const DEFAULT_REMOTE_PROMPT =
	"You are a routing classifier for a coding agent. The prompt contains the current request and a bounded summary of recent conversation. Answer the choice questions about THIS request: which execution model should handle it, how complex the work is, and what kind of work it is. Base the choice primarily on the current request; use context only to resolve references.";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Classifier backend selection: local Laya (default) or a remote Jev backend (openrouter|litellm). */
export interface ClassifierConfig {
	/** "local" = ONNX Laya in-process (default, free); "openrouter"/"litellm" = remote decision model. */
	provider: "local" | "openrouter" | "litellm";
	/** Remote decision model id (e.g. "typesafe/jev-1.13"). Unused for local. */
	model: string;
	/** Remote wire protocol: "decisions" (OpenRouter /api/alpha/decisions) or "systemone"
	 * (TypeSafe native /systemone - served by OpenRouter and LiteLLM proxies). */
	protocol: "decisions" | "systemone";
	/** Pi auth file for credential + base-URL lookup (default ~/.pi/agent/auth.json). */
	authFile: string | null;
	/** Classification timeout in ms (local inference and each remote attempt). Default 5000. */
	timeoutMs: number;
	/** Total remote attempts: first try + retries, on 429/5xx/network errors only.
	 * Local inference never retries. Default 1 (no retries). Bounds: 1..10. */
	maxAttempts: number;
	/** Delay between remote attempts in ms. Default 0. */
	retryDelayMs: number;
	/** Remote state budget in tokens (the JSON {role, prompt} payload). Default 512. */
	maxInputTokens: number;
	/** Role text for local Laya's state (measured room still applies). */
	localPrompt: string;
	/** Role text for the remote classifier's state. */
	remotePrompt: string;
	/** What happens when the SELECTED backend fails: "deterministic" = the documented
	 * deterministic fallback policy decides the execution model (no provider switch),
	 * "error" = surface the routing error. Default "deterministic". */
	failurePolicy: "deterministic" | "error";
	/** Environment variable holding the OpenRouter API key (never read from config/logs). */
	apiKeyEnv: string;
	/** Optional HTTP(S) proxy; falls back to env https_proxy. */
	proxyUrl: string | null;
}

export interface ModelEntry {
	/** Provider id as known to Pi's model registry (e.g. "litellm"). */
	provider: string;
	/** Model id as known to Pi's model registry (e.g. "deepseek/deepseek-v4-flash"). */
	id: string;
	/** Criterion Laya scores against: when should this model be used? One concrete line. */
	description: string;
	/** Input price in USD per 1M tokens. Optional; absent = unknown. 0 is a legitimate value. */
	priceIn?: number;
	/** Output price in USD per 1M tokens. Optional; absent = unknown. */
	priceOut?: number;
	/** Set false to keep the entry in the config without ever routing to it. Default true. */
	enabled?: boolean;
	/** Measured internal benchmark results overriding the cached OpenRouter data for this
	 * model (same scale as the named metric; null = explicitly no data). Optional. */
	capabilities?: CapabilityOverrides;
}

export interface RouterConfig {
	models: ModelEntry[];
	/** Preferred deterministic fallback model as "provider/id". Optional. */
	defaultModel?: string;
	/** Manual override: route everything here while set, as "provider/id". Optional. */
	override?: string;
	/** Fallback policy when Laya fails or is outvoted: "cheapest" (first eligible entry) or "strongest" (last eligible). Default "cheapest". */
	fallback: "cheapest" | "strongest";
	/** Named model groups ("saved filters"), each a list of refs:
	 * "provider/id" explicit, "*" all catalog models, "@group" include
	 * another group, "-provider/id" exclude. Optional. */
	groups: Record<string, string[]>;
	/** The group classification and routing are filtered to. null = full catalog. */
	activeGroup: string | null;
	/** Group to try when the active group has no eligible model. Optional. */
	defaultGroup: string | null;
	/** Override for the ONNX weight cache directory. Default: ~/.cache/receptron-laya. */
	cacheDir?: string | null;
	/** Total character budget for the context sections (recent turns + summary) sent to Laya.
	 * Upper bound only: Laya's state room (see LAYA_STATE_ROOM_TOKENS) is the binding limit.
	 * Default 1800 (Laya truncates its state at ~316 tokens). */
	contextBudgetChars: number;
	/** How many recent short turns to include as context. Default 4. */
	recentTurns: number;
	/** Laya inference timeout in ms. Default 5000. Legacy alias: feeds classifier.timeoutMs when the classifier section omits it. */
	layaTimeoutMs: number;
	/** Classifier backend selection (default: local Laya - unchanged for existing installs). */
	classifier: ClassifierConfig;
	/** Optional remote-only backend for the second virtual model "jev/auto"
	 * (openrouter or litellm). null = jev/auto is not registered. */
	jev: ClassifierConfig | null;
	/** Benchmark data source + cache settings. */
	benchmarks: BenchmarkConfig;
	/** Auto-refreshed OpenRouter price cache (24h background refresh, keyless). */
	prices: PriceConfig;
	/** Deterministic benchmark-driven scoring policy (disabled by default). */
	scoring: ScoringConfig;
	/** Problems found while loading the config (non-fatal; logged once per (re)load). */
	problems: string[];
}

export interface LayaState {
	/** Chosen model as "provider/id". */
	model: string;
	/** Rolling compact conversation summary, bounded. */
	summary: string;
}

export type Complexity = "standard" | "moderate" | "complex";
export type Category =
	| "question"
	| "coding"
	| "infra"
	| "debugging"
	| "reasoning"
	| "research"
	| "long_horizon"
	| "budget";

export const COMPLEXITIES: Complexity[] = ["standard", "moderate", "complex"];
export const CATEGORIES: Category[] = [
	"question",
	"coding",
	"infra",
	"debugging",
	"reasoning",
	"research",
	"long_horizon",
	"budget",
];

export interface CostEstimate {
	/** Estimated input tokens for the upcoming request, or null if not estimable. */
	inputTokens: number | null;
	/** Estimated input cost in USD using the selected model's price, or null. */
	estCostIn: number | null;
	/** "usage" = from provider usage data (actual, previous request), "context" = Pi context estimate, "chars" = chars/4 heuristic. */
	source: "usage" | "context" | "chars" | null;
	/** Estimated output tokens, from the previous request's actual usage when available. */
	outputTokens: number | null;
	/** Estimated output cost in USD, or null (never fabricated). */
	estCostOut: number | null;
}

export interface RoutingDecision {
	entry: ModelEntry;
	/** Which deterministic rule produced the final selection. */
	rule:
		| "override"
		| "sticky"
		| "primary"
		| "scored"
		| "escalated"
		| "downgraded"
		| "fallback"
		| "direct";
	complexity: Complexity;
	category: Category;
	/** Laya's probability for the selected model, or null when no classification ran. */
	confidence: number | null;
	/** Human-readable rationale lines for logging/diagnostics. */
	notes: string[];
	estimate: CostEstimate;
}

export interface Classifier {
	/** Laya-style classifier: returns answer probabilities per question, or throws. */
	systemOne(
		input: { role: string; prompt: string },
		questions: Record<string, unknown>,
	): Promise<{ answers: Record<string, { probabilities?: Record<string, number> } | undefined> }>;
	/** Optional real-Laya surface for exact state fitting: tokenizer ids for text. */
	encode?: (text: string) => number[];
	/** Optional real-Laya sequence budget. */
	config?: { max_len?: number; head_max_len?: number };
}

type LayaRouteRequest = ModelRouteRequest<LayaState>;

// ---------------------------------------------------------------------------
// Configuration loading and validation
// ---------------------------------------------------------------------------

const KNOWN_MODEL_KEYS = new Set(["provider", "id", "description", "priceIn", "priceOut", "enabled", "capabilities"]);
const KNOWN_CONFIG_KEYS = new Set([
	"models",
	"default",
	"override",
	"fallback",
	"groups",
	"activeGroup",
	"defaultGroup",
	"cacheDir",
	"contextBudgetChars",
	"recentTurns",
	"layaTimeoutMs",
	"classifier",
	"jev",
	"benchmarks",
	"prices",
	"scoring",
]);
const CLASSIFIER_KEYS = new Set(["provider", "model", "protocol", "authFile", "timeoutMs", "maxAttempts", "retryDelayMs", "maxInputTokens", "localPrompt", "remotePrompt", "failurePolicy", "apiKeyEnv", "proxyUrl"]);
const BENCHMARK_KEYS = new Set(["enabled", "source", "cachePath", "refreshIntervalMs", "staleAfterMs", "timeoutMs", "apiKeyEnv", "proxyUrl"]);
const PRICE_KEYS = new Set(["enabled", "cachePath", "refreshIntervalMs", "timeoutMs", "proxyUrl"]);
const SCORING_KEYS = new Set(["enabled", "missingPolicy", "missingPenalty", "stalePolicy", "expectedInputTokens", "expectedOutputTokens", "complexityQualityDelta", "classifierWeight", "weights"]);
const CAPABILITY_KEYS = new Set(["coding", "reasoning", "agentic", "infra", "taskSuccess", "research"]);

function validateModelEntry(raw: unknown, index: number, problems: string[]): ModelEntry | null {
	if (typeof raw !== "object" || raw === null) {
		problems.push(`models[${index}]: not an object`);
		return null;
	}
	const m = raw as Record<string, unknown>;
	if (typeof m.provider !== "string" || m.provider.trim() === "") {
		problems.push(`models[${index}]: missing or invalid "provider"`);
		return null;
	}
	if (typeof m.id !== "string" || m.id.trim() === "") {
		problems.push(`models[${index}]: missing or invalid "id"`);
		return null;
	}
	if (typeof m.description !== "string" || m.description.trim() === "") {
		problems.push(`models[${index}]: missing or invalid "description"`);
		return null;
	}
	if (m.description.length > 200) {
		// Criteria text shares Laya's ~195-token head across all options; long
		// descriptions get tail-truncated inside Laya exactly like the old
		// price line did. One short sentence, task-fit first.
		problems.push(
			`models[${index}] "${m.provider}/${m.id}": description is ${m.description.length} chars; keep it one short sentence (<= 200 chars), task-fit first - Laya's head budget truncates long criteria`,
		);
	}
	for (const key of ["priceIn", "priceOut"]) {
		if (m[key] !== undefined && (typeof m[key] !== "number" || !Number.isFinite(m[key]) || m[key] < 0)) {
			problems.push(`models[${index}]: "${key}" must be a number >= 0 (USD per 1M tokens)`);
			return null;
		}
	}
	if (m.enabled !== undefined && typeof m.enabled !== "boolean") {
		problems.push(`models[${index}]: "enabled" must be boolean`);
		return null;
	}
	let capabilities: CapabilityOverrides | undefined;
	if (m.capabilities !== undefined) {
		if (typeof m.capabilities !== "object" || m.capabilities === null || Array.isArray(m.capabilities)) {
			problems.push(`models[${index}]: "capabilities" must be an object; ignoring`);
		} else {
			const caps: Record<string, number | null> = {};
			for (const [key, value] of Object.entries(m.capabilities as Record<string, unknown>)) {
				if (!CAPABILITY_KEYS.has(key)) {
					problems.push(`models[${index}] "capabilities": unknown key "${key}" ignored`);
					continue;
				}
				if (value === null || (typeof value === "number" && Number.isFinite(value) && value >= 0)) {
					caps[key] = value as number | null;
				} else {
					problems.push(`models[${index}] "capabilities.${key}": must be a number >= 0 or null`);
				}
			}
			if (Object.keys(caps).length > 0) capabilities = caps as CapabilityOverrides;
		}
	}
	for (const key of Object.keys(m)) {
		if (!KNOWN_MODEL_KEYS.has(key)) problems.push(`models[${index}]: unknown key "${key}" ignored`);
	}
	return {
		provider: m.provider,
		id: m.id,
		description: m.description,
		priceIn: m.priceIn as number | undefined,
		priceOut: m.priceOut as number | undefined,
		enabled: m.enabled === false ? false : true,
		...(capabilities ? { capabilities } : {}),
	};
}

/** Shared classifier default; timeoutMs differs by call site (laya budget vs jev's own 5000). */
export function defaultClassifier(timeoutMs: number): ClassifierConfig {
	return {
		provider: "local",
		model: "typesafe/jev-1.13",
		timeoutMs,
		maxAttempts: 1,
		retryDelayMs: 0,
		maxInputTokens: 512,
		localPrompt: ROLE_TEXT,
		remotePrompt: DEFAULT_REMOTE_PROMPT,
		failurePolicy: "deterministic",
		apiKeyEnv: "OPENROUTER_API_KEY",
		proxyUrl: null,
		protocol: "decisions",
		authFile: null,
	};
}

/** Conventional credential env var per remote provider. Only a default: an
 * explicit `apiKeyEnv` in the section always wins (applyClassifierSection).
 * Kept provider-aware so a litellm section never inherits OPENROUTER_API_KEY
 * and sends OpenRouter credentials to the LiteLLM proxy. */
export function defaultApiKeyEnv(provider: string): string {
	return provider === "litellm" ? "LITELLM_API_KEY" : "OPENROUTER_API_KEY";
}

/** Benchmark cache defaults (shared by the initial literal and the mid-validation rebuild). */
function defaultBenchmarks(): BenchmarkConfig {
	return {
		enabled: true,
		source: "openrouter",
		cachePath: join(piAgentDir(), "triage-router.benchmarks.json"),
		seedPath: join(HERE, "triage-router.benchmarks.json"),
		refreshIntervalMs: 86_400_000,
		staleAfterMs: 2_592_000_000,
		timeoutMs: 15_000,
		apiKeyEnv: "OPENROUTER_API_KEY",
		proxyUrl: null,
	};
}

/** Price cache defaults (shared by the initial literal and the mid-validation rebuild). */
function defaultPrices(): PriceConfig {
	return {
		enabled: true,
		cachePath: join(piAgentDir(), "triage-router.prices.json"),
		seedPath: join(HERE, "triage-router.prices.json"),
		refreshIntervalMs: 86_400_000,
		timeoutMs: 15_000,
		proxyUrl: null,
	};
}

/** Scoring defaults, disabled (shared by the initial literal and the mid-validation rebuild). */
function defaultScoring(): ScoringConfig {
	return {
		enabled: false,
		missingPolicy: "neutral",
		missingPenalty: 0.15,
		stalePolicy: "use",
		expectedInputTokens: 4000,
		expectedOutputTokens: 1000,
		complexityQualityDelta: 0.15,
		classifierWeight: 0,
		weights: Object.fromEntries(CATEGORIES.map((cat) => [cat, { ...DEFAULT_SCORING_WEIGHTS[cat] }])) as Record<Category, CategoryWeights>,
	};
}

/** Validate a raw parsed config object. Returns a validated config plus non-fatal problems. */
export function validateConfig(raw: unknown): RouterConfig {
	const problems: string[] = [];
	const obj = (typeof raw === "object" && raw !== null ? raw : {}) as Record<string, unknown>;
	const config: RouterConfig = {
		models: DEFAULT_MODELS,
		fallback: obj.fallback === "strongest" ? "strongest" : "cheapest",
		groups: {},
		activeGroup: null,
		defaultGroup: null,
		contextBudgetChars: 1800,
		recentTurns: 4,
		layaTimeoutMs: 5000,
		classifier: defaultClassifier(5000),
		jev: null,
		benchmarks: defaultBenchmarks(),
		prices: defaultPrices(),
		scoring: defaultScoring(),
		problems,
	};
	for (const key of Object.keys(obj)) {
		if (!KNOWN_CONFIG_KEYS.has(key)) problems.push(`unknown config key "${key}" ignored`);
	}
	if (obj.contextBudgetChars !== undefined) {
		if (typeof obj.contextBudgetChars !== "number" || obj.contextBudgetChars < 400) {
			problems.push('"contextBudgetChars" must be a number >= 400; using default 1800');
		} else {
			config.contextBudgetChars = obj.contextBudgetChars;
		}
	}
	if (obj.recentTurns !== undefined) {
		if (typeof obj.recentTurns !== "number" || obj.recentTurns < 0 || obj.recentTurns > 12) {
			problems.push('"recentTurns" must be a number 0..12; using default 4');
		} else {
			config.recentTurns = obj.recentTurns;
		}
	}
	if (obj.layaTimeoutMs !== undefined) {
		if (typeof obj.layaTimeoutMs !== "number" || obj.layaTimeoutMs < 100) {
			problems.push('"layaTimeoutMs" must be a number >= 100; using default 5000');
		} else {
			config.layaTimeoutMs = obj.layaTimeoutMs;
		}
	}
	if (obj.cacheDir !== undefined && obj.cacheDir !== null && typeof obj.cacheDir !== "string") {
		problems.push('"cacheDir" must be a string or null; ignoring');
	} else {
		config.cacheDir = (obj.cacheDir as string | null | undefined) ?? null;
	}

	// --- classifier sections (laya/auto local by default; optional jev/auto remote) ---
	function applyClassifierSection(
		raw: unknown,
		label: string,
		target: ClassifierConfig,
		allowLocal: boolean,
		problems: string[],
	): void {
		if (raw === undefined) return;
		if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
			problems.push(`"${label}" must be an object; using defaults`);
			return;
		}
		const c = raw as Record<string, unknown>;
		for (const key of Object.keys(c)) {
			if (!CLASSIFIER_KEYS.has(key)) problems.push(`"${label}": unknown key "${key}" ignored`);
		}
		if (c.provider !== undefined) {
			if (c.provider === "local" && !allowLocal) {
				problems.push(`"${label}.provider" must be "openrouter" or "litellm" (the jev entry is the remote backend)`);
			} else if (c.provider === "local" || c.provider === "openrouter" || c.provider === "litellm") {
				target.provider = c.provider;
			} else {
				problems.push(`"${label}.provider" must be "local", "openrouter", or "litellm"`);
			}
		}
		if (c.model !== undefined) {
			if (typeof c.model === "string" && c.model.trim() !== "") target.model = c.model;
			else problems.push(`"${label}.model" must be a non-empty string`);
		}
		if (c.protocol !== undefined) {
			if (c.protocol === "decisions" || c.protocol === "systemone") target.protocol = c.protocol;
			else problems.push(`"${label}.protocol" must be "decisions" or "systemone"`);
		}
		if (c.authFile !== undefined) {
			if (c.authFile === null || (typeof c.authFile === "string" && c.authFile.trim() !== "")) {
				target.authFile = (c.authFile as string | null) ?? null;
			} else problems.push(`"${label}.authFile" must be a non-empty string or null`);
		}
		if (c.timeoutMs !== undefined) {
			if (typeof c.timeoutMs === "number" && c.timeoutMs >= 100) target.timeoutMs = c.timeoutMs;
			else problems.push(`"${label}.timeoutMs" must be a number >= 100`);
		}
		if (c.maxAttempts !== undefined) {
			if (typeof c.maxAttempts === "number" && Number.isInteger(c.maxAttempts) && c.maxAttempts >= 1 && c.maxAttempts <= 10) {
				target.maxAttempts = c.maxAttempts;
			} else problems.push(`"${label}.maxAttempts" must be an integer in 1..10`);
		}
		if (c.retryDelayMs !== undefined) {
			if (typeof c.retryDelayMs === "number" && c.retryDelayMs >= 0 && c.retryDelayMs <= 60_000) {
				target.retryDelayMs = c.retryDelayMs;
			} else problems.push(`"${label}.retryDelayMs" must be a number in 0..60000`);
		}
		if (c.maxInputTokens !== undefined) {
			if (typeof c.maxInputTokens === "number" && c.maxInputTokens >= 64 && c.maxInputTokens <= 1_000_000) {
				target.maxInputTokens = c.maxInputTokens;
			} else problems.push(`"${label}.maxInputTokens" must be a number in 64..1000000`);
		}
		if (c.localPrompt !== undefined) {
			if (typeof c.localPrompt === "string" && c.localPrompt.trim() !== "") target.localPrompt = c.localPrompt;
			else problems.push(`"${label}.localPrompt" must be a non-empty string`);
		}
		if (c.remotePrompt !== undefined) {
			if (typeof c.remotePrompt === "string" && c.remotePrompt.trim() !== "") target.remotePrompt = c.remotePrompt;
			else problems.push(`"${label}.remotePrompt" must be a non-empty string`);
		}
		if (c.failurePolicy !== undefined) {
			if (c.failurePolicy === "deterministic" || c.failurePolicy === "error") target.failurePolicy = c.failurePolicy;
			else problems.push(`"${label}.failurePolicy" must be "deterministic" or "error"; using "deterministic"`);
		}
		if (c.apiKeyEnv !== undefined) {
			if (typeof c.apiKeyEnv === "string" && c.apiKeyEnv.trim() !== "") target.apiKeyEnv = c.apiKeyEnv;
			else problems.push(`"${label}.apiKeyEnv" must be a non-empty string`);
		} else if (target.provider !== "local") {
			// Provider-aware credential default; an explicit value above wins.
			target.apiKeyEnv = defaultApiKeyEnv(target.provider);
		}
		if (c.proxyUrl !== undefined) {
			if (c.proxyUrl === null || (typeof c.proxyUrl === "string" && c.proxyUrl.trim() !== "")) {
				target.proxyUrl = c.proxyUrl as string | null;
			} else problems.push(`"${label}.proxyUrl" must be a string or null ("direct" = never proxy)`);
		}
	}
	config.classifier = defaultClassifier(config.layaTimeoutMs);
	applyClassifierSection(obj.classifier, "classifier", config.classifier, true, problems);
	if (obj.jev === undefined || obj.jev === null) {
		config.jev = null;
	} else {
		const jev: ClassifierConfig = { ...defaultClassifier(5000), provider: "openrouter" };
		applyClassifierSection(obj.jev, "jev", jev, false, problems);
		config.jev = jev;
	}
	// Startup visibility: a selected remote backend without credentials is a
	// clear, non-fatal problem (the extension still starts; no paid provider is
	// required unless the user selects it). Credentials = env var or Pi auth file.
	for (const [label, section] of [
		["classifier", config.classifier],
		["jev", config.jev],
	] as const) {
		if (
			section &&
			section.provider !== "local" &&
			!hasApiKey(section.apiKeyEnv, section.provider, section.authFile)
		) {
			problems.push(
				`${label} provider "${section.provider}" selected but no credentials found ` +
					`(env ${section.apiKeyEnv} or auth file entry "${section.provider}"): classification will follow ` +
					`failurePolicy=${section.failurePolicy} (extension still starts)`,
			);
		}
	}

	// --- benchmark cache section ---
	config.benchmarks = defaultBenchmarks();
	if (obj.benchmarks !== undefined) {
		if (typeof obj.benchmarks !== "object" || obj.benchmarks === null || Array.isArray(obj.benchmarks)) {
			problems.push('"benchmarks" must be an object; using defaults');
		} else {
			const b = obj.benchmarks as Record<string, unknown>;
			for (const key of Object.keys(b)) {
				if (!BENCHMARK_KEYS.has(key)) problems.push(`"benchmarks": unknown key "${key}" ignored`);
			}
			if (b.enabled !== undefined) {
				if (typeof b.enabled === "boolean") config.benchmarks.enabled = b.enabled;
				else problems.push('"benchmarks.enabled" must be boolean');
			}
			if (b.source !== undefined) {
				if (b.source === "openrouter" || b.source === "openrouter-models") config.benchmarks.source = b.source;
				else problems.push('"benchmarks.source" must be "openrouter" or "openrouter-models"; using "openrouter"');
			}
			if (b.cachePath !== undefined) {
				if (typeof b.cachePath === "string" && b.cachePath.trim() !== "") config.benchmarks.cachePath = b.cachePath;
				else problems.push('"benchmarks.cachePath" must be a non-empty string');
			}
			if (b.refreshIntervalMs !== undefined) {
				if (typeof b.refreshIntervalMs === "number" && b.refreshIntervalMs >= 60_000) config.benchmarks.refreshIntervalMs = b.refreshIntervalMs;
				else problems.push('"benchmarks.refreshIntervalMs" must be a number >= 60000');
			}
			if (b.staleAfterMs !== undefined) {
				if (typeof b.staleAfterMs === "number" && b.staleAfterMs >= 3_600_000) config.benchmarks.staleAfterMs = b.staleAfterMs;
				else problems.push('"benchmarks.staleAfterMs" must be a number >= 3600000');
			}
			if (b.timeoutMs !== undefined) {
				if (typeof b.timeoutMs === "number" && b.timeoutMs >= 1000) config.benchmarks.timeoutMs = b.timeoutMs;
				else problems.push('"benchmarks.timeoutMs" must be a number >= 1000');
			}
			if (b.apiKeyEnv !== undefined) {
				if (typeof b.apiKeyEnv === "string" && b.apiKeyEnv.trim() !== "") config.benchmarks.apiKeyEnv = b.apiKeyEnv;
				else problems.push('"benchmarks.apiKeyEnv" must be a non-empty string');
			}
			if (b.proxyUrl !== undefined) {
				if (b.proxyUrl === null || (typeof b.proxyUrl === "string" && b.proxyUrl.trim() !== "")) {
					config.benchmarks.proxyUrl = b.proxyUrl as string | null;
				} else problems.push('"benchmarks.proxyUrl" must be a string or null');
			}
		}
	}

	// --- price cache section ---
	config.prices = defaultPrices();
	if (obj.prices !== undefined) {
		if (typeof obj.prices !== "object" || obj.prices === null || Array.isArray(obj.prices)) {
			problems.push('"prices" must be an object; using defaults');
		} else {
			const p = obj.prices as Record<string, unknown>;
			for (const key of Object.keys(p)) {
				if (!PRICE_KEYS.has(key)) problems.push(`"prices": unknown key "${key}" ignored`);
			}
			if (p.enabled !== undefined) {
				if (typeof p.enabled === "boolean") config.prices.enabled = p.enabled;
				else problems.push('"prices.enabled" must be boolean');
			}
			if (p.cachePath !== undefined) {
				if (typeof p.cachePath === "string" && p.cachePath.trim() !== "") config.prices.cachePath = p.cachePath;
				else problems.push('"prices.cachePath" must be a non-empty string');
			}
			if (p.refreshIntervalMs !== undefined) {
				if (typeof p.refreshIntervalMs === "number" && p.refreshIntervalMs >= 60_000) config.prices.refreshIntervalMs = p.refreshIntervalMs;
				else problems.push('"prices.refreshIntervalMs" must be a number >= 60000');
			}
			if (p.timeoutMs !== undefined) {
				if (typeof p.timeoutMs === "number" && p.timeoutMs >= 1000) config.prices.timeoutMs = p.timeoutMs;
				else problems.push('"prices.timeoutMs" must be a number >= 1000');
			}
			if (p.proxyUrl !== undefined) {
				if (p.proxyUrl === null || (typeof p.proxyUrl === "string" && p.proxyUrl.trim() !== "")) {
					config.prices.proxyUrl = p.proxyUrl as string | null;
				} else problems.push('"prices.proxyUrl" must be a string or null');
			}
		}
	}

	// --- scoring section (default: disabled - existing behavior unchanged) ---
	config.scoring = defaultScoring();
	if (obj.scoring !== undefined) {
		if (typeof obj.scoring !== "object" || obj.scoring === null || Array.isArray(obj.scoring)) {
			problems.push('"scoring" must be an object; using defaults (disabled)');
		} else {
			const s = obj.scoring as Record<string, unknown>;
			for (const key of Object.keys(s)) {
				if (!SCORING_KEYS.has(key)) problems.push(`"scoring": unknown key "${key}" ignored`);
			}
			if (s.enabled !== undefined) {
				if (typeof s.enabled === "boolean") config.scoring.enabled = s.enabled;
				else problems.push('"scoring.enabled" must be boolean');
			}
			if (s.missingPolicy !== undefined) {
				if (s.missingPolicy === "neutral" || s.missingPolicy === "penalize") config.scoring.missingPolicy = s.missingPolicy;
				else problems.push('"scoring.missingPolicy" must be "neutral" or "penalize"; using "neutral"');
			}
			if (s.stalePolicy !== undefined) {
				if (s.stalePolicy === "use" || s.stalePolicy === "missing") config.scoring.stalePolicy = s.stalePolicy;
				else problems.push('"scoring.stalePolicy" must be "use" or "missing"; using "use"');
			}
			if (s.missingPenalty !== undefined) {
				if (typeof s.missingPenalty === "number" && s.missingPenalty >= 0 && s.missingPenalty <= 1) config.scoring.missingPenalty = s.missingPenalty;
				else problems.push('"scoring.missingPenalty" must be a number in 0..1');
			}
			if (s.expectedInputTokens !== undefined) {
				if (typeof s.expectedInputTokens === "number" && s.expectedInputTokens >= 1) config.scoring.expectedInputTokens = s.expectedInputTokens;
				else problems.push('"scoring.expectedInputTokens" must be a number >= 1');
			}
			if (s.expectedOutputTokens !== undefined) {
				if (typeof s.expectedOutputTokens === "number" && s.expectedOutputTokens >= 0) config.scoring.expectedOutputTokens = s.expectedOutputTokens;
				else problems.push('"scoring.expectedOutputTokens" must be a number >= 0');
			}
			if (s.complexityQualityDelta !== undefined) {
				if (typeof s.complexityQualityDelta === "number" && s.complexityQualityDelta >= 0 && s.complexityQualityDelta <= 0.5) {
					config.scoring.complexityQualityDelta = s.complexityQualityDelta;
				} else problems.push('"scoring.complexityQualityDelta" must be a number in 0..0.5');
			}
			if (s.classifierWeight !== undefined) {
				if (typeof s.classifierWeight === "number" && Number.isFinite(s.classifierWeight) && s.classifierWeight >= 0 && s.classifierWeight <= 1) {
					config.scoring.classifierWeight = s.classifierWeight;
				} else problems.push('"scoring.classifierWeight" must be a number in 0..1');
			}
			if (s.weights !== undefined) {
				if (typeof s.weights !== "object" || s.weights === null || Array.isArray(s.weights)) {
					problems.push('"scoring.weights" must be an object of per-category {quality, cost}');
				} else {
					for (const [cat, value] of Object.entries(s.weights as Record<string, unknown>)) {
						if (!CATEGORIES.includes(cat as Category)) {
							problems.push(`"scoring.weights": unknown category "${cat}" ignored`);
							continue;
						}
						const w = value as { quality?: unknown; cost?: unknown };
						if (
							typeof w === "object" && w !== null &&
							typeof w.quality === "number" && Number.isFinite(w.quality) && w.quality >= 0 &&
							typeof w.cost === "number" && Number.isFinite(w.cost) && w.cost >= 0 &&
							w.quality + w.cost > 0
						) {
							config.scoring.weights[cat as Category] = { quality: w.quality, cost: w.cost };
						} else {
							problems.push(`"scoring.weights.${cat}" must be {quality, cost} numbers >= 0 summing > 0; using default`);
						}
					}
				}
			}
		}
	}
	for (const key of ["default", "override"]) {
		if (obj[key] !== undefined && obj[key] !== null) {
			if (typeof obj[key] !== "string" || !obj[key].includes("/")) {
				problems.push(`"${key}" must be "provider/id" or null; ignoring`);
			} else if (key === "default") {
				config.defaultModel = obj[key] as string;
			} else {
				config.override = obj[key] as string;
			}
		}
	}
	const seen = new Set<string>();
	if (Array.isArray(obj.models)) {
		const entries: ModelEntry[] = [];
		for (let i = 0; i < obj.models.length; i++) {
			const entry = validateModelEntry(obj.models[i], i, problems);
			if (!entry) continue;
			const composite = `${entry.provider}/${entry.id}`;
			if (seen.has(composite)) {
				problems.push(`duplicate entry "${composite}" ignored (models[${i}])`);
				continue;
			}
			seen.add(composite);
			entries.push(entry);
		}
		if (entries.length > 0) config.models = entries;
		else if (obj.models.length > 0) problems.push("no valid model entries; using defaults");
	} else {
		problems.push('"models" missing or not an array; using defaults');
	}

	// --- named model groups ("saved filters"; optional) ---
	// Parsed after models: refs expand against the final catalog, and
	// unknown/circular refs become load problems, never silent wrong sets.
	const catalog = config.models.map((entry) => `${entry.provider}/${entry.id}`);
	config.groups = validateGroupsSection(obj.groups, catalog, problems) ?? {};
	for (const [key, target] of [
		["activeGroup", "active"],
		["defaultGroup", "default"],
	] as const) {
		const value = (obj as Record<string, unknown>)[key];
		if (value === undefined || value === null) continue;
		if (typeof value !== "string" || value.trim() === "") {
			problems.push(`"${key}" must be a non-empty string or null; ignoring`);
		} else if (!Object.prototype.hasOwnProperty.call(config.groups, value)) {
			// Own-property check: "constructor"/"toString" otherwise resolve via
			// the prototype chain and pass. Warn + ignore at load: routing falls
			// back to the full catalog (decision #7) instead of breaking.
			problems.push(`"${key}" names no defined group ("${value}"); group filtering off`);
		} else {
			config[key] = value;
		}
	}

	return config;
}

let cachedConfig: { path: string; mtimeMs: number; config: RouterConfig } | null = null;
let configOverrideForTests: RouterConfig | null = null;

/** Test hook: make loadConfig() (and therefore route()) use a fixed config
 * instead of the shipped file, so route-level tests stay hermetic. */
export function setConfigForTests(config: RouterConfig | null): void {
	configOverrideForTests = config;
}

/** Load the config file, re-reading only when its mtime changes. Logs validation problems once per (re)load. */
export function loadConfig(path: string = CONFIG_PATH): RouterConfig {
	if (configOverrideForTests) return configOverrideForTests;
	try {
		const mtimeMs = statSync(path).mtimeMs;
		if (cachedConfig?.path === path && cachedConfig.mtimeMs === mtimeMs) return cachedConfig.config;
		const raw = JSON.parse(readFileSync(path, "utf8"));
		const config = validateConfig(raw);
		cachedConfig = { path, mtimeMs, config };
		for (const problem of config.problems) console.error(`${LOG_PREFIX} config: ${problem}`);
		return config;
	} catch (err) {
		console.error(`${LOG_PREFIX} config unreadable, using defaults:`, err);
		return validateConfig({});
	}
}

/** The effective config serialized for writing: defaults minus runtime-only fields.
 * `problems` and the package-internal `seedPath` values never belong in a
 * config file (validateConfig rejects both as unknown keys). */
function effectiveConfigRaw(): Record<string, unknown> {
	const { problems, ...rest } = validateConfig({});
	const raw = JSON.parse(JSON.stringify(rest)) as Record<string, unknown>;
	if (raw.benchmarks) delete (raw.benchmarks as Record<string, unknown>).seedPath;
	if (raw.prices) delete (raw.prices as Record<string, unknown>).seedPath;
	return raw;
}

/**
 * Apply a patch to the config file and write it atomically. Patch values that
 * are plain objects merge one level into the existing section; everything else
 * (including arrays and null) is assigned. Unknown keys in the file survive.
 * Returns the revalidated config; problems are logged like loadConfig does.
 * Only the /triage menu calls this - routing itself stays read-only.
 */
export function updateConfigFile(path: string, patch: Record<string, unknown>): RouterConfig {
	let raw: Record<string, unknown>;
	try {
		const parsed = JSON.parse(readFileSync(path, "utf8"));
		raw =
			typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)
				? (parsed as Record<string, unknown>)
				: effectiveConfigRaw();
	} catch {
		// Missing or unreadable file: start from the effective defaults so the
		// first menu edit materializes a complete, valid config.
		raw = effectiveConfigRaw();
	}
	for (const [key, value] of Object.entries(patch)) {
		if (
			value !== null &&
			typeof value === "object" &&
			!Array.isArray(value) &&
			typeof raw[key] === "object" &&
			raw[key] !== null &&
			!Array.isArray(raw[key])
		) {
			raw[key] = { ...(raw[key] as Record<string, unknown>), ...value };
		} else if (value === undefined) {
			delete raw[key];
		} else {
			raw[key] = value;
		}
	}
	return writeConfigRaw(path, raw);
}

/** Validate, write atomically, and invalidate the loadConfig cache - shared by
 * every config writer. `seedPath` is runtime-injected (package-internal seed),
 * never a config-file key, so it is stripped before writing. */
function writeConfigRaw(path: string, raw: Record<string, unknown>): RouterConfig {
	// `seedPath` is runtime-injected (package-internal seed), never a config-file
	// key - strip it so menu patches never materialize it in the file.
	for (const section of ["benchmarks", "prices"] as const) {
		const value = raw[section];
		if (typeof value === "object" && value !== null && !Array.isArray(value)) {
			delete (value as Record<string, unknown>).seedPath;
		}
	}
	const config = validateConfig(raw);
	mkdirSync(dirname(path), { recursive: true });
	const tmp = `${path}.tmp`;
	writeFileSync(tmp, `${JSON.stringify(raw, null, "\t")}\n`);
	renameSync(tmp, path);
	// Invalidate the loadConfig cache outright: mtimeMs granularity varies by
	// filesystem, so two rapid saves could otherwise alias to the same key and
	// serve a stale config.
	cachedConfig = null;
	for (const problem of config.problems) console.error(`${LOG_PREFIX} config: ${problem}`);
	return config;
}

/** Reset the config file to the shipped defaults (the bundled triage-router.json,
 * falling back to the effective defaults when the bundle is unreadable). The
 * current file is copied to a timestamped backup first, so a reset is never
 * destructive beyond undo. Returns the revalidated config and the backup path
 * (null when there was no previous file). Only the /triage menu calls this. */
export function resetConfigFile(path: string = CONFIG_PATH): { config: RouterConfig; backupPath: string | null } {
	let current: string | null = null;
	try {
		current = readFileSync(path, "utf8");
	} catch {
		// No previous file: nothing to back up.
	}
	let backupPath: string | null = null;
	if (current !== null) {
		// A failed backup aborts the reset (the error propagates) - the reset
		// never destroys the only copy of the user's config.
		backupPath = `${path}.backup-${new Date().toISOString().replace(/[:.]/g, "-")}`;
		writeFileSync(backupPath, current);
	}
	let raw: Record<string, unknown>;
	try {
		raw = JSON.parse(readFileSync(join(HERE, "triage-router.json"), "utf8")) as Record<string, unknown>;
	} catch {
		raw = effectiveConfigRaw();
	}
	return { config: writeConfigRaw(path, raw), backupPath };
}

// ---------------------------------------------------------------------------
// Conversation context for Laya
// ---------------------------------------------------------------------------

function textOf(content: Message["content"]): string {
	if (typeof content === "string") return content;
	return (content ?? [])
		.filter((block) => block.type === "text")
		.map((block) => (block as { type: "text"; text: string }).text)
		.join("\n");
}

export function lastUserText(messages: readonly Message[]): string {
	const user = messages.filter((message) => message.role === "user").at(-1);
	return user ? textOf(user.content) : "";
}

/** Short texts of turns BEFORE the current request, oldest first. */
export function priorTurnTexts(messages: readonly Message[]): string[] {
	const texts: string[] = [];
	let seenCurrent = false;
	for (let i = messages.length - 1; i >= 0; i--) {
		const message = messages[i];
		if (message.role !== "user" && message.role !== "assistant") continue;
		const text = textOf(message.content).trim();
		if (!text) continue;
		if (!seenCurrent && message.role === "user") {
			seenCurrent = true; // newest user message = current request; skip it
			continue;
		}
		texts.unshift(text);
	}
	return texts;
}

function clampText(text: string, max: number, label: string): string {
	if (text.length <= max) return text;
	const marker = `\n…[${label} truncated]`;
	return `${text.slice(0, Math.max(0, max - marker.length))}${marker}`;
}

export interface RoutingInput {
	prompt: string;
	currentRequest: string;
	recent: string;
	summary: string;
	/** Tokens the JSON-dumped state actually takes, when the tokenizer is available. */
	stateTokens: number | null;
	/** State room in tokens for this classification pass. */
	roomTokens: number;
}

/** State room in tokens for a classifier: derived from Laya's own config when present. */
export function stateRoomTokens(classifier: Pick<Classifier, "config"> | null | undefined): number {
	const cfg = classifier?.config;
	if (cfg && typeof cfg.max_len === "number" && typeof cfg.head_max_len === "number") {
		return Math.max(64, cfg.max_len - cfg.head_max_len - 4);
	}
	return LAYA_STATE_ROOM_TOKENS;
}

function assemblePrompt(currentRequest: string, recent: string, summary: string): string {
	const sections: string[] = [];
	if (recent) sections.push(`${HEADER_RECENT}\n${recent}`);
	if (summary) sections.push(`${HEADER_SUMMARY}\n${summary}`);
	sections.push(`${HEADER_REQUEST}\n${currentRequest}`);
	return sections.join("\n\n");
}

/** An input with no context (used by log lines for non-classified routes). */
export function emptyRoutingInput(currentRequest = ""): RoutingInput {
	return {
		prompt: "",
		currentRequest,
		recent: "",
		summary: "",
		stateTokens: null,
		roomTokens: LAYA_STATE_ROOM_TOKENS,
	};
}

/**
 * Build the text Laya sees, bounded to Laya's state room (measured ~316
 * tokens; the state is the whole JSON-dumped {role, prompt} object, truncated
 * from the tail by Laya beyond the room). Allocation priority: the current
 * request, then recent turns, then the summary - older context is dropped
 * first, and every truncation carries an explicit marker. No tokenizer
 * available: conservative chars/token allocation. Tokenizer available: exact
 * token counts, shrink-then-verify fit loop.
 */
export function buildRoutingInput(
	messages: readonly Message[],
	stateSummary: string,
	sessionSummary: string | null,
	config: RouterConfig,
	classifier?: Classifier | null,
	roleText: string = ROLE_TEXT,
): RoutingInput {
	const roomTokens = stateRoomTokens(classifier);
	const roomChars = Math.max(
		240,
		Math.floor((roomTokens - STATE_ROLE_MARGIN_TOKENS) * CONSERVATIVE_CHARS_PER_TOKEN),
	);
	const overhead =
		HEADER_RECENT.length + HEADER_SUMMARY.length + HEADER_REQUEST.length + 8 + "\n---\n".length;

	const currentRequestRaw = lastUserText(messages);
	const turnTexts = priorTurnTexts(messages);
	const recentCount = Math.max(0, config.recentTurns);
	// Newest turns first; older context is dropped first.
	const recentFull = turnTexts
		.slice(-recentCount)
		.reverse()
		.map((text) => clampText(text, 250, "recent turn"))
		.join("\n---\n");
	const summaryBase = stateSummary.trim() || (sessionSummary ?? "").trim();

	// Allocation priority: current request, then recent turns, then summary.
	let requestCap = Math.max(120, Math.min(6000, roomChars - overhead));
	let currentRequest = clampText(currentRequestRaw, requestCap, "current request");
	// Context sections are bounded by both the room and the configured budget.
	let remaining = Math.max(
		0,
		Math.min(config.contextBudgetChars, roomChars - overhead - currentRequest.length),
	);
	let recentCap = Math.min(recentFull.length, remaining);
	let recent = recentCap > 0 ? clampText(recentFull, recentCap, "recent turns") : "";
	remaining -= recent.length;
	let summaryCap = Math.min(600, remaining);
	let summary = summaryCap > 0 ? tailText(summaryBase, summaryCap) : "";

	const encode = classifier?.encode;
	let stateTokens: number | null = null;
	if (typeof encode === "function") {
		// Exact fit: shrink oldest context first, then the request tail.
		for (let guard = 0; guard < 12; guard++) {
			const prompt = assemblePrompt(currentRequest, recent, summary);
			stateTokens = encode(JSON.stringify({ role: roleText, prompt })).length;
			if (stateTokens <= roomTokens) break;
			if (summaryCap > 0) summaryCap = Math.floor(summaryCap / 2);
			else if (recentCap > 0) recentCap = Math.floor(recentCap / 2);
			else requestCap = Math.max(120, requestCap - 240);
			summary = summaryCap > 0 ? tailText(summaryBase, summaryCap) : "";
			recent = recentCap > 0 ? clampText(recentFull, recentCap, "recent turns") : "";
			currentRequest = clampText(currentRequestRaw, requestCap, "current request");
		}
	}
	const prompt = assemblePrompt(currentRequest, recent, summary);
	return { prompt, currentRequest, recent, summary, stateTokens, roomTokens };
}

/** Keep the newest `max` characters of `text` (recency bias) with an explicit truncation marker. */
function tailText(text: string, max: number): string {
	if (text.length <= max) return text;
	const marker = "…[summary truncated] ";
	return `${marker}${text.slice(Math.max(0, text.length - max + marker.length))}`;
}

/**
 * Refresh the rolling summary: previous summary + turns that just left the
 * recent window, bounded (600 chars) and recency-biased (oldest content is
 * dropped first, so stale context fades out). Pure and deterministic.
 */
export function refreshSummary(
	previousSummary: string,
	messages: readonly Message[],
	config: RouterConfig,
): string {
	const turnTexts = priorTurnTexts(messages);
	const older = turnTexts.slice(0, Math.max(0, turnTexts.length - config.recentTurns));
	const merged = [previousSummary.trim(), ...older.map((t) => clampText(t, 200, "turn"))]
		.filter(Boolean)
		.join("\n");
	return tailText(merged, 600);
}

// ---------------------------------------------------------------------------
// Laya (local classifier) - loaded lazily, always free of API charges
// ---------------------------------------------------------------------------

// Lazily loaded Laya singleton. null = load failed; do not retry this session.
let layaPromise: Promise<Classifier | null> | null | undefined;

export function resetLayaForTests(): void {
	layaPromise = undefined;
	layaLoadStartedAt = null;
	layaLoadSettled = false;
}

/** Test hook: inject a fake classifier (or null) without loading the real one.
 * A promise simulates a load still in progress (first-run weight download);
 * `loadStartedAt` fakes how long ago that load began. */
export function setLayaForTests(
	classifier: Classifier | null | Promise<Classifier | null>,
	loadStartedAt?: number,
): void {
	if (classifier instanceof Promise) {
		layaPromise = classifier;
		layaLoadStartedAt = loadStartedAt ?? Date.now();
		layaLoadSettled = false;
		void classifier.then(
			() => (layaLoadSettled = true),
			() => (layaLoadSettled = true),
		);
	} else {
		layaPromise = Promise.resolve(classifier);
		layaLoadStartedAt = loadStartedAt ?? Date.now();
		layaLoadSettled = true;
	}
}

let layaLoadWarned = false;
/** When the lazy local load started (null = not started) and whether it has
 * settled. A FIRST load can download ~2 GB of ONNX weights from Hugging Face
 * and take minutes on a cold cache, so requests must not wait on it forever. */
let layaLoadStartedAt: number | null = null;
let layaLoadSettled = false;
/** Bounds how long a single request waits for the local load; the load itself
 * keeps running past the wait and classification resumes once it resolves. */
const LAYA_LOAD_TIMEOUT_MS = 20_000;
const LAYA_LOAD_MIN_WAIT_MS = 2_000;

/** Progress of the first-run ONNX weight download (per-file bytes as they arrive). */
export type LayaLoadProgress = (info: { file: string; received: number; total: number | null }) => void;

export async function getLaya(cacheDir?: string | null, onProgress?: LayaLoadProgress): Promise<Classifier | null> {
	if (layaPromise !== undefined) return layaPromise;
	layaLoadStartedAt = Date.now();
	layaLoadSettled = false;
	const opts = { ...(cacheDir ? { cacheDir } : {}), ...(onProgress ? { onProgress } : {}) };
	layaPromise = import("@receptron/laya")
		.then(({ Laya }) => Laya.load(opts) as unknown as Promise<Classifier>)
		.catch((err) => {
			// Once per process: a failing load is retried per request and would
			// otherwise spam the terminal on every routed message.
			if (!layaLoadWarned) {
				layaLoadWarned = true;
				console.error(`${LOG_PREFIX} Laya unavailable (local inference failed to load):`, err);
			}
			return null;
		})
		.then((classifier) => {
			layaLoadSettled = true;
			return classifier;
		});
	return layaPromise;
}

/** Criteria Laya scores model choices against: the description only - one
 * short, task-fit-first sentence. Pricing is never appended here: all criteria
 * share Laya's ~195-token head budget, and appended text truncates last-first,
 * which silently dropped the price line anyway. Price is handled by the
 * deterministic rules instead (budget category, escalation, fallback policy). */
export function describe(entry: ModelEntry): string {
	return entry.description;
}

// ---------------------------------------------------------------------------
// Classifier backends (local Laya default; remote OpenRouter Decisions API)
// ---------------------------------------------------------------------------

/** Which backend ran (or failed) plus per-call usage for cost accounting. */
export interface ClassifierHandle {
	kind: "local" | "openrouter" | "litellm";
	/** "laya" for local, or the configured remote decision model id. */
	model: string;
	/** The classifier itself; null = local backend failed to load. */
	classifier: Classifier | null;
	/** Latest remote usage (set after each successful remote call). */
	lastCall: { inputTokens: number | null; outputTokens: number | null; cost: number | null } | null;
}

/** Virtual model entry backing: "classifier" = laya/auto, "jev" = jev/auto. */
export type ClassifierBackend = "classifier" | "jev";

/**
 * Resolve the configured classifier backend for a virtual model entry. Local
 * returns the ONNX Laya singleton (null when local inference failed to load -
 * logged, never retried via a remote provider). OpenRouter/LiteLLM return a
 * handle whose systemOne speaks the same answers contract over the decisions
 * or systemone protocol. Credentials: env var first, Pi auth file second.
 */
export async function getClassifier(
	config: RouterConfig,
	backend: ClassifierBackend = "classifier",
	onProgress?: LayaLoadProgress,
): Promise<ClassifierHandle> {
	const cfg = backend === "jev" ? config.jev ?? config.classifier : config.classifier;
	if (cfg.provider === "local") {
		const classifier = await getLaya(config.cacheDir, onProgress);
		return { kind: "local", model: "laya", classifier, lastCall: null };
	}
	const label = cfg.provider === "litellm" ? "LiteLLM" : "OpenRouter";
	// Base URL comes from the Pi auth file's provider entry (never plugin config,
	// so private endpoints stay out of shared config files); null = OpenRouter default.
	const endpoint = classifierEndpoint(cfg.protocol, resolveBaseUrl(cfg.provider, cfg.authFile));
	const handle: ClassifierHandle = { kind: cfg.provider, model: cfg.model, classifier: null, lastCall: null };
	handle.classifier = {
		systemOne: async (input, questions) => {
			const apiKey = resolveApiKey(cfg.apiKeyEnv, cfg.provider, cfg.authFile);
			if (!apiKey) {
				throw new RemoteClassifierError(
					`missing credentials: set the ${cfg.apiKeyEnv} environment variable or the "${cfg.provider}" key in the Pi auth file (classifier provider "${cfg.provider}")`,
				);
			}
			const proxyUrl = resolveProxyUrl(cfg.proxyUrl);
			const result = await callDecisions({
				model: cfg.model,
				role: input.role,
				prompt: input.prompt,
				questions,
				timeoutMs: cfg.timeoutMs,
				maxAttempts: cfg.maxAttempts,
				retryDelayMs: cfg.retryDelayMs,
				apiKey,
				proxyUrl,
				endpoint,
				providerLabel: label,
			});
			handle.lastCall = {
				inputTokens: result.usage.inputTokens,
				outputTokens: result.usage.outputTokens,
				cost: result.usage.cost,
			};
			return { answers: result.answers };
		},
		// Remote room: the configured state budget (tokens). head_max_len 0 - the
		// questions are sent as a separate field, not inside the state.
		config: { max_len: cfg.maxInputTokens + 4, head_max_len: 0 },
	};
	return handle;
}

// ---------------------------------------------------------------------------
// Eligibility, decision rules, fallback policy
// ---------------------------------------------------------------------------

/** Minimal registry model surface the router relies on (cost + limits). */
export interface RegistryModel {
	cost?: { input?: number; output?: number; cacheRead?: number; cacheWrite?: number };
	contextWindow?: number;
}

export interface RegistryLike {
	find(provider: string, id: string): RegistryModel | undefined;
}

/** Models that are enabled and actually present in the live registry,
 * before any group filtering. */
export function presentModels(config: RouterConfig, registry: RegistryLike): ModelEntry[] {
	return config.models.filter((entry) => entry.enabled !== false && registry.find(entry.provider, entry.id));
}

/** Apply the active-group filter ("saved filters") with the fallback chain:
 * activeGroup -> defaultGroup -> `default` model -> empty (the caller's
 * no-eligible-models error). The group is the universe: escalation, budget,
 * scoring, and fallback all operate on its models only. An unknown group
 * name degrades to the full catalog with a warning (warned once per process). */
function applyActiveGroup(config: RouterConfig, present: ModelEntry[]): ModelEntry[] {
	const active = config.activeGroup;
	if (active == null) return present;
	if (groupComposites(config, active) == null) {
		// Unknown name (only reachable when a config bypassed validation):
		// warn + ignore - the full catalog stays routable.
		warnOnce(`unknown:${active}`, `active group "${active}" is not defined; group filtering off`);
		return present;
	}
	const filter = (name: string): ModelEntry[] => {
		const composites = groupComposites(config, name);
		if (composites == null) return [];
		return filterByComposites(present, composites);
	};
	let pool = filter(active);
	if (pool.length === 0) {
		warnOnce(
			`group:${active}`,
			`group "${active}" has no eligible models${config.defaultGroup ? `; trying defaultGroup "${config.defaultGroup}"` : ""}`,
		);
	}
	if (pool.length === 0 && config.defaultGroup) {
		pool = filter(config.defaultGroup);
		if (pool.length > 0) {
			warnOnce(
				`defaultGroup:${config.defaultGroup}`,
				`group "${active}" empty; using defaultGroup "${config.defaultGroup}" (${pool.length} model${pool.length === 1 ? "" : "s"})`,
			);
			return pool;
		}
	}
	if (pool.length === 0 && config.defaultModel) {
		const match = present.find((entry) => composite(entry) === config.defaultModel);
		if (match) {
			warnOnce(
				`default:${config.defaultModel}`,
				`group "${active}" and defaultGroup empty; using default model ${config.defaultModel}`,
			);
			return [match];
		}
	}
	return pool;
}

/** Filter a model list by context-window fit (window unknown = kept; never
 * rejected on fabricated limits). When nothing fits, everything is kept
 * (logged) - routing degrades rather than throwing. */
export function windowEligible(
	models: readonly ModelEntry[],
	registry: RegistryLike,
	requiredTokens: number | null,
): ModelEntry[] {
	if (requiredTokens == null) return [...models];
	const excluded: string[] = [];
	const fits = models.filter((entry) => {
		const win = registry.find(entry.provider, entry.id)?.contextWindow;
		if (typeof win === "number" && win < requiredTokens) {
			excluded.push(composite(entry));
			return false;
		}
		return true;
	});
	if (fits.length === 0) {
		console.error(
			`${LOG_PREFIX} context window: every eligible model is below ~${requiredTokens} tokens; keeping all`,
		);
		return [...models];
	}
	if (excluded.length > 0) {
		const plural = excluded.length === 1 ? "" : "s";
		console.error(
			`${LOG_PREFIX} context window: excluding ${excluded.length} model${plural} below ~${requiredTokens} tokens: ${excluded.join(", ")}`,
		);
	}
	return fits;
}

/** Models that are enabled, valid, present in the live registry, and inside
 * the active group (when one is set). With `requiredTokens`, also excludes
 * models whose context window cannot hold the upcoming request. */
export function eligibleModels(
	config: RouterConfig,
	registry: RegistryLike,
	requiredTokens: number | null = null,
): ModelEntry[] {
	const present = presentModels(config, registry);
	const pool = applyActiveGroup(config, present);
	if (pool.length === 0) return pool;
	return windowEligible(pool, registry, requiredTokens);
}

/** The pool a `direct` request (compaction summaries, extension calls)
 * picks from: the models of `defaultGroup` when one is configured, else the
 * given models so the caller's `default`-then-cheapest preference applies.
 * Callers pass the present-but-not-activeGroup-filtered set: the active
 * group filters the chat loop, never compaction. Never throws: a group with
 * no eligible models degrades to the full given set. */
function directModels(config: RouterConfig, present: readonly ModelEntry[]): ModelEntry[] {
	const group = config.defaultGroup ? groupComposites(config, config.defaultGroup) : null;
	if (!group) return [...present];
	const pool = filterByComposites(present, group);
	if (pool.length === 0) {
		warnOnce(
			`direct:${config.defaultGroup}`,
			`defaultGroup "${config.defaultGroup}" has no eligible models for direct requests; using the full set`,
		);
		return [...present];
	}
	return pool;
}

function composite(entry: ModelEntry): string {
	return `${entry.provider}/${entry.id}`;
}

/** Effective input price (USD/1M tokens) for routing rules: live registry
 * cost, then the auto-refreshed price cache, then the config `priceIn`. */
export type PriceResolver = (entry: ModelEntry) => number | undefined;

function priceResolver(ctx: ExtensionContext, config: RouterConfig): PriceResolver {
	const dataset = config.prices.enabled ? readPriceDataset(config.prices.cachePath, config.prices.seedPath) : null;
	return (entry) =>
		ctx.modelRegistry.find(entry.provider, entry.id)?.cost?.input ??
		dataset?.prices[entry.id]?.input ??
		entry.priceIn;
}

/** Cached catalog price for a model id (null-cache-aware). */
function cachedPriceFor(config: RouterConfig, id: string): ModelPrice | undefined {
	return config.prices.enabled ? readPriceDataset(config.prices.cachePath, config.prices.seedPath)?.prices[id] : undefined;
}

/** Cheapest eligible entry: lowest input price when all entries carry prices, else file order first. */
export function cheapestEligible(eligible: ModelEntry[], priceIn?: PriceResolver): ModelEntry {
	const price = (e: ModelEntry) => priceIn?.(e) ?? e.priceIn;
	if (eligible.every((e) => price(e) !== undefined)) {
		return eligible.reduce((a, b) => ((price(a) ?? 0) <= (price(b) ?? 0) ? a : b));
	}
	return eligible[0];
}

/** Strongest eligible entry: highest input price when all entries carry prices, else file order last. */
export function strongestEligible(eligible: ModelEntry[], priceIn?: PriceResolver): ModelEntry {
	const price = (e: ModelEntry) => priceIn?.(e) ?? e.priceIn;
	if (eligible.every((e) => price(e) !== undefined)) {
		return eligible.reduce((a, b) => ((price(a) ?? 0) >= (price(b) ?? 0) ? a : b));
	}
	return eligible[eligible.length - 1];
}

function fallbackEntry(config: RouterConfig, eligible: ModelEntry[], priceIn?: PriceResolver): ModelEntry {
	if (eligible.length === 0) throw new Error(`${LOG_PREFIX} no eligible models: config entries missing from the model registry`);
	if (config.defaultModel) {
		const match = eligible.find((entry) => composite(entry) === config.defaultModel);
		if (match) return match;
		console.error(`${LOG_PREFIX} configured "default" ${config.defaultModel} is not eligible; using ${config.fallback}`);
	}
	return config.fallback === "strongest" ? strongestEligible(eligible, priceIn) : cheapestEligible(eligible, priceIn);
}

function argmaxModel(probabilities: Record<string, number>, eligible: ModelEntry[]): ModelEntry | null {
	let best: ModelEntry | null = null;
	let bestP = -1;
	let seen = false;
	for (const entry of eligible) {
		const raw = probabilities[composite(entry)];
		if (raw === undefined) continue; // absent from Laya's answer: not a pick
		if (!seen || raw > bestP) {
			seen = true;
			bestP = raw;
			best = entry;
		}
	}
	// All-zero or empty probabilities mean "no classification signal" -> null,
	// so the deterministic fallback policy decides instead.
	if (best && bestP <= 0) return null;
	return seen ? best : null;
}

const ESCALATE_CATEGORIES = new Set<Category>(["coding", "debugging", "reasoning", "infra"]);

/** Inputs for the deterministic scoring policy (absent = classifier pick only). */
export interface ScoreContext {
	dataset: BenchmarkDataset | null;
	/** Estimated total task cost in USD for an entry (null = missing price). */
	estimatedCost: (entry: ModelEntry) => number | null;
	/** Per-entry measured internal benchmark overrides from the config. */
	capabilities?: (entry: ModelEntry) => CapabilityOverrides | undefined;
}

/**
 * Deterministic post-classification rules. The model pick comes from the
 * classifier's probabilities, or - when scoring is enabled and benchmark data
 * is available - from the deterministic scoring policy over the same eligible
 * set. The pick is validated against the eligible set, then refined: complex
 * demanding work never stays on the weakest tier, and budget-tagged work
 * never pays above the weakest tier.
 */
export function decide(
	probabilities: {
		model: Record<string, number>;
		complexity: Record<string, number>;
		category: Record<string, number>;
	},
	config: RouterConfig,
	eligible: ModelEntry[],
	scoreContext?: ScoreContext,
	priceIn?: PriceResolver,
): RoutingDecision {
	const notes: string[] = [];
	const complexity = argmaxKey(probabilities.complexity, COMPLEXITIES) ?? "standard";
	const category = argmaxKey(probabilities.category, CATEGORIES) ?? "coding";
	let picked = argmaxModel(probabilities.model, eligible);
	let scored = false;
	if (config.scoring.enabled && scoreContext) {
		// Benchmark-driven selection: same eligible set, same safety rules - only
		// the pick source changes. Missing/stale data falls back explicitly.
		const result = scoreEligible({
			eligible,
			category,
			complexity,
			config,
			dataset: scoreContext.dataset,
			estimatedCost: scoreContext.estimatedCost,
			capabilities: scoreContext.capabilities,
			classifierProbability: (entry: ModelEntry) => probabilities.model[composite(entry)] ?? 0,
		});
		if (result.ranked.length > 0) {
			const top = result.ranked[0];
			picked = top.entry;
			scored = true;
			notes.push(
				`scored pick score=${top.score} quality=${top.quality} cost=${top.cost}` +
					(top.notes.length ? ` (${top.notes.join("; ")})` : ""),
			);
		} else {
			notes.push("benchmark scoring unavailable (no measured quality): classifier pick used");
		}
	}
	// Scored picks are made by benchmark scoring, not the classifier; its
	// probability would misattribute the decision. The score is in the notes.
	const confidence = picked && !scored ? (probabilities.model[composite(picked)] ?? null) : null;

	if (!picked) {
		notes.push("Laya selection rejected: outside eligible set");
		const entry = fallbackEntry(config, eligible, priceIn);
		return {
			entry,
			rule: "fallback",
			complexity,
			category,
			confidence,
			notes,
			estimate: emptyEstimate(),
		};
	}
	let entry = picked;
	let rule: RoutingDecision["rule"] = scored ? "scored" : "primary";
	if (
		complexity === "complex" &&
		ESCALATE_CATEGORIES.has(category) &&
		entry === cheapestEligible(eligible, priceIn) &&
		eligible.length > 1
	) {
		// Quality rule, independent of the failure fallback policy: complex
		// demanding work escalates to the strongest eligible model.
		const strongest = strongestEligible(eligible, priceIn);
		if (strongest && strongest !== picked) {
			entry = strongest;
			rule = "escalated";
			notes.push(`complex ${category} work escalated off cheapest tier`);
		}
	} else if (category === "budget" && entry !== cheapestEligible(eligible, priceIn)) {
		const cheapest = cheapestEligible(eligible, priceIn);
		if (cheapest && cheapest !== entry) {
			entry = cheapest;
			rule = "downgraded";
			notes.push("budget-tagged work downgraded to cheapest eligible");
		}
	}
	return {
		entry,
		rule,
		complexity,
		category,
		confidence,
		notes,
		estimate: emptyEstimate(),
	};
}

export function emptyEstimate(): CostEstimate {
	return { inputTokens: null, outputTokens: null, estCostIn: null, estCostOut: null, source: null };
}

function roundUsd(value: number): number {
	return Number(value.toFixed(6));
}

function argmaxKey<T extends string>(probabilities: Record<string, number>, keys: readonly T[]): T | null {
	let best: T | null = null;
	let bestP = -1;
	for (const key of keys) {
		const p = probabilities[key] ?? 0;
		if (p > bestP) {
			bestP = p;
			best = key;
		}
	}
	// All-zero or empty probabilities mean "no classification signal" -> null,
	// mirroring argmaxModel, so the documented "standard"/"coding" defaults apply.
	if (bestP <= 0) return null;
	return best;
}

// ---------------------------------------------------------------------------
// Cost estimation and observability
// ---------------------------------------------------------------------------

export interface SessionPeek {
	/** Latest compaction/branch summary text, if any. */
	summary: string | null;
	/** Actual usage of the most recent billed request, if any. */
	lastUsage: { provider: string; model: string; inputTokens: number; outputTokens: number; cost: number } | null;
}

type SessionManagerLike = {
	getEntries?: () => unknown[];
};

/** Read trustworthy session data (read-only): latest summary and latest usage entry. */
export function peekSession(ctx: ExtensionContext): SessionPeek {
	const peek: SessionPeek = { summary: null, lastUsage: null };
	try {
		const entries = ctx.sessionManager?.getEntries?.() as unknown[] | undefined ?? [];
		for (let i = entries.length - 1; i >= 0; i--) {
			const entry = entries[i] as { type?: string; summary?: string };
			if ((entry.type === "compaction" || entry.type === "branch_summary") && typeof entry.summary === "string") {
				peek.summary = entry.summary;
				break;
			}
		}
		for (let i = entries.length - 1; i >= 0; i--) {
			const entry = entries[i] as {
				type?: string;
				provider?: string;
				model?: string;
				usage?: Usage & { totalTokens?: number };
			};
			if (entry.type === "usage" && entry.usage && typeof entry.model === "string") {
				peek.lastUsage = {
					provider: entry.provider ?? "?",
					model: entry.model,
					inputTokens: entry.usage.input ?? 0,
					outputTokens: entry.usage.output ?? 0,
					cost: entry.usage.cost?.total ?? 0,
				};
				break;
			}
		}
	} catch {
		// Session data is best-effort context; never fabricate it.
	}
	return peek;
}

/** Estimate the cost of the upcoming request. Never fabricated: falls back to null.
 * Prices are USD per 1M tokens. Input tokens: previous request's actual usage
 * (only when it ran on the SAME provider+model), then Pi context tokens, then
 * chars/4 of the assembled routing input (not the clamped current request).
 * Output tokens: only from the previous request's actual usage. */
export function estimateCost(
	entry: ModelEntry,
	registryModel: RegistryModel | undefined,
	promptChars: number,
	contextTokens: number | null,
	lastUsage: SessionPeek["lastUsage"],
	/** Auto-refreshed catalog price; registry cost still wins. */
	cachedPrice?: ModelPrice | null,
): CostEstimate {
	const priceIn = registryModel?.cost?.input ?? cachedPrice?.input ?? entry.priceIn ?? null;
	const priceOut = registryModel?.cost?.output ?? cachedPrice?.output ?? entry.priceOut ?? null;
	const usageMatches =
		lastUsage != null && lastUsage.model === entry.id && lastUsage.provider === entry.provider;
	let inputTokens: number | null = null;
	let outputTokens: number | null = null;
	let source: CostEstimate["source"] = null;
	if (usageMatches) {
		inputTokens = lastUsage.inputTokens;
		outputTokens = lastUsage.outputTokens;
		source = "usage";
	} else if (typeof contextTokens === "number" && contextTokens > 0) {
		inputTokens = contextTokens;
		source = "context";
	} else if (promptChars > 0) {
		inputTokens = Math.ceil(promptChars / 4);
		source = "chars";
	}
	const estCostIn =
		priceIn !== null && inputTokens !== null ? roundUsd((inputTokens / 1e6) * priceIn) : null;
	const estCostOut =
		priceOut !== null && outputTokens !== null ? roundUsd((outputTokens / 1e6) * priceOut) : null;
	return { inputTokens, outputTokens, estCostIn, estCostOut, source };
}

/** Per-request routing diagnostics. In the TUI, stderr output corrupts the
 * input line, so info lines are shown in the status bar (ctx.ui.setStatus)
 * instead; warnings still go to stderr AND notify(). Non-TUI modes (print,
 * JSON, RPC) always log to stderr. TRIAGE_ROUTER_LOG=1 forces console output. */
function reportRoute(
	ctx: ExtensionContext | undefined,
	message: string,
	statusText?: string | null,
	level: "info" | "warn" = "info",
): void {
	const tui =
		!!ctx && ctx.mode === "tui" && ctx.hasUI && typeof ctx.ui?.setStatus === "function";
	if (level === "warn" || !tui || process.env.TRIAGE_ROUTER_LOG === "1") {
		console.error(`${LOG_PREFIX} ${message}`);
	}
	if (!tui || !ctx) return;
	if (statusText != null) ctx.ui.setStatus("laya-route", `→ ${statusText}`);
	if (level === "warn" && typeof ctx.ui.notify === "function") {
		ctx.ui.notify(`${LOG_PREFIX} ${message}`, "warning");
	}
}

function logDecision(
	decision: RoutingDecision,
	layaMs: number | null,
	input: RoutingInput,
	lastCost: number | null,
	ctx?: ExtensionContext,
): void {
	const est = decision.estimate;
	reportRoute(
		ctx,
		`route=${decision.rule} model=${composite(decision.entry)} ` +
			`cat=${decision.category} cplx=${decision.complexity} conf=${decision.confidence ?? "-"} ` +
			`laya_ms=${layaMs ?? "-"} state_tokens=${input.stateTokens ?? "-"}/${input.roomTokens} ` +
			`ctx_chars=(recent=${input.recent.length} summary=${input.summary.length} ` +
			`request=${input.currentRequest.length}) ` +
			`est_in=${est?.inputTokens ?? "?"}(${est?.source ?? "-"}) est_cost_in=${est?.estCostIn ?? "?"} ` +
			`est_out=${est?.outputTokens ?? "?"} est_cost_out=${est?.estCostOut ?? "?"} ` +
			`last_cost=${lastCost ?? "-"}(actual, previous request) ` +
			`note="${decision.notes.join("; ") || "-"}"`,
		`${decision.entry.id} · ${decision.rule}` +
			(est?.estCostIn != null ? ` · ~$${Number(est.estCostIn).toFixed(4)}/req` : ""),
	);
	// Session history for the /triage dashboard (in-memory, bounded).
	recordRoute({
		model: composite(decision.entry),
		rule: decision.rule,
		complexity: decision.complexity,
		category: decision.category,
		cost: recordCost(est?.estCostIn, est?.estCostOut),
	});
}

// ---------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------

/** Resolve a config entry to a live registry model. Throws for missing models. */
export function buildRoute(
	request: LayaRouteRequest,
	ctx: ExtensionContext,
	entry: ModelEntry,
	state: LayaState,
): ModelRoute<LayaState> {
	const model = ctx.modelRegistry.find(entry.provider, entry.id);
	if (!model) throw new Error(`${LOG_PREFIX} model ${composite(entry)} disappeared from the registry`);
	return { model, thinkingLevel: request.thinkingLevel, state };
}

function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
	return new Promise<T>((resolve, reject) => {
		const timer = setTimeout(() => reject(new Error(`Classification timed out after ${ms}ms`)), ms);
		promise.then(
			(value) => {
				clearTimeout(timer);
				resolve(value);
			},
			(err) => {
				clearTimeout(timer);
				reject(err);
			},
		);
	});
}

function emptyProbabilities(): { model: Record<string, number>; complexity: Record<string, number>; category: Record<string, number> } {
	return { model: {}, complexity: {}, category: {} };
}

/** Per-classification cost/latency accounting - kept strictly separate from execution-model cost. */
export interface ClassifierCallStats {
	provider: "local" | "openrouter" | "litellm";
	model: string;
	ms: number | null;
	inputTokens: number | null;
	outputTokens: number | null;
	/** Actual USD when the provider reports it; 0 for local (no API charge); null = unknown/failed. */
	cost: number | null;
	/** "actual" = provider-reported, "none" = free local run, "unavailable" = remote call ok but no cost reported, null = no call. */
	costKind: "actual" | "none" | "unavailable" | null;
}

/**
 * Classify with the configured backend (local Laya by default; the remote
 * Decisions API only when explicitly selected) and decide. On failure the
 * configured failure policy applies - never a silent switch to another
 * classifier provider.
 */
export async function classifyAndDecide(
	request: LayaRouteRequest,
	ctx: ExtensionContext,
	config: RouterConfig,
	eligible: ModelEntry[],
	backend: ClassifierBackend = "classifier",
): Promise<{
		decision: RoutingDecision;
		layaMs: number | null;
		input: RoutingInput;
		summary: string;
		lastCost: number | null;
		clf: ClassifierCallStats;
	}> {
	const section = backend === "jev" ? config.jev ?? config.classifier : config.classifier;
	const stateSummary = request.state?.summary ?? "";
	const session = peekSession(ctx);
	if (eligible.length === 0) {
		throw new Error(`${LOG_PREFIX} no eligible models to route to`);
	}
	// First local load can include a ~2 GB weight download that takes many
	// minutes: bound how long this request waits (the load keeps running in the
	// background) and fall back to deterministic routing instead of hanging the
	// turn - classification resumes normally on a later request once it resolves.
	const onProgress: LayaLoadProgress | undefined =
		section.provider === "local"
			? (() => {
					let lastAt = 0;
					return (info): void => {
						const now = Date.now();
						if (now - lastAt < 800) return;
						lastAt = now;
						const mb = (n: number): string => `${Math.round(n / 1048576)}MB`;
						// Some servers report no/zero content-length: a "0MB" total is
						// unknown, not 0 bytes, so omit it rather than print a wrong one.
						const total = info.total != null && info.total > 0 ? info.total : null;
						const text = `downloading ${info.file} ${mb(info.received)}` +
							(total != null ? `/${mb(total)}` : "");
						reportRoute(ctx, `laya: ${text}`, `laya: ${text}…`);
					};
				})()
			: undefined;
	let waitMs = LAYA_LOAD_TIMEOUT_MS;
	if (section.provider === "local" && layaLoadStartedAt !== null && !layaLoadSettled) {
		const elapsed = Date.now() - layaLoadStartedAt;
		waitMs = Math.max(LAYA_LOAD_MIN_WAIT_MS, LAYA_LOAD_TIMEOUT_MS - elapsed);
		if (elapsed > 1_500) {
			reportRoute(
				ctx,
				`local classifier still loading (${Math.round(elapsed / 1000)}s) - the first run downloads ~2 GB of weights; routing deterministically until it is ready`,
				`laya: loading ${Math.round(elapsed / 1000)}s (first run downloads ~2 GB)…`,
			);
		}
	}
	let handle: ClassifierHandle;
	try {
		handle = await withTimeout(getClassifier(config, backend, onProgress), waitMs);
	} catch (err) {
		const message = err instanceof Error ? err.message : String(err);
		if (section.failurePolicy === "error") {
			throw new Error(`${LOG_PREFIX} classifier load failed (failurePolicy=error): ${message}`);
		}
		handle = {
			kind: section.provider,
			model: section.provider === "local" ? "laya" : section.model,
			classifier: null,
			lastCall: null,
		};
		reportRoute(
			ctx,
			`classifier ${handle.kind}/${handle.model} not ready after ${waitMs}ms (${message}); deterministic policy decides`,
			`laya: not ready → deterministic route`,
		);
	}
	const roleText = handle.kind === "local" ? section.localPrompt : section.remotePrompt;
	// Roll the summary forward FIRST so the prompt and the stored state are the
	// same text: the turn that just left the recent window is visible now, not
	// one request later (no one-turn blind spot).
	const summary = refreshSummary(stateSummary, request.messages, config);
	const input = buildRoutingInput(request.messages, summary, session.summary, config, handle.classifier, roleText);
	const criteria: Record<string, string> = {};
	for (const entry of eligible) criteria[composite(entry)] = describe(entry);
	const questions = {
		model: {
			type: "choice",
			instructions:
				"Which model should handle the work in the current request? Base the choice primarily on the current request; use the conversation context only to resolve references and constraints.",
			criteria,
		},
		complexity: {
			type: "choice",
			instructions: "How demanding is the engineering work in the current request?",
			criteria: {
				standard: "A question, explanation, or a small single-file change",
				moderate: "A routine multi-step task or moderate refactor",
				complex: "Subtle design, cross-cutting changes, or hard debugging",
			},
		},
		category: {
			type: "choice",
			instructions: "What kind of work is the current request?",
			criteria: {
				question: "A conceptual question or explanation request",
				coding: "Implementing or changing code",
				infra: "Infrastructure or DevOps: Kubernetes, Linux, networking, databases, deployment",
				debugging: "Diagnosing and fixing a failure",
				reasoning: "Architecture design or hard analysis",
				research: "Gathering information or exploring a topic",
				long_horizon: "A long-running multi-step agentic task",
				budget: "Work where the user explicitly prioritizes low cost",
			},
		},
	};
	let probabilities = emptyProbabilities();
	let clfMs: number | null = null;
	if (handle.classifier) {
		const started = Date.now();
		try {
			// Remote retries extend the wall-clock budget: per-attempt timeouts sum,
			// plus the gaps between attempts. Local inference keeps a single budget.
			const budget = handle.kind === "local"
				? section.timeoutMs
				: section.timeoutMs * section.maxAttempts + section.retryDelayMs * (section.maxAttempts - 1);
			const result = await withTimeout(
				handle.classifier.systemOne({ role: roleText, prompt: input.prompt }, questions),
				budget,
			);
			clfMs = Date.now() - started;
			probabilities = {
				model: result.answers.model?.probabilities ?? {},
				complexity: result.answers.complexity?.probabilities ?? {},
				category: result.answers.category?.probabilities ?? {},
			};
		} catch (err) {
			clfMs = Date.now() - started;
			if (section.failurePolicy === "error") {
				throw new Error(
					`${LOG_PREFIX} classifier ${handle.kind}/${handle.model} failed (failurePolicy=error): ${err instanceof Error ? err.message : String(err)}`,
				);
			}
			console.error(
				`${LOG_PREFIX} classifier ${handle.kind}/${handle.model} failed; deterministic policy decides:`,
				err instanceof Error ? err.message : err,
			);
		}
	}
	const clf: ClassifierCallStats =
		handle.kind === "local"
			? {
					provider: "local",
					model: handle.model,
					ms: clfMs,
					inputTokens: null,
					outputTokens: null,
					cost: clfMs !== null ? 0 : null,
					costKind: clfMs !== null ? "none" : null,
				}
			: {
					provider: handle.kind,
					model: handle.model,
					ms: clfMs,
					inputTokens: handle.lastCall?.inputTokens ?? null,
					outputTokens: handle.lastCall?.outputTokens ?? null,
					cost: handle.lastCall?.cost ?? null,
					costKind: handle.lastCall
						? handle.lastCall.cost !== null
							? "actual"
							: "unavailable"
						: null,
				};
	const scoreContext: ScoreContext | undefined = config.scoring.enabled
		? {
				dataset: readDataset(config.benchmarks.cachePath, config.benchmarks.seedPath),
				estimatedCost: (entry: ModelEntry) => {
					const reg = ctx.modelRegistry.find(entry.provider, entry.id);
					const cached = cachedPriceFor(config, entry.id);
					const pIn = reg?.cost?.input ?? cached?.input ?? entry.priceIn;
					const pOut = reg?.cost?.output ?? cached?.output ?? entry.priceOut;
					if (typeof pIn !== "number" || typeof pOut !== "number") return null;
					return (
						(config.scoring.expectedInputTokens / 1e6) * pIn +
						(config.scoring.expectedOutputTokens / 1e6) * pOut
					);
				},
				capabilities: (entry: ModelEntry) => entry.capabilities,
			}
		: undefined;
	const priceIn = priceResolver(ctx, config);
	const decision = decide(probabilities, config, eligible, scoreContext, priceIn);
	const registryModel = ctx.modelRegistry.find(decision.entry.provider, decision.entry.id);
	const contextTokens = safeContextTokens(ctx);
	decision.estimate = estimateCost(
		decision.entry,
		registryModel,
		input.prompt.length,
		contextTokens,
		session.lastUsage,
		cachedPriceFor(config, decision.entry.id) ?? null,
	);
	return { decision, layaMs: clfMs, input, summary, lastCost: session.lastUsage?.cost ?? null, clf };
}

function safeContextTokens(ctx: ExtensionContext): number | null {
	try {
		const usage = ctx.getContextUsage?.();
		return usage && typeof usage.tokens === "number" ? usage.tokens : null;
	} catch {
		return null;
	}
}

// ---------------------------------------------------------------------------
// /triage interactive menu (TUI dialogs; never writes to the chat transcript)
// ---------------------------------------------------------------------------

const MENU_BACK = "Back";
const MENU_DONE = "Done";

/** Extracts the "provider/id" prefix from a menu option line. */
function compositeFromOptionLine(line: string, config: RouterConfig): ModelEntry | null {
	const id = line.split(" · ")[0];
	return config.models.find((m) => composite(m) === id) ?? null;
}

/** A /triage menu save: write the patch, refresh the shared config, confirm. */
type MenuSave = (patch: Record<string, unknown>, note: string) => Promise<void>;
/** Reads the current config (post-save) so menu loops never use stale state. */
type GetConfig = () => RouterConfig;

function makeSave(ctx: ExtensionContext, setConfig: (c: RouterConfig) => void): MenuSave {
	return async (patch, note) => {
		let config: RouterConfig;
		try {
			config = updateConfigFile(CONFIG_PATH, patch);
		} catch (err) {
			ctx.ui.notify(`Save failed: ${err instanceof Error ? err.message : String(err)}`, "error");
			return;
		}
		setConfig(config);
		ctx.ui.notify(note, "info");
	};
}

/** Build the next models array with `enabled` flipped for one id. */
function toggleModelEnabled(config: RouterConfig, id: string): ModelEntry[] {
	return config.models.map((m) => (m.id === id ? { ...m, enabled: m.enabled === false } : m));
}

async function lineupMenu(ctx: ExtensionContext, getConfig: GetConfig, save: MenuSave): Promise<void> {
	for (;;) {
		const config = getConfig();
		const pick = await ctx.ui.select("Model lineup", [...modelOptions(config), MENU_BACK]);
		if (!pick || pick === MENU_BACK) return;
		const entry = compositeFromOptionLine(pick, config);
		if (!entry) continue;
		ctx.ui.notify(modelDetail(config, entry.id), "info");
		const toggleLabel = entry.enabled === false ? "Enable" : "Disable";
		const action = await ctx.ui.select(composite(entry), ["Set as manual override", toggleLabel, MENU_BACK]);
		if (!action || action === MENU_BACK) continue;
		if (action === "Set as manual override") {
			await save({ override: composite(entry) }, `override = ${composite(entry)} saved`);
		} else {
			await save({ models: toggleModelEnabled(config, entry.id) }, `${composite(entry)} ${toggleLabel.toLowerCase()}d`);
		}
	}
}

async function historyMenu(ctx: ExtensionContext): Promise<void> {
	for (;;) {
		const lines = historyOptions();
		if (lines.length === 0) {
			ctx.ui.notify("No routes recorded yet this session.", "info");
			return;
		}
		const pick = await ctx.ui.select("Route history (newest first)", [...lines, MENU_BACK]);
		if (!pick || pick === MENU_BACK) return;
		ctx.ui.notify(pick, "info");
	}
}

/** Shared prompt for a classifier timeout (ms >= 100); null = cancelled/invalid. */
async function promptTimeoutMs(ctx: ExtensionContext, current: number): Promise<number | null> {
	const value = await ctx.ui.input(`Timeout in ms, >= 100 (current ${current})`, String(current));
	if (!value) return null;
	const parsed = Number(value);
	if (!Number.isFinite(parsed) || parsed < 100) {
		ctx.ui.notify("Invalid timeout; needs a number >= 100.", "warning");
		return null;
	}
	return Math.round(parsed);
}

/** Shared prompt for classifier retry attempts (integer 1..10); null = cancelled/invalid. */
async function promptAttempts(ctx: ExtensionContext, current: number): Promise<number | null> {
	const value = await ctx.ui.input(`Attempts 1..10 (current ${current})`, String(current));
	if (!value) return null;
	const parsed = Number(value);
	if (!Number.isInteger(parsed) || parsed < 1 || parsed > 10) {
		ctx.ui.notify("Invalid attempts; needs an integer 1..10.", "warning");
		return null;
	}
	return parsed;
}

/** Edits the laya classifier section: backs laya/auto on the local Laya
 * classifier. The provider is intentionally not editable here - the laya
 * classifier stays local; remote providers remain a config-file option. */
async function classifierMenu(ctx: ExtensionContext, getConfig: GetConfig, save: MenuSave): Promise<void> {
	for (;;) {
		const clf = getConfig().classifier;
		const pick = await ctx.ui.select("Laya classifier (laya/auto)", [
			`Model id — ${clf.provider === "local" ? "(unused for local)" : clf.model}`,
			`Timeout ms — ${clf.timeoutMs}`,
			`Max attempts — ${clf.maxAttempts}`,
			MENU_BACK,
		]);
		if (!pick || pick === MENU_BACK) return;
		if (pick.startsWith("Model id")) {
			if (clf.provider === "local") {
				ctx.ui.notify("Model id is remote-only; the laya classifier stays local.", "warning");
				continue;
			}
			const model = await ctx.ui.input("Classifier model id (e.g. typesafe/jev-1.13)", clf.model);
			const trimmed = model?.trim();
			if (!trimmed) continue;
			await save({ classifier: { model: trimmed } }, `classifier.model = ${trimmed} saved`);
		} else if (pick.startsWith("Timeout ms")) {
			const timeoutMs = await promptTimeoutMs(ctx, clf.timeoutMs);
			if (timeoutMs != null) await save({ classifier: { timeoutMs } }, `classifier.timeoutMs = ${timeoutMs} saved`);
		} else if (pick.startsWith("Max attempts")) {
			const maxAttempts = await promptAttempts(ctx, clf.maxAttempts);
			if (maxAttempts != null) await save({ classifier: { maxAttempts } }, `classifier.maxAttempts = ${maxAttempts} saved`);
		}
	}
}

/** Validate a pending jev provider+model change before saving it.
 * Denies definite failures (no credentials, model confirmed absent) with the
 * check message; saves with a warning when the provider is unreachable
 * (network errors never lock an offline laptop out of configuring). */
async function validatedJevSave(
	ctx: ExtensionContext,
	save: MenuSave,
	jev: ClassifierConfig | null,
	patch: Record<string, unknown>,
	next: { provider: string; model: string },
	note: string,
): Promise<void> {
	ctx.ui.notify(`Validating ${next.provider}/${next.model}...`, "info");
	const base = jev ?? { ...defaultClassifier(5000), provider: "openrouter" };
	const section: RemoteSectionLike = {
		provider: next.provider,
		model: next.model,
		// Fresh setup: the provider's conventional env var. An existing jev
		// section carries its (validated) apiKeyEnv; switching providers keeps it.
		apiKeyEnv: jev ? jev.apiKeyEnv : defaultApiKeyEnv(next.provider),
		authFile: base.authFile,
		proxyUrl: base.proxyUrl,
	};
	const check = await checkRemoteClassifier(section, "jev");
	if (check.level === "problem") {
		ctx.ui.notify(check.message, "error");
		return;
	}
	await save(patch, note);
	if (check.level === "warn") ctx.ui.notify(check.message, "warning");
}

/** Edits the jev classifier section: backs jev/auto (remote-only, openrouter or
 * litellm). When unconfigured, offers to write a full jev section - jev/auto
 * registers at extension load, so a restart is needed for it to appear.
 * Provider and model edits are validated (credentials + model existence on
 * the provider) and denied when the check fails. */
async function jevClassifierMenu(ctx: ExtensionContext, getConfig: GetConfig, save: MenuSave): Promise<void> {
	for (;;) {
		const jev = getConfig().jev;
		if (!jev) {
			const pick = await ctx.ui.select("Jev classifier (jev/auto)", ["Not configured - set it up", MENU_BACK]);
			if (!pick || pick === MENU_BACK) return;
			const provider = await ctx.ui.select("Jev provider", ["openrouter", "litellm"]);
			if (!provider) continue;
			const model = await ctx.ui.input("Jev model id (e.g. typesafe/jev-1.13)", "typesafe/jev-1.13");
			const trimmed = model?.trim();
			if (!trimmed) continue;
			await validatedJevSave(
				ctx,
				save,
				null,
				{ jev: { provider, model: trimmed } },
				{ provider, model: trimmed },
				`jev = ${provider}/${trimmed} saved - jev/auto appears in /models after restart`,
			);
			continue;
		}
		const pick = await ctx.ui.select("Jev classifier (jev/auto)", [
			`Provider — ${jev.provider}`,
			`Model id — ${jev.model}`,
			`Timeout ms — ${jev.timeoutMs}`,
			`Max attempts — ${jev.maxAttempts}`,
			MENU_BACK,
		]);
		if (!pick || pick === MENU_BACK) return;
		if (pick.startsWith("Provider")) {
			// No "local": the jev entry is the remote backend by definition.
			const provider = await ctx.ui.select("Jev provider", ["openrouter", "litellm"]);
			if (!provider) continue;
			// The unchanged model id must exist on the new provider too.
			await validatedJevSave(ctx, save, jev, { jev: { provider } }, { provider, model: jev.model }, `jev.provider = ${provider} saved`);
		} else if (pick.startsWith("Model id")) {
			const model = await ctx.ui.input("Jev model id (e.g. typesafe/jev-1.13)", jev.model);
			const trimmed = model?.trim();
			if (!trimmed) continue;
			await validatedJevSave(ctx, save, jev, { jev: { model: trimmed } }, { provider: jev.provider, model: trimmed }, `jev.model = ${trimmed} saved`);
		} else if (pick.startsWith("Timeout ms")) {
			const timeoutMs = await promptTimeoutMs(ctx, jev.timeoutMs);
			if (timeoutMs != null) await save({ jev: { timeoutMs } }, `jev.timeoutMs = ${timeoutMs} saved`);
		} else if (pick.startsWith("Max attempts")) {
			const maxAttempts = await promptAttempts(ctx, jev.maxAttempts);
			if (maxAttempts != null) await save({ jev: { maxAttempts } }, `jev.maxAttempts = ${maxAttempts} saved`);
		}
	}
}

async function editMenu(ctx: ExtensionContext, getConfig: GetConfig, save: MenuSave): Promise<void> {
	for (;;) {
		const config = getConfig();
		const eligible = config.models.filter((m) => m.enabled !== false).map((m) => composite(m));
		const clf = config.classifier;
		const jev = config.jev;
		const groupNames = Object.keys(config.groups);
		const edit = await ctx.ui.select("Edit config", [
			`Manual override — ${config.override ?? "none"}`,
			`Active group — ${activeGroupLabel(config)}`,
			`Default group — ${groupLabel(config, "defaultGroup")}`,
			`Fallback policy — ${config.fallback}`,
			`Scoring — ${config.scoring.enabled ? "enabled" : "disabled"}`,
			`Price auto-refresh — ${config.prices.enabled ? "enabled" : "disabled"}`,
			`Laya classifier — ${clf.provider}${clf.provider === "local" ? "" : ` (${clf.model})`}`,
			`Jev classifier — ${jev ? `${jev.provider} (${jev.model})` : "not configured"}`,
			"Enable/disable a model",
			MENU_BACK,
		]);
		if (!edit || edit === MENU_BACK) return;
		if (edit.startsWith("Manual override")) {
			const pick = await ctx.ui.select("Manual override", ["(none)", ...eligible]);
			if (!pick) continue;
			await save(
				{ override: pick === "(none)" ? null : pick },
				pick === "(none)" ? "override cleared" : `override = ${pick} saved`,
			);
		} else if (edit.startsWith("Active group")) {
			if (groupNames.length === 0) {
				// No groups defined in the config file: nothing to switch to.
				ctx.ui.notify("No groups defined in the config file (add a \"groups\" section first)", "info");
				continue;
			}
			const picks = ["(none — full catalog)", ...groupNames];
			const pick = await ctx.ui.select("Active group", picks);
			if (!pick) continue;
			if (pick === "(none — full catalog)") {
				if (config.activeGroup == null) continue;
				await save({ activeGroup: null }, "activeGroup cleared (full catalog)");
			} else {
				if (config.activeGroup === pick) continue;
				await save({ activeGroup: pick }, `activeGroup = "${pick}" saved`);
			}
		} else if (edit.startsWith("Default group")) {
			if (groupNames.length === 0) {
				// Mirror of the Active group guard: no groups defined, nothing to
				// pick from. (A stale defaultGroup cannot exist here: validation
				// warns and ignores names not in "groups" on load.)
				ctx.ui.notify("No groups defined in the config file (add a \"groups\" section first)", "info");
				continue;
			}
			// defaultGroup drives /compact and out-of-loop picks, and rescues an
			// empty activeGroup. (none) clears it: /compact then prefers the
			// `default` model, then the cheapest present model.
			const picks = ["(none — default model, then cheapest)", ...groupNames];
			const pick = await ctx.ui.select("Default group (drives /compact picks)", picks);
			if (!pick) continue;
			if (pick === "(none — default model, then cheapest)") {
				if (config.defaultGroup == null) continue;
				await save({ defaultGroup: null }, "defaultGroup cleared");
			} else {
				if (config.defaultGroup === pick) continue;
				await save({ defaultGroup: pick }, `defaultGroup = "${pick}" saved`);
			}
		} else if (edit.startsWith("Fallback policy")) {
			const pick = await ctx.ui.select("Fallback policy", ["cheapest", "strongest"]);
			if (pick) await save({ fallback: pick }, `fallback = ${pick} saved`);
		} else if (edit.startsWith("Scoring")) {
			const pick = await ctx.ui.select("Scoring", ["enabled", "disabled"]);
			if (pick) await save({ scoring: { enabled: pick === "enabled" } }, `scoring ${pick} saved`);
		} else if (edit.startsWith("Price auto-refresh")) {
			const pick = await ctx.ui.select("Price auto-refresh", ["enabled", "disabled"]);
			if (pick) await save({ prices: { enabled: pick === "enabled" } }, `price auto-refresh ${pick} saved`);
		} else if (edit.startsWith("Laya classifier")) {
			await classifierMenu(ctx, getConfig, save);
		} else if (edit.startsWith("Jev classifier")) {
			await jevClassifierMenu(ctx, getConfig, save);
		} else if (edit === "Enable/disable a model") {
			const pick = await ctx.ui.select("Models", modelOptions(config));
			if (!pick) continue;
			const entry = compositeFromOptionLine(pick, config);
			if (!entry) continue;
			const label = entry.enabled === false ? "enabled" : "disabled";
			await save({ models: toggleModelEnabled(config, entry.id) }, `${composite(entry)} ${label} saved`);
		}
	}
}

/** Interactive /triage dashboard loop. Esc/"Done" exits. */
async function triageMenu(ctx: ExtensionContext): Promise<void> {
	let config = loadConfig();
	const getConfig: GetConfig = () => config;
	const save = makeSave(ctx, (c) => {
		config = c;
	});
	for (;;) {
		const enabled = config.models.filter((m) => m.enabled !== false).length;
		const total = sessionCost().total;
		const choice = await ctx.ui.select("pi-triage-router — dashboard", [
			`Model lineup (${enabled}/${config.models.length} active)`,
			`Session cost${total != null ? ` — ${usdLabel(total)}` : ""}`,
			`Route history (${routeHistory().length} routes)`,
			"Cache status",
			"Current config",
			"Edit config",
			"Reset config to defaults",
			MENU_DONE,
		]);
		if (!choice || choice === MENU_DONE) return;
		if (choice.startsWith("Model lineup")) await lineupMenu(ctx, getConfig, save);
		else if (choice.startsWith("Session cost")) ctx.ui.notify(costSummary(), "info");
		else if (choice.startsWith("Route history")) await historyMenu(ctx);
		else if (choice === "Cache status") ctx.ui.notify(cacheSummary(config), "info");
		else if (choice === "Current config") ctx.ui.notify(configSummary(config), "info");
		else if (choice === "Edit config") await editMenu(ctx, getConfig, save);
		else if (choice === "Reset config to defaults") await resetMenu(ctx, getConfig, (c) => (config = c));
	}
}

/** Confirm-and-reset flow: back up the current file, restore the shipped
 * defaults. /models registration happens at extension load, so a restart is
 * mentioned when the jev backend's presence changes. */
async function resetMenu(ctx: ExtensionContext, getConfig: GetConfig, setConfig: (c: RouterConfig) => void): Promise<void> {
	const confirm = await ctx.ui.select(
		"Overwrite the config file with the shipped defaults?",
		["Back up current config and reset", MENU_BACK],
	);
	if (!confirm || confirm === MENU_BACK) return;
	const hadJev = getConfig().jev != null;
	let result: ReturnType<typeof resetConfigFile>;
	try {
		result = resetConfigFile();
	} catch (err) {
		ctx.ui.notify(`Reset failed: ${err instanceof Error ? err.message : String(err)}`, "error");
		return;
	}
	setConfig(result.config);
	const jevChanged = hadJev !== (result.config.jev != null);
	ctx.ui.notify(
		`Config reset to shipped defaults${result.backupPath ? ` (backup: ${result.backupPath})` : " (no previous file to back up)"}` +
			(jevChanged ? " — restart pi for /models changes" : ""),
		"info",
	);
}

// ---------------------------------------------------------------------------
// Extension entry point
// ---------------------------------------------------------------------------

export default function (pi: ExtensionAPI) {
	// Virtual models are the whole point of this extension; on older pi builds the
	// API is absent and the default load error would be a bare "not a function".
	if (typeof pi.registerVirtualModel !== "function") {
		throw new Error(
			"pi-triage-router needs the virtual models API (pi.registerVirtualModel, added in pi 0.99.0). " +
				"Upgrade pi (e.g. `npm install -g @earendil-works/pi-coding-agent@latest` or `brew upgrade pi-coding-agent`) and restart.",
		);
	}
	const meta = {
		thinkingLevels: ["off", "low", "medium", "high"] as ["off", "low", "medium", "high"],
		// Shown before the first response; the physical model's limits take over.
		contextWindow: 1_048_576,
		maxTokens: 128_000,
	};
	pi.registerVirtualModel<LayaState>({
		provider: "laya",
		id: "auto",
		name: "Auto (Laya local)",
		...meta,
		async route(request, ctx) {
			return handleRequest(request, ctx, loadConfig(), "classifier");
		},
	});
	// The remote Jev entry is registered only when the config defines it:
	// presence in /models mirrors the presence of a configured remote backend.
	if (loadConfig().jev) {
		pi.registerVirtualModel<LayaState>({
			provider: "jev",
			id: "auto",
			name: "Auto (Jev, remote classifier)",
			...meta,
			async route(request, ctx) {
				return handleRequest(request, ctx, loadConfig(), "jev");
			},
		});
		// Startup validation of the jev backend (non-blocking, like the price
		// refresh): warns when credentials are missing, the model id is not on
		// the provider, or the provider cannot be reached.
		const jev = loadConfig().jev;
		if (jev) {
			void checkRemoteClassifier(jev, "jev").then((check) => {
				if (check.level !== "ok") console.warn(`${LOG_PREFIX} warning: ${check.message}`);
			});
		}
	}
	// /triage dashboard: interactive menu in the TUI (no chat output);
	// JSON/print modes have no dialogs and fall back to the Markdown report.
	// Guarded so older pi builds without registerCommand still load the router.
	if (typeof pi.registerCommand === "function") {
		pi.registerCommand("triage", {
			description:
				"pi-triage-router dashboard: model lineup, session cost, routing history, live config edits",
			handler: async (_args, ctx) => {
				if (ctx.mode === "tui" && ctx.hasUI && typeof ctx.ui?.select === "function") {
					await triageMenu(ctx);
				} else {
					pi.sendMessage({
						customType: "triage-dashboard",
						content: formatDashboard(loadConfig()),
						display: true,
						details: undefined,
					});
				}
			},
		});
	}
}

/** Shared routing pipeline for both virtual models; only the classifier
 * backend differs. Exported for tests. */
export async function handleRequest(
	request: LayaRouteRequest,
	ctx: ExtensionContext,
	config: RouterConfig,
	backend: ClassifierBackend,
): Promise<Awaited<ReturnType<typeof buildRoute>>> {
	const contextTokens = safeContextTokens(ctx);
	// Required context for the upcoming request: current conversation +
	// the new request (chars/4, documented approximation) + output reserve.
	const requiredTokens =
		contextTokens == null
			? null
			: contextTokens + Math.ceil(lastUserText(request.messages).length / 4) + OUTPUT_RESERVE_TOKENS;
	const eligible = eligibleModels(config, ctx.modelRegistry, requiredTokens);
	// Present-but-not-group-filtered set: manual override and the sticky model
	// win even when the active group excludes them (override = user command,
	// sticky = conversation continuity; both outrank the group filter).
	const present = windowEligible(presentModels(config, ctx.modelRegistry), ctx.modelRegistry, requiredTokens);

	// Manual override: validated, takes precedence, visible in diagnostics.
	if (config.override) {
		const entry = present.find((e) => composite(e) === config.override);
		if (entry) {
			const state = { model: composite(entry), summary: request.state?.summary ?? "" };
			logDecision(
				{
					entry,
					rule: "override",
					complexity: "standard",
					category: "question",
					confidence: null,
					notes: ["config override"],
					estimate: emptyEstimate(),
				},
				null,
				emptyRoutingInput(lastUserText(request.messages)),
				null,
				ctx,
			);
			return buildRoute(request, ctx, entry, state);
		}
		console.error(`${LOG_PREFIX} configured "override" ${config.override} is not eligible; routing normally`);
	}

	// Direct requests (compaction summaries, extension calls): cheapest in
	// defaultGroup, else cheapest in defaultGroup's fallback chain, else the
	// configured default model, else cheapest eligible. Deterministic - no
	// classification for a /compact summary; see decision rule "direct".
	if (request.reason === "direct") {
		const priceIn = priceResolver(ctx, config);
		// present, not eligible: the active group filters the chat loop, not
		// /compact - direct requests use defaultGroup/defaultModel per spec.
		const directPool = directModels(config, present);
		const entry = fallbackEntry(config, directPool, priceIn);
		logDecision(
			{
				entry,
				rule: "direct",
				complexity: "standard",
				category: "question",
				confidence: null,
				notes: [
					config.defaultGroup
						? `request outside agent loop (pool: defaultGroup "${config.defaultGroup}")`
						: "request outside agent loop",
				],
				estimate: emptyEstimate(),
			},
			null,
			emptyRoutingInput(),
			null,
			ctx,
		);
		return buildRoute(request, ctx, entry, { model: composite(entry), summary: request.state?.summary ?? "" });
	}

	// Continuations and retries stay on the model that handled the turn
	// (bounded: no re-escalation loops), unless it is no longer eligible.
	if (request.reason !== "user" && request.state) {
		// Sticky lookup ignores the active group (decision: sticky wins until
		// the next user turn); a new user turn re-classifies within the group.
		const sticky = present.find((e) => composite(e) === request.state!.model);
		if (sticky) {
			logDecision(
				{
					entry: sticky,
					rule: "sticky",
					complexity: "standard",
					category: "question",
					confidence: null,
					notes: [`${request.reason} keeps session model`],
					estimate: emptyEstimate(),
				},
				null,
				emptyRoutingInput(),
				null,
				ctx,
			);
			return buildRoute(request, ctx, sticky, request.state);
		}
		// State model gone (disabled/removed): fall through to fresh classification.
	}

	// New user request (or lost state): classify with context.
	const benchmarkIds = config.models.filter((m) => m.enabled !== false).map((m) => m.id);
	// Keyless public catalog: price refresh independent of scoring/benchmarks.
	if (config.prices.enabled) schedulePriceRefresh(config.prices, benchmarkIds);
	if (config.benchmarks.enabled && config.scoring.enabled) {
		// Non-blocking: refresh runs in the background, routing reads the cache.
		scheduleRefresh(config.benchmarks, benchmarkIds);
	}
	const { decision, layaMs, input, summary, lastCost, clf } = await classifyAndDecide(
		request,
		ctx,
		config,
		eligible,
		backend,
	);
	// Classifier cost/latency, logged SEPARATELY from execution-model cost.
	reportRoute(
		ctx,
		`clf=${clf.provider} model=${clf.model} ms=${clf.ms ?? "-"} ` +
			`in=${clf.inputTokens ?? "-"} out=${clf.outputTokens ?? "-"} ` +
			`cost=${clf.cost ?? "-"}(${clf.costKind ?? "no-call"})`,
		`clf ${clf.provider}${clf.cost != null ? ` · $${clf.cost}` : ""}`,
	);
	if (config.benchmarks.enabled && config.scoring.enabled) {
		// Freshness + coverage observability for the scoring inputs.
		reportRoute(ctx, datasetStatus(config.benchmarks, benchmarkIds), null);
	}
	// logDecision LAST: its status line is the selection, which must stay
	// visible after the clf/bench diagnostics.
	logDecision(decision, layaMs, input, lastCost, ctx);
	return buildRoute(
		request,
		ctx,
		decision.entry,
		{ model: composite(decision.entry), summary },
	);
}