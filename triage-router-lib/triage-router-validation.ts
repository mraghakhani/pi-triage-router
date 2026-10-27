/**
 * Validation of remote classifier sections (the `jev` backend; the menu scope
 * per user choice). Two checks, in order:
 *
 *  1. Credentials: env var (apiKeyEnv) or Pi auth file entry for the provider.
 *  2. Model id serves on the provider. The provider's model list
 *     (OpenRouter public catalog GET /api/v1/models, LiteLLM proxy GET
 *     {baseUrl}/v1/models) is checked first because it is cheap; when the id
 *     is absent from it, the actual scoring endpoint is probed with a minimal
 *     valid request - the list is NOT authoritative: LiteLLM proxies filter it
 *     per team (a real model the key cannot access looks like a typo), and
 *     OpenRouter's public catalog omits decision models (e.g. typesafe/jev-*)
 *     entirely even though they serve.
 *
 * Results:
 *  - "problem" = definite failure (token missing, model rejected by the
 *    endpoint): the /triage menu denies the edit; startup prints a warning.
 *  - "warn" = provider unreachable (network, proxy down, no baseUrl, timeout):
 *    the menu saves the edit with a warning; startup prints the connection
 *    problem. Offline laptops stay usable.
 *  - "ok" = provider and model verified (in the list, or the endpoint answered
 *    the probe).
 *
 * Model lists and probe verdicts are cached for LIST_TTL_MS so a menu session
 * (provider switch + model edit + startup) does not refetch per keystroke.
 */

import { httpGetJson } from "./triage-router-benchmarks.ts";
import {
	callDecisions,
	classifierEndpoint,
	hasApiKey,
	resolveApiKey,
	resolveBaseUrl,
	resolveProxyUrl,
	type FetchLike,
} from "./triage-router-openrouter.ts";

/** The subset of ClassifierConfig that validation needs. */
export interface RemoteSectionLike {
	provider: string;
	model: string;
	apiKeyEnv: string;
	authFile?: string | null;
	proxyUrl?: string | null;
	/** Wire protocol of the scoring endpoint; absent = systemone (the routing
	 * default for a missing protocol). */
	protocol?: "decisions" | "systemone";
}

export type RemoteCheckLevel = "ok" | "warn" | "problem";

export interface RemoteCheckResult {
	level: RemoteCheckLevel;
	message: string;
}

export interface RemoteCheckDeps {
	fetchImpl?: FetchLike;
	/** Test seam for the model-list cache. */
	now?: () => number;
}

const OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models";
const LIST_TTL_MS = 10 * 60_000;
const CHECK_TIMEOUT_MS = 5_000;

let listCache = new Map<string, { at: number; ids: Set<string> }>();

/** One endpoint-probe outcome for a (provider, model) pair. */
interface ProbeVerdict {
	kind: "verified" | "not-found" | "restricted" | "unreachable";
	/** Error text for restricted/unreachable verdicts (credential-free). */
	detail?: string;
}
let probeCache = new Map<string, { at: number; verdict: ProbeVerdict }>();

export function resetValidationCacheForTests(): void {
	listCache = new Map();
	probeCache = new Map();
}

/** Proxy for validation traffic (list GET and probe POST): explicit config
 * value first, then Pi's settings/env fallbacks - the same resolution routing
 * uses, so validation exercises the transport routing will actually take. */
export function remoteProxyUrl(section: RemoteSectionLike): string | null {
	return resolveProxyUrl(section.proxyUrl ?? null);
}

/** Model-list endpoint for a provider; null = cannot construct one. */
export function modelsUrlFor(provider: string, baseUrl?: string | null): string | null {
	if (provider === "openrouter") return OPENROUTER_MODELS_URL;
	if (provider !== "litellm") return null;
	const base = baseUrl?.trim().replace(/\/+$/u, "");
	if (!base) return null;
	return /\/v1$/u.test(base) ? `${base}/models` : `${base}/v1/models`;
}

/** Fetch the provider's model ids (cached). Never throws; failure is a message. */
async function fetchModelIds(provider: string, section: RemoteSectionLike, deps: RemoteCheckDeps): Promise<{ ok: true; ids: Set<string> } | { ok: false; error: string }> {
	const now = deps.now ?? Date.now;
	const cached = listCache.get(provider);
	if (cached && now() - cached.at < LIST_TTL_MS) return { ok: true, ids: cached.ids };
	const url = modelsUrlFor(provider, resolveBaseUrl(provider, section.authFile));
	if (!url) {
		return { ok: false, error: `no baseUrl for ${provider} in the Pi auth file - add "baseUrl" to the "${provider}" entry` };
	}
	const apiKey = resolveApiKey(section.apiKeyEnv, provider, section.authFile) ?? "";
	try {
		const payload = (await httpGetJson(url, {
			timeoutMs: CHECK_TIMEOUT_MS,
			proxyUrl: remoteProxyUrl(section),
			apiKey: apiKey || undefined,
			fetchImpl: deps.fetchImpl,
			maxAttempts: 1,
		})) as { data?: unknown };
		if (!Array.isArray(payload?.data)) return { ok: false, error: "model list response missing `data` array" };
		const ids = new Set<string>();
		for (const row of payload.data as Array<{ id?: unknown }>) {
			if (typeof row?.id === "string" && row.id !== "") ids.add(row.id);
		}
		listCache.set(provider, { at: now(), ids });
		return { ok: true, ids };
	} catch (err) {
		return { ok: false, error: err instanceof Error ? err.message : String(err) };
	}
}

/** Minimal valid scoring request for the endpoint probe: the same body shape
 * routing sends, with one criterion per question. Rejections of THIS payload
 * are attributable to the model id, not the request shape. */
function probeQuestions(): Record<string, unknown> {
	const question = (instructions: string) => ({
		type: "choice",
		instructions,
		criteria: { any: instructions },
	});
	return {
		model: question("Which model should handle the request?"),
		complexity: question("How demanding is the work?"),
		category: question("What kind of work is it?"),
	};
}

/** Classify a probe failure. Pattern source: real provider errors - LiteLLM
 * proxies reject out-of-team ids with "team not allowed ..." (also for ids the
 * proxy does not know); OpenRouter returns "Model X does not exist" for typos
 * and 401/403 for account problems. */
function classifyProbeFailure(message: string): ProbeVerdict {
	if (/does not exist|not found|unknown model|no such model/i.test(message)) {
		return { kind: "not-found", detail: message };
	}
	if (/not allowed|permission|forbidden|denied|unauthorized|credential|authentication|insufficient credits/i.test(message)) {
		return { kind: "restricted", detail: message };
	}
	return { kind: "unreachable", detail: message };
}

/** Probe the scoring endpoint with a minimal valid request; cached per
 * (provider, model, endpoint). A real (tiny) inference call - only paid when
 * the model list does not know the id. */
async function probeRemoteModel(section: RemoteSectionLike, deps: RemoteCheckDeps): Promise<ProbeVerdict> {
	const now = deps.now ?? Date.now;
	const apiKey = resolveApiKey(section.apiKeyEnv, section.provider, section.authFile);
	if (!apiKey) return { kind: "unreachable", detail: "no credentials" };
	const endpoint = classifierEndpoint(section.protocol ?? "systemone", resolveBaseUrl(section.provider, section.authFile));
	const cacheKey = `${section.provider}|${section.model}|${endpoint}`;
	const proxyUrl = remoteProxyUrl(section);
	const cached = probeCache.get(cacheKey);
	if (cached && now() - cached.at < LIST_TTL_MS) return cached.verdict;
	let verdict: ProbeVerdict;
	try {
		await callDecisions({
			model: section.model,
			role: "user",
			prompt: "validation probe",
			questions: probeQuestions(),
			timeoutMs: CHECK_TIMEOUT_MS,
			maxAttempts: 1,
			retryDelayMs: 0,
			apiKey,
			proxyUrl,
			endpoint,
			providerLabel: section.provider,
			fetchImpl: deps.fetchImpl,
		});
		verdict = { kind: "verified" };
	} catch (err) {
		verdict = classifyProbeFailure(err instanceof Error ? err.message : String(err));
	}
	probeCache.set(cacheKey, { at: now(), verdict });
	return verdict;
}

/**
 * Validate a remote classifier section (provider + model). Credentials are
 * checked first; a missing token is a problem even though the OpenRouter
 * catalog is public, because routing would fail the same way.
 */
export async function checkRemoteClassifier(
	section: RemoteSectionLike,
	label: string,
	deps: RemoteCheckDeps = {},
): Promise<RemoteCheckResult> {
	if (!hasApiKey(section.apiKeyEnv, section.provider, section.authFile)) {
		return {
			level: "problem",
			message: `${label}: no ${section.provider} credentials found - set the ${section.apiKeyEnv} environment variable or add a "${section.provider}" entry to the Pi auth file`,
		};
	}
	const list = await fetchModelIds(section.provider, section, deps);
	if (!list.ok) {
		return {
			level: "warn",
			message: `could not connect to ${section.provider} (${list.error}) - ${label} model "${section.model}" left unverified`,
		};
	}
	if (!list.ids.has(section.model)) {
		// The list is not authoritative: LiteLLM proxies team-filter it (a real
		// model the key cannot access looks like a typo) and OpenRouter's public
		// catalog omits decision models entirely. Probe the endpoint that
		// routing would actually use before reporting a problem.
		const probe = await probeRemoteModel(section, deps);
		if (probe.kind === "verified") {
			return { level: "ok", message: `${label}: ${section.provider}/${section.model} verified` };
		}
		if (probe.kind === "not-found") {
			return {
				level: "problem",
				message: `${label}: model "${section.model}" not found on ${section.provider} - check the model id`,
			};
		}
		if (probe.kind === "restricted") {
			return {
				level: "problem",
				message: `${label}: ${section.provider} rejected model "${section.model}" for your credentials - routing would fail (${(probe.detail ?? "").slice(0, 140)})`,
			};
		}
		return {
			level: "warn",
			message: `could not reach the ${section.provider} scoring endpoint to verify ${label} model "${section.model}" (${(probe.detail ?? "").slice(0, 140)}) - left unverified`,
		};
	}
	return { level: "ok", message: `${label}: ${section.provider}/${section.model} verified` };
}
