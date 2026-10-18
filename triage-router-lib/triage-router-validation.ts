/**
 * Validation of remote classifier sections (the `jev` backend; the menu scope
 * per user choice). Two checks, in order:
 *
 *  1. Credentials: env var (apiKeyEnv) or Pi auth file entry for the provider.
 *  2. Model id exists on the provider: OpenRouter public catalog
 *     (GET https://openrouter.ai/api/v1/models) or the LiteLLM proxy's
 *     OpenAI-compatible model list (GET {baseUrl}/v1/models).
 *
 * Results:
 *  - "problem" = definite failure (token missing, model confirmed absent):
 *    the /triage menu denies the edit; startup prints a warning.
 *  - "warn" = provider unreachable (network, proxy down, no baseUrl): the
 *    menu saves the edit with a warning; startup prints the connection
 *    problem. Offline laptops stay usable.
 *  - "ok" = provider and model verified.
 *
 * Model lists are cached per provider for LIST_TTL_MS so a menu session
 * (provider switch + model edit + startup) does not refetch per keystroke.
 */

import { httpGetJson } from "./triage-router-benchmarks.ts";
import { hasApiKey, resolveApiKey, resolveBaseUrl, type FetchLike } from "./triage-router-openrouter.ts";

/** The subset of ClassifierConfig that validation needs. */
export interface RemoteSectionLike {
	provider: string;
	model: string;
	apiKeyEnv: string;
	authFile?: string | null;
	proxyUrl?: string | null;
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

export function resetValidationCacheForTests(): void {
	listCache = new Map();
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
			proxyUrl: section.proxyUrl ?? null,
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
		return {
			level: "problem",
			message: `${label}: model "${section.model}" not found on ${section.provider} - check the model id`,
		};
	}
	return { level: "ok", message: `${label}: ${section.provider}/${section.model} verified` };
}
