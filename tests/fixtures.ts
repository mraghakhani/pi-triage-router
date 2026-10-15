/** Shared fixtures for triage-router tests. No network, no real models. */
import * as router from "../triage-router.ts";

export type FakeModel = {
	id: string;
	name: string;
	api: string;
	provider: string;
	baseUrl: string;
	input: string[];
	cost: { input: number; output: number; cacheRead: number; cacheWrite: number };
	contextWindow: number;
	maxTokens: number;
	reasoning: boolean;
	type?: "chat";
};

/** Mirrors the shipped config's enabled entries (all provider "litellm"). */
export const CATALOG: Record<string, FakeModel> = {
	"litellm/deepseek/deepseek-v4-flash": {
		id: "deepseek/deepseek-v4-flash",
		name: "DeepSeek V4 Flash",
		api: "openai-completions",
		provider: "litellm",
		baseUrl: "https://example.test",
		input: ["text"],
		cost: { input: 0.01, output: 1.28, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 1_000_000,
		maxTokens: 128_000,
		reasoning: true,
		type: "chat",
	},
	"litellm/z-ai/glm-5.3-flash": {
		id: "z-ai/glm-5.3-flash",
		name: "GLM 5.3 Flash",
		api: "openai-completions",
		provider: "litellm",
		baseUrl: "https://example.test",
		input: ["text"],
		cost: { input: 0.15, output: 0.5, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 250_000,
		maxTokens: 32_000,
		reasoning: true,
		type: "chat",
	},
	"litellm/z-ai/glm-5.3": {
		id: "z-ai/glm-5.3",
		name: "GLM 5.3",
		api: "openai-completions",
		provider: "litellm",
		baseUrl: "https://example.test",
		input: ["text"],
		cost: { input: 0.04, output: 4.8, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 1_000_000,
		maxTokens: 128_000,
		reasoning: true,
		type: "chat",
	},
	"litellm/deepseek/deepseek-v4-pro": {
		id: "deepseek/deepseek-v4-pro",
		name: "DeepSeek V4 Pro",
		api: "openai-completions",
		provider: "litellm",
		baseUrl: "https://example.test",
		input: ["text"],
		cost: { input: 0.3, output: 0.59, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 128_000,
		maxTokens: 32_000,
		reasoning: true,
		type: "chat",
	},
	"litellm/qwen/qwen3.8-max": {
		id: "qwen/qwen3.8-max",
		name: "Qwen3.8 Max",
		api: "openai-completions",
		provider: "litellm",
		baseUrl: "https://example.test",
		input: ["text"],
		cost: { input: 2.0, output: 6.0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 1_000_000,
		maxTokens: 128_000,
		reasoning: true,
		type: "chat",
	},
	"litellm/xiaomi/mimo-v2.5": {
		id: "xiaomi/mimo-v2.5",
		name: "MiMo v2.5",
		api: "openai-completions",
		provider: "litellm",
		baseUrl: "https://example.test",
		input: ["text"],
		cost: { input: 0.14, output: 0.28, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 128_000,
		maxTokens: 32_000,
		reasoning: true,
		type: "chat",
	},
};

export const ALL_KEYS = Object.keys(CATALOG);

/** A registry-like object that only knows `keys` (composite "provider/id"). */
export function registryFor(keys: string[]): any {
	return {
		find(provider: string, id: string) {
			const composite = `${provider}/${id}`;
			const model = CATALOG[composite];
			return keys.includes(composite) ? (model as never) : undefined;
		},
	};
}

/** The shipped config's six enabled entries, as a validated config. */
export function testConfig(overrides: Record<string, unknown> = {}): router.RouterConfig {
	return router.validateConfig({
		models: [
			{ provider: "litellm", id: "deepseek/deepseek-v4-flash", description: "Cheap default", priceIn: 0.01 },
			{ provider: "litellm", id: "z-ai/glm-5.3-flash", description: "Cheap fast tier", priceIn: 0.15 },
			{ provider: "litellm", id: "z-ai/glm-5.3", description: "Reasoning multi-file", priceIn: 0.04 },
			{ provider: "litellm", id: "deepseek/deepseek-v4-pro", description: "Balanced strong", priceIn: 0.3 },
			{ provider: "litellm", id: "qwen/qwen3.8-max", description: "Top coder", priceIn: 2.0 },
			{ provider: "litellm", id: "xiaomi/mimo-v2.5", description: "Long-horizon cheap", priceIn: 0.14 },
		],
		fallback: "cheapest",
		// Hermetic tests: the default-enabled price refresh must never run here.
		prices: { enabled: false },
		...overrides,
	});
}

export function userMsg(text: string): never {
	return { role: "user", content: text } as never;
}
export function asstMsg(text: string): never {
	return { role: "assistant", content: [{ type: "text", text }] } as never;
}

export function makeCtx(
	usage: { tokens: number; contextWindow: number } | null = null,
	entries: unknown[] = [],
): any {
	return {
		modelRegistry: registryFor(ALL_KEYS),
		getContextUsage: () => (usage ? { ...usage, percent: 1 } : undefined),
		sessionManager: { getEntries: () => entries },
	} as never;
}

/** A request carrying route-facing state. */
export function routeRequest(messages: unknown[], state: router.LayaState | undefined, reason = "user"): any {
	return {
		model: { id: "auto", name: "Auto", api: "pi-virtual", provider: "laya", baseUrl: "", input: [], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } },
		thinkingLevel: "medium",
		reason,
		state,
		messages,
	} as never;
}

/** A classifier mock returning fixed Laya-style answers. */
export function mockClassifier(answers: {
	model: Record<string, number>;
	complexity: Record<string, number>;
	category: Record<string, number>;
}): router.Classifier {
	return {
		systemOne: async () => ({
			answers: {
				model: { probabilities: answers.model },
				complexity: { probabilities: answers.complexity },
				category: { probabilities: answers.category },
			},
		}),
	};
}