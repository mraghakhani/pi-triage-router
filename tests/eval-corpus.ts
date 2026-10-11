/**
 * Routing-quality evaluation corpus for triage-router. 25 representative
 * engineering requests: coding, Kubernetes, networking, databases, debugging,
 * reasoning, research, simple questions, long-context, follow-ups, and task
 * changes. Each case records the expected category, complexity, and the
 * acceptable model tiers, so regressions in the deterministic routing rules
 * are caught without any paid API call.
 *
 * Tiers map to the fixtures catalog by input price:
 *   cheap  = deepseek/deepseek-v4-flash (0.01)
 *   mid    = z-ai/glm-5.3, xiaomi/mimo-v2.5, z-ai/glm-5.3-flash, deepseek/deepseek-v4-pro
 *   strong = qwen/qwen3.8-max (2.00)
 */
import type * as router from "../triage-router.ts";
import { testConfig } from "./fixtures.ts";

export type Tier = "cheap" | "mid" | "strong";

export interface EvalCase {
	id: string;
	/** Current request text. */
	request: string;
	/** Prior conversation turns (oldest first), for follow-up cases. */
	history?: { role: "user" | "assistant"; text: string }[];
	/** What Laya should classify the current request as. */
	expectedCategory: router.Category;
	expectedComplexity: router.Complexity;
	/** Tiers whose models are acceptable picks for this case. */
	acceptableTiers: Tier[];
	/** The tier the deterministic mock nominates for this case. */
	mockTier: Tier;
	/** Text the classifier context must retain for this case (mustSee). */
	mustSee?: string[];
}

/** Deterministic tier -> model mapping (first model of each tier's price rank). */
export const TIER_MODELS: Record<Tier, string> = {
	cheap: "litellm/deepseek/deepseek-v4-flash",
	mid: "litellm/z-ai/glm-5.3",
	strong: "litellm/qwen/qwen3.8-max",
};

const longLog = Array.from(
	{ length: 90 },
	(_, i) => `2026-10-09T1${i}:00:03Z svc[${i % 7}] step=deploy status=${i % 13 === 0 ? "FAIL" : "OK"} latency=${i}ms`,
).join("\n");

export const CORPUS: EvalCase[] = [
	{ id: "fix-typo", request: "Fix the typo in the README install command.", expectedCategory: "coding", expectedComplexity: "standard", acceptableTiers: ["cheap", "mid"], mockTier: "cheap" },
	{ id: "unit-tests", request: "Add unit tests for the retry helper in payments.js.", expectedCategory: "coding", expectedComplexity: "moderate", acceptableTiers: ["cheap", "mid"], mockTier: "mid" },
	{ id: "refactor-billing", request: "Refactor the billing service into a repository and handler split across 6 files.", expectedCategory: "coding", expectedComplexity: "complex", acceptableTiers: ["mid", "strong"], mockTier: "mid" },
	{ id: "k8s-crashloop", request: "Our Kubernetes pods crash-loop; the readiness probe fails on port 8080.", expectedCategory: "debugging", expectedComplexity: "complex", acceptableTiers: ["mid", "strong"], mockTier: "mid" },
	{ id: "k8s-hpa", request: "Tune the HPA so scale-ups stop lagging behind queue depth.", expectedCategory: "infra", expectedComplexity: "moderate", acceptableTiers: ["cheap", "mid"], mockTier: "cheap" },
	{ id: "k8s-netpol", request: "Write a NetworkPolicy that lets only the gateway talk to the payments API.", expectedCategory: "infra", expectedComplexity: "moderate", acceptableTiers: ["cheap", "mid"], mockTier: "mid" },
	{ id: "mesh-dns", request: "Dig into why api.internal returns NXDOMAIN from the service mesh but works from pods.", expectedCategory: "debugging", expectedComplexity: "complex", acceptableTiers: ["mid", "strong"], mockTier: "mid" },
	{ id: "gateway-504", request: "The Envoy gateway returns 504 on uploads over 10MB; fix the timeout chain.", expectedCategory: "debugging", expectedComplexity: "complex", acceptableTiers: ["mid", "strong"], mockTier: "strong" },
	{ id: "pg-connections", request: "Postgres hits 'too many connections' every night; diagnose and fix it.", expectedCategory: "debugging", expectedComplexity: "moderate", acceptableTiers: ["cheap", "mid"], mockTier: "mid" },
	{ id: "pg-index", request: "Add an index so the orders report query stops seq-scanning.", expectedCategory: "infra", expectedComplexity: "moderate", acceptableTiers: ["cheap", "mid"], mockTier: "cheap" },
	{ id: "adr-auth-boundary", request: "Design the auth boundary between our monolith and the new billing service; write the ADR.", expectedCategory: "reasoning", expectedComplexity: "complex", acceptableTiers: ["mid", "strong"], mockTier: "strong" },
	{ id: "capacity-reasoning", request: "Reason through whether our current Postgres primary survives 3x write traffic.", expectedCategory: "reasoning", expectedComplexity: "complex", acceptableTiers: ["mid", "strong"], mockTier: "strong" },
	{ id: "research-structured", request: "Research which of our models support structured outputs and summarize the tradeoffs.", expectedCategory: "research", expectedComplexity: "moderate", acceptableTiers: ["cheap", "mid"], mockTier: "cheap" },
	{ id: "research-ratelimit", request: "Compare the top three rate-limiting libraries for our stack.", expectedCategory: "research", expectedComplexity: "standard", acceptableTiers: ["cheap", "mid"], mockTier: "cheap" },
	{ id: "question-locking", request: "What is the difference between optimistic and pessimistic locking?", expectedCategory: "question", expectedComplexity: "standard", acceptableTiers: ["cheap"], mockTier: "cheap" },
	{ id: "question-build-slow", request: "Why does our build get slower after the Docker cache prune?", expectedCategory: "question", expectedComplexity: "moderate", acceptableTiers: ["cheap", "mid"], mockTier: "cheap" },
	{ id: "long-horizon-migration", request: "Migrate all 40 services off the shared library onto the new one, start to finish.", expectedCategory: "long_horizon", expectedComplexity: "complex", acceptableTiers: ["mid", "strong"], mockTier: "strong" },
	{ id: "budget-docs", request: "Cheapest option that works: refresh the docs table.", expectedCategory: "budget", expectedComplexity: "standard", acceptableTiers: ["cheap"], mockTier: "cheap" },
	{
		id: "long-context-logs",
		request: `Here is the deploy log:\n${longLog}\n\nFind the failing step and fix it.`,
		expectedCategory: "debugging",
		expectedComplexity: "complex",
		acceptableTiers: ["mid", "strong"],
		mockTier: "mid",
	},
	{
		id: "followup-same-fix",
		request: "Apply the same fix to the refunds service.",
		history: [
			{ role: "user", text: "We're fixing the PostgreSQL connection pool exhaustion in the orders service." },
			{ role: "assistant", text: "Found it: the pool leak is in checkout.js where the client is never released." },
		],
		expectedCategory: "coding",
		expectedComplexity: "moderate",
		acceptableTiers: ["mid"],
		mockTier: "mid",
		mustSee: ["PostgreSQL", "pool leak"],
	},
	{
		id: "followup-phase-2",
		request: "Start with phase 2 only.",
		history: [
			{ role: "user", text: "Refactor the auth middleware into three phases." },
			{ role: "assistant", text: "Planned: phase 1 extract helpers, phase 2 split token parsing, phase 3 drop legacy." },
		],
		expectedCategory: "coding",
		expectedComplexity: "moderate",
		acceptableTiers: ["mid"],
		mockTier: "mid",
		mustSee: ["Refactor the auth middleware", "phase 2"],
	},
	{
		id: "task-change-terraform",
		request: "New topic: write the Terraform for the S3 bucket replication.",
		history: [
			{ role: "user", text: "Debug the flaky payments e2e test." },
			{ role: "assistant", text: "Fixed: it was clock skew in the mock server." },
		],
		expectedCategory: "infra",
		expectedComplexity: "moderate",
		acceptableTiers: ["cheap", "mid"],
		mockTier: "cheap",
	},
	{
		id: "task-change-haiku",
		request: "Completely unrelated: write a haiku about UDP packets.",
		history: [
			{ role: "user", text: "Long discussion about the PostgreSQL connection pool sizes." },
			{ role: "assistant", text: "Pool size 100 seems wrong for our workload." },
		],
		expectedCategory: "question",
		expectedComplexity: "standard",
		acceptableTiers: ["cheap"],
		mockTier: "cheap",
	},
	{ id: "distributed-retry", request: "Rework the retry, circuit-breaker, and idempotency interaction across the payments gateway; the failure modes are subtle.", expectedCategory: "debugging", expectedComplexity: "complex", acceptableTiers: ["strong"], mockTier: "strong" },
	{ id: "intermittent-503", request: "Root-cause the intermittent 503s across the gateway; it is a hard distributed bug.", expectedCategory: "debugging", expectedComplexity: "complex", acceptableTiers: ["mid", "strong"], mockTier: "cheap" },
	{ id: "oracle-locks", request: "Oracle deadlocks started after the 19c upgrade in the order pipeline; diagnose the lock chain and fix the tx order.", expectedCategory: "debugging", expectedComplexity: "moderate", acceptableTiers: ["cheap", "mid"], mockTier: "mid" },
	{ id: "rke2-etcd-recovery", request: "An RKE2 etcd member is unhealthy after a disk-full event; recover the cluster without data loss.", expectedCategory: "infra", expectedComplexity: "complex", acceptableTiers: ["mid", "strong"], mockTier: "mid" },
	{ id: "tool-heavy-migration", request: "Run the billing migration end to end: inspect the live schema, generate the SQL, apply it to staging, verify row counts, and report every diff.", expectedCategory: "long_horizon", expectedComplexity: "moderate", acceptableTiers: ["cheap", "mid"], mockTier: "mid" },
	{ id: "ambiguous-slow-api", request: "The payments API feels slow lately.", expectedCategory: "question", expectedComplexity: "standard", acceptableTiers: ["cheap"], mockTier: "cheap" },
	{ id: "deploy-rollback-edge", request: "Roll back yesterday's nginx deploy on the edge nodes and find out why the canary went 502.", expectedCategory: "infra", expectedComplexity: "moderate", acceptableTiers: ["cheap", "mid"], mockTier: "cheap" },
	{ id: "expensive-unnecessary", request: "Which HTTP status code fits a 409 conflict?", expectedCategory: "question", expectedComplexity: "standard", acceptableTiers: ["cheap"], mockTier: "cheap" },
];

/** Build corpus case messages (history + current request) as fixture messages. */
export function caseMessages(c: EvalCase): unknown[] {
	const msgs = (c.history ?? []).map((h) => (h.role === "user" ? { role: "user", content: h.text } : { role: "assistant", content: [{ type: "text", text: h.text }] }));
	msgs.push({ role: "user", content: c.request });
	return msgs;
}

/** A deterministic mock classifier answer matching a case's expectations. */
export function caseMock(c: EvalCase): { model: Record<string, number>; complexity: Record<string, number>; category: Record<string, number> } {
	return {
		model: { [TIER_MODELS[c.mockTier]]: 0.9 },
		complexity: { [c.expectedComplexity]: 0.9 },
		category: { [c.expectedCategory]: 0.9 },
	};
}

/** Input price of a catalog model (USD per 1M tokens), from the fixtures catalog. */
export function priceOf(composite: string): number {
	const config = testConfig();
	const [provider, ...rest] = composite.split("/");
	const id = rest.join("/");
	const entry = config.models.find((m) => m.provider === provider && m.id === id);
	if (!entry || entry.priceIn === undefined) throw new Error(`corpus: no price for ${composite}`);
	return entry.priceIn;
}

/** Which tier a catalog model belongs to, by price rank. */
export function tierOf(composite: string): Tier {
	const price = priceOf(composite);
	if (price <= 0.02) return "cheap";
	if (price >= 1) return "strong";
	return "mid";
}