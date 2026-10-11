/**
 * Strategy comparison over the shared evaluation corpus (tests/eval-corpus.ts).
 *
 * Strategies (same corpus, same acceptable-tier ground truth):
 *  A. local Laya routing (classifier picks the model)      - opt-in LAYA_INTEGRATION=1
 *  B. remote Jev classifier (Decisions API) picks          - opt-in REMOTE_INTEGRATION=1 (+ key)
 *  C. benchmark-and-price scored selection (scoring on, real OpenRouter
 *     snapshot in tests/fixtures/benchmarks.json, controlled classifier
 *     answers for category/complexity)                     - always runs (deterministic)
 *  D. fixed budget baseline (always cheapest eligible)     - always runs
 *  E. fixed strong baseline (always strongest eligible)    - always runs
 *
 * Standard runs are deterministic and free: no network, no paid API call.
 * A/B are measurements (opt-in), clearly labeled; C uses REAL benchmark data
 * with SYNTHETIC classifier answers; D/E need no classifier at all.
 *
 * Measured per strategy: category accuracy, complexity accuracy, tier
 * accuracy vs acceptable tiers, eligible violations, escalation/fallback
 * frequency, classification latency, classifier API cost, execution-model
 * input price (estimate, USD/1M).
 */
import { test, afterEach } from "node:test";
import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import * as router from "../triage-router.ts";
import { resetDatasetReadCacheForTests } from "../triage-router-lib/triage-router-benchmarks.ts";
import { setFetchForTests } from "../triage-router-lib/triage-router-openrouter.ts";
import { testConfig, makeCtx, routeRequest, mockClassifier } from "./fixtures.ts";
import { CORPUS, caseMessages, caseMock, TIER_MODELS, tierOf, priceOf, type EvalCase } from "./eval-corpus.ts";

const remoteEnabled = process.env.REMOTE_INTEGRATION === "1";
const localEnabled = process.env.LAYA_INTEGRATION === "1";
const FIXTURE_CACHE = fileURLToPath(new URL("./fixtures/benchmarks.json", import.meta.url));

afterEach(() => {
	router.resetLayaForTests();
	setFetchForTests(null);
	resetDatasetReadCacheForTests();
});

interface StrategyResult {
	strategy: string;
	categoryHits: number;
	complexityHits: number;
	violations: number;
	eligibleMisses: number;
	escalations: number;
	fallbacks: number;
	scores: number;
	clsMsTotal: number;
	clsCount: number;
	clsCostTotal: number;
	execPriceSum: number;
	runs: number;
}

function newResult(strategy: string): StrategyResult {
	return {
		strategy, categoryHits: 0, complexityHits: 0, violations: 0, eligibleMisses: 0,
		escalations: 0, fallbacks: 0, scores: 0, clsMsTotal: 0, clsCount: 0,
		clsCostTotal: 0, execPriceSum: 0, runs: 0,
	};
}

function record(
	r: StrategyResult,
	c: EvalCase,
	decision: router.RoutingDecision,
	eligible: router.ModelEntry[],
	clf: router.ClassifierCallStats | null,
): void {
	const composite = `${decision.entry.provider}/${decision.entry.id}`;
	if (clf?.ms != null) {
		r.clsMsTotal += clf.ms;
		r.clsCount++;
		if (clf.provider === "openrouter" && clf.cost != null) r.clsCostTotal += clf.cost;
	}
	if (!eligible.some((e) => e === decision.entry)) r.eligibleMisses++;
	if (!c.acceptableTiers.includes(tierOf(composite))) r.violations++;
	if (decision.rule === "escalated") r.escalations++;
	if (decision.rule === "fallback") r.fallbacks++;
	if (decision.rule === "scored") r.scores++;
	if (decision.category === c.expectedCategory) r.categoryHits++;
	if (decision.complexity === c.expectedComplexity) r.complexityHits++;
	r.execPriceSum += priceOf(composite);
	r.runs++;
}

function pct(n: number, d: number): string {
	return d ? `${((n / d) * 100).toFixed(1)}%` : "-";
}

function report(r: StrategyResult, extra = "", noClassifier = false): void {
	console.log(
		`[laya-strategy] ${r.strategy} n=${r.runs} ` +
			(noClassifier
				? "category=n/a complexity=n/a "
				: `category=${pct(r.categoryHits, r.runs)} complexity=${pct(r.complexityHits, r.runs)} `) +
			`tier_acc=${pct(r.runs - r.violations, r.runs)} ` +
			`violations=${r.violations} eligible_misses=${r.eligibleMisses} escalations=${r.escalations} ` +
			`fallbacks=${r.fallbacks} scored=${r.scores} ` +
			`clf_ms_mean=${r.clsCount ? Math.round(r.clsMsTotal / r.clsCount) : 0} ` +
			`clf_cost_total_usd=${r.clsCostTotal.toFixed(6)} ` +
			`exec_input_price_mean=${(r.execPriceSum / Math.max(1, r.runs)).toFixed(3)} (estimate, USD/1M in)` +
			(extra ? ` ${extra}` : ""),
	);
}

function scoredConfig(overrides: Record<string, unknown> = {}) {
	return testConfig({
		benchmarks: { enabled: true, cachePath: FIXTURE_CACHE },
		scoring: { enabled: true },
		...overrides,
	});
}

// ---------------------------------------------------------------------------
// C: benchmark-and-price scored selection (deterministic, always runs)
// ---------------------------------------------------------------------------

test("strategy C: benchmark-driven scoring routes the corpus deterministically", async () => {
	const config = scoredConfig();
	const ctx = makeCtx({ tokens: 30_000, contextWindow: 1_000_000 });
	const eligible = router.eligibleModels(config, ctx.modelRegistry);
	const result = newResult("C benchmark-scored (real benchmarks, synthetic classification)");
	for (const c of CORPUS) {
		// Controlled classifier answers: category/complexity are given; only the
		// model pick is decided by the scoring policy over real benchmark data.
		router.setLayaForTests(mockClassifier(caseMock(c)));
		const { decision, clf } = await router.classifyAndDecide(
			routeRequest(caseMessages(c), undefined) as never,
			ctx as never,
			config,
			eligible,
		);
		record(result, c, decision, eligible, clf);
		assert.ok(eligible.some((e) => e === decision.entry), `${c.id}: decision must be eligible`);
	}
	report(result);
	// Regression bounds: measured 2026-10-09 with the checked-in snapshot; the
	// scored path must never be less eligible-safe than the baselines below.
	assert.equal(result.eligibleMisses, 0, "scoring never escapes the eligible set");
	assert.equal(result.runs, CORPUS.length);
	assert.ok(result.scores >= 1, "at least some cases were decided by scoring, not fallback");
	// C must beat the always-cheapest baseline on violations (measured: D violates
	// on every mid/strong-acceptable case; C keeps quality for demanding work).
	const d = newResult("D");
	for (const c of CORPUS) {
		const entry = router.cheapestEligible(eligible);
		d.violations += c.acceptableTiers.includes(tierOf(`${entry.provider}/${entry.id}`)) ? 0 : 1;
		d.execPriceSum += priceOf(`${entry.provider}/${entry.id}`);
		d.runs++;
	}
	assert.ok(
		result.violations <= d.violations,
		`C violations (${result.violations}) must not exceed always-cheapest (${d.violations})`,
	);
	// C must not cost more than the always-strong baseline (mean input price).
	const eMean = priceOf(TIER_MODELS.strong);
	assert.ok(
		result.execPriceSum / result.runs < eMean,
		"scored mean input price must stay below the strong baseline",
	);
});

// ---------------------------------------------------------------------------
// D / E: fixed baselines (deterministic, always runs)
// ---------------------------------------------------------------------------

test("strategy D: fixed budget baseline (always cheapest)", async () => {
	const config = testConfig();
	const ctx = makeCtx();
	const eligible = router.eligibleModels(config, ctx.modelRegistry);
	const result = newResult("D always-cheapest");
	for (const c of CORPUS) {
		const entry = router.cheapestEligible(eligible);
		const composite = `${entry.provider}/${entry.id}`;
		if (!c.acceptableTiers.includes(tierOf(composite))) result.violations++;
		result.execPriceSum += priceOf(composite);
		result.runs++;
	}
	report(result, "(no classifier: fixed pick)", true);
	assert.equal(result.runs, CORPUS.length);
	assert.ok(result.violations > 0, "the corpus deliberately contains work the cheap tier cannot own");
});

test("strategy E: fixed strong baseline (always strongest)", async () => {
	const config = testConfig();
	const ctx = makeCtx();
	const eligible = router.eligibleModels(config, ctx.modelRegistry);
	const result = newResult("E always-strongest");
	for (const c of CORPUS) {
		const entry = router.strongestEligible(eligible);
		const composite = `${entry.provider}/${entry.id}`;
		if (!c.acceptableTiers.includes(tierOf(composite))) result.violations++;
		result.execPriceSum += priceOf(composite);
		result.runs++;
	}
	report(result, "(no classifier: fixed pick)", true);
	assert.equal(result.runs, CORPUS.length);
	assert.ok(result.violations > 0, "the strong tier over-routes cheap-only work: the baseline's cost of simplicity");
	assert.ok(result.execPriceSum / result.runs > priceOf(TIER_MODELS.cheap), "strong baseline costs more than cheap");
});

// ---------------------------------------------------------------------------
// A: local Laya (opt-in measurement)
// ---------------------------------------------------------------------------

test("strategy A: local Laya classification over the corpus", { skip: !localEnabled }, async () => {
	const config = testConfig(); // classifier pick path (scoring off): the audited baseline
	const ctx = makeCtx({ tokens: 30_000, contextWindow: 1_000_000 });
	const eligible = router.eligibleModels(config, ctx.modelRegistry);
	const result = newResult("A local-Laya (real classifier, real measurement)");
	for (const c of CORPUS) {
		const { decision, clf } = await router.classifyAndDecide(
			routeRequest(caseMessages(c), undefined) as never,
			ctx as never,
			config,
			eligible,
		);
		record(result, c, decision, eligible, clf);
	}
	report(result, "(measurement, not a guarantee)");
	assert.equal(result.eligibleMisses, 0);
	assert.equal(result.clsCostTotal, 0, "local classification has no API charge");
});

// ---------------------------------------------------------------------------
// B: remote Jev classifier (opt-in measurement, paid)
// ---------------------------------------------------------------------------

test("strategy B: remote Jev classifier over the corpus", { skip: !remoteEnabled }, async () => {
	const config = testConfig({
		classifier: { provider: "openrouter", proxyUrl: null, timeoutMs: 10_000 },
	});
	const ctx = makeCtx({ tokens: 30_000, contextWindow: 1_000_000 });
	const eligible = router.eligibleModels(config, ctx.modelRegistry);
	const result = newResult("B remote-Jev (real Decisions API, real measurement)");
	for (const c of CORPUS) {
		const { decision, clf } = await router.classifyAndDecide(
			routeRequest(caseMessages(c), undefined) as never,
			ctx as never,
			config,
			eligible,
		);
		record(result, c, decision, eligible, clf);
		assert.equal(clf.provider, "openrouter", "backend identity");
		if (clf.costKind !== null) {
			assert.ok(clf.costKind === "actual" || clf.costKind === "unavailable", "cost is labeled, never invented");
		}
	}
	report(result, "(measurement, not a guarantee; classifier cost is provider-reported actual)");
	assert.equal(result.eligibleMisses, 0);
	assert.equal(result.clsCount, CORPUS.length, "every case produced a measured classifier call");
	assert.ok(result.clsCostTotal > 0, "remote classifier cost is captured as reported by the provider");
});
