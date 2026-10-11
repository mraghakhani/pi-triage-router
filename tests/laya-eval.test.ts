/**
 * Routing-quality evaluation over the corpus (tests/eval-corpus.ts).
 *
 * Deterministic mode (default): every case's classifier answer is mocked to
 * the case's expected category/complexity/nominated tier, then the routing
 * rules are evaluated: category preservation, eligible-set membership, tier
 * violations, escalation policy, fallback policy, and cost baselines
 * (always-cheapest vs always-strongest). No Laya load, no network, no paid
 * API calls, fully reproducible.
 *
 * Semantic mode (LAYA_INTEGRATION=1): the REAL local Laya classifies every
 * case (with history) and real metrics are printed (category accuracy,
 * violations, escalation and fallback frequency, latency, cost baselines).
 * Only structural assertions run; accuracy numbers are measurements, not
 * guarantees - no task-success or cost-improvement claims are made here.
 */
import { test, afterEach } from "node:test";
import assert from "node:assert/strict";
import * as router from "../triage-router.ts";
import { testConfig, makeCtx, routeRequest, mockClassifier, registryFor, ALL_KEYS } from "./fixtures.ts";
import { CORPUS, caseMessages, caseMock, TIER_MODELS, tierOf, priceOf } from "./eval-corpus.ts";

const enabled = process.env.LAYA_INTEGRATION === "1";

afterEach(() => router.resetLayaForTests());

// ---------------------------------------------------------------------------
// Corpus structure
// ---------------------------------------------------------------------------

test("corpus covers >= 20 cases and every category", () => {
	assert.ok(CORPUS.length >= 20, `expected >= 20 cases, got ${CORPUS.length}`);
	const categories = new Set(CORPUS.map((c) => c.expectedCategory));
	for (const cat of router.CATEGORIES) {
		assert.ok(categories.has(cat), `corpus must cover category "${cat}"`);
	}
	const complexities = new Set(CORPUS.map((c) => c.expectedComplexity));
	assert.deepEqual([...complexities].sort(), ["complex", "moderate", "standard"]);
	const followUps = CORPUS.filter((c) => c.history?.length);
	assert.ok(followUps.length >= 3, "corpus must include follow-up cases");
	assert.ok(CORPUS.some((c) => c.mustSee?.length), "corpus must include context-dependent cases");
});

test("corpus cases nominate acceptable mock tiers (escalation cases excepted)", () => {
	const escalateCategories = new Set(["coding", "debugging", "reasoning", "infra"]);
	for (const c of CORPUS) {
		// A case may deliberately nominate the cheap tier for complex demanding
		// work: the escalation rule must then land on an acceptable strong tier.
		const escalationCase =
			c.expectedComplexity === "complex" &&
			escalateCategories.has(c.expectedCategory) &&
			c.mockTier === "cheap";
		assert.ok(
			c.acceptableTiers.includes(c.mockTier) || (escalationCase && c.acceptableTiers.includes("strong")),
			`${c.id}: mock tier ${c.mockTier} must be acceptable, or escalate to an acceptable strong tier`,
		);
		assert.ok(router.CATEGORIES.includes(c.expectedCategory), `${c.id}: bad category`);
		assert.ok(router.COMPLEXITIES.includes(c.expectedComplexity), `${c.id}: bad complexity`);
	}
});

// ---------------------------------------------------------------------------
// Deterministic rule evaluation over the corpus
// ---------------------------------------------------------------------------

test("corpus routes deterministically: categories preserved, zero violations, policy counted", async () => {
	const config = testConfig();
	const ctx = makeCtx(null);
	const eligible = router.eligibleModels(config, ctx.modelRegistry);
	let violations = 0;
	let unnecessaryEscalations = 0;
	let escalations = 0;
	const categoryAccuracy: string[] = [];
	for (const c of CORPUS) {
		router.setLayaForTests(mockClassifier(caseMock(c)));
		const { decision } = await router.classifyAndDecide(
			routeRequest(caseMessages(c), undefined) as never,
			ctx as never,
			config,
			eligible,
		);
		const tier = tierOf(`${decision.entry.provider}/${decision.entry.id}`);
		if (!c.acceptableTiers.includes(tier)) violations++;
		if (decision.rule === "escalated") {
			escalations++;
			if (!c.acceptableTiers.includes("strong")) unnecessaryEscalations++;
		}
		if (decision.category !== c.expectedCategory || decision.complexity !== c.expectedComplexity) {
			categoryAccuracy.push(c.id);
		}
		assert.ok(
			eligible.some((e) => e === decision.entry),
			`${c.id}: decision must land on an eligible model`,
		);
	}
	assert.equal(violations, 0, "no case may route outside its acceptable tiers");
	assert.equal(unnecessaryEscalations, 0, "escalation must only happen when a strong tier is acceptable");
	assert.equal(categoryAccuracy.length, 0, "rules must preserve the classified category and complexity");
	// Escalations do happen by design: the intermittent-503 case nominates the
	// cheap tier for complex debugging work.
	assert.ok(escalations >= 1, "the corpus includes at least one policy escalation");
	console.log(`[laya-eval] corpus=${CORPUS.length} escalations=${escalations} violations=${violations}`);
});

test("corpus escalation lands on the strongest eligible model, not the fallback direction", async () => {
	const case503 = CORPUS.find((c) => c.id === "intermittent-503");
	assert.ok(case503);
	const config = testConfig({ fallback: "cheapest" }); // shipped direction
	const ctx = makeCtx(null);
	const eligible = router.eligibleModels(config, ctx.modelRegistry);
	router.setLayaForTests(mockClassifier(caseMock(case503)));
	const { decision } = await router.classifyAndDecide(
		routeRequest(caseMessages(case503), undefined) as never,
		ctx as never,
		config,
		eligible,
	);
	// fallback:"cheapest" must not neuter escalation: complex debugging work
	// nominated on the cheap tier escalates to the strong tier.
	assert.equal(decision.rule, "escalated");
	assert.equal(decision.entry.id, "qwen/qwen3.8-max");
});

test("corpus fallback policy routes every case deterministically under no signal", async () => {
	const config = testConfig();
	const ctx = makeCtx(null);
	const eligible = router.eligibleModels(config, ctx.modelRegistry);
	router.setLayaForTests(mockClassifier({ model: {}, complexity: {}, category: {} }));
	for (const c of CORPUS.slice(0, 8)) {
		const { decision } = await router.classifyAndDecide(
			routeRequest(caseMessages(c), undefined) as never,
			ctx as never,
			config,
			eligible,
		);
		assert.equal(decision.rule, "fallback", `${c.id}: no signal must hit the fallback policy`);
		assert.equal(decision.entry.id, "deepseek/deepseek-v4-flash");
	}
});

test("corpus cost stays within fixed baselines (no improvement claims)", () => {
	const chosen = CORPUS.map((c) => TIER_MODELS[c.mockTier]);
	const routerCost = chosen.reduce((sum, m) => sum + priceOf(m), 0);
	const alwaysCheapest = CORPUS.length * priceOf(TIER_MODELS.cheap);
	const alwaysStrongest = CORPUS.length * priceOf(TIER_MODELS.strong);
	assert.ok(routerCost <= alwaysStrongest + 1e-9, "router corpus cost must not exceed the strong baseline");
	assert.ok(routerCost >= alwaysCheapest - 1e-9, "router corpus cost is bounded below by the cheap baseline");
	console.log(
		`[laya-eval] input-cost baselines per request: router=${(routerCost / CORPUS.length).toFixed(3)} ` +
			`always-cheap=${priceOf(TIER_MODELS.cheap)} always-strong=${priceOf(TIER_MODELS.strong)} (estimates, USD/1M in)`,
	);
});

test("follow-up corpus cases carry the referenced context to the classifier", async () => {
	const config = testConfig({ recentTurns: 4 });
	const ctx = makeCtx(null);
	for (const c of CORPUS.filter((x) => x.mustSee?.length)) {
		const input = router.buildRoutingInput(
			caseMessages(c) as never,
			"",
			null,
			config,
		);
		for (const needle of c.mustSee ?? []) {
			assert.ok(input.prompt.includes(needle), `${c.id}: context must retain "${needle}"`);
		}
		assert.ok(
			input.prompt.includes(c.request.slice(0, 60)),
			`${c.id}: current request must be in the prompt`,
		);
	}
});

// ---------------------------------------------------------------------------
// Semantic evaluation with the real local Laya (opt-in)
// ---------------------------------------------------------------------------

test("real Laya over the corpus: measure routing quality", { skip: !enabled }, async () => {
	const config = testConfig();
	const ctx = makeCtx({ tokens: 30_000, contextWindow: 1_000_000 });
	const eligible = router.eligibleModels(config, ctx.modelRegistry);
	let categoryHits = 0;
	let complexityHits = 0;
	let violations = 0;
	let escalations = 0;
	let fallbacks = 0;
	let cost = 0;
	let layaMsTotal = 0;
	for (const c of CORPUS) {
		const { decision, layaMs, input } = await router.classifyAndDecide(
			routeRequest(caseMessages(c), undefined) as never,
			ctx as never,
			config,
			eligible,
		);
		if (decision.category === c.expectedCategory) categoryHits++;
		if (decision.complexity === c.expectedComplexity) complexityHits++;
		if (!c.acceptableTiers.includes(tierOf(`${decision.entry.provider}/${decision.entry.id}`))) violations++;
		if (decision.rule === "escalated") escalations++;
		if (decision.rule === "fallback") fallbacks++;
		const entry = config.models.find(
			(m) => m.provider === decision.entry.provider && m.id === decision.entry.id,
		);
		cost += entry?.priceIn ?? 0;
		layaMsTotal += layaMs ?? 0;
		assert.ok(
			eligible.some((e) => e === decision.entry),
			`${c.id}: real decision must land on an eligible model`,
		);
		assert.ok(input.stateTokens === null || input.stateTokens <= input.roomTokens, `${c.id}: state must fit Laya's room`);
	}
	console.log(
		`[laya-eval] REAL corpus=${CORPUS.length} category=${((categoryHits / CORPUS.length) * 100).toFixed(1)}% ` +
			`complexity=${((complexityHits / CORPUS.length) * 100).toFixed(1)}% violations=${violations} ` +
			`escalations=${escalations} fallbacks=${fallbacks} mean_laya_ms=${Math.round(layaMsTotal / CORPUS.length)} ` +
			`mean_input_price=${(cost / CORPUS.length).toFixed(3)} (measurement, not a guarantee)`,
	);
});