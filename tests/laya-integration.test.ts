/**
 * Integration tests for triage-router using the REAL local Laya model.
 * No network beyond the initial ONNX weight download (already cached at
 * ~/.cache/receptron-laya); no paid API calls; classification runs locally.
 *
 * Opt-in because the first run downloads ~1.7 GB of weights:
 *   LAYA_INTEGRATION=1 node --test tests/laya-integration.test.ts
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import * as router from "../triage-router.ts";
import { testConfig, registryFor, ALL_KEYS, userMsg, asstMsg, makeCtx, routeRequest } from "./fixtures.ts";

const enabled = process.env.LAYA_INTEGRATION === "1";

test("real Laya: classify a small request end-to-end", { skip: !enabled }, async () => {
	const config = testConfig();
	const ctx = makeCtx({ tokens: 30_000, contextWindow: 1_000_000 });
	const { decision, layaMs, input } = await router.classifyAndDecide(
		routeRequest([userMsg("Fix the typo in the README file.")], undefined),
		ctx,
		config,
		router.eligibleModels(config, registryFor(ALL_KEYS)),
	);
	assert.ok(["primary", "escalated", "downgraded", "fallback"].includes(decision.rule));
	assert.ok(decision.entry, "decision must name a real config entry");
	assert.ok(ctx.modelRegistry.find(decision.entry.provider, decision.entry.id), "decision must exist in the registry");
	assert.ok(router.CATEGORIES.includes(decision.category));
	assert.ok(router.COMPLEXITIES.includes(decision.complexity));
	assert.equal(typeof layaMs, "number");
	assert.ok(layaMs !== null && layaMs < 5000, `local inference must be fast, got ${layaMs}ms`);
	assert.ok(input.prompt.includes("Fix the typo in the README file."));
	assert.ok(decision.estimate.inputTokens !== null, "cost estimate should be present");
});

test("real Laya: a long request still fits the measured state room (regression)", { skip: !enabled }, async () => {
	const config = testConfig();
	const ctx = makeCtx({ tokens: 30_000, contextWindow: 1_000_000 });
	const longRequest = "Root-cause the intermittent 503s across the Envoy gateway: ".repeat(90) +
		"Check timeouts, retries, and upstream health for every dependency.";
	const history = Array.from({ length: 8 }, (_, i) => userMsg(`Turn ${i} about the legacy SOAP gateway and its quirks.`));
	const { input } = await router.classifyAndDecide(
		routeRequest([...history, userMsg(longRequest)], undefined),
		ctx,
		config,
		router.eligibleModels(config, ctx.modelRegistry),
	);
	// Regression: the state (the whole JSON-dumped {role, prompt}) used to
	// exceed Laya's ~316-token room, so Laya truncated it silently from the
	// tail - cutting the [Current request] section first. Now the router fits it.
	assert.ok(input.stateTokens !== null, "the real tokenizer must report state tokens");
	assert.ok(
		input.stateTokens <= input.roomTokens,
		`state must fit Laya's room: ${input.stateTokens} > ${input.roomTokens}`,
	);
	assert.ok(input.prompt.includes("[Current request"), "the current request section must survive");
});

test("real Laya: conversation context reaches the classifier for a follow-up", { skip: !enabled }, async () => {
	const config = testConfig();
	const ctx = makeCtx(null);
	const messages = [
		userMsg("We are debugging a PostgreSQL connectivity problem in our Kubernetes cluster."),
		asstMsg("I checked the pg service endpoints; the pool is exhausted."),
		userMsg("Fix the same issue in the other service."),
	];
	const { decision, input } = await router.classifyAndDecide(
		routeRequest(messages, undefined),
		ctx,
		config,
		router.eligibleModels(config, registryFor(ALL_KEYS)),
	);
	assert.ok(input.prompt.includes("PostgreSQL"), "prior conversation must reach Laya");
	assert.ok(input.prompt.includes("Kubernetes"));
	assert.equal(input.currentRequest, "Fix the same issue in the other service.");
	assert.ok(["primary", "escalated", "downgraded", "fallback"].includes(decision.rule));
});

test("real Laya: a hard architecture task and a docs tweak produce decisions", { skip: !enabled }, async () => {
	const config = testConfig({ fallback: "strongest" });
	const ctx = makeCtx(null);
	for (const [request, note] of [
		["Redesign our auth system: migrate 40 services from session cookies to OAuth2 with zero downtime.", "hard task"],
		["Rename the variable 'foo' to 'bar' in utils.js.", "small task"],
	] as const) {
		const { decision } = await router.classifyAndDecide(
			routeRequest([userMsg(request)], undefined),
			ctx,
			config,
			router.eligibleModels(config, registryFor(ALL_KEYS)),
		);
		assert.ok(decision.entry, `${note}: decision must name a model`);
		console.log(`[laya-integration] ${note}: model=${decision.entry.id} rule=${decision.rule} cplx=${decision.complexity} cat=${decision.category}`);
	}
});