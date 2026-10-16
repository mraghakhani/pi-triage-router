/** Tests for the /triage dashboard: session routing history, cost totals, report format. */
import { test, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import * as router from "../triage-router.ts";
import {
	formatDashboard,
	HISTORY_CAP,
	recordCost,
	recordRoute,
	resetDashboardForTests,
	routeHistory,
	sessionCost,
} from "../triage-router-lib/triage-router-dashboard.ts";
import { testConfig, makeCtx, routeRequest, mockClassifier, userMsg } from "./fixtures.ts";

beforeEach(() => resetDashboardForTests());
afterEach(() => resetDashboardForTests());

test("recordRoute appends and sessionCost sums known costs", () => {
	assert.equal(routeHistory().length, 0);
	assert.equal(sessionCost().total, null, "empty history has no total");

	recordRoute({ model: "litellm/a", rule: "primary", complexity: "standard", category: "question", cost: 0.01 });
	recordRoute({ model: "litellm/a", rule: "sticky", complexity: "standard", category: "question", cost: 0.02 });
	recordRoute({ model: "litellm/b", rule: "direct", complexity: "standard", category: "question", cost: null });

	assert.equal(routeHistory().length, 3);
	const { total, byModel } = sessionCost();
	assert.equal(total, 0.03);
	assert.deepEqual(byModel, { "litellm/a": 0.03 });
});

test("history stays bounded at HISTORY_CAP (oldest dropped)", () => {
	for (let i = 0; i < HISTORY_CAP + 10; i++) {
		recordRoute({ model: `litellm/m${i}`, rule: "primary", complexity: "standard", category: "question", cost: 1 });
	}
	const history = routeHistory();
	assert.equal(history.length, HISTORY_CAP);
	assert.equal(history[0].model, `litellm/m${HISTORY_CAP + 10 - HISTORY_CAP}`);
	assert.equal(history[history.length - 1].model, `litellm/m${HISTORY_CAP + 10 - 1}`);
	// Totals only cover the retained records.
	assert.equal(sessionCost().total, HISTORY_CAP * 1);
});

test("recordCost sums known halves and stays null when both unknown", () => {
	assert.equal(recordCost(null, null), null);
	assert.equal(recordCost(undefined, undefined), null);
	assert.equal(recordCost(0.01, null), 0.01);
	assert.equal(recordCost(null, 0.02), 0.02);
	assert.equal(recordCost(0.01, 0.02), 0.03);
});

test("formatDashboard renders lineup, history, cache, and config sections", () => {
	recordRoute({ model: "litellm/deepseek/deepseek-v4-flash", rule: "primary", complexity: "standard", category: "question", cost: 0.01 });
	const config = testConfig();
	const report = formatDashboard(config);
	assert.ok(report.includes("# pi-triage-router dashboard"));
	assert.ok(report.includes("## Model lineup"));
	assert.ok(report.includes("deepseek/deepseek-v4-flash"));
	assert.ok(report.includes("## Session cost (estimated)"));
	assert.ok(report.includes("$0.0100"), "recorded cost is shown");
	assert.ok(report.includes("## Route history"));
	assert.ok(report.includes("## Cache status"));
	assert.ok(report.includes("- Prices: disabled"), "test config disables the price cache");
	assert.ok(report.includes("## Config"));
	assert.ok(report.includes("- Classifier: local"));
	assert.ok(report.includes("- Jev backend: not configured"));
});

test("formatDashboard with empty history says so and shows no cost", () => {
	const report = formatDashboard(testConfig());
	assert.ok(report.includes("_No requests routed yet this session._"));
	assert.ok(report.includes("Total this session: **-**"));
});

test("handleRequest records each routing decision into the session history", async () => {
	const fake = mockClassifier({ model: {}, complexity: {}, category: {} });
	router.setLayaForTests(fake);
	const config = testConfig();
	await router.handleRequest(routeRequest([userMsg("hello")], undefined), makeCtx(null, []), config, "classifier");
	const history = routeHistory();
	assert.equal(history.length, 1, "each route decision is recorded once");
	assert.ok(history[0].model.startsWith("litellm/"), `recorded model is a config composite: ${history[0].model}`);
	assert.equal(typeof history[0].rule, "string");
	assert.equal(typeof history[0].timestamp, "number");
});