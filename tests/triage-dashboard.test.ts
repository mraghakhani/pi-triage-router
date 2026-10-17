/** Tests for the /triage dashboard: session routing history, cost totals, report format. */
import { test, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import * as router from "../triage-router.ts";
import {
	cacheSummary,
	configSummary,
	costSummary,
	formatDashboard,
	HISTORY_CAP,
	historyOptions,
	modelOptions,
	recordCost,
	recordRoute,
	resetDashboardForTests,
	routeHistory,
	sessionCost,
	usdLabel,
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

// ---------------------------------------------------------------------------
// Menu helpers (interactive /triage dialogs)
// ---------------------------------------------------------------------------

test("modelOptions lists every configured model with price and status", () => {
	const lines = modelOptions(testConfig());
	assert.equal(lines.length, 6);
	assert.ok(lines[0].startsWith("litellm/deepseek/deepseek-v4-flash · in "));
	assert.ok(lines[0].endsWith("active"), `line shows status: ${lines[0]}`);
	const disabled = modelOptions(testConfig({ models: [{ provider: "litellm", id: "a/b", description: "x", enabled: false }] }));
	assert.ok(disabled[0].endsWith("disabled"));
});

test("historyOptions is newest first and matches the recorded history", () => {
	recordRoute({ timestamp: 1_000, model: "litellm/a", rule: "primary", complexity: "standard", category: "question", cost: 0.01 });
	recordRoute({ timestamp: 2_000, model: "litellm/b", rule: "sticky", complexity: "standard", category: "question", cost: null });
	const lines = historyOptions();
	assert.equal(lines.length, 2);
	assert.ok(lines[0].includes("litellm/b"), "newest record first");
	assert.ok(lines[1].includes("litellm/a"));
	assert.ok(lines[1].includes("$0.0100"), "known cost shown");
	assert.ok(lines[0].includes("-"), "unknown cost renders as dash");
});

test("costSummary, cacheSummary, configSummary are chat-free multiline texts", () => {
	recordRoute({ model: "litellm/a", rule: "primary", complexity: "standard", category: "question", cost: 0.02 });
	const cost = costSummary();
	assert.ok(cost.includes("Total this session: $0.0200"));
	assert.ok(cost.includes("- litellm/a: $0.0200"));
	const cache = cacheSummary(testConfig());
	assert.ok(!cache.includes("##"), "no markdown headers in dialog text");
	assert.ok(cache.includes("Prices: disabled"));
	const conf = configSummary(testConfig());
	assert.ok(!conf.includes("##"));
	assert.ok(conf.includes("Classifier: local"));
	assert.ok(conf.includes("Fallback: cheapest"));
});

test("usdLabel formats known and unknown values", () => {
	assert.equal(usdLabel(0.5), "$0.5000");
	assert.equal(usdLabel(null), "-");
	assert.equal(usdLabel(undefined), "-");
});

// ---------------------------------------------------------------------------
// updateConfigFile (the /triage menu's persistence)
// ---------------------------------------------------------------------------

let configDir: string;
beforeEach(() => {
	configDir = mkdtempSync(join(tmpdir(), "triage-config-"));
});
afterEach(() => {
	rmSync(configDir, { recursive: true, force: true });
});

test("updateConfigFile creates a complete config when the file is missing", () => {
	const path = join(configDir, "triage-router.json");
	const config = router.updateConfigFile(path, { override: "litellm/deepseek/deepseek-v4-flash" });
	assert.equal(config.override, "litellm/deepseek/deepseek-v4-flash");
	const raw = JSON.parse(readFileSync(path, "utf8"));
	assert.equal(raw.override, "litellm/deepseek/deepseek-v4-flash");
	assert.ok(Array.isArray(raw.models), "defaults materialize on first edit");
	assert.ok(!("problems" in raw), "problems never serialize into the file");
	assert.equal(config.problems.length, 0, "materialized defaults are valid");
});

test("updateConfigFile patches an existing file and keeps unknown keys", () => {
	const path = join(configDir, "triage-router.json");
	writeFileSync(
		path,
		JSON.stringify({ fallback: "cheapest", scoring: { enabled: false, missingPolicy: "penalize" }, futureKey: true, models: [{ provider: "litellm", id: "a/b", description: "x" }] }),
	);
	const config = router.updateConfigFile(path, { fallback: "strongest", scoring: { enabled: true } });
	assert.equal(config.fallback, "strongest");
	assert.equal(config.scoring.enabled, true);
	assert.equal(config.scoring.missingPolicy, "penalize", "nested patch merges, siblings survive");
	const raw = JSON.parse(readFileSync(path, "utf8"));
	assert.equal(raw.futureKey, true, "unknown keys survive the patch");
	assert.equal(raw.scoring.missingPolicy, "penalize");
});

test("updateConfigFile toggles one model's enabled flag", () => {
	const path = join(configDir, "triage-router.json");
	writeFileSync(
		path,
		JSON.stringify({ models: [
			{ provider: "litellm", id: "a/b", description: "x" },
			{ provider: "litellm", id: "c/d", description: "y" },
		] }),
	);
	const config = router.updateConfigFile(path, {
		models: [
			{ provider: "litellm", id: "a/b", description: "x", enabled: false },
			{ provider: "litellm", id: "c/d", description: "y" },
		],
	});
	assert.equal(config.models.find((m) => m.id === "a/b")?.enabled, false);
	assert.equal(config.models.find((m) => m.id === "c/d")?.enabled, true, "untouched models stay enabled");
});

test("updateConfigFile clearing the override writes null and validateConfig drops it", () => {
	const path = join(configDir, "triage-router.json");
	router.updateConfigFile(path, { override: "litellm/a/b" });
	assert.equal(router.loadConfig(path).override, "litellm/a/b");
	const config = router.updateConfigFile(path, { override: null });
	assert.equal(config.override, undefined, "null override is treated as none");
	assert.equal(JSON.parse(readFileSync(path, "utf8")).override, null);
});

test("menu patches never materialize the runtime seedPath in the config file", () => {
	const path = join(configDir, "triage-router.json");
	writeFileSync(
		path,
		JSON.stringify({ models: [{ provider: "litellm", id: "a/b", description: "x" }], prices: { enabled: true }, benchmarks: { enabled: true } }),
	);
	const config = router.updateConfigFile(path, {
		prices: { enabled: false, seedPath: "/bogus/seed.json" },
		benchmarks: { enabled: false, seedPath: "/bogus/seed.json" },
	});
	assert.equal(config.problems.length, 0, "no unknown-key problems");
	const raw = JSON.parse(readFileSync(path, "utf8"));
	assert.ok(!("seedPath" in raw.prices), "prices.seedPath stripped");
	assert.ok(!("seedPath" in raw.benchmarks), "benchmarks.seedPath stripped");
	assert.equal(raw.prices.enabled, false, "the actual edit survives");
});

test("loadConfig picks up menu edits without a restart", () => {
	const path = join(configDir, "triage-router.json");
	router.updateConfigFile(path, { fallback: "cheapest" });
	assert.equal(router.loadConfig(path).fallback, "cheapest");
	router.updateConfigFile(path, { fallback: "strongest" });
	assert.equal(router.loadConfig(path).fallback, "strongest", "mtime change reloads the config");
});