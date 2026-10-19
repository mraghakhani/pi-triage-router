/**
 * Regression tests for the /triage command handler wiring (e61f0c6): the
 * Markdown report must be posted in EVERY mode — including interactive TUI,
 * where it used to be skipped and report-only sections like "## Model
 * groups" were invisible. In TUI the menu then opens on top of the report.
 */
import { test, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import routerDefault, * as router from "../triage-router.ts";
import { formatDashboard } from "../triage-router-lib/triage-router-dashboard.ts";
import { testConfig } from "./fixtures.ts";

/** Minimal ExtensionAPI mock capturing registerCommand + sendMessage. */
function makePi() {
	const messages: any[] = [];
	let triageHandler: ((args: string, ctx: any) => Promise<void>) | null = null;
	const pi: any = {
		registerVirtualModel: () => {},
		registerCommand: (_name: string, cmd: any) => {
			triageHandler = cmd.handler;
		},
		sendMessage: (msg: any) => {
			messages.push(msg);
		},
	};
	return { pi, messages, handler: () => triageHandler as (args: string, ctx: any) => Promise<void> };
}

test.beforeEach(() => {
	router.setConfigForTests(testConfig() as router.RouterConfig);
});

test.afterEach(() => {
	router.setConfigForTests(null);
});

test("tui mode: report is posted before the menu opens", async () => {
	const { pi, messages, handler } = makePi();
	routerDefault(pi);
	const h = handler();
	assert.ok(h, "triage command registered");
	// ui.select throws a sentinel the moment the menu opens: proves the menu
	// ran AFTER the report and lets the handler's promise settle.
	const ctx = {
		mode: "tui",
		hasUI: true,
		ui: { select: async () => { throw new Error("menu-opened"); } },
	} as never;
	await assert.rejects(h("", ctx), { message: "menu-opened" });
	assert.equal(messages.length, 1, "exactly one report message before the menu");
	assert.equal(messages[0].customType, "triage-dashboard");
	assert.equal(messages[0].display, true);
	assert.equal(messages[0].content, formatDashboard(testConfig() as router.RouterConfig),
		"report content must be the full formatDashboard output, not a trimmed variant");
});

test("print/json mode: report is posted, menu is not opened", async () => {
	const { pi, messages, handler } = makePi();
	routerDefault(pi);
	const h = handler();
	assert.ok(h, "triage command registered");
	const ctx = { mode: "json" } as never;
	await h("", ctx); // must resolve without touching ui
	assert.equal(messages.length, 1);
	assert.equal(messages[0].customType, "triage-dashboard");
});

test("older pi builds without registerCommand still load the router", () => {
	const messages: any[] = [];
	const pi: any = { registerVirtualModel: () => {}, sendMessage: (m: any) => messages.push(m) };
	assert.doesNotThrow(() => routerDefault(pi));
	assert.equal(messages.length, 0);
});
