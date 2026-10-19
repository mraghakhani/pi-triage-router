/**
 * Tests for the interactive groups menu (config-file-only group management
 * until now): create/delete a group, add/remove refs, and the delete guard
 * that clears a dangling activeGroup/defaultGroup in the same patch.
 *
 * The menu's persistence goes through updateConfigFile against a temp config
 * file; the menu state is fed via setConfigForTests + a GetConfig closure so
 * no real user config is touched.
 */
import { test, beforeEach, afterEach } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import * as router from "../triage-router.ts";
import { groupRefsMenu, groupsMenu } from "../triage-router.ts";

const MENU_BACK = "Back";

let configDir: string;
let configPath: string;
let notifications: { message: string; level: string }[];

/** Scripted ui: select/input answer from a queue; notify records. */
function makeCtx(answers: string[]) {
	return {
		ui: {
			select: async (_title: string, options: string[]) => {
				const next = answers.shift();
				assert.ok(next !== undefined, `scripted answer missing (options were ${options.join(" | ")})`);
				assert.ok(options.includes(next), `scripted answer "${next}" not among options ${options.join(" | ")}`);
				return next;
			},
			input: async (_prompt: string, _initial: string) => answers.shift() ?? "",
			notify: (message: string, level: string) => {
				notifications.push({ message, level });
			},
		},
	} as never;
}

/** save wired to the temp file; the menu loop re-reads through loadConfig. */
async function save(patch: Record<string, unknown>): Promise<void> {
	router.setConfigForTests(router.updateConfigFile(configPath, patch));
}

const getConfig = () => router.loadConfig(configPath);

beforeEach(() => {
	configDir = mkdtempSync(join(tmpdir(), "triage-groups-menu-"));
	configPath = join(configDir, "triage-router.json");
	notifications = [];
	router.setConfigForTests(null);
});

afterEach(() => {
	router.setConfigForTests(null);
	rmSync(configDir, { recursive: true, force: true });
});

function seedConfig(raw: Record<string, unknown>): void {
	writeFileSync(configPath, JSON.stringify(raw));
}

test("groupsMenu creates a group then adds a ref to it", async () => {
	seedConfig({ models: [{ provider: "litellm", id: "a/b", description: "x" }] });
	// Script: groups list -> "New group…", name input, ref menu -> "Add ref…",
	// ref input, ref menu -> Back, groups list -> Back.
	const ctx = makeCtx([
		"New group…",
		"econ",
		"Add ref…",
		"-litellm/a/b",
		MENU_BACK,
		MENU_BACK,
	]);
	await groupsMenu(ctx, getConfig, save);
	const raw = JSON.parse(readFileSync(configPath, "utf8"));
	assert.deepEqual(raw.groups.econ, ["-litellm/a/b"]);
});

test("groupsMenu refuses __proto__ and duplicate names without writing", async () => {
	seedConfig({ groups: { econ: ["a/b"] }, models: [] });
	// Script: New group… -> "__proto__" (rejected), New group… -> "econ"
	// (duplicate -> opens existing ref menu), Back, Back.
	const ctx = makeCtx(["New group…", "__proto__", "New group…", "econ", MENU_BACK, MENU_BACK]);
	await groupsMenu(ctx, getConfig, save);
	const raw = JSON.parse(readFileSync(configPath, "utf8"));
	assert.deepEqual(Object.keys(raw.groups), ["econ"], "no new group was persisted");
	assert.ok(notifications.some((n) => n.message.includes("reserved")), "__proto__ notified");
	assert.ok(notifications.some((n) => n.message.includes("already exists")), "duplicate notified");
});

test("groupRefsMenu removes a selected ref and reports the change", async () => {
	seedConfig({ groups: { econ: ["a/b", "c/d"] }, models: [] });
	router.setConfigForTests(router.loadConfig(configPath));
	const ctx = makeCtx(["c/d", MENU_BACK]);
	await groupRefsMenu(ctx, "econ", getConfig, save);
	const raw = JSON.parse(readFileSync(configPath, "utf8"));
	assert.deepEqual(raw.groups.econ, ["a/b"]);
});

test("groupRefsMenu delete clears a dangling activeGroup in the same patch", async () => {
	seedConfig({ groups: { econ: ["a/b"], gone: [] }, activeGroup: "gone", defaultGroup: "gone", models: [] });
	router.setConfigForTests(router.loadConfig(configPath));
	const ctx = makeCtx(["Delete group", "delete"]);
	await groupRefsMenu(ctx, "gone", getConfig, save);
	const raw = JSON.parse(readFileSync(configPath, "utf8"));
	assert.ok(!("gone" in raw.groups), "group key removed from the file");
	assert.ok("econ" in raw.groups, "sibling groups survive");
	assert.equal(raw.activeGroup, null, "dangling activeGroup cleared");
	assert.equal(raw.defaultGroup, null, "dangling defaultGroup cleared");
});

test("groupRefsMenu delete keeps confirm=false a no-op", async () => {
	seedConfig({ groups: { econ: ["a/b"] }, models: [] });
	router.setConfigForTests(router.loadConfig(configPath));
	const ctx = makeCtx(["Delete group", "keep", MENU_BACK]);
	await groupRefsMenu(ctx, "econ", getConfig, save);
	const raw = JSON.parse(readFileSync(configPath, "utf8"));
	assert.ok("econ" in raw.groups);
});
