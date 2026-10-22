/**
 * Session routing history + dashboard report for the triage-router extension.
 *
 * Rules honored here:
 *  - In-memory only: history and totals are per-session, never persisted; the
 *    dashboard is a live view, not an audit log.
 *  - Bounded: at most HISTORY_CAP records are kept (oldest dropped).
 *  - Costs are ESTIMATES from the routing decision (estCostIn/estCostOut),
 *    never invented; unknown stays null and renders as "-".
 *  - No network: the report reads only config, the local price cache/seed,
 *    and the local benchmark cache/seed.
 *
 * This module type-imports shared types from triage-router.ts but imports no
 * runtime values from it (the router imports values from here).
 */
import type { Category, Complexity, ModelEntry, RouterConfig } from "../triage-router.ts";
import { datasetStatus, readDataset, type MetricKey } from "./triage-router-benchmarks.ts";
import { readPriceDataset } from "./triage-router-prices.ts";

// ---------------------------------------------------------------------------
// Routing history
// ---------------------------------------------------------------------------

/** One routing decision as recorded for the dashboard (per-session). */
export interface RoutingRecord {
	/** ms epoch of the decision. */
	timestamp: number;
	/** Chosen model as "provider/id". */
	model: string;
	/** Which deterministic rule produced the selection. */
	rule: string;
	complexity: Complexity;
	category: Category;
	/** Estimated request cost estCostIn + estCostOut in USD, or null. */
	cost: number | null;
}

/** Ring-buffer cap for in-memory session history. */
export const HISTORY_CAP = 50;

let history: RoutingRecord[] = [];

/** Estimated request cost estCostIn + estCostOut in USD, or null when unknown. */
export function recordCost(
	estCostIn: number | null | undefined,
	estCostOut: number | null | undefined,
): number | null {
	const inUsd = typeof estCostIn === "number" ? estCostIn : null;
	const outUsd = typeof estCostOut === "number" ? estCostOut : null;
	if (inUsd === null && outUsd === null) return null;
	return (inUsd ?? 0) + (outUsd ?? 0);
}

/** Append a record (oldest dropped past HISTORY_CAP); returns a copy. */
export function recordRoute(
	record: Omit<RoutingRecord, "timestamp"> & Partial<Pick<RoutingRecord, "timestamp">>,
): RoutingRecord {
	const entry: RoutingRecord = { ...record, timestamp: record.timestamp ?? Date.now() };
	history.push(entry);
	if (history.length > HISTORY_CAP) {
		history = history.slice(history.length - HISTORY_CAP);
	}
	return { ...entry };
}

/** Current session history, oldest first (copy). */
export function routeHistory(): RoutingRecord[] {
	return [...history];
}

/** Session cost totals from the recorded history. */
export function sessionCost(): { total: number | null; byModel: Record<string, number> } {
	const byModel: Record<string, number> = {};
	let total: number | null = null;
	for (const r of history) {
		if (r.cost == null) continue;
		total = (total ?? 0) + r.cost;
		byModel[r.model] = (byModel[r.model] ?? 0) + r.cost;
	}
	return { total, byModel };
}

export function resetDashboardForTests(): void {
	history = [];
}

// ---------------------------------------------------------------------------
// Report formatting
// ---------------------------------------------------------------------------

function usd(value: number | null | undefined): string {
	return typeof value === "number" ? `$${value.toFixed(4)}` : "-";
}

/** USD label for menu lines (e.g. "$0.0123", "-" when unknown). */
export function usdLabel(value: number | null | undefined): string {
	return usd(value);
}

function composite(entry: Pick<ModelEntry, "provider" | "id">): string {
	return `${entry.provider}/${entry.id}`;
}

function timeLabel(timestamp: number): string {
	// Hour+minute only: full toLocaleTimeString made the history row 102
	// chars with a realistic-long model id (past the 100-char audit).
	return new Date(timestamp).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}

/** Price used by the report. Config pins win: OpenRouter catalog prices for
 * dynamic-priced models (time-windowed snapshots) fluctuate between refreshes,
 * so a user-declared `priceIn`/`priceOut` is the reference and the refreshed
 * catalog cache only fills models without pins. */
function reportPrice(
	config: RouterConfig,
	entry: ModelEntry,
): { input: number | null; output: number | null; source: "cache" | "config" } {
	const cached = config.prices.enabled
		? readPriceDataset(config.prices.cachePath, config.prices.seedPath)?.prices[entry.id]
		: undefined;
	return {
		input: entry.priceIn ?? cached?.input ?? null,
		output: entry.priceOut ?? cached?.output ?? null,
		source: entry.priceIn === undefined && entry.priceOut === undefined && cached ? "cache" : "config",
	};
}

/** One quality metric from the benchmark cache for the lineup table. */
function benchmarkQuality(config: RouterConfig, id: string): string | null {
	if (!config.benchmarks.enabled) return null;
	const dataset = readDataset(config.benchmarks.cachePath, config.benchmarks.seedPath);
	const metrics = dataset?.models[id];
	if (!metrics) return null;
	const key: MetricKey = "coding";
	const value = metrics[key];
	return typeof value === "number" ? value.toFixed(0) : null;
}

function lineupSection(config: RouterConfig): string {
	const lines = ["## Model lineup", "", "| model | $/M in | $/M out | coding | status |", "|---|---|---|---|---|"];
	for (const entry of config.models) {
		const price = reportPrice(config, entry);
		const quality = benchmarkQuality(config, entry.id);
		const status =
			entry.enabled === false ? "disabled" : price.source === "cache" ? "active (catalog price)" : "active";
		lines.push(
			`| ${composite(entry)} | ${usd(price.input)} | ${usd(price.output)} | ${quality ?? "-"} | ${status} |`,
		);
	}
	lines.push("");
	return lines.join("\n");
}

function costSection(): string {
	const { total, byModel } = sessionCost();
	const lines = ["## Session cost (estimated)", ""];
	lines.push(`Total this session: **${usd(total)}** across ${history.length} routed request(s).`);
	const models = Object.entries(byModel).sort((a, b) => b[1] - a[1]);
	if (models.length > 0) {
		lines.push("");
		lines.push("| model | estimated spend |", "|---|---|");
		for (const [model, cost] of models) lines.push(`| ${model} | ${usd(cost)} |`);
	}
	lines.push("");
	return lines.join("\n");
}

function historySection(): string {
	const lines = ["## Route history (newest first, session only)", ""];
	if (history.length === 0) {
		lines.push("_No requests routed yet this session._");
		lines.push("");
		return lines.join("\n");
	}
	lines.push("| time | model | rule | complexity | category | est cost |", "|---|---|---|---|---|---|");
	for (const r of [...history].reverse()) {
		lines.push(
			`| ${timeLabel(r.timestamp)} | ${r.model} | ${r.rule} | ${r.complexity} | ${r.category} | ${usd(r.cost)} |`,
		);
	}
	lines.push("");
	return lines.join("\n");
}

function cacheSection(config: RouterConfig): string {
	const lines = ["## Cache status", ""];
	const ids = config.models.filter((m) => m.enabled !== false).map((m) => m.id);
	if (config.benchmarks.enabled) {
		lines.push(`- Benchmarks: ${datasetStatus(config.benchmarks, ids)}`);
	} else {
		lines.push("- Benchmarks: disabled");
	}
	if (config.prices.enabled) {
		const dataset = readPriceDataset(config.prices.cachePath, config.prices.seedPath);
		lines.push(
			dataset
				? `- Prices: ${Object.keys(dataset.prices).length} model(s), fetched ${dataset.fetchedAt.slice(0, 10)}`
				: "- Prices: no dataset (cache/seed absent)",
		);
	} else {
		lines.push("- Prices: disabled (config prices only)");
	}
	lines.push("");
	return lines.join("\n");
}

/** Short unset labels, shared by every call site so future merges need one
 * change. Verbose fallback chains live in the README and the edit-menu
 * submenus, not in these labels (80-col wrap safety). */
const GROUP_UNSET_TEXT = {
	activeGroup: "none",
	defaultGroup: "none",
} as const;

/** One-line group label for the Config section and the edit menu:
 * name + ref count when the field names a defined group, the short unset
 * text otherwise (override via unsetText for specialized labels). */
export function groupLabel(
	config: RouterConfig,
	field: "activeGroup" | "defaultGroup",
	unsetText: string = GROUP_UNSET_TEXT[field],
): string {
	const name = config[field];
	if (!name) return unsetText;
	const refs = config.groups?.[name];
	// Array.isArray: hand-built configs can hold anything on the prototype chain.
	return Array.isArray(refs) ? `"${name}" (${refs.length} ref${refs.length === 1 ? "" : "s"})` : `"${name}"`;
}

/** One-line active-group label for the Config section and the edit menu:
 * name + ref count when a group is active, "none (full catalog)"
 * otherwise. */
export function activeGroupLabel(config: RouterConfig): string {
	return groupLabel(config, "activeGroup", "none (full catalog)");
}

function configSection(config: RouterConfig): string {
	const lines = ["## Config", ""];
	const classifier = config.classifier;
	lines.push(
		`- Laya classifier: ${classifier.provider}${classifier.provider === "local" ? "" : ` (${classifier.provider}/${classifier.model})`}`,
	);
	const jev = config.jev;
	lines.push(
		jev
			? `- Jev backend: ${jev.provider}${jev.provider === "local" ? "" : ` (${jev.provider}/${jev.model})`}`
			: "- Jev backend: not configured",
	);
	lines.push(`- Fallback: ${config.fallback}`);
	// Both labels use the short "none" when unset: the descriptive fallbacks
	// ("full catalog", "default model, then cheapest") made the merged line
	// 82 chars and wrapped on 80-col terminals. Detail lives in the README.
	lines.push(
		`- Groups: active ${groupLabel(config, "activeGroup")} · default ${groupLabel(config, "defaultGroup")}`,
	);
	lines.push(`- Override: ${config.override ?? "none"}`);
	lines.push(`- Scoring: ${config.scoring.enabled ? `enabled (${config.scoring.missingPolicy})` : "disabled"}`);
	lines.push("");
	return lines.join("\n");
}

// ---------------------------------------------------------------------------
// Interactive menu helpers (TUI dialogs; no chat output)
// ---------------------------------------------------------------------------

/** One select-option line per configured model. */
export function modelOptionLine(config: RouterConfig, entry: ModelEntry): string {
	const price = reportPrice(config, entry);
	const quality = benchmarkQuality(config, entry.id);
	const status = entry.enabled === false ? "disabled" : "active";
	return `${composite(entry)} · in ${usd(price.input)} · out ${usd(price.output)} · coding ${quality ?? "-"} · ${status}`;
}

/** Select-option lines for the model lineup. */
export function modelOptions(config: RouterConfig): string[] {
	return config.models.map((entry) => modelOptionLine(config, entry));
}

/** Multiline detail (for a notify) of one configured model. */
export function modelDetail(config: RouterConfig, id: string): string {
	const entry = config.models.find((m) => m.id === id);
	if (!entry) return "unknown model";
	const price = reportPrice(config, entry);
	const quality = benchmarkQuality(config, entry.id);
	return [
		composite(entry),
		`prices: in ${usd(price.input)}/M · out ${usd(price.output)}/M (${price.source})`,
		`coding index: ${quality ?? "-"}`,
		`status: ${entry.enabled === false ? "disabled" : "active"}`,
		`description: ${entry.description}`,
	].join("\n");
}

/** Select-option lines for the session route history, newest first. */
export function historyOptions(): string[] {
	return [...history]
		.reverse()
		.map((r) => `${timeLabel(r.timestamp)} · ${r.model} · ${r.rule} · ${r.complexity} · ${r.category} · ${usd(r.cost)}`);
}

/** Multiline session-cost summary (for a notify). */
export function costSummary(): string {
	const { total, byModel } = sessionCost();
	const lines = [`Total this session: ${usd(total)} across ${history.length} routed request(s).`];
	for (const [model, cost] of Object.entries(byModel).sort((a, b) => b[1] - a[1])) {
		lines.push(`- ${model}: ${usd(cost)}`);
	}
	return lines.join("\n");
}

/** Multiline cache-status summary (for a notify). */
export function cacheSummary(config: RouterConfig): string {
	return cacheSection(config)
		.replace(/^##.*\n+/, "")
		.replace(/\n$/, "");
}

/** Model-group definitions: one bullet per group, one sub-bullet per ref.
 * Sub-bullets keep lines short (the 100-char audit rejects long joined
 * ref lists) and "*" is shown as "* (all models)". */
function groupsSection(config: RouterConfig): string {
	const groups = config.groups;
	if (!groups) return "";
	const names = Object.keys(groups);
	if (names.length === 0) return "";
	const lines = ["## Model groups", ""];
	for (const name of names) {
		const refs = groups[name];
		// Array.isArray: hand-built configs can hold anything on the prototype chain.
		if (!Array.isArray(refs)) continue;
		lines.push(`- ${name} (${refs.length} ref${refs.length === 1 ? "" : "s"}):`);
		if (refs.length === 0) {
			lines.push("  - (empty)");
			continue;
		}
		for (const ref of refs) {
			lines.push(`  - ${ref === "*" ? "* (all models)" : ref}`);
		}
	}
	lines.push("");
	return lines.join("\n");
}

/** Multiline config summary (for a notify). */
export function configSummary(config: RouterConfig): string {
	return configSection(config)
		.replace(/^##.*\n+/, "")
		.replace(/\n$/, "");
}

/** Full dashboard report (Markdown) for the /triage command. */
export function formatDashboard(config: RouterConfig): string {
	return [
		"# pi-triage-router dashboard",
		"",
		lineupSection(config),
		costSection(),
		historySection(),
		cacheSection(config),
		configSection(config),
		groupsSection(config),
	]
		.join("\n")
		.replace(/\n{3,}/g, "\n\n");
}
