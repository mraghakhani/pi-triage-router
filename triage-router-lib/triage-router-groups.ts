/**
 * Named model groups ("saved filters") for the triage-router.
 *
 * A group is a named subset of the model catalog. When `activeGroup` is set,
 * classification and every deterministic rule operate on that group's models
 * only - the group is the universe.
 *
 * Refs inside a group's list:
 *  - `"provider/id"`  explicit model (must exist in the catalog; else a problem)
 *  - `"*"`            every model in the catalog (`models[]`, enabled or not)
 *  - `"@otherGroup"`  include another group's members (recursive; cycles and
 *                     unknown names are validation problems)
 *  - `"-provider/id"` exclude a model (order-independent set semantics; a
 *                     no-op when the model was never included)
 *
 * The result of expansion is a SET: order does not matter, duplicates are
 * collapsed, and exclusions always apply after inclusions.
 *
 * Rules honored here:
 *  - Pure functions: no I/O, no registry access, no config loading. Everything
 *    takes its inputs and returns its outputs; callers own the side effects.
 *  - Never fatal: invalid input collects problems; it never throws.
 *  - Warnings about group resolution are printed at most once per process
 *    (`warnOnce`) so a per-route resolution cannot spam the console.
 *
 * This module type-imports shared types from triage-router.ts but imports no
 * runtime values from it (the router imports values from here).
 */
import type { RouterConfig } from "../triage-router.ts";

const LOG_PREFIX = "[pi-triage-router]";

/** Composite model id ("provider/id"), the form refs and members use. */
export type Composite = string;

/** Warnings already emitted (key -> message); bounded by config size, per process. */
const warned = new Set<string>();

/** Print a group-related warning at most once per process. */
export function warnOnce(key: string, message: string): void {
	const dedupe = `${key}: ${message}`;
	if (warned.has(dedupe)) return;
	warned.add(dedupe);
	console.error(`${LOG_PREFIX} ${message}`);
}

/** Expand one ref list into catalog-ordered composite ids. Pure.
 * `groups` maps group names to their raw ref lists (used for `@group` refs);
 * `catalog` is every configured composite (`models[]`); `problems` collects
 * unknown models, unknown groups, and circular references. */
export function expandGroupRefs(
	refs: readonly string[],
	groups: Readonly<Record<string, readonly string[]>>,
	catalog: readonly Composite[],
	problems: string[],
): Composite[] {
	const inCatalog = new Set(catalog);
	const included: Composite[] = [];
	const seen = new Set<Composite>();
	const excluded = new Set<Composite>();

	const includeComposite = (composite: Composite): void => {
		if (seen.has(composite)) return;
		seen.add(composite);
		included.push(composite);
	};

	const visitGroup = (name: string, stack: readonly string[]): void => {
		if (stack.includes(name)) {
			problems.push(`groups: circular reference ${[...stack, name].join(" -> ")}`);
			return;
		}
		const member = groups[name];
		if (!member) {
			problems.push(`groups: unknown group "@${name}"`);
			return;
		}
		visitRefs(member, [...stack, name]);
	};

	const visitRefs = (list: readonly string[], stack: readonly string[]): void => {
		for (const ref of list) {
			if (ref === "*") {
				for (const composite of inCatalog) includeComposite(composite);
			} else if (ref.startsWith("@")) {
				visitGroup(ref.slice(1), stack);
			} else if (ref.startsWith("-")) {
				excluded.add(ref.slice(1));
			} else if (inCatalog.has(ref)) {
				includeComposite(ref);
			} else {
				problems.push(`groups: unknown model "${ref}"`);
			}
		}
	};

	visitRefs(refs, []);
	return included.filter((composite) => !excluded.has(composite));
}

/** The `groups` section of a raw config, type-checked. Returns:
 *  - `{ groups }`  the validated ref lists (problems inside `problems`)
 *  - `null`       the section is absent (no groups feature in use)
 * Never throws; bad shapes degrade to problems + empty result. */
export function validateGroupsSection(
	raw: unknown,
	catalog: readonly Composite[],
	problems: string[],
): Record<string, string[]> | null {
	if (raw === undefined || raw === null) return null;
	if (typeof raw !== "object" || Array.isArray(raw)) {
		problems.push('"groups" must be an object of name -> [refs]; ignoring');
		return {};
	}
	const groups: Record<string, string[]> = {};
	for (const [name, refs] of Object.entries(raw as Record<string, unknown>)) {
		if (!Array.isArray(refs) || refs.some((ref) => typeof ref !== "string" || ref.trim() === "")) {
			problems.push(`groups: "${name}" must be an array of non-empty strings; ignoring`);
			continue;
		}
		groups[name] = (refs as unknown[]).map((ref) => ref as string);
	}
	// Unknown refs and circular references: detect now, at load, so a bad
	// group never silently resolves to the wrong catalog subset later.
	// Recursion reports a referenced group's problems from every referrer;
	// dedupe so each unique problem is reported exactly once.
	const seen = new Set<string>();
	for (const refs of Object.values(groups)) {
		expandGroupRefs(refs, groups, catalog, problems);
	}
	for (let i = problems.length - 1; i >= 0; i--) {
		if (seen.has(problems[i])) problems.splice(i, 1);
		else seen.add(problems[i]);
	}
	return groups;
}

/** Expand a named group from a validated config. Returns its catalog-ordered
 * composites, or null when the group name does not exist. */
export function groupComposites(config: RouterConfig, name: string): Composite[] | null {
	const refs = config.groups?.[name];
	if (!refs) return null;
	return expandGroupRefs(refs, config.groups, config.models.map((entry) => `${entry.provider}/${entry.id}`), []);
}

/** Filter a model list to a group's composites (catalog order preserved). */
export function filterByComposites<T extends { provider: string; id: string }>(
	models: readonly T[],
	composites: readonly Composite[],
): T[] {
	const set = new Set(composites);
	return models.filter((entry) => set.has(`${entry.provider}/${entry.id}`));
}