#!/usr/bin/env node
// Regenerates CHANGELOG.md from git history (conventional commits, newest first).
// Usage: npm run changelog
// Pure Node + git; no dependencies. Tags matching `v<semver>` become version
// sections; commits after the newest tag land under "Unreleased".
import { execFileSync } from "node:child_process";
import { writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const ROOT = dirname(fileURLToPath(import.meta.url)) + "/..";
const REPO = "mraghakhani/pi-triage-router";
const BASE_URL = `https://github.com/${REPO}`;

const run = (cmd, args) => execFileSync(cmd, args, { encoding: "utf8", cwd: ROOT }).trim();

const BUCKET_ORDER = ["Added", "Changed", "Fixed", "Performance", "Documentation", "Other"];
const OTHER = "Other";
const BUCKET = {
	feat: "Added",
	perf: "Performance",
	docs: "Documentation",
	fix: "Fixed",
	refactor: "Changed",
};
const SKIP_TYPES = new Set(["chore", "test", "ci", "merge"]);

/** [{ sha, msg }] for the range prev..tag (tag open-ended when prev omitted). */
export function commitsBetween(prev, tag) {
	const range = prev ? `${prev}..${tag}` : tag;
	// --no-decorate + color.ui=never: ambient git config must not corrupt subject parsing.
	return run("git", ["-c", "color.ui=never", "log", "--no-decorate", "--format=%h %s", range])
		.split("\n")
		.filter(Boolean)
		.map((line) => {
			const m = line.match(/^([0-9a-f]{7,64}) (.+)$/);
			return { sha: m?.[1] ?? "", msg: m?.[2] ?? line };
		});
}

export function bucketsFor(commits) {
	const buckets = new Map();
	const put = (bucket, line) => {
		if (!buckets.has(bucket)) buckets.set(bucket, []);
		buckets.get(bucket).push(line);
	};
	for (const c of commits) {
		const m = c.msg.match(/^(\w+)(?:\(([^)]+)\))?(!)?: (.+)$/);
		if (!m) {
			// Unparsable subject: keep it visible under Other instead of silently dropping it.
			if (!c.msg.startsWith("Merge ")) put(OTHER, `- ${c.msg} (${c.sha})`);
			continue;
		}
		const [, rawType, scope, breaking, subject] = m;
		const type = rawType.toLowerCase();
		if (SKIP_TYPES.has(type)) continue;
		const bucket = BUCKET[type] ?? OTHER;
		let label = scope ? `**${scope}:** ${subject}` : subject;
		if (breaking) label = `**BREAKING:** ${label}`;
		put(bucket, `- ${label} (${c.sha})`);
	}
	return buckets;
}

function renderSection(title, tag, prev, commits = commitsBetween(prev, tag)) {
	// Deterministic: always the ref's commit date (HEAD included), never wall clock.
	const date = run("git", ["log", "-1", "--format=%ad", "--date=short", tag]);
	const compareLink = prev
		? `[${prev}...${tag}](${BASE_URL}/compare/${prev}...${tag})`
		: `[${tag}](${BASE_URL}/releases/tag/${tag})`;
	const buckets = bucketsFor(commits);
	const out = [`## ${title}`, "", `_${date} — ${compareLink}_`, ""];
	if (buckets.size === 0) {
		out.push("_No notable changes._", "");
		return out;
	}
	for (const name of BUCKET_ORDER) {
		const lines = buckets.get(name);
		if (!lines) continue;
		out.push(`### ${name}`, "", ...lines, "");
	}
	return out;
}

function main() {
	const tags = run("git", ["tag", "--sort=version:refname"])
		.split("\n")
		.filter((t) => /^v\d/.test(t));
	if (tags.length === 0) {
		console.error("no tags found (shallow clone?); refusing to dump full history into one section");
		process.exit(1);
	}

	const out = [
		"# Changelog",
		"",
		"All notable changes to this project are documented here, generated from",
		"`git log` by `npm run changelog` (scripts/changelog.mjs).",
		"",
	];

	// Newest first: Unreleased, then tags descending.
	const latest = tags[tags.length - 1];
	const unreleased = commitsBetween(latest, "HEAD");
	if (unreleased.length > 0) {
		out.push(...renderSection("Unreleased", "HEAD", latest, unreleased));
	}
	for (let i = tags.length - 1; i >= 0; i--) {
		out.push(...renderSection(tags[i], tags[i], tags[i - 1]));
	}

	writeFileSync(join(ROOT, "CHANGELOG.md"), out.join("\n"));
	const sections = out.filter((l) => l.startsWith("## ")).length;
	console.log(`wrote CHANGELOG.md (${sections} sections)`);
}

// Only run when executed directly; importing for tests must not rewrite CHANGELOG.md.
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
	main();
}
