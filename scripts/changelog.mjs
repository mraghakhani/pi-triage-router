#!/usr/bin/env node
// Regenerates CHANGELOG.md from git history (conventional commits, newest first).
// Usage: npm run changelog
// Pure Node + git; no dependencies. Tags matching `v<semver>` become version
// sections; commits after the newest tag land under "Unreleased".
import { execFileSync } from "node:child_process";
import { writeFileSync } from "node:fs";

const REPO = "mraghakhani/pi-triage-router";
const BASE_URL = `https://github.com/${REPO}`;

const run = (cmd, args) => execFileSync(cmd, args, { encoding: "utf8" }).trim();

const tags = run("git", ["tag", "--sort=version:refname"])
	.split("\n")
	.filter((t) => /^v\d/.test(t));

const BUCKET = {
	feat: "Added",
	feature: "Added",
	fix: "Fixed",
	refactor: "Changed",
	perf: "Performance",
	docs: "Documentation",
};
const OTHER = "Other";

/** [{ sha, msg }] for the range prev..tag (tag open-ended when prev omitted). */
function commitsBetween(prev, tag) {
	const range = prev ? `${prev}..${tag}` : tag;
	return run("git", ["log", "--oneline", range])
		.split("\n")
		.filter(Boolean)
		.map((line) => {
			const m = line.match(/^([0-9a-f]{7,40}) (.+)$/);
			return { sha: m?.[1] ?? "", msg: m?.[2] ?? line };
		});
}

function bucketsFor(commits) {
	const buckets = new Map();
	for (const c of commits) {
		const m = c.msg.match(/^(\w+)(?:\(([^)]+)\))?: (.+)$/);
		if (!m) continue;
		const [, type, scope, subject] = m;
		const bucket = BUCKET[type] ?? OTHER;
		if (bucket === OTHER && ["chore", "test", "ci", "merge"].includes(type)) continue;
		if (!buckets.has(bucket)) buckets.set(bucket, []);
		const label = scope ? `**${scope}:** ${subject}` : subject;
		buckets.get(bucket).push(`- ${label} (${c.sha})`);
	}
	return buckets;
}

function renderSection(title, tag, prev) {
	const date = tag
		? run("git", ["log", "-1", "--format=%ad", "--date=short", tag])
		: new Date().toISOString().slice(0, 10);
	const compareLink = prev
		? `[${prev}...${tag}](${BASE_URL}/compare/${prev}...${tag})`
		: `[${tag}](${BASE_URL}/releases/tag/${tag})`;
	const commits = commitsBetween(prev, tag);
	const buckets = bucketsFor(commits);
	if (buckets.size === 0) return [];
	const out = [`## ${title}`, "", `_${date} — ${compareLink}_`, ""];
	for (const [name, lines] of buckets) {
		out.push(`### ${name}`, "", ...lines, "");
	}
	return out;
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
if (commitsBetween(latest, "HEAD").length > 0) {
	out.push(...renderSection("Unreleased", "HEAD", latest));
}
for (let i = tags.length - 1; i >= 0; i--) {
	out.push(...renderSection(tags[i], tags[i], tags[i - 1]));
}

writeFileSync("CHANGELOG.md", out.join("\n"));
console.log(`wrote CHANGELOG.md (${tags.length + (out[5] ? 1 : 0)} sections)`);