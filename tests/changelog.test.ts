// Unit tests for the changelog parser (scripts/changelog.mjs).
// The parser is untyped JS; import via createRequire to sidestep TS typing.
import { createRequire } from "node:module";
import assert from "node:assert/strict";
import { test } from "node:test";

const require = createRequire(import.meta.url);
const { bucketsFor, commitsBetween } = await import("../scripts/changelog.mjs");
const c = (sha: string, msg: string) => ({ sha, msg });

test("unparsable subject lands in Other instead of being dropped (regression: 348223a)", () => {
	const buckets = bucketsFor([c("348223a", "fix(menu)+tests: review follow-ups")]);
	assert.ok(buckets.has("Other"), "non-matching subject must be kept, not dropped");
	assert.equal(buckets.get("Other")[0], "- fix(menu)+tests: review follow-ups (348223a)");
});

test("matched conventional commit goes to its bucket", () => {
	const buckets = bucketsFor([c("abc1234", "feat(groups): add economy pool")]);
	assert.ok(buckets.has("Added"));
	assert.equal(buckets.get("Added")[0], "- **groups:** add economy pool (abc1234)");
});

test("breaking marker is promoted to a BREAKING label", () => {
	const buckets = bucketsFor([c("abc1234", "feat(config)!: rename activeGroup")]);
	assert.equal(buckets.get("Added")[0], "- **BREAKING:** **config:** rename activeGroup (abc1234)");
});

test("type matching is case-insensitive", () => {
	const buckets = bucketsFor([c("abc1234", "FIX(core): handle empty group")]);
	assert.ok(buckets.has("Fixed"));
	assert.equal(buckets.get("Fixed").length, 1);
});

test("chore/test/ci/merge with valid prefix are skipped", () => {
	const buckets = bucketsFor([
		c("abc1234", "chore: bump version"),
		c("abc2345", "test(core): cover fallback"),
		c("abc3456", "ci: run tests"),
		c("abc4567", "merge: feature branch"),
	]);
	assert.ok(!buckets.has("Other"));
	assert.equal(buckets.size, 0);
});

test("merge-prefixed unparsable subject is skipped", () => {
	const buckets = bucketsFor([c("abc1234", "Merge branch 'main' of github.com:x/y")]);
	assert.equal(buckets.size, 0);
});

test("unknown type falls through to Other", () => {
	const buckets = bucketsFor([c("abc1234", "wip: half-done thing")]);
	assert.equal(buckets.get("Other").length, 1);
});

test("commitsBetween: empty range returns empty list (smoke)", () => {
	// HEAD..HEAD is always empty; exercises the git invocation + line parser
	// without depending on any particular history.
	assert.deepEqual(commitsBetween("HEAD", "HEAD"), []);
});
