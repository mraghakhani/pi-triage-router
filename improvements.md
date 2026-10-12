# Improvements — pi-triage-router

Ordered by value. Items marked ⚑ also close a bug from `bug-report.md` (cross-referenced by B#). No code changed yet; this is the proposal list.

---

## 1. Consolidate `validateConfig` default initialization ⚑ (B17 family)

`validateConfig()` initializes `config.classifier`, `config.benchmarks`, and `config.scoring` **twice** — once inside the `config` object literal, then again immediately before applying the user's section (`classifier` re-init at ~L525, `benchmarks` re-init at ~L585, `scoring` re-init at ~L620). The two copies must be kept in sync by hand; they have already drifted stylistically (`layaTimeoutMs` feeds `classifier.timeoutMs` only in the second copy).

**Status: implemented in this commit.**
**Suggestion:** extract `defaultClassifier()`, `defaultBenchmarks()`, `defaultScoring()` helpers and call each exactly once. Removes ~40 duplicated lines and the drift risk. The `jev` section's separate default literal belongs in `defaultClassifier()` too.

## 2. Harden `parseDataset` at the boundary ⚑ (fixes B2)

**Status: implemented in B2 fix (`1cd4085`).** No code change needed here — `parseDataset` already sanitizes every row via `num()`, drops non-object rows, validates `fetchedAt`/`updatedAt` dates, and `datasetStatus` guards null dataset. `metricValue` has no per-read re-validation to remove.

Trust-boundary principle already used elsewhere in the codebase ("a corrupt cache file is discarded, not trusted") is only applied at the top level. Extend it:

- Validate each `models[id]` row: all metric fields numeric-finite or `null` (reuse the ingest-time `num()`), drop invalid rows instead of trusting the whole file.
- Validate `fetchedAt` parses as a date (also fixes the `age=NaNh` output in `datasetStatus`).
- One small helper `validMetrics(row): boolean` covers it; `metricValue()` then needs no per-read re-validation.

## 3. Fix free-model cost normalization ⚑ (fixes B6)

Two-line change in `scoreEligible` step 3 (see B6): treat `est === 0` as a legitimate best cost. Also stops wrongly flagging `missingUsed` for fully-priced free models. Add one regression test (`$0` model + paid model, expect the free model ranked first on `budget` category).

## 4. Unify proxy resolution, honor `"direct"` everywhere ⚑ (fixes B1)

**Status: implemented in this commit.**

Extract one helper used by both `classifyAndDecide` (classifier path) and `refreshBenchmarks`:

```ts
function proxyFor(explicit: string | null | undefined): string | null // explicit "direct" wins, else settings/env chain
```

Currently the classifier path is right and the benchmark path is wrong for the same documented config value. One helper makes the divergence impossible.

## 5. Return correct state on override/direct routes ⚑ (fixes B4)

`handleRequest()`: replace `request.state ?? { model: composite(entry), summary: "" }` with `{ model: composite(entry), summary: request.state?.summary ?? "" }` in both the override and `direct` branches. Small diff, removes the sticky-model inconsistency. One test: override route → continuation sticks to the override model, not the previous one.

## 6. Fix `cacheDir` validation and slug-date validation ⚑ (fixes B3, B5)

- `cacheDir`: one-line fix (assign only when `typeof === "string"`).
- `resolveSlug` 4-digit branch: reject non-calendar MMDD values (`month 01–12, day 01–31`). Keeps `-2025`-style year suffixes from matching as dates.

## 7. Allow bounded retry of a failed local Laya load (B12)

Replace the permanent per-session `null` with one background retry after a cool-down:

- Keep per-request behavior identical (no waiting, deterministic routing while unsettled).
- After a failed load, schedule a single retry after ~5 min; reset `layaLoadStartedAt` on retry so the bounded-wait logic stays correct.
- Reuses the existing `layaPromise`/`layaLoadSettled` machinery; ~15 lines.

## 8. Improve diagnostics fidelity (B7, B9, B10, B11)

Cheap, isolated polish:

- `argmaxKey`: return `null` on all-zero/empty (mirror `argmaxModel`), keep documented `"standard"`/`"coding"` defaults meaningful.
  **Status: implemented in the B-fixes (B7).**
- `decide()`: report `confidence: null` for `rule === "scored"` picks (or add a `score` field to `RoutingDecision` and log that instead).
  **Status: implemented — confidence: null for scored picks.**
- `loadConfig`: include path in the cache key; cache the default-fallback too.
  **Status: implemented in the B-fixes (B10).**
- Delete `lastAttemptAt`; fix the `callDecisions` JSDoc ("per attempt" timeout); fix `handleRequest` indentation.
  **Status: implemented.**
- Remove the nonexistent `"scripts/"` from `package.json` `"files"`.
  **Status: implemented.**

## 9. Quiet the per-model context-window logging

**Status: implemented in this commit.**

`eligibleModels()` `console.error`s one line per excluded model, on **every** routed request when the context is large (the common case for big sessions). Collapse to one line per request (`excluding N models below ~X tokens: a, b`) or fold into `logDecision` notes. Purely cosmetic for the terminal, but it fires every turn.

## 10. Better cost estimation without a tokenizer (optional)

`estimateCost`'s `chars` fallback uses the **clamped** `currentRequest.length` (≤ ~786 chars) and ignores recent turns + summary that are part of the real prompt — a systematic underestimate exactly when the better sources (`usage`, `contextTokens`) are unavailable. Options, cheapest first:

- Estimate from `prompt.length` (the assembled routing input) instead of the clamped request.
- Or reuse `stateTokens`/`roomTokens` when the classifier tokenizer is available.

Optional because `source: "context"` (Pi's own estimate) usually shadows this path.

## 11. Document restart requirements for `jev` registration

**Status: implemented (documented in README; lazy registration not pursued).**

`default()` decides `jev/auto` registration from the config **at startup**. Editing the config to add/remove `jev` later requires restarting Pi; nothing in the README says so. Either document it, or check `loadConfig().jev` at route time and register lazily (the extension API permitting).

## 12. Test-coverage additions matching the above

- Proxy: `benchmarks.proxyUrl: "direct"` + `https_proxy` set → refresh fetches direct (B1).
- Corrupt cache fixtures: string/undefined metric values, garbage `fetchedAt` (B2).
- Override + continuation state-model continuity (B4).
- `$0`-price model in scoring (B6).
- 4-digit non-date slug suffix (`gpt-5` vs `gpt-5-2025`) (B5).

---

### Deliberately not suggested

- **No new dependencies or abstractions** — the two-module split (`benchmarks` / `openrouter` with a strict no-cycle rule) is clean and worth preserving as-is.
- **No changes to the deterministic rule set** (escalation/downgrade semantics) beyond the optional scored-pick precedence note (B8) — that is a policy decision, not a defect; flag it to the maintainer, don't silently change it.
- **No per-request Laya retry** — the existing bounded-wait design is right; only the session-permanence of a failed load (item 7) is worth touching.
