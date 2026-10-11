# Bug Report — pi-triage-router

Scope reviewed: `triage-router.ts`, `triage-router-lib/triage-router-benchmarks.ts`, `triage-router-lib/triage-router-openrouter.ts`, `tests/*`, `package.json`, `tsconfig.json`, sample/shipped configs.

Baseline at review time: `tsc --noEmit` clean; `node --test tests/` → 122 tests, 115 pass, 7 skipped, 0 fail. All findings below were confirmed by reading code paths; none are speculative.

Severity: **High** = wrong routing/cost behavior in supported configs · **Medium** = wrong behavior under plausible inputs · **Low** = edge cases, diagnostics, dead code.

---

## B1 · Medium — Benchmark refresh ignores `proxyUrl: "direct"` ✅ fixed

**Location:** `triage-router-lib/triage-router-benchmarks.ts`, `refreshBenchmarks()` (proxy resolution line).

```ts
const proxyUrl = normalizeProxy(config.proxyUrl) ?? resolveProxyUrl(null) ?? env("https_proxy") ?? env("HTTPS_PROXY") ?? null;
```

`normalizeProxy("direct")` returns `null` ("unspecified"), so `"direct"` falls through the `??` chain into `resolveProxyUrl(null)` → Pi's `settings.json` `httpProxy` → `https_proxy`/`HTTPS_PROXY` env. Result: a user who sets `benchmarks.proxyUrl: "direct"` to force no proxy still gets benchmark refresh traffic pushed through their proxy.

The classifier path handles this correctly: `resolveProxyUrl(cfg.proxyUrl)` short-circuits on any non-null explicit value, treating `"direct"` as "never proxy". The two paths disagree on the same documented meaning of `"direct"` ("never proxy", per the config-validation error message in `validateConfig`).

**Fix:** make `null` (unspecified) and `"direct"` distinguishable, e.g. a sentinel, or:

```ts
const explicit = config.proxyUrl !== undefined && config.proxyUrl !== null ? normalizeProxy(config.proxyUrl) : undefined;
const proxyUrl = explicit !== undefined ? explicit : (resolveProxyUrl(null) ?? env("https_proxy") ?? env("HTTPS_PROXY") ?? null);
```

Note `config.proxyUrl` may be `null` by default (unspecified) while `"direct"` must mean never — the current expression cannot tell them apart.

---

## B2 · Medium — Corrupt benchmark cache poisons scores with `NaN` ✅ fixed

**Location:** `triage-router-lib/triage-router-benchmarks.ts` — `parseDataset()` (L480) and `scoreEligible()`/`metricValue` (L~800-860).

`parseDataset` validates only that the root has `fetchedAt: string`, `source: string`, `models: object`. It does **not** validate individual model rows. Then in `scoreEligible`:

```ts
const raw = metricValue(entry, metric);   // returns m[metric] unvalidated
if (raw !== null) values.set(entry.id, raw);  // undefined / string passes this check
```

`m[metric]` can be `undefined`, a string, or `NaN` for any hand-edited, truncated, or older-schema cache file (`triage-router.benchmarks.json` ships with the package and is also user-editable). `normalize()` then computes `Math.min(...[undefined])` → `NaN`, quality becomes `NaN`, the final sort comparator returns `NaN` (interpreted as 0), and `decide()` can route to an effectively unranked model.

**Fix (either suffices):**
- In `parseDataset`, validate each `models[id]` entry's numeric fields with the same `num()` helper used at ingest; drop invalid rows.
- Or in `metricValue`, re-validate: `const raw = m[metric]; return typeof raw === "number" && Number.isFinite(raw) ? raw : null;`

Related, same root cause: `datasetStatus()` does `Date.parse(dataset.fetchedAt)` with no guard → prints `age=NaNh` for a corrupt cache.

---

## B3 · Low — Invalid `cacheDir` type is assigned despite "ignoring" message ❌ false positive

**Correction (during fix pass):** this finding was wrong — the original code is correct. `validateConfig()` (L448-452) only assigns `config.cacheDir` in the `else` branch, which is reached only for `undefined` / `null` / `string` values; any other type (e.g. `42`) hits the `if` branch, logs the problem, and **does not assign**. The message *"ignoring"* is accurate. No code change made; no commit.

---

## B4 · Low — Override / direct routes keep a stale `state.model`

**Location:** `triage-router.ts` L1775 (override path) and the `reason === "direct"` path (~L1800), `handleRequest()`.

```ts
const state = request.state ?? { model: composite(entry), summary: "" };
...
return buildRoute(request, ctx, entry, state);
```

When `request.state` already exists (model `X`) and the override (or direct-fallback) routes to a different entry `Y`, the returned state still says `model: X`. The extension's own sticky rule ("continuations and retries stay on the model that handled the turn") then sticks the *next* continuation to `X` — not the model that actually handled the turn. Same pattern on the `direct` path.

**Fix:** always return the state of the model that actually served the request, preserving the summary:

```ts
{ model: composite(entry), summary: request.state?.summary ?? "" }
```

---

## B5 · Low — `resolveSlug` matches 4-digit suffixes that are not MMDD dates

**Location:** `triage-router-lib/triage-router-benchmarks.ts` — `DATED_SLUG_RE = /^(.*)-(\d{8}|\d{4})$/`, `resolveSlug()`, `slugDateRank()`.

The 4-digit branch is meant to be `-MMDD`, but any 4 digits match, e.g. config id `gpt-5` matches permaslug `gpt-5-2025` (year suffix) or `model-1234`, as long as a dataset year is known. `slugDateRank()` then treats `2025` as `MMDD` (month 20 — invalid date) for recency ordering. Consequence: a model can be served another release's benchmark rows via a coincidental numeric suffix.

**Fix:** validate the 4-digit branch as a real month/day (`month 01–12, day 01–31`), or require `datasetYear`-anchored `-MMDD` semantics explicitly. Low real-world likelihood, but the fix is two lines.

---

## B6 · Low — Free (0-cost) models are excluded from cost normalization

**Location:** `triage-router-lib/triage-router-benchmarks.ts`, `scoreEligible()` step 3 (L~851).

```ts
if (est !== null && est > 0) { rawCost.set(key, est); }
...
const cost = costs.get(key) ?? neutralCost;
```

`estimatedCost === 0` (free tier, `$0` price, or internal endpoint) is dropped from the cost map — evidently to avoid `minCost/est` → `0/0 = NaN` when the cheapest model is free. But the side effect is that a **free model receives `neutralCost` instead of cost 1.0**, so the cost term systematically under-ranks exactly the models a cost-aware policy should favor most. The exclusion also silently marks `missingUsed = true` for a model that *has* complete price data.

**Fix:** handle `minCost === 0` explicitly: `cost = est === 0 ? 1.0 : minCost / est` (paid models then score 0 on the cost axis, which is the intended monotonic meaning). Add a regression test with a `$0` model.

---

## B7 · Info — `argmaxKey` all-zero/empty probabilities return the first key, defeating the documented defaults

**Location:** `triage-router.ts`, `argmaxKey()` and its call sites in `decide()`.

`decide()` writes `argmaxKey(...) ?? "standard"` and `?? "coding"`, implying empty/all-zero probabilities should default to those values. In practice `argmaxKey` **never returns null**: with all `p = 0` the first key wins (`p > bestP` with `bestP = -1`). Empty classifier output therefore yields `category = "question"` (first in `CATEGORIES`), not the intended `"coding"` default. Deterministic and harmless for correctness, but the `?? "coding"` branches are dead code and the logged category for a failed classification is misleading.

**Fix:** return `null` when `bestP <= 0` (mirroring `argmaxModel`'s "no signal → null" rule), or drop the dead `??` defaults.

---

## B8 · Info — Escalation rule can override the benchmark-scored pick with the strongest model

**Location:** `triage-router.ts`, `decide()` escalation branch.

With `scoring.enabled`, the scored pick already applies `complexityQualityDelta` (+0.15 quality weight for complex work). The escalation rule then still fires when the scored pick happens to be the cheapest entry and complexity is `complex` + category is escalate-able — replacing a deliberate benchmark-driven choice with the priciest eligible model. Deterministic and safe, but it partially contradicts the scoring policy it just ran. Worth either suppressing escalation when `scored === true`, or documenting the precedence explicitly.

---

## B9 · Info — `confidence` is misleading for scored picks

**Location:** `triage-router.ts` L1257.

When the pick came from benchmark scoring (`rule: "scored"`), `confidence` still reports the classifier's probability for that model — often near 0 or absent (`null`) — implying Laya chose it. Diagnostic-only; suggest reporting `null` (or a separate `score` field) when `scored`.

---

## B10 · Low — `loadConfig` cache is keyed by mtime only

**Location:** `triage-router.ts`, `loadConfig()`.

`cachedConfig?.mtimeMs === mtimeMs` ignores the path. Any second `loadConfig(otherPath)` call in the same process whose file happens to share an mtime with the cached one returns the wrong config. Today all call sites use the default path, so impact is latent (tests that pass an explicit path are exposed). Also: when the config file is unreadable, `validateConfig({})` is re-run on **every** route (not cached) — negligible but unbounded.

**Fix:** include `path` in the cache key; cache the default-fallback result too.

---

## B11 · Info — Dead code and doc mismatches (cosmetic)

- `triage-router-benchmarks.ts`: `lastAttemptAt` is written (`scheduleRefresh`) and reset for tests but **never read**.
- `triage-router-openrouter.ts`, `callDecisions()` JSDoc: *"Total attempt budget in ms"* — the implementation applies `timeoutMs` **per attempt** (2 attempts → up to ~2× timeout). Doc should say "per attempt".
- `triage-router.ts`, `handleRequest()` body is indented one extra level (formatting artifact; no behavior change).
- `package.json` `"files"` lists `"scripts/"`, which does not exist in the repo — npm silently ignores it.

---

## B12 · Low (design) — A failed first Laya load is permanent for the whole session

**Location:** `triage-router.ts`, `getLaya()`.

The load promise settles to `null` on failure and `getLaya` returns it for the rest of the process ("do not retry this session"). A transient failure (network hiccup during the ~2 GB first-run weight download, disk-full during extraction) silently degrades every subsequent request to deterministic routing until the user restarts Pi. Intentional per the comment (it prevents per-request retry spam), but the cure is total.

**Fix idea:** keep the "no retry per request" guarantee but allow one background retry after a cool-down (e.g. 5 min), or a single retry on next session start; the existing `layaLoadWarned` flag already handles log spam.

---

## B13 · Low — `mergeOpenrouter` research metric is an order-sensitive running average

**Location:** `triage-router-lib/triage-router-benchmarks.ts`, `mergeOpenrouter()`.

Multiple `search_*` benchmark rows for one model are blended with `(target.research + mean) / 2` per iteration, so the weighting of each `search_*` eval depends on row arrival order rather than an equal or documented weighting. Deterministic for a fixed API response, but arbitrary and non-obvious. Collect all `search_*` means first, then average once.

---

## What holds up well (checked, no findings)

- Credential handling: keys read from env / `~/.pi/agent/auth.json`, never logged or embedded in errors; `RemoteClassifierError` messages are credential-free (verified error paths).
- Proxy loopback bypass (`makeFetch`) correctly excludes localhost targets.
- Remote failure handling: never silently switches classifier provider; `failurePolicy=deterministic` is honored on both load and call failures; bounded retries only on 429/5xx/network.
- Config hot-reload via mtime; atomic cache write (tmp + rename); last-known-good dataset survives refresh failures.
- Context budgeting: current request prioritized, explicit truncation markers, exact tokenizer fit loop with bounded iterations.
- Deterministic decision rules are well-tested (115 passing tests cover escalation, downgrade, fallback, override, sticky, cost estimation).
