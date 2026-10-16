# pi-triage-router

A Pi extension that registers the virtual models `laya/auto` and `jev/auto`. On
every new user message, a classifier - **local Laya by default** (Apache-2.0 ONNX System 1
decision model, free, in-process), optionally the **remote Jev Decisions API**
via an explicit config switch - picks the execution model from a
configurable catalog, using the current request **plus conversation context**.
Model selection can additionally be driven by **real OpenRouter benchmark
data** through a deterministic scoring policy (config `scoring.enabled`).

```sh
pi --model laya/auto
```

## Install

```sh
pi install npm:pi-triage-router
```

That is the whole install: pi pulls the package, installs its runtime
dependencies (`@receptron/laya`, `undici`) and registers `triage-router.ts`.
No setup script, no symlink. Restart pi, then confirm it loads:

```sh
pi --model laya/auto -p "reply ok"
```

Remove it with `pi remove npm:pi-triage-router`.

#### If install fails on `linux/x64`

`onnxruntime-node` (a transitive dependency of `@receptron/laya`) has a
postinstall that downloads CUDA/GPU provider libraries from nuget.org — and
`linux/x64` is the **only** platform it does this for. It needs direct internet
access, so on a restricted or proxied network the download dies with
`socket hang up` and aborts the whole install.

That is safe to skip: the CPU binaries every platform needs ship inside the
npm tarball, and Laya runs CPU inference. Set the flag via the environment:

```sh
ONNXRUNTIME_NODE_INSTALL=skip pi install npm:pi-triage-router
```

If you are installing from the git clone (dev build), the repo ships a root
`.npmrc` with `onnxruntime-node-install=skip` that npm reads inside the
cloned directory — the download self-skips without the env var.

### Where files live

Sources land in different directories (`npm:` under
`~/.pi/agent/npm/node_modules/...`, `git:` under `~/.pi/agent/git/...`). The
rows below are for the **npm** install:

| What | Path | Edit it? |
|---|---|---|
| Config | `~/.pi/agent/triage-router.json` | **yes - the only file you edit** |
| Bundled defaults | `~/.pi/agent/npm/node_modules/pi-triage-router/triage-router.json` | no - `pi update` replaces it |
| Benchmark cache | `~/.pi/agent/triage-router.benchmarks.json` | no - regenerated automatically |
| Price cache | `~/.pi/agent/triage-router.prices.json` | no - regenerated automatically |
| Laya weights (~2 GB) | `~/.cache/receptron-laya` | no - downloaded on first use |

Resolution order is `TRIAGE_ROUTER_CONFIG` > `~/.pi/agent/triage-router.json` >
bundled `triage-router.json`, so your edits always survive `pi update`. To
customise:

```sh
cp ~/.pi/agent/npm/node_modules/pi-triage-router/triage-router.json \
   ~/.pi/agent/triage-router.json
$EDITOR ~/.pi/agent/triage-router.json
```

### Credentials

`laya/auto` is fully local and free - nothing to configure. `jev/auto` needs
`LITELLM_API_KEY` in the environment, or a `litellm.key` entry in
`~/.pi/agent/auth.json` (which also carries the litellm `baseUrl`). Benchmark
refreshes work keyless against OpenRouter.
If your network needs a proxy, set `httpProxy` in `~/.pi/agent/settings.json`.

## How routing works

1. **Every new user message** (`reason: "user"`) is classified by Laya in one
   local forward pass (~75-200 ms warm): which model, how complex, what kind of
   work.
2. Tool follow-ups (`continuation`) and retries stay **sticky** on the model
   that handled the turn (no re-escalation loops, no prompt-cache thrash).
   If the sticky model vanished from the registry/config, it re-classifies.
3. Requests outside the agent loop (`direct`: compaction summaries, extension
   calls) always route to the cheapest eligible model.
4. The chosen model id + a rolling conversation summary are stored as router
   state on the session branch (survives compaction, session trees, forks).

The decision is then validated and refined deterministically (see
"Decision rules" below). The log line for every decision goes to stderr:

```
[pi-triage-router] route=primary model=litellm/z-ai/glm-5.3 cat=infra cplx=complex conf=0.8 laya_ms=142 state_tokens=212/316 ctx_chars=(recent=210 summary=95 request=64) est_in=42000(context) est_cost_in=0.00168 est_out=800 est_cost_out=0.00384 last_cost=0.12(actual, previous request) note="-"
```

## How Laya sees conversation context

Laya's whole sequence is capped at 512 tokens; each question's head
(instructions + option criteria) is capped at ~195 tokens, leaving a measured
**state room of ~316 tokens** (`max_len - head_max_len - 4`, refined per
instance from Laya's own config). The state Laya sees is the JSON-dumped
`{role, prompt}` object, and Laya truncates it **from the tail** beyond the
room - which would silently cut the `[Current request]` section placed last.
The router therefore bounds the entire assembled prompt to that room: with
the real tokenizer available (the loaded Laya instance), prompts are fit by
exact token counts (shrink-then-verify, oldest context dropped first);
without one (unit tests), a conservative 3 chars/token allocation is used.
`contextBudgetChars` (default 1800) is an upper bound on the context
sections - the room is the binding limit. Allocation priority: current
request, then recent turns, then summary:

```
[Recent conversation]
<last N short turns, newest last, each <= 250 chars>        (recentTurns, default 4)
[Conversation summary]
<rolling summary, <= 600 chars>
[Current request - base the decision primarily on this]
<latest user message, verbatim, hard cap 6000 chars>
```

- **Current request first**: truncated only to fit the room (hard cap 6000
  chars), always with an explicit `[current request truncated]` marker;
  context sections are dropped before the request is cut further.
- **Rolling summary**: refreshed FIRST (the just-aged turns roll in before
  the prompt is built - no one-turn blind spot), maintained per user message
  in router state (JSON, per session branch). When the router state is empty
  (fresh session, or after compaction), Pi's own compaction/branch summary
  from the session log seeds it - a trustworthy summary Pi already computed.
  The prompt and the stored state carry the same summary text.
- **Recency bias**: the summary keeps the newest ~600 chars. Old topics fade
  out as new turns roll in, so stale context cannot dominate routing.
- **No duplication**: turns inside the recent window never roll into the
  summary until they leave it.
- **Never fabricated**: if session data is unavailable, only the request is
  used; nothing is invented.
- Compaction summarization is **never** an extra LLM call: context assembly is
  deterministic string work done locally, in-process.

## Decision rules (deterministic, after classification)

- Eligible models must be `enabled !== false` **and** present in the live
  model registry (`ctx.modelRegistry.find`). Laya picks are validated against
  this set; picks outside it are rejected -> fallback policy.
- **Escalation**: `complex` work in a demanding category
  (`coding|debugging|reasoning|infra`) that lands on the cheapest eligible
  model is escalated to the **strongest eligible model** - independent of the
  fallback direction, so `fallback: "cheapest"` cannot neuter escalation.
- **Context-window eligibility**: a model is additionally excluded when its
  registry `contextWindow` cannot hold the upcoming request - estimated as
  the current conversation tokens (Pi's `getContextUsage`), plus the new
  request (chars/4, a documented approximation - no tokenizer exists for the
  execution models), plus a fixed 16K output/safety reserve. Unknown windows
  are never treated as limits; if nothing fits, every model stays eligible
  (logged) - routing degrades rather than throwing. Sticky models that
  outgrow their window are re-classified.
- **Budget**: `budget`-tagged work never pays above the cheapest eligible
  model.
- Empty/zero/all-invalid probabilities (Laya unavailable, timeout, garbage)
  count as **no signal** -> fallback policy. Laya's self-reported confidence
  is logged but never trusted for decisions.

## Fallback policy (in order)

1. `"default": "provider/id"` in the config, if it is eligible.
2. `"fallback": "strongest"` -> highest `priceIn` eligible model (file order
   last when prices are missing); `"cheapest"` -> lowest `priceIn` (file order
   first).
3. No eligible models at all -> routing throws an observable error
   (`[pi-triage-router] no eligible models...`); nothing is silently routed.

A routing failure never falls back to a **paid classifier**. Laya failing
just means the deterministic policy decides.

## Manual override

- `"override": "provider/id"` in the config routes everything there while
  set (validated against the catalog; visible in diagnostics as
  `route=override`). Remove the key to resume automatic routing. Nothing is
  persisted.
- Pi's native `/model <provider>/<id>` switches the session away from
  `laya/auto` entirely, as usual.

## Configuration (`triage-router.json`)

Hot-reloaded (re-read when the file mtime changes; no Pi restart needed).
Prices are **USD per 1M tokens**; `priceIn: 0` is a legitimate free model,
an absent price means unknown. Unknown keys and bad values are reported once

**Auto-updated prices** (`prices` section, default on): the public OpenRouter
catalog (`/api/v1/models`, keyless) refreshes in the background at most once
every 24h into `~/.pi/agent/triage-router.prices.json` - same pattern as the
benchmark cache: routing only reads the local file, a refresh never blocks a
route, and failures keep the last known good cache. A bundled snapshot
(`triage-router.prices.json`, fetched at release time) serves reads until the
first background refresh writes the agent-dir cache, so a fresh install has
prices without any network wait. Lookup precedence:
live registry cost -> cached catalog price -> config `priceIn`/`priceOut`.
Models absent from the catalog (renamed/withdrawn upstream) fall back to the
config prices. Set `"prices": { "enabled": false }` to pin config prices only.
per (re)load and skipped; the router keeps working with the valid entries.

```jsonc
{
  "models": [
    // Order is documentation; routing uses prices. Fewer than ~20 entries
    // per choice question is what Laya recommends.
    { "provider": "litellm", "id": "deepseek/deepseek-v4-flash",
      "description": "Simple questions, explanations, small fixes, and routine code edits.",
      "priceIn": 0.01, "priceOut": 1.28 },          // enabled by default
    { "provider": "litellm", "id": "moonshotai/kimi-k3",
      "description": "Top-tier coding quality when output cost matters less; expensive output.",
      "priceIn": 0.50, "priceOut": 12.00 }         // litellm only; all entries are litellm
  ],
  "default": null,           // "provider/id": preferred fallback model
  "override": null,          // "provider/id": manual override while set
  "fallback": "cheapest",    // or "strongest"
  "cacheDir": null,          // ONNX weight cache; default ~/.cache/receptron-laya
  "contextBudgetChars": 1800, // total context text sent to Laya
  "recentTurns": 4,          // short turns included verbatim-ish
  "layaTimeoutMs": 5000,     // legacy: feeds classifier.timeoutMs

  "classifier": {            // backend for the laya/auto entry (default: local Laya)
    "provider": "local",     // "local" | "openrouter" | "litellm" (remote backends)
    "timeoutMs": 5000,       // local inference / each remote attempt
    "maxAttempts": 1,        // remote attempts total (first try + retries); 1..10
    "retryDelayMs": 0,       // gap between remote attempts (ms)
    "maxInputTokens": 512,   // remote state budget (tokens, conservative 3 chars/token)
    "localPrompt": "...",    // role text for local Laya (default: ROLE_TEXT)
    "remotePrompt": "...",   // role text for the remote classifier (longer allowed)
    "failurePolicy": "deterministic", // or "error"
    "apiKeyEnv": "OPENROUTER_API_KEY", // remote backends only; ignored for local
    "proxyUrl": null         // default: settings.json httpProxy, then env; "direct" = never
  },

  "jev": {                   // backend for the jev/auto entry; delete the section to hide it
    "provider": "litellm",   // "openrouter" | "litellm"
    "model": "typesafe/jev-1.13",
    "protocol": "systemone", // "systemone" (LiteLLM) | "decisions" (OpenRouter)
    "timeoutMs": 5000,
    "maxAttempts": 1,        // remote attempts total; retries only on 429/5xx/network
    "retryDelayMs": 0,
    "maxInputTokens": 28672, // Jev state room: 32k (state+question) limit, 28k leaves 4k for questions
    "failurePolicy": "deterministic",
    "apiKeyEnv": "LITELLM_API_KEY" // env first, then auth.json["litellm"].key; base URL from auth.json["litellm"]
  },

  "benchmarks": {            // OpenRouter benchmark cache
    "enabled": true,
    "source": "openrouter",  // unified endpoint (key) | "openrouter-models" (public, keyless)
    "refreshIntervalMs": 86400000,  // background refresh at most once a day
    "staleAfterMs": 2592000000      // 30 days: stale rows per scoring.stalePolicy
  },

  "prices": {               // OpenRouter price cache (public catalog, keyless)
    "enabled": true,
    "refreshIntervalMs": 86400000   // background refresh at most once a day
  },

  "scoring": {               // benchmark-and-price selection (default: off)
    "enabled": true,         // classifier still decides category+complexity
    "missingPolicy": "neutral",  // or "penalize" (with missingPenalty)
    "stalePolicy": "use",    // or "missing"
    "expectedInputTokens": 4000,  // for the estimated task cost
    "expectedOutputTokens": 1000,
    "complexityQualityDelta": 0.15,
    "classifierWeight": 0.35,  // share of the score from the classifier's model answer
    "weights": { "coding": { "quality": 0.65, "cost": 0.35 } } // per category
  }
}
```

`description` is the criterion Laya scores - one short sentence (<= 200
chars, task-fit first; the validator flags longer ones) so all six criteria
fit Laya's ~195-token question head. Pricing is **never** appended text: the
old auto-appended price line sat last in each criterion and was the first
casualty of head truncation. Price is handled by deterministic rules instead:
`budget`-tagged work -> cheapest eligible, escalation -> strongest eligible,
no-signal fallback -> cheapest/strongest by `priceIn`.

## Classifier backends (`classifier` section)

The classifier model is **conceptually separate from the execution models**:
the model that classifies a request is never required to be one of the
models that executes it.

### Two router entries, one config file

The extension registers **two virtual models** visible in `/models`:

| entry | classifier backend | registered when |
|---|---|---|
| `laya/auto` | the `classifier` section (local Laya by default) | always |
| `jev/auto` | the `jev` section (remote Jev: openrouter or litellm) | the config has a `jev` section |

Registration is decided once at startup: adding or removing the `jev` section in the config requires restarting Pi for `jev/auto` to appear in or disappear from `/models`.

Both entries run the **identical** pipeline - same model catalog, context
system, decision rules, benchmark scoring, eligibility checks, logs - only
the classifier backend differs. Pick the entry per session (`pi --model
laya/auto` or `pi --model jev/auto`, or switch in `/models`); switching is
always explicit and visible, never silent.

### Backends

| | `provider: "local"` (default) | `provider: "openrouter"` | `provider: "litellm"` |
|---|---|---|---|
| Engine | Laya ONNX in-process | OpenRouter **Decisions API** with Jev (`typesafe/jev-1.13`) | TypeSafe **System One** protocol (`/systemone`) through a LiteLLM proxy |
| Endpoint | - | `openrouter.ai/api/alpha/decisions` | `{baseUrl}/systemone` (verified live, incl. `/v1/systemone`); `baseUrl` from the auth file |
| Cost | free, no API key | input-priced per call; `usage.cost` reported per call and logged as actual | same - LiteLLM fronts TypeSafe/OpenRouter and reports actual cost |
| Proxy | none | settings.json `httpProxy` / env | works direct and via proxy (loopback hosts are never proxied) |

```jsonc
"jev": {                       // the jev/auto entry (delete the section to hide it)
  "provider": "litellm",       // "openrouter" | "litellm" (remote only; local stays with laya/auto)
  "model": "typesafe/jev-1.13",
  "protocol": "systemone",    // "systemone" (LiteLLM/OpenRouter) | "decisions" (OpenRouter)
  "timeoutMs": 5000, "maxInputTokens": 28672,
  "failurePolicy": "deterministic",
  "apiKeyEnv": "LITELLM_API_KEY",   // optional; see credentials below
  "proxyUrl": null            // null = settings.json httpProxy -> env; "direct" = never proxy
}
```

A complete, loadable reference lives in **`triage-router.sample.json`** (validates
with zero problems; your current `triage-router.json` is that file plus your live
catalog choices).

**Credentials** (remote backends): env var first (`apiKeyEnv`), then Pi's auth
file (`~/.pi/agent/auth.json`, keyed by provider name - `"openrouter"` or
`"litellm"`). A selected remote backend without credentials is a loud,
non-fatal startup problem and follows `failurePolicy` at runtime.

**Base URL** (remote backends): also read from the auth file's provider entry -
never from this plugin's config, so private endpoints stay out of shared config
files. Set `"baseUrl"` on the entry (or any `env.<NAME>_BASE_URL` string, e.g.
`"env": { "LITELLM_BASE_URL": "https://your-litellm" }`); without one, the
OpenRouter defaults for the protocol are used.

**Proxy**: explicit `"direct"` = never proxy; otherwise settings.json
`httpProxy` (e.g. `http://127.0.0.1:10808`), then env proxies. Loopback
targets (local LiteLLM, llama.cpp) are never proxied.

Rules that hold for every remote backend:

- Switching back to local Laya: set the entry's `"provider": "local"`
  (`classifier` section only) or delete the section - existing configs keep
  their old behavior.
- **Shared contract**: both answer the identical `questions` object with
  `answers.<name>.probabilities`; the schema, category definitions,
  eligibility rules, deterministic post-processing, and evaluation fixtures
  are one implementation, not two.
- **Separate prompts**: `localPrompt` / `remotePrompt` are the role text of
  the classification state. The local prompt stays within the measured ~316
  token state room; the remote prompt may be more detailed (bounded by
  `maxInputTokens`). Both feed the same validated decision schema.
- **Structured output is validated regardless of provider**: probabilities
  must be finite numbers in [0,1] or they are dropped; a structurally
  invalid payload is a classification failure, never a decision.
- **Bounded retries and timeouts**: `maxAttempts` remote attempts total
  (default 1 = no retries; configurable 1..10), only on 429/5xx /
  network errors, `retryDelayMs` between attempts (default 0), each attempt
  inside `timeoutMs`. The wall-clock budget scales with the attempt count.
- **No silent provider switching.** A failing backend follows
  `failurePolicy`: `"deterministic"` (default) = the documented deterministic
  fallback policy picks the execution model with no classification signal;
  `"error"` = the routing error surfaces. Neither path switches classifiers.
- **Never selects disabled/unknown models**: the questions only contain the
  eligible composites, and the argmax/score runs over the eligible set.
- **No tools**: classification is a typed question/answer call; it cannot
  execute commands.
- Diagnostics log the active backend: `clf=local|openrouter model=... ms=...
  in=... out=... cost=...(actual|none|unavailable)`.
- **Privacy**: with the default local backend, conversation context never
  leaves the machine. Only an explicit `provider: "openrouter"` sends the
  bounded routing input (current request + compact summary/recent turns,
  capped by `maxInputTokens`) to OpenRouter for classification - and nothing
  else: summaries are not stored remotely, and no prompt, summary, or
  conversation content is ever logged (only ids, sizes, latencies, rule
  names, and cost figures). Credentials live in the environment, are sent
  only as the `Authorization` header, and never appear in logs or errors.

### Jev: what was verified, and the two modes

Verified against official OpenRouter sources (2026-10-09):

1. **`typesafe/jev-1.13` (Jev, the decision model)** - NOT an LLM: send
   state + typed questions to `POST /api/alpha/decisions`, get typed answers
   with probabilities plus `usage` (tokens + actual USD cost). This is the
   remote classifier backend implemented here. Docs:
   `/docs/guides/community/jev-tutorial`.
2. **`typesafe/jev-router` (Jev Router)** - a model-*selection* router that
   also **executes**: it picks a model + reasoning effort and that model
   generates the response; the selection is visible in the response `model`
   field; optional plugin params (`cost_tier`, `allowed_models`,
   `excluded_models`); billed at the selected model's rate (pricing `-1`).
   This is a separate **execution mode**, not a classifier: it is deliberately
   NOT wired through this router's model-selection pipeline. Use it directly:
   `pi --model openrouter/typesafe/jev-router` (or add it as a catalog entry
   you route TO, not THROUGH). No unsupported integration was assumed.

## Benchmark data (`benchmarks` section)

Sources (documented machine-readable endpoints, all verified live):

- `GET https://openrouter.ai/api/v1/benchmarks` (auth: any OpenRouter key;
  30 req/min, 500 req/day) - unified rows from **Artificial Analysis**
  (intelligence / coding / agentic indices), **Design Arena**
  (`codecategories` ELO), and OpenRouter's own evals: `gpqa_diamond`,
  `tau_bench_verified_airline` (accuracy **and** `avg_cost_per_task` - real
  cost-per-success where measured), `search_*` retrieval evals. Each row
  carries `last_run_timestamp`; the response carries `meta.as_of`.
- `GET https://openrouter.ai/api/v1/models` (public, keyless) - per-model AA
  indices + pricing; narrower fallback (`source: "openrouter-models"`).

Behavior:

- **Cached locally** in `~/.pi/agent/triage-router.benchmarks.json` (the Pi
  agent dir, `PI_CODING_AGENT_DIR`-aware); routing NEVER fetches per decision.
  The package ships a bundled snapshot (fetched 2026-10-09, `as_of` preserved
  inside) that serves reads until the first background refresh writes the
  agent-dir cache - so `pi update` can never wipe your benchmark data.
  Override the location with `benchmarks.cachePath`.
- Config candidates (`TRIAGE_ROUTER_CONFIG` env > `~/.pi/agent/triage-router.json` >
  legacy extensions-dir path > bundled defaults) honor `PI_CODING_AGENT_DIR`.
- **Background refresh** at most once per `refreshIntervalMs`, fire-and-forget
  (routing never waits on it). Failures (timeout, 429, malformed body,
  unreachable) keep the **last known good** dataset byte-for-byte.
- **Explicit refresh**: `node --input-type=module -e 'const {refreshNow}=await
  import("./triage-router-benchmarks.ts"); const {validateConfig}=await
  import("./triage-router.ts"); ...'` or simply delete the cache file.
- **Freshness + coverage are observable**: each scored route logs
  `bench=covered/total age=Hh[ stale] as_of=...`.
- **Identity discipline**: rows attach to a model by exact permaslug or by
  stripping a trailing release date (`-YYYYMMDD`/`-MMDD`); newest release
  wins; same-release display variants are averaged (`variants` recorded).
  Family prefixes never match (`glm-5.3-flash` cannot lend scores to
  `glm-5.3`). `match`, `variants`, `sources`, `updatedAt` stay attached.
  Missing values stay `null` - never invented, never borrowed.

## Scoring policy (`scoring` section, default off)

When `scoring.enabled` is true, the classifier answers **category +
complexity**, and a deterministic policy picks the execution model from
benchmark data + prices, blended with the classifier's own **model** answer
when `classifierWeight` > 0:

```
quality_m  = weighted blend of normalized, task-relevant metrics   (in [0,1])
cost_m     = minEstimatedCost / estimatedCost_m                     (in (0,1])
support_m  = classifier P(model_m) / max over eligible                (in [0,1])
score_m    = wQ*quality_m + wC*cost_m + wK*support_m   (wQ+wC+wK = 1)
estimatedCost = expectedInputTokens/1M * priceIn + expectedOutputTokens/1M * priceOut   [estimate]
```

- **Normalization first**: each metric is min-max normalized across the
  eligible set (all-equal -> 0.5 for everyone); raw percentages, ELOs, and
  USD are never averaged directly. Blending happens between normalized
  values only.
- **Task-specific weights** (`weights.<category> = {quality, cost}`, defaults):

  | category | quality | cost | primary quality metrics |
  |---|---|---|---|
  | coding | 0.65 | 0.35 | AA coding (0.7) + arena code ELO (0.3) |
  | debugging | 0.75 | 0.25 | AA coding + AA intelligence |
  | infra | 0.60 | 0.40 | `capabilities.infra` (internal) + coding |
  | reasoning | 0.80 | 0.20 | gpqa accuracy + AA intelligence |
  | long_horizon | 0.70 | 0.30 | AA agentic + tau task success + coding |
  | research | 0.60 | 0.40 | search-eval accuracy |
  | question | 0.30 | 0.70 | AA intelligence |
  | budget | 0.20 | 0.80 | AA intelligence |

  `complexityQualityDelta` (0.15): `complex` shifts weight to quality,
  `standard` shifts to cost (clamped 0.05..0.95).
- **Missing data is never zero**: a model without a metric gets the neutral
  treatment (mean of available quality; or `neutral * (1 - missingPenalty *
  (1 - coverage))` under `missingPolicy: "penalize"`). Partial coverage is
  renormalized over the metrics the model actually has and reported in the
  decision note. If NO model has measured quality, scoring abstains and the
  classifier pick (then the deterministic fallback) decides - explicitly
  noted, never silent.
- **Staleness**: rows older than `staleAfterMs` are used-but-flagged
  (`stalePolicy: "use"`, default) or dropped (`"missing"`).
- **Internal results win**: `capabilities` on a model entry (e.g.
  `{ "infra": 92 }`) overrides cached data for that metric - use it for
  infrastructure, where no public benchmark maps.
- **Safety rules still apply after scoring**: escalation (complex demanding
  work never stays on the cheapest tier) and the budget downgrade run on top
  of the scored pick; eligibility, enabled status, registry presence, and
  context-window checks all happen BEFORE scoring.
- **Deterministic tie-break**: score desc, quality desc, price asc, config
  order. Same inputs -> same decision, always.
- **`classifierWeight`** (default 0 = legacy benchmark-only): the
  classifier's **model** answer is blended in as `support_m` - its
  probability relative to the eligible argmax - while quality and cost are
  scaled by `1 - classifierWeight` so the three weights still sum to 1 and
  the score stays in 0..1. Without it, a model the classifier scored 0.0 can
  still win on benchmark quality alone. The shipped config uses 0.35;
  raising it hands more authority to the classifier, lowering it leans on
  benchmarks. Absent classifier support the ranking is unchanged.
- Descriptions remain human-facing / classifier-criteria metadata; they no
  longer drive selection when scoring is enabled.

## Strategy evaluation (same corpus, five strategies)

Corpus: `tests/eval-corpus.ts` - 31 engineering requests (coding, multi-file,
difficult debugging, K8s/RKE2, networking/deploy, PostgreSQL/Oracle,
distributed systems, tool-heavy agent flows, context follow-ups,
large-context, ambiguous + task changes, budget cases). Ground truth:
expected category, complexity, acceptable model tiers. Run: `node --test
tests/strategy-eval.test.ts` (+ opt-in env gates below).

Measured 2026-10-09 on this corpus (prices = mean input price, USD/1M,
estimate):

| Strategy | category | complexity | tier acc | violations | clf latency | classifier cost | exec price mean |
|---|---|---|---|---|---|---|---|
| A local Laya (`LAYA_INTEGRATION=1`) | 35.5% | 51.6% | 83.9% | 5 | 952 ms | $0 | 0.082 |
| B remote Jev (`REMOTE_INTEGRATION=1`) | 90.3% | 58.1% | 90.3% | 3 | 1495 ms | **$0.001146 total (actual)** | 0.442 |
| C benchmark-scored (deterministic) | 100%* | 100%* | 83.9% | 5 | - | $0 | 0.788 |
| D always-cheapest | n/a | n/a | 58.1% | 13 | - | $0 | 0.010 |
| E always-strongest | n/a | n/a | 35.5% | 20 | - | $0 | 2.000 |

\* C uses **controlled classifier answers** (category/complexity given by
the fixtures) with **real OpenRouter benchmark data** - it measures the
selection policy, not classifier accuracy. A and B are real classifier
measurements (no guarantees). C, D, E are deterministic and run in every
standard test pass; A/B are opt-in and never required for CI. No strategy is
declared a winner: B buys +54.8pp category accuracy over A for ~$0.000037/call
and +0.5s latency; C dominates the baselines on tier accuracy at 2.5x the
cost of D but 0.39x the cost of E.

## Cost handling and observability

- **Laya is local**: zero API charges. Only its latency (`laya_ms`) and
  failures are logged, never token costs.
- **Classifier cost is tracked separately from execution cost**: every
  classified route logs a dedicated `clf=` line - backend, model, latency,
  token usage, and cost labeled `actual` (provider-reported, remote), `none`
  (local run), or `unavailable` (remote call ok, no cost reported - never
  invented). Classifier cost never enters `est_cost_in`/`est_cost_out`, which
  belong to the execution model; the strategy comparison adds them explicitly
  when comparing total workflow cost.
- **Execution-model cost** is always labeled, never confused with actuals:
  `est_in` prefers (in order) the latest usage entry from the session log
  (`source=usage` - the **previous** request's actual tokens, attributed only
  when it ran on the same provider AND model), then Pi's current context
  tokens (`source=context`), then `chars/4` (`source=chars`), multiplied by
  the registry model's `cost.input` price. `est_cost_in` / `est_cost_out` are
  labeled estimates (`est_out` only from previous usage - never invented);
  `last_cost` is the actual cost of the previous completed request, logged as
  `(actual, previous request)`. Zero prices estimate zero; missing prices
  estimate nothing (null).
- No prompts, summaries, or conversation content are logged - only ids,
  sizes, latencies, and rule names. No secrets are read.

### `/triage` dashboard

The extension registers a `/triage` slash command that prints a session
dashboard into the transcript:

- **Model lineup** - every configured entry with its USD/1M-token prices
  (auto-refreshed catalog price wins over the config `priceIn`/`priceOut`),
  the cached `coding` benchmark index, and its status (active/disabled).
- **Session cost** - the running **estimated** total across the routes
  recorded this session, plus a per-model breakdown. Estimates come from the
  routing decision's `estCostIn`/`estCostOut` (see above); they are never
  invented.
- **Route history** - the last 50 routes of this session (newest first):
  time, model, rule, complexity, category, and estimated cost.
- **Cache status** - benchmark and price cache freshness/coverage.
- **Config** - classifier backend, jev backend, fallback policy, override,
  and scoring state.

History and totals are in-memory and per-session: nothing is written to
disk, and no conversation content is recorded - only ids, rules, and costs.
The command exists on any pi build with `registerCommand`; older builds skip
registration and keep routing unchanged.

## Tests

The test suite lives in the [git repository](https://github.com/mraghakhani/pi-triage-router)
(the npm tarball omits test files). Clone the repo or install via git:

```sh
pi install git:github.com/mraghakhani/pi-triage-router
cd ~/.pi/agent/git/github.com/mraghakhani/pi-triage-router

# Unit suites (deterministic, mocked classifier/fetch, no Laya load, no network)
node --test tests/triage-router.test.ts         # core routing, config validation
node --test tests/benchmarks.test.ts          # ingestion, cache, refresh, scoring
node --test tests/remote-classifier.test.ts   # remote backend, retries, policies
node --test tests/strategy-eval.test.ts       # strategies C/D/E (deterministic)
node --test tests/laya-eval.test.ts           # corpus regression (deterministic part)

# Integration suite (real local Laya, ~2 GB RAM; weights cached once)
LAYA_INTEGRATION=1 node --test tests/laya-integration.test.ts

# Optional real-classifier measurements (paid for B; never required for CI)
LAYA_INTEGRATION=1 node --test tests/strategy-eval.test.ts tests/laya-eval.test.ts
REMOTE_INTEGRATION=1 OPENROUTER_API_KEY=... https_proxy=... \
  node --test tests/strategy-eval.test.ts

# Type check
./node_modules/.bin/tsc --noEmit -p /tmp/tsconfig.laya.json   # or any equivalent config
```

`tests/fixtures.ts` mirrors the shipped config's eligible models against a
fake registry; `tests/capture.ts` captures the registered virtual model so
`route()` itself is tested end-to-end with a mocked classifier.

### Routing-quality evaluation corpus

`tests/eval-corpus.ts` holds 31 representative engineering requests (coding,
multi-file changes, difficult debugging, Kubernetes/RKE2, networking and
deployment, PostgreSQL/Oracle, distributed systems, tool-heavy agent
workflows, research, simple questions, large-context, context-dependent
follow-ups, ambiguous requests and task changes, cheapest-insufficient and
expensive-unnecessary cases), each with its expected category, complexity,
and acceptable model tiers. `tests/laya-eval.test.ts`:

- deterministically (mocked classifier answers): asserts category and
  complexity survive the rules, every decision lands in an acceptable tier
  (violations = 0), escalation happens only where a strong tier is
  acceptable, the no-signal fallback policy holds, and corpus routing cost
  stays within the fixed cheap/strong baselines - **no task-success or
  cost-improvement claims**;
- with `LAYA_INTEGRATION=1`: the real local Laya classifies the whole corpus
  with history and prints measured category/complexity accuracy, violations,
  escalation/fallback frequency, latency, and mean input price (a
  measurement, not a guarantee).

## Requirements / installation

- `pi` 0.99.0+ (the virtual-models API this extension is built on; check with
  `pi --version` - older builds fail the extension load with "pi.registerVirtualModel
  is not a function").
- Node.js 20+ (runs fine on 26), ~2 GB RAM for the local classifier.
- `npm install` **next to `package.json` inside the dotfiles repo**
  (`~/dotfiles/pi/.pi/agent/extensions/`), not in `~/.pi/agent/extensions`.
  The extension files are symlinks and Node resolves their imports from the
  real path (the repo dir), so `undici`/`@receptron/laya`/`onnxruntime-node`
  must be installed there. `triage-router.setup.sh` does this for you.
  (`undici` is only imported when an HTTP proxy is configured.)
- First use downloads ~1.7 GB ONNX weights to `~/.cache/receptron-laya`.

### Diagnostics: where routing output goes

Per-request routing info (selected model, classifier latency/cost, benchmark
freshness) goes to **the TUI status bar** (`ctx.ui.setStatus`, key
`laya-route`) — stderr spam would corrupt the input line. Warnings (config
problems, classifier failures) still print to stderr AND raise a
notification. In print/JSON/RPC modes everything logs to stderr as before.
Set `TRIAGE_ROUTER_LOG=1` to force the machine-readable stderr lines in the
TUI too.

### First run: the ~2 GB classifier weight download

`laya/auto` runs Laya locally, which needs ~2 GB of ONNX weights cached in
`~/.cache/receptron-laya`. **`triage-router.setup.sh` pre-downloads them** (with
progress, borrowing your `settings.json` `httpProxy` if huggingface.co is
blocked). If you skipped that or the download died mid-way, the first
`laya/auto` message downloads them inline — which on a slow or blocked
network can look *stuck*.

The router no longer hangs on this: it waits at most **20 s** for the load,
shows `laya: loading… / downloading …` in the status bar, then routes
deterministically until the weights arrive; the next message classifies
normally. To check or re-run just the download:

```bash
du -sh ~/.cache/receptron-laya                 # ~2 GB when ready
bash ~/dotfiles/pi/.pi/agent/extensions/triage-router.setup.sh   # idempotent; re-downloads if <2 GB
#   (pass --skip-weights to set up files/deps without touching the weights)
```

### Advanced: symlinks from a dotfiles repo (local dev)

This setup is for developers who track their pi config in a dotfiles repo.
For a standard `npm:` install you don't need it — pi manages everything under
`~/.pi/agent/npm/node_modules/pi-triage-router/`.

The runtime files live in the dotfiles repo and are symlinked locally:

```
~/dotfiles/pi/.pi/agent/extensions/triage-router.ts  <-> ~/.pi/agent/extensions/triage-router.ts
~/dotfiles/pi/.pi/agent/extensions/triage-router-lib/ <-> ~/.pi/agent/extensions/triage-router-lib/
+ triage-router.json, triage-router.sample.json, README.md,
  triage-router.benchmarks.json, package.json, package-lock.json
```

On another machine: pull the repo, then run
`bash ~/dotfiles/pi/.pi/agent/extensions/triage-router.setup.sh`. It creates
the symlinks above, runs `npm install` in the repo dir (where the imports
actually resolve from), and verifies the deps. `tests/` stays local.

**Why `triage-router-lib/`:** pi loads every top-level `extensions/*.ts` as
an extension and rejects files without a default-export factory. The two
helper modules therefore live in a subdirectory without an `index.ts`
(subdirectories only load when they have an `index.ts`/`index.js`). If you
rename or relocate them, update the imports in `triage-router.ts` and
`tests/`, or pi will print "does not export a valid factory function".

## Known limitations

- Laya's state room is ~316 tokens: context is compact by design, and the
  current request is cut to fit (explicitly, with a marker) when needed.
- Context-window eligibility uses conversation tokens + chars/4 for the new
  request + a fixed 16K reserve: an approximation, documented, conservative
  on the side of routing rather than throwing.
- Benchmarks/prices in the shipped config come from the OpenRouter snapshot
  of 2026-10-09; `enabled` entries are the LiteLLM proxy catalog. The 12
  `enabled: false` OpenRouter-only entries are kept as reference data.
- `route()` failures (e.g. every eligible model disappearing mid-session)
  surface as a routing error in Pi; the session keeps its previous model.
- The estimate uses the **last** provider usage entry, which may lag the
  current turn on fast follow-ups (labeled `usage` = actual, previous).
- Escalation/quality rules key off the classifier's category+complexity
  answers; measured real-corpus accuracy (see evaluation corpus above) is
  modest for local Laya (35.5% category / 51.6% complexity, 2026-10-09) and
  much better for the remote Jev backend (90.3% / 58.1%) - choose the backend
  and tune criterion texts for the stakes involved.
- Remote classification adds ~0.5 s latency and sends the bounded routing
  input to OpenRouter; its cost is provider-reported `usage.cost` (actual)
  and stays in the `clf=` log line, never in execution cost.
- Benchmark coverage is incomplete by design: tau/gpqa rows are missing for
  some models, search/research rows were absent for all six enabled models in
  the 2026-10-09 snapshot (research scoring therefore abstains and falls back),
  and there is no public infra benchmark - use `capabilities.infra`.
- The unified benchmark endpoint is rate-limited (30/min, 500/day per key);
  refreshes are background, bounded to one per interval, and failures keep
  the last known good cache.
- Strategy B (remote) results come from one run on one day on a synthetic
  ground-truth corpus; they are measurements, not guarantees, and no
  strategy is declared a winner without comparable real task outcomes.