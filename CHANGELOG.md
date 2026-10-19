# Changelog

All notable changes to this project are documented here, generated from
`git log` by `npm run changelog` (scripts/changelog.mjs).

## Unreleased

_2026-10-19 — [v0.5.0...HEAD](https://github.com/mraghakhani/pi-triage-router/compare/v0.5.0...HEAD)_

### Added

- **menu:** validate group refs before saving them (6998bca)
- **menu:** interactive groups management in /triage (5408ed8)

## v0.5.0

_2026-10-19 — [v0.4.0...v0.5.0](https://github.com/mraghakhani/pi-triage-router/compare/v0.4.0...v0.5.0)_

### Fixed

- **tui:** /triage must post the report in TUI too, not only print mode (e61f0c6)
- **dashboard:** history row wrapped past 100 chars - shorten time cell (ac778ad)
- **dashboard:** merged Groups line must not wrap on 80-col terminals (348a635)
- **menu:** Default group zero-groups guard + test indentation (4f98203)
- **direct:** compact must not inherit the activeGroup filter (b6a98e2)
- **menu:** polish Active group entry (e9b291e)
- **groups:** harden ref resolution against prototype-chain names (1f13aab)

### Added

- **dashboard:** Model groups section lists every group and its model refs (0e3f439)
- **menu,logs:** defaultGroup menu entry + direct-pick pool logging (05e4b3a)
- **direct:** /compact and out-of-loop requests pick from defaultGroup (4763b99)
- default groups in shipped config + Active group menu entry (a25238e)
- **groups:** named model groups filter classification and routing (071b2b6)

### Changed

- **labels,dashboard:** shared unset label map + dashboard wrap audit (52c667b)
- **dashboard,menu:** merge group lines, shorten menu default label (70e27b8)

## v0.4.0

_2026-10-18 — [v0.3.0...v0.4.0](https://github.com/mraghakhani/pi-triage-router/compare/v0.3.0...v0.4.0)_

### Fixed

- **triage:** the laya classifier provider is not editable from the menu (f38395a)
- **classifier:** litellm sections no longer inherit OPENROUTER_API_KEY (0fc2b41)
- **triage:** expose the jev classifier in the /triage edit menu (4adb5f4)
- **triage:** harden config editing from deep review (bf51af5)

### Added

- **triage:** reset config to shipped defaults from the dashboard (29eb7c6)
- **triage:** validate jev classifier edits and warn at startup (7a350f0)
- **triage:** interactive /triage menu with live config editing (050718f)
- **triage:** add /triage dashboard with session routing history and cost report (4c48a2d)

## v0.3.0

_2026-10-16 — [v0.2.3...v0.3.0](https://github.com/mraghakhani/pi-triage-router/compare/v0.2.3...v0.3.0)_

### Added

- auto-refreshed price cache (24h background, keyless OpenRouter catalog, bundled seed) (46164fa)
- configurable classifier retries (maxAttempts, retryDelayMs) (7afad1d)

## v0.2.3

_2026-10-14 — [v0.2.2...v0.2.3](https://github.com/mraghakhani/pi-triage-router/compare/v0.2.2...v0.2.3)_

### Fixed

- **paths:** write mutable state outside the package dir (e29041c)

### Documentation

- **readme:** switch primary install to npm, update all paths (a1d9835)

## v0.2.2

_2026-10-14 — [v0.2.1...v0.2.2](https://github.com/mraghakhani/pi-triage-router/compare/v0.2.1...v0.2.2)_

### Fixed

- **config:** default must be provider/id composite, not bare model id (944ff90)

## v0.2.1

_2026-10-13 — [v0.2.0...v0.2.1](https://github.com/mraghakhani/pi-triage-router/compare/v0.2.0...v0.2.1)_

### Fixed

- **install:** restore .npmrc so onnxruntime-node's postinstall self-skips (d6354d7)
- **router:** clear upgrade error when pi lacks the virtual models API (9852b81)
- **router:** read classifier base URL from the Pi auth file, never config (e5a7c1c)

## v0.2.0

_2026-10-12 — [v0.2.0](https://github.com/mraghakhani/pi-triage-router/releases/tag/v0.2.0)_

### Fixed

- **benchmarks:** fall back to the Pi auth file's openrouter key when the env var is unset (62930df)
- **estimate:** use assembled prompt length for the chars/4 cost fallback (9e41a82)
- **decide:** report confidence null for scored picks (7b42534)
- **benchmarks:** make research metric an order-independent equal-weight mean (cbd5352)
- **router:** key loadConfig cache by path as well as mtime (1f3171e)
- **router:** empty classification defaults to standard/coding via argmaxKey null (67cfa20)
- **scoring:** rank zero-price models as the cheapest tier instead of neutral (d3f74ff)
- **benchmarks:** require -MMDD suffixes to be real calendar dates (8f911f8)
- **router:** override/direct routes persist the model that actually served the turn (23f01e4)
- **benchmarks:** sanitize corrupt cache rows so NaN cannot reach scoring (fa14a10)
- **benchmarks:** honor proxyUrl \direct\ in benchmark refresh (5470f04)

### Documentation

- mark items 3, 5, 6, 12 as covered by the bug-fix pass (b242855)
- jev/auto registration requires restart after config changes (29c0bce)
- mark item 2 (parseDataset hardening) as already covered by B2 fix (ea3b5e3)
- add improvements report (cbb3982)
- annotate remaining findings as fixed/deferred/not-a-bug (f7a0ddc)
- retract B3 as false positive (cacheDir validation was correct) (16a7a8c)

### Changed

- **proxy:** unify classifier and benchmark proxy resolution into one chain (c86467e)
- **config:** extract default classifier/benchmarks/scoring helpers (c2076f6)

### Other

- correct the install paths to the git source directory (e35f94a)
