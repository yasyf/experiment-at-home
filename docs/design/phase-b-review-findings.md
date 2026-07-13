# Phase B review findings (accepted)

Two codex finder lanes (security/secrets/batch/modal/hf + correctness) over the
v0.2 code. fable synthesis: accept the list below; fix with fail-first regression
tests, then a refuter pass. The finders confirmed clean: keys flow through pydantic
settings (never logged/URL'd/journaled), `ensure_write_auth` precedes the HF push,
`custom_id` (not order) correlates results, `expired`≠`failed`, Modal parity
asserts before ready with no local fallback, and the 73 ty diagnostics are
Wire-union/`| None` narrowing artifacts (not runtime bugs).

## serve (`athome/serve.py`)

- **SV1** — `uvx --from mlx-vlm` executes an *unpinned* release (rapid-mlx is
  pinned `==`). Add a required `version` to `MlxVlmSettings` and pin it like
  rapid-mlx.
- **SV2** — any 2xx on `/v1/models` is adopted as ready without checking identity.
  On adopt, verify the served model id matches the recipe's configured model;
  mismatch → don't adopt (raise), so private prompts never hit a stray process.
- **SV3** — a readiness timeout leaves the just-launched process alive (and
  `stop()` misses the detached process when a launchd agent also exists). On
  `HealthTimeout`, tear down what `ensure()` started before raising; `stop()` must
  cover both detached and launchd state.
- **SV4** — the deadline is only checked after a failed probe, so a probe
  succeeding past the deadline is accepted. Check the deadline in the loop
  condition. (Low.)

## batch (`athome/llm/batch/`)

- **B1** — the provider batch is submitted *before* any state record is written; a
  crash/lost-response then re-submits (double-bill). Journal the submit intent
  (custom_ids + an attempt marker) *before* the provider call; on resume, a dangling
  attempt with no batch_id reconciles (query/warn), never blind-resubmits.
- **B2** — expired-item resubmit is an unlocked read→submit→append; launchd + a
  manual collector can each submit (double-bill). Take an `fcntl` lock on the state
  file across submit/collect, and journal-before-submit as in B1.
- **B3** — a retry batch (expired A → new B) is journaled but never polled/collected;
  collection keeps polling A and strands B. Register B as the active batch to poll;
  collection follows the retry chain.
- **B4** — `max_usd=NaN` bypasses the cap (NaN comparisons are always false).
  Require `max_usd` finite and > 0.
- **B5** — duplicate `custom_id`s silently collapse in a dict (last body wins). The
  correlation key must be unique — raise at submit on a duplicate.

## modal (`athome/modal.py`)

- **MO1** — parity checks only locally-known keys; a remote that *adds* a
  behavior-affecting package/param still matches. Compare the full fingerprint dicts
  for exact equality (keys and values), both directions.
- **MO2** — packages and params share a flat namespace (`params["numpy"]` overwrites
  the `numpy` version). Namespace them (`pkg:`/`param:` prefixes or nested dicts).
- **MO3** (document, not code) — Modal auth is the SDK's ambient token by design;
  `app_name` is workspace-scoped so a wrong workspace fails loud on `from_name`.
  Document that athome does not route the Modal token through settings.

## llm lanes (`athome/llm/__init__.py`)

- **L1** — spend `check` and `record` straddle the provider `await` with no
  reservation, so concurrent calls all pass against the same balance. `SpendGuard`
  must *reserve* the projected spend atomically at check, then reconcile at record.
- **L2** — telemetry (`CallLog.add`) is written before `guard.record()`; a sink
  failure skips the spend record (a completed call vanishes from the total). Record
  spend before the fallible sink write.
- **L3** (document) — every lane records `served_model=None`, so the drift detector
  is inert for lanes (spawnllm surfaces no served model). Document it; optionally
  `local()` reads the served model from its endpoint response.

## ocr (`athome/ocr/`)

- **O1** — `AppleVision` imports/runs GIL-bound `ocrmac` in-process via a worker
  thread; on 3.14t it must run out-of-process (a sidecar/subprocess via
  `athome.workers`), per the plan pitfall. Move ocrmac execution to a subprocess.
- **O2** — `documents_agree()` returns True when *either* transcription is empty, so
  an empty VLM result vs a full Apple read counts as agreement and the merger never
  fires. Empty-vs-nonempty is maximal disagreement.
- **O3** — `LlmMerger.merge()` ignores its `image` argument and sends only the text
  candidates; the merger must see the source image to adjudicate (`$100` vs `$700`).
- **O4** — a primary token stays fully trusted if *any* nearby supplement token
  agrees, even when another nearby token contradicts it (and the contradiction is
  discarded). Fix the cross-validation so a nearby contradiction is not suppressed.

## embed (`athome/embed.py`)

- **E1** — `upsert()` is an unlocked read-modify-write; concurrent writers to a
  namespace lose updates (atomic blob write prevents corruption, not lost updates).
  Serialize with a lock (and document single-writer), or merge-on-write.

## bakeoff (`athome/bakeoff.py`)

- **BO1** — aggregation/disagreement silently drop missing fields instead of using
  the full-corpus denominator, so an arm that omits `viable` on failed items scores
  `1.0` and passes the "viable on every item" constraint. Missing field ⇒ counts
  against the arm; the hard constraint fails on any missing item.
- **BO2** — the gate picks the highest observed arm then tests *that* arm at the
  unadjusted per-test alpha — no multiple-comparison control, so noise approves a
  regression. Apply a multiple-comparison correction (Bonferroni or a proper
  multi-arm procedure).
- **BO3** — the leaderboard sorts by primary metric only, ignoring the tiebreak and
  viability `WinnerPicker` uses (a nonviable arm can show first). Sort consistently
  with the winner selection. (Low.)
