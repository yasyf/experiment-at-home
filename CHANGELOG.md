# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **`athome.research.meta`** — the campaign outer loop (A6b): `run_campaign` proposes, admits, and runs one experiment per round under a campaign-wide flock, with reservation-based cumulative caps — each launch reserves the admitted spec's declared worst case (`max_usd`/`max_wall_s`) against the `CampaignBudget` and is refused, never stretched, when the reservation would breach it; the reservation is released to journal-summed actuals when the experiment ends. An append-only `Ledger` (the Journal idiom: torn-final-line refusal, crash-resume recomputes totals, the consecutive-failure streak, and the next seq from disk; an unreleased reservation stays counted at its worst case, so a crash can only over-reserve) records one JSONL row per campaign event (`proposed`/`rejected`/`pending`/`approved`/`reserved`/`started`/`preflight_failed`/`aborted`/`infra_aborted`/`completed`/`stopped`). The per-experiment loop wires A2 preflight before any paid work and the A3 taxonomy unchanged: an `InfraFailure` ledgers `infra_aborted` without consuming a candidate-experiment slot, and rejected/preflight-failed/infra-aborted rounds count toward the `max_consecutive_failures` stop. Each completed experiment generates its A5 retro (durable in `retros.jsonl` before it feeds the next round's proposer context); a retro-generation error aborts loudly after the accounting row is durable. Gated mode writes admitted specs to `pending/` and never runs them — `meta approve` SHA-256-verifies the file into the queue (drained through the identical codepath), `meta reject` records why. `meta stop` arms an atomic stop file the runner checks at the experiment boundary: the current experiment finishes, nothing new launches. CLI: `athome research meta run|stop|report|install|approve|reject`, with `MetaSettings` (`[research.meta]`: root, backend, tier).
- **`athome.research.retro`** — post-experiment retrospective generation and durable, cc-notes-mirrored records.
- A quiet-alarm research watchdog with persistent journal/log byte-offset tracking, A3-compatible event alerts, cc-notes mirroring, and a ten-minute launchd agent.
- **`athome.research.policy`** — the outer loop's operator policy as pure frozen data: `ExperimentTemplate` (the verbatim-copied executable fields plus a `mutable_allowlist`), per-experiment `Budget` ceilings, a `CampaignBudget` (cumulative experiment/spend/wall/failure caps), and an `auto`/`gated` mode. `ProposalPolicy.load` refuses unknown TOML fields at every level, and a self-inconsistent policy — duplicate templates, an unbounded allowlist glob, `auto` mode without the `max_usd`/`max_wall_s` ceilings the reservation math needs — raises `PolicyViolation` at load, before any proposal is read.
- **`athome.research.propose`** — the LLM proposer, admitted only by refusal. The `Proposal` schema has no slot for any executable field (metric command/key, direction, immutable paths are copied verbatim from the chosen template; extra fields are refused by the model itself), and `validate_proposal` raises `ProposalViolation` on every violation, never clamping: allowlist membership is exact string equality (no glob subsumption), every budget number must sit at or under its operator ceiling with `max_usd`/`max_wall_s` mandatory in auto mode, and the name must fully match `NAME_RE` before the harness-supplied ledger `seq` prefixes it. `propose()` runs `spawnllm.extract` on a concretely-bound backend under the judge's backoff, re-prompts a refused proposal with its violation at most twice (≤3 extract calls per round), and returns a `ProposalRound` audit record; its `ProposerContext` carries only harness-authored history (retros, morning-report stats, leaderboard cells, the promoted version, prior failure reasons) — never run-log bytes.
- `ExperimentSpec` gained an optional `hypothesis` field: the proposer's free text renders into the contract's new `## Hypothesis` section and the audit record, and nowhere else — never a shell, path, or key.
- A `qwen` model family in the judge's alias table (`qwen`/`qwq`/`qvq`/`alibaba` tokens), so `family_of` resolves qwen model strings and `ensure_cross_family` can pass a qwen generator against an anthropic judge instead of failing closed with `UnknownFamilyError`.
- `athome research run --mirror-cc-notes` mirrors each research journal row to the installed `cc-notes` service; `athome research nightly install --mirror-cc-notes` persists the same opt-in behavior in the launchd agent.
- Tinker pricing for `Qwen/Qwen3.5-9B`: $0.44/Mtok prefill and $1.33/Mtok for sampled and training tokens.

### Fixed
- Training baseline comparability now fingerprints the evaluation task and identifying arm configuration as well as its ranking metrics, so changing the bake-off cannot reuse an unrelated frozen baseline. The baseline database is also opened and read during preflight, before a paid backend starts, and concurrent baseline writes are atomic while preserving exact-value conflicts.

### Security
- `metric_file` must now be a relative path confined to the work directory — no absolute paths, no `..` traversal — refused with `PolicyViolation` at policy load and `UnconfinedPath` at `ExperimentSpec` construction. An absolute `metric_file` previously produced an admitted spec whose pre-measure cleanup unlinked that external file: an arbitrary-file-deletion primitive.
- Every budget number and operator ceiling must be finite and positive, enforced in `Budget` and `CampaignBudget` themselves (`InvalidBudget` / `PolicyViolation`). A `max_usd = inf` ceiling previously admitted `inf` proposal budgets (`inf <= inf`), disabling all spend limits.
- The harness-supplied ledger `seq` must be a positive non-boolean int before it forms the experiment identity; `seq=True` previously collided with `seq=1` as `001-…` (shared journal/lock/branch identity), and zero/negative values were admitted.
- Template names must fully match `NAME_RE` at policy load, so a name carrying newlines or Markdown headings cannot inject sections into the proposer prompt.

### Changed
- `ExperimentSpec.load` now refuses TOML fields the spec or its budget does not declare with a named `UnknownSpecField` error instead of an incidental `TypeError`, so a hostile TOML cannot smuggle a field past the declared schema (and the refusal survives any future loosening of the constructors).

## [0.6.0] - 2026-07-17

### Added
- **`athome.train.engine`** — the Tinker lane rebuilt from scratch as an ordered op stream over the SDK's clock-cycle model. A schedule is data — `TrainOp` (one optimization step: `forward_backward` + `optim_step` as a single inseparable value, so the sequential await-between-them mistake has no representation), `ScoreOp` (one batched prefill-billed forward), `SnapshotOp` (`save_weights_for_sampler` with optional eval datums scored against those same weights) — and `execute()` is the only code in the package that ever holds an SDK future: it submits each step's pair together before awaiting either, keeps a submit-ahead queue so the worker never idles between steps, and drains results strictly in submission order. One step now costs one clock cycle where the old lock-step loop paid ~3. `projection()` prices a schedule by folding over the same value `execute()` runs, so a run can never bill differently than it projected.
- **`TinkerBackend.fit(spec, *, sink, checkpoints, eval_rows) -> TrainReport`** — the one training entry point. `CheckpointPolicy` places intermediate snapshots at fractions of the run (7-day TTL; the final save is always taken and kept forever), and pre-tokenized `EvalRow`s are scored against every snapshot's exact weights by riding the op stream — one batched forward per checkpoint instead of hundreds of serial round trips. The report carries per-step `StepRecord`s and per-checkpoint `SavedCheckpoint`s with `ScoredSequence` arrays, so checkpoint selection is caller-side math over evidence, not a second network pass.
- **`TinkerBackend.materialize` / `fuse` split** — `materialize` downloads a saved checkpoint and converts it to a servable non-fused mlx-lm `Adapter`; `fuse` folds an adapter into the base model as the fused `Checkpoint` when a consumer needs one. Consumers that serve the bare adapter no longer pay a base-model load and multi-GB fuse they never use. `train()` is unchanged in contract: `fit` + `materialize` + `fuse`.
- **`TinkerBackend.score(path, rows, *, base, max_usd)`** — post-hoc scoring of what Tinker actually serves: a sampling client against any saved checkpoint path, rows fanned out under bounded concurrency, reduced with the same weighted arithmetic as `fit`'s in-train eval so the two are directly comparable (serving-drift checks are a subtraction).
- **`athome.train.retrain`** — the retrain pipeline shape, hosted: `fit` → argmax a caller-supplied `select` over the saved checkpoints → `materialize` the winner → a caller-supplied `artifact_scorer` reads the local artifact → a caller-supplied `gate` returns the `GateVerdict`. `RetrainOutcome` carries the full evidence chain; `retrain` itself takes no side effect — registration, promotion, and journaling stay with the caller, in the caller's registry.
- **`athome.train.gate`** (new `gate` extra: numpy, scipy, scikit-learn) — promotion-gate statistics: exact sign test, budget-matched fire thresholds with conservative tie handling, sentinel AUC, and the corrected paired gate. Every score input has an explicit higher-is-fire contract — the silent `1 - p` orientation flips in the donor implementation are gone — and domain strata arrive as one caller-supplied `warranted` mask. Budget matching also fixes a latent donor defect: an integer fire budget no longer round-trips through a per-100 float rate, which could floor away one fire (19 scores at budget 5 fired only 4).
- **`Rows`** — an in-memory `DatasetSource` for pools the caller already holds, ending the write-a-temp-JSONL detour.
- `InsufficientData` — a pool smaller than one batch now raises before any billable call (it previously trained silently on an under-filled batch); `OverlongEvalRows` rejects eval rows longer than `max_seq_len` up front.

### Changed
- **Spend projection is single and complete**: `fit` runs exactly one spend-guard check covering training tokens (twice for DPO's two-pass custom loss), the DPO reference prefill, and every checkpoint's eval prefill — before any client exists. Spend is recorded and journaled only as results drain, in step order.
- DPO reference logprobs now ride the batched scorer (`ScoreOp` on the frozen client) instead of a per-pair loop, and are fully consumed before the training schedule is compiled.
- `gather_bounded` moved from `athome.research.judge` to **`athome.concurrency`**, with `concurrency` now a required keyword — no compat re-export.

### Removed
- `TinkerBackend.run_sft` / `run_dpo` — the lock-step loops are gone; schedule compilers behind `fit()` replaced them, and there is no public surface that can express the sequential shape.

## [0.5.1] - 2026-07-16

### Fixed
- `PipeWorker` now spawns each sidecar in its own session/process group (`start_new_session=True`) and `reap()` kills the whole group with `os.killpg` rather than signalling only the direct child. A worker launched behind a supervisor that forks without exec — `uv run <worker>` being the common case — previously left the real worker orphaned when only the supervisor was killed: the orphan kept the inherited stdin/stdout/stderr pipe ends open, so teardown of a poisoned worker (a cancelled in-flight `call()`) or a graceful-close timeout leaked the process and stalled on its still-open pipes. Killing the session group reaps the supervisor and its descendants together, so `poison()` and `aclose()` return promptly and leave no orphan.

## [0.5.0] - 2026-07-15

### Added
- `athome.embed.VoyageEmbedBackend` — the Voyage AI `/embeddings` API as an `EmbedBackend`, behind the new `embed-voyage` extra (`voyageai`, `numpy`). Dual-budget batching caps each request by both item count (`batch_texts`, ≤256) and total characters (`batch_chars`, ≤240k); batches run concurrently under a `concurrency` semaphore and the vectors are reassembled in input order. `input_type` (`query`/`document`) and `normalize` are per-instance construction state, so a consumer builds one document backend and one query backend. `VoyageSettings` binds `[embed.voyage]` / `ATHOME_EMBED_VOYAGE_*` and reads the canonical `VOYAGE_API_KEY`.
- `athome.registry.components` / `rollback` / `prune` — generic registry verbs. `components` lists the artifact families that have at least one registered version under a root, so a registration that died before its commit no longer leaves a ghost family listed forever; `rollback` repoints `current` to the version registered before the current promotion (raising when none exists); `prune` deletes all but the newest `keep` versions while always retaining the one `current` points to, and refuses a negative `keep` rather than treating it as zero and deleting everything. Deleting versions made the family lock load-bearing for `promote` too, which until now took no lock: every verb that moves `current` or removes a version — `register`, `promote`, `rollback`, `prune` — serialises on it, so a promotion can no longer land on a version a concurrent prune is deleting and strand `current` on a dangling symlink. `prune` renames each doomed version out of discovery before removing it, so a reader never observes a half-deleted version and a crash mid-delete leaves nothing discoverable behind.
- `athome.llm.pricing` prices `Qwen/Qwen3-8B` at $0.13 in / $0.40 out per Mtok, so a lane pointed at the Qwen base costs out instead of raising `UnpricedModel`.
- `athome.train.spec.TinkerPrice` — one Tinker base model's price split into the three token classes Tinker actually meters: `prefill`, `sample`, `train`.
- `AgentSpec.log_dir` — a per-agent log directory, so a consumer keeps its agents' logs beside its own state instead of in the shared `[athome].logs_root`. It overrides the directory only: the filename stays `{log_name or label}.log`, and the default `None` still writes to `logs_root`, so an existing spec renders the same plist it always did.

### Changed
- **`athome serve up|down|status` default their `RECIPE` argument to `rapid-mlx`.** The argument was required; naming a recipe is now how you opt out of the default rather than a precondition for running at all. `serve.DEFAULT_RECIPE` names the policy for consumers.
- The `RapidMlxSettings` and `LlamaServerSettings` docstrings now carry **engine-selection guidance**, and `athome.serve` has an entry in the API reference — it had none, so nothing about picking an engine had anywhere to surface. Measured on one model (Qwen3.6-35B-A3B) across 42 tool calls per engine, the two are equivalent on ordinary tool calling: zero dispatch errors and fully valid, schema-conformant arguments on both, over flat, enum, nested, arrays-of-objects, multi-tool, and numeric-verbatim shapes. They separate only at two edges, one per engine, and each docstring points at the other: rapid-mlx's delimiter-based parser truncates a string argument at an embedded `</function>`, dropping data silently behind well-formed JSON; llama-server ignores `tool_choice="required"`, answering in prose and running to `max_tokens` rather than forcing a call.
- **`TinkerSettings.price_per_mtok` is now a per-class price, not a flat rate.** It binds `dict[TinkerModelId, TinkerPrice]` rather than `dict[str, float]`, and `TinkerBackend.cost` takes a token count per class — `cost(model=…, prefill=…, sample=…, train=…)` — instead of one undifferentiated `tokens`. Tinker bills a forward-only pass, a sampled token, and a training token at three different rates, so the old flat rate charged DPO's frozen reference pass at the training rate: that pass now bills at prefill ($0.13 vs $0.40 per Mtok on Qwen3-8B). Training rates are unchanged ($0.40 Qwen3-8B, $0.67 Qwen3.5-4B), so an SFT run costs exactly what it did. A `[train.tinker.price_per_mtok]` TOML override written against the old flat shape no longer validates.

### Known limitations
- **The `embed-voyage` extra does not install on free-threaded 3.14t**, the same as `train`. `voyageai` pulls `langchain-*` → `langsmith` → `orjson`, and `orjson` publishes no `cp314t` wheel, so the resolve fails outright rather than degrading. The extra installs normally on GIL 3.13 and 3.14. `athome` core stays 3.14t-clean either way: `VoyageEmbedBackend` imports `voyageai` lazily, so nothing in the core import graph reaches it.

## [0.4.0] - 2026-07-14

### Added
- `athome.train` — LoRA fine-tuning over three backends, preferred **tinker > local > modal** and picked by availability (`athome train run|status|register`). SFT is first-class on all three; DPO runs on tinker and modal.
  - Every backend converges on the same artifact: a **fused, standalone 4-bit MLX model**. `rapid-mlx` serves one model and has no adapter flag, so an unfused adapter is unservable — tinker and modal convert their PEFT adapter, local fuses.
  - Tinker DPO goes through the SDK's custom-loss seam (`forward_backward_custom`): the SDK ships no preference loss, so athome renders chosen/rejected pairs, caches reference-policy logprobs, and backprops a log-sigmoid margin loss in torch. That path — and only that path — needs the new `train-dpo` extra.
  - `athome.train.run` trains, serves the result, scores it through a bake-off, and writes a `.athome-metric.json` scalar — the same structured metric channel `athome.research` already reads, so pointing an `ExperimentSpec` at `athome train run` turns the overnight loop into a training-recipe search with no new machinery.
  - Spend caps are hard aborts on both paid backends (tinker per-Mtok, modal per-GPU-hour).
- `athome.registry` — the content-addressed version-dir + `current`-symlink registry, promoted out of `athome.research` so training and research share one promotion path. `athome.research.registry` keeps its imports and its on-disk root.
- `ManagedServer.ensure(model=...)` — serve an arbitrary model path without mutating the process-global serve settings, so a freshly trained artifact can be evaluated in place.
- `Store.open(..., synchronous=)` — callers that need durability can demand `FULL`.

### Fixed
- **The sidecar wire dropped Apple Vision entirely.** `ocrmac` returns text as `objc.pyobjc_unicode`, a `str` subclass: it passed `validate()`, then the restricted unpickler refused it (`refused builtins.str`), so OCR over a worker failed outright. `validate()` now coerces every primitive to its exact builtin — what the wire accepts is what a frame can carry — for every producer, not just this one. (`bool` is matched before `int`, so it survives as `bool`.)
- **An OAuth `HF_TOKEN` failed the write preflight outright.** `ensure_write_auth` read `whoami()["auth"]["accessToken"]["role"]`, but an OAuth token — what `huggingface-cli login` mints — reports `{"type": "oauth", "expiresAt": …}` and carries no role at all, so the preflight raised `KeyError` instead of passing. Every `hf.push` under an OAuth login died before it started. The preflight now matches on the auth type: OAuth and classic-write pass, a non-write classic role is rejected by name, and an unrecognized shape raises `HfAuthError` rather than a `KeyError`. OAuth scope is not exposed by `whoami`, so a narrowly-scoped OAuth token is still caught by the hub at push.
- Apple Vision boxes round their true corners instead of origin and size independently; consumers rebuild the far edge as `x + width`, so an independently rounded size drifted that edge off the real box.
- `Cache.write` cleans its staging directory and marker under a shielded scope, so a cancelled write no longer strands them until the stale-temp sweep (which remains the backstop for process death).
- `Store.open` arms `busy_timeout` **before** the WAL flip. Converting a brand-new database takes an exclusive lock, so a concurrent open was returning `SQLITE_BUSY` instead of waiting for it.

## [0.3.2] - 2026-07-14

### Added
- `Arm.client_factory` — an optional zero-argument builder for a bake-off arm's `AsyncOpenAI` client, so a remote or authenticated arm can supply a real API key (and optionally the `athome.llmcache` record-replay transport) instead of the default local client. An arm that omits it behaves exactly as before.

## [0.3.1] - 2026-07-13

### Added
- `athome.cache.atomic_write_bytes` / `atomic_write_text` — atomic same-filesystem writes to any caller-chosen path (temp sibling, `fsync`, `os.replace`), for standalone files that live outside a cache namespace.
- `Cache.open(..., root=)` — a per-call cache-root override, so a CLI flag or a test fixture can point one cache at its own directory without mutating the process-wide `cache_root` setting.
- `Store.open(..., busy_timeout_ms=)` plus retry-on-locked in `Store.execute` (bounded exponential backoff), for higher-contention sqlite consumers.

## [0.3.0] - 2026-07-13

### Added
- `athome.research` — the overnight autoresearch harness: a greedy keep/discard loop (`athome research run`) that drives a coding agent against a git-isolated checkout and keeps each proposal only when a trusted metric improves.
  - A structured metric channel: the score is read only from the JSON file the metric command writes, never grepped from stdout; run logs are captured but withheld from the agent's next contract, closing the prompt-injection surface.
  - `ExperimentSpec` (TOML) with blocking pre-flight invariants, a typed append-only resumable journal, monotone + bootstrap-CI promotion gates, and a content-addressed registry with atomic promotion.
  - An evaluation kit: position-debiased pairwise spawnllm judges with cross-family enforcement, blind golden-labeling packets whose panel-vs-human agreement gate blocks LLM spend until green, saturation-aware calibration, and topic-leakage probes.
  - A morning report (`athome research report`), a nightly launchd agent, and the `athome research` CLI.

### Security
- The research harness enforces its anti-reward-hacking boundary structurally, not by prompt: immutability is checked by diffing the staged index against the incumbent over a tight `mutable_paths` allowlist — symlinks and Python auto-loaders rejected — so a violating candidate is discarded before a commit object exists, and scoring runs from a clean checkout containing no `.git`, which closes the git-config RCE class.
- Per-experiment single-writer locks, work-unit/wall-clock/dollar budgets with journal cost validation, and cross-family judging plus the golden gate as LLM spend controls.
- [docs/design/research-security-model.md](docs/design/research-security-model.md) documents the threat model and the limitations only an OS sandbox closes: metric authorship, process-group escape, and cost forgery via the shared log.

## [0.2.0] - 2026-07-13

### Added
- `athome.serve` — recipe-driven `ManagedServer` for OpenAI-compatible local servers (`rapid-mlx` text lane, `mlx-vlm` vision/OCR lane, `llama-server` bake-off arm) with health, detached/launchd lifecycle, and a top-level `athome status` (`athome serve up|down|status`).
- `athome.llm` — explicit call-policy lanes `small`/`extract`/`local` over spawnllm, each routed through a shared spend guard and call log (estimate-based telemetry with a per-tier price map).
- `athome.llm.batch` — hand-rolled Anthropic/OpenAI/Gemini batch adapters (50%-off, `custom_id`-keyed, JSONL state, pre-submit `max_usd` cap, `expired`→auto-resubmit) via `athome batch submit|status|collect`.
- `athome.ocr` engines — `AppleVision`, `PaddleOcr` (sidecar), `VlmOcr` (dots.ocr via mlx-vlm), `EnsembleTokenOcr`, `LlmMerger`, and `realtime`/`quality` profiles behind `athome ocr`.
- `athome-ocr-paddle` — a separate `>=3.13,<3.14` sidecar dist (PP-OCRv6) with local and Modal transports, so free-threaded hosts reach onnxruntime out-of-process.
- `athome.modal` — an elastic Modal backend: a wire-compatible worker relay with mandatory parity fingerprinting before ready and no local fallback.
- `athome.hf` — pinned-revision snapshots and a write-role auth preflight (`athome hf pull`).
- `athome.embed` — a digest-keyed incremental embedding index with API/local backends and MMR rerank.
- `athome.bakeoff` — A/B/N endpoint bake-offs with a scored leaderboard and go/no-go gate (`athome bakeoff run`); gates the stream-trade llama.cpp→Rapid-MLX swap.

### Changed
- Core dependency bump: the `llm` extra now requires `spawnllm>=0.6.1` (its new OpenAI-endpoint backend powers `athome.llm.local`).

## [0.1.0] - 2026-07-13

### Added
- `athome.config` — the pydantic-settings surface over `~/.athome/config.toml` (`ATHOME_` env prefix, per-module sections, `load()` accessor).
- `athome.cache` — namespaced, content+version-keyed cache with atomic staging writes, stale-tmp sweeping, and the `@cached` decorator (`athome cache stats`).
- `athome.llmcache` — record/replay httpx transport for OpenAI-SDK-shaped clients, with the retry layer beneath the cache so only final responses persist.
- `athome.workers` + `athome.wire` — sidecar subprocess workers over a version-handshaked, length-prefixed wire protocol, plus a digest-keyed prefetch/lease pool.
- `athome.progress` — resumable JSONL work-sets (error units retry), crash-safe run sinks with failure budgets, and phase gating.
- `athome.launchd` — declarative launchd agents (`Calendar`/`Interval`/`KeepAlive` schedules) with install/uninstall/status (`athome launchd`).
- `athome.detach` — detached overnight runs with completion sentinels (`athome run --detach`, `athome run wait`).
- `athome.sync` — verified rsync mirroring: sha256 manifest loop, AppleDouble sweep, and source deletion blocked until the tree checks out (`athome sync`).
- `athome.store` — the shared aiosqlite scaffold (WAL, busy timeout, idempotent schema).
- `athome.ocr` protocol types — `TokenOcr`, `DocOcr`, `Document`, `OcrToken`, `Box` (engines land in 0.2).
- CI gate asserting the core imports on Python 3.14 free-threaded with the GIL still disabled.
- The in-repo `athome` Claude Code plugin (marketplace + `cache`/`overnight` skills).

[Unreleased]: https://github.com/yasyf/experiment-at-home/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/yasyf/experiment-at-home/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/yasyf/experiment-at-home/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/yasyf/experiment-at-home/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/yasyf/experiment-at-home/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/yasyf/experiment-at-home/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/yasyf/experiment-at-home/releases/tag/v0.1.0
