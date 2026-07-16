# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
