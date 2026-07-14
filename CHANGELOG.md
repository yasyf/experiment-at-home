# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/yasyf/experiment-at-home/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/yasyf/experiment-at-home/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/yasyf/experiment-at-home/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/yasyf/experiment-at-home/releases/tag/v0.1.0
