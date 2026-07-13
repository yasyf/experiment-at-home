# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/yasyf/experiment-at-home/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/yasyf/experiment-at-home/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/yasyf/experiment-at-home/releases/tag/v0.1.0
