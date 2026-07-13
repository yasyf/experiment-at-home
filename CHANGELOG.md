# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- The in-repo `athome` Claude Code plugin (marketplace + skills skeleton).

[Unreleased]: https://github.com/yasyf/experiment-at-home/commits/main
