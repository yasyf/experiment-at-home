# experiment-at-home Development Guide

The plumbing every local AI experiment rebuilds, built once. Shared primitives for AI experiments on your own hardware — MLX serving, OCR ensembles, caching, batch LLM calls, and overnight autoresearch loops — extracted from cc-steer, stream-trade, and write-like-me, which all consume this package. Published to PyPI as `experiment-at-home`; the import package and CLI are both `athome` (run as `uvx --from experiment-at-home athome`).

## Repository Structure

```
experiment-at-home/
├── athome/            # The package — config, cache, llmcache, workers, progress,
│   │                  #   serve, launchd, detach, cli, store, sync, hf, embed,
│   │                  #   bakeoff, modal
│   ├── ocr/           # OCR protocol types, engines, ensemble, profiles
│   ├── stt/           # Speech-to-text: transcribe.cpp engine, GGUF catalog, OpenAI-shim server
│   ├── llm/           # Lane functions (small/extract/local) + batch adapters
│   └── research/      # Autoresearch harness: spec, loop, journal, gates, judges
├── engines/           # Sidecar dists (athome-ocr-paddle: >=3.13,<3.14, own pyproject)
├── plugin/            # Claude Code plugin shipping the athome:* skills
├── tests/             # Pytest suite
├── docs/              # Great Docs assets + scripts
├── .github/           # GitHub Actions workflows (CI incl. 3.14t GIL job, docs, release)
├── AGENTS.md          # This file — shared conventions
└── README.md          # Project overview
```

## Hard Constraints

- **Python 3.14 free-threaded must stay viable.** `athome` core imports cleanly on 3.14t with the GIL still disabled (CI asserts `sys._is_gil_enabled()` is False). Native-dep features live behind extras with lazy imports; GIL-bound engines run as sidecar subprocesses (`athome.workers`), never in-process.
- **Machine-specific detail lives in `~/.athome/config.toml`**, never in code. All settings flow through pydantic-settings models (`ATHOME_` env prefix); no scattered `os.environ` reads.
- **Abstractions stay generic.** OCR knows nothing about trading; caching knows nothing about transcripts. Domain logic belongs in the consumer repos.
