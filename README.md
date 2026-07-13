# ![experiment-at-home](https://github.com/yasyf/experiment-at-home/raw/main/docs/assets/readme-banner.webp)

**The plumbing every local AI experiment rebuilds, built once.** athome is one
SDK, CLI, and skill set for the machinery every experiment repo grows anyway —
content-keyed caching, sidecar workers, record/replay LLM caching, launchd
scheduling, detached overnight runs, and resumable progress.

[![CI](https://img.shields.io/github/actions/workflow/status/yasyf/experiment-at-home/ci.yml?branch=main&label=ci)](https://github.com/yasyf/experiment-at-home/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/experiment-at-home.svg)](https://pypi.org/project/experiment-at-home/)
[![Docs](https://img.shields.io/github/actions/workflow/status/yasyf/experiment-at-home/docs.yml?branch=main&label=docs)](https://yasyf.github.io/experiment-at-home/)
[![License: PolyForm-Noncommercial-1.0.0](https://img.shields.io/badge/License-PolyForm--Noncommercial--1.0.0-blue.svg)](https://github.com/yasyf/experiment-at-home/blob/main/LICENSE)

## Get started

```bash
uv add experiment-at-home
uv run athome --help
```

<img src="https://github.com/yasyf/experiment-at-home/raw/main/docs/assets/demo.png" alt="Terminal running 'uv run athome --help' — the cache, launchd, run, and sync command groups" width="700">

Driving with an agent? Paste this:

```text
Add experiment-at-home (uv add experiment-at-home) and wrap my most expensive
pipeline step in its @cached decorator, namespaced after this project.
Docs: https://yasyf.github.io/experiment-at-home/
```

<details>
<summary>Claude Code plugin (the athome skills)</summary>

```text
/plugin marketplace add yasyf/experiment-at-home
/plugin install athome@experiment-at-home
```

</details>

---

## Use cases

### Stop re-paying for the same expensive artifact every rerun

A pipeline rerun that re-derives every intermediate burns minutes (or API
dollars) to reproduce bytes you already had. Wrap the step; the cache is
content-keyed under `~/.athome/cache`, shared across every repo on the machine:

```python
from athome import cached


@cached(ns="ingest", version=1)
async def parsed_pages(pdf_digest: str) -> list[str]:
    # the expensive part: OCR, chunking, whatever your pipeline re-derives
    ...
```

The second call with the same arguments returns from disk. Invalidation is one
edit: bump `version=2` and the old entries are never read again.

### Send tonight's sweep off before you close the laptop

An overnight script tied to a terminal dies with the terminal, and `nohup`
leaves you grepping for whether it finished. Detach it with a name:

```bash
uv run athome run --detach --name sweep -- uv run python sweep.py
```

The run gets its own log under `~/.athome/logs/runs/`, and a completion
sentinel records the exit code. In the morning, `uv run athome run wait sweep`
returns instantly with that exit code — 0 or the failure you need to chase.

### Move a corpus without wondering if it all arrived

`rsync` reports transfer success, not tree integrity — and macOS salts remote
copies with `._*` AppleDouble droppings. Mirror with verification:

```bash
uv run athome sync ~/corpora/traces backup-host:corpora/traces --move
```

Every file is sha256-verified against the source before `--move` deletes
anything; a mismatch repairs and re-verifies, and an unverifiable tree blocks
deletion outright.

## More in the docs

- **Record/replay LLM caching** — replay yesterday's API responses in today's tests, byte-for-byte — [llmcache](https://yasyf.github.io/experiment-at-home/reference/llmcache.CachingTransport.html)
- **Sidecar workers** — call GIL-bound engines from free-threaded Python over a version-handshaked pipe — [workers](https://yasyf.github.io/experiment-at-home/reference/workers.PipeWorker.html)
- **launchd agents** — declarative Calendar/Interval/KeepAlive schedules, no plist authoring — [launchd](https://yasyf.github.io/experiment-at-home/reference/launchd.AgentSpec.html)
- **Resumable progress** — JSONL work-sets where crashed units retry and finished ones skip — [progress](https://yasyf.github.io/experiment-at-home/reference/progress.WorkSet.html)
- **One config file** — every knob in `~/.athome/config.toml`, every module a pydantic-settings section — [config](https://yasyf.github.io/experiment-at-home/reference/config.AthomeSettings.html)

Read the [docs](https://yasyf.github.io/experiment-at-home/) for the full guide.

Status: alpha — the primitives above are shipped and tested (including on
free-threaded Python 3.14); MLX serving recipes, OCR engines, batch LLM lanes,
and the autoresearch harness land in 0.2–0.3.

Licensed under [PolyForm-Noncommercial-1.0.0](LICENSE).
