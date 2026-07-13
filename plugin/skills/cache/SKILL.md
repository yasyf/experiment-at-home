---
name: cache
description: >-
  Cache expensive pipeline artifacts (OCR passes, embeddings, parsed corpora,
  derived datasets) in the shared athome cache instead of ad-hoc pickle files,
  temp dirs, or hand-rolled digest schemes. Use whenever code re-derives bytes
  it already computed, needs atomic multi-file artifact writes, or needs
  record/replay caching of LLM API calls in tests.
---

# athome cache

The shared, content-keyed cache under `~/.athome/cache` (override:
`cache_root` in `~/.athome/config.toml` or `ATHOME_CACHE_ROOT`). Namespaced
per concern, versioned per schema — a version bump is the ONLY invalidation.
Never hand-roll a digest-named pickle dir; reach for this.

## Decorator (the 90% case)

```python
from athome import cached

@cached(ns="ingest", version=1)
async def parsed_pages(pdf_digest: str) -> list[str]: ...
```

- Key = qualname + args/kwargs (hashable primitives only — a list/dict arg
  raises `CacheKeyError`; pass a digest instead of the payload).
- Changed the function's semantics? Bump `version=` — never delete entries.

## Object API (bytes, directories, streams)

```python
from athome.cache import Cache

cache = Cache.open("ocr", version=2)
key = cache.key(image_digest, "dots-ocr", 1024)

if (hit := await cache.get_bytes(key)) is not None:
    return hit
await cache.put_bytes(key, result)

# multi-file / incremental entries: stage, then atomic publish
async with cache.write(key) as staging:
    ...  # write file(s) under `staging`; published atomically on clean exit
```

`cache.get(key)` returns the published `Path` (file or directory) or `None`.
Everything routes through one temp-sibling + `os.replace` codepath — a crash
never leaves a half-written entry visible.

## LLM record/replay (tests, offline reruns)

```python
import httpx
from athome.llmcache import LlmCacheMode, transport

client = httpx.AsyncClient(transport=transport(mode=LlmCacheMode.REPLAY_OR_RECORD))
# hand `client` to AsyncOpenAI(http_client=client) or any OpenAI-shaped SDK
```

Modes: `RECORD`, `REPLAY` (miss raises `LlmCacheMiss` — fail loud), `REPLAY_OR_RECORD`,
`BYPASS` (default). Set per-run via `ATHOME_LLMCACHE_MODE=replay` or `[llmcache]`
in config.toml. The retry layer sits below the cache: only final responses persist.

## Inspect

```bash
athome cache stats [--json]
```

## Rules

- Namespaces are per-concern (`"ingest"`, `"ocr"`), not per-repo — sharing hits
  across repos is the point.
- No gc/eviction exists (deliberate: eviction breaks replay consumers). Don't
  build one; don't delete entries by hand mid-run.
- Cache values must be derivable: never cache anything you can't regenerate by
  re-running the function.
