# Phase A review findings (accepted)

Two codex finder passes (security/subprocess + correctness/atomicity) over the v0.1
core. fable synthesis: accept the list below; fix before tagging v0.1.0. Each fix
lands with a regression test that fails before and passes after.

## wire + workers (`athome/wire.py`, `athome/workers.py`)

- **W1 — pickle deserializes before validation.** `decode()` runs `pickle.loads()`
  then `validate()`; a `__reduce__` gadget executes during `loads`. Validation must
  gate deserialization, not follow it. Fix: a restricted `pickle.Unpickler` whose
  `find_class` refuses everything outside a small builtin allowlist (the Wire type
  is primitives + list/tuple/dict only), so a hostile frame raises `WireError`
  instead of executing. Matters at the Phase B Modal boundary where the "child" is
  a remote container.
- **W2 — unbounded frame length.** A child sending a `0xffffffff` prefix forces
  multi-GB buffering / holds the lease forever. Fix: `MAX_FRAME_BYTES` cap; a prefix
  over it raises `WorkerCrashed` (or `WireError`) rather than allocating.
- **W3 — handshake not failure-atomic.** After `HandshakeMismatch`, the process,
  pipes, and stderr thread stay installed, so the next `call()` skips the handshake
  and talks to the incompatible worker. Fix: tear down (kill + join + null out) in
  the mismatch path before raising.
- **W4 — pool acquisition not cancellation-safe.** Cancellation after the semaphore
  acquire but before the guard lock leaks a permit; `size` repeats deadlock the pool
  with workers free. Fix: acquire/release in a shielded try/finally so a cancel
  returns the permit.

## detach (`athome/detach.py`)

- **D1 — `name` shell-injects.** `name` is interpolated unquoted into the sentinel
  echo inside `/bin/sh -c`; `name='x$(touch /tmp/pwn)'` executes. Fix: validate
  `name` against `^[A-Za-z0-9._-]+$` at the API boundary (it's also a filename), and
  `shlex.quote` it in the composed string as defense in depth.

## launchd (`athome/launchd.py`)

- **L1 — `label` used as a filesystem path.** `uninstall("/tmp/victim")` unlinks
  `/tmp/victim.plist`. Fix: validate `label` (reverse-DNS, `^[A-Za-z0-9.-]+$`, no
  slashes) in `AgentSpec`, `install`, `uninstall`, `status`.
- **L2 — uninstall unlinks after a transient bootout failure.** Ignoring *absent*
  is intended; ignoring a real launchctl failure then unlinking orphans a running
  agent with no on-disk definition. Fix: distinguish "not loaded" (ignore) from a
  genuine bootout error (raise `LaunchdError`, do not unlink).

## sync (`athome/sync.py`)

- **S1 — rsync operands unprotected.** No `--` end-of-options (leading-dash paths
  become flags) and remote args ride the remote shell (`host:/safe; touch /tmp/pwn`
  executes). Fix: `--` before operands; for remote endpoints pass `-e ssh` and never
  compose the remote side through a shell; reject paths/hosts with shell metacharacters.
- **S2 — `--move` TOCTOU.** A source file created/modified between verify and
  `find -delete` is deleted unverified. Fix: verify immediately before delete under
  the same manifest, and delete only files present in the verified manifest.
- **S3 — IPv6 bracket parse.** First-colon split breaks `user@[::1]:/dst`. Fix:
  parse `[...]:` bracketed hosts before the colon split. (Minor.)

## cache (`athome/cache.py`)

- **C1 — path traversal / arbitrary overwrite.** `namespace`/digest components are
  unchecked: `Cache.open("../../x")` escapes `cache_root`, and a hand-built
  `CacheKey(digest="/etc/…")` lets `put_bytes` replace an arbitrary path. Fix:
  validate `namespace` (`^[A-Za-z0-9._-]+$`) in `open`, and assert `digest` is
  lowercase hex of the expected length in `entry_path`.
- **C2 — stale-tmp sweep can delete a live staging dir.** A staging dir written for
  >24h is swept by a concurrent `Cache.open`. Fix: sweep by age *and* skip anything
  under an active `write()` (track open staging paths, or use a sweep threshold well
  above any real write and document it). (Minor — long single writes are rare.)
- **C3 — key non-uniqueness.** Closures sharing a qualname collide; distinct NaN
  payloads all encode `f:nan`. Fix: encode floats via `struct`/`float.hex` so NaN
  bit-patterns differ; document that `@cached` keys on qualname (module-level
  functions only, not per-instance closures).

## llmcache (`athome/llmcache.py`)

- **M1 — key omits scheme and port.** `http://h`, `https://h`, `https://h:8443`
  collide. Fix: include scheme and port in the key tuple.
- **M2 — `RECORD` never refreshes.** `INSERT OR IGNORE` keeps the stale row while
  returning the fresh upstream response, so later `REPLAY` serves stale data. Fix:
  `INSERT OR REPLACE` in `RECORD` (REPLAY_OR_RECORD still writes only on miss).
- **M3 — header fidelity.** Duplicate response headers (e.g. `Set-Cookie`) coalesce
  on replay. Fix: store headers as an ordered multi-item list and rebuild
  `httpx.Headers` from it. (Note the `stream=True` full-buffering behavior in the
  docstring — acceptable for a record/replay cache, but stated.)

## store (`athome/store.py`)

- **T1 — execute/commit not task-atomic on the shared connection.** A cancelled
  task's uncommitted write is persisted by another task's later commit. Fix: serialize
  `execute`/`commit` with an `anyio.Lock` on the `Store`, and document single-writer
  semantics.

## progress (`athome/progress.py`)

- **P1 — append doesn't preserve record boundaries.** Buffered text writes mean one
  logical record is not one kernel write, so a crash can splice records and the
  truncated-tail loader can eat a valid line. Fix: `os.write(fd, (line+"\n").encode())`
  on an `O_APPEND` fd — one atomic write per record (POSIX guarantees append
  atomicity for a single write under `PIPE_BUF`/regular files).
- **P2 — `done(**extra)` overwrites reserved fields.** `done("a", status="error")`
  marks done in memory but journals an error → pending after restart. Fix: reject
  reserved keys (`unit`, `status`) in `**extra`.
- **P3 — failure budget not recounted on resume.** *Review false positive —
  already satisfied at HEAD.* `RunSink.open` already recounts `failed` records and
  seeds the counter (`test_failure_budget_survives_resume` covers it); no change.
