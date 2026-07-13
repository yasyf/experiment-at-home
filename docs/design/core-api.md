# athome core API design (Phase A)

The public-surface contract for the v0.1 core modules. Implementation agents code
against this spec; deviations require stopping and reporting back (a sketch that
doesn't survive contact with the donor semantics is a shape-change, not a detail).

Donor code supplies **semantics and invariants only** — atomic write discipline,
version-bump invalidation, lock-serialized pipes, error-units-retry — never the
API. Every surface below is the redesigned one; donors adapt to it at migration.

House rules that bind every module (see STYLEGUIDE.md):

- `from __future__ import annotations` everywhere; full annotations; no `Any` widening.
- Frozen `@dataclass(frozen=True, slots=True)` for immutable/config data.
- Options and flags are keyword-only (after `*`).
- Async-only SDK: every public I/O function is `async def`; sync exists only at
  the CLI boundary via `athome.cli.coro`. No `_sync` twins.
- No scattered `os.environ` reads — every knob is a field on a settings model
  (`athome.config`). Secrets are required env-sourced fields.
- Fail fast: no fallbacks, no sentinel returns; raise typed errors rooted at
  `athome.cli.AthomeError` (importable from `athome.errors` — see below).
- Docstrings (Google style) on public API only; no comments except TODOs and
  non-obvious workarounds.
- Every module must import cleanly on Python 3.14 free-threaded — pure stdlib +
  the core deps (anyio, click, httpx, loguru, pydantic, pydantic-settings,
  aiosqlite) only.

Module file map (one module = one file unless noted; tests in
`tests/test_<module>.py`):

| Module | Files | CLI verb |
|---|---|---|
| errors | `athome/errors.py` | — |
| config | `athome/config.py` | — |
| cli | `athome/cli.py` | root group |
| store | `athome/store.py` | — |
| cache | `athome/cache.py` | `athome cache stats` |
| llmcache | `athome/llmcache.py` | — |
| workers | `athome/workers.py`, `athome/wire.py` | — |
| progress | `athome/progress.py` | — |
| launchd | `athome/launchd.py` | `athome launchd install\|uninstall\|list\|status` |
| detach | `athome/detach.py` | `athome run --detach -- <cmd>`, `athome run wait <name>` |
| sync | `athome/sync.py` | `athome sync SRC DST [--move]` |
| ocr types | `athome/ocr/types.py` (+ `athome/ocr/__init__.py` re-exports) | — |

## errors — `athome/errors.py`

The exception root. Small, no logic:

```python
class AthomeError(Exception):
    """Root of every athome error; the CLI renders these as clean stderr + exit 1."""

    exit_code: ClassVar[int] = 1
```

Every module defines its typed errors as subclasses (named below per module).

## config — `athome/config.py`

pydantic-settings is the single settings surface. One TOML file
(`~/.athome/config.toml`), one env prefix scheme, per-module section binding.

```python
CONFIG_FILE = Path.home() / ".athome" / "config.toml"

class SectionSettings(BaseSettings):
    """Base for every athome settings model; binds one [section] of ~/.athome/config.toml.

    Precedence: init kwargs > env (ATHOME_<SECTION>_<FIELD>) > TOML section > defaults.
    """

    section: ClassVar[tuple[str, ...]] = ()
    model_config = SettingsConfigDict(frozen=True, extra="ignore")

class AthomeSettings(SectionSettings):
    """Root settings: the shared filesystem roots and the machine env prefix."""

    cache_root: Path = Path("~/.athome/cache")
    logs_root: Path = Path("~/.athome/logs")
    batches_root: Path = Path("~/.athome/batches")
    env_prefix_cmd: str | None = None

def load[S: SectionSettings](model: type[S]) -> S: ...
```

- `load` is the one accessor: `load(AthomeSettings)`, `load(LlmCacheSettings)`.
  It is `@lru_cache`-backed so a settings model is constructed once per process;
  tests call `load.cache_clear()` (the conftest fixture does this).
- Env var derivation: `ATHOME_` + section path upper-cased, `-`/`.` → `_`, then
  `_<FIELD>`. Root section (empty tuple) → `ATHOME_CACHE_ROOT` etc. Implement via
  `settings_customise_sources` + a `TomlConfigSettingsSource` subclass that
  plucks the nested section (`section=("llmcache",)` reads `[llmcache]`). A
  missing config file is fine (defaults apply); a malformed one raises.
- Every `Path` field expands `~` (one shared field validator on `SectionSettings`).
- No other module touches `os.environ` or reads TOML.

Test isolation contract (goes in `tests/conftest.py`, written by the foundations
agent): an autouse fixture that points `ATHOME_CACHE_ROOT`/`ATHOME_LOGS_ROOT`/
`ATHOME_BATCHES_ROOT` at `tmp_path` subdirs via `monkeypatch.setenv`, sets
`ATHOME_CONFIG_FILE`? — no: `CONFIG_FILE` is a module constant; the fixture
monkeypatches `athome.config.CONFIG_FILE` to `tmp_path / "config.toml"` and calls
`load.cache_clear()` before and after. Modules must always resolve roots through
`load(...)` at call time (never import-time constants) so the fixture works.

## cli — `athome/cli.py`

The shared Click idioms, dogfooded by the `athome` CLI itself.

```python
def coro[**P, R](f: Callable[P, Coroutine[Any, Any, R]]) -> Callable[P, R]: ...
    # wraps an async click callback in anyio.run

json_option = click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")

def emit(data: object, *, as_json: bool) -> None: ...
    # as_json → json.dumps(default=str); human → key: value lines / plain rows

class LazyGroup(click.Group):
    """Click group whose subcommands import lazily from 'module:attr' strings."""

class AthomeGroup(LazyGroup):
    """Root group: catches AthomeError from subcommands → loguru error + exit(err.exit_code)."""

main = AthomeGroup(
    name="athome",
    lazy_subcommands={
        "cache": "athome.cache:cli",
        "launchd": "athome.launchd:cli",
        "run": "athome.detach:cli",
        "sync": "athome.sync:cli",
    },
)
```

- Each module with a CLI exposes a `cli` click group/command at module level;
  `LazyGroup` imports it on first use so `athome --help` stays fast and core
  imports stay lean.
- The starter `hello` command is deleted; `tests/test_cli.py` is rewritten to
  assert the group lists the four subcommands and that `AthomeError` renders as
  stderr + exit 1 (invoke a tiny in-test command that raises).
- `coro` uses `anyio.run` (asyncio backend default).

## store — `athome/store.py`

The aiosqlite scaffold every consumer rebuilt. Thin: lifecycle + PRAGMAs +
schema, then get out of the way.

```python
@dataclass(slots=True)
class Store:
    """An open aiosqlite database with WAL, busy_timeout, and a schema applied."""

    db: aiosqlite.Connection

    @classmethod
    @asynccontextmanager
    async def open(cls, path: Path, *, schema: str) -> AsyncIterator[Store]: ...
        # mkdir parents; PRAGMA journal_mode=WAL, synchronous=NORMAL, busy_timeout=5000,
        # foreign_keys=ON; executescript(schema) (schemas are idempotent:
        # CREATE TABLE IF NOT EXISTS ...); row_factory = aiosqlite.Row; commit; close on exit

    async def fetch_one(self, sql: str, params: Sequence[object] = ()) -> aiosqlite.Row | None: ...
    async def fetch_all(self, sql: str, params: Sequence[object] = ()) -> list[aiosqlite.Row]: ...
    async def execute(self, sql: str, params: Sequence[object] = ()) -> None: ...  # commits
```

Anything richer (transactions, upserts) uses `store.db` directly. The
event-journal convention (single write codepath, side effects as listeners) is a
documented rule, not machinery.

Add `aiosqlite>=0.21` to core deps in `pyproject.toml` (foundations agent).

## cache — `athome/cache.py`

One coherent cache subsuming the donor's `BlobCache`/`StreamWriter` split
(donor semantics: `stream_trade/caching.py` — content+version keying, blake2b,
atomic temp-sibling writes, stale-tmp sweeping, version-bump-only invalidation).

```python
@dataclass(frozen=True, slots=True)
class CacheKey:
    digest: str  # blake2b-128 hex

@dataclass(frozen=True, slots=True)
class CacheStats:
    namespace: str
    entries: int
    bytes: int

@dataclass(frozen=True, slots=True)
class Cache:
    """A namespaced, versioned, content-keyed blob/directory cache under cache_root."""

    namespace: str
    version: int
    root: Path

    @classmethod
    def open(cls, namespace: str, *, version: int) -> Cache: ...
        # root = load(AthomeSettings).cache_root / namespace / f"v{version}";
        # mkdir parents; sweep stale *.tmp-* older than 1 day

    def key(self, *parts: bytes | str | int | float | bool | None) -> CacheKey: ...
        # blake2b over a canonical, type-tagged encoding of parts (b"s:" prefix for
        # str, b"b:" for bytes, repr for numbers) — collision-safe across types

    async def get(self, key: CacheKey) -> Path | None: ...        # published entry (file OR dir), None on miss
    async def get_bytes(self, key: CacheKey) -> bytes | None: ...
    async def put_bytes(self, key: CacheKey, data: bytes) -> Path: ...

    @asynccontextmanager
    async def write(self, key: CacheKey) -> AsyncIterator[Path]: ...
        # yields a temp-sibling staging path (caller writes a file there, or
        # mkdir + fill for a directory entry); atomic os.replace publish on clean
        # exit, unlink/rmtree on error. Incremental writers (the StreamWriter
        # case) append to the staging file across the block.

    async def stats(self) -> CacheStats: ...

async def stats_all() -> list[CacheStats]: ...  # every namespace under cache_root

def cached[F](*, ns: str, version: int) -> Callable[[F], F]: ...
    # decorator for async functions: key = cache.key(qualname, canonical(args, kwargs));
    # value stored via pickle in Cache.open(ns, version=version). Args must be
    # hashable primitives/tuples — raise CacheKeyError otherwise (no silent repr()).
```

- Entry layout: `<root>/<digest[:2]>/<digest>`; version lives in the directory
  path so a version bump is the only invalidation. No delete/gc API in v0.1.
- All filesystem writes go through one temp-sibling + `os.replace` codepath.
- Errors: `CacheKeyError(AthomeError)`.
- CLI `cli` group: `athome cache stats [--json]` → table of `stats_all()`.

## llmcache — `athome/llmcache.py`

Record-replay HTTP caching for OpenAI-SDK-shaped clients (donor semantics:
`stream_trade/llm/cache.py` — key excludes headers; retry sits *below* the cache
so only final responses persist).

```python
class LlmCacheMode(StrEnum):
    RECORD = "record"
    REPLAY = "replay"
    REPLAY_OR_RECORD = "replay_or_record"
    BYPASS = "bypass"

class LlmCacheSettings(SectionSettings):
    section = ("llmcache",)
    mode: LlmCacheMode = LlmCacheMode.BYPASS
    schema_version: int = 1

class LlmCacheMiss(AthomeError): ...   # REPLAY mode, key absent

class RetryTransport(httpx.AsyncBaseTransport):
    """Retries 429/5xx/transport errors with capped exponential backoff (default 3 tries)."""

class CachingTransport(httpx.AsyncBaseTransport):
    """Record/replay layer keyed on sha256(schema_version, method, host, path, sorted-query, body)."""
    # headers excluded from the key by design (auth/session noise);
    # only 2xx responses are recorded; sqlite store under
    # load(AthomeSettings).cache_root / "llmcache" / "llmcache.db" via athome.store

def transport(*, mode: LlmCacheMode | None = None) -> httpx.AsyncBaseTransport: ...
    # mode=None → load(LlmCacheSettings).mode; stack: CachingTransport(RetryTransport(httpx.AsyncHTTPTransport()))
    # BYPASS returns RetryTransport(...) only
```

Usage (documented, not wrapped): `AsyncOpenAI(http_client=httpx.AsyncClient(transport=athome.llmcache.transport()))`.
Schema: `responses(key TEXT PRIMARY KEY, status INTEGER, headers TEXT, body BLOB, created_at TEXT)`.
Response replay reconstructs `httpx.Response` with stored status/headers/body.

## workers — `athome/wire.py` + `athome/workers.py`

Sidecar execution (donor semantics: `stream_trade/worker.py` — length-prefixed
framing, lock-serialized round-trips, FD-inheritance hygiene, lazy spawn; pool +
digest-keyed prefetch/lease in `stream_trade/ocr.py`). Parent and sidecar run
*different environments and Python versions* — the wire layer is the contract.

`athome/wire.py` (importable by sidecar dists that don't depend on athome — keep
it dependency-free stdlib):

```python
WIRE_VERSION = 1

type Wire = None | bool | int | float | str | bytes | list[Wire] | tuple[Wire, ...] | dict[str, Wire]

class WireError(Exception): ...  # NOT AthomeError — wire.py must import without athome
# Vendoring contract: sidecar dists copy wire.py into their own tree (file copy,
# stdlib-only). `import athome.wire` from a bare interpreter is NOT a supported
# path — the package __init__ eagerly re-exports Cache and pulls the core deps.

def validate(obj: object) -> Wire: ...      # structural walk; raises WireError on anything else
def encode(obj: Wire) -> bytes: ...          # validate + pickle protocol 5, 4-byte BE length prefix
def read_frame(stream) / write_frame(stream, obj)  # sync file-object versions for the sidecar side
```

`athome/workers.py`:

```python
class WorkerError(AthomeError): ...
class WorkerCrashed(WorkerError): ...        # carries returncode + tail of stderr
class HandshakeMismatch(WorkerError): ...

class WorkerTransport(Protocol):
    async def call(self, method: str, payload: Wire) -> Wire: ...
    async def aclose(self) -> None: ...

@dataclass(frozen=True, slots=True)
class WorkerSpec:
    command: tuple[str, ...]                  # anything spawnable, e.g. ("uvx", "--python", "3.13", "athome-ocr-paddle")
    env: tuple[tuple[str, str], ...] = ()
    cwd: Path | None = None

class PipeWorker:  # implements WorkerTransport
    """A lazily-spawned sidecar subprocess speaking length-prefixed wire frames over stdio."""
    # anyio.open_process, stdin/stdout pipes, stderr → parent stderr passthrough;
    # spawn on first call; handshake: child's first frame is
    # {"wire": WIRE_VERSION, "fingerprint": {...}} — mismatched wire version raises
    # HandshakeMismatch BEFORE the first call returns. Round-trips serialized with
    # one anyio.Lock (interleaved writes deadlock the pipe). Request frame:
    # {"method": str, "payload": Wire}; reply: {"ok": Wire} | {"err": str} (err → WorkerError).

class WorkerPool:
    """N PipeWorkers with a free-queue and digest-keyed prefetch/lease."""
    def __init__(self, spec: WorkerSpec, *, size: int) -> None: ...
    @asynccontextmanager
    async def lease(self, key: str | None = None) -> AsyncIterator[WorkerTransport]: ...
        # key → the worker that prefetched it when free, else any free worker
    async def prefetch(self, key: str, method: str, payload: Wire) -> None: ...
    async def aclose(self) -> None: ...

def serve(handler: object) -> None: ...
    # sidecar main loop (sync; the sidecar owns its own process): handshake frame
    # first ({"wire": WIRE_VERSION, "fingerprint": handler.fingerprint() if defined else {}}),
    # then dispatch frames to handler methods by name; exceptions → {"err": traceback string}.
```

Tests use a tiny inline sidecar script (`sys.executable -c "...serve(Echo())..."`)
— real subprocess, no mocks.

## progress — `athome/progress.py`

The resumable-work substrate all three repos built independently (donor
semantics: stream-trade JSONL skip-lists where error units count as **not**
done; write-like-me `RunSink` incremental crash-safe writes + failure budget;
cc-steer-lab phase gating). Pure stdlib + anyio.

```python
class FailureBudgetExceeded(AthomeError): ...
class PhaseMissing(AthomeError): ...

class WorkSet:
    """A resumable set of work units journaled as JSONL; errors are retried on resume."""

    @classmethod
    def open(cls, path: Path) -> WorkSet: ...          # loads existing journal into memory
    def is_done(self, unit: str) -> bool: ...
    def pending(self, units: Iterable[str]) -> list[str]: ...   # order-preserving
    async def done(self, unit: str, **extra: object) -> None: ...
    async def error(self, unit: str, message: str) -> None: ...  # journaled, still pending on resume

class RunSink:
    """Crash-safe incremental JSONL sink with a failure budget."""

    @classmethod
    def open(cls, path: Path, *, failure_budget: int = 0) -> RunSink: ...
    async def append(self, record: Mapping[str, object]) -> None: ...
    async def fail(self, record: Mapping[str, object]) -> None: ...
        # appended with "failed": true; raises FailureBudgetExceeded once budget exceeded
    @property
    def failures(self) -> int: ...

class Phases:
    """Ordered phase markers gating multi-stage runs."""

    @classmethod
    def open(cls, path: Path) -> Phases: ...
    async def mark(self, name: str) -> None: ...
    def require(self, name: str) -> None: ...          # raises PhaseMissing
    def done(self, name: str) -> bool: ...
```

Every append is one `O_APPEND` write of one `\n`-terminated JSON line + flush —
crash mid-run leaves at most one truncated final line, which loaders skip with a
logged warning (the one sanctioned lenient parse; a truncated tail is an
expected crash artifact, not corruption).

## launchd — `athome/launchd.py`

Declarative launchd agents (donor mechanics: `cc_steer/launchd.py` — plistlib
generation, `/bin/sh -lc '<env-prefix>; exec <cmd>'`, bootout + bootstrap in
`gui/<uid>`; that donor is fully hardcoded, this is the spec-driven redesign).

```python
@dataclass(frozen=True, slots=True)
class Calendar:
    hour: int
    minute: int
    weekday: int | None = None

@dataclass(frozen=True, slots=True)
class Interval:
    seconds: int

@dataclass(frozen=True, slots=True)
class KeepAlive:
    pass

type Schedule = Calendar | Interval | KeepAlive

@dataclass(frozen=True, slots=True)
class AgentSpec:
    label: str                      # e.g. "com.athome.batch-collect"
    command: tuple[str, ...]
    schedule: Schedule
    log_name: str | None = None     # default: label; log at logs_root/<log_name>.log
    working_dir: Path | None = None
    env: tuple[tuple[str, str], ...] = ()

@dataclass(frozen=True, slots=True)
class AgentStatus:
    label: str
    installed: bool
    running: bool
    pid: int | None
    last_exit: int | None

def plist_dict(spec: AgentSpec) -> dict[str, object]: ...
    # match spec.schedule: Calendar → StartCalendarInterval, Interval → StartInterval,
    # KeepAlive → {"KeepAlive": True, "RunAtLoad": True}.
    # ProgramArguments: ["/bin/sh", "-lc", "<env_prefix_cmd>; exec <cmd...>"] when
    # load(AthomeSettings).env_prefix_cmd is set, else the exec form without prefix.
    # StandardOut/ErrorPath → logs_root/<log_name>.log

async def install(spec: AgentSpec) -> Path: ...     # write plist, bootout (ignore absent), bootstrap gui/<uid>
async def uninstall(label: str) -> None: ...        # bootout + unlink plist
async def status(label: str) -> AgentStatus: ...    # parse `launchctl print gui/<uid>/<label>`
def installed(*, prefix: str = "") -> list[str]: ...  # scan ~/Library/LaunchAgents for athome-written plists
```

Plists live at `~/Library/LaunchAgents/<label>.plist`. launchctl invocations run
via `anyio.run_process`; a non-zero bootstrap raises `LaunchdError(AthomeError)`
with stderr attached. CLI: `athome launchd list|status LABEL|uninstall LABEL`
(install is SDK-driven — specs are code; the CLI installs nothing in v0.1).

## detach — `athome/detach.py`

Detached overnight runs on top of `progress` conventions (donor semantics:
stream-trade's nohup + completion-sentinel scripts).

```python
class DetachError(AthomeError): ...

@dataclass(frozen=True, slots=True)
class DetachedRun:
    name: str
    pid: int
    log_path: Path

async def launch(command: Sequence[str], *, name: str) -> DetachedRun: ...
    # spawn via /bin/sh -c '<cmd>; rc=$?; echo "ATHOME-RUN-DONE name=<name> exit=$rc" >> <log>'
    # with start_new_session=True, stdout+stderr appended to logs_root/runs/<name>.log;
    # env_prefix_cmd prepended like launchd. Refuses (DetachError) if a live run
    # with the same name exists (pid file logs_root/runs/<name>.pid).

async def wait(name: str, *, poll: float = 5.0, timeout: float | None = None) -> int: ...
    # tail the log for the sentinel line; returns the exit code; TimeoutError on timeout

def running(name: str) -> int | None: ...   # pid if alive (pid file + kill -0), else None
```

CLI (`cli` group named `run`): `athome run --detach --name NAME -- CMD...`
(prints pid + log path), `athome run wait NAME`, `athome run log NAME`
(prints log path).

## sync — `athome/sync.py`

Verified rsync/migrate: replication that refuses to lose data (semantics from
the stream-trade corpus move: manifest + shasum verification loop that blocks
source deletion until the diff is clean; AppleDouble sweep after every hop).

```python
class SyncVerificationError(AthomeError): ...   # carries the mismatch list

@dataclass(frozen=True, slots=True)
class SyncReport:
    files: int
    bytes: int
    verified: bool
    swept_appledoubles: int

async def mirror(src: Path, dst: str, *, delete_source: bool = False) -> SyncReport: ...
    # 1. rsync -a --exclude '._*' src/ dst/  (dst may be local path or "host:path")
    # 2. sweep `._*` AppleDouble files on dst (find -delete; via ssh for remote)
    # 3. build src manifest: shasum -a 256 over every file, sorted
    # 4. build dst manifest the same way (ssh for remote); diff
    # 5. mismatch → re-rsync the differing files, re-verify (max 3 rounds), then
    #    SyncVerificationError — NEVER delete on an unverified tree
    # 6. delete_source=True and verified → remove src contents
```

Subprocess calls via `anyio.run_process`; require `rsync` and `shasum` on PATH
(raise `SyncVerificationError` naming the missing tool — do not degrade).
CLI: `athome sync SRC DST [--move] [--json]`.

## ocr types — `athome/ocr/types.py`

Protocol types only in v0.1 (engines are Phase B) — they ship early so
stream-trade's re-parent never waits on engines.

```python
@dataclass(frozen=True, slots=True)
class Box:
    x: int
    y: int
    width: int
    height: int

@dataclass(frozen=True, slots=True)
class OcrToken:
    text: str
    box: Box
    confidence: float

@dataclass(frozen=True, slots=True)
class Document:
    """A page read by a document-OCR engine."""

    markdown: str
    tokens: tuple[OcrToken, ...] = ()

    @property
    def text(self) -> str: ...   # markdown stripped of formatting (headings/emphasis/tables → plain lines)

class TokenOcr(Protocol):
    """Token-level OCR: geometry + confidence per token (the ensemble substrate)."""

    async def tokens(self, image: bytes, *, region: Box | None = None, upscale: float = 1.0) -> tuple[OcrToken, ...]: ...

class DocOcr(Protocol):
    """Page-level document OCR (VLM engines): image in, structured markdown out."""

    async def read(self, image: bytes) -> Document: ...
```

`athome/ocr/__init__.py` re-exports all of the above (plain imports, no `__all__`).

## Package `__init__` and docs

`athome/__init__.py` re-exports the stable entry points (plain imports): `load`,
`AthomeSettings`, `Cache`, `cached`, `AthomeError`. Great Docs documents the
rest per-module; the curated `reference:` section in `great-docs.yml` is a
Phase A follow-up (orchestrator-owned), not an implementation-agent concern.

## Testing conventions

- `tests/test_<module>.py`, anyio auto mode (already configured).
- The conftest isolation fixture (see config section) is autouse; no test may
  touch the real `~/.athome`.
- Strict assertions on exact values; mock nothing that can run for real —
  workers tests spawn real subprocesses, launchd tests assert on `plist_dict`
  output and mock only the `launchctl` boundary (`anyio.run_process`), sync
  tests round-trip real temp dirs with an injected corruption.
- Every error path in this spec gets a test (replay miss, wire violation,
  handshake mismatch, failure budget, unverified deletion block, duplicate
  detached run).
