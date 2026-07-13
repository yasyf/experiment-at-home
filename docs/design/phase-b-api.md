# athome Phase B API design (serve · llm · ocr · modal · data)

The binding public-surface contract for v0.2. Same rules as `core-api.md`
(frozen dataclasses+slots, keyword-only options, async-only SDK, pydantic-settings
per-module sections, no `os.environ` outside config, typed errors rooted at
`AthomeError`, no defensive coding, Google docstrings on public API only, imports
clean on 3.14t — heavy deps behind extras with lazy imports).

Phase B adds native-dep features. **Every heavy import is lazy, inside a function
body, behind an extra.** The 3.14t CI job imports only what stays pure-Python; the
serve/ocr/llm modules must import (module-load) without their extras installed —
heavy imports happen at call time. Extras (in `pyproject.toml`):

```toml
[project.optional-dependencies]
serve = ["httpx>=0.28"]                      # client only; rapid-mlx/mlx-vlm are `uv tool`/subprocess, not deps
llm   = ["spawnllm>=0.5.5"]
ocr   = ["ocrmac>=1.0"]                      # Apple Vision; VLM via the mlx-vlm serve recipe (subprocess)
modal = ["modal>=0.66"]
hf    = ["huggingface-hub>=0.26"]
embed = ["numpy>=2", "sentence-transformers>=3"]  # sentence-transformers optional-within-extra (lazy)
```

The `athome-ocr-paddle` sidecar is a **separate dist** (`engines/ocr-paddle/`,
`requires-python >=3.13,<3.14`) — never in athome's own deps.

Module → file map:

| Module | Files | CLI |
|---|---|---|
| serve | `athome/serve.py` | `athome serve up\|down\|status <recipe>`, `athome status` |
| llm lanes | `athome/llm/__init__.py` | — |
| llm batch | `athome/llm/batch/{__init__,state,anthropic,openai,gemini}.py` | `athome batch submit\|status\|collect` |
| llm cost | `athome/llm/{pricing,spend,telemetry}.py` | — |
| ocr engines | `athome/ocr/{apple,paddle,vlm,ensemble,merge,profiles}.py` | `athome ocr <img> --profile` |
| paddle sidecar | `engines/ocr-paddle/` (own pyproject) | (sidecar entrypoint) |
| modal | `athome/modal.py` | — |
| hf | `athome/hf.py` | `athome hf pull <repo>` |
| embed | `athome/embed.py` | — |
| bakeoff | `athome/bakeoff.py` | `athome bakeoff run <spec>` |

---

## serve — `athome/serve.py`

Generic managed-process lifecycle for OpenAI-compatible local servers, driven by
**config recipes** (no engine ever enters athome's deps — rapid-mlx/mlx-vlm are
installed via `uv tool` / run as subprocesses).

```python
class ServeError(AthomeError): ...
class HealthTimeout(ServeError): ...

class RapidMlxSettings(SectionSettings):
    section = ("serve", "rapid-mlx")
    version: str            # exact pin — rapid-mlx ships daily; REQUIRED (no default)
    model: str
    port: int = 8400

class MlxVlmSettings(SectionSettings):
    section = ("serve", "mlx-vlm")
    model: str = "mlx-community/dots.ocr-4bit"
    port: int = 8401

class LlamaServerSettings(SectionSettings):
    section = ("serve", "llama-server")
    command: str            # full command string (bake-off comparison arm)
    port: int = 8402

type Recipe = Literal["rapid-mlx", "mlx-vlm", "llama-server"]

@dataclass(frozen=True, slots=True)
class ServerHandle:
    recipe: Recipe
    port: int
    pid: int | None         # None when run as a launchd KeepAlive service
    base_url: str           # http://127.0.0.1:<port>/v1

@dataclass(frozen=True, slots=True)
class ManagedServer:
    """A recipe-configured OpenAI-compatible local server with lifecycle + health."""

    recipe: Recipe

    async def ensure(self, *, persistent: bool = False) -> ServerHandle: ...
        # persistent=False: spawn detached via athome.detach (own log), wait for health;
        # persistent=True: install a launchd KeepAlive agent (athome.launchd) and wait.
        # Idempotent: a healthy server on the port returns its handle without respawning.
        # Command per recipe:
        #   rapid-mlx  -> ("uvx","--from",f"rapid-mlx=={ver}","rapid-mlx","serve",model,"--port",port)
        #   mlx-vlm    -> ("uvx","mlx-vlm.server","--model",model,"--port",port) (lazy: exact entrypoint resolved at build)
        #   llama-server-> shlex.split(settings.command)
    async def health(self) -> bool: ...            # GET /v1/models or /health, 1s timeout
    async def stop(self) -> None: ...              # detach kill or launchd uninstall
    def client(self, *, cached: bool = False) -> AsyncOpenAI: ...
        # AsyncOpenAI(base_url=handle.base_url, api_key="local",
        #   http_client=httpx.AsyncClient(transport=athome.llmcache.transport() if cached else None))

async def up(recipe: Recipe, *, persistent: bool = False) -> ServerHandle: ...
async def down(recipe: Recipe) -> None: ...
async def probe_all() -> list[tuple[Recipe, bool]]: ...   # health of every configured recipe

# CLI `cli` group: serve up|down|status RECIPE. Plus a TOP-LEVEL `athome status`
# verb (wired in athome/cli.py lazy map): launchd agents (label/pid/last-log) +
# probe_all() health rows. Add "status": ("athome.serve:status_cli", "...") to main.
```

`ManagedServer.ensure` uses `athome.detach.launch` (name = `f"serve-{recipe}"`) or
`athome.launchd.install`. Health-poll loop: `HealthTimeout` after N seconds.

---

## llm lanes — `athome/llm/__init__.py`

The codified call policy — visible, greppable, telemetered. Behind the `llm` extra.
Every lane routes through `telemetry.CallLog` + `spend` guard (the value over a bare
spawnllm import).

```python
async def small(prompt: str, *, model: Tier = "small", timeout: int = 180) -> str: ...
    # thin spawnllm.call passthrough (interactive/cheap tiers), wrapped in telemetry+spend

async def extract[T: BaseModel](prompt: str, schema: type[T], *, model: Tier = "large",
                                timeout: int = 180) -> T: ...
    # spawnllm.extract passthrough

async def local[T: BaseModel](prompt: str, *, schema: type[T] | None = None,
                              recipe: Recipe = "rapid-mlx") -> str | T: ...
    # calls the rapid-mlx recipe's endpoint through spawnllm's NEW OpenAI-endpoint
    # backend (the spawnllm PR below). base_url from ManagedServer(recipe).ensure().
    # schema=None -> str; schema given -> structured output validated to T.
    # Record/replay-cacheable: the client wraps athome.llmcache.transport().

TierName = Literal["small", "medium", "large"]   # re-exported alias of spawnllm.TModel
```

`batch` is exposed as `athome.llm.batch` (submodule, below) — NOT a lane function
(the interaction model differs: submit now, collect tomorrow).

### spawnllm PR (separate repo `~/Code/spawnllm`)

Add an **OpenAI-compatible-endpoint backend**: given `base_url` + `model`, it POSTs
to `<base_url>/chat/completions` (and structured output via `response_format`
json_schema) over raw httpx, implementing the `LlmBackend` protocol so
`call`/`extract`/`run` accept it. Provider name `"openai_endpoint"` (or extend
`MlxBackend`'s sibling). athome's `llm.local` constructs it with the recipe's
`base_url`. Ship as a spawnllm minor; athome pins `spawnllm>=0.5.5`. The PR keeps
spawnllm's own test suite green. (Per the ecosystem rule: the *call surface* lives
in spawnllm; the *server lifecycle* lives in athome.serve.)

---

## llm batch — `athome/llm/batch/`

Hand-rolled 3-provider batch adapter on raw httpx (batch endpoints are plain REST;
no provider SDKs). 50%-off, async, correlate ONLY by `custom_id`, own the state file.

```python
# state.py
class BatchError(AthomeError): ...
class BudgetExceeded(BatchError): ...

class BatchStatus(StrEnum):
    PENDING = "pending"; RUNNING = "running"; COMPLETED = "completed"
    FAILED = "failed"; EXPIRED = "expired"      # EXPIRED distinct: unbilled -> safe resubmit

@dataclass(frozen=True, slots=True)
class BatchRequest:
    custom_id: str            # the ONLY correlation key
    body: dict[str, Wire]     # provider chat-completion payload

@dataclass(frozen=True, slots=True)
class BatchResult:
    custom_id: str
    body: dict | None         # None when this item failed/expired
    status: BatchStatus

@dataclass(frozen=True, slots=True)
class BatchJob:
    provider: Provider
    provider_batch_id: str
    state_path: Path          # batches_root/<job>.jsonl
    schema_version: int = 1

class BatchProvider(Protocol):     # anthropic.py / openai.py / gemini.py
    async def submit(self, reqs: Sequence[BatchRequest]) -> str: ...   # -> provider_batch_id
    async def poll(self, batch_id: str) -> BatchStatus: ...
    async def collect(self, batch_id: str) -> list[BatchResult]: ...
    def estimate_usd(self, reqs: Sequence[BatchRequest]) -> float: ...

type Provider = Literal["anthropic", "openai", "gemini"]

# __init__.py — the public batch surface
async def submit(reqs: Sequence[BatchRequest], *, provider: Provider,
                 max_usd: float) -> BatchJob: ...
    # estimate_usd; raise BudgetExceeded if over max_usd BEFORE submit; write JSONL state
    # (schema_version, provider, batch_id, custom_ids, submitted_at placeholder) via progress.RunSink
async def status(job: BatchJob) -> BatchStatus: ...
async def collect(job: BatchJob) -> list[BatchResult]: ...
    # poll -> if COMPLETED: provider.collect, keyed by custom_id, append to state;
    # EXPIRED items -> auto-resubmit with a FRESH custom_id (logged as retry), NOT counted as failure
```

State file is the sole idempotency/resume layer. Provider adapters use each
provider's batch REST (Anthropic Message Batches, OpenAI `/v1/batches` + files,
Gemini batch). Secrets: required env fields on a `BatchSettings` per provider
(`ANTHROPIC_API_KEY` etc.) — fail at startup. CLI `athome batch submit|status|collect`;
a daily `com.athome.batch-collect` launchd agent (morning collection — retention
windows make it load-bearing). Auth keys read via settings, spend routed through
`env_prefix_cmd` where present.

Secrets note: batch settings read the provider key from env (`os.environ["ANTHROPIC_API_KEY"]`
is the sanctioned exception only inside a settings model's required field — implement
as a pydantic field with `validation_alias`, not a bare read).

---

## llm cost — `pricing.py` / `spend.py` / `telemetry.py`

```python
# pricing.py  (donor: wlm lab/pricing.py — unpriced model RAISES, never $0)
class UnpricedModel(AthomeError): ...

@dataclass(frozen=True, slots=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float

PRICES: dict[str, Price] = { ... }     # model id -> Price; extend as needed

def cost(model: str, *, input_tokens: int, output_tokens: int) -> float: ...
    # PRICES[model] or raise UnpricedModel(model)

# spend.py  (donor: lab tinker_pilot spend guard)
class SpendExceeded(AthomeError): ...

@dataclass(slots=True)
class SpendGuard:
    """Aborts loudly when projected + running cost crosses max_usd."""
    max_usd: float
    spent: float = 0.0
    def check(self, projected: float) -> None: ...   # raise SpendExceeded if spent+projected > max_usd
    def record(self, actual: float) -> None: ...

# telemetry.py  (donor: stream-trade CallRecord/CallLog)
@dataclass(frozen=True, slots=True)
class CallRecord:
    model: str; latency_s: float; input_tokens: int; output_tokens: int
    cost_usd: float; served_model: str | None; system_fingerprint: str | None

class CallLog:
    """In-memory + optional JSONL sink of CallRecords; warns on served-model / fingerprint drift."""
    def add(self, record: CallRecord) -> None: ...
    @property
    def total_usd(self) -> float: ...
```

The lanes wire these: each lane estimates cost (`pricing.cost`), `SpendGuard.check`s
before the call, runs it, then `CallLog.add` + `SpendGuard.record`. A process-wide
default `CallLog`/`SpendGuard` is created from `BatchSettings`/an `LlmSettings`
(`ATHOME_LLM_MAX_USD`).

---

## ocr engines — `athome/ocr/`

Types shipped in v0.1 (`ocr/types.py`). Phase B adds engines + ensemble + merge +
profiles. `athome.ocr.read(image, profile=...) -> Document` is the 90% entry.

```python
# apple.py  (ocrmac; GIL-bound -> on 3.14t hosts run via athome.workers sidecar,
#            but ocrmac itself is macOS-only; import lazy inside methods)
@dataclass(frozen=True, slots=True)
class AppleVision:   # implements TokenOcr
    async def tokens(self, image: bytes, *, region: Box | None = None, upscale: float = 1.0) -> tuple[OcrToken, ...]: ...

# paddle.py  (PP-OCRv6 via the athome-ocr-paddle sidecar dist over athome.workers)
@dataclass(frozen=True, slots=True)
class PaddleOcr:     # implements TokenOcr; spawns WorkerSpec(("uvx","athome-ocr-paddle"))
    pool: WorkerPool
    async def tokens(self, image: bytes, *, region: Box | None = None, upscale: float = 1.0) -> tuple[OcrToken, ...]: ...

# vlm.py  (dots.ocr etc. through the mlx-vlm serve recipe -> DocOcr)
@dataclass(frozen=True, slots=True)
class VlmOcr:        # implements DocOcr
    recipe: Recipe = "mlx-vlm"
    async def read(self, image: bytes) -> Document: ...   # posts image to the mlx-vlm OpenAI endpoint

# ensemble.py  (donor SEMANTICS: stream_trade EnsembleTokenOcr — generic primary +
#               cross-validating supplement; NO trading-domain postproc)
@dataclass(frozen=True, slots=True)
class EnsembleTokenOcr:   # implements TokenOcr
    primary: TokenOcr
    supplement: TokenOcr
    async def tokens(self, image: bytes, *, region: Box | None = None, upscale: float = 1.0) -> tuple[OcrToken, ...]: ...
        # union + cross-validate; where primary/supplement disagree on a region, flag low-confidence

# merge.py  (LLM-as-merger on disagreement — feeds divergent outputs + the image)
@dataclass(frozen=True, slots=True)
class LlmMerger:
    async def merge(self, image: bytes, candidates: Sequence[Document]) -> Document: ...
        # only invoked on conflict; uses athome.llm.local (a vision-capable recipe) or extract

# profiles.py
class OcrError(AthomeError): ...

class OcrSettings(SectionSettings):
    section = ("ocr",)
    profile: Literal["realtime", "quality"] = "quality"

async def read(image: bytes, *, profile: Literal["realtime", "quality"] | None = None) -> Document: ...
    # realtime -> EnsembleTokenOcr(PaddleOcr, AppleVision) tokens -> Document (markdown from tokens)
    # quality  -> VlmOcr(dots.ocr).read + AppleVision cross-check; LlmMerger.merge only on disagreement
    # BlobCache-backed: key = cache.key(image_digest, profile, stack_version)
```

CLI: `athome ocr <image|pdf> --profile quality [--json]`. All engine outputs
BlobCache-keyed on image digest + params + stack version.

### paddle sidecar — `engines/ocr-paddle/`

Own dist `athome-ocr-paddle` (`requires-python >=3.13,<3.14`, deps: `rapidocr-onnxruntime`
or `paddleocr`, `pillow`, `numpy`). Vendors `athome/wire.py` (file copy — the wire
vendoring contract). Exposes a PipeWorker `serve(handler)` entrypoint whose handler
has `tokens(jpeg, region, upscale) -> list[OcrToken-as-wire]` and `fingerprint() -> dict`.
Also ships its **Modal app + relay** (below), mirroring stream-trade's `ocr-engine`.
Published to PyPI separately; invoked via `uvx athome-ocr-paddle`.

---

## modal — `athome/modal.py`

Modal as an elastic remote backend, generalizing stream-trade's
`ocr-engine/ppocr_worker/modal_{app,client}.py` (commit 3bd1b9c). Behind `modal` extra.

```python
class ModalError(AthomeError): ...
class ParityMismatch(ModalError): ...     # remote image/params != local pins

@dataclass(frozen=True, slots=True)
class ModalSettings(SectionSettings):
    section = ("modal",)
    app_prefix: str = "athome"

class ModalWorkerTransport:    # implements workers.WorkerTransport
    """Wire-compatible relay to a modal.Cls — a drop-in for a local PipeWorker.

    On first call: invokes the remote fingerprint() (forces hydration so a missing
    token / undeployed app fails BEFORE ready), asserts remote package versions +
    params match local pins -> ParityMismatch on skew. NO local fallback: Modal
    unreachable = the worker dies loudly, same as a crashed local engine.
    """
    async def call(self, method: str, payload: Wire) -> Wire: ...
    async def aclose(self) -> None: ...

@dataclass(frozen=True, slots=True)
class ServiceSpec:
    """Declares the Modal side from the SAME engine module that runs locally."""
    name: str                              # app name = f"{app_prefix}-{name}"
    version_packages: tuple[str, ...]      # parity fingerprint inputs
    params: dict[str, Wire]                # engine params folded into the fingerprint

def image_recipe(spec: ServiceSpec, *, python: str, local_source: str,
                 download_models: Callable) -> object: ...
    # modal.Image.debian_slim(python).uv_sync().add_local_python_source(local_source)
    #   .run_function(download_models)  — bakes HF weights into the image
def fingerprint_for(spec: ServiceSpec) -> dict[str, Wire]: ...
    # {pkg: importlib.metadata.version(pkg) for pkg in version_packages} | spec.params
```

Parity fingerprinting is **mandatory and load-bearing** (byte-comparable
remote/local results for bake-offs + replay caches). The `athome-ocr-paddle`
sidecar ships both transports + its Modal app, selectable per run
(`ST_OCR_BACKEND`-style: an `OcrSettings.backend: Literal["local","modal"]`).
Modal token via env (settings). Out of scope (noted): GPU VLM OCR on Modal, a
serve recipe for Modal GPU endpoints, `Function.map` bulk fan-out.

---

## hf — `athome/hf.py`

HF hub discipline. Behind `hf` extra.

```python
class HfError(AthomeError): ...
class HfAuthError(HfError): ...          # write-role preflight failed

REVISIONS: dict[str, str] = { ... }      # model repo -> pinned commit SHA (threaded into fingerprints)

async def snapshot(repo: str, *, revision: str | None = None) -> Path: ...
    # revision default = REVISIONS[repo] (raise if absent — no silent floating);
    # huggingface_hub.snapshot_download into the HF cache; returns the local path
async def ensure_write_auth() -> None: ...
    # whoami + role check; a read-scoped token that 403s on push must fail HERE, loudly
    # (a read token whoamis fine but 403s on push — the past bite). Raise HfAuthError.
async def push(repo: str, local_dir: Path, *, revision: str = "main") -> None: ...
    # ensure_write_auth() FIRST, then upload_folder
```

Secrets: `HF_TOKEN` required env field. CLI: `athome hf pull <repo> [--revision]`.

---

## embed — `athome/embed.py`

Incremental embedding cache/index (merges cc-steer `exemplars.py` + wlm
`embeddings.py`). Behind `embed` extra (numpy always; sentence-transformers lazy).

```python
class EmbedError(AthomeError): ...

class EmbedBackend(Protocol):
    async def embed(self, texts: Sequence[str]) -> "np.ndarray": ...   # (n, dim) float32

@dataclass(frozen=True, slots=True)
class ApiBackend:      # OpenAI-compatible embeddings endpoint
    base_url: str; model: str
@dataclass(frozen=True, slots=True)
class LocalBackend:    # sentence-transformers (lazy import)
    model: str = "all-MiniLM-L6-v2"

@dataclass(frozen=True, slots=True)
class EmbedIndex:
    """Content-digest-keyed incremental embedding matrix persisted under cache_root."""
    namespace: str
    backend: EmbedBackend
    async def upsert(self, items: Mapping[str, str]) -> None: ...   # id -> text; re-embeds only changed digests
    async def matrix(self) -> "np.ndarray": ...
    def mmr(self, query_vec: "np.ndarray", *, k: int, lambda_: float = 0.5) -> list[str]: ...  # MMR rerank -> ids
```

Matrix persisted via `athome.cache` atomic writes (npz). Only the digest-keyed
incremental index + backend fallback + MMR are shared; scoring/rendering stays in
consumers (both donors are domain-shaped).

---

## bakeoff — `athome/bakeoff.py`

Generic A/B/N endpoint bake-off (donor: lab `WinnerPicker`). **Also the tool that
gates the Phase D llama.cpp→Rapid-MLX swap.**

```python
class BakeoffError(AthomeError): ...

@dataclass(frozen=True, slots=True)
class Arm:
    name: str
    base_url: str          # an OpenAI-compatible endpoint (a serve recipe's client)
    model: str

@dataclass(frozen=True, slots=True)
class BakeoffSpec:
    task: Callable[[AsyncOpenAI, object], Awaitable[dict]]   # runs one corpus item against one arm
    corpus: tuple[object, ...]
    arms: tuple[Arm, ...]
    primary_metric: str
    tiebreak: str | None = None

@dataclass(frozen=True, slots=True)
class ArmResult:
    arm: str
    metrics: dict[str, float]
    per_field_disagreement: dict[str, float]

@dataclass(frozen=True, slots=True)
class Leaderboard:
    results: tuple[ArmResult, ...]
    winner: str
    passed_gate: bool       # statistical go/no-go on the primary metric

async def run(spec: BakeoffSpec) -> Leaderboard: ...
    # each arm over the shared corpus (pipeline/bounded concurrency), per-arm metrics
    # + pairwise agreement + per-field disagreement, WinnerPicker (primary, tiebreak,
    # hard-constraint flag), statistical gate.
```

CLI: `athome bakeoff run <spec-module>`. The Phase D fidelity gate builds a
`BakeoffSpec` with two arms (llama-server recipe vs rapid-mlx recipe) over
stream-trade's fidelity corpus; a regression is stop-and-report.

---

## Build order within Phase B

1. **serve** + the spawnllm PR (llm.local needs the endpoint backend) — unblocks live smokes.
2. **llm** lanes + **cost** (pricing/spend/telemetry) + **batch** adapters.
3. **ocr** engines + ensemble + merge + profiles; **paddle sidecar** dist (local + Modal transports).
4. **modal** transport + service factory.
5. **hf**, **embed**, **bakeoff** (independent; parallel).

## Testing conventions (Phase B)

Mock the external boundary, keep the unit real:
- serve: mock `athome.detach.launch`/`launchd.install` + a fake health endpoint
  (httpx.MockTransport); assert the exact per-recipe command vectors and idempotent
  ensure. One `@pytest.mark.live` smoke actually starts rapid-mlx (skipped without it).
- llm lanes: mock spawnllm.call/extract; assert telemetry + spend-guard wiring.
- batch: httpx.MockTransport fake provider REST; cover submit budget-abort, custom_id
  mapping, `expired`→auto-resubmit-fresh-id, partial failure, state-file resume.
- ocr: mock the engines (fake TokenOcr/DocOcr); test ensemble union/cross-validate,
  merger-only-on-disagreement, profile dispatch, cache keying. `@live` smoke: real
  dots.ocr via mlx-vlm on a fixture page (skipped without the recipe).
- modal: fake `modal.Cls` stub; assert parity fingerprint asserted-before-ready,
  ParityMismatch on skew, no-fallback (unreachable → raises). `@live` smoke gated on
  Modal auth.
- hf: mock huggingface_hub; auth-preflight test (read-scoped token → HfAuthError
  before any push).
- embed: real numpy, fake backend; incremental re-embed only on changed digest, MMR.
- bakeoff: two fake local arms; leaderboard + gate verdict.

`@pytest.mark.live` marks every test needing real MLX/models/API/Modal; CI runs the
mocked suite; live smokes run locally where the environment permits and are reported
per the plan's Phase B verification.
