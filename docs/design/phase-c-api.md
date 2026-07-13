# athome Phase C API design — the research harness (`athome/research/`)

The autoresearch harness: karpathy's greedy keep/discard loop primitives merged
with cc-steer-lab / write-like-me evaluation machinery, redesigned to athome's
style. v0.3. Same house rules as the other specs (frozen dataclasses+slots,
keyword-only options, async-only SDK, pydantic-settings, typed errors under
`AthomeError`, no defensive coding, 3.14t-clean core with heavy deps lazy behind
extras).

**What we keep from karpathy/autoresearch (design, not code):** mutable-target vs
immutable-evaluator split (the anti-reward-hack boundary), a scored keep/discard
gate over a monotone scalar with git as the ledger, a results journal.
**What we fix (its known bites):** enforce the immutable boundary **structurally**
(a git-diff check, not prose); make budgets **work-unit-first** (wall-clock isn't
cross-machine comparable) with wall-clock + hard-kill as backstops; return the
metric over a **structured channel** (a JSON result file the harness reads — never
grepped from stdout, the prompt-injection surface); **worktree isolation** per
experiment; **resumability + a morning report**; treat run logs as **untrusted**
before they re-enter agent context.

Extra: `research = ["numpy>=2"]` (bootstrap CI); Claude Code driver shells out to
the `claude` CLI (no dep). spawnllm (`llm` extra) powers judges.

Lands in **four increments** (each its own opus agent + verify):
(i) spec + journal + gates + commons/invariants/cells + registry;
(ii) the loop with a stub driver;
(iii) the Claude Code driver + nightly wiring;
(iv) golden labeling + calibration + probes + judge panel.

File map:

| File | Increment | Purpose |
|---|---|---|
| `research/spec.py` | i | `ExperimentSpec` (TOML), budgets, comparability fingerprint |
| `research/journal.py` | i | typed append-only JSONL journal (on `progress.RunSink`) |
| `research/gate.py` | i | monotone metric gate + bootstrap-CI promotion gate + blocking invariants |
| `research/common.py` | i | `canonical_json`, `Hasher`, order-invariant `dataset_digest`, `StratifiedSplitter`, `ConfusionMatrix` |
| `research/invariants.py` | i | blocking pre-flight invariants (draft-diversity, NaN guard, constant-decider) |
| `research/cells.py` | i | sqlite cell index + leaderboard queries |
| `research/registry.py` | i | content-addressed artifact registry + `current`-symlink promote |
| `research/loop.py` | ii | greedy keep/discard loop in a worktree; structural immutability; structured metric channel |
| `research/contract.py` | ii/iii | the agent instruction contract (the `program.md` role), generated |
| `research/driver.py` | ii/iii | `StubDriver` (ii) + `ClaudeCodeDriver` (iii) |
| `research/nightly.py` | iii | launchd Calendar wiring + morning report |
| `research/judge.py` | iv | `spawnllm.extract` pydantic judges + judge-health controls |
| `research/golden.py` | iv | blind labeling packets + panel-vs-human agreement gate |
| `research/calibrate.py` | iv | ceiling/floor normalization + saturation refusals |
| `research/probes.py` | iv | topic-leakage probes |

CLI: `athome research init|run|status|report|nightly install`.

---

## Increment i — spec, journal, gates, commons, registry

### `research/spec.py`

```python
class ResearchError(AthomeError): ...
class ImmutableViolation(ResearchError): ...      # a candidate touched an immutable path
class BudgetExhausted(ResearchError): ...

@dataclass(frozen=True, slots=True)
class Budget:
    max_units: int                    # work-unit iterations — FIRST-CLASS, cross-machine comparable
    max_wall_s: float | None = None   # backstop
    hard_kill_s: float | None = None  # per-iteration hard kill -> crash
    max_usd: float | None = None

@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """A TOML-loaded research experiment (the `program.md`+`prepare.py` split, typed)."""

    name: str
    metric_command: tuple[str, ...]   # writes the structured metric file (NOT stdout-grepped)
    metric_key: str                   # key read from the metric JSON
    direction: Literal["min", "max"]  # monotone gate direction
    mutable_paths: tuple[str, ...]     # globs the agent MAY edit
    immutable_paths: tuple[str, ...]   # globs the agent must NOT edit (the scoring boundary)
    budget: Budget
    metric_file: str = ".athome-metric.json"   # the structured channel the metric_command writes

    @classmethod
    def load(cls, path: Path) -> ExperimentSpec: ...   # tomllib

@dataclass(frozen=True, slots=True)
class Comparability:
    """config_hash + dataset_digest — two runs are comparable iff these match (lab/wlm)."""
    config_hash: str
    dataset_digest: str
```

### `research/journal.py`

```python
class Verdict(StrEnum):
    KEEP = "keep"; DISCARD = "discard"; CRASH = "crash"

@dataclass(frozen=True, slots=True)
class JournalRow:
    unit: int
    commit: str
    metric: float | None      # None on crash
    verdict: Verdict
    resources: dict[str, float]   # wall_s, peak_rss_mb, usd
    description: str

class Journal:
    """Typed append-only JSONL journal on progress.RunSink; journal-BEFORE-result (lab rule).

    Optional cc-notes mirror invokes the INSTALLED cc-notes binary
    (/opt/homebrew/bin/cc-notes) — never `uvx cc-notes` (not on PyPI).
    """
    @classmethod
    def open(cls, path: Path, *, mirror_cc_notes: bool = False) -> Journal: ...
    async def append(self, row: JournalRow) -> None: ...
    def rows(self) -> list[JournalRow]: ...
    def best(self, direction: Literal["min","max"]) -> JournalRow | None: ...
    def resume_unit(self) -> int: ...   # next unit index from the journal (resumability)
```

### `research/gate.py`

```python
def monotone_gate(candidate: float, incumbent: float | None, *, direction: Literal["min","max"],
                  floor: float = 0.0) -> bool: ...
    # KEEP iff candidate strictly beats incumbent by > floor (the greedy loop's decision)

@dataclass(frozen=True, slots=True)
class PromotionVerdict:
    promote: bool
    reason: str

def bootstrap_ci_gate(incumbent: Sequence[float], candidate: Sequence[float], *,
                      direction: Literal["min","max"], n_boot: int = 10_000,
                      floor: float = 0.0) -> PromotionVerdict: ...
    # overlapping CIs -> incumbent stays; candidate promotes only on a clear, floored win
    # (donor: lab run_gate; the judged-experiment promotion gate). numpy (research extra).

def blocking_invariants(rows: Sequence[object], checks: Sequence[Callable]) -> None: ...
    # raise BEFORE any verdict is journaled if any invariant fails (donor: lab invariants.py)
```

### `research/common.py`, `invariants.py`, `cells.py`, `registry.py`

- `common.py`: `canonical_json(obj)->bytes` (sorted keys, stable separators);
  `Hasher` (blake2b accumulator); `dataset_digest(items)` order-invariant
  (sorted per-item hashes); `StratifiedSplitter` with disjointness asserts;
  `ConfusionMatrix`. Donor: `cc-steer-lab/harness/common.py` (ported once already).
- `invariants.py`: blocking pre-flight checks born from real post-mortems —
  `draft_diversity` (reject an N-identical-drafts batch), `nan_guard`,
  `constant_decider` (a judge that returns one label for everything). Each raises
  a typed `ResearchError` subclass.
- `cells.py`: an `athome.store`-backed sqlite cell index (one row per
  (experiment, unit, arm)) + leaderboard queries. Donor: lab `results_store.py`.
- `registry.py`: content-addressed version dirs (`v<NNN>-<date>-<digest12>`) +
  atomic `current`-symlink promote. Donor: `cc_steer/registry.py` (lift the
  mechanics: `VERSION_PATTERN`, promote-by-symlink-rename). Promoted loop winners
  land here.

---

## Increment ii — the loop (stub driver)

### `research/loop.py`

```python
@dataclass(frozen=True, slots=True)
class Driver(Protocol):
    async def propose(self, contract: str, workdir: Path) -> str: ...  # returns a description; edits files in workdir

@dataclass(frozen=True, slots=True)
class LoopResult:
    kept: int
    best: JournalRow | None

async def run(spec: ExperimentSpec, *, driver: Driver, repo: Path) -> LoopResult:
    """Greedy keep/discard loop in a git worktree. Per unit until budget:
    1. worktree add a fresh branch off the incumbent (isolation).
    2. driver.propose(contract, workdir) edits mutable files, returns a description.
    3. STRUCTURAL immutability check: `git diff --name-only` in the worktree; if any
       path matches spec.immutable_paths -> reset the candidate, journal DISCARD
       (ImmutableViolation reason), do NOT run the metric. (Enforced, not prose.)
    4. run spec.metric_command (hard_kill_s timeout -> CRASH); read spec.metric_file
       (the STRUCTURED channel) for metric_key. Never grep stdout. The run log is
       captured but treated as untrusted (never fed back to the driver verbatim).
    5. monotone_gate(candidate, incumbent): KEEP -> commit the worktree onto the
       experiment branch + advance incumbent; DISCARD -> reset.
    6. journal.append(row) BEFORE moving on; budget-low warning injected into the
       next contract. Resumable: on restart, journal.resume_unit() skips done units.
    """
```

Immutability is enforced by the git-diff gate in step 3 — a candidate commit
touching `immutable_paths` is rejected and reset, never scored. The metric only
ever comes from the JSON file the metric_command writes.

### `research/driver.py` (ii) + `research/contract.py`

- `StubDriver` — a deterministic test driver: applies a scripted edit + writes a
  metric file, so the loop is testable end-to-end without an LLM.
- `contract.py` — `build_contract(spec, *, budget_low: bool) -> str`: generates the
  agent instruction spec (the `program.md` role) from the ExperimentSpec —
  mutable/immutable manifest, the metric command + file, the keep/discard rule, the
  simplicity criterion, "return the metric in the JSON file", and (when budget_low)
  the budget-low warning.

---

## Increment iii — Claude Code driver + nightly

- `driver.py` gains `ClaudeCodeDriver`: launches the `claude` CLI detached
  (`athome.detach`) in the worktree with the generated contract as its prompt;
  waits for it to finish one proposal; the driver reads only the metric FILE and the
  git diff, never the agent's stdout, closing the prompt-injection surface.
- `nightly.py`: `install(spec_path)` writes a launchd Calendar agent
  (`athome.launchd`) that runs `athome research run <spec>` overnight; `report()`
  is the morning summary (journal rows, best, kept count, crashes) — CLI
  `athome research report`.

---

## Increment iv — golden labeling + calibration + probes + judges

- `judge.py`: `Judge[T]` wraps `spawnllm.extract` with a pydantic verdict model;
  position-debiased pairwise A/B via a seeded coin-flip; sha256-keyed vote cache
  (re-runs never re-buy votes); `gather_bounded`/`with_backoff`; judge-health
  controls — embedded paraphrase/garbage control pairs that **fail the batch** when
  the judge fails them; cross-family enforcement (a judge may not grade its own
  family). Donors: lab judge kit + wlm cross-family rule.
- `golden.py`: blind labeling packets — outcome-stripped stratified rows, a labels
  template + sha256 manifest, and a panel-vs-human agreement gate that **blocks LLM
  spend until green**. Donor: lab `e10_golden.py`. A cc-present board is the
  labeling UI.
- `calibrate.py`: ceiling/floor normalization with saturation refusals (a metric
  pinned at the ceiling refuses to calibrate rather than reporting a fake spread).
  Donor: wlm `calibrate.py`.
- `probes.py`: topic-leakage probes (detect a judge keying on topic rather than
  quality). Donor: wlm `probes.py`.
- Also absorbs `cc_steer/evaluate.py`'s three-gate eval trio: a **frozen
  golden-fixture mechanical gate** (enforced), a **seeded-sampler precision/
  contamination estimator** cross-validated against a blind stronger-model auditor
  (advisory), and prompt-version-keyed idempotent judge/audit persistence. The
  golden gate is enforced; precision/contamination stay advisory/observed.

---

## Testing conventions (Phase C)

- Sandbox toy repo (a temp git repo with a mutable `train.py` + an immutable
  `score.py` writing a metric JSON): run the loop with `StubDriver` for 3 units;
  assert keep/discard commits, journal rows, a **rejected immutable-path mutation**
  (the structural gate), the **structured-metric channel** (metric read from the
  file, never stdout), and a **budget-cap stop**.
- Golden: a synthetic labeler round-trips a packet; the agreement gate blocks then
  passes.
- gate/common/registry: unit tests with strict assertions (bootstrap-CI overlap →
  incumbent stays; dataset_digest order-invariance; registry promote atomicity).
- All heavy paths (`claude` CLI, spawnllm) mocked; `@pytest.mark.live` for a real
  Claude Code proposal.
