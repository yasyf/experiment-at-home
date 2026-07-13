# Security model

The research harness runs an LLM agent that edits code and gets scored on the result — a reward-hacking machine by construction. The security model draws one line through it: everything that decides *what counts as a result* is enforced structurally, in the harness; containing a *hostile process* is the operating system's job, and v0.3 deliberately does not attempt it. This page says which side of the line each defense sits on, so nobody mistakes the first list for the second.

## What the harness enforces

The immutability boundary checks exactly the content that would be committed, before any commit object exists. The harness stages the agent's proposal into a scratch index and diffs that index against the incumbent (`git diff-index --cached`, in `athome/research/loop.py`): every changed path must fall inside the `mutable_paths` allowlist and outside `immutable_paths`, symlinks are rejected by mode, and a rename or deletion of an immutable file or an undeclared new file is a violation. A violating candidate is journaled `DISCARD` without ever being committed, let alone scored — the harness writes the commit object only after the boundary passes, from the same index it just checked.

Scoring never sees a `.git` directory. The candidate agent works in a plain directory materialized from the incumbent with `git archive | tar`, and the scorer runs in a second plain directory materialized the same way from the candidate commit (both extracted through tarfile's `filter="data"`), so only committed content runs and the candidate never possesses a git dir the harness reads. That is what closes the config-RCE class: a linked worktree would leave a shared `.git` the candidate could plant a `core.fsmonitor` or hook in, but here there is no git dir to plant anything in, and the harness's own git calls run hermetic (`GIT_CONFIG_GLOBAL`/`GIT_CONFIG_SYSTEM` pointed at `/dev/null`, `core.hooksPath=/dev/null`).

Around the boundary sit the economic and judgment defenses: work-unit, wall-clock, and dollar budgets with a per-iteration hard-kill timeout (`Budget` in `athome/research/spec.py`); cross-family judge enforcement, so the judging model family never matches the generating one (`athome/research/judge.py`); the golden gate, which blocks all judge spend until the panel agrees with human golden labels (`athome/research/golden.py`); and single-writer discipline on the journal and vote cache, so overlapping batches cannot corrupt the record.

## The operator contract

Two knobs are yours, and the boundary's soundness depends on both. `mutable_paths` must be a tight allowlist: spec load rejects unbounded globs, and the boundary rejects package initializers and site hooks (`__init__.py`, `conftest.py`, `sitecustomize.py`) because Python auto-loads them into any process that imports the package — including your scorer's. And `metric_command` must be a trusted scorer you wrote; the harness verifies where the metric comes from, not what your scorer chooses to execute.

## Deferred: what only an OS sandbox closes

Everything below shares one root cause: candidate code and the harness run as the same UID on the same filesystem. These are open limitations. The boundary does nothing about them, and they stay open until a future sandboxed-execution increment (separate UID, container, or seatbelt profile) lands.

- **Metric authorship.** Any real ML scorer executes candidate code — `metric_command` runs `train.py` — and that code shares the score directory's filesystem. A hostile `train.py` can write a forged metric file and `os._exit(0)` before the trusted scorer overwrites it. The freshness check (delete before, must exist after) proves the file is *fresh*, not that a trusted writer produced it.
- **Process-group escape.** A candidate tool subprocess can `setsid` or double-fork out of the process group the harness kills, outliving both `hard_kill_s` and the driver timeout.
- **Cost forgery via the shared log.** The candidate, same UID, can append a raw `total_cost_usd` line to the detached run log after the real CLI envelope; cost capture takes the last match, so spend is under-reported toward `max_usd`.

The posture, plainly: the built-in defenses target structural reward hacking. A candidate cannot redefine the metric, touch the scorer, edit the eval, or judge itself — the same posture as karpathy/autoresearch. For a truly adversarial code-executing agent, run the harness under an OS sandbox; until the sandboxed-execution increment lands, that containment is yours to provide.
