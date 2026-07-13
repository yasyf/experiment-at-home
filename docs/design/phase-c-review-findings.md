# Phase C review findings (accepted)

codex finder over the research harness. fable synthesis: accept the list below —
they concentrate in the harness's core guarantees (the anti-reward-hack immutability
boundary, budget/process control over the detached agent, metric integrity, and
making the judge/golden gates actually *enforced* rather than advisory). Fix with
fail-first regression tests, then a refuter pass. Finder confirmed clean: normal
staged/unstaged/untracked changes are caught, metric stdout is never parsed as a
score, and no argv/shell injection (metric uses an argv vector; claude/worktree/
contract are shell-quoted).

## Immutability gate + scoring — the architectural rework (loop.py, spec.py)

The gate trusts candidate-controlled git state and is checked once, before commit,
against the agent's dirty worktree. **Redesign: score from a fresh checkout of the
candidate commit in a clean environment, and enforce immutability by diffing commit
trees.** This closes most of the cluster at once.

- **IG1 (crit)** — the gate reads the candidate-controlled index, so `assume-unchanged`/
  `skip-worktree` hide edits to tracked immutable files.
- **IG2 (crit)** — `.gitignore`d untracked files are invisible to `git add -A`, so an
  agent-created ignored (auto-loaded) immutable file evades the diff.
- **IG3 (crit)** — a rename reports only the destination, hiding an immutable file's
  deletion/move.
- **IG4 (crit)** — the `mutable_paths` allowlist is never enforced; only `immutable_paths`
  is rejected, so an undeclared new file (e.g. a `json.py` shadowing stdlib, or an edit
  to the scorer's dependencies) forges the metric.
- **IG5 (crit)** — immutability is checked only before commit, never around scoring, so a
  `core.hooksPath` post-commit hook can overwrite the scorer after the check passes.
- **IG6 (high)** — `PurePosixPath.match()` lacks recursive `**` semantics (`eval/**` misses
  `eval/a/b.py`; `**/score.py` misses top-level `score.py`).
- **IG7 (high)** — allowed symlinks can point outside the worktree, so scoring executes
  external content absent from the candidate commit.

**Fix:** (a) after the candidate commits, check out that commit into a *fresh, clean*
worktree and run the metric there (only committed content runs — kills IG1, IG2, IG5,
IG7 by construction); (b) run the metric with hooks disabled and hermetic config
(`GIT_CONFIG_GLOBAL=/dev/null`, `-c core.hooksPath=/dev/null`); (c) enforce immutability
by `git diff-tree --name-status --no-renames base candidate` — every changed path must
be inside `mutable_paths` (allowlist) AND outside `immutable_paths`, catching
renames/deletions (IG3, IG4); (d) use a real recursive glob matcher (`pathspec`, or
stdlib with `**` handling) for both manifests (IG6); (e) reject symlinks among changed
paths (IG7 belt-and-suspenders).

## Metric integrity (loop.py, gate.py)

- **M1 (crit)** — the metric file is neither removed before the run nor proven freshly
  written, so a prewritten/stale `.athome-metric.json` is accepted. Fix: delete it before
  the metric command, require it to exist and be written after the command started (fresh
  path or mtime check).
- **M2 (high)** — metric values aren't required finite and the first candidate is kept
  unconditionally, so `NaN`/`±Inf` becomes the incumbent and can permanently win a
  direction. Fix: a non-finite metric is a CRASH, never a KEEP.

## Budget + process control over the detached agent (loop.py, driver.py, detach.py)

- **BP1 (crit)** — `hard_kill_s` wraps only `metric_command`, not the agent; a driver/agent
  timeout doesn't kill the detached `claude` process, which keeps running and billing. Fix:
  the ClaudeCodeDriver bounds the proposal with a timeout and **kills the detached process**
  on timeout/cancel; wait on process exit, not just a log line.
- **BP2 (crit)** — `max_usd` is stored but never measured or enforced. Fix: capture the
  claude CLI's per-run cost and enforce `max_usd` across units (abort loudly when exceeded);
  journal spend.
- **BP3 (high)** — `max_wall_s` is checked only between units, so an in-flight proposal/metric
  can blow past it. Fix: enforce wall time within a unit (bound the proposal + metric).
- **BP4 (high)** — the completion sentinel is a forgeable log line the agent's stdout shares;
  the agent can emit `ATHOME-RUN-DONE name=<run> exit=0` to fake completion. Fix: the driver
  waits on the detached process's actual exit (pid gone via `detach.running`), not a
  grepped sentinel — and/or `detach` writes the exit code to a separate `<name>.exit` file
  the child can't forge.

## Worktree / resume / crash (loop.py, journal.py, registry.py)

- **WR1 (high)** — a KEEP is journaled before the branch update and resume never reconciles
  the journaled commit with the branch, so a crash between them builds from old code while
  comparing against the recorded metric. Fix: update the branch before journaling the KEEP,
  and reconcile on resume.
- **WR2 (high)** — no per-experiment single-writer lock/atomic unit claim; two invocations
  both claim unit 0, double spend, append duplicate rows, and race the branch. Fix: an
  flock'd experiment lock (or atomic unit claim).
- **WR3 (high)** — candidate-caused exceptions (empty-commit failure, missing/invalid metric
  JSON) abort the whole overnight run instead of journaling a CRASH and continuing. Fix:
  per-unit try → journal CRASH → continue (the "NEVER STOP" overnight contract).
- **WR4 (med)** — concurrent registry promotions share a fixed `.current.tmp` staging symlink;
  one can clobber another's staging. Fix: a unique temp name per promotion.

## Judge + golden — make the gates enforced, not advisory (judge.py, golden.py)

- **JG1 (high)** — vote-cache keys omit the reference, dataset/config digest, prompt version,
  verdict model, and judge identity, so a cached WIN is reused against a new reference and
  cached healthy controls mask a now-poisoned judge. Fix: include all of these in the key.
- **JG2 (high)** — cross-family enforcement is an unused opt-in helper; `verdict()`/`pairwise_vote()`
  don't take the generator's family, so an Anthropic output can be graded by an Anthropic
  judge. Fix: thread the candidate family into the vote path and refuse same-family
  (normalize aliases before the equality check).
- **JG3 (high)** — `run_controls()` returns an unchecked report; a poisoned judge that fails
  garbage controls still returns usable votes unless the caller remembers `report.check()`.
  Fix: the judge **raises** (refuses to vote) when controls fail — enforced, not advisory.
- **JG4 (high)** — golden labels aren't verified against the manifest, and `AgreementReport.check()`
  doesn't require `report.n == gate.n`, so a 4-row subset passes a 6-row/floor-4 gate. Fix:
  verify labels against the manifest sha256 and require `report.n == gate.n`.
- **JG5 (high)** — the golden gate isn't wired to any judging/spending entrypoint, so callers
  can buy votes without it ever blocking. Fix: the judge spend path checks the golden gate
  (a red/missing gate blocks spend).
