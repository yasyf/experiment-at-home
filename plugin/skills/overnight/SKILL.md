---
name: overnight
description: >-
  Run scripts overnight or on a schedule on macOS: detach long runs from the
  terminal with completion sentinels, schedule recurring jobs as launchd agents,
  and make sweeps resumable so a crash retries only the failed units. Use
  instead of nohup, screen/tmux, cron, or hand-written plists whenever a
  command should outlive the session or recur on a schedule.
---

# athome overnight runs

Three layers, pick by lifetime: one detached run (`athome run`), a recurring
schedule (`athome.launchd`), resumable progress inside the script
(`athome.progress`). Logs land under `~/.athome/logs` (`logs_root` in config).

## Detach one run (tonight's sweep)

```bash
athome run --detach --name sweep -- uv run python sweep.py
athome run wait sweep     # blocks until the sentinel; exits with the run's code
athome run log sweep      # prints the log path
```

- Names are unique per live run — a duplicate name refuses loudly.
- The log ends with `ATHOME-RUN-DONE name=<name> exit=<code>`; `wait` reads it.
- Machine env prefixes (e.g. spend routing) come from `env_prefix_cmd` in
  `~/.athome/config.toml` — never hardcode them in the command.

## Schedule a recurring job (SDK, not CLI)

```python
from athome.launchd import AgentSpec, Calendar, install

await install(AgentSpec(
    label="com.athome.nightly-sweep",
    command=("uv", "run", "python", "sweep.py"),
    schedule=Calendar(hour=2, minute=30),
))
```

Schedules: `Calendar(hour, minute, weekday=None)`, `Interval(seconds)`,
`KeepAlive()` (restart-on-exit services). Inspect with `athome launchd list`
and `athome launchd status LABEL`; remove with `athome launchd uninstall LABEL`.
KeepAlive agents go quietly stale — check `status` before trusting one.

## Make the script resumable

```python
from athome.progress import WorkSet

work = WorkSet.open(out_dir / "progress.jsonl")
for unit in work.pending(unit_ids):
    try:
        await process(unit)
    except ProcessingError as err:
        await work.error(unit, str(err))   # journaled; retried next run
    else:
        await work.done(unit)
```

Error units count as NOT done — rerunning after a crash retries exactly the
failed and unprocessed units. For append-only result files with a failure
budget, use `athome.progress.RunSink`; for multi-stage gating,
`athome.progress.Phases`.

## Rules

- Detached commands must be self-contained: absolute paths or a `cwd` you set,
  no inherited shell state.
- One job, one label, `com.<project>.<job>` — never reuse a label across
  different commands (launchd orphans the old agent).
- Always write progress via `WorkSet`/`RunSink` for anything longer than a few
  minutes — an unresumable overnight run wastes the night on a crash.
