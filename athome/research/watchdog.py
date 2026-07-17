from __future__ import annotations

import fcntl
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import anyio

from athome import launchd
from athome.config import AthomeSettings, load
from athome.progress import append_line
from athome.research import nightly
from athome.research.errors import ResearchError
from athome.research.journal import CC_NOTES_BIN
from athome.research.spec import ExperimentSpec

if TYPE_CHECKING:
    from collections.abc import Callable

WATCH_INTERVAL = launchd.Interval(seconds=600)
WATCH_LABEL_PREFIX = f"{launchd.LABEL_NAMESPACE}research.watch."
CC_NOTES_LABEL = "athome-research-watchdog"


class WatchdogStateError(ResearchError):
    """A persisted watchdog state file is malformed."""


@dataclass(frozen=True, slots=True)
class OffsetState:
    """A path's persisted byte-offset observation.

    Attributes:
        offset: The path's byte size at the latest check, or zero when missing.
        last_growth_ts: Unix timestamp when byte growth was last observed.
        checked_at: Unix timestamp of the latest check.
        device: Filesystem device for the observed path, or ``None`` when missing.
        inode: Filesystem inode for the observed path, or ``None`` when missing.
    """

    offset: int
    last_growth_ts: float
    checked_at: float
    device: int | None
    inode: int | None


@dataclass(frozen=True, slots=True)
class WatchResult:
    """The liveness and quiet-alarm result for one watchdog check.

    Attributes:
        live: Whether the experiment lock is held exclusively by a run.
        alarm: Whether the live run exceeded the quiet interval.
    """

    live: bool
    alarm: bool


def probe_live(lock_path: Path) -> bool:
    """Checks whether a writer holds ``lock_path`` without blocking or taking its lock.

    Args:
        lock_path: The experiment lock file probed with ``LOCK_SH | LOCK_NB``.

    Returns:
        ``True`` when an exclusive holder prevents the shared probe, otherwise ``False``.
    """
    fd = os.open(lock_path, os.O_RDONLY | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def _state_path(path: Path) -> Path:
    return Path(f"{path}.watch-state.json")


def _decode_offset_state(record: object, path: Path) -> OffsetState:
    match record:
        case {
            "offset": int(offset),
            "last_growth_ts": int(last_growth_ts) | float(last_growth_ts),
            "checked_at": int(checked_at) | float(checked_at),
            "device": int(device),
            "inode": int(inode),
        } if (
            type(offset) is int
            and type(device) is int
            and type(inode) is int
            and not isinstance(last_growth_ts, bool)
            and not isinstance(checked_at, bool)
            and offset >= 0
        ):
            return OffsetState(offset, float(last_growth_ts), float(checked_at), device, inode)
        case {
            "offset": int(offset),
            "last_growth_ts": int(last_growth_ts) | float(last_growth_ts),
            "checked_at": int(checked_at) | float(checked_at),
            "device": None,
            "inode": None,
        } if (
            type(offset) is int
            and not isinstance(last_growth_ts, bool)
            and not isinstance(checked_at, bool)
            and offset >= 0
        ):
            return OffsetState(offset, float(last_growth_ts), float(checked_at), None, None)
        case _:
            raise WatchdogStateError(f"invalid watchdog offset state in {path}")


async def _read_offset_state(path: Path) -> OffsetState | None:
    try:
        payload = await anyio.Path(path).read_text()
    except FileNotFoundError:
        return None
    record: object = json.loads(payload)
    return _decode_offset_state(record, path)


async def _write_json(path: Path, record: object) -> None:
    await anyio.Path(path.parent).mkdir(parents=True, exist_ok=True)
    await anyio.Path(path).write_text(json.dumps(record) + "\n")


async def _snapshot(path: Path) -> tuple[int, int | None, int | None]:
    try:
        stat = await anyio.Path(path).stat()
    except FileNotFoundError:
        return 0, None, None
    return stat.st_size, stat.st_dev, stat.st_ino


async def observe_progress(path: Path, *, now: float) -> OffsetState:
    """Persists and returns one append-only path's byte-offset progress.

    A new inode is measured from offset zero. Same-inode truncation resets the
    stored offset without counting as growth, and a missing path is stored at
    offset zero.

    Args:
        path: Any journal-shaped, ledger-shaped, or log-shaped append-only file.
        now: Injected Unix timestamp for this observation.

    Returns:
        The updated persisted observation.
    """
    state_path = _state_path(path)
    previous = await _read_offset_state(state_path)
    offset, device, inode = await _snapshot(path)
    baseline = (
        0
        if previous is not None and device is not None and (device, inode) != (previous.device, previous.inode)
        else previous.offset
        if previous is not None
        else offset
    )
    state = OffsetState(
        offset=offset,
        last_growth_ts=now if previous is None or offset > baseline else previous.last_growth_ts,
        checked_at=now,
        device=device,
        inode=inode,
    )
    await _write_json(state_path, asdict(state))
    return state


def _decode_live_since(record: object, path: Path) -> float | None:
    match record:
        case {"live_since_ts": None}:
            return None
        case {"live_since_ts": int(live_since) | float(live_since)} if not isinstance(live_since, bool):
            return float(live_since)
        case _:
            raise WatchdogStateError(f"invalid watchdog liveness state in {path}")


async def _read_live_since(path: Path) -> float | None:
    try:
        payload = await anyio.Path(path).read_text()
    except FileNotFoundError:
        return None
    record: object = json.loads(payload)
    return _decode_live_since(record, path)


async def _observe_live_since(lock_path: Path, *, live: bool, now: float) -> float | None:
    state_path = _state_path(lock_path)
    previous = await _read_live_since(state_path)
    live_since = previous if live and previous is not None else now if live else None
    await _write_json(state_path, {"live_since_ts": live_since})
    return live_since


def _run_log_path(name: str) -> Path:
    return load(AthomeSettings).logs_root / f"{nightly.RESEARCH_LABEL_PREFIX}{name}.log"


def _events_path(journal: Path) -> Path:
    return journal.with_name(f"{journal.stem}.events.jsonl")


async def _alert(journal: Path, *, unit: str, detail: str) -> None:
    await append_line(_events_path(journal), {"type": "quiet_alarm", "unit": unit, "detail": detail})
    await anyio.run_process(
        [
            str(CC_NOTES_BIN),
            "note",
            "add",
            f"athome quiet alarm [{unit}]",
            "--body",
            detail,
            "--label",
            CC_NOTES_LABEL,
        ]
    )


async def check(
    spec: ExperimentSpec,
    *,
    repo: Path,
    quiet_s: float = 5400.0,
    now: Callable[[], float] = time.time,
) -> WatchResult:
    """Checks one experiment for live-but-quiet journal and launchd-log progress.

    Args:
        spec: The experiment whose lock and append-only outputs are checked.
        repo: The experiment's git repository.
        quiet_s: Seconds a live run may show no byte growth before alerting.
        now: Injectable Unix-time source.

    Returns:
        Whether the run is live and whether this check raised a quiet alarm.
    """
    checked_at = now()
    journal = await nightly.journal_path(repo, spec.name)
    await anyio.Path(journal.parent).mkdir(parents=True, exist_ok=True)
    lock_path = journal.with_suffix(".lock")
    live = probe_live(lock_path)
    progress = (
        await observe_progress(journal, now=checked_at),
        await observe_progress(_run_log_path(spec.name), now=checked_at),
    )
    live_since = await _observe_live_since(lock_path, live=live, now=checked_at)
    quiet_since = max(
        *(state.last_growth_ts for state in progress),
        live_since if live_since is not None else checked_at,
    )
    alarm = live and checked_at - quiet_since >= quiet_s
    if alarm:
        await _alert(
            journal,
            unit=spec.name,
            detail=f"no journal or launchd log growth for {checked_at - quiet_since:.0f}s while the experiment lock is held",
        )
    return WatchResult(live=live, alarm=alarm)


async def install(spec_path: Path, *, interval: launchd.Interval = WATCH_INTERVAL) -> Path:
    """Installs a launchd Interval agent that checks the experiment's quiet alarm.

    The agent runs ``athome research watch <spec>`` every ten minutes by default;
    a re-install replaces the existing watchdog for the same experiment.

    Args:
        spec_path: The experiment TOML watched by the agent.
        interval: How often to check; defaults to 600 seconds.

    Returns:
        The path of the written launchd plist.
    """
    spec = ExperimentSpec.load(spec_path)
    agent = launchd.AgentSpec(
        label=f"{WATCH_LABEL_PREFIX}{spec.name}",
        command=("athome", "research", "watch", str(spec_path.resolve())),
        schedule=interval,
        working_dir=await nightly.repo_root(spec_path),
    )
    return await launchd.install(agent)
