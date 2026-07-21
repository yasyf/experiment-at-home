from __future__ import annotations

import dataclasses
import hashlib
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, NewType

from athome.store import Store
from athome.train.spec import HfDatasetRef, LocalJsonlRef, Rows
from athome.train.state import handle_from_json, handle_to_json

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from athome.train.spec import DatasetSource, TrainSpec
    from athome.train.state import StateHandle

RunKey = NewType("RunKey", str)

RUNSTATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_state (
    run_key          TEXT NOT NULL PRIMARY KEY,
    spec_fingerprint TEXT NOT NULL,
    step             INTEGER NOT NULL,
    handle           TEXT NOT NULL,
    reference        TEXT,
    cost_usd         REAL NOT NULL,
    status           TEXT NOT NULL,
    updated_at       TEXT NOT NULL
) WITHOUT ROWID;
"""


@dataclass(frozen=True, slots=True)
class RunState:
    """One run's most-progressed persisted checkpoint, keyed by its deterministic identity.

    Attributes:
        run_key: The digest of the spec's identity; the store's primary key.
        spec_fingerprint: The canonical identity string ``run_key`` digests, kept so a resume can
            confirm it is restoring the run it thinks it is.
        step: The absolute step the ``handle`` was saved at; a same-run resume continues from here.
        handle: The policy training state saved at ``step``.
        reference: The DPO reference anchor the run started from, or None when the reference is base.
        cost_usd: The cumulative metered spend at ``step``, carried so ``max_usd`` binds per run.
        status: ``running`` while a run is resumable; ``complete`` once it finished.
        updated_at: When this record was last written.
    """

    run_key: RunKey
    spec_fingerprint: str
    step: int
    handle: StateHandle
    reference: StateHandle | None
    cost_usd: float
    status: Literal["running", "complete"]
    updated_at: datetime


def dataset_identity(dataset: DatasetSource) -> dict[str, object]:
    """The identity of ``dataset`` folded into a run key — a changed corpus mints a fresh key."""
    match dataset:
        case HfDatasetRef(repo=repo, config=config, split=split):
            return {"kind": "hf", "repo": repo, "config": config, "split": split}
        case LocalJsonlRef(path=path):
            return {"kind": "local", "path": str(path)}
        case Rows(examples=examples):
            return {"kind": "rows", "examples": [dataclasses.asdict(example) for example in examples]}


def spec_fingerprint(spec: TrainSpec) -> str:
    """The canonical identity of ``spec`` — everything that changes what a run trains, resume seeds excluded.

    ``resume_from`` and ``resume_reference`` are the continuation seeds, so they are deliberately
    absent: a continued run must still recover *itself* by key, and threading a seed must not mint a
    new identity that can never find its own prior state.
    """
    return json.dumps(
        {
            "name": spec.name,
            "base": dataclasses.asdict(spec.base),
            "method": spec.method,
            "lora": dataclasses.asdict(spec.lora),
            "hyperparams": dataclasses.asdict(spec.hyperparams),
            "dataset": dataset_identity(spec.dataset),
        },
        sort_keys=True,
        default=str,
    )


def run_key(spec: TrainSpec) -> RunKey:
    """The deterministic resume key for ``spec``: a digest over its identity, excluding resume seeds."""
    return RunKey(hashlib.sha256(spec_fingerprint(spec).encode()).hexdigest())


@dataclass(slots=True)
class RunStateStore:
    """The one persistence codepath for run state, aiosqlite-backed and async-native.

    Example:
        >>> async with RunStateStore.open(path) as store:
        ...     await store.put(state)
        ...     restored = await store.get(state.run_key)
    """

    store: Store

    @classmethod
    @asynccontextmanager
    async def open(cls, db_path: Path) -> AsyncIterator[RunStateStore]:
        """Open the run-state store at ``db_path`` with a durability-critical ``synchronous=FULL``."""
        async with Store.open(db_path, schema=RUNSTATE_SCHEMA, synchronous="FULL") as store:
            yield cls(store)

    async def get(self, key: RunKey) -> RunState | None:
        """The persisted run state for ``key``, or None when no run has recorded one."""
        row = await self.store.fetch_one("SELECT * FROM run_state WHERE run_key = ?", (key,))
        if row is None:
            return None
        return RunState(
            run_key=RunKey(row["run_key"]),
            spec_fingerprint=row["spec_fingerprint"],
            step=row["step"],
            handle=handle_from_json(json.loads(row["handle"])),
            reference=handle_from_json(json.loads(row["reference"])) if row["reference"] is not None else None,
            cost_usd=float(row["cost_usd"]),
            status=row["status"],
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    async def put(self, state: RunState) -> None:
        """Upsert ``state`` under its key — the only writer, so two attempts never diverge a record."""
        await self.store.execute(
            """
            INSERT INTO run_state(run_key, spec_fingerprint, step, handle, reference, cost_usd, status, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_key) DO UPDATE SET
                spec_fingerprint = excluded.spec_fingerprint,
                step = excluded.step,
                handle = excluded.handle,
                reference = excluded.reference,
                cost_usd = excluded.cost_usd,
                status = excluded.status,
                updated_at = excluded.updated_at
            """,
            (
                state.run_key,
                state.spec_fingerprint,
                state.step,
                json.dumps(handle_to_json(state.handle)),
                None if state.reference is None else json.dumps(handle_to_json(state.reference)),
                state.cost_usd,
                state.status,
                state.updated_at.isoformat(),
            ),
        )

    async def mark_complete(self, key: RunKey) -> None:
        """Flip ``key``'s run from ``running`` to ``complete`` so a later run never resumes it."""
        await self.store.execute(
            "UPDATE run_state SET status = 'complete', updated_at = ? WHERE run_key = ?",
            (datetime.now(UTC).isoformat(), key),
        )
