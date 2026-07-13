from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol
from uuid import uuid4

import anyio
from loguru import logger

from athome.detach import launch, wait

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path

    from athome.research.spec import ExperimentSpec

RUN_PREFIX = "athome-research"
DEFAULT_CLAUDE_COMMAND = ("claude", "-p", "--dangerously-skip-permissions")


class Driver(Protocol):
    """A proposer that edits the mutable files in a worktree and describes its change.

    An implementation receives the generated contract and the worktree path, mutates
    the mutable files in place, and returns a one-line description of the proposal. It
    never writes the metric file — the immutable ``metric_command`` owns that channel.
    """

    async def propose(self, contract: str, workdir: Path) -> str: ...


@dataclass(frozen=True, slots=True)
class StubProposal:
    """One scripted edit the :class:`StubDriver` replays into a worktree.

    Attributes:
        files: Worktree-relative path to file content, written verbatim.
        description: The one-line description the driver returns for this proposal.
    """

    files: Mapping[str, str]
    description: str = "stub proposal"


@dataclass(frozen=True, slots=True)
class StubDriver:
    """A deterministic, LLM-free driver that replays a scripted sequence of edits.

    Each :meth:`propose` writes the next :class:`StubProposal`'s files into the
    worktree, so the greedy loop runs end to end without a live agent. The iterator
    is the driver's only mutable state; the dataclass itself stays frozen.

    Example:
        >>> driver = StubDriver(iter([StubProposal({"train.py": "LOSS = 0.3\\n"})]))
    """

    proposals: Iterator[StubProposal]

    async def propose(self, contract: str, workdir: Path) -> str:
        proposal = next(self.proposals)
        for relative, content in proposal.files.items():
            target = anyio.Path(workdir) / relative
            await target.parent.mkdir(parents=True, exist_ok=True)
            await target.write_text(content)
        return proposal.description


async def changed_files(workdir: Path) -> list[str]:
    listing = (await anyio.run_process(["git", "-C", str(workdir), "status", "--porcelain", "-z"])).stdout.decode()
    return sorted({entry[3:] for entry in listing.split("\0") if len(entry) > 3})


@dataclass(frozen=True, slots=True)
class ClaudeCodeDriver:
    """Drives one proposal per work-unit by running the ``claude`` CLI in the worktree.

    The generated contract is handed to a detached ``claude`` process
    (:func:`athome.detach.launch`) whose working directory is the candidate worktree.
    :meth:`propose` waits for that single run to finish, then builds its description
    from the git diff and the structured metric file alone — never the agent's stdout,
    which stays an untrusted, prompt-injection surface. The ``claude`` CLI is shelled
    out to, never imported, so this driver loads without the ``llm``/``research`` extras.

    Example:
        >>> driver = ClaudeCodeDriver(spec)
        >>> await run(spec, driver=driver, repo=repo)
    """

    spec: ExperimentSpec
    command: tuple[str, ...] = DEFAULT_CLAUDE_COMMAND
    poll: float = 5.0
    timeout_s: float | None = None

    async def propose(self, contract: str, workdir: Path) -> str:
        inner = f"cd {shlex.quote(str(workdir))} && exec {shlex.join([*self.command, contract])}"
        run = await launch(["/bin/sh", "-c", inner], name=f"{RUN_PREFIX}-{self.spec.name}-{uuid4().hex[:12]}")
        if exit_code := await wait(run.name, poll=self.poll, timeout=self.timeout_s):
            logger.warning("claude driver {} exited {}; scoring the worktree as it stands", run.name, exit_code)
        return await self.describe(workdir)

    async def describe(self, workdir: Path) -> str:
        files = ", ".join(await changed_files(workdir)) or "no files"
        match await self.reported_metric(workdir):
            case None:
                return f"claude edited {files}"
            case metric:
                return f"claude edited {files} (reported {self.spec.metric_key}={metric})"

    async def reported_metric(self, workdir: Path) -> float | None:
        metric_file = anyio.Path(workdir) / self.spec.metric_file
        if not await metric_file.exists():
            return None
        return float(json.loads(await metric_file.read_text())[self.spec.metric_key])
