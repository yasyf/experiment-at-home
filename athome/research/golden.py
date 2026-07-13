"""Blind golden-labeling packets and the panel-vs-human agreement gate.

The packet renders, per sampled row, only the outcome-stripped window a labeler
should judge — every outcome-revealing field is withheld by construction, because
the packet contains nothing but ``window_of(row)``. A sha256 manifest pins the
exact rows so a later judge panel provably scores the same rows the human labeled.
The agreement gate then blocks downstream LLM spend until the panel agrees with the
human labels and is not a constant decider. ``verify_packet`` is the gate's entry:
it matches the manifest's ``packet_sha256`` against the rendered packet and mints a
``VerifiedManifest`` that ``read_labels`` and ``prove_gate`` both require, so no label
set and no ``GoldenProof`` can be built from a packet whose content drifted from its
manifest. This module is UI-agnostic (a cc-present board is one labeling front end);
it only samples, renders, and gates. Donor: cc-steer-lab ``e10_golden.py``.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import anyio

from athome.research.common import canonical_json
from athome.research.errors import ResearchError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    Row = Mapping[str, object]

PACKET_NAME = "packet.md"
LABELS_NAME = "labels_template.json"
MANIFEST_NAME = "manifest.json"
LABELS = {"yes": True, "no": False}


class GoldenSampleError(ResearchError):
    """A stratum has fewer eligible rows than the packet requires."""


class GoldenGateViolation(ResearchError):
    """The judge panel disagreed with the human golden labels or is a constant decider; spend stays blocked."""


@dataclass(frozen=True, slots=True)
class Stratum:
    """One sampling stratum: a named pool and how many rows the packet draws from it."""

    name: str
    size: int


@dataclass(frozen=True, slots=True)
class GoldenRow:
    """One packet row: the blind window a labeler judges, plus opaque provenance.

    Attributes:
        number: The 1-based row number shown in the packet.
        row_id: The opaque provenance key linking back to the source row.
        stratum: The stratum the row was drawn from.
        window: The outcome-stripped context the labeler sees — the only source content in the packet.
    """

    number: int
    row_id: str
    stratum: str
    window: str


@dataclass(frozen=True, slots=True)
class GoldenGate:
    """The agreement gate: agree on at least ``floor`` of ``n`` rows, and no constant-decider panel.

    Attributes:
        n: The number of labeled rows.
        floor: The minimum panel-vs-human agreements required to pass.
    """

    n: int
    floor: int


@dataclass(frozen=True, slots=True)
class GoldenPacket:
    """A built labeling packet: the sampled rows and their rendered artifacts.

    Attributes:
        rows: The sampled, numbered, outcome-stripped rows.
        packet_md: The rendered blind labeling document.
        labels_template: The JSON labels template a labeler fills in.
        manifest: The same-rows manifest, including ``packet_sha256`` and the gate.
    """

    rows: tuple[GoldenRow, ...]
    packet_md: str
    labels_template: str
    manifest: dict[str, object]


@dataclass(frozen=True, slots=True)
class AgreementReport:
    """The panel's agreement with the human golden labels over the same rows.

    Attributes:
        n: The rows both the human and the panel labeled.
        agree: The rows they labeled identically.
        panel_constant: Whether the panel returned one label for every row.
    """

    n: int
    agree: int
    panel_constant: bool

    @property
    def agreement_rate(self) -> float:
        return self.agree / self.n if self.n else 0.0

    def check(self, gate: GoldenGate) -> None:
        """Pass the gate, or block spend.

        Raises:
            GoldenGateViolation: the report covers fewer rows than the gate pins, the
                panel is a constant decider, or agreement fell below ``gate.floor``.
        """
        if self.n != gate.n:
            raise GoldenGateViolation(
                f"agreement covers {self.n} rows but the gate pins {gate.n}; an agreeing subset cannot pass"
            )
        if self.panel_constant:
            raise GoldenGateViolation(f"panel is a constant decider over {self.n} rows (one label for everything)")
        if self.agree < gate.floor:
            raise GoldenGateViolation(
                f"panel-human agreement {self.agree}/{self.n} < floor {gate.floor}/{gate.n}; spend stays blocked"
            )


@dataclass(frozen=True, slots=True)
class VerifiedManifest:
    """A manifest whose ``packet_sha256`` matched the rendered packet.

    Constructed only by :func:`verify_packet`, so holding one is proof that the
    packet a labeler saw is the packet the manifest pins. ``read_labels`` and
    :func:`prove_gate` both require it, making packet verification a precondition
    of reading human labels and of building a gate.

    Attributes:
        manifest: The verified manifest mapping.
    """

    manifest: Mapping[str, object]

    @property
    def rows_sha256(self) -> str:
        return cast(str, self.manifest["rows_sha256"])

    @property
    def gate(self) -> GoldenGate:
        return GoldenGate(**cast("Mapping[str, int]", self.manifest["gate"]))


def verify_packet(*, packet_md: str, manifest: Mapping[str, object]) -> VerifiedManifest:
    """Match the rendered packet against its manifest's ``packet_sha256``, or block spend.

    Raises:
        GoldenGateViolation: the packet's content hash differs from the manifest's
            ``packet_sha256`` (the packet drifted from what the manifest pins).
    """
    if (actual := hashlib.sha256(packet_md.encode()).hexdigest()) != manifest["packet_sha256"]:
        raise GoldenGateViolation(
            f"packet content hash {actual} != manifest packet_sha256 {manifest['packet_sha256']!r}"
        )
    return VerifiedManifest(manifest)


@dataclass(frozen=True, slots=True)
class GoldenProof:
    """A golden gate that passed, minted only by :func:`prove_gate`.

    Carrying one is proof the panel agreed with the human golden labels over the
    manifest's pinned rows; a judge batch requires it before any spend.

    Attributes:
        gate: The gate the report cleared, taken from the verified manifest.
        report: The panel-vs-human agreement the gate cleared.
        rows_sha256: The manifest's row fingerprint the labels were pinned to.
    """

    gate: GoldenGate
    report: AgreementReport
    rows_sha256: str

    def check(self) -> None:
        """Re-verify the gate at the trust boundary.

        Raises:
            GoldenGateViolation: the report no longer clears the gate.
        """
        self.report.check(self.gate)


def prove_gate(*, report: AgreementReport, manifest: VerifiedManifest) -> GoldenProof:
    """Mint a :class:`GoldenProof` from a green report; the gate comes from the manifest, never loose.

    Raises:
        GoldenGateViolation: the report fails the manifest's gate (too few
            agreements, a constant-decider panel, or partial row coverage).
    """
    report.check(manifest.gate)
    return GoldenProof(gate=manifest.gate, report=report, rows_sha256=manifest.rows_sha256)


def sample(
    rows: Sequence[Row],
    *,
    strata: Sequence[Stratum],
    stratum_of: Callable[[Row], str],
    window_of: Callable[[Row], str],
    row_id: Callable[[Row], str],
    seed: int,
) -> tuple[GoldenRow, ...]:
    """Draw a deterministic per-stratum sample, then number it under a seeded shuffle.

    ``window_of`` is the sole source of packet content, so outcome-revealing fields
    never enter the packet. ``stratum_of`` buckets rows and ``row_id`` supplies the
    opaque provenance key.

    Raises:
        GoldenSampleError: a stratum holds fewer rows than its draw size.
    """
    rng = random.Random(seed)
    pools: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        pools[stratum_of(row)].append(row)
    picked = [(row, stratum.name) for stratum in strata for row in draw(rng, pools[stratum.name], stratum)]
    rng.shuffle(picked)
    return tuple(
        GoldenRow(number=number, row_id=row_id(row), stratum=stratum, window=window_of(row))
        for number, (row, stratum) in enumerate(picked, start=1)
    )


def draw(rng: random.Random, pool: Sequence[Row], stratum: Stratum) -> list[Row]:
    if len(pool) < stratum.size:
        raise GoldenSampleError(f"stratum {stratum.name!r} needs {stratum.size} rows, has {len(pool)}")
    return rng.sample(list(pool), stratum.size)


def render_packet(rows: Sequence[GoldenRow], *, question: str, header: str) -> str:
    return "\n".join([header, *(render_row(row, question=question) for row in rows)])


def render_row(row: GoldenRow, *, question: str) -> str:
    return "\n".join(
        [
            f"## Row {row.number}",
            "",
            "~~~text",
            row.window,
            "~~~",
            "",
            f"**Q: {question}**",
            "",
            "- Answer (`yes` / `no`): ",
            "",
            "---",
            "",
        ]
    )


def render_labels_template(rows: Sequence[GoldenRow]) -> str:
    return json.dumps([{"row": row.number, "row_id": row.row_id, "label": None} for row in rows], indent=2) + "\n"


def rows_fingerprint(identities: Sequence[tuple[int, str]]) -> str:
    """A sha256 over the sorted ``(number, row_id)`` identities that pins the exact label set."""
    return hashlib.sha256(canonical_json(sorted(identities))).hexdigest()


def build_manifest(
    rows: Sequence[GoldenRow],
    *,
    dataset_digest: str,
    seed: int,
    strata: Sequence[Stratum],
    gate: GoldenGate,
    packet_sha256: str,
) -> dict[str, object]:
    return {
        "seed": seed,
        "dataset_digest": dataset_digest,
        "packet_sha256": packet_sha256,
        "rows_sha256": rows_fingerprint([(row.number, row.row_id) for row in rows]),
        "strata": {stratum.name: stratum.size for stratum in strata},
        "gate": {"n": gate.n, "floor": gate.floor},
        "rows": [{"row": row.number, "row_id": row.row_id, "stratum": row.stratum} for row in rows],
    }


def build_packet(
    rows: Sequence[Row],
    *,
    strata: Sequence[Stratum],
    stratum_of: Callable[[Row], str],
    window_of: Callable[[Row], str],
    row_id: Callable[[Row], str],
    seed: int,
    dataset_digest: str,
    question: str,
    header: str,
) -> GoldenPacket:
    """Sample the rows and render the blind packet, labels template, and same-rows manifest.

    The gate is derived from the sample: ``n`` is the row count and ``floor`` is the
    strict majority (``n // 2 + 1``) unless a caller tightens it via the manifest.
    """
    sampled = sample(rows, strata=strata, stratum_of=stratum_of, window_of=window_of, row_id=row_id, seed=seed)
    packet_md = render_packet(sampled, question=question, header=header)
    gate = GoldenGate(n=len(sampled), floor=len(sampled) // 2 + 1)
    return GoldenPacket(
        rows=sampled,
        packet_md=packet_md,
        labels_template=render_labels_template(sampled),
        manifest=build_manifest(
            sampled,
            dataset_digest=dataset_digest,
            seed=seed,
            strata=strata,
            gate=gate,
            packet_sha256=hashlib.sha256(packet_md.encode()).hexdigest(),
        ),
    )


async def write_packet(packet: GoldenPacket, directory: anyio.Path) -> None:
    """Write ``packet.md``, ``labels_template.json``, and ``manifest.json`` into ``directory``."""
    await directory.mkdir(parents=True, exist_ok=True)
    await (directory / PACKET_NAME).write_text(packet.packet_md)
    await (directory / LABELS_NAME).write_text(packet.labels_template)
    await (directory / MANIFEST_NAME).write_text(json.dumps(packet.manifest, indent=2) + "\n")


async def read_labels(path: anyio.Path, *, manifest: VerifiedManifest) -> dict[str, bool]:
    """Parse a filled labels template into ``row_id -> yes/no`` booleans, verified against the manifest.

    Requiring a :class:`VerifiedManifest` makes packet verification a precondition
    of reading labels. The label file's ``(row, row_id)`` identities are then
    fingerprinted and checked against the manifest's ``rows_sha256``, so a tampered
    or mismatched label set — a dropped row (an easier agreeing subset), an added
    row, or a substituted ``row_id`` — is rejected before a single label is read.

    Raises:
        GoldenGateViolation: the labels do not match the manifest, a row was left
            unlabeled, or a value falls outside ``yes``/``no``.
    """
    entries = json.loads(await path.read_text())
    fingerprint = rows_fingerprint([(entry["row"], entry["row_id"]) for entry in entries])
    if fingerprint != manifest.rows_sha256:
        raise GoldenGateViolation(
            f"labels do not match the manifest: rows_sha256 {fingerprint} != {manifest.rows_sha256!r} "
            "(the label set was tampered or covers a different row set)"
        )
    return {entry["row_id"]: label_bool(entry) for entry in entries}


def label_bool(entry: Mapping[str, object]) -> bool:
    match entry.get("label"):
        case str(value) if value in LABELS:
            return LABELS[value]
        case other:
            raise GoldenGateViolation(f"row {entry.get('row_id')!r} label {other!r} is not 'yes' or 'no'")


def agreement(human: Mapping[str, bool], panel: Mapping[str, bool]) -> AgreementReport:
    """Compare panel labels against the human golden labels over the same rows.

    Raises:
        GoldenGateViolation: the panel did not label exactly the human's rows.
    """
    if human.keys() != panel.keys():
        raise GoldenGateViolation("panel labeled a different row set than the human golden labels")
    return AgreementReport(
        n=len(human),
        agree=sum(1 for key in human if human[key] == panel[key]),
        panel_constant=len(set(panel.values())) == 1,
    )
