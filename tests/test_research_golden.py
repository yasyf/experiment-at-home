from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import anyio
import pytest

from athome.research.golden import (
    AgreementReport,
    GoldenGate,
    GoldenGateViolation,
    GoldenSampleError,
    Stratum,
    VerifiedManifest,
    agreement,
    build_packet,
    prove_gate,
    read_labels,
    rows_fingerprint,
    verify_packet,
    write_packet,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

STRATA = (Stratum("positive", 3), Stratum("negative", 3))
QUESTION = "Was steering warranted here? (yes/no)"
HEADER = "# Golden Packet\n\nLabel each row from its window alone.\n\n---\n"


def make_rows() -> list[dict[str, object]]:
    return [
        {"id": f"p{i}", "kind": "positive", "window": f"positive window {i}", "outcome": "fired"} for i in range(5)
    ] + [{"id": f"n{i}", "kind": "negative", "window": f"negative window {i}", "outcome": "quiet"} for i in range(5)]


def make_packet() -> tuple[list[dict[str, object]], object]:
    rows = make_rows()
    packet = build_packet(
        rows,
        strata=STRATA,
        stratum_of=lambda row: str(row["kind"]),
        window_of=lambda row: str(row["window"]),
        row_id=lambda row: str(row["id"]),
        seed=7,
        dataset_digest="deadbeef",
        question=QUESTION,
        header=HEADER,
    )
    return rows, packet


def verified_with_gate(*, n: int, floor: int) -> VerifiedManifest:
    body = "packet-body"
    return verify_packet(
        packet_md=body,
        manifest={
            "packet_sha256": hashlib.sha256(body.encode()).hexdigest(),
            "rows_sha256": rows_fingerprint([(i, f"r{i}") for i in range(n)]),
            "gate": {"n": n, "floor": floor},
        },
    )


def test_packet_is_outcome_stripped_and_stratified() -> None:
    _, packet = make_packet()
    assert len(packet.rows) == 6
    assert {row.stratum for row in packet.rows} == {"positive", "negative"}
    assert "outcome" not in packet.packet_md
    assert "fired" not in packet.packet_md and "quiet" not in packet.packet_md
    assert all(row.window in packet.packet_md for row in packet.rows)


def test_manifest_pins_the_rows_and_the_packet_sha() -> None:
    _, packet = make_packet()
    assert packet.manifest["packet_sha256"] == hashlib.sha256(packet.packet_md.encode()).hexdigest()
    assert packet.manifest["dataset_digest"] == "deadbeef"
    assert [entry["row_id"] for entry in packet.manifest["rows"]] == [row.row_id for row in packet.rows]
    assert packet.manifest["gate"] == {"n": 6, "floor": 4}


def test_build_is_deterministic_under_the_seed() -> None:
    rows = make_rows()

    def build() -> tuple[str, ...]:
        packet = build_packet(
            rows,
            strata=STRATA,
            stratum_of=lambda row: str(row["kind"]),
            window_of=lambda row: str(row["window"]),
            row_id=lambda row: str(row["id"]),
            seed=7,
            dataset_digest="deadbeef",
            question=QUESTION,
            header=HEADER,
        )
        return tuple(row.row_id for row in packet.rows)

    assert build() == build()


def test_sample_raises_when_a_stratum_is_underfilled() -> None:
    with pytest.raises(GoldenSampleError, match="positive"):
        build_packet(
            make_rows(),
            strata=(Stratum("positive", 99),),
            stratum_of=lambda row: str(row["kind"]),
            window_of=lambda row: str(row["window"]),
            row_id=lambda row: str(row["id"]),
            seed=7,
            dataset_digest="d",
            question=QUESTION,
            header=HEADER,
        )


async def label_from_windows(
    path: anyio.Path, *, warranted: Mapping[str, bool], manifest: VerifiedManifest
) -> dict[str, bool]:
    entries = json.loads(await path.read_text())
    for entry in entries:
        entry["label"] = "yes" if warranted[entry["row_id"]] else "no"
    await path.write_text(json.dumps(entries, indent=2) + "\n")
    return await read_labels(path, manifest=manifest)


async def test_synthetic_labeler_round_trips_a_packet(tmp_path: Path) -> None:
    _, packet = make_packet()
    directory = anyio.Path(tmp_path / "golden")
    await write_packet(packet, directory)

    assert await (directory / "packet.md").read_text() == packet.packet_md
    verified = verify_packet(packet_md=packet.packet_md, manifest=packet.manifest)
    warranted = {row.row_id: row.stratum == "positive" for row in packet.rows}
    labels = await label_from_windows(directory / "labels_template.json", warranted=warranted, manifest=verified)
    assert labels == warranted


async def test_agreement_gate_blocks_then_passes(tmp_path: Path) -> None:
    _, packet = make_packet()
    directory = anyio.Path(tmp_path / "golden")
    await write_packet(packet, directory)
    verified = verify_packet(packet_md=packet.packet_md, manifest=packet.manifest)
    human = {row.row_id: row.stratum == "positive" for row in packet.rows}
    await label_from_windows(directory / "labels_template.json", warranted=human, manifest=verified)

    gate = GoldenGate(n=len(human), floor=packet.manifest["gate"]["floor"])
    disagreeing = {key: not value for key, value in human.items()}
    with pytest.raises(GoldenGateViolation, match="agreement"):
        agreement(human, disagreeing).check(gate)

    agreeing = dict(human)
    agreement(human, agreeing).check(gate)
    assert agreement(human, agreeing).agreement_rate == 1.0


def test_constant_decider_panel_fails_even_when_it_agrees() -> None:
    human = {"a": True, "b": True, "c": False}
    panel = {"a": True, "b": True, "c": True}
    report = agreement(human, panel)
    assert report.panel_constant is True
    with pytest.raises(GoldenGateViolation, match="constant decider"):
        report.check(GoldenGate(n=3, floor=2))


def test_agreement_refuses_a_mismatched_row_set() -> None:
    with pytest.raises(GoldenGateViolation, match="different row set"):
        agreement({"a": True}, {"b": False})


async def test_read_labels_rejects_an_unlabeled_row(tmp_path: Path) -> None:
    path = anyio.Path(tmp_path / "labels.json")
    await path.write_text(json.dumps([{"row": 1, "row_id": "x", "label": None}]))
    manifest = verify_packet(
        packet_md="body",
        manifest={"packet_sha256": hashlib.sha256(b"body").hexdigest(), "rows_sha256": rows_fingerprint([(1, "x")])},
    )
    with pytest.raises(GoldenGateViolation, match="not 'yes' or 'no'"):
        await read_labels(path, manifest=manifest)


async def test_read_labels_rejects_a_tampered_label_set(tmp_path: Path) -> None:
    # JG4: a hostile shrinks the pinned 6-row set to an easier 4-row subset to game the gate.
    _, packet = make_packet()
    directory = anyio.Path(tmp_path / "golden")
    await write_packet(packet, directory)
    verified = verify_packet(packet_md=packet.packet_md, manifest=packet.manifest)
    labels_path = directory / "labels_template.json"

    entries = json.loads(await labels_path.read_text())
    for entry in entries:
        entry["label"] = "yes"
    subset = entries[:4]
    await labels_path.write_text(json.dumps(subset, indent=2) + "\n")
    with pytest.raises(GoldenGateViolation, match="do not match the manifest"):
        await read_labels(labels_path, manifest=verified)

    # a substituted row_id is likewise rejected — the fingerprint pins exact identities.
    substituted = [{"row": e["row"], "row_id": f"forged-{e['row_id']}", "label": "yes"} for e in entries]
    await labels_path.write_text(json.dumps(substituted, indent=2) + "\n")
    with pytest.raises(GoldenGateViolation, match="do not match the manifest"):
        await read_labels(labels_path, manifest=verified)

    # the untouched, complete label set still passes.
    await labels_path.write_text(json.dumps(entries, indent=2) + "\n")
    assert set(await read_labels(labels_path, manifest=verified)) == {row.row_id for row in packet.rows}


def test_check_requires_full_row_coverage() -> None:
    # JG4: an agreeing 4-row subset must NOT pass a 6-row / floor-4 gate.
    with pytest.raises(GoldenGateViolation, match="the gate pins 6"):
        AgreementReport(n=4, agree=4, panel_constant=False).check(GoldenGate(n=6, floor=4))
    # full coverage over the same gate passes.
    AgreementReport(n=6, agree=6, panel_constant=False).check(GoldenGate(n=6, floor=4))


def test_verify_packet_accepts_untouched_and_rejects_tampered() -> None:
    # JG: packet verification gates label-reading and gate-building on the packet's content hash.
    _, packet = make_packet()
    verified = verify_packet(packet_md=packet.packet_md, manifest=packet.manifest)
    assert isinstance(verified, VerifiedManifest)
    assert verified.rows_sha256 == packet.manifest["rows_sha256"]
    assert verified.gate == GoldenGate(n=packet.manifest["gate"]["n"], floor=packet.manifest["gate"]["floor"])
    with pytest.raises(GoldenGateViolation, match="packet content hash"):
        verify_packet(packet_md=packet.packet_md + "x", manifest=packet.manifest)


def test_read_labels_requires_a_verified_manifest() -> None:
    # JG: read_labels no longer accepts a raw dict — packet verification is a precondition.
    import inspect

    assert inspect.signature(read_labels).parameters["manifest"].annotation == "VerifiedManifest"


def test_prove_gate_rejects_a_red_report_and_makes_no_proof() -> None:
    # JG5: a red agreement report can never mint a GoldenProof; the gate comes from the manifest.
    report = AgreementReport(n=10, agree=4, panel_constant=False)
    with pytest.raises(GoldenGateViolation, match="agreement 4/10"):
        prove_gate(report=report, manifest=verified_with_gate(n=10, floor=8))


def test_prove_gate_mints_a_checkable_proof_on_green() -> None:
    _, packet = make_packet()
    verified = verify_packet(packet_md=packet.packet_md, manifest=packet.manifest)
    proof = prove_gate(report=AgreementReport(n=6, agree=6, panel_constant=False), manifest=verified)
    assert proof.gate == verified.gate
    assert proof.rows_sha256 == verified.rows_sha256
    proof.check()  # green: re-checking at the boundary does not raise
