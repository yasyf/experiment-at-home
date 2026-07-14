from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import sys
from dataclasses import dataclass, field
from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from athome.config import load
from athome.llm.spend import SpendExceeded
from athome.modal import ParityMismatch, fingerprint_for
from athome.progress import RunSink
from athome.train import modal as train_modal
from athome.train.backend import TrainBackend
from athome.train.modal import (
    OVERSHOOT_SECONDS,
    PACKAGES,
    STARTUP_SECONDS,
    ModalTrainBackend,
    RemoteConfig,
    RemoteResult,
    baked_commit,
    billed_usd,
    budget_seconds,
    fingerprint_remote,
    lora_config,
    lora_params,
    projected_usd,
    service_spec,
    train_remote,
)
from athome.train.spec import (
    BASE_MODELS,
    STD_MODULES,
    Hyperparams,
    LocalJsonlRef,
    LoraSpec,
    ModalTrainSettings,
    TrainSpec,
    UnservableBase,
    UnsupportedLoraShape,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from athome.train.spec import Method

BASE = BASE_MODELS["qwen3-8b"]
UNSERVABLE = BASE_MODELS["qwen3.5-4b"]
GPU_USD_PER_HOUR = 3.95
PINS: dict[str, str] = {
    "param:trl": "0.21.0",
    "param:peft": "0.17.0",
    "param:torch": "2.6.0",
    "param:transformers": "4.55.4",
    "param:datasets": "5.0.0",
}
SHAPE: dict[str, object] = {
    "param:lora_rank": 16,
    "param:lora_alpha": 32,
    "param:lora_dropout": 0.0,
    "param:lora_target_modules": sorted(STD_MODULES),
    "param:lora_train_mlp": True,
    "param:lora_train_attn": True,
    "param:lora_train_unembed": False,
}
WEIGHTS: dict[str, object] = {
    "param:base_hf": "Qwen/Qwen3-8B",
    "param:base_hf_revision": BASE.hf_revision,
    "param:base_mlx": "mlx-community/Qwen3-8B-4bit",
    "param:base_mlx_revision": BASE.mlx_revision,
}
MATCHING_FINGERPRINT: dict[str, object] = PINS | SHAPE | WEIGHTS
TRAINED = RemoteResult(repo="athome-train/pilot", revision="c0ffee", step=4, seconds=1800.0)
TRAINED_USD = (STARTUP_SECONDS + 1800.0) / 3600 * GPU_USD_PER_HOUR
SFT_ROWS = [{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}]
DPO_ROWS = [
    {
        "prompt": [{"role": "user", "content": "q"}],
        "chosen": [{"role": "assistant", "content": "yes"}],
        "rejected": [{"role": "assistant", "content": "no"}],
    }
]


@dataclass(frozen=True, slots=True)
class FakeDataset:
    rows: list[dict[str, object]]

    @staticmethod
    def from_list(rows: list[dict[str, object]]) -> FakeDataset:
        return FakeDataset(rows)


@dataclass(slots=True)
class Remote:
    """What the fake Modal side returns, and what it was asked to do."""

    fingerprint: dict[str, object]
    result: RemoteResult
    image: list[tuple[object, ...]] = field(default_factory=list)
    options: dict[str, dict[str, object]] = field(default_factory=dict)
    calls: list[tuple[str, tuple[object, ...]]] = field(default_factory=list)
    app: tuple[str, object] | None = None
    secrets: list[dict[str, str]] = field(default_factory=list)
    ran: bool = False

    def called(self, name: str) -> list[tuple[object, ...]]:
        return [args for called, args in self.calls if called == name]


@dataclass(slots=True)
class FakeFunction:
    remote_side: Remote
    name: str

    @property
    def remote(self) -> SimpleNamespace:
        async def aio(*args: object) -> object:
            self.remote_side.calls.append((self.name, args))
            return self.remote_side.fingerprint if self.name == "fingerprint_remote" else self.remote_side.result

        return SimpleNamespace(aio=aio)


def install_fake_modal(monkeypatch: pytest.MonkeyPatch, remote: Remote) -> Remote:
    class Chain:
        def __getattr__(self, name: str) -> Callable[..., Chain]:
            def step(*args: object, **kwargs: object) -> Chain:
                remote.image.append((name, args, kwargs))
                return self

            return step

    class Image:
        @staticmethod
        def debian_slim(*, python_version: str) -> Chain:
            remote.image.append(("debian_slim", (python_version,), {}))
            return Chain()

    class Run:
        async def __aenter__(self) -> None:
            remote.ran = True

        async def __aexit__(self, *exc: object) -> None: ...

    class App:
        def __init__(self, name: str, *, image: object) -> None:
            remote.app = (name, image)

        def function(self, **options: object) -> Callable[[Callable[..., object]], FakeFunction]:
            def decorate(fn: Callable[..., object]) -> FakeFunction:
                remote.options[fn.__name__] = options
                return FakeFunction(remote, fn.__name__)

            return decorate

        def run(self) -> Run:
            return Run()

    class Secret:
        @staticmethod
        def from_dict(values: dict[str, str]) -> object:
            remote.secrets.append(values)
            return values

    module = ModuleType("modal")
    module.__spec__ = ModuleSpec("modal", None)
    module.Image = Image
    module.App = App
    module.Secret = Secret
    monkeypatch.setitem(sys.modules, "modal", module)
    return remote


@dataclass(frozen=True, slots=True)
class FakeLoraConfig:
    """Stands in for ``peft.LoraConfig``, which lives only in the Modal image."""

    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]
    task_type: str


@dataclass(slots=True)
class Converge:
    """The mocked HF-and-sidecar tail: snapshot, PEFT-to-mlx convert, then fuse."""

    peft_dir: Path
    adapter_dir: Path
    mlx_path: Path
    order: list[str] = field(default_factory=list)


@pytest.fixture(autouse=True)
def fake_datasets(monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("datasets")
    module.Dataset = FakeDataset
    monkeypatch.setitem(sys.modules, "datasets", module)


@pytest.fixture
def fake_peft(monkeypatch: pytest.MonkeyPatch) -> None:
    """peft is a Modal-image package; the container's half of the fingerprint needs it importable."""
    module = ModuleType("peft")
    module.LoraConfig = FakeLoraConfig
    monkeypatch.setitem(sys.modules, "peft", module)


@pytest.fixture
def baked(monkeypatch: pytest.MonkeyPatch, fake_peft: None) -> Callable[[str], None]:
    """Runs the container's fingerprint half locally: pinned versions, and the commit baked into the image."""

    def bake(commit: str) -> None:
        module = ModuleType("huggingface_hub")
        module.snapshot_download = lambda repo, *, revision, local_files_only=False: f"/models/hf/snapshots/{commit}"
        monkeypatch.setitem(sys.modules, "huggingface_hub", module)
        monkeypatch.setattr(
            importlib.metadata, "version", lambda package: PINS[f"param:{package}"] if package in PACKAGES else "0"
        )

    return bake


@pytest.fixture(autouse=True)
def settings_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_write_token")
    load.cache_clear()


@pytest.fixture
def converge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Converge:
    state = Converge(tmp_path / "peft", tmp_path / "adapter", tmp_path / "mlx")

    async def ensure_write_auth() -> None:
        state.order.append("preflight")

    async def snapshot(repo: str, *, revision: str | None = None) -> Path:
        state.order.append(f"snapshot:{repo}@{revision}")
        return state.peft_dir

    async def convert_peft_to_mlx(peft_dir: Path, out_dir: Path, *, base: object) -> Path:
        state.order.append(f"convert:{peft_dir}->{out_dir}")
        return state.adapter_dir

    async def fuse(adapter_dir: Path, out_dir: Path, *, base: object) -> Path:
        state.order.append(f"fuse:{adapter_dir}->{out_dir}")
        return state.mlx_path

    monkeypatch.setattr(train_modal, "ensure_write_auth", ensure_write_auth)
    monkeypatch.setattr(train_modal, "snapshot", snapshot)
    monkeypatch.setattr(train_modal, "convert_peft_to_mlx", convert_peft_to_mlx)
    monkeypatch.setattr(train_modal, "fuse", fuse)
    return state


def corpus(tmp_path: Path, rows: Sequence[dict[str, object]]) -> LocalJsonlRef:
    path = tmp_path / "corpus.jsonl"
    path.write_text("".join(f"{json.dumps(row)}\n" for row in rows))
    return LocalJsonlRef(path)


def train_spec(tmp_path: Path, *, method: Method = "sft", **overrides: object) -> TrainSpec:
    return dataclasses.replace(
        TrainSpec(
            name="pilot",
            base=BASE,
            dataset=corpus(tmp_path, SFT_ROWS if method == "sft" else DPO_ROWS),
            hyperparams=Hyperparams(steps=4, batch_size=2, max_seq_len=128),
            method=method,
            lora=LoraSpec(),
        ),
        **overrides,
    )


def sink(tmp_path: Path) -> RunSink:
    return RunSink.open(tmp_path / "run.jsonl")


def records(tmp_path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in (tmp_path / "run.jsonl").read_text().splitlines()]


async def train(tmp_path: Path, spec: TrainSpec, settings: ModalTrainSettings | None = None) -> object:
    return await ModalTrainBackend(settings or ModalTrainSettings()).train(
        spec, sink=sink(tmp_path), work_dir=tmp_path / "run"
    )


def test_backend_satisfies_the_protocol_and_trains_both_trl_methods() -> None:
    assert ModalTrainBackend.name == "modal"
    assert ModalTrainBackend.supports("sft")
    assert ModalTrainBackend.supports("dpo")
    assert isinstance(ModalTrainBackend(ModalTrainSettings()), TrainBackend)


def test_available_needs_modal_and_a_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_modal(monkeypatch, Remote(MATCHING_FINGERPRINT, TRAINED))
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-123")
    assert ModalTrainBackend.available()

    monkeypatch.setattr(train_modal.importlib.util, "find_spec", lambda name: None)
    assert not ModalTrainBackend.available()


def test_the_image_pins_are_a_co_installable_set() -> None:
    """#2 — TRL 0.21 requires transformers>=4.55; the old 4.48 pin made the image unbuildable.

    The proof that these resolve is a `uv pip compile` for the image's linux/py3.13; this guards
    the floor that made the old pair unsatisfiable, so a downgrade cannot land unnoticed.
    """
    settings = ModalTrainSettings()

    assert tuple(int(part) for part in settings.trl_version.split(".")) >= (0, 21, 0)
    assert tuple(int(part) for part in settings.transformers_version.split(".")) >= (4, 55, 0)
    assert set(PACKAGES) == {"trl", "peft", "torch", "transformers", "datasets"}


def test_service_spec_fingerprints_the_pins_the_lora_shape_and_the_base_weights() -> None:
    assert fingerprint_for(service_spec(ModalTrainSettings(), LoraSpec(), BASE)) == MATCHING_FINGERPRINT


@pytest.mark.parametrize(
    ("skew", "expected"),
    [
        pytest.param({"param:trl": "0.22.0"}, "trl", id="trl-skew"),
        pytest.param({"param:peft": "0.18.0"}, "peft", id="peft-skew"),
        pytest.param({"param:torch": "2.7.0"}, "torch", id="torch-skew"),
        pytest.param({"param:transformers": "4.56.0"}, "transformers", id="transformers-skew"),
        pytest.param({"param:datasets": "4.0.0"}, "datasets", id="datasets-skew"),
        pytest.param({"param:lora_rank": 8}, "lora_rank", id="lora-rank-skew"),
        pytest.param({"param:lora_alpha": 64}, "lora_alpha", id="lora-alpha-skew"),
        pytest.param({"param:lora_target_modules": ["mlp.gate_proj"]}, "lora_target_modules", id="lora-modules-skew"),
        pytest.param({"param:lora_train_mlp": False}, "lora_train_mlp", id="mlp-toggle-skew"),
        pytest.param({"param:lora_train_attn": False}, "lora_train_attn", id="attn-toggle-skew"),
        pytest.param({"param:base_hf_revision": "deadbeef"}, "base_hf_revision", id="baked-base-weight-skew"),
        pytest.param({"param:base_mlx_revision": "deadbeef"}, "base_mlx_revision", id="mlx-base-weight-skew"),
    ],
)
async def test_parity_mismatch_raises_before_the_gpu_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, converge: Converge, skew: dict[str, object], expected: str
) -> None:
    """#5/#7 — a toggle or a base-weight skew must be caught, not just a version one."""
    remote = install_fake_modal(monkeypatch, Remote(MATCHING_FINGERPRINT | skew, TRAINED))

    with pytest.raises(ParityMismatch) as excinfo:
        await train(tmp_path, train_spec(tmp_path))

    assert expected in str(excinfo.value)
    assert "athome-train" in str(excinfo.value)
    assert remote.called("fingerprint_remote")
    assert remote.called("train_remote") == []
    assert converge.order == ["preflight"]


async def test_a_modal_that_ignored_the_lora_toggles_is_caught_by_the_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, converge: Converge
) -> None:
    """#5 — Modal trained MLP weights for `train_mlp=False` and its fingerprint still passed."""
    remote = install_fake_modal(monkeypatch, Remote(MATCHING_FINGERPRINT, TRAINED))
    spec = train_spec(tmp_path, lora=LoraSpec(train_mlp=False))

    with pytest.raises(ParityMismatch) as excinfo:
        await train(tmp_path, spec)

    assert "lora_target_modules" in str(excinfo.value)
    assert "lora_train_mlp" in str(excinfo.value)
    assert remote.called("train_remote") == []


def test_lora_config_targets_only_the_modules_the_toggles_select(fake_peft: None) -> None:
    """#5 — the PEFT config Modal builds is the one shape rule, same as local's."""
    assert lora_config(LoraSpec()).target_modules == list(STD_MODULES)
    assert lora_config(LoraSpec(train_mlp=False)).target_modules == list(STD_MODULES[:4])
    assert lora_config(LoraSpec(train_attn=False)).target_modules == list(STD_MODULES[4:])

    with pytest.raises(UnsupportedLoraShape, match="unembedding"):
        lora_config(LoraSpec(train_unembed=True))


def test_the_containers_fingerprint_reports_the_toggles_and_the_baked_commit(
    baked: Callable[[str], None],
) -> None:
    """#5/#7 — the remote half is what parity compares against, so it must carry both."""
    baked(BASE.hf_revision)
    config = RemoteConfig(
        base=BASE, repo="athome-train/pilot", method="sft", lora=LoraSpec(), hyperparams=Hyperparams(steps=1)
    )

    assert fingerprint_remote(config) == MATCHING_FINGERPRINT
    assert fingerprint_remote(dataclasses.replace(config, lora=LoraSpec(train_mlp=False))) == MATCHING_FINGERPRINT | {
        "param:lora_train_mlp": False,
        "param:lora_target_modules": sorted(STD_MODULES[:4]),
    }


def test_the_baked_commit_is_read_off_the_image_not_echoed_back(baked: Callable[[str], None]) -> None:
    """#7 — an image built over other weights must report those weights, so parity can catch it."""
    baked("deadbeef")

    assert baked_commit(BASE) == "deadbeef"
    assert baked_commit(BASE) != BASE.hf_revision


async def test_sft_feeds_the_trainer_prompt_and_completion_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, converge: Converge
) -> None:
    remote = install_fake_modal(monkeypatch, Remote(MATCHING_FINGERPRINT, TRAINED))

    await train(tmp_path, train_spec(tmp_path, method="sft"))

    ((config, dataset),) = remote.called("train_remote")
    assert config.method == "sft"
    assert config.base == BASE
    assert dataset.rows == [
        {"prompt": [{"role": "user", "content": "q"}], "completion": [{"role": "assistant", "content": "a"}]}
    ]


async def test_dpo_feeds_the_trainer_chosen_and_rejected_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, converge: Converge
) -> None:
    remote = install_fake_modal(monkeypatch, Remote(MATCHING_FINGERPRINT, TRAINED))

    await train(tmp_path, train_spec(tmp_path, method="dpo"))

    ((config, dataset),) = remote.called("train_remote")
    assert config.method == "dpo"
    assert dataset.rows == [
        {
            "prompt": [{"role": "user", "content": "q"}],
            "chosen": [{"role": "assistant", "content": "yes"}],
            "rejected": [{"role": "assistant", "content": "no"}],
        }
    ]


async def test_the_gpu_loads_the_base_at_its_pinned_revision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """#7 — the weights TRL trains must be the weights the fingerprint names."""
    loaded: list[tuple[str, str]] = []

    class Auto:
        @staticmethod
        def from_pretrained(repo: str, *, revision: str, **kwargs: object) -> str:
            loaded.append((repo, revision))
            return "model"

    trainer = SimpleNamespace(train=lambda: None, save_model=lambda path: None)
    transformers = ModuleType("transformers")
    transformers.AutoModelForCausalLM = transformers.AutoTokenizer = Auto
    trl = ModuleType("trl")
    trl.SFTConfig = trl.DPOConfig = dict
    trl.SFTTrainer = trl.DPOTrainer = lambda **kwargs: trainer
    hub = ModuleType("huggingface_hub")
    hub.HfApi = lambda token: SimpleNamespace(
        create_repo=lambda repo, private, exist_ok: None,
        upload_folder=lambda repo_id, folder_path: SimpleNamespace(oid="c0ffee"),
    )
    for name, module in (("transformers", transformers), ("trl", trl), ("huggingface_hub", hub)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "peft", ModuleType("peft"))
    sys.modules["peft"].LoraConfig = FakeLoraConfig
    monkeypatch.setenv("HF_TOKEN", "hf_write_token")

    result = train_remote(
        RemoteConfig(
            base=BASE, repo="athome-train/pilot", method="sft", lora=LoraSpec(), hyperparams=Hyperparams(steps=1)
        ),
        FakeDataset([]),
    )

    assert loaded == [(BASE.hf, BASE.hf_revision), (BASE.hf, BASE.hf_revision)]
    assert result.revision == "c0ffee"


async def test_an_unservable_base_is_refused_before_a_dollar_is_spent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, converge: Converge
) -> None:
    """#7 — Modal accepted `qwen3.5-4b` and only failed at the fuse, after the GPU had billed."""
    remote = install_fake_modal(monkeypatch, Remote(MATCHING_FINGERPRINT, TRAINED))

    with pytest.raises(UnservableBase, match="mlx-lm LoRA counterpart"):
        await train(tmp_path, train_spec(tmp_path, base=UNSERVABLE))

    assert remote.app is None
    assert not remote.ran
    assert remote.calls == []
    assert converge.order == []


async def test_a_lora_shape_the_fuse_cannot_express_is_refused_before_the_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, converge: Converge
) -> None:
    """#6 — `train_unembed=True` spent GPU money on a tensor the converter then threw away."""
    remote = install_fake_modal(monkeypatch, Remote(MATCHING_FINGERPRINT, TRAINED))

    with pytest.raises(UnsupportedLoraShape, match="unembedding"):
        await train(tmp_path, train_spec(tmp_path, lora=LoraSpec(train_unembed=True)))

    assert remote.app is None
    assert remote.calls == []


async def test_a_projected_breach_aborts_before_modal_is_touched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, converge: Converge
) -> None:
    remote = install_fake_modal(monkeypatch, Remote(MATCHING_FINGERPRINT, TRAINED))
    settings = ModalTrainSettings()
    spec = train_spec(tmp_path, max_usd=0.10)
    assert projected_usd(spec, settings) > 0.10

    with pytest.raises(SpendExceeded, match=r"exceeds cap \$0.1000"):
        await train(tmp_path, spec, settings)

    assert not remote.ran
    assert remote.calls == []
    assert remote.app is None


async def test_a_zero_cap_spends_nothing_rather_than_the_configured_sixty_dollars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, converge: Converge
) -> None:
    """#4 — `spec.max_usd or settings.spend_cap_usd` read a 0.0 cap as unset and authorized $60."""
    remote = install_fake_modal(monkeypatch, Remote(MATCHING_FINGERPRINT, TRAINED))
    assert ModalTrainSettings().spend_cap_usd == 60.0

    with pytest.raises(SpendExceeded, match=r"exceeds cap \$0.0000"):
        await train(tmp_path, train_spec(tmp_path, max_usd=0.0))

    assert not remote.ran
    assert remote.calls == []
    assert remote.app is None
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    ("max_usd", "timeout"),
    [
        pytest.param(1.0, 551, id="a-dollar-buys-551s-after-startup-and-overshoot"),
        pytest.param(60.0, 54323, id="the-configured-cap"),
        pytest.param(1_000_000.0, 86400, id="modals-own-ceiling-still-binds"),
    ],
)
def test_the_timeout_is_what_the_cap_has_left_after_startup_and_overshoot(max_usd: float, timeout: int) -> None:
    """#3 — the whole cap used to become the timeout, ignoring the startup it is also billed for."""
    assert budget_seconds(max_usd, ModalTrainSettings()) == timeout


@pytest.mark.parametrize("max_usd", [0.4, 1.0, 5.0, 60.0, 95.0])
def test_a_run_that_burns_its_whole_timeout_still_settles_under_the_cap(max_usd: float) -> None:
    """#3 — the cap is a hard dollar bound: worst case is the granted timeout plus Modal's overshoot."""
    settings = ModalTrainSettings()

    worst = billed_usd(budget_seconds(max_usd, settings) + OVERSHOOT_SECONDS, settings)

    assert worst <= max_usd
    assert worst == pytest.approx(max_usd, rel=0.01)


def test_a_cap_beyond_modals_own_ceiling_is_bounded_by_the_ceiling() -> None:
    """A 24h function is all Modal will run, so a huge cap simply goes unspent."""
    settings = ModalTrainSettings()

    assert budget_seconds(250.0, settings) == train_modal.MODAL_MAX_TIMEOUT
    assert billed_usd(train_modal.MODAL_MAX_TIMEOUT + OVERSHOOT_SECONDS, settings) < 250.0


def test_a_cap_that_cannot_even_pay_for_startup_grants_no_gpu_time() -> None:
    """#3 — with a $1 cap the old code granted 911s and its own model then billed $1.3287."""
    settings = ModalTrainSettings()
    unaffordable = billed_usd(OVERSHOOT_SECONDS, settings)

    with pytest.raises(SpendExceeded, match=r"buys no GPU time"):
        budget_seconds(unaffordable - 0.01, settings)

    assert budget_seconds(1.0, settings) * GPU_USD_PER_HOUR / 3600 < 1.0


def test_projection_and_settlement_are_billed_the_same_way(tmp_path: Path) -> None:
    """#3 — the projection counted 300s of startup while the in-function timer did not."""
    settings = ModalTrainSettings()
    spec = train_spec(tmp_path)
    tokens = spec.hyperparams.steps * spec.hyperparams.batch_size * spec.hyperparams.max_seq_len

    assert projected_usd(spec, settings) == billed_usd(tokens / train_modal.LORA_TOKENS_PER_SECOND, settings)
    assert billed_usd(0.0, settings) == STARTUP_SECONDS / 3600 * GPU_USD_PER_HOUR


async def test_a_run_that_bills_past_the_cap_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, converge: Converge
) -> None:
    install_fake_modal(
        monkeypatch, Remote(MATCHING_FINGERPRINT, RemoteResult("athome-train/pilot", "c0ffee", 4, 7200.0))
    )

    with pytest.raises(SpendExceeded, match=r"billed \$8.2292, over the \$1.0000 cap"):
        await train(tmp_path, train_spec(tmp_path, max_usd=1.0))

    assert "convert" not in " ".join(converge.order)


async def test_dpo_projects_two_passes_of_the_corpus(tmp_path: Path) -> None:
    settings = ModalTrainSettings()
    sft, dpo = train_spec(tmp_path, method="sft"), train_spec(tmp_path, method="dpo")
    startup = projected_usd(
        TrainSpec(name="empty", base=BASE, dataset=sft.dataset, hyperparams=Hyperparams(steps=0)), settings
    )

    assert projected_usd(dpo, settings) - startup == pytest.approx(2 * (projected_usd(sft, settings) - startup))


async def test_the_gpu_function_carries_the_gpu_class_the_budget_and_the_hf_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, converge: Converge
) -> None:
    brief = RemoteResult(repo="athome-train/pilot", revision="c0ffee", step=4, seconds=500.0)
    remote = install_fake_modal(monkeypatch, Remote(MATCHING_FINGERPRINT, brief))

    checkpoint = await train(tmp_path, train_spec(tmp_path, max_usd=1.0))

    assert remote.options["train_remote"]["gpu"] == "H100"
    assert remote.options["train_remote"]["timeout"] == 551
    assert "gpu" not in remote.options["fingerprint_remote"]
    assert remote.secrets == [{"HF_TOKEN": "hf_write_token"}]
    assert remote.app == ("athome-train", remote.app[1])
    assert checkpoint.train_cost_usd == pytest.approx(billed_usd(500.0, ModalTrainSettings()))
    assert checkpoint.train_cost_usd < 1.0


async def test_the_image_pins_the_trl_stack_and_bakes_the_base_at_its_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, converge: Converge
) -> None:
    remote = install_fake_modal(monkeypatch, Remote(MATCHING_FINGERPRINT, TRAINED))

    await train(tmp_path, train_spec(tmp_path))

    steps = {name: (args, kwargs) for name, args, kwargs in remote.image}
    assert steps["pip_install"][0] == (
        "trl==0.21.0",
        "peft==0.17.0",
        "torch==2.6.0",
        "transformers==4.55.4",
        "datasets==5.0.0",
        "huggingface_hub",
    )
    assert steps["add_local_python_source"] == (("athome",), {"copy": True})
    assert steps["run_function"] == ((train_modal.download_base,), {"args": (BASE.hf, BASE.hf_revision)})
    assert list(steps) == ["debian_slim", "uv_sync", "pip_install", "env", "add_local_python_source", "run_function"]


async def test_the_checkpoint_is_the_fused_mlx_model_the_sidecar_produced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, converge: Converge
) -> None:
    install_fake_modal(monkeypatch, Remote(MATCHING_FINGERPRINT, TRAINED))

    checkpoint = await train(tmp_path, train_spec(tmp_path))

    assert checkpoint.mlx_path == converge.mlx_path
    assert checkpoint.adapter_dir == converge.adapter_dir
    assert checkpoint.backend == "modal"
    assert checkpoint.method == "sft"
    assert checkpoint.base == BASE
    assert checkpoint.step == 4
    assert checkpoint.train_cost_usd == pytest.approx(TRAINED_USD)
    work_dir = tmp_path / "run"
    assert converge.order == [
        "preflight",
        "snapshot:athome-train/pilot@c0ffee",
        f"convert:{converge.peft_dir}->{work_dir / 'adapter'}",
        f"fuse:{converge.adapter_dir}->{work_dir / 'mlx'}",
    ]


async def test_the_run_is_journaled_from_launch_to_the_fused_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, converge: Converge
) -> None:
    install_fake_modal(monkeypatch, Remote(MATCHING_FINGERPRINT, TRAINED))

    await train(tmp_path, train_spec(tmp_path))

    journal = records(tmp_path)
    assert [record["stage"] for record in journal] == ["launch", "trained", "converged"]
    assert journal[0]["gpu"] == "H100"
    assert journal[0]["timeout"] == 54323
    assert journal[1]["revision"] == "c0ffee"
    assert journal[1]["usd"] == pytest.approx(TRAINED_USD)
    assert journal[2]["mlx_path"] == str(converge.mlx_path)


def test_lora_params_carry_every_toggle_into_the_fingerprint() -> None:
    """#5 — the toggles had no fingerprint entry, so a skew on them was invisible."""
    assert lora_params(LoraSpec(train_mlp=False)) == {
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "lora_target_modules": sorted(STD_MODULES[:4]),
        "lora_train_mlp": False,
        "lora_train_attn": True,
        "lora_train_unembed": False,
    }
