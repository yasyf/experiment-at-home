from __future__ import annotations

import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from athome import llm
from athome.config import load
from athome.llm import TIER_PRICE_MODEL, extract, local, small
from athome.llm.pricing import cost
from athome.llm.spend import SpendExceeded
from athome.serve import ServerHandle

if TYPE_CHECKING:
    from collections.abc import Iterator


class Sentiment(BaseModel):
    label: str
    score: float


class FakeBackend:
    def __init__(self, base_url: str, model: str, *, api_key: str = "local", transport: object | None = None) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.transport = transport


@pytest.fixture(autouse=True)
def reset_llm_singletons() -> Iterator[None]:
    llm.default_log.cache_clear()
    llm.default_guard.cache_clear()
    yield
    llm.default_log.cache_clear()
    llm.default_guard.cache_clear()


@pytest.fixture
def fake_spawn(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module = ModuleType("spawnllm")
    module.call = AsyncMock(return_value="")
    module.extract = AsyncMock(return_value=None)
    module.OpenAiEndpointBackend = FakeBackend
    monkeypatch.setitem(sys.modules, "spawnllm", module)
    return module


@pytest.fixture
def local_env(monkeypatch: pytest.MonkeyPatch, fake_spawn: ModuleType) -> SimpleNamespace:
    from athome import serve

    monkeypatch.setenv("ATHOME_SERVE_RAPID_MLX_VERSION", "0.10.9")
    monkeypatch.setenv("ATHOME_SERVE_RAPID_MLX_MODEL", "mlx-community/Qwen3-4bit")
    load.cache_clear()
    handle = ServerHandle(recipe="rapid-mlx", port=8400, pid=None, base_url="http://127.0.0.1:8400/v1")

    async def fake_ensure(self: serve.ManagedServer, *, persistent: bool = False) -> ServerHandle:
        return handle

    sentinel = object()
    monkeypatch.setattr(serve.ManagedServer, "ensure", fake_ensure)
    monkeypatch.setattr("athome.llmcache.transport", lambda **_: sentinel)
    return SimpleNamespace(handle=handle, transport=sentinel, spawn=fake_spawn)


async def test_small_returns_and_records(fake_spawn: ModuleType) -> None:
    fake_spawn.call.return_value = "the answer"
    prompt = "some prompt text here"
    result = await small(prompt)
    assert result == "the answer"
    records = llm.default_log().records
    assert len(records) == 1
    record = records[0]
    assert record.model == "claude-haiku-4-5"
    assert record.served_model is None
    assert record.system_fingerprint is None
    assert record.input_tokens == len(prompt) // 4
    assert record.output_tokens == len("the answer") // 4
    expected = cost("claude-haiku-4-5", input_tokens=record.input_tokens, output_tokens=record.output_tokens)
    assert record.cost_usd == pytest.approx(expected)
    assert llm.default_guard().spent == pytest.approx(expected)
    assert fake_spawn.call.call_args.kwargs["timeout"] == 180


@pytest.mark.parametrize(
    ("tier", "price_model"),
    [
        pytest.param("small", "claude-haiku-4-5", id="small"),
        pytest.param("medium", "claude-sonnet-5", id="medium"),
        pytest.param("large", "claude-opus-4-8", id="large"),
    ],
)
async def test_small_tier_maps_to_price_model(fake_spawn: ModuleType, tier: str, price_model: str) -> None:
    fake_spawn.call.return_value = "ok"
    await small("hello world prompt", model=tier)
    record = llm.default_log().records[-1]
    assert record.model == price_model == TIER_PRICE_MODEL[tier]
    assert fake_spawn.call.call_args.kwargs["model"] == tier


async def test_extract_returns_validated_and_records(fake_spawn: ModuleType) -> None:
    answer = Sentiment(label="positive", score=0.9)
    fake_spawn.extract.return_value = answer
    result = await extract("classify this", Sentiment)
    assert result is answer
    record = llm.default_log().records[-1]
    assert record.model == "claude-opus-4-8"
    assert record.output_tokens == len(answer.model_dump_json()) // 4
    call_args = fake_spawn.extract.call_args
    assert call_args.args[1] is Sentiment
    assert call_args.kwargs["model"] == "large"


async def test_spend_guard_blocks_before_call(fake_spawn: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ATHOME_LLM_MAX_USD", "0.0")
    load.cache_clear()
    with pytest.raises(SpendExceeded):
        await small("a prompt that would cost real money")
    fake_spawn.call.assert_not_called()
    assert llm.default_log().records == []
    assert llm.default_guard().spent == 0.0


async def test_local_text_wraps_transport_and_records(local_env: SimpleNamespace) -> None:
    local_env.spawn.call.return_value = "local text"
    result = await local("hi there")
    assert result == "local text"
    backend = local_env.spawn.call.call_args.kwargs["backend"]
    assert backend.transport is local_env.transport
    assert backend.base_url == local_env.handle.base_url
    assert backend.model == "mlx-community/Qwen3-4bit"
    record = llm.default_log().records[-1]
    assert record.model == "claude-haiku-4-5"
    assert record.served_model is None


async def test_local_schema_validates_and_wraps_transport(local_env: SimpleNamespace) -> None:
    answer = Sentiment(label="neg", score=0.1)
    local_env.spawn.extract.return_value = answer
    result = await local("classify", schema=Sentiment)
    assert result is answer
    backend = local_env.spawn.extract.call_args.kwargs["backend"]
    assert backend.transport is local_env.transport
    assert local_env.spawn.extract.call_args.args[1] is Sentiment
    assert llm.default_log().records[-1].model == "claude-haiku-4-5"


def test_module_import_does_not_eagerly_pull_spawnllm() -> None:
    # The lazy-import contract: importing athome.llm must not import spawnllm at
    # module load (it lives behind the `llm` extra). Assert in a fresh interpreter
    # so the result is independent of whether spawnllm happens to be installed here.
    probe = "import sys, athome.llm; assert 'spawnllm' not in sys.modules; assert callable(athome.llm.small)"
    subprocess.run([sys.executable, "-c", probe], check=True)
