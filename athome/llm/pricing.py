from __future__ import annotations

from dataclasses import dataclass

from athome.errors import AthomeError


class UnpricedModel(AthomeError):
    """Raised when a model has no :data:`PRICES` entry — cost is never silently zero."""


@dataclass(frozen=True, slots=True)
class Price:
    """Per-million-token input and output pricing for one model."""

    input_per_mtok: float
    output_per_mtok: float


PRICES: dict[str, Price] = {
    "claude-fable-5": Price(10.0, 50.0),
    "claude-opus-4-8": Price(5.0, 25.0),
    "claude-opus-4-7": Price(5.0, 25.0),
    "claude-sonnet-5": Price(3.0, 15.0),
    "claude-haiku-4-5": Price(1.0, 5.0),
    "gpt-4o": Price(2.5, 10.0),
    "gpt-4o-mini": Price(0.15, 0.6),
    "Qwen/Qwen3-8B": Price(0.13, 0.40),
}


def cost(model: str, *, input_tokens: int, output_tokens: int) -> float:
    """Return the USD cost of a call, raising :class:`UnpricedModel` for an unknown model."""
    if (price := PRICES.get(model)) is None:
        raise UnpricedModel(model)
    return input_tokens / 1_000_000 * price.input_per_mtok + output_tokens / 1_000_000 * price.output_per_mtok
