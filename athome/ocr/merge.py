from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from athome.ocr.types import Document

if TYPE_CHECKING:
    from collections.abc import Sequence

MERGE_RECIPE = "mlx-vlm"
MERGE_INSTRUCTION = (
    "These are divergent OCR transcriptions of one page. Reconcile them into the single most accurate "
    "GitHub-flavored Markdown transcription. Output only the reconciled Markdown."
)


def merge_prompt(candidates: Sequence[Document]) -> str:
    return "\n\n".join(
        [MERGE_INSTRUCTION, *(f"--- Reading {index} ---\n{doc.markdown}" for index, doc in enumerate(candidates, 1))]
    )


@dataclass(frozen=True, slots=True)
class LlmMerger:
    """Reconciles divergent OCR readings with a local LLM, invoked only when engines disagree.

    :meth:`merge` feeds the candidate transcriptions to :func:`athome.llm.local` on a vision-capable
    recipe and returns the reconciled :class:`~athome.ocr.types.Document`. It runs on the conflict
    path alone — the profile calls it only after its cross-check fails.

    Example:
        >>> await LlmMerger().merge(page, (vlm_document, apple_document))
    """

    async def merge(self, image: bytes, candidates: Sequence[Document]) -> Document:
        from athome import llm

        # TODO: feed `image` to the merge once athome.llm exposes a vision entry (llm.local is text-only).
        return Document(markdown=await llm.local(merge_prompt(candidates), recipe=MERGE_RECIPE))
