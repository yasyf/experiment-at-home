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
    """Reconciles divergent OCR readings with a local vision LLM, invoked only when engines disagree.

    :meth:`merge` feeds the source image *and* the divergent candidate transcriptions to the
    ``mlx-vlm`` recipe's vision endpoint, so the model adjudicates conflicts (``$100`` vs ``$700``)
    against the pixels rather than the text alone, and returns the reconciled
    :class:`~athome.ocr.types.Document`. It runs on the conflict path alone — the profile calls it
    only after its cross-check fails.

    Example:
        >>> await LlmMerger().merge(page, (vlm_document, apple_document))
    """

    async def merge(self, image: bytes, candidates: Sequence[Document]) -> Document:
        from athome.ocr import vlm

        message = vlm.vision_message(image, prompt=merge_prompt(candidates))
        return Document(markdown=await vlm.complete(MERGE_RECIPE, [message]))
