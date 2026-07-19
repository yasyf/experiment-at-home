from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

from athome.ocr.types import Document
from athome.serve import (
    LlamaServerSettings,
    ManagedServer,
    MlxVlmSettings,
    ModalVllmSettings,
    RapidMlxSettings,
    settings_for,
)

if TYPE_CHECKING:
    from athome.serve import Recipe
    from athome.wire import Wire

VLM_PROMPT = (
    "Transcribe this page into GitHub-flavored Markdown. Preserve reading order, headings, lists, "
    "and tables. Output only the Markdown."
)
ENDPOINT_TIMEOUT_S = 300.0
IMAGE_MEDIA_TYPES = {
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
    b"GIF8": "image/gif",
    b"RIFF": "image/webp",
}


def endpoint_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=ENDPOINT_TIMEOUT_S)


def image_media_type(image: bytes) -> str:
    from athome.ocr.profiles import OcrError

    for magic, media in IMAGE_MEDIA_TYPES.items():
        if image.startswith(magic):
            return media
    raise OcrError(f"unrecognized image format: {image[:8]!r}")


def image_data_uri(image: bytes) -> str:
    return f"data:{image_media_type(image)};base64,{base64.b64encode(image).decode()}"


def vlm_model(recipe: Recipe) -> str:
    from athome.ocr.profiles import OcrError

    match settings_for(recipe):
        case MlxVlmSettings(model=model) | RapidMlxSettings(model=model):
            return model
        case LlamaServerSettings() | ModalVllmSettings():
            raise OcrError(f"recipe {recipe!r} has no vision model")


def vision_message(image: bytes, *, prompt: str) -> Wire:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data_uri(image)}},
        ],
    }


async def complete(recipe: Recipe, messages: list[Wire]) -> str:
    handle = await ManagedServer(recipe).ensure()
    async with endpoint_client() as client:
        response = await client.post(
            f"{handle.base_url}/chat/completions", json={"model": vlm_model(recipe), "messages": messages}
        )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


@dataclass(frozen=True, slots=True)
class VlmOcr:
    """Document OCR via a vision LLM (dots.ocr) served over the ``mlx-vlm`` serve recipe.

    :meth:`read` ensures the recipe's OpenAI-compatible server is healthy, posts the page as a
    base64 image to its ``/chat/completions`` endpoint, and returns the model's Markdown as a
    :class:`~athome.ocr.types.Document`. The engine itself is a ``uvx`` subprocess (never an
    athome dependency); only ``httpx`` is used to reach it.

    Example:
        >>> await VlmOcr().read(page_png)
    """

    recipe: Recipe = "mlx-vlm"

    async def read(self, image: bytes) -> Document:
        return Document(markdown=await complete(self.recipe, [vision_message(image, prompt=VLM_PROMPT)]))
