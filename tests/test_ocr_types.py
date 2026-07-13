from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from athome.ocr import Box, DocOcr, Document, OcrToken, TokenOcr


class FakeTokenOcr:
    async def tokens(self, image: bytes, *, region: Box | None = None, upscale: float = 1.0) -> tuple[OcrToken, ...]:
        return (OcrToken(text="hi", box=Box(0, 0, 10, 10), confidence=0.9),)


class FakeDocOcr:
    async def read(self, image: bytes) -> Document:
        return Document(markdown="# Hi")


def test_box_is_frozen_and_slotted() -> None:
    box = Box(1, 2, 3, 4)
    assert (box.x, box.y, box.width, box.height) == (1, 2, 3, 4)
    assert not hasattr(box, "__dict__")
    with pytest.raises(FrozenInstanceError):
        setattr(box, "x", 5)


def test_ocr_token_is_frozen_and_slotted() -> None:
    token = OcrToken(text="es", box=Box(0, 0, 4, 4), confidence=0.5)
    assert (token.text, token.box, token.confidence) == ("es", Box(0, 0, 4, 4), 0.5)
    assert not hasattr(token, "__dict__")
    with pytest.raises(FrozenInstanceError):
        setattr(token, "text", "nq")


def test_document_is_frozen_slotted_with_empty_default_tokens() -> None:
    doc = Document(markdown="hello")
    assert doc.tokens == ()
    assert not hasattr(doc, "__dict__")
    with pytest.raises(FrozenInstanceError):
        setattr(doc, "markdown", "bye")


def test_document_carries_tokens() -> None:
    token = OcrToken(text="hi", box=Box(0, 0, 1, 1), confidence=1.0)
    assert Document(markdown="hi", tokens=(token,)).tokens == (token,)


@pytest.mark.parametrize(
    ("markdown", "expected"),
    [
        pytest.param("# Title", "Title", id="heading"),
        pytest.param("### Deep Heading ", "Deep Heading", id="deep-heading"),
        pytest.param("## **Bold Title**", "Bold Title", id="heading-with-emphasis"),
        pytest.param("Some **bold** word", "Some bold word", id="bold"),
        pytest.param("An *italic* and _also_ word", "An italic and also word", id="italic"),
        pytest.param("Use `code()` inline", "Use code() inline", id="inline-code"),
        pytest.param("| Name | Age |\n| --- | --- |\n| Alice | 30 |", "Name Age\nAlice 30", id="table"),
        pytest.param("| Sym | Px |\n|:-:|--:|\n| ES | *4200* |", "Sym Px\nES 4200", id="table-aligned-emphasis"),
        pytest.param("```python\nx = 1\ny = 2\n```", "x = 1\ny = 2", id="code-fence"),
        pytest.param(
            "# Report\n\nRows below:\n\n| A | B |\n| - | - |\n| 1 | 2 |",
            "Report\nRows below:\nA B\n1 2",
            id="mixed-document",
        ),
    ],
)
def test_document_text_strips_markdown(markdown: str, expected: str) -> None:
    assert Document(markdown=markdown).text == expected


def test_token_ocr_runtime_checkable() -> None:
    assert isinstance(FakeTokenOcr(), TokenOcr)
    assert not isinstance(FakeDocOcr(), TokenOcr)


def test_doc_ocr_runtime_checkable() -> None:
    assert isinstance(FakeDocOcr(), DocOcr)
    assert not isinstance(FakeTokenOcr(), DocOcr)


async def test_token_ocr_conformance_runs() -> None:
    result = await FakeTokenOcr().tokens(b"img", region=Box(0, 0, 5, 5), upscale=2.0)
    assert result == (OcrToken(text="hi", box=Box(0, 0, 10, 10), confidence=0.9),)


async def test_doc_ocr_conformance_runs() -> None:
    doc = await FakeDocOcr().read(b"img")
    assert doc == Document(markdown="# Hi")
    assert doc.text == "Hi"
