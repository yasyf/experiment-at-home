from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import BinaryIO

WIRE_VERSION = 1
LENGTH_PREFIX = 4
PICKLE_PROTOCOL = 5

type Wire = None | bool | int | float | str | bytes | list[Wire] | tuple[Wire, ...] | dict[str, Wire]


class WireError(Exception):
    """A value is not a valid :data:`Wire`, or a frame cannot be decoded.

    Rooted at :class:`Exception`, deliberately *not* at ``athome.errors.AthomeError``:
    ``wire`` must import with only the standard library so sidecar distributions that
    do not depend on ``athome`` can vendor it.
    """


def validate(obj: object) -> Wire:
    """Structurally verify that ``obj`` is a :data:`Wire` value, returning it unchanged.

    Walks containers recursively. Dict keys must be ``str``. Raises :class:`WireError`
    on any other type (a ``set``, a custom object, a non-``str`` mapping key).
    """
    match obj:
        case None | bool() | int() | float() | str() | bytes():
            return obj
        case list():
            return [validate(item) for item in obj]
        case tuple():
            return tuple(validate(item) for item in obj)
        case dict() if all(isinstance(key, str) for key in obj):
            return {key: validate(value) for key, value in obj.items()}
        case dict():
            raise WireError("wire dict keys must be str")
        case _:
            raise WireError(f"not a wire value: {type(obj).__name__}")


def encode(obj: Wire) -> bytes:
    """Validate ``obj`` and serialize it into a length-prefixed wire frame.

    The frame is a 4-byte big-endian body length followed by a pickle (protocol 5) of
    the validated value.
    """
    body = pickle.dumps(validate(obj), protocol=PICKLE_PROTOCOL)
    return len(body).to_bytes(LENGTH_PREFIX, "big") + body


def decode(frame: bytes) -> Wire:
    """Decode a length-prefixed wire frame produced by :func:`encode` back into a :data:`Wire`.

    Raises :class:`WireError` if the declared length disagrees with the body or the
    payload is not a valid :data:`Wire`.
    """
    size = int.from_bytes(frame[:LENGTH_PREFIX], "big")
    if len(frame) - LENGTH_PREFIX != size:
        raise WireError(f"frame declares {size} bytes, body carries {len(frame) - LENGTH_PREFIX}")
    return validate(pickle.loads(frame[LENGTH_PREFIX:]))


def read_exactly(stream: BinaryIO, size: int) -> bytes:
    buffer = bytearray()
    while len(buffer) < size:
        if not (chunk := stream.read(size - len(buffer))):
            raise EOFError("stream closed before a full frame arrived")
        buffer += chunk
    return bytes(buffer)


def read_frame(stream: BinaryIO) -> Wire:
    """Read and decode one length-prefixed wire frame from a binary stream (sync, sidecar side).

    Raises :class:`EOFError` when the writer closes before a full frame, so a sidecar
    serve loop can exit cleanly on parent shutdown.
    """
    size = int.from_bytes(read_exactly(stream, LENGTH_PREFIX), "big")
    return validate(pickle.loads(read_exactly(stream, size)))


def write_frame(stream: BinaryIO, obj: Wire) -> None:
    """Validate ``obj`` and write it as a length-prefixed wire frame, flushing (sync, sidecar side)."""
    stream.write(encode(obj))
    stream.flush()
