"""The sidecar half of the wire protocol — a verbatim mirror of ``athome.workers.serve``.

Vendored (not imported) so this dist carries no ``athome`` dependency: it speaks the
same length-prefixed frames a :class:`athome.workers.PipeWorker` parent expects, using
only the local :mod:`athome_ocr_paddle.wire` copy and the standard library.
"""

from __future__ import annotations

import sys
import traceback
from typing import TYPE_CHECKING

from athome_ocr_paddle.wire import WIRE_VERSION, read_frame, validate, write_frame

if TYPE_CHECKING:
    from athome_ocr_paddle.wire import Wire


def handler_fingerprint(handler: object) -> Wire:
    provider = getattr(handler, "fingerprint", None)
    return provider() if callable(provider) else {}


def dispatch(handler: object, frame: Wire) -> Wire:
    match frame:
        case {"method": str() as method, "payload": payload}:
            try:
                return {"ok": validate(getattr(handler, method)(payload))}
            except Exception:
                return {"err": traceback.format_exc()}
        case _:
            return {"err": f"malformed request frame: {frame!r}"}


def serve(handler: object) -> None:
    """Run a sidecar's frame loop until the parent closes the pipe.

    Emits the handshake frame (wire version plus ``handler.fingerprint()`` when defined), then
    reads request frames and dispatches each to the named handler method, replying ``{"ok": ...}``
    on success or ``{"err": <traceback>}`` on any exception. Returns when stdin reaches EOF.
    """
    write_frame(sys.stdout.buffer, {"wire": WIRE_VERSION, "fingerprint": handler_fingerprint(handler)})
    while True:
        try:
            frame = read_frame(sys.stdin.buffer)
        except EOFError:
            return
        write_frame(sys.stdout.buffer, dispatch(handler, frame))
