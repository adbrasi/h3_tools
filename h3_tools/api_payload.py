"""Pure builders for the modal_inferencito /generate bodies (API.md).

Stdlib-only: imports and tests without ComfyUI, like refs.py. The frame grid
mirrors the server exactly (17k+5 at 24 fps) so the duration shown to the
local enhancer equals the duration the server will generate.
"""

import os

FPS = 24


class ApiPayloadError(ValueError):
    """User-facing payload error; the message is shown as-is."""


def align_frames(n):
    n = max(5, int(n))
    while n % 17 != 5:
        n += 1
    return n


def target_duration(duration_s):
    """The real clip duration after the server's frame snap, in seconds."""
    return align_frames(round(float(duration_s) * FPS)) / float(FPS)


def upload_name(path, taken):
    """Unique upload basename for a local file path (server flattens paths)."""
    base = os.path.basename(str(path).replace("\\", "/"))
    stem, dot, ext = base.partition(".")
    name, i = base, 2
    while name in taken:
        name = "%s_%d%s%s" % (stem, i, dot, ext)
        i += 1
    taken.add(name)
    return name


def ref_body(*, prompt, references, duration_s, duration_mode=None):
    body = {
        "mode": "ref_extend" if duration_mode is not None else "ref",
        "prompt": prompt,
        "references": list(references),
        "duration_s": float(duration_s),
    }
    if duration_mode is not None:
        body["duration_mode"] = duration_mode
    return body


def flf_body(*, prompt, duration_s, first_name=None, last_name=None,
             size_mode="aspect"):
    """No frames at all is text-to-video: the server's strict per-mode
    whitelists (API.md) demand the explicit t2v mode (and reject size_mode
    there — t2v sizes by aspect only)."""
    if size_mode == "source" and not first_name:
        raise ApiPayloadError("size_mode 'source' needs a first frame image")
    if not first_name and not last_name:
        return {"mode": "t2v", "prompt": prompt,
                "duration_s": float(duration_s)}
    body = {"mode": "flf", "prompt": prompt, "duration_s": float(duration_s),
            "size_mode": size_mode}
    if first_name:
        body["first_image"] = first_name
    if last_name:
        body["last_image"] = last_name
    return body


def extend_extra(context_seconds, source_name):
    if float(context_seconds) <= 0:
        raise ApiPayloadError("context_seconds must be > 0")
    return {"source_video": source_name,
            "context_seconds": float(context_seconds)}
