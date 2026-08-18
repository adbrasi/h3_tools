"""Continuation logic for H3RefToVideoContinuePro (SPEC §12) — stdlib-pure.

Timeline math, the LLM context sentence, and OpenRouter video-support
detection. The frame-grid `align` function is injected by the caller (it lives
in comfy_extras) so this module imports and tests without ComfyUI; `requests`
is imported lazily inside fetch_video_models only.
"""

import logging
import time

VIDEO_MODELS_URL = "https://openrouter.ai/api/v1/models?input_modalities=video"
VIDEO_MODELS_TTL = 3600.0  # seconds; the catalog changes rarely

_MODELS_CACHE = {"at": None, "models": None}


class ContinuationError(ValueError):
    """User-facing continuation failure; the message is shown as-is."""


def timeline(mode, length, source_duration, align):
    """Latent length + the real clock the LLM must write against (SPEC §12.3).

    `align` is the native align_frame_count (17k+5 grid, snapping up).
    Image mode (source_duration is None): the source clock is unknown, so the
    prompt's timestamps start at zero and `length` is the new part.
    """
    if source_duration is None:
        return {"latent_length": length,
                "total_duration": align(max(5, length)) / 24.0,
                "source_duration": None, "from_zero": True}
    if mode == "total":
        total_duration = align(max(5, length)) / 24.0
        if source_duration >= total_duration:
            raise ContinuationError(
                "the source video (%.1fs) is not shorter than the total "
                "duration (%.1fs = length %d at 24 fps) — increase length or "
                "set duration_mode to new_only"
                % (source_duration, total_duration, length))
        return {"latent_length": length, "total_duration": total_duration,
                "source_duration": source_duration, "from_zero": False}
    source_frames = int(round(source_duration * 24.0))
    latent_length = source_frames + length
    return {"latent_length": latent_length,
            "total_duration": align(max(5, latent_length)) / 24.0,
            "source_duration": source_duration, "from_zero": False}


def context_text(source_duration, sent_duration, total_duration, from_zero):
    """The continuation sentence shown to the LLM before the footage part."""
    if from_zero:
        return ("This is the final frame of the footage to continue. The "
                "target video continues from it and lasts %.1f seconds; "
                "timestamps start at 00:00.000." % total_duration)
    if sent_duration is not None and sent_duration < source_duration - 0.05:
        opening = ("This is the last %.1fs of the footage to continue (full "
                   "length %.1fs)." % (sent_duration, source_duration))
    else:
        opening = "This is the footage to continue (%.1fs)." % source_duration
    return ("%s The target video continues it from %.1fs to %.1fs total."
            % (opening, source_duration, total_duration))


def parse_video_models(payload):
    """models?input_modalities=video response -> lowercase slug set."""
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return set()
    return {str(m.get("id")).lower() for m in data
            if isinstance(m, dict) and m.get("id")}


def fetch_video_models(get=None, now=None):
    """Video-capable model slugs, cached in memory for ~1 h.

    Returns None on network failure (the caller then relies on the runtime
    guards); failures are never cached, the next execution tries again.
    """
    if now is None:
        now = time.monotonic
    at = _MODELS_CACHE["at"]
    if at is not None and now() - at < VIDEO_MODELS_TTL:
        return _MODELS_CACHE["models"]
    if get is None:
        import requests
        get = requests.get
    try:
        resp = get(VIDEO_MODELS_URL, timeout=(10, 20))
        if getattr(resp, "status_code", 0) != 200:
            raise OSError("HTTP %s" % getattr(resp, "status_code", "?"))
        models = parse_video_models(resp.json())
    except Exception as e:
        logging.warning("h3_tools: video-model preflight failed (%s); relying "
                        "on the runtime guard", e)
        return None
    if not models:
        # a 200 with an unexpected/empty body must never hard-block every
        # model for an hour — treat it like a failed preflight, uncached
        logging.warning("h3_tools: video-model preflight returned no models; "
                        "relying on the runtime guard")
        return None
    _MODELS_CACHE.update(at=now(), models=models)
    return models


def _supports_video(model, supported):
    slug = model.lower()
    # routing suffixes (:nitro, :floor, :online, ...) are not in the catalog
    return slug in supported or slug.split(":")[0] in supported


def filter_models_for_video(models, supported):
    """Split the fallback chain into (usable, skipped) while a video part is
    present. supported=None (preflight failed) keeps everything usable."""
    if supported is None:
        return list(models), []
    usable = [m for m in models if _supports_video(m, supported)]
    skipped = [m for m in models if not _supports_video(m, supported)]
    return usable, skipped


def is_video_rejection(status, text):
    """OpenRouter rejects video for a non-video model with an instant 404."""
    return status == 404 and "input video" in str(text).lower()


def is_silent_video_drop(data):
    """HTTP 200 from a Google provider that silently dropped the video part:
    video_tokens == 0 despite a video being sent. Non-Google providers
    legitimately report 0, so the guard only fires on Google."""
    provider = str(data.get("provider") or "")
    if "google" not in provider.lower():
        return False
    usage = data.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return details.get("video_tokens") == 0
