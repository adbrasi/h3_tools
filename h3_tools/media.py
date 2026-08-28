"""Media decode for references — thin wrappers over the native loader nodes.

Decoding calls the core loader node classes themselves (LoadImage, LoadVideo,
LoadAudio) so results are byte-identical to a hand-wired native graph. Video
components come from VIDEO.get_components(), the same call GetVideoComponents
makes -- its socket list grows between ComfyUI releases, its data does not.
The only pack-owned media logic is the 24 fps resample, which has no native
equivalent. All ComfyUI/torch/PIL imports are lazy so this
module imports (and resample_indices tests run) without ComfyUI.
"""

import logging
import math

TARGET_FPS = 24.0
MIN_VIDEO_FRAMES = 5  # native H3 floor for reference videos

# how any video reaches the LLM
LLM_MAX_EDGE = 768
LLM_FPS = 10.0
LLM_MAX_BYTES = 20 * 1024 * 1024

# reference-video resolution handed to the H3 encoder, by short edge.
# Resolution is the dominant cost of a reference video: the same clip at
# 480p samples several times faster than at 720p, and motion/camera
# structure survives the downscale intact.
REF_VIDEO_SIZES = {"source": None, "720p": 720, "480p": 480, "360p": 360}


class MediaError(ValueError):
    """User-facing media error; the message is shown as-is."""


def resample_indices(n_src, src_fps, target_fps=TARGET_FPS):
    """Source-frame index per output frame, preserving real duration.

    Output frame i takes source frame min(floor(i * src_fps / target + 0.5),
    n-1), for i in 0 .. floor(duration * target) - 1. Arithmetic rounding on
    purpose: Python's banker's round() gives an uneven duplication cadence at
    half-integer ratios (e.g. 12->24 fps duplicating frames 3x/1x/3x/1x).
    """
    if n_src <= 0 or src_fps <= 0:
        return []
    count = int(math.floor((n_src / src_fps) * target_fps))
    return [min(int(i * src_fps / target_fps + 0.5), n_src - 1) for i in range(count)]


def _has_audio(audio):
    if audio is None:
        return False
    waveform = audio.get("waveform")
    return waveform is not None and waveform.shape[-1] > 0


def load_image_ref(file):
    """IMAGE tensor [B, H, W, 3]; sizing is owned by the native H3 node."""
    import nodes
    return nodes.LoadImage().load_image(file)[0]


def load_video_ref(file, name):
    """{"frames": [T, H, W, C] at 24 fps, "audio": dict | None, "duration": float}"""
    from comfy_extras import nodes_video
    video = nodes_video.LoadVideo.execute(file=file).args[0]
    components = video.get_components()
    frames, audio = components.images, components.audio
    n_src = int(frames.shape[0])
    fps = float(components.frame_rate)
    if fps <= 0 or n_src <= 0:
        raise MediaError('could not read frames from reference video "@%s" (%s)'
                         % (name, file))
    indices = resample_indices(n_src, fps)
    if len(indices) < MIN_VIDEO_FRAMES:
        raise MediaError('reference video "@%s" is shorter than ~0.21s '
                         "(MiniMax H3 needs at least 5 frames at 24 fps)" % name)
    return {"frames": frames[indices],
            "audio": audio if _has_audio(audio) else None,
            "duration": n_src / fps}


def load_audio_ref(file):
    """{"audio": {"waveform": [1, C, L], "sample_rate": int}, "duration": float}"""
    from comfy_extras import nodes_audio
    audio = nodes_audio.LoadAudio.execute(audio=file).args[0]
    duration = audio["waveform"].shape[-1] / float(audio["sample_rate"])
    return {"audio": audio, "duration": duration}


def soundtrack_of(video_payload, name):
    audio = video_payload.get("audio")
    if not _has_audio(audio):
        raise MediaError('"@%s" has use_soundtrack enabled but the file has no '
                         "audio track" % name)
    return audio


def _fit_long_edge(frames, max_edge):
    """Scale [T, H, W, C] so the long edge is <= max_edge; dims stay even
    (yuv420p). Never scales up."""
    import comfy.utils

    h, w = int(frames.shape[1]), int(frames.shape[2])
    scale = min(1.0, max_edge / max(h, w))
    new_w = max(2, int(w * scale) // 2 * 2)
    new_h = max(2, int(h * scale) // 2 * 2)
    if (new_w, new_h) == (w, h):
        return frames
    return comfy.utils.common_upscale(
        frames.movedim(-1, 1), new_w, new_h, "bilinear", "disabled"
    ).movedim(1, -1)


def _mp4_data_url(frames, fps, what, max_bytes=LLM_MAX_BYTES):
    """[T, H, W, C] -> silent mp4 data URL via the native writer."""
    import base64
    import io as _io

    from comfy_api.latest import InputImpl, Types

    buf = _io.BytesIO()
    InputImpl.VideoFromComponents(
        Types.VideoComponents(images=frames, audio=None, frame_rate=fps)
    ).save_to(buf)
    blob = buf.getvalue()
    if len(blob) > max_bytes:
        raise MediaError("%s is %.1f MB after re-encoding (limit %d MB) — "
                         "shorten or downscale the source"
                         % (what, len(blob) / 1e6, max_bytes // (1024 * 1024)))
    return "data:video/mp4;base64," + base64.b64encode(blob).decode("ascii")


def llm_video_fps(duration, target_fps=LLM_FPS, max_frames=360):
    """Frame rate for a clip sent to the LLM whole.

    A reference video is read for its structure, so it is never truncated: a
    source too long for the frame budget loses frame rate, never its ending.
    """
    fps = min(target_fps, TARGET_FPS)
    if duration > 0 and duration * fps > max_frames:
        fps = round(max_frames / duration, 2)
    return fps


def encode_video_ref(payload, name, max_edge=LLM_MAX_EDGE, target_fps=LLM_FPS,
                     max_frames=360, max_bytes=LLM_MAX_BYTES):
    """Decoded reference video -> silent mp4 data URL for the LLM.

    A still frame says nothing about a camera path, a choreography or a
    timing — the three things a reference video exists to carry — so the
    enhancer gets the clip itself. The WHOLE clip, always: a source too long
    for the frame budget loses frame rate, never its ending.
    """
    frames = payload["frames"]  # already resampled to TARGET_FPS
    duration = float(payload["duration"])
    out_fps = llm_video_fps(duration, target_fps, max_frames)
    if out_fps < min(target_fps, TARGET_FPS):
        logging.info("h3_tools: reference video @%s is %.1fs; sending it to the "
                     "LLM at %.2f fps to stay inside the frame budget",
                     name, duration, out_fps)
    indices = resample_indices(int(frames.shape[0]), TARGET_FPS, out_fps)
    if not indices:
        raise MediaError('reference video "@%s" is too short to send to the '
                         "LLM (%.2fs)" % (name, duration))
    frames = _fit_long_edge(frames[indices], max_edge)
    return _mp4_data_url(frames, out_fps, 'reference video "@%s"' % name,
                         max_bytes)


def ref_video_target(h, w, size):
    """(new_h, new_w) for a reference-video downscale, or None to keep as is.

    `size` is a key of REF_VIDEO_SIZES (a short-edge target). Only ever scales
    down; dimensions snap to a multiple of 16 for the VAE.
    """
    short_edge = REF_VIDEO_SIZES.get(size)
    if not short_edge:
        return None
    scale = short_edge / min(h, w)
    if scale >= 1.0:
        return None
    new_h = max(16, int(round(h * scale)) // 16 * 16)
    new_w = max(16, int(round(w * scale)) // 16 * 16)
    return None if (new_h, new_w) == (h, w) else (new_h, new_w)


def fit_ref_video(frames, size, name):
    """Downscale a reference video's frames for the H3 encoder."""
    import comfy.utils

    h, w = int(frames.shape[1]), int(frames.shape[2])
    target = ref_video_target(h, w, size)
    if target is None:
        return frames
    new_h, new_w = target
    logging.info("h3_tools: downscaling reference video @%s from %dx%d to %dx%d "
                 "(%s)", name, w, h, new_w, new_h, size)
    return comfy.utils.common_upscale(
        frames.movedim(-1, 1), new_w, new_h, "bilinear", "disabled"
    ).movedim(1, -1)


def encode_video_context(video, max_edge=LLM_MAX_EDGE, target_fps=LLM_FPS,
                         max_seconds=30.0, max_bytes=LLM_MAX_BYTES):
    """VIDEO socket -> (data_url, full_duration, sent_duration) for the LLM.

    mp4/h264 via the native writer (VideoFromComponents.save_to), audio
    stripped, long side <= max_edge (dims snapped even for yuv420p), resampled
    down to ~target_fps. Longer sources send only the LAST max_seconds —
    continuation cares about the end. Duration is the only token-cost lever
    (Gemini bills ~258 tokens/s regardless of resolution); size only affects
    upload latency.
    """
    full_duration = float(video.get_duration())
    if full_duration > max_seconds:
        logging.info("h3_tools: continuation video is %.1fs; sending only the "
                     "last %.1fs to the LLM", full_duration, max_seconds)
        # trim BEFORE decoding: a negative start_time seeks from the end, so
        # only the window is ever turned into tensors (a long 1080p source
        # decoded whole would exhaust RAM before any post-decode cut)
        trimmed = None
        if hasattr(video, "as_trimmed"):
            trimmed = video.as_trimmed(start_time=-max_seconds,
                                       duration=max_seconds,
                                       strict_duration=False)
        if trimmed is not None:
            video = trimmed
    components = video.get_components()
    frames = components.images
    fps = float(components.frame_rate)
    n_src = int(frames.shape[0])
    if n_src <= 0 or fps <= 0:
        raise MediaError("could not read frames from the video to continue")
    if n_src / fps > max_seconds + 0.05:  # as_trimmed unavailable or a no-op
        keep = int(round(max_seconds * fps))
        frames = frames[n_src - keep:]
    sent_duration = int(frames.shape[0]) / fps
    out_fps = min(target_fps, fps)
    if out_fps < fps:
        indices = resample_indices(int(frames.shape[0]), fps, out_fps)
        if not indices:
            raise MediaError("the video to continue is shorter than %.2fs — "
                             "connect image_last_frame instead" % (1.0 / out_fps))
        frames = frames[indices]
    url = _mp4_data_url(_fit_long_edge(frames, max_edge), out_fps,
                        "the video to continue", max_bytes)
    return url, full_duration, sent_duration


def image_data_url(image, max_edge=1024, quality=85):
    """IMAGE tensor -> JPEG data URL for the enhancer's vision parts."""
    import base64
    import io as _io

    import numpy as np
    from PIL import Image

    arr = image
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().float().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    arr = (np.clip(arr[..., :3], 0.0, 1.0) * 255.0).astype("uint8")
    img = Image.fromarray(arr)
    scale = max_edge / max(img.size)
    if scale < 1.0:
        img = img.resize((max(1, round(img.size[0] * scale)),
                          max(1, round(img.size[1] * scale))), Image.LANCZOS)
    buf = _io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
