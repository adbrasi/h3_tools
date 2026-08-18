"""Media decode for references — thin wrappers over the native loader nodes.

Decoding calls the core loader node classes themselves (LoadImage, LoadVideo +
GetVideoComponents, LoadAudio) so results are byte-identical to a hand-wired
native graph. The only pack-owned media logic is the 24 fps resample, which
has no native equivalent. All ComfyUI/torch/PIL imports are lazy so this
module imports (and resample_indices tests run) without ComfyUI.
"""

import math

TARGET_FPS = 24.0
MIN_VIDEO_FRAMES = 5  # native H3 floor for reference videos


class MediaError(ValueError):
    """User-facing media error; the message is shown as-is."""


def resample_indices(n_src, src_fps, target_fps=TARGET_FPS):
    """Source-frame index per output frame, preserving real duration.

    Output frame i takes source frame min(round(i * src_fps / target), n-1),
    for i in 0 .. floor(duration * target) - 1.
    """
    if n_src <= 0 or src_fps <= 0:
        return []
    count = int(math.floor((n_src / src_fps) * target_fps))
    return [min(round(i * src_fps / target_fps), n_src - 1) for i in range(count)]


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
    frames, audio, fps, _bit_depth = nodes_video.GetVideoComponents.execute(video=video).args
    n_src = int(frames.shape[0])
    fps = float(fps)
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


def middle_frame(frames):
    """One [1, H, W, C] frame from the middle of the clip (for enhancer vision)."""
    mid = frames.shape[0] // 2
    return frames[mid:mid + 1]


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
