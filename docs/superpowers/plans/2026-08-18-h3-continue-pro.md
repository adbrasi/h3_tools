# H3RefToVideoContinuePro Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans.
> Design source of truth: `SPEC.md` §12 (all owner decisions are locked there).

**Goal:** Add the `MiniMax H3 Reference to Video Continue (Pro)` node (continuation
context for the LLM enhancer) plus the shared `vision_format` combo on both nodes.

**Architecture:** All continuation logic that can be pure-python lives in a new
`h3_tools/continuation.py` (timeline math, context text, video-model support
detection) so it is testable without ComfyUI. `enhancer.py` grows a message
builder (`build_user_messages`) covering `default`/`cascade` and the two runtime
video guards. `media.py` grows the ComfyUI-side video re-encode
(`encode_video_context`). `node.py` refactors `_enhance` so both node classes
share it; the Continue node subclasses nothing — it is its own `io.ComfyNode`
calling the same helpers.

**Tech Stack:** stdlib + requests (lazy); native ComfyUI APIs
(`VideoFromComponents(...).save_to(BytesIO)`, `comfy.utils.common_upscale`,
`align_frame_count`, `io.Video.Input`).

## Global Constraints

- `H3RefToVideoPro` behavior unchanged except the added `vision_format` widget.
- Pure modules (`refs.py`, `continuation.py`, testable parts of `enhancer.py`)
  must import without ComfyUI/torch/requests.
- The API key never appears in logs, errors or cache files.
- Video/image continuation inputs feed ONLY the enhancer (never conditioning).

---

### Task 1: `continuation.py` (pure) + tests

Create `h3_tools/continuation.py`, `tests/test_continuation.py`.

Produces:
- `timeline(mode, length, source_duration, align) -> dict` with keys
  `latent_length` (int passed to the native node), `total_duration` (float s,
  post-align), `source_duration` (float | None), `from_zero` (bool).
  Rules per SPEC §12.3; `ContinuationError` when `total` and source ≥ total.
- `context_text(source_duration, sent_duration, total_duration, from_zero) -> str`
  — the sentence shown to the LLM (image variant when `source_duration is None`).
- `parse_video_models(payload) -> set[str]` from `GET /models?input_modalities=video`.
- `fetch_video_models(get=None, now=None) -> set | None` with ~1 h in-memory
  cache; `None` on network failure (caller then relies on runtime guards).
- `filter_models_for_video(models, supported) -> (usable, skipped)`;
  `supported=None` → everything usable.
- `is_video_rejection(status, text) -> bool` (404 + "no endpoints found …
  input video", case-insensitive).
- `is_silent_video_drop(data) -> bool` (Google provider + video_tokens == 0).

Steps: write failing tests (timeline both modes + image mode + error; context
text variants; models parsing/filter/rejection/silent-drop), implement, run
`pytest tests/test_continuation.py`, commit.

### Task 2: `enhancer.py` — message builder, video guards, cache key

Modify `h3_tools/enhancer.py`, `tests/test_enhancer.py`.

Produces:
- `build_user_messages(user_text, *, context_parts=None, vision_parts=None,
  vision_format="default") -> list[message]`:
  - default: one message; content `context_parts + [text] + vision_parts`
    (plain string when no parts — current behavior preserved).
  - cascade: one message per (text, media) pair from
    `context_parts`+`vision_parts`, then the main text message.
- `enhance(...)` gains `context_parts=None, vision_format="default",
  video_sent=False`; uses the builder; on non-200 checks
  `continuation.is_video_rejection` → clear EnhancerError suggesting
  `image_last_frame` or a video-capable model; on 200 with `video_sent` checks
  `continuation.is_silent_video_drop` → EnhancerError (lets the fallback model
  try).
- `cache_key(..., duration_mode=None, vision_format=None, context_sha=None)`.

Steps: failing tests (default single-message parity, cascade shapes, 404 video
message, silent-drop error, cache-key sensitivity ×3), implement, run
`pytest tests/test_enhancer.py`, commit.

### Task 3: `system_prompts.py` — `SYSTEM_PROMPTS_CONTINUE`

`_CONTINUATION` block injected between OBJECTIVE and `_TAIL`; same 3 preset
keys, exported as `SYSTEM_PROMPTS_CONTINUE`. Author constraints (SPEC §12.5):
footage = established fact (subjects, light, camera, motion continue from the
final frame); timestamps: video mode = absolute on the full timeline, the
continuation occupies the given range; image mode = start at 00:00.000;
`[video continuation]` prefix encouraged; the source footage gets NO @token and
no `<Video N>` label. Test: keys match, block present in continue and absent in
the plain prompts. Commit.

### Task 4: `media.py` — `encode_video_context`

`encode_video_context(video, max_edge=768, target_fps=10, max_seconds=30.0,
max_bytes=20*1024*1024) -> (data_url, full_duration, sent_duration)`:
components via `video.get_components()`; keep the LAST `max_seconds` (log
info); resample with `resample_indices(..., target_fps)`; downscale long side
to ≤ `max_edge` with `comfy.utils.common_upscale` snapped to even dims (h264
yuv420p); `VideoFromComponents(VideoComponents(images, audio=None,
frame_rate=target_fps)).save_to(BytesIO)`; `MediaError` above `max_bytes`.
Lazy imports; manual integration test only (needs ComfyUI). Commit.

### Task 5: `node.py` + `__init__.py` — both nodes

- Refactor `_enhance` signature: add `vision_format`, `system_prompts`
  (dict to pick presets from), `context_parts=None`, `context_sha=None`,
  `duration_mode=None`, `video_sent=False`, `supported_models=None`.
  Fallback-chain filtering via `continuation.filter_models_for_video` when
  `video_sent` (log skips; error when nothing remains).
- Pro schema gains `vision_format` combo after `enhancer_vision`; execute
  passes it through (cache key + cascade for reference images).
- New `H3RefToVideoContinuePro` class: Pro inputs + `video`
  (`io.Video.Input`, optional) + `image_last_frame` (`io.Image.Input`,
  optional) + `duration_mode` combo; presets from `SYSTEM_PROMPTS_CONTINUE`;
  execute per SPEC §12 (video wins + log; neither → error; timeline via
  `continuation.timeline(..., align=align_frame_count)`; context payload built
  only when the enhancer runs; preflight `fetch_video_models` only when video
  present; latent length from `timeline["latent_length"]`).
- Register both in `__init__.py`.
Commit.

### Task 6: frontend + docs + full gate

- `web/board.js`: `NODE_IDS` set gating both class names.
- `README.md`: Continue node section + `vision_format`; SPEC §12 header drops
  "pending implementation".
- Run the full suite `pytest`. Commit.

### Task 7: Opus 5 review

Dispatch Opus subagent review of the new/changed code; fix confirmed findings;
final commit + push (push pre-authorized for this repo).
