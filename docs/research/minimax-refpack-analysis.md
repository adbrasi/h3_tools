# Technical Map: ComfyUI-MiniMaxRefPack (v0.3.5, by Hearmeman24)

Analysis date: 2026-08-18, against a fresh clone of
https://github.com/Hearmeman24/ComfyUI-MiniMaxRefPack. Line refs are to that tree.
Python dep: only `requests`; torch/PIL/numpy/av lazy-imported via ComfyUI. 15 pytest
files cover refs/media/prompt/routes/migration/workflow.

---

## 1. Node inventory

**Exactly one node.** `MiniMaxH3ReferencePack` (display "MiniMax References Manager"),
`minimax_refpack/nodes.py:92-415`. Plain V1 node (INPUT_TYPES / IS_CHANGED), category
"MiniMax H3", not io.ComfyNode v3.

**Inputs** (nodes.py:110-218) — order is load-bearing (see §8):
- required: `direction` (multiline STRING; hidden on canvas, bound to a DOM textarea).
- optional: `references_json` (hidden STRING, "do not hand-edit"), `system_prompt`
  (hidden STRING, edited via modal), `prompt_provider` (combo:
  openrouter/local/none), `openrouter_api_key` (STRING), `openrouter_model` (combo
  from `prompt.available_models()`), `reasoning_effort` (none/low/medium/high),
  `api_base` (STRING), `local_model_slug` (STRING), `job_type`
  (auto/standard/replacement), `width`/`height` INT, `length_seconds` FLOAT,
  `max_reference_edge` INT (default 2048).
- Legacy kwargs `use_openrouter`, `model`, `model_override` still accepted by
  `build()`/`IS_CHANGED` (nodes.py:232, 261-267) for pre-0.3.2/0.3.3 API prompts.

**Outputs**: 20, defined in `refs.output_names()` (refs.py:258-279), **append-only by
contract** (links stored by slot index): `image_1..9` (IMAGE), `video_1..3` (IMAGE
batch), `video_audio_1..3` (AUDIO), `audio_1..3` (AUDIO), `prompt` (STRING), `debug`
(STRING). Empty slots emit `None` (strings emit `""`), see `refs.empty_outputs()`.

**Relationship to native `comfy_extras/nodes_minimax_h3.py`** (the pack cites ComfyUI
v0.32.0 line numbers throughout):
- **Nothing native is imported or wrapped.** The pack is a pure *fan-out upstream* of
  the native `MiniMaxH3ReferenceToVideo`. The example workflow wires image_1..9 →
  `ref_images.ref_image_0..8`, video_1..3 → `ref_videos.ref_video_0..2`,
  video_audio_1..3 → `ref_video_audios.ref_video_audio_0..2`, audio_1..3 →
  `ref_audios.ref_audio_0..2`, `prompt` → the native node's `prompt` input (19
  hand-wired links; `debug` goes to a Display Any node). Design rationale in
  nodes.py:6-9: the native node already skips None per reference group (native
  :219,236,270), so "always connected, mostly None" is what upstream anticipates; the
  native node stays the sole encoder/VAE/tokenizer owner.
- **Reused knowledge, reimplemented code**: caps 9/3/3 copied from the native Autogrow
  groups (:183-195, refs.py:29-33); the tag-ordinal rule read out of native :216-274
  (refs.py docstring); the ≥5 frame minimum (:250) enforced early in
  `media._decode_video`; the image short-edge behavior (:29, :301) motivates
  `max_reference_edge`. 24fps resample is the pack's own addition — the native node
  never touches framerate (media.py:29-31).
- **Native code actually reused at runtime**:
  `comfy_api.latest._input_impl.video_types.VideoFromFile` for video decode and
  `comfy_extras.nodes_audio.load` for audio decode, both behind test-friendly
  indirections `media._video_from_file_cls()`/`_audio_load_fn()` (media.py:75-86).

`IS_CHANGED` (nodes.py:225-253) returns a `|`-joined key of: `_files_signature`
(sha256 of mtime_ns+size of every referenced file, nodes.py:73-89 — exists because
re-uploading under the same filename with `overwrite=true` can leave
`references_json` byte-identical) plus direction, model, references_json,
system_prompt, width/height/length, provider, effort, job_type, cap, api_base, slug.
The API key is deliberately excluded.

## 2. Media import

- **Upload**: the frontend uses **stock ComfyUI `/upload/image`** with `type=input`,
  `overwrite=true` for all three kinds — images, videos and audio all go through the
  same endpoint (`apiUpload`, refpack.js:510-519). No custom upload route. No
  drag-&-drop onto the node (only file pickers: three toolbar buttons + per-row "+"
  squares). Files land in ComfyUI's **input directory** under whatever name the server
  returns.
- **Reference at execution**: filenames only, inside the hidden `references_json`
  widget value — JSON `{"references":[{"kind","file",("use_soundtrack"),("crop"),
  ("trim")}...]}` (`refs.Reference.to_dict`, refs.py:97-105). No custom socket type;
  media never travels through links into the pack node.
- **Decoding at execution** (`media.py`): images via **PIL** (`load_image`,
  EXIF-transposed, fraction-crop via `_crop_box` half-up rounding, then `thumbnail()`
  to `max_reference_edge`); video via **VideoFromFile.get_components()** (core, PyAV
  underneath), then the pack's own `resample_indices` to 24fps preserving real
  duration, trim on source frames with 1e-9 epsilon, crop on tensor slices
  (media.py:142-202); audio via core `comfy_extras.nodes_audio.load` + batch-dim wrap
  + `_slice_audio` trim. For the VLM payload, an untouched video's raw file bytes are
  sent as-is if the container is in `VLM_VIDEO_MIMES` (mp4/mov/webm/mpeg),
  otherwise/when edited it is **re-encoded in-memory to mp4 with PyAV/libx264,
  video-only** (`_transcode_window`, media.py:243-293), soundtrack re-sent separately
  as PCM16 WAV.
- **Previews**: `GET /minimax_refpack/thumb` renders a PNG server-side (PIL + PyAV
  seek-then-decode for a video frame at the trim in-point, crop applied through the
  same `_crop_box`); playable previews use stock `/view?filename=&type=input` directly
  (refpack.js:533-535).

## 3. Reference management UI (web/refpack.js, 2845 lines + refpack.css)

- **Architecture**: a hybrid — real DOM for the toolbar row (⬆Image/⬆Video/⬆Audio,
  Save config, Load config, Local LLM, ⚙) and the prompt textarea; a **single
  `<canvas>`** ("black slab") for all tiles, drawn by one `draw()` function with
  hand-rolled hit regions (`node._mmrpHit.regions`, scanned in reverse;
  refpack.js:1104-1300). No per-cell DOM ("kept producing overlap/leakage bugs"). One
  shared `<video>` overlay + one `<audio>` per node for click-to-play. **Fixed node
  size 1340px wide** — `resizable=false` plus monkeypatched
  `onResize`/`computeSize`/`setSize` all clamp to `fixedSize()`
  (refpack.js:2712-2741); CSS pins matching heights, the two must agree by hand.
- **Operations**: add (upload or "+" square, hard caps 9/3/3 with an `alert` on
  overflow), delete (chip or Delete/Backspace on selection — a capture-phase document
  keydown handler that swallows the event so ComfyUI doesn't delete the node,
  refpack.js:2674-2691), select (red stroke), per-video soundtrack toggle ♪ (gated on
  `/minimax_refpack/probe` `has_audio`; silent clip = permanently struck through;
  probe answer `false` **force-clears** a saved `use_soundtrack` via `syncProbes`,
  refpack.js:588-605), crop/trim modal (scissors chip or double-click; fraction-space
  rect + aspect presets, trim bar + 2dp fields, "play edit" preview that reframes the
  media to the crop). **No reorder and no rename** — README line 9 admits it:
  "reorder nothing". Order within a kind is insertion order; deleting ref N renumbers
  everything after it.
- **Serialization**: every mutation goes through `applyRefs` → `syncReferencesWidget`
  (refpack.js:1440-1458), which writes the JSON into the ordinary `references_json`
  STRING widget — so it serializes into `widgets_values` in the workflow JSON and
  into the API prompt like any widget. A UI-only `missing: true` flag rides in that
  JSON (flagged by comparing against `/minimax_refpack/files` listings on config
  load); Python's `Reference.from_dict` just ignores it. Three widgets
  (`references_json`, `direction`, `system_prompt`) are hidden with a heavy
  `hideWidget()` (Object.defineProperty getter-locks on `hidden`/`type`,
  `computeSize=[0,0]`, plus a 50ms×1s poll for the async-created V3 DOM element;
  refpack.js:629-674).
- **Thumbnails**: module-wide `thumbCache` keyed by **filename only**, with a retry
  ladder (1s→15s), plus retry-all on window `focus`/`online` (exists because
  `thumb_route` decodes synchronously on the aiohttp loop and stalls during pod boot;
  refpack.js:720-808).
- **Config portability**: "Save config" downloads a JSON to the browser; "Load config"
  reads one via file picker; no server state (refpack.js:1659-1856). Restores
  direction + references and cross-checks file existence per pod.

## 4. Prompt & mention system

- **No @mention/autocomplete/tag-insertion.** The `direction` textarea is a plain
  `<textarea>`; the tags are simply painted on each tile's badge (`<Picture 2>`,
  `vid_audio <Audio 1>`, `<Audio 1>`), and the user types tags by hand if they want to
  address a reference. The mapping from typed tags to actual assets is delegated
  entirely to the LLM.
- **Ordinal rule** (mirrored in two places, `refs.ReferenceSet.assign_tags()`
  refs.py:222-245 and JS `assignTags()` refpack.js:302-326): (1) images in slot order
  → `<Picture 1..n>`; (2) per video in slot order, its soundtrack (if
  `use_soundtrack`) takes the **next `<Audio j>` first**, then the video takes
  `<Video k>`; (3) standalone audio continues the same `<Audio j>` counter. Per-type,
  1-based, over the compacted list — copied from native nodes_minimax_h3.py:216-274 so
  the tags the LLM writes match what the native tokenizer will assign to the wired
  sockets.
- **Payload mapping** (`prompt._build_content`, prompt.py:463-604): one leading text
  part — `USER DIRECTION:` + optional `TARGET FORMAT:` (frame + gcd aspect +
  duration) + a manifest — then each asset preceded by its own inline label part
  (`image_reference <Picture N> (file) - the next image:` etc.). Two manifest styles:
  verbose `Reference manifest:` for OpenRouter, and a concise grouped `REFERENCES`
  block for degraded (local) endpoints, where a video's 6 sampled stills get **one**
  label declaring "the next K images are still frames from ONE video" and withheld
  audio is listed as `<Audio N> (not sent)` so numbering never shifts. Tags are never
  renumbered on degrade (comment at prompt.py:481-484: tags are the contract with the
  sockets).

## 5. LLM auto-prompting (prompt.py + endpoint.py)

- **When it runs**: **during graph execution**, inside `build()` (nodes.py:354-369)
  via synchronous `requests.post` — no frontend "enhance" button. The only frontend
  LLM interaction is the Local LLM discovery modal.
- **Providers** (`endpoint.resolve`, endpoint.py:134-169): `openrouter` (fixed base
  `https://openrouter.ai/api/v1`, accepts text+image+audio+video, chat timeout 120s,
  sends `reasoning:{effort}`), `local` (any OpenAI-compatible `api_base` ending in
  /v1, accepts text+image only, chat timeout **900s**, never sends `reasoning`),
  `none` (no call; `direction` passes through verbatim, nodes.py:334-341). Empty refs
  + empty direction skips the call.
- **Key resolution** (`_key_for`, prompt.py:223-237): OpenRouter → node box, then env
  `OPENROUTER_API_KEY`, then `LLM_KEY`, else raise. Local → **only** the typed key,
  never env (deliberate: a pasted api_base must not receive an ambient paid
  credential). Key is only ever a header; scrubbed from raised messages and the debug
  socket by literal string replace (nodes.py:379-403); log fields ending in
  key/token/secret/password print `***` (logs.py:31).
- **Model routing**: two mutually invisible fields — `openrouter_model` (dropdown
  filtered to models whose OpenRouter `architecture.input_modalities ⊇
  {text,image,audio,video}`, default `google/gemini-3-flash-preview`, fetched in
  `available_models()` **inside INPUT_TYPES** with 5s timeout + 1h cache + hardcoded
  fallback) vs `local_model_slug` (typed/picked; empty on local = hard error,
  `_model_for` nodes.py:44-70, born of a real 400 when a shared field leaked a local
  slug to OpenRouter).
- **job_type=auto**: `classify_mode` (prompt.py:370-420) — only runs with ≥1 video AND
  ≥1 image AND non-empty direction; a ~141-token system prompt asks for one word
  REPLACEMENT/STANDARD; on OpenRouter it uses `google/gemini-2.5-flash-lite`
  (max_tokens 6, temp 0, 20s timeout), locally it reuses the writer's own
  model/endpoint. **Every failure falls back to "standard" silently.**
- **System prompt strategy**: two separate ~22KB/~7KB register files, read at call
  time. `system_prompt.md` (standard): forces the six-section Ref2VA format
  (subject_definitions / summary / retention_analysis / detailed_description /
  overall_soundscape / non_diegetic_music), precedence rules (direction outranks
  references, scoped not total), per-type reference handling, degraded-input rules
  ("you did NOT watch that clip", "invent no voice for `<Audio N>` (not sent)"), fixed
  camera-motion/style/retention-marker/task-type vocabularies,
  `[Shot N]`+`MM:SS.mmm` timestamp grammar, `<d>[English]…</d>` dialogue rules, an
  extensive realism section, banned-vocabulary lists, and an 8-point self-check.
  `system_prompt_replacement.md`: `[video editing]` register — swap one thing in the
  plate video for the image's subject. A non-blank `system_prompt` widget **overrides
  both registers verbatim** (prompt.py:767-768). Tag preservation is enforced purely
  by prompt instruction + manifest/label redundancy — no post-hoc validation of the
  LLM's output.
- **Error handling**: `PromptError` on timeout (actionable text), non-200 (first 200
  chars of provider error), malformed body, empty completion, and the observed
  "reasoning-only, content empty" local-model case (prompt.py:891-907). `build()`
  re-raises as ValueError so the queue fails visibly — **no retry, no fallback
  prompt**; only `classify_mode` degrades silently. The full rendered payload (base64
  stubbed, part index, user-message-first display reorder) is always pushed into the
  `debug` output even when the call fails (prompt.py:793-804).

## 6. Custom routes (all GET, registered at import of `minimax_refpack/routes.py`)

| Route | Handler | Frontend use |
|---|---|---|
| `/minimax_refpack/probe?file=` | `probe_route` → `media.probe` | soundtrack-toggle gating (`has_audio`), kind/duration |
| `/minimax_refpack/thumb?file=&crop=x,y,w,h&t=s` | `thumb_route` → `media.thumbnail_png` | tile thumbnails, crop/trim baked in |
| `/minimax_refpack/files?kind=` | `list_files_route` | missing-file check on config load |
| `/minimax_refpack/system_prompt` | `system_prompt_route` (read-only) | ⚙ modal's default prefill |
| `/minimax_refpack/detect[?base=]` | `detect_route` | Local LLM modal sweep (loopback-only, SSRF defense) |

Path safety: `_safe_join` (routes.py:37-54) = `..` substring reject + abspath +
commonpath. No write route anywhere (uploads use core `/upload/image`).

## 7. API/headless behavior

**Executes fully headless.** `build()` needs only: files present in the input dir, a
valid `references_json` string, and (for openrouter) an env or widget key. What
breaks/hurts headless:
- The client must **hand-author `references_json`** and upload files itself via
  `/upload/image` first.
- `openrouter_model` is a **combo validated against `available_models()`** — an
  offline/firewalled server returns only the fallback, so a stored prompt with any
  other model id fails ComfyUI's combo validation. Also `INPUT_TYPES` performs a
  network GET at node-enumeration time.
- The LLM call is synchronous inside execution — up to 120s/900s blocking the worker.
- The 18 media outputs must already be wired to the native node in the stored graph.

## 8. Weaknesses / bugs / design smells

1. **Positional `widgets_values` treated as a wire format** — three shipped layouts,
   a migration engine (`ORDER_0_3_1/2/3`, `detectLayout` by value shape,
   `remapWidgetValues`) + server-side legacy kwargs. Heuristic; browser-only.
2. **Confirmed bug — Save/Load config loses the model**: `buildConfig` reads widget
   `"model"`, renamed `openrouter_model` in 0.3.3. Configs save `model: ""` forever.
3. **Hidden state in a STRING widget** + same-filename `overwrite=true` uploads forced
   the `_files_signature` mtime/size hash into IS_CHANGED — workaround on workaround.
4. **API key serialized into workflow JSON** (widget), leak defense via literal
   string-replace on error/debug text — best-effort.
5. **Network in `INPUT_TYPES`**, combo-validation coupled to a live third-party API.
6. **Blocking sync HTTP inside execution** (up to 900s); `thumb_route`/`probe_route`
   decode synchronously on the aiohttp event loop — server stalls at boot, JS grew a
   retry ladder to paper over it.
7. **Module-wide caches keyed by filename only** (`thumbCache`, `probeCache`): two
   node instances sharing a file with different crops share one thumb; `syncProbes`
   mutates saved `use_soundtrack` behind the user's back.
8. **No reorder, no rename, tags shift on delete** — deleting `<Picture 1>` silently
   renumbers everything; no mention system to re-bind.
9. **Extreme frontend coupling to litegraph internals**: defineProperty-locked
   `hidden`/`type`, 50ms DOM polls, monkeypatched `computeSize`/`setSize`/`onResize`.
10. **Fixed 1340px node** with JS/CSS lockstep constants.
11. **Silent degradation in `classify_mode`**; typo'd provider silently normalized to
    "openrouter" (data sent to OpenRouter unintentionally).
12. **Dual implementation of the tag rule** (Python + JS) — hand-mirrored, drift-prone.
13. Minor: `..` substring reject blocks legit filenames; `alert()` for errors; debug
    output ships the whole system prompt; documented retention_analysis contradiction.

**Strengths worth copying**: append-only 20-socket contract with tag ordinals matched
byte-for-byte to the native tokenizer; pure/torch-free `refs.py` as a frozen tested
contract; endpoint value object with capability sets + degraded-manifest honesty;
key-egress asymmetry (env keys never follow a typed URL); loopback-only discovery;
pre-call debug sink surviving failures.
