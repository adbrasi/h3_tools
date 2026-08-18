# h3_tools — MiniMax H3 Reference to Video (Pro)

Design spec, 2026-08-18. Status: approved by owner, pending implementation.

One ComfyUI custom node that replaces the native `MiniMaxH3ReferenceToVideo` workflow of
"18 separate loader nodes + hand-written `<Picture 1>` tags + memorized ordering" with:

- a built-in **media importer** (image / video / audio upload with previews, up to 18 refs),
- an **@mention prompt** (`@garota` instead of `<Picture 1>`), with an autocomplete popup
  showing real thumbnails (arrow keys + Enter, Dreamina-style),
- an optional **LLM prompt enhancer** (OpenRouter),
- first-class **API/headless** operation: the workflow JSON is self-contained text, no
  frontend required for anything to work.

The node's job is **text and organization only**. It never touches model code, sampling,
or how conditioning works — all encoding is delegated to the untouched native node.

---

## 1. Context and target environment

- **Model**: MiniMax H3, an open-weights audio+video generation DiT. Its ref2va
  (reference-to-video) mode accepts up to 18 references: 9 images, 3 videos,
  3 video soundtracks, 3 standalone audios.
- **Target ComfyUI**: current `master` (has `comfy_extras/nodes_minimax_h3.py`,
  `comfy/ldm/minimax/`, `comfy/text_encoders/minimax.py`). Any install too old to have
  `comfy_extras.nodes_minimax_h3` cannot run H3 at all; the pack should fail to import
  with a clear message in that case ("h3_tools requires a ComfyUI version with MiniMax
  H3 support").
- **Node API**: v3 schema (`comfy_api.latest.io.ComfyNode` / `io.Schema`), same style as
  the native H3 nodes. `validate_inputs`, `fingerprint_inputs` and `NodeOutput` (with
  `.args`) are available on this API.
- **Language/runtime**: Python backend + one vanilla-JS frontend extension. No build
  step, no npm. Python deps: only `requests` (everything else — torch, PIL, av — ships
  with ComfyUI).

### 1.1 Native mechanics you must not break (source of truth)

From `comfy_extras/nodes_minimax_h3.py` and `comfy/text_encoders/minimax.py` (read both
before implementing):

- **Tag grammar**: the prompt refers to references as literal text tags `<Picture i>`,
  `<Video k>`, `<Audio j>` — 1-based ordinals **per type**.
- **Ordinal assignment**: the tokenizer assigns ordinals in the order reference items
  are presented (`minimax_ref_items` list). The native node presents them in a fixed
  order: **all images first, then videos, then standalone audios**; a video's
  soundtrack, when present, consumes the **next `<Audio j>` ordinal immediately before
  its `<Video k>`** (the `<Audio j>: ` label is emitted right before `<Video k>: `).
  Standalone audios continue the same audio counter after all videos.
- **Native input schema** (`MiniMaxH3ReferenceToVideo`): `clip` (CLIP), `vae` (video
  VAE), `audio_vae` (audio VAE), `prompt` (string), `width`/`height` (int, step 32,
  defaults 1344/768), `length` (int frames @24fps, default 124, min 5, step 17, snapped
  up to the `17k+5` grid), `ref_image_size` (combo `match`|`max`), plus four Autogrow
  dict inputs: `ref_images` (max 9 IMAGE), `ref_videos` (max 3 IMAGE batches),
  `ref_video_audios` (max 3 AUDIO), `ref_audios` (max 3 AUDIO).
- **Soundtrack pairing convention**: `ref_video_audios["ref_video_audio_N"]` is the
  soundtrack of `ref_videos["ref_video_N"]` (paired by trailing index in the key).
- **Video constraints**: reference video frames are expected at 24 fps; a video needs
  **≥ 5 frames**; the native node trims frame count down to the `17k+5` grid
  (5, 22, 39, …) and caps it at the generation's frame count.
- **Outputs**: positive CONDITIONING (carrying `minimax_refs` metadata) + AV LATENT
  (NestedTensor video+audio pair).

## 2. Product decisions (locked with the owner)

1. **Media enters by upload only.** No IMAGE/VIDEO/AUDIO graph input sockets for
   references. Files are uploaded from the node UI to ComfyUI's standard `input/`
   directory; headless users upload via the standard `/upload/image` endpoint and pass
   filenames.
2. **Single node.** Importer, @mentions, enhancer, and encode delegation all live in
   one node. (The enhancer cannot be a separate node because references live in this
   node's widget.)
3. **Mention rendering**: standard textarea + highlight overlay (`@nome` colored inline;
   the value stays plain text). NOT contentEditable chips — that was the biggest bug
   source in the competitor analyzed (custom undo stack, caret sentinels, breaks on
   ComfyUI frontend updates).
4. **Names are automatic, renaming is optional.** Importing `garota.png` creates the
   reference `@garota`; the user only ever types `@` and picks from the popup.
5. **This is a text job**: the backend converts `@name` → `<Picture X>` and organizes
   media. Model/conditioning behavior is the native node's, unchanged.
6. **Maximum native reuse, including decode.** Media decoding calls the core loader
   node classes themselves (`LoadImage`, `LoadVideo` + `GetVideoComponents`,
   `LoadAudio`) so behavior is byte-identical to a hand-wired native graph; encoding
   is delegated to the native H3 node (§5.5). The only pack-owned media logic is the
   24 fps resample, which has no native equivalent. Custom code concentrates on text
   and organization (`refs.py`).

## 3. The node

- **node_id**: `H3RefToVideoPro`
- **display_name**: `MiniMax H3 Reference to Video (Pro)`
- **category**: `conditioning/minimax`
- **API style**: v3 `io.ComfyNode` with `define_schema`.

### 3.1 Inputs

Connections:

| name | type | notes |
|---|---|---|
| `clip` | CLIP | the MiniMax H3 text encoder (Qwen3-VL) |
| `vae` | VAE | video VAE |
| `audio_vae` | VAE | audio VAE |

Widgets (all serialize normally into `widgets_values` / API JSON — **no hidden
transport inputs, no state in `node.properties`**):

| name | type | default | notes |
|---|---|---|---|
| `prompt` | String, multiline, `dynamic_prompts=False` | `""` | text with `@name` mentions; literal `<Picture i>` etc. tags also allowed and passed through untouched |
| `references` | String (JSON) | `"[]"` | managed by the board UI; hidden by the frontend; hand-writable headless. Schema in §4 |
| `width` | Int, step 32 | 1344 | same as native |
| `height` | Int, step 32 | 768 | same as native |
| `length` | Int, min 5, max 3600, step 17 | 124 | frames @24fps, same as native |
| `ref_image_size` | Combo `match`\|`max` | `match` | passed through to native |
| `enhance_prompt` | Boolean | `false` | master switch for the LLM enhancer |
| `enhancer_model` | String | `"google/gemini-3-flash-preview"` | free text, OpenRouter model slug. NEVER a combo validated against a live API (breaks offline/headless validation) |
| `openrouter_api_key` | String | `""` | empty → fall back to env `OPENROUTER_API_KEY`; missing both (with enhancer on) is an execution error |
| `enhancer_vision` | Boolean | `false` | send reference images (and 1 frame per video) to the LLM |
| `system_prompt_preset` | Combo (keys of `SYSTEM_PROMPTS`) | `default` | built-in system prompt presets synthesized from the official H3 guide (`h3_tools/system_prompts.py`): `default`, `multishot`, `single_take`, `dialogue`, `music_video` |
| `system_prompt_override` | STRING **connection** (`force_input`), optional | — | a connected non-empty string replaces the chosen preset verbatim |
| `enhancer_seed` | Int | 0 | part of the enhancer cache key; bump to re-roll the LLM |

All enhancer widgets are plain optional widgets — no dynamic show/hide of widget rows
in v1 (competitors' conditional-row code was a major fragility source).

### 3.2 Outputs

| # | type | name | content |
|---|---|---|---|
| 0 | CONDITIONING | `positive` | from the native node, untouched |
| 1 | LATENT | `latent` | AV latent from the native node, untouched |
| 2 | STRING | `final_prompt` | the exact prompt string handed to the native node (post-enhancer, post-substitution, including provenance lines) — for inspection/preview nodes |

## 4. `references` JSON schema

A JSON **array** (bare list, no wrapper object). Each element:

```json
{
  "name": "garota",            // optional; derived from filename when absent
  "type": "image",             // required: "image" | "video" | "audio"
  "file": "h3_refs/garota.png",// required: path relative to the input directory,
                               // ComfyUI annotated-filename rules apply
  "use_soundtrack": true       // optional, videos only, default false
}
```

Rules:

- **Order matters** within each type: it defines ordinal numbering (first image in the
  array = `<Picture 1>`, etc.). Order across types does not matter (backend groups by
  type in the native order).
- **`name`**: `^[a-z0-9_]{1,64}$`. When absent, derive from the file's basename:
  lowercase, strip extension, replace every non-`[a-z0-9]` run with `_`, trim `_`,
  prefix with `m_` if it would start with a digit or be empty. Collisions (given or
  derived) get `_2`, `_3`, … suffixes **only for derived names**; duplicated explicit
  names are a validation error.
- **`file`**: resolved with `folder_paths.get_annotated_filepath(file)` (so
  `subfolder/name.ext` and `name.ext [input]` forms both work). Must exist.
- Unknown keys are ignored (forward compatibility). Schema evolution is additive-only.
- Limits (validation errors when exceeded): ≤ 9 images, ≤ 3 videos, ≤ 3 audios
  (standalone), and ≤ 3 videos with `use_soundtrack` (implied by the video cap).

## 5. Backend execution flow

Module layout keeps the pure logic testable without ComfyUI:

```
h3_tools/
├── __init__.py               # guarded import of comfy_extras.nodes_minimax_h3;
│                             # NODE_CLASS_MAPPINGS via the v3 extension entrypoint
├── h3_tools/
│   ├── refs.py               # PURE (stdlib only): JSON parse/validate, name
│   │                         # derivation, ordinal assignment, @→tag substitution,
│   │                         # provenance lines, error messages
│   ├── media.py              # decode: image (PIL), video (VideoFromFile + 24fps
│   │                         # resample + soundtrack extraction), audio (core loader)
│   ├── enhancer.py           # OpenRouter call, system prompt, JSON parsing,
│   │                         # retries, disk cache
│   └── node.py               # the io.ComfyNode; wires the above; delegates encode
├── web/
│   ├── board.js              # the whole frontend extension
│   └── board.css
├── tests/                    # pytest, runs WITHOUT ComfyUI (see §10)
├── SPEC.md                   # this file
└── pyproject.toml            # ComfyUI registry metadata; requests dependency
```

### 5.1 `validate_inputs` (runs before execution; fail fast, fail clear)

- `references` parses as JSON array matching §4 (types, names, caps, no duplicate
  explicit names).
- Every referenced file exists.
- Every `@name` mention in `prompt` (see §5.3 for the token grammar) resolves to a
  reference name. Unknown mention → error string listing the valid names.
- Returns `True` or a human-actionable error string. Never raises for user data errors.

### 5.2 Media loading (`media.py`)

Decoding literally calls the core loader node classes — not reimplementations of
them — so results are byte-identical to a hand-wired native graph:

- **Image**: `nodes.LoadImage().load_image(annotated_file)` → take the IMAGE output
  (`[1, H, W, 3]` float 0..1; EXIF, RGB, multi-frame handling all native), drop the
  mask. No resizing here — the native H3 node owns reference sizing via
  `ref_image_size`.
- **Video**: `comfy_extras.nodes_video.LoadVideo.execute(file=...)` → VIDEO object →
  `comfy_extras.nodes_video.GetVideoComponents.execute(video=...)` → frames
  `[T, H, W, C]`, audio, fps. (Equivalent to wiring LoadVideo → GetVideoComponents.)
  Then the pack's one piece of media logic, **resample to 24 fps preserving real
  duration** (no native equivalent exists): output frame `i` takes source frame
  `min(floor(i * src_fps / 24 + 0.5), n_src - 1)`, for `i` in
  `0 .. floor(duration * 24) - 1` (arithmetic rounding — banker's `round()`
  gives an uneven duplication cadence at half-integer ratios). Enforce ≥ 5
  frames after resampling (error: "reference video @name is shorter than ~0.21s").
  Do NOT trim to the `17k+5` grid here — the native H3 node does that.
- **Video soundtrack** (`use_soundtrack: true`): the audio output of
  `GetVideoComponents`. If the container has no audio track → clear execution error
  ("@name has use_soundtrack enabled but the file has no audio track"), never a silent
  skip.
- **Audio**: `comfy_extras.nodes_audio.LoadAudio.execute(audio=annotated_file)` →
  `{"waveform": [1, C, L], "sample_rate": sr}`. No resampling — the native H3 node
  resamples to the audio VAE's rate.

If a loader class's method signature shifts upstream, prefer adapting the call over
copying its body; these are stable public nodes that API workflows depend on.

### 5.3 Mention grammar and resolution (`refs.py`)

- **Token regex**: `(?<![A-Za-z0-9_])@([A-Za-z0-9_]{1,64})(?![A-Za-z0-9_])` — an `@`
  not glued to a preceding word character, capturing the name; the trailing lookahead
  keeps >64-char tokens (never valid names) as plain text instead of partially
  matching. Matching against reference names is case-insensitive (names are stored
  lowercase).
- **Ordinal assignment** (must mirror the native presentation exactly):
  1. images in array order → `<Picture 1..N>`; picture counter `i`.
  2. videos in array order → `<Video 1..K>`; a video with `use_soundtrack` first
     consumes the next audio ordinal `j` (its soundtrack is `<Audio j>`), then the
     video takes `<Video k>`.
  3. standalone audios in array order continue the audio counter → `<Audio j>`.
- **Substitution**: every resolved `@name` in the prompt is replaced by its tag.
  Literal `<Picture i>` / `<Video k>` / `<Audio j>` text already in the prompt is left
  untouched (manual-control escape hatch; no validation of literal tags).
- **Provenance lines**: for every video whose soundtrack is used, prepend one line to
  the final prompt: `<Audio j> is the synchronized audio track of <Video k>.` —
  always, deterministically (one line per soundtrack, in video order, joined before the
  user prompt with a blank line).
- **Unmentioned references** are still sent to the model (native behavior: every
  connected ref enters the conditioning). Log an info line naming them.
- Referencing a video's soundtrack directly from the prompt is done with its literal
  `<Audio j>` tag (visible on the board tile). A `@name.audio` sugar is a possible
  future addition, out of scope for v1.

### 5.4 Enhancer (`enhancer.py`) — runs only when `enhance_prompt` is true

Runs **inside node execution** (works headless), **before** mention substitution — the
LLM reads and must preserve `@name` tokens, never raw `<Picture i>` tags.

- **Request**: `POST https://openrouter.ai/api/v1/chat/completions`, headers
  `Authorization: Bearer <key>` (widget value, else env `OPENROUTER_API_KEY`, else
  error), `Content-Type: application/json`. Body:
  `model`, `messages` (see below), `response_format: {"type": "json_object"}`,
  `temperature: 0.8`, `reasoning: {"effort": "medium"}` (sent unconditionally;
  OpenRouter drops it upstream for non-reasoning models). Connect timeout 10 s,
  read timeout 120 s. The key must never appear in logs, error messages, or any
  output socket. Caveat outside the pack's control: ComfyUI includes widget
  values in execution-error payloads (`/history`), so shared/serverless hosts
  should use the env var, not the widget (documented in the tooltip + README).
- **Messages**: one `system` (built-in prompt below, or `system_prompt_override`
  verbatim) + one `user`. User content parts:
  1. text: the raw prompt, then a reference manifest — one line per ref:
     `- @name (image | video Xs | audio Xs, file "basename.ext")`, and for used
     soundtracks: `- @name's soundtrack (audio)`.
  2. if `enhancer_vision`: for each image ref, a text part `@name (image):` followed by
     an `image_url` part (data URL, JPEG q85, long edge ≤ 1024); for each video ref,
     the same with one frame sampled from the middle of the clip. Audio is never sent.
- **Built-in system prompts**: preset per `system_prompt_preset`, defined in
  `h3_tools/system_prompts.py` (synthesized from the official MiniMax guide,
  `guide.md`). Contract every preset implements: rewrite the draft into the H3
  six-section full-reference format; preserve every `@name` token spelled
  identically; never invent `@` tokens; never describe media content not shown;
  fit all timestamps inside the provided "Target video duration: X.Xs" line;
  output only `{"prompt_final": "..."}`.
- **Response parsing**: strip code fences if present → first balanced `{...}` →
  `json.loads` → `prompt_final` must be a non-empty string.
- **Retry/error policy**: up to 3 total attempts. Network / 429 / 5xx failures
  retry with a short backoff (1 s then 2 s, overridden by `Retry-After` capped at
  30 s; no sleep after the final attempt); a parse failure retries with an appended
  system line "Return ONLY the JSON object, nothing else." (the nudge stays for any
  later attempt); other 4xx fail immediately. After 3 failures → execution error
  carrying the provider's message (first ~200 chars), never a silent fallback to
  the raw prompt (silent quality degradation is worse than a visible failure).
- **Post-enhancement sanitation** (asymmetric on purpose): `@name` tokens in the
  **user's** prompt that don't resolve = validation error (§5.1). Unknown `@` tokens in
  the **LLM's** output = stripped, with a logged warning (a hallucinating LLM must not
  kill a serverless job). If the LLM dropped a `@name` the user had written, log a
  warning; the ref still reaches the model unmentioned.
- **Disk cache**: key = sha256 of canonical JSON
  `{prompt, refs: [{name, type, file, use_soundtrack, mtime_ns, size}], model,
  system, vision, seed}` (`use_soundtrack` changes the manifest the LLM sees, so
  it must change the key);
  value = `{"prompt_final": ..., "model": ..., "created": ...}` stored as
  `<user_dir>/h3_tools/enhancer_cache/<hash>.json`
  (`folder_paths.get_user_directory()`); prune oldest beyond 500 entries. Cache means
  changing `width`/`length`/etc. re-executes the node without re-calling the LLM.

### 5.5 Encode delegation (`node.py`)

Build the native Autogrow dicts and call the native classmethod directly:

```python
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo

out = MiniMaxH3ReferenceToVideo.execute(
    clip=clip, vae=vae, audio_vae=audio_vae,
    prompt=final_prompt, width=width, height=height, length=length,
    ref_image_size=ref_image_size,
    ref_images={"ref_image_0": img0, ...},          # array order
    ref_videos={"ref_video_0": frames0, ...},
    ref_video_audios={"ref_video_audio_0": audio0},  # only for use_soundtrack videos,
                                                     # index MUST match its video's key
    ref_audios={"ref_audio_0": audio, ...},
)
cond, latent = out.args
return io.NodeOutput(cond, latent, final_prompt)
```

Dict keys follow the native `prefix + index` convention; the trailing index pairs
soundtracks to videos. Empty groups pass `None`. If a future upstream refactor changes
this signature, the fallback is to reimplement the ~60-line body of the native
`execute` using the same public helpers (`adapt_canvas`, `_resize`,
`_encode_ref_audio`, `_empty_av_latent`, `clip.tokenize(prompt,
minimax_ref_items=...)`) — the spec's ordinal rules in §5.3 already match the
tokenizer contract directly.

### 5.6 `fingerprint_inputs` (cache correctness)

Return a hash of: every widget value **plus** `(mtime_ns, size)` of every referenced
file. This makes re-uploading a changed file under the same name invalidate ComfyUI's
execution cache (a known bug class in a competitor). One exception: the API key
contributes only `bool(key)` to the hash — its value must never be hashed or logged.

## 6. Frontend (`web/board.js` + `board.css`)

One `app.registerExtension` scoped strictly to `H3RefToVideoPro` via
`beforeRegisterNodeDef` / `nodeCreated`. **Forbidden techniques** (each one is a
documented bug source in the analyzed competitors): patching `app.graphToPrompt`,
patching canvas/`processKey`/link-dispatch prototypes, virtual links, state in
`node.properties`, hidden transport inputs, fixed node sizes, network calls in
`INPUT_TYPES`, `setInterval` polling loops.

The only widget-internals touch allowed: hiding the `references` JSON widget
(`widget.type = "hidden"`, `computeSize = () => [0, -4]`). Degradation if a frontend
update breaks hiding: the JSON shows as an ugly-but-functional text widget. Nothing
else may depend on frontend internals.

### 6.1 Reference board (DOM widget)

`node.addDOMWidget("h3_board", ...)` rendering, in normal document flow (no absolute
overlay positioning beyond what addDOMWidget provides):

- Toolbar: `+ Imagem`, `+ Vídeo`, `+ Áudio` buttons → hidden `<input type=file>` with
  the proper `accept` filter → `POST /api/upload/image` (multipart: `image=<file>`,
  `type=input`, `subfolder=h3_refs`, `overwrite=false` so the server dedups names) →
  on response, append `{name: derivedSlug, type, file: "h3_refs/<returned name>"}` to
  the JSON widget value. Slug derivation duplicates §4's rule (trivial, and drift is
  harmless: the backend re-derives only when `name` is absent).
- Tiles (one per ref, grid): thumbnail (§6.3), editable name (click-to-edit text,
  validated against `^[a-z0-9_]{1,64}$` + uniqueness), type badge, the **current
  official tag** (`<Picture 2>`, `<Video 1>`, `♪ <Audio 1>`) computed client-side with
  the same ordinal rule as §5.3 (display-only duplication: the backend's assignment is
  authoritative, so drift can only mislabel a badge, never change behavior), delete
  button, and for videos a ♪ soundtrack toggle
  (freely togglable; the backend errors at run time if the file has no audio track —
  no probe route in v1).
- Counters with limits: `imagens 3/9 · vídeos 1/3 · áudios 0/3`; the add buttons
  disable at the cap.
- Every mutation rewrites the `references` widget value (single source of truth) and
  refreshes tags/counters. Reordering is NOT in v1 (names make order cosmetic).

### 6.2 Prompt editor: textarea + highlight overlay + mention popup

- Replace the default multiline widget's visual with: a wrapper containing a `<div>`
  highlight layer behind a transparent-text `<textarea>` (same font, padding, line
  height, wrap; scroll positions synced on `input`/`scroll`/resize). The overlay
  re-renders the text with `@name` spans colored (known-name = accent color,
  unknown-name = red). The textarea remains the real widget input — undo, IME,
  clipboard, selection all native.
- **Popup**: on input, if the text before the caret matches `/(?<![A-Za-z0-9_])@([a-z0-9_]*)$/i`,
  show a fixed-position menu near the caret listing references filtered by the partial
  name, ordered image → video → audio. Each row: thumbnail (48px), `@name`, type
  badge, current tag. Keyboard: ↑/↓ move, Enter/Tab insert `@name ` (plain text) at
  the caret, Esc closes; click inserts; clicking elsewhere or moving the caret out of
  the token closes. The menu element lives on `document.body` and is removed on node
  collapse/removal.
- Hovering a highlighted `@name` shows a small thumbnail tooltip (P2, optional).

### 6.3 Thumbnails (all client-side, zero custom routes)

- Image: `/api/view?filename=<name>&subfolder=h3_refs&type=input` directly.
- Video: off-DOM `<video preload="metadata">` with the same `/api/view` URL → seek to
  `min(1s, duration/2)` → draw to a 96px canvas → `toDataURL`; cached in a module map
  keyed by file path. On failure fall back to a type icon.
- Audio: type icon; clicking the tile toggles play/pause on a shared `<audio>` element.

### 6.4 Serialization guarantees

- Everything the backend needs lives in the two String widgets (`prompt`,
  `references`) — plain `widgets_values`. Workflow JSON reloads identically with or
  without this extension installed; "Save (API format)" needs no patching; third-party
  graph→prompt converters work.
- No migrations by widget position: the JSON schema is the wire format and is
  additive-only.

## 7. Headless / API contract

```jsonc
// node entry in an API prompt
"37": {
  "class_type": "H3RefToVideoPro",
  "inputs": {
    "clip": ["10", 0], "vae": ["11", 0], "audio_vae": ["12", 0],
    "prompt": "@garota dança na chuva em frente ao @predio, com a voz de @voz_ana",
    "references": "[{\"type\":\"image\",\"file\":\"h3_refs/garota.png\"},{\"type\":\"image\",\"file\":\"h3_refs/predio.png\"},{\"type\":\"audio\",\"file\":\"h3_refs/voz_ana.wav\"}]",
    "width": 1344, "height": 768, "length": 124, "ref_image_size": "match",
    "enhance_prompt": true, "enhancer_model": "google/gemini-3-flash-preview",
    "openrouter_api_key": "", "enhancer_vision": false,
    "system_prompt_override": "", "enhancer_seed": 0
  }
}
```

- Serverless flow: upload media via standard `/upload/image` (or pre-bake files into
  `input/h3_refs/`), then queue the prompt. `name` may be omitted per ref (derived
  from filename) as long as derived names are what the prompt mentions.
- All failure modes surface as ComfyUI validation/execution errors with actionable
  messages (§5.1, §5.4) — no partial successes, no silent substitutions beyond the
  documented LLM-output sanitation.
- The API key should come from the env var in serverless deployments; the widget field
  exists for local convenience and is excluded (as a value) from fingerprints.

## 8. Error message catalog (exact behaviors)

| condition | stage | behavior |
|---|---|---|
| `references` not valid JSON / bad schema | validate | error naming the field and first offending element |
| duplicate explicit names | validate | error listing the duplicate |
| over caps (>9/>3/>3) | validate | error with the counts and the model's limits |
| file missing | validate | error with the resolved path |
| unknown `@name` typed by the user | validate | error listing valid names |
| video < 5 frames after 24fps resample | execute | error naming the ref |
| `use_soundtrack` on a file without audio track | execute | error naming the ref |
| enhancer on, no key anywhere | execute | error pointing at widget + env var |
| LLM: 3 failed attempts | execute | error with provider message excerpt |
| LLM output invents `@x` | execute | stripped + warning log |
| LLM output drops a user `@name` | execute | warning log, ref stays unmentioned |

## 9. Out of scope for v1 (explicitly)

- Crop/trim editors, drag-reorder, `@name.audio` soundtrack mentions, dialogue-block
  UI (typing `<d>...</d>` by hand works — it's plain prompt text), local/self-hosted
  LLM providers (OpenRouter only; the env var + free-text model keep it flexible),
  negative conditioning outputs, first/last-frame keyframe modes (the native
  `MiniMaxH3ImageToVideo` and `MiniMaxH3AddGuide` already cover those), i18n,
  prompt-guide libraries, any custom server route.

## 10. Testing plan

`tests/` runs under plain pytest with no ComfyUI import (that's why `refs.py` is
stdlib-pure and `media.py`/`node.py` import ComfyUI lazily):

1. **refs**: schema validation (good/bad/caps/duplicates), name derivation (unicode,
   digits-first, collisions), mention regex (word boundaries, case, emails like
   `a@b` not matching after word chars), ordinal assignment incl. soundtrack-before-
   video and mixed sets, substitution + provenance lines, unknown-mention error text.
2. **enhancer**: response parsing (clean JSON, fenced, prose-wrapped, garbage),
   sanitation (invented/dropped mentions), cache key stability, retry sequencing
   (HTTP mocked with `requests` monkeypatching).
3. **media (pure parts)**: the 24fps resample index math as a pure function
   (`src_fps`, `duration` → index list) across common rates (23.976/25/30/60).
4. **Manual/integration checklist** (with a running ComfyUI + H3 models): board
   upload of each type, popup navigation, headless API replay of an exported
   workflow, enhancer on/off, soundtrack on/off, cache-hit behavior, and a diff-check
   that our node + native node produce identical conditioning for an equivalent
   hand-wired graph.

## 11. Competitor pitfalls this design exists to avoid (for the implementer)

Analyzed: `Hearmeman24/ComfyUI-MiniMaxRefPack` (v0.3.5) and
`nkxx188/ComfyUI-MiniMaxH3-Easy`. Full technical reports: `docs/research/`. The
distilled don'ts: virtual links in `node.properties` + `graphToPrompt` monkey-patch (workflow
JSON loses all media without the exact frontend); placeholder tokens
(`__MINIMAX_H3_REF_n__`) between UI and backend; positional identity (delete one ref
and every tag silently retargets); positional `widgets_values` migrations; model combo
populated/validated from a live HTTP API inside `INPUT_TYPES`; canvas-drawn tile UI
with hand-rolled hit regions and a fixed 1340px node; filename-keyed module caches
shared across node instances; plaintext API keys served by unauthenticated GET routes;
`IS_CHANGED = NaN` re-running the LLM every execution; silent `""` substitution of
stale mentions. Every one of these has a corresponding positive decision above.

## Appendix A — original default enhancer system prompt (superseded)

> Superseded in v1.1: the live prompts are the guide-derived presets in
> `h3_tools/system_prompts.py` (source of truth). Kept for history only.

```
You are an expert prompt writer for MiniMax H3, a reference-to-video model that
generates video with synchronized audio. You receive a draft prompt and a manifest of
media references. Each reference is addressed by a token like @name.

Rewrite the draft into one rich, production-quality video prompt:
- Structure it temporally: what is on screen and audible from start to end.
- Cover subject and action, camera (framing, movement), lighting, atmosphere, and
  style, keeping every concrete detail the draft already states.
- When audio references exist, describe the soundscape and how each audio reference
  is used (voice, music, ambience). When none exist, you may still describe diegetic
  sound briefly.
- Stay faithful to the draft's intent. Enrich, never replace it.

Hard rules:
1. Every @name token present in the draft MUST appear in your output, spelled exactly
   the same. Refer to the referenced media ONLY through these tokens.
2. NEVER write an @ token that is not in the manifest.
3. Do not invent visual or audio content for references you were not shown; describe
   only their role in the video.
4. Write in English, unless the draft is clearly and deliberately in another language.
5. Target 80-250 words in prompt_final.

Output: respond with ONLY this JSON object, no markdown fences, no commentary:
{"prompt_final": "<the rewritten prompt>"}
```
