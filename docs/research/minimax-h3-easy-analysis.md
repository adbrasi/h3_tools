# Technical map: ComfyUI-MiniMaxH3-Easy (by nkxx188)

Analysis date: 2026-08-18, against a fresh clone of
https://github.com/nkxx188/ComfyUI-MiniMaxH3-Easy. `nodes.py` (1914 lines, backend),
`web/minimax_h3_easy_ui.js` (~5.5k lines, single frontend module), `prompt_guides/`.

## 1. Node inventory (nodes.py)

| Class | node_id | Signature | Purpose |
|---|---|---|---|
| `MiniMaxH3EasyLoader` (L1304) | `MiniMaxH3EasyLoader` | 5 required combos: `fl2va_model`, `ref2va_model`, `text_encoder`, `video_vae`, `audio_vae` → `MINIMAX_H3_BUNDLE` | Bundled loader. Fuzzy filename-role matching (`_has_role` L288) accepts community/GGUF variants; `None`/`无` sentinel allowed for one transformer. GGUF via optional ComfyUI-GGUF classes looked up at runtime. Text encoder: `nodes.CLIPLoader().load_clip(name, "minimax", "default")` (L1191). |
| `MiniMaxH3EasyModelAdapter` (L1345) | same | CLIP + video VAE + audio VAE (+ optional MODELs) → `MINIMAX_H3_BUNDLE` | "Model Bridge" for native/GGUF loaders. |
| `MiniMaxH3Easy` (L1637) | same | see §2/§4 → `("MODEL", "MINIMAX_H3_CONTEXT")` | Main node: prompt, media ordering, canvas/duration math, conditioning + empty AV latent. |
| `MiniMaxH3EasyOutput` (L1791) | same | `MINIMAX_H3_CONTEXT` → `CONDITIONING, LATENT, VAE, VAE, FLOAT` | Unpacks the context dataclass. |
| `MiniMaxH3EasyAspectRatio` (L1819) | same | context → label e.g. `"16:9 (Widescreen)"` | Feeds Pass-2 resolution combos. |
| `MiniMaxH3EasySecondPassConditioning` (L1846) | same | context + 24ch video LATENT → CONDITIONING | Re-encodes stored keyframe source images at pass-2 resolution. |
| `MiniMaxH3PromptOptimizer` (L1009) | **not registered** | prompt/mode/api params → STRING | Dead code, absent from `NODE_CLASS_MAPPINGS`. |

Lazy transformer management: `MiniMaxH3Bundle.model_for(kind)` (L1255) loads
FL2VA/REF2VA on demand, falls back to the other role if one is None, drops +
`soft_empty_cache()` when swapping.

## 2. Media reference flow — virtual links, not real graph links

- One visible `media` input of type `"*"` (L1647) + **30 hidden transport inputs**:
  `media_1..media_15` (`("*", {"hidden": True})`) and `media_type_1..15` (STRING,
  hidden), declared purely so execution can resolve links (L1652-1659).
- The frontend never keeps a real LiteGraph link on `media`. All media connections
  live in **`node.properties["minimax_h3_virtual_media_links"]`**: an array of
  `{source_id, source_slot, source_type, media_type, order}`. Any native connection to
  `media`/`media_N` is converted to a virtual link and *disconnected*
  (`convertNativeMediaConnection` js L708, hooked from `onConnectionsChange` and
  `onConfigure`).
- Virtual wires are drawn by wrapping `canvas.drawConnections` (`patchCanvas` L1357):
  bezier + numbered midpoint dot; clicking the dot opens a Delete menu. Dragging from
  `media` to empty canvas spawns a LoadImage/LoadVideo/LoadAudio quick-create menu
  while suppressing ComfyUI's native node-search dialog (CSS class + synthetic Escape
  keydowns, L1016/L1037).
- **Execution wiring happens only in a monkey-patched `app.graphToPrompt`**
  (`patchGraphToPrompt` L1472): deletes `media`/`media_*` from the API prompt, then
  re-emits `inputs["media_{i+1}"] = [source_id, slot]` + `media_type_{i+1}` in stored
  link order. Also canonicalizes localized combo values and injects `prompt`,
  `prompt_optimizer_resources` (JSON), `prompt_optimizer_marker`,
  `prompt_optimizer_prompt_connected`.
- Backend `_collect_media` (L1685) rebuilds `_MediaInput(input_index, media_type,
  value)`; type falls back to `_infer_media_type` (L1388: Tensor→image, `waveform`
  Mapping→audio, `get_components`→video, else video).
- Limits: frontend `mediaLimits` (L526): image-mode = 2; ref-mode = 9/3/3, total 15.
  Backend re-validates (L1744-1754).

## 3. @mention frontend UX

- Prompt UI is a `contentEditable` DIV (`.h3-prompt-editor`) added via
  **`node.addDOMWidget("h3_prompt_mentions", ..., {serialize:false})`** (L4742); the
  native `prompt` widget is hidden (`type="hidden"`, `computeSize→[0,-4]`, L3016) but
  keeps holding the canonical string value.
- **Trigger**: typing `@` (`beforeinput` L4553 + `input`/`keyup`/`focus`) →
  `getMentionRange` (L2567) walks text units backwards from the caret matching
  `/@[^@\n]*$/` (chips/dialogue/BR are hard boundaries).
- **Menu**: `openMentionMenu` (L2886) appends a fixed-position `.h3-mention-menu` to
  `document.body`. Options from `mentionOptions` (L1596): links sorted
  image→video→audio, ordinal counted per type; label per `reference_mention_mode`
  widget (`index` → "Image1"; `filename` → truncated filename). Previews: images via
  `/view?filename=...`; videos via an off-DOM `<video>`+canvas frame grab with
  brightness scoring and multi-timestamp sampling (`getVideoFrameThumbnail` L1828,
  cached); audio = inline SVG icon. Arrow/Enter/Tab/Escape in
  `handleMentionMenuKeydown` (L3949).
- **Selection**: `chooseMention` (L2817) replaces the `@query` range with `​`
  sentinel + `<span class="h3-mention-chip" contentEditable=false>` + sentinel. Chip
  state all in `dataset` (token, tag, label, mediaType, ordinal, sourceId, sourceSlot,
  previewUrl...). A huge caret/deletion helper suite (L4204-4280, sentinel management
  L3973-4180) plus a custom 120-entry undo stack (`pushPromptHistory` L2421) because
  native contentEditable undo is blocked at three layers.
- **Storage**: every edit → `serializeEditorDoc` (L2228) producing `{version:1, text,
  parts:[{type:"text"|"mention"|"dialogue", ...}]}`, stored in
  `node.properties["minimax_h3_prompt_reference_doc"]`. The `prompt` widget value is
  set to `doc.text`, where mentions render as their official tag (`<Picture 2>`) and
  dialogue as `<d>...</d>`. View mode (`structured`/`raw`) in properties; a corner
  button toggles. Pasting text containing `<Picture N>` or `@Image1`-style aliases
  re-chips it (L4413, L4368).
- **What is actually sent to the backend** (`buildRuntimePrompt` L1418, called from
  the graphToPrompt patch): each mention becomes **`__MINIMAX_H3_REF_{n}__`** where n
  = 1-based index into the runtime link array — resolved by ordinal-within-type in
  `index` mode, or by `sourceId+sourceSlot+mediaType` in `filename` mode.
  Unresolvable mentions become `__MINIMAX_H3_UNRESOLVED_REF_{type}__`. So the API
  prompt JSON contains placeholder tokens, *not* `<Picture N>` tags.

## 4. Backend prompt translation

`_reference_conditioning` (L1513) assigns ordinals in the official H3 presentation
order and builds `ref_items` (for `clip.tokenize(prompt, minimax_ref_items=...)`),
`ref_blocks` (attached as `minimax_refs`), and `tag_by_input: {input_index → tag}`:

1. **Images** in link order → `<Picture 1..N>`; per-image resize per `ref_image_size`
   (`match`/1k/1.5k/2k area with uniform-scale grid search L137; `original` =
   center-crop to 32-px grid).
2. **Videos** in link order → `<Video 1..N>`; resampled to `h3.FPS` (L1419), canvas
   via `h3.adapt_canvas`, trimmed to `%17==5` frames. An embedded soundtrack
   **consumes an audio ordinal first** (L1591).
3. **Standalone audios** continue the audio ordinal → `<Audio k>` (L1611-1618).

`_resolve_reference_prompt` (L1437) regex-substitutes
`__MINIMAX_H3_REF_(\d+)__` with `tag_by_input[input_index]` (missing index →
**silently replaced with ""**), and when ambiguity exists prepends provenance lines:
`"<Audio i> is the synchronized audio track of <Video k>."`.

**Native reuse**: never instantiates native node classes. From
`comfy_extras.nodes_minimax_h3 as h3` it reuses `h3.CANVAS_MULTIPLE`, `h3.FPS`,
`h3._empty_av_latent`, `h3._resize`, `h3.adapt_canvas` — then re-implements
conditioning assembly itself via `clip.tokenize(...)` + `encode_from_tokens_scheduled`
+ `conditioning_set_values` with `minimax_keyframes`/`minimax_frame_count`/
`minimax_refs`. Frame count = `5 + 17n` (`_frame_length` L1477); canvas = megapixel
budget × aspect ratio, 32-aligned (`_canvas_dimensions` L1466).

## 5. API / headless behavior

- The hidden `media_N`/`media_type_N` inputs are legitimate `INPUT_TYPES` entries, so
  a hand-written API prompt that supplies `"media_1": ["12", 0], "media_type_1":
  "image"` works with zero frontend. `prompt` is a plain string; **literal
  `<Picture N>` tags pass through untouched** — headless users can write official
  tags directly.
- A workflow exported via "Save (API Format)" from the patched frontend contains baked
  placeholders + aligned `media_N` links → replays fine headless.
- **The saved workflow JSON (non-API) is not headless-safe**: loader nodes have no
  real link to the main node (links are virtual, in `properties`). Any tool doing its
  own graph→prompt conversion without this extension produces a main node with *no
  media at all* and orphaned loaders.
- Optimizer: `_optimize_prompt_on_run` (L927) only fires if `prompt_optimizer.json`
  has `optimize_on_run:true`; the dedupe marker round-trips through
  `ui.auto_optimization_marker` → frontend `onExecuted` → properties → next
  graphToPrompt. **Headless, the marker never persists, so every run re-calls the
  LLM.** `IS_CHANGED` returns NaN whenever optimize_on_run is on (L1699) — full
  re-execution every run.
- Server routes (registered with a polling retry thread, L1104):
  `GET/POST /minimax_h3_easy/prompt_optimizer_settings` and
  `POST /minimax_h3_easy/prompt_optimize`. Config file `prompt_optimizer.json` beside
  the module, plaintext API key, served to any client via unauthenticated GET.

## 6. Custom widget technique (frontend APIs used)

- Single `app.registerExtension({name, setup, beforeRegisterNodeDef})` (L5485).
  Everything is prototype patching in `beforeRegisterNodeDef` (`installNode` L5022):
  wraps `onNodeCreated`, `onExecuted`, `onAdded`, `onConfigure`,
  `onConnectionsChange`, `onSerialize`, `onDrawForeground` (used as an
  editor-install retry trigger), `onRemoved`. Also patches
  LoadImage/LoadVideo/LoadAudio prototypes to watch filename widgets.
- DOM widget via `node.addDOMWidget`; conditional widget rows are collapsed by
  mutating `widget.type="hidden"` + `computeSize=()=>[0,-4]` and manually resizing
  the node, with a `__h3EditorStableSize` restore dance on configure to fight
  cumulative growth.
- Global monkey-patches: `app.graphToPrompt`, `canvas.drawConnections`,
  `canvas.processMouseDown`, `LGraphCanvas.prototype.processKey` (twice),
  `canvas.linkConnector.events.dispatch/dispatchEvent`, plus capture-phase listeners
  on window/document. Theme sync is a 1s `setInterval` poll. Client-side i18n
  rewrites combo option values zh/en and maps back through
  `OPTION_ALIASES`/`canonicalOption`; `repairConfiguredWidgetValues` (L4963) restores
  positional `widgets_values` including a known extra-null off-by-one.

## 7. Dialogue blocks

Typing `#` in structured mode inserts `<span class="h3-dialogue-block">` (L2685; dual
handling in keydown/beforeinput with a dedupe flag). Enter exits the block,
Shift+Enter inserts `<br>` inside. Serialization: part `{type:"dialogue", text}` →
literal **`<d>...</d>`** in both the widget text and the runtime prompt; parsed back
with `/<d>([\s\S]*?)<\/d>/gi`. The backend does nothing with `<d>` — it flows to the
H3 tokenizer as prompt text.

## 8. Weaknesses / design smells

1. **Execution correctness lives in a `graphToPrompt` monkey-patch.** Virtual links
   exist only in `properties`; without this exact frontend, saved workflows lose all
   media (and localized widget values like `"参考生视频"` fail server combo validation).
2. **30 hidden transport inputs** with a nonstandard `{"hidden": True}` option; JS
   must mutate `nodeData` *and* live nodes to strip them. Other frontends show a wall
   of sockets.
3. **Ordinal identity is positional.** In `index` mode a chip means "the Nth image",
   so reordering/removing links silently retargets references. In `filename` mode
   chips bind to `sourceId`, which breaks on copy/paste or graph-merge ID remapping.
   The UI numbers standalone audio starting at 1 while runtime counts video
   soundtracks first — the chip label "Audio1" can serialize to `<Audio 2>`.
4. **Silent reference loss**: stale `__MINIMAX_H3_REF_n__` resolve to `""` (L1450);
   `__MINIMAX_H3_UNRESOLVED_REF_x__` strings reach the model verbatim
   (`UNRESOLVED_REFERENCE_RE` is compiled and never used).
5. **Fragile UI plumbing**: native search suppressed via CSS class + synthetic Escape
   keydowns at 0/16/50/120ms; drop suppression windows of 800–1000ms; editor install
   retried with exponential backoff and even from `onDrawForeground`. The code's own
   comments catalog Nodes 2.0 breakages papered over.
6. **Hidden state sprawl**: four custom `properties` keys, dozens of `__h3*` instance
   fields, a module-level `activePromptNode`, dual-write of widget `.value` and
   `._state.value`.
7. **Optimizer smells**: plaintext API key in `prompt_optimizer.json` served
   unauthenticated; `IS_CHANGED = NaN` when optimize-on-run; headless re-calls the
   LLM every run; a BOOLEAN parameter abused as a popup button.
8. **Misc**: dead `MiniMaxH3PromptOptimizer` class; `_registered_node_class` scans
   `sys.modules` and is `lru_cache`d (late-installed GGUF loader never found);
   `dynamicPrompts: True` on the prompt widget risks wildcard mangling of
   `<...>`/`{...}`; `_infer_media_type` defaults unknown objects to "video".

## 9. Prompt guides (brief)

`prompt_guides/manifest.json` declares a `general` entry (base-en.txt for
T2VA/I2VA/FL2VA/L2VA, ref-en.txt for Ref2VA) and 8 scene guides. Guides are
agent-skill-style markdown with YAML frontmatter describing the official H3 prompt
schema (`integrated_multimodal_description`, `overall_soundscape`,
`non_diegetic_music`; Ref2VA's six-section format). `_prompt_guide_bundle` (L375)
concatenates general + mode reference + scene guide into the optimizer system prompt,
appending a "media evidence rule" anti-hallucination block (L787).

## Takeaway for h3_tools

The two structural decisions everything else fights against are (a) virtual links
stored in properties instead of real graph links, and (b) a placeholder-token prompt
layer between UI and backend. Plain-text `@name` mentions resolved server-side, with
media as filenames in a widget, eliminate the graphToPrompt patch, the hidden-input
hack, and most headless failure modes. The mention popup UX itself (thumbnails,
arrows+Enter, image→video→audio ordering, client-side video frame grabs) is the part
worth imitating.
