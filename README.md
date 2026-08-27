# h3_tools — MiniMax H3 Reference to Video (Pro)

Two nodes that replace the native MiniMax H3 ref2va workflow of "up to 18
loader nodes + hand-written `<Picture 1>` tags" with:

- a **media board**: upload images / videos / audios straight on the node,
  with previews, rename, delete and a ♪ soundtrack toggle per video;
- an **@mention prompt**: type `@` and pick a reference from a popup with real
  thumbnails (↑/↓ + Enter). `@garota` becomes `<Picture 1>` automatically;
- an optional **LLM prompt enhancer** (OpenRouter) that rewrites the draft
  while preserving every `@name` token;
- first-class **headless/API** operation — the workflow JSON is
  self-contained text, no frontend needed.

The node's job is text and organization only. Media decoding calls the core
loader nodes (`LoadImage`, `LoadVideo` + `GetVideoComponents`, `LoadAudio`)
and all encoding is delegated, untouched, to the native
`MiniMaxH3ReferenceToVideo` — conditioning is byte-identical to a hand-wired
native graph.

## Requirements

- ComfyUI recent enough to ship MiniMax H3 support
  (`comfy_extras/nodes_minimax_h3.py`); the pack refuses to load otherwise.
- `pip install requests` (usually already present).

## Node

`MiniMax H3 Reference to Video (Pro)` (`H3RefToVideoPro`), category
`conditioning/minimax`.

Inputs: `clip`, `vae`, `audio_vae` + widgets `prompt`, `references` (JSON,
managed by the board), `width`, `height`, `length`, `ref_image_size`, and the
enhancer group (`enhance_prompt`, `enhancer_model`, `openrouter_api_key`,
`enhancer_vision`, `vision_format`, `system_prompt_override`, `enhancer_seed`).

Outputs: `positive` (CONDITIONING), LATENT (AV latent ready for sampling),
`final_prompt` (STRING — the exact text handed to the native node).

### `MiniMax H3 Reference to Video Continue (Pro)` (`H3RefToVideoContinuePro`)

The Pro node plus continuation context **for the LLM only** — the pixel-level
continuation stays with the native `MiniMaxH3AddGuide`, which you wire
downstream (`positive`/`latent` → AddGuide with the source frames → sampler).

- `video` (VIDEO socket): the footage being continued, sent WHOLE to the LLM —
  re-encoded small (h264, no audio, ≤768px, ~10fps; only the last 30s of
  longer sources). Needs a video-capable model
  (openrouter.ai/models?input_modalities=video — Gemini works, GPT/Claude
  don't); non-video models are detected and skipped/erroed with guidance.
- `image_last_frame` (IMAGE socket): fallback when no video — only this frame
  is sent. `video` wins when both are connected.
- `duration_mode`: `total` = `length` is the FINAL length, the continuation
  spans source end → total (errors if the source is longer); `new_only` =
  `length` is the new part, the latent grows to source + length. With
  `image_last_frame` the source clock is unknown, so `length` is always the
  new part and timestamps start at 00:00.000.
- The continuation footage is sent whenever the enhancer is on;
  `enhancer_vision` keeps controlling reference images only. System prompts
  are the same 3 presets with a CONTINUATION block (the footage is established
  fact, gets no `@token`, and the shot script covers the full timeline).

`vision_format` (both nodes): how media reaches the LLM — `default` = one
message with interleaved label+media parts; `cascade` = one message per media
item. Part of the enhancer cache key; A/B them freely.

## Headless / API usage

Upload media via the standard `/upload/image` endpoint (multipart with
`type=input`, `subfolder=h3_refs`) or pre-bake files into `input/h3_refs/`,
then queue a prompt like:

```jsonc
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

Each reference is `{"name"?, "type": "image"|"video"|"audio", "file",
"use_soundtrack"?}`. `name` defaults to a slug of the filename
(`Garota Linda.png` → `@garota_linda`). Order within a type defines the
native ordinals. Caps: 9 images, 3 videos, 3 standalone audios.

## Modal API client nodes

Six nodes that run generation on the modal_inferencito API instead of the
local GPU — use them on any ComfyUI, including CPU-only frontends (this pack
loads them even when the ComfyUI has no MiniMax H3 support; only the local
Pro nodes need it).

- **Prepare Reference / Prepare Reference Continue**: media board +
  `@mentions` + the same OpenRouter enhancer as the Pro nodes, running
  LOCALLY (media is sent to the LLM from your machine, not from the server).
  Output feeds the matching Generate node.
- **Generate Reference / Reference Continue / First-Last Frame / First-Last
  Continue**: upload media, submit, poll with live progress + animated
  preview, return the produced VIDEO (saved under `output/h3_api/`) plus the
  exact `final_prompt` the server encoded and a cost line.
- First/Last nodes take a ready prompt STRING — no enhancer, no mentions;
  with neither frame connected they are text-to-video, and the Continue
  variants never need frames (the source video provides the pixels).
- Connection: set `accounts_json` to your modal_inferencito `accounts.json`
  (or export `H3_MODAL_ACCOUNTS_JSON`), or fill `api_url` + `api_key`.
  From Windows, the WSL path works:
  `\\wsl.localhost\Debian\home\adolfocesar\projects\modal_inferencito\accounts.json`.
- Quality tiers: `fast` (turbo lora, 8 steps), `normal` (20), `quality` (32).

## Enhancer

Set `enhance_prompt` and either fill `openrouter_api_key` or export
`OPENROUTER_API_KEY`. Prefer the env var on shared or serverless hosts:
ComfyUI itself includes widget values in execution-error payloads
(`/history`), which is outside this pack's control. Requests route to the fastest provider (`provider.sort: throughput`), cap
`max_tokens` per reasoning level, and default `enhancer_reasoning` to `none`
(thinking off — reasoning tokens are generated serially and dominate
latency); they retry
429/5xx with `Retry-After`-aware backoff, logging every retry; set
`enhancer_model_fallback` to try a second model when the main one fails for
good (timeout, provider error, unparseable output). The LLM receives the draft
+ a reference manifest (+ downscaled images when `enhancer_vision` is on),
must answer `{"prompt_final": "..."}`, and its output is sanitized: the six
H3 sections are restored to the official order when the LLM shuffles them;
invented `@tokens` are stripped with a warning; a dropped `@name` only logs a
warning. Responses stream over SSE and are logged live in the console
(`h3_tools: enhancer ▸ ...`) while they arrive.
Results are cached on disk (`user/h3_tools/enhancer_cache/`) keyed by
prompt/refs/model/system/vision/seed/duration — bump `enhancer_seed` to
re-roll. Failures after 3 attempts are hard errors, never silent fallbacks.

`system_prompt_preset` picks the objective the LLM directs for — `default`,
`multishot` (cuts on exact timecodes) or `single_take` (plano-sequência).
All presets share one core built from the official MiniMax guide
(`guide.md`): the H3 six-section format, shot-script craft rules and a full
worked output example. Connect a STRING into `system_prompt_override` to
replace the preset verbatim. The LLM always receives the target video
duration and each reference's duration, so timestamps land on the real
clock.

## `Show Text Brabo` (`ShowTextBrabo`)

A text preview that **stays in the workflow**. Feed it any socket (`source`) —
a `STRING`, a number, a dict — and it renders the value in a multiline widget
and passes the same text through its `text` output, so it can sit mid-chain.

The built-in *Preview as Text* renders into a widget flagged `serialize: false`,
so its text is gone the moment you reopen the workflow. Here the widget is a
declared `STRING` input, which litegraph saves into `widgets_values` and
restores on load — the last run's output travels inside the workflow JSON, and
is still on screen when the node is served from cache.

The widget is editable (the frontend has no read-only mode for a serialising
string widget). Anything typed there is display only: it is discarded and
overwritten by the next execution, and never reaches the `text` output.

## Design docs

`SPEC.md` is the full design (behavior contract, error catalog §8, JSON
schema §4); `docs/research/` analyzes the competitor packs this design
avoids repeating.
