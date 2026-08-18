# h3_tools — MiniMax H3 Reference to Video (Pro)

One node that replaces the native MiniMax H3 ref2va workflow of "up to 18
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
`enhancer_vision`, `system_prompt_override`, `enhancer_seed`).

Outputs: `positive` (CONDITIONING), LATENT (AV latent ready for sampling),
`final_prompt` (STRING — the exact text handed to the native node).

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

## Enhancer

Set `enhance_prompt` and either fill `openrouter_api_key` or export
`OPENROUTER_API_KEY`. Prefer the env var on shared or serverless hosts:
ComfyUI itself includes widget values in execution-error payloads
(`/history`), which is outside this pack's control. Requests default to
`reasoning: {"effort": "medium"}` (ignored by non-reasoning models) and retry
429/5xx with `Retry-After`-aware backoff, logging every retry; set
`enhancer_model_fallback` to try a second model when the main one fails for
good (timeout, provider error, unparseable output). The LLM receives the draft
+ a reference manifest (+ downscaled images when `enhancer_vision` is on),
must answer `{"prompt_final": "..."}`, and its output is sanitized: invented
`@tokens` are stripped with a warning; a dropped `@name` only logs a warning.
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

## Design docs

`SPEC.md` is the full design (behavior contract, error catalog §8, JSON
schema §4); `docs/research/` analyzes the competitor packs this design
avoids repeating.
