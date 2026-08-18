# MiniMax H3 Reference to Video (Pro) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One ComfyUI custom node (`H3RefToVideoPro`) that replaces "18 loader nodes + hand-written `<Picture 1>` tags" with a media board, `@name` mentions, and an optional OpenRouter prompt enhancer — delegating all encoding to the native `MiniMaxH3ReferenceToVideo`.

**Architecture:** Pure text/organization logic in stdlib-only `refs.py`; ComfyUI-coupled decode in `media.py` (calls native loader node classes); OpenRouter client in `enhancer.py`; the v3 `io.ComfyNode` in `node.py` wires them and calls the native node's `execute` classmethod. One vanilla-JS frontend extension (`web/board.js`) renders the board + mention editor on top of two plain String widgets (`prompt`, `references`).

**Tech Stack:** Python 3 (stdlib + `requests`; torch/PIL only inside ComfyUI-coupled paths), ComfyUI v3 node API (`comfy_api.latest.io`), vanilla JS (no build step), pytest.

**Authoritative spec:** `SPEC.md` at repo root (approved). Where this plan and the spec disagree, the spec wins.

## Global Constraints

- Target ComfyUI: current master (must have `comfy_extras.nodes_minimax_h3`); pack fails import with message "h3_tools requires a ComfyUI version with MiniMax H3 support" otherwise.
- `tests/` MUST run under plain pytest with no ComfyUI importable: `refs.py` is stdlib-pure; `media.py`/`enhancer.py` import ComfyUI/requests lazily inside functions.
- The OpenRouter API key must never be logged, hashed, or emitted on any output; fingerprints use only `bool(key)`.
- No custom server routes, no `graphToPrompt`/prototype patches, no state in `node.properties`, no hidden transport inputs, no network in schema definition.
- Reference caps: 9 images, 3 videos, 3 standalone audios. Name regex `^[a-z0-9_]{1,64}$`. Mention regex `(?<![A-Za-z0-9_])@([A-Za-z0-9_]{1,64})`, matched case-insensitively against stored-lowercase names.
- Node id `H3RefToVideoPro`, display "MiniMax H3 Reference to Video (Pro)", category `conditioning/minimax`.
- Commit after each task (checkpoint commits, pt-BR messages, `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`). Never push.

## Verified infrastructure facts (do not re-derive)

- Custom node loading: `nodes.py` accepts either `NODE_CLASS_MAPPINGS` or `comfy_entrypoint() -> ComfyExtension`; `WEB_DIRECTORY = "./web"` is honored in both cases.
- `validate_inputs(cls, prompt, references)` with explicit named args receives ONLY those inputs (execution.py filters by argspec). Linked inputs would arrive as non-str; guard for that.
- `fingerprint_inputs(cls, **kwargs)` receives ALL inputs with links resolved to `None` (constants only) — so `clip`/`vae`/`audio_vae` arrive as `None` and must be ignored.
- `nodes.LoadImage().load_image(annotated_file)` → tuple `(IMAGE [B,H,W,3], MASK)` (legacy class).
- `comfy_extras.nodes_video.LoadVideo.execute(file=...)` → `io.NodeOutput`; `.args[0]` is a `VideoFromFile`.
- `comfy_extras.nodes_video.GetVideoComponents.execute(video=...)` → `.args` = `(images [T,H,W,C], audio|None, fps: float, bit_depth)`.
- `comfy_extras.nodes_audio.LoadAudio.execute(audio=...)` → `.args[0]` = `{"waveform": [1,C,L], "sample_rate": int}`.
- `MiniMaxH3ReferenceToVideo.execute(clip, vae, audio_vae, prompt, width, height, length, ref_image_size, ref_images=None, ref_videos=None, ref_video_audios=None, ref_audios=None)` → `.args = (conditioning, latent)`. Soundtrack pairing: `ref_video_audios["ref_video_audio_N"]` belongs to `ref_videos["ref_video_N"]`.

---

### Task 1: Scaffolding + `refs.py` (pure core) + tests

**Files:**
- Create: `h3_tools/__init__.py` (empty inner-package marker)
- Create: `h3_tools/refs.py`
- Create: `tests/conftest.py`, `tests/test_refs.py`
- Create: `.gitignore` (`__pycache__/`, `.pytest_cache/`)

**Interfaces (Produces):**
```python
# h3_tools/refs.py — stdlib only (json, re, dataclasses)
MAX_IMAGES = 9; MAX_VIDEOS = 3; MAX_AUDIOS = 3
MENTION_RE  # compiled, (?<![A-Za-z0-9_])@([A-Za-z0-9_]{1,64}), used with re.IGNORECASE at call sites? NO —
            # pattern chars cover both cases already; lookup lowercases the captured name.
NAME_RE     # ^[a-z0-9_]{1,64}$

class RefError(ValueError): ...  # message is the user-facing error

@dataclass
class Ref:
    name: str; type: str; file: str; use_soundtrack: bool = False
    tag: str = ""            # "<Picture 1>" / "<Video 2>" / "<Audio 1>"
    soundtrack_tag: str = "" # "<Audio j>" when use_soundtrack

def derive_name(filename: str) -> str
def parse_references(raw: str) -> list[Ref]        # §4 rules; raises RefError; assigns ordinals before returning
def assign_ordinals(refs: list[Ref]) -> list[Ref]  # fills tag/soundtrack_tag per §5.3 (mutates + returns)
def find_mentions(text: str) -> list[str]          # captured names, as typed
def unknown_mentions(text: str, refs: list[Ref]) -> list[str]
def substitute_mentions(text: str, refs: list[Ref]) -> str   # unknown mentions left untouched
def provenance_lines(refs: list[Ref]) -> list[str]
def build_final_prompt(text: str, refs: list[Ref]) -> str    # provenance + "\n\n" + substituted (or just substituted)
def strip_unknown_mentions(text: str, refs: list[Ref]) -> tuple[str, list[str]]  # for LLM output sanitation
```

Key algorithms (implement exactly):
- `derive_name`: basename → strip extension → lowercase → every `[^a-z0-9]+` run → `_` → strip `_` → if empty or starts with digit, prefix `m_` → truncate to 64.
- `parse_references`: `json.loads`; must be a bare list of dicts. Two passes: (1) register explicit names (must match `NAME_RE`; duplicate explicit → `RefError` naming it); (2) derive missing names, suffix `_2`, `_3`… while colliding with anything taken. Validate `type` ∈ {image, video, audio}; `file` non-empty str; `use_soundtrack` only on videos (else `RefError`); per-type caps. Unknown keys ignored. Errors name the field and the offending element index.
- `assign_ordinals`: images in array order → `<Picture 1..>`; then videos in array order — a `use_soundtrack` video consumes the next audio ordinal for `soundtrack_tag` BEFORE taking its `<Video k>`; then standalone audios continue the audio counter.
- `substitute_mentions`: `MENTION_RE.sub` with a lookup dict keyed by lowercase name → tag; no match → return the original text of the match.
- `provenance_lines`: for each soundtrack video in video order: `f"{soundtrack_tag} is the synchronized audio track of {tag}."`

**Steps:**
- [ ] Write `tests/conftest.py` inserting the repo root into `sys.path`, and `tests/test_refs.py` covering: derive_name (unicode `garota-α.png`, digit-first `1girl.png` → `m_1girl`, empty stem), parse (good full set; not-a-list; bad type; missing file key; dup explicit names; caps 10 images; `use_soundtrack` on image → error; unknown keys ignored; uppercase explicit name → error), collisions (`a.png` + `a.jpg` → `a`, `a_2`), ordinals (mixed set incl. 2 soundtrack videos + 1 standalone audio → standalone audio is `<Audio 3>`), mentions (word boundary: `mail@a` no match; `(@a)` match; case-insensitive `@Garota`), substitution + untouched literal `<Picture 1>` text + untouched unknown mention, provenance line text, build_final_prompt joining, strip_unknown_mentions.
- [ ] Run `python -m pytest tests/ -q` → all FAIL (module missing).
- [ ] Implement `h3_tools/refs.py`.
- [ ] Run `python -m pytest tests/ -q` → all PASS.
- [ ] Commit `feat: refs.py — validação, nomes, ordinais e substituição @`.

### Task 2: `enhancer.py` + tests

**Files:**
- Create: `h3_tools/enhancer.py`
- Create: `tests/test_enhancer.py`

**Interfaces:**
- Consumes: `refs.Ref`, `refs.strip_unknown_mentions`, `refs.find_mentions`.
- Produces:
```python
DEFAULT_SYSTEM_PROMPT: str        # SPEC.md Appendix A, verbatim
class EnhancerError(RuntimeError)  # user-facing message, provider excerpt ≤ 200 chars

def build_manifest(refs, durations: dict[str, float]) -> str
    # one line per ref: - @name (image, file "x.png") / (video 3.2s, file "v.mp4") / (audio 5.0s, file "a.wav")
    # + for use_soundtrack videos: - @name's soundtrack (audio)
def parse_response(text: str) -> str          # fences → first balanced {...} → json.loads → non-empty prompt_final
def cache_key(prompt, ref_stats, model, system, vision, seed) -> str
    # sha256 hex of json.dumps({...}, sort_keys=True, ensure_ascii=False)
    # ref_stats = [{name, type, file, mtime_ns, size}] in array order
def cache_get(cache_dir, key) -> str | None
def cache_put(cache_dir, key, prompt_final, model) -> None    # prune oldest by mtime beyond 500 entries
def sanitize_output(text, refs, original_prompt) -> tuple[str, list[str]]
    # strip @tokens not in refs (warning list); warn (not fix) user mentions the LLM dropped
def enhance(prompt, refs, *, api_key, model, system_prompt, manifest, vision_parts, timeout=120, post=None) -> str
    # post=None → import requests lazily and use requests.post. 3 total attempts:
    # network error / 429 / 5xx → retry as-is; parse failure → retry once with system line
    # "Return ONLY the JSON object, nothing else."; other 4xx → immediate EnhancerError.
    # body: model, messages, response_format={"type":"json_object"}, temperature=0.8.
```
- `vision_parts`: pre-built list of OpenAI-style content parts (`{"type":"text",...}` / `{"type":"image_url","image_url":{"url":"data:..."}}`) — built in `media.py`/`node.py`, so `enhancer.py` stays torch/PIL-free.

**Steps:**
- [ ] Write `tests/test_enhancer.py`: parse_response (clean JSON, ```json fenced, prose-wrapped `Sure! {...}`, nested braces inside strings, garbage → EnhancerError, empty prompt_final → EnhancerError); sanitize (invented `@ghost` stripped + warned; dropped `@garota` warned, text unchanged); cache_key stable across dict order / changes with seed; cache_put/get roundtrip + prune (tmp_path, 502 entries → 500); manifest formatting; enhance() with fake `post` callables: success on attempt 1; 500,500,200 succeeds; 500×3 → EnhancerError with excerpt; 401 → immediate error; parse-fail then success (assert the retry body contains the nudge line); assert api_key appears in no exception message.
- [ ] Run tests → FAIL.
- [ ] Implement `h3_tools/enhancer.py` (copy Appendix A verbatim into `DEFAULT_SYSTEM_PROMPT`).
- [ ] Run tests → PASS.
- [ ] Commit `feat: enhancer.py — cliente OpenRouter com cache e sanitização`.

### Task 3: `media.py` + pure resample tests

**Files:**
- Create: `h3_tools/media.py`
- Create: `tests/test_media.py`

**Interfaces:**
- Produces:
```python
def resample_indices(n_src: int, src_fps: float, target_fps: float = 24.0) -> list[int]
    # PURE. count = floor((n_src / src_fps) * target_fps);
    # [min(round(i * src_fps / target_fps), n_src - 1) for i in range(count)]
class MediaError(ValueError)
def load_image_ref(file: str) -> "torch.Tensor"     # nodes.LoadImage().load_image(file)[0]
def load_video_ref(file: str, name: str) -> dict    # {"frames": [T,H,W,C] @24fps, "audio": dict|None, "duration": float}
    # LoadVideo.execute(file=file).args[0] → GetVideoComponents.execute(video=v).args
    # resample via resample_indices; < 5 frames → MediaError("reference video @name is shorter than ~0.21s")
def load_audio_ref(file: str) -> dict               # {"audio": {...}, "duration": float}
def soundtrack_of(video_payload, name: str) -> dict # video_payload["audio"] or MediaError("@name has use_soundtrack enabled but the file has no audio track")
def image_data_url(tensor, max_edge=1024, quality=85) -> str   # JPEG data URL (PIL imported lazily)
def middle_frame(frames) -> "torch.Tensor"          # frames[len//2] as [1,H,W,C]
```
- All ComfyUI/torch/PIL imports live INSIDE functions; module import is stdlib-only (`math`).
- An audio dict with a zero-length waveform (`waveform.shape[-1] == 0`) counts as "no audio track".

**Steps:**
- [ ] Write `tests/test_media.py` for `resample_indices` only: 23.976→24 keeps count ≈ duration*24 and last index ≤ n_src-1; 25→24 drops frames monotonically; 30→24; 60→24 picks every 2.5th; exact 24→identity `list(range(n))`; tiny clip (3 frames @24) → len 3 (caller rejects <5); indices monotone non-decreasing for all cases.
- [ ] Run tests → FAIL; implement `h3_tools/media.py`; tests → PASS.
- [ ] `python -c "import h3_tools.media"` in a venv WITHOUT ComfyUI → must succeed (lazy-import guard).
- [ ] Commit `feat: media.py — decode via loaders nativos + resample 24fps`.

### Task 4: `node.py` + pack `__init__.py` + `pyproject.toml`

**Files:**
- Create: `h3_tools/node.py`
- Rewrite: `__init__.py` (pack root)
- Create: `pyproject.toml`

**Interfaces:**
- Consumes: everything from Tasks 1–3, plus the verified native signatures in "Verified infrastructure facts".
- Produces: class `H3RefToVideoPro(io.ComfyNode)`; pack exports `comfy_entrypoint` + `WEB_DIRECTORY = "./web"`.

Schema (exact widgets, in this order after clip/vae/audio_vae): `prompt` (String, multiline, `dynamic_prompts=False`, default `""`), `references` (String, default `"[]"`), `width` 1344 / `height` 768 (min 32, max `nodes.MAX_RESOLUTION`, step 32), `length` (default 124, min 5, max 3600, step 17), `ref_image_size` Combo `["match","max"]` default `"match"`, `enhance_prompt` Bool false, `enhancer_model` String `"google/gemini-3-flash-preview"`, `openrouter_api_key` String `""`, `enhancer_vision` Bool false, `system_prompt_override` String multiline `""`, `enhancer_seed` Int 0 (min 0, max 2**31-1). Outputs: Conditioning `positive`, Latent, String `final_prompt`.

Execution flow in `execute` (order matters):
1. `refs.parse_references(references)` (ordinals already assigned).
2. Decode every ref via `media.py` in array order; collect payloads + durations; resolve soundtracks (`soundtrack_of`) for `use_soundtrack` videos.
3. Re-check `unknown_mentions(prompt, refs)` → raise ValueError (defense; validate_inputs already covers the UI path).
4. If `enhance_prompt`: key = widget or `os.environ.get("OPENROUTER_API_KEY")` or error naming both sources; system = override or default; vision_parts built only if `enhancer_vision` (images + `middle_frame` of each video via `image_data_url`); cache under `folder_paths.get_user_directory()/h3_tools/enhancer_cache`; on cache miss call `enhancer.enhance`, then `sanitize_output` (log warnings via `logging`), `cache_put`. Working prompt = result.
5. `final_prompt = refs.build_final_prompt(working_prompt, refs)`; log info line naming unmentioned refs.
6. Build dicts in array order: `{"ref_image_%d" % i: tensor}`, `{"ref_video_%d" % i: frames}`, `{"ref_video_audio_%d" % i: audio}` (same `i` as its video), `{"ref_audio_%d" % i: audio}`; empty group → `None`.
7. `out = MiniMaxH3ReferenceToVideo.execute(...)`; `cond, latent = out.args`; `return io.NodeOutput(cond, latent, final_prompt)`.

`validate_inputs(cls, prompt, references)`: if either is not a `str` → `True` (linked; execute re-checks). Else parse refs (RefError → its message), check each file with `folder_paths.exists_annotated_filepath` (missing → message with resolved path), unknown mentions → message listing valid names. Returns `True` or str, never raises.

`fingerprint_inputs(cls, **kwargs)`: drop non-str/int/bool/float values (links → None); replace `openrouter_api_key` with `bool(value)`; try to parse `references` and append `(file, mtime_ns, size)` per existing file via `os.stat(folder_paths.get_annotated_filepath(f))`; return sha256 hex of the canonical JSON. Any internal error → return `float("NaN")` (always re-run — safe fallback).

Pack `__init__.py`:
```python
try:
    import comfy_extras.nodes_minimax_h3  # noqa: F401
except ImportError as e:
    raise ImportError(
        "h3_tools requires a ComfyUI version with MiniMax H3 support "
        "(comfy_extras.nodes_minimax_h3). Update ComfyUI to the latest master."
    ) from e

from comfy_api.latest import ComfyExtension
from .h3_tools.node import H3RefToVideoPro

WEB_DIRECTORY = "./web"

class H3ToolsExtension(ComfyExtension):
    async def get_node_list(self):
        return [H3RefToVideoPro]

async def comfy_entrypoint() -> H3ToolsExtension:
    return H3ToolsExtension()
```

`pyproject.toml`: `[project]` name `h3_tools`, version `1.0.0`, description, `dependencies = ["requests"]`; `[tool.comfy]` DisplayName "h3_tools".

**Steps:**
- [ ] Implement `h3_tools/node.py`, rewrite `__init__.py`, create `pyproject.toml`.
- [ ] `python -m py_compile h3_tools/*.py __init__.py` → OK; `python -m pytest tests/ -q` still green.
- [ ] Commit `feat: node H3RefToVideoPro + entrypoint do pack`.

### Task 5: Frontend `web/board.js` + `web/board.css`

**Files:**
- Create: `web/board.js`, `web/board.css`

**Interfaces:**
- Consumes: the `prompt` and `references` String widgets by name; standard `api.fetchApi("/upload/image", ...)` and `/view` endpoints; `references` JSON schema §4.
- Produces: no backend-visible artifacts — widgets stay the single source of truth.

Single `app.registerExtension({name: "h3_tools.board"})` with `beforeRegisterNodeDef` gated on `nodeData.name === "H3RefToVideoPro"`, chaining `onNodeCreated`. Structure inside the module (plain functions, no classes needed):

1. **Widget takeover:** find `prompt` + `references` widgets by name; hide both (`computeSize = () => [0, -4]`, `widget.type = "hidden"`, hide `.element` if present) inside try/catch — on failure, log and keep stock widgets visible (degraded but functional). Build ONE container div (board + editor) and attach with `node.addDOMWidget("h3_board", "div", el, {serialize: false})`.
2. **State:** parse `referencesWidget.value` (fallback `[]` on bad JSON, show inline error banner). Every mutation: `referencesWidget.value = JSON.stringify(list)`, re-render tiles, tags, counters, overlay highlights. Client-side tag calc mirrors §5.3 (display-only; comment says backend is authoritative).
3. **Board:** toolbar `+ Imagem` / `+ Vídeo` / `+ Áudio` → hidden `<input type=file>` (accept `image/*`, `video/*`, `audio/*`) → `FormData` {image: file, type: "input", subfolder: "h3_refs"} → `POST /upload/image` → push `{name: deriveName(resp.name), type, file: "h3_refs/" + resp.name}` (dedupe derived name against current list, `_2` suffixes). Buttons disable at caps; counters `imagens 3/9 · vídeos 1/3 · áudios 0/3`. Tiles: thumbnail, click-to-edit name (validate `^[a-z0-9_]{1,64}$` + uniqueness, revert + tooltip on invalid), type badge, current tag badge, ♪ toggle (videos), delete.
4. **Thumbnails:** image → `api.apiURL("/view?filename=...&subfolder=h3_refs&type=input")`; video → off-DOM `<video preload="metadata">` seek `min(1, duration/2)` → 96px canvas → dataURL, cached in a module `Map` keyed by file; audio → icon, click toggles a shared `<audio>`.
5. **Editor:** our own `<textarea>` (value synced both ways with the hidden `prompt` widget) over a `<div>` highlight layer (identical font/padding/wrap; scroll synced). Overlay renders text with `@name` spans — known = accent, unknown = red. On input/caret move, if text before caret matches `/(?<![A-Za-z0-9_])@([a-z0-9_]*)$/i` show the popup on `document.body` near the caret (mirror-div caret measurement), rows ordered image→video→audio filtered by prefix: thumb 48px, `@name`, badge, tag. ↑/↓/Enter/Tab/Esc + click; insert `@name ` plain text. Remove popup on blur, node removal (`onRemoved` chain), scroll.
6. **CSS:** `board.css` loaded via `import "./board.css"`? NO — plain JS: inject a `<link>`/`<style>` once (extensions can't import css); read the file via `import cssUrl from` is unavailable — use `const el = document.createElement("link"); el.rel="stylesheet"; el.href = new URL("board.css", import.meta.url)`.

**Steps:**
- [ ] Implement `web/board.css` then `web/board.js` (target ≲ 600 lines, one file each).
- [ ] Syntax gate: `node --check web/board.js` → OK.
- [ ] Commit `feat: frontend — board de referências e editor @ com popup`.

### Task 6: README + full gate

**Files:**
- Create: `README.md` (usage: install, board, mentions, headless JSON example from §7, enhancer env var, error catalog pointer to SPEC §8)
- Test: full suite

**Steps:**
- [ ] Write `README.md` (English, concise, includes the §7 API example verbatim).
- [ ] Full gate: `python -m pytest tests/ -q` (all green), `python -m py_compile` all py files, `node --check web/board.js`.
- [ ] Commit `docs: README de uso e API headless`.

### Task 7: Opus 5 subagent reviews (user-mandated) + fixes

- [ ] Dispatch TWO Opus 5 review subagents in parallel (`model: "opus"`), each reading SPEC.md + full diff: (a) backend correctness review — refs/enhancer/media/node vs the native contract (ordinals, soundtrack pairing, native call signatures, validate/fingerprint semantics, key-leak scan); (b) frontend + headless review — board.js vs the forbidden-techniques list, serialization guarantees, popup/overlay edge cases, API-contract fidelity.
- [ ] Triage findings: fix confirmed bugs, re-run the test suite, note rejected findings with reasons.
- [ ] Commit `fix: ajustes das reviews Opus 5` (if any) and report results to the user.
