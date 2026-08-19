# API Modal Nodes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ComfyUI client nodes (running on the user's CPU ComfyUI) that call the
modal_inferencito `/generate` API: 2 prepare nodes (reference family — media
board + @mentions + LOCAL OpenRouter enhancer) and 4 generate nodes (one per
mode: ref, ref continue, first/last frame, first/last continue). All prompt
enhancement happens client-side; the server is always called with
`enhance: false`.

**Architecture:** New modules inside the existing h3_tools pack. Pure logic
(payload building, HTTP client) lives in stdlib-testable modules mirroring the
pack's refs.py/enhancer.py pattern; comfy-touching code lives in one new
`api_nodes.py`. Shared helpers currently in `node.py` (which imports
`comfy_extras.nodes_minimax_h3` at module top and therefore cannot load on the
CPU frontend) move verbatim to a new `enhancer_flow.py`; `node.py` re-imports
them — zero behavior change. `__init__.py` registers the API nodes
unconditionally and the original Pro nodes only when `nodes_minimax_h3`
exists (today it hard-raises, which would block the CPU frontend entirely).

**Tech Stack:** Python stdlib + `requests` (already a dependency), comfy_api
v3 io schema (same as node.py), pytest for the pure modules.

## Global Constraints

- Original nodes stay behavior-identical: `H3RefToVideoPro` /
  `H3RefToVideoContinuePro` keep working exactly as today on GPU ComfyUI.
  Only allowed edits to existing files: `node.py` swaps 4 private helpers for
  imports from `enhancer_flow.py` (moved verbatim); `__init__.py` makes the
  minimax import soft; `web/board.js` adds 2 node ids to `NODE_IDS`;
  `README.md` gains a section.
- FLF nodes have NO enhancer and NO @mentions (user decision 2026-08-19): the
  prompt arrives ready via STRING input.
- New pure modules (`api_payload.py`, `modal_client.py`) must import and pass
  pytest WITHOUT ComfyUI on the path (same rule as refs.py — see
  tests/conftest.py).
- The server API contract is `modal_inferencito/API.md` (WSL:
  `/home/adolfocesar/projects/modal_inferencito/API.md`). No server changes
  are needed or allowed in this plan.
- Work happens in the h3_tools repo: `/mnt/d/Comfyui/comfyOficial/custom_nodes/h3_tools`.
  Commit messages in pt-BR. Do NOT push (user pushes after testing).
- Commands run from WSL; the pack also targets Windows ComfyUI
  (`D:\Comfyui\venv`) — never introduce POSIX-only runtime code paths
  (`/tmp`, `os.symlink`, shell calls).

## File Structure

| File | Responsibility |
|---|---|
| `h3_tools/api_payload.py` (new, pure) | frame-grid math, upload-name mapping, `/generate` body builders for the 4 modes |
| `h3_tools/modal_client.py` (new, pure + lazy requests) | accounts.json loading, account rotation, submit/poll/download against the API |
| `h3_tools/enhancer_flow.py` (new, comfy-safe) | `_decode_refs`, `_reference_vision_parts`, `_run_enhancer`, `_enhancer_inputs` moved verbatim from node.py |
| `h3_tools/api_nodes.py` (new, comfy-only) | the 6 nodes: `H3ApiPrepareRef`, `H3ApiPrepareRefContinue`, `H3ApiGenerateRef`, `H3ApiGenerateRefContinue`, `H3ApiGenerateFlf`, `H3ApiGenerateFlfContinue` |
| `h3_tools/node.py` (edit) | imports the 4 helpers from enhancer_flow |
| `__init__.py` (edit) | soft minimax import; register API nodes always |
| `web/board.js` (edit, line 11) | add prepare node ids to `NODE_IDS` |
| `tests/test_api_payload.py`, `tests/test_modal_client.py` (new) | pure-module coverage |
| `scripts/e2e_api.py` (new, WSL-run) | drives modal_client against the real deployed API without ComfyUI |

---

### Task 1: `api_payload.py` — pure payload builders

**Files:**
- Create: `h3_tools/api_payload.py`
- Test: `tests/test_api_payload.py`

**Interfaces:**
- Consumes: `refs.parse_references(raw) -> list[Ref]` (existing).
- Produces (used by Tasks 3–5):
  - `align_frames(n: int) -> int` — snap up to 17k+5.
  - `target_duration(duration_s: float) -> float` — aligned seconds at 24 fps.
  - `upload_name(path: str, taken: set) -> str` — unique basename.
  - `ref_body(*, prompt, references, duration_s, duration_mode=None) -> dict`
  - `flf_body(*, prompt, duration_s, first_name=None, last_name=None, size_mode="aspect") -> dict`
  - `extend_extra(context_seconds: float, source_name: str) -> dict`
  - `ApiPayloadError(ValueError)`
- A "payload" (prepare→generate socket value) is
  `{"mode": str, "body": dict, "uploads": [{"name": str, "path": str}], "preview_prompt": str}`.

- [x] **Step 1: Write the failing tests**

```python
# tests/test_api_payload.py
import pytest

from h3_tools import api_payload as ap


def test_align_frames_snaps_up_to_17k_plus_5():
    assert ap.align_frames(5) == 5
    assert ap.align_frames(6) == 22
    assert ap.align_frames(96) == 107   # 4 s -> 107, matches the server
    assert ap.align_frames(240) == 243  # 10 s -> 243


def test_target_duration_matches_server_grid():
    assert ap.target_duration(4.0) == pytest.approx(107 / 24.0)
    assert ap.target_duration(10.0) == pytest.approx(243 / 24.0)


def test_upload_name_dedupes_basenames():
    taken = set()
    a = ap.upload_name("h3_refs/girl.png", taken)
    b = ap.upload_name("other/girl.png", taken)
    assert a == "girl.png" and b != a and b.endswith(".png")
    assert taken == {a, b}


def test_ref_body_shape():
    body = ap.ref_body(
        prompt="a @garota dança",
        references=[{"name": "garota", "type": "image", "file": "garota.png"}],
        duration_s=8.0)
    assert body == {
        "mode": "ref", "prompt": "a @garota dança",
        "references": [{"name": "garota", "type": "image", "file": "garota.png"}],
        "duration_s": 8.0, "enhance": False,
    }


def test_ref_body_continue_adds_mode_and_duration_mode():
    body = ap.ref_body(prompt="p", references=[], duration_s=10,
                       duration_mode="new_only")
    assert body["mode"] == "ref_extend"
    assert body["duration_mode"] == "new_only"


def test_flf_body_variants():
    t2v = ap.flf_body(prompt="p", duration_s=6)
    assert t2v == {"mode": "flf", "prompt": "p", "duration_s": 6,
                   "enhance": False, "size_mode": "aspect"}
    i2v = ap.flf_body(prompt="p", duration_s=6, first_name="a.png",
                      last_name="b.png", size_mode="source")
    assert i2v["first_image"] == "a.png" and i2v["last_image"] == "b.png"
    assert i2v["size_mode"] == "source"


def test_flf_body_rejects_source_size_without_first():
    with pytest.raises(ap.ApiPayloadError):
        ap.flf_body(prompt="p", duration_s=6, size_mode="source")


def test_extend_extra():
    assert ap.extend_extra(2.0, "prev.mp4") == {
        "source_video": "prev.mp4", "context_seconds": 2.0}
    with pytest.raises(ap.ApiPayloadError):
        ap.extend_extra(0, "prev.mp4")
```

- [x] **Step 2: Run tests to verify they fail**

Run: `cd /mnt/d/Comfyui/comfyOficial/custom_nodes/h3_tools && python3 -m pytest tests/test_api_payload.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'h3_tools.api_payload'`

- [x] **Step 3: Write the implementation**

```python
# h3_tools/api_payload.py
"""Pure builders for the modal_inferencito /generate bodies (API.md).

Stdlib-only: imports and tests without ComfyUI, like refs.py. The frame grid
mirrors the server exactly (17k+5 at 24 fps) so the duration shown to the
local enhancer equals the duration the server will generate.
"""

import os

FPS = 24


class ApiPayloadError(ValueError):
    """User-facing payload error; the message is shown as-is."""


def align_frames(n):
    n = max(5, int(n))
    while n % 17 != 5:
        n += 1
    return n


def target_duration(duration_s):
    """The real clip duration after the server's frame snap, in seconds."""
    return align_frames(round(float(duration_s) * FPS)) / float(FPS)


def upload_name(path, taken):
    """Unique upload basename for a local file path (server flattens paths)."""
    base = os.path.basename(str(path).replace("\\", "/"))
    stem, dot, ext = base.partition(".")
    name, i = base, 2
    while name in taken:
        name = "%s_%d%s%s" % (stem, i, dot, ext)
        i += 1
    taken.add(name)
    return name


def ref_body(*, prompt, references, duration_s, duration_mode=None):
    body = {
        "mode": "ref_extend" if duration_mode is not None else "ref",
        "prompt": prompt,
        "references": list(references),
        "duration_s": float(duration_s),
        "enhance": False,
    }
    if duration_mode is not None:
        body["duration_mode"] = duration_mode
    return body


def flf_body(*, prompt, duration_s, first_name=None, last_name=None,
             size_mode="aspect"):
    if size_mode == "source" and not first_name:
        raise ApiPayloadError("size_mode 'source' needs a first frame image")
    body = {"mode": "flf", "prompt": prompt, "duration_s": float(duration_s),
            "enhance": False, "size_mode": size_mode}
    if first_name:
        body["first_image"] = first_name
    if last_name:
        body["last_image"] = last_name
    return body


def extend_extra(context_seconds, source_name):
    if float(context_seconds) <= 0:
        raise ApiPayloadError("context_seconds must be > 0")
    return {"source_video": source_name,
            "context_seconds": float(context_seconds)}
```

- [x] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_api_payload.py -q`
Expected: all PASS

- [x] **Step 5: Commit**

```bash
git -C /mnt/d/Comfyui/comfyOficial/custom_nodes/h3_tools add h3_tools/api_payload.py tests/test_api_payload.py
git -C /mnt/d/Comfyui/comfyOficial/custom_nodes/h3_tools commit -m "api: builders puros dos bodies do /generate (grade 17k+5, nomes de upload)"
```

---

### Task 2: `modal_client.py` — HTTP client with rotation

**Files:**
- Create: `h3_tools/modal_client.py`
- Test: `tests/test_modal_client.py`

**Interfaces:**
- Consumes: nothing from the pack (self-contained; `requests` lazy like enhancer.py).
- Produces (used by Tasks 3–5):
  - `load_config(accounts_json="", api_url="", api_key="") -> dict` with keys
    `api_key`, `accounts` (list of `{"name", "url"}`), `gpu`, `rate_usd_h`.
    Resolution order: explicit path → env `H3_MODAL_ACCOUNTS_JSON` → url+key pair.
  - `generate(cfg, body, files, *, timeout_s, account="auto", progress=None,
    interrupt=None, post=None, get=None, sleep=None, clock=None)
    -> (result: dict, meta: {"account": str, "duration_s": float})`
    where `files` is `[{"name": str, "b64": str}]` and `progress` receives the
    raw `/status` progress dict each poll.
  - `ModalApiError(RuntimeError)` — all accounts failed;
    `WorkflowFailed(ModalApiError)` — server-side ComfyUI error, NOT retried
    on other accounts (it would reproduce, burning credits).

- [x] **Step 1: Write the failing tests**

```python
# tests/test_modal_client.py
import json

import pytest

from h3_tools import modal_client as mc


class Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload


def cfg_two():
    return {"api_key": "k", "gpu": "RTX-PRO-6000", "rate_usd_h": 3.03,
            "accounts": [{"name": "a", "url": "https://a"},
                         {"name": "b", "url": "https://b"}]}


def test_load_config_from_file(tmp_path):
    p = tmp_path / "accounts.json"
    p.write_text(json.dumps({
        "api_key": "secret", "gpu": "RTX-PRO-6000",
        "accounts": [
            {"name": "x", "url": "https://x", "enabled": True},
            {"name": "off", "url": "https://off", "enabled": False},
            {"name": "nourl", "enabled": True},
        ]}), encoding="utf-8")
    cfg = mc.load_config(accounts_json=str(p))
    assert cfg["api_key"] == "secret"
    assert [a["name"] for a in cfg["accounts"]] == ["x"]


def test_load_config_from_pair():
    cfg = mc.load_config(api_url="https://solo", api_key="kk")
    assert cfg["accounts"] == [{"name": "api_url", "url": "https://solo"}]
    assert cfg["api_key"] == "kk"


def test_load_config_nothing_raises():
    with pytest.raises(mc.ModalApiError):
        mc.load_config()


def test_generate_happy_path_with_progress():
    calls = {"get": [], "post": []}
    seen = []

    def post(url, **kw):
        calls["post"].append(url)
        assert kw["headers"]["X-API-Key"] == "k"
        assert kw["json"]["mode"] == "flf"
        return Resp(200, {"call_id": "fc-1", "seed": 7})

    ticks = [
        Resp(200, {"ok": True}),                                     # health
        Resp(202, {"status": "running",
                   "progress": {"value": 1, "max": 8}}),             # poll 1
        Resp(200, {"status": "done",
                   "result": {"outputs": [], "texts": [],
                              "duration_s": 12.5}}),                 # poll 2
    ]

    def get(url, **kw):
        calls["get"].append(url)
        return ticks.pop(0)

    result, meta = mc.generate(
        cfg_two(), {"mode": "flf"}, [], timeout_s=60,
        progress=lambda p: seen.append(p), post=post, get=get,
        sleep=lambda s: None, clock=iter(range(100)).__next__)
    assert result["duration_s"] == 12.5
    assert meta["account"] == "a"
    assert seen == [{"value": 1, "max": 8}]
    assert calls["post"] == ["https://a/generate"]


def test_generate_rotates_on_account_failure():
    def post(url, **kw):
        return Resp(200, {"call_id": "fc-2"})

    ticks = {"https://a/health": Resp(503, {"error": "down"}),
             "https://b/health": Resp(200, {"ok": True}),
             "https://b/status/fc-2": Resp(200, {"status": "done",
                                                 "result": {"duration_s": 1}})}

    def get(url, **kw):
        return ticks[url]

    result, meta = mc.generate(cfg_two(), {"mode": "flf"}, [], timeout_s=60,
                               post=post, get=get, sleep=lambda s: None,
                               clock=iter(range(100)).__next__)
    assert meta["account"] == "b"


def test_generate_workflow_failure_does_not_rotate():
    def post(url, **kw):
        return Resp(200, {"call_id": "fc-3"})

    def get(url, **kw):
        if url.endswith("/health"):
            return Resp(200, {"ok": True})
        return Resp(200, {"status": "failed", "error": "node exploded"})

    with pytest.raises(mc.WorkflowFailed, match="node exploded"):
        mc.generate(cfg_two(), {"mode": "flf"}, [], timeout_s=60,
                    post=post, get=get, sleep=lambda s: None,
                    clock=iter(range(100)).__next__)


def test_generate_400_body_error_fails_hard():
    def post(url, **kw):
        return Resp(400, {"error": "prompt is required"})

    def get(url, **kw):
        return Resp(200, {"ok": True})

    with pytest.raises(mc.WorkflowFailed, match="prompt is required"):
        mc.generate(cfg_two(), {"mode": "flf"}, [], timeout_s=60,
                    post=post, get=get, sleep=lambda s: None,
                    clock=iter(range(100)).__next__)


def test_generate_interrupt_stops_polling():
    def post(url, **kw):
        return Resp(200, {"call_id": "fc-4"})

    def get(url, **kw):
        if url.endswith("/health"):
            return Resp(200, {"ok": True})
        return Resp(202, {"status": "running", "progress": {}})

    class Stop(Exception):
        pass

    def interrupt():
        raise Stop()

    with pytest.raises(Stop):
        mc.generate(cfg_two(), {"mode": "flf"}, [], timeout_s=60,
                    interrupt=interrupt, post=post, get=get,
                    sleep=lambda s: None, clock=iter(range(100)).__next__)
```

- [x] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_modal_client.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'h3_tools.modal_client'`

- [x] **Step 3: Write the implementation**

```python
# h3_tools/modal_client.py
"""HTTP client for the modal_inferencito /generate API, with account rotation.

Stdlib at import time; `requests` is imported lazily inside generate() so the
module stays testable with injected doubles (same pattern as enhancer.py).
Contract: modal_inferencito/API.md. Account-level failures (unreachable,
suspended) rotate to the next account; workflow-level failures raise
immediately — retrying them elsewhere would burn credits on the same bug.
"""

import json
import logging
import os
import time

GPU_RATES_USD_H = {"RTX-PRO-6000": 3.03, "L40S": 1.95, "A100-80GB": 2.50,
                   "H100": 3.95, "H200": 4.54, "B200": 6.25}


class ModalApiError(RuntimeError):
    """All accounts failed (or config is unusable); message shown as-is."""


class WorkflowFailed(ModalApiError):
    """The server-side ComfyUI graph failed — reproducible, never rotated."""


def load_config(accounts_json="", api_url="", api_key=""):
    path = (accounts_json or "").strip() or os.environ.get(
        "H3_MODAL_ACCOUNTS_JSON", "")
    if path:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            raise ModalApiError("could not read accounts json %r: %s"
                                % (path, e)) from e
        accounts = [{"name": a.get("name", "?"), "url": a["url"]}
                    for a in data.get("accounts", [])
                    if a.get("url") and a.get("enabled", True)]
        if not accounts:
            raise ModalApiError("accounts json %r has no enabled account with "
                                "a url (run deploy_all.py first)" % path)
        gpu = data.get("gpu", "RTX-PRO-6000")
        return {"api_key": data.get("api_key", ""), "accounts": accounts,
                "gpu": gpu, "rate_usd_h": GPU_RATES_USD_H.get(gpu, 3.03)}
    if api_url.strip() and api_key.strip():
        return {"api_key": api_key.strip(),
                "accounts": [{"name": "api_url", "url": api_url.strip().rstrip("/")}],
                "gpu": "RTX-PRO-6000", "rate_usd_h": 3.03}
    raise ModalApiError(
        "no API configured: set the accounts_json widget (path to "
        "modal_inferencito/accounts.json), or the H3_MODAL_ACCOUNTS_JSON "
        "environment variable, or fill api_url + api_key")


def generate(cfg, body, files, *, timeout_s, account="auto", progress=None,
             interrupt=None, post=None, get=None, sleep=None, clock=None):
    if post is None or get is None:
        import requests
        post = post or requests.post
        get = get or requests.get
    sleep = sleep or time.sleep
    clock = clock or time.monotonic

    accounts = cfg["accounts"]
    if account != "auto":
        chosen = [a for a in accounts if a["name"] == account]
        if not chosen:
            raise ModalApiError("account %r not found (have: %s)" % (
                account, ", ".join(a["name"] for a in accounts)))
        accounts = chosen
    headers = {"X-API-Key": cfg["api_key"]}
    payload = dict(body, files=files)

    errors = []
    for acc in accounts:
        try:
            r = get(acc["url"] + "/health", headers=headers, timeout=20)
            if getattr(r, "status_code", 0) != 200:
                raise OSError("health HTTP %s" % getattr(r, "status_code", "?"))
        except Exception as e:
            errors.append("%s: health failed (%s)" % (acc["name"], e))
            continue
        try:
            r = post(acc["url"] + "/generate", json=payload, headers=headers,
                     timeout=300)
        except Exception as e:
            errors.append("%s: submit failed (%s)" % (acc["name"], e))
            continue
        status = getattr(r, "status_code", 0)
        if status == 400:
            # invalid request body: our bug or the user's — same everywhere
            raise WorkflowFailed("generate rejected: %s"
                                 % r.json().get("error", "HTTP 400"))
        if status != 200:
            errors.append("%s: submit HTTP %s" % (acc["name"], status))
            continue
        call_id = r.json()["call_id"]
        logging.info("h3_tools: api job %s submitted to %s", call_id,
                     acc["name"])

        started = clock()
        while True:
            if interrupt is not None:
                interrupt()
            if clock() - started > timeout_s:
                raise ModalApiError("timed out after %ds waiting for %s on %s"
                                    % (timeout_s, call_id, acc["name"]))
            try:
                s = get(acc["url"] + "/status/" + call_id, headers=headers,
                        timeout=60)
            except Exception as e:
                logging.warning("h3_tools: poll error (%s); retrying", e)
                sleep(5.0)
                continue
            if getattr(s, "status_code", 0) == 202:
                if progress is not None:
                    try:
                        progress(s.json().get("progress") or {})
                    except Exception:
                        logging.exception("h3_tools: progress callback failed")
                sleep(3.0)
                continue
            data = s.json()
            if data.get("status") == "failed":
                raise WorkflowFailed("workflow failed on %s: %s"
                                     % (acc["name"], data.get("error", "?")))
            result = data["result"]
            duration = float(result.get("duration_s") or (clock() - started))
            return result, {"account": acc["name"], "duration_s": duration}
    raise ModalApiError("all accounts failed:\n  " + "\n  ".join(
        errors or ["(no accounts configured)"]))
```

- [x] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_modal_client.py tests/test_api_payload.py -q`
Expected: all PASS

- [x] **Step 5: Commit**

```bash
git -C /mnt/d/Comfyui/comfyOficial/custom_nodes/h3_tools add h3_tools/modal_client.py tests/test_modal_client.py
git -C /mnt/d/Comfyui/comfyOficial/custom_nodes/h3_tools commit -m "api: cliente HTTP do /generate com rotacao de contas e polling"
```

---

### Task 3: `enhancer_flow.py` extraction + soft `__init__` + board binding

**Files:**
- Create: `h3_tools/enhancer_flow.py`
- Modify: `h3_tools/node.py` (lines 27–218: the four helpers move out; an
  import replaces them), `__init__.py`, `web/board.js:11`
- Test: existing suite must stay green: `python3 -m pytest tests/ -q`

**Interfaces:**
- Produces (used by node.py and Task 4): `enhancer_flow._decode_refs(ref_list, pbar)`,
  `enhancer_flow._reference_vision_parts(ref_list, payloads)`,
  `enhancer_flow._run_enhancer(prompt, ref_list, payloads, durations, *, model,
  fallback_model, reasoning, api_key_widget, vision, preset, system_override,
  seed, target_duration, vision_format="default", system_prompts=None,
  context_parts=None, context_sha=None, duration_mode=None, video_sent=False)`,
  `enhancer_flow._enhancer_inputs(presets)`,
  `enhancer_flow._unknown_mentions_message(unknown, ref_list)`.
  All FIVE move VERBATIM (bodies unchanged) from node.py.

- [x] **Step 1: Create `h3_tools/enhancer_flow.py`**

Header + the five helpers cut verbatim from node.py (`_unknown_mentions_message`,
`_decode_refs`, `_reference_vision_parts`, `_run_enhancer`, `_enhancer_inputs`),
with exactly these imports (note: NO `comfy_extras.nodes_minimax_h3` — that is
the whole point of the move):

```python
# h3_tools/enhancer_flow.py
"""Enhancer orchestration shared by the Pro nodes and the API client nodes.

Moved verbatim from node.py so it can load on a frontend ComfyUI that lacks
comfy_extras.nodes_minimax_h3 (node.py imports the native encoder at module
top). No behavior change.
"""

import logging
import os

import folder_paths
from comfy_api.latest import io

from . import continuation, enhancer, media, refs

# ... the five functions, bodies byte-identical to node.py ...
```

- [x] **Step 2: Replace the definitions in `node.py` with the import**

Delete the five function definitions from node.py and add to its imports:

```python
from .enhancer_flow import (_decode_refs, _enhancer_inputs,
                            _reference_vision_parts, _run_enhancer,
                            _unknown_mentions_message)
```

- [x] **Step 3: Make `__init__.py` degrade softly and register API nodes**

```python
"""h3_tools — MiniMax H3 Reference to Video (Pro) + Modal API client nodes."""

import logging

from comfy_api.latest import ComfyExtension

WEB_DIRECTORY = "./web"

try:
    import comfy_extras.nodes_minimax_h3  # noqa: F401
    from .h3_tools.node import H3RefToVideoContinuePro, H3RefToVideoPro
    _NATIVE_NODES = [H3RefToVideoPro, H3RefToVideoContinuePro]
except ImportError:
    _NATIVE_NODES = []
    logging.warning(
        "h3_tools: this ComfyUI has no MiniMax H3 support "
        "(comfy_extras.nodes_minimax_h3) — loading only the Modal API "
        "client nodes. Update ComfyUI to master for the local Pro nodes.")

from .h3_tools.api_nodes import API_NODES  # noqa: E402


class H3ToolsExtension(ComfyExtension):
    async def get_node_list(self):
        return _NATIVE_NODES + API_NODES


async def comfy_entrypoint() -> H3ToolsExtension:
    return H3ToolsExtension()
```

(`API_NODES` will not exist until Task 4 — create a placeholder
`h3_tools/api_nodes.py` with `API_NODES = []` in this task so the pack loads.)

- [x] **Step 4: Add the prepare ids to `web/board.js` line 11**

```javascript
const NODE_IDS = new Set(["H3RefToVideoPro", "H3RefToVideoContinuePro",
                          "H3ApiPrepareRef", "H3ApiPrepareRefContinue"]);
```

- [x] **Step 5: Run the full existing suite**

Run: `python3 -m pytest tests/ -q`
Expected: all PASS (the pure modules never import node.py, but this guards the refactor).

- [x] **Step 6: Import-smoke node.py on the GPU-style install (has minimax) via grep sanity**

node.py cannot be imported outside ComfyUI. Instead verify the refactor is
textually complete:

Run: `grep -c "def _run_enhancer\|def _decode_refs\|def _reference_vision_parts\|def _enhancer_inputs\|def _unknown_mentions_message" h3_tools/node.py h3_tools/enhancer_flow.py`
Expected: `h3_tools/node.py:0` and `h3_tools/enhancer_flow.py:5`

- [x] **Step 7: Commit**

```bash
git -C /mnt/d/Comfyui/comfyOficial/custom_nodes/h3_tools add -A
git -C /mnt/d/Comfyui/comfyOficial/custom_nodes/h3_tools commit -m "refactor: helpers do enhancer movidos p/ enhancer_flow; __init__ tolera comfy sem minimax (frontend CPU)"
```

---

### Task 4: `api_nodes.py` — the 6 client nodes

**Files:**
- Modify: `h3_tools/api_nodes.py` (replaces the Task 3 placeholder)
- Test: `python3 -m pytest tests/ -q` (regression) + Task 5 e2e + user's visual test

**Interfaces:**
- Consumes: everything from Tasks 1–3 plus `media.encode_video_context`,
  `media.image_data_url`, `continuation.timeline`, `continuation.context_text`,
  `refs.parse_references`, `refs.build_final_prompt`, `refs.unknown_mentions`.
- Produces: `API_NODES` list with the 6 io.ComfyNode classes; payload dicts per
  Task 1's shape flowing over `io.Custom("H3_API_PAYLOAD")` sockets.

- [x] **Step 1: Write the module skeleton with shared helpers**

```python
# h3_tools/api_nodes.py
"""Client nodes for the modal_inferencito /generate API (SPEC: repo API.md).

Prepare nodes (reference family) run the media board + @mentions + OpenRouter
enhancer LOCALLY and emit a payload; generate nodes upload the media, submit,
poll with live progress/preview, and return the produced VIDEO. First/last
frame nodes take a ready prompt STRING — no enhancer, no mentions (the user
enhances upstream with their own nodes).
"""

import base64
import logging
import os
import time

import comfy.model_management
import comfy.utils
import folder_paths
from comfy_api.latest import InputImpl, io

from . import api_payload, continuation, enhancer, media, modal_client, refs
from .enhancer_flow import (_decode_refs, _enhancer_inputs,
                            _reference_vision_parts, _run_enhancer,
                            _unknown_mentions_message)
from .system_prompts import SYSTEM_PROMPTS_CONTINUE

Payload = io.Custom("H3_API_PAYLOAD")


def _connection_inputs():
    return [
        io.String.Input("accounts_json", default="",
            tooltip="Path to modal_inferencito/accounts.json. Empty falls back "
                    "to the H3_MODAL_ACCOUNTS_JSON environment variable, then "
                    "to api_url + api_key below."),
        io.String.Input("api_url", default="",
            tooltip="Single-account fallback: https://<ws>--inferencito-api.modal.run"),
        io.String.Input("api_key", default="",
            tooltip="X-API-Key for api_url. Prefer accounts_json/env on shared hosts."),
        io.String.Input("account", default="auto",
            tooltip="Account name from accounts.json, or 'auto' for rotation."),
        io.Combo.Input("quality", options=["fast", "normal", "quality"],
            default="fast",
            tooltip="fast = turbo lora + 8 steps; normal = 20; quality = 32."),
        io.Int.Input("seed", default=0, min=-1, max=2**48 - 1,
            tooltip="-1 = random on the server (returned in the log)."),
        io.Int.Input("timeout_min", default=30, min=1, max=180),
    ]


def _stage_image(image, taken):
    """IMAGE tensor -> png file in the input dir; returns (upload_name, path)."""
    import numpy as np
    from PIL import Image as PILImage

    arr = image
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().float().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    arr = (np.clip(arr[..., :3], 0.0, 1.0) * 255.0).astype("uint8")
    stage_dir = os.path.join(folder_paths.get_input_directory(), "h3_api_stage")
    os.makedirs(stage_dir, exist_ok=True)
    name = api_payload.upload_name("frame_%d.png" % int(time.time() * 1000), taken)
    path = os.path.join(stage_dir, name)
    PILImage.fromarray(arr).save(path)
    return name, path


def _video_source_path(video, taken):
    """VIDEO socket -> (upload_name, existing file path or staged copy)."""
    src = None
    if hasattr(video, "get_stream_source"):
        try:
            candidate = video.get_stream_source()
            if isinstance(candidate, str) and os.path.exists(candidate):
                src = candidate
        except Exception:
            src = None
    if src is None:
        stage_dir = os.path.join(folder_paths.get_input_directory(), "h3_api_stage")
        os.makedirs(stage_dir, exist_ok=True)
        src = os.path.join(stage_dir, "source_%d.mp4" % int(time.time() * 1000))
        video.save_to(src)
    return api_payload.upload_name(src, taken), src


def _b64_file(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _run_remote(body, uploads, *, accounts_json, api_url, api_key, account,
                quality, seed, timeout_min, extra=None):
    cfg = modal_client.load_config(accounts_json, api_url, api_key)
    body = dict(body, quality=quality, **(extra or {}))
    if seed >= 0:
        body["seed"] = int(seed)
    files = [{"name": u["name"], "b64": _b64_file(u["path"])} for u in uploads]

    pbar = comfy.utils.ProgressBar(100)

    def on_progress(prog):
        value, total = prog.get("value") or 0, prog.get("max") or 0
        preview = None
        if prog.get("preview_b64"):
            try:
                import io as pyio

                from PIL import Image as PILImage
                img = PILImage.open(pyio.BytesIO(
                    base64.b64decode(prog["preview_b64"])))
                preview = ("JPEG", img.convert("RGB"), None)
            except Exception:
                preview = None
        if total:
            pbar.update_absolute(int(value * 100 / total), 100, preview)

    result, meta = modal_client.generate(
        cfg, body, files, timeout_s=timeout_min * 60, account=account,
        progress=on_progress,
        interrupt=comfy.model_management.throw_exception_if_processing_interrupted)

    videos = [o for o in result["outputs"] if o.get("kind") == "video"]
    if not videos:
        raise modal_client.WorkflowFailed("the job finished but returned no video")
    out_dir = os.path.join(folder_paths.get_output_directory(), "h3_api")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "%s_%s" % (
        time.strftime("%Y%m%d_%H%M%S"),
        os.path.basename(videos[0]["filename"])))
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(videos[0]["b64"]))

    final_prompt = next((t["text"] for t in result.get("texts", [])
                         if t.get("title") == "final_prompt"), "")
    cost = meta["duration_s"] / 3600.0 * cfg["rate_usd_h"]
    info = ("account=%s | gpu=%.1fs | est=$%.3f | %s"
            % (meta["account"], meta["duration_s"], cost,
               os.path.basename(out_path)))
    logging.info("h3_tools: api job done — %s", info)
    return InputImpl.VideoFromFile(out_path), final_prompt, info
```

- [x] **Step 2: Write the two prepare nodes**

```python
class H3ApiPrepareRef(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3ApiPrepareRef",
            display_name="MiniMax H3 API — Prepare Reference",
            category="api/minimax",
            description="Media board + @mentions + LOCAL OpenRouter enhancer. "
                        "Feeds a Generate Reference node; nothing runs on the "
                        "GPU here.",
            inputs=[
                io.String.Input("prompt", multiline=True, dynamic_prompts=False,
                                default=""),
                io.String.Input("references", default="[]",
                    tooltip="JSON array managed by the board UI (same format "
                            "as the Pro nodes)."),
                io.Float.Input("duration_s", default=10.0, min=0.3, max=150.0,
                    step=0.1,
                    tooltip="Final video duration in seconds (snapped to the "
                            "17k+5 frame grid). Lives here because the "
                            "enhancer writes against this exact clock."),
            ] + _enhancer_inputs(enhancer.SYSTEM_PROMPTS),
            outputs=[
                Payload.Output(display_name="payload"),
                io.String.Output(display_name="final_prompt",
                    tooltip="Preview of the exact prompt the server will "
                            "encode (post-enhancer, post-substitution)."),
            ],
        )

    @classmethod
    def execute(cls, prompt, references, duration_s, enhance_prompt,
                enhancer_model, enhancer_model_fallback, enhancer_reasoning,
                openrouter_api_key, enhancer_vision, vision_format,
                system_prompt_preset, enhancer_seed,
                system_prompt_override=None) -> io.NodeOutput:
        ref_list = refs.parse_references(references)
        unknown = refs.unknown_mentions(prompt, ref_list)
        if unknown:
            raise ValueError(_unknown_mentions_message(unknown, ref_list))
        pbar = comfy.utils.ProgressBar(len(ref_list) + (1 if enhance_prompt else 0))
        payloads, durations = _decode_refs(ref_list, pbar)

        working_prompt = prompt
        if enhance_prompt:
            working_prompt = _run_enhancer(
                prompt, ref_list, payloads, durations, model=enhancer_model,
                fallback_model=enhancer_model_fallback,
                reasoning=enhancer_reasoning, api_key_widget=openrouter_api_key,
                vision=enhancer_vision, preset=system_prompt_preset,
                system_override=system_prompt_override, seed=enhancer_seed,
                target_duration=api_payload.target_duration(duration_s),
                vision_format=vision_format)
            pbar.update(1)

        taken = set()
        uploads, refs_out = [], []
        for r in ref_list:
            path = folder_paths.get_annotated_filepath(r.file)
            name = api_payload.upload_name(path, taken)
            uploads.append({"name": name, "path": path})
            item = {"name": r.name, "type": r.type, "file": name}
            if r.use_soundtrack:
                item["use_soundtrack"] = True
            refs_out.append(item)

        body = api_payload.ref_body(prompt=working_prompt, references=refs_out,
                                    duration_s=duration_s)
        preview = refs.build_final_prompt(working_prompt, ref_list)
        payload = {"mode": body["mode"], "body": body, "uploads": uploads,
                   "preview_prompt": preview}
        return io.NodeOutput(payload, preview)
```

`H3ApiPrepareRefContinue` is the same node plus:
- inputs `io.Video.Input("video")` (REQUIRED — the footage to continue) and
  `io.Combo.Input("duration_mode", options=["total", "new_only"], default="total")`,
  and it uses `_enhancer_inputs(SYSTEM_PROMPTS_CONTINUE)`;
- before the enhancer it builds the LLM context exactly like
  `H3RefToVideoContinuePro.execute` does (node.py:441–476), with
  `align=api_payload.align_frames` injected into `continuation.timeline`:

```python
        source_duration = float(video.get_duration())
        t = continuation.timeline(duration_mode, api_payload.align_frames(
            round(duration_s * 24)), source_duration, api_payload.align_frames)
        if enhance_prompt:
            data_url, _full, sent_duration = media.encode_video_context(video)
            context_parts = [
                {"type": "text", "text": continuation.context_text(
                    t["source_duration"], sent_duration, t["total_duration"],
                    t["from_zero"])},
                {"type": "video_url", "video_url": {"url": data_url}},
            ]
            import hashlib
            context_sha = hashlib.sha256(data_url.encode("ascii")).hexdigest()
            working_prompt = _run_enhancer(
                prompt, ref_list, payloads, durations, model=enhancer_model,
                fallback_model=enhancer_model_fallback,
                reasoning=enhancer_reasoning, api_key_widget=openrouter_api_key,
                vision=enhancer_vision, preset=system_prompt_preset,
                system_override=system_prompt_override, seed=enhancer_seed,
                target_duration=t["total_duration"], vision_format=vision_format,
                system_prompts=SYSTEM_PROMPTS_CONTINUE,
                context_parts=context_parts, context_sha=context_sha,
                duration_mode=duration_mode, video_sent=True)
```
- after building `refs_out`/`uploads` it stages the source video and stores it
  in the payload (the generate node adds context_seconds):

```python
        vid_name, vid_path = _video_source_path(video, taken)
        uploads.append({"name": vid_name, "path": vid_path})
        body = api_payload.ref_body(prompt=working_prompt, references=refs_out,
                                    duration_s=duration_s,
                                    duration_mode=duration_mode)
        payload = {"mode": body["mode"], "body": body, "uploads": uploads,
                   "preview_prompt": preview, "source_name": vid_name}
```

- [x] **Step 3: Write the four generate nodes**

```python
class H3ApiGenerateRef(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3ApiGenerateRef",
            display_name="MiniMax H3 API — Generate Reference",
            category="api/minimax",
            description="Submits a prepared reference payload to the Modal "
                        "API and returns the generated video.",
            inputs=[
                Payload.Input("payload"),
                io.Combo.Input("aspect",
                    options=["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"],
                    default="16:9"),
                io.Float.Input("megapixels", default=0.4, min=0.1, max=1.05,
                    step=0.05,
                    tooltip="Canvas area; the server caps at 768x1344."),
            ] + _connection_inputs(),
            outputs=[
                io.Video.Output(display_name="video"),
                io.String.Output(display_name="final_prompt"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, payload, aspect, megapixels, accounts_json, api_url,
                api_key, account, quality, seed, timeout_min) -> io.NodeOutput:
        if payload.get("mode") != "ref":
            raise ValueError("this node takes a payload from 'Prepare "
                             "Reference' (got mode %r) — the Continue payload "
                             "goes into Generate Reference Continue"
                             % payload.get("mode"))
        video, final_prompt, info = _run_remote(
            payload["body"], payload["uploads"], accounts_json=accounts_json,
            api_url=api_url, api_key=api_key, account=account, quality=quality,
            seed=seed, timeout_min=timeout_min,
            extra={"aspect": aspect, "megapixels": float(megapixels)})
        return io.NodeOutput(video, final_prompt or payload["preview_prompt"], info)
```

The other three follow the same shape; differences only:

- `H3ApiGenerateRefContinue` — expects `payload["mode"] == "ref_extend"`; no
  aspect widget (server sizes from the source); widgets `megapixels`
  (default 0.5) and `io.Float.Input("context_seconds", default=2.0, min=0.2,
  max=10.0, step=0.1, tooltip="Trailing seconds of the source anchored by "
  "AddGuide on the server.")`; `extra=dict(
  api_payload.extend_extra(context_seconds, payload["source_name"]),
  megapixels=float(megapixels))`.
- `H3ApiGenerateFlf` — NO payload input. Inputs: `io.String.Input("prompt",
  multiline=True, force_input=True, tooltip="Ready prompt (enhance upstream "
  "with your own nodes — this node sends it verbatim).")`,
  `io.Image.Input("first_frame", optional=True)`,
  `io.Image.Input("last_frame", optional=True)`,
  `io.Float.Input("duration_s", default=10.0, min=0.3, max=150.0, step=0.1)`,
  `io.Combo.Input("size_mode", options=["aspect", "source"], default="aspect")`,
  aspect + megapixels (default 0.5) + `_connection_inputs()`. Execute stages
  the frames and builds the body inline:

```python
    @classmethod
    def execute(cls, prompt, duration_s, size_mode, aspect, megapixels,
                accounts_json, api_url, api_key, account, quality, seed,
                timeout_min, first_frame=None, last_frame=None) -> io.NodeOutput:
        taken, uploads = set(), []
        first_name = last_name = None
        if first_frame is not None:
            first_name, path = _stage_image(first_frame, taken)
            uploads.append({"name": first_name, "path": path})
        if last_frame is not None:
            last_name, path = _stage_image(last_frame, taken)
            uploads.append({"name": last_name, "path": path})
        body = api_payload.flf_body(prompt=prompt, duration_s=duration_s,
                                    first_name=first_name, last_name=last_name,
                                    size_mode=size_mode)
        video, final_prompt, info = _run_remote(
            body, uploads, accounts_json=accounts_json, api_url=api_url,
            api_key=api_key, account=account, quality=quality, seed=seed,
            timeout_min=timeout_min,
            extra={"aspect": aspect, "megapixels": float(megapixels)})
        return io.NodeOutput(video, final_prompt or prompt, info)
```

- `H3ApiGenerateFlfContinue` — inputs: prompt STRING (force_input),
  `io.Video.Input("video")` (required), duration_s, megapixels (default 0.6),
  context_seconds (default 2.0) + `_connection_inputs()`. No first/last, no
  size_mode (server always sizes from the source). Execute:

```python
        taken = set()
        vid_name, vid_path = _video_source_path(video, taken)
        body = dict({"mode": "flf_extend", "prompt": prompt,
                     "duration_s": float(duration_s), "enhance": False})
        video_out, final_prompt, info = _run_remote(
            body, [{"name": vid_name, "path": vid_path}],
            accounts_json=accounts_json, api_url=api_url, api_key=api_key,
            account=account, quality=quality, seed=seed,
            timeout_min=timeout_min,
            extra=dict(api_payload.extend_extra(context_seconds, vid_name),
                       megapixels=float(megapixels)))
        return io.NodeOutput(video_out, final_prompt or prompt, info)
```

Close the module with:

```python
API_NODES = [H3ApiPrepareRef, H3ApiPrepareRefContinue, H3ApiGenerateRef,
             H3ApiGenerateRefContinue, H3ApiGenerateFlf,
             H3ApiGenerateFlfContinue]
```

- [x] **Step 4: Regression + syntax check**

Run: `python3 -m pytest tests/ -q && python3 -m py_compile h3_tools/api_nodes.py h3_tools/enhancer_flow.py && echo SYNTAX-OK`
Expected: tests PASS, `SYNTAX-OK`

- [x] **Step 5: Commit**

```bash
git -C /mnt/d/Comfyui/comfyOficial/custom_nodes/h3_tools add h3_tools/api_nodes.py
git -C /mnt/d/Comfyui/comfyOficial/custom_nodes/h3_tools commit -m "api: 6 nodes cliente (2 prepare ref + 4 generate) contra o /generate do Modal"
```

---

### Task 5: E2E without ComfyUI (WSL, real API)

**Files:**
- Create: `scripts/e2e_api.py` (in the h3_tools repo)

**Interfaces:**
- Consumes: `api_payload`, `modal_client` (Tasks 1–2), the deployed API, and
  the WSL accounts file `/home/adolfocesar/projects/modal_inferencito/accounts.json`.

- [x] **Step 1: Write the driver**

```python
#!/usr/bin/env python3
"""E2E of api_payload + modal_client against the real deployed API (no ComfyUI).

Usage (WSL):
  python3 scripts/e2e_api.py --dry-run     # validates the 4 bodies, free
  python3 scripts/e2e_api.py               # one real flf fast 3s job (~$0.03)
"""

import argparse
import base64
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from h3_tools import api_payload, modal_client  # noqa: E402

ACCOUNTS = "/home/adolfocesar/projects/modal_inferencito/accounts.json"
ASSETS = Path("/home/adolfocesar/projects/modal_inferencito/scripts/test_assets")


def b64(p):
    return base64.b64encode(p.read_bytes()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = modal_client.load_config(accounts_json=ACCOUNTS)

    bodies = {
        "ref": (api_payload.ref_body(
            prompt="a @garota acena", duration_s=3,
            references=[{"name": "garota", "type": "image", "file": "ref_image.png"}]),
            [{"name": "ref_image.png", "b64": b64(ASSETS / "ref_image.png")}]),
        "ref_extend": (dict(api_payload.ref_body(
            prompt="a @garota acena", duration_s=6, duration_mode="total",
            references=[{"name": "garota", "type": "image", "file": "ref_image.png"}]),
            **api_payload.extend_extra(2.0, "source_video.mp4")),
            [{"name": "ref_image.png", "b64": b64(ASSETS / "ref_image.png")},
             {"name": "source_video.mp4", "b64": b64(ASSETS / "source_video.mp4")}]),
        "flf": (api_payload.flf_body(
            prompt="ela sorri para a camera", duration_s=3,
            first_name="first_frame.jpg"),
            [{"name": "first_frame.jpg", "b64": b64(ASSETS / "first_frame.jpg")}]),
        "flf_extend": (dict(api_payload.flf_body(
            prompt="a cena continua", duration_s=6),
            **api_payload.extend_extra(2.0, "source_video.mp4")),
            [{"name": "source_video.mp4", "b64": b64(ASSETS / "source_video.mp4")}]),
    }
    bodies["flf_extend"][0]["mode"] = "flf_extend"

    if args.dry_run:
        import requests
        acc = cfg["accounts"][0]
        for mode, (body, files) in bodies.items():
            r = requests.post(acc["url"] + "/generate",
                              json=dict(body, files=files, dry_run=True,
                                        quality="fast", seed=1),
                              headers={"X-API-Key": cfg["api_key"]}, timeout=120)
            ok = r.status_code == 200 and "workflow" in r.json()
            print(f"{mode}: {'ok' if ok else 'FAIL ' + r.text[:200]}")
            if not ok:
                sys.exit(1)
        return

    body, files = bodies["flf"]
    t0 = time.time()
    result, meta = modal_client.generate(
        cfg, dict(body, quality="fast", seed=7), files, timeout_s=1800,
        progress=lambda p: print("  progress", p.get("value"), "/", p.get("max")))
    out = Path("/tmp/e2e_api_flf.mp4")
    out.write_bytes(base64.b64decode(result["outputs"][0]["b64"]))
    print(f"ok: {out} | account {meta['account']} | gpu {meta['duration_s']}s "
          f"| wall {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
```

- [x] **Step 2: Run the dry-run**

Run: `python3 scripts/e2e_api.py --dry-run`
Expected: 4 lines ending in `ok`

- [x] **Step 3: Run the real job**

Run: `python3 scripts/e2e_api.py`
Expected: progress lines, then `ok: /tmp/e2e_api_flf.mp4 ...`; verify with
`ffprobe -v error -show_entries stream=codec_type -of csv=p=0 /tmp/e2e_api_flf.mp4`
showing `video` and `audio` streams.

- [x] **Step 4: Commit**

```bash
git -C /mnt/d/Comfyui/comfyOficial/custom_nodes/h3_tools add scripts/e2e_api.py
git -C /mnt/d/Comfyui/comfyOficial/custom_nodes/h3_tools commit -m "api: driver e2e sem ComfyUI contra a API real"
```

---

### Task 6: README + example workflow + user handoff

**Files:**
- Modify: `README.md` (new section after "Headless / API usage")
- Create: `example_workflows/api_generate_flf.json`

- [x] **Step 1: README section**

Add (verbatim, adjusting nothing else):

```markdown
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
```

- [x] **Step 2: Example workflow JSON**

`example_workflows/api_generate_flf.json` — a 3-node API-format graph:
`LoadImage` → `H3ApiGenerateFlf` (prompt widget text "ela sorri para a camera",
duration 4, quality fast) → `SaveVideo`. Build it by hand mirroring the node
schemas (widgets in schema order), validate with
`python3 -c "import json; json.load(open('example_workflows/api_generate_flf.json'))"`.

- [x] **Step 3: Commit**

```bash
git -C /mnt/d/Comfyui/comfyOficial/custom_nodes/h3_tools add README.md example_workflows/
git -C /mnt/d/Comfyui/comfyOficial/custom_nodes/h3_tools commit -m "docs: nodes cliente da API Modal + workflow de exemplo"
```

- [x] **Step 4: User acceptance (manual, blocking)**

Ask the user to restart their local ComfyUI (D:), confirm the 6 nodes appear
under `api/minimax`, run the example workflow, and test a Prepare Reference →
Generate Reference chain with the board. Fix whatever they hit; only then they
push (per their rule: push only after they test).

---

## Self-Review notes

- FLF nodes have no enhancer/@mentions (user's mid-plan clarification) — Tasks
  4/6 reflect it; the videovibe system prompt is intentionally NOT ported.
- `duration_s` lives on prepare nodes (ref family) because the local enhancer
  writes against that clock; generate nodes for flf carry their own since no
  LLM is involved.
- `continuation.timeline` gets `api_payload.align_frames` injected — the plan
  never imports `comfy_extras.nodes_minimax_h3` in client-loadable modules.
- Type check: payload dict keys (`mode`, `body`, `uploads`, `preview_prompt`,
  `source_name`) are consistent across Tasks 1, 4, and the generate nodes'
  validations; `modal_client.generate` signature matches every call site.
