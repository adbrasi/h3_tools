"""Client nodes for the modal_inferencito /generate API (contract: API.md).

Prepare nodes (reference family) run the media board + @mentions + OpenRouter
enhancer LOCALLY and emit a payload; generate nodes upload the media, submit,
poll with live progress/preview, and return the produced VIDEO. First/last
frame nodes take a ready prompt STRING — no enhancer, no mentions (the user
enhances upstream with their own nodes).
"""

import base64
import hashlib
import json
import logging
import os
import time

import comfy.model_management
import comfy.utils
import folder_paths
from comfy_api.latest import InputImpl, io

from . import api_payload, continuation, enhancer, media, modal_client, refs
from .enhancer_flow import (_decode_refs, _enhancer_inputs, _run_enhancer,
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
            control_after_generate=True,
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
            # a trimmed VideoFromFile must go through save_to (which applies
            # the trim); the raw stream source would upload the whole file
            trim = (video.get_active_trim_window()
                    if hasattr(video, "get_active_trim_window") else (0.0, 0.0))
            candidate = video.get_stream_source()
            if (tuple(trim) == (0.0, 0.0) and isinstance(candidate, str)
                    and os.path.exists(candidate)):
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


def _validate_prompt_and_refs(prompt, references):
    """Shared static validation (same rules as the Pro nodes)."""
    if not isinstance(prompt, str) or not isinstance(references, str):
        return True
    try:
        ref_list = refs.parse_references(references)
    except refs.RefError as e:
        return str(e)
    for r in ref_list:
        try:
            if folder_paths.exists_annotated_filepath(r.file):
                continue
            resolved = folder_paths.get_annotated_filepath(r.file)
        except (OSError, ValueError):
            return ('reference file not found or outside the input '
                    'directory: "%s"' % r.file)
        return ('reference file not found: "%s" (resolved to %s)'
                % (r.file, resolved))
    unknown = refs.unknown_mentions(prompt, ref_list)
    if unknown:
        return _unknown_mentions_message(unknown, ref_list)
    return True


def _prepare_fingerprint(kwargs):
    """Widget values + (mtime_ns, size) per referenced file (as the Pro nodes:
    re-uploading a changed file under the same name must invalidate the
    execution cache; the API key contributes only bool(key))."""
    try:
        data = {k: v for k, v in kwargs.items()
                if isinstance(v, (str, int, float, bool))}
        data["openrouter_api_key"] = bool(kwargs.get("openrouter_api_key"))
        stats = []
        for r in refs.parse_references(kwargs.get("references", "[]")):
            try:
                st = os.stat(folder_paths.get_annotated_filepath(r.file))
                stats.append([r.file, st.st_mtime_ns, st.st_size])
            except (OSError, ValueError):
                stats.append([r.file, -1, -1])
        data["_file_stats"] = stats
        blob = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()
    except Exception:
        return float("NaN")  # always re-run rather than serve a stale cache


def _stage_reference_uploads(ref_list, taken):
    """Resolve board files to (uploads, references-for-the-body)."""
    uploads, refs_out = [], []
    for r in ref_list:
        path = folder_paths.get_annotated_filepath(r.file)
        name = api_payload.upload_name(path, taken)
        uploads.append({"name": name, "path": path})
        item = {"name": r.name, "type": r.type, "file": name}
        if r.use_soundtrack:
            item["use_soundtrack"] = True
        refs_out.append(item)
    return uploads, refs_out


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
                                default="",
                    tooltip="Prompt with @name mentions (the server converts "
                            "them to the native <Picture i>/<Video k>/<Audio j> "
                            "tags)."),
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
    def validate_inputs(cls, prompt, references):
        return _validate_prompt_and_refs(prompt, references)

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return _prepare_fingerprint(kwargs)

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
        pbar = comfy.utils.ProgressBar(
            len(ref_list) + (1 if enhance_prompt else 0))
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
        uploads, refs_out = _stage_reference_uploads(ref_list, taken)
        body = api_payload.ref_body(prompt=working_prompt, references=refs_out,
                                    duration_s=duration_s)
        preview = refs.build_final_prompt(working_prompt, ref_list)
        payload = {"mode": body["mode"], "body": body, "uploads": uploads,
                   "preview_prompt": preview}
        return io.NodeOutput(payload, preview)


class H3ApiPrepareRefContinue(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3ApiPrepareRefContinue",
            display_name="MiniMax H3 API — Prepare Reference Continue",
            category="api/minimax",
            description="Prepare Reference plus the footage being continued: "
                        "the video is sent whole to the LLM enhancer (locally) "
                        "and uploaded as the continuation source. Feeds the "
                        "Generate Reference Continue node.",
            inputs=[
                io.Video.Input("video",
                    tooltip="The footage being continued — LLM context and the "
                            "server-side continuation source."),
                io.String.Input("prompt", multiline=True, dynamic_prompts=False,
                                default="",
                    tooltip="Prompt with @name mentions (the server converts "
                            "them to the native <Picture i>/<Video k>/<Audio j> "
                            "tags)."),
                io.String.Input("references", default="[]",
                    tooltip="JSON array managed by the board UI (same format "
                            "as the Pro nodes)."),
                io.Float.Input("duration_s", default=10.0, min=0.3, max=150.0,
                    step=0.1,
                    tooltip="Meaning depends on duration_mode: 'total' = final "
                            "video length (context included); 'new_only' = "
                            "seconds added on top of the source."),
                io.Combo.Input("duration_mode", options=["total", "new_only"],
                    default="total"),
            ] + _enhancer_inputs(SYSTEM_PROMPTS_CONTINUE),
            outputs=[
                Payload.Output(display_name="payload"),
                io.String.Output(display_name="final_prompt",
                    tooltip="Preview of the exact prompt the server will "
                            "encode (post-enhancer, post-substitution)."),
            ],
        )

    @classmethod
    def validate_inputs(cls, prompt, references):
        return _validate_prompt_and_refs(prompt, references)

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return _prepare_fingerprint(kwargs)

    @classmethod
    def execute(cls, video, prompt, references, duration_s, duration_mode,
                enhance_prompt, enhancer_model, enhancer_model_fallback,
                enhancer_reasoning, openrouter_api_key, enhancer_vision,
                vision_format, system_prompt_preset, enhancer_seed,
                system_prompt_override=None) -> io.NodeOutput:
        ref_list = refs.parse_references(references)
        unknown = refs.unknown_mentions(prompt, ref_list)
        if unknown:
            raise ValueError(_unknown_mentions_message(unknown, ref_list))

        source_duration = float(video.get_duration())
        t = continuation.timeline(
            duration_mode, api_payload.align_frames(round(duration_s * 24)),
            source_duration, api_payload.align_frames)
        logging.info("h3_tools: continuation timeline — source %.1fs, total "
                     "%.1fs", t["source_duration"], t["total_duration"])

        pbar = comfy.utils.ProgressBar(
            len(ref_list) + (2 if enhance_prompt else 0))
        payloads, durations = _decode_refs(ref_list, pbar)

        working_prompt = prompt
        if enhance_prompt:
            logging.info("h3_tools: re-encoding continuation video for the LLM")
            data_url, _full, sent_duration = media.encode_video_context(video)
            context_parts = [
                {"type": "text", "text": continuation.context_text(
                    t["source_duration"], sent_duration, t["total_duration"],
                    t["from_zero"])},
                {"type": "video_url", "video_url": {"url": data_url}},
            ]
            context_sha = hashlib.sha256(data_url.encode("ascii")).hexdigest()
            pbar.update(1)
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
            pbar.update(1)

        taken = set()
        uploads, refs_out = _stage_reference_uploads(ref_list, taken)
        vid_name, vid_path = _video_source_path(video, taken)
        uploads.append({"name": vid_name, "path": vid_path})
        body = api_payload.ref_body(prompt=working_prompt, references=refs_out,
                                    duration_s=duration_s,
                                    duration_mode=duration_mode)
        preview = refs.build_final_prompt(working_prompt, ref_list)
        payload = {"mode": body["mode"], "body": body, "uploads": uploads,
                   "preview_prompt": preview, "source_name": vid_name}
        return io.NodeOutput(payload, preview)


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


class H3ApiGenerateRefContinue(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3ApiGenerateRefContinue",
            display_name="MiniMax H3 API — Generate Reference Continue",
            category="api/minimax",
            description="Submits a prepared continuation payload to the Modal "
                        "API. Width/height follow the source video on the "
                        "server (scaled to megapixels).",
            inputs=[
                Payload.Input("payload"),
                io.Float.Input("megapixels", default=0.5, min=0.1, max=1.05,
                    step=0.05,
                    tooltip="Canvas area; the server caps at 768x1344."),
                io.Float.Input("context_seconds", default=2.0, min=0.2,
                    max=10.0, step=0.1,
                    tooltip="Trailing seconds of the source anchored by "
                            "AddGuide on the server."),
            ] + _connection_inputs(),
            outputs=[
                io.Video.Output(display_name="video"),
                io.String.Output(display_name="final_prompt"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, payload, megapixels, context_seconds, accounts_json,
                api_url, api_key, account, quality, seed,
                timeout_min) -> io.NodeOutput:
        if payload.get("mode") != "ref_extend":
            raise ValueError("this node takes a payload from 'Prepare "
                             "Reference Continue' (got mode %r)"
                             % payload.get("mode"))
        video, final_prompt, info = _run_remote(
            payload["body"], payload["uploads"], accounts_json=accounts_json,
            api_url=api_url, api_key=api_key, account=account, quality=quality,
            seed=seed, timeout_min=timeout_min,
            extra=dict(api_payload.extend_extra(context_seconds,
                                                payload["source_name"]),
                       megapixels=float(megapixels)))
        return io.NodeOutput(video, final_prompt or payload["preview_prompt"], info)


class H3ApiGenerateFlf(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3ApiGenerateFlf",
            display_name="MiniMax H3 API — Generate First/Last Frame",
            category="api/minimax",
            description="First/last frame (or pure t2v with neither) on the "
                        "Modal API. The prompt is sent verbatim — enhance it "
                        "upstream with your own nodes.",
            inputs=[
                io.String.Input("prompt", multiline=True, force_input=True,
                    tooltip="Ready prompt (enhance upstream with your own "
                            "nodes — this node sends it verbatim)."),
                io.Image.Input("first_frame", optional=True),
                io.Image.Input("last_frame", optional=True),
                io.Float.Input("duration_s", default=10.0, min=0.3, max=150.0,
                    step=0.1),
                io.Combo.Input("size_mode", options=["aspect", "source"],
                    default="aspect",
                    tooltip="'aspect' = aspect + megapixels below; 'source' = "
                            "width/height derived from first_frame on the "
                            "server."),
                io.Combo.Input("aspect",
                    options=["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"],
                    default="16:9"),
                io.Float.Input("megapixels", default=0.5, min=0.1, max=1.05,
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


class H3ApiGenerateFlfContinue(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3ApiGenerateFlfContinue",
            display_name="MiniMax H3 API — Generate First/Last Continue",
            category="api/minimax",
            description="Continues a video in the first/last-frame register on "
                        "the Modal API. No frame inputs — the source video "
                        "provides the pixels; sizing follows the source. The "
                        "prompt is sent verbatim.",
            inputs=[
                io.String.Input("prompt", multiline=True, force_input=True,
                    tooltip="Ready prompt (enhance upstream with your own "
                            "nodes — this node sends it verbatim)."),
                io.Video.Input("video",
                    tooltip="The footage being continued."),
                io.Float.Input("duration_s", default=10.0, min=0.3, max=150.0,
                    step=0.1,
                    tooltip="Final total length, context head included."),
                io.Float.Input("megapixels", default=0.6, min=0.1, max=1.05,
                    step=0.05,
                    tooltip="Canvas area; the server caps at 768x1344."),
                io.Float.Input("context_seconds", default=2.0, min=0.2,
                    max=10.0, step=0.1,
                    tooltip="Trailing seconds of the source anchored by "
                            "AddGuide on the server."),
            ] + _connection_inputs(),
            outputs=[
                io.Video.Output(display_name="video"),
                io.String.Output(display_name="final_prompt"),
                io.String.Output(display_name="info"),
            ],
        )

    @classmethod
    def execute(cls, prompt, video, duration_s, megapixels, context_seconds,
                accounts_json, api_url, api_key, account, quality, seed,
                timeout_min) -> io.NodeOutput:
        taken = set()
        vid_name, vid_path = _video_source_path(video, taken)
        body = {"mode": "flf_extend", "prompt": prompt,
                "duration_s": float(duration_s), "enhance": False}
        video_out, final_prompt, info = _run_remote(
            body, [{"name": vid_name, "path": vid_path}],
            accounts_json=accounts_json, api_url=api_url, api_key=api_key,
            account=account, quality=quality, seed=seed,
            timeout_min=timeout_min,
            extra=dict(api_payload.extend_extra(context_seconds, vid_name),
                       megapixels=float(megapixels)))
        return io.NodeOutput(video_out, final_prompt or prompt, info)


API_NODES = [H3ApiPrepareRef, H3ApiPrepareRefContinue, H3ApiGenerateRef,
             H3ApiGenerateRefContinue, H3ApiGenerateFlf,
             H3ApiGenerateFlfContinue]
