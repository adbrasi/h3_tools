"""Enhancer orchestration shared by the Pro nodes and the API client nodes.

Moved verbatim from node.py so it can load on a frontend ComfyUI that lacks
comfy_extras.nodes_minimax_h3 (node.py imports the native encoder at module
top). No behavior change.
"""

import logging
import os

import folder_paths
from comfy_api.latest import io

from . import continuation, enhancer, media, system_prompts


def _unknown_mentions_message(unknown, ref_list):
    valid = ", ".join("@" + r.name for r in ref_list) or "(none — the board is empty)"
    return ("unknown reference%s %s in prompt; valid names: %s"
            % ("s" if len(unknown) > 1 else "",
               ", ".join("@" + u for u in unknown), valid))


def _decode_refs(ref_list, pbar):
    """Decode every reference in array order via the native loader nodes."""
    payloads = {}
    durations = {}
    for r in ref_list:
        logging.info("h3_tools: decoding @%s (%s: %s)", r.name, r.type, r.file)
        if r.type == "image":
            payloads[r.name] = media.load_image_ref(r.file)
        elif r.type == "video":
            payload = media.load_video_ref(r.file, r.name)
            if r.use_soundtrack:
                # resolve now: a missing audio track must fail before any
                # (paid) enhancer call, not after
                payload["soundtrack"] = media.soundtrack_of(payload, r.name)
            payloads[r.name] = payload
            durations[r.name] = payload["duration"]
        else:
            payload = media.load_audio_ref(r.file)
            payloads[r.name] = payload
            durations[r.name] = payload["duration"]
        pbar.update(1)
    return payloads, durations


def _reference_vision_parts(ref_list, payloads):
    """(label, media) part pairs for enhancer_vision; audio is never sent.

    Reference videos go as VIDEO, not as a still: a camera path, a
    choreography and a timing — the three things a reference video exists
    to carry — are invisible in a single frame, and an enhancer that cannot
    see them can only write "the camera motion is referenced from @clip".
    """
    parts = []
    for r in ref_list:
        if r.type == "image":
            part = {"type": "image_url",
                    "image_url": {"url": media.image_data_url(payloads[r.name])}}
        elif r.type == "video":
            part = {"type": "video_url",
                    "video_url": {"url": media.encode_video_ref(payloads[r.name],
                                                                r.name)}}
        else:
            continue
        parts.append({"type": "text", "text": "@%s (%s):" % (r.name, r.type)})
        parts.append(part)
    return parts


def _run_enhancer(prompt, ref_list, payloads, durations, *, model,
                  fallback_model, reasoning, api_key_widget, vision, preset,
                  system_override, seed, target_duration,
                  vision_format="default", continue_job=False,
                  context_parts=None, context_sha=None, duration_mode=None,
                  video_sent=False):
    api_key = api_key_widget or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError(
            "enhance_prompt is on but no OpenRouter key was found: fill the "
            "openrouter_api_key widget or set the OPENROUTER_API_KEY "
            "environment variable")
    # a reference video is only worth prompting for if the LLM can see it:
    # "seen" licenses describing the clip, "unseen" limits the rewrite to what
    # the draft says about it
    n_ref_videos = sum(1 for r in ref_list if r.type == "video")
    refs_video_sent = bool(vision and n_ref_videos)
    video_mode = ("none" if not n_ref_videos
                  else "seen" if refs_video_sent else "unseen")
    if system_override and system_override.strip():
        system_prompt = system_override
    else:
        system_prompt = system_prompts.system_prompt(
            preset, continuation=continue_job, video=video_mode)
    if video_mode != "none":
        logging.info("h3_tools: %d reference video(s), system prompt video "
                     "mode %r", n_ref_videos, video_mode)

    models = [model]
    fallback_model = (fallback_model or "").strip()
    if fallback_model and fallback_model != model:
        models.append(fallback_model)
    context_video_sent = video_sent
    video_sent = video_sent or refs_video_sent
    if video_sent:
        supported = continuation.fetch_video_models()
        models, skipped = continuation.filter_models_for_video(models, supported)
        for m in skipped:
            logging.warning("h3_tools: model %s does not accept video input; "
                            "skipping it in the fallback chain", m)
        if not models:
            ways = []
            if refs_video_sent:
                ways.append("turn enhancer_vision off (the reference videos "
                            "then reach the LLM as text only)")
            if context_video_sent:
                ways.append("connect image_last_frame instead of video")
            raise ValueError(
                "none of the configured models (%s) accepts video input — "
                "pick a video-capable model (see openrouter.ai/models?"
                "input_modalities=video), or %s"
                % (", ".join(skipped), " or ".join(ways)))

    ref_stats = []
    for r in ref_list:
        try:
            st = os.stat(folder_paths.get_annotated_filepath(r.file))
            mtime_ns, size = st.st_mtime_ns, st.st_size
        except (OSError, ValueError):
            mtime_ns, size = -1, -1
        # use_soundtrack changes the manifest the LLM sees, so it must
        # change the cache key too
        ref_stats.append({"name": r.name, "type": r.type, "file": r.file,
                          "use_soundtrack": r.use_soundtrack,
                          "mtime_ns": mtime_ns, "size": size})
    keys = {m: enhancer.cache_key(prompt, ref_stats, m, system_prompt, vision,
                                  seed, duration=target_duration,
                                  reasoning=reasoning,
                                  duration_mode=duration_mode,
                                  vision_format=vision_format,
                                  context_sha=context_sha)
            for m in models}
    cache_dir = os.path.join(folder_paths.get_user_directory(),
                             "h3_tools", "enhancer_cache")
    for m in models:
        cached = enhancer.cache_get(cache_dir, keys[m])
        if cached is not None:
            logging.info("h3_tools: enhancer cache hit (%s)", m)
            return cached

    vision_parts = _reference_vision_parts(ref_list, payloads) if vision else None
    manifest = enhancer.build_manifest(ref_list, durations)
    used_model, result = enhancer.enhance_with_fallback(
        prompt, models=models, api_key=api_key, system_prompt=system_prompt,
        manifest=manifest, target_duration=target_duration,
        vision_parts=vision_parts, context_parts=context_parts,
        vision_format=vision_format, video_sent=video_sent,
        reasoning_effort=reasoning)
    clean, warnings = enhancer.sanitize_output(result, ref_list, prompt)
    for warning in warnings:
        logging.warning("h3_tools: %s", warning)
    enhancer.cache_put(cache_dir, keys[used_model], clean, used_model)
    return clean


def _enhancer_inputs():
    """The enhancer widget block, shared by both nodes."""
    return [
        io.Boolean.Input("enhance_prompt", default=False,
            tooltip="Rewrite the prompt with an LLM (OpenRouter) before encoding. "
                    "@name mentions are preserved."),
        io.String.Input("enhancer_model", default="google/gemini-3-flash-preview",
            tooltip="OpenRouter model slug (free text)."),
        io.String.Input("enhancer_model_fallback", default="",
            tooltip="Optional second model slug, tried when the main model "
                    "fails for good (timeout, provider error, unparseable "
                    "output). Empty disables the fallback."),
        io.Combo.Input("enhancer_reasoning",
            options=["none", "low", "medium", "high", "xhigh"], default="none",
            tooltip="Reasoning effort sent to OpenRouter. 'none' disables "
                    "thinking (fastest — reasoning tokens are generated "
                    "serially and dominate wall-clock). Higher = better "
                    "structure, much slower and pricier."),
        io.String.Input("openrouter_api_key", default="",
            tooltip="Empty falls back to the OPENROUTER_API_KEY environment "
                    "variable. Never logged or cached by this pack — but "
                    "ComfyUI itself includes widget values in execution-error "
                    "payloads (/history), so prefer the env var on shared or "
                    "serverless hosts."),
        io.Boolean.Input("enhancer_vision", default=False,
            tooltip="Send the references to the LLM: images as pictures, reference "
                    "videos as VIDEO. Required for a reference video to be used for "
                    "its camera path, motion or timing — the model must accept video "
                    "input (e.g. gemini-3-flash-preview)."),
        io.Combo.Input("system_prompt_preset",
            options=list(system_prompts.PRESETS), default="default",
            tooltip="Built-in enhancer system prompt: what kind of video the "
                    "LLM should write for (multishot, single take, ...)."),
        io.String.Input("system_prompt_override", optional=True, force_input=True,
            tooltip="Connect a STRING to replace the preset verbatim."),
        io.Int.Input("enhancer_seed", default=0, min=0, max=2**31 - 1,
            tooltip="Part of the enhancer cache key only — bump to re-roll the LLM."),
        # appended LAST on purpose: workflow JSON stores widget values by
        # position, so new widgets must never enter the middle of the list
        io.Combo.Input("vision_format", options=["default", "cascade"],
            default="default",
            tooltip="How media reaches the LLM: 'default' = one message with "
                    "interleaved label+media parts; 'cascade' = one message "
                    "per media item. Part of the enhancer cache key."),
    ]
