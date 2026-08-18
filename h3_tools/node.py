"""H3RefToVideoPro / H3RefToVideoContinuePro — board references + @mentions +
enhancer.

Text and organization only: media decode goes through the native loader nodes
(media.py) and ALL encoding is delegated, untouched, to the native
MiniMaxH3ReferenceToVideo. The Continue node adds continuation context for the
LLM only — pixel-level continuation stays with the native MiniMaxH3AddGuide the
user wires downstream. This module is only imported inside ComfyUI.
"""

import hashlib
import json
import logging
import os
import time

import comfy.utils
import folder_paths
import nodes
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo, align_frame_count
from comfy_api.latest import io

from . import continuation, enhancer, media, refs
from .system_prompts import SYSTEM_PROMPTS_CONTINUE


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
    """(label, image) part pairs for enhancer_vision; audio is never sent."""
    parts = []
    for r in ref_list:
        if r.type == "image":
            url = media.image_data_url(payloads[r.name])
        elif r.type == "video":
            url = media.image_data_url(media.middle_frame(payloads[r.name]["frames"]))
        else:
            continue
        parts.append({"type": "text", "text": "@%s (%s):" % (r.name, r.type)})
        parts.append({"type": "image_url", "image_url": {"url": url}})
    return parts


def _encode_native(clip, vae, audio_vae, final_prompt, width, height, length,
                   ref_image_size, ref_list, payloads, target_duration):
    """Assemble the native Autogrow dicts and delegate the encode."""
    # keys paired by trailing index (soundtrack N <-> video N)
    groups = {"ref_images": {}, "ref_videos": {}, "ref_video_audios": {}, "ref_audios": {}}
    for name, group, key in refs.autogrow_slots(ref_list):
        payload = payloads[name]
        if group == "ref_images":
            groups[group][key] = payload
        elif group == "ref_videos":
            groups[group][key] = payload["frames"]
        elif group == "ref_video_audios":
            groups[group][key] = payload["soundtrack"]
        else:
            groups[group][key] = payload["audio"]

    logging.info("h3_tools: encoding via native MiniMaxH3ReferenceToVideo "
                 "(%d refs, %.1fs target)", len(ref_list), target_duration)
    out = MiniMaxH3ReferenceToVideo.execute(
        clip=clip, vae=vae, audio_vae=audio_vae, prompt=final_prompt,
        width=width, height=height, length=length, ref_image_size=ref_image_size,
        ref_images=groups["ref_images"] or None,
        ref_videos=groups["ref_videos"] or None,
        ref_video_audios=groups["ref_video_audios"] or None,
        ref_audios=groups["ref_audios"] or None)
    return out.args


def _run_enhancer(prompt, ref_list, payloads, durations, *, model,
                  fallback_model, reasoning, api_key_widget, vision, preset,
                  system_override, seed, target_duration,
                  vision_format="default", system_prompts=None,
                  context_parts=None, context_sha=None, duration_mode=None,
                  video_sent=False):
    api_key = api_key_widget or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError(
            "enhance_prompt is on but no OpenRouter key was found: fill the "
            "openrouter_api_key widget or set the OPENROUTER_API_KEY "
            "environment variable")
    if system_override and system_override.strip():
        system_prompt = system_override
    else:
        system_prompt = (system_prompts or enhancer.SYSTEM_PROMPTS)[preset]

    models = [model]
    fallback_model = (fallback_model or "").strip()
    if fallback_model and fallback_model != model:
        models.append(fallback_model)
    if video_sent:
        supported = continuation.fetch_video_models()
        models, skipped = continuation.filter_models_for_video(models, supported)
        for m in skipped:
            logging.warning("h3_tools: model %s does not accept video input; "
                            "skipping it in the fallback chain", m)
        if not models:
            raise ValueError(
                "none of the configured models (%s) accepts video input — "
                "pick a video-capable model (see openrouter.ai/models?"
                "input_modalities=video) or connect image_last_frame instead "
                "of video" % ", ".join(skipped))

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


def _enhancer_inputs(presets):
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
            tooltip="Send reference images (and one frame per video) to the LLM."),
        io.Combo.Input("vision_format", options=["default", "cascade"],
            default="default",
            tooltip="How media reaches the LLM: 'default' = one message with "
                    "interleaved label+media parts; 'cascade' = one message "
                    "per media item. Part of the enhancer cache key."),
        io.Combo.Input("system_prompt_preset",
            options=list(presets.keys()), default="default",
            tooltip="Built-in enhancer system prompt: what kind of video the "
                    "LLM should write for (multishot, single take, ...)."),
        io.String.Input("system_prompt_override", optional=True, force_input=True,
            tooltip="Connect a STRING to replace the preset verbatim."),
        io.Int.Input("enhancer_seed", default=0, min=0, max=2**31 - 1,
            tooltip="Part of the enhancer cache key only — bump to re-roll the LLM."),
    ]


class H3RefToVideoPro(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3RefToVideoPro",
            display_name="MiniMax H3 Reference to Video (Pro)",
            category="conditioning/minimax",
            description="Reference-to-video conditioning with a built-in media board, "
                        "@name mentions (converted to the native <Picture i>/<Video k>/"
                        "<Audio j> tags) and an optional OpenRouter prompt enhancer. "
                        "Encoding is delegated to the native MiniMax H3 node.",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=False, default="",
                    tooltip="Prompt with @name mentions. Literal <Picture i>/<Video k>/"
                            "<Audio j> tags also work and pass through untouched."),
                io.String.Input("references", default="[]",
                    tooltip='JSON array of references, managed by the board UI. Headless: '
                            '[{"type":"image","file":"h3_refs/garota.png"}, ...] — '
                            'see SPEC.md §4.'),
                io.Int.Input("width", default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17,
                    tooltip="Frame count at 24 fps, snapped up to the model's 17k+5 grid "
                            "(124 = ~5s, trained range is ~124-362)"),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match",
                    tooltip="Reference image sizing, passed through to the native node. "
                            "'match' scales refs to the generation's pixel area; 'max' "
                            "keeps up to a 2048px short edge (better identity, slower)."),
            ] + _enhancer_inputs(enhancer.SYSTEM_PROMPTS),
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(),
                io.String.Output(display_name="final_prompt",
                                 tooltip="The exact prompt handed to the native node "
                                         "(post-enhancer, post-substitution)."),
            ],
        )

    @classmethod
    def validate_inputs(cls, prompt, references):
        # linked (non-literal) inputs can't be validated here; execute re-checks
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
                # e.g. a path-traversal file value; still a message, never a raise
                return ('reference file not found or outside the input '
                        'directory: "%s"' % r.file)
            return ('reference file not found: "%s" (resolved to %s)'
                    % (r.file, resolved))
        unknown = refs.unknown_mentions(prompt, ref_list)
        if unknown:
            return _unknown_mentions_message(unknown, ref_list)
        return True

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        # widget values + (mtime_ns, size) per referenced file, so re-uploading a
        # changed file under the same name invalidates the execution cache. The
        # API key contributes only bool(key) — its value is never hashed.
        # Linked inputs (models, video/image tensors) are excluded: the graph
        # cache already tracks their upstream changes.
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

    @classmethod
    def execute(cls, clip, vae, audio_vae, prompt, references, width, height, length,
                ref_image_size, enhance_prompt, enhancer_model, enhancer_model_fallback,
                enhancer_reasoning, openrouter_api_key, enhancer_vision, vision_format,
                system_prompt_preset, enhancer_seed,
                system_prompt_override=None) -> io.NodeOutput:
        ref_list = refs.parse_references(references)
        unknown = refs.unknown_mentions(prompt, ref_list)
        if unknown:
            raise ValueError(_unknown_mentions_message(unknown, ref_list))

        # progress: one tick per decoded ref (+ enhancer) + the native encode
        pbar = comfy.utils.ProgressBar(len(ref_list) + (1 if enhance_prompt else 0) + 1)
        payloads, durations = _decode_refs(ref_list, pbar)

        # the generation's real duration (after the 17k+5 frame snap) — the
        # enhancer needs it to place shots/beats at exact timestamps
        target_duration = align_frame_count(max(5, length)) / 24.0

        working_prompt = prompt
        if enhance_prompt:
            logging.info("h3_tools: enhancing prompt via OpenRouter (%s, preset %s)",
                         enhancer_model, system_prompt_preset)
            started = time.monotonic()
            working_prompt = _run_enhancer(
                prompt, ref_list, payloads, durations, model=enhancer_model,
                fallback_model=enhancer_model_fallback,
                reasoning=enhancer_reasoning, api_key_widget=openrouter_api_key,
                vision=enhancer_vision, preset=system_prompt_preset,
                system_override=system_prompt_override, seed=enhancer_seed,
                target_duration=target_duration, vision_format=vision_format)
            logging.info("h3_tools: enhancer done in %.1fs",
                         time.monotonic() - started)
            pbar.update(1)

        final_prompt = refs.build_final_prompt(working_prompt, ref_list)
        unmentioned = refs.unmentioned_refs(working_prompt, ref_list)
        if unmentioned:
            logging.info("h3_tools: references never mentioned in the prompt (still "
                         "sent to the model): %s",
                         ", ".join("@" + n for n in unmentioned))

        cond, latent = _encode_native(clip, vae, audio_vae, final_prompt, width,
                                      height, length, ref_image_size, ref_list,
                                      payloads, target_duration)
        pbar.update(1)
        return io.NodeOutput(cond, latent, final_prompt)


class H3RefToVideoContinuePro(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3RefToVideoContinuePro",
            display_name="MiniMax H3 Reference to Video Continue (Pro)",
            category="conditioning/minimax",
            description="The Pro node plus continuation context for the LLM "
                        "enhancer: the connected video (sent whole) or last "
                        "frame tells the LLM what it is continuing, and length "
                        "is interpreted per duration_mode. Conditioning is "
                        "identical to the Pro node — wire positive/latent into "
                        "the native MiniMaxH3AddGuide with the source frames "
                        "for the pixel-level continuation.",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.Video.Input("video", optional=True,
                    tooltip="The footage being continued — context for the LLM "
                            "only (sent whole, re-encoded small). Wire the "
                            "pixel continuation via MiniMaxH3AddGuide."),
                io.Image.Input("image_last_frame", optional=True,
                    tooltip="Fallback context when no video is connected: only "
                            "this final frame is sent to the LLM."),
                io.String.Input("prompt", multiline=True, dynamic_prompts=False, default="",
                    tooltip="Prompt with @name mentions. Literal <Picture i>/<Video k>/"
                            "<Audio j> tags also work and pass through untouched."),
                io.String.Input("references", default="[]",
                    tooltip='JSON array of references, managed by the board UI. Headless: '
                            '[{"type":"image","file":"h3_refs/garota.png"}, ...] — '
                            'see SPEC.md §4.'),
                io.Int.Input("width", default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17,
                    tooltip="Frame count at 24 fps, snapped up to the model's "
                            "17k+5 grid. Meaning depends on duration_mode."),
                io.Combo.Input("duration_mode", options=["total", "new_only"],
                    default="total",
                    tooltip="'total': length is the FINAL video length; the "
                            "continuation spans source end -> total. 'new_only': "
                            "length is the NEW part; the latent grows to source "
                            "+ length. With image_last_frame the source clock "
                            "is unknown, so length is always the new part."),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match",
                    tooltip="Reference image sizing, passed through to the native node. "
                            "'match' scales refs to the generation's pixel area; 'max' "
                            "keeps up to a 2048px short edge (better identity, slower)."),
            ] + _enhancer_inputs(SYSTEM_PROMPTS_CONTINUE),
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(),
                io.String.Output(display_name="final_prompt",
                                 tooltip="The exact prompt handed to the native node "
                                         "(post-enhancer, post-substitution)."),
            ],
        )

    validate_inputs = H3RefToVideoPro.validate_inputs
    fingerprint_inputs = H3RefToVideoPro.fingerprint_inputs

    @classmethod
    def execute(cls, clip, vae, audio_vae, prompt, references, width, height,
                length, duration_mode, ref_image_size, enhance_prompt,
                enhancer_model, enhancer_model_fallback, enhancer_reasoning,
                openrouter_api_key, enhancer_vision, vision_format,
                system_prompt_preset, enhancer_seed, video=None,
                image_last_frame=None, system_prompt_override=None) -> io.NodeOutput:
        ref_list = refs.parse_references(references)
        unknown = refs.unknown_mentions(prompt, ref_list)
        if unknown:
            raise ValueError(_unknown_mentions_message(unknown, ref_list))
        if video is None and image_last_frame is None:
            raise ValueError(
                "connect video or image_last_frame — the LLM needs to see what "
                "it is continuing (or use the plain MiniMax H3 Reference to "
                "Video (Pro) node)")
        if video is not None and image_last_frame is not None:
            logging.info("h3_tools: both video and image_last_frame are "
                         "connected; using the video")

        source_duration = float(video.get_duration()) if video is not None else None
        t = continuation.timeline(duration_mode, length, source_duration,
                                  align_frame_count)
        logging.info("h3_tools: continuation timeline — source %s, total %.1fs, "
                     "latent length %d",
                     "%.1fs" % t["source_duration"]
                     if t["source_duration"] is not None else "unknown (image)",
                     t["total_duration"], t["latent_length"])

        # progress: refs + (context encode + enhancer) + the native encode
        pbar = comfy.utils.ProgressBar(
            len(ref_list) + (2 if enhance_prompt else 0) + 1)
        payloads, durations = _decode_refs(ref_list, pbar)

        working_prompt = prompt
        if enhance_prompt:
            if video is not None:
                logging.info("h3_tools: re-encoding continuation video for the LLM")
                data_url, _full, sent_duration = media.encode_video_context(video)
                context_media = {"type": "video_url", "video_url": {"url": data_url}}
                video_sent = True
            else:
                data_url = media.image_data_url(image_last_frame)
                context_media = {"type": "image_url", "image_url": {"url": data_url}}
                sent_duration = None
                video_sent = False
            context_parts = [
                {"type": "text", "text": continuation.context_text(
                    t["source_duration"], sent_duration, t["total_duration"],
                    t["from_zero"])},
                context_media,
            ]
            context_sha = hashlib.sha256(data_url.encode("ascii")).hexdigest()
            pbar.update(1)

            logging.info("h3_tools: enhancing prompt via OpenRouter (%s, preset %s)",
                         enhancer_model, system_prompt_preset)
            started = time.monotonic()
            working_prompt = _run_enhancer(
                prompt, ref_list, payloads, durations, model=enhancer_model,
                fallback_model=enhancer_model_fallback,
                reasoning=enhancer_reasoning, api_key_widget=openrouter_api_key,
                vision=enhancer_vision, preset=system_prompt_preset,
                system_override=system_prompt_override, seed=enhancer_seed,
                target_duration=t["total_duration"], vision_format=vision_format,
                system_prompts=SYSTEM_PROMPTS_CONTINUE,
                context_parts=context_parts, context_sha=context_sha,
                duration_mode=duration_mode, video_sent=video_sent)
            logging.info("h3_tools: enhancer done in %.1fs",
                         time.monotonic() - started)
            pbar.update(1)

        final_prompt = refs.build_final_prompt(working_prompt, ref_list)
        unmentioned = refs.unmentioned_refs(working_prompt, ref_list)
        if unmentioned:
            logging.info("h3_tools: references never mentioned in the prompt (still "
                         "sent to the model): %s",
                         ", ".join("@" + n for n in unmentioned))

        cond, latent = _encode_native(clip, vae, audio_vae, final_prompt, width,
                                      height, t["latent_length"], ref_image_size,
                                      ref_list, payloads, t["total_duration"])
        pbar.update(1)
        return io.NodeOutput(cond, latent, final_prompt)
