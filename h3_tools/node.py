"""H3RefToVideoPro — the single node: board references + @mentions + enhancer.

Text and organization only: media decode goes through the native loader nodes
(media.py) and ALL encoding is delegated, untouched, to the native
MiniMaxH3ReferenceToVideo. This module is only imported inside ComfyUI.
"""

import hashlib
import json
import logging
import os

import folder_paths
import nodes
from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo

from . import enhancer, media, refs


def _unknown_mentions_message(unknown, ref_list):
    valid = ", ".join("@" + r.name for r in ref_list) or "(none — the board is empty)"
    return ("unknown reference%s %s in prompt; valid names: %s"
            % ("s" if len(unknown) > 1 else "",
               ", ".join("@" + u for u in unknown), valid))


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
                io.Boolean.Input("enhance_prompt", default=False,
                    tooltip="Rewrite the prompt with an LLM (OpenRouter) before encoding. "
                            "@name mentions are preserved."),
                io.String.Input("enhancer_model", default="google/gemini-3-flash-preview",
                    tooltip="OpenRouter model slug (free text)."),
                io.String.Input("openrouter_api_key", default="",
                    tooltip="Empty falls back to the OPENROUTER_API_KEY environment "
                            "variable. Never logged or cached by this pack — but "
                            "ComfyUI itself includes widget values in execution-error "
                            "payloads (/history), so prefer the env var on shared or "
                            "serverless hosts."),
                io.Boolean.Input("enhancer_vision", default=False,
                    tooltip="Send reference images (and one frame per video) to the LLM."),
                io.String.Input("system_prompt_override", multiline=True, default="",
                    tooltip="Non-empty replaces the built-in enhancer system prompt verbatim."),
                io.Int.Input("enhancer_seed", default=0, min=0, max=2**31 - 1,
                    tooltip="Part of the enhancer cache key only — bump to re-roll the LLM."),
            ],
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
                ref_image_size, enhance_prompt, enhancer_model, openrouter_api_key,
                enhancer_vision, system_prompt_override, enhancer_seed) -> io.NodeOutput:
        ref_list = refs.parse_references(references)
        unknown = refs.unknown_mentions(prompt, ref_list)
        if unknown:
            raise ValueError(_unknown_mentions_message(unknown, ref_list))

        # decode every reference in array order via the native loader nodes
        payloads = {}
        durations = {}
        for r in ref_list:
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

        working_prompt = prompt
        if enhance_prompt:
            working_prompt = cls._enhance(
                prompt, ref_list, payloads, durations, enhancer_model,
                openrouter_api_key, enhancer_vision, system_prompt_override,
                enhancer_seed)

        final_prompt = refs.build_final_prompt(working_prompt, ref_list)
        unmentioned = refs.unmentioned_refs(working_prompt, ref_list)
        if unmentioned:
            logging.info("h3_tools: references never mentioned in the prompt (still "
                         "sent to the model): %s",
                         ", ".join("@" + n for n in unmentioned))

        # native Autogrow dicts, keys paired by trailing index (soundtrack N <-> video N)
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

        out = MiniMaxH3ReferenceToVideo.execute(
            clip=clip, vae=vae, audio_vae=audio_vae, prompt=final_prompt,
            width=width, height=height, length=length, ref_image_size=ref_image_size,
            ref_images=groups["ref_images"] or None,
            ref_videos=groups["ref_videos"] or None,
            ref_video_audios=groups["ref_video_audios"] or None,
            ref_audios=groups["ref_audios"] or None)
        cond, latent = out.args
        return io.NodeOutput(cond, latent, final_prompt)

    @classmethod
    def _enhance(cls, prompt, ref_list, payloads, durations, model, api_key_widget,
                 vision, system_override, seed):
        api_key = api_key_widget or os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise ValueError(
                "enhance_prompt is on but no OpenRouter key was found: fill the "
                "openrouter_api_key widget or set the OPENROUTER_API_KEY "
                "environment variable")
        system_prompt = system_override if system_override.strip() \
            else enhancer.DEFAULT_SYSTEM_PROMPT

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
        key = enhancer.cache_key(prompt, ref_stats, model, system_prompt, vision, seed)
        cache_dir = os.path.join(folder_paths.get_user_directory(),
                                 "h3_tools", "enhancer_cache")
        cached = enhancer.cache_get(cache_dir, key)
        if cached is not None:
            logging.info("h3_tools: enhancer cache hit")
            return cached

        vision_parts = None
        if vision:
            vision_parts = []
            for r in ref_list:
                if r.type == "image":
                    url = media.image_data_url(payloads[r.name])
                elif r.type == "video":
                    url = media.image_data_url(
                        media.middle_frame(payloads[r.name]["frames"]))
                else:
                    continue  # audio is never sent to the LLM
                vision_parts.append({"type": "text", "text": "@%s (%s):" % (r.name, r.type)})
                vision_parts.append({"type": "image_url", "image_url": {"url": url}})

        manifest = enhancer.build_manifest(ref_list, durations)
        result = enhancer.enhance(prompt, api_key=api_key, model=model,
                                  system_prompt=system_prompt, manifest=manifest,
                                  vision_parts=vision_parts)
        clean, warnings = enhancer.sanitize_output(result, ref_list, prompt)
        for warning in warnings:
            logging.warning("h3_tools: %s", warning)
        enhancer.cache_put(cache_dir, key, clean, model)
        return clean
