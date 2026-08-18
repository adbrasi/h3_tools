"""OpenRouter prompt enhancer: request building, parsing, sanitation, disk cache.

Stdlib-only at import time; `requests` is imported lazily inside enhance() so
the parsing/cache logic stays testable without it. Vision parts are built by
the caller (media/node) — this module never touches torch or PIL.

The API key must never appear in logs, error messages, or cache files; every
error path passes through _redact().
"""

import hashlib
import json
import os
import re
import time

from . import refs

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
CACHE_MAX_ENTRIES = 500

DEFAULT_SYSTEM_PROMPT = """You are an expert prompt writer for MiniMax H3, a reference-to-video model that
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
{"prompt_final": "<the rewritten prompt>"}"""


class EnhancerError(RuntimeError):
    """User-facing enhancer failure; the message is shown as-is."""


def _redact(text, api_key):
    if api_key:
        text = text.replace(api_key, "***")
    return text


def _excerpt(resp, api_key):
    return _redact(str(getattr(resp, "text", ""))[:200], api_key)


def _first_json_object(text):
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find("{", start + 1)
    return None


def parse_response(text):
    """Model output -> prompt_final. Tolerates fences and surrounding prose."""
    candidate = _first_json_object(text or "")
    if candidate is None:
        raise EnhancerError("enhancer response contains no JSON object")
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise EnhancerError("enhancer response is not valid JSON: %s" % e) from e
    value = data.get("prompt_final")
    if not isinstance(value, str) or not value.strip():
        raise EnhancerError('enhancer response is missing a non-empty "prompt_final"')
    return value.strip()


def build_manifest(refs_list, durations=None):
    """One line per reference for the LLM; durations keyed by ref name (seconds)."""
    durations = durations or {}
    lines = []
    for r in refs_list:
        base = r.file.replace("\\", "/").rsplit("/", 1)[-1]
        d = durations.get(r.name)
        kind = r.type if d is None else "%s %.1fs" % (r.type, d)
        lines.append('- @%s (%s, file "%s")' % (r.name, kind, base))
        if r.use_soundtrack:
            lines.append("- @%s's soundtrack (audio)" % r.name)
    return "\n".join(lines)


def sanitize_output(text, refs_list, original_prompt):
    """Asymmetric sanitation (SPEC §5.4): strip invented @tokens, warn on drops."""
    clean, removed = refs.strip_unknown_mentions(text, refs_list)
    warnings = ["enhancer output mentioned unknown reference %s; stripped" % tok
                for tok in removed]
    known = {r.name for r in refs_list}
    wanted = {m.lower() for m in refs.find_mentions(original_prompt)} & known
    kept = {m.lower() for m in refs.find_mentions(clean)}
    for name in sorted(wanted - kept):
        warnings.append('enhancer output dropped "@%s"; the reference will reach '
                        "the model unmentioned" % name)
    return clean, warnings


def cache_key(prompt, ref_stats, model, system, vision, seed):
    payload = {"prompt": prompt, "refs": ref_stats, "model": model,
               "system": system, "vision": bool(vision), "seed": seed}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def cache_get(cache_dir, key):
    try:
        with open(os.path.join(cache_dir, key + ".json"), "r", encoding="utf-8") as f:
            value = json.load(f).get("prompt_final")
        return value if isinstance(value, str) and value else None
    except (OSError, ValueError):
        return None


def cache_put(cache_dir, key, prompt_final, model, max_entries=CACHE_MAX_ENTRIES):
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, key + ".json"), "w", encoding="utf-8") as f:
        json.dump({"prompt_final": prompt_final, "model": model,
                   "created": time.time()}, f, ensure_ascii=False)
    try:
        entries = [p for p in os.listdir(cache_dir) if p.endswith(".json")]
        entries.sort(key=lambda p: os.path.getmtime(os.path.join(cache_dir, p)))
        for oldest in entries[:max(0, len(entries) - max_entries)]:
            os.remove(os.path.join(cache_dir, oldest))
    except OSError:
        pass  # pruning is best-effort; a failed prune must not kill the job


def enhance(prompt, refs, *, api_key, model, system_prompt, manifest,
            vision_parts=None, timeout=120, post=None):
    """Call OpenRouter and return prompt_final. 3 total attempts:
    network / 429 / 5xx retry as-is; the first parse failure retries with a
    "JSON only" nudge; other 4xx fail immediately. Never falls back silently.
    """
    if post is None:
        import requests
        post = requests.post

    user_text = "Draft prompt:\n%s\n\nReferences:\n%s" % (prompt, manifest)
    nudged = False
    last_error = "no attempts made"
    for _ in range(3):
        system_text = system_prompt
        if nudged:
            system_text += "\nReturn ONLY the JSON object, nothing else."
        if vision_parts:
            user_content = [{"type": "text", "text": user_text}] + list(vision_parts)
        else:
            user_content = user_text
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system_text},
                         {"role": "user", "content": user_content}],
            "response_format": {"type": "json_object"},
            "temperature": 0.8,
        }
        try:
            resp = post(OPENROUTER_URL,
                        headers={"Authorization": "Bearer %s" % api_key,
                                 "Content-Type": "application/json"},
                        json=body, timeout=timeout)
        except Exception as e:
            last_error = _redact(str(e), api_key)
            continue
        status = getattr(resp, "status_code", 0)
        if status == 429 or status >= 500:
            last_error = "HTTP %d: %s" % (status, _excerpt(resp, api_key))
            continue
        if status != 200:
            raise EnhancerError("OpenRouter request failed (HTTP %d): %s"
                                % (status, _excerpt(resp, api_key)))
        try:
            content = resp.json()["choices"][0]["message"]["content"]
        except Exception:
            last_error = "malformed OpenRouter response body: %s" % _excerpt(resp, api_key)
            continue
        try:
            return parse_response(content)
        except EnhancerError as e:
            last_error = str(e)
            nudged = True
            continue
    raise EnhancerError("prompt enhancer failed after 3 attempts: %s" % last_error)
