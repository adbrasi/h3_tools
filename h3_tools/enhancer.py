"""OpenRouter prompt enhancer: request building, parsing, sanitation, disk cache.

Stdlib-only at import time; `requests` is imported lazily inside enhance() so
the parsing/cache logic stays testable without it. Vision parts are built by
the caller (media/node) — this module never touches torch or PIL.

The API key must never appear in logs, error messages, or cache files; every
error path passes through _redact().
"""

import hashlib
import json
import logging
import os
import re
import time

from . import refs
from .system_prompts import DEFAULT_SYSTEM_PROMPT, SYSTEM_PROMPTS  # noqa: F401

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
CACHE_MAX_ENTRIES = 500


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
    """Sanitation (SPEC §5.4): repair near-miss @tokens (LLM typos), strip the
    rest, warn on drops — user mistakes fail earlier, LLM mistakes never kill
    the job."""
    clean, repairs = refs.repair_mentions(text, refs_list)
    warnings = ['enhancer output misspelled %s; corrected to "@%s"' % (typed, fixed)
                for typed, fixed in dict.fromkeys(repairs)]
    clean, removed = refs.strip_unknown_mentions(clean, refs_list)
    warnings += ["enhancer output mentioned unknown reference %s; stripped" % tok
                 for tok in dict.fromkeys(removed)]
    known = {r.name for r in refs_list}
    wanted = {m.lower() for m in refs.find_mentions(original_prompt)} & known
    kept = {m.lower() for m in refs.find_mentions(clean)}
    for name in sorted(wanted - kept):
        warnings.append('enhancer output dropped "@%s"; the reference will reach '
                        "the model unmentioned" % name)
    return clean, warnings


def cache_key(prompt, ref_stats, model, system, vision, seed, duration=None,
              reasoning=None):
    # duration and reasoning are part of the LLM's input/behavior, so they are
    # part of the key; width/height stay out on purpose
    payload = {"prompt": prompt, "refs": ref_stats, "model": model,
               "system": system, "vision": bool(vision), "seed": seed,
               "duration": duration, "reasoning": reasoning}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def cache_get(cache_dir, key):
    try:
        with open(os.path.join(cache_dir, key + ".json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        value = data.get("prompt_final")
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


def enhance(prompt, *, api_key, model, system_prompt, manifest,
            target_duration=None, vision_parts=None, reasoning_effort="low",
            timeout=60, post=None, sleep=None):
    """Call OpenRouter and return prompt_final. 3 total attempts:
    network / 429 / 5xx retry with a short Retry-After-aware backoff; a parse
    failure retries with a "JSON only" nudge (kept for later attempts); other
    4xx fail immediately. A timeout also fails immediately — a model that
    blew the read budget will blow it again, and the user is waiting.
    Never falls back silently to the raw prompt.
    """
    if post is None:
        import requests
        post = requests.post
    if sleep is None:
        sleep = time.sleep

    def backoff(attempt, reason, resp=None):
        if attempt >= 2:
            return  # that was the final attempt; the error is about to raise
        delay = 1.0 * (attempt + 1)
        headers = getattr(resp, "headers", None) or {}
        try:
            delay = min(30.0, float(headers.get("Retry-After") or delay))
        except (TypeError, ValueError):
            pass
        logging.warning("h3_tools: enhancer attempt %d/3 failed (%s); retrying in %.1fs",
                        attempt + 1, reason, delay)
        sleep(delay)

    user_text = "Draft prompt:\n%s\n\nReferences:\n%s" % (prompt, manifest)
    if target_duration:
        user_text = ("Target video duration: %.1f seconds (24 fps).\n\n"
                     % target_duration) + user_text
    nudged = False
    last_error = "no attempts made"
    for attempt in range(3):
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
            # dropped upstream by models without reasoning support
            "reasoning": {"effort": reasoning_effort},
        }
        try:
            resp = post(OPENROUTER_URL,
                        headers={"Authorization": "Bearer %s" % api_key,
                                 "Content-Type": "application/json"},
                        json=body, timeout=(10, timeout))
        except Exception as e:
            last_error = _redact(str(e), api_key)
            if "timeout" in type(e).__name__.lower():
                raise EnhancerError(
                    "enhancer request timed out (%ds read limit) with model %s: %s"
                    % (timeout, model, last_error)) from e
            backoff(attempt, last_error)
            continue
        status = getattr(resp, "status_code", 0)
        if status == 429 or status >= 500:
            last_error = "HTTP %d: %s" % (status, _excerpt(resp, api_key))
            backoff(attempt, last_error, resp)
            continue
        if status != 200:
            raise EnhancerError("OpenRouter request failed (HTTP %d): %s"
                                % (status, _excerpt(resp, api_key)))
        try:
            content = resp.json()["choices"][0]["message"]["content"]
        except Exception:
            last_error = "malformed OpenRouter response body: %s" % _excerpt(resp, api_key)
            backoff(attempt, last_error, resp)
            continue
        try:
            return parse_response(content)
        except EnhancerError as e:
            last_error = str(e)
            if attempt < 2:
                logging.warning("h3_tools: enhancer attempt %d/3 returned unparseable "
                                "output (%s); retrying with a JSON-only nudge",
                                attempt + 1, last_error)
            nudged = True
            continue
    raise EnhancerError("prompt enhancer failed after 3 attempts: %s" % last_error)


def enhance_with_fallback(prompt, *, models, **kwargs):
    """Try each model in order with the full enhance() policy.

    Returns (model_used, prompt_final); re-raises the last EnhancerError when
    every model failed.
    """
    last_error = None
    for i, model in enumerate(models):
        try:
            return model, enhance(prompt, model=model, **kwargs)
        except EnhancerError as e:
            last_error = e
            if i + 1 < len(models):
                logging.warning("h3_tools: enhancer model %s failed (%s); falling "
                                "back to %s", model, str(e)[:200], models[i + 1])
    raise last_error
