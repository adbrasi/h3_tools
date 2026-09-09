import json
import logging
import os

import pytest

from h3_tools import enhancer, refs
from h3_tools.enhancer import EnhancerError


def make_refs(items):
    return refs.parse_references(json.dumps(items))


REFS = make_refs([
    {"type": "image", "file": "h3_refs/garota.png"},
    {"type": "video", "file": "h3_refs/danca.mp4", "use_soundtrack": True},
    {"type": "audio", "file": "h3_refs/voz.wav"},
])


# ---- parse_response -----------------------------------------------------

def test_parse_clean_json():
    assert enhancer.parse_response('{"prompt_final": "ok"}') == "ok"


def test_parse_fenced_json():
    text = '```json\n{"prompt_final": "ok"}\n```'
    assert enhancer.parse_response(text) == "ok"


def test_parse_prose_wrapped():
    text = 'Sure! Here you go: {"prompt_final": "ok"} hope it helps'
    assert enhancer.parse_response(text) == "ok"


def test_parse_nested_braces_in_string():
    text = '{"prompt_final": "a {weird} \\"quoted\\" prompt"}'
    assert enhancer.parse_response(text) == 'a {weird} "quoted" prompt'


def test_parse_garbage_raises():
    with pytest.raises(EnhancerError):
        enhancer.parse_response("no json here at all")


def test_parse_empty_prompt_final_raises():
    with pytest.raises(EnhancerError):
        enhancer.parse_response('{"prompt_final": "  "}')


def test_parse_missing_key_raises():
    with pytest.raises(EnhancerError):
        enhancer.parse_response('{"other": "x"}')


# ---- sanitize_output ----------------------------------------------------

def test_sanitize_strips_invented_mention():
    out, warnings = enhancer.sanitize_output(
        "a @ghost scene with @garota", REFS, "@garota dança")
    assert out == "a scene with @garota"
    assert any("@ghost" in w for w in warnings)


def test_sanitize_warns_dropped_user_mention():
    out, warnings = enhancer.sanitize_output(
        "a scene with @garota", REFS, "@garota dança com @voz")
    assert out == "a scene with @garota"
    assert any("voz" in w for w in warnings)


def test_sanitize_repairs_llm_typo():
    out, warnings = enhancer.sanitize_output(
        "a scene with @garotaa dancing", REFS, "@garota dança")
    assert out == "a scene with @garota dancing"
    assert any("misspelled" in w and "@garotaa" in w for w in warnings)
    assert not any("dropped" in w for w in warnings)


def test_sanitize_clean_output_no_warnings():
    out, warnings = enhancer.sanitize_output(
        "@garota dança ao som de @voz", REFS, "@garota e @voz")
    assert out == "@garota dança ao som de @voz"
    assert warnings == []


# ---- manifest -----------------------------------------------------------

def test_build_manifest():
    manifest = enhancer.build_manifest(REFS, {"danca": 3.2, "voz": 5.0})
    assert manifest.splitlines() == [
        '- @garota (image, file "garota.png")',
        '- @danca (video 3.2s, file "danca.mp4")',
        "- @danca's soundtrack (audio)",
        '- @voz (audio 5.0s, file "voz.wav")',
    ]


def test_build_manifest_without_durations():
    manifest = enhancer.build_manifest(REFS, {})
    assert '- @danca (video, file "danca.mp4")' in manifest.splitlines()


# ---- cache --------------------------------------------------------------

def test_cache_key_stable_and_seed_sensitive():
    stats = [{"name": "a", "type": "image", "file": "a.png",
              "mtime_ns": 1, "size": 2}]
    k1 = enhancer.cache_key("p", stats, "m", "s", False, 0)
    k2 = enhancer.cache_key("p", list(stats), "m", "s", False, 0)
    k3 = enhancer.cache_key("p", stats, "m", "s", False, 1)
    assert k1 == k2
    assert k1 != k3
    assert len(k1) == 64


def test_cache_key_sensitive_to_every_field():
    stats = [{"name": "a", "type": "image", "file": "a.png",
              "mtime_ns": 1, "size": 2}]
    base = enhancer.cache_key("p", stats, "m", "s", False, 0)
    assert enhancer.cache_key("q", stats, "m", "s", False, 0) != base
    assert enhancer.cache_key("p", stats, "m2", "s", False, 0) != base
    assert enhancer.cache_key("p", stats, "m", "s2", False, 0) != base
    assert enhancer.cache_key("p", stats, "m", "s", True, 0) != base
    changed = [dict(stats[0], use_soundtrack=True)]
    assert enhancer.cache_key("p", changed, "m", "s", False, 0) != base
    assert enhancer.cache_key("p", stats, "m", "s", False, 0, duration=5.2) != base


def test_cache_roundtrip(tmp_path):
    enhancer.cache_put(str(tmp_path), "k" * 64, "the prompt", "model/x")
    assert enhancer.cache_get(str(tmp_path), "k" * 64) == "the prompt"


def test_cache_miss_and_corrupt(tmp_path):
    assert enhancer.cache_get(str(tmp_path), "absent") is None
    (tmp_path / "bad.json").write_text("{not json")
    assert enhancer.cache_get(str(tmp_path), "bad") is None
    (tmp_path / "arr.json").write_text("[]")  # valid JSON, wrong shape
    assert enhancer.cache_get(str(tmp_path), "arr") is None


def test_cache_prune_keeps_newest(tmp_path):
    for i in range(501):
        p = tmp_path / ("old%03d.json" % i)
        p.write_text('{"prompt_final": "x"}')
        os.utime(p, (1000 + i, 1000 + i))
    enhancer.cache_put(str(tmp_path), "newest", "fresh", "m")
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 500
    assert enhancer.cache_get(str(tmp_path), "newest") == "fresh"


# ---- enhance (HTTP loop) ------------------------------------------------

OK = "a plausible rewritten prompt, long enough to pass the substance check"
OK_JSON = '{"prompt_final": "%s"}' % OK


class FakeResp:
    def __init__(self, status, content=None, text="", headers=None, extra=None):
        self.status_code = status
        self._content = content
        self._extra = extra or {}
        self.text = text or (content or "")
        self.headers = headers or {}

    def json(self):
        data = {"choices": [{"message": {"content": self._content}}]}
        data.update(self._extra)
        return data


def call(post, **kw):
    args = dict(prompt="@garota dança", api_key="sk-secret-123",
                model="m/x", system_prompt="SYS", manifest="- @garota (image)",
                vision_parts=None, post=post, sleep=lambda s: None)
    args.update(kw)
    return enhancer.enhance(**args)


def test_enhance_success_first_try():
    bodies = []
    timeouts = []

    def post(url, headers=None, json=None, timeout=None):
        bodies.append(json)
        timeouts.append(timeout)
        return FakeResp(200, OK_JSON)

    assert call(post) == OK
    assert bodies[0]["response_format"] == {"type": "json_object"}
    assert "temperature" not in bodies[0]  # model default; reasoning models reject it
    assert "max_tokens" not in bodies[0]  # owner decision: never cap the answer
    assert bodies[0]["reasoning"] == {"enabled": False}  # default: no thinking
    assert bodies[0]["provider"] == {"sort": "throughput"}
    assert bodies[0]["usage"] == {"include": True}
    assert bodies[0]["messages"][0]["role"] == "system"
    assert timeouts[0] == (10, 60)  # fast connect failure, 60s read budget


def test_enhance_retries_5xx_then_succeeds():
    seq = [FakeResp(500, text="err"), FakeResp(502, text="err"),
           FakeResp(200, OK_JSON)]
    assert call(lambda *a, **k: seq.pop(0)) == OK


def test_enhance_three_5xx_fails_with_excerpt():
    with pytest.raises(EnhancerError, match="boom-detail"):
        call(lambda *a, **k: FakeResp(500, text="boom-detail"))


def test_enhance_backoff_honors_retry_after():
    sleeps = []
    seq = [FakeResp(429, text="slow down", headers={"Retry-After": "3"}),
           FakeResp(200, OK_JSON)]
    assert call(lambda *a, **k: seq.pop(0), sleep=sleeps.append) == OK
    assert sleeps == [3.0]


def test_enhance_timeout_fails_fast():
    class ReadTimeout(Exception):
        pass

    sleeps = []
    calls = []

    def post(*a, **k):
        calls.append(1)
        raise ReadTimeout("read timed out")

    with pytest.raises(EnhancerError, match="timed out"):
        call(post, sleep=sleeps.append)
    assert len(calls) == 1 and sleeps == []  # no retry, no backoff


def test_enhance_logs_each_retry(caplog):
    seq = [FakeResp(500, text="err-500"), FakeResp(200, OK_JSON)]
    with caplog.at_level(logging.WARNING):
        assert call(lambda *a, **k: seq.pop(0)) == OK
    assert any("attempt 1/3" in r.message and "retrying" in r.message
               for r in caplog.records)


def test_enhance_with_fallback_uses_second_model():
    calls = []

    def post(url, headers=None, json=None, timeout=None):
        calls.append(json["model"])
        if json["model"] == "bad/one":
            return FakeResp(400, text="no such model")
        return FakeResp(200, OK_JSON)

    used, out = enhancer.enhance_with_fallback(
        "@garota dança", models=["bad/one", "good/two"], api_key="k",
        system_prompt="SYS", manifest="- @garota (image)", vision_parts=None,
        post=post, sleep=lambda s: None)
    assert (used, out) == ("good/two", OK)
    assert calls == ["bad/one", "good/two"]


def test_enhance_with_fallback_raises_when_all_fail():
    def post(*a, **k):
        return FakeResp(400, text="nope")

    with pytest.raises(EnhancerError, match="nope"):
        enhancer.enhance_with_fallback(
            "p", models=["a/b", "c/d"], api_key="k", system_prompt="S",
            manifest="m", vision_parts=None, post=post, sleep=lambda s: None)


def test_enhance_no_sleep_after_final_attempt():
    sleeps = []
    with pytest.raises(EnhancerError):
        call(lambda *a, **k: FakeResp(500, text="err"), sleep=sleeps.append)
    assert sleeps == [1.0, 2.0]  # backoff between attempts only


def test_enhance_401_fails_immediately():
    calls = []

    def post(*a, **k):
        calls.append(1)
        return FakeResp(401, text="bad key")

    with pytest.raises(EnhancerError, match="401"):
        call(post)
    assert len(calls) == 1


def test_enhance_parse_failure_retries_with_nudge():
    bodies = []
    seq = [FakeResp(200, "not json"), FakeResp(200, OK_JSON)]

    def post(url, headers=None, json=None, timeout=None):
        bodies.append(json)
        return FakeResp(*[]) if False else seq.pop(0)

    assert call(post) == OK
    assert "Return ONLY the JSON object" not in bodies[0]["messages"][0]["content"]
    assert "Return ONLY the JSON object" in bodies[1]["messages"][0]["content"]


def test_enhance_network_error_never_leaks_key():
    def post(*a, **k):
        raise RuntimeError("connect fail sk-secret-123")

    with pytest.raises(EnhancerError) as exc:
        call(post)
    assert "sk-secret-123" not in str(exc.value)


def test_enhance_sends_chosen_reasoning_effort():
    bodies = []

    def post(url, headers=None, json=None, timeout=None):
        bodies.append(json)
        return FakeResp(200, OK_JSON)

    call(post, reasoning_effort="xhigh")
    assert bodies[0]["reasoning"] == {"effort": "xhigh"}


def test_enhance_strips_rejected_reasoning_field():
    bodies = []
    seq = [FakeResp(400, text='{"error": "reasoning is not supported"}'),
           FakeResp(200, OK_JSON)]

    def post(url, headers=None, json=None, timeout=None):
        bodies.append(json)
        return seq.pop(0)

    assert call(post) == OK
    assert "reasoning" in bodies[0]
    assert "reasoning" not in bodies[1]


def test_enhance_deadline_stops_the_loop():
    import time as _time
    calls = []

    def post(*a, **k):
        calls.append(1)
        return FakeResp(500, text="err")

    with pytest.raises(EnhancerError, match="time budget"):
        call(post, deadline=_time.monotonic() - 1)
    assert calls == []  # expired before the first attempt


def test_cache_key_sensitive_to_reasoning():
    stats = [{"name": "a", "type": "image", "file": "a.png",
              "mtime_ns": 1, "size": 2}]
    base = enhancer.cache_key("p", stats, "m", "s", False, 0, reasoning="low")
    assert enhancer.cache_key("p", stats, "m", "s", False, 0,
                              reasoning="high") != base


def test_enhance_sends_target_duration():
    bodies = []

    def post(url, headers=None, json=None, timeout=None):
        bodies.append(json)
        return FakeResp(200, OK_JSON)

    call(post, target_duration=5.2)
    user = bodies[0]["messages"][1]["content"]
    assert user.startswith("Target video duration: 5.2 seconds")


def test_enhance_sends_vision_parts():
    bodies = []

    def post(url, headers=None, json=None, timeout=None):
        bodies.append(json)
        return FakeResp(200, OK_JSON)

    parts = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,x"}}]
    call(post, vision_parts=parts)
    user = bodies[0]["messages"][1]
    assert isinstance(user["content"], list)
    assert user["content"][0]["type"] == "text"
    assert user["content"][-1] == parts[0]


# ---- build_user_messages (vision_format) --------------------------------

IMG = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,x"}}
VID = {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,y"}}


def test_build_user_messages_plain_string_without_parts():
    msgs = enhancer.build_user_messages("draft")
    assert msgs == [{"role": "user", "content": "draft"}]


def test_build_user_messages_default_orders_context_text_vision():
    ctx = [{"type": "text", "text": "footage:"}, VID]
    vis = [{"type": "text", "text": "@a (image):"}, IMG]
    msgs = enhancer.build_user_messages("draft", context_parts=ctx,
                                        vision_parts=vis)
    assert len(msgs) == 1
    content = msgs[0]["content"]
    assert content[0]["text"] == "footage:"
    assert content[1] == VID
    assert content[2] == {"type": "text", "text": "draft"}
    assert content[3]["text"] == "@a (image):"
    assert content[4] == IMG


def test_build_user_messages_cascade_one_message_per_media():
    ctx = [{"type": "text", "text": "footage:"}, VID]
    vis = [{"type": "text", "text": "@a (image):"}, IMG,
           {"type": "text", "text": "@b (image):"}, IMG]
    msgs = enhancer.build_user_messages("draft", context_parts=ctx,
                                        vision_parts=vis,
                                        vision_format="cascade")
    assert len(msgs) == 4
    assert msgs[0]["content"] == ctx
    assert msgs[1]["content"] == vis[0:2]
    assert msgs[2]["content"] == vis[2:4]
    assert msgs[3] == {"role": "user", "content": "draft"}


def test_build_user_messages_cascade_without_media_is_plain():
    msgs = enhancer.build_user_messages("draft", vision_format="cascade")
    assert msgs == [{"role": "user", "content": "draft"}]


# ---- video guards in enhance() ------------------------------------------

def test_enhance_video_rejection_fails_with_guidance():
    calls = []

    def post(url, headers=None, json=None, timeout=None):
        calls.append(json)
        return FakeResp(404, text='{"error": {"message": "No endpoints found '
                                  'that support input video"}}')

    ctx = [{"type": "text", "text": "footage:"}, VID]
    with pytest.raises(EnhancerError) as e:
        call(post, context_parts=ctx, video_sent=True)
    assert "image_last_frame" in str(e.value)
    assert "input_modalities=video" in str(e.value)
    assert len(calls) == 1  # instant rejection, no retries


def test_enhance_plain_404_without_video_keeps_generic_error():
    def post(url, headers=None, json=None, timeout=None):
        return FakeResp(404, text="not found")

    with pytest.raises(EnhancerError) as e:
        call(post)
    assert "HTTP 404" in str(e.value)


def test_enhance_silent_video_drop_is_an_error():
    def post(url, headers=None, json=None, timeout=None):
        return FakeResp(200, OK_JSON,
                        extra={"provider": "Google AI Studio",
                               "usage": {"prompt_tokens_details":
                                         {"video_tokens": 0}}})

    ctx = [{"type": "text", "text": "footage:"}, VID]
    with pytest.raises(EnhancerError) as e:
        call(post, context_parts=ctx, video_sent=True)
    assert "dropped the video" in str(e.value)


def test_enhance_video_tokens_billed_is_fine():
    def post(url, headers=None, json=None, timeout=None):
        return FakeResp(200, OK_JSON,
                        extra={"provider": "Google",
                               "usage": {"prompt_tokens_details":
                                         {"video_tokens": 2580}}})

    ctx = [{"type": "text", "text": "footage:"}, VID]
    assert call(post, context_parts=ctx, video_sent=True) == OK


def test_enhance_cascade_sends_separate_user_messages():
    bodies = []

    def post(url, headers=None, json=None, timeout=None):
        bodies.append(json)
        return FakeResp(200, OK_JSON)

    vis = [{"type": "text", "text": "@a (image):"}, IMG]
    call(post, vision_parts=vis, vision_format="cascade")
    messages = bodies[0]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["content"] == vis
    assert isinstance(messages[2]["content"], str)


# ---- cache_key new fields -----------------------------------------------

def test_cache_key_sensitive_to_continue_fields():
    base = dict(prompt="p", ref_stats=[], model="m", system="s",
                vision=False, seed=0)
    k0 = enhancer.cache_key(**base)
    assert enhancer.cache_key(**base, duration_mode="total") != k0
    assert enhancer.cache_key(**base, vision_format="cascade") != k0
    assert enhancer.cache_key(**base, context_sha="abc") != k0
    assert enhancer.cache_key(**base) == k0


# ---- reorder_sections ---------------------------------------------------

SHUFFLED = """subject_definitions:
<Subject 1> is the woman.
summary:
[video continuation] She kisses her.
retention_analysis:
<Subject 1>: fully_preserved - kept.
overall_soundscape:
Forest sounds.
non_diegetic_music:
N/A
detailed_description:
The target video is static.
[Shot 1] She enters."""


def test_reorder_sections_restores_official_order():
    fixed = enhancer.reorder_sections(SHUFFLED)
    order = [fixed.index(l + ":") for l in enhancer._SECTION_ORDER]
    assert order == sorted(order)
    assert "[Shot 1] She enters." in fixed


def test_reorder_sections_already_ordered_untouched():
    text = "subject_definitions:\na\nsummary:\nb\ndetailed_description:\nc"
    assert enhancer.reorder_sections(text) is text


def test_reorder_sections_prose_untouched():
    assert enhancer.reorder_sections("just a prompt") == "just a prompt"


def test_sanitize_reorders_and_warns():
    clean, warnings = enhancer.sanitize_output(SHUFFLED, [], "draft")
    assert clean.rstrip().endswith("[Shot 1] She enters.") is False
    assert any("out of order" in w for w in warnings)
    order = [clean.index(l + ":") for l in enhancer._SECTION_ORDER]
    assert order == sorted(order)


# ---- streaming ----------------------------------------------------------

def sse(obj):
    return "data: " + json.dumps(obj)


class FakeStreamResp:
    status_code = 200
    headers = {"Content-Type": "text/event-stream"}
    text = ""

    def __init__(self, lines):
        self._lines = lines

    def iter_lines(self, decode_unicode=True):
        return iter(self._lines)


@pytest.mark.parametrize("encoding", [None, "ISO-8859-1"])
def test_stream_preserves_utf8_without_charset(encoding):
    import requests

    text = "Eu volto amanhã. Espere por mim. 日本語"
    chunk = {"choices": [{"delta": {"content": text}}]}
    response = requests.Response()
    response.headers["Content-Type"] = "text/event-stream"
    response.encoding = encoding
    response._content = ("data: " + json.dumps(chunk, ensure_ascii=False)
                         + "\n\ndata: [DONE]\n\n").encode("utf-8")
    response._content_consumed = True
    content, _ = enhancer._drain_stream(response)
    assert content == text


def test_enhance_streams_and_assembles_content():
    bodies = []
    lines = [
        sse({"choices": [{"delta": {"content": '{"prompt_final": "'}}]}),
        ": keep-alive comment",
        sse({"choices": [{"delta": {"content": OK + '"}'}}]}),
        sse({"provider": "Google", "usage": {"completion_tokens": 7}}),
        "data: [DONE]",
    ]

    def post(url, headers=None, json=None, timeout=None):
        bodies.append(json)
        return FakeStreamResp(lines)

    assert call(post) == OK
    assert bodies[0]["stream"] is True


def test_enhance_stream_error_chunk_retries():
    seq = [FakeStreamResp([sse({"error": {"message": "provider blew up"}})]),
           FakeResp(200, OK_JSON)]
    assert call(lambda *a, **k: seq.pop(0)) == OK


def test_enhance_stream_silent_video_drop_guard():
    lines = [
        sse({"choices": [{"delta": {"content": '{"prompt_final": "x"}'}}]}),
        sse({"provider": "Google",
             "usage": {"prompt_tokens_details": {"video_tokens": 0}}}),
        "data: [DONE]",
    ]
    ctx = [{"type": "text", "text": "footage:"}, VID]
    with pytest.raises(EnhancerError, match="dropped the video"):
        call(lambda *a, **k: FakeStreamResp(lines),
             context_parts=ctx, video_sent=True)


# ---- substance check ----------------------------------------------------

def test_enhance_tiny_answer_retries_then_fallback():
    # a reasoning model burning its budget can emit a valid-but-empty answer
    # like {"prompt_final": ".}"}; that must never reach the encoder
    seq = [FakeResp(200, '{"prompt_final": ".}"}'),
           FakeResp(200, '{"prompt_final": ".}"}'),
           FakeResp(200, '{"prompt_final": ".}"}'),
           FakeResp(200, OK_JSON)]

    def post(url, headers=None, json=None, timeout=None):
        return seq.pop(0)

    used, out = enhancer.enhance_with_fallback(
        "@garota dança", models=["bad/reasoner", "good/model"],
        api_key="sk-secret-123", system_prompt="SYS", manifest="- @garota",
        post=post, sleep=lambda s: None)
    assert (used, out) == ("good/model", OK)


def test_enhance_tiny_answer_alone_fails_clearly():
    with pytest.raises(EnhancerError, match="too short"):
        call(lambda *a, **k: FakeResp(200, '{"prompt_final": ".}"}'))


def test_enhance_warns_when_disabled_reasoning_is_ignored(caplog):
    def post(url, headers=None, json=None, timeout=None):
        return FakeResp(200, OK_JSON,
                        extra={"usage": {"completion_tokens": 4462,
                                         "completion_tokens_details":
                                         {"reasoning_tokens": 4454}}})

    with caplog.at_level(logging.WARNING):
        assert call(post) == OK
    assert any("ignored disabled reasoning" in r.message for r in caplog.records)
