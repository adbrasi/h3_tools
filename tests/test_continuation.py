import pytest

from h3_tools import continuation
from h3_tools.continuation import ContinuationError


def align(n):
    # the native 17k+5 grid, snapping up (mirror of align_frame_count)
    n = max(5, n)
    return ((n - 5 + 16) // 17) * 17 + 5


# ---- timeline -----------------------------------------------------------

def test_timeline_total_mode():
    t = continuation.timeline("total", 124, 2.0, align)
    assert t["latent_length"] == 124
    assert t["total_duration"] == pytest.approx(124 / 24.0)
    assert t["source_duration"] == 2.0
    assert t["from_zero"] is False


def test_timeline_total_mode_aligns_total_duration():
    # 130 snaps up to 141 frames; the LLM must see the real clock
    t = continuation.timeline("total", 130, 2.0, align)
    assert t["latent_length"] == 130
    assert t["total_duration"] == pytest.approx(141 / 24.0)


def test_timeline_total_mode_source_too_long():
    with pytest.raises(ContinuationError, match="not shorter"):
        continuation.timeline("total", 124, 6.0, align)  # total is ~5.17s


def test_timeline_new_only_mode():
    t = continuation.timeline("new_only", 124, 2.0, align)
    # 2.0s * 24 = 48 source frames + 124 new = 172, aligned to 175
    assert t["latent_length"] == 48 + 124
    assert t["total_duration"] == pytest.approx(align(172) / 24.0)
    assert t["from_zero"] is False


def test_timeline_new_only_rounds_source_frames():
    t = continuation.timeline("new_only", 124, 2.02, align)  # 48.48 -> 48
    assert t["latent_length"] == 48 + 124


def test_timeline_image_mode_ignores_duration_mode():
    for mode in ("total", "new_only"):
        t = continuation.timeline(mode, 124, None, align)
        assert t["latent_length"] == 124
        assert t["total_duration"] == pytest.approx(124 / 24.0)
        assert t["source_duration"] is None
        assert t["from_zero"] is True


# ---- context_text -------------------------------------------------------

def test_context_text_video_full():
    text = continuation.context_text(2.0, 2.0, 5.2, from_zero=False)
    assert "footage to continue (2.0s)" in text
    assert "from 2.0s to 5.2s total" in text


def test_context_text_video_truncated():
    text = continuation.context_text(45.0, 30.0, 50.0, from_zero=False)
    assert "last 30.0s" in text
    assert "full length 45.0s" in text
    assert "from 45.0s to 50.0s total" in text


def test_context_text_image():
    text = continuation.context_text(None, None, 5.2, from_zero=True)
    assert "final frame" in text
    assert "5.2" in text
    assert "00:00.000" in text


# ---- video model support ------------------------------------------------

def test_parse_video_models():
    payload = {"data": [{"id": "google/gemini-3-flash-preview"},
                        {"id": "Qwen/qwen3-vl"}]}
    models = continuation.parse_video_models(payload)
    assert models == {"google/gemini-3-flash-preview", "qwen/qwen3-vl"}


def test_parse_video_models_garbage():
    assert continuation.parse_video_models({"data": "nope"}) == set()
    assert continuation.parse_video_models({}) == set()


def test_filter_models_for_video():
    supported = {"google/gemini-3-flash-preview"}
    usable, skipped = continuation.filter_models_for_video(
        ["google/Gemini-3-Flash-Preview", "openai/gpt-5o"], supported)
    assert usable == ["google/Gemini-3-Flash-Preview"]
    assert skipped == ["openai/gpt-5o"]


def test_filter_models_without_preflight_keeps_all():
    models = ["a/b", "c/d"]
    usable, skipped = continuation.filter_models_for_video(models, None)
    assert usable == models and skipped == []


def test_fetch_video_models_caches_for_an_hour():
    calls = []

    class Resp:
        status_code = 200

        def json(self):
            return {"data": [{"id": "google/g"}]}

    def get(url, timeout=None):
        calls.append(url)
        return Resp()

    continuation._MODELS_CACHE.update(at=None, models=None)
    clock = [1000.0]
    now = lambda: clock[0]
    assert continuation.fetch_video_models(get=get, now=now) == {"google/g"}
    clock[0] += 300
    assert continuation.fetch_video_models(get=get, now=now) == {"google/g"}
    assert len(calls) == 1  # served from cache
    clock[0] += 3600
    continuation.fetch_video_models(get=get, now=now)
    assert len(calls) == 2  # TTL expired
    assert "input_modalities=video" in calls[0]


def test_fetch_video_models_network_failure_returns_none():
    def get(url, timeout=None):
        raise OSError("boom")

    continuation._MODELS_CACHE.update(at=None, models=None)
    assert continuation.fetch_video_models(get=get, now=lambda: 0.0) is None
    # a failure is not cached: the next call tries again
    assert continuation.fetch_video_models(get=get, now=lambda: 1.0) is None


# ---- runtime guards -----------------------------------------------------

def test_is_video_rejection():
    body = '{"error": {"message": "No endpoints found that support input video"}}'
    assert continuation.is_video_rejection(404, body) is True
    assert continuation.is_video_rejection(404, "plain not found") is False
    assert continuation.is_video_rejection(400, body) is False


def test_is_silent_video_drop_google_zero_tokens():
    data = {"provider": "Google AI Studio",
            "usage": {"prompt_tokens_details": {"video_tokens": 0}}}
    assert continuation.is_silent_video_drop(data) is True


def test_is_silent_video_drop_google_with_tokens():
    data = {"provider": "Google",
            "usage": {"prompt_tokens_details": {"video_tokens": 2580}}}
    assert continuation.is_silent_video_drop(data) is False


def test_is_silent_video_drop_non_google_reports_zero():
    # non-Google providers legitimately report 0 — never a drop
    data = {"provider": "Alibaba",
            "usage": {"prompt_tokens_details": {"video_tokens": 0}}}
    assert continuation.is_silent_video_drop(data) is False


def test_is_silent_video_drop_missing_usage():
    assert continuation.is_silent_video_drop({"provider": "Google"}) is False
    assert continuation.is_silent_video_drop({}) is False
