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


def test_load_config_nothing_raises(monkeypatch):
    monkeypatch.delenv("H3_MODAL_ACCOUNTS_JSON", raising=False)
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
