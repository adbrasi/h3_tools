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
