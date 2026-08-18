import json

import pytest

from h3_tools import refs
from h3_tools.refs import RefError


def make(items):
    return refs.parse_references(json.dumps(items))


# ---- name derivation ----------------------------------------------------

def test_derive_name_basic():
    assert refs.derive_name("h3_refs/Garota Linda.PNG") == "garota_linda"


def test_derive_name_unicode_stripped():
    assert refs.derive_name("garota-α.png") == "garota"


def test_derive_name_digit_first_gets_prefix():
    assert refs.derive_name("1girl.png") == "m_1girl"


def test_derive_name_empty_stem_gets_prefix():
    assert refs.derive_name("!!!.png") == "m_"


def test_derive_name_multi_dot_keeps_full_stem():
    assert refs.derive_name("a.b.c.png") == "a_b_c"


def test_derive_name_truncates_to_64():
    assert len(refs.derive_name("x" * 200 + ".png")) == 64


# ---- parse_references ---------------------------------------------------

def test_parse_full_set():
    got = make([
        {"type": "image", "file": "h3_refs/garota.png"},
        {"name": "cidade", "type": "image", "file": "h3_refs/predio.png"},
        {"type": "video", "file": "h3_refs/danca.mp4", "use_soundtrack": True},
        {"type": "audio", "file": "h3_refs/voz.wav"},
    ])
    assert [r.name for r in got] == ["garota", "cidade", "danca", "voz"]
    assert [r.type for r in got] == ["image", "image", "video", "audio"]
    assert got[2].use_soundtrack is True


def test_parse_rejects_bad_json():
    with pytest.raises(RefError, match="not valid JSON"):
        refs.parse_references("[")


def test_parse_rejects_non_list():
    with pytest.raises(RefError, match="array"):
        refs.parse_references("{}")


def test_parse_rejects_bad_type():
    with pytest.raises(RefError, match=r"references\[0\].type"):
        make([{"type": "gif", "file": "a.gif"}])


def test_parse_rejects_missing_file():
    with pytest.raises(RefError, match=r"references\[0\].file"):
        make([{"type": "image"}])


def test_parse_rejects_duplicate_explicit_names():
    with pytest.raises(RefError, match='duplicate reference name "a"'):
        make([{"name": "a", "type": "image", "file": "a.png"},
              {"name": "a", "type": "image", "file": "b.png"}])


def test_parse_rejects_uppercase_explicit_name():
    with pytest.raises(RefError, match="invalid"):
        make([{"name": "Garota", "type": "image", "file": "a.png"}])


def test_parse_rejects_over_image_cap():
    items = [{"type": "image", "file": "i%d.png" % i} for i in range(10)]
    with pytest.raises(RefError, match="at most 9"):
        make(items)


def test_parse_rejects_soundtrack_on_image():
    with pytest.raises(RefError, match="only valid on videos"):
        make([{"type": "image", "file": "a.png", "use_soundtrack": True}])


def test_parse_ignores_unknown_keys():
    got = make([{"type": "image", "file": "a.png", "future_key": 42}])
    assert got[0].name == "a"


def test_derived_name_collision_gets_suffix():
    got = make([{"type": "image", "file": "h3_refs/a.png"},
                {"type": "image", "file": "other/a.jpg"}])
    assert [r.name for r in got] == ["a", "a_2"]


def test_derived_name_avoids_explicit_name():
    got = make([{"type": "image", "file": "b.png"},
                {"name": "b", "type": "image", "file": "c.png"}])
    assert [r.name for r in got] == ["b_2", "b"]


# ---- ordinal assignment -------------------------------------------------

def test_ordinals_mixed_set():
    got = make([
        {"type": "image", "file": "a.png"},
        {"type": "video", "file": "v1.mp4", "use_soundtrack": True},
        {"type": "audio", "file": "x.wav"},
        {"type": "image", "file": "b.png"},
        {"type": "video", "file": "v2.mp4", "use_soundtrack": True},
        {"type": "audio", "file": "y.wav"},
    ])
    by_name = {r.name: r for r in got}
    assert by_name["a"].tag == "<Picture 1>"
    assert by_name["b"].tag == "<Picture 2>"
    assert by_name["v1"].tag == "<Video 1>"
    assert by_name["v1"].soundtrack_tag == "<Audio 1>"
    assert by_name["v2"].tag == "<Video 2>"
    assert by_name["v2"].soundtrack_tag == "<Audio 2>"
    assert by_name["x"].tag == "<Audio 3>"
    assert by_name["y"].tag == "<Audio 4>"


def test_ordinals_video_without_soundtrack_consumes_no_audio():
    got = make([
        {"type": "video", "file": "v1.mp4"},
        {"type": "audio", "file": "x.wav"},
    ])
    assert got[0].tag == "<Video 1>"
    assert got[0].soundtrack_tag == ""
    assert got[1].tag == "<Audio 1>"


# ---- mentions -----------------------------------------------------------

def test_mention_not_matched_after_word_char():
    assert refs.find_mentions("mail@a and code@b") == []


def test_mention_matched_after_punctuation():
    assert refs.find_mentions("(@garota) e @predio!") == ["garota", "predio"]


def test_unknown_mentions_case_insensitive_dedup():
    got = make([{"type": "image", "file": "garota.png"}])
    assert refs.unknown_mentions("@Garota @nope @NOPE", got) == ["nope"]


def test_substitute_case_insensitive():
    got = make([{"type": "image", "file": "garota.png"}])
    assert refs.substitute_mentions("a @Garota b", got) == "a <Picture 1> b"


def test_substitute_leaves_unknown_and_literal_tags():
    got = make([{"type": "image", "file": "garota.png"}])
    text = "use <Picture 1> com @garota e @ghost"
    assert refs.substitute_mentions(text, got) == \
        "use <Picture 1> com <Picture 1> e @ghost"


# ---- provenance + final prompt -----------------------------------------

def test_provenance_lines():
    got = make([
        {"type": "video", "file": "v1.mp4", "use_soundtrack": True},
        {"type": "video", "file": "v2.mp4"},
        {"type": "video", "file": "v3.mp4", "use_soundtrack": True},
    ])
    assert refs.provenance_lines(got) == [
        "<Audio 1> is the synchronized audio track of <Video 1>.",
        "<Audio 2> is the synchronized audio track of <Video 3>.",
    ]


def test_build_final_prompt_with_provenance():
    got = make([{"type": "video", "file": "v1.mp4", "use_soundtrack": True}])
    out = refs.build_final_prompt("@v1 dançando", got)
    assert out == ("<Audio 1> is the synchronized audio track of <Video 1>.\n\n"
                   "<Video 1> dançando")


def test_build_final_prompt_without_provenance():
    got = make([{"type": "image", "file": "a.png"}])
    assert refs.build_final_prompt("@a parada", got) == "<Picture 1> parada"


# ---- LLM output sanitation ---------------------------------------------

def test_strip_unknown_mentions():
    got = make([{"type": "image", "file": "garota.png"}])
    out, removed = refs.strip_unknown_mentions("a @ghost b @garota", got)
    assert out == "a b @garota"
    assert removed == ["@ghost"]


def test_strip_unknown_keeps_clean_text_intact():
    got = make([{"type": "image", "file": "garota.png"}])
    out, removed = refs.strip_unknown_mentions("só a @garota aqui", got)
    assert out == "só a @garota aqui"
    assert removed == []


def test_unmentioned_refs():
    got = make([{"type": "image", "file": "a.png"},
                {"type": "image", "file": "b.png"}])
    assert refs.unmentioned_refs("olha a @a", got) == ["b"]
