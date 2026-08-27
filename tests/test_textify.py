import json

from h3_tools.textify import stringify


class Unserialisable:
    def __repr__(self):
        return "<Unserialisable>"


def test_string_passes_through_untouched():
    assert stringify("um prompt\ncom quebra") == "um prompt\ncom quebra"


def test_none_renders_as_none():
    assert stringify(None) == "None"


def test_scalars():
    assert stringify(7) == "7"
    assert stringify(1.5) == "1.5"
    assert stringify(True) == "True"


def test_dict_is_indented_json():
    out = stringify({"b": 1, "a": [1, 2]})
    assert json.loads(out) == {"b": 1, "a": [1, 2]}
    assert "\n" in out  # indent=2, not a one-liner


def test_json_keeps_accents_readable():
    assert stringify({"k": "ação"}) == '{\n  "k": "ação"\n}'


def test_unserialisable_falls_back_to_repr():
    assert stringify(Unserialisable()) == "<Unserialisable>"
