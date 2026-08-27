"""Value -> display text. Stdlib only, so the rules stay unit-testable.

Mirrors the conversion core's "Preview as Text" (comfy_extras/nodes_preview_any)
applies, so a value looks the same in either node.
"""

import json


def stringify(value):
    """Render any node output as the text a preview widget should show."""
    if isinstance(value, str):
        return value
    if value is None:
        return "None"
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        try:
            return str(value)
        except Exception:
            return "source exists, but could not be serialized."
