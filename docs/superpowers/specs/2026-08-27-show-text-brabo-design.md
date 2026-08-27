# Show Text Brabo — design

**Goal.** A node that shows a run's output as text and keeps that text inside
the saved workflow, the way pythongosssss' *Show Text* does.

## Why the built-in node is not enough

`comfy_extras/nodes_preview_any.py` (*Preview as Text*) returns
`{"ui": {"text": (value,)}}`, and the frontend renders it with
`addTextPreviewWidgets`, a `TEXT_PREVIEW` widget created with
`options.serialize = false` **and** `widget.serialize = false`
(frontend 1.45.21). The two flags are separate concerns, documented in
`src/utils/executionUtil.ts`:

- `widget.options.serialize` — include the widget in the **API prompt**.
- `widget.serialize` — include the widget in **`widgets_values`**, i.e. in the
  saved workflow (`LGraphNode.configure` / `LGraphNode.serialize`).

The core preview turns both off, so its text never reaches the workflow JSON.
That is the entire gap.

## Design

`ShowTextBrabo` (display name *Show Text Brabo*, category `utils`):

| Piece | Decision |
| --- | --- |
| `source` input | `io.AnyType` (`*`) — strings pass through, everything else is JSON-dumped, matching *Preview as Text* |
| `text` input | declared `io.String` widget, `multiline=True` — this is the display, and the reason persistence works |
| `text` output | `STRING` passthrough, so the node can sit mid-chain |
| node kind | `is_output_node=True`, returns `ui.PreviewText(value)` |
| frontend | `web/show_text.js`: on `onExecuted`, copy `message.text` into the `text` widget |

Persistence is therefore native: a declared widget serialises to
`widgets_values` on save and is restored on load with no code of ours. No
`onConfigure` hook, and no `extra_pnginfo` mutation (pythongosssss needs that
only to inject text into a PNG saved in the same run; h3_tools emits video).

`execute` ignores the incoming `text`. It is only ever the previous run's
display, and letting it influence the output would emit a stale value.

## The divergence: the widget is editable

The original design called for a read-only widget, like both reference nodes.
That is not reachable on frontend 1.45.21: **read-only and persistent are
mutually exclusive** for a text widget.

- The only read-only text component is `WidgetTextPreview` (`TEXT_PREVIEW`),
  which is hardcoded `serialize: false` and keeps its value in a Vue store
  rather than on the widget — it cannot persist.
- The ordinary multiline string widget persists, but honours no `read_only`
  option (`options.read_only` is read by the number/slider widgets only).

Rendering both — a read-only preview for display plus a hidden serialising
widget for storage — would be two channels carrying one value, so it was
rejected. Persistence is the requirement; read-only was polish. The widget
stays editable, and the README states that edits are display-only.

## Files

- `h3_tools/textify.py` — `stringify(value)`, stdlib-only so it is unit-testable
  without ComfyUI on the path (`tests/conftest.py` rule)
- `h3_tools/show_text.py` — the node class + `SHOW_TEXT_NODES`
- `web/show_text.js` — `onExecuted` → widget value
- `tests/test_textify.py` — conversion rules
- `__init__.py` — registration, outside the MiniMax `ImportError` guard so the
  node loads on any ComfyUI
