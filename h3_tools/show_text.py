"""Show Text Brabo — a text preview that survives saving and reopening a workflow.

Core's "Preview as Text" renders into a TEXT_PREVIEW widget flagged
`serialize: false`, so its text is gone the moment the workflow is reloaded.
This node instead declares an ordinary multiline String widget ("text") as one
of its inputs: litegraph writes every serialising widget into `widgets_values`
on save and restores it on load, so the last run's output travels inside the
workflow JSON with no extra machinery.

`execute` ignores the incoming `text` — it is only ever the previous run's
display — and publishes the new value through the ui message that
`web/show_text.js` pushes back into the widget.
"""

from comfy_api.latest import io, ui

from .textify import stringify


class ShowTextBrabo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ShowTextBrabo",
            display_name="Show Text Brabo",
            category="utils",
            description="Show any value as text and keep it saved in the "
                        "workflow. Unlike the built-in Preview as Text, the "
                        "text is still there after reopening the workflow.",
            inputs=[
                io.AnyType.Input("source",
                    tooltip="Anything. Strings pass through; everything else "
                            "is JSON-dumped."),
                io.String.Input("text", multiline=True, default="",
                    dynamic_prompts=False,
                    tooltip="The last run's output, kept in the saved "
                            "workflow. Overwritten on every execution."),
            ],
            outputs=[
                io.String.Output(display_name="text",
                    tooltip="The same text, so the node can sit mid-chain."),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, source, text="") -> io.NodeOutput:
        value = stringify(source)
        return io.NodeOutput(value, ui=ui.PreviewText(value))


SHOW_TEXT_NODES = [ShowTextBrabo]
