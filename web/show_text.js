// h3_tools — "Show Text Brabo": the run's output text, kept in the workflow.
//
// Persistence needs no code here: "text" is a declared String input, so
// litegraph serialises it into widgets_values on save and restores it on load
// (LGraphNode.configure). All this extension does is push each run's result
// into that widget — the same thing core does for "Preview as Text", except
// core's widget is flagged serialize:false and therefore forgets.

import { app } from "../../scripts/app.js";

const NODE_ID = "ShowTextBrabo";
const WIDGET = "text";

app.registerExtension({
  name: "h3_tools.ShowTextBrabo",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_ID) return;

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const widget = this.widgets?.find((w) => w.name === WIDGET);
      if (!widget) return;
      const text = message?.text ?? "";
      widget.value = Array.isArray(text) ? text.join("\n\n") : String(text);
    };
  },
});
