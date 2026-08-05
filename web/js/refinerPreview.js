import { app } from "../../../scripts/app.js";

const style = document.createElement("style");
style.textContent = `
.mmh3-refiner-preview {
  width: 100%;
  min-width: 0;
  height: 180px;
  box-sizing: border-box;
  padding: 4px;
}
.mmh3-refiner-preview-text {
  width: 100%;
  height: 100%;
  box-sizing: border-box;
  resize: none;
  border: 1px solid var(--border-color, #3a3f4b);
  border-radius: 8px;
  background: var(--comfy-input-bg, #171a20);
  color: var(--input-text, #d5d8de);
  font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
  font-size: 12px;
  line-height: 1.45;
  padding: 8px;
  overflow: auto;
  white-space: pre-wrap;
  word-break: break-word;
}
`;
document.head.appendChild(style);

const REFINER_NODES = new Set([
  "MiniMaxH3ContextIRRefiner",
  "MiniMaxH3OpenAICompatibleRefiner",
]);

function setupPreview(node) {
  if (node.__mmh3RefinerPreview) {
    return;
  }

  const root = document.createElement("div");
  root.className = "mmh3-refiner-preview";
  const textarea = document.createElement("textarea");
  textarea.className = "mmh3-refiner-preview-text";
  textarea.readOnly = true;
  textarea.placeholder = "Refiner preview will appear here after execution.";
  root.appendChild(textarea);

  const widget = node.addDOMWidget(
    "mmh3_refiner_preview",
    "mmh3_refiner_preview",
    root,
    {
      getMinHeight: () => 180,
      getMaxHeight: () => 600,
      serialize: false,
    },
  );

  node.__mmh3RefinerPreview = { widget, textarea };
}

app.registerExtension({
  name: "ComfyUI-MiniMaxH3.RefinerPreview",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!REFINER_NODES.has(nodeData?.name)) {
      return;
    }

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      setupPreview(this);
      return result;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      const result = onExecuted?.apply(this, arguments);
      const preview = message?.text?.[0];
      if (preview && this.__mmh3RefinerPreview) {
        this.__mmh3RefinerPreview.textarea.value = preview;
        requestAnimationFrame(() => {
          this.setSize?.(this.size);
          app.graph.setDirtyCanvas(true, false);
        });
      }
      return result;
    };
  },
});
