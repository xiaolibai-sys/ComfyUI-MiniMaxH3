import { readFileSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

/* minimal DOM stub to execute flConstraint.js outside the browser */
function makeEl(tag) {
  const el = {
    tagName: tag.toUpperCase(),
    children: [],
    style: {},
    dataset: {},
    classList: {
      _set: new Set(),
      add(c) { this._set.add(c); },
      remove(c) { this._set.delete(c); },
      toggle(c, force) {
        const on = force === undefined ? !this._set.has(c) : force;
        on ? this._set.add(c) : this._set.delete(c);
      },
      contains(c) { return this._set.has(c); },
    },
    set className(v) {
      this._cls = v;
      this.classList._set = new Set(String(v).split(/\s+/).filter(Boolean));
    },
    get className() { return this._cls || ""; },
    append(...kids) { this.children.push(...kids); },
    appendChild(k) { this.children.push(k); return k; },
    replaceChildren(...kids) { this.children = [...kids]; },
    querySelectorAll() { return []; },
    querySelector() { return null; },
    addEventListener() {},
    removeEventListener() {},
    setAttribute() {},
    getBoundingClientRect() { return { width: 800, left: 0, top: 0 }; },
    remove() {},
    click() { this.onclick?.({ stopPropagation() {}, target: this }); },
    focus() {},
    setSelectionRange() {},
    clientWidth: 800,
    innerHTML: "",
    title: "",
    value: "",
  };
  return el;
}

global.document = {
  createElement: (tag) => makeEl(tag),
  createTextNode: (t) => ({ text: t }),
  head: { appendChild() {} },
  addEventListener() {},
};
global.window = { addEventListener() {}, removeEventListener() {} };
global.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
let registered = null;
const appStub = {
  graph: null,
  registerExtension(ext) { registered = ext; },
};
const apiStub = {
  apiURL: (u) => u,
  fetchApi: async () => ({ ok: true, json: async () => ({}) }),
  addEventListener() {},
  dispatchCustomEvent() {},
};

let source = readFileSync(new URL("../web/js/flConstraint.js", import.meta.url), "utf8");
source = source
  .replace('import { app } from "../../../scripts/app.js";', "const app = globalThis.__app;")
  .replace('import { api } from "../../../scripts/api.js";', "const api = globalThis.__api;");
globalThis.__app = appStub;
globalThis.__api = apiStub;

const tmpFile = join(tmpdir(), "_fl_ui_smoke_module.mjs");
writeFileSync(tmpFile, source);
await import("file://" + tmpFile.replace(/\\/g, "/"));

if (!registered) {
  throw new Error("extension was not registered");
}

const nodeType = function () {};
await registered.beforeRegisterNodeDef(nodeType, { name: "MiniMaxH3FLConstraint" }, appStub);

const node = {
  properties: {},
  id: 1,
  graph: null,
  widgets: [],
  addDOMWidget(name, type, element, options) {
    this.widgets.push({ name, element, options });
    return { computeSize: null, element };
  },
  setSize() {},
};
nodeType.prototype.onNodeCreated.call(node);

const root = node.widgets[0].element;
const [header, globalPanel, timeline, editor] = root.children;
console.log("header children:", header.children.length);
console.log("global panel children:", globalPanel.children.length);
console.log("timeline children:", timeline.children.length);
const [, ruler, stage] = timeline.children;
console.log("ruler ticks:", ruler.children.length);
console.log("stage children:", stage.children.length);
console.log("editor children:", editor.children.length);

/* simulate clicking "Add keyframe" */
const tlHead = timeline.children[0];
const segAdd = tlHead.children[2];
const addBtn = segAdd.children[2];
console.log("add button text:", addBtn.textContent);
addBtn.click();
console.log("after add -> stage children:", stage.children.length);
addBtn.click();
console.log("after 2nd add -> stage children:", stage.children.length);
console.log("fl_data:", node.properties.fl_data);
console.log("SMOKE OK");
