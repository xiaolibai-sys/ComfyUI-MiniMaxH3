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
  removeEventListener() {},
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

const tmpFile = join(tmpdir(), "_fl_ui_repro_module.mjs");
writeFileSync(tmpFile, source);
await import("file://" + tmpFile.replace(/\\/g, "/"));

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

const root = node.widgets.find((w) => w.name === "fl_constraint_ui").element;
const [header, globalPanel, timeline, editor] = root.children;
const [tlHead, ruler, stage] = timeline.children;
const segAdd = tlHead.children[2];
const addBtn = segAdd.children[2];
const data = () => JSON.parse(node.properties.fl_data || "{}");

const fail = (msg) => { console.error("FAIL:", msg); process.exitCode = 1; };
const check = (cond, msg) => {
  console.log(`${cond ? "ok" : "FAIL"} - ${msg}`);
  if (!cond) process.exitCode = 1;
};
const findAll = (rootEl, pred, out = []) => {
  for (const child of rootEl?.children || []) {
    if (pred(child)) out.push(child);
    findAll(child, pred, out);
  }
  return out;
};

/* 1. add two segments -> 3 keyframes */
addBtn.click();
addBtn.click();
check(data().keyframes.length === 3, `after 2 adds: 3 keyframes (got ${data().keyframes.length})`);
check(Math.abs(data().duration - 5) < 1e-6, `duration 5 (got ${data().duration})`);

/* 2. delete every keyframe via the editor's Delete button */
for (let i = 0; i < 3; i++) {
  const marker = stage.children.find((c) => c.classList.contains("mmh3-fl-kf"));
  marker.click(); /* select */
  const removeBtn = findAll(editor, (c) => c.textContent === "Delete period")[0];
  if (!removeBtn) { fail("no Delete button after selecting keyframe"); break; }
  removeBtn.click();
  if (i < 2) {
    /* selection should move to a neighbor, keeping deletion fluid */
    const stillThere = findAll(editor, (c) => c.textContent === "Delete period")[0];
    check(!!stillThere, `after delete ${i + 1}: neighbor keyframe auto-selected`);
  }
}
check(data().keyframes.length === 0, `after deleting all: 0 keyframes (got ${data().keyframes.length})`);
console.log("persisted after wipe:", node.properties.fl_data);

/* 3. add again after full wipe */
addBtn.click();
const after = data();
check(after.keyframes.length === 2, `re-add after wipe: 2 keyframes (got ${after.keyframes.length})`);
check(after.keyframes[0]?.time === 0, `re-add starts at 0 (got ${after.keyframes[0]?.time})`);
check(Math.abs(after.keyframes[1]?.time - 2.5) < 1e-6, `re-add end at 2.5 (got ${after.keyframes[1]?.time})`);
const ids = after.keyframes.map((k) => k.id);
check(new Set(ids).size === ids.length, `ids unique: ${ids}`);
check(Math.abs(after.duration - 2.5) < 1e-6, `duration 2.5 (got ${after.duration})`);

/* 4. duration semantics: editing Duration grows the segment and shifts later keyframes */
node.__h3FLState.setData(JSON.stringify({
  duration: 5,
  fps: 24,
  keyframes: [
    { id: 1, time: 0, image: null, prompt: "" },
    { id: 2, time: 2.5, image: null, prompt: "" },
    { id: 3, time: 5, image: null, prompt: "" },
  ],
}));
stage.children.find((c) => c.classList.contains("mmh3-fl-kf")).click(); /* first kf */
const durInput = findAll(editor, (c) => c.tagName === "INPUT")[0];
check(!!durInput, "duration input shown for non-last keyframe");
check(Math.abs(Number(durInput.value) - 2.5) < 1e-6,
  `duration shows this segment's length 2.5 (got ${durInput.value})`);
durInput.value = "4";
durInput.onchange?.();
const shifted = data();
check(Math.abs(shifted.keyframes[1].time - 4) < 1e-6,
  `next keyframe moved to 4 (got ${shifted.keyframes[1]?.time})`);
check(Math.abs(shifted.keyframes[2].time - 6.5) < 1e-6,
  `later keyframe shifted to 6.5, keeping its own 2.5s segment (got ${shifted.keyframes[2]?.time})`);
check(shifted.duration >= 6.5 - 1e-6,
  `persisted duration ${shifted.duration} covers max kf time`);

/* the last keyframe anchors the chain end: no duration field */
stage.children.filter((c) => c.classList.contains("mmh3-fl-kf")).pop().click();
check(findAll(editor, (c) => c.tagName === "INPUT").length === 0,
  "last keyframe has no duration input");

/* 5. selected keyframe with image: preview clickable, clearable, marker re-click picks */
node.__h3FLState.setData(JSON.stringify({
  duration: 2.5,
  fps: 24,
  keyframes: [
    { id: 1, time: 0, image: { name: "a.png" }, prompt: "" },
    { id: 2, time: 2.5, image: null, prompt: "" },
  ],
}));
const firstMarker = stage.children.find((c) => c.classList.contains("mmh3-fl-kf"));
firstMarker.click();
const preview = editor.children[0];
const info = editor.children[1];
const thumbBtn = findAll(preview, (c) => c.classList.contains("mmh3-fl-ed-thumbbtn"))[0];
check(!!thumbBtn && typeof thumbBtn.onclick === "function",
  "editor thumbnail is clickable to re-pick image");
/* image actions live in the preview column, next to the image */
const clearBtn = findAll(preview, (c) => c.tagName === "BUTTON" && c.textContent === "Remove")[0];
check(!!clearBtn && clearBtn.classList.contains("danger"),
  "Remove image button sits under the preview image (danger)");
const replaceBtn = findAll(preview, (c) => c.tagName === "BUTTON" && c.textContent === "Replace")[0];
check(replaceBtn?.classList.contains("primary"), "Replace button sits under the preview image (primary)");
/* timeline deletion lives with the keyframe identity in the info header */
const head = info.children[0];
const deleteBtn = findAll(head, (c) => c.textContent === "Delete period")[0];
check(deleteBtn?.classList.contains("danger"), "Delete period button sits in the info header (danger)");
check(!findAll(preview, (c) => c.textContent === "Delete period").length
  && !findAll(info, (c) => c.tagName === "BUTTON" && c.textContent === "Remove").length,
  "image removal and keyframe deletion are separated into their own areas");

/* second click on the already-selected marker opens the picker */
const fileInput = findAll(root, (c) => c.tagName === "INPUT" && c.type === "file")[0];
fileInput.onchange = null;
firstMarker.click();
check(typeof fileInput.onchange === "function",
  "re-clicking selected keyframe marker opens image picker");

/* clear image removes it without deleting the keyframe */
fileInput.onchange = null;
clearBtn.click();
const cleared = data();
check(cleared.keyframes.length === 2 && !cleared.keyframes[0].image,
  "Clear image keeps the keyframe but drops the image");

/* 6. deleting a period pulls later periods earlier; re-add starts fresh */
node.__h3FLState.setData(JSON.stringify({
  duration: 5,
  fps: 24,
  keyframes: [
    { id: 1, time: 0, image: null, prompt: "p0" },
    { id: 2, time: 2.5, image: null, prompt: "p1" },
    { id: 3, time: 5, image: null, prompt: "" },
  ],
}));
stage.children.find((c) => c.classList.contains("mmh3-fl-kf")).click(); /* first kf */
findAll(editor, (c) => c.textContent === "Delete period")[0].click();
let cur = data();
check(cur.keyframes.length === 2
  && Math.abs(cur.keyframes[0].time) < 1e-6 && Math.abs(cur.keyframes[1].time - 2.5) < 1e-6,
  `after deleting first period: later periods shifted to [0, 2.5] (got ${cur.keyframes.map((k) => k.time)})`);
check(cur.keyframes[0].id === 2 && cur.keyframes[0].prompt === "p1",
  "the following period slid into place with its own identity");
findAll(editor, (c) => c.textContent === "Delete period")[0].click();
findAll(editor, (c) => c.textContent === "Delete period")[0].click();
check(data().keyframes.length === 0, "deleting every period leaves no dangling end anchor");
addBtn.click();
cur = data();
check(cur.keyframes.length === 2
  && Math.abs(cur.keyframes[0].time) < 1e-6 && Math.abs(cur.keyframes[1].time - 2.5) < 1e-6,
  `re-add after full deletion is a fresh chain [0, 2.5] (got ${cur.keyframes.map((k) => k.time)})`);

/* 7. a lone empty end anchor from old saved state never poisons the next add */
node.__h3FLState.setData(JSON.stringify({
  duration: 5,
  fps: 24,
  keyframes: [{ id: 9, time: 5, image: null, prompt: "" }],
}));
addBtn.click();
cur = data();
check(cur.keyframes.length === 2
  && Math.abs(cur.keyframes[0].time) < 1e-6 && Math.abs(cur.keyframes[1].time - 2.5) < 1e-6,
  `lone end anchor dropped on add: fresh chain [0, 2.5] (got ${cur.keyframes.map((k) => k.time)})`);

/* node removal releases listeners and resets backend state */
check(typeof node.onRemoved === "function", "onRemoved cleanup is registered");
node.onRemoved();

/* 8. panel toggles travel inside fl_data (no ComfyUI widgets involved) */
const toggleLabels = findAll(header, (c) => c.classList.contains("mmh3-fl-toggle"));
check(toggleLabels.length === 2, `panel renders 2 toggle switches (got ${toggleLabels.length})`);
const toggleInput = (text) => {
  const label = toggleLabels.find((l) =>
    findAll(l, (c) => c.textContent === text).length > 0);
  return findAll(label, (c) => c.tagName === "INPUT")[0];
};
check(toggleInput("Offload DiT")?.checked === true, "Offload DiT defaults to on");
check(toggleInput("Loudness match")?.checked === true, "Loudness match defaults to on");
toggleInput("Loudness match").checked = false;
toggleInput("Loudness match").onchange?.();
check(data().audio_loudness_match === false,
  "toggle switch writes into fl_data (audio_loudness_match=false)");
check(data().offload_dit === true, "untouched toggle keeps its value in fl_data");
/* restored fl_data (workflow load / undo) drives the switches */
node.__h3FLState.setData(node.properties.fl_data);
check(toggleInput("Loudness match").checked === false,
  "switch re-syncs from restored fl_data (stays off)");
node.__h3FLState.setData("{}");
check(toggleInput("Loudness match").checked === true
  && toggleInput("Offload DiT").checked === true,
  "missing keys in old fl_data default both toggles to on");

console.log(process.exitCode ? "REPRO DONE (failures)" : "REPRO DONE (all ok)");
