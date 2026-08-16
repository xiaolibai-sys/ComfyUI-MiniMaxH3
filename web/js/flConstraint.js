import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const style = document.createElement("style");
style.textContent = `
.mmh3-fl-root {
  --fl-text: var(--input-text, #d8dbe2);
  --fl-dim: var(--descrip-text, #8b91a0);
  --fl-border: var(--border-color, #3a3f4b);
  --fl-border-strong: #4a5163;
  --fl-surface: var(--comfy-input-bg, #22262f);
  --fl-raised: rgba(255, 255, 255, 0.035);
  --fl-accent: #4cc2a8;
  --fl-accent-dim: rgba(76, 194, 168, 0.16);
  --fl-warn: #f5b04c;
  --fl-danger: var(--error-text, #f26d6d);
  display: flex;
  flex-direction: column;
  gap: 8px;
  width: 100%;
  min-width: 0;
  max-width: 100%;
  height: 460px;
  box-sizing: border-box;
  padding: 8px;
  overflow: hidden;
  font-size: 12px;
  color: var(--fl-text);
}
.mmh3-fl-root *,
.mmh3-fl-root *::before,
.mmh3-fl-root *::after {
  box-sizing: border-box;
}
.mmh3-fl-root button {
  font: inherit;
  color: inherit;
  background: none;
  border: none;
  cursor: pointer;
  padding: 0;
}
.mmh3-fl-root input {
  font: inherit;
  color: var(--fl-text);
  background: var(--fl-surface);
  border: 1px solid var(--fl-border);
  border-radius: 6px;
  padding: 5px 8px;
  outline: none;
}
.mmh3-fl-root input:focus {
  border-color: var(--fl-accent);
}

/* header */
.mmh3-fl-header {
  flex: none;
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 10px;
}
.mmh3-fl-title {
  font-size: 13px;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 7px;
}
.mmh3-fl-title svg {
  color: var(--fl-accent);
}
.mmh3-fl-hfield {
  display: flex;
  align-items: center;
  gap: 6px;
  color: var(--fl-dim);
  font-size: 11px;
}
.mmh3-fl-hfield input {
  height: 28px;
  width: 62px;
}
.mmh3-fl-toggle {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  cursor: pointer;
  font-size: 11px;
  color: var(--fl-dim);
  user-select: none;
}
.mmh3-fl-toggle:hover {
  color: var(--fl-text);
}
.mmh3-fl-toggle input {
  display: none;
}
.mmh3-fl-toggle-track {
  position: relative;
  width: 28px;
  height: 16px;
  border-radius: 9px;
  background: var(--fl-surface);
  border: 1px solid var(--fl-border-strong);
  flex: none;
  transition: background 0.15s ease, border-color 0.15s ease;
}
.mmh3-fl-toggle-thumb {
  position: absolute;
  top: 2px;
  left: 2px;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: var(--fl-dim);
  transition: left 0.15s ease, background 0.15s ease;
}
.mmh3-fl-toggle input:checked + .mmh3-fl-toggle-track {
  background: var(--fl-accent-dim);
  border-color: var(--fl-accent);
}
.mmh3-fl-toggle input:checked + .mmh3-fl-toggle-track .mmh3-fl-toggle-thumb {
  left: 14px;
  background: var(--fl-accent);
}
.mmh3-fl-spacer {
  flex: 1;
}
.mmh3-fl-pill {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  padding: 4px 11px;
  border-radius: 20px;
  border: 1px solid var(--fl-border);
  background: var(--fl-surface);
  max-width: 300px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.mmh3-fl-pill.ok {
  color: var(--fl-accent);
  border-color: var(--fl-accent-dim);
}
.mmh3-fl-pill.bad {
  color: var(--fl-warn);
  border-color: rgba(245, 176, 76, 0.35);
}

/* timeline */
.mmh3-fl-timeline {
  flex: none;
  border: 1px solid var(--fl-border);
  border-radius: 8px;
  background: var(--fl-raised);
  padding: 8px 10px;
}
.mmh3-fl-tl-head {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 6px;
}
.mmh3-fl-tl-label {
  font-size: 10px;
  color: var(--fl-dim);
  font-weight: 700;
  letter-spacing: 0.5px;
  text-transform: uppercase;
}
.mmh3-fl-tl-hint {
  margin-left: auto;
  font-size: 10px;
  color: var(--fl-dim);
  opacity: 0.8;
}
.mmh3-fl-ruler {
  position: relative;
  height: 15px;
  margin: 0 26px 2px;
}
.mmh3-fl-tick {
  position: absolute;
  top: 0;
  font-size: 9px;
  color: var(--fl-dim);
  transform: translateX(-50%);
  user-select: none;
  white-space: nowrap;
}
.mmh3-fl-tick::after {
  content: "";
  position: absolute;
  left: 50%;
  top: 11px;
  width: 1px;
  height: 4px;
  background: var(--fl-border-strong);
}
.mmh3-fl-stage {
  position: relative;
  height: 96px;
  margin: 0 26px;
}
.mmh3-fl-seg {
  position: absolute;
  top: 34px;
  height: 26px;
  border-radius: 5px;
  border: 1px solid var(--fl-border);
  background: var(--fl-surface);
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 4px;
  overflow: hidden;
  cursor: pointer;
}
.mmh3-fl-seg:hover {
  border-color: var(--fl-border-strong);
}
.mmh3-fl-seg.sel {
  border-color: var(--fl-accent);
  background: var(--fl-accent-dim);
}
.mmh3-fl-seg-label {
  font-size: 9px;
  color: var(--fl-dim);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding: 0 4px;
}
.mmh3-fl-seg-label .chain {
  color: var(--fl-accent);
}
.mmh3-fl-add {
  font-size: 10px;
  font-weight: 600;
  color: var(--fl-dim);
  border: 1px dashed var(--fl-border-strong);
  border-radius: 6px;
  padding: 3px 10px;
}
.mmh3-fl-add:hover {
  color: var(--fl-accent);
  border-color: var(--fl-accent);
}
.mmh3-fl-kf {
  position: absolute;
  top: 0;
  width: 52px;
  transform: translateX(-50%);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
  cursor: pointer;
  z-index: 2;
}
.mmh3-fl-kf-thumb {
  width: 52px;
  height: 34px;
  border-radius: 6px;
  border: 1px solid var(--fl-border-strong);
  background: #000;
  object-fit: cover;
  display: block;
}
.mmh3-fl-kf-empty {
  width: 52px;
  height: 34px;
  border-radius: 6px;
  border: 1px dashed var(--fl-border-strong);
  color: var(--fl-dim);
  font-size: 15px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--fl-surface);
}
.mmh3-fl-kf:hover .mmh3-fl-kf-empty {
  color: var(--fl-accent);
  border-color: var(--fl-accent);
}
.mmh3-fl-kf-time {
  font-size: 9px;
  color: var(--fl-dim);
  white-space: nowrap;
}
.mmh3-fl-kf.sel .mmh3-fl-kf-thumb,
.mmh3-fl-kf.sel .mmh3-fl-kf-empty {
  border-color: var(--fl-accent);
  box-shadow: 0 0 0 1px var(--fl-accent);
}
.mmh3-fl-kf.sel .mmh3-fl-kf-time {
  color: var(--fl-accent);
}
.mmh3-fl-kf-pin {
  position: absolute;
  top: -4px;
  right: -4px;
  width: 14px;
  height: 14px;
  border-radius: 50%;
  background: var(--fl-accent);
  color: #10221d;
  font-size: 8px;
  font-weight: 700;
  display: flex;
  align-items: center;
  justify-content: center;
}

/* editor */
.mmh3-fl-editor {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  scrollbar-width: thin;
  display: flex;
  gap: 12px;
  border: 1px solid var(--fl-border);
  border-radius: 8px;
  background: var(--fl-raised);
  padding: 12px;
}
.mmh3-fl-ed-empty {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--fl-dim);
}
.mmh3-fl-ed-preview {
  flex: none;
  width: 176px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.mmh3-fl-ed-thumb {
  width: 176px;
  height: 99px;
  border-radius: 8px;
  border: 1px solid var(--fl-border);
  background: #000;
  object-fit: contain;
  display: block;
}
.mmh3-fl-ed-thumbbtn {
  position: relative;
  width: 176px;
  height: 99px;
  border-radius: 8px;
  border: 1px solid var(--fl-border);
  overflow: hidden;
  display: block;
  padding: 0;
  background: #000;
}
.mmh3-fl-ed-thumbbtn .mmh3-fl-ed-thumb {
  border: none;
  border-radius: 0;
}
.mmh3-fl-ed-thumbbtn:hover {
  border-color: var(--fl-accent);
}
.mmh3-fl-ed-thumbhint {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  padding: 4px 0;
  font-size: 10px;
  font-weight: 600;
  text-align: center;
  color: #eafff8;
  background: rgba(20, 24, 30, 0.72);
  opacity: 0;
  transition: opacity 0.15s ease;
}
.mmh3-fl-ed-thumbbtn:hover .mmh3-fl-ed-thumbhint {
  opacity: 1;
}
.mmh3-fl-ed-imgactions {
  display: flex;
  gap: 6px;
}
.mmh3-fl-ed-imgactions .mmh3-fl-btn {
  flex: 1;
  justify-content: center;
  padding: 5px 6px;
}
.mmh3-fl-ed-drop {
  width: 176px;
  height: 99px;
  border-radius: 8px;
  border: 1px dashed var(--fl-border-strong);
  color: var(--fl-dim);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 6px;
  font-size: 11px;
}
.mmh3-fl-ed-drop:hover {
  color: var(--fl-accent);
  border-color: var(--fl-accent);
}
.mmh3-fl-ed-info {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.mmh3-fl-ed-title {
  font-size: 13px;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 8px;
}
.mmh3-fl-ed-title .tag {
  font-size: 9px;
  font-weight: 700;
  color: var(--fl-accent);
  background: var(--fl-accent-dim);
  padding: 2px 8px;
  border-radius: 10px;
}
.mmh3-fl-ed-title .mmh3-fl-btn {
  margin-left: auto;
  flex: none;
}
.mmh3-fl-ed-row {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.mmh3-fl-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 12px;
  border-radius: 6px;
  border: 1px solid var(--fl-border);
  background: var(--fl-surface);
  color: var(--fl-dim);
  font-size: 11px;
  font-weight: 600;
}
.mmh3-fl-btn:hover:not(:disabled) {
  color: var(--fl-text);
  border-color: var(--fl-border-strong);
}
.mmh3-fl-btn.danger {
  color: var(--fl-danger);
  border-color: rgba(242, 109, 109, 0.45);
  background: rgba(242, 109, 109, 0.08);
}
.mmh3-fl-btn.danger:hover:not(:disabled) {
  color: #ffe1e1;
  background: rgba(242, 109, 109, 0.2);
  border-color: var(--fl-danger);
}
.mmh3-fl-btn:disabled {
  opacity: 0.4;
  cursor: default;
}
.mmh3-fl-btn.primary {
  color: var(--fl-accent);
  border-color: var(--fl-accent);
  background: var(--fl-accent-dim);
}
.mmh3-fl-btn.primary:hover:not(:disabled) {
  color: #eafff8;
  background: rgba(76, 194, 168, 0.3);
  border-color: var(--fl-accent);
}
.mmh3-fl-seg-add {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 6px;
  color: var(--fl-dim);
  font-size: 11px;
}
.mmh3-fl-seg-add input {
  height: 28px;
  width: 58px;
}
.mmh3-fl-total-ro {
  color: var(--fl-text);
  font-weight: 600;
}
.mmh3-fl-ed-note {
  font-size: 11px;
  color: var(--fl-dim);
  line-height: 1.6;
}
.mmh3-fl-ed-note .chain {
  color: var(--fl-accent);
}
.mmh3-fl-ed-prompt {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.mmh3-fl-ed-prompt label {
  font-size: 10px;
  color: var(--fl-dim);
  font-weight: 700;
  letter-spacing: 0.3px;
}
.mmh3-fl-ed-prompt textarea {
  font: inherit;
  font-size: 12px;
  line-height: 1.5;
  color: var(--fl-text);
  background: var(--fl-surface);
  border: 1px solid var(--fl-border);
  border-radius: 6px;
  padding: 6px 8px;
  outline: none;
  min-height: 64px;
  resize: none;
  width: 100%;
}
.mmh3-fl-ed-prompt textarea:focus {
  border-color: var(--fl-accent);
}
.mmh3-fl-global {
  flex: none;
  display: flex;
  gap: 10px;
  align-items: flex-start;
  border: 1px solid var(--fl-border);
  border-radius: 8px;
  background: var(--fl-raised);
  padding: 8px 10px;
}
.mmh3-fl-global .mmh3-fl-ed-prompt {
  flex: 1;
}
.mmh3-fl-ed-note-input textarea {
  min-height: 44px;
}
`;
document.head.appendChild(style);

const ICONS = {
  film: '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="4" width="20" height="16" rx="2"/><path d="M7 4v16M17 4v16M2 9h5M2 15h5M17 9h5M17 15h5"/></svg>',
  check: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M20 6 9 17l-5-5"/></svg>',
  alert: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/></svg>',
  image: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.1-3.1a2 2 0 0 0-2.8 0L6 21"/></svg>',
};

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) {
    node.className = className;
  }
  if (text !== undefined) {
    node.textContent = text;
  }
  return node;
}

function defaultData() {
  return {
    duration: 5,
    fps: 24,
    offload_dit: true,
    audio_loudness_match: true,
    global_negative_prompt: "",
    keyframes: [],
  };
}

function parseData(raw) {
  const data = defaultData();
  try {
    Object.assign(data, JSON.parse(raw || "{}"));
  } catch (error) {
    console.error("MiniMax H3 FLConstraint parse error", error);
  }
  data.duration = Math.max(1, Number(data.duration) || 5);
  data.fps = Number(data.fps) || 24;
  /* toggle defaults are "on"; only an explicit false turns them off */
  data.offload_dit = data.offload_dit !== false;
  data.audio_loudness_match = data.audio_loudness_match !== false;
  data.global_negative_prompt = String(data.global_negative_prompt || "");
  data.keyframes = (Array.isArray(data.keyframes) ? data.keyframes : [])
    .map((kf, index) => ({
      id: Number(kf?.id) || index + 1,
      time: Math.min(data.duration, Math.max(0, Number(kf?.time) || 0)),
      image: kf?.image?.name ? kf.image : null,
      prompt: String(kf?.prompt || ""),
      negative_prompt: String(kf?.negative_prompt || ""),
      note: String(kf?.note || ""),
    }))
    .sort((a, b) => a.time - b.time);
  return data;
}

function mediaUrl(name) {
  return api.apiURL("/view?" + new URLSearchParams({ filename: name, type: "input" }));
}

function syncFLConstraint(node, data) {
  return api.fetchApi("/minimax-h3/fl_constraint", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ node_id: String(node.id), data }),
  }).catch((error) => {
    console.error("MiniMax H3 FLConstraint sync error", error);
  });
}

function formatTimeShort(seconds) {
  const safe = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(safe / 60);
  const whole = Math.floor(safe % 60);
  return `${String(minutes).padStart(2, "0")}:${String(whole).padStart(2, "0")}`;
}

function rulerStep(total, width) {
  const minPx = 54;
  for (const step of [0.5, 1, 2, 5]) {
    if ((total / step) * minPx <= width) {
      return step;
    }
  }
  return 10;
}

function setupFLConstraint(node) {
  node.properties = node.properties || {};
  if (node.__h3FLState) {
    node.__h3FLState.setData(node.properties.fl_data);
    /* onConfigure loads persisted keyframes after onNodeCreated pushed
       defaults, so re-sync the backend with the real data */
    syncFLConstraint(node, parseData(node.properties.fl_data));
    return;
  }

  let data = parseData(node.properties.fl_data);
  let selectedId = null;
  let nextId = data.keyframes.reduce((max, kf) => Math.max(max, kf.id), 0) + 1;

  /* ---------- skeleton ---------- */
  const root = el("div", "mmh3-fl-root");

  const header = el("div", "mmh3-fl-header");
  const title = el("div", "mmh3-fl-title");
  title.innerHTML = ICONS.film;
  title.append(el("span", "", "FL Constraint"));
  const durField = el("div", "mmh3-fl-hfield", "Total");
  const durValue = el("span", "mmh3-fl-total-ro", "0.0s");
  durField.append(durValue);
  const fpsField = el("div", "mmh3-fl-hfield", "FPS");
  const fpsInput = document.createElement("input");
  fpsInput.type = "number";
  fpsInput.min = "1";
  fpsInput.max = "60";
  fpsField.append(fpsInput);
  /* panel toggles travel inside fl_data itself (synced to the backend by
     persist) - no ComfyUI widgets involved */
  const toggles = [];
  const makeToggle = (label, key, tooltip) => {
    const wrap = el("label", "mmh3-fl-toggle");
    wrap.title = tooltip;
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = Boolean(data[key]);
    const track = el("span", "mmh3-fl-toggle-track");
    track.append(el("span", "mmh3-fl-toggle-thumb"));
    wrap.append(input, track, el("span", "", label));
    input.onchange = () => {
      data[key] = input.checked;
      persist();
    };
    toggles.push({ wrap, input, key });
  };
  makeToggle("Offload DiT", "offload_dit",
    "Offload DiT blocks to RAM during rolling segment-boundary VAE re-encoding.");
  makeToggle("Loudness match", "audio_loudness_match",
    "Match per-segment audio loudness (BS.1770 gated LUFS) across rolling segment boundaries.");
  const spacer = el("div", "mmh3-fl-spacer");
  const pill = el("div", "mmh3-fl-pill bad");
  header.append(
    title, durField, fpsField,
    ...toggles.map((t) => t.wrap),
    spacer, pill,
  );

  const timeline = el("div", "mmh3-fl-timeline");
  const tlHead = el("div", "mmh3-fl-tl-head");
  const segAdd = el("div", "mmh3-fl-seg-add", "Segment");
  const segDurInput = document.createElement("input");
  segDurInput.type = "number";
  segDurInput.min = "0.1";
  segDurInput.step = "0.5";
  segDurInput.value = "2.5";
  const addKfBtn = el("button", "mmh3-fl-btn primary", "+ Add segment");
  segAdd.append(segDurInput, el("span", "", "s"), addKfBtn);
  tlHead.append(
    el("span", "mmh3-fl-tl-label", "Keyframes"),
    el("span", "mmh3-fl-tl-hint", "← → nudges selected keyframe"),
    segAdd,
  );
  const ruler = el("div", "mmh3-fl-ruler");
  const stage = el("div", "mmh3-fl-stage");
  timeline.append(tlHead, ruler, stage);

  const globalPanel = el("div", "mmh3-fl-global");
  const globalPromptWrap = el("div", "mmh3-fl-ed-prompt");
  globalPromptWrap.append(el("label", "", "GLOBAL NEGATIVE PROMPT"));
  const globalNegInput = document.createElement("textarea");
  globalNegInput.placeholder = "Negative guidance shared by all segments…";
  globalNegInput.value = data.global_negative_prompt || "";
  globalNegInput.oninput = () => {
    data.global_negative_prompt = globalNegInput.value;
    persist();
  };
  globalPromptWrap.append(globalNegInput);
  globalPanel.append(globalPromptWrap);

  const editor = el("div", "mmh3-fl-editor");
  root.append(header, globalPanel, timeline, editor);

  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.accept = "image/*";
  fileInput.style.display = "none";
  root.append(fileInput);

  /* ---------- helpers ---------- */
  const sortedKfs = () => [...data.keyframes].sort((a, b) => a.time - b.time);
  const selectedKf = () => data.keyframes.find((kf) => kf.id === selectedId) || null;
  /* keep stored state self-consistent before it is serialized or synced:
     duration always derives from the last keyframe, keyframes stay sorted */
  const normalizeData = () => {
    data.keyframes.sort((a, b) => a.time - b.time);
    const last = data.keyframes[data.keyframes.length - 1];
    data.duration = last ? Math.max(1, last.time) : 5;
  };
  const persist = () => {
    normalizeData();
    node.properties.fl_data = JSON.stringify(data);
    syncFLConstraint(node, data);
    node.graph?.setDirtyCanvas?.(true, true);
  };

  /* ---------- timeline ---------- */
  const renderRuler = () => {
    ruler.replaceChildren();
    const width = ruler.clientWidth || root.clientWidth - 88 || 700;
    const step = rulerStep(data.duration, width);
    for (let t = 0; t <= data.duration + 1e-6; t += step) {
      const tick = el("span", "mmh3-fl-tick", formatTimeShort(t));
      tick.style.left = `${(t / data.duration) * 100}%`;
      ruler.append(tick);
    }
  };

  const renderStage = () => {
    stage.replaceChildren();
    const kfs = sortedKfs();
    /* segment blocks between neighbouring keyframes */
    for (let i = 0; i < kfs.length - 1; i++) {
      const from = kfs[i].time;
      const to = kfs[i + 1].time;
      if (to - from < 1e-6) {
        continue;
      }
      const seg = el("div", "mmh3-fl-seg");
      seg.style.left = `${(from / data.duration) * 100}%`;
      seg.style.width = `${((to - from) / data.duration) * 100}%`;
      seg.classList.toggle("sel", kfs[i].id === selectedId);
      seg.title = kfs[i].prompt || "Click to edit this segment";
      seg.onclick = () => {
        selectedId = kfs[i].id;
        renderAll();
      };
      const label = el("span", "mmh3-fl-seg-label");
      /* every segment shows its own length; chained ones start at the
         previous segment's end, marked with the chain symbol */
      label.innerHTML = `${i > 0 ? '<span class="chain">⟲</span> ' : ""}${(to - from).toFixed(1)}s`;
      seg.append(label);
      stage.append(seg);
    }
    /* keyframe markers */
    kfs.forEach((kf) => {
      const marker = el("div", "mmh3-fl-kf");
      marker.classList.toggle("sel", kf.id === selectedId);
      marker.style.left = `${(kf.time / data.duration) * 100}%`;
      if (kf.image) {
        const img = document.createElement("img");
        img.className = "mmh3-fl-kf-thumb";
        img.src = mediaUrl(kf.image.name);
        img.alt = "";
        marker.append(img);
      } else {
        marker.append(el("div", "mmh3-fl-kf-empty", "+"));
      }
      marker.append(el("span", "mmh3-fl-kf-time", `${kf.time.toFixed(1)}s`));
      if (kf.time <= 1e-6 || Math.abs(kf.time - data.duration) <= 1e-6) {
        marker.append(el("span", "mmh3-fl-kf-pin", kf.time <= 1e-6 ? "F" : "L"));
      }
      marker.onclick = (event) => {
        event.stopPropagation();
        if (kf.id === selectedId) {
          /* already selected: clicking the marker again re-picks its image */
          pickImage(kf);
          return;
        }
        selectedId = kf.id;
        renderAll();
      };
      stage.append(marker);
    });
  };

  addKfBtn.onclick = () => {
    const segDur = Math.max(0.1, Number(segDurInput.value) || 0.1);
    /* a lone end anchor with no content is a leftover from deletions (or old
       saved state); it bounds no segment, so drop it instead of letting the
       anchor pin below resurrect a phantom span across the old timeline */
    if (data.keyframes.length === 1) {
      const lone = data.keyframes[0];
      if (lone.time > 1e-6 && !lone.image && !lone.prompt && !lone.note) {
        data.keyframes = [];
      }
    }
    const existing = sortedKfs();
    if (!existing.length || existing[0].time > 1e-6) {
      /* chain must start at 0: pin a first-frame anchor when missing */
      data.keyframes.push({
        id: nextId++,
        time: 0,
        image: null,
        prompt: "",
        negative_prompt: "",
        note: "",
      });
    }
    const kfs = sortedKfs();
    const lastTime = kfs.length ? kfs[kfs.length - 1].time : 0;
    const endKf = {
      id: nextId++,
      time: Math.round((lastTime + segDur) * 10) / 10,
      image: null,
      prompt: "",
      negative_prompt: "",
      note: "",
    };
    data.keyframes.push(endKf);
    data.duration = Math.max(1, endKf.time);
    selectedId = endKf.id;
    persist();
    renderAll();
  };

  /* ---------- editor ---------- */
  const uploadImage = async (file) => {
    const body = new FormData();
    body.append("image", file);
    const response = await api.fetchApi("/upload/image", { method: "POST", body });
    if (!response.ok) {
      throw new Error(`upload failed: ${response.status}`);
    }
    const info = await response.json();
    const subfolder = info.subfolder || "";
    return {
      name: subfolder ? `${subfolder}/${info.name}` : info.name,
      filename: info.name,
      subfolder,
      type: "input",
    };
  };

  const pickImage = (kf) => {
    fileInput.onchange = async () => {
      const file = fileInput.files?.[0];
      if (!file) {
        return;
      }
      try {
        kf.image = await uploadImage(file);
        persist();
        renderAll();
      } catch (error) {
        alert(error.message || String(error));
      } finally {
        fileInput.value = "";
      }
    };
    fileInput.click();
  };

  const renderEditor = () => {
    editor.replaceChildren();
    const kf = selectedKf();
    if (!kf) {
      editor.append(el("div", "mmh3-fl-ed-empty",
        "Click a keyframe to edit its image, segment prompt, or image note."));
      return;
    }
    const kfs = sortedKfs();
    const index = kfs.findIndex((item) => item.id === kf.id);
    const isFirst = kf.time <= 1e-6;
    const isLast = Math.abs(kf.time - data.duration) <= 1e-6;

    const preview = el("div", "mmh3-fl-ed-preview");
    if (kf.image) {
      const thumbBtn = el("button", "mmh3-fl-ed-thumbbtn");
      thumbBtn.title = "Click to replace this image";
      const img = document.createElement("img");
      img.className = "mmh3-fl-ed-thumb";
      img.src = mediaUrl(kf.image.name);
      img.alt = "";
      thumbBtn.append(img, el("span", "mmh3-fl-ed-thumbhint", "Replace image"));
      thumbBtn.onclick = () => pickImage(kf);
      preview.append(thumbBtn);
      /* image actions stay with the image they operate on */
      const imgActions = el("div", "mmh3-fl-ed-imgactions");
      const replaceBtn = el("button", "mmh3-fl-btn primary", "Replace");
      replaceBtn.title = "Replace this keyframe's image";
      replaceBtn.onclick = () => pickImage(kf);
      const clearImgBtn = el("button", "mmh3-fl-btn danger", "Remove");
      clearImgBtn.title = "Remove the image from this keyframe (the keyframe itself stays)";
      clearImgBtn.onclick = () => {
        kf.image = null;
        persist();
        renderAll();
      };
      imgActions.append(replaceBtn, clearImgBtn);
      preview.append(imgActions);
    } else {
      const drop = el("button", "mmh3-fl-ed-drop");
      const icon = el("span");
      icon.innerHTML = ICONS.image;
      drop.append(icon, el("span", "", "Choose image"));
      drop.onclick = () => pickImage(kf);
      preview.append(drop);
    }
    if (kf.image) {
      const noteWrap = el("div", "mmh3-fl-ed-prompt mmh3-fl-ed-note-input");
      noteWrap.append(el("label", "", "IMAGE NOTE"));
      const imageNoteInput = document.createElement("textarea");
      imageNoteInput.placeholder = "Optional note for this image…";
      imageNoteInput.value = kf.note || "";
      imageNoteInput.oninput = () => {
        kf.note = imageNoteInput.value;
        persist();
      };
      noteWrap.append(imageNoteInput);
      preview.append(noteWrap);
    }

    const info = el("div", "mmh3-fl-ed-info");
    const head = el("div", "mmh3-fl-ed-title");
    head.append(el("span", "", `Keyframe ${index + 1} · starts at ${kf.time.toFixed(1)}s`));
    if (isFirst) {
      head.append(el("span", "tag", "FIRST FRAME"));
    }
    if (isLast) {
      head.append(el("span", "tag", "LAST FRAME"));
    }
    /* deleting the period (timeline) is kept apart from removing its image */
    const removeBtn = el("button", "mmh3-fl-btn danger", "Delete period");
    removeBtn.title = "Delete this period from the timeline; later periods shift earlier";
    removeBtn.onclick = () => {
      const removedIndex = kfs.findIndex((item) => item.id === kf.id);
      const hasFollowing = removedIndex >= 0 && removedIndex < kfs.length - 1;
      /* deleting a period removes its start boundary and pulls later periods
         earlier by the deleted length, so the chain stays gap-free and no
         dangling end anchor survives to poison the next "Add segment" */
      const shift = hasFollowing ? kfs[removedIndex + 1].time - kf.time : 0;
      data.keyframes = data.keyframes.filter((item) => item.id !== kf.id);
      if (shift > 1e-6) {
        for (const later of data.keyframes) {
          if (later.time > kf.time + 1e-6) {
            later.time = Math.max(0, Math.round((later.time - shift) * 10) / 10);
          }
        }
      }
      /* keep a sensible selection so deleting several keyframes stays fluid */
      const remaining = sortedKfs();
      const neighbor = remaining[Math.min(removedIndex, remaining.length - 1)] || null;
      selectedId = neighbor ? neighbor.id : null;
      persist();
      renderAll();
    };
    head.append(removeBtn);

    /* segments chain: this one starts where the previous one ends, so the
       editable value is the segment length, not an absolute time */
    const hasNext = index < kfs.length - 1;
    if (hasNext) {
      const row = el("div", "mmh3-fl-ed-row");
      const durField = el("div", "mmh3-fl-hfield", "Duration");
      const durInput = document.createElement("input");
      durInput.type = "number";
      durInput.min = "0.1";
      durInput.step = "0.1";
      durInput.title = "Length of the segment starting at this keyframe; later segments shift along";
      durInput.value = (kfs[index + 1].time - kf.time).toFixed(1);
      durField.append(durInput, el("span", "", "s"));
      row.append(durField);
      durInput.onchange = () => {
        const oldDur = kfs[index + 1].time - kf.time;
        const newDur = Math.max(0.1, Math.round((Number(durInput.value) || 0) * 10) / 10);
        const delta = newDur - oldDur;
        if (Math.abs(delta) > 1e-6) {
          /* shift every later keyframe so following segments keep their length */
          for (const later of data.keyframes) {
            if (later.time > kf.time + 1e-6) {
              later.time = Math.round((later.time + delta) * 10) / 10;
            }
          }
        }
        persist();
        renderAll();
      };
      info.append(head, row);
    } else {
      info.append(head);
    }

    const note = el("div", "mmh3-fl-ed-note");
    if (index > 0) {
      note.innerHTML = 'The segment before this keyframe <span class="chain">⟲ reuses the previous keyframe\'s image as its end frame</span>, so motion chains continuously.';
    } else {
      note.textContent = "This is the first keyframe; the opening segment starts from this image.";
    }
    info.append(note);
    if (index < kfs.length - 1) {
      const promptWrap = el("div", "mmh3-fl-ed-prompt");
      promptWrap.append(el("label", "",
        `SEGMENT PROMPT · ${kf.time.toFixed(1)}s → ${kfs[index + 1].time.toFixed(1)}s`));
      const promptInput = document.createElement("textarea");
      promptInput.placeholder = "Describe the motion in this segment…";
      promptInput.value = kf.prompt || "";
      promptInput.oninput = () => {
        kf.prompt = promptInput.value;
        persist();
      };
      promptWrap.append(promptInput);
      promptWrap.append(el("label", "", "SEGMENT NEGATIVE PROMPT"));
      const negativeInput = document.createElement("textarea");
      negativeInput.placeholder = "Negative guidance for this segment…";
      negativeInput.value = kf.negative_prompt || "";
      negativeInput.oninput = () => {
        kf.negative_prompt = negativeInput.value;
        persist();
      };
      promptWrap.append(negativeInput);
      info.append(promptWrap);
    }
    editor.append(preview, info);
  };

  /* ---------- validation ---------- */
  const updatePill = () => {
    const kfs = sortedKfs();
    const imaged = kfs.filter((kf) => kf.image);
    const hasFirst = imaged.some((kf) => kf.time <= 1e-6);
    const hasLast = imaged.some((kf) => Math.abs(kf.time - data.duration) <= 1e-6);
    const problems = [];
    if (!imaged.length) {
      problems.push("no keyframe image");
    } else {
      if (!hasFirst) {
        problems.push("no first frame at 0s");
      }
      if (!hasLast) {
        problems.push("no last frame at end");
      }
    }
    const pending = kfs.filter((kf) => !kf.image).length;
    if (pending) {
      problems.push(`${pending} keyframe${pending > 1 ? "s" : ""} missing image`);
    }
    pill.className = `mmh3-fl-pill ${problems.length ? "bad" : "ok"}`;
    pill.innerHTML = problems.length ? ICONS.alert : ICONS.check;
    pill.append(el("span", "",
      problems.length ? problems.join(" · ") : `${imaged.length} keyframes · ${data.duration}s`));
  };

  /* ---------- bindings ---------- */
  fpsInput.value = data.fps;
  fpsInput.onchange = () => {
    data.fps = Number(fpsInput.value) || 24;
    persist();
  };

  let rootHover = false;
  root.addEventListener("mouseenter", () => {
    rootHover = true;
  });
  root.addEventListener("mouseleave", () => {
    rootHover = false;
  });
  const onKeydown = (event) => {
    if (!rootHover || (event.key !== "ArrowLeft" && event.key !== "ArrowRight")) {
      return;
    }
    const tag = event.target?.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") {
      return;
    }
    const kf = selectedKf();
    if (!kf) {
      return;
    }
    event.preventDefault();
    const step = event.shiftKey ? 0.5 : 0.1;
    const delta = event.key === "ArrowRight" ? step : -step;
    kf.time = Math.min(data.duration,
      Math.max(0, Math.round((kf.time + delta) * 10) / 10));
    persist();
    renderAll();
  };
  document.addEventListener("keydown", onKeydown);

  const resizeObserver = new ResizeObserver(() => {
    renderRuler();
  });
  resizeObserver.observe(root);

  /* release listeners and backend state when the node is deleted,
     so deleting and re-adding the node never leaks stale handlers */
  const originalOnRemoved = node.onRemoved;
  node.onRemoved = function () {
    document.removeEventListener("keydown", onKeydown);
    resizeObserver.disconnect();
    syncFLConstraint(node, defaultData());
    originalOnRemoved?.apply(this, arguments);
  };

  const renderAll = () => {
    normalizeData();
    durValue.textContent = `${data.duration.toFixed(1)}s`;
    renderRuler();
    renderStage();
    renderEditor();
    updatePill();
  };

  const setData = (value) => {
    data = parseData(value);
    nextId = data.keyframes.reduce((max, kf) => Math.max(max, kf.id), 0) + 1;
    selectedId = null;
    /* restored fl_data (workflow load / undo) drives the toggle switches */
    for (const t of toggles) {
      t.input.checked = Boolean(data[t.key]);
    }
    renderAll();
  };

  const domWidget = node.addDOMWidget("fl_constraint_ui", "fl_constraint_ui", root, {
    serialize: false,
    hideOnZoom: false,
    getMinHeight: () => 460,
    getMaxHeight: () => 460,
  });
  domWidget.computeSize = function (width) {
    return [width, 460];
  };
  node.__h3FLState = { setData, render: renderAll };
  node.setSize?.([840, 500]);
  renderAll();
  syncFLConstraint(node, data);
}

app.registerExtension({
  name: "ComfyUI-MiniMaxH3.FLConstraint",
  setup() {
    const originalQueuePrompt = api.queuePrompt.bind(api);
    api.queuePrompt = async (...args) => {
      const nodes = app.graph?.nodes || [];
      await Promise.all(nodes
        .filter((node) => node.__h3FLState && node.properties?.fl_data)
        .map((node) => syncFLConstraint(
          node,
          parseData(node.properties.fl_data),
        )));
      return originalQueuePrompt(...args);
    };
  },
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData?.name !== "MiniMaxH3FLConstraint") {
      return;
    }
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      setupFLConstraint(this);
      return result;
    };
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      setupFLConstraint(this);
      return result;
    };
  },
});
