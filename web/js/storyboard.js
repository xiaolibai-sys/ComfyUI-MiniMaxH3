import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const style = document.createElement("style");
style.textContent = `
.mmh3-sb-root {
  --sb-text: var(--input-text, #d8dbe2);
  --sb-dim: var(--descrip-text, #8b91a0);
  --sb-border: var(--border-color, #3a3f4b);
  --sb-border-strong: #4a5163;
  --sb-surface: var(--comfy-input-bg, #22262f);
  --sb-raised: rgba(255, 255, 255, 0.035);
  --sb-accent: #4cc2a8;
  --sb-accent-dim: rgba(76, 194, 168, 0.16);
  --sb-warn: #f5b04c;
  --sb-danger: var(--error-text, #f26d6d);
  --sb-c-picture: #38bdf8;
  --sb-c-video: #f5b04c;
  --sb-c-audio: #b78cf0;
  display: flex;
  flex-direction: column;
  gap: 8px;
  width: 100%;
  min-width: 0;
  max-width: 100%;
  height: 720px;
  box-sizing: border-box;
  padding: 8px;
  overflow: hidden;
  font-size: 12px;
  color: var(--sb-text);
}
.mmh3-sb-root *,
.mmh3-sb-root *::before,
.mmh3-sb-root *::after {
  box-sizing: border-box;
}
.mmh3-sb-root button {
  font: inherit;
  color: inherit;
  background: none;
  border: none;
  cursor: pointer;
  padding: 0;
}
.mmh3-sb-root input,
.mmh3-sb-root select,
.mmh3-sb-root textarea {
  font: inherit;
  color: var(--sb-text);
  background: var(--sb-surface);
  border: 1px solid var(--sb-border);
  border-radius: 6px;
  padding: 5px 8px;
  outline: none;
}
.mmh3-sb-root input:focus,
.mmh3-sb-root select:focus,
.mmh3-sb-root textarea:focus {
  border-color: var(--sb-accent);
}
.mmh3-sb-root textarea {
  resize: none;
}
.mmh3-sb-scroll {
  scrollbar-width: thin;
}

/* header */
.mmh3-sb-header {
  flex: none;
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 10px;
}
.mmh3-sb-seg {
  display: flex;
  background: var(--sb-surface);
  border: 1px solid var(--sb-border);
  border-radius: 7px;
  padding: 2px;
  gap: 2px;
}
.mmh3-sb-seg button {
  padding: 3px 12px;
  border-radius: 5px;
  color: var(--sb-dim);
  font-size: 11px;
  white-space: nowrap;
}
.mmh3-sb-seg button.on {
  background: var(--sb-accent-dim);
  color: var(--sb-accent);
}
.mmh3-sb-hfield {
  display: flex;
  align-items: center;
  gap: 6px;
  color: var(--sb-dim);
  font-size: 11px;
}
.mmh3-sb-hfield select,
.mmh3-sb-hfield input {
  height: 28px;
}
.mmh3-sb-hfield input[type="number"] {
  width: 58px;
}
.mmh3-sb-spacer {
  flex: 1;
}
.mmh3-sb-pill {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  padding: 4px 11px;
  border-radius: 20px;
  border: 1px solid var(--sb-border);
  background: var(--sb-surface);
  max-width: 260px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.mmh3-sb-pill.ok {
  color: var(--sb-accent);
  border-color: var(--sb-accent-dim);
}
.mmh3-sb-pill.bad {
  color: var(--sb-warn);
  border-color: rgba(245, 176, 76, 0.35);
}

/* timeline */
.mmh3-sb-timeline {
  flex: none;
  border: 1px solid var(--sb-border);
  border-radius: 8px;
  background: var(--sb-raised);
  padding: 8px 10px 8px;
}
.mmh3-sb-tl-head {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 6px;
}
.mmh3-sb-tl-label {
  font-size: 10px;
  color: var(--sb-dim);
  font-weight: 700;
  letter-spacing: 0.5px;
  text-transform: uppercase;
}
.mmh3-sb-tl-total {
  font-size: 11px;
  color: var(--sb-dim);
}
.mmh3-sb-tl-total b {
  color: var(--sb-text);
  font-weight: 600;
}
.mmh3-sb-tl-total.mismatch b {
  color: var(--sb-warn);
}
.mmh3-sb-tl-hint {
  margin-left: auto;
  font-size: 10px;
  color: var(--sb-dim);
  opacity: 0.8;
}
.mmh3-sb-ruler {
  position: relative;
  height: 15px;
  margin: 0 2px 2px;
}
.mmh3-sb-tick {
  position: absolute;
  top: 0;
  font-size: 9px;
  color: var(--sb-dim);
  transform: translateX(-50%);
  user-select: none;
  white-space: nowrap;
}
.mmh3-sb-tick::after {
  content: "";
  position: absolute;
  left: 50%;
  top: 11px;
  width: 1px;
  height: 4px;
  background: var(--sb-border-strong);
}
.mmh3-sb-track {
  display: flex;
  gap: 3px;
  height: 50px;
}
.mmh3-sb-shot {
  position: relative;
  min-width: 42px;
  border-radius: 6px;
  background: var(--sb-surface);
  border: 1px solid var(--sb-border);
  display: flex;
  flex-direction: column;
  justify-content: center;
  padding: 0 8px;
  overflow: hidden;
  cursor: pointer;
  user-select: none;
}
.mmh3-sb-shot:hover {
  border-color: var(--sb-border-strong);
}
.mmh3-sb-shot:focus-visible {
  outline: 1px solid var(--sb-accent);
  outline-offset: 1px;
}
.mmh3-sb-shot.sel {
  border-color: var(--sb-accent);
  background: var(--sb-accent-dim);
}
.mmh3-sb-shot-title {
  font-size: 11px;
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.mmh3-sb-shot-sub {
  font-size: 9px;
  color: var(--sb-dim);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.mmh3-sb-shot-warn {
  position: absolute;
  top: 3px;
  right: 4px;
  color: var(--sb-warn);
  display: none;
}
.mmh3-sb-shot.invalid .mmh3-sb-shot-warn {
  display: block;
}
.mmh3-sb-add {
  flex: none;
  width: 36px;
  border: 1px dashed var(--sb-border-strong);
  border-radius: 6px;
  color: var(--sb-dim);
  font-size: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
}
.mmh3-sb-add:hover {
  color: var(--sb-accent);
  border-color: var(--sb-accent);
}
.mmh3-sb-tl-warn {
  margin-top: 6px;
  font-size: 11px;
  color: var(--sb-warn);
  display: none;
  align-items: center;
  gap: 6px;
}
.mmh3-sb-tl-warn.show {
  display: flex;
}

/* main area */
.mmh3-sb-main {
  flex: 1;
  display: flex;
  min-height: 0;
  gap: 8px;
}
.mmh3-sb-editor {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
  overflow-y: auto;
  padding: 2px;
}
.mmh3-sb-ed-head {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.mmh3-sb-ed-title {
  font-size: 13px;
  font-weight: 600;
}
.mmh3-sb-ed-range {
  font-size: 11px;
  color: var(--sb-dim);
}
.mmh3-sb-icon {
  width: 26px;
  height: 26px;
  border-radius: 6px;
  border: 1px solid var(--sb-border);
  background: var(--sb-raised);
  color: var(--sb-dim);
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex: none;
}
.mmh3-sb-icon:hover:not(:disabled) {
  color: var(--sb-text);
  border-color: var(--sb-border-strong);
}
.mmh3-sb-icon.danger:hover:not(:disabled) {
  color: var(--sb-danger);
  border-color: var(--sb-danger);
}
.mmh3-sb-icon:disabled {
  opacity: 0.35;
  cursor: default;
}
.mmh3-sb-chipbar {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.mmh3-sb-chiprow {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
}
.mmh3-sb-chip {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 10px;
  padding: 2px 8px 2px 6px;
  border-radius: 20px;
  background: var(--sb-surface);
  border: 1px solid var(--sb-border);
}
.mmh3-sb-chip:hover {
  border-color: currentColor;
}
.mmh3-sb-chip .mmh3-sb-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: currentColor;
  flex: none;
}
.mmh3-sb-chip .mmh3-sb-chip-label {
  color: var(--sb-text);
}
.mmh3-sb-chip.k-picture { color: var(--sb-c-picture); }
.mmh3-sb-chip.k-video { color: var(--sb-c-video); }
.mmh3-sb-chip.k-audio { color: var(--sb-c-audio); }
.mmh3-sb-prompt {
  width: 100%;
  min-height: 96px;
  line-height: 1.6;
  font-size: 12px;
}
.mmh3-sb-fields {
  display: grid;
  grid-template-columns: 1fr;
  gap: 8px;
}
.mmh3-sb-fcol {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}
.mmh3-sb-fcol label {
  font-size: 10px;
  color: var(--sb-dim);
  font-weight: 700;
  letter-spacing: 0.3px;
}
.mmh3-sb-fcol textarea {
  width: 100%;
  min-height: 34px;
  font-size: 12px;
  line-height: 1.5;
}
.mmh3-sb-empty {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--sb-dim);
}

/* sidebar */
.mmh3-sb-sidebar {
  flex: none;
  width: 280px;
  border: 1px solid var(--sb-border);
  border-radius: 8px;
  background: var(--sb-raised);
  display: flex;
  flex-direction: column;
  min-height: 0;
  overflow: hidden;
}
.mmh3-sb-tabs {
  display: flex;
  border-bottom: 1px solid var(--sb-border);
  flex: none;
}
.mmh3-sb-tabs button {
  flex: 1;
  padding: 8px 0;
  font-size: 11px;
  font-weight: 600;
  color: var(--sb-dim);
  border-bottom: 2px solid transparent;
}
.mmh3-sb-tabs button.on {
  color: var(--sb-text);
  border-bottom-color: var(--sb-accent);
}
.mmh3-sb-side-body {
  flex: 1;
  overflow-y: auto;
  padding: 10px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.mmh3-sb-pane {
  display: none;
  flex-direction: column;
  gap: 8px;
}
.mmh3-sb-pane.on {
  display: flex;
}
.mmh3-sb-media-row {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  color: var(--sb-dim);
}
.mmh3-sb-media-row select {
  flex: 1;
  min-width: 0;
  height: 26px;
}
.mmh3-sb-media-none {
  font-size: 10px;
  color: var(--sb-dim);
  font-style: italic;
}
.mmh3-sb-subject {
  border: 1px solid var(--sb-border);
  border-radius: 7px;
  background: var(--sb-surface);
  padding: 8px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.mmh3-sb-subject-top {
  display: flex;
  align-items: center;
  gap: 7px;
}
.mmh3-sb-subject-top .mmh3-sb-dot {
  width: 9px;
  height: 9px;
  border-radius: 50%;
  flex: none;
}
.mmh3-sb-subject-label {
  font-size: 10px;
  font-weight: 700;
  white-space: nowrap;
}
.mmh3-sb-subject-top input {
  flex: 1;
  min-width: 0;
  height: 26px;
}
.mmh3-sb-subject textarea {
  min-height: 42px;
  font-size: 11px;
  width: 100%;
}
.mmh3-sb-ghost {
  font-size: 11px;
  color: var(--sb-dim);
  border: 1px dashed var(--sb-border-strong);
  border-radius: 6px;
  padding: 6px 12px;
  width: 100%;
  text-align: center;
}
.mmh3-sb-ghost:hover {
  color: var(--sb-accent);
  border-color: var(--sb-accent);
}
.mmh3-sb-gfield {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.mmh3-sb-gfield label {
  font-size: 10px;
  color: var(--sb-dim);
  font-weight: 700;
  letter-spacing: 0.3px;
}
.mmh3-sb-gfield textarea {
  min-height: 52px;
  font-size: 11px;
  width: 100%;
}

/* compact layout for narrow nodes */
.mmh3-sb-root.compact .mmh3-sb-main {
  flex-direction: column;
  overflow-y: auto;
}
.mmh3-sb-root.compact .mmh3-sb-editor {
  overflow: visible;
  flex: none;
}
.mmh3-sb-root.compact .mmh3-sb-sidebar {
  width: 100%;
  flex: none;
  max-height: 260px;
}

/* preview */
.mmh3-sb-preview {
  flex: none;
  border: 1px solid var(--sb-border);
  border-radius: 8px;
  background: var(--sb-raised);
  overflow: hidden;
}
.mmh3-sb-preview-toggle {
  width: 100%;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 7px 10px;
  color: var(--sb-dim);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.4px;
}
.mmh3-sb-preview-toggle:hover {
  color: var(--sb-text);
}
.mmh3-sb-chev {
  transition: transform 0.15s;
}
.mmh3-sb-preview.open .mmh3-sb-chev {
  transform: rotate(180deg);
}
.mmh3-sb-copy {
  font-size: 10px;
  font-weight: 400;
  color: var(--sb-dim);
  border: 1px solid var(--sb-border);
  border-radius: 5px;
  padding: 2px 10px;
  letter-spacing: 0;
  text-transform: none;
}
.mmh3-sb-copy:hover {
  color: var(--sb-accent);
  border-color: var(--sb-accent);
}
.mmh3-sb-preview-body {
  display: none;
  max-height: 150px;
  overflow-y: auto;
  padding: 2px 10px 10px;
}
.mmh3-sb-preview.open .mmh3-sb-preview-body {
  display: block;
}
.mmh3-sb-preview-body pre {
  white-space: pre-wrap;
  word-break: break-word;
  font: 11px/1.7 Consolas, monospace;
  color: var(--sb-dim);
  margin: 0;
}
.mmh3-sb-tok {
  border-radius: 4px;
  padding: 0 4px;
  font-weight: 600;
}
.mmh3-sb-tok.k-picture { color: var(--sb-c-picture); background: rgba(56, 189, 248, 0.1); }
.mmh3-sb-tok.k-video { color: var(--sb-c-video); background: rgba(245, 176, 76, 0.1); }
.mmh3-sb-tok.k-audio { color: var(--sb-c-audio); background: rgba(183, 140, 240, 0.1); }
.mmh3-sb-tok.k-subject { color: var(--sb-accent); background: var(--sb-accent-dim); }
.mmh3-sb-shot-head {
  color: var(--sb-accent);
  font-weight: 700;
}
`;
document.head.appendChild(style);

const MODES = ["T2VA", "full_reference"];
const RATIOS = ["adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"];
const SUBJECT_COLORS = ["#5fd68b", "#f2728c", "#6aa8f5", "#e8c455", "#4cc2c2", "#d98cf0"];
const COMPACT_WIDTH = 640;

const ICONS = {
  warn: '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 9v4M12 17h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/></svg>',
  check: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M20 6 9 17l-5-5"/></svg>',
  alert: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/></svg>',
  left: '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m15 18-6-6 6-6"/></svg>',
  right: '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m9 18 6-6-6-6"/></svg>',
  trash: '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2m3 0v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/></svg>',
  close: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 6 6 18M6 6l12 12"/></svg>',
  chev: '<svg class="mmh3-sb-chev" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m18 15-6-6-6 6"/></svg>',
};

function newShot(duration) {
  return {
    duration: Number(duration) || 0,
    prompt: "",
    camera: "",
    dialogue: "",
    sound: "",
  };
}

function defaultStoryboard() {
  return {
    mode: "T2VA",
    ratio: "16:9",
    fps: 24,
    total_duration: 5,
    negative_prompt: "",
    soundscape: "",
    music_style: "",
    subjects: [],
    shots: [newShot(5)],
  };
}

function parseStoryboard(raw) {
  const data = { ...defaultStoryboard() };
  try {
    const parsed = JSON.parse(raw || "{}");
    Object.assign(data, parsed);
  } catch (error) {
    console.error("MiniMax H3 Storyboard parse error", error);
  }
  data.mode = MODES.includes(data.mode) ? data.mode : "T2VA";
  data.ratio = RATIOS.includes(data.ratio) ? data.ratio : "16:9";
  data.fps = Number(data.fps) || 24;
  data.total_duration = Number(data.total_duration) || 5;
  data.negative_prompt = data.negative_prompt || "";
  data.soundscape = data.soundscape || "";
  data.music_style = data.music_style || "";
  data.subjects = Array.isArray(data.subjects)
    ? data.subjects
      .map((subject, index) => ({
        ...newSubject(),
        ...(subject || {}),
        name: String(subject?.name || "").trim(),
        definition: String(subject?.definition || "").trim(),
        label: `<Subject ${index + 1}>`,
      }))
      .filter((subject) => subject.name || subject.definition)
    : [];
  data.shots = Array.isArray(data.shots) && data.shots.length
    ? data.shots.map((shot) => ({
        ...newShot(Number(shot?.duration || 0)),
        prompt: String(shot?.prompt || ""),
        camera: String(shot?.camera || ""),
        dialogue: String(shot?.dialogue || ""),
        sound: String(shot?.sound || ""),
      }))
    : [newShot(data.total_duration)];
  return data;
}

function serializeStoryboard(data) {
  return JSON.stringify(data);
}

function syncStoryboard(node, data) {
  return api.fetchApi("/minimax-h3/storyboard", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ node_id: String(node.id), data }),
  }).catch((error) => {
    console.error("MiniMax H3 Storyboard sync error", error);
  });
}

function formatTime(seconds) {
  const safe = Math.max(0, Number(seconds) || 0);
  let minutes = Math.floor(safe / 60);
  let whole = Math.floor(safe % 60);
  let millis = Math.round((safe - Math.floor(safe)) * 1000);
  if (millis >= 1000) {
    millis = 0;
    whole += 1;
  }
  if (whole >= 60) {
    minutes += 1;
    whole -= 60;
  }
  return `${String(minutes).padStart(2, "0")}:${String(whole).padStart(2, "0")}.${String(millis).padStart(3, "0")}`;
}

function formatTimeShort(seconds) {
  const safe = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(safe / 60);
  const whole = Math.floor(safe % 60);
  return `${String(minutes).padStart(2, "0")}:${String(whole).padStart(2, "0")}`;
}

function escapeRegExp(value) {
  return String(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function normalizeMediaAnchors(text) {
  if (!text) {
    return text;
  }
  const canonical = {
    picture: "Picture",
    video: "Video",
    audio: "Audio",
    subject: "Subject",
  };
  return String(text).replace(
    /<(picture|video|audio|subject)\s*(\d+)>/gi,
    (match, kind, number) => `<${canonical[kind.toLowerCase()]} ${number}>`
  );
}

function newSubject() {
  return { name: "", definition: "", label: "" };
}

function packageDataFromProperties(node) {
  try {
    return JSON.parse(node?.properties?.package_data || "{}");
  } catch (error) {
    console.error("MiniMax H3 PackageData parse error", error);
    return {};
  }
}

function labelsFromPackageData(data) {
  return {
    images: (data?.images || []).map((item, index) => item?.label || `<Picture ${index + 1}>`),
    videos: (data?.videos || []).map((item, index) => item?.label || `<Video ${index + 1}>`),
    audios: (data?.audios || []).map((item, index) => item?.label || `<Audio ${index + 1}>`),
  };
}

function replaceSubjectRefs(text, subjects, used, inlineDefinitions) {
  if (!text || !subjects?.length) {
    return normalizeMediaAnchors(text);
  }
  text = normalizeMediaAnchors(text);
  const refs = [];
  for (const subject of subjects) {
    const name = String(subject.name || "").trim();
    const label = subject.label;
    if (name) {
      refs.push({ raw: name, label, subject });
    }
    if (label) {
      refs.push({ raw: label, label, subject });
    }
  }
  if (!refs.length) {
    return text;
  }
  refs.sort((a, b) => b.raw.length - a.raw.length);
  const pattern = new RegExp(refs
    .map((item) => `(?<!\\w)${escapeRegExp(item.raw)}(?!\\w)`)
    .join("|"), "gi");
  return String(text).replace(pattern, (value) => {
    for (const item of refs) {
      if (value.toLowerCase() === item.raw.toLowerCase()) {
        const name = String(item.subject.name || "").trim();
        const definition = String(item.subject.definition || "").trim();
        const first = !used.has(item.label);
        if (first) {
          used.add(item.label);
        }
        const anchored = name && value.toLowerCase() === name.toLowerCase()
          ? `${name} (${item.label})`
          : item.label;
        if (inlineDefinitions && first && definition) {
          return `${anchored}, ${definition}`;
        }
        return anchored;
      }
    }
    return value;
  });
}

function normalizeDialogueTag(tag) {
  const match = String(tag).match(/<d>\s*(\[[A-Za-z-]+\])\s*([\s\S]*?)<\/d>/i);
  if (!match) {
    return tag;
  }
  return `<d>${match[1]}${match[2].trim()}</d>`;
}

function replaceOutsideDialogue(text, subjects, used, inlineDefinitions) {
  if (!text) {
    return text;
  }
  text = String(text).replace(/\bsays\s*[：:]\s*/gi, "says: ");
  const pattern = /<d>[\s\S]*?<\/d>/gi;
  if (!pattern.test(text)) {
    return replaceSubjectRefs(text, subjects, used, inlineDefinitions);
  }
  pattern.lastIndex = 0;
  const parts = [];
  let cursor = 0;
  for (const match of text.matchAll(pattern)) {
    parts.push(replaceSubjectRefs(
      text.slice(cursor, match.index), subjects, used, inlineDefinitions));
    parts.push(normalizeDialogueTag(match[0]));
    cursor = match.index + match[0].length;
  }
  parts.push(replaceSubjectRefs(
    text.slice(cursor), subjects, used, inlineDefinitions));
  return parts.join("");
}

function subjectDefinitionText(data, mediaLabels) {
  const lines = [];
  data.subjects.forEach((subject, index) => {
    if (!subject.name && !subject.definition) {
      return;
    }
    const label = subject.label || `<Subject ${index + 1}>`;
    const name = normalizeMediaAnchors(subject.name || "");
    const definition = normalizeMediaAnchors(subject.definition || "");
    if (name && definition) {
      lines.push(`${label} is ${name}, ${definition}.`);
    } else if (name) {
      lines.push(`${label} is ${name}.`);
    } else {
      lines.push(`${label} is ${definition}.`);
    }
  });
  const cited = new Set();
  for (const subject of data.subjects) {
    if (!subject.name && !subject.definition) {
      continue;
    }
    const text = normalizeMediaAnchors(`${subject.name} ${subject.definition}`);
    for (const match of text.matchAll(/<(?:Picture|Video|Audio) \d+>/g)) {
      cited.add(match[0]);
    }
  }
  for (const kind of ["images", "videos", "audios"]) {
    for (const label of mediaLabels?.[kind] || []) {
      if (!cited.has(label)) {
        lines.push(`${label} is a reference media source.`);
      }
    }
  }
  return lines.length ? lines.join("\n") : "N/A";
}

function compilePreview(data, mediaLabels) {
  const shots = data.shots;
  if (!shots.length) {
    return { ok: false, error: "At least one shot is required." };
  }
  const total = Number(data.total_duration) || 0;
  if (total <= 0) {
    return { ok: false, error: "Total duration must be positive." };
  }

  const parts = [];
  const starts = [];
  let cursor = 0;
  const usedSubjects = new Set();
  const inlineDefinitions = data.mode !== "full_reference";
  for (let index = 0; index < shots.length; index++) {
    const shot = shots[index];
    starts.push(cursor);
    const duration = Number(shot.duration) || 0;
    if (duration <= 0 && index !== shots.length - 1) {
      return { ok: false, error: `Shot ${index + 1} needs a positive duration.` };
    }
    if (cursor + (duration || (total - cursor)) > total + 1e-6) {
      return { ok: false, error: `Shot ${index + 1} ends after total duration.` };
    }

    const prompt = [
      replaceSubjectRefs(shot.prompt, data.subjects, usedSubjects, inlineDefinitions),
      replaceSubjectRefs(shot.camera, data.subjects, usedSubjects, inlineDefinitions),
      replaceOutsideDialogue(shot.dialogue, data.subjects, usedSubjects, inlineDefinitions),
      replaceSubjectRefs(shot.sound, data.subjects, usedSubjects, inlineDefinitions),
    ].filter(Boolean).join(" ").trim();
    if (!prompt) {
      return { ok: false, error: `Shot ${index + 1} prompt is empty.` };
    }
    const start = starts[index];
    parts.push(index === 0
      ? `[Shot 1] ${prompt}`
      : `[Shot ${index + 1}] At ${formatTime(start)}, ${prompt}`);
    cursor += duration || (total - cursor);
  }

  const body = parts.join(" ");
  const soundscape = data.soundscape || "N/A";
  const music = data.music_style || "N/A";
  const core = [
    `integrated_multimodal_description: ${body}`,
    `overall_soundscape: ${soundscape}`,
    `non_diegetic_music: ${music}`,
  ].join("\n\n");

  let text = core;
  if (data.mode === "full_reference") {
    text = [
      "subject_definitions:",
      subjectDefinitionText(data, mediaLabels),
      "",
      "summary:",
      "[reference generation] The target video follows the shot-by-shot storyboard.",
      "",
      "retention_analysis:",
      "N/A",
      "",
      `detailed_description: ${body}`,
      "",
      `overall_soundscape: ${soundscape}`,
      "",
      `non_diegetic_music: ${music}`,
    ].join("\n");
  }
  return { ok: true, text };
}

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

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function highlightCompiled(text) {
  return escapeHtml(text)
    .replace(/&lt;(Picture|Video|Audio|Subject) (\d+)&gt;/g, (match, kind, number) =>
      `<span class="mmh3-sb-tok k-${kind.toLowerCase()}">&lt;${kind} ${number}&gt;</span>`)
    .replace(/\[Shot (\d+)\]/g, '<span class="mmh3-sb-shot-head">[Shot $1]</span>');
}

function rulerStep(total, width) {
  const candidates = [0.5, 1, 2, 5];
  const minPx = 54;
  for (const step of candidates) {
    if ((total / step) * minPx <= width) {
      return step;
    }
  }
  return 10;
}

function setupStoryboard(node) {
  node.properties = node.properties || {};
  if (node.__h3StoryState) {
    node.__h3StoryState.setData(node.properties.storyboard_data);
    return;
  }

  let data = parseStoryboard(node.properties.storyboard_data);
  let selectedShot = 0;
  let mediaSourceNodeId = node.properties.mmh3_media_source || null;
  let mediaLabels = { images: [], videos: [], audios: [] };
  let syncTimer = null;
  let insertTarget = null;

  /* ---------- skeleton ---------- */
  const root = el("div", "mmh3-sb-root");

  const header = el("div", "mmh3-sb-header");
  const modeSeg = el("div", "mmh3-sb-seg");
  const modeButtons = MODES.map((mode) => {
    const button = el("button", "", mode === "full_reference" ? "Full Ref" : mode);
    button.dataset.mode = mode;
    button.title = mode;
    modeSeg.append(button);
    return button;
  });
  const ratioField = el("div", "mmh3-sb-hfield", "Ratio");
  const ratioSelect = document.createElement("select");
  RATIOS.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    ratioSelect.append(option);
  });
  ratioField.append(ratioSelect);
  const fpsField = el("div", "mmh3-sb-hfield", "FPS");
  const fpsInput = document.createElement("input");
  fpsInput.type = "number";
  fpsInput.min = "1";
  fpsInput.max = "60";
  fpsField.append(fpsInput);
  const totalField = el("div", "mmh3-sb-hfield", "Total");
  const totalInput = document.createElement("input");
  totalInput.type = "number";
  totalInput.min = "1";
  totalInput.max = "15";
  totalInput.step = "0.5";
  totalField.append(totalInput, el("span", "", "s"));
  const headerSpacer = el("div", "mmh3-sb-spacer");
  const pill = el("div", "mmh3-sb-pill ok");
  header.append(modeSeg, ratioField, fpsField, totalField, headerSpacer, pill);

  const timeline = el("div", "mmh3-sb-timeline");
  const tlHead = el("div", "mmh3-sb-tl-head");
  const tlTotal = el("span", "mmh3-sb-tl-total");
  tlHead.append(
    el("span", "mmh3-sb-tl-label", "Timeline"),
    tlTotal,
    el("span", "mmh3-sb-tl-hint", "← → adjusts selected shot"),
  );
  const ruler = el("div", "mmh3-sb-ruler");
  const track = el("div", "mmh3-sb-track");
  const tlWarn = el("div", "mmh3-sb-tl-warn");
  tlWarn.innerHTML = ICONS.warn;
  const tlWarnText = el("span");
  tlWarn.append(tlWarnText);
  timeline.append(tlHead, ruler, track, tlWarn);

  const main = el("div", "mmh3-sb-main");
  const editor = el("div", "mmh3-sb-editor mmh3-sb-scroll");

  const sidebar = el("div", "mmh3-sb-sidebar");
  const tabs = el("div", "mmh3-sb-tabs");
  const globalTab = el("button", "on", "Global");
  const subjectsTab = el("button", "", "Subjects");
  tabs.append(globalTab, subjectsTab);
  const sideBody = el("div", "mmh3-sb-side-body mmh3-sb-scroll");

  const subjectsPane = el("div", "mmh3-sb-pane");
  const mediaRow = el("div", "mmh3-sb-media-row", "Media source");
  const mediaSourceSelect = document.createElement("select");
  const mediaRefreshBtn = el("button", "mmh3-sb-icon");
  mediaRefreshBtn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12a9 9 0 1 1-2.64-6.36M21 3v6h-6"/></svg>';
  mediaRefreshBtn.title = "Refresh media labels";
  mediaRefreshBtn.style.flex = "none";
  mediaRow.append(mediaSourceSelect, mediaRefreshBtn);
  const mediaNone = el("div", "mmh3-sb-media-none", "No PackageData node in this graph.");
  const subjectList = el("div");
  subjectList.style.cssText = "display:flex;flex-direction:column;gap:8px;";
  const addSubjectBtn = el("button", "mmh3-sb-ghost", "+ Add Subject");
  subjectsPane.append(mediaRow, mediaNone, subjectList, addSubjectBtn);

  const globalPane = el("div", "mmh3-sb-pane on");
  const negField = el("div", "mmh3-sb-gfield");
  negField.append(el("label", "", "NEGATIVE PROMPT"));
  const negativeInput = document.createElement("textarea");
  negativeInput.placeholder = "low quality, watermark…";
  negField.append(negativeInput);
  const soundField = el("div", "mmh3-sb-gfield");
  soundField.append(el("label", "", "OVERALL SOUNDSCAPE"));
  const soundInput = document.createElement("textarea");
  soundInput.placeholder = "rainy city street ambience…";
  soundField.append(soundInput);
  const musicField = el("div", "mmh3-sb-gfield");
  musicField.append(el("label", "", "NON-DIEGETIC MUSIC"));
  const musicInput = document.createElement("textarea");
  musicInput.placeholder = "melancholic piano";
  musicField.append(musicInput);
  globalPane.append(negField, soundField, musicField);

  sideBody.append(subjectsPane, globalPane);
  sidebar.append(tabs, sideBody);
  main.append(editor, sidebar);

  const previewBar = el("div", "mmh3-sb-preview");
  const previewToggle = el("button", "mmh3-sb-preview-toggle");
  previewToggle.innerHTML = ICONS.chev;
  previewToggle.append(
    el("span", "", "COMPILED PROMPT PREVIEW"),
    el("span", "mmh3-sb-spacer"),
  );
  const copyBtn = el("span", "mmh3-sb-copy", "Copy");
  previewToggle.append(copyBtn);
  const previewBody = el("div", "mmh3-sb-preview-body mmh3-sb-scroll");
  const previewPre = document.createElement("pre");
  previewBody.append(previewPre);
  previewBar.append(previewToggle, previewBody);

  root.append(header, timeline, main, previewBar);

  /* chips always insert into the last focused text field of this node */
  root.addEventListener("focusin", (event) => {
    if (event.target?.__mmh3InsertApply) {
      insertTarget = { el: event.target, apply: event.target.__mmh3InsertApply };
    }
  });

  /* ---------- state helpers ---------- */
  const shotStarts = () => {
    const starts = [];
    let cursor = 0;
    for (const shot of data.shots) {
      starts.push(cursor);
      cursor += Number(shot.duration) || 0;
    }
    return starts;
  };
  const shotSum = () => data.shots
    .reduce((sum, shot) => sum + (Number(shot.duration) || 0), 0);
  const relabelSubjects = () => {
    data.subjects.forEach((subject, index) => {
      subject.label = `<Subject ${index + 1}>`;
    });
  };

  const persist = () => {
    node.properties.storyboard_data = serializeStoryboard(data);
    clearTimeout(syncTimer);
    syncTimer = setTimeout(() => syncStoryboard(node, data), 250);
  };

  /* ---------- timeline ---------- */
  const renderRuler = () => {
    ruler.replaceChildren();
    const total = Number(data.total_duration) || 0;
    if (total <= 0) {
      return;
    }
    const width = ruler.clientWidth || root.clientWidth - 36 || 700;
    const step = rulerStep(total, width);
    for (let t = 0; t <= total + 1e-6; t += step) {
      const tick = el("span", "mmh3-sb-tick", formatTimeShort(t));
      tick.style.left = `${(t / total) * 100}%`;
      ruler.append(tick);
    }
  };

  const renderTotal = () => {
    const sum = shotSum();
    const total = Number(data.total_duration) || 0;
    const ok = Math.abs(sum - total) < 0.05;
    tlTotal.className = `mmh3-sb-tl-total${ok ? "" : " mismatch"}`;
    tlTotal.replaceChildren();
    const bold = el("b", "", `${sum.toFixed(1)}s`);
    tlTotal.append(bold, ` / ${total.toFixed(1)}s`);
    tlWarn.classList.toggle("show", !ok);
    tlWarnText.textContent = ok ? "" : sum > total
      ? `Shots exceed total duration by ${(sum - total).toFixed(1)}s — the last shot will overflow.`
      : `${(total - sum).toFixed(1)}s unused — the last shot will stretch to fill it.`;
  };

  const renderTrack = () => {
    track.replaceChildren();
    const total = Number(data.total_duration) || 0;
    const starts = shotStarts();
    data.shots.forEach((shot, index) => {
      const block = el("div", "mmh3-sb-shot");
      block.classList.toggle("sel", index === selectedShot);
      block.classList.toggle("invalid", !String(shot.prompt || "").trim());
      block.style.flexBasis = total > 0
        ? `${Math.max(2, ((Number(shot.duration) || 0) / total) * 100)}%`
        : "10%";
      block.append(
        el("div", "mmh3-sb-shot-title", `Shot ${index + 1}`),
        el("div", "mmh3-sb-shot-sub",
          `${formatTimeShort(starts[index])} · ${(Number(shot.duration) || 0).toFixed(1)}s`),
      );
      const warnIcon = el("span", "mmh3-sb-shot-warn");
      warnIcon.innerHTML = ICONS.warn;
      warnIcon.title = "Prompt is empty";
      block.append(warnIcon);

      block.onclick = () => {
        selectedShot = index;
        renderAll();
      };
      track.append(block);
    });

    const add = el("button", "mmh3-sb-add", "+");
    add.title = "Add shot (splits the unfilled shot in half, or fills remaining time)";
    add.onclick = () => {
      const isEmpty = (shot) => !String(shot?.prompt || "").trim();
      let target = isEmpty(data.shots[selectedShot]) ? selectedShot : -1;
      if (target < 0) {
        target = data.shots.findIndex(isEmpty);
      }
      const half = target >= 0
        ? Math.round(((Number(data.shots[target].duration) || 0) / 2) * 10) / 10
        : 0;
      if (target >= 0 && half >= 0.1) {
        data.shots[target].duration = half;
        data.shots.splice(target + 1, 0, newShot(half));
        selectedShot = target + 1;
      } else {
        const remaining = Math.max(0.5,
          Math.round(((Number(data.total_duration) || 0) - shotSum()) * 10) / 10);
        data.shots.push(newShot(remaining));
        selectedShot = data.shots.length - 1;
      }
      persist();
      renderAll();
    };
    track.append(add);
  };

  /* ---------- editor ---------- */
  const makeChip = (label, colorClass, dotColor, display) => {
    const chip = el("button", `mmh3-sb-chip${colorClass ? ` ${colorClass}` : ""}`);
    const dot = el("span", "mmh3-sb-dot");
    if (dotColor) {
      dot.style.background = dotColor;
    }
    chip.append(dot, el("span", "mmh3-sb-chip-label", display || label));
    chip.dataset.label = label;
    chip.title = `Insert ${label}`;
    return chip;
  };

  const renderEditor = () => {
    editor.replaceChildren();
    const index = selectedShot;
    const shot = data.shots[index];
    if (!shot) {
      editor.append(el("div", "mmh3-sb-empty", "Select a shot on the timeline"));
      return;
    }
    const starts = shotStarts();

    const head = el("div", "mmh3-sb-ed-head");
    head.append(
      el("span", "mmh3-sb-ed-title", `Shot ${index + 1}`),
      el("span", "mmh3-sb-ed-range",
        `${formatTimeShort(starts[index])} → ${formatTimeShort(starts[index] + (Number(shot.duration) || 0))}`),
      el("div", "mmh3-sb-spacer"),
    );
    const durField = el("div", "mmh3-sb-hfield", "Duration");
    const durInput = document.createElement("input");
    durInput.type = "number";
    durInput.step = "0.1";
    durInput.min = "0.1";
    durInput.value = shot.duration;
    durField.append(durInput, el("span", "", "s"));
    const moveLeft = el("button", "mmh3-sb-icon");
    moveLeft.innerHTML = ICONS.left;
    moveLeft.title = "Move left";
    moveLeft.disabled = index === 0;
    const moveRight = el("button", "mmh3-sb-icon");
    moveRight.innerHTML = ICONS.right;
    moveRight.title = "Move right";
    moveRight.disabled = index === data.shots.length - 1;
    const deleteShot = el("button", "mmh3-sb-icon danger");
    deleteShot.innerHTML = ICONS.trash;
    deleteShot.title = "Delete shot";
    deleteShot.disabled = data.shots.length <= 1;
    head.append(durField, moveLeft, moveRight, deleteShot);

    const chips = [
      ...mediaLabels.images.map((label) => makeChip(label, "k-picture", "", label.replace(/[<>]/g, ""))),
      ...mediaLabels.videos.map((label) => makeChip(label, "k-video", "", label.replace(/[<>]/g, ""))),
      ...mediaLabels.audios.map((label) => makeChip(label, "k-audio", "", label.replace(/[<>]/g, ""))),
      ...data.subjects
        .map((subject, subjectIndex) => ({ subject, subjectIndex }))
        .filter(({ subject }) => subject.name || subject.definition)
        .map(({ subject, subjectIndex }) => makeChip(
          subject.name || subject.label || `<Subject ${subjectIndex + 1}>`,
          "",
          SUBJECT_COLORS[subjectIndex % SUBJECT_COLORS.length],
          subject.name || subject.label || `<Subject ${subjectIndex + 1}>`,
        )),
    ];
    const chipbar = el("div", "mmh3-sb-chipbar");
    [
      chips.filter((chip) =>
        !chip.classList.contains("k-picture")
        && !chip.classList.contains("k-video")
        && !chip.classList.contains("k-audio")),
      chips.filter((chip) => chip.classList.contains("k-picture")),
      chips.filter((chip) => chip.classList.contains("k-video")),
      chips.filter((chip) => chip.classList.contains("k-audio")),
    ].filter((groupChips) => groupChips.length)
      .forEach((groupChips) => {
        const row = el("div", "mmh3-sb-chiprow");
        groupChips.forEach((chip) => row.append(chip));
        chipbar.append(row);
      });

    const prompt = document.createElement("textarea");
    prompt.className = "mmh3-sb-prompt";
    prompt.placeholder = "Describe what happens in this shot…";
    prompt.value = shot.prompt || "";
    prompt.__mmh3InsertApply = (value) => {
      shot.prompt = value;
      persist();
      renderTrack();
      renderTotal();
      updatePreview();
    };

    const fields = el("div", "mmh3-sb-fields");
    const cameraCol = el("div", "mmh3-sb-fcol");
    cameraCol.append(el("label", "", "CAMERA"));
    const cameraInput = document.createElement("textarea");
    cameraInput.placeholder = "e.g. slow dolly-in";
    cameraInput.value = shot.camera || "";
    cameraCol.append(cameraInput);
    cameraInput.__mmh3InsertApply = (value) => {
      shot.camera = value;
      persist();
      updatePreview();
    };
    const dialogueCol = el("div", "mmh3-sb-fcol");
    dialogueCol.append(el("label", "", "DIALOGUE"));
    const dialogueInput = document.createElement("textarea");
    dialogueInput.placeholder = "Lena says: <d>[Chinese] your line</d>";
    dialogueInput.value = shot.dialogue || "";
    dialogueCol.append(dialogueInput);
    dialogueInput.__mmh3InsertApply = (value) => {
      shot.dialogue = value;
      persist();
      updatePreview();
    };
    const soundCol = el("div", "mmh3-sb-fcol");
    soundCol.append(el("label", "", "DIEGETIC SOUND"));
    const shotSoundInput = document.createElement("textarea");
    shotSoundInput.placeholder = "e.g. rain, distant siren";
    shotSoundInput.value = shot.sound || "";
    soundCol.append(shotSoundInput);
    shotSoundInput.__mmh3InsertApply = (value) => {
      shot.sound = value;
      persist();
      updatePreview();
    };
    fields.append(cameraCol, dialogueCol, soundCol);

    editor.append(head, chipbar, prompt, fields);

    chips.forEach((chip) => {
      chip.onclick = () => {
        const label = chip.dataset.label;
        const target = insertTarget && root.contains(insertTarget.el)
          ? insertTarget
          : { el: prompt, apply: prompt.__mmh3InsertApply };
        const start = target.el.selectionStart ?? target.el.value.length;
        const end = target.el.selectionEnd ?? start;
        const before = target.el.value.slice(0, start);
        const after = target.el.value.slice(end);
        const insertion = before && !/\s$/.test(before) ? ` ${label}` : label;
        target.el.value = `${before}${insertion}${after}`;
        const caret = start + insertion.length;
        target.el.focus();
        target.el.setSelectionRange(caret, caret);
        target.apply(target.el.value);
      };
    });

    prompt.oninput = () => {
      shot.prompt = prompt.value;
      persist();
      renderTrack();
      renderTotal();
      updatePreview();
    };
    cameraInput.oninput = () => {
      shot.camera = cameraInput.value;
      persist();
      updatePreview();
    };
    dialogueInput.oninput = () => {
      shot.dialogue = dialogueInput.value;
      persist();
      updatePreview();
    };
    shotSoundInput.oninput = () => {
      shot.sound = shotSoundInput.value;
      persist();
      updatePreview();
    };
    durInput.onchange = () => {
      shot.duration = Math.max(0.1, Number(durInput.value) || 0.1);
      persist();
      renderTrack();
      renderTotal();
      renderEditor();
      updatePreview();
    };
    moveLeft.onclick = () => {
      [data.shots[index - 1], data.shots[index]] = [data.shots[index], data.shots[index - 1]];
      selectedShot = index - 1;
      persist();
      renderAll();
    };
    moveRight.onclick = () => {
      [data.shots[index + 1], data.shots[index]] = [data.shots[index], data.shots[index + 1]];
      selectedShot = index + 1;
      persist();
      renderAll();
    };
    deleteShot.onclick = () => {
      if (data.shots.length <= 1) {
        return;
      }
      data.shots.splice(index, 1);
      selectedShot = Math.max(0, index - 1);
      persist();
      renderAll();
    };
  };

  /* ---------- subjects ---------- */
  const renderSubjects = () => {
    subjectList.replaceChildren();
    data.subjects.forEach((subject, index) => {
      const color = SUBJECT_COLORS[index % SUBJECT_COLORS.length];
      const card = el("div", "mmh3-sb-subject");
      const top = el("div", "mmh3-sb-subject-top");
      const dot = el("span", "mmh3-sb-dot");
      dot.style.background = color;
      const label = el("span", "mmh3-sb-subject-label", `<Subject ${index + 1}>`);
      label.style.color = color;
      const name = document.createElement("input");
      name.placeholder = "Name (e.g. Lena)";
      name.value = subject.name || "";
      name.__mmh3InsertApply = (value) => {
        subject.name = value.trim();
        persist();
        renderEditor();
        updatePreview();
      };
      const remove = el("button", "mmh3-sb-icon danger");
      remove.innerHTML = ICONS.close;
      remove.title = "Remove subject";
      top.append(dot, label, name, remove);
      const definition = document.createElement("textarea");
      definition.placeholder = "Visual definition — appearance, clothing, reference tokens like <Picture 1>…";
      definition.value = subject.definition || "";
      definition.__mmh3InsertApply = (value) => {
        subject.definition = value.trim();
        persist();
        renderEditor();
        updatePreview();
      };
      card.append(top, definition);

      name.oninput = () => {
        subject.name = name.value.trim();
        persist();
        renderEditor();
        updatePreview();
      };
      definition.oninput = () => {
        subject.definition = definition.value.trim();
        persist();
        renderEditor();
        updatePreview();
      };
      remove.onclick = () => {
        data.subjects.splice(index, 1);
        relabelSubjects();
        persist();
        renderSubjects();
        renderEditor();
        updatePreview();
      };
      subjectList.append(card);
    });
  };

  /* ---------- preview & validation ---------- */
  const updatePreview = () => {
    const result = compilePreview(data, mediaLabels);
    previewPre.innerHTML = result.ok
      ? highlightCompiled(result.text)
      : `<span style="color:var(--sb-danger)">${escapeHtml(result.error)}</span>`;
    pill.className = `mmh3-sb-pill ${result.ok ? "ok" : "bad"}`;
    pill.innerHTML = result.ok ? ICONS.check : ICONS.alert;
    pill.append(el("span", "", result.ok ? "Ready to compile" : result.error));
    pill.title = result.ok ? "" : result.error;
  };

  /* ---------- bindings ---------- */
  const bindGlobal = () => {
    modeButtons.forEach((button) => {
      button.classList.toggle("on", button.dataset.mode === data.mode);
      button.onclick = () => {
        data.mode = button.dataset.mode;
        modeButtons.forEach((item) => item.classList.toggle("on", item === button));
        persist();
        updatePreview();
      };
    });
    ratioSelect.value = data.ratio;
    ratioSelect.onchange = () => {
      data.ratio = ratioSelect.value;
      persist();
    };
    fpsInput.value = data.fps;
    fpsInput.onchange = () => {
      data.fps = Number(fpsInput.value) || 24;
      persist();
    };
    totalInput.value = data.total_duration;
    totalInput.onchange = () => {
      data.total_duration = Math.max(1, Number(totalInput.value) || 1);
      persist();
      renderRuler();
      renderTrack();
      renderTotal();
      renderEditor();
      updatePreview();
    };
    negativeInput.value = data.negative_prompt;
    negativeInput.__mmh3InsertApply = (value) => {
      data.negative_prompt = value;
      persist();
    };
    negativeInput.oninput = () => {
      data.negative_prompt = negativeInput.value;
      persist();
    };
    soundInput.value = data.soundscape;
    soundInput.__mmh3InsertApply = (value) => {
      data.soundscape = value;
      persist();
      updatePreview();
    };
    soundInput.oninput = () => {
      data.soundscape = soundInput.value;
      persist();
      updatePreview();
    };
    musicInput.value = data.music_style;
    musicInput.__mmh3InsertApply = (value) => {
      data.music_style = value;
      persist();
      updatePreview();
    };
    musicInput.oninput = () => {
      data.music_style = musicInput.value;
      persist();
      updatePreview();
    };
    addSubjectBtn.onclick = () => {
      data.subjects.push(newSubject());
      relabelSubjects();
      persist();
      renderSubjects();
      updatePreview();
    };
  };

  subjectsTab.onclick = () => {
    subjectsTab.classList.add("on");
    globalTab.classList.remove("on");
    subjectsPane.classList.add("on");
    globalPane.classList.remove("on");
  };
  globalTab.onclick = () => {
    globalTab.classList.add("on");
    subjectsTab.classList.remove("on");
    globalPane.classList.add("on");
    subjectsPane.classList.remove("on");
  };
  previewToggle.onclick = (event) => {
    if (event.target === copyBtn) {
      return;
    }
    previewBar.classList.toggle("open");
  };
  copyBtn.onclick = (event) => {
    event.stopPropagation();
    const result = compilePreview(data, mediaLabels);
    if (result.ok) {
      navigator.clipboard?.writeText(result.text).catch(() => {});
      copyBtn.textContent = "Copied";
      setTimeout(() => {
        copyBtn.textContent = "Copy";
      }, 1200);
    }
  };

  /* ---------- media source ---------- */
  const getPackageNodes = () => (node.graph?.nodes || app.graph?.nodes || [])
    .filter((item) => item.type === "MiniMaxH3PackageData");

  const refreshMediaLabels = async () => {
    const nodes = getPackageNodes();
    mediaSourceSelect.replaceChildren();
    const hasNodes = nodes.length > 0;
    mediaRow.style.display = hasNodes ? "" : "none";
    mediaNone.style.display = hasNodes ? "none" : "";
    if (!hasNodes) {
      mediaSourceNodeId = null;
      mediaLabels = { images: [], videos: [], audios: [] };
      renderEditor();
      updatePreview();
      return;
    }
    nodes.forEach((item, index) => {
      const option = document.createElement("option");
      option.value = String(item.id);
      option.textContent = item.title || `PackageData ${index + 1}`;
      mediaSourceSelect.append(option);
    });
    if (!nodes.some((item) => String(item.id) === String(mediaSourceNodeId))) {
      mediaSourceNodeId = String(nodes[0].id);
      node.properties.mmh3_media_source = mediaSourceNodeId;
    }
    mediaSourceSelect.value = mediaSourceNodeId;
    const source = nodes.find((item) => String(item.id) === String(mediaSourceNodeId));
    let packageData = packageDataFromProperties(source);
    if (!packageData?.images?.length && !packageData?.videos?.length && !packageData?.audios?.length) {
      try {
        const response = await api.fetchApi(
          `/minimax-h3/package-data?node_id=${encodeURIComponent(mediaSourceNodeId)}`
        );
        if (response.ok) {
          packageData = await response.json();
        }
      } catch (error) {
        console.error("MiniMax H3 Storyboard package refresh error", error);
      }
    }
    mediaLabels = labelsFromPackageData(packageData);
    renderEditor();
    updatePreview();
  };
  mediaSourceSelect.onchange = () => {
    mediaSourceNodeId = mediaSourceSelect.value;
    node.properties.mmh3_media_source = mediaSourceNodeId;
    refreshMediaLabels();
  };
  mediaRefreshBtn.onclick = () => refreshMediaLabels();

  /* ---------- render ---------- */
  const renderAll = () => {
    renderRuler();
    renderTrack();
    renderTotal();
    renderEditor();
    renderSubjects();
    updatePreview();
    node.graph?.setDirtyCanvas?.(true, true);
  };

  const render = () => {
    bindGlobal();
    renderAll();
    syncStoryboard(node, data);
  };

  const setData = (value) => {
    data = parseStoryboard(value);
    selectedShot = Math.min(selectedShot, data.shots.length - 1);
    render();
  };

  /* ---------- widget ---------- */
  const domWidget = node.addDOMWidget("storyboard_ui", "storyboard_ui", root, {
    serialize: false,
    hideOnZoom: false,
    getMinHeight: () => 720,
    getMaxHeight: () => 720,
  });
  domWidget.computeSize = function (width) {
    return [width, 720];
  };
  node.__h3StoryState = { setData, render };
  node.setSize?.([840, 740]);

  const resizeObserver = new ResizeObserver(() => {
    root.classList.toggle("compact", root.clientWidth > 0 && root.clientWidth < COMPACT_WIDTH);
    renderRuler();
  });
  resizeObserver.observe(root);

  /* arrow keys nudge the selected shot's duration while hovering the node */
  let rootHover = false;
  root.addEventListener("mouseenter", () => {
    rootHover = true;
  });
  root.addEventListener("mouseleave", () => {
    rootHover = false;
  });
  document.addEventListener("keydown", (event) => {
    if (!rootHover || (event.key !== "ArrowLeft" && event.key !== "ArrowRight")) {
      return;
    }
    const tag = event.target?.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") {
      return;
    }
    const shot = data.shots[selectedShot];
    if (!shot) {
      return;
    }
    event.preventDefault();
    const step = event.shiftKey ? 0.5 : 0.1;
    const delta = event.key === "ArrowRight" ? step : -step;
    shot.duration = Math.max(0.1,
      Math.round(((Number(shot.duration) || 0) + delta) * 10) / 10);
    persist();
    renderTrack();
    renderTotal();
    renderEditor();
    updatePreview();
  });

  api.addEventListener("minimax-h3/package-data-changed", refreshMediaLabels);
  render();
  refreshMediaLabels();
  /* node configure order is not guaranteed on workflow load; retry until
     the PackageData node's properties are available */
  [500, 1500, 3000].forEach((delay) => {
    setTimeout(() => {
      const empty = !mediaLabels.images.length
        && !mediaLabels.videos.length
        && !mediaLabels.audios.length;
      if (empty && getPackageNodes().length) {
        refreshMediaLabels();
      }
    }, delay);
  });
}

app.registerExtension({
  name: "ComfyUI-MiniMaxH3.Storyboard",
  setup() {
    const originalQueuePrompt = api.queuePrompt.bind(api);
    api.queuePrompt = async (...args) => {
      const nodes = app.graph?.nodes || [];
      await Promise.all(nodes
        .filter((node) => node.__h3StoryState && node.properties?.storyboard_data)
        .map((node) => syncStoryboard(node, parseStoryboard(node.properties.storyboard_data))));
      return originalQueuePrompt(...args);
    };
  },
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData?.name !== "MiniMaxH3Storyboard") {
      return;
    }
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      setupStoryboard(this);
      return result;
    };
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      setupStoryboard(this);
      return result;
    };
  },
});
