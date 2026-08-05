import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const style = document.createElement("style");
style.textContent = `
.mmh3-story-root {
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
  color: var(--input-text, #d5d8de);
}
.mmh3-story-topbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}
.mmh3-story-field {
  display: flex;
  align-items: center;
  gap: 6px;
}
.mmh3-story-field label {
  opacity: 0.78;
}
.mmh3-story-select,
.mmh3-story-number,
.mmh3-story-text {
  height: 28px;
  box-sizing: border-box;
  border: 1px solid var(--border-color, #525a68);
  border-radius: 8px;
  background: var(--comfy-input-bg, #262a32);
  color: var(--input-text, #d5d8de);
  padding: 0 8px;
}
.mmh3-story-number {
  width: 76px;
}
.mmh3-story-text {
  width: 160px;
}
.mmh3-story-add {
  margin-left: auto;
  min-height: 28px;
  border: 1px solid var(--border-color, #525a68);
  border-radius: 8px;
  background: var(--comfy-input-bg, #262a32);
  color: var(--input-text, #d5d8de);
  cursor: pointer;
  padding: 0 12px;
}
.mmh3-story-global {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}
.mmh3-story-global textarea {
  width: 100%;
  height: 44px;
  resize: none;
  box-sizing: border-box;
  border: 1px solid var(--border-color, #525a68);
  border-radius: 8px;
  background: var(--comfy-input-bg, #262a32);
  color: var(--input-text, #d5d8de);
  padding: 6px 8px;
}
.mmh3-story-shots {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
  overflow-y: auto;
  scrollbar-width: thin;
  padding: 2px;
}
.mmh3-story-subjects {
  flex: none;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.mmh3-story-subjects-header {
  display: flex;
  align-items: center;
  gap: 8px;
}
.mmh3-story-section-title {
  font-weight: 600;
}
.mmh3-story-subjects-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
  max-height: 150px;
  overflow-y: auto;
}
.mmh3-story-subject {
  display: flex;
  align-items: center;
  gap: 6px;
}
.mmh3-story-subject-label {
  flex: none;
  min-width: 82px;
  opacity: 0.82;
}
.mmh3-story-subject-definition {
  flex: 1;
  min-width: 0;
  height: 26px;
  box-sizing: border-box;
  border: 1px solid var(--border-color, #525a68);
  border-radius: 8px;
  background: var(--comfy-input-bg, #262a32);
  color: var(--input-text, #d5d8de);
  padding: 0 6px;
}
.mmh3-story-start {
  opacity: 0.72;
  font-size: 11px;
}
.mmh3-story-media {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
}
.mmh3-story-chip {
  min-height: 22px;
  padding: 0 8px;
  border: 1px solid var(--border-color, #525a68);
  border-radius: 8px;
  background: var(--comfy-input-bg, #262a32);
  color: var(--input-text, #d5d8de);
  cursor: pointer;
}
.mmh3-story-shot {
  flex: none;
  display: flex;
  flex-direction: column;
  gap: 6px;
  box-sizing: border-box;
  border: 1px solid var(--border-color, #3a3f4b);
  border-radius: 10px;
  background: var(--comfy-input-bg, #171a20);
  padding: 8px;
}
.mmh3-story-shot-header {
  display: flex;
  align-items: center;
  gap: 8px;
}
.mmh3-story-shot-title {
  font-weight: 600;
}
.mmh3-story-shot-actions {
  margin-left: auto;
  display: flex;
  gap: 6px;
}
.mmh3-story-button {
  min-width: 28px;
  height: 28px;
  border: 1px solid var(--border-color, #525a68);
  border-radius: 8px;
  background: var(--comfy-input-bg, #262a32);
  color: var(--input-text, #d5d8de);
  cursor: pointer;
}
.mmh3-story-prompt {
  width: 100%;
  min-height: 64px;
  resize: vertical;
  box-sizing: border-box;
  border: 1px solid var(--border-color, #525a68);
  border-radius: 8px;
  background: var(--comfy-input-bg, #1b1e24);
  color: var(--input-text, #d5d8de);
  padding: 6px 8px;
}
.mmh3-story-details {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 6px;
}
.mmh3-story-details input {
  width: 100%;
  height: 26px;
  box-sizing: border-box;
  border: 1px solid var(--border-color, #525a68);
  border-radius: 7px;
  background: var(--comfy-input-bg, #262a32);
  color: var(--input-text, #d5d8de);
  padding: 0 6px;
}
.mmh3-story-preview {
  flex: none;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.mmh3-story-preview textarea {
  width: 100%;
  height: 100px;
  resize: vertical;
  box-sizing: border-box;
  border: 1px solid var(--border-color, #525a68);
  border-radius: 8px;
  background: var(--comfy-input-bg, #1b1e24);
  color: var(--input-text, #d5d8de);
  padding: 6px 8px;
}
.mmh3-story-status {
  min-height: 16px;
  opacity: 0.82;
}
.mmh3-story-status.mmh3-story-error {
  color: #ff7b72;
}
`;
document.head.appendChild(style);

const MODES = ["T2VA", "full_reference"];
const RATIOS = ["adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"];

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

function setupStoryboard(node) {
  node.properties = node.properties || {};
  if (node.__h3StoryState) {
    node.__h3StoryState.setData(node.properties.storyboard_data);
    return;
  }

  let data = parseStoryboard(node.properties.storyboard_data);
  const root = document.createElement("div");
  const topbar = document.createElement("div");
  const modeField = document.createElement("div");
  const ratioField = document.createElement("div");
  const durationField = document.createElement("div");
  const modeLabel = document.createElement("label");
  const modeSelect = document.createElement("select");
  const ratioLabel = document.createElement("label");
  const ratioSelect = document.createElement("select");
  const durationLabel = document.createElement("label");
  const durationInput = document.createElement("input");
  const addButton = document.createElement("button");
  const globalFields = document.createElement("div");
  const negativeInput = document.createElement("textarea");
  const soundInput = document.createElement("textarea");
  const musicInput = document.createElement("input");
  const subjectsRoot = document.createElement("div");
  const subjectsHeader = document.createElement("div");
  const subjectsTitle = document.createElement("span");
  const mediaSourceLabel = document.createElement("label");
  const mediaSourceSelect = document.createElement("select");
  const addSubjectButton = document.createElement("button");
  const subjectsList = document.createElement("div");
  const shotsRoot = document.createElement("div");
  const previewRoot = document.createElement("div");
  const preview = document.createElement("textarea");
  const status = document.createElement("div");
  let mediaSourceNodeId = null;
  let mediaLabels = { images: [], videos: [], audios: [] };

  root.className = "mmh3-story-root";
  topbar.className = "mmh3-story-topbar";
  modeField.className = "mmh3-story-field";
  ratioField.className = "mmh3-story-field";
  durationField.className = "mmh3-story-field";
  globalFields.className = "mmh3-story-global";
  subjectsRoot.className = "mmh3-story-subjects";
  subjectsHeader.className = "mmh3-story-subjects-header";
  subjectsTitle.className = "mmh3-story-section-title";
  mediaSourceLabel.className = "mmh3-story-field";
  mediaSourceSelect.className = "mmh3-story-select";
  addSubjectButton.className = "mmh3-story-add";
  subjectsList.className = "mmh3-story-subjects-list";
  shotsRoot.className = "mmh3-story-shots";
  previewRoot.className = "mmh3-story-preview";
  addButton.className = "mmh3-story-add";
  preview.className = "mmh3-story-prompt";
  status.className = "mmh3-story-status";
  negativeInput.placeholder = "Negative prompt";
  soundInput.placeholder = "Overall soundscape";
  musicInput.placeholder = "Non-diegetic music style";
  musicInput.className = "mmh3-story-text";
  musicInput.style.width = "100%";

  MODES.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    modeSelect.append(option);
  });
  RATIOS.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    ratioSelect.append(option);
  });
  modeSelect.className = "mmh3-story-select";
  ratioSelect.className = "mmh3-story-select";
  durationInput.className = "mmh3-story-number";
  durationInput.type = "number";
  durationInput.min = "1";
  durationInput.max = "15";
  durationInput.step = "0.1";

  modeLabel.textContent = "Mode";
  ratioLabel.textContent = "Ratio";
  durationLabel.textContent = "Duration";
  subjectsTitle.textContent = "Subjects";
  mediaSourceLabel.textContent = "Media";
  addButton.textContent = "Add Shot";
  addSubjectButton.textContent = "Add Subject";
  modeField.append(modeLabel, modeSelect);
  ratioField.append(ratioLabel, ratioSelect);
  durationField.append(durationLabel, durationInput);
  topbar.append(modeField, ratioField, durationField, addButton);
  subjectsHeader.append(subjectsTitle, mediaSourceLabel, mediaSourceSelect, addSubjectButton);
  subjectsRoot.append(subjectsHeader, subjectsList);
  globalFields.append(
    negativeInput,
    soundInput,
    document.createTextNode(""),
    musicInput,
  );
  previewRoot.append(preview, status);
  root.append(topbar, globalFields, subjectsRoot, shotsRoot, previewRoot);

  const updatePreview = () => {
    const result = compilePreview(data, mediaLabels);
    preview.value = result.ok ? result.text : "";
    status.textContent = result.ok ? "Official prompt format is valid." : result.error;
    status.classList.toggle("mmh3-story-error", !result.ok);
  };

  const persist = () => {
    node.properties.storyboard_data = serializeStoryboard(data);
    syncStoryboard(node, data);
  };

  const insertLabel = (textarea, shot, label) => {
    const start = textarea.selectionStart ?? textarea.value.length;
    const end = textarea.selectionEnd ?? start;
    const prefix = textarea.value.slice(0, start);
    const suffix = textarea.value.slice(end);
    const insertion = prefix && !/\s$/.test(prefix) ? ` ${label}` : label;
    textarea.value = `${prefix}${insertion}${suffix}`;
    const caret = start + insertion.length;
    textarea.focus();
    textarea.setSelectionRange(caret, caret);
    shot.prompt = textarea.value;
    persist();
    updatePreview();
  };

  const renderSubjects = () => {
    subjectsList.replaceChildren();
    data.subjects.forEach((subject, index) => {
      const row = document.createElement("div");
      const label = document.createElement("span");
      const name = document.createElement("input");
      const definition = document.createElement("input");
      const remove = document.createElement("button");

      row.className = "mmh3-story-subject";
      label.className = "mmh3-story-subject-label";
      name.className = "mmh3-story-text";
      name.style.width = "150px";
      definition.className = "mmh3-story-subject-definition";
      remove.className = "mmh3-story-button";
      label.textContent = `<Subject ${index + 1}>`;
      name.value = subject.name || "";
      name.placeholder = "Name";
      definition.value = subject.definition || "";
      definition.placeholder = "Definition";
      remove.textContent = "x";

      name.oninput = () => {
        subject.name = name.value.trim();
        subject.label = `<Subject ${index + 1}>`;
        persist();
        updatePreview();
      };
      definition.oninput = () => {
        subject.definition = definition.value.trim();
        persist();
        updatePreview();
      };
      remove.onclick = () => {
        data.subjects.splice(index, 1);
        data.subjects.forEach((item, itemIndex) => {
          item.label = `<Subject ${itemIndex + 1}>`;
        });
        persist();
        renderSubjects();
        updatePreview();
      };

      row.append(label, name, definition, remove);
      subjectsList.append(row);
    });
  };

  const renderShots = () => {
    shotsRoot.replaceChildren();
    const starts = [];
    let cursor = 0;
    data.shots.forEach((shot, index) => {
      starts.push(cursor);
      const card = document.createElement("div");
      const header = document.createElement("div");
      const title = document.createElement("span");
      const actions = document.createElement("div");
      const up = document.createElement("button");
      const down = document.createElement("button");
      const remove = document.createElement("button");
      const duration = document.createElement("input");
      const startLabel = document.createElement("span");
      const mediaChips = document.createElement("div");
      const prompt = document.createElement("textarea");
      const details = document.createElement("div");
      const camera = document.createElement("input");
      const dialogue = document.createElement("input");
      const sound = document.createElement("input");

      card.className = "mmh3-story-shot";
      header.className = "mmh3-story-shot-header";
      title.className = "mmh3-story-shot-title";
      actions.className = "mmh3-story-shot-actions";
      up.className = "mmh3-story-button";
      down.className = "mmh3-story-button";
      remove.className = "mmh3-story-button";
      startLabel.className = "mmh3-story-start";
      mediaChips.className = "mmh3-story-media";
      prompt.className = "mmh3-story-prompt";
      details.className = "mmh3-story-details";
      duration.className = "mmh3-story-number";
      camera.className = "mmh3-story-text";
      dialogue.className = "mmh3-story-text";
      sound.className = "mmh3-story-text";
      duration.type = "number";
      duration.step = "0.1";

      title.textContent = `Shot ${index + 1}`;
      up.textContent = "^";
      down.textContent = "v";
      remove.textContent = "x";
      startLabel.textContent = `Start ${formatTime(starts[index])}`;
      duration.value = shot.duration;
      prompt.value = shot.prompt || "";
      camera.value = shot.camera || "";
      camera.placeholder = "Camera";
      dialogue.value = shot.dialogue || "";
      dialogue.placeholder = "Dialogue";
      sound.value = shot.sound || "";
      sound.placeholder = "Diegetic sound/music";

      const chips = [
        ...mediaLabels.images,
        ...mediaLabels.videos,
        ...mediaLabels.audios,
        ...data.subjects
          .filter((subject) => subject.name || subject.definition)
          .map((subject) => subject.label),
      ];
      for (const label of chips) {
        const chip = document.createElement("button");
        chip.className = "mmh3-story-chip";
        chip.textContent = label;
        chip.title = label;
        chip.onclick = () => insertLabel(prompt, shot, label);
        mediaChips.append(chip);
      }
      if (chips.length) {
        card.append(mediaChips);
      }

      duration.onchange = () => {
        shot.duration = Number(duration.value) || 0;
        persist();
        updatePreview();
      };
      prompt.oninput = () => {
        shot.prompt = prompt.value;
        persist();
        updatePreview();
      };
      camera.oninput = () => {
        shot.camera = camera.value;
        persist();
        updatePreview();
      };
      dialogue.oninput = () => {
        shot.dialogue = dialogue.value;
        persist();
        updatePreview();
      };
      sound.oninput = () => {
        shot.sound = sound.value;
        persist();
        updatePreview();
      };

      up.disabled = index === 0;
      down.disabled = index === data.shots.length - 1;
      up.onclick = () => {
        [data.shots[index - 1], data.shots[index]] = [data.shots[index], data.shots[index - 1]];
        persist();
        renderShots();
        updatePreview();
      };
      down.onclick = () => {
        [data.shots[index + 1], data.shots[index]] = [data.shots[index], data.shots[index + 1]];
        persist();
        renderShots();
        updatePreview();
      };
      remove.onclick = () => {
        if (data.shots.length <= 1) {
          return;
        }
        data.shots.splice(index, 1);
        persist();
        renderShots();
        updatePreview();
      };

      const startField = document.createElement("div");
      const durationFieldInner = document.createElement("div");
      const durationLabelInner = document.createElement("label");
      startField.className = "mmh3-story-field";
      durationFieldInner.className = "mmh3-story-field";
      durationLabelInner.textContent = "Duration";
      startField.append(startLabel);
      durationFieldInner.append(durationLabelInner, duration);
      header.append(title, startField, durationFieldInner, actions);
      actions.append(up, down, remove);
      details.append(camera, dialogue, sound);
      card.append(header, prompt, details);
      shotsRoot.append(card);
      cursor += Number(shot.duration) || 0;
    });
  };

  const bindGlobal = () => {
    modeSelect.value = data.mode;
    ratioSelect.value = data.ratio;
    durationInput.value = data.total_duration;
    negativeInput.value = data.negative_prompt;
    soundInput.value = data.soundscape;
    musicInput.value = data.music_style;

    modeSelect.onchange = () => {
      data.mode = modeSelect.value;
      persist();
      updatePreview();
    };
    ratioSelect.onchange = () => {
      data.ratio = ratioSelect.value;
      persist();
    };
    durationInput.onchange = () => {
      data.total_duration = Number(durationInput.value) || 0;
      persist();
      updatePreview();
    };
    negativeInput.oninput = () => {
      data.negative_prompt = negativeInput.value;
      persist();
    };
    soundInput.oninput = () => {
      data.soundscape = soundInput.value;
      persist();
      updatePreview();
    };
    musicInput.oninput = () => {
      data.music_style = musicInput.value;
      persist();
      updatePreview();
    };
    addButton.onclick = () => {
      const start = data.shots.reduce((sum, shot) => sum + (Number(shot.duration) || 0), 0);
      const remaining = Math.max(0.1, Number(data.total_duration) - start);
      data.shots.push(newShot(remaining));
      persist();
      renderShots();
      updatePreview();
    };
    addSubjectButton.onclick = () => {
      data.subjects.push(newSubject());
      persist();
      renderSubjects();
      updatePreview();
    };
  };

  const getPackageNodes = () => (node.graph?.nodes || app.graph?.nodes || [])
    .filter((item) => item.type === "MiniMaxH3PackageData");

  const refreshMediaLabels = async () => {
    const nodes = getPackageNodes();
    mediaSourceSelect.replaceChildren();
    if (!nodes.length) {
      mediaSourceNodeId = null;
      mediaLabels = { images: [], videos: [], audios: [] };
      render();
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
    render();
  };

  const render = () => {
    bindGlobal();
    renderSubjects();
    renderShots();
    updatePreview();
    syncStoryboard(node, data);
    node.graph?.setDirtyCanvas?.(true, true);
  };

  const setData = (value) => {
    data = parseStoryboard(value);
    render();
  };

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
  node.setSize?.([800, 740]);
  mediaSourceSelect.onchange = () => {
    mediaSourceNodeId = mediaSourceSelect.value;
    refreshMediaLabels();
  };
  api.addEventListener("minimax-h3/package-data-changed", refreshMediaLabels);
  refreshMediaLabels();
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
