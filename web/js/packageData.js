import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const style = document.createElement("style");
style.textContent = `
.mmh3-pkg-root {
  --pkg-text: var(--input-text, #d8dbe2);
  --pkg-dim: var(--descrip-text, #8b91a0);
  --pkg-border: var(--border-color, #3a3f4b);
  --pkg-border-strong: #4a5163;
  --pkg-surface: var(--comfy-input-bg, #22262f);
  --pkg-raised: rgba(255, 255, 255, 0.035);
  --pkg-accent: #4cc2a8;
  --pkg-c-images: #38bdf8;
  --pkg-c-videos: #f5b04c;
  --pkg-c-audios: #b78cf0;
  display: flex;
  flex-direction: column;
  gap: 8px;
  width: 100%;
  min-width: 0;
  max-width: 100%;
  height: 480px;
  box-sizing: border-box;
  padding: 8px;
  overflow: hidden;
  font-size: 12px;
  color: var(--pkg-text);
}
.mmh3-pkg-root *,
.mmh3-pkg-root *::before,
.mmh3-pkg-root *::after {
  box-sizing: border-box;
}
.mmh3-pkg-root button {
  font: inherit;
  color: inherit;
  background: none;
  border: none;
  cursor: pointer;
  padding: 0;
}
.mmh3-pkg-root input,
.mmh3-pkg-root select {
  font: inherit;
  font-size: 11px;
  color: var(--pkg-text);
  background: var(--pkg-surface);
  border: 1px solid var(--pkg-border);
  border-radius: 6px;
  padding: 4px 7px;
  outline: none;
  width: 100%;
}
.mmh3-pkg-root input:focus,
.mmh3-pkg-root select:focus {
  border-color: var(--pkg-accent);
}
.mmh3-pkg-root ::-webkit-scrollbar {
  width: 8px;
}
.mmh3-pkg-root ::-webkit-scrollbar-thumb {
  background: var(--pkg-border-strong);
  border-radius: 4px;
}
.mmh3-pkg-root ::-webkit-scrollbar-track {
  background: transparent;
}

/* tab bar */
.mmh3-pkg-tabs {
  flex: none;
  display: flex;
  align-items: center;
  gap: 6px;
}
.mmh3-pkg-tab {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  padding: 5px 12px;
  border-radius: 7px;
  border: 1px solid var(--pkg-border);
  background: var(--pkg-raised);
  color: var(--pkg-dim);
  font-size: 11px;
  font-weight: 600;
  transition: border-color 0.12s, background 0.12s, color 0.12s;
}
.mmh3-pkg-tab:hover {
  border-color: currentColor;
}
.mmh3-pkg-tab .mmh3-pkg-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: currentColor;
}
.mmh3-pkg-tab .mmh3-pkg-tab-count {
  font-weight: 400;
  opacity: 0.75;
}
.mmh3-pkg-tab.k-images { color: var(--pkg-c-images); }
.mmh3-pkg-tab.k-videos { color: var(--pkg-c-videos); }
.mmh3-pkg-tab.k-audios { color: var(--pkg-c-audios); }
.mmh3-pkg-tab span.tt { color: var(--pkg-text); }
.mmh3-pkg-tab.on {
  border-color: currentColor;
  background: var(--pkg-surface);
}
.mmh3-pkg-spacer {
  flex: 1;
}
.mmh3-pkg-add {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 5px 14px;
  border-radius: 7px;
  border: 1px dashed var(--pkg-border-strong);
  color: var(--pkg-dim);
  font-size: 11px;
  font-weight: 600;
  transition: color 0.12s, border-color 0.12s, background 0.12s;
}
.mmh3-pkg-add:hover:not(:disabled) {
  color: var(--pkg-accent);
  border-color: var(--pkg-accent);
  background: rgba(76, 194, 168, 0.08);
}
.mmh3-pkg-add:disabled {
  opacity: 0.4;
  cursor: default;
}

/* content */
.mmh3-pkg-content {
  flex: 1;
  min-height: 0;
  overflow-y: auto;
  scrollbar-width: thin;
  border: 1px solid var(--pkg-border);
  border-radius: 8px;
  background: var(--pkg-raised);
  padding: 10px;
}
.mmh3-pkg-empty {
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  color: var(--pkg-dim);
  font-size: 12px;
  border: 1px dashed var(--pkg-border);
  border-radius: 8px;
}
.mmh3-pkg-empty .icon {
  opacity: 0.5;
}
.mmh3-pkg-empty .hint {
  font-size: 10px;
  opacity: 0.75;
}

/* image grid */
.mmh3-pkg-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
}
.mmh3-pkg-card {
  display: flex;
  flex-direction: column;
  gap: 6px;
  border: 1px solid var(--pkg-border);
  border-radius: 8px;
  background: var(--pkg-surface);
  padding: 7px;
  min-width: 0;
  transition: border-color 0.12s, box-shadow 0.12s;
}
.mmh3-pkg-card:hover {
  border-color: var(--pkg-border-strong);
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25);
}
.mmh3-pkg-thumbwrap {
  position: relative;
  aspect-ratio: 4 / 3;
  border-radius: 6px;
  overflow: hidden;
  background: #000;
}
.mmh3-pkg-thumbwrap img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.mmh3-pkg-thumbname {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  padding: 14px 7px 5px;
  font-size: 9px;
  color: rgba(255, 255, 255, 0.92);
  background: linear-gradient(transparent, rgba(0, 0, 0, 0.72));
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  pointer-events: none;
}
.mmh3-pkg-chip {
  position: absolute;
  top: 5px;
  left: 5px;
  font-size: 9px;
  font-weight: 700;
  padding: 2px 7px;
  border-radius: 10px;
  background: rgba(0, 0, 0, 0.62);
  backdrop-filter: blur(2px);
}
.mmh3-pkg-chip.k-images { color: var(--pkg-c-images); }
.mmh3-pkg-chip.k-videos { color: var(--pkg-c-videos); }
.mmh3-pkg-chip.k-audios { color: var(--pkg-c-audios); }
.mmh3-pkg-x {
  position: absolute;
  top: 5px;
  right: 5px;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: rgba(0, 0, 0, 0.62);
  color: #fff;
  font-size: 11px;
  line-height: 1;
  display: none;
  align-items: center;
  justify-content: center;
}
.mmh3-pkg-thumbwrap:hover .mmh3-pkg-x,
.mmh3-pkg-row:hover .mmh3-pkg-x {
  display: inline-flex;
}
.mmh3-pkg-x:hover {
  background: var(--error-text, #f26d6d);
}
.mmh3-pkg-name {
  font-size: 10px;
  color: var(--pkg-dim);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* video / audio rows */
.mmh3-pkg-rows {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.mmh3-pkg-row {
  position: relative;
  display: flex;
  gap: 12px;
  border: 1px solid var(--pkg-border);
  border-radius: 8px;
  background: var(--pkg-surface);
  padding: 10px;
}
.mmh3-pkg-row .mmh3-pkg-x {
  display: none;
  position: absolute;
  top: 7px;
  right: 7px;
}
.mmh3-pkg-preview {
  flex: none;
  width: 128px;
  display: flex;
  flex-direction: column;
  gap: 5px;
  min-width: 0;
}
.mmh3-pkg-vthumb {
  width: 128px;
  height: 72px;
  object-fit: contain;
  border-radius: 6px;
  background: #000;
}
.mmh3-pkg-info {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.mmh3-pkg-rowtop {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
  padding-right: 24px;
}
.mmh3-pkg-rowtop .mmh3-pkg-chip {
  position: static;
  flex: none;
  background: var(--pkg-raised);
}
.mmh3-pkg-duration {
  flex: none;
  font-size: 10px;
  color: var(--pkg-dim);
}
.mmh3-pkg-audio {
  flex: 1;
  min-width: 0;
  display: flex;
  align-items: center;
  gap: 8px;
}
.mmh3-pkg-play {
  flex: none;
  width: 30px;
  height: 30px;
  border-radius: 50%;
  border: 1px solid var(--pkg-c-audios);
  color: var(--pkg-c-audios);
  display: inline-flex;
  align-items: center;
  justify-content: center;
}
.mmh3-pkg-play:hover:not(:disabled) {
  background: rgba(183, 140, 240, 0.12);
}
.mmh3-pkg-play:disabled {
  opacity: 0.4;
  cursor: default;
}
.mmh3-pkg-wave {
  flex: 1;
  min-width: 0;
  display: block;
}
`;
document.head.appendChild(style);

const LIMITS = {
  images: 9,
  videos: 3,
  audios: 3,
};

const DURATIONS = {
  video: { min: 2, max: 15, total: 15 },
  audio: { min: 2, max: 15, total: 15 },
};

const KIND_COLORS = {
  images: "var(--pkg-c-images)",
  videos: "var(--pkg-c-videos)",
  audios: "var(--pkg-c-audios)",
};

const KIND_PREFIX = {
  images: "Picture",
  videos: "Video",
  audios: "Audio",
};

const ROLE_OPTIONS = {
  images: [
    ["reference_image", "Reference Image (conservative)"],
    ["subject_reference", "Subject / Appearance"],
    ["scene_reference", "Scene / Environment"],
    ["style_reference", "Style / Mood"],
    ["storyboard_anchor", "Storyboard Anchor"],
  ],
  videos: [
    ["reference_video", "Reference Video (conservative)"],
    ["motion_reference", "Motion Reference"],
    ["camera_reference", "Camera Reference"],
    ["continuation_source", "Continuation Source"],
    ["edit_source", "Edit Source"],
    ["structure_reference", "Video Structure"],
  ],
  audios: [
    ["reference_audio", "Reference Audio (conservative)"],
    ["music_reference", "Music Style"],
    ["sound_effect_reference", "Sound Effect"],
    ["voice_reference", "Voice / Timbre"],
    ["audio_copy", "Audio Reuse"],
  ],
};

const DEFAULT_ROLE = {
  images: "reference_image",
  videos: "reference_video",
  audios: "reference_audio",
};

const ICONS = {
  close: '<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M18 6 6 18M6 6l12 12"/></svg>',
  play: '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>',
  pause: '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M6 5h4v14H6zM14 5h4v14h-4z"/></svg>',
  image: '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.1-3.1a2 2 0 0 0-2.8 0L6 21"/></svg>',
  video: '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="2" y="4" width="20" height="16" rx="2"/><path d="m10 9 5 3-5 3z" fill="currentColor" stroke="none"/></svg>',
  audio: '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>',
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

function normalizeItem(item, kind) {
  return {
    ...item,
    role: ROLE_OPTIONS[kind].some(([value]) => value === item.role)
      ? item.role
      : DEFAULT_ROLE[kind],
  };
}

function parsePackage(raw) {
  try {
    const data = JSON.parse(raw || "{}");
    return {
      images: Array.isArray(data.images)
        ? data.images.map((item) => normalizeItem(item, "images"))
        : [],
      videos: Array.isArray(data.videos)
        ? data.videos.map((item) => normalizeItem(item, "videos"))
        : [],
      audios: Array.isArray(data.audios)
        ? data.audios.map((item) => normalizeItem(item, "audios"))
        : [],
    };
  } catch (error) {
    console.error("MiniMax H3 PackageData parse error", error);
    return { images: [], videos: [], audios: [] };
  }
}

function serializePackage(data) {
  return JSON.stringify(data);
}

function mediaUrl(name) {
  return api.apiURL("/view?" + new URLSearchParams({ filename: name, type: "input" }));
}

function syncPackage(node, data) {
  return api.fetchApi("/minimax-h3/package-data", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ node_id: String(node.id), data }),
  }).catch((error) => {
    console.error("MiniMax H3 PackageData sync error", error);
  });
}

function readDuration(file, kind) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const media = kind === "video" ? document.createElement("video") : new Audio();
    media.preload = "metadata";
    media.onloadedmetadata = () => {
      const duration = media.duration;
      URL.revokeObjectURL(url);
      resolve(duration);
    };
    media.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error(`Could not read ${kind} duration`));
    };
    media.src = url;
  });
}

async function drawWaveform(item, canvas) {
  let peaks = item._peaks;
  if (!peaks) {
    try {
      const response = await fetch(mediaUrl(item.name));
      const buffer = await response.arrayBuffer();
      const context = new (window.AudioContext || window.webkitAudioContext)();
      const decoded = await context.decodeAudioData(buffer);
      const channel = decoded.getChannelData(0);
      const buckets = 64;
      const block = Math.max(1, Math.floor(channel.length / buckets));
      peaks = [];
      for (let i = 0; i < buckets; i++) {
        let peak = 0;
        const start = i * block;
        const end = Math.min(channel.length, start + block);
        for (let j = start; j < end; j++) {
          peak = Math.max(peak, Math.abs(channel[j]));
        }
        peaks.push(peak);
      }
      item._peaks = peaks;
      await context.close();
    } catch (error) {
      peaks = Array(64).fill(0.18);
      item._peaks = peaks;
    }
  }

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#b78cf0";
  const barWidth = Math.max(2, canvas.width / peaks.length - 2);
  peaks.forEach((peak, index) => {
    const height = Math.max(2, peak * canvas.height);
    ctx.fillRect(index * (barWidth + 2), (canvas.height - height) / 2, barWidth, height);
  });
}

function makeChip(kind, item, index) {
  const chip = el("span", `mmh3-pkg-chip k-${kind}`,
    (item.label || `<${KIND_PREFIX[kind]} ${index + 1}>`).replace(/[<>]/g, ""));
  return chip;
}

function makeRoleSelect(kind, item, onChange) {
  const role = document.createElement("select");
  ROLE_OPTIONS[kind].forEach(([value, label]) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    role.append(option);
  });
  role.value = item.role;
  role.onchange = () => {
    item.role = role.value;
    onChange();
  };
  return role;
}

function makeNoteInput(item, onChange) {
  const note = document.createElement("input");
  note.placeholder = "Optional note";
  note.value = item.note || "";
  note.onchange = () => {
    item.note = note.value.trim();
    onChange();
  };
  return note;
}

function makeRemoveButton(onClick) {
  const remove = el("button", "mmh3-pkg-x");
  remove.innerHTML = ICONS.close;
  remove.title = "Remove";
  remove.onclick = onClick;
  return remove;
}

function setupPackageData(node) {
  node.properties = node.properties || {};
  if (node.__h3PackageState) {
    node.__h3PackageState.setData(node.properties.package_data);
    return;
  }

  let data = parsePackage(node.properties.package_data);
  let activeTab = "images";

  const root = el("div", "mmh3-pkg-root");
  const tabbar = el("div", "mmh3-pkg-tabs");
  const content = el("div", "mmh3-pkg-content");
  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.style.display = "none";

  const tabButtons = {};
  ["images", "videos", "audios"].forEach((kind) => {
    const button = el("button", `mmh3-pkg-tab k-${kind}`);
    const dot = el("span", "mmh3-pkg-dot");
    const title = el("span", "tt", kind === "images" ? "Images" : kind === "videos" ? "Videos" : "Audio");
    const count = el("span", "mmh3-pkg-tab-count");
    button.append(dot, title, count);
    button.onclick = () => {
      activeTab = kind;
      renderTabs();
      renderContent();
    };
    tabButtons[kind] = { button, count };
    tabbar.append(button);
  });
  const spacer = el("div", "mmh3-pkg-spacer");
  const addButton = el("button", "mmh3-pkg-add");
  tabbar.append(spacer, addButton);
  root.append(tabbar, content, fileInput);

  const persist = () => {
    node.properties.package_data = serializePackage(data);
    syncPackage(node, data);
    api.dispatchCustomEvent("minimax-h3/package-data-changed", {
      nodeId: String(node.id),
    });
    node.graph?.setDirtyCanvas?.(true, true);
  };

  const renderTabs = () => {
    for (const kind of ["images", "videos", "audios"]) {
      tabButtons[kind].button.classList.toggle("on", kind === activeTab);
      tabButtons[kind].count.textContent = `${data[kind].length}/${LIMITS[kind]}`;
    }
    const singular = activeTab === "images" ? "Image" : activeTab === "videos" ? "Video" : "Audio";
    addButton.textContent = `+ Add ${singular}`;
    addButton.disabled = data[activeTab].length >= LIMITS[activeTab];
    addButton.onclick = () => pickFile(activeTab);
  };

  const makeImageCard = (item, index) => {
    const card = el("div", "mmh3-pkg-card");
    card.title = item.name || "";
    const wrap = el("div", "mmh3-pkg-thumbwrap");
    const img = document.createElement("img");
    img.src = mediaUrl(item.name);
    img.alt = "";
    img.onerror = () => img.remove();
    wrap.append(img, makeChip("images", item, index));
    const thumbName = el("div", "mmh3-pkg-thumbname", item.filename || item.name || "");
    wrap.append(thumbName);
    wrap.append(makeRemoveButton(() => {
      data.images.splice(index, 1);
      persist();
      renderAll();
    }));
    card.append(
      wrap,
      makeRoleSelect("images", item, persist),
      makeNoteInput(item, persist),
    );
    return card;
  };

  const makeVideoRow = (item, index) => {
    const row = el("div", "mmh3-pkg-row");
    row.title = item.name || "";
    const preview = el("div", "mmh3-pkg-preview");
    const video = document.createElement("video");
    video.className = "mmh3-pkg-vthumb";
    video.src = mediaUrl(item.name);
    video.muted = true;
    video.controls = true;
    video.preload = "metadata";
    preview.append(video);

    const info = el("div", "mmh3-pkg-info");
    const top = el("div", "mmh3-pkg-rowtop");
    top.append(
      makeChip("videos", item, index),
      el("span", "mmh3-pkg-name", item.filename || item.name || ""),
      el("span", "mmh3-pkg-spacer"),
      el("span", "mmh3-pkg-duration", item.duration ? `${Number(item.duration).toFixed(1)}s` : ""),
    );
    info.append(top, makeRoleSelect("videos", item, persist), makeNoteInput(item, persist));
    row.append(preview, info, makeRemoveButton(() => {
      data.videos.splice(index, 1);
      persist();
      renderAll();
    }));
    return row;
  };

  const makeAudioRow = (item, index) => {
    const row = el("div", "mmh3-pkg-row");
    row.title = item.name || "";

    const info = el("div", "mmh3-pkg-info");
    const top = el("div", "mmh3-pkg-rowtop");
    top.append(
      makeChip("audios", item, index),
      el("span", "mmh3-pkg-name", item.filename || item.name || ""),
      el("span", "mmh3-pkg-spacer"),
      el("span", "mmh3-pkg-duration", item.duration ? `${Number(item.duration).toFixed(1)}s` : ""),
    );

    const audioRow = el("div", "mmh3-pkg-audio");
    const play = el("button", "mmh3-pkg-play");
    play.innerHTML = ICONS.play;
    const canvas = document.createElement("canvas");
    canvas.className = "mmh3-pkg-wave";
    canvas.style.height = "34px";
    canvas.width = 640;
    canvas.height = 68;
    const audio = new Audio(mediaUrl(item.name));
    audio.preload = "metadata";
    item._audio = audio;
    play.onclick = () => {
      if (audio.paused) {
        audio.play();
        play.innerHTML = ICONS.pause;
      } else {
        audio.pause();
        play.innerHTML = ICONS.play;
      }
    };
    audio.onended = () => {
      play.innerHTML = ICONS.play;
    };
    audio.onerror = () => {
      play.disabled = true;
    };
    audioRow.append(play, canvas);
    drawWaveform(item, canvas);

    info.append(top, audioRow, makeRoleSelect("audios", item, persist), makeNoteInput(item, persist));
    row.append(info, makeRemoveButton(() => {
      item._audio?.pause();
      data.audios.splice(index, 1);
      persist();
      renderAll();
    }));
    return row;
  };

  const renderContent = () => {
    content.replaceChildren();
    const kind = activeTab;
    const items = data[kind];
    if (!items.length) {
      const empty = el("div", "mmh3-pkg-empty");
      const singular = kind === "images" ? "image" : kind === "videos" ? "video" : "audio";
      const icon = el("span", "icon");
      icon.innerHTML = ICONS[kind === "images" ? "image" : kind === "videos" ? "video" : "audio"];
      empty.append(
        icon,
        el("span", "", `No ${kind} yet`),
        el("span", "hint", `Add up to ${LIMITS[kind]} ${singular} ${kind === "images" ? "files" : "clips"} as reference media`),
      );
      content.append(empty);
      return;
    }
    if (kind === "images") {
      const grid = el("div", "mmh3-pkg-grid");
      items.forEach((item, index) => grid.append(makeImageCard(item, index)));
      content.append(grid);
    } else {
      const rows = el("div", "mmh3-pkg-rows");
      const maker = kind === "videos" ? makeVideoRow : makeAudioRow;
      items.forEach((item, index) => rows.append(maker(item, index)));
      content.append(rows);
    }
  };

  const renderAll = () => {
    renderTabs();
    renderContent();
  };

  const setData = (value) => {
    data = parsePackage(value);
    renderAll();
  };

  const validateTimed = (kind, duration) => {
    const limits = DURATIONS[kind.slice(0, -1)];
    if (!Number.isFinite(duration) || duration < limits.min || duration > limits.max) {
      throw new Error(`${kind} must be ${limits.min}-${limits.max}s`);
    }
    const total = data[kind].reduce((sum, item) => sum + Number(item.duration || 0), 0) + duration;
    if (total > limits.total) {
      throw new Error(`${kind} total must not exceed ${limits.total}s`);
    }
  };

  const uploadFile = async (file) => {
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

  const pickFile = async (kind) => {
    if (data[kind].length >= LIMITS[kind]) {
      alert(`Max ${LIMITS[kind]} ${kind}`);
      return;
    }

    fileInput.accept = kind === "images" ? "image/*" : kind === "videos" ? "video/*" : "audio/*";
    fileInput.onchange = async () => {
      const file = fileInput.files?.[0];
      if (!file) {
        return;
      }
      try {
        let duration = 0;
        if (kind !== "images") {
          duration = await readDuration(file, kind.slice(0, -1));
          validateTimed(kind, duration);
        }
        const info = await uploadFile(file);
        data[kind].push({ duration, role: DEFAULT_ROLE[kind], ...info });
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

  const domWidget = node.addDOMWidget("package_data_ui", "package_data_ui", root, {
    serialize: false,
    hideOnZoom: false,
    getMinHeight: () => 480,
    getMaxHeight: () => 480,
  });
  domWidget.computeSize = function (width) {
    return [width, 480];
  };
  node.__h3PackageState = { setData, render: renderAll };
  node.setSize?.([780, 500]);
  renderAll();
  /* re-publish after page reload so storyboard nodes can pull media labels */
  persist();
}

app.registerExtension({
  name: "ComfyUI-MiniMaxH3.PackageData",
  setup() {
    const originalQueuePrompt = api.queuePrompt.bind(api);
    api.queuePrompt = async (...args) => {
      const nodes = app.graph?.nodes || [];
      await Promise.all(nodes
        .filter((node) => node.type === "MiniMaxH3PackageData" && node.properties?.package_data)
        .map((node) => syncPackage(node, parsePackage(node.properties.package_data))));
      return originalQueuePrompt(...args);
    };
  },
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (nodeData?.name !== "MiniMaxH3PackageData") {
      return;
    }
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      setupPackageData(this);
      return result;
    };
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = onConfigure?.apply(this, arguments);
      setupPackageData(this);
      return result;
    };
  },
});
