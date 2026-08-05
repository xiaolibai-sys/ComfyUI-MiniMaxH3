import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const style = document.createElement("style");
style.textContent = `
.mmh3-package-root {
  display: grid;
  grid-template-rows: repeat(3, minmax(0, 1fr));
  gap: 8px;
  width: 100%;
  min-width: 0;
  max-width: 100%;
  height: 600px;
  box-sizing: border-box;
  padding: 8px;
  overflow: hidden;
  font-size: 12px;
  color: var(--input-text, #d5d8de);
}
.mmh3-package-section {
  display: flex;
  flex-direction: column;
  min-height: 0;
  overflow: hidden;
  border: 1px solid var(--border-color, #3a3f4b);
  border-radius: 10px;
  background: var(--comfy-input-bg, #1e2127);
  padding: 8px;
}
.mmh3-package-section-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}
.mmh3-package-section-title {
  font-weight: 600;
}
.mmh3-package-section-count {
  opacity: 0.72;
}
.mmh3-package-section-button {
  margin-left: auto;
  min-height: 26px;
  border: 1px solid var(--border-color, #525a68);
  border-radius: 8px;
  background: var(--comfy-input-bg, #262a32);
  color: var(--input-text, #d5d8de);
  cursor: pointer;
  padding: 0 10px;
}
.mmh3-package-section-button:disabled {
  opacity: 0.45;
  cursor: default;
}
.mmh3-package-list {
  flex: 1;
  min-height: 0;
  max-height: 100%;
  display: flex;
  flex-direction: column;
  gap: 8px;
  overflow-y: auto;
  scrollbar-width: thin;
  padding: 2px;
}
.mmh3-package-item {
  width: 100%;
  min-width: 0;
  display: flex;
  align-items: center;
  gap: 10px;
  box-sizing: border-box;
  border: 1px solid var(--border-color, #3a3f4b);
  border-radius: 10px;
  background: var(--comfy-input-bg, #171a20);
  padding: 8px;
}
.mmh3-package-thumb {
  flex: none;
  width: 84px;
  height: 44px;
  object-fit: contain;
  border-radius: 6px;
  background: #000;
}
.mmh3-package-badge {
  flex: none;
  min-width: 52px;
  height: 20px;
  box-sizing: border-box;
  padding: 0 8px;
  border-radius: 7px;
  background: rgba(255, 255, 255, 0.08);
  color: #fff;
  font-size: 11px;
  line-height: 20px;
  text-align: center;
}
.mmh3-package-info {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.mmh3-package-info-row {
  min-width: 0;
  display: flex;
  align-items: center;
  gap: 8px;
}
.mmh3-package-name {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  opacity: 0.82;
}
.mmh3-package-role {
  min-width: 0;
  max-width: 220px;
  height: 24px;
  border: 1px solid var(--border-color, #525a68);
  border-radius: 8px;
  background: var(--comfy-input-bg, #262a32);
  color: var(--input-text, #d5d8de);
  padding: 0 6px;
}
.mmh3-package-duration {
  flex: none;
  opacity: 0.8;
}
.mmh3-package-note {
  width: 100%;
  min-width: 0;
  height: 24px;
  box-sizing: border-box;
  border: 1px solid var(--border-color, #525a68);
  border-radius: 8px;
  background: var(--comfy-input-bg, #262a32);
  color: var(--input-text, #d5d8de);
  padding: 0 6px;
}
.mmh3-package-remove {
  flex: none;
  width: 24px;
  height: 24px;
  border: 1px solid var(--border-color, #525a68);
  border-radius: 8px;
  background: var(--comfy-input-bg, #262a32);
  color: var(--input-text, #d5d8de);
  cursor: pointer;
}
.mmh3-audio-item {
  flex: 1;
  min-width: 0;
  display: flex;
  align-items: center;
  gap: 8px;
}
.mmh3-audio-play {
  flex: none;
  min-width: 52px;
  min-height: 26px;
  border: 1px solid var(--border-color, #525a68);
  border-radius: 8px;
  background: var(--comfy-input-bg, #262a32);
  color: var(--input-text, #d5d8de);
  cursor: pointer;
}
.mmh3-audio-wave {
  display: block;
  min-width: 0;
}
.mmh3-audio-duration {
  flex: none;
  opacity: 0.8;
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
      const buckets = 48;
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
      peaks = Array(48).fill(0.18);
      item._peaks = peaks;
    }
  }

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#4f9cf7";
  const barWidth = Math.max(2, canvas.width / peaks.length - 2);
  peaks.forEach((peak, index) => {
    const height = Math.max(2, peak * canvas.height);
    ctx.fillRect(index * (barWidth + 2), (canvas.height - height) / 2, barWidth, height);
  });
}

function makePreview(item, kind) {
  if (kind === "images") {
    const img = document.createElement("img");
    img.className = "mmh3-package-thumb";
    img.src = mediaUrl(item.name);
    img.alt = "";
    img.onerror = () => {
      img.remove();
    };
    return img;
  }
  if (kind === "videos") {
    const video = document.createElement("video");
    video.className = "mmh3-package-thumb";
    video.src = mediaUrl(item.name);
    video.muted = true;
    video.controls = true;
    video.preload = "metadata";
    return video;
  }

  const row = document.createElement("div");
  const play = document.createElement("button");
  const canvas = document.createElement("canvas");
  const durationLabel = document.createElement("span");
  const audio = new Audio(mediaUrl(item.name));

  row.className = "mmh3-audio-item";
  play.className = "mmh3-audio-play";
  play.textContent = "Play";
  canvas.className = "mmh3-audio-wave";
  durationLabel.className = "mmh3-audio-duration";

  const duration = Number(item.duration || 0);
  const ratio = Math.min(1, Math.max(0, (duration - DURATIONS.audio.min) / (DURATIONS.audio.max - DURATIONS.audio.min)));
  const cssWidth = Math.round(120 + ratio * 220);
  canvas.style.width = `${cssWidth}px`;
  canvas.style.height = "46px";
  canvas.width = cssWidth * 2;
  canvas.height = 92;
  durationLabel.textContent = duration ? `${duration.toFixed(1)}s` : "";

  audio.preload = "metadata";
  item._audio = audio;
  play.onclick = () => {
    if (audio.paused) {
      audio.play();
      play.textContent = "Pause";
    } else {
      audio.pause();
      play.textContent = "Play";
    }
  };
  audio.onended = () => {
    play.textContent = "Play";
  };
  audio.onerror = () => {
    play.disabled = true;
  };

  drawWaveform(item, canvas);
  row.append(play, canvas, durationLabel);
  return row;
}

function makeInfo(item, kind, index, onRoleChange) {
  const info = document.createElement("div");
  const rowTop = document.createElement("div");
  const badge = document.createElement("span");
  const name = document.createElement("span");
  const rowBottom = document.createElement("div");
  const role = document.createElement("select");
  const duration = document.createElement("span");
  const note = document.createElement("input");
  const prefix = kind === "images" ? "Picture" : kind === "videos" ? "Video" : "Audio";

  info.className = "mmh3-package-info";
  rowTop.className = "mmh3-package-info-row";
  rowBottom.className = "mmh3-package-info-row";
  badge.className = "mmh3-package-badge";
  name.className = "mmh3-package-name";
  role.className = "mmh3-package-role";
  duration.className = "mmh3-package-duration";
  note.className = "mmh3-package-note";

  badge.textContent = item.label || `${prefix} ${index + 1}`;
  name.textContent = item.filename || item.name || "";
  name.title = item.name || "";
  duration.textContent = item.duration ? `${Number(item.duration).toFixed(1)}s` : "";

  ROLE_OPTIONS[kind].forEach(([value, label]) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    role.append(option);
  });
  role.value = item.role;
  role.onchange = () => {
    item.role = role.value;
    onRoleChange();
  };
  note.placeholder = "Optional note";
  note.value = item.note || "";
  note.onchange = () => {
    item.note = note.value.trim();
    onRoleChange();
  };

  rowTop.append(badge, name);
  rowBottom.append(role, duration);
  info.append(rowTop, rowBottom, note);
  return info;
}

function setupPackageData(node) {
  node.properties = node.properties || {};
  if (node.__h3PackageState) {
    node.__h3PackageState.setData(node.properties.package_data);
    return;
  }

  let data = parsePackage(node.properties.package_data);
  const root = document.createElement("div");
  const fileInput = document.createElement("input");
  const sections = {};

  fileInput.type = "file";
  fileInput.style.display = "none";
  root.className = "mmh3-package-root";

  const makeSection = (kind, title) => {
    const section = document.createElement("div");
    const header = document.createElement("div");
    const titleEl = document.createElement("span");
    const count = document.createElement("span");
    const button = document.createElement("button");
    const list = document.createElement("div");

    section.className = "mmh3-package-section";
    header.className = "mmh3-package-section-header";
    titleEl.className = "mmh3-package-section-title";
    count.className = "mmh3-package-section-count";
    button.className = "mmh3-package-section-button";
    list.className = "mmh3-package-list";

    titleEl.textContent = title;
    button.textContent = kind === "images" ? "Add Image" : kind === "videos" ? "Add Video" : "Add Audio";
    button.onclick = () => pickFile(kind);

    list.dataset.kind = kind;
    header.append(titleEl, count, button);
    section.append(header, list);
    sections[kind] = { section, list, count, button };
    return section;
  };

  root.append(
    makeSection("images", "Images"),
    makeSection("videos", "Videos"),
    makeSection("audios", "Audio"),
    fileInput,
  );

  const render = () => {
    for (const kind of ["images", "videos", "audios"]) {
      const prefix = kind === "images" ? "Picture" : kind === "videos" ? "Video" : "Audio";
      let nextLabel = 1;
      for (const item of data[kind]) {
        const match = String(item.label || "").match(new RegExp(`^<${prefix} (\\d+)>$`));
        if (match) {
          nextLabel = Math.max(nextLabel, Number(match[1]) + 1);
        }
      }
      data[kind].forEach((item, index) => {
        if (!item.label) {
          item.label = `<${prefix} ${nextLabel}>`;
          nextLabel += 1;
        }
      });
      const items = data[kind].map((item, index) => {
        const wrapper = document.createElement("div");
        const remove = document.createElement("button");

        wrapper.className = "mmh3-package-item";
        wrapper.title = item.name || "";
        remove.className = "mmh3-package-remove";
        remove.textContent = "x";
        remove.onclick = () => {
          item._audio?.pause();
          data[kind].splice(index, 1);
          render();
        };
        wrapper.append(
          makePreview(item, kind),
          makeInfo(item, kind, index, render),
          remove,
        );
        return wrapper;
      });

      sections[kind].list.replaceChildren(...items);
      sections[kind].count.textContent = `${data[kind].length}/${LIMITS[kind]}`;
      sections[kind].button.disabled = data[kind].length >= LIMITS[kind];
    }
    node.properties.package_data = serializePackage(data);
    syncPackage(node, data);
    api.dispatchCustomEvent("minimax-h3/package-data-changed", {
      nodeId: String(node.id),
    });
    node.graph?.setDirtyCanvas?.(true, true);
  };

  const setData = (value) => {
    data = parsePackage(value);
    render();
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

  const uploadFile = async (file, kind) => {
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
        const info = await uploadFile(file, kind);
        data[kind].push({ duration, role: DEFAULT_ROLE[kind], ...info });
        render();
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
    getMinHeight: () => 600,
    getMaxHeight: () => 600,
  });
  domWidget.computeSize = function (width) {
    return [width, 600];
  };
  node.__h3PackageState = { setData, render };
  node.setSize?.([780, 620]);
  render();
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
