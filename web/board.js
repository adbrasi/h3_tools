// h3_tools — reference board + @mention editor for the H3 Pro nodes.
//
// Everything the backend needs lives in the two String widgets ("prompt",
// "references"); this extension is a pure view/controller over them. No
// graphToPrompt patches, no prototype patches beyond this node's own
// callbacks, no state in node.properties, no custom server routes.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_IDS = new Set(["H3RefToVideoPro", "H3RefToVideoContinuePro"]);
const SUBFOLDER = "h3_refs";
const CAPS = { image: 9, video: 3, audio: 3 };
const NAME_RE = /^[a-z0-9_]{1,64}$/;
const MENTION_RE = /(?<![A-Za-z0-9_])@([A-Za-z0-9_]{1,64})(?![A-Za-z0-9_])/g;
const TRIGGER_RE = /(?<![A-Za-z0-9_])@([A-Za-z0-9_]*)$/;
const TYPE_ICONS = { image: "\u{1F5BC}", video: "\u{1F3AC}", audio: "♪" };
const TYPE_LABELS = { image: "Imagem", video: "Vídeo", audio: "Áudio" };
const ACCEPT = { image: "image/*", video: "video/*", audio: "audio/*" };

// ---- shared helpers -------------------------------------------------------

let cssInjected = false;
function injectCss() {
  if (cssInjected) return;
  cssInjected = true;
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = new URL("board.css", import.meta.url).href;
  document.head.appendChild(link);
}

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function escapeRegExp(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// replace [start, end) with text, keeping the browser's native undo stack alive
function replaceRange(textarea, start, end, text) {
  textarea.focus();
  textarea.setSelectionRange(start, end);
  let ok = false;
  try {
    ok = document.execCommand("insertText", false, text);
  } catch (e) {
    ok = false;
  }
  if (!ok) textarea.setRangeText(text, start, end, "end");
}

// mirrors refs.derive_name in the backend (drift only mislabels, backend rules)
function deriveName(filename) {
  const base = filename.replace(/\\/g, "/").split("/").pop();
  const stem = base.includes(".") ? base.slice(0, base.lastIndexOf(".")) : base;
  let name = stem.toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
  // drop trailing save-counter runs ("krea2_turbo_00017" -> "krea2_turbo")
  const stripped = name.replace(/(?:_\d{3,})+$/, "");
  if (stripped) name = stripped;
  if (!name || /^[0-9]/.test(name)) name = "m_" + name;
  return name.slice(0, 64);
}

// mirrors refs.parse_references naming: explicit names win, derived get suffixes
function effectiveNames(list) {
  const taken = new Set();
  for (const r of list) if (r.name) taken.add(r.name);
  return list.map((r) => {
    if (r.name) return r.name;
    const base = deriveName(String(r.file || ""));
    let name = base;
    for (let n = 2; taken.has(name); n++) {
      const suffix = `_${n}`;
      name = base.slice(0, 64 - suffix.length) + suffix;
    }
    taken.add(name);
    return name;
  });
}

// mirrors refs.assign_ordinals (display only — the backend is authoritative)
function computeTags(list) {
  const tags = list.map(() => ({}));
  let pic = 0, vid = 0, aud = 0;
  list.forEach((r, i) => { if (r.type === "image") tags[i].tag = `<Picture ${++pic}>`; });
  list.forEach((r, i) => {
    if (r.type !== "video") return;
    if (r.use_soundtrack) tags[i].soundtrackTag = `<Audio ${++aud}>`;
    tags[i].tag = `<Video ${++vid}>`;
  });
  list.forEach((r, i) => { if (r.type === "audio") tags[i].tag = `<Audio ${++aud}>`; });
  return tags;
}

function viewURL(file) {
  const path = String(file).replace(/\\/g, "/");
  const slash = path.lastIndexOf("/");
  const subfolder = slash === -1 ? "" : path.slice(0, slash);
  const filename = slash === -1 ? path : path.slice(slash + 1);
  return api.apiURL(
    `/view?filename=${encodeURIComponent(filename)}` +
    `&subfolder=${encodeURIComponent(subfolder)}&type=input`);
}

// off-DOM <video> seek + canvas grab; cached per file path
const videoThumbCache = new Map();
function videoThumb(file) {
  if (videoThumbCache.has(file)) return videoThumbCache.get(file);
  const promise = new Promise((resolve) => {
    const video = document.createElement("video");
    video.muted = true;
    video.preload = "metadata";
    video.crossOrigin = "anonymous";
    const fail = () => {
      videoThumbCache.delete(file); // allow a retry on the next render
      resolve(null);
    };
    video.onerror = fail;
    video.onloadedmetadata = () => {
      video.currentTime = Math.min(1, (video.duration || 0) / 2);
    };
    video.onseeked = () => {
      try {
        const scale = 96 / (video.videoWidth || 96);
        const canvas = document.createElement("canvas");
        canvas.width = 96;
        canvas.height = Math.max(1, Math.round((video.videoHeight || 54) * scale));
        canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
        resolve(canvas.toDataURL("image/jpeg", 0.7));
      } catch (e) {
        fail();
      }
      video.src = "";
    };
    video.src = viewURL(file);
  });
  videoThumbCache.set(file, promise);
  return promise;
}

// one shared preview player for audio tiles
const sharedAudio = new Audio();
let sharedAudioFile = null;
function toggleAudio(file) {
  if (sharedAudioFile === file) {
    if (sharedAudio.paused) sharedAudio.play().catch(() => {});
    else sharedAudio.pause();
    return;
  }
  sharedAudioFile = file;
  sharedAudio.src = viewURL(file);
  sharedAudio.play().catch(() => {});
}

function stopAudio(file) {
  if (file === undefined || sharedAudioFile === file) {
    sharedAudio.pause();
    sharedAudioFile = null;
    sharedAudio.removeAttribute("src");
  }
}

function setThumb(el, ref) {
  el.textContent = "";
  el.style.backgroundImage = "";
  if (ref.type === "image") {
    const probe = new Image();
    probe.onload = () => { el.style.backgroundImage = `url("${viewURL(ref.file)}")`; };
    probe.onerror = () => { el.textContent = TYPE_ICONS.image; };
    probe.src = viewURL(ref.file);
  } else if (ref.type === "video") {
    el.textContent = TYPE_ICONS.video;
    videoThumb(ref.file).then((url) => {
      if (url) { el.style.backgroundImage = `url("${url}")`; el.textContent = ""; }
    });
  } else {
    el.textContent = TYPE_ICONS.audio;
  }
}

// caret pixel position inside a textarea (mirror-div technique)
const MIRROR_PROPS = [
  "boxSizing", "width", "paddingTop", "paddingRight", "paddingBottom",
  "paddingLeft", "borderTopWidth", "borderRightWidth", "borderBottomWidth",
  "borderLeftWidth", "fontFamily", "fontSize", "fontWeight", "fontStyle",
  "letterSpacing", "lineHeight", "textTransform", "wordSpacing", "textIndent",
];
function caretPosition(textarea) {
  const style = getComputedStyle(textarea);
  const div = document.createElement("div");
  for (const prop of MIRROR_PROPS) div.style[prop] = style[prop];
  div.style.position = "absolute";
  div.style.visibility = "hidden";
  div.style.whiteSpace = "pre-wrap";
  div.style.wordWrap = "break-word";
  div.style.top = "0";
  div.style.left = "-9999px";
  div.textContent = textarea.value.slice(0, textarea.selectionStart);
  const marker = document.createElement("span");
  marker.textContent = "​";
  div.appendChild(marker);
  document.body.appendChild(div);
  // the node's DOM container is transform-scaled with the canvas, so the
  // rect is in scaled viewport px while offsets/scroll are unscaled CSS px
  const scale = app.canvas?.ds?.scale ?? 1;
  const rect = textarea.getBoundingClientRect();
  const top = rect.top + (marker.offsetTop - textarea.scrollTop) * scale;
  const left = rect.left + (marker.offsetLeft - textarea.scrollLeft) * scale;
  const lineHeight =
    (marker.offsetHeight || parseFloat(style.lineHeight) || 16) * scale;
  div.remove();
  return { top, left, lineHeight };
}

// ---- mention popup (module singleton, parented to document.body) ----------

let popup = null; // { el, items, index, insert }

function closePopup() {
  if (!popup) return;
  popup.el.remove();
  window.removeEventListener("wheel", onPopupWheel, true);
  popup = null;
}

function onPopupWheel(e) {
  if (popup && !popup.el.contains(e.target)) closePopup();
}

function renderPopupRows() {
  const { el, items, index } = popup;
  el.textContent = "";
  if (!items.length) {
    const empty = document.createElement("div");
    empty.className = "h3tools-popup-empty";
    empty.textContent = "nenhuma referência";
    el.appendChild(empty);
    return;
  }
  items.forEach((item, i) => {
    const row = document.createElement("div");
    row.className = "h3tools-popup-row" + (i === index ? " h3tools-popup-active" : "");
    const thumb = document.createElement("div");
    thumb.className = "h3tools-popup-thumb";
    setThumb(thumb, item.ref);
    const name = document.createElement("span");
    name.className = "h3tools-popup-name";
    name.textContent = "@" + item.name;
    const badge = document.createElement("span");
    badge.className = "h3tools-badge";
    badge.textContent = item.ref.type;
    const tag = document.createElement("span");
    tag.className = "h3tools-popup-tag";
    tag.textContent = item.tag || "";
    row.append(thumb, name, badge, tag);
    // mousedown (not click) so the textarea never loses focus
    row.addEventListener("mousedown", (e) => {
      e.preventDefault();
      popup.insert(item.name);
    });
    row.addEventListener("mousemove", () => {
      if (popup.index !== i) { popup.index = i; renderPopupRows(); }
    });
    el.appendChild(row);
  });
  el.children[index]?.scrollIntoView({ block: "nearest" });
}

function openPopup(textarea, items, insert) {
  if (!popup) {
    const el = document.createElement("div");
    el.className = "h3tools-popup";
    document.body.appendChild(el);
    popup = { el, items, index: 0, insert };
    window.addEventListener("wheel", onPopupWheel, true);
  } else {
    const prev = popup.items[popup.index]?.name;
    popup.items = items;
    popup.insert = insert;
    popup.index = Math.max(0, items.findIndex((i) => i.name === prev));
  }
  renderPopupRows();
  const pos = caretPosition(textarea);
  const el = popup.el;
  el.style.left = Math.min(pos.left, window.innerWidth - 330) + "px";
  const below = pos.top + pos.lineHeight + 2;
  if (below + el.offsetHeight > window.innerHeight - 8) {
    el.style.top = Math.max(4, pos.top - el.offsetHeight - 2) + "px";
  } else {
    el.style.top = below + "px";
  }
}

// ---- widget hiding --------------------------------------------------------

function hideWidget(widget) {
  if (!widget) return;
  try {
    widget.computeSize = () => [0, -4];
    widget.type = "hidden";
    widget.hidden = true;
    const el = widget.element || widget.inputEl;
    if (el) el.style.display = "none";
  } catch (e) {
    // degraded but functional: the raw widget stays visible
    console.warn("h3_tools: could not hide widget", widget?.name, e);
  }
}

// ---- node setup -----------------------------------------------------------

function setupNode(node) {
  injectCss();
  const promptWidget = node.widgets?.find((w) => w.name === "prompt");
  const refsWidget = node.widgets?.find((w) => w.name === "references");
  if (!promptWidget || !refsWidget) {
    console.warn("h3_tools: prompt/references widgets not found; UI disabled");
    return;
  }
  hideWidget(promptWidget);
  hideWidget(refsWidget);

  // ---- DOM skeleton
  const root = document.createElement("div");
  root.className = "h3tools-root";
  const error = document.createElement("div");
  error.className = "h3tools-error";
  error.style.display = "none";
  const toolbar = document.createElement("div");
  toolbar.className = "h3tools-toolbar";
  const counters = document.createElement("span");
  counters.className = "h3tools-counters";
  const grid = document.createElement("div");
  grid.className = "h3tools-grid";
  const editor = document.createElement("div");
  editor.className = "h3tools-editor";
  const highlight = document.createElement("div");
  highlight.className = "h3tools-highlight";
  const textarea = document.createElement("textarea");
  textarea.spellcheck = false;
  textarea.placeholder = "Digite o prompt — use @ para mencionar uma referência";
  editor.append(highlight, textarea);
  root.append(error, toolbar, grid, editor);

  let errorTimer = null;
  function showError(message) {
    error.textContent = message;
    error.style.display = "";
    clearTimeout(errorTimer);
    errorTimer = setTimeout(() => { error.style.display = "none"; }, 8000);
  }

  // ---- references state (the JSON widget is the single source of truth)
  function readRefs() {
    try {
      const parsed = JSON.parse(refsWidget.value || "[]");
      if (Array.isArray(parsed)) return parsed;
    } catch (e) { /* fall through */ }
    showError("references inválido (JSON malformado) — corrija o widget");
    return null;
  }

  function writeRefs(list) {
    refsWidget.value = JSON.stringify(list);
    render();
  }

  function syncPrompt() {
    promptWidget.value = textarea.value;
    renderOverlay();
  }

  function renderOverlay() {
    const list = readRefs() || [];
    const known = new Set(effectiveNames(list));
    const text = textarea.value;
    let html = "";
    let last = 0;
    for (const m of text.matchAll(MENTION_RE)) {
      html += escapeHtml(text.slice(last, m.index));
      const cls = known.has(m[1].toLowerCase())
        ? "h3tools-mention" : "h3tools-mention-unknown";
      html += `<span class="${cls}">${escapeHtml(m[0])}</span>`;
      last = m.index + m[0].length;
    }
    html += escapeHtml(text.slice(last)) + "\n";
    highlight.innerHTML = html;
    highlight.scrollTop = textarea.scrollTop;
    highlight.scrollLeft = textarea.scrollLeft;
  }

  // ---- board rendering
  function render() {
    const list = readRefs();
    grid.textContent = "";
    if (!list) {
      // still paint the prompt (the textarea text is transparent — the
      // overlay is what the user reads) and keep the counters honest
      counters.textContent = "references inválido";
      renderOverlay();
      return;
    }
    const names = effectiveNames(list);
    const tags = computeTags(list);
    const counts = { image: 0, video: 0, audio: 0 };
    for (const r of list) if (counts[r.type] !== undefined) counts[r.type]++;
    counters.textContent =
      `imagens ${counts.image}/${CAPS.image} · ` +
      `vídeos ${counts.video}/${CAPS.video} · ` +
      `áudios ${counts.audio}/${CAPS.audio}`;
    for (const [type, button] of Object.entries(addButtons)) {
      button.disabled = counts[type] >= CAPS[type];
    }
    list.forEach((ref, i) => grid.appendChild(makeTile(ref, i, names[i], tags[i], list)));
    renderOverlay();
  }

  function makeTile(ref, index, name, tagInfo, list) {
    const tile = document.createElement("div");
    tile.className = "h3tools-tile";
    tile.dataset.type = ref.type;

    const thumb = document.createElement("div");
    thumb.className = "h3tools-thumb";
    setThumb(thumb, ref);
    if (ref.type === "audio") {
      thumb.addEventListener("click", () => toggleAudio(ref.file));
      thumb.title = "tocar / pausar";
    }

    const nameEl = document.createElement("div");
    nameEl.className = "h3tools-name";
    nameEl.textContent = "@" + name;
    nameEl.title = "clique para renomear";
    nameEl.addEventListener("click", () => startRename(nameEl, ref, name, list));

    const tagline = document.createElement("div");
    tagline.className = "h3tools-tagline";
    const badge = document.createElement("span");
    badge.className = "h3tools-badge";
    badge.textContent = ref.type;
    const tagEl = document.createElement("span");
    tagEl.textContent = tagInfo.soundtrackTag
      ? `${tagInfo.tag} ${TYPE_ICONS.audio} ${tagInfo.soundtrackTag}`
      : (tagInfo.tag || "");
    tagline.append(badge, tagEl);

    const del = document.createElement("div");
    del.className = "h3tools-del";
    del.textContent = "✕";
    del.title = "remover referência";
    del.addEventListener("click", () => {
      if (ref.type === "audio") stopAudio(ref.file);
      list.splice(index, 1);
      writeRefs(list);
    });

    tile.append(thumb, nameEl, tagline, del);

    if (ref.type === "video") {
      const snd = document.createElement("div");
      snd.className = "h3tools-snd" + (ref.use_soundtrack ? " h3tools-snd-on" : "");
      snd.textContent = TYPE_ICONS.audio;
      snd.title = ref.use_soundtrack
        ? "trilha sonora ativa (o backend falha se o arquivo não tiver áudio)"
        : "usar trilha sonora do vídeo";
      snd.addEventListener("click", () => {
        if (ref.use_soundtrack) delete ref.use_soundtrack;
        else ref.use_soundtrack = true;
        writeRefs(list);
      });
      tile.appendChild(snd);
    }
    return tile;
  }

  function startRename(nameEl, ref, currentName, list) {
    const input = document.createElement("input");
    input.value = currentName;
    nameEl.textContent = "";
    nameEl.appendChild(input);
    input.focus();
    input.select();
    let done = false;
    const restore = () => { nameEl.textContent = "@" + currentName; };
    const commit = () => {
      if (done) return;
      done = true;
      const value = input.value.trim();
      // no rebuild on a no-op: a rebuild would detach whatever the user is
      // mousedown-ing on and swallow that click
      if (!value || value === currentName) { restore(); return; }
      const others = new Set(effectiveNames(list));
      others.delete(currentName);
      if (!NAME_RE.test(value)) {
        showError(`nome inválido "${value}" — use ^[a-z0-9_]{1,64}$`);
        restore();
        return;
      }
      if (others.has(value)) {
        showError(`nome "${value}" já existe`);
        restore();
        return;
      }
      // freeze derived names first so freeing this name can't retarget one
      const names = effectiveNames(list);
      list.forEach((r, i) => { if (!r.name) r.name = names[i]; });
      ref.name = value;
      // keep existing mentions working: rewrite @old -> @new in the prompt
      const re = new RegExp(
        `(?<![A-Za-z0-9_])@${escapeRegExp(currentName)}(?![A-Za-z0-9_])`, "gi");
      const rewritten = textarea.value.replace(re, "@" + value);
      if (rewritten !== textarea.value) {
        replaceRange(textarea, 0, textarea.value.length, rewritten);
        syncPrompt();
      }
      writeRefs(list);
    };
    input.addEventListener("blur", commit);
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); input.blur(); }
      if (e.key === "Escape") { done = true; render(); }
    });
  }

  // ---- uploads
  const addButtons = {};
  for (const type of ["image", "video", "audio"]) {
    const button = document.createElement("button");
    button.textContent = "+ " + TYPE_LABELS[type];
    button.addEventListener("click", () => {
      const input = document.createElement("input");
      input.type = "file";
      input.accept = ACCEPT[type];
      input.multiple = type === "image";
      input.onchange = () => addFiles(type, [...input.files]);
      input.click();
    });
    addButtons[type] = button;
    toolbar.appendChild(button);
  }
  toolbar.appendChild(counters);

  async function addFiles(type, files) {
    for (const file of files) {
      let list = readRefs();
      if (!list) return;
      if (list.filter((r) => r.type === type).length >= CAPS[type]) {
        showError(`limite de ${CAPS[type]} referências de tipo ${type} atingido`);
        break;
      }
      try {
        const body = new FormData();
        body.append("image", file);
        body.append("type", "input");
        body.append("subfolder", SUBFOLDER);
        const resp = await api.fetchApi("/upload/image", { method: "POST", body });
        if (resp.status !== 200) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        // re-read: the board may have changed while the upload was in flight
        list = readRefs();
        if (!list) return;
        // simple sequential names per type — short, predictable, and the LLM
        // never typos them; rename on the tile when a semantic name helps
        const taken = new Set(effectiveNames(list));
        const base = { image: "imagem", video: "video", audio: "audio" }[type];
        let name;
        for (let n = 1; ; n++) {
          name = `${base}_${n}`;
          if (!taken.has(name)) break;
        }
        list.push({ name, type, file: `${SUBFOLDER}/${data.name}` });
        writeRefs(list);
      } catch (e) {
        showError(`upload de "${file.name}" falhou: ${e.message || e}`);
      }
    }
  }

  // ---- mention popup wiring
  function popupItems(filter) {
    const list = readRefs() || [];
    const names = effectiveNames(list);
    const tags = computeTags(list);
    const order = { image: 0, video: 1, audio: 2 };
    return list
      .map((ref, i) => ({ ref, name: names[i], tag: tags[i].tag }))
      .filter((item) => item.name.startsWith(filter.toLowerCase()))
      .sort((a, b) => (order[a.ref.type] ?? 3) - (order[b.ref.type] ?? 3));
  }

  function insertMention(name) {
    const upto = textarea.value.slice(0, textarea.selectionStart);
    const match = TRIGGER_RE.exec(upto);
    if (!match) { closePopup(); return; }
    const start = upto.length - match[0].length;
    // consume the word chars right of the caret too (caret mid-token)
    const rest = textarea.value.slice(textarea.selectionStart);
    const rightLen = (/^[A-Za-z0-9_]*/.exec(rest) || [""])[0].length;
    const end = textarea.selectionStart + rightLen;
    closePopup();
    replaceRange(textarea, start, end, "@" + name + " ");
    syncPrompt();
  }

  function maybePopup() {
    const upto = textarea.value.slice(0, textarea.selectionStart);
    const match = TRIGGER_RE.exec(upto);
    if (!match) { closePopup(); return; }
    openPopup(textarea, popupItems(match[1]), insertMention);
  }

  textarea.addEventListener("input", () => { syncPrompt(); maybePopup(); });
  textarea.addEventListener("scroll", () => {
    highlight.scrollTop = textarea.scrollTop;
    highlight.scrollLeft = textarea.scrollLeft;
  });
  textarea.addEventListener("click", maybePopup);
  textarea.addEventListener("blur", closePopup);
  textarea.addEventListener("keydown", (e) => {
    // no stopPropagation: ComfyUI's keybinding service already ignores plain
    // keys in textareas, and stopping it would kill Ctrl+Enter (queue) etc.
    if (e.isComposing) return; // let the IME own Enter/arrows mid-composition
    if (!popup) return;
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      const n = popup.items.length;
      if (n) {
        popup.index = (popup.index + (e.key === "ArrowDown" ? 1 : n - 1)) % n;
        renderPopupRows();
      }
    } else if (e.key === "Enter" || e.key === "Tab") {
      if (popup.items.length) {
        e.preventDefault();
        popup.insert(popup.items[popup.index].name);
      } else {
        closePopup();
      }
    } else if (e.key === "Escape") {
      e.preventDefault();
      closePopup();
    }
  });
  textarea.addEventListener("keyup", (e) => {
    if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) maybePopup();
  });

  // ---- attach as a DOM widget (kept LAST so widgets_values stays aligned
  // for workflows loaded without this extension)
  const boardWidget = node.addDOMWidget("h3_board", "div", root, { serialize: false });
  // options.serialize=false excludes it from the API prompt; the top-level
  // flag is what LGraphNode.serialize checks for workflow persistence
  if (boardWidget) boardWidget.serialize = false;

  node.__h3refresh = () => {
    textarea.value = promptWidget.value ?? "";
    render();
  };
  node.__h3cleanup = () => {
    closePopup();
    stopAudio();
    clearTimeout(errorTimer);
  };

  node.__h3refresh();
  const size = node.computeSize();
  node.setSize([Math.max(node.size[0], 380), Math.max(node.size[1], size[1], 560)]);
}

// ---- extension ------------------------------------------------------------

app.registerExtension({
  name: "h3_tools.board",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODE_IDS.has(nodeData.name)) return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);
      try {
        setupNode(this);
      } catch (e) {
        console.error("h3_tools: UI setup failed; plain widgets remain usable", e);
      }
    };
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      onConfigure?.apply(this, arguments);
      this.__h3refresh?.();
    };
    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      this.__h3cleanup?.();
      onRemoved?.apply(this, arguments);
    };
  },
});
