"use strict";

const $ = (id) => document.getElementById(id);
const STATUS_ICON = { HELMET: "i-check", NO_HELMET: "i-alert", UNKNOWN: "i-help" };
const STATUS_TEXT = { HELMET: "Helmet Detected", NO_HELMET: "No Helmet", UNKNOWN: "Checking..." };
const ADD_CAMERA = "__add_camera__";

let state = null;
let streamOn = false;
let shownAlertId = null;
let openAlertId = null;
let lastRecordingsCheck = 0;
let recordingsKey = "";

function icon(name, cls = "icon") {
  return `<svg class="${cls}"><use href="#${name}"/></svg>`;
}

function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function toast(text, bad = false) {
  const el = $("toast");
  el.textContent = text;
  el.className = "toast" + (bad ? " bad" : "");
  el.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (el.hidden = true), bad ? 6000 : 2500);
}

async function api(path, body) {
  const options = body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  };
  const response = await fetch(path, options);
  let data = {};
  try { data = await response.json(); } catch (e) { /* empty body */ }
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}

/* ---------- Camera list ---------- */

async function loadSources(scan = false) {
  try {
    const data = await api("/api/sources" + (scan ? "?scan=1" : ""));
    const select = $("source");
    const groups = { webcam: "This computer", phone: "Phone", network: "IP / CCTV cameras", screen: "Screen", file: "Video", other: "Current" };
    select.innerHTML = "";
    const byKind = {};
    for (const item of data.sources) (byKind[item.kind] ||= []).push(item);
    for (const kind of ["other", "webcam", "phone", "network", "screen", "file"]) {
      if (!byKind[kind]) continue;
      const group = document.createElement("optgroup");
      group.label = groups[kind];
      for (const item of byKind[kind]) {
        const option = new Option(item.label, item.source);
        if (item.detail) option.title = item.detail;
        group.append(option);
      }
      select.append(group);
    }
    if (!byKind.webcam) {
      const group = document.createElement("optgroup");
      group.label = groups.webcam;
      const option = new Option("No webcam found (plug in, then pick again)", "__rescan__");
      group.append(option);
      select.prepend(group);
    }
    const extra = document.createElement("optgroup");
    extra.label = "More";
    extra.append(new Option("+ Add IP / CCTV / phone app camera...", ADD_CAMERA));
    extra.append(new Option("Search for webcams again", "__rescan__"));
    select.append(extra);
    select.value = data.current;
    select.dataset.current = data.current;
  } catch (error) {
    toast(error.message, true);
  }
}

async function chooseSource(source) {
  const select = $("source");
  if (source === ADD_CAMERA) {
    select.value = select.dataset.current;
    openDialog("dlg-camera");
    $("cam-address").focus();
    return;
  }
  if (source === "__rescan__") {
    select.value = select.dataset.current;
    toast("Searching for webcams...");
    await loadSources(true);
    toast("Camera list updated.");
    return;
  }
  if (source === "phone") showPhone();
  showOverlay(true, "Connecting to the camera...");
  try {
    await api("/api/source", { source });
    select.dataset.current = source;
    shownAlertId = null;
    resetStream();
  } catch (error) {
    select.value = select.dataset.current;
    toast(error.message, true);
  }
}

/* ---------- Video ---------- */

function resetStream() {
  streamOn = false;
}

function updateStream(running) {
  const img = $("stream");
  if (running && !streamOn) {
    img.src = "/stream.mjpg?t=" + Date.now();
    streamOn = true;
  } else if (!running && streamOn) {
    // Keep the last picture; just stop the connection.
    fetch("/api/frame.jpg").then((r) => (r.status === 200 ? r.blob() : null)).then((blob) => {
      if (blob) img.src = URL.createObjectURL(blob);
    });
    streamOn = false;
  }
}

function showOverlay(show, text = "", spinner = true, retry = false) {
  $("overlay").hidden = !show;
  $("overlay-text").textContent = text;
  $("spinner").hidden = !spinner;
  $("overlay-retry").hidden = !retry;
}

function formatTime(seconds) {
  seconds = Math.max(0, Math.floor(seconds));
  const h = Math.floor(seconds / 3600), m = Math.floor((seconds % 3600) / 60), s = seconds % 60;
  const mm = String(m).padStart(2, "0"), ss = String(s).padStart(2, "0");
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

/* ---------- Render state ---------- */

function render(s) {
  const running = s.state === "running";
  const waitingPhone = running && s.source === "phone" && !s.phone_connected && !s.counts.people && !s.fps;

  $("scan").className = "scan" + (running ? " on" : s.state === "error" ? " err" : "");
  $("scan-text").textContent = running ? (s.live ? "Scanning in real time..." : "Scanning video...")
    : s.state === "error" ? "Camera problem" : s.state === "ended" ? "Video finished"
    : s.state === "stopped" ? "Stopped" : "Starting...";

  if (s.state === "error") showOverlay(true, s.message, false, true);
  else if (s.state === "loading" || s.state === "starting") showOverlay(true, s.message || "Starting...");
  else if (waitingPhone) showOverlay(true, "Waiting for the phone... scan the QR code and tap Start camera.", true);
  else if (s.state === "stopped") showOverlay(true, "Monitoring is stopped. Press play to start again.", false, true);
  else showOverlay(false);
  $("overlay-retry").textContent = s.state === "stopped" ? "Start" : "Try again";

  updateStream(running);
  $("live-badge").classList.toggle("off", !running);
  $("live-badge").lastChild.textContent = s.live ? "LIVE" : "VIDEO";
  $("rec-badge").hidden = !(running && (s.recording_file || s.clip_recording));
  $("rec-badge").title = s.clip_recording && !s.recording_file ? "Recording the alert clip" : "Recording";
  $("btn-record").classList.toggle("on", s.recording);
  $("btn-record").title = s.recording ? "Stop recording" : "Record video";
  $("btn-stop").innerHTML = icon(running || s.state === "loading" || s.state === "starting" ? "i-stop" : "i-play");
  $("btn-stop").title = running ? "Stop" : "Start";
  $("timer").textContent = running && s.started_at ? formatTime(Date.now() / 1000 - s.started_at) : "00:00";
  $("fps").textContent = running && s.fps ? `${s.fps} FPS` : "";
  $("alerts-on").checked = s.alerts_enabled;

  const summary = s.summary || {};
  $("s-people").textContent = s.counts.people;
  $("s-helmet").textContent = s.counts.helmet;
  $("s-nohelmet").textContent = s.counts.no_helmet;
  $("s-total").textContent = summary.total_people ?? 0;
  $("s-alerts").textContent = summary.alerts ?? 0;
  $("s-rate").textContent = Math.round((summary.violation_rate ?? 0) * 100) + "%";

  renderAlert(s.alerts);
  renderPeople(s.people);
  if (openAlertId !== null) renderAlertDialog(s.alerts.find((a) => a.id === openAlertId));
  if ($("dlg-phone").open) {
    const status = $("phone-status");
    status.classList.toggle("on", s.phone_connected);
    status.lastChild.textContent = s.phone_connected
      ? (s.source === "phone" ? "Phone connected - monitoring the phone camera." : "Phone connected.")
      : "Waiting for the phone...";
  }
}

function renderAlert(alerts) {
  const active = alerts.filter((a) => !a.dismissed);
  const box = $("alert-box");
  if (!active.length) {
    box.innerHTML = "";
    shownAlertId = null;
    return;
  }
  const alert = active[0];
  if (alert.id === shownAlertId && box.firstChild) {
    box.querySelector(".more").textContent = active.length > 1 ? `+ ${active.length - 1} more alert(s)` : "";
    return;
  }
  shownAlertId = alert.id;
  box.innerHTML = `
    <div class="alert" role="alert">
      <div class="top">
        ${icon("i-alert")}
        <div class="msg"><b>Alert!</b><span>${escapeHtml(alert.message)}</span><small>${alert.time}</small></div>
      </div>
      <div class="buttons">
        <button class="btn danger" data-view="${alert.id}">View Details</button>
        <button class="btn danger-soft" data-dismiss="${alert.id}">Dismiss</button>
      </div>
      <div class="more">${active.length > 1 ? `+ ${active.length - 1} more alert(s)` : ""}</div>
    </div>`;
}

function renderPeople(people) {
  const list = $("people");
  if (!people.length) {
    if (!list.querySelector(".empty")) list.innerHTML = '<p class="empty">No one detected yet.</p>';
    return;
  }
  list.querySelector(".empty")?.remove();
  const seen = new Set();
  const now = Date.now();
  people.forEach((person, index) => {
    const key = "p" + person.id;
    seen.add(key);
    let card = $(key);
    if (!card) {
      card = document.createElement("div");
      card.className = "person";
      card.id = key;
      card.tabIndex = 0;
      card.dataset.person = person.id;
      card.innerHTML = `<div class="ph"></div><div class="info"><span class="name"></span><span class="chip"></span><span class="when"></span></div>${icon("i-chevron")}`;
    }
    if (list.children[index] !== card) list.insertBefore(card, list.children[index] || null);

    card.classList.toggle("gone", !person.in_view);
    card.querySelector(".name").textContent = `Person ${person.id}`;
    const chip = card.querySelector(".chip");
    if (chip.dataset.status !== person.status) {
      chip.dataset.status = person.status;
      chip.className = "chip " + person.status;
      chip.innerHTML = icon(STATUS_ICON[person.status]) + STATUS_TEXT[person.status];
    }
    card.querySelector(".when").textContent = person.in_view ? person.first_seen : `${person.first_seen} · left`;

    const due = !card.dataset.photoAt || (person.in_view && now - Number(card.dataset.photoAt) > 2000);
    if (person.has_photo && due) {
      const img = new Image();
      img.alt = `Person ${person.id}`;
      img.onload = () => card.querySelector(".ph, img").replaceWith(img);
      img.src = `/api/people/${person.id}.jpg?t=${now}`;
      card.dataset.photoAt = now;
    }
  });
  for (const card of [...list.children]) if (!seen.has(card.id)) card.remove();
}

function renderAlertDialog(alert) {
  if (!alert) return;
  $("dlg-alert-title").innerHTML = `${icon("i-alert")} Person ${alert.track_id} - No Helmet`;
  const media = $("dlg-alert-media");
  const want = alert.clip ? "clip:" + alert.clip : alert.clip_recording ? "recording" : "image";
  if (media.dataset.showing !== want) {
    media.dataset.showing = want;
    if (alert.clip) {
      media.innerHTML = `<video src="/media/recordings/${encodeURIComponent(alert.clip)}" controls autoplay muted playsinline></video>
        <p class="note">Video clip of the alert. <a href="/media/recordings/${encodeURIComponent(alert.clip)}" download>Download</a></p>`;
    } else {
      media.innerHTML = `<img src="/api/alerts/${alert.id}.jpg" alt="Picture at the moment of the alert">
        <p class="note">${alert.clip_recording ? "Recording the video clip... it appears here when the person has left or put a helmet on."
          : "Picture at the moment of the alert."}</p>`;
    }
  }
  $("dlg-alert-facts").innerHTML = `
    <dt>Time</dt><dd>${alert.date} ${alert.time}</dd>
    <dt>Person</dt><dd>#${alert.track_id} (temporary number for this session)</dd>
    <dt>Confidence</dt><dd>${alert.confidence !== null ? Math.round(alert.confidence * 100) + "%" : "-"}</dd>
    <dt>Camera</dt><dd>${escapeHtml(state?.source_label || "")}</dd>
    <dt>Status</dt><dd>${alert.dismissed ? "Dismissed" : "Open"}</dd>`;
  $("dlg-alert-dismiss").hidden = alert.dismissed;
}

async function loadRecordings() {
  try {
    const data = await api("/api/recordings");
    const key = data.recordings.map((r) => r.file + r.size_mb).join();
    if (key === recordingsKey) return;
    recordingsKey = key;
    const list = $("rec-list");
    if (!data.recordings.length) {
      list.innerHTML = '<p class="empty">No recordings yet.</p>';
      return;
    }
    list.innerHTML = data.recordings.map((r) => `
      <a class="rec-item${r.alert ? " alert-clip" : ""}" href="${r.url}" target="_blank" rel="noopener" title="${escapeHtml(r.file)}">
        ${icon(r.alert ? "i-alert" : "i-film")}
        <span class="name">${r.alert ? "Alert clip" : "Recording"} · ${r.time}</span>
        <span class="size">${r.size_mb} MB</span>
      </a>`).join("");
  } catch (error) { /* try again next time */ }
}

/* ---------- Dialogs ---------- */

function openDialog(id) {
  const dialog = $(id);
  if (!dialog.open) dialog.showModal();
}

document.addEventListener("click", (event) => {
  const close = event.target.closest("[data-close]");
  if (close) close.closest("dialog").close();
  const view = event.target.closest("[data-view]");
  if (view) {
    openAlertId = Number(view.dataset.view);
    $("dlg-alert-media").dataset.showing = "";
    renderAlertDialog(state.alerts.find((a) => a.id === openAlertId));
    openDialog("dlg-alert");
  }
  const personCard = event.target.closest("[data-person]");
  if (personCard) openPerson(Number(personCard.dataset.person));
  const dismiss = event.target.closest("[data-dismiss]");
  if (dismiss) dismissAlert(Number(dismiss.dataset.dismiss));
});

for (const dialog of document.querySelectorAll("dialog")) {
  dialog.addEventListener("click", (event) => { if (event.target === dialog) dialog.close(); });
}
$("dlg-alert").addEventListener("close", () => {
  openAlertId = null;
  $("dlg-alert-media").innerHTML = "";
  $("dlg-alert-media").dataset.showing = "";
});
$("dlg-alert-dismiss").addEventListener("click", () => {
  if (openAlertId !== null) dismissAlert(openAlertId);
  $("dlg-alert").close();
});

document.addEventListener("keydown", (event) => {
  const card = event.target.closest?.("[data-person]");
  if (card && (event.key === "Enter" || event.key === " ")) {
    event.preventDefault();
    openPerson(Number(card.dataset.person));
  }
});

function openPerson(id) {
  // A person with an alert opens that alert; anyone else shows their photo and status.
  const alert = state?.alerts.find((a) => a.track_id === id);
  if (alert) {
    openAlertId = alert.id;
    $("dlg-alert-media").dataset.showing = "";
    renderAlertDialog(alert);
    openDialog("dlg-alert");
    return;
  }
  const person = state?.people.find((p) => p.id === id);
  if (!person) return;
  openAlertId = null;
  $("dlg-alert-title").innerHTML = `${icon(STATUS_ICON[person.status])} Person ${person.id} - ${STATUS_TEXT[person.status]}`;
  const media = $("dlg-alert-media");
  media.dataset.showing = "person";
  media.innerHTML = person.has_photo
    ? `<img class="portrait" src="/api/people/${person.id}.jpg?t=${Date.now()}" alt="Person ${person.id}">`
    : '<p class="note">No photo yet.</p>';
  $("dlg-alert-facts").innerHTML = `
    <dt>Status</dt><dd>${STATUS_TEXT[person.status]}</dd>
    <dt>First seen</dt><dd>${person.first_seen}</dd>
    <dt>Last seen</dt><dd>${person.last_seen}${person.in_view ? " (in view now)" : ""}</dd>
    <dt>Person</dt><dd>#${person.id} (temporary number for this session)</dd>
    <dt>Camera</dt><dd>${escapeHtml(state?.source_label || "")}</dd>`;
  $("dlg-alert-dismiss").hidden = true;
  openDialog("dlg-alert");
}

async function dismissAlert(id) {
  await api(`/api/alerts/${id}/dismiss`, {}).catch((e) => toast(e.message, true));
  poll();
}

async function showPhone() {
  openDialog("dlg-phone");
  try {
    const data = await api("/api/phone");
    $("phone-qr").innerHTML = data.qr;
    $("phone-url").textContent = data.url;
  } catch (error) {
    $("phone-qr").textContent = "Could not start the phone page.";
    toast(error.message, true);
  }
}

$("camera-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = $("cam-connect");
  const error = $("cam-error");
  error.hidden = true;
  button.disabled = true;
  button.textContent = "Connecting...";
  try {
    await api("/api/cameras", { name: $("cam-name").value, address: $("cam-address").value.trim() });
    $("dlg-camera").close();
    $("camera-form").reset();
    resetStream();
    await loadSources();
    toast("Camera connected.");
  } catch (e) {
    error.textContent = e.message;
    error.hidden = false;
  } finally {
    button.disabled = false;
    button.textContent = "Connect";
  }
});

/* ---------- Controls ---------- */

$("source").addEventListener("change", (event) => chooseSource(event.target.value));
$("btn-record").addEventListener("click", async () => {
  const on = !state?.recording;
  await api("/api/record", { on }).catch((e) => toast(e.message, true));
  toast(on ? "Recording started." : "Recording saved.");
  setTimeout(loadRecordings, 1500);
  poll();
});
$("btn-snapshot").addEventListener("click", async () => {
  try {
    const data = await api("/api/snapshot", {});
    toast(`Photo saved: snapshots/${data.file}`);
  } catch (e) { toast(e.message, true); }
});
$("btn-stop").addEventListener("click", async () => {
  const running = state && ["running", "loading", "starting"].includes(state.state);
  showOverlay(true, running ? "Stopping..." : "Starting...");
  await api(running ? "/api/stop" : "/api/start", {}).catch((e) => toast(e.message, true));
  resetStream();
  poll();
});
$("overlay-retry").addEventListener("click", async () => {
  showOverlay(true, "Starting...");
  await api("/api/start", {}).catch((e) => toast(e.message, true));
  resetStream();
  poll();
});
$("btn-full").addEventListener("click", () => {
  if (document.fullscreenElement) document.exitFullscreen();
  else $("video").requestFullscreen?.();
});
$("alerts-on").addEventListener("change", (event) => {
  api("/api/alerts-enabled", { on: event.target.checked }).then(() =>
    toast(event.target.checked ? "Alerts on." : "Alerts off."));
});

/* ---------- Polling ---------- */

async function poll() {
  try {
    state = await api("/api/state");
    render(state);
    if (Date.now() - lastRecordingsCheck > 3000) {
      lastRecordingsCheck = Date.now();
      loadRecordings();
    }
  } catch (error) {
    $("scan").className = "scan err";
    $("scan-text").textContent = "Dashboard not running";
    showOverlay(true, "Cannot reach the app. Is python main.py --web still running?", false);
    streamOn = false;
  }
}

loadSources();
poll();
setInterval(poll, 500);
setTimeout(() => loadSources(), 4000);   // The webcam search finishes in the background
