// /compute dashboard. Polls /api/overview and updates the widgets in place;
// forms are static markup, so polling never clobbers what you are typing.

const T = 1e12;
const SETTING_KEYS = [
  "mode", "url", "gpu_percent", "contribute", "finish", "session_token", "grant_split",
  "training", "memory_gb", "inference", "inference_memory_gb", "inference_head", "models_dir", "transport",
];
const GIB = 1024 ** 3;
const GRANTS_REFRESH_MS = 20000;
const POLL_MS = 2000;
const API_TIMEOUT_MS = 20000;       // polls: a hung one must not hold up the next
const ACTION_TIMEOUT_MS = 120000;   // starts, stops, saves: the controller may wait on a process
// Reads made every poll or on a timer; everything else is an action (or the dataset upload).
const POLL_PATHS = ["/api/overview", "/api/grants", "/api/coord/auth/me", "/api/coord/auth/terms", "/api/status", "/inference/status", "/v1/models"];
const MAX_UPLOAD_GIB = 64;          // the shell refuses a bigger GGUF: no Mac pool loads one
const RATE_WINDOW = 5;          // polls averaged into the "now" rate
const OUTDATED_COORDINATOR = "This pool's coordinator has no LLM inference: it runs an older /compute. "
  + "Ask whoever hosts it to update and restart it, or host a pool on this Mac.";
const STATUS_TONE = {
  running: "ok", completed: "line", starting: "line", queued: "warn", recovering: "warn",
  failed: "hot", cancelled: "",
};

const state = {
  tab: "contributions",
  ov: null,
  settings: null,
  saving: 0,          // saves queued or in flight
  settingsVersion: 0, // bumped when a save starts: a poll begun earlier may not overwrite it
  polling: false,
  pollSeq: 0,         // the latest poll sent
  pollApplied: 0,     // the latest poll whose answer was applied; older answers are dropped
  rates: [],
  last: null,
  rateKey: "",        // pool + node the FLOPs counter belongs to; a change restarts the samples
  grants: null,
  grantsUp: null,     // coordinator_up when the board was last loaded
  grantsAt: 0,
  grantsSeq: 0,
  sort: "top",
  admin: false,
  fundOpen: null,
  fundMsg: "",
  busy: new Set(),
  dirty: { url: false, publicUrl: false },   // address fields being edited: polls leave them alone
  dragging: new Set(),                        // slider ids mid-drag: polls leave them alone
  grantsWho: "",      // user id + admin flag the grant board was rendered for
  dataset: null,
  modelsShown: "",
  keys: {},
  user: null,
  credits: null,
  authMode: "login",
  terms: "",
  llm: {
    net: null,          // /inference/status
    models: [],         // /v1/models: servable right now
    model: "",
    messages: [],       // {role, content, meta?, error?}
    streaming: false,
    abort: null,
    upload: null,       // {name, pct} while a GGUF is on its way
    error: "",          // why the pool's LLM service could not be read
    why: null,          // why Send is unavailable: {text, tone?, action?}
    starting: false,    // the pending Start/Stop serving click is a start
  },
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

// Gives up on a request that gets no answer: a hung poll must not block the next one.
function timeoutSignal(ms) {
  if (typeof AbortSignal !== "undefined" && typeof AbortSignal.timeout === "function") return AbortSignal.timeout(ms);
  if (typeof AbortController === "undefined") return undefined;   // no abort support: just no timeout
  const ctl = new AbortController();
  window.setTimeout(() => ctl.abort(), ms);
  return ctl.signal;
}

// The message in an error body: {"detail": "..."} from the shell, FastAPI's 422 list of {msg},
// or an OpenAI-style {"error": {"message"}} from the coordinator.
function detailText(data, fallback) {
  const d = data && typeof data === "object" ? (data.detail ?? (data.error && data.error.message)) : null;
  if (typeof d === "string" && d) return d;
  if (Array.isArray(d)) return d.map((x) => (x && x.msg) || JSON.stringify(x)).join("; ");
  if (d != null) return JSON.stringify(d);
  return fallback;
}

// How long a request may take: a poll gives up fast so the next is not stuck behind it; an action
// (start, stop, save) waits for the controller; `timeout: 0` (the dataset upload) waits as long as it takes.
function timeoutFor(path, opts) {
  if (opts.timeout != null) return opts.timeout;
  const p = String(path).split("?")[0];
  const read = (opts.method || "GET").toUpperCase() === "GET";
  return read && POLL_PATHS.some((x) => p === x || p.endsWith(x)) ? API_TIMEOUT_MS : ACTION_TIMEOUT_MS;
}

async function api(path, opts = {}) {
  const ms = timeoutFor(path, opts);
  const rest = { ...opts };
  delete rest.timeout;   // ours, not fetch's
  const next = { signal: ms ? timeoutSignal(ms) : undefined, ...rest, credentials: "include", headers: { ...(opts.headers || {}) } };
  if (next.body && !(next.body instanceof FormData) && !next.headers["content-type"]) {
    next.headers["content-type"] = "application/json";
  }
  const r = await fetch(path, next);
  const text = await r.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = text; }
  if (!r.ok) {
    const err = new Error(detailText(data, text || r.statusText));
    err.status = r.status;
    throw err;
  }
  return data;
}

const post = (path, body) => api(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });
const patch = (path, body) => api(path, { method: "PATCH", body: JSON.stringify(body) });
const signedIn = () => !!(state.user && state.user.id);
// The controller's "Restarting the training agent / the LLM node after the current step finishes":
// news that new settings are on their way, not a failure. Only the tone of a toast or the banner
// hangs on it; what the agent is doing comes from the status flags.
const isRestartNote = (msg) => typeof msg === "string" && msg.startsWith("Restarting ");
// The agent drains with a restart queued behind it, to start again with new settings: on screen a
// restart, not a stop.
const agentRestarting = (st) => !!(st.agent_running && st.agent_draining && st.agent_restart_pending);

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

// ------------------------------------------------------------ formatting

// Scale |x| by `base` through `units`, rounding before the unit is chosen: 999.6 reads "1.00 K",
// never "1000", and 1023.99 bytes read "1.0 KB", never "1024.0 B".
function scaled(x, base, units, fmt) {
  const n = Number(x) || 0;
  let v = Math.abs(n);
  let i = 0;
  while (i < units.length - 1 && Number(fmt(v)) >= base) {
    v /= base;
    i += 1;
  }
  return { text: `${n < 0 ? "-" : ""}${fmt(v)}`, unit: units[i] };
}

const sigFigs = (v) => (v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2));

function fmtFlops(x) {
  if (!(Number(x) || 0)) return "0";
  const { text, unit } = scaled(x, 1000, ["", "K", "M", "G", "T", "P", "E"], sigFigs);
  return unit ? `${text} ${unit}` : text;
}

function withUnit(x, unit = "FLOPs") {
  const s = fmtFlops(x);
  return /[A-Z]$/.test(s) ? s + unit : `${s} ${unit}`;
}

function fmtBytes(n) {
  if (!(Number(n) || 0)) return "0 B";
  const { text, unit } = scaled(n, 1024, ["B", "KB", "MB", "GB", "TB"], (v) => (v >= 100 ? v.toFixed(0) : v.toFixed(1)));
  return `${text} ${unit}`;
}

const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
const shortModel = (m) => String(m || "").split("/").pop();
const modelLabel = (m) => shortModel(m).replace(/-Instruct-4bit$/, "").replace(/-/g, " ");
const setText = (sel, text) => { const el = $(sel); if (el && el.textContent !== text) el.textContent = text; };

function setTag(sel, text, tone) {
  const el = $(sel);
  el.textContent = text;
  el.className = `tag ${tone || ""}`.trim();
}

function setDot(sel, tone, live = false) {
  const el = $(sel);
  el.className = `sq ${tone || ""} ${live ? "live" : ""}`.trim();
}

// Re-render a list only when its data changed, so hover and focus survive polls.
function renderOnce(key, data, el, html) {
  const sig = JSON.stringify(data);
  if (state.keys[key] === sig) return false;
  state.keys[key] = sig;
  el.innerHTML = html();
  return true;
}

function rangeFill(el) {
  const pct = ((el.value - el.min) / (el.max - el.min)) * 100;
  el.style.setProperty("--pct", `${pct}%`);
}

// ------------------------------------------------------------ memory sliders

// Both sliders stop at the GPU working set (75% of RAM); 0 is "Auto". Until the service has
// reported this Mac's memory nothing is assumed: `known` is false and the labels say a plain "Auto".
function memoryLimits() {
  const st = status();
  const total = (Number(st.memory_total_bytes) || 0) / GIB;
  const known = total > 0;
  return { total, known, cap: known ? Math.max(1, Math.round(total * 0.75)) : 0, free: (Number(st.memory_available_bytes) || 0) / GIB };
}

const trainingAutoGb = () => { const m = memoryLimits(); return m.known && m.free > 0 ? Math.min(m.free, m.cap) : 0; };   // what is free at start
const inferenceAutoGb = () => { const m = memoryLimits(); return m.known ? Math.max(2, Math.round(m.total * 0.75 - 4)) : 0; };
const memLabel = (gb, autoGb) => (gb ? `${gb} GB` : autoGb > 0 ? `Auto · ≈${Math.round(autoGb)} GB` : "Auto");
const upTo = () => (memoryLimits().known ? ` Up to ${memoryLimits().cap} GB.` : "");

function syncMemSlider(el, out, saved, autoGb) {
  const { cap } = memoryLimits();
  el.max = String(Math.max(cap || Number(el.max) || 12, Number(saved) || 0));
  if (!state.dragging.has(el.id) && saved != null) el.value = saved;
  rangeFill(el);
  out.textContent = memLabel(Number(el.value), autoGb);
}

function toast(text, tone = "ok") {
  const el = $("#toast");
  el.querySelector("span").textContent = text;
  el.className = `toast ${tone === "bad" ? "bad" : ""}`.trim();
  el.hidden = false;
  window.clearTimeout(toast.timer);
  toast.timer = window.setTimeout(() => { el.hidden = true; }, tone === "bad" ? 6000 : 3500);
}

// ------------------------------------------------------------ derived values

const status = () => (state.ov && state.ov.status) || {};
const pool = () => (state.ov && state.ov.pool) || { online: false, nodes: [], jobs: [], capacity: {} };
const me = () => (state.ov && state.ov.me) || { flops: 0 };
// The share the running agent actually uses; a slider move only saves the next one.
const liveGpuPercent = () => status().agent_gpu_percent
  ?? (me().node || {}).gpu_percent ?? (state.settings || {}).gpu_percent ?? 0;

function credits() {
  const s = state.settings || {};
  // Mid-drag the split under the thumb is the one on screen, before the account has saved it.
  const split = Number(state.dragging.has("split") ? s.grant_split
    : ((state.user && state.user.grant_split) ?? s.grant_split)) || 0;
  if (state.credits) {
    const earned = Math.max(0, Number(state.credits.lifetime_earned) || 0);
    const toGrants = earned * split / 100;
    return {
      earned, toGrants, kept: earned - toGrants, split,
      balance: Math.max(0, Number(state.credits.balance) || 0),
    };
  }
  const earned = Math.max(0, Number(me().flops) || 0);
  const toGrants = earned * split / 100;
  return { earned, toGrants, kept: earned - toGrants, split, balance: 0 };
}

function grantBalance() {
  if (state.credits) return Math.max(0, Number(state.credits.balance) || 0);
  if (state.grants && state.grants.available != null) return Math.max(0, Number(state.grants.available) || 0);
  return 0;
}

function agentActivity(st) {
  if (st.agent_fetch_done_bytes == null) return st.agent_status || "running";
  const total = st.agent_fetch_total_bytes ? ` of ${fmtBytes(st.agent_fetch_total_bytes)}` : "";
  return `downloading the model (${fmtBytes(st.agent_fetch_done_bytes)}${total})`;
}

function fmtWait(s) {
  if (s == null || !Number.isFinite(Number(s))) return "unknown wait";
  const n = Number(s);
  if (n < 60) return `${Math.max(1, Math.round(n))}s`;
  if (n < 3600) return `${Math.max(1, Math.round(n / 60))} min`;
  return `${(n / 3600).toFixed(1)} h`;
}

// ------------------------------------------------------------ settings

function pickSettings(src) {
  return Object.fromEntries(SETTING_KEYS.map((k) => [k, src[k]]));
}

// Saves run one after another, each posting only what it changes (the server merges that over what
// it stores), so the value set last is the one stored last, and a save queued behind Connect or
// Start never carries the mode and address they have just replaced. Only the final answer in the
// chain is applied: an earlier one would briefly undo a change still waiting its turn.
const saveQueue = { tail: Promise.resolve(), seq: 0 };

function saveSettings(changes) {
  if (!state.settings) return Promise.resolve();   // nothing to merge into before the first overview
  const delta = { ...changes };
  state.settings = { ...state.settings, ...delta };
  state.settingsVersion += 1;
  state.saving += 1;
  const seq = ++saveQueue.seq;
  saveQueue.tail = saveQueue.tail.then(async () => {
    try {
      const saved = pickSettings(await post("/api/settings", delta));
      if (seq === saveQueue.seq) state.settings = saved;
    } catch (e) {
      toast(e.message, "bad");
    } finally {
      state.saving -= 1;
    }
  });
  return saveQueue.tail;
}

// Settings from a poll replace ours only when no save began after the poll did and none is in
// flight: the save's own answer is the newer truth. A split slider mid-drag keeps its value.
function applySettings(st, version) {
  if (!st) return;
  if (state.settings && (version !== state.settingsVersion || state.saving > 0)) return;
  const next = pickSettings(st);
  if (state.settings && state.dragging.has("split")) next.grant_split = state.settings.grant_split;
  state.settings = next;
}

// ------------------------------------------------------------ polling

// One poll in flight at a time, the next scheduled once it has finished, so a hung request never
// piles up behind itself. poll() while one is running waits for it and then runs a fresh one: an
// action that awaits poll() sees the state after it, not an answer fetched before it.
const pollRun = { inflight: null, next: null, timer: null };

function poll() {
  if (pollRun.inflight) {
    if (!pollRun.next) {
      pollRun.next = pollRun.inflight.catch(() => {}).then(() => {
        pollRun.next = null;
        return poll();
      });
    }
    return pollRun.next;
  }
  window.clearTimeout(pollRun.timer);
  state.polling = true;
  pollRun.inflight = pollOnce().finally(() => {
    pollRun.inflight = null;
    state.polling = false;
    pollRun.timer = window.setTimeout(poll, POLL_MS);
  });
  return pollRun.inflight;
}

async function pollOnce() {
  const seq = ++state.pollSeq;
  const version = state.settingsVersion;
  try {
    const ov = await api("/api/overview");
    if (seq < state.pollApplied) return;   // a later poll has already answered
    state.pollApplied = seq;
    trackRate(ov);
    // Another pool now: whatever account the old one knew is not this one's; loadAuth asks afresh.
    const before = status().coordinator_url;
    if (before && ov.status && ov.status.coordinator_url !== before) {
      state.user = null;
      state.credits = null;
    }
    state.ov = ov;
    applySettings(ov.status, version);
    if (ov.status && ov.status.coordinator_up) {
      await loadAuth();
      if (state.tab === "llm") await loadLlm();
    } else {
      dropPool();
    }
  } catch (e) {
    if (seq < state.pollApplied) return;
    state.pollApplied = seq;
    state.ov = null;
    dropPool();
  }
  // The board comes from the coordinator: reload it when the pool comes or goes,
  // and now and then while the tab is open, so it never reads Live while offline.
  const up = !!status().coordinator_up;
  const stale = state.tab === "grants" && Date.now() - state.grantsAt > GRANTS_REFRESH_MS;
  // It is also per account (pledges, the admin's review queue): a session that ended or began
  // elsewhere redraws it at once and reloads it.
  const who = `${(state.user || {}).id || ""}:${!!(state.user && state.user.admin)}`;
  const whoChanged = who !== state.grantsWho;
  state.grantsWho = who;
  if (whoChanged && state.grants) renderGrants();
  if (up !== state.grantsUp || stale || (whoChanged && up)) loadGrants(true);
  render();
}

// The coordinator is gone: nothing read from it is current any more.
function dropPool() {
  state.user = null;
  state.credits = null;
  state.llm.net = null;
  state.llm.models = [];
  state.llm.error = "";
}

async function loadAuth() {
  try {
    const me = await api("/api/coord/auth/me");
    state.user = (me && me.user) || null;
    state.credits = (me && me.credits) || null;
    if (state.user && !state.user.accepted_terms && !state.terms) {
      const t = await api("/api/coord/auth/terms");
      state.terms = (t && t.text) || "";
    }
  } catch (e) {
    // Only a refused session signs us out; a timeout or a proxy hiccup keeps what we know.
    if (e.status === 401) {
      state.user = null;
      state.credits = null;
    }
  }
}

function trackRate(ov) {
  if (!ov.pool.online) return;
  // The counter belongs to one pool and one node id: another pool's ledger would read as a burst.
  const key = `${(ov.status || {}).coordinator_url || ""}|${(ov.me || {}).node_id || ""}`;
  if (key !== state.rateKey) {
    state.rateKey = key;
    state.last = null;
    state.rates = [];
  }
  const now = Date.now() / 1000;
  const flops = Number(ov.me.flops) || 0;
  if (state.last && now > state.last[0]) {
    state.rates.push(Math.max(0, (flops - state.last[1]) / (now - state.last[0])));
    if (state.rates.length > 48) state.rates.shift();
  }
  state.last = [now, flops];
}

// The rate shown as "now": the mean of the last few samples, since one poll's step is noisy.
function currentRate() {
  const recent = state.rates.slice(-RATE_WINDOW);
  return recent.length ? recent.reduce((a, b) => a + b, 0) / recent.length : 0;
}

function render() {
  renderGate();
  renderSidebar();
  renderAuth();
  renderContributions();
  renderUsage();
  renderGrantsLive();
  renderPool();
  renderLlm();
}

// Nothing that saves a setting is usable before the first overview: there is nothing to merge into.
function renderGate() {
  const locked = !state.settings;
  $$("#gpu, #mem, #split, #l-mem, #l-dir, #p-mode button, #l-role button, #l-transport-seg button")
    .forEach((el) => { el.disabled = locked; });
}

// ------------------------------------------------------------ sidebar + tabs

function showTab(name) {
  state.tab = name;
  $$(".nav-item").forEach((b) => {
    const on = b.dataset.tab === name;
    b.classList.toggle("is-on", on);
    b.setAttribute("aria-selected", on ? "true" : "false");
  });
  $$(".view").forEach((v) => v.classList.toggle("is-on", v.id === `view-${name}`));
  $("#main").scrollTop = 0;
  if (document.scrollingElement) document.scrollingElement.scrollTop = 0;   // narrow windows scroll the page itself
  if (name === "grants") loadGrants();
  if (name === "llm" && status().coordinator_up) loadLlm().then(renderLlm);
}

// The pool on screen is ours only in host mode, not one left running from before joining another.
const hostingHere = (st) => (state.settings || {}).mode === "host" && st.coordinator_pid != null;

function renderSidebar() {
  const st = status();
  if (!state.ov) {
    setDot("#side-dot", "hot");
    setText("#side-title", "No connection");
    setText("#side-sub", "The local /compute service is not answering.");
  } else if (st.coordinator_up) {
    setDot("#side-dot", "ok");
    setText("#side-title", hostingHere(st) ? "Hosting" : "Joined");
    setText("#side-sub", `${plural(st.nodes || 0, "Mac")} · ${plural(st.jobs || 0, "job")}`);
  } else {
    setDot("#side-dot", "");
    setText("#side-title", "Offline");
    setText("#side-sub", "No pool running");
  }
  const agent = $("#side-agent");
  agent.hidden = !st.agent_running;
  // Set whether shown or hidden: it must never lag the Contributions pill.
  agent.textContent = !st.agent_running ? "Not contributing"
    : agentRestarting(st) ? "Restarting…" : st.agent_draining ? "Stopping…" : `Contributing · ${liveGpuPercent()}%`;
}

function renderAuth() {
  if (state.busy.has("auth")) return;   // a sign-in or sign-out owns the form meanwhile
  const gate = $("#auth-gate");
  const form = $("#auth-form");
  const terms = $("#auth-terms");
  const userBox = $("#auth-user");
  const online = !!(status().coordinator_up);
  $("#auth-submit").disabled = !online;
  if (signedIn() && !state.user.accepted_terms) {
    gate.hidden = true;
    form.hidden = true;
    terms.hidden = false;
    userBox.hidden = true;
    if (state.terms) $("#terms-text").textContent = state.terms;
    return;
  }
  if (signedIn()) {
    gate.hidden = true;
    form.hidden = true;
    terms.hidden = true;
    userBox.hidden = false;
    setText("#auth-who", state.user.name || state.user.email);
    setText("#auth-sub", state.user.admin ? "Admin on this pool" : state.user.email);
    return;
  }
  userBox.hidden = true;
  terms.hidden = true;
  if (!form.hidden) {
    gate.hidden = true;
    const register = state.authMode === "register";
    $("#auth-submit").textContent = register ? "Create account" : "Sign in";
    $("#auth-name").hidden = !register;
    $("#auth-password").setAttribute("autocomplete", register ? "new-password" : "current-password");
  } else {
    gate.hidden = false;
  }
}

// ------------------------------------------------------------ contributions

function sparkline(svg, values) {
  const w = 200;
  const h = 40;
  const vals = values.slice(-48);
  const top = Math.max(0, ...vals);
  if (vals.length < 2 || top <= 0) {
    svg.innerHTML = `<line x1="0" y1="${h - 1}" x2="${w}" y2="${h - 1}" stroke="#3a3a3a" stroke-width="2" stroke-dasharray="4 5"/>`;
    return;
  }
  const step = w / (vals.length - 1);
  const pts = vals.map((v, i) => `${(i * step).toFixed(1)},${(h - 2 - (v / top) * (h - 6)).toFixed(1)}`);
  svg.innerHTML = `<polygon points="0,${h} ${pts.join(" ")} ${w},${h}" fill="rgba(214,255,0,0.12)"/>`
    + `<polyline points="${pts.join(" ")}" fill="none" stroke="#d6ff00" stroke-width="2" vector-effect="non-scaling-stroke"/>`;
}

function renderContributions() {
  const st = status();
  const m = me();
  const c = credits();
  const rate = currentRate();

  const given = Math.max(0, Number(m.flops) || 0);   // this Mac's work; the account's credits are the next tile
  setText("#c-flops", fmtFlops(given));
  setText("#c-flops-sub", given
    ? `FLOPs total · ${rate > 0 ? withUnit(rate, "FLOP/s") + " now" : "idle right now"}`
    : "Nothing yet. Start contributing to earn.");
  sparkline($("#c-spark"), state.rates);
  setText("#c-earned", fmtFlops(c.earned));
  setText("#c-earned-sub", signedIn()
    ? `On this account · you keep ${withUnit(c.kept)}`
    : `Estimated · you keep ${withUnit(c.kept)}`);
  setText("#c-grants", fmtFlops(c.toGrants));
  setText("#c-grants-sub", signedIn()
    ? `${c.split}% of lifetime earnings`
    : `Estimated · ${c.split}% of what you earn`);
  setText("#c-rank", m.rank ? `#${m.rank}` : "—");
  setText("#c-rank-sub", m.rank ? `of ${plural(m.of, "Mac")} in the pool` : "Not ranked yet");

  const running = !!st.agent_running;
  const draining = running && !!st.agent_draining;   // the agent exits after its step: Stop was pressed...
  const restarting = draining && agentRestarting(st); // ...or new settings make it start again
  const s = state.settings || {};
  setTag("#c-pill", restarting ? "Restarting…" : draining ? "Stopping" : running ? `Contributing · ${liveGpuPercent()}%` : "Not contributing",
    running && !draining ? "ok" : restarting ? "line" : "");
  setDot("#c-dot", running ? "ok" : "", running && !draining);
  if (restarting) {
    setText("#c-state", "Restarting…");
    setText("#c-detail", "The current step finishes, then the agent starts again with the new settings.");
  } else if (draining) {
    setText("#c-state", "Stopping");
    setText("#c-detail", "The current step finishes, then this Mac is released.");
  } else if (running) {
    const job = st.agent_job_id ? ` · job ${String(st.agent_job_id).slice(0, 8)}` : "";
    setText("#c-state", `Contributing${job}`);
    setText("#c-detail", `Agent is ${agentActivity(st)}. Earning whenever the pool has work.`);
  } else {
    const where = s.mode === "host" ? "hosted on this Mac"
      : s.mode === "public" ? (s.url || status().public_url || "public pool")
      : (s.url || "no address set yet");
    setText("#c-state", "Not contributing");
    setText("#c-detail", `Start to lend this Mac to the pool (${where}). It runs in the background.`);
  }
  const toggle = $("#c-toggle");
  if (draining) {
    toggle.textContent = restarting ? "Restarting…" : "Stopping…";
    toggle.className = "btn block ghost";
    toggle.disabled = true;
  } else if (!state.busy.has("contribute")) {
    toggle.textContent = running ? "Stop contributing" : "Start contributing";
    toggle.className = `btn block ${running ? "ghost" : "primary"}`;
    toggle.disabled = !state.ov;
  }

  const gpu = $("#gpu");
  if (!state.dragging.has("gpu") && s.gpu_percent != null) gpu.value = s.gpu_percent;
  rangeFill(gpu);
  setText("#gpu-out", `${gpu.value}%`);
  const live = st.agent_gpu_percent ?? (m.node || {}).gpu_percent;
  setText("#gpu-hint", running && live != null && Number(live) !== Number(gpu.value)
    ? `Running at ${live}%. The new share applies next time you start.`
    : "Higher shares earn credits faster but leave less GPU for you.");

  syncMemSlider($("#mem"), $("#mem-out"), s.memory_gb, trainingAutoGb());
  const lent = m.node && m.node.memory_contrib_bytes;
  setText("#mem-hint", running && lent
    ? `Lending ${fmtBytes(lent)} now. A new amount applies next time you start.`
    : `Auto lends what is free when you start.${upTo()} More than is free makes macOS squeeze other apps.`);

  const split = $("#split");
  if (!state.dragging.has("split")) split.value = c.split;
  rangeFill(split);
  setText("#split-grants", `${split.value}%`);
  setText("#split-keep", `${100 - split.value}%`);
  setText("#split-grants-amt", `≈ ${withUnit(c.toGrants)}`);
  setText("#split-keep-amt", `≈ ${withUnit(c.kept)}`);
  setText("#split-hint", signedIn()
    ? "Credits are 1:1 with FLOPs. This split is saved on your account."
    : "Credits are 1:1 with FLOPs. Sign in so this split is saved on your account.");

  const node = m.node;
  renderOnce("mac", [node, m.node_id], $("#c-mac"), () => node ? `
    <dl class="kv">
      <dt>Name</dt><dd>${esc(node.name)}</dd>
      <dt>Chip</dt><dd>${esc(node.chip || "—")}</dd>
      <dt>Memory lent</dt><dd>${fmtBytes(node.memory_contrib_bytes)}</dd>
      <dt>Matmul speed</dt><dd>${Number(node.matmul_tflops || 0).toFixed(1)} TFLOPS</dd>
      <dt>GPU share in use</dt><dd>${esc(node.gpu_percent)}%</dd>
      <dt>Data address</dt><dd>${esc(node.data_addr || "—")}</dd>
      <dt>Node id</dt><dd class="mono-sm">${esc(String(m.node_id || "").slice(0, 12))}</dd>
    </dl>` : `<p class="empty">Start contributing to register this Mac with the pool.</p>`);
}

// ------------------------------------------------------------ usage

function renderUsage() {
  const p = pool();
  const cap = p.capacity || {};
  setText("#u-macs", String(cap.macs || 0));
  setText("#u-macs-sub", p.online ? "contributing right now" : "pool offline");
  setText("#u-tflops", Number(cap.tflops || 0).toFixed(1));
  setText("#u-mem", fmtBytes(cap.memory_bytes || 0));
  setText("#u-jobs", String(cap.running || 0));
  setText("#u-jobs-sub", `running · ${cap.waiting || 0} waiting`);
  setText("#u-pill", signedIn()
    ? `${withUnit(credits().balance)} available`
    : `≈ ${withUnit(credits().kept)} to spend`);
  $("#flop-budget").hidden = !signedIn();
  const minMacs = $("#min_stages");
  if ((state.settings || {}).mode === "public") {
    minMacs.value = "1";
    minMacs.disabled = true;
  } else {
    minMacs.disabled = false;
  }

  const models = status().models || [];
  if (models.length && state.modelsShown !== models.join()) {
    const sel = $("#model");
    const keep = sel.value;
    sel.innerHTML = models.map((m, i) => `<option value="${esc(m)}">${esc(modelLabel(m))}${i === 0 ? " · quick test" : ""}</option>`).join("");
    if (models.includes(keep)) sel.value = keep;
    state.modelsShown = models.join();
  }

  const submit = $("#job-submit");
  if (!state.busy.has("submit")) submit.disabled = !p.online;
  $("#u-open-pool").hidden = p.online;
  const msg = $("#job-msg");
  if (!p.online && !msg.dataset.sticky) setMsg("#job-msg", "Start or join a pool to submit jobs.", "");
  else if (p.online && msg.dataset.offline) setMsg("#job-msg", "", "");
  msg.dataset.offline = p.online ? "" : "1";

  setText("#u-count", String(p.jobs.length));
  // Rebuilt only when the cards themselves change. A starting job's progress changes every poll,
  // and rebuilding then replaced its Cancel button under the pointer, losing the click.
  renderOnce("jobs", p.jobs.map((j) => [j.id, j.status, j.can_cancel, j.model]), $("#jobs"),
    () => p.jobs.length ? p.jobs.map(jobCard).join("")
      : `<p class="empty">No jobs yet. Submit one and it will appear here.</p>`);
  $$("#jobs .job").forEach((card, i) => patchJob(card, p.jobs[i]));
}

// What a starting job waits for on each Mac: the model download, loading it, or nothing.
function startNote(j) {
  const parts = (j.stages || []).map((st) => {
    const who = st.node_name || String(st.node_id || "").slice(0, 8);
    if (st.phase === "ready") return `${who} ready`;
    if (st.phase !== "fetching") return `Loading the model on ${who}`;
    const done = st.fetch_done_bytes;
    const total = st.fetch_total_bytes;
    if (done == null) return `Downloading the model on ${who}`;
    const pct = total ? ` (${Math.floor((done / total) * 100)}%)` : "";
    return `Downloading the model on ${who}: ${fmtBytes(done)}${total ? ` of ${fmtBytes(total)}` : ""}${pct}`;
  });
  if (!parts.length) parts.push("Starting");
  if (j.starting_s != null) parts.push(`assigned ${fmtWait(j.starting_s)} ago`);
  return parts.join(" · ");
}

function jobNotes(j, status) {
  const waiting = ["queued", "recovering"].includes(status);
  const notes = [];   // [text, class]
  if (status === "starting") notes.push([startNote(j), "info"]);
  // While it waits or starts again, the error is why the last try ended, not the news.
  if (j.error) notes.push([waiting || status === "starting" ? `Last try: ${j.error}` : j.error, ""]);
  if (waiting) {
    const bits = [];
    if (j.queue_position != null) bits.push(`queue #${j.queue_position}`);
    if (j.wait_reason) bits.push(j.wait_reason);
    else if (j.queue_position > 1) bits.push("behind the job ahead of it");
    if (Number(j.wait_s) > 0) bits.push(`~${fmtWait(j.wait_s)}`);
    if (bits.length) notes.push([`Waitlist ${bits.join(" · ")}`, "info"]);
  } else if (j.wait_reason) notes.push([`Waiting: ${j.wait_reason}`, "info"]);
  if (!notes.length && j.adapter_dir) return `<p class="note ok">Adapter ready at ${esc(j.adapter_dir)}</p>`;
  return notes.map(([text, cls]) => `<p class="note ${cls}">${esc(text)}</p>`).join("");
}

function jobCard(j) {
  const status = String(j.status || "");
  const cls = ["running", "starting"].includes(status) ? "is-active"
    : ["queued", "recovering"].includes(status) ? "is-waiting" : "";
  const cancel = j.can_cancel
    ? `<button type="button" class="btn ghost sm" data-cancel="${esc(j.id)}">Cancel</button>` : "";
  return `<div class="job ${cls}">
    <div class="job-top"><b>${esc(modelLabel(j.model))}</b><span class="tag ${STATUS_TONE[status] || ""}">${esc(status)}</span>${cancel}</div>
    <div class="bar ${status === "completed" ? "done" : ""}"><i></i></div>
    <p class="meta"></p><div class="notes"></div>
  </div>`;
}

// The parts of a job card that change between polls, set in place.
function patchJob(card, j) {
  if (!j) return;
  const status = String(j.status || "");
  const meta = [`Step ${j.progress_step ?? 0} of ${j.steps ?? 0}`];
  if (j.last_loss != null) meta.push(`loss ${Number(j.last_loss).toFixed(3)}`);
  if (j.stages && j.stages.length) meta.push(plural(j.stages.length, "Mac"));
  meta.push(String(j.id).slice(0, 8));
  card.querySelector(".bar i").style.width = `${(Number(j.progress) * 100).toFixed(1)}%`;
  card.querySelector(".meta").textContent = meta.join(" · ");
  const notes = card.querySelector(".notes");
  const html = jobNotes(j, status);
  if (notes.dataset.html !== html) {
    notes.innerHTML = html;
    notes.dataset.html = html;
  }
}

function setMsg(sel, text, tone) {
  const el = $(sel);
  el.textContent = text;
  el.className = `msg ${tone || ""}`.trim();
}

// ------------------------------------------------------------ grants

async function loadGrants(quiet = false) {
  const seq = ++state.grantsSeq;
  state.grantsUp = !!status().coordinator_up;
  state.grantsAt = Date.now();
  let board = null;
  try {
    board = await api(`/api/grants?sort=${encodeURIComponent(state.sort)}`);
  } catch (e) {
    if (!quiet) toast(e.message, "bad");
  }
  if (seq !== state.grantsSeq) return;
  state.grants = board;
  renderGrants();
  renderGrantsLive();
}

function renderGrantsLive() {
  const g = state.grants;
  const bal = grantBalance();
  setText("#g-avail", g || signedIn() ? fmtFlops(bal) : "—");
  setText("#g-avail-sub", signedIn()
    ? `${withUnit(credits().balance)} personal balance`
    : "Sign in to fund grants with your credits.");
  setText("#g-pledged", g ? fmtFlops(g.pledged) : "—");
  setText("#g-open", g ? String(g.grants.length) : "—");
  setText("#g-open-sub", g ? (g.sample ? "demo grants" : `${g.pending.length} waiting for review`) : "");
  setTag("#g-pill", g && g.sample ? "Demo" : g && g.online ? "Live" : "Offline",
    g && g.sample ? "line" : g && g.online ? "ok" : "");

  const board = (g && g.leaders && g.leaders.length)
    ? g.leaders
    : ((state.ov && state.ov.leaderboard) || []);
  const online = pool().online;
  renderOnce("leaders", [board, online], $("#leaders"), () => {
    if (!board.length) {
      return `<li class="empty">${online ? "No contributions yet." : "Start or join a pool to see who is giving the most."}</li>`;
    }
    return board.slice(0, 8).map((r) => `<li class="${r.is_me ? "is-me" : ""}">
      <span class="rank">${esc(Number(r.rank))}</span>
      <span class="name">${esc(r.name)}${r.is_me ? `<span class="tag ok">You</span>` : ""}</span>
      <span class="flops">${esc(withUnit(r.flops))}</span>
    </li>`).join("");
  });
}

function renderGrants() {
  const g = state.grants;
  $$("#g-sort button").forEach((b) => b.classList.toggle("is-on", b.dataset.sort === state.sort));
  const admin = !!(state.user && state.user.admin);
  const online = !!(g && g.online);
  const grants = g ? g.grants : [];
  const pending = g ? g.pending : [];

  // Rebuilding the cards would reset the pledge being typed: carry its value (and the focus) over.
  const fund = state.fundOpen ? document.getElementById(`fund-${state.fundOpen}`) : null;
  const typed = fund ? { id: fund.id, value: fund.value, focused: document.activeElement === fund } : null;
  const rebuilt = renderOnce("grants", [grants, online, !!(g && g.sample), state.fundOpen, state.fundMsg], $("#g-list"), () => grants.length
    ? grants.map(grantCard).join("")
    : `<article class="card"><p class="empty">${online ? "No public grants yet." : "Start or join a pool to see live grants."}</p></article>`);
  const again = rebuilt && typed ? document.getElementById(typed.id) : null;
  if (again) {
    again.value = typed.value;
    if (typed.focused) again.focus();
  }

  if (!signedIn()) $("#g-request").hidden = true;   // the request form belongs to a session
  $("#g-review").hidden = !admin || !!(g && g.sample);
  setTag("#g-review-count", String(pending.length), pending.length ? "hot" : "");
  renderOnce("pending", pending, $("#g-pending"), () => pending.length ? pending.map((p) => `<div class="pending">
      <div class="top"><b>${esc(p.title)}</b><span class="muted">${esc(withUnit(p.goal))} goal</span></div>
      <p class="hint">by ${esc(p.author)}</p>
      <p class="summary">${esc(p.summary)}</p>
      <div class="btn-row">
        <button type="button" class="btn primary sm" data-review="${esc(p.id)}" data-approve="1">Approve</button>
        <button type="button" class="btn danger sm" data-review="${esc(p.id)}" data-approve="">Decline</button>
      </div>
    </div>`).join("") : `<p class="empty">Nothing waiting for review.</p>`);
}

function grantCard(g) {
  const funded = g.remaining <= 0;
  const open = state.fundOpen === g.id;
  const demo = !!(state.grants && state.grants.sample);
  const actions = demo ? `<p class="hint">Demo grant — not spendable.</p>`
    : open ? `
      <input type="number" min="1" step="1" id="fund-${esc(g.id)}" value="${Math.max(1, Math.min(25, Math.floor(g.remaining / T)))}" aria-label="TFLOPs to pledge">
      <span class="unit">TFLOPs</span>
      <button type="button" class="btn primary sm" data-fund-go="${esc(g.id)}">Pledge</button>
      <button type="button" class="btn ghost sm" data-fund-close>Cancel</button>`
    : `<button type="button" class="btn ${funded ? "ghost" : "primary"} sm" data-fund-open="${esc(g.id)}" ${funded ? "disabled" : ""}>${funded ? "Funded" : "Fund this grant"}</button>`;
  return `<article class="card grant">
    <div class="top"><span class="tag line">${esc(g.tag)}</span><span class="hint">${plural(g.backers, "backer")}</span></div>
    <h3>${esc(g.title)}</h3>
    <p class="by">by ${esc(g.author)}</p>
    <p class="summary">${esc(g.summary)}</p>
    <div class="bar ${funded ? "done" : ""}"><i style="width:${(g.progress * 100).toFixed(1)}%"></i></div>
    <div class="nums"><span>${esc(fmtFlops(g.raised))} of ${esc(withUnit(g.goal))}</span><span>${Math.round(g.progress * 100)}%</span></div>
    <div class="actions">${actions}</div>
    ${open && state.fundMsg ? `<p class="msg bad">${esc(state.fundMsg)}</p>` : ""}
  </article>`;
}

async function grantAction(path, body, okText) {
  state.grantsSeq += 1;   // a board load already in flight is older than this answer
  try {
    state.grants = await post(path, { ...body, sort: state.sort });
    if (okText) toast(okText);
    return true;
  } catch (e) {
    return e.message;
  } finally {
    renderGrants();
    renderGrantsLive();
  }
}

// ------------------------------------------------------------ pool

function renderPool() {
  const st = status();
  const s = state.settings || {};
  const mode = s.mode === "join" || s.mode === "public" ? s.mode : "host";
  $$("#p-mode button").forEach((b) => b.classList.toggle("is-on", b.dataset.mode === mode));
  $("#p-host").hidden = mode !== "host";
  $("#p-join").hidden = mode !== "join";
  $("#p-public").hidden = mode !== "public";
  setText("#p-ip", st.lan_ip || "—");
  // An address being typed is left alone until Connect, or until what is stored matches it.
  const url = $("#url");
  const urlVal = s.url || "";
  if (state.dirty.url && url.value.trim() === urlVal) state.dirty.url = false;
  if (!state.dirty.url && url.value !== urlVal) url.value = urlVal;
  const pub = $("#public-url");
  // One stored address serves Join and Public: the LAN one must not turn up in the Public field.
  const pubVal = (s.mode === "public" && s.url) || st.public_url || "";
  if (state.dirty.publicUrl && pub.value.trim() === pubVal) state.dirty.publicUrl = false;
  if (!state.dirty.publicUrl && pub.value !== pubVal) pub.value = pubVal;

  const banner = $("#p-banner");
  banner.hidden = !st.last_error;
  banner.textContent = st.last_error || "";
  banner.className = `banner ${isRestartNote(st.last_error) ? "info" : ""}`.trim();

  if (!state.busy.has("pool")) {
    // A coordinator that failed (port taken, crashed) must leave Start hosting free for a retry.
    const hosting = st.coordinator_pid != null && !(st.last_error && !st.coordinator_up);
    $("#p-start").textContent = hosting ? "Hosting" : "Start hosting";
    $("#p-start").disabled = hosting || !state.ov;
    // Anything of ours still running (a coordinator process, the agent, the LLM node) can be stopped.
    $("#p-stop").disabled = !(st.coordinator_up || st.agent_running || st.coordinator_pid != null || st.inference_running);
  }

  const job = st.agent_job_id ? ` · job ${String(st.agent_job_id).slice(0, 8)}` : "";
  const rows = [
    ["Coordinator", st.coordinator_up, st.coordinator_up ? (hostingHere(st) ? "hosting here" : "reachable") : "offline"],
    ["This Mac's agent", st.agent_running,
      !st.agent_running ? "not contributing" : agentRestarting(st) ? "restarting" : st.agent_draining ? "stopping"
        : `${agentActivity(st)}${job}`],
    ["Macs in pool", (st.nodes || 0) > 0, st.coordinator_up ? String(st.nodes || 0) : "—"],
    ["Jobs", (st.jobs || 0) > 0, st.coordinator_up ? String(st.jobs || 0) : "—"],
    ["LLM node", !!st.inference_running, st.inference_running ? llmNodeState(st) : "not serving"],
    ["Address", st.coordinator_up, st.coordinator_url || "—"],
  ];
  // Left the Host tab with our pool still up: it keeps serving whoever joined it until Connect stops it.
  if (mode !== "host" && st.coordinator_pid != null) rows.push(["Pool hosted here", true, "still running · Connect stops it"]);
  renderOnce("status", rows, $("#p-status"), () => rows.map(([name, ok, value]) =>
    `<dt><i class="sq ${ok ? "ok" : ""}"></i>${esc(name)}</dt><dd>${esc(value)}</dd>`).join(""));

  const nodes = pool().nodes;
  setText("#p-count", String(nodes.length));
  renderOnce("macs", nodes, $("#p-macs"), () => nodes.length ? `<table class="macs">
      <thead><tr><th>Mac</th><th>Chip</th><th>Memory lent</th><th>GPU</th><th>FLOPs given</th><th>Status</th></tr></thead>
      <tbody>${nodes.map((n) => {
        const ok = n.canary_passed !== false && !n.draining;
        return `<tr>
          <td><span class="who"><i class="sq ${ok ? "ok" : "hot"}"></i>${esc(n.name || String(n.node_id).slice(0, 8))}${n.is_me ? ` <span class="tag ok">You</span>` : ""}</span></td>
          <td>${esc(n.chip || "—")}</td>
          <td class="num">${fmtBytes(n.memory_contrib_bytes)}</td>
          <td class="num">${esc(n.gpu_percent)}%</td>
          <td class="num">${esc(withUnit(n.flops))}</td>
          <td>${esc(n.draining ? "draining" : (n.status || "—"))}</td>
        </tr>`;
      }).join("")}</tbody></table>`
    : `<p class="empty">No Macs connected yet. Start contributing here, or have others join.</p>`);

  setText("#p-hint", mode === "host"
    ? "Firewall: allow incoming TCP 8765 on this Mac and 9700 on each contributing Mac."
    : mode === "public"
      ? "Each Take runs on one signed-in Mac. Nobody opens ports at home."
      : "Firewall: allow incoming TCP 9700 on this Mac so pipeline peers can reach it.");
}

// ------------------------------------------------------------ LLMs

// One busy key per model: its buttons in the Models list, on the Pipeline card and under Send share it.
const llmKey = (id) => `llm-model:${id}`;
// Who may unload, stop or remove a model: anyone on a LAN pool, only an admin on a public one
// (the coordinator's require_admin, said before a click is refused).
const canManageModels = () => (state.settings || {}).mode !== "public" || !!(state.user && state.user.admin);
// The pool's coordinator has Unload, Serve and Remove per model; an older one only stops a pipeline.
const controls = () => !!(state.llm.net && state.llm.net.model_controls);
const llmModel = (id) => ((state.llm.net && state.llm.net.models) || []).find((m) => m.id === id) || null;
const LLM_TAG_TONE = {
  loaded: "ok", loading: "line", unloading: "line", "not served": "warn", removing: "line", removed: "", rejected: "hot",
};

function llmNodeState(st) {
  const n = st.inference_status || {};
  if (n.available === false) return n.reason || "paused";
  const busy = (n.pipelines || []).length ? "serving" : "ready";
  return `${busy} · ${String(n.transport || "").toLowerCase() || "—"}`;
}

async function loadLlm() {
  const l = state.llm;
  try {
    const net = await api("/api/coord/inference/status");
    const models = ((await api("/api/coord/v1/models")) || {}).data || [];
    l.net = net;
    l.models = models;
    l.error = "";
  } catch (e) {
    l.net = null;
    l.models = [];
    l.error = e.message || String(e);   // said under Send, not hidden behind "No models yet"
    return;   // the pick stands: a blip must not switch models under the user
  }
  const choices = llmChoices(l);
  if (!choices.some((m) => m.id === l.model)) {
    const served = choices.find((m) => llmServable(l, m.id));
    l.model = (served || choices[0] || {}).id || "";
  }
}

// Every known model can be picked, served or not: one nobody serves yet says why under Send.
// A removed one is listed under Models only until its last copy is gone; it can't be chatted with.
function llmChoices(l) {
  const known = l.net ? l.net.models.filter((m) => !["rejected", "removed"].includes(m.status)) : [];
  return known.length ? known : l.models;
}

const llmServable = (l, id) => !!id && l.models.some((m) => m.id === id);

// Why Send can't be used now (or why a reply would fail), and the one click that fixes it.
function llmWhy(st, s, l, unsupported) {
  if (!st.coordinator_up) return { text: "Start or join a pool first (Pool tab)." };
  if (unsupported) return { text: OUTDATED_COORDINATOR, tone: "bad" };
  if (l.error) return { text: `Can't reach the pool's LLM service: ${l.error}`, tone: "bad" };
  const choices = llmChoices(l);
  if (!choices.length) return { text: "No models yet. Upload a GGUF below, or put one in a head's models folder." };
  const m = choices.find((c) => c.id === l.model) || choices[0];
  const n = st.inference_status || {};
  const running = !!st.inference_running;
  if (state.busy.has("llm")) return { text: l.starting ? "Starting llama.cpp on this Mac…" : "Stopping serving on this Mac…" };
  if (m.status === "disabled") {
    const text = `${m.id} is not being served: it was stopped for the whole pool.`;
    return canManageModels()
      ? { text: `${text} Serve lets the next chat load it again.`, action: { label: "Serve", act: "llm-serving", model: m.id, on: "1" } }
      : { text: `${text} Ask the pool's admin to serve it again.` };
  }
  const me = l.net && n.node_id ? l.net.nodes.find((x) => x.id === n.node_id) : null;
  const lent = me ? Number(me.committed_gb) || 0 : 0;
  const need = Number(m.min_memory_gb) || 0;
  const short = running && need > 0 && lent > 0 && lent < need;
  const lend = memoryLimits().known && need > memoryLimits().cap
    ? { text: `${m.id} needs ${need} GB lent to run on one Mac, more than this Mac can lend (${memoryLimits().cap} GB). Add a Mac to the pool to split it.` }
    : { text: `${m.id} needs ${need} GB lent to run on one Mac; this Mac lends ${lent} GB.`,
        action: { label: `Lend ${need} GB and restart serving`, act: "llm-lend", gb: need } };
  if (llmServable(l, m.id)) {
    const serving = l.net ? l.net.nodes.filter((x) => x.online && x.available).length : 0;
    return short && serving <= 1 ? { ...lend, text: `${lend.text} A reply would fail.` } : null;
  }
  if (!running) {
    return { text: `No Mac is serving ${m.id} yet.`, action: { label: "Start serving on this Mac", act: "llm-serve" } };
  }
  if (n.reason === "unsupported") return { text: n.last_error || OUTDATED_COORDINATOR, tone: "bad" };
  if (!n.node_id || ["connecting", "joining"].includes(n.reason)) {
    return { text: n.last_error ? `Starting llama.cpp on this Mac: ${n.last_error}` : "Starting llama.cpp on this Mac…" };
  }
  if (n.available === false) return { text: `This Mac serves LLMs but is not taking work right now: ${n.reason || "paused"}.` };
  if (s.inference_head === false) {
    return { text: `This Mac only lends layers; a head has to hold ${m.id}.`,
             action: { label: "Make this Mac a head", act: "llm-head" } };
  }
  const copying = me && m.downloading ? m.downloading[me.name] : null;
  if (copying != null) return { text: `Copying ${m.id} to this Mac: ${Math.round(copying * 100)}%.` };
  if (!(n.models || []).includes(m.id)) return { text: `Waiting for ${m.id} to reach this Mac.` };
  if (m.status && m.status !== "ready") return { text: `${m.id} is ${m.status}${m.status_reason ? `: ${m.status_reason}` : ""}.` };
  if (short) return lend;
  return { text: `This Mac has ${m.id}; it is listed once the pool hears from it (a few seconds).` };
}

function renderWhy(why) {
  const box = $("#l-why");
  box.hidden = !why;
  if (!why) return;
  box.className = `why ${why.tone || ""}`.trim();
  setText("#l-why-text", why.text);
  const btn = $("#l-why-act");
  const a = why.action;
  // withBusy owns the button's label meanwhile: this Mac's Start serving, or a model's Serve.
  if (state.busy.has("llm") || (btn.dataset.model && state.busy.has(llmKey(btn.dataset.model)))) return;
  btn.hidden = !a;
  if (!a) return;
  btn.textContent = a.label;
  btn.dataset.act = a.act;
  btn.dataset.gb = a.gb != null ? String(a.gb) : "";
  btn.dataset.model = a.model || "";
  btn.dataset.on = a.on || "";
}

function renderLlm() {
  const st = status();
  const s = state.settings || {};
  const l = state.llm;
  const net = l.net;
  const up = !!st.coordinator_up;
  const serving = net ? net.nodes.filter((n) => n.online && n.available) : [];
  const pipes = net ? net.pipelines.filter((p) => p.model === l.model && !["stopped", "broken"].includes(p.state)) : [];
  // The pipeline serving now, before one that is only finishing its last reply.
  const pipe = pipes.find((p) => p.state !== "draining") || pipes[0];
  const live = st.inference_transport || (net && net.transport) || "";

  const unsupported = up && st.inference_supported === false;
  const choices = llmChoices(l);
  setTag("#l-pill", !up ? "Offline" : unsupported ? "Pool has no LLMs"
    : l.models.length ? `${plural(l.models.length, "model")} ready`
    : choices.length ? `${plural(choices.length, "model")} · none served` : "No models yet",
  !up || unsupported ? "hot" : l.models.length ? "ok" : "warn");
  setText("#l-models", String(l.models.length));
  const known = net ? net.models.filter((m) => m.status !== "removed").length : 0;
  setText("#l-models-sub", net ? `${plural(known, "known model")}, ${l.models.length} ready` : "ready to chat");
  setText("#l-nodes", String(serving.length));
  setText("#l-nodes-sub", net ? `${plural(net.nodes.length, "Mac")} registered` : "running llama.cpp");
  setText("#l-speed", pipe && (pipe.live_tok_s || pipe.est_tok_s) ? (pipe.live_tok_s || pipe.est_tok_s).toFixed(1) : "—");
  setText("#l-speed-sub", pipe ? (pipe.live_tok_s ? "tokens/s, last reply" : "tokens/s, estimated") : "tokens per second");
  setText("#l-transport", live ? live.toUpperCase() : "—");
  setText("#l-transport-sub", live === "relay" ? "RPC through the coordinator" : live ? "Macs talk directly on the LAN" : "");
  setTag("#l-chat-tag", l.streaming ? "Replying" : pipe ? pipe.state : "Idle", l.streaming ? "ok" : "");

  const sel = $("#l-model");
  renderOnce("llm-models", [choices.map((m) => [m.id, m.size_gb, m.status, llmServable(l, m.id)]), l.model], sel, () => choices.length
    ? choices.map((m) => `<option value="${esc(m.id)}"${m.id === l.model ? " selected" : ""}>${esc(m.id)} · ${esc(m.size_gb)} GB${
      m.status === "disabled" ? " · stopped" : llmServable(l, m.id) ? "" : " · not served yet"}</option>`).join("")
    : `<option value="">No models yet</option>`);
  $("#l-send").disabled = l.streaming || !llmServable(l, l.model) || !up;
  l.why = llmWhy(st, s, l, unsupported);
  renderWhy(l.why);
  $("#l-stop").disabled = !l.streaming;

  renderModels();

  const up_ = l.upload;
  $("#l-upbar").hidden = !up_;
  if (up_) {
    $("#l-upbar i").style.width = `${Math.round(up_.pct * 100)}%`;
    setText("#l-file-name", up_.processing ? `${up_.name} · processing on the pool…` : up_.name);
  }
  $("#l-pick").classList.toggle("is-disabled", !!up_ || !up || unsupported);

  setTag("#l-pipe-tag", pipe ? pipe.state : "None", pipe && pipe.state === "active" ? "ok" : "");
  const stopped = !pipe && (llmModel(l.model) || {}).status === "disabled";
  // Keyed on what the card shows, not the live speed: a rebuild every reply replaced Unload under the pointer.
  renderOnce("llm-pipe", [pipe ? [pipe.id, pipe.state, pipe.members, pipe.explanation] : null, !!l.model, stopped,
    controls(), canManageModels()], $("#l-pipe"), () => pipe ? `
    <table class="macs">
      <thead><tr><th>Mac</th><th>Role</th><th>Layers</th><th>Share</th><th>Memory</th></tr></thead>
      <tbody>${pipe.members.map((m) => `<tr>
        <td><span class="who"><i class="sq ok"></i>${esc(m.node)}</span></td><td>${esc(m.role)}</td>
        <td class="num">${esc(Number(m.layer_start))}–${esc(Number(m.layer_end) - 1)}</td><td class="num">${Math.round(m.share * 100)}%</td>
        <td class="num">${esc(m.memory_gb)} GB</td></tr>`).join("")}</tbody></table>
    <p class="hint">${esc(pipe.explanation || "")}</p>
    ${pipeUnload(pipe)}`
    : `<p class="empty">${!l.model ? "Pick a model to see how it's split."
      : stopped ? `Not served: it was stopped for the whole pool. ${canManageModels() ? "Press Serve to use it again." : "Only the pool's admin can serve it again."}`
      : "Not loaded. Your first message plans a split across the pool and loads it."}</p>`);
  $$("#l-pipe [data-act]").forEach((b) => { b.disabled = state.busy.has(llmKey(b.dataset.model)); });

  const running = !!st.inference_running;
  const n = st.inference_status || {};
  setDot("#l-dot", running ? (n.available === false ? "hot" : "ok") : "", running && (n.pipelines || []).length > 0);
  setText("#l-state", running ? `Serving · ${llmNodeState(st)}` : "Not serving");
  setText("#l-detail", running
    ? `${plural((n.models || []).length, "model")} on disk${Object.keys(n.downloads || {}).length ? ", downloading" : ""}. ${n.last_error || ""}`
    : "Lend memory to run LLM layers. Uploaded models are pushed here if this Mac may head.");
  syncMemSlider($("#l-mem"), $("#l-mem-out"), s.inference_memory_gb, inferenceAutoGb());
  setText("#l-mem-hint", running
    ? "A new amount applies next time you start serving."
    : `Auto lends 75% of RAM minus 4 GB.${upTo()}`);
  const dir = $("#l-dir");
  if (document.activeElement !== dir && s.models_dir) dir.value = s.models_dir;
  $$("#l-role button").forEach((b) => b.classList.toggle("is-on", (b.dataset.llmHead === "1") === (s.inference_head !== false)));
  $("#l-transport-field").hidden = s.mode !== "host";   // the transport is the hosted pool's
  $$("#l-transport-seg button").forEach((b) => b.classList.toggle("is-on", b.dataset.transport === (s.transport || "direct")));
  const toggle = $("#l-toggle");
  if (!state.busy.has("llm")) {
    toggle.textContent = running ? "Stop serving" : "Start serving";
    toggle.className = `btn block ${running ? "ghost" : "primary"}`;
    toggle.disabled = !state.ov;
  }
  renderChat();
}

// One row of the Models list: its tag, its buttons (only for someone who may manage models), the line
// under its name, and a note per thing worth saying, [text, class]. Pure: a poll's model in, the row out.
function llmRow(m, manage) {
  const held = m.held_by || [];
  const removed = m.status === "removed";
  const inFlight = held.some((h) => h.removing);
  const tag = m.status === "rejected" ? "rejected"
    : removed ? (inFlight ? "removing" : "removed")
    : m.status === "disabled" ? "not served"
    : m.status === "ready" && m.load ? m.load : "";
  const tone = m.status === "rejected" || (removed && held.some((h) => h.error)) ? "is-waiting" : m.servable ? "is-active" : "";

  const acts = [];
  const act = (a, label, cls, more = {}) => acts.push({ act: a, label, cls, ...more });
  if (manage) {
    if (m.status === "ready" && ["loaded", "loading"].includes(m.load)) {
      act("llm-unload", "Unload", "ghost", { title: "Frees its memory on every Mac. The next chat loads it again." });
    }
    if (m.status === "ready") {
      act("llm-serving", "Stop serving", "ghost", { on: "0",
        title: "Unloads it on every Mac in the pool and keeps it from loading until Serve is pressed. Its files stay on disk." });
    }
    if (m.status === "disabled") act("llm-serving", "Serve", "primary", { on: "1" });
    if (!removed) {
      act("llm-remove", "Remove", "danger", { title: "Unloads it and deletes its copies across the pool. Asks first." });
    } else if (!inFlight) {
      if (m.restorable) act("llm-serving", "Add back", "ghost", { on: "1", title: "Lists it again, for the Macs that still have it." });
      if (held.length) act("llm-remove", "Remove again", "danger");
    }
  }

  const dl = Object.entries(m.downloading || {}).map(([n, f]) => `${n} ${Math.round(f * 100)}%`).join(", ");
  const heads = m.heads || [];
  const where = m.status === "rejected" ? (m.status_reason || "rejected")
    : removed ? "removed from this pool"
    : dl ? `downloading on ${dl}` : heads.length ? `on ${heads.join(", ")}` : "no head has it yet";
  const meta = `${m.size_gb} GB${m.arch ? ` · ${m.arch}` : ""} · ${where}`;

  const notes = [];
  if (m.status === "disabled") notes.push(["Stopped for the whole pool: no Mac loads it until Serve is pressed. Its files stay on disk.", "info"]);
  if (m.load === "unloading") notes.push(["Unloading: the reply in progress finishes first.", "info"]);
  if (removed) {
    const again = manage ? " Press Remove again once it is back." : "";
    held.forEach((h) => {
      if (h.removing) notes.push([`Removing from ${h.name}…`, "info"]);
      else if (h.error) notes.push([`${h.name}: ${h.error}`, ""]);
      else if (!h.online) notes.push([`${h.name} is offline and keeps its copy.${again}`, "info"]);
      else if (h.kept) notes.push([`Kept in ${h.name}'s own models folder: delete it there by hand if you want it gone.`, "info"]);
      else notes.push([`Still on ${h.name}.${manage ? " Press Remove again." : ""}`, "info"]);
    });
  }
  return { tag, tone, acts, meta, notes };
}

function modelCard(m, v) {
  const tag = v.tag ? `<span class="tag ${esc(LLM_TAG_TONE[v.tag] || "")}">${esc(v.tag)}</span>` : "";
  const acts = v.acts.map((a) => `<button type="button" class="btn ${esc(a.cls)} sm" data-act="${esc(a.act)}" data-model="${esc(m.id)}"`
    + `${a.on ? ` data-on="${esc(a.on)}"` : ""}${a.title ? ` title="${esc(a.title)}"` : ""}>${esc(a.label)}</button>`).join("");
  return `<div class="job ${esc(v.tone)}">
    <div class="job-top"><b title="${esc(m.id)}">${esc(m.id)}</b>${tag}</div>
    <p class="meta"></p><div class="notes"></div>${acts ? `<div class="acts">${acts}</div>` : ""}
  </div>`;
}

// The parts of a model row that change between polls (downloads, notes, busy buttons), set in place.
function patchModelCard(card, v, busy) {
  if (!card || !v) return;
  const meta = card.querySelector(".meta");
  if (meta && meta.textContent !== v.meta) meta.textContent = v.meta;
  const notes = card.querySelector(".notes");
  const html = v.notes.map(([text, cls]) => `<p class="note ${esc(cls)}">${esc(text)}</p>`).join("");
  if (notes && notes.dataset.html !== html) {
    notes.innerHTML = html;
    notes.dataset.html = html;
  }
  card.querySelectorAll(".acts .btn").forEach((b) => { b.disabled = busy; });
}

// The Models list. Rebuilt only when a row's tag or buttons change, never for a download's progress,
// so a button is not replaced under the pointer; the rest of each row is patched in place.
function renderModels() {
  const net = state.llm.net;
  const up = !!status().coordinator_up;
  const unsupported = up && status().inference_supported === false;
  const catalog = net ? net.models : [];
  const manage = controls() && canManageModels();
  const rows = catalog.map((m) => llmRow(m, manage));
  setText("#l-count", String(catalog.filter((m) => m.status !== "removed").length));
  const hint = !catalog.length ? ""
    : !controls() ? "This pool's coordinator runs an older /compute: update the hosting Mac to unload, stop or remove models here."
    : !manage ? "Only the pool's admin can unload, stop or remove models." : "";
  renderOnce("llm-catalog", [catalog.map((m, i) => [m.id, rows[i].tag, rows[i].tone, rows[i].acts]), up, unsupported, controls(), hint],
    $("#l-catalog"), () => (catalog.length ? catalog.map((m, i) => modelCard(m, rows[i])).join("")
      : `<p class="empty">${unsupported ? esc(OUTDATED_COORDINATOR)
        : up ? "No models yet. Upload a GGUF, or put one in a head's models folder." : "Start or join a pool first."}</p>`)
      + (hint ? `<p class="hint">${esc(hint)}</p>` : ""));
  $$("#l-catalog .job").forEach((card, i) => patchModelCard(card, rows[i], !!catalog[i] && state.busy.has(llmKey(catalog[i].id))));
}

// Unload on the Pipeline card: per model on a pool with model controls, per pipeline on an older one.
function pipeUnload(pipe) {
  if (pipe.state === "draining") return `<button type="button" class="btn ghost sm" disabled>Unloading…</button>`;
  if (!controls()) {
    return `<button type="button" class="btn ghost sm" data-act="llm-unload" data-model="${esc(pipe.model)}" data-pipeline="${esc(pipe.id)}">Unload</button>`;
  }
  return canManageModels() ? `<button type="button" class="btn ghost sm" data-act="llm-unload" data-model="${esc(pipe.model)}">Unload</button>` : "";
}

// What Remove is about to do, asked before anything happens.
function removeQuestion(m, publicPool = (state.settings || {}).mode === "public") {
  const held = m.held_by || [];
  const online = held.filter((h) => h.online).map((h) => h.name);
  const offline = held.filter((h) => !h.online).map((h) => h.name);
  const lines = [`Remove ${m.id} from this pool?`, "",
    "It is unloaded now on every Mac; a reply in progress finishes first."];
  if (m.uploaded) lines.push("The copy uploaded to the pool is deleted.");
  if (online.length) {
    lines.push(publicPool
      ? `On ${online.join(", ")}: the app's copy is deleted. A copy in a Mac's own models folder stays: a public pool never moves files to the Trash.`
      : `On ${online.join(", ")}: the app's copy is deleted, and a copy in the Mac's own models folder goes to its Trash.`);
  }
  if (offline.length) lines.push(`Macs that are offline keep their copy for now: ${offline.join(", ")}. Press Remove again once they are back.`);
  if (!held.length) lines.push("No Mac has a copy.");
  lines.push("", "It stays out of the pool until it is added back or uploaded again.");
  return lines.join("\n");
}

function removedNote(r) {
  const parts = [`Removed ${r.model}.`];
  if ((r.macs || []).length) parts.push(`Deleting its copies on ${r.macs.join(", ")}.`);
  if ((r.offline || []).length) parts.push(`Offline, still holding it: ${r.offline.join(", ")}.`);
  return parts.join(" ");
}

function chatFooter(net) {
  if (!net) return "";
  const who = (net.members || []).map((m) => `${m.node} ${m.layers.length ? `L${m.layers[0]}–${m.layers[1]}` : ""}`.trim()).join(", ");
  const tok = net.tok_s ? ` · ${net.tok_s.toFixed(1)} tok/s` : "";
  return `≈ ${withUnit(net.flops)} · w=${(net.gen_weight || 1).toFixed(1)}${tok} · ${who}`;
}

// Fold one SSE event from /api/chat into the reply being streamed.
function applyChatEvent(reply, ev) {
  if (ev.error) reply.error = ev.error.message;
  if (ev.network) reply.meta = chatFooter(ev.network);
  const choice = ev.choices && ev.choices[0];
  if (!choice) return;
  const delta = choice.delta || {};
  // thinking models (Qwen3 etc.) stream their reasoning separately, before the answer
  if (delta.reasoning_content) reply.reasoning = (reply.reasoning || "") + delta.reasoning_content;
  if (delta.content) reply.content += delta.content;
  if (choice.finish_reason) reply.finish = choice.finish_reason;
}

// A thinking model's reasoning: open while it streams in, folded away once the answer starts
// (unless the user opened or closed it themselves).
function thinkBlock(m, i) {
  if (!m.reasoning) return "";
  const thinking = m.live && !m.content;
  const open = m.thinkOpen ?? thinking;
  return `<details class="think" data-think="${i}"${open ? ` open=""` : ""}>
      <summary>${thinking ? "Thinking…" : "Thoughts"}</summary><p>${esc(m.reasoning)}</p></details>`;
}

function replyText(m) {
  if (m.role === "user") return `<p>${esc(m.content)}</p>`;
  // A failure mid-stream keeps what did arrive, with the error under it.
  const body = m.content ? `<p>${esc(m.content)}</p>` : "";
  const error = m.error ? `<p><span class="hot">${esc(m.error)}</span></p>` : "";
  if (body || error) return body + error;
  if (m.live) return m.reasoning ? "" : `<p>…</p>`;
  if (m.reasoning && m.finish === "length") return `<p class="empty">Ran out of tokens while thinking, before answering.</p>`;
  return `<p class="empty">(no reply)</p>`;
}

function renderChat() {
  const log = $("#l-log");
  const msgs = state.llm.messages;
  const stick = log.scrollHeight - log.scrollTop - log.clientHeight < 32;
  const html = msgs.length ? msgs.map((m, i) => `<div class="say ${m.role === "user" ? "you" : "bot"}">
      <label>${m.role === "user" ? "You" : "Pool"}</label>
      ${thinkBlock(m, i)}${replyText(m)}
      ${m.meta ? `<small>${esc(m.meta)}</small>` : ""}</div>`).join("")
    : `<p class="empty">Replies stream in here, token by token.</p>`;
  // Compared with what we last set, not innerHTML: the browser re-serialises entities, so the
  // two never matched and every poll rebuilt the log (losing the selection and open Thoughts).
  if (log.dataset.html !== html) {
    log.innerHTML = html;
    log.dataset.html = html;
  }
  if (stick) log.scrollTop = log.scrollHeight;
}

// The turns a request carries. A question whose reply failed or came back empty is asked again by
// its retry, not twice; an empty assistant turn says nothing worth a model's attention.
function chatHistory(msgs, reply) {
  const dead = (m) => m && m.role === "assistant" && (m.error || !m.content);
  return msgs.flatMap((m, i) => {
    if (m === reply) return [];
    if (m.role === "assistant") return dead(m) ? [] : [{ role: m.role, content: m.content }];
    const next = msgs[i + 1];
    if (next && next !== reply && dead(next)) return [];
    return [{ role: m.role, content: m.content }];
  });
}

// One SSE event (the lines between blank lines): each data: line is a JSON chunk, or [DONE].
function foldChatEvent(reply, event) {
  for (const ln of event.split("\n")) {
    const payload = ln.startsWith("data:") ? ln.slice(5).trim() : "";
    if (!payload || payload === "[DONE]") continue;
    try {
      applyChatEvent(reply, JSON.parse(payload));
    } catch { /* a truncated or non-JSON chunk: skip it, keep streaming */ }
  }
}

async function readChatStream(body, reply) {
  const reader = body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let cut;
      while ((cut = buf.indexOf("\n\n")) >= 0) {
        foldChatEvent(reply, buf.slice(0, cut));
        buf = buf.slice(cut + 2);
        renderChat();
      }
    }
    buf += dec.decode();
    if (buf.trim()) foldChatEvent(reply, buf);   // a last event that came without its blank line
  } finally {
    reader.cancel().catch(() => {});   // Stop (or an error) must release the connection
  }
}

async function sendChat() {
  const l = state.llm;
  const draft = $("#l-draft");
  const text = draft.value.trim();
  if (!text || l.streaming) return;
  if (!llmServable(l, l.model)) return setMsg("#l-msg", (l.why && l.why.text) || "No Mac is serving this model yet.", "bad");
  setMsg("#l-msg", "", "");
  l.messages.push({ role: "user", content: text });
  const reply = { role: "assistant", content: "", live: true };
  l.messages.push(reply);
  draft.value = "";
  l.streaming = true;
  l.abort = new AbortController();
  renderLlm();
  const history = chatHistory(l.messages, reply);
  try {
    const r = await fetch("/api/chat", {
      method: "POST", credentials: "include", signal: l.abort.signal,
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: l.model, messages: history, max_tokens: 512 }),
    });
    if (!r.ok) {
      const t = await r.text();
      let data = null;
      try { data = JSON.parse(t); } catch { /* plain text */ }
      throw new Error(detailText(data, t || r.statusText));
    }
    await readChatStream(r.body, reply);
  } catch (e) {
    if (e.name === "AbortError") reply.meta = "Stopped.";
    else reply.error = e.message;
  } finally {
    reply.live = false;
    l.streaming = false;
    l.abort = null;
    await loadLlm();
    renderLlm();
  }
}

function uploadModel(file) {
  const l = state.llm;
  if (l.upload) return setMsg("#l-upmsg", `Still sending ${l.upload.name}; one upload at a time.`, "bad");
  if (!/\.gguf$/i.test(file.name)) return setMsg("#l-upmsg", "Pick a .gguf model file.", "bad");
  // Said before a byte goes: a refusal mid-upload reaches the window as a reset connection, not a reason.
  if (!status().coordinator_up) return setMsg("#l-upmsg", "Start or join a pool first (Pool tab).", "bad");
  if (status().inference_supported === false) return setMsg("#l-upmsg", OUTDATED_COORDINATOR, "bad");
  if (file.size > MAX_UPLOAD_GIB * GIB) {
    return setMsg("#l-upmsg", `${file.name} is ${fmtBytes(file.size)}: no Mac pool loads a GGUF over ${MAX_UPLOAD_GIB} GB.`, "bad");
  }
  const pooled = ((l.net && l.net.nodes) || []).filter((n) => n.online).reduce((sum, n) => sum + (Number(n.committed_gb) || 0), 0);
  const needs = file.size / GIB;
  const tooBig = l.net && needs > pooled
    ? `Heads-up: its weights alone need ≈${needs.toFixed(1)} GB but the pool's Macs lend ${pooled.toFixed(1)} GB in total. `
      + "It will upload, but won't load until more memory joins." : "";
  l.upload = { name: file.name, pct: 0, processing: false };
  setText("#l-file-name", file.name);
  setMsg("#l-upmsg", tooBig, tooBig ? "warn" : "");
  const xhr = new XMLHttpRequest();   // fetch() has no upload progress
  // The name travels as a query parameter (any Unicode); the header stays for older shells, but
  // only when it can carry the name (headers take ASCII).
  xhr.open("POST", `/api/models/upload?name=${encodeURIComponent(file.name)}`);
  xhr.withCredentials = true;
  if (/^[\x20-\x7e]+$/.test(file.name)) xhr.setRequestHeader("x-filename", file.name);
  xhr.upload.onprogress = (e) => { if (e.lengthComputable && l.upload) { l.upload.pct = e.loaded / e.total; renderLlm(); } };
  // Every byte is sent: the coordinator now reads the GGUF and pushes it to the heads.
  xhr.upload.onload = () => { if (l.upload) { l.upload.pct = 1; l.upload.processing = true; renderLlm(); } };
  const fail = (text) => { l.upload = null; setMsg("#l-upmsg", text, "bad"); renderLlm(); };
  xhr.onload = async () => {
    let data = null;
    try { data = JSON.parse(xhr.responseText); } catch { /* plain text */ }
    l.upload = null;
    if (xhr.status >= 200 && xhr.status < 300) {
      if (data && typeof data === "object") {
        const pushed = Number(data.pushed) || 0;
        const none = pushed ? "" : "No Mac is serving yet: start serving on this Mac to host it.";
        setMsg("#l-upmsg", `${data.name || file.name}: ${data.layers ?? "?"} layers, sent to ${plural(pushed, "head")}. ${none} ${tooBig}`.trim(),
          tooBig ? "warn" : "ok");
      } else {
        setMsg("#l-upmsg", `${file.name} uploaded. ${tooBig}`.trim(), tooBig ? "warn" : "ok");
      }
    } else {
      // 409 (older coordinator) and 503 (no pool) come with a reason: say it, not the status line.
      setMsg("#l-upmsg", detailText(data, xhr.responseText || xhr.statusText || "Upload failed."), "bad");
    }
    await loadLlm();
    renderLlm();
  };
  xhr.onerror = () => fail("Upload failed: the pool is not reachable.");
  xhr.onabort = () => fail("Upload cancelled.");
  xhr.ontimeout = () => fail("Upload timed out before the pool answered.");
  xhr.send(file);
  renderLlm();
}

// ------------------------------------------------------------ actions

async function withBusy(key, button, busyText, fn) {
  if (state.busy.has(key)) return;
  state.busy.add(key);
  const idle = button ? button.textContent : "";
  if (button) { button.disabled = true; button.textContent = busyText; }
  try {
    await fn();
  } catch (e) {
    toast(e.message, "bad");
  } finally {
    // Fresh state first, then the button: its label must say what is true now, and a second click
    // must not repeat an action whose effect is still on its way.
    try { await poll(); } catch { /* the next tick polls again */ }
    state.busy.delete(key);
    if (button) { button.disabled = false; button.textContent = idle; }
    render();
  }
}

// /api/start lays the body over the stored settings and starts from the result, so only the keys an
// action means to change travel (mode, address, what this Mac lends): a copy of every setting would
// carry values a save still waiting its turn is about to replace. A poll begun before it may not
// bring the old ones back.
function startWith(body) {
  state.settingsVersion += 1;
  return post("/api/start", body);
}

// "Connected to …", with the controller's note when what this Mac lends restarts on the new pool.
const connectedNote = (ov, snap) => `Connected to ${ov.status.coordinator_url}`
  + (isRestartNote(snap.last_error) ? `. ${snap.last_error}` : "");

// What this Mac lends right now, kept as it is when the pool it lends to changes.
const lending = (st) => ({
  contribute: !!(st.agent_running || st.inference_running),
  training: !!st.agent_running,
  inference: !!st.inference_running,
});

// Start or stop serving LLMs on this Mac. Only the LLM node changes: the training agent is left
// alone (resending every setting through /api/start used to restart it, and it hung mid-download).
function serveLlm(btn, on, changes = {}) {
  if (!state.busy.has("llm")) state.llm.starting = on;
  return withBusy("llm", btn, on ? "Starting…" : "Stopping…", async () => {
    const s = state.settings || {};
    const snap = await post("/api/inference", {
      on, inference_memory_gb: s.inference_memory_gb, inference_head: s.inference_head !== false,
      models_dir: s.models_dir, transport: s.transport, ...changes,
    });
    state.settingsVersion += 1;   // a poll begun before this answer may not undo it
    state.settings = pickSettings(snap);
    if (isRestartNote(snap.last_error)) toast(snap.last_error);
    else if (snap.last_error) toast(snap.last_error, "bad");
    else if (on && !snap.inference_running) toast("The LLM node did not start: see ~/.slashcompute/logs/inference.log.", "bad");
    else toast(on ? "Serving. This Mac hosts LLM layers whenever a chat needs them."
      : "Stopped serving on this Mac.");
  });
}

// Joining another pool stops the coordinator hosted here once that pool answers: ask first.
function leaveHostedPool(what = "Connecting") {
  return status().coordinator_pid == null || window.confirm(
    `${what} stops the pool hosted on this Mac. Macs that joined it lose their coordinator. Continue?`);
}

const actions = {
  "toggle-llm": (btn) => serveLlm(btn, !status().inference_running),
  "llm-serve": (btn) => serveLlm(btn, true),
  "llm-lend": (btn) => serveLlm(btn, true, { inference_memory_gb: Number(btn.dataset.gb) }),
  "llm-head": (btn) => serveLlm(btn, true, { inference_head: true }),
  "llm-stop": () => { if (state.llm.abort) state.llm.abort.abort(); },
  "llm-clear": () => {
    if (state.llm.abort) state.llm.abort.abort();   // the reply under way would land in an empty log
    state.llm.messages = [];
    renderChat();
  },
  // The model travels in the body, never the path: the shell's proxy rebuilds decoded paths.
  "llm-unload": (btn) => {
    const id = btn.dataset.model || "";
    const loading = (llmModel(id) || {}).load === "loading";
    return withBusy(llmKey(id), btn, "Unloading…", async () => {
      if (btn.dataset.pipeline) {
        // An older coordinator: stop the one pipeline on the Pipeline card.
        await post(`/api/coord/inference/pipelines/${encodeURIComponent(btn.dataset.pipeline)}/stop`);
        toast("Unloading after the current reply.");
      } else {
        const r = await post("/api/coord/inference/models/unload", { model: id });
        toast(!(r && r.pipelines) ? `${id} was not loaded.`
          : `Unloading ${id}. The reply in progress finishes first${loading ? "; a chat waiting for it loads it again" : ""}.`);
      }
      await loadLlm();
    });
  },
  "llm-serving": (btn) => {
    const id = btn.dataset.model || "";
    const on = btn.dataset.on === "1";   // a real boolean: the coordinator refuses anything else
    const back = (llmModel(id) || {}).status === "removed";
    return withBusy(llmKey(id), btn, !on ? "Stopping…" : back ? "Adding…" : "Starting…", async () => {
      await post("/api/coord/inference/models/serving", { model: id, on });
      toast(!on ? `Stopped serving ${id} on every Mac. The reply in progress finishes first; its files stay on disk.`
        : back ? `Added ${id} back. Macs that still have it can serve it.`
        : `Serving ${id} again. The next chat loads it.`);
      await loadLlm();
    });
  },
  "llm-remove": (btn) => {
    const id = btn.dataset.model || "";
    if (state.busy.has(llmKey(id))) return;
    if (!window.confirm(removeQuestion(llmModel(id) || { id }))) return;
    return withBusy(llmKey(id), btn, "Removing…", async () => {
      const r = await post("/api/coord/inference/models/remove", { model: id });
      toast(removedNote({ model: id, ...r }));
      await loadLlm();
    });
  },
  "toggle-contribute": (btn) => {
    const st = status();
    if (st.agent_draining) return;   // already stopping: the agent exits on its own
    const running = !!st.agent_running;
    return withBusy("contribute", btn, running ? "Stopping…" : "Starting…", async () => {
      if (running) {
        await post("/api/stop-agent");
        toast("Stopped contributing. The current step finishes first.");
        return;
      }
      // Lending to another pool stops a coordinator still hosted here, as Connect does: ask first.
      if ((state.settings || {}).mode !== "host" && !leaveHostedPool("Contributing to another pool")) return;
      const snap = await startWith({ contribute: true, training: true });
      if (isRestartNote(snap.last_error)) toast(snap.last_error);
      else if (snap.last_error) toast(snap.last_error, "bad");
      else toast("Contributing. This Mac picks up work whenever the pool has some.");
    });
  },

  host: (btn) => {
    // Hosting only switches the mode: whatever this Mac already lends keeps running, now to its own pool.
    const keep = lending(status());
    return withBusy("pool", btn, "Starting…", async () => {
      const snap = await startWith({ ...keep, mode: "host" });
      if (isRestartNote(snap.last_error)) toast(`Pool is up. ${snap.last_error}`);
      else if (snap.last_error) toast(snap.last_error, "bad");
      else toast("Pool is up. Share this Mac's address with the others.");
    });
  },

  "stop-pool": (btn) => withBusy("pool", btn, "Stopping…", async () => {
    await post("/api/stop");
    toast("Pool stopped.");
  }),

  connect: (btn) => {
    // Joining moves whatever this Mac lends to the new pool (and stops a pool hosted here).
    const keep = lending(status());
    return withBusy("connect", btn, "Connecting…", async () => {
      const url = $("#url").value.trim();
      if (!url) throw new Error("Enter the host Mac's address, or press Find on LAN.");
      if (!leaveHostedPool()) return;   // declined: the typed address stays, polls still leave it alone
      state.dirty.url = false;
      const snap = await startWith({ ...keep, mode: "join", url });
      if (snap.last_error && !isRestartNote(snap.last_error)) throw new Error(snap.last_error);
      const ov = await api("/api/overview");
      if (!ov.status.coordinator_up) throw new Error(`No coordinator answering at ${ov.status.coordinator_url}.`);
      toast(connectedNote(ov, snap));
    });
  },

  "connect-public": (btn) => {
    const keep = lending(status());
    return withBusy("connect", btn, "Connecting…", async () => {
      const url = $("#public-url").value.trim() || status().public_url || "";
      if (!url) throw new Error("Enter the public coordinator URL.");
      state.dirty.publicUrl = false;
      // The sign-in goes to the pool at this address: store it first, or there is nothing to sign in
      // to, then ask that pool whether the stored session is one of its own.
      await saveSettings({ mode: "public", url });
      await loadAuth();
      if (!signedIn()) throw new Error("Sign in first (in the sidebar), then press Connect.");
      if (!leaveHostedPool()) return;
      const snap = await startWith({ ...keep, mode: "public", url });
      if (snap.last_error && !isRestartNote(snap.last_error)) throw new Error(snap.last_error);
      const ov = await api("/api/overview");
      if (!ov.status.coordinator_up) throw new Error(`No coordinator answering at ${ov.status.coordinator_url}.`);
      toast(connectedNote(ov, snap));
    });
  },

  discover: (btn) => withBusy("discover", btn, "Searching…", async () => {
    const r = await post("/api/discover");
    state.settingsVersion += 1;   // the shell stored the address: a poll begun earlier may not undo it
    state.settings = { ...state.settings, url: r.url };
    state.dirty.url = false;
    $("#url").value = r.url;
    toast(`Found a pool at ${r.url}`);
  }),

  "copy-ip": async (btn) => {
    const ip = status().lan_ip || "";
    try {
      await navigator.clipboard.writeText(ip);
    } catch {
      const t = Object.assign(document.createElement("textarea"), { value: ip });
      document.body.append(t);
      t.select();
      document.execCommand("copy");
      t.remove();
    }
    btn.textContent = "Copied";
    window.setTimeout(() => { btn.textContent = "Copy"; }, 1400);
  },

  "show-auth": () => {
    $("#auth-gate").hidden = true;
    $("#auth-form").hidden = false;
    $("#auth-email").focus();
    renderAuth();
  },

  "auth-mode": () => {
    state.authMode = state.authMode === "register" ? "login" : "register";
    const btn = document.querySelector("[data-act='auth-mode']");
    if (btn) btn.textContent = state.authMode === "register" ? "Have an account" : "Create account";
    renderAuth();
  },

  "accept-terms": (btn) => withBusy("terms", btn, "Saving…", async () => {
    await post("/api/coord/auth/accept-terms", {});
    await loadAuth();
    toast("Terms accepted.");
  }),

  logout: (btn) => withBusy("auth", btn, "Signing out…", async () => {
    let remote = true;
    try {
      await post("/api/coord/auth/logout", {});
    } catch {
      remote = false;   // the pool did not answer: this Mac still forgets the session
    }
    // The stored session goes even when the pool did not answer: nothing here may keep earning for it.
    await saveSettings({ session_token: "" });
    state.user = null;
    state.credits = null;
    $("#auth-password").value = "";
    // Back to the sign-in form, even if this session started with a registration.
    state.authMode = "login";
    const mode = document.querySelector("[data-act='auth-mode']");
    if (mode) mode.textContent = "Create account";
    // The grant board is per user (admin review queue, pledges): rebuild it
    // so nothing from the old session stays on screen.
    $("#g-request").hidden = true;
    await loadGrants();
    if (remote) toast("Signed out.");
    else toast("Signed out on this Mac (the pool did not answer)", "bad");
  }),

  "toggle-request": () => {
    if (!signedIn()) return toast("Sign in to request a grant.", "bad");
    const card = $("#g-request");
    card.hidden = !card.hidden;
    setMsg("#g-msg", "", "");
    if (!card.hidden) $("#g-title").focus();
  },
};

document.addEventListener("click", async (e) => {
  const el = e.target.closest("button, [data-tab]");
  if (!el || el.disabled) return;
  const d = el.dataset;
  if (d.tab) return showTab(d.tab);
  if (d.act && actions[d.act]) return actions[d.act](el);
  if (d.mode) {
    // One stored address serves Join and Public: a switch between them drops it, so the LAN address
    // never turns up in the Public field (nor the public URL in the LAN one). Host keeps it.
    const prev = (state.settings || {}).mode;
    const remote = (m) => m === "join" || m === "public";
    const swap = prev !== d.mode && remote(prev) && remote(d.mode);
    await saveSettings(swap ? { mode: d.mode, url: "" } : { mode: d.mode });
    return poll();
  }
  if (d.llmHead) {
    await saveSettings({ inference_head: d.llmHead === "1" });
    return renderLlm();
  }
  if (d.transport) {
    await saveSettings({ transport: d.transport });
    toast(status().inference_running || status().coordinator_pid
      ? "Saved. Start serving again to restart the pool on the new transport."
      : "Saved. Used next time this Mac hosts.");
    return renderLlm();
  }
  if (d.sort) {
    state.sort = d.sort;
    return loadGrants();
  }
  if (d.cancel) {
    return withBusy(`cancel-${d.cancel}`, el, "Cancelling…", () => post(`/api/coord/jobs/${encodeURIComponent(d.cancel)}/cancel`));
  }
  if (d.fundOpen) {
    state.fundOpen = d.fundOpen;
    state.fundMsg = "";
    renderGrants();
    const input = $(`#fund-${CSS.escape(d.fundOpen)}`);
    if (input) input.focus();
    return;
  }
  if ("fundClose" in d) {
    state.fundOpen = null;
    return renderGrants();
  }
  if ((d.fundGo || d.review) && state.grants && state.grants.sample) return;
  if (d.fundGo) {
    const input = $(`#fund-${CSS.escape(d.fundGo)}`);
    const tflops = Number(input ? input.value : 0);
    const grant = state.grants.grants.find((g) => g.id === d.fundGo);
    // Busy per grant: a double-click must not pledge twice.
    return withBusy(`fund-${d.fundGo}`, el, "Pledging…", async () => {
      const res = await grantAction(`/api/grants/${encodeURIComponent(d.fundGo)}/fund`,
        { amount: tflops * T }, `Pledged ${withUnit(tflops * T)} to ${grant ? grant.title : "the grant"}`);
      state.fundMsg = res === true ? "" : res;
      if (res === true) state.fundOpen = null;
      renderGrants();
    });
  }
  if (d.review) {
    const approve = !!d.approve;
    return withBusy(`review-${d.review}`, el, approve ? "Approving…" : "Declining…", async () => {
      const res = await grantAction(`/api/grants/${encodeURIComponent(d.review)}/review`, { approve },
        approve ? "Approved. It's public now." : "Declined.");
      if (res !== true) toast(res, "bad");
    });
  }
});

// A slider mid-drag keeps the value under the thumb; polls sync it again once its change is saved.
// (Focus would tell us the same, but WKWebView does not report it reliably for range inputs.)
function bindSlider(el, onInput, onChange) {
  const key = el.id;
  let moved = false;
  let start = null;   // the value under the thumb when the press began
  el.addEventListener("pointerdown", () => {
    start = el.value;
    state.dragging.add(key);
  });
  el.addEventListener("input", () => {
    moved = true;
    state.dragging.add(key);
    onInput(el);
  });
  // A press that never moved the thumb, or a drag that came back to where it began, fires no change:
  // release the slider here so polls sync it again (a drag that did move is released by its change).
  const release = () => {
    if (moved && el.value !== start) return;
    const wandered = moved;
    moved = false;
    state.dragging.delete(key);
    if (wandered) render();   // the split's input wrote the value into settings: draw from them again
  };
  window.addEventListener("pointerup", release);
  window.addEventListener("pointercancel", release);
  el.addEventListener("change", async () => {
    try {
      await onChange(Number(el.value));
    } finally {
      moved = false;
      state.dragging.delete(key);
      render();
    }
  });
}

bindSlider($("#gpu"), (el) => {
  rangeFill(el);
  setText("#gpu-out", `${el.value}%`);
}, (gpu_percent) => saveSettings({ gpu_percent }));

bindSlider($("#mem"), (el) => {
  rangeFill(el);
  setText("#mem-out", memLabel(Number(el.value), trainingAutoGb()));
}, (memory_gb) => saveSettings({ memory_gb }));

bindSlider($("#split"), (el) => {
  state.settings = { ...state.settings, grant_split: Number(el.value) };
  renderContributions();
  renderGrantsLive();
}, async (grant_split) => {
  await saveSettings({ grant_split });
  if (!signedIn()) return;
  try {
    const r = await patch("/api/coord/auth/me", { grant_split });
    if (r && r.user) state.user = r.user;
  } catch (err) {
    toast(err.message, "bad");
  }
});

$("#url").addEventListener("input", () => { state.dirty.url = true; });
$("#url").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.isComposing) actions.connect($("#p-connect"));
});

$("#public-url").addEventListener("input", () => { state.dirty.publicUrl = true; });
// Leaving the field stores the address, so a sign-in reaches that pool before Connect is pressed.
$("#public-url").addEventListener("change", (e) => {
  const url = e.target.value.trim();
  state.dirty.publicUrl = false;
  if (url && url !== (state.settings || {}).url) saveSettings({ mode: "public", url });
});
$("#public-url").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.isComposing) actions["connect-public"]($("#p-connect-public"));
});

$("#dataset").addEventListener("change", (e) => {
  state.dataset = e.target.files[0] || null;
  const name = $("#dataset-name");
  name.textContent = state.dataset ? state.dataset.name : "No file chosen";
  name.classList.toggle("sig", !!state.dataset);
});

$("#job-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const msg = $("#job-msg");
  if (!state.dataset) return setMsg("#job-msg", "Choose a JSONL dataset first.", "bad");
  const steps = Number($("#steps").value);
  const publicPool = (state.settings || {}).mode === "public";
  const minStages = publicPool ? 1 : Number($("#min_stages").value);
  if (!Number.isInteger(steps) || steps < 1) return setMsg("#job-msg", "Steps must be a whole number above zero.", "bad");
  if (!Number.isInteger(minStages) || minStages < 1) return setMsg("#job-msg", "Min Macs must be at least 1.", "bad");
  const body = new FormData();
  body.append("dataset", state.dataset);
  body.append("model", $("#model").value);
  body.append("steps", String(steps));
  body.append("min_stages", String(minStages));
  if (signedIn()) {
    const tflops = Number($("#max_flops").value);
    if (!Number.isFinite(tflops) || tflops <= 0) {
      return setMsg("#job-msg", "Set a FLOP budget above zero.", "bad");
    }
    body.append("max_flops", String(tflops * T));
  }
  setMsg("#job-msg", `Uploading ${state.dataset.name}…`, "");
  msg.dataset.sticky = "1";
  return withBusy("submit", $("#job-submit"), "Uploading…", async () => {
    try {
      const job = await api("/api/coord/jobs/upload", { method: "POST", body, timeout: 0 });
      setMsg("#job-msg", `Submitted job ${String(job.id).slice(0, 8)}. It starts when enough Macs are free.`, "ok");
      toast("Job submitted");
    } catch (err) {
      setMsg("#job-msg", err.message, "bad");
    } finally {
      delete msg.dataset.sticky;
    }
  });
});

$("#grant-form").addEventListener("submit", (e) => {
  e.preventDefault();
  return withBusy("grant-request", e.target.querySelector("[type=submit]"), "Sending…", async () => {
    const res = await grantAction("/api/grants", {
      title: $("#g-title").value,
      summary: $("#g-summary").value,
      goal: Number($("#g-goal").value) * T,
    }, "Request sent. It goes public once an admin approves it.");
    if (res !== true) return setMsg("#g-msg", res, "bad");
    $("#g-title").value = "";
    $("#g-summary").value = "";
    $("#g-request").hidden = true;
  });
});

$("#auth-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const email = $("#auth-email").value.trim();
  const password = $("#auth-password").value;
  const name = $("#auth-name").value.trim();
  if (!email || !password) return setMsg("#auth-msg", "Email and password are required.", "bad");
  if (state.authMode === "register" && !name) return setMsg("#auth-msg", "Give the account a name.", "bad");
  return withBusy("auth", $("#auth-submit"), "Working…", async () => {
    const path = state.authMode === "register" ? "/api/coord/auth/register" : "/api/coord/auth/login";
    const body = { email, password };
    if (state.authMode === "register") body.name = name;
    // The shell stores the sign-in token as it relays the reply: nothing to save here.
    const r = await post(path, body);
    state.user = r.user || null;
    await loadAuth();
    if (state.user && !state.user.accepted_terms) {
      try {
        const t = await api("/api/coord/auth/terms");
        state.terms = (t && t.text) || "";
      } catch { state.terms = ""; }
    }
    setMsg("#auth-msg", "", "");
    $("#auth-password").value = "";
    toast(state.authMode === "register" ? "Account created." : "Signed in.");
    loadGrants();
  });
});

$("#l-model").addEventListener("change", (e) => { state.llm.model = e.target.value; renderLlm(); });
bindSlider($("#l-mem"), (el) => {
  rangeFill(el);
  setText("#l-mem-out", memLabel(Number(el.value), inferenceAutoGb()));
}, (inference_memory_gb) => saveSettings({ inference_memory_gb }));
$("#l-dir").addEventListener("change", (e) => saveSettings({ models_dir: e.target.value.trim() || "~/models" }));
$("#l-file").addEventListener("change", (e) => {
  const f = e.target.files[0];
  e.target.value = "";
  if (f) uploadModel(f);
});
$("#l-draft").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); sendChat(); }
});
$("#chat-form").addEventListener("submit", (e) => { e.preventDefault(); sendChat(); });
// Remember a Thinking block the user opened or closed, so the next streamed token doesn't undo it.
$("#l-log").addEventListener("click", (e) => {
  const block = e.target.closest("summary") && e.target.closest("[data-think]");
  if (block) state.llm.messages[block.dataset.think].thinkOpen = !block.open;
});

// A drop must never replace the dashboard: pywebview hands it to WKWebView, which would navigate to
// a dropped file or link. Files are refused everywhere; anything else is left to the browser only
// when it lands in a field, where the default is to insert the text.
const guardDrop = (e) => {
  const types = e.dataTransfer && e.dataTransfer.types;
  const files = !!types && Array.from(types).includes("Files");
  const t = e.target;
  const inField = !!(t && typeof t.closest === "function" && t.closest("input, textarea, [contenteditable]"));
  if (files || !inField) e.preventDefault();
};
window.addEventListener("dragover", guardDrop);
window.addEventListener("drop", guardDrop);

renderGate();   // nothing that saves is usable before the first overview answers
poll();   // each poll schedules the next once it has finished (see poll())
