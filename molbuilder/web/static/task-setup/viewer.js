/* task-setup/viewer.js — the Task Setup tab's page controller.
 *
 * WHAT THIS TAB DOES TODAY, and it is deliberately less than the design:
 *   1. Follow the projects sidebar's selected DIRECTORY.
 *   2. Read that folder's `task.json` (and notice the template beside it).
 *   3. Show what the description says — its stages, and the machine settings
 *      you either chose or asked to have measured.
 *   4. Let you read and edit the file itself in the vendored CodeMirror.
 *
 * IT WRITES NOTHING.  Save is disabled and says why.  `molbuilder jobset
 * describe` is what writes a description today, and building a browser write
 * path against a design neither of us has seen running is how you get a page
 * that is confidently wrong.
 *
 * The contract is `docs/web/task-setup.md`; where this disagrees, that wins.
 *
 * NO NEW ENDPOINT.  Reading a folder is `/api/files/list` + `/api/files/read`,
 * both already shipped and both already inside the roots guard.  `missing_ok`
 * exists precisely for the "this folder is not a calculation yet" case, so the
 * normal empty answer costs no failed-resource console error.
 */

import { loadCodeMirror, modeFor } from "../lib/codemirror-load.js";

const TASK_JSON     = "task.json";
/* The hand-over a parameter tab leaves (`stages.md` § 6.5a).  Read ONLY when
 * there is no `task.json` -- a folder holding both is a save that did not
 * finish, and the description wins because it is the one that passed the
 * preflight. */
const TASK_HANDOVER = "task.1st.json";

/* The three items whose value the MACHINE answers.  Each names an allocation
 * resolver, and `read_template` refuses a value on one — a description may
 * state the question and never the answer (`engines/template.md` § 6.4).  So
 * on this page they can only ever be points to measure, never a choice.
 * Hard-coded here rather than derived because the catalogue is not fetched by
 * this page; a fourth one would show up as `chosen`, which reads wrong but
 * breaks nothing, and the fix is one line. */
const MACHINE_ANSWERED = new Set(["mpi_np", "omp_threads", "max_memory_mb"]);

/* Friendly second lines.  A name the catalogue carries would be better and is
 * what the built version should read; this page does not load the catalogue. */
const ROW_NOTE = {
    enable_gpu:    "Use the GPU — yours to choose",
    use_gpu:       "Use the GPU — yours to choose",
    mpi_np:        "MPI ranks — the scheduler answers",
    omp_threads:   "Threads per rank — the scheduler answers",
    max_memory_mb: "Memory cap — the scheduler answers",
};

let _cm = null;
let _loadedText = "";
let _dir        = "";     // the folder currently open
let _shape      = "";     // "" until chosen — never defaulted (§ 4)
let _mode       = "";     // "description" | "handover" | "empty"
let _handover   = null;   // the parsed task.1st.json, in handover mode

const $ = (id) => document.getElementById(id);

/* ---------- small DOM helpers ---------- */

function el(tag, attrs, ...kids) {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
        if (v !== null && v !== undefined) n.setAttribute(k, v);
    }
    for (const kid of kids) {
        if (kid === null || kid === undefined) continue;
        n.appendChild(typeof kid === "string" ? document.createTextNode(kid) : kid);
    }
    return n;
}

function setState(kind, title, body) {
    $("ts-state").setAttribute("data-state", kind);
    $("ts-state-title").textContent = title;
    $("ts-state-body").textContent  = body;
}

function showPath(dir) {
    const host = $("ts-path");
    host.textContent = "";
    if (!dir) {
        host.appendChild(el("span", { class: "ts-path-seg" }, "no folder selected"));
        return;
    }
    const parts = String(dir).split("/").filter(Boolean);
    parts.forEach((seg, i) => {
        const last = i === parts.length - 1;
        host.appendChild(el("span",
            { class: last ? "ts-path-here" : "ts-path-seg" }, seg));
        if (!last) host.appendChild(el("span", { class: "ts-path-sep" }, "/"));
    });
}

function markFile(id, exists, presentText) {
    const n = $(id);
    n.setAttribute("data-exists", exists ? "yes" : "no");
    n.textContent = exists ? presentText : "not there yet";
}

/* ---------- rendering the description ---------- */

function renderStages(task) {
    const card = $("ts-stages-card");
    const stages = Array.isArray(task && task.stages) ? task.stages : [];
    const varies = Array.isArray(task && task.varies) ? task.varies : [];
    if (!stages.length) { card.hidden = true; return; }

    const table = $("ts-stage-table");
    const thead = table.querySelector("thead");
    const tbody = table.querySelector("tbody");
    thead.textContent = "";
    tbody.textContent = "";

    const hrow = el("tr", null, el("th", null, "Stage"));
    for (const col of varies) hrow.appendChild(el("th", null, col));
    thead.appendChild(hrow);

    for (const st of stages) {
        const name = (st && st.name) || "(unnamed)";
        const on   = st && st.enabled !== false;
        const tr = el("tr", null,
            el("td", null, on ? name : name + " (off)"));
        const ov = (st && st.overrides) || {};
        for (const col of varies) {
            const has = Object.prototype.hasOwnProperty.call(ov, col);
            // Absent is a real state: "this stage uses the template's value"
            // (`stages.md` § 6.2).  Shown muted, never blank-and-ambiguous.
            tr.appendChild(el("td",
                { "data-template": has ? null : "yes" },
                has ? String(ov[col]) : "template"));
        }
        tbody.appendChild(tr);
    }
    card.hidden = false;
}

function renderMachine(task) {
    const card = $("ts-machine-card");
    const host = $("ts-machine-rows");
    host.textContent = "";

    const bench = (task && task.bench) || {};
    const names = Object.keys(bench);
    if (!names.length) { card.hidden = true; return; }

    for (const name of names) {
        const pts = Array.isArray(bench[name]) ? bench[name] : [bench[name]];
        // The tab's one idea: length decides what this row IS.
        const kind = MACHINE_ANSWERED.has(name)
            ? "machine"
            : (pts.length === 1 ? "chosen" : "measured");
        const verdict = kind === "chosen"
            ? "chosen · 1 point"
            : `measured · ${pts.length} trial${pts.length === 1 ? "" : "s"}`;

        const pointEls = pts.map((p) => el("span", { class: "ts-pt" }, String(p)));
        host.appendChild(el("div", { class: "ts-row", "data-kind": kind },
            el("div", { class: "ts-row-name" }, name,
                el("small", null, ROW_NOTE[name] || "")),
            el("div", { class: "ts-points" }, ...pointEls),
            el("div", { class: "ts-verdict" }, verdict)));
    }
    card.hidden = false;
}

/* ---------- the editor ---------- */

async function ensureEditor() {
    if (_cm) return _cm;
    const CM = await loadCodeMirror();
    // Highlighting comes from the SUFFIX, and the mode file is fetched only
    // when a file of that kind is first opened (`lib/codemirror-load.js`).
    // `task.json` resolves to the javascript mode with `json: true` — the
    // JSON dialect, since CodeMirror ships no separate json mode.
    const mode = await modeFor(TASK_JSON);
    _cm = CM($("ts-editor"), {
        value:       "",
        mode:        mode,
        lineNumbers: true,
        lineWrapping: true,
        readOnly:    false,
    });
    _cm.on("change", () => {
        const dirty = _cm.getValue() !== _loadedText;
        $("ts-dirty").hidden = !dirty;
    });
    return _cm;
}

async function setEditorText(text) {
    const cm = await ensureEditor();
    _loadedText = text;
    cm.setValue(text);
    cm.clearHistory();          // a fresh file is not an undo step of the last
    $("ts-dirty").hidden = true;
    cm.refresh();               // it mounted inside a card that may have been
                                // hidden or resized since
}

/* ---------- loading a folder ---------- */

async function readOptional(projects, path) {
    try {
        // `missingOk` is camelCase HERE and `missing_ok` on the wire —
        // `lib/projects/api.js` maps it.  Passing the wire spelling is
        // silently ignored, and the 404 it then takes logs a failed-resource
        // console error for the perfectly normal "no description yet" case.
        const r = await projects.readFile(path, { missingOk: true });
        if (r && r.ok === false) return null;
        if (r && r.exists === false) return null;
        return (r && typeof r.text === "string") ? r.text : null;
    } catch (_) {
        return null;                       // absent is an answer, not a fault
    }
}

async function loadFolder(projects, dir) {
    _dir = dir;
    showPath(dir);
    if (!dir) {
        _mode = "empty"; refreshSave();
        setState("empty", "Nothing selected",
                 "Pick a calculation folder in the sidebar to read its description.");
        return;
    }

    const taskText = await readOptional(projects, dir + "/" + TASK_JSON);
    const overText = taskText
        ? null
        : await readOptional(projects, dir + "/" + TASK_HANDOVER);

    // What is already on disk, so the file list tells the truth rather than a
    // guess.  A listing failure is not fatal — the read above is what matters.
    let templateName = null;
    try {
        const listing = await projects.listDir(dir);
        const entries = (listing && listing.entries) || [];
        const tmpl = entries.find((e) => e && typeof e.name === "string"
                                      && e.name.endsWith(".template.toml"));
        if (tmpl) templateName = tmpl.name;
    } catch (_) { /* listing is a nicety here */ }

    markFile("ts-f-task", !!taskText, "already here — saving would update it");
    markFile("ts-f-tmpl", !!templateName, "already here — saving would update it");
    if (templateName) $("ts-f-tmpl-name").textContent = templateName;

    // A hand-over: the parameters arrived, the description has not been
    // finished.  Show the file so it can be read, and say what is missing.
    if (!taskText && overText) {
        let over = null;
        try { over = JSON.parse(overText); } catch (_) { /* shown raw below */ }
        const awaiting = (over && Array.isArray(over.awaiting))
            ? over.awaiting.join(" and ") : "shape and stages";
        _mode = "handover"; _handover = over;
        $("ts-shape-card").hidden = false;
        setShape(_shape);                       // re-assert / repaint the choice
        setState("handover",
                 "Handed over — not a description yet",
                 `${TASK_HANDOVER} is here, carrying the parameters. Still `
                 + `needed: ${awaiting}. Saving writes task.json and removes `
                 + `this file.`);
        $("ts-stages-card").hidden = true;
        renderMachine(over || {});
        if (!_shape) await setEditorText(overText);   // until a shape is picked
        refreshSave();
        return;
    }

    if (!taskText) {
        _mode = "empty"; _handover = null;
        $("ts-shape-card").hidden = true;
        refreshSave();
        setState("empty", "No description here yet",
                 "This folder carries no task.json and no hand-over. Send "
                 + "parameters here from the Structure-optimization tab, or "
                 + "run `molbuilder jobset describe`.");
        $("ts-stages-card").hidden = true;
        $("ts-machine-card").hidden = true;
        await setEditorText("");
        return;
    }

    let task = null;
    try {
        task = JSON.parse(taskText);
    } catch (e) {
        // Show the bytes anyway: a file you cannot parse is exactly the one
        // you want to look at.
        setState("refuse", "task.json is here but does not parse",
                 String(e && e.message ? e.message : e));
        $("ts-stages-card").hidden = true;
        $("ts-machine-card").hidden = true;
        await setEditorText(taskText);
        return;
    }

    const run  = (task && task.run) || {};
    const name = run.name || run.id || "(unnamed)";
    setState("loaded", `Loaded — ${name}`,
             `engine ${(task.engine && task.engine.name) || "?"}`
             + ` · shape ${task.shape || "?"}`
             + ` · ${(task.stages || []).length} stage(s)`);

    _mode = "description"; _handover = null;
    _shape = String(task.shape || "");
    $("ts-shape-card").hidden = false;
    setShape(_shape);                            // shows which one it carries
    renderStages(task);
    renderMachine(task);
    await setEditorText(taskText);
}

/* ---------- the shape, and what a hand-over becomes ---------- */

/** A description built from a hand-over plus the chosen shape.
 *
 * The editor shows WHAT WILL BE WRITTEN, not the hand-over file — a person
 * checking a description before a week of compute should be reading the thing
 * that lands, not its input.  `varies: []` and one stage named `coarse` are the
 * ordinary starting ladder (`stages.md` § 6.5): one stage is not a special
 * shape, and empty `varies` is a real state.
 */
function proposedFromHandover(over, shape) {
    const run = (over && over.run) || {};
    return JSON.stringify({
        schema:    "molbuilder/task@1",
        engine:    (over && over.engine) || { name: "siesta" },
        shape:     shape,
        run:       { name: run.name || "", id: run.id || "",
                     created: run.created || "" },
        structure: (over && over.structure) || {},
        varies:    [],
        stages:    [{ name: "coarse", enabled: true, overrides: {} }],
    }, null, 2) + "\n";
}

function setShape(shape) {
    _shape = shape;
    for (const b of document.querySelectorAll("#ts-shape-card .opt")) {
        b.setAttribute("aria-pressed",
                       b.getAttribute("data-shape") === shape ? "true" : "false");
    }
    const needs = $("ts-shape-needs");
    if (needs) needs.hidden = !!shape;
    if (_mode === "handover" && shape) {
        setEditorText(proposedFromHandover(_handover, shape));
    }
    refreshSave();
}

/** Save is enabled only when it could actually succeed. */
function refreshSave() {
    const btn = $("ts-save");
    const why = $("ts-save-why");
    if (!btn) return;
    let blocked = "";
    if (!_dir)                                    blocked = "Pick a folder first.";
    else if (_mode === "empty")                   blocked = "Nothing to save — this folder carries no description and no hand-over.";
    else if (_mode === "handover" && !_shape)     blocked = "Choose how the files are kept apart, above.";
    btn.disabled = !!blocked;
    if (why) {
        why.textContent = blocked
            || "Writes task.json into this folder. The text in the editor is "
             + "what gets written.";
    }
}

async function save() {
    if (!_cm || !_dir) return;
    const btn = $("ts-save");
    if (btn) btn.disabled = true;
    setState(_mode === "handover" ? "handover" : "loaded", "Saving…", "");
    let body;
    try {
        const r = await fetch("/api/task-setup/save", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ dest: _dir, text: _cm.getValue() }),
        });
        body = await r.json();
        if (!r.ok || !body || body.ok === false) {
            // Refused, not repaired — show the reader's own words.
            setState("refuse", "Not written", (body && body.error)
                     || ("save failed (" + r.status + ")"));
            refreshSave();
            return;
        }
    } catch (e) {
        setState("refuse", "Not written",
                 "Could not reach the server: " + (e && e.message ? e.message : e));
        refreshSave();
        return;
    }
    // The hand-over's REMOVAL is the browser's, through the content-blind
    // file layer (`web/projects.md` § 1) -- the server writes task.json
    // because it owns that schema, the way /api/structure/save owns the
    // sidecar, but moving bytes is this layer's job.  AFTER the write
    // succeeded: the reverse order loses the parameters if the write fails.
    const projects = window.molbuilder && window.molbuilder.projects;
    if (projects && body.handover_here
        && typeof projects.deleteEntry === "function") {
        const gone = await projects.deleteEntry(
            _dir.replace(/\/$/, "") + "/" + body.handover_name, false)
            .catch(() => null);
        if (!gone || gone.ok === false) {
            // The description is written and correct; a surviving hand-over is
            // untidy, not wrong, and § 6.5a says the description wins.
            console.warn("[task-setup] task.json written, but the hand-over "
                         + "could not be removed:", gone && gone.error);
        }
    }
    // Re-open the folder: it is now a description.
    if (projects) await loadFolder(projects, _dir);
}

/* ---------- wiring ---------- */

function start() {
    const projects = window.molbuilder && window.molbuilder.projects;
    if (!projects) {
        setState("refuse", "The projects sidebar did not load",
                 "This page follows the sidebar's selected folder, so it cannot "
                 + "read anything without it.");
        return;
    }

    // `getCurrentDir` is the public accessor; reading sessionStorage directly
    // would duplicate the sidebar's own key name in a second place.
    const startDir = typeof projects.getCurrentDir === "function"
        ? projects.getCurrentDir()
        : "";
    loadFolder(projects, startDir);

    // Directory changes arrive on onChange; a dblclick commit also lands on
    // onCommit.  Both carry `dir`, and this page cares about the folder only.
    if (typeof projects.onChange === "function") {
        projects.onChange((sel) => loadFolder(projects, (sel && sel.dir) || ""));
    }
    if (typeof projects.onCommit === "function") {
        projects.onCommit((sel) => loadFolder(projects, (sel && sel.dir) || ""));
    }

    for (const b of document.querySelectorAll("#ts-shape-card .opt")) {
        b.addEventListener("click",
            () => setShape(b.getAttribute("data-shape") || ""));
    }
    const saveBtn = $("ts-save");
    if (saveBtn) saveBtn.addEventListener("click", save);
    refreshSave();
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start, { once: true });
} else {
    start();
}
