/* job-prep/viewer.js — the Job Prep tab's page controller.
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
 * The contract is `docs/web/job-prep.md`; where this disagrees, that wins.
 *
 * NO NEW ENDPOINT.  Reading a folder is `/api/files/list` + `/api/files/read`,
 * both already shipped and both already inside the roots guard.  `missing_ok`
 * exists precisely for the "this folder is not a calculation yet" case, so the
 * normal empty answer costs no failed-resource console error.
 */

import { loadCodeMirror } from "../lib/codemirror-load.js";

const TASK_JSON = "task.json";

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
    $("jp-state").setAttribute("data-state", kind);
    $("jp-state-title").textContent = title;
    $("jp-state-body").textContent  = body;
}

function showPath(dir) {
    const host = $("jp-path");
    host.textContent = "";
    if (!dir) {
        host.appendChild(el("span", { class: "jp-path-seg" }, "no folder selected"));
        return;
    }
    const parts = String(dir).split("/").filter(Boolean);
    parts.forEach((seg, i) => {
        const last = i === parts.length - 1;
        host.appendChild(el("span",
            { class: last ? "jp-path-here" : "jp-path-seg" }, seg));
        if (!last) host.appendChild(el("span", { class: "jp-path-sep" }, "/"));
    });
}

function markFile(id, exists, presentText) {
    const n = $(id);
    n.setAttribute("data-exists", exists ? "yes" : "no");
    n.textContent = exists ? presentText : "not there yet";
}

/* ---------- rendering the description ---------- */

function renderStages(task) {
    const card = $("jp-stages-card");
    const stages = Array.isArray(task && task.stages) ? task.stages : [];
    const varies = Array.isArray(task && task.varies) ? task.varies : [];
    if (!stages.length) { card.hidden = true; return; }

    const table = $("jp-stage-table");
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
    const card = $("jp-machine-card");
    const host = $("jp-machine-rows");
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

        const pointEls = pts.map((p) => el("span", { class: "jp-pt" }, String(p)));
        host.appendChild(el("div", { class: "jp-row", "data-kind": kind },
            el("div", { class: "jp-row-name" }, name,
                el("small", null, ROW_NOTE[name] || "")),
            el("div", { class: "jp-points" }, ...pointEls),
            el("div", { class: "jp-verdict" }, verdict)));
    }
    card.hidden = false;
}

/* ---------- the editor ---------- */

async function ensureEditor() {
    if (_cm) return _cm;
    const CM = await loadCodeMirror();
    // `mode: null` is plain text and is deliberate: the vendored 5.65.16
    // bundle ships the markdown mode and no javascript/json one, and adding
    // one is a vendor-inventory decision (`static/vendor/README.md`), not a
    // convenience.  Line numbers, editing, undo and the search addons all
    // work without it.
    _cm = CM($("jp-editor"), {
        value:       "",
        mode:        null,
        lineNumbers: true,
        lineWrapping: true,
        readOnly:    false,
    });
    _cm.on("change", () => {
        const dirty = _cm.getValue() !== _loadedText;
        $("jp-dirty").hidden = !dirty;
    });
    return _cm;
}

async function setEditorText(text) {
    const cm = await ensureEditor();
    _loadedText = text;
    cm.setValue(text);
    cm.clearHistory();          // a fresh file is not an undo step of the last
    $("jp-dirty").hidden = true;
    cm.refresh();               // it mounted inside a card that may have been
                                // hidden or resized since
}

/* ---------- loading a folder ---------- */

async function readOptional(projects, path) {
    try {
        const r = await projects.readFile(path, { missing_ok: true });
        if (r && r.ok === false) return null;
        if (r && r.exists === false) return null;
        return (r && typeof r.text === "string") ? r.text : null;
    } catch (_) {
        return null;                       // absent is an answer, not a fault
    }
}

async function loadFolder(projects, dir) {
    showPath(dir);
    if (!dir) {
        setState("empty", "Nothing selected",
                 "Pick a calculation folder in the sidebar to read its description.");
        return;
    }

    const taskText = await readOptional(projects, dir + "/" + TASK_JSON);

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

    markFile("jp-f-task", !!taskText, "already here — saving would update it");
    markFile("jp-f-tmpl", !!templateName, "already here — saving would update it");
    if (templateName) $("jp-f-tmpl-name").textContent = templateName;

    if (!taskText) {
        setState("empty", "No description here yet",
                 "This folder carries no task.json. Saving would write a new "
                 + "one — once saving is built.");
        $("jp-stages-card").hidden = true;
        $("jp-machine-card").hidden = true;
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
        $("jp-stages-card").hidden = true;
        $("jp-machine-card").hidden = true;
        await setEditorText(taskText);
        return;
    }

    const run  = (task && task.run) || {};
    const name = run.name || run.id || "(unnamed)";
    setState("loaded", `Loaded — ${name}`,
             `engine ${(task.engine && task.engine.name) || "?"}`
             + ` · shape ${task.shape || "?"}`
             + ` · ${(task.stages || []).length} stage(s)`);

    renderStages(task);
    renderMachine(task);
    await setEditorText(taskText);
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

    const currentDir = () => {
        try { return sessionStorage.getItem("molbuilder.current_dir") || ""; }
        catch (_) { return ""; }
    };

    loadFolder(projects, currentDir());

    // Directory changes arrive on onChange; a dblclick commit also lands on
    // onCommit.  Both carry `dir`, and this page cares about the folder only.
    if (typeof projects.onChange === "function") {
        projects.onChange((sel) => loadFolder(projects, (sel && sel.dir) || ""));
    }
    if (typeof projects.onCommit === "function") {
        projects.onCommit((sel) => loadFolder(projects, (sel && sel.dir) || ""));
    }
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start, { once: true });
} else {
    start();
}
