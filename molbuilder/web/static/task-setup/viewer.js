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
/* The parsed description the table is a VIEW of.
 *
 * ONE DIRECTION AT A TIME, because two-way binding between a table and a text
 * buffer is how you get an edit loop.  A table edit mutates this, re-serialises
 * into the editor, and re-renders; a hand edit in the editor re-parses into
 * this when it is valid.  The BUFFER stays what `save` sends (`task-setup.md`
 * § 9a) -- the model is a convenience for the table, never the source. */
let _task       = null;
let _reparse    = null;   // debounce for the editor -> model re-parse
let _runs       = {};     // stage name -> attempts on disk (T5)

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
    for (const col of varies) {
        const x = el("button", {
            type: "button",
            class: "ts-rowbtn ts-rowbtn-drop"
                   + (_pendingDrop === col ? " is-pending" : ""),
            title: "Remove this column",
        }, "\u00d7");
        x.addEventListener("click", () => removeColumn(col));
        const th = el("th", { title: helpText(col) }, col, x);
        hrow.appendChild(th);
    }
    thead.appendChild(hrow);

    stages.forEach((st, i) => {
        const name = (st && st.name) || "(unnamed)";
        const on   = st && st.enabled !== false;

        const nameInput = el("input", {
            class: "ts-cell ts-cell-name", value: name,
            "aria-label": "stage " + (i + 1) + " name",
        });
        nameInput.addEventListener("change", () => {
            const v = nameInput.value.trim();
            // The name keys filenames, so the rule is the description's, not
            // this page's (`stages.md` § 2): letters, digits, underscore.
            if (!/^[A-Za-z0-9_]+$/.test(v)) {
                setState("refuse", "That stage name cannot be used",
                         "A stage name is letters, digits and underscore — no "
                         + "hyphen, because a hyphen means \'a counter "
                         + "follows\' everywhere else in the system.");
                nameInput.value = name;
                return;
            }
            _task.stages[i].name = v;
            syncFromModel();
        });

        const toggle = el("button", {
            type: "button", class: "ts-rowbtn",
            "aria-pressed": on ? "true" : "false",
            title: on ? "Disable this stage" : "Enable this stage",
        }, on ? "on" : "off");
        toggle.addEventListener("click", () => toggleStage(i));

        const drop = el("button", {
            type: "button", class: "ts-rowbtn ts-rowbtn-drop",
            title: "Remove this stage",
        }, "\u00d7");
        drop.addEventListener("click", () => removeStage(i));

        // Fill this row from a shipped tier (`tuning.md` § 4).
        const preset = el("select", { class: "ts-preset",
                                      "aria-label": "apply a preset to " + name });
        preset.appendChild(el("option", { value: "" }, "preset\u2026"));
        for (const ps of (_presets || [])) {
            preset.appendChild(el("option", { value: String(ps.tier) }, ps.name));
        }
        preset.addEventListener("change", () => {
            const ps = (_presets || []).find(
                (x) => String(x.tier) === preset.value);
            preset.value = "";
            if (ps) applyPreset(i, ps.values);
        });

        const ran = _runs[name];
        const ranEl = (ran === undefined) ? null
            : el("span", { class: "ts-ran",
                           title: ran ? ran + " attempt(s) on disk"
                                      : "nothing has run for this stage yet" },
                 ran ? ran + "\u00d7" : "\u2014");

        const tr = el("tr", { "data-off": on ? null : "yes" },
            el("td", null, nameInput, toggle, drop, preset, ranEl));

        const ov = (st && st.overrides) || {};
        for (const col of varies) {
            const has = Object.prototype.hasOwnProperty.call(ov, col);
            // An empty cell is not blank: it says what the stage will USE,
            // which is the template's value.  Showing the number rather than
            // the word "template" is what makes "adding a column changes
            // nothing on screen" (§ 9) visible instead of merely true.
            const fallback = defaultText(col);
            const cell = el("input", {
                class: "ts-cell", value: has ? String(ov[col]) : "",
                placeholder: fallback || "template",
                title: helpText(col),
                "aria-label": name + " " + col,
                "data-template": has ? null : "yes",
            });
            cell.addEventListener("change", () => setCell(i, col, cell.value));
            tr.appendChild(el("td", { "data-template": has ? null : "yes" }, cell));
        }
        tbody.appendChild(tr);
    });

    const actions = $("ts-stage-actions");
    if (actions) actions.hidden = false;
    card.hidden = false;
}

function renderMachine(task) {
    const card = $("ts-machine-card");
    const host = $("ts-machine-rows");
    host.textContent = "";

    const bench = (task && task.bench) || {};
    const names = Object.keys(bench);

    for (const name of names) {
        const pts = Array.isArray(bench[name]) ? bench[name] : [bench[name]];
        // The tab's one idea: length decides what this row IS.  A machine-
        // answered setting stays `machine` at any length -- a description may
        // never assert a value for it, so even one point is a point to TRY.
        const kind = MACHINE_ANSWERED.has(name)
            ? "machine"
            : (pts.length === 1 ? "chosen" : "measured");
        const verdict = kind === "chosen"
            ? "chosen · 1 point"
            : `${kind === "machine" ? "to try" : "measured"} · ${pts.length} `
              + `point${pts.length === 1 ? "" : "s"}`;

        const chips = pts.map((p, i) => {
            const drop = el("button", { type: "button", class: "ts-pt-x",
                                        title: "Remove this point" }, "\u00d7");
            drop.addEventListener("click", () => removePoint(name, i));
            return el("span", { class: "ts-pt" }, String(p), drop);
        });

        const add = el("input", { class: "ts-pt-add", placeholder: "+ add",
                                  "aria-label": "add a point to " + name });
        add.addEventListener("change", () => { addPoint(name, add.value); add.value = ""; });

        const dropRow = el("button", { type: "button", class: "ts-rowbtn ts-rowbtn-drop",
                                       title: "Stop measuring " + name }, "\u00d7");
        dropRow.addEventListener("click", () => removeSetting(name));

        host.appendChild(el("div", { class: "ts-row", "data-kind": kind },
            el("div", { class: "ts-row-name", title: helpText(name) }, name,
                el("small", null, ROW_NOTE[name] || "")),
            el("div", { class: "ts-points" }, ...chips, add),
            el("div", { class: "ts-verdict" }, verdict, dropRow)));
    }
    const acts = $("ts-machine-actions");
    if (acts) acts.hidden = false;
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
        // A hand edit in the editor re-parses into the model, so the table
        // keeps showing what the buffer says.  Debounced, and silent when the
        // text is mid-typing and does not parse — an editor that flashed a
        // refusal on every keystroke would be unusable.
        clearTimeout(_reparse);
        _reparse = setTimeout(() => {
            let next = null;
            try { next = JSON.parse(_cm.getValue()); } catch (_) { return; }
            if (!next || typeof next !== "object") return;
            _task = next;
            renderStages(_task);
            refreshSave();
        }, 400);
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
    // Before anything renders: what this folder's template answers is the
    // baseline every empty cell names.
    await loadTemplateValues(dir);
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
        renderCameOver(over);
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
        renderCameOver(null);
        $("ts-shape-card").hidden = true;
        $("ts-next-card").hidden = true;
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

    _mode = "description"; _handover = null; _task = task;
    renderCameOver(task);
    _runs = await runsForStages(projects, dir, task);
    _shape = String(task.shape || "");
    $("ts-shape-card").hidden = false;
    setShape(_shape);                            // shows which one it carries
    renderStages(task);
    renderNext(task);
    renderMachine(task);
    refreshPickers();
    await setEditorText(taskText);
}

/* ---------- the table edits the description ---------- */

/** Push the model into the buffer and repaint. */
async function syncFromModel() {
    if (!_task) return;
    await setEditorText(JSON.stringify(_task, null, 2) + "\n");
    renderStages(_task);
    renderNext(_task);
    refreshPickers();
    refreshSave();
}

/** `stages.md` § 6.5: removing the last stage is refused — a job always has
 *  at least one, so there is no stage-less shape to fall back to. */
function removeStage(i) {
    if (!_task || !Array.isArray(_task.stages)) return;
    if (_task.stages.length <= 1) {
        setState("refuse", "Cannot remove the last stage",
                 "A job always has at least one stage — one is the ordinary "
                 + "case, not a special shape. Rename it or change its values "
                 + "instead.");
        return;
    }
    _task.stages.splice(i, 1);
    syncFromModel();
}

/** `task-setup.md` § 9: a new stage COPIES the previous one's overrides.
 *  A refinement starts from what came before; a stage that inherits nothing
 *  is a different calculation, not a next step. */
function addStage() {
    if (!_task) return;
    const stages = _task.stages || (_task.stages = []);
    const prev = stages[stages.length - 1];
    let name = "stage" + (stages.length + 1);
    const taken = new Set(stages.map((x) => String(x.name || "").toLowerCase()));
    let n = stages.length + 1;
    while (taken.has(name.toLowerCase())) { n += 1; name = "stage" + n; }
    stages.push({
        name,
        enabled: true,
        overrides: Object.assign({}, (prev && prev.overrides) || {}),
    });
    syncFromModel();
}

function toggleStage(i) {
    if (!_task || !_task.stages || !_task.stages[i]) return;
    // Disabling changes what `prep` builds; it does NOT delete the row's
    // values (`task-setup.md` § 9).
    _task.stages[i].enabled = _task.stages[i].enabled === false;
    syncFromModel();
}

function setCell(i, col, raw) {
    if (!_task || !_task.stages || !_task.stages[i]) return;
    const ov = _task.stages[i].overrides || (_task.stages[i].overrides = {});
    const text = String(raw).trim();
    if (text === "") {
        // Empty means "this stage uses the template's value" — a real state,
        // expressed by the key being ABSENT (`stages.md` § 6.2).
        delete ov[col];
    } else {
        const n = Number(text);
        ov[col] = (text !== "" && Number.isFinite(n)) ? n : text;
    }
    syncFromModel();
}

/* ---------- what came over ---------- */

/** The identity facts, read-only (`task-setup.md` § 3).
 *
 * Shown because you are about to commit a week of compute against them, not so
 * they can be changed.  The **id** is displayed and never recomputed: it is
 * read from the file and checked, which is what makes a rename DETECTABLE
 * rather than silent (`run-identity.md` § 3).
 */
function renderCameOver(obj) {
    const host = $("ts-facts");
    const card = $("ts-came-card");
    if (!host || !card) return;
    host.textContent = "";
    if (!obj) { card.hidden = true; return; }
    const run = obj.run || {};
    const st  = obj.structure || {};
    const rows = [
        ["Calculation", run.name || "\u2014"],
        ["Run id",      run.id   || "\u2014"],
        ["Engine",      (obj.engine && obj.engine.name) || "\u2014"],
        ["Structure",   st.source ? String(st.source).split("/").pop() : "\u2014"],
        ["Formula",     st.formula || "\u2014"],
        ["Atoms",       (st.atoms === undefined ? "\u2014" : String(st.atoms))],
    ];
    for (const [k, v] of rows) {
        host.appendChild(el("div", null,
            el("dt", null, k), el("dd", null, v)));
    }
    card.hidden = false;
}

/* ---------- the pickers: what may be added, from the catalogue ---------- */

let _cols = null;       // every parameter this engine has  {name,label,group}
/* name -> the catalogue's own item, so the table can show what a parameter
 * DEFAULTS to and what it is for.  The catalogue already carries `default`,
 * `unit` and `help`; a second copy here would be the drift the one-source rule
 * exists to prevent, so this is a lookup into what the schema returned. */
const _meta = Object.create(null);
let _sweep = null;      // the ones a benchmark may sweep

/** Every parameter, for the column picker.
 *
 * ANY field may be promoted (`stages.md` § 1.2 — the group is a default, never
 * a restriction), so this is the whole form schema, which the parameter tab
 * already builds from the catalogue.  No second list to keep in step.
 */
async function loadColumnChoices(engine) {
    if (_cols) return _cols;
    const r = await fetch("/api/build/schema/"
                          + encodeURIComponent(engine || "siesta"));
    const j = await r.json();
    const secs = (j && j.schema && j.schema.sections) || [];
    _cols = [];
    for (const sec of secs) {
        for (const f of (sec.fields || [])) {
            _cols.push({ name: f.name, label: f.label || f.name,
                         group: f.workflow_group || "" });
            _meta[f.name] = f;
        }
    }
    return _cols;
}

/* A STARTING SWEEP for the settings the machine answers.
 *
 * They can only ever be points to try -- a description may never carry a value
 * for one (`template.md` § 6.4) -- so an empty bench leaves the card with
 * nothing in it and the user typing point lists from scratch.  These are the
 * shipped starting points, and `stages.md` § 6.8's rule is what makes them
 * safe to propose: `bench` records POINTS TO TRY and never an answer, so a
 * proposed grid costs nothing but a measurement, and every row can be edited
 * or dropped.
 *
 * Powers of two for ranks because that is how the block distributes
 * (`tuning.md` § 2.11); 1 and 2 threads because hybrid runs are the comparison
 * worth making first. */
const BENCH_START = { mpi_np: [4, 8, 16], omp_threads: [1, 2] };

/** The sweepable set — `execution` category only (`stages.md` § 6.8).
 *  A separate read because the FORM filters `staging` out, and those are
 *  exactly the knobs a benchmark measures. */
async function loadSweepChoices(engine) {
    if (_sweep) return _sweep;
    const r = await fetch("/api/task-setup/sweepable?engine="
                          + encodeURIComponent(engine || "siesta"));
    const j = await r.json();
    _sweep = (j && j.items) || [];
    // `staging` items are filtered out of the form schema, so this is the
    // only place their note arrives -- fold it in so the table hovers work.
    for (const i of _sweep) {
        if (!_meta[i.name]) {
            _meta[i.name] = { name: i.name, label: i.label, help: i.help };
        }
    }
    return _sweep;
}

/** Fill a <select> with what is NOT already used. */
function fillPicker(sel, items, taken, empty) {
    if (!sel) return;
    sel.textContent = "";
    const left = items.filter((i) => taken.indexOf(i.name) === -1);
    if (!left.length) {
        sel.appendChild(el("option", { value: "" }, empty));
        sel.disabled = true;
        return;
    }
    sel.disabled = false;
    sel.appendChild(el("option", { value: "" }, "choose a parameter\u2026"));
    for (const i of left) {
        const d = defaultText(i.name);
        sel.appendChild(el("option", {
            value: i.name,
            title: helpText(i.name) || i.help || "",
        }, i.name
           + (i.label && i.label !== i.name ? "  \u2014  " + i.label : "")
           + (d ? "  [" + d + "]" : "")
           + (i.machine_answers ? "  (the machine answers this)" : "")));
    }
}

async function refreshPickers() {
    const engine = (_task && _task.engine && _task.engine.name) || "siesta";
    try {
        const [cols, sweep] = await Promise.all([
            loadColumnChoices(engine), loadSweepChoices(engine),
            loadPresets(engine)]);
        fillPicker($("ts-add-col"), cols,
                   Array.isArray(_task && _task.varies) ? _task.varies : [],
                   "every parameter is already a column");
        fillPicker($("ts-add-setting"), sweep,
                   Object.keys((_task && _task.bench) || {}),
                   "every sweepable setting is already listed");
    } catch (_) { /* the pickers stay empty; nothing else breaks */ }
}

/* What THIS FOLDER's template answers, which is not the same thing as what the
 * catalogue recommends.  `stages.md` § 6.2: a stage that sets nothing uses the
 * TEMPLATE's value -- so the k-grid a person chose in the parameter tab is the
 * number an empty cell must name here, and the catalogue default is only the
 * answer when the template is silent. */
let _tmpl = { name: null, values: Object.create(null) };

async function loadTemplateValues(dir) {
    _tmpl = { name: null, values: Object.create(null) };
    if (!dir) return;
    try {
        const r = await fetch("/api/task-setup/template-values?dir="
                              + encodeURIComponent(dir));
        const j = await r.json();
        if (j && j.ok) _tmpl = { name: j.name, values: j.values || {} };
    } catch (_) { /* the cells fall back to the catalogue, as before */ }
}

/** Rendered the way it will be written, unit and all. */
function renderValue(v, unit) {
    if (v === undefined || v === null) return "";
    const t = Array.isArray(v) ? v.join(", ") : String(v);
    return t + (unit ? " " + unit : "");
}

/** What a stage that sets nothing will actually use: the template first. */
function defaultText(name) {
    const m = _meta[name];
    const unit = m && m.unit;
    if (name in _tmpl.values) return renderValue(_tmpl.values[name], unit);
    if (!m) return "";
    return renderValue(m.default, unit);
}

/** The parameter's own note, for a hover. */
function helpText(name) {
    const m = _meta[name];
    if (!m) return "";   // caller falls back to its own note
    const bits = [m.label && m.label !== name ? m.label : "", m.help || ""];
    /* When the template answers, BOTH numbers are worth reading: one is what
     * this job runs, the other is what the catalogue recommends, and a person
     * checking a description before a week of compute wants to see that they
     * differ. */
    if (name in _tmpl.values) {
        bits.push("This job (" + (_tmpl.name || "template") + "): "
                  + renderValue(_tmpl.values[name], m.unit));
    }
    const d = renderValue(m.default, m.unit);
    if (d) bits.push("Recommended: " + d);
    if (m.engine_key) bits.push("Writes: " + m.engine_key);
    return bits.filter(Boolean).join("\n\n");
}

/* ---------- the tier presets ---------- */

let _presets = null;

async function loadPresets(engine) {
    if (_presets) return _presets;
    const r = await fetch("/api/task-setup/presets?engine="
                          + encodeURIComponent(engine || "siesta"));
    const j = await r.json();
    _presets = (j && j.presets) || [];
    return _presets;
}

/**
 * Fill a stage's row from a tier.
 *
 * `task-setup.md` § 9: *"a preset knows several fields.  If some are not
 * columns yet it **adds them first** -- a preset that half-applied would be
 * worse than one that refused."*  So every field the preset carries is
 * promoted before any value is written, and the whole thing lands or nothing
 * does.
 *
 * The values come from the SAME table `default_siesta_stages` builds the
 * shipped ladder from, so a stage filled here and stage N of that ladder
 * cannot drift (`tuning.md` § 4 is the authority for the numbers).
 */
function applyPreset(i, values) {
    if (!_task || !_task.stages || !_task.stages[i]) return;
    const v = variesOf(); if (!v) return;
    const added = [];
    for (const key of Object.keys(values)) {
        if (v.indexOf(key) === -1) { v.push(key); added.push(key); }
    }
    const ov = _task.stages[i].overrides || (_task.stages[i].overrides = {});
    Object.assign(ov, values);
    if (added.length) {
        setState(_mode === "handover" ? "handover" : "loaded",
                 "Preset applied",
                 "Added " + added.join(", ") + " as column"
                 + (added.length === 1 ? "" : "s") + ", because the preset "
                 + "sets them.");
    }
    syncFromModel();
}

/* ---------- the columns: which parameters vary ---------- */

/** `varies` is the column set; a stage's `overrides` fills the cells it
 *  chooses to (`stages.md` § 6.2). */
function variesOf() {
    if (!_task) return null;
    if (!Array.isArray(_task.varies)) _task.varies = [];
    return _task.varies;
}

/**
 * Add a column.
 *
 * `task-setup.md` § 9: it "seeds every stage with the current base value, so
 * promoting changes nothing on screen — a statement about STRUCTURE, never
 * about values".  Here that costs nothing to honour: an ABSENT override
 * already means "this stage uses the template's value" (`stages.md` § 6.2),
 * so adding the column and touching no cell IS seeding them all.
 */
function addColumn(name) {
    const v = variesOf(); if (!v) return;
    if (!/^[a-z_][a-z0-9_]*$/i.test(name)) {
        setState("refuse", "That is not a parameter name",
                 "Use the parameter's own name — letters, digits and "
                 + "underscore, as the catalogue spells it.");
        return;
    }
    if (v.indexOf(name) !== -1) return;
    v.push(name);
    syncFromModel();
}

/* Removing a column destroys values, so it is a TWO-CLICK act: the first says
 * what would be lost, the second does it (`task-setup.md` § 9 — "the page says
 * which value it kept, and says it BEFORE the click").  No browser dialog:
 * a `confirm()` blocks everything, including the page's own scripts. */
let _pendingDrop = "";

function removeColumn(name) {
    const v = variesOf(); if (!v) return;
    const stages = (_task && _task.stages) || [];
    const enabled = stages.filter((st) => st && st.enabled !== false);
    const last = enabled.length ? enabled[enabled.length - 1] : stages[stages.length - 1];
    const survivor = last && last.overrides
        ? last.overrides[name] : undefined;

    if (_pendingDrop !== name) {
        _pendingDrop = name;
        // The value the LAST ENABLED stage carries is the one § 9 keeps —
        // it is the production stage, and the value a single run would use.
        // This page cannot write it into the template, so it says so rather
        // than implying the value survives somewhere.
        setState("refuse", "Remove the column " + name + "?",
                 (survivor === undefined
                    ? "No stage overrides it, so nothing is lost. "
                    : "The last enabled stage (" + ((last && last.name) || "?")
                      + ") has " + JSON.stringify(survivor) + ", and every "
                      + "stage's value for it is dropped — this page edits "
                      + "task.json, not the template, so set it there if you "
                      + "want it kept. ")
                 + "Click × again to remove it.");
        renderStages(_task);
        return;
    }

    _pendingDrop = "";
    v.splice(v.indexOf(name), 1);
    for (const st of stages) {
        if (st && st.overrides) delete st.overrides[name];
    }
    syncFromModel();
}

/* ---------- the hand-off: the next command ---------- */

/** The exact commands, per stage, in order.
 *
 * `task-setup.md` § 1: this page "turns that into a description on disk and
 * **hands you the command to run it somewhere else**".  Half the tab's purpose,
 * and it has to be EXACT rather than a generic snippet — a stage that continues
 * names what it continues from, because `prep` refuses to guess (`stages.md`
 * § 6.5: every verb is given the stage's name).
 */
function renderNext(task) {
    const card = $("ts-next-card");
    const host = $("ts-next");
    if (!card || !host) return;
    host.textContent = "";
    const stages = ((task && task.stages) || []).filter(
        (st) => st && st.enabled !== false);
    if (!stages.length) { card.hidden = true; return; }

    stages.forEach((st, i) => {
        const name = st.name || "";
        const ov = st.overrides || {};
        // `continue` carries from the stage before it — and `prep` is TOLD
        // which attempt, never left to guess (`project-layout.md` § 1.6).
        let from = "";
        if (i > 0 && String(ov.restart || "") === "continue") {
            const prev = stages[i - 1];
            const token = String(i).padStart(2, "0") + "_" + (prev.name || "");
            from = " --from " + token + "/run-0";
        }
        const runs = _runs[name];
        host.appendChild(el("div", { class: "ts-next-step" },
            el("div", { class: "ts-next-stage" }, name,
               (runs ? el("span", { class: "ts-ran" }, runs + "\u00d7 run") : null)),
            el("pre", { class: "ts-cmd" },
               "molbuilder jobset prep run " + name + from + "\n"
               + "molbuilder jobset submit run " + name)));
    });
    card.hidden = false;
}

/* ---------- what has already run ---------- */

/** Attempts per stage, read from the DIRECTORY — no target machine needed,
 *  which is why this belongs here and not on Results (`task-setup.md` § 10).
 *
 *  Both shapes, from `job-contracts.md` § 6.3's Files table:
 *    hierarchical  <NN>_<name>/run-<n>/
 *    flat          <label>_<NN>_<name>-run<N>.out   (beside the deck)
 *
 *  It counts attempts and does not judge them.  Whether a run CONVERGED is in
 *  its output, and parsing engine output is the Results tab's contract, not a
 *  claim this page should make from a filename.
 */
async function runsForStages(projects, dir, task) {
    const out = {};
    const stages = (task && task.stages) || [];
    let entries = [];
    try {
        const listing = await projects.listDir(dir);
        entries = (listing && listing.entries) || [];
    } catch (_) { return out; }

    const names = entries.map((e) => (e && e.name) || "");
    for (let i = 0; i < stages.length; i++) {
        const stage = stages[i];
        const token = String(i + 1).padStart(2, "0") + "_" + (stage.name || "");
        const dirHit = names.filter((n) => n === token);
        let attempts = 0;
        if (dirHit.length) {
            try {
                const sub = await projects.listDir(dir + "/" + token);
                attempts = ((sub && sub.entries) || [])
                    .filter((e) => /^run-\d+$/.test((e && e.name) || "")).length;
            } catch (_) { /* unreadable is not zero, but it is all we can say */ }
        } else {
            // flat: one output per attempt, carrying the same token
            attempts = names.filter(
                (n) => n.indexOf("_" + token + "-run") !== -1).length;
        }
        out[stage.name] = attempts;
    }
    return out;
}

/* ---------- the machine rows: a point is a choice, several a measurement ---- */

/** Coerce a typed point to what `task.json` should carry.
 *  `bench` takes scalars only — the reader refuses a nested value. */
function coercePoint(raw) {
    const t = String(raw).trim();
    if (t === "") return null;
    if (/^(true|on|yes)$/i.test(t))  return true;
    if (/^(false|off|no)$/i.test(t)) return false;
    const n = Number(t);
    return Number.isFinite(n) ? n : t;
}

function benchOf() {
    if (!_task) return null;
    return _task.bench || (_task.bench = {});
}

function addPoint(name, raw) {
    const v = coercePoint(raw);
    if (v === null) return;
    const b = benchOf(); if (!b) return;
    const pts = Array.isArray(b[name]) ? b[name] : (b[name] === undefined ? [] : [b[name]]);
    // Adding a point to a CHOSEN setting keeps the value as the first point,
    // so measuring never discards what you chose (`task-setup.md` § 9).
    if (pts.some((p) => String(p) === String(v))) return;
    pts.push(v);
    b[name] = pts;
    syncFromModel();
}

function removePoint(name, idx) {
    const b = benchOf(); if (!b || !Array.isArray(b[name])) return;
    b[name].splice(idx, 1);
    // `bench` takes a NON-EMPTY list, so a setting with no points is not a
    // setting with zero points -- it is a setting that is not being measured.
    if (!b[name].length) delete b[name];
    if (!Object.keys(b).length) delete _task.bench;
    syncFromModel();
}

function removeSetting(name) {
    const b = benchOf(); if (!b) return;
    delete b[name];
    if (!Object.keys(b).length) delete _task.bench;
    syncFromModel();
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
function proposedFromHandover(over, shape, varies, bench) {
    const run = (over && over.run) || {};
    return JSON.stringify({
        schema:    "molbuilder/task@1",
        engine:    (over && over.engine) || { name: "siesta" },
        shape:     shape,
        run:       { name: run.name || "", id: run.id || "",
                     created: run.created || "" },
        structure: (over && over.structure) || {},
        varies:    varies || [],
        stages:    [{ name: "coarse", enabled: true, overrides: {} }],
        bench:     bench || undefined,
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
        /* THE STARTING MATRIX.  `stages.md` § 1.3: *"`varies` defaults to the
         * engine's `stage` group, and the user adds to or removes from it"* —
         * the settings that typically vary across a sequence.  A description
         * that arrived with an empty `varies` would open on a table with no
         * columns and nothing to edit, which is not a neutral starting point,
         * it is a dead end.  The group is a DEFAULT, never a restriction: any
         * parameter can be added and any of these removed (§ 1.2). */
        const engine = (_handover && _handover.engine
                        && _handover.engine.name) || "siesta";
        Promise.all([loadColumnChoices(engine),
                     loadSweepChoices(engine)]).then(([cols, sweep]) => {
            const seed = cols.filter((c) => c.group === "stage")
                             .map((c) => c.name);
            // Only propose a sweep for settings this engine actually has.
            const grid = {};
            for (const it of sweep) {
                if (it.machine_answers && BENCH_START[it.name]) {
                    grid[it.name] = BENCH_START[it.name].slice();
                }
            }
            const text = proposedFromHandover(_handover, shape, seed,
                             Object.keys(grid).length ? grid : undefined);
            try { _task = JSON.parse(text); } catch (_) { _task = null; }
            return setEditorText(text).then(() => {
                if (_task) { renderStages(_task); renderNext(_task);
                             renderMachine(_task); refreshPickers(); }
            });
        }).catch(() => {
            const text = proposedFromHandover(_handover, shape, []);
            try { _task = JSON.parse(text); } catch (_) { _task = null; }
            setEditorText(text).then(() => { if (_task) renderStages(_task); });
        });
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

    /* STEP 1 — the folder's current state (`task-setup.md` § 8).
     *
     * OFFERED, NEVER TAKEN SILENTLY (`checkpointing.md` § 9): the tick is the
     * offer, and clearing it is a real answer.  Through the public API
     * (`projects.md` § 5), not a fetch of our own — the panel showing this
     * folder's history is refreshed by the same call.
     *
     * A refusal here STOPS the save.  The step exists so what you are about to
     * change can be brought back; writing anyway would silently spend the
     * safety net you asked for. */
    const projects0 = window.molbuilder && window.molbuilder.projects;
    const wantCkpt = $("ts-ckpt") && $("ts-ckpt").checked;
    if (wantCkpt && projects0 && projects0.checkpoint) {
        const note = ($("ts-ckpt-note") && $("ts-ckpt-note").value) || "";
        const st = await projects0.checkpoint.status(_dir).catch(() => null);
        if (st && st.ok && !st.initialised) {
            const started = await projects0.checkpoint
                .init(_dir, { engine: (_task && _task.engine
                                       && _task.engine.name) || undefined })
                .catch(() => null);
            if (!started || !started.ok) {
                setState("refuse", "No state was saved, so nothing was written",
                         "Could not start a history here: "
                         + ((started && started.error) || "unknown reason")
                         + ". Untick the box to write without one.");
                refreshSave();
                return;
            }
        }
        const kept = await projects0.checkpoint.saveState(_dir, note)
            .catch((e) => ({ ok: false, error: String(e && e.message || e) }));
        if (!kept || !kept.ok) {
            setState("refuse", "No state was saved, so nothing was written",
                     (kept && kept.error) || "the checkpoint failed");
            refreshSave();
            return;
        }
        // `changed: false` is honest, not a failure — nothing differed from
        // the state the folder already stands at.
    }

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

function start(projects) {

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
    const addSetting = () => {
        const sel = $("ts-add-setting");
        if (!sel || !sel.value) return;
        addPoint(sel.value, "1");       // a row starts as ONE point: a choice
        sel.value = "";
    };
    const goBtn = $("ts-add-setting-go");
    if (goBtn) goBtn.addEventListener("click", addSetting);
    const addBtn = $("ts-add-stage");
    if (addBtn) addBtn.addEventListener("click", addStage);
    const colBtn = $("ts-add-col-go");
    if (colBtn) colBtn.addEventListener("click", () => {
        const sel = $("ts-add-col");
        if (sel && sel.value) { addColumn(sel.value); sel.value = ""; }
    });
    const saveBtn = $("ts-save");
    if (saveBtn) saveBtn.addEventListener("click", save);
    refreshSave();
}

/* WAIT for the sidebar, never read it.
 *
 * `web/projects.md` § 1: *"A tab waits for it with
 * `runtime.whenReady("projects")` instead of polling."*  The sidebar is a
 * `type=module` script, so its deferred initialisation has NOT run at
 * DOMContentLoaded -- reading `window.molbuilder.projects` there finds
 * `undefined` and the page reported "the projects sidebar did not load" on
 * every load.  That is what the runtime registry exists to prevent, and
 * `structure-optimization/viewer.js` already waits this way.
 */
function boot() {
    const rt = window.molbuilder && window.molbuilder.runtime;
    if (rt && typeof rt.whenReady === "function") {
        rt.whenReady("projects").then(start).catch((e) => {
            setState("refuse", "The projects sidebar did not load",
                     "This page follows the sidebar's selected folder, so it "
                     + "cannot read anything without it. "
                     + (e && e.message ? e.message : ""));
        });
        return;
    }
    // No runtime at all is a page-assembly fault, not a timing one.
    setState("refuse", "The page did not assemble",
             "molbuilder-runtime.js must load before every other script.");
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot, { once: true });
} else {
    boot();
}
