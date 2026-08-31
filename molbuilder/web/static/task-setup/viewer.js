/* task-setup/viewer.js — the Task Setup tab's page controller.
 *
 * WHAT THIS TAB DOES:
 *   1. Follow the projects sidebar's selected DIRECTORY.
 *   2. Read that folder's `task.json`, or the `task.1st.json` hand-over a
 *      parameter tab left when there is no description yet.
 *   3. Show what the description says — its stages, and the machine settings
 *      you either chose or asked to have measured.
 *   4. Let you read and edit the file itself in the vendored CodeMirror.
 *   5. WRITE IT — POST /api/task-setup/save, which puts `task.json` into the
 *      folder.  An offered checkpoint runs first (`checkpointing.md` § 9):
 *      the tick is the offer and clearing it is a real answer.
 *   6. Read what the page cannot derive: the sweepable set
 *      (`/api/task-setup/sweepable`), the picker columns (`…/columns`),
 *      the tier presets (`…/presets`) and the folder's own template
 *      values (`…/template-values`); write through the save door and the
 *      launcher door (`…/launcher`).
 *
 * The contract is `docs/web/task-setup.md`; where this disagrees, that wins.
 *
 * THIS HEADER SAID *"IT WRITES NOTHING.  Save is disabled and says why"*
 * until 2026-08-17, while `refreshSave()` computed the button's state per
 * folder and the hint under it read *"Writes task.json into this folder."*
 * The claim was true when the page was read-only and was never revisited when
 * saving landed.  A header drifts in one direction — it keeps describing the
 * smaller, earlier page — so the claim to distrust is always the categorical
 * one: *it writes nothing*, *these are the only calls*.
 */

import { loadCodeMirror, modeFor } from "../lib/codemirror-load.js";
// § 5.2: the declared type decides what a cell's text means.
import { CELL_READERS } from "./cell-readers.js";

const TASK_JSON     = "task.json";
/* The hand-over a parameter tab leaves (`stages.md` § 6.5a).  Read ONLY when
 * there is no `task.json` -- a folder holding both is a save that did not
 * finish, and the description wins because it is the one that passed the
 * preflight. */
const TASK_HANDOVER = "task.1st.json";

/* Which items the MACHINE answers — DERIVED, never listed here.
 *
 * An item whose `resolver` is an allocation resolver may state the question
 * and never the answer (`engines/template.md` § 6.4), so on this page it can
 * only be a point to measure, never a choice.  The server already computes
 * exactly that (`/api/task-setup/sweepable` ships `machine_answers` per
 * item, from the item's own `allocation` flag) and this page already reads
 * that field in two other places.
 *
 * It was a hard-coded `new Set(["mpi_np", "omp_threads", "max_memory_mb"])`
 * until 2026-08-17 — a THIRD answer to a question the page already had two
 * answers to, and not even in the same vocabulary: the constant listed ITEM
 * names while `ALLOCATION_RESOLVERS` holds RESOLVER names, which collide on
 * `omp_threads` by coincidence.  Its own comment named the drift it invited:
 * *"a fourth one would show up as `chosen`, which reads wrong but breaks
 * nothing"* — that is a machine-answered value presented as a person's
 * choice, silently. */
/* WHICH ENGINE the hand-over is for.
 *
 * `engine` is an OBJECT in every artifact this system writes -- `{name: ...}`
 * in `task.1st.json` and in `task.json` alike -- so the name is `.engine.name`.
 * This read `.engine` and wrapped it in `String()`, which produced the literal
 * `"[object Object]"`, and everything downstream of it went quiet rather than
 * loud: the sweepable fetch answered `400 unknown engine`, the cached answer
 * became the empty list, and the machine card then reported the OPPOSITE of
 * the truth -- *"every sweepable setting is already listed"*, disabled, with
 * nine settings available and none listed.  `use_gpu` is one of the nine,
 * so the surface `task-setup.md` § 6.2 makes the ONE place a GPU is chosen was
 * inert on the hand-over path, which is the only path the UI offers.
 *
 * `setShape` twelve lines down had always read it correctly.  Two accessors
 * for one field is the shape of this bug; one of them is now gone. */
function _handoverEngine(over) {
    const from = (o) => (o && o.engine && o.engine.name) || "";
    return String(from(over) || from(_handover) || "siesta");
}

function machineAnswers(name) {
    return (_sweep || []).some(i => i.name === name && i.machine_answers);
}

/* The second line under a row's name — DERIVED, never a table here.
 *
 * The catalogue already carries each item's `label`, and the server already
 * ships it (`/api/task-setup/sweepable`).  Whether the scheduler answers it is
 * the `machine_answers` flag beside it.  So the note is those two facts read
 * together, and there is nothing to keep in step.
 *
 * It was a hand-typed `ROW_NOTE` map until 2026-08-17 — the LAST of the
 * copies of "the scheduler answers mpi_np", which that one fact had been
 * written in six times across the tree.  Its own comment said what it was:
 * *"a name the catalogue carries would be better."*
 */
function rowNote(name) {
    const it = (_sweep || []).find(i => i.name === name);
    if (!it) return "";
    return it.machine_answers
        ? `${it.label || name} — the scheduler answers`
        : `${it.label || name} — yours to choose`;
}

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
            // The cell follows the value SHAPE (user, 2026-08-20): an
            // enum or bool column edits through a dropdown of its legal
            // values -- the empty option IS the template state, labeled
            // with the value it means -- and every other column keeps the
            // free input.  One rule with the machine card's adder
            // (`legalValues`); a value's look never picks the widget.
            const legal = legalValues(col);
            let cell;
            if (legal) {
                cell = el("select", {
                    class: "ts-cell",
                    title: helpText(col),
                    "aria-label": name + " " + col,
                    "data-template": has ? null : "yes",
                });
                cell.appendChild(el("option", { value: "" },
                    "(" + (fallback || "template") + ")"));
                for (const v of legal) {
                    cell.appendChild(el("option", { value: String(v) },
                                        String(v)));
                }
                cell.value = has ? String(ov[col]) : "";
            } else {
                cell = el("input", {
                    class: "ts-cell", value: has ? String(ov[col]) : "",
                    placeholder: fallback || "template",
                    title: helpText(col),
                    "aria-label": name + " " + col,
                    "data-template": has ? null : "yes",
                });
            }
            cell.addEventListener("change", () => setCell(i, col, cell.value));
            tr.appendChild(el("td", { "data-template": has ? null : "yes" }, cell));
        }
        tbody.appendChild(tr);
    });

    const actions = $("ts-stage-actions");
    if (actions) actions.hidden = false;
    card.hidden = false;
}

/* ---------- what this machine would actually run ---------- */

/* The grid the axes above would produce, and which queues would take each
 * cell -- painted into the card that sets them, live (user, 2026-08-30:
 * "can't this list be just updated in the same card where the parameters
 * are set... this does not need to be a message with a window").
 *
 * THE BROWSER DOES NOT ENUMERATE IT.  `/api/task-setup/bench-grid` hands
 * the axes to `_bench_inputs` -- the one enumerator, the same one `prep`
 * runs -- and returns its report.  A grid computed here would be the
 * second, drifting decider that the whole cross-out rule was rebuilt to
 * remove: the browser would say a cell is fine and `launch` would refuse
 * it, which is exactly the failure the person hit.
 *
 * THE AXES SENT ARE THE MODEL'S, NOT THE FILE'S, so the list tracks
 * typing rather than the last save. */
let _fitTimer = null;
let _fitSeq   = 0;
/* THE AXES THE ROWS WERE PAINTED FROM.  `renderMachine` takes the task as
 * an ARGUMENT and is called with the handover object in handover mode --
 * so reading the module's `_task` here described a different object than
 * the rows above, which is the two-sources bug in miniature.  The rows and
 * the list answer from one value or they can disagree. */
let _fitBench = {};

function scheduleFitRefresh(bench) {
    _fitBench = bench || {};
    if (_fitTimer) clearTimeout(_fitTimer);
    // Every keystroke repaints the rows; the server answer is worth one
    // request per pause, not one per character.
    _fitTimer = setTimeout(refreshFit, 300);
}

async function refreshFit() {
    const host = $("ts-machine-fit");
    if (!host) return;
    const bench = _fitBench || {};
    /* ONLY A REAL DESCRIPTION HAS A GRID.  In handover mode `task.json`
     * does not exist yet, and the door reads it -- so this would fire a
     * request that can only 400.  "Empty" likewise. */
    if (_mode !== "description") { host.hidden = true; return; }
    if (!_dir || !Object.keys(bench).length) { host.hidden = true; return; }

    /* STALE ANSWERS ARE DROPPED.  Typing outruns the network, and an
     * earlier reply landing after a later one would leave the card showing
     * a grid for axes that no longer exist. */
    const seq = ++_fitSeq;
    let body;
    try {
        const r = await fetch("/api/task-setup/bench-grid", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ dest: _dir, target: _machine || "",
                                   bench: bench }),
        });
        body = await r.json();
    } catch (e) {
        body = null;
    }
    if (seq !== _fitSeq) return;

    /* A LIST THAT CANNOT LOAD HIDES ITSELF; it never breaks the card.
     * The rows above are the substance and are usable without this --
     * the same rule the label note follows, for the same reason (a card
     * that vanishes gets reported as the feature being gone). */
    if (!body || !body.ok || !Array.isArray(body.cells)) {
        host.hidden = true;
        return;
    }
    paintFit(host, body);
}

function paintFit(host, body) {
    const cells   = body.cells;
    const kept    = cells.filter((c) => !(c.why || []).length);
    const crossed = cells.filter((c) => (c.why || []).length);
    host.textContent = "";
    host.hidden = false;

    host.appendChild(el("div", { class: "ts-fit-head" },
        /* "this machine" would be wrong: the queues these were checked
         * against are the TARGET's, and preparing for another machine is
         * an ordinary thing to do here. */
        cells.length + " combination(s) \u2014 " + kept.length
        + " fit a queue"
        + (crossed.length ? ", " + crossed.length + " do not" : "")));

    for (const c of kept) {
        host.appendChild(el("div", { class: "ts-fit-row" },
            el("code", { class: "ts-fit-label" }, c.label),
            el("span", { class: "ts-fit-shape" }, c.shape),
            el("span", { class: "ts-fit-where" },
               (c.fits || []).slice(0, 4).join(", ")
               + ((c.fits || []).length > 4 ? " \u2026" : ""))));
    }
    for (const c of crossed) {
        host.appendChild(el("div", { class: "ts-fit-row ts-fit-row--out" },
            el("code", { class: "ts-fit-label" }, c.label),
            el("span", { class: "ts-fit-shape" }, c.shape),
            // THE NUMBERS, not a verdict word (scheduler.md R4): the reason
            // says what to change, which is the whole point of showing the
            // struck rows rather than silently dropping them.
            el("span", { class: "ts-fit-why" }, (c.why || [])[0] || "")));
    }
    /* A `note` means the grid resolved this far and then stopped -- the
     * rows above are still the honest answer, and the reason belongs with
     * them rather than swallowed.  Its own words, never a paraphrase. */
    if (body.note) {
        host.appendChild(el("p", { class: "status warn" }, body.note));
    } else if (!kept.length) {
        host.appendChild(el("p", { class: "status warn" },
            "No queue on the chosen machine takes any of these \u2014 "
            + "adjust the points above, or choose a different machine."));
    }
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
        const kind = machineAnswers(name)
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

        // The adder follows the value SHAPE (user, 2026-08-20): a bool or
        // an enum offers exactly the legal values not already listed -- a
        // dropdown, never a number box -- and a numeric axis keeps the
        // free input it always had.
        const legal = legalValues(name);
        let add;
        if (legal) {
            const left = legal.filter(
                (v) => !pts.some((p) => String(p) === String(v)));
            add = el("select", { class: "ts-pt-add",
                                 "aria-label": "add a point to " + name });
            add.appendChild(el("option", { value: "" },
                               left.length ? "+ add" : "all values listed"));
            for (const v of left) {
                add.appendChild(el("option", { value: String(v) },
                                   String(v)));
            }
            add.disabled = !left.length;
            add.addEventListener("change", () => {
                if (add.value) addPoint(name, add.value);
            });
        } else {
            add = el("input", { class: "ts-pt-add", placeholder: "+ add",
                                "aria-label": "add a point to " + name });
            add.addEventListener("change",
                () => { addPoint(name, add.value); add.value = ""; });
        }

        const dropRow = el("button", { type: "button", class: "ts-rowbtn ts-rowbtn-drop",
                                       title: "Stop measuring " + name }, "\u00d7");
        dropRow.addEventListener("click", () => removeSetting(name));

        // A VALUE axis with several points SWEEPS (generator.md § 4.3a,
        // built 2026-08-21 -- it was refused by name until then): the
        // points multiply the machine grid and every trial's deck
        // carries its coordinate.  use_gpu decides which trials run
        // on the GPU at all; submission groups trials by their exact
        // resource ask (one exact-fit job per shelf), so CPU trials
        // never hold a device.
        const _isMachine = machineAnswers(name);
        const _tooMany = (!_isMachine && pts.length > 1)
            ? el("small", { class: "ts-row-note" },
                 name === "use_gpu"
                   ? (pts.length + " points \u2014 the cpu-vs-gpu axis: "
                      + "the grid enumerates per flag, and submit bench "
                      + "groups trials by their exact resource ask \u2014 "
                      + "CPU trials never hold a GPU.")
                   : (pts.length + " points sweep as a value axis \u2014 "
                      + "the machine grid multiplies per value; the "
                      + "winning combination lands in run-config.toml."))
            : null;
        host.appendChild(el("div", { class: "ts-row", "data-kind": kind },
            el("div", { class: "ts-row-name", title: helpText(name) }, name,
                el("small", null, rowNote(name)), _tooMany),
            el("div", { class: "ts-points" }, ...chips, add),
            el("div", { class: "ts-verdict" }, verdict, dropRow)));
    }
    const acts = $("ts-machine-actions");
    if (acts) acts.hidden = false;
    // From the task THIS render was given -- see `_fitBench`.
    scheduleFitRefresh(bench);

    /* SAY WHEN THE LABELS ARE MISSING.  The rows above are the real setting
     * names and are usable as they are; the server-side vocabulary that turns
     * `mpi_np` into "MPI ranks (np)" is a separate fetch, and when it fails
     * the honest thing is a card that works and admits what it lacks.  The
     * alternative -- what happened on 2026-08-23, when the server was
     * restarting under a loaded page -- is a card that never appears at all
     * and a user who reports the feature as gone. */
    let note = $("ts-machine-labels-note");
    if (_vocabFailed) {
        if (!note) {
            note = el("p", { class: "status warn",
                             id: "ts-machine-labels-note" }, "");
            card.insertBefore(note, $("ts-machine-rows"));
        }
        note.textContent =
            "Showing raw setting names — the label list did not load "
            + "(reload the page once the server is back).";
        note.hidden = false;
    } else if (note) {
        note.hidden = true;
    }
    card.hidden = false;
}

/* ---------- the editor ---------- */

/* SINGLE-FLIGHT, because this is async and the guard used to be on the
 * RESULT.  Two callers arriving before the first finished both saw `_cm ===
 * null`, both awaited the loader, and both constructed an editor into
 * `#ts-editor` -- three of them stacked in the live page, the top two showing
 * text from before the shape was chosen.  Every edit went to the newest
 * instance and every reading came from the oldest, so the whole panel looked
 * dead: rows would not add, points would not drop, and the JSON on screen
 * never moved.  Caching the PROMISE is what makes the guard hold across the
 * await. */
let _cmBooting = null;

function ensureEditor() {
    if (_cm) return Promise.resolve(_cm);
    if (_cmBooting) return _cmBooting;
    _cmBooting = _bootEditor();
    return _cmBooting;
}

async function _bootEditor() {
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
    /* The resolved-config view is per FOLDER (the bundle's own
     * .molbuilder.json is one of the scopes), so it repaints with one.
     * Fire-and-forget: it is a read of what a prep WOULD do, and nothing
     * below waits on it. */
    setTimeout(loadResolved, 0);
    _dir = dir;
    /* EVERY per-folder fact resets before the branch (U6 close): the
     * hand-over and empty branches never wrote _task/_shape, so a
     * SIESTA description opened first leaked into the next folder --
     * setShape(_shape) re-fired on the STALE _task, syncFromModel()
     * overwrote the editor with the previous folder's task.json, and
     * Save was enabled over the wrong calculation. */
    _task = null;
    _shape = "";
    _handover = null;
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
    let names = new Set();
    try {
        const listing = await projects.listDir(dir);
        const entries = (listing && listing.entries) || [];
        names = new Set(entries.map((e) => e && e.name));
        const tmpl = entries.find((e) => e && typeof e.name === "string"
                                      && e.name.endsWith(".template.toml"));
        if (tmpl) templateName = tmpl.name;
    } catch (_) { /* listing is a nicety here */ }

    markFile("ts-f-task", !!taskText, "already here — saving would update it");
    markFile("ts-f-tmpl", !!templateName, "already here — saving would update it");
    if (templateName) $("ts-f-tmpl-name").textContent = templateName;

    /* THE STRUCTURE, by the name the description itself gives it.  Globbing
     * for a `.xyz` would answer a different question — "is there a geometry
     * here" rather than "is the one this calculation names here" — and those
     * differ in exactly the case worth showing: a folder that holds someone
     * else's structure and not its own.  `prep` refuses on that, late; this
     * says it on the page where it can still be fixed. */
    let ref = {};
    let docKind = "optimization";
    try {
        const doc = JSON.parse(taskText || overText || "{}");
        ref = (doc && doc.structure) || {};
        docKind = (doc && doc.calculation) || "optimization";
    } catch (_) { /* a file that does not parse is refused further down */ }
    /* THE COMPOSITE'S FOLDER IS task.json ALONE (transport-design.md
     * 4.1): no template, no structure pair -- the contract and the
     * geometry arrive from the citation at prep.  Listing "<label>.xyz
     * ... not there yet" would be the page claiming files are missing
     * that this kind never writes. */
    for (const id of ["ts-f-tmpl", "ts-f-struct", "ts-f-side"]) {
        const el = $(id);
        const li = el && el.closest("li");
        if (li) li.hidden = (docKind === "transport");
    }
    if (docKind === "transport" && taskText) {
        markFile("ts-f-task", true,
                 "already here — the WHOLE description (the composite "
                 + "has no template; everything else derives from the "
                 + "citation at prep)");
    }
    const structName = String(ref.source || "").split("/").pop();
    const sideName = structName
        ? structName.replace(/\.[^.]+$/, "") + ".molstruct.json" : "";
    if (structName) $("ts-f-struct-name").textContent = structName;
    if (sideName)   $("ts-f-side-name").textContent = sideName;
    markFile("ts-f-struct", structName && names.has(structName),
             "here — " + (ref.atoms || "?") + " atoms");
    markFile("ts-f-side", sideName && names.has(sideName),
             "here — the cell and the labels");

    // A hand-over: the parameters arrived, the description has not been
    // finished.  Show the file so it can be read, and say what is missing.
    if (!taskText && overText) {
        let over = null;
        try { over = JSON.parse(overText); } catch (_) { /* shown raw below */ }
        const awaiting = "Still needed: "
            + ((over && Array.isArray(over.awaiting) && over.awaiting.length)
               ? over.awaiting.join(" and ") : "shape and stages") + ".";
        _mode = "handover"; _handover = over;
        renderCameOver(over);
        $("ts-shape-card").hidden = false;
        setShape(_shape);                       // re-assert / repaint the choice
        setState("handover",
                 "Handed over — not a description yet",
                 `${TASK_HANDOVER} is here, carrying the parameters. `
                 + `${awaiting} Saving writes task.json and removes `
                 + `this file.`);
        $("ts-stages-card").hidden = true;
        { const c = $("ts-notify-card"); if (c) c.hidden = true; }
        // The machine rows are labelled from the SERVER's answer, so it has to
        // be here before they paint.  `loadSweepChoices` is memoised, so this
        // costs one fetch per engine per page-load, and nothing on repeat.
        await loadSweepChoices(_handoverEngine(over));
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
                 + "run `molbuilder jobset init`.");
        $("ts-stages-card").hidden = true;
        { const c = $("ts-notify-card"); if (c) c.hidden = true; }
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
        { const c = $("ts-notify-card"); if (c) c.hidden = true; }
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
    // THE CATALOGUE META COMES FIRST, and it is AWAITED (2026-08-24).
    // `renderStages` asks `legalValues()` per cell to decide whether the
    // column edits through a DROPDOWN of its legal values or a free input
    // -- and `_meta` was still empty here, because the only two fills ran
    // after it (`loadSweepChoices` on the next line, `refreshPickers`
    // four lines down, unawaited, which also CLEARS `_meta` first).  So
    // every enum and bool rendered as a text box and had to be typed by
    // hand.  Rendering a widget from data that has not arrived is the
    // defect; awaiting the data is the fix.
    await refreshPickers();
    // Through the ONE accessor -- `String({name})` is "[object Object]",
    // which 400s the sweepable fetch and sticks an empty cache for the
    // whole page-load (2026-08-21 review, E-A1).
    await loadSweepChoices(_handoverEngine(task));
    renderStages(task);
    renderNext(task);
    renderMachine(task);
    // The card reflects the DESCRIPTION on open, so reopening a folder
    // shows what it already asks for instead of an empty card.
    readAsksFromTask(task);
    readNotifyFromTask(task);
    // Shown once there IS a description: the card writes into it, so with
    // nothing open there is nothing for a tick to land in.
    { const c = $("ts-notify-card"); if (c) c.hidden = false; }
    renderQueues();
    await setEditorText(taskText);
}

/* ---------- the table edits the description ---------- */

/** Push the model into the buffer and repaint. */
async function syncFromModel() {
    if (!_task) return;
    await setEditorText(JSON.stringify(_task, null, 2) + "\n");
    renderStages(_task);
    // THE MACHINE ROWS TOO.  Every bench verb -- addPoint, removePoint,
    // removeSetting -- ends here, and this function re-rendered everything
    // except the card those verbs act on.  So a point was added to the model,
    // written into the JSON on screen, and the row it belonged to went on
    // showing the old chips: the panel looked inert while the file underneath
    // it was changing.
    await loadSweepChoices(_handoverEngine(_task || {}));
    renderMachine(_task);
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
        syncFromModel();
        return;
    }
    /* THE DECLARED TYPE DECIDES, and nothing else may.  A value's LOOK
     * never picks its type -- that is the rule `legalValues` keeps for the
     * widget ("inventing a widget from the value's look is how `use_gpu`
     * became a number box"), and it is the same rule here.  So a column
     * whose type the catalogue did not give us is stored AS TYPED and
     * named by the save door, rather than guessed at as a number. */
    const read = CELL_READERS[(_meta[col] || {}).type];
    const v = read ? read(text) : undefined;
    ov[col] = (v === undefined) ? text : v;
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
let _colsKey = null;
/* name -> the catalogue's own item, so the table can show what a parameter
 * DEFAULTS to and what it is for.  The catalogue already carries `default`,
 * `unit` and `help`; a second copy here would be the drift the one-source rule
 * exists to prevent, so this is a lookup into what the schema returned. */
const _meta = Object.create(null);
let _sweep = null;      // the ones a benchmark may sweep
//: Whether ANY server-side vocabulary failed to load this page-load.
//: One fact, one flag: the card's note says the names are raw, and it
//: does not matter which of the four lookups was the one that failed.
let _vocabFailed = false;
let _sweepKey = null;

/** Every parameter that may become a column.
 *
 * `stages.md` § 6.2: anything the description is ALLOWED TO HOLD may be a
 * column; the settings the machine answers may not, because the description
 * may not hold them at all.  The server answers that from the catalogue —
 * there is no list here and none there.
 *
 * It read `/api/build/schema` until 2026-08-18, which is the PARAMETER FORM's
 * schema and filters the whole `staging` group out on purpose.  Filtering a
 * panel and limiting a table are different jobs, and borrowing the first
 * answer for the second cost this table `restart` — the field that decides
 * whether a ladder is a ladder — so a ladder built here ran every stage clean.
 */
/** ONE guarded read of a server-side vocabulary (2026-08-23).
 *
 * Four functions here memoise a fetch of something the server names --
 * columns, sweepable settings, tier presets, template values.  Three of the
 * four wrote the fetch out by hand and NONE of those three caught anything,
 * so a request that failed rejected out through `loadFolder` and stranded
 * whatever card sat behind it.  The fourth, `loadTemplateValues`, had the
 * try/catch -- the right answer was already in this file and the copies did
 * not get it, which is what copied code does.
 *
 * That is how the bench card "disappeared" on 2026-08-23: the server was
 * restarting under a loaded page, the label lookup failed, and every card
 * above it painted while the one behind it never got its turn.
 *
 * Returns `{ok, body}`.  A failure is the CALLER'S to degrade around -- these
 * are vocabularies, and a page that has the substance can show it with the
 * raw names.
 */
async function fetchVocabulary(url, what) {
    try {
        const r = await fetch(url);
        if (!r.ok) throw new Error("HTTP " + r.status);
        return { ok: true, body: await r.json() };
    } catch (e) {
        _vocabFailed = true;
        if (window.console) {
            console.warn("[task-setup] " + what + " unavailable:", e);
        }
        return { ok: false, body: null };
    }
}


async function loadColumnChoices(engine) {
    // The folder's KIND narrows the columns (template.md § 6.3's sibling
    // rule) -- a vibration description's picker offers the vibration
    // items beside the shared ones; an optimization's never sees them.
    // The cache is keyed by (engine, kind): a bare `if (_cols)` served an
    // optimization folder's columns to the vibration folder opened next.
    const kind = (_task && _task.calculation)
        || (_handover && _handover.calculation) || "optimization";
    const key = (engine || "siesta") + ":" + kind;
    if (_cols && _colsKey === key) {
        // REFILL `_meta` EVEN ON THE CACHED PATH.  `refreshPickers` clears
        // `_meta` before calling the loaders, so an early return here left
        // it empty -- and `legalValues()` reads it to decide whether a cell
        // is a DROPDOWN or a text box.  Every enum and bool in the stage
        // table therefore had to be typed by hand from the second load on
        // (reported 2026-08-24).  The cache is about not re-FETCHING; it
        // was never meant to skip publishing what was fetched.
        _fillMeta(_cols);
        return _cols;
    }
    _colsKey = key;
    const got = await fetchVocabulary(
        "/api/task-setup/columns?engine="
        + encodeURIComponent(engine || "siesta")
        + "&calculation=" + encodeURIComponent(kind), "column list");
    if (!got.ok) { _cols = []; return _cols; }
    const j = got.body;
    _cols = (j && j.items) || [];
    _fillMeta(_cols);
    return _cols;
}

/** Publish a vocabulary's items into `_meta` -- the ONE place a cell's
 *  widget question (`legalValues`) gets its answer from.  A function, not
 *  a loop at each call site, because the two loaders each have a cached
 *  path and a fetching one: four places to remember, and the two cached
 *  ones were forgotten. */
function _fillMeta(items) {
    for (const it of (items || [])) {
        if (it && it.name) _meta[it.name] = it;
    }
}

/* A STARTING SWEEP for the settings the machine answers.
 *
 * They can only ever be points to try -- a description may never carry a value
 * for one (`template.md` § 6.4) -- so an empty bench leaves the card with
 * nothing in it and the user typing point lists from scratch.  These are the
 * shipped starting points, and `stages.md` § 6.8's rule is what makes them
 * safe to propose: a MACHINE-ANSWERED `bench` entry records points to try
 * and never an answer (stages.md § 6.8 -- the one-point-is-a-pin rule is
 * for the non-machine entries only), so a proposed grid costs nothing but
 * a measurement, and every row can be edited or dropped.
 *
 * Powers of two for ranks because that is how the block distributes
 * (`tuning.md` § 2.11); 1 and 2 threads because hybrid runs are the comparison
 * worth making first. */
const BENCH_START = { mpi_np: [4, 8, 16], omp_threads: [1, 2] };

/** The sweepable set — `execution` category only (`stages.md` § 6.8).
 *  A separate read because the FORM filters `staging` out, and those are
 *  exactly the knobs a benchmark measures. */
async function loadSweepChoices(engine) {
    /* Keyed by ENGINE, like `_cols` (R2-1): a bare `if (_sweep)` served
     * the first folder's engine to every folder opened after it -- a
     * PySCF description got SIESTA's machine rows. */
    const key = engine || "siesta";
    if (_sweep && _sweepKey === key) {
        _fillSweepMeta(_sweep);      // same reason as the column cache above
        return _sweep;
    }
    _sweepKey = key;
    /* A LABEL LOOKUP MUST NOT BE ABLE TO STRAND THE PAGE (2026-08-23).
     *
     * This is an ENRICHMENT: it turns `mpi_np` into "MPI ranks (np)".  It
     * used to be an unguarded `await fetch` with `renderMachine` behind it as
     * the last step of `loadFolder`, so a slow or failed request left every
     * card above it painted and the bench card simply absent -- no error, no
     * empty state, nothing to retry.  The user's report was "the bench setup
     * is gone", and the page looked completely normal.
     *
     * Now it degrades: rows paint with their raw names and the card says the
     * labels are missing.  A surface that cannot get its nicety shows what it
     * has; only a surface that cannot get its SUBSTANCE may refuse. */
    const got = await fetchVocabulary(
        "/api/task-setup/sweepable?engine=" + encodeURIComponent(key),
        "sweepable settings");
    if (!got.ok) { _sweep = []; return _sweep; }
    const j = got.body;
    _sweep = (j && j.items) || [];
    _fillSweepMeta(_sweep);
    return _sweep;
}

/** Fold the sweepable items into `_meta` WITHOUT overwriting a column's
 *  richer record -- `staging` items are filtered out of the form schema,
 *  so this is the only place their note arrives.  Called from both the
 *  cached and the fetching path, for the reason `_fillMeta` records. */
function _fillSweepMeta(items) {
    for (const i of (items || [])) {
        if (i && i.name && !_meta[i.name]) {
            _meta[i.name] = { name: i.name, label: i.label, help: i.help,
                              type: i.type, choices: i.choices,
                              default: i.default };
        }
    }
}

/* The LEGAL VALUES of a bool/enum parameter, from the catalogue meta --
 * null for a free-typed (numeric) one.  The one place the widget question
 * is answered (user, 2026-08-20): a value's look must never pick the
 * widget, which is how use_gpu became a number box. */
function legalValues(name) {
    const m = _meta[name] || {};
    if (m.type === "bool") return [true, false];
    if (m.type === "enum" && Array.isArray(m.choices)) return m.choices;
    return null;
}

/* The value IN FORCE for a parameter -- what a new machine-card row is
 * born holding: this folder's template answer first, else the catalogue
 * default, else 1 (the numeric axes' old birth value).  RAW value --
 * `defaultText` below is its DISPLAY sibling (rendered, with the unit);
 * conflating the two puts "300 Ry" into a point list. */
function valueInForce(name) {
    if (name in _tmpl.values) return _tmpl.values[name];
    const m = _meta[name] || {};
    if (m.default !== undefined && m.default !== null) return m.default;
    return 1;
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
    /* _meta is rebuilt from this engine's answers: its neighbours are
     * keyed (_colsKey/_sweepKey/_presetsKey) but _meta accreted across
     * engines -- a sweep-only item's record was written once and never
     * refreshed, and the previous engine's names backed legalValues /
     * valueInForce / helpText forever. */
    for (const k of Object.keys(_meta)) delete _meta[k];
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
let _presetsKey = null;

async function loadPresets(engine) {
    /* Keyed by ENGINE, like `_cols` (R2-1): a stale cache here APPLIED
     * SIESTA tier values into a PySCF description opened second. */
    const key = engine || "siesta";
    if (_presets && _presetsKey === key) return _presets;
    _presetsKey = key;
    const got = await fetchVocabulary(
        "/api/task-setup/presets?engine=" + encodeURIComponent(key),
        "tier presets");
    if (!got.ok) { _presets = []; return _presets; }
    const j = got.body;
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
    // The prep widgets are rebuilt with the card, so the list of them is
    // emptied with the card too -- otherwise it accretes one page's worth
    // per folder opened, and `_syncPrepButtons` walks buttons nobody can
    // see.  Owned here because this is the one place they are created.
    _PREP_WIDGETS.length = 0;
    // ORDINALS COME FROM THE FULL LADDER (`stages.md` § 6.5: seq is
    // assigned once and never renumbered), so a disabled stage still
    // occupies its number -- the enabled-filtered index mis-named the
    // previous attempt's directory whenever one was skipped
    // (2026-08-21 review, E-T4).
    const ladder = (task && task.stages) || [];
    const enabled = [];
    ladder.forEach((st, full) => {
        if (st && st.enabled !== false) enabled.push({ st: st, full: full });
    });
    if (!enabled.length) { card.hidden = true; return; }

    // The bench lane, when the description PLANS a measurement
    // (`task.bench` non-empty -- stages.md § 6.8): the whole sequence,
    // taught once with the first stage as the example.  summarize
    // writes bench-result.json (the record) AND run-config.toml (the
    // editable proposal); `prep run` then applies what the accepted
    // proposal says -- template < declaration < run-config < flags.
    /* `--bundle <path from the projects root>` for every command this
     * tab teaches.  Naming the bundle is what lets the line be pasted
     * from anywhere; the sidebar already knows the folder, so the user
     * never types it.
     *
     * Built from what already exists -- the sidebar's `getCurrentDir` +
     * `getProjectsRoot`, and `path.relativeFromDir` to subtract one from
     * the other.  That last one had had NO caller since `d1c8a871` took
     * deck-rendering out of the browser; this is the job it was written
     * for.  Empty when the folder is not under the tree, where the verb's
     * own refusal says more than a truncated command could. */
    /* `--target <machine>` for every command taught, once a machine is
     * chosen and it is not this one.  Preparing for the box you are on is
     * the case the flag is not for, so it is omitted there -- the same
     * shape as `--bundle`, which is omitted when the cwd already is it. */
    function _targetArg() {
        if (!_machine || _machine === "(this machine)") return "";
        return " --target " + _machine;
    }

    function _bundleArg() {
        const mb = window.molbuilder || {};
        const proj = mb.projects, pathUtil = mb.path;
        if (!proj || !pathUtil || !pathUtil.relativeFromDir) return "";
        const dir  = proj.getCurrentDir && proj.getCurrentDir();
        const root = proj.getProjectsRoot && proj.getProjectsRoot();
        if (!dir || !root) return "";
        const rel = pathUtil.relativeFromDir(dir, root);
        if (!rel || rel === "." || rel.indexOf("..") === 0) return "";
        return " --bundle " + rel;
    }

    /* ONE BLOCK PER ENABLED STAGE, and both things you can do with it
     * (`task-setup.md` § 11).  A stage is either something to MEASURE or
     * something to RUN, and which one is a decision only the user has.
     *
     * The bench half used to be a single block hardwired to
     * `enabled[0]` -- a guess dressed as an answer.  The bench axes are
     * declared once for the calculation, so ANY enabled stage can be
     * measured; which is worth measuring is a judgement, and the page
     * hints rather than choosing. */
    const benchKeys = Object.keys((task && task.bench) || {});

    enabled.forEach((e, i) => {
        const name = e.st.name || "";
        const ov = e.st.overrides || {};
        // `continue` carries from the stage before it — and `prep` is TOLD
        // which attempt, never left to guess (`project-layout.md` § 1.6).
        let from = "";
        if (i > 0 && String(ov.restart || "") === "continue") {
            const prev = enabled[i - 1];
            const token = String(prev.full + 1).padStart(2, "0")
                        + "_" + (prev.st.name || "");
            from = " --from " + token + "/run-0";
        }
        const runs = _runs[name];
        const block = el("div", { class: "ts-next-step" },
            el("div", { class: "ts-next-stage" }, name,
               (runs ? el("span", { class: "ts-ran" }, runs + "\u00d7 run") : null)));

        /* MEASURE first, when the description declares axes to measure.
         * The order is shown because it is load-bearing: `summarize`
         * writes run-config.toml and `prep run` APPLIES it to any
         * allocation field you did not state, so skipping the middle step
         * does not fail -- it quietly prepares a run with no verdict
         * behind it. */
        if (benchKeys.length) {
            block.appendChild(el("p", { class: "hint" },
                "Measure it \u2014 varying " + benchKeys.join(", ")
                + ". Worth doing on the cheapest rung that still has the "
                + "expensive stage's shape; the verdict carries to the run."));
            block.appendChild(el("pre", { class: "ts-cmd" },
                "molbuilder jobset prep bench " + name + _bundleArg() + _targetArg() + "\n"
                + "molbuilder jobset launch bench " + name + _bundleArg()
                + "      # one job per resource shelf; wait for the queue\n"
                + "molbuilder jobset summarize bench " + name + _bundleArg()
                + "   # writes bench-result.json + run-config.toml"));
            block.appendChild(prepButton("bench", name));
        }

        block.appendChild(el("p", { class: "hint" },
            benchKeys.length
                ? "Run it \u2014 applies run-config.toml for anything you "
                  + "did not state. Add --np / --omp / --time to override "
                  + "the measured verdict."
                : "Run it \u2014 the wrapper sizes the launch on the "
                  + "machine it lands on. Add --np / --omp / --time to say "
                  + "otherwise."));
        block.appendChild(el("pre", { class: "ts-cmd" },
            // The bundle is NAMED, from the projects root, so the line
            // works from wherever the user is standing
            // (job-contracts.md 2.5b).
            "molbuilder jobset prep run " + name + from + _bundleArg() + _targetArg() + "\n"
            + "molbuilder jobset launch run " + name + _bundleArg()));
        // `--from` is deliberately NOT offered by the button: which run you
        // continue from is a scientific choice the CLI makes you say out
        // loud (`project-layout.md` § 1.6), and a button would have to pick
        // a default.  The command above still shows it when it applies.
        if (!from) block.appendChild(prepButton("run", name));
        host.appendChild(block);
    });

    card.hidden = false;
}

//: Every prep widget on the page, so the machine choice can reach them.
//: Declared ABOVE its users: a `const` is hoisted but not initialised, so
//: a push from `prepButton` before this line would throw.
const _PREP_WIDGETS = [];

/** A "Prep this here" button for one stage.
 *
 * **Prep, never launch** (user, 2026-08-24).  `prep` writes files into the
 * calculation and can be run again; `launch` spends a queue slot and
 * refuses batch submission by design.  The two differ in what they cost to
 * get wrong, so only the cheap one gets a button.
 *
 * **Nothing is prepped unseen.**  The first click asks the server what it
 * WOULD do -- which stage, which machine, what the description asks the
 * scheduler for -- and shows it; the second click runs it.  The same rule
 * the launch door keeps (`submission.md` S4), for the same reason.
 */
function prepButton(kind, stage) {
    const wrap = el("div", { class: "ts-prep" });
    const btn = el("button", { type: "button", class: "btn" },
                   "Prep " + kind + " here");
    const say = el("div", { class: "ts-prep-say" });
    let planned = null;

    btn.addEventListener("click", async () => {
        // A MACHINE IS THE FIRST QUESTION, and this asked it by firing into
        // the server's refusal -- a paragraph about targets and probe
        // commands, which reads as a fault rather than as "you skipped a
        // step".  After any page load nothing is chosen (`loadMachines`
        // ends at `setMachine("")` whenever more than one could be meant),
        // so this was the ORDINARY path, not an edge case (reported
        // 2026-08-24).  Answered here, in the page's own words, and the
        // card that answers it is scrolled to.
        if (!_machine) {
            say.textContent = "Pick a machine first \u2014 \u201cWhich "
                + "machine is this for\u201d, just above.";
            say.setAttribute("data-state", "warn");
            const card = $("ts-target-card");
            if (card && card.scrollIntoView) {
                card.scrollIntoView({ behavior: "smooth", block: "center" });
            }
            return;
        }
        btn.disabled = true;
        try {
            if (!planned) {
                const r = await _prepCall(kind, stage, true);
                if (!r.ok) { say.textContent = r.error; say.setAttribute("data-state", "bad"); return; }
                planned = r;
                const bits = ["for " + r.machine];
                const axes = Object.keys(r.bench_axes || {});
                if (axes.length) bits.push("varying " + axes.join(", "));
                const a = r.allocation || {};
                bits.push(a.domain ? "queue " + a.domain : "NO QUEUE STATED");
                bits.push(a.mem ? "memory " + a.mem
                                : "NO MEMORY STATED \u2014 the scheduler's "
                                  + "own default decides");
                bits.push(a.time ? "time " + a.time : "no time stated");
                say.textContent = bits.join(" \u00b7 ") + ".  Click again to write it.";
                say.setAttribute("data-state", (a.mem && a.domain) ? "ok" : "warn");
                btn.textContent = "Write it";
                return;
            }
            say.textContent = "Preparing\u2026";
            say.setAttribute("data-state", "ok");
            const r = await _prepCall(kind, stage, false);
            if (!r.ok) {
                say.textContent = r.error;
                say.setAttribute("data-state", "bad");
                planned = null; btn.textContent = "Prep " + kind + " here";
                return;
            }
            say.textContent = "Prepared for " + r.machine + " \u2014 "
                + r.dirs.length + " director" + (r.dirs.length === 1 ? "y" : "ies")
                + ": " + r.dirs.slice(0, 3).join(", ")
                + (r.dirs.length > 3 ? ", \u2026" : "");
            say.setAttribute("data-state", "ok");
            planned = null; btn.textContent = "Prep " + kind + " here";
            // The folder now holds decks and wrappers it did not before --
            // the same announcement a restore makes, so every open view
            // re-reads rather than showing the folder as it was.
            const p = window.molbuilder && window.molbuilder.projects;
            if (p && typeof p.publishFolderChanged === "function") {
                p.publishFolderChanged(_dir);
            }
        } finally {
            btn.disabled = false;
        }
    });
    wrap.append(btn, say);
    _PREP_WIDGETS.push({ btn, say, kind });
    _syncPrepButtons();
    return wrap;
}

/** Keep the buttons honest about whether they can run yet.
 *
 *  Disabled-with-a-reason rather than enabled-and-refusing: the question
 *  ("which machine") is answered in a card directly above, and a control
 *  that looks ready but is not is how a person ends up reading a server
 *  refusal to find out they missed a step.
 */
function _syncPrepButtons() {
    // NO CONNECTEDNESS TEST AT ALL.  Two attempts got this wrong: skipping
    // detached widgets meant each one missed its OWN first sync (it is
    // returned before the caller appends it), and pruning them dropped a
    // stage's bench widget the moment its run widget was made, because the
    // block holding both is not in the page until the stage finishes.  The
    // list belongs to `renderNext`, which empties it when it rebuilds --
    // one owner, no liveness guessing.
    for (const w of _PREP_WIDGETS) {
        const ready = !!_machine;
        w.btn.disabled = !ready;
        w.btn.title = ready
            ? "Runs prep for " + _machine
            : "Pick a machine first";
        if (!ready && !w.say.textContent) {
            w.say.textContent = "pick a machine above";
            w.say.setAttribute("data-state", "unset");
        }
        if (ready && w.say.getAttribute("data-state") === "unset") {
            w.say.textContent = "";
            w.say.removeAttribute("data-state");
        }
    }
}

async function _prepCall(kind, stage, plan) {
    const body = { dest: _dir, kind, stage, plan };
    // The local machine has a NAME, not just a label: the server maps
    // `(this machine)` to it, so sending the label is enough and the two
    // surfaces keep one vocabulary.
    if (_machine) body.target = _machine;
    try {
        const r = await fetch("/api/task-setup/prep", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        const j = await r.json();
        return (j && typeof j.ok === "boolean")
            ? j : { ok: false, error: "prep failed (" + r.status + ")" };
    } catch (e) {
        return { ok: false, error: String((e && e.message) || e) };
    }
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
    // THE KIND rides the hand-over (absent = optimization, the same
    // absent-is-a-state rule task.json uses).  A vibration hand-over
    // proposes the kind's own ladder -- ONE `freq` stage
    // (spectra-migration plan § 2: the relaxation is the deck's
    // precondition, not a rung) -- where an optimization proposes the
    // ordinary `coarse` start.
    const kind = (over && over.calculation) || "optimization";
    const stages = kind === "vibration"
        ? [{ name: "freq", enabled: true, overrides: {} }]
        : [{ name: "coarse", enabled: true, overrides: {} }];
    const out = {
        schema:    "molbuilder/task@1",
        engine:    (over && over.engine) || { name: "siesta" },
        shape:     shape,
        run:       { name: run.name || "", id: run.id || "",
                     created: run.created || "" },
        structure: (over && over.structure) || {},
        varies:    varies || [],
        stages:    stages,
        bench:     bench || undefined,
    };
    if (kind !== "optimization") out.calculation = kind;
    return JSON.stringify(out, null, 2) + "\n";
}

/* ---------- which machine this is prepared for ---------- */
/* The SAME component and the same rule as the shape chooser below: a
 * choice with no default when several answers are possible.  It is the
 * user's because only they know it -- `preparing-for-another-machine.md`
 * § 4 -- and the CLI refuses without it, so the tab asks rather than
 * inventing a second rule. */

let _machine = "";          // the chosen record's name, "" until chosen
let _machines = [];         // what /api/task-setup/machines answered

function setMachine(name) {
    _machine = name;
    for (const b of document.querySelectorAll("#ts-target-choice .opt")) {
        b.setAttribute("aria-pressed",
                       b.getAttribute("data-machine") === name ? "true" : "false");
    }
    const needs = $("ts-target-needs");
    if (needs) needs.hidden = !!name;
    // A machine's QUEUES are its own -- switching machines invalidates a
    // queue chosen under the old one, so the choice is cleared rather
    // than carried across (a name that means nothing here would default
    // the asks from a ceiling this machine never stated).
    _queue = "";
    renderQueues();
    _syncPrepButtons();      // the prep buttons wait on this answer
    loadResolved();          // the warning depends on WHICH machine
    const chosen = _machines.find((m) => m.name === name);
    const st = $("ts-target-state");
    if (!st) return;
    if (!chosen) { st.hidden = true; return; }
    /* An unreadable record is shown as a refusal rather than hidden: the
     * user picked it, so silence would leave them waiting for something
     * that cannot happen. */
    st.hidden = false;
    st.setAttribute("data-state", chosen.readable ? "loaded" : "refuse");
    const title = $("ts-target-state-title");
    const body = $("ts-target-state-body");
    if (title) title.textContent = chosen.readable
        ? "Prepared for " + chosen.name
        : "Cannot prepare for " + chosen.name;
    if (body) body.textContent = chosen.readable
        ? chosen.summary + (chosen.detected_at
            ? "\nmeasured " + chosen.detected_at : "")
        : chosen.summary;
}

/* ===================================================================== *
 *  The queue, and what it allows  (user, 2026-08-24)                    *
 *                                                                       *
 *  Three facts, one card: WHICH queue, how long, how much memory.  The  *
 *  queue's own probed ceilings are the defaults -- the most that queue  *
 *  allows -- because a person sizing a job wants to start from what is  *
 *  possible and come down.  NOTHING is invented: a queue that states no *
 *  ceiling leaves the field empty and says so, and an empty field means *
 *  the scheduler decides (submission.md S1).                            *
 * ===================================================================== */

let _queue = "";            // the chosen domain name, "" = none

/** Seconds -> the shortest spelling a person would type back. */
function _humanTime(sec) {
    if (!sec) return "";
    if (sec % 86400 === 0) return (sec / 86400) + "-00:00:00";
    if (sec % 3600 === 0) return (sec / 3600) + "h";
    if (sec % 60 === 0) return (sec / 60) + "m";
    return sec + "s";
}

/* ---- the record's one spelling -------------------------------------
 * `task.json` holds SLURM's spelling and nothing else (user, 2026-08-24:
 * *"your record should set unified time format while it is the UI that
 * can do some translation for human readability/input"*).  So this file
 * translates in BOTH directions and the file sees only one:
 *
 *   _slurmTime / _canonTime   -> what gets WRITTEN   ("0-04:00:00")
 *   _humanTime                -> what gets SHOWN in prose ("4h")
 *
 * Typing "4h" in the box still works -- that is the human edge, and
 * `_canonTime` is the door it goes through.  Mirrors `ask.slurm_time` /
 * `ask.canonical_time`; the parity test pins the two against each other.
 */

/** seconds -> "D-HH:MM:SS", the spelling sbatch takes and the file holds. */
function _slurmTime(sec) {
    const n = Math.max(0, Math.round(Number(sec) || 0));
    const d = Math.floor(n / 86400);
    const h = Math.floor((n % 86400) / 3600);
    const m = Math.floor((n % 3600) / 60);
    const s = n % 60;
    const p2 = (x) => String(x).padStart(2, "0");
    return d + "-" + p2(h) + ":" + p2(m) + ":" + p2(s);
}

/** GB -> "<n>G", or "<n>M" when not a whole GB.  Mirrors `ask.slurm_mem`. */
function _slurmMem(gb) {
    const v = Number(gb);
    if (!(v > 0)) return "0";
    const mb = Math.round(v * 1024);
    // A POSITIVE ASK NEVER ROUNDS TO ZERO -- SLURM reads `--mem=0` as ALL
    // the node's memory, so rounding a sliver down would turn the smallest
    // request into the largest one.  Mirrors `ask.slurm_mem`; the parity
    // test carries a sub-megabyte value precisely to hold the two together
    // here, which it did not until 2026-08-24.
    if (mb <= 0) return "1M";
    return (mb % 1024 === 0) ? (mb / 1024) + "G" : mb + "M";
}

/** Whatever the person typed -> the record's spelling.  An unparseable
 *  value is passed through untouched so the backend refuses it BY NAME
 *  ("allocation.time: ...") instead of this quietly writing a zero. */
function _canonTime(txt) {
    const s = String(txt || "").trim();
    if (!s) return "";
    const sec = _parseTime(s);
    return Number.isFinite(sec) && sec > 0 ? _slurmTime(sec) : s;
}

function _canonMem(txt) {
    const s = String(txt || "").trim();
    if (!s) return "";
    if (s === "0") return "0";          // SLURM's "all of the node's"
    const gb = _parseMem(s);
    return Number.isFinite(gb) && gb > 0 ? _slurmMem(gb) : s;
}

/** "4h" / "90m" / "2-00:00:00" / "45" -> seconds, or null. Mirrors
 *  `scheduler.quantities.parse_duration` plus SLURM's own D-HH:MM:SS,
 *  which is what the
 *  ceilings are spelled in. */
function _parseTime(txt) {
    const t = String(txt || "").trim().toLowerCase();
    if (!t) return null;
    let m = t.match(/^(?:(\d+)-)?(\d+):(\d{2})(?::(\d{2}))?$/);
    if (m) {
        return (+(m[1] || 0)) * 86400 + (+m[2]) * 3600
             + (+m[3]) * 60 + (+(m[4] || 0));
    }
    m = t.match(/^([0-9.]+)\s*([hms]?)$/);
    if (!m) return NaN;                       // says nothing usable
    const v = parseFloat(m[1]);
    if (!(v > 0)) return NaN;
    return Math.round(v * ({ h: 3600, m: 60, s: 1 }[m[2]] || 60));
}

/** "128G" / "0.5T" / "128" -> GB, or null.
 *  Mirrors `scheduler.quantities.parse_memory`. */
function _parseMem(txt) {
    const t = String(txt || "").trim().toUpperCase();
    if (!t) return null;
    const m = t.match(/^([0-9.]+)\s*([KMGT]?)B?$/);
    if (!m) return NaN;
    const v = parseFloat(m[1]);
    if (!(v > 0)) return NaN;
    return v * ({ T: 1024, G: 1, M: 1 / 1024, K: 1 / 1048576 }[m[2]] || 1);
}

function _queuesOf(machineName) {
    const m = _machines.find((x) => x.name === machineName);
    return (m && m.domains) || [];
}

/** Paint the queue buttons for the chosen machine. */
function renderQueues() {
    const card = $("ts-queue-card");
    const host = $("ts-queue-choice");
    const needs = $("ts-queue-needs");
    if (!card || !host) return;
    const qs = _queuesOf(_machine);
    card.hidden = false;
    host.innerHTML = "";
    if (!_machine) {
        needs.hidden = false;
        needs.textContent = "Pick a machine first";
        $("ts-queue-asks").hidden = true;
        return;
    }
    if (!qs.length) {
        /* A workstation, or a cluster nobody has probed: there is no menu
         * to choose from, and saying so is better than an empty row. */
        needs.hidden = false;
        needs.textContent = _machine + " states no queues \u2014 it runs "
            + "directly, so there is nothing to choose. No queue means no "
            + "wall, so a blank time is unlimited; memory still has a real "
            + "ceiling \u2014 this machine's own RAM.";
        $("ts-queue-asks").hidden = false;
        _queue = "";
        // A MACHINE WITH NO QUEUES STILL HAS A MEMORY CEILING (user,
        // 2026-08-24): its RAM.  The suggestion came only from a QUEUE, so
        // the one kind of machine that cannot have one got none at all --
        // while `mem_total_gb` sat measured in its own record.  Time is
        // genuinely different and correctly stays blank: no scheduler
        // means no wall to state.
        const me = _machines.find((x) => x.name === _machine);
        if (me && me.mem_total_gb) {
            _fillIfUnanswered($("ts-ask-mem"), _defaultMemMB(me.mem_total_gb));
        }
        paintAskNotes();
        return;
    }
    needs.hidden = true;
    for (const d of qs) {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "opt";
        b.setAttribute("data-queue", d.name);
        b.setAttribute("aria-pressed", d.name === _queue ? "true" : "false");
        const bits = [];
        if (d.max_time) bits.push(d.max_time);
        if (d.max_cores) bits.push(d.max_cores + " cores");
        if (d.max_mem_gb) bits.push(_fmtGB(d.max_mem_gb));
        if (d.gpu) bits.push("GPU");
        // The SAME shape the machine buttons use -- <b> is what the
        // chosen-tick hooks (style.css `.ts-choice .opt b::before`), so a
        // parallel markup here would silently lose the tick.
        const nm = document.createElement("b");
        nm.textContent = d.name;
        const wh = document.createElement("span");
        wh.textContent = bits.join(" \u00b7 ");
        b.append(nm, wh);
        b.addEventListener("click", () => setQueue(d.name));
        host.appendChild(b);
    }
    $("ts-queue-asks").hidden = false;
    paintAskNotes();
}

function _fmtGB(gb) {
    /* Never rounds UP: this renders a CEILING, and a ceiling shown larger
     * than it is invites an ask the queue will refuse. */
    if (gb >= 1024) return (Math.floor(gb / 102.4) / 10) + " TB";
    return Math.floor(gb) + " GB";
}

/** The memory a queue's ceiling should DEFAULT to, in exact MB.
 *
 * `MEM_HEADROOM` of the node total, floored (user, 2026-08-24).  Two
 * reasons, and the first is a bug this replaces: the default was
 * `Math.round(max_mem_gb) + "G"`, which on a 503.5 GB queue asked for
 * 504 GB -- MORE than the ceiling -- so every non-integral queue filled
 * itself with a value its own hint then called too large.  The second is
 * why the fix is not merely `floor`: `max_mem_gb` is the node's TOTAL, and
 * a job asking all of it is commonly unschedulable because the OS and
 * SLURM itself need some, so the headroom is what makes the default a
 * value that actually runs.
 *
 * MB, not GB, because that is SLURM's own unit here (`sinfo %m`) and
 * rounding to whole GB is what lost the 0.5 in the first place.
 */
const MEM_HEADROOM = 0.95;

function _defaultMemMB(gb) {
    return Math.floor(gb * 1024 * MEM_HEADROOM) + "M";
}

/** Choosing a queue FILLS the two asks with that queue's ceilings --
 *  its own measured limits, which is the most this job could ask there. */
function setQueue(name) {
    _queue = name;
    for (const b of document.querySelectorAll("#ts-queue-choice .opt")) {
        b.setAttribute("aria-pressed",
                       b.getAttribute("data-queue") === name ? "true" : "false");
    }
    const d = _queuesOf(_machine).find((x) => x.name === name);
    // A CEILING FILLS WHAT NOBODY ANSWERED -- it does not overwrite an
    // answer (user, 2026-08-24).  This assigned unconditionally, so
    // choosing a queue merely to CHECK a value against it destroyed the
    // value: a 256G loaded from the folder's own `task.json` became the
    // queue's 487372M, and a figure just typed by hand went the same way
    // on the next queue click.  A default is for an empty field; replacing
    // a stated one is data loss wearing a default's clothes.
    // Canonical, not "4h": this fills the FIELD, and the field is what
    // lands in `task.json`.  The human spelling lives on in the note
    // beneath it (paintAskNotes), which is prose and not a record.
    _fillIfUnanswered($("ts-ask-time"),
                      d && d.max_time_s ? _slurmTime(d.max_time_s) : "");
    _fillIfUnanswered($("ts-ask-mem"),
                      d && d.max_mem_gb ? _defaultMemMB(d.max_mem_gb) : "");
    paintAskNotes();
    applyAsksToDoc();
    refreshSave();
}

/** Write a queue's ceiling into a field ONLY if nothing has answered it.
 *
 *  "Answered" means typed by a person or loaded from `task.json`.  A value
 *  this function put there is not an answer -- it is a suggestion -- so the
 *  next queue may replace it, which is what makes the fields track the
 *  queue you are looking at until the moment you disagree with one.
 *
 *  Marked on the element rather than held beside it: the field IS the
 *  state, and a parallel record of what is in it is a second answer to the
 *  same question.
 */
function _fillIfUnanswered(el, suggested) {
    if (!el) return;
    const mine = el.dataset.mbAuto === "1";
    if (el.value && !mine) return;          // a person answered; leave it
    el.value = suggested;
    if (suggested) el.dataset.mbAuto = "1";
    else delete el.dataset.mbAuto;
}

/** Say, under each field, what the queue allows and whether this ask
 *  fits -- while changing it is still free. */
function paintAskNotes() {
    const d = _queuesOf(_machine).find((x) => x.name === _queue);
    // With no queue, the MACHINE's own RAM is the memory ceiling -- a
    // workstation's field otherwise read "no queue chosen" and checked the
    // ask against nothing at all.
    const _me = _machines.find((x) => x.name === _machine);
    const _nodeMem = (!d && _me && !(_me.domains || []).length)
        ? _me.mem_total_gb : null;
    const pairs = [
        ["ts-ask-time", "ts-ask-time-note", _parseTime,
         d && d.max_time_s, (v) => _humanTime(Math.round(v)),
         "this queue states no time ceiling"],
        ["ts-ask-mem", "ts-ask-mem-note", _parseMem,
         (d && d.max_mem_gb) || _nodeMem, (v) => _fmtGB(v),
         "this queue states no memory ceiling"],
    ];
    for (const [inId, noteId, parse, cap, fmt, noCap] of pairs) {
        const el = $(inId);
        const note = $(noteId);
        if (!el || !note) continue;
        const raw = el.value;
        const val = parse(raw);
        let state = "ok";
        let msg;
        if (Number.isNaN(val)) {
            state = "bad";
            msg = "not a value I can read";
        } else if (val === null) {
            state = "unset";
            msg = "left blank \u2014 the scheduler's own default decides";
        } else if (cap && val > cap) {
            state = "bad";
            msg = "more than " + _queue + " allows (" + fmt(cap) + ")";
        } else if (cap) {
            msg = "allowed here: up to " + fmt(cap);
        } else {
            msg = d ? noCap : "no queue chosen";
        }
        note.textContent = msg;
        note.setAttribute("data-state", state);
        el.setAttribute("aria-invalid", state === "bad" ? "true" : "false");
    }
}

/** The three asks as `task.json`'s `allocation` carries them. */
function askValues() {
    const t = ($("ts-ask-time") || {}).value || "";
    const m = ($("ts-ask-mem") || {}).value || "";
    const out = {};
    if (_queue) out.domain = _queue;
    // THE RECORD GETS ONE SPELLING.  The box accepts "4h" because that is
    // the human edge; the file never sees it.  Until 2026-08-24 this wrote
    // the box verbatim, so the browser's own "4h" reached `sbatch` as
    // `-t 4h` and SLURM refused the tool's written value.
    if (t.trim()) out.time = _canonTime(t);
    if (m.trim()) out.mem = _canonMem(m);
    return out;
}

/** Run `fn`, then put the page back exactly where it was.
 *
 * **Writing the editor's whole document moves the page.**  `cm.setValue`
 * followed by `setCursor` makes CodeMirror scroll the cursor into view,
 * and the browser drags the scrolling container (`.app-content`) with it
 * -- so ticking a checkbox in a card near the TOP threw the view 1704px
 * down to the editor at the bottom, every single time.  Measured in the
 * browser 2026-08-27, reported as *"the page always jumps off to another
 * place down"*, which is exactly what it did.
 *
 * The invariant is the point: **a card edit must not move the page.**  A
 * person ticking a box has said nothing about where they want to be
 * looking, so the answer is to leave them there.  Restored twice --
 * synchronously and on the next frame -- because CodeMirror's own
 * `refresh` can scroll again after this returns.
 *
 * Both card writers use it, not only the one that was reported: the asks
 * card wrote the document the same way and moved the page the same way.
 */
function keepingPagePut(fn) {
    const sc = document.querySelector(".app-content");
    const top = sc ? sc.scrollTop : 0;
    const put = () => { if (sc && sc.scrollTop !== top) sc.scrollTop = top; };
    try {
        return fn();
    } finally {
        put();
        requestAnimationFrame(put);
    }
}


/** Write the asks INTO the open `task.json`, which is what save sends.
 *
 * The page's one source of truth is the editor's text (this file's header
 * rule), so a control that kept its value beside it would be a second
 * answer to what the description says -- and the one that never reached
 * disk.  Written on `change` rather than on every keystroke: rewriting
 * the document under a moving cursor is how an editor fights its user.
 *
 * Absent-is-a-state, matching `task.py`: nothing asked writes NO key, so
 * a description that says nothing round-trips byte-identical.
 */
function applyAsksToDoc() {
    if (!_cm) return;
    const text = _cm.getValue();
    let task;
    try {
        task = JSON.parse(text);
    } catch (e) {
        return;              // mid-edit and unparseable; say nothing, lose nothing
    }
    if (!task || typeof task !== "object") return;
    const asks = askValues();
    const had = JSON.stringify(task.allocation || null);
    if (Object.keys(asks).length) task.allocation = asks;
    else delete task.allocation;
    if (JSON.stringify(task.allocation || null) === had) return;   // no-op
    const cur = _cm.getCursor && _cm.getCursor();
    keepingPagePut(() => {
        _cm.setValue(JSON.stringify(task, null, 2) + "\n");
        if (cur && _cm.setCursor) _cm.setCursor(cur);
    });
}

/** Fill the card FROM the open description, so reopening a folder shows
 *  what it already asks for rather than an empty card. */
function readAsksFromTask(task) {
    const a = (task && task.allocation) || {};
    _queue = a.domain || "";
    const t = $("ts-ask-time");
    const m = $("ts-ask-mem");
    // Loaded from the DESCRIPTION: a person put these there, so they carry
    // no auto mark and no queue click may replace them.
    if (t) { t.value = a.time || ""; delete t.dataset.mbAuto; }
    if (m) { m.value = a.mem || ""; delete m.dataset.mbAuto; }
}

/* ---------- when this run should tell you something ---------- */

/** The policy as `task.json`'s `notify` block carries it.
 *
 * WHEN only.  Where to send it is the user's own file on the machine that
 * runs the job, because a description travels and a token must not travel
 * with it (`plans/bench-and-junction-plan.md` § 2.9).
 *
 * The triggers are independent -- checkboxes, not a picker -- so either,
 * both or neither is a valid answer.  Finish is not among them: a run
 * ending always reports, which is why it is not offered as a box.
 */
function notifyValues() {
    const scf = $("ts-notify-scf");
    const per = $("ts-notify-periodic");
    const hrs = $("ts-notify-hours");
    const out = {};
    if (scf && scf.checked) out.on_scf_converged = true;
    if (per && per.checked) {
        const n = parseFloat((hrs || {}).value);
        // A number, in HOURS, on both sides -- `task.py` refuses "6h" and a
        // string, and it is right to: a value that changes meaning crossing
        // a boundary is how "4h" reached sbatch as `-t 4h`.
        if (isFinite(n) && n > 0) out.every_hours = n;
    }
    return out;
}

/** Write the policy INTO the open `task.json`, which is what save sends.
 *
 * Same rule as the queue card: the editor's text is the page's one source
 * of truth, so a control holding its value beside it would be a second
 * answer -- and the one that never reached disk.
 *
 * Absent-is-a-state, matching `task.Notify`: nothing ticked writes NO key,
 * so a description that reports on nothing round-trips byte-identical.
 */
function applyNotifyToDoc() {
    if (!_cm) return;
    let task;
    try {
        task = JSON.parse(_cm.getValue());
    } catch (e) {
        return;              // mid-edit and unparseable; say nothing, lose nothing
    }
    if (!task || typeof task !== "object") return;
    const want = notifyValues();
    const had = JSON.stringify(task.notify || null);
    if (Object.keys(want).length) task.notify = want;
    else delete task.notify;
    if (JSON.stringify(task.notify || null) === had) return;   // no-op
    const cur = _cm.getCursor && _cm.getCursor();
    keepingPagePut(() => {
        _cm.setValue(JSON.stringify(task, null, 2) + "\n");
        if (cur && _cm.setCursor) _cm.setCursor(cur);
    });
}

/** Fill the card FROM the open description, so reopening a folder shows
 *  what it already asks for rather than an empty card. */
function readNotifyFromTask(task) {
    const n = (task && task.notify) || {};
    const scf = $("ts-notify-scf");
    const per = $("ts-notify-periodic");
    const hrs = $("ts-notify-hours");
    if (scf) scf.checked = n.on_scf_converged === true;
    const every = parseFloat(n.every_hours);
    if (per) per.checked = isFinite(every) && every > 0;
    // The box keeps its offered default when the description says nothing,
    // so ticking the row does not first make the user think of a number.
    if (hrs && isFinite(every) && every > 0) hrs.value = String(every);
    paintNotifyNote();
}

/** One line saying what this calculation will actually send. */
function paintNotifyNote() {
    const note = $("ts-notify-note");
    if (!note) return;
    const v = notifyValues();
    const parts = [];
    if (v.on_scf_converged) parts.push("each SCF convergence");
    if (v.every_hours) parts.push(`every ${v.every_hours} h`);
    parts.push("and when it ends");
    note.textContent = parts.length > 1
        ? "Reports " + parts.slice(0, -1).join(", ") + " " + parts[parts.length - 1]
        : "Reports only when it ends";
}

/** What a `prep` would resolve for the open folder, and from which file.
 *
 *  The block `prep` itself prints, served by the same producer -- a
 *  hand-written notice here would be a second account of the same facts,
 *  free to drift from the one the terminal shows.
 */
async function loadResolved() {
    const facts = $("ts-resolved");
    if (!facts) return;
    const proj = (window.molbuilder || {}).projects;
    const dir = proj && proj.getCurrentDir && proj.getCurrentDir();
    if (!dir) { facts.hidden = true; return; }
    let d = null;
    try {
        // Provenance is a property of the FOLDER, not of the machine you
        // are preparing for; the `target` this used to send fed only the
        // bootstrap warning, retired 2026-08-25.
        d = await fetch("/api/task-setup/resolved?dest="
                        + encodeURIComponent(dir)).then((r) => r.json());
    } catch (e) { d = null; }
    if (!d || !d.ok) { facts.hidden = true; return; }

    facts.textContent = "";
    /* WHICH FILE SUPPLIED EACH SETTING -- the question provenance exists
     * to answer. */
    for (const [key, v] of Object.entries(d.effective || {})) {
        facts.appendChild(el("div", null,
            el("dt", null, key.replace("script_generation.", "")),
            el("dd", null, String(v.value || "\u2014")),
            el("dd", { class: "hint" }, "from " + (v.from || "?"))));
    }
    const found = (d.sources || []).filter((s) => s.found);
    if (found.length) {
        facts.appendChild(el("div", null,
            el("dt", null, "read from"),
            el("dd", { class: "hint" },
               found.map((s) => s.scope + " (" + s.via + ")").join(", "))));
    }
    /* AND WHAT IS WRONG WITH WHERE IT WAS READ FROM.  Showing the resolved
     * path without these would be the worse half of an answer: a person with
     * a `molbuilder.json` in their working directory would see a card naming
     * a different file and nothing saying the one they are editing is never
     * opened.  The terminal prints both; a tab that printed neither is how
     * the two came to disagree.
     *
     * The words are the server's, verbatim -- rewording them here would put a
     * second author on a message that has one home (§ 2.1a, § 2.1b). */
    for (const warning of [d.shadow, d.mode_warning]) {
        if (!warning) continue;
        facts.appendChild(el("div", null,
            el("dt", null, "warning"),
            el("dd", { class: "hint ts-config-warning" }, String(warning))));
    }
    facts.hidden = !facts.children.length;

}

async function loadMachines() {
    const card = $("ts-target-card");
    if (!card) return;
    let data = null;
    try {
        data = await fetch("/api/task-setup/machines").then((r) => r.json());
    } catch (e) { data = null; }
    if (!data || !data.ok) { card.hidden = true; return; }
    _machines = data.machines || [];
    const host = $("ts-target-choice");
    if (!host) return;
    host.textContent = "";
    for (const m of _machines) {
        host.appendChild(el("button", {
            type: "button", class: "opt", "data-machine": m.name,
            "aria-pressed": "false",
        }, el("b", {}, m.name), el("span", {}, m.summary)));
    }
    for (const b of host.querySelectorAll(".opt")) {
        b.addEventListener("click",
            () => setMachine(b.getAttribute("data-machine") || ""));
    }
    /* One machine is not a question.  Choosing it silently is the same
     * rule the CLI applies: there is no ambiguity, so nothing is asked. */
    card.hidden = false;
    if (!data.choice_required && _machines.length === 1) {
        setMachine(_machines[0].name);
    } else {
        setMachine("");
    }
}

function setShape(shape) {
    _shape = shape;
    for (const b of document.querySelectorAll("#ts-shape-card .opt")) {
        b.setAttribute("aria-pressed",
                       b.getAttribute("data-shape") === shape ? "true" : "false");
    }
    const needs = $("ts-shape-needs");
    if (needs) needs.hidden = !!shape;
    /* A SHAPE CHANGE IS NOT A RESET.  Below, a hand-over is turned into a
     * proposal -- stages, varies and a seeded bench.  That has to happen ONCE.
     * Re-running it on every click meant picking flat, building a two-stage
     * table with its overrides and a bench grid, then changing your mind to
     * hierarchical, threw all of it away and silently rebuilt the starting
     * proposal.  Shape is one field of the description; changing it edits that
     * field. */
    /* WRITING IT THROUGH IS NOT THE HAND-OVER'S PRIVILEGE.  This required
     * `_mode === "handover"`, so on a SAVED description the buttons repainted
     * and the model was never touched: the page showed `flat` chosen while the
     * buffer Save posts still said `hierarchical`, and the write was silently
     * the old value.  `task-setup.md` § 4 says the shape is free to change
     * before the first produce -- a refusal would have been a fair answer, and
     * showing the new choice while writing the old one is not one.
     *
     * The condition that matters is whether there is a DESCRIPTION to edit,
     * which is what `_task.schema` asks; the hand-over branch below is about
     * BUILDING one, and only that has to happen once. */
    if (shape && _task && _task.schema === "molbuilder/task@1") {
        _task.shape = shape;
        syncFromModel();
        refreshSave();
        return;
    }
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
/** Is a state being kept before this write? */
function wantsCheckpoint() {
    const box = $("ts-ckpt");
    return !!(box && box.checked);
}

/** The note that state would carry, trimmed. */
function checkpointNote() {
    const el = $("ts-ckpt-note");
    return ((el && el.value) || "").trim();
}

/* The button's answer depends on these two, so it is recomputed when they
 * change -- otherwise it is only ever right at the moment the page loaded. */
function watchCheckpointControls() {
    for (const id of ["ts-ckpt", "ts-ckpt-note"]) {
        const el = $(id);
        if (!el || el.dataset.mbWatched) continue;
        el.dataset.mbWatched = "1";
        el.addEventListener("input", refreshSave);
        el.addEventListener("change", refreshSave);
    }
}

/* The two asks validate as they are typed: "more than htc allows" is
 * worth knowing at the keystroke, not after a queue wait. */
function watchAskControls() {
    for (const id of ["ts-ask-time", "ts-ask-mem"]) {
        const el = $(id);
        if (!el || el.dataset.mbWatched) continue;
        el.dataset.mbWatched = "1";
        el.addEventListener("input", () => {
            // The moment a person types, the field is THEIRS -- so no
            // later queue click may overwrite it.  Clearing it hands it
            // back, and the next queue fills it again.
            if (el.value) el.dataset.mbAuto = "";
            else delete el.dataset.mbAuto;
            paintAskNotes();
            refreshSave();
        });
        el.addEventListener("change", () => { paintAskNotes(); applyAsksToDoc(); refreshSave(); });
    }
    // The notify card writes on `change` for the same reason the asks do:
    // rewriting the document under a moving cursor is how an editor fights
    // its user.  A number box also fires `change` on blur, which is when a
    // half-typed "1" has become the "12" the person meant.
    for (const id of ["ts-notify-scf", "ts-notify-periodic", "ts-notify-hours"]) {
        const el = $(id);
        if (!el || el.dataset.mbNotifyWired) continue;
        el.dataset.mbNotifyWired = "1";
        el.addEventListener("change", () => {
            paintNotifyNote();
            applyNotifyToDoc();
            refreshSave();
        });
    }
}

function refreshSave() {
    watchCheckpointControls();
    watchAskControls();
    const btn = $("ts-save");
    const why = $("ts-save-why");
    if (!btn) return;
    let blocked = "";
    if (!_dir)                                    blocked = "Pick a folder first.";
    else if (_mode === "empty")                   blocked = "Nothing to save — this folder carries no description and no hand-over.";
    else if (_mode === "handover" && !_shape)     blocked = "Choose how the files are kept apart, above.";
    /* A ticked checkpoint with no note cannot succeed: `saveState` requires
     * one (`checkpointing.md` L4 -- nothing writes a message on your behalf),
     * and the save aborts rather than writing without the state it was asked
     * to keep.  Leaving the button live meant you found that out by pressing
     * it and reading a refusal.  The condition was always knowable here. */
    else if (wantsCheckpoint() && !checkpointNote())
        blocked = "The checkpoint needs a note — say what this state is, or untick it.";
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
    const wantCkpt = wantsCheckpoint();
    if (wantCkpt && projects0 && projects0.checkpoint) {
        const note = checkpointNote();
        /* `status` answers `ok:false` for a folder that simply has no history
         * yet -- `ok` there means "this folder is under checkpointing", not
         * "the query worked".  Reading it as the latter skipped `init` for
         * exactly the folders that need it, and the save then died on
         * `saveState`'s "not a checkpoint folder; run init first" -- a message
         * about a step the page had decided to skip.  What this branch needs
         * is the question `initialized` already answers; `error` separates a
         * real failure from a fine answer. */
        const st = await projects0.checkpoint.status(_dir).catch(() => null);
        if (st && !st.error && !st.initialized) {
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
    /* Gate ③'s NON-refusing findings (sequence warnings) ride the OK
     * response, exactly what the CLI would have echoed -- and until the
     * U6 close they went on the floor while loadFolder repainted the
     * box to "loaded": the warned-and-navigated-past failure mode the
     * hand-over's notices arm just had fixed. */
    const warns = (Array.isArray(body.findings) ? body.findings : [])
        .filter((f) => f && f.severity !== "error");
    if (warns.length) {
        setState("loaded",
                 "Saved — the preflight has "
                 + (warns.length === 1 ? "a note" : warns.length + " notes"),
                 warns.map((f) => "• "
                     + (f.where ? "[" + f.where + "] " : "")
                     + (f.message || "")).join("\n"));
    }
}

/* ---------- wiring ---------- */

function start(projects) {

    // WIRED BEFORE THE FOLDER LOADS, and independent of it: the
    // destination is a MACHINE setting, so it is neither gated on a
    // calculation being open nor reset when you pick a different one.
    wireDestination();

    // `getCurrentDir` is the public accessor; reading sessionStorage directly
    // would duplicate the sidebar's own key name in a second place.
    const startDir = typeof projects.getCurrentDir === "function"
        ? projects.getCurrentDir()
        : "";
    loadFolder(projects, startDir);

    // Directory changes arrive on onChange; a dblclick commit also lands on
    // onCommit.  Both carry `dir`, and this page cares about the folder only.
    /* A restore rewrites the folder while we are showing it -- same
     * selection, different bytes -- so re-read on the folder-changed
     * channel as well as on selection.  Without this the tab kept
     * displaying a description the folder no longer had (2026-08-24). */
    if (typeof projects.onFolderChanged === "function") {
        projects.onFolderChanged((ev) => {
            const changed = (ev && ev.dir) || "";
            if (changed && _dir && changed !== _dir) return;   // not ours
            loadFolder(projects, _dir);
        });
    }
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
    // The machine options are built from the server's records, so the
    // buttons (and their listeners) are wired inside loadMachines.
    loadMachines();
    const addSetting = () => {
        const sel = $("ts-add-setting");
        if (!sel || !sel.value) return;
        // A row starts as ONE point -- a choice -- and the choice starts
        // at the value IN FORCE (template, else catalogue default).  It
        // started as the literal 1 for every type until 2026-08-20, which
        // for use_gpu or diag_algorithm was not a value at all.
        addPoint(sel.value, String(valueInForce(sel.value)));
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

/* ---------- where the reports go: a MACHINE setting ---------- */
/*
 * A different card from the policy one above, and a different file.  The
 * policy card writes task.json, which TRAVELS -- so it never sees a key.
 * This writes config_dir()/notify on this machine, which never travels
 * (`run-reports.md` 1 and 3.1).
 *
 * It is also machine-wide, so it does NOT hide with the calculation: what
 * it sets outlives whichever folder you happen to have open, and hiding
 * it would suggest otherwise.
 */

/** The key is write-only: shown as a placeholder, never fetched back. */
function paintDestination(d) {
    const state = $("ts-reports-state");
    const path = $("ts-reports-path");
    const form = $("ts-reports-form");
    const away = $("ts-reports-elsewhere");
    if (!state) return;
    if (path) path.textContent = d.path ? ("It lives at " + d.path + ".") : "";

    if (d.problem) {
        // A BROKEN file and NO file both mean nothing is sent, and they
        // look identical from the outside.  Saying which is the whole
        // reason this card is worth having.
        state.textContent = "There is a file, but it cannot be read: "
            + d.problem + " \u2014 nothing is being sent.";
        state.setAttribute("data-state", "bad");
    } else if (d.configured) {
        state.textContent = "Reports go to " + d.url
            + (d.has_key ? " (signed)" : " (no key \u2014 the address is the credential)")
            + (d.mode && d.mode !== "0o600"
               ? "  \u2014 warning: the file is " + d.mode + ", not 0600" : "");
        state.setAttribute("data-state", "set");
    } else {
        state.textContent = "Nothing is set up, so no reports are sent.";
        state.setAttribute("data-state", "none");
    }

    // WHICH MACHINE RUNS THE JOBS decides what this card can do.  On a
    // submit machine the file belongs on the cluster, which this server
    // cannot write -- so it hands over the exact command instead of
    // offering a button that would write the file somewhere useless.
    const here = d.can_write_here !== false;
    if (form) form.hidden = !here;
    if (away) away.hidden = here;
    const cmd = $("ts-reports-cmd");
    if (cmd && !here) {
        const url = ($("ts-reports-url") && $("ts-reports-url").value.trim())
            || d.url || "https://YOUR-SERVER:8888/api/<segment>";
        /* THE PATH IS RESOLVED ON THE FAR MACHINE, so the rule has to travel
         * as shell rather than as an answer -- this server's config directory
         * is not the cluster's.  It must be the WHOLE rule
         * (`configuration.md` § 2.1c): MOLBUILDER_CONFIG_DIR first and exactly
         * as given, then XDG_CONFIG_HOME/molbuilder, then ~/.config/molbuilder.
         *
         * It implemented only the last two until 2026-08-31, so on any machine
         * with MOLBUILDER_CONFIG_DIR set, following this card wrote the file
         * where the monitor does not look -- and silently, because an absent
         * notify file simply means "no notifier".  That is the same failure
         * `notify_setup.py` records from the card's previous life, when it
         * named `~/.molbuilder/notify` while the monitor read
         * `config_dir()/notify`.  Fixing the spelling last time left the
         * hardcoding in place; this states the rule instead.
         *
         * `tests/test_task_setup_notify_js.py` pins all three branches. */
        cmd.textContent =
            'cfg="${MOLBUILDER_CONFIG_DIR:-${XDG_CONFIG_HOME:-$HOME/.config}/molbuilder}"\n'
            + 'mkdir -p -m 700 "$cfg"\n'
            + 'cat > "$cfg/notify" <<\'EOF\'\n'
            + '{\n  "url": "' + url + '",\n  "key": "PASTE-THE-KEY"\n}\nEOF\n'
            + 'chmod 600 "$cfg/notify"';
    }
    if (d.configured && $("ts-reports-url") && !$("ts-reports-url").value) {
        $("ts-reports-url").value = d.url;
    }
    // THE EMPTY KEY FIELD MEANS "UNCHANGED", so it must not read as
    // "none set".  The field is cleared after every save -- a secret left
    // in the DOM ends up in a screenshot -- and saving again with it blank
    // KEEPS the stored key, so the placeholder has to say so or the next
    // person assumes they have wiped it.
    const kf = $("ts-reports-key");
    if (kf) {
        kf.placeholder = d.has_key
            ? "unchanged \u2014 type a new one to replace it"
            : "from `molbuilder notify-token`";
    }
}

async function loadDestination() {
    try {
        const r = await fetch("/api/notify/destination");
        paintDestination(await r.json());
    } catch (e) {
        const s = $("ts-reports-state");
        if (s) s.textContent = "Could not ask this server: " + e;
    }
}

/** Say something under the destination controls.
 *
 * `data-state`, because that is what every other `.ts-ask-note` on this
 * page already uses and the sheet already styles (`[data-state="bad"]`).
 * A first version set an `is-bad` CLASS instead -- a second convention for
 * one idea on one page, and shaped exactly like `modify/viewer.js`'s own
 * status setter, which `test_no_duplicated_ui_components` caught: *one
 * function written twice is two places for a fix to miss.*
 */
function destNote(text, bad) {
    const n = $("ts-reports-note");
    if (!n) return;
    n.textContent = text;
    n.setAttribute("data-state", bad ? "bad" : "ok");
}

async function saveDestination() {
    const url = ($("ts-reports-url") || {}).value || "";
    const key = ($("ts-reports-key") || {}).value || "";
    destNote("saving\u2026", false);
    try {
        const r = await fetch("/api/notify/destination", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url: url.trim(), key: key.trim() }),
        });
        const d = await r.json();
        if (!d.ok) { destNote(d.error || "could not save", true); return; }
        // CLEARED ON SUCCESS: the field is write-only, and leaving a
        // secret sitting in the DOM is how it ends up in a screenshot.
        if ($("ts-reports-key")) $("ts-reports-key").value = "";
        paintDestination(d);
        destNote("Saved. Send a test report to be sure it works.", false);
    } catch (e) {
        destNote(String(e), true);
    }
}

async function clearDestination() {
    destNote("removing\u2026", false);
    try {
        const d = await (await fetch("/api/notify/destination",
                                     { method: "DELETE" })).json();
        if ($("ts-reports-url")) $("ts-reports-url").value = "";
        if ($("ts-reports-key")) $("ts-reports-key").value = "";
        paintDestination(d);
        destNote("Removed \u2014 nothing is sent now.", false);
    } catch (e) { destNote(String(e), true); }
}

async function testDestination() {
    destNote("sending one report\u2026", false);
    try {
        const d = await (await fetch("/api/notify/destination/test",
                                     { method: "POST" })).json();
        if (d.ok) { destNote("It arrived.", false); return; }
        // The listener refuses every way identically, so the hint names
        // all of the possibilities rather than guessing between them.
        destNote(d.error || d.hint
                 || ("the destination answered " + d.status), true);
    } catch (e) { destNote(String(e), true); }
}

function wireDestination() {
    // The destination lives INSIDE the notify card now (user, 2026-08-27:
    // "where the notification should be sent should be configurable in
    // this card too"), so there is no card of its own to reveal -- the
    // notify card's own gate governs both halves.  Which is a behaviour
    // change worth naming: the destination is machine-wide, so it is now
    // only reachable with a calculation open.  That is the trade for
    // having it where a person is actually thinking about it, and the CLI
    // (`molbuilder notify-token`) is the door that needs no calculation.
    if (!$("ts-reports-state")) return;
    const on = (id, fn) => { const b = $(id); if (b) b.addEventListener("click", fn); };
    on("ts-reports-save", saveDestination);
    on("ts-reports-clear", clearDestination);
    on("ts-reports-test", testDestination);
    loadDestination();
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot, { once: true });
} else {
    boot();
}
