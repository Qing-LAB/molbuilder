/* projects/checkpoint.js -- the run-history VIEW of the projects sidebar.
 *
 * The checkpoint UI shares the file-list space (2026-08-19, user): a small
 * status-colored button beside the filter box (#ps-checkpoint-toggle)
 * exists only for a run directory, and clicking it swaps #ps-list for
 * #ps-checkpoint in place -- one column, one owner for the swap (this
 * module).  While the checkpoint view shows, the filter input is disabled:
 * it filters files, and a control acting on a hidden list would act
 * invisibly.  Subscribes to projects.onChange to react when the user
 * navigates into a directory.  For each run directory:
 *
 *   1. GET  /api/checkpoint/state    -> sensor pill + visibility decision
 *   2. GET  /api/checkpoint/list     -> the states, when the folder has any
 *   3. POST /api/checkpoint/init     -> empty-state CTA
 *   4. POST /api/checkpoint/save     -> "Save state…" button
 *   5. POST /api/checkpoint/tag      -> "Name this state…" button
 *   6. POST /api/checkpoint/restore  -> per-row "Restore" button
 *
 * Vocabulary is the contract's and nothing else: a STATE is a saved snapshot
 * of the folder, a TAG is a name you gave one, and the folder always STANDS AT
 * exactly one state.  There are no branches and no "HEAD" here -- going back
 * and saving is how you fork, so there is nothing to declare and nothing to
 * name (docs/execution/checkpointing.md § 5, § 7.1).
 *
 * Activation gate (docs/web/projects.md): the panel only appears
 * for a *run directory* -- a dir at projects rel-depth 3, in the
 * canonical layout projects/PROJECT/CATEGORY/RUNNING_DIR.  Selecting
 * anything shallower (or a file) hides the panel entirely.
 *
 * Refresh model (docs/web/projects.md): explicit only --
 * NO background polling.  State refreshes on (a) directory-enter into
 * such a run dir (so the toggle's color is honest without opening the
 * view), (b) opening the view, and (c) the manual Refresh control.
 * There is no setInterval and no visibility-driven timer.
 *
 * Graph viewer: lazy-loaded @gitgraph/js (see _ensureGitgraph() below).
 *
 * HTTP contract: web/blueprints/checkpoint.py + docs/web/web-api.md.
 * Invariants: docs/execution/checkpointing.md.
 */

// projects rel-depth of a canonical run dir:
// projects/PROJECT_NAME/CATEGORY/RUNNING_DIR_NAME (docs/web/projects.md).
const RUN_DIR_DEPTH = 3;
const _state = {
    /** Currently selected directory path (relative to projects root,
     *  resolved to absolute by the API).  null when no dir selected. */
    currentDir:    null,
    /** Latest /api/checkpoint/state snapshot (or null on init). */
    repoState:     null,
    /** The states this folder has, newest first. */
    states:        [],
    /** True while the checkpoint view occupies the list space.  The
     *  preference survives dir changes and reloads (sessionStorage), so
     *  stepping out of a run dir and back returns the view you had. */
    open:          false,
};

// DOM handles, populated by _attach().  elFileList / elFilterInput are the
// OTHER half of the swap -- same module (the sidebar is one module in nine
// files), one owner for the exclusivity.
let elPanel, elSensor, elToggle, elEmpty, elInitBtn, elActions,
    elCommitBtn, elTagBtn, elRefreshBtn, elList, elGraph, elAdvisory,
    elViewListBtn, elViewGraphBtn, elFileList, elFilterInput;

// Current view mode: "list" (default) or "graph".  Persisted in
// sessionStorage so a refresh keeps the user's preference.
let _viewMode = "list";

// Lazy-loaded @gitgraph/js global.  Loaded on first switch to graph
// view -- avoids a 96 KB blocking download for users who never use
// the graph.
let _gitgraphPromise = null;

/* ---------- DOM bootstrap ---------- */

function _attach() {
    elPanel      = document.getElementById("ps-checkpoint");
    elSensor     = document.getElementById("ps-checkpoint-sensor");
    elToggle     = document.getElementById("ps-checkpoint-toggle");
    elEmpty      = document.getElementById("ps-checkpoint-empty");
    elInitBtn    = document.getElementById("ps-checkpoint-init");
    elActions    = document.getElementById("ps-checkpoint-actions");
    elCommitBtn  = document.getElementById("ps-checkpoint-commit-btn");
    elTagBtn     = document.getElementById("ps-checkpoint-tag-btn");
    elRefreshBtn = document.getElementById("ps-checkpoint-refresh-btn");
    elList         = document.getElementById("ps-checkpoint-list");
    elGraph        = document.getElementById("ps-checkpoint-graph");
    elAdvisory     = document.getElementById("ps-checkpoint-advisory");
    elViewListBtn  = document.getElementById("ps-checkpoint-view-list");
    elViewGraphBtn = document.getElementById("ps-checkpoint-view-graph");
    elFileList     = document.getElementById("ps-list");
    elFilterInput  = document.getElementById("ps-filter-input");

    if (!elPanel) return false;   // template not loaded; skip wiring

    // BIND ONCE.  `initCheckpointPanel` documents itself as safe to call from
    // several bootstrap paths, and this function is what that promise rests
    // on -- but every listener below was added unconditionally, so a second
    // call would double-bind them and one click on Save would POST twice.
    // Only one caller exists today; the docstring invites a second, which is
    // exactly how a latent trap gets sprung by an unrelated change.
    if (elPanel.dataset && elPanel.dataset.psCheckpointWired === "1") {
        return true;
    }
    if (elPanel.dataset) elPanel.dataset.psCheckpointWired = "1";

    // Restore view-mode preference from sessionStorage.
    try {
        const saved = sessionStorage.getItem("ws.ui.checkpoint.view");
        if (saved === "graph" || saved === "list") _viewMode = saved;
    } catch (_) { /* sessionStorage disabled — fall through to default */ }
    _updateViewButtons();

    // Restore the open-view preference -- the SAME mechanism as the
    // view-mode pref above.  A UI pref is per-origin sessionStorage, owned
    // here, never smuggled through the workspace.
    try {
        if (sessionStorage.getItem("ps.checkpoint.open") === "1") {
            _state.open = true;
        }
    } catch (_) { /* sessionStorage disabled -- default to files */ }

    if (elToggle) elToggle.addEventListener("click",
        () => _setOpen(!_state.open));
    elInitBtn.addEventListener("click", _onInitClick);
    elCommitBtn.addEventListener("click", _onCommitClick);
    elTagBtn.addEventListener("click", _onTagClick);
    // The Refresh control is the "check content now" affordance: the automatic
    // read on directory-enter compares size and timestamp, which is enough to
    // draw a badge and cannot lose anything by being briefly wrong.  Pressing
    // Refresh says you want certainty.
    elRefreshBtn.addEventListener("click", () => _refresh({ deep: true }));
    // Explicit arg: bound bare, the click Event arrives as `opts` and only
    // works because Event.deep happens to be undefined.  The pill is a quick
    // re-read; the Refresh button is the one that asks for certainty.
    elSensor.addEventListener("click", () => _refresh());
    elList.addEventListener("click", _onListClick);
    if (elViewListBtn)  elViewListBtn.addEventListener("click",
        () => _setViewMode("list"));
    if (elViewGraphBtn) elViewGraphBtn.addEventListener("click",
        () => _setViewMode("graph"));

    return true;
}

/**
 * Whether ``dirPath`` is a canonical run directory -- projects
 * rel-depth 3 (projects/PROJECT/CATEGORY/RUNNING_DIR), the only place
 * a checkpoint viewer activates (docs/web/projects.md).  The
 * ``.git/`` presence is confirmed separately by /api/checkpoint/state;
 * this is the cheap structural gate that runs before any fetch.
 */
function _isRunDir(dirPath) {
    if (!dirPath) return false;
    const projects = window.molbuilder && window.molbuilder.projects;
    const root = projects && typeof projects.getProjectsRoot === "function"
        ? projects.getProjectsRoot() : "";
    if (!root) return false;
    const norm = (p) => p.replace(/\/+$/, "");
    const dir = norm(dirPath);
    const base = norm(root);
    if (dir === base || !dir.startsWith(base + "/")) return false;
    const rel = dir.slice(base.length + 1);
    return rel.split("/").filter(Boolean).length === RUN_DIR_DEPTH;
}

/* ---------- Public API ---------- */

/**
 * Notify the panel that the user navigated into a directory.
 * Called by the projects sidebar entry on every selection change.
 *
 * @param {string|null} dirPath - absolute path of selected directory,
 *                                or null when no directory is current.
 */
export function onDirectoryChange(dirPath) {
    // Activation gate: the viewer exists ONLY for a canonical run dir
    // (rel-depth 3).  Anywhere else -- a project dir, a category dir,
    // the projects root, or a file -- the panel is hidden entirely
    // (docs/web/projects.md).
    if (!_isRunDir(dirPath)) {
        _state.currentDir = null;
        _hide();
        return;
    }
    // No-op guard: projects.onChange fires on every single-click FILE selection too
    // (the cursor moves within the same dir).  Re-entering the SAME run dir must not
    // re-fetch -- the contract refreshes on directory-ENTER + the manual Refresh only
    // (checkpoint.js header), and re-running GET /api/checkpoint/state per file click
    // is exactly the "don't rewrite on a no-op tick" rule.
    if (dirPath === _state.currentDir) return;
    _state.currentDir = dirPath;
    if (elToggle) elToggle.hidden = false;
    // The stored preference decides which view greets a run dir; the state
    // fetch happens either way, because the BUTTON's color must be honest
    // without the view ever opening.
    _applyOpen();
    _refresh();
}

/* ---------- Internal: the view swap ---------- */

function _hide() {
    // Not a run dir: the button does not exist and the files are back.
    // The open PREFERENCE survives (stepping out and back keeps your view).
    if (!elPanel) return;
    elPanel.hidden = true;
    if (elToggle) elToggle.hidden = true;
    if (elFileList) elFileList.hidden = false;
    _setFilterDisabled(false);
}

function _setOpen(open) {
    _state.open = !!open;
    try {
        if (_state.open) sessionStorage.setItem("ps.checkpoint.open", "1");
        else sessionStorage.removeItem("ps.checkpoint.open");
    } catch (_) { /* sessionStorage disabled -- pref just won't persist */ }
    _applyOpen();
    if (_state.open) _refresh();
}

function _applyOpen() {
    // ONE place decides the exclusivity: exactly one of the file list and
    // the checkpoint view occupies the space below the filter bar.
    const open = _state.open;
    if (elPanel)    elPanel.hidden    = !open;
    if (elFileList) elFileList.hidden = open;
    if (elToggle)   elToggle.setAttribute("aria-pressed", String(open));
    _setFilterDisabled(open);
}

function _setFilterDisabled(disabled) {
    // The filter acts on the FILE list.  Acting on a hidden list is acting
    // invisibly, so the input says it is out of scope instead.
    if (!elFilterInput) return;
    elFilterInput.disabled = disabled;
    elFilterInput.title = disabled
        ? "Filtering applies to files -- close the run history to filter"
        : "";
}

function _paintToggle(state, text) {
    if (!elToggle) return;
    elToggle.setAttribute("data-state", state);
    elToggle.title = "Run history -- " + text
        + (_state.open ? " (click for files)" : " (click to open)");
}

function _renderState(repoState) {
    _state.repoState = repoState;
    if (!repoState || !repoState.initialized) {
        elSensor.textContent = "no states saved";
        elSensor.setAttribute("data-state", "uninit");
        _paintToggle("uninit", "no states saved");
        // Clear the tooltip too.  The other two branches SET it, so without
        // this the pill kept the previous folder's list of unsaved files while
        // reading "no states saved" -- naming files that are not in this
        // directory at all.
        elSensor.title    = "";
        elEmpty.hidden    = false;
        elActions.hidden  = true;
        elList.hidden     = true;
        if (elGraph) elGraph.hidden = true;
        return;
    }
    // "unsaved", not "dirty": the folder differs from the state it STANDS AT,
    // which is the only baseline (§ 5).  Measured against the newest state
    // instead, this would light up after every restore about work that is
    // already saved -- and a warning that fires when nothing is wrong is one
    // people learn to ignore.
    const unsaved = (repoState.unsaved || []).length;
    if (unsaved > 0) {
        elSensor.textContent = `${unsaved} unsaved`;
        elSensor.setAttribute("data-state", "dirty");
        elSensor.title = (repoState.unsaved || []).join("\n");
        _paintToggle("dirty", `${unsaved} unsaved`);
    } else {
        elSensor.textContent = "saved";
        elSensor.setAttribute("data-state", "clean");
        elSensor.title = repoState.standing_at
            ? `standing at ${repoState.standing_at.short}`
            : "";
        _paintToggle("clean", "saved");
    }
    elEmpty.hidden   = true;
    elActions.hidden = false;

    // A HAND EDIT INSIDE THE GENERATED BLOCK IS NOT SUPPOSED TO HAPPEN, so say
    // so.  molbuilder rewrites that block from the classification on every save,
    // which is what makes the edit detectable at all -- and also what makes it
    // vanish.  Detecting it and staying quiet leaves somebody editing a file
    // that never survives, and wondering why their rule keeps coming undone.
    // The note names the way to get what they wanted instead.
    if (repoState.ignore_edited) {
        _showAdvisory(
            "The molbuilder block in .gitignore was edited by hand. It is "
            + "rewritten from the classification on every save, so that edit "
            + "will be lost. Put your own entries outside the molbuilder "
            + "markers and they are left alone.");
    }
}

function _renderError(message) {
    elSensor.textContent = "error";
    elSensor.setAttribute("data-state", "error");
    _paintToggle("error", "error");
    _showAdvisory("Sensor error: " + message);
}

function _renderStates(states) {
    _state.states = states || [];
    _renderActiveView(_state.states);
}

function _buildRow(state) {
    const li = document.createElement("li");
    li.className   = "ps-checkpoint-list-item";
    li.dataset.sha = state.id;
    if (_state.repoState && _state.repoState.standing_at
        && _state.repoState.standing_at.id === state.id) {
        li.setAttribute("data-standing", "true");
    }

    const id = document.createElement("span");
    id.className   = "ps-checkpoint-row-sha";
    id.textContent = state.short;
    li.appendChild(id);

    // The NOTE, never a generated stand-in.  It is the only thing that answers
    // the question you bring to this list a month later (L3), and the panel is
    // where that question gets asked.
    const note = document.createElement("span");
    note.className   = "ps-checkpoint-row-summary";
    note.textContent = state.note;
    li.appendChild(note);

    // Which state this one came from -- two rows sharing a parent ARE the fork
    // (§ 7.1), and nothing had to be named for it.
    if (state.parent) {
        const from = document.createElement("span");
        from.className   = "ps-checkpoint-row-parent";
        from.textContent = "from " + state.parent.slice(0, 7);
        li.appendChild(from);
    }

    if (state.tags && state.tags.length) {
        const tags = document.createElement("span");
        tags.className = "ps-checkpoint-row-refs";
        for (const name of state.tags) {
            const chip = document.createElement("span");
            chip.className = "ps-checkpoint-ref-chip";
            chip.setAttribute("data-kind", "tag");
            chip.textContent = name;
            tags.appendChild(chip);
        }
        li.appendChild(tags);
    }

    // Inline action buttons -- shown when the row is expanded
    // (click toggles .is-expanded).
    const actions = document.createElement("div");
    actions.className = "ps-checkpoint-row-actions";
    const btnRestore = document.createElement("button");
    btnRestore.type        = "button";
    btnRestore.className   = "ps-checkpoint-row-action-btn";
    btnRestore.dataset.action = "restore";
    btnRestore.textContent  = "Restore to here";
    btnRestore.title        = "Rewind text + binaries to this checkpoint";
    actions.appendChild(btnRestore);
    li.appendChild(actions);

    return li;
}

/* ---------- View mode swap (List <-> Graph) ---------- */

function _updateViewButtons() {
    if (!elViewListBtn || !elViewGraphBtn) return;
    elViewListBtn.classList.toggle("is-active",  _viewMode === "list");
    elViewGraphBtn.classList.toggle("is-active", _viewMode === "graph");
    elViewListBtn.setAttribute("aria-selected",
        _viewMode === "list" ? "true" : "false");
    elViewGraphBtn.setAttribute("aria-selected",
        _viewMode === "graph" ? "true" : "false");
}

function _setViewMode(mode) {
    if (mode !== "list" && mode !== "graph") return;
    _viewMode = mode;
    try { sessionStorage.setItem("ws.ui.checkpoint.view", mode); }
    catch (_) { /* sessionStorage disabled */ }
    _updateViewButtons();
    _renderActiveView(_state.states);
}

function _renderActiveView(states) {
    // Hide both, then show the active one (single source of layout
    // truth -- no race between toggles).
    if (elList)  elList.hidden  = true;
    if (elGraph) elGraph.hidden = true;
    if (_viewMode === "graph") {
        if (elGraph) elGraph.hidden = false;
        _renderGraph(states);
    } else {
        if (elList)  elList.hidden  = false;
        _renderListRows(states);
    }
}

/* ---------- Lazy @gitgraph/js loader ---------- */

function _loadGitGraph() {
    // Cached promise — one network fetch + one global mount per page.
    if (_gitgraphPromise) return _gitgraphPromise;
    _gitgraphPromise = new Promise((resolve, reject) => {
        if (window.GitgraphJS) { resolve(window.GitgraphJS); return; }
        const tag = document.createElement("script");
        tag.src   = "/static/vendor/gitgraph/gitgraph.umd.js";
        tag.async = true;
        tag.onload  = () => {
            if (window.GitgraphJS) {
                resolve(window.GitgraphJS);
            } else {
                reject(new Error(
                    "gitgraph.umd.js loaded but window.GitgraphJS is "
                    + "missing -- vendor file may be stale or wrong."));
            }
        };
        tag.onerror = () =>
            reject(new Error("failed to load gitgraph.umd.js"));
        document.head.appendChild(tag);
    });
    return _gitgraphPromise;
}

/* ---------- Graph view rendering ---------- */

async function _renderGraph(states) {
    if (!elGraph) return;
    elGraph.innerHTML = "";
    if (!states || !states.length) {
        const empty = document.createElement("p");
        empty.className   = "ps-checkpoint-graph-empty";
        empty.textContent = "(no states to graph)";
        elGraph.appendChild(empty);
        return;
    }
    let GitgraphJS;
    try {
        GitgraphJS = await _loadGitGraph();
    } catch (e) {
        _showAdvisory("Graph viewer unavailable: " +
            String(e?.message || e));
        return;
    }

    // The @gitgraph/js renderer needs a tag <svg> in our container.
    const svg = document.createElementNS(
        "http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("width", "100%");
    elGraph.appendChild(svg);

    // Tie the graph palette to the sidebar's CSS variables so a
    // future palette change in projects-sidebar.css propagates
    // without touching this file.
    const cssVar = (name, fallback) => {
        const v = getComputedStyle(document.documentElement)
                      .getPropertyValue(name).trim();
        return v || fallback;
    };
    const template = GitgraphJS.templateExtend(
        GitgraphJS.TemplateName.Metro, {
        colors:  [
            cssVar("--ps-accent",   "#4a7aa7"),
            cssVar("--ps-accent-2", "#7a5b2d"),
            cssVar("--ps-accent-3", "#2d4a5a"),
            "#2d4a2d",
            "#4a2d4a",
        ],
        branch:  { spacing: 24, lineWidth: 2 },
        commit:  {
            spacing: 36,
            dot: { size: 6 },
            message: {
                color:    cssVar("--ps-fg", "#dcdcdc"),
                font:     "11px system-ui, sans-serif",
                displayAuthor: false,
                displayHash:   true,
            },
        },
    });

    const gitgraph = GitgraphJS.createGitgraph(svg, {
        template,
        orientation: "vertical",
    });

    // Oldest first: a child is drawn after the state it came from.
    const ordered = states.slice().reverse();

    // THE SHAPE COMES FROM PARENTAGE, NOT FROM BRANCH NAMES.  There are no
    // branches in this system -- going back to a state and saving from it is
    // how you fork (§ 7.1), so a fork is simply a state with a second child.
    // The old renderer read branch names off ref decorations, which is a
    // concept the contract removed; it drew every history as one line.
    //
    // @gitgraph/js needs a branch object per line, so one is created the
    // moment a parent acquires its second child.  Those names are a drawing
    // artifact and are never shown to the user as something they chose.
    const lineOf   = new Map();   // state id -> gitgraph branch
    const children = new Map();   // parent id -> how many children drawn
    let trunk = null;

    for (const st of ordered) {
        let line;
        if (!st.parent || !lineOf.has(st.parent)) {
            line = trunk || (trunk = gitgraph.branch("main"));
        } else {
            const seen = children.get(st.parent) || 0;
            line = seen === 0
                ? lineOf.get(st.parent)                    // continue the line
                : lineOf.get(st.parent).branch(st.short);  // a second child = a fork
            children.set(st.parent, seen + 1);
        }
        line.commit({
            subject: st.note,
            hash:    st.short,
            onClick: () => _showStateDetail(st),
        });
        lineOf.set(st.id, line);
        for (const name of (st.tags || [])) line.tag(name);
    }
}

function _showStateDetail(state) {
    // Click on a node = an inline line with what the list cannot fit.
    const tags = (state.tags || []).join(", ");
    const from = state.parent ? `from ${state.parent.slice(0, 7)}` : "the first state";
    _showAdvisory(
        `${state.short} — ${state.note} · ${from}` +
        (tags ? ` · tagged ${tags}` : ""));
}

/* ---------- List view extraction (so View toggle can call it) ---------- */

function _renderListRows(states) {
    if (!elList) return;
    elList.innerHTML = "";
    for (const st of (states || [])) {
        elList.appendChild(_buildRow(st));
    }
}


/* ---------- Network calls ---------- */

async function _fetchJSON(method, url, body) {
    const opts = { method, headers: { "Content-Type": "application/json" } };
    if (body !== undefined) opts.body = JSON.stringify(body);
    const r = await fetch(url, opts);
    let payload = null;
    try { payload = await r.json(); } catch (_) { /* empty body */ }
    return { http: r.status, body: payload };
}

async function _refresh(opts = {}) {
    if (!_state.currentDir) return;
    _hideAdvisory();
    try {
        const stRes = await _fetchJSON("GET",
            `/api/checkpoint/state?path=${encodeURIComponent(_state.currentDir)}`
            + (opts.deep ? "&deep=1" : ""));
        if (stRes.http >= 500 || !stRes.body) {
            _renderError(stRes.body?.error || "HTTP " + stRes.http);
            return;
        }
        if (stRes.http !== 200) {
            // 4xx: bad path or similar.  Sensor reads "error".
            _renderError(stRes.body.error || "HTTP " + stRes.http);
            return;
        }
        // The status fields are the response, not a nested object: `state`
        // now means a saved snapshot, so the folder's condition cannot also
        // be called that (§ 5).
        _renderState(stRes.body);

        if (stRes.body.initialized && _state.open) {
            // The list is fetched only while the view is showing -- a
            // closed view needs the STATE (the button's color), never the
            // fifty rows nobody is looking at.
            const lsRes = await _fetchJSON("GET",
                `/api/checkpoint/list?path=${encodeURIComponent(_state.currentDir)}&limit=50`);
            if (lsRes.body && lsRes.body.ok && lsRes.body.states) {
                _renderStates(lsRes.body.states);
            }
        } else {
            elList.hidden = true;
            if (elGraph) elGraph.hidden = true;
        }
    } catch (e) {
        _renderError(String(e && e.message || e));
    }
}

/* ---------- Action handlers ---------- */

/**
 * Set the folder up -- and, when its own name cannot be used, ask for one.
 *
 * A calculation's name is written verbatim into every state and is never
 * repaired silently (L3), so a folder called `BDT relax!` is refused.  Left at
 * that, the panel was a dead end: the refusal is the same whichever button you
 * press, and nothing here could supply the one thing that resolves it.  The
 * server marks that refusal `where: "calculation"` precisely so a surface can
 * tell it apart from the other one (a folder holding several calculations,
 * which a name does NOT fix) and ask the right question.
 *
 * Two steps, the same shape `_restore` uses: try, and only if the server
 * refuses for a reason the user can answer, ask -- with the reason in front of
 * them.  The ordinary case never sees a prompt.
 */
async function _onInitClick() {
    if (!_state.currentDir) return;
    _hideAdvisory();
    elInitBtn.disabled = true;
    try {
        let res = await _fetchJSON("POST", "/api/checkpoint/init",
            { path: _state.currentDir });
        const refusal = (res.body && Array.isArray(res.body.errors_only)
                         && res.body.errors_only.length)
            ? res.body.errors_only[0] : null;
        if (refusal && refusal.where === "calculation") {
            const name = prompt(
                "This folder's name cannot be used as a calculation name -- "
                + "it is written into every saved state as you type it, and is "
                + "never changed for you.\n\n"
                + "Letters, digits, - and _ only:", "");
            if (name === null) return;               // user cancelled
            if (!name.trim()) {
                _showAdvisory("A calculation needs a name. Nothing was set up.");
                return;
            }
            res = await _fetchJSON("POST", "/api/checkpoint/init",
                { path: _state.currentDir, calculation: name.trim() });
        }
        if (res.body && res.body.ok) {
            await _refresh();
            return;
        }
        const again = (res.body && Array.isArray(res.body.errors_only)
                       && res.body.errors_only.length)
            ? res.body.errors_only[0].message : null;
        _showAdvisory(again || ("Set-up failed: " +
            (res.body?.error || "HTTP " + res.http)));
    } catch (e) {
        _showAdvisory("Set-up failed: " + String(e?.message || e));
    } finally {
        elInitBtn.disabled = false;
    }
}

/**
 * Save the folder as a new state.  THE NOTE IS REQUIRED AND NEVER GENERATED
 * (L3): it is the only thing that answers the question you bring to this list a
 * month later -- why did I stop here, and what was I about to do?  A timestamp
 * answers neither and repeats the column beside it, so an empty answer is
 * refused here rather than filled in.
 */
async function _onCommitClick() {
    if (!_state.currentDir) return;
    const note = prompt(
        "What happened, and what were you about to do?\n"
        + "(e.g. \"stage 1 converged, 41 steps -- before retightening\")", "");
    if (note === null) return;     // user cancelled
    if (!note.trim()) {
        _showAdvisory("A state needs a note. Nothing was saved.");
        return;
    }
    _hideAdvisory();
    elCommitBtn.disabled = true;
    try {
        // Through the public API (§ 5), so the panel and a tab that must take
        // a state before writing share ONE implementation.
        const out = await saveState(_state.currentDir, note.trim());
        const res = { body: { ok: out.ok, changed: out.changed,
                              error: out.error } };
        if (res.body && res.body.ok) {
            // `changed:false` is the honest "nothing differed from where you
            // stand" -- not a failure, and not a state that was invented.
            if (res.body.changed === false) {
                _showAdvisory(
                    "Nothing changed since the state this folder stands at.");
            }
        } else {
            _showAdvisory("Save failed: " +
                (res.body?.error || "HTTP " + res.http));
        }
    } catch (e) {
        _showAdvisory("Save failed: " + String(e?.message || e));
    } finally {
        elCommitBtn.disabled = false;
    }
}

/**
 * Give a state a name so you can find it again (§ 5, L4).  Nothing tags on your
 * behalf, so this namespace is the user's alone -- which is what makes their
 * own tags easy to see in a list a month later.
 */
async function _onTagClick() {
    if (!_state.currentDir) return;
    const name = prompt("Name this state (e.g. stage1-good):", "");
    if (name === null) return;                     // user cancelled
    if (!name.trim()) return;
    const note = prompt("Why is it worth coming back to?", "");
    if (note === null) return;
    if (!note.trim()) {
        _showAdvisory("A tag needs a note saying why. Nothing was tagged.");
        return;
    }
    _hideAdvisory();
    elTagBtn.disabled = true;
    try {
        const res = await _fetchJSON("POST", "/api/checkpoint/tag", {
            path: _state.currentDir,
            name: name.trim(),
            note: note.trim(),
        });
        if (res.body && res.body.ok) {
            await _refresh();
        } else {
            _showAdvisory("Tag failed: " +
                (res.body?.error || "HTTP " + res.http));
        }
    } catch (e) {
        _showAdvisory("Tag failed: " + String(e?.message || e));
    } finally {
        elTagBtn.disabled = false;
    }
}

function _onListClick(ev) {
    const li = ev.target.closest(".ps-checkpoint-list-item");
    if (!li) return;
    const btn = ev.target.closest(".ps-checkpoint-row-action-btn");
    if (btn) {
        if (btn.dataset.action === "restore") {
            _onRestoreClick(li.dataset.sha);
            ev.stopPropagation();
            return;
        }
    }
    // Toggle expanded state (shows the inline action buttons).
    li.classList.toggle("is-expanded");
}

async function _onRestoreClick(sha) {
    if (!_state.currentDir) return;
    if (!sha) return;
    const target = _state.states.find(s => s.id === sha);
    const label = target
        ? (target.tags && target.tags.length ? target.tags[0] : target.short)
        : sha.slice(0, 7);
    if (!confirm(
        `Put this folder back to ${label}?\n\n` +
        `The whole folder returns to that state -- text and big files\n` +
        `together.  Anything here that is not saved is named first, and\n` +
        `you get the choice before anything changes.`)) {
        return;
    }
    _hideAdvisory();
    await _restore(sha, label, false);
}

/**
 * A5 in two steps: try, and if there is unsaved work the server names it and
 * refuses.  Only then is the user asked whether to accept the loss -- with the
 * files in front of them, which is the whole difference between an informed
 * decision and a shrug.  Nothing is stashed or set aside either way.
 */
async function _restore(sha, label, force) {
    try {
        const res = await _fetchJSON("POST", "/api/checkpoint/restore", {
            path:  _state.currentDir,
            state: sha,
            force: !!force,
        });
        if (res.body && res.body.ok) {
            await _refresh();
            /* THE FOLDER'S FILES JUST CHANGED UNDERNEATH EVERY OPEN TAB.
             * `_refresh()` above repaints THIS panel's own list and nothing
             * else, so a restore that swapped `task.json` for
             * `task.1st.json` left Task setup showing stages and a bench
             * the folder no longer has (reported 2026-08-24).  Announced
             * rather than reached-into: this panel does not know which tabs
             * are open or what they cache. */
            const _p = window.molbuilder && window.molbuilder.projects;
            if (_p && typeof _p.publishFolderChanged === "function") {
                _p.publishFolderChanged(_state.currentDir);
            }
            _showAdvisory(`This folder is now ${label}.`);
            return;
        }
        const advisory = res.body && Array.isArray(res.body.errors_only)
            && res.body.errors_only.length
            ? res.body.errors_only[0].message : null;
        if (advisory && !force) {
            if (confirm(advisory + "\n\nGo ahead and lose it?")) {
                await _restore(sha, label, true);
            } else {
                _showAdvisory("Nothing was changed.");
            }
            return;
        }
        _showAdvisory(advisory || ("Restore failed: " +
            (res.body?.error || "HTTP " + res.http)));
    } catch (e) {
        _showAdvisory("Restore failed: " + String(e && e.message || e));
    }
}

/* ---------- Advisory surface ---------- */

function _showAdvisory(message) {
    if (!elAdvisory) return;
    elAdvisory.textContent = message;
    elAdvisory.hidden      = false;
}

function _hideAdvisory() {
    if (!elAdvisory) return;
    elAdvisory.hidden = true;
    elAdvisory.textContent = "";
}

/* ---------- Public init ---------- */

/**
 * Wire the panel.  Idempotent; safe to call from multiple bootstrap
 * paths.  Returns true if the panel template was present and got
 * attached, false otherwise (i.e. running on a page without the
 * sidebar).
 */
export function initCheckpointPanel() {
    if (!_attach()) return false;
    if (window.molbuilder && window.molbuilder.projects
            && typeof window.molbuilder.projects.onChange === "function") {
        window.molbuilder.projects.onChange((info) => {
            const dir = info && info.dir || null;
            onDirectoryChange(dir);
        });
        // Also seed from the current state if a dir is already
        // selected when this module loads (slow page, deferred
        // script, etc.).
        if (typeof window.molbuilder.projects.getCurrentDir === "function") {
            const dir = window.molbuilder.projects.getCurrentDir();
            if (dir) onDirectoryChange(dir);
        }
    }
    return true;
}

/* ---------------------------------------------------------------- *
 *  The public API — `projects.checkpoint`                           *
 * ---------------------------------------------------------------- *
 * Contract: `docs/web/projects.md` § 5.  A sub-namespace on the one door,
 * the way `projects.parser` is.
 *
 * Before this the panel's save was a private click handler, so a tab that
 * had to take a state before writing (Task setup, `task-setup.md` § 8) had
 * two bad options: POST the route itself -- a second caller with its own
 * error handling and no sidebar refresh -- or reach into the panel's DOM.
 *
 * Restore and tag are NOT here, and that is deliberate: restoring rewinds a
 * folder and tagging writes in the namespace you are meant to be naming
 * things in yourself (`checkpointing.md` L4).  Both are decisions taken at
 * the panel, looking at the history.
 */

/** Does this folder have a history, and does it differ from where it stands?
 *
 * Served by GET /api/checkpoint/state and speaking ITS field names
 * (`initialized`).  Until 2026-08-19 this called /api/checkpoint/status --
 * a route that never existed -- and read a British-spelled field the server
 * never wrote, so every caller got `{ok:false}` forever and survived only
 * where `init` happened to be idempotent. */
export async function status(dir) {
    if (!dir) return { ok: false, error: "no folder given" };
    const res = await _fetchJSON("GET",
        "/api/checkpoint/state?path=" + encodeURIComponent(dir));
    const b = (res && res.body) || {};
    return {
        ok:          !!b.ok,
        initialized: !!b.initialized,
        clean:       b.clean !== false,
        error:       b.error || null,
    };
}

/** Start a history here.  Refuses with a reason a caller can show. */
export async function init(dir, opts) {
    if (!dir) return { ok: false, error: "no folder given" };
    const res = await _fetchJSON("POST", "/api/checkpoint/init",
        { path: dir, engine: (opts && opts.engine) || undefined });
    const b = (res && res.body) || {};
    const refusal = (Array.isArray(b.errors_only) && b.errors_only.length)
        ? b.errors_only[0] : null;
    return { ok: !!b.ok, error: b.error || refusal || null };
}

/**
 * Save the folder's current state.
 *
 * A NOTE IS REQUIRED.  `checkpointing.md` L4 retired automatic messages —
 * "a history where most tags are machine-made is one where your own are hard
 * to find" — so nothing writes one on your behalf, and this refuses rather
 * than inventing one.
 *
 * `changed: false` is HONEST, not a failure: nothing differed from the state
 * the folder already stands at.
 */
export async function saveState(dir, note) {
    if (!dir) return { ok: false, error: "no folder given" };
    const text = String(note == null ? "" : note).trim();
    if (!text) {
        return { ok: false, error: "A state needs a note. Nothing was saved." };
    }
    const res = await _fetchJSON("POST", "/api/checkpoint/save",
                                 { path: dir, note: text });
    const b = (res && res.body) || {};
    if (b.ok) {
        // The panel is showing this folder's history; keep it truthful.
        try { await _refresh(); } catch (_) { /* the state is saved either way */ }
    }
    return { ok: !!b.ok, changed: b.changed !== false, error: b.error || null };
}
