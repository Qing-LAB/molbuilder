/* projects/checkpoint.js -- Run-history panel inside the projects sidebar.
 *
 * Mounts the panel declared in templates/_projects_sidebar.html
 * (#ps-checkpoint).  Subscribes to projects.onChange to react when
 * the user navigates into a directory.  For each directory:
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
 * such a run dir, and (b) the manual Refresh control.  There is no
 * setInterval and no visibility-driven timer.
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
    /** True if the user has collapsed the panel via the chevron. */
    userCollapsed: false,
};

// DOM handles, populated by _attach().
let elPanel, elSensor, elCollapse, elEmpty, elInitBtn, elActions,
    elCommitBtn, elTagBtn, elRefreshBtn, elList, elGraph, elAdvisory,
    elViewListBtn, elViewGraphBtn;

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
    elCollapse   = document.getElementById("ps-checkpoint-collapse");
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

    if (!elPanel) return false;   // template not loaded; skip wiring

    // Restore view-mode preference from sessionStorage.
    try {
        const saved = sessionStorage.getItem("ws.ui.checkpoint.view");
        if (saved === "graph" || saved === "list") _viewMode = saved;
    } catch (_) { /* sessionStorage disabled — fall through to default */ }
    _updateViewButtons();

    // Restore collapse preference from sessionStorage -- the SAME mechanism as the
    // view-mode pref above.  (This used to read
    // ws.readPersistedSnapshot().state.ui.checkpoint.collapsed, but MolView's session
    // snapshot has no `ui` slot and nothing ever wrote it -- the restore was dead code
    // AND a projects-sidebar module reaching into MolView's persisted session, which
    // the workspace contract forbids.  A UI pref is per-origin sessionStorage, owned
    // here, not smuggled through the workspace.)
    try {
        if (sessionStorage.getItem("ps.checkpoint.collapsed") === "1") {
            _state.userCollapsed = true;
        }
    } catch (_) { /* sessionStorage disabled -- default to expanded */ }

    elCollapse.addEventListener("click", _onCollapseClick);
    elInitBtn.addEventListener("click", _onInitClick);
    elCommitBtn.addEventListener("click", _onCommitClick);
    elTagBtn.addEventListener("click", _onTagClick);
    elRefreshBtn.addEventListener("click", _refresh);
    elSensor.addEventListener("click", _refresh);
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
    if (_state.userCollapsed) {
        elPanel.hidden = false;
        _renderCollapsedHeader();
        return;
    }
    elPanel.hidden = false;
    _refresh();
}

/* ---------- Internal: state-driven rendering ---------- */

function _hide() {
    if (!elPanel) return;
    elPanel.hidden = true;
}

function _renderCollapsedHeader() {
    // Show only the header + sensor pill; hide the rest.  Used when
    // the user has explicitly collapsed the panel.
    elEmpty.hidden    = true;
    elActions.hidden  = true;
    elList.hidden     = true;
    if (elGraph) elGraph.hidden = true;
    elAdvisory.hidden = true;
}

function _renderState(repoState) {
    _state.repoState = repoState;
    if (!repoState || !repoState.initialized) {
        elSensor.textContent = "no states saved";
        elSensor.setAttribute("data-state", "uninit");
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
    } else {
        elSensor.textContent = "saved";
        elSensor.setAttribute("data-state", "clean");
        elSensor.title = repoState.standing_at
            ? `standing at ${repoState.standing_at.short}`
            : "";
    }
    elEmpty.hidden   = true;
    elActions.hidden = false;
}

function _renderError(message) {
    elSensor.textContent = "error";
    elSensor.setAttribute("data-state", "error");
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

async function _refresh() {
    if (!_state.currentDir) return;
    _hideAdvisory();
    try {
        const stRes = await _fetchJSON("GET",
            `/api/checkpoint/state?path=${encodeURIComponent(_state.currentDir)}`);
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

        if (stRes.body.initialized) {
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

function _onCollapseClick() {
    _state.userCollapsed = !_state.userCollapsed;
    elCollapse.textContent = _state.userCollapsed ? "▸" : "▾";
    elCollapse.setAttribute("aria-expanded", String(!_state.userCollapsed));
    // Persist the preference (per-origin sessionStorage; mirrors the view-mode pref)
    // so it survives a reload -- the restore in _attach reads this key.
    try {
        if (_state.userCollapsed) sessionStorage.setItem("ps.checkpoint.collapsed", "1");
        else sessionStorage.removeItem("ps.checkpoint.collapsed");
    } catch (_) { /* sessionStorage disabled -- pref just won't persist */ }
    if (_state.userCollapsed) {
        _renderCollapsedHeader();
    } else {
        _refresh();
    }
}

async function _onInitClick() {
    if (!_state.currentDir) return;
    _hideAdvisory();
    elInitBtn.disabled = true;
    try {
        const res = await _fetchJSON("POST", "/api/checkpoint/init",
            { path: _state.currentDir });
        if (res.body && res.body.ok) {
            await _refresh();
        } else if (res.body && Array.isArray(res.body.errors_only)
                   && res.body.errors_only.length) {
            // Bucket B advisory (web-api.md § 1): surface inline.
            _showAdvisory(res.body.errors_only[0].message);
        } else {
            _showAdvisory("Init failed: " +
                (res.body?.error || "HTTP " + res.http));
        }
    } catch (e) {
        _showAdvisory("Init failed: " + String(e?.message || e));
    } finally {
        elInitBtn.disabled = false;
    }
}

async function _onCommitClick() {
    if (!_state.currentDir) return;
    const msg = prompt(
        "Checkpoint message (leave blank for ISO timestamp):", "");
    if (msg === null) return;     // user cancelled
    _hideAdvisory();
    elCommitBtn.disabled = true;
    try {
        const res = await _fetchJSON("POST", "/api/checkpoint/save", {
            path:    _state.currentDir,
            message: msg.trim(),
        });
        if (res.body && res.body.ok) {
            await _refresh();
            if (res.body.checkpoint === null) {
                _showAdvisory(res.body.note ||
                    "Nothing to checkpoint (working tree clean).");
            }
        } else {
            _showAdvisory("Checkpoint failed: " +
                (res.body?.error || "HTTP " + res.http));
        }
    } catch (e) {
        _showAdvisory("Checkpoint failed: " + String(e?.message || e));
    } finally {
        elCommitBtn.disabled = false;
    }
}

async function _onTagClick() {
    if (!_state.currentDir) return;
    const label = prompt("Tag label (e.g. stage3-converged):", "");
    if (!label || !label.trim()) return;
    const message = prompt(
        "Tag message (required; the audit trail wants meaning):", "");
    if (message === null) return;
    if (!message.trim()) {
        _showAdvisory("Tag message is required.");
        return;
    }
    _hideAdvisory();
    elTagBtn.disabled = true;
    try {
        const res = await _fetchJSON("POST", "/api/checkpoint/tag", {
            path:    _state.currentDir,
            label:   label.trim(),
            message: message.trim(),
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
