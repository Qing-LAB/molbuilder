/* projects/checkpoint.js -- Run-history panel inside the projects sidebar.
 *
 * Mounts the panel declared in templates/_projects_sidebar.html
 * (#ps-checkpoint).  Subscribes to projects.onChange to react when
 * the user navigates into a directory.  For each directory:
 *
 *   1. GET  /api/checkpoint/state    -> sensor pill + visibility decision
 *   2. GET  /api/checkpoint/list     -> populate row list when init'd
 *   3. POST /api/checkpoint/init     -> empty-state CTA
 *   4. POST /api/checkpoint/commit   -> "Checkpoint now" button
 *   5. POST /api/checkpoint/tag      -> "Tag HEAD…" button
 *   6. POST /api/checkpoint/restore  -> per-row "Restore" button
 *
 * Activation gate (run-checkpoints.md § 6.1): the panel only appears
 * for a *run directory* -- a dir at projects rel-depth 3, in the
 * canonical layout projects/PROJECT/CATEGORY/RUNNING_DIR.  Selecting
 * anything shallower (or a file) hides the panel entirely.
 *
 * Refresh model (run-checkpoints.md § 6.2, § 11.7): explicit only --
 * NO background polling.  State refreshes on (a) directory-enter into
 * such a run dir, and (b) the manual Refresh control.  There is no
 * setInterval and no visibility-driven timer.
 *
 * No graph viewer in Phase 2 -- that's PR-B Phase 3 with @gitgraph/js.
 *
 * Spec: docs/protocols/run-checkpoints.md § 6 (sidebar UI).
 * HTTP contract: web/blueprints/checkpoint.py + § 8 of run-checkpoints.md.
 */

// projects rel-depth of a canonical run dir:
// projects/PROJECT_NAME/CATEGORY/RUNNING_DIR_NAME  (§ 6.1).
const RUN_DIR_DEPTH = 3;
const _state = {
    /** Currently selected directory path (relative to projects root,
     *  resolved to absolute by the API).  null when no dir selected. */
    currentDir:    null,
    /** Latest /api/checkpoint/state snapshot (or null on init). */
    repoState:     null,
    /** Cached checkpoints list to avoid re-rendering on every refresh. */
    checkpoints:   [],
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

    // Restore collapse state from sessionStorage (per workspace-
    // contract.md § 4.1; ws.ui.checkpoint.collapsed is the key).
    const ws = window.molbuilder && window.molbuilder.workspace;
    if (ws && typeof ws.readPersistedSnapshot === "function") {
        const snap = ws.readPersistedSnapshot();
        if (snap && snap.state && snap.state.ui
                && snap.state.ui.checkpoint
                && snap.state.ui.checkpoint.collapsed === true) {
            _state.userCollapsed = true;
        }
    }

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
 * a checkpoint viewer activates (run-checkpoints.md § 6.1).  The
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
    // (run-checkpoints.md § 6.1).
    if (!_isRunDir(dirPath)) {
        _state.currentDir = null;
        _hide();
        return;
    }
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
        elSensor.textContent = "no checkpoints";
        elSensor.setAttribute("data-state", "uninit");
        elEmpty.hidden    = false;
        elActions.hidden  = true;
        elList.hidden     = true;
        if (elGraph) elGraph.hidden = true;
        return;
    }
    if (repoState.dirty) {
        const n = repoState.untracked || 0;
        elSensor.textContent = `${n > 0 ? n + " new + " : ""}dirty`;
        elSensor.setAttribute("data-state", "dirty");
    } else {
        elSensor.textContent = "clean";
        elSensor.setAttribute("data-state", "clean");
    }
    elEmpty.hidden   = true;
    elActions.hidden = false;
}

function _renderError(message) {
    elSensor.textContent = "error";
    elSensor.setAttribute("data-state", "error");
    _showAdvisory("Sensor error: " + message);
}

function _renderCheckpoints(checkpoints) {
    _state.checkpoints = checkpoints || [];
    _renderActiveView(_state.checkpoints);
}

function _buildRow(cp) {
    const li = document.createElement("li");
    li.className     = "ps-checkpoint-list-item";
    li.dataset.sha   = cp.sha;

    const sha = document.createElement("span");
    sha.className   = "ps-checkpoint-row-sha";
    sha.textContent = cp.short_sha;
    li.appendChild(sha);

    const summary = document.createElement("span");
    summary.className   = "ps-checkpoint-row-summary";
    summary.textContent = cp.summary || "(no message)";
    li.appendChild(summary);

    if (cp.refs && cp.refs.length) {
        const refs = document.createElement("span");
        refs.className = "ps-checkpoint-row-refs";
        for (const r of cp.refs) {
            const chip = document.createElement("span");
            chip.className = "ps-checkpoint-ref-chip";
            const t = r.trim();
            if (t.startsWith("tag:")) {
                chip.setAttribute("data-kind", "tag");
                chip.textContent = t.slice(4).trim();
            } else if (t === "HEAD -> master" || t === "HEAD") {
                chip.setAttribute("data-kind", "branch");
                chip.textContent = "HEAD";
            } else {
                chip.setAttribute("data-kind", "branch");
                chip.textContent = t.replace(/^HEAD -> /, "");
            }
            refs.appendChild(chip);
        }
        li.appendChild(refs);
    }

    if (cp.has_archive && cp.archive_bytes != null) {
        const arch = document.createElement("span");
        arch.className   = "ps-checkpoint-row-archive";
        arch.textContent = _fmtBytes(cp.archive_bytes) + " archived";
        li.appendChild(arch);
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
    _renderActiveView(_state.checkpoints);
}

function _renderActiveView(checkpoints) {
    // Hide both, then show the active one (single source of layout
    // truth -- no race between toggles).
    if (elList)  elList.hidden  = true;
    if (elGraph) elGraph.hidden = true;
    if (_viewMode === "graph") {
        if (elGraph) elGraph.hidden = false;
        _renderGraph(checkpoints);
    } else {
        if (elList)  elList.hidden  = false;
        _renderListRows(checkpoints);
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

async function _renderGraph(checkpoints) {
    if (!elGraph) return;
    elGraph.innerHTML = "";
    if (!checkpoints || !checkpoints.length) {
        const empty = document.createElement("p");
        empty.className   = "ps-checkpoint-graph-empty";
        empty.textContent = "(no checkpoints to graph)";
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

    // @gitgraph/js wants commits in chronological order (oldest first).
    // Our /api/checkpoint/list returns newest-first, so reverse.
    const ordered = checkpoints.slice().reverse();

    // Build per-branch hashes-known map so we can wire merges if any.
    // PR-A's CLI is single-branch by default; this is a no-op for
    // linear histories, but ready for the day branches appear.
    const branches = new Map();
    let mainBranch = null;
    for (const cp of ordered) {
        // Determine the branch this commit lives on (best-effort
        // from ref decorations).  Default to "main".
        let branchName = "main";
        for (const r of (cp.refs || [])) {
            const m = r.trim().match(/^([^:>\s]+)$/);
            if (m && m[1] !== "HEAD" && m[1] !== "tag") {
                branchName = m[1];
                break;
            }
        }
        if (!branches.has(branchName)) {
            const b = gitgraph.branch(branchName);
            branches.set(branchName, b);
            if (!mainBranch) mainBranch = b;
        }
        const b = branches.get(branchName);
        b.commit({
            subject: cp.summary || cp.short_sha,
            hash:    cp.short_sha,
            onClick: () => _showCommitDetail(cp),
        });
        // Decorate with tag chips inline.
        for (const r of (cp.refs || [])) {
            const t = r.trim();
            if (t.startsWith("tag:")) {
                b.tag(t.slice(4).trim());
            }
        }
    }
}

function _showCommitDetail(cp) {
    // Click on a graph node = inline advisory line with the full
    // commit info.  Keeps the panel chrome minimal (no popover
    // library); user can then act via the list view or sidebar
    // context menu in future Phase 4 work.
    const refs = (cp.refs || []).join(", ");
    const arch = cp.has_archive
        ? `${_fmtBytes(cp.archive_bytes)} archived`
        : "no binaries archived";
    _showAdvisory(
        `${cp.short_sha} — ${cp.summary || "(no message)"} ` +
        `[${refs || "no refs"}] · ${arch}`);
}

/* ---------- List view extraction (so View toggle can call it) ---------- */

function _renderListRows(checkpoints) {
    if (!elList) return;
    elList.innerHTML = "";
    for (const cp of (checkpoints || [])) {
        elList.appendChild(_buildRow(cp));
    }
}

function _fmtBytes(n) {
    if (n == null) return "";
    if (n >= 1024 * 1024) return (n / (1024 * 1024)).toFixed(1) + " MB";
    if (n >= 1024)        return (n / 1024).toFixed(1) + " KB";
    return n + " B";
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
        _renderState(stRes.body.state);

        if (stRes.body.state && stRes.body.state.initialized) {
            const lsRes = await _fetchJSON("GET",
                `/api/checkpoint/list?path=${encodeURIComponent(_state.currentDir)}&limit=50`);
            if (lsRes.body && lsRes.body.ok && lsRes.body.checkpoints) {
                _renderCheckpoints(lsRes.body.checkpoints);
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
            // Bucket B advisory (web-api.md § 1.6): surface inline.
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
        const res = await _fetchJSON("POST", "/api/checkpoint/commit", {
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
    const cp = _state.checkpoints.find(c => c.sha === sha);
    const label = cp && cp.refs && cp.refs.length
        ? cp.refs[0].replace(/^tag:\s*/, "").replace(/^HEAD -> /, "")
        : cp ? cp.short_sha : sha.slice(0, 7);
    if (!confirm(
        `Restore working tree to ${label}?\n\n` +
        `This rewinds text files via git restore AND copies archived\n` +
        `binaries (.DM, .HSX, ...) back over the working dir.  Refuses\n` +
        `if there are uncommitted changes.`)) {
        return;
    }
    _hideAdvisory();
    try {
        const res = await _fetchJSON("POST", "/api/checkpoint/restore", {
            path: _state.currentDir,
            ref:  sha,
        });
        if (res.body && res.body.ok) {
            await _refresh();
            _showAdvisory(
                `Restored ${label}.  ` +
                (res.body.restored && res.body.restored.length
                    ? `Binaries: ${res.body.restored.join(", ")}.`
                    : "Text only (no archived binaries for this ref)."));
        } else if (res.body && Array.isArray(res.body.errors_only)
                   && res.body.errors_only.length) {
            _showAdvisory(res.body.errors_only[0].message);
        } else {
            _showAdvisory("Restore failed: " +
                (res.body?.error || "HTTP " + res.http));
        }
    } catch (e) {
        _showAdvisory("Restore failed: " + String(e?.message || e));
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
