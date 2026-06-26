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
 * Sensor poll cadence: 5 s (per design § 11.7), only when document is
 * visible AND the panel is expanded AND a directory is selected.
 *
 * No graph viewer in Phase 2 -- that's PR-B Phase 3 with @gitgraph/js.
 *
 * Spec: docs/protocols/run-checkpoints.md § 6 (sidebar UI).
 * HTTP contract: web/blueprints/checkpoint.py + § 8 of run-checkpoints.md.
 */

const POLL_MS = 5000;          // § 11.7 sensor cadence
const _state = {
    /** Currently selected directory path (relative to projects root,
     *  resolved to absolute by the API).  null when no dir selected. */
    currentDir:    null,
    /** Latest /api/checkpoint/state snapshot (or null on init). */
    repoState:     null,
    /** Cached checkpoints list to avoid re-rendering on every poll. */
    checkpoints:   [],
    /** Poll handle (setInterval id) for the sensor. */
    pollHandle:    null,
    /** True if the user has collapsed the panel via the chevron. */
    userCollapsed: false,
};

// DOM handles, populated by _attach().
let elPanel, elSensor, elCollapse, elEmpty, elInitBtn, elActions,
    elCommitBtn, elTagBtn, elRefreshBtn, elList, elAdvisory;

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
    elList       = document.getElementById("ps-checkpoint-list");
    elAdvisory   = document.getElementById("ps-checkpoint-advisory");

    if (!elPanel) return false;   // template not loaded; skip wiring

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

    // Suspend polling when the document is hidden -- the user isn't
    // looking; saves a request per cadence interval per tab.
    document.addEventListener("visibilitychange", _maybePoll);

    return true;
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
    _state.currentDir = dirPath;
    if (!dirPath) {
        _hide();
        return;
    }
    if (_state.userCollapsed) {
        elPanel.hidden = false;
        _renderCollapsedHeader();
        return;
    }
    elPanel.hidden = false;
    _refresh();
    _maybePoll();
}

/* ---------- Internal: state-driven rendering ---------- */

function _hide() {
    if (!elPanel) return;
    elPanel.hidden = true;
    _stopPoll();
}

function _renderCollapsedHeader() {
    // Show only the header + sensor pill; hide the rest.  Used when
    // the user has explicitly collapsed the panel.
    elEmpty.hidden    = true;
    elActions.hidden  = true;
    elList.hidden     = true;
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
    elList.innerHTML = "";
    elList.hidden    = false;
    for (const cp of _state.checkpoints) {
        elList.appendChild(_buildRow(cp));
    }
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
        }
    } catch (e) {
        _renderError(String(e && e.message || e));
    }
}

/* ---------- Polling ---------- */

function _maybePoll() {
    _stopPoll();
    if (document.hidden) return;
    if (!_state.currentDir) return;
    if (_state.userCollapsed) return;
    _state.pollHandle = setInterval(_refresh, POLL_MS);
}

function _stopPoll() {
    if (_state.pollHandle != null) {
        clearInterval(_state.pollHandle);
        _state.pollHandle = null;
    }
}

/* ---------- Action handlers ---------- */

function _onCollapseClick() {
    _state.userCollapsed = !_state.userCollapsed;
    elCollapse.textContent = _state.userCollapsed ? "▸" : "▾";
    elCollapse.setAttribute("aria-expanded", String(!_state.userCollapsed));
    if (_state.userCollapsed) {
        _renderCollapsedHeader();
        _stopPoll();
    } else {
        _refresh();
        _maybePoll();
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
        } else if (res.body && res.body.errors_only) {
            // Bucket B advisory: surface inline.
            _showAdvisory(res.body.errors[0].message);
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
        } else if (res.body && res.body.errors_only) {
            _showAdvisory(res.body.errors[0].message);
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
