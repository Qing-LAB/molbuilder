/* Markdown inspector for the /results tab.
 *
 * Split-pane: CodeMirror 5 (markdown mode, left) + marked.js rendered
 * HTML (right).  Explicit Save button posts to /api/files/write with
 * expected_mtime so a concurrent edit returns 409 instead of silently
 * clobbering.  Ctrl-S triggers Save inside the editor.
 *
 * Registers BEFORE the generic source inspector (lib/inspectors/
 * source.js) so .md files dispatch here -- the source inspector's
 * match still includes .md but the registry order favours earlier
 * registrations for ties (see lib/inspectors/registry.js § match).
 *
 * Lazy-loads CodeMirror 5 + the markdown mode + marked.js on first
 * mount.  Subsequent .md views reuse the cached globals.
 *
 * Spec: task #32 (2026-06-26).  Save-flow contract:
 *   docs/protocols/save-flow.md § 1.2 (edit-save uses expected_mtime).
 */
(function (root) {
    "use strict";

    let _libsPromise = null;

    /** Lazy-load CodeMirror core + markdown mode + marked.js once. */
    function _loadLibs() {
        if (_libsPromise) return _libsPromise;
        _libsPromise = new Promise((resolve, reject) => {
            const loadScript = (src) => new Promise((ok, ko) => {
                const t = document.createElement("script");
                t.src = src;
                t.onload  = () => ok();
                t.onerror = () => ko(new Error("failed to load " + src));
                document.head.appendChild(t);
            });
            const loadCSS = (href) => new Promise((ok) => {
                const l = document.createElement("link");
                l.rel  = "stylesheet";
                l.href = href;
                l.onload = () => ok();
                document.head.appendChild(l);
            });
            (async () => {
                if (!window.CodeMirror) {
                    await loadCSS("/static/vendor/codemirror/codemirror.min.css");
                    await loadScript("/static/vendor/codemirror/codemirror.min.js");
                }
                if (!window.CodeMirror.modes.markdown) {
                    await loadScript("/static/vendor/codemirror/markdown.min.js");
                }
                if (!window.marked) {
                    await loadScript("/static/vendor/marked/marked.min.js");
                }
                resolve();
            })().catch(reject);
        });
        return _libsPromise;
    }

    const inspector = {
        name:        "markdown",
        displayName: "Markdown notes",
        isResult:    false,
        match: (file) => file.toLowerCase().endsWith(".md"),

        async mount(host, file, ctx) {
            host.innerHTML = "";
            const wrap = document.createElement("div");
            wrap.className = "md-inspector";
            wrap.innerHTML = `
              <div class="md-toolbar">
                <span class="md-toolbar-title"></span>
                <span class="md-toolbar-spacer"></span>
                <span class="md-dirty-flag" hidden>● unsaved</span>
                <button type="button" class="md-save-btn" disabled>Save (Ctrl-S)</button>
              </div>
              <div class="md-split">
                <div class="md-edit-pane"></div>
                <div class="md-render-pane"></div>
              </div>
              <div class="md-status" aria-live="polite"></div>
            `;
            host.appendChild(wrap);
            const elTitle  = wrap.querySelector(".md-toolbar-title");
            const elEdit   = wrap.querySelector(".md-edit-pane");
            const elRender = wrap.querySelector(".md-render-pane");
            const elStatus = wrap.querySelector(".md-status");
            const elDirty  = wrap.querySelector(".md-dirty-flag");
            const elSave   = wrap.querySelector(".md-save-btn");

            elTitle.textContent = file.split("/").pop();
            elStatus.textContent = "Loading…";

            // 1. Load text + capture mtime for conflict detection.
            let initialText = "";
            let mtime       = null;
            try {
                const r = await fetch("/api/files/read?path=" +
                    encodeURIComponent(file));
                const body = await r.json();
                if (!body.ok) {
                    elStatus.textContent = "Read failed: " +
                        (body.error || ("HTTP " + r.status));
                    return;
                }
                initialText = body.text || "";
                mtime       = body.mtime || null;
            } catch (e) {
                elStatus.textContent = "Read failed: " +
                    String(e && e.message || e);
                return;
            }

            // 2. Lazy-load CodeMirror + marked.
            try {
                await _loadLibs();
            } catch (e) {
                elStatus.textContent = "Editor unavailable: " +
                    String(e && e.message || e);
                return;
            }

            // 3. Mount CodeMirror.
            const cm = window.CodeMirror(elEdit, {
                value:        initialText,
                mode:         "markdown",
                lineNumbers:  true,
                lineWrapping: true,
                viewportMargin: Infinity,
                extraKeys: {
                    "Ctrl-S":   () => save(),
                    "Cmd-S":    () => save(),
                },
            });

            // 4. Initial render.
            const render = () => {
                try {
                    elRender.innerHTML = window.marked.parse(cm.getValue(), {
                        breaks: false,
                        gfm:    true,
                    });
                } catch (e) {
                    elRender.textContent = "Render error: " + e.message;
                }
            };
            render();
            elStatus.textContent = "Loaded.  Edit on the left; "
                + "preview on the right.";

            // 5. Live preview (debounced) + dirty tracking.
            let renderTimer  = null;
            let dirty        = false;
            cm.on("change", () => {
                if (!dirty) {
                    dirty = true;
                    elDirty.hidden = false;
                    elSave.disabled = false;
                }
                clearTimeout(renderTimer);
                renderTimer = setTimeout(render, 200);
            });

            // 6. Save handler — POST /api/files/write with expected_mtime.
            async function save() {
                if (!dirty) return;
                elSave.disabled = true;
                elStatus.textContent = "Saving…";
                try {
                    const r = await fetch("/api/files/write", {
                        method:  "POST",
                        headers: { "Content-Type": "application/json" },
                        body:    JSON.stringify({
                            path:            file,
                            text:            cm.getValue(),
                            expected_mtime:  mtime,
                        }),
                    });
                    const body = await r.json();
                    if (r.status === 409) {
                        elStatus.textContent =
                            "Conflict: this file was modified on disk "
                            + "since you opened it.  Reload to see the "
                            + "current version (your edits will be "
                            + "lost), or save under a new name.";
                        elSave.disabled = false;
                        return;
                    }
                    if (!body.ok) {
                        elStatus.textContent = "Save failed: " +
                            (body.error || ("HTTP " + r.status));
                        elSave.disabled = false;
                        return;
                    }
                    mtime         = body.mtime || null;
                    dirty         = false;
                    elDirty.hidden = true;
                    elStatus.textContent = "Saved at " +
                        new Date().toLocaleTimeString();
                } catch (e) {
                    elStatus.textContent = "Save failed: " +
                        String(e && e.message || e);
                    elSave.disabled = false;
                }
            }
            elSave.addEventListener("click", save);

            // Warn the user if they navigate away with unsaved edits.
            const beforeunload = (ev) => {
                if (dirty) {
                    ev.preventDefault();
                    ev.returnValue = "";
                }
            };
            window.addEventListener("beforeunload", beforeunload);
            // Cleanup hook for the registry to call when remounting.
            wrap._cleanup = () =>
                window.removeEventListener("beforeunload", beforeunload);
        },
    };

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.inspectors = root.molbuilder.inspectors || {};
    if (typeof root.molbuilder.inspectors.register === "function") {
        // Register BEFORE source.js by relying on the registry's
        // first-match-wins-for-ties policy (registry.js § match).
        root.molbuilder.inspectors.register(inspector);
    } else {
        // Registry not loaded yet -- queue.
        root.molbuilder.inspectors._pending =
            root.molbuilder.inspectors._pending || [];
        root.molbuilder.inspectors._pending.push(inspector);
    }
})(typeof window !== "undefined" ? window : globalThis);
