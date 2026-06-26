/* Markdown inspector for the /results tab.
 *
 * Split-pane: CodeMirror 5 (markdown mode, left) + DOMPurify-sanitised
 * marked.js render (right).  Explicit Save button posts to
 * /api/files/write with expected_mtime so a concurrent edit returns
 * 409 instead of silently clobbering.  Ctrl-S triggers Save inside
 * the editor.
 *
 * Contract — registry.js § "Handle contract":
 *   mount(host, file, ctx) returns SYNCHRONOUSLY a handle with
 *   ``dispose()``; the dispatcher calls dispose() before mounting the
 *   next inspector.  We do all async work (lazy library load, file
 *   read) inside the mount body, paint a "Loading…" status, and the
 *   handle's dispose() flips an ``aborted`` flag so late callbacks
 *   become no-ops.
 *
 * Spec: task #32 (2026-06-26).  Save-flow contract:
 *   docs/protocols/save-flow.md § 1.2 (edit-save uses expected_mtime).
 */
(function (root) {
    "use strict";

    let _libsPromise = null;

    /** Lazy-load CodeMirror core + markdown mode + marked + DOMPurify
     *  once.  Cached promise — first call kicks off the network fetch,
     *  every subsequent call waits on the cached promise. */
    function _loadLibs() {
        if (_libsPromise) return _libsPromise;
        _libsPromise = (async () => {
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
            if (!window.DOMPurify) {
                await loadScript("/static/vendor/dompurify/purify.min.js");
            }
        })();
        return _libsPromise;
    }

    /** Render text -> sanitised HTML.  Single render path so a future
     *  switch to a different sanitiser (or a custom marked extension)
     *  has one site to update. */
    function _renderToHTML(text) {
        const raw = window.marked.parse(text, {
            breaks: false,
            gfm:    true,
        });
        // DOMPurify defaults strip <script>, on* attributes, javascript:
        // URLs, and iframes.  Configure to keep ``target=_blank`` for
        // links (the GFM tables / lists / code blocks we use don't
        // need anything beyond the default allow-list).
        return window.DOMPurify.sanitize(raw, {
            ADD_ATTR: ["target"],
        });
    }

    const inspector = {
        name:        "markdown",
        displayName: "Markdown notes",
        isResult:    false,
        match: (file) => file.toLowerCase().endsWith(".md"),

        mount(host, file, ctx) {
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

            elTitle.textContent  = file.split("/").pop();
            elStatus.textContent = "Loading…";

            // State captured by the handle's dispose() so late
            // callbacks become no-ops after the user navigates away.
            let aborted     = false;
            let cm          = null;
            let renderTimer = null;
            let dirty       = false;
            let mtime       = null;

            // The Save handler is shared between Ctrl-S and the button
            // click.  Re-entrant-safe via the elSave.disabled check.
            async function save() {
                if (aborted || !dirty || elSave.disabled) return;
                elSave.disabled      = true;
                elStatus.textContent = "Saving…";
                let r;
                try {
                    r = await fetch("/api/files/write", {
                        method:  "POST",
                        headers: { "Content-Type": "application/json" },
                        body:    JSON.stringify({
                            path:            file,
                            text:            cm ? cm.getValue() : "",
                            expected_mtime:  mtime,
                        }),
                    });
                } catch (e) {
                    if (aborted) return;
                    elStatus.textContent = "Save failed: " +
                        String(e && e.message || e);
                    elSave.disabled = false;
                    return;
                }
                if (aborted) return;
                // CHECK STATUS BEFORE PARSING -- a 409 response body
                // shape isn't guaranteed; we surface the conflict
                // regardless of whether the server replied with JSON.
                if (r.status === 409) {
                    elStatus.textContent =
                        "Conflict: this file was modified on disk "
                        + "since you opened it.  Reload to see the "
                        + "current version (your edits will be "
                        + "lost), or save under a new name.";
                    elSave.disabled = false;
                    return;
                }
                let body = null;
                try { body = await r.json(); } catch (_) { /* swallow */ }
                if (aborted) return;
                if (!body || !body.ok) {
                    elStatus.textContent = "Save failed: " +
                        (body && body.error ? body.error :
                            "HTTP " + r.status);
                    elSave.disabled = false;
                    return;
                }
                mtime          = body.mtime || null;
                dirty          = false;
                elDirty.hidden = true;
                elSave.disabled = true;   // explicit: nothing to save now
                elStatus.textContent = "Saved at " +
                    new Date().toLocaleTimeString();
            }

            // Browser navigation warning if there are unsaved edits.
            const beforeunload = (ev) => {
                if (dirty && !aborted) {
                    ev.preventDefault();
                    ev.returnValue = "";
                }
            };
            window.addEventListener("beforeunload", beforeunload);
            const onSaveClick = () => save();
            elSave.addEventListener("click", onSaveClick);

            // Kick off async work.  Aborted flag flips to true if the
            // user navigates away mid-load; everything below short-
            // circuits on it.
            (async () => {
                // 1. Read file + capture mtime.
                let initialText = "";
                try {
                    const r = await fetch("/api/files/read?path=" +
                        encodeURIComponent(file));
                    if (aborted) return;
                    const body = await r.json();
                    if (aborted) return;
                    if (!body.ok) {
                        elStatus.textContent = "Read failed: " +
                            (body.error || ("HTTP " + r.status));
                        return;
                    }
                    initialText = body.text || "";
                    mtime       = body.mtime || null;
                } catch (e) {
                    if (aborted) return;
                    elStatus.textContent = "Read failed: " +
                        String(e && e.message || e);
                    return;
                }

                // 2. Lazy-load libraries.
                try {
                    await _loadLibs();
                } catch (e) {
                    if (aborted) return;
                    elStatus.textContent = "Editor unavailable: " +
                        String(e && e.message || e);
                    return;
                }
                if (aborted) return;

                // 3. Mount CodeMirror with shared Save handler.
                cm = window.CodeMirror(elEdit, {
                    value:          initialText,
                    mode:           "markdown",
                    lineNumbers:    true,
                    lineWrapping:   true,
                    viewportMargin: Infinity,
                    extraKeys: {
                        "Ctrl-S": save,
                        "Cmd-S":  save,
                    },
                });

                // 4. Initial render + live preview pipeline.
                const render = () => {
                    if (aborted) return;
                    try {
                        elRender.innerHTML = _renderToHTML(cm.getValue());
                    } catch (e) {
                        elRender.textContent = "Render error: " +
                            (e && e.message || e);
                    }
                };
                render();
                cm.on("change", () => {
                    if (aborted) return;
                    if (!dirty) {
                        dirty = true;
                        elDirty.hidden  = false;
                        elSave.disabled = false;
                    }
                    clearTimeout(renderTimer);
                    renderTimer = setTimeout(render, 200);
                });

                elStatus.textContent = "Loaded.  Edit on the left; "
                    + "preview on the right.";
            })();

            // Synchronous handle per registry.js § "Handle contract".
            return {
                dispose() {
                    aborted = true;
                    clearTimeout(renderTimer);
                    window.removeEventListener(
                        "beforeunload", beforeunload);
                    elSave.removeEventListener("click", onSaveClick);
                    // CodeMirror has no explicit destroy(); detaching
                    // its DOM is enough for GC to reclaim.
                    host.innerHTML = "";
                },
            };
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
