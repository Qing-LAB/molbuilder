/* Documents tab — browse + read every docs/*.md in-app.
 *
 * Left: a grouped list from /api/docs/list.  Right: the picked doc, fetched
 * from /api/docs/read and rendered through the shared markdown renderer
 * (lib/markdown-render.js — the ONE sanitise policy).  Read-only: no editor,
 * no save.  Deep-linkable via the ``?doc=<rel-path>`` query + hash.
 */
(function () {
    "use strict";

    var listEl   = document.getElementById("docs-list");
    var statusEl = document.getElementById("docs-list-status");
    var emptyEl  = document.getElementById("docs-view-empty");
    var renderEl = document.getElementById("docs-render");

    var _activeLink = null;

    function _setStatus(msg) {
        if (statusEl) statusEl.textContent = msg;
    }

    /** Put a plain-text status/message into the render pane WITHOUT innerHTML
     *  (textContent can't inject markup) -- used for loading + error states so
     *  the only innerHTML in this file is the DOMPurify-sanitised render. */
    function _showMessage(cls, msg) {
        renderEl.textContent = "";
        var p = document.createElement("p");
        p.className = cls;
        p.textContent = msg;
        renderEl.appendChild(p);
    }

    /** Fetch + render one doc into the right pane. */
    async function openDoc(path, link) {
        if (_activeLink) _activeLink.classList.remove("is-active");
        if (link) { link.classList.add("is-active"); _activeLink = link; }

        emptyEl.hidden = true;
        renderEl.hidden = false;
        _showMessage("docs-loading", "Loading…");

        var r;
        try {
            r = await fetch("/api/docs/read?path=" + encodeURIComponent(path))
                    .then(function (x) { return x.json(); });
        } catch (e) {
            _showMessage("docs-error", "Network error: "
                + (e && e.message ? e.message : String(e)));
            return;
        }
        if (!r || !r.ok) {
            _showMessage("docs-error",
                (r && r.error) || "Could not load document.");
            return;
        }
        try {
            await window.molbuilder.markdownRender.loadRenderLibs();
            // The ONE innerHTML in this file.  RHS is markdownRender.render(),
            // the shared marked + DOMPurify pipeline (lib/markdown-render.js):
            // the HTML is sanitised (scripts / on* / javascript: stripped)
            // before it reaches the DOM.  Source is app-shipped docs/*.md.
            // Allowlisted in tests/test_xss_audit.py.
            renderEl.innerHTML = window.molbuilder.markdownRender.render(r.text);
        } catch (e) {
            _showMessage("docs-error", "Render failed: "
                + (e && e.message ? e.message : String(e)));
            return;
        }
        // Links inside docs open in a new tab (many are external / cross-doc).
        renderEl.querySelectorAll("a[href]").forEach(function (a) {
            a.setAttribute("target", "_blank");
            a.setAttribute("rel", "noopener noreferrer");
        });
        // Render any ```mermaid diagrams into SVG in place (lazy-loads mermaid
        // only when the doc actually has one).  Non-fatal: a diagram failure
        // leaves the rest of the doc intact.
        try {
            await window.molbuilder.markdownRender.renderMermaidIn(renderEl);
        } catch (_) { /* one bad diagram must not blank the doc */ }
        renderEl.scrollTop = 0;
        // Reflect the selection in the URL so a doc is shareable / reloadable.
        try {
            var u = new URL(window.location.href);
            u.searchParams.set("doc", path);
            window.history.replaceState(null, "", u.toString());
        } catch (_) { /* history API absent — non-fatal */ }
    }

    /** Build the grouped list of docs in the left rail. */
    function renderList(groups) {
        listEl.innerHTML = "";
        var linksByPath = {};
        groups.forEach(function (g) {
            var section = document.createElement("div");
            section.className = "docs-group";
            var h = document.createElement("div");
            h.className = "docs-group-name";
            h.textContent = g.name;
            section.appendChild(h);

            g.docs.forEach(function (doc) {
                var a = document.createElement("a");
                a.className = "docs-link";
                a.href = "?doc=" + encodeURIComponent(doc.path);
                a.textContent = doc.title;
                a.title = doc.path;
                a.addEventListener("click", function (ev) {
                    ev.preventDefault();
                    openDoc(doc.path, a);
                });
                linksByPath[doc.path] = a;
                section.appendChild(a);
            });
            listEl.appendChild(section);
        });
        return linksByPath;
    }

    async function init() {
        var r;
        try {
            r = await fetch("/api/docs/list").then(function (x) { return x.json(); });
        } catch (e) {
            _setStatus("Failed to load the document list.");
            return;
        }
        if (!r || !r.ok || !r.groups || !r.groups.length) {
            _setStatus((r && r.note) || "No documents found under docs/.");
            return;
        }
        statusEl.remove();
        var linksByPath = renderList(r.groups);

        // Deep link: ?doc=<path> opens that doc on load.
        var initial = null;
        try {
            initial = new URL(window.location.href).searchParams.get("doc");
        } catch (_) { initial = null; }
        if (initial && linksByPath[initial]) {
            openDoc(initial, linksByPath[initial]);
        }
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
