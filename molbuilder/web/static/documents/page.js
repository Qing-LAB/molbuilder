/* Documents tab — browse + read every docs/*.md in-app.
 *
 * Left: a hierarchical tree from /api/docs/toc (built from docs/toc.json).
 * Right: the picked doc, fetched from /api/docs/read and rendered through
 * the shared markdown renderer (lib/markdown-render.js).
 */
(function () {
    "use strict";

    var listEl   = document.getElementById("docs-list");
    var statusEl = document.getElementById("docs-list-status");
    var emptyEl  = document.getElementById("docs-view-empty");
    var renderEl = document.getElementById("docs-render");
    var barEl    = document.getElementById("docs-view-bar");
    var themeBtn = document.getElementById("docs-theme-btn");

    var _activeLink = null;
    var _linksByPath = {};       // path -> DOM element

    /* ---- theme toggle ------------------------------------------- */

    /* THE APP IS DARK, so the isolated `docs-render.css` theme starts dark;
     * the toggle offers a light reading surface (some people print from this
     * pane), and the choice is a reading preference kept in localStorage,
     * with a sessionStorage fallback where storage is blocked. */
    function _store(key, value) {
        try { localStorage.setItem(key, value); return; } catch (_) {}
        try { sessionStorage.setItem(key, value); } catch (_) {}
    }
    function _read(key) {
        try {
            var v = localStorage.getItem(key);
            if (v !== null) return v;
        } catch (_) {}
        try { return sessionStorage.getItem(key); } catch (_) {}
        return null;
    }

    function _applyTheme(dark) {
        renderEl.classList.toggle("docs-render-dark", dark);
        if (themeBtn) {
            themeBtn.textContent = dark ? "\u25D1" : "\u25D0";
            themeBtn.title = dark ? "Read on a light background"
                                  : "Read on a dark background";
        }
        _store("docs-theme", dark ? "dark" : "light");
    }
    if (themeBtn) {
        // Dark unless the reader has said otherwise -- matching the app.
        _applyTheme(_read("docs-theme") !== "light");
        themeBtn.addEventListener("click", function () {
            _applyTheme(!renderEl.classList.contains("docs-render-dark"));
        });
    } else {
        // No toggle rendered (a trimmed page): still match the app.
        renderEl.classList.add("docs-render-dark");
    }

    /* ---- rendering helpers -------------------------------------- */

    // THE SERVER'S WORDS, DRAWN (`lib/dom.js`, ui-contract.md § 12.1).
    function _setStatus(msg) {
        if (statusEl) statusEl.replaceChildren(window.molbuilder.dom.codeWords(msg));
    }

    function _showMessage(cls, msg) {
        renderEl.textContent = "";
        var p = document.createElement("p");
        p.className = cls;  p.appendChild(window.molbuilder.dom.codeWords(msg));
        renderEl.appendChild(p);
    }

    /** Shared link builder: path (link target) + subtitle (dimmed).
     *  `displayPath` overrides the visible filename (e.g. to hide
     *  the ``../`` prefix on the root README). */
    function _makeLink(path, subtitle, extraClass, displayPath) {
        var a = document.createElement("a");
        a.className = "docs-link" + (extraClass ? " " + extraClass : "");
        a.href = "?doc=" + encodeURIComponent(path);

        var nm = document.createElement("span");
        nm.className = "docs-link-name";
        nm.textContent = displayPath || path;
        a.appendChild(nm);

        if (subtitle) {
            var sb = document.createElement("span");
            sb.className = "docs-link-subtitle";
            sb.textContent = subtitle;
            a.appendChild(sb);
        }

        a.addEventListener("click", function (ev) {
            ev.preventDefault();
            openDoc(path, a);
        });
        _linksByPath[path] = a;
        return a;
    }

    /* ---- document loading --------------------------------------- */

    async function openDoc(path, link) {
        if (_activeLink) _activeLink.classList.remove("is-active");
        if (link) { link.classList.add("is-active"); _activeLink = link; }

        emptyEl.hidden = true;
        renderEl.hidden = false;
        if (barEl) barEl.hidden = false;
        _showMessage("docs-loading", "Loading…");

        // THE ONE READER (ui-contract.md § 12), ready since init waited.
        var a = await window.molbuilder.server.ask(
            "/api/docs/read?path=" + encodeURIComponent(path));
        if (!a.ok) {
            _showMessage("docs-error", a.error);
            return;
        }
        var r = a.body;
        try {
            await window.molbuilder.markdownRender.loadRenderLibs();
            renderEl.innerHTML = window.molbuilder.markdownRender.render(r.text);
        } catch (e) {
            _showMessage("docs-error", "Render failed: " + (e && e.message || ""));
            return;
        }
        // Keep root-README links portable: GitHub resolves docs/<path>.md and
        // LICENSE directly. This reader rewrites only those root-README links
        // to their documented internal ?doc= paths.
        function wireDocLink(a, docPath) {
            a.setAttribute("href", "?doc=" + encodeURIComponent(docPath));
            a.addEventListener("click", function (ev) {
                ev.preventDefault();
                openDoc(docPath, a);
            });
        }
        // Wire links + images.
        renderEl.querySelectorAll("a[href]").forEach(function (a) {
            var h = a.getAttribute("href") || "";
            if (/^https?:/.test(h)) {
                a.setAttribute("target", "_blank");
                a.setAttribute("rel", "noopener noreferrer");
            } else if (h.startsWith("?doc=")) {
                wireDocLink(a, h.slice(5));
            } else if (path === "../README.md" && h === "LICENSE") {
                wireDocLink(a, "../LICENSE");
            } else if (path === "../README.md" && /^docs\/.+\.md(?:#.*)?$/.test(h)) {
                wireDocLink(a, h.slice(5).split("#", 1)[0]);
            }
        });
        renderEl.querySelectorAll("img[src]").forEach(function (img) {
            var s = img.getAttribute("src");
            if (s && !/^https?:/.test(s) && !s.startsWith("/")) {
                s = s.replace(/^\.?\/?/, "");
                if (s.startsWith("docs/")) s = s.slice(5);
                if (s.startsWith("img/"))  s = s.slice(4);
                img.setAttribute("src", "/api/docs/img/" + s);
            }
        });
        try { await window.molbuilder.markdownRender.renderMathIn(renderEl); }
        catch (_) {}
        try { await window.molbuilder.markdownRender.renderMermaidIn(renderEl); }
        catch (_) {}
        renderEl.scrollTop = 0;
        try {
            var u = new URL(window.location.href);
            u.searchParams.set("doc", path);
            window.history.replaceState(null, "", u.toString());
        } catch (_) {}
    }

    /* ---- tree building ------------------------------------------ */

    /* A FOLDER'S ARROW IS ITS BUTTON (ui-contract.md § 4.1): the keyboard's
     * way to what a click on the folder's row does, saying whether it is
     * open. */
    function _makeToggle(collapsed) {
        var t = document.createElement("button");
        t.type = "button";
        t.className = "docs-tree-toggle plain";
        t.textContent = collapsed ? "\u25B8" : "\u25BE";
        t.setAttribute("aria-expanded", collapsed ? "false" : "true");
        return t;
    }

    /** A folder opened or closed: its documents shown or not, its button
     *  saying which. */
    function _setOpen(body, toggle, open) {
        body.classList.toggle("is-collapsed", !open);
        toggle.textContent = open ? "\u25BE" : "\u25B8";
        toggle.setAttribute("aria-expanded", open ? "true" : "false");
    }

    /** Build a collapsible folder: header row + children body. */
    function _makeDir(toggle, headerContent, children, collapsed) {
        var hdr = document.createElement("div");
        hdr.className = "docs-tree-header";
        hdr.appendChild(toggle);
        hdr.appendChild(headerContent);

        var body = document.createElement("div");
        body.className = "docs-tree-body";
        if (collapsed) body.classList.add("is-collapsed");
        children.forEach(function (c) { body.appendChild(c); });

        // NAMED BY THE FOLDER'S NAME -- a document's path, never its
        // subtitle run on after it.
        var named = headerContent.querySelector
            && headerContent.querySelector(".docs-link-name");
        toggle.setAttribute("aria-label", "Documents under "
            + ((named || headerContent).textContent || "this folder"));
        hdr.addEventListener("click", function (ev) {
            if (ev.target.closest("a")) return;  // a document's link opens it
            _setOpen(body, toggle, body.classList.contains("is-collapsed"));
        });

        var dir = document.createElement("div");
        dir.className = "docs-tree-dir";
        dir.appendChild(hdr);
        dir.appendChild(body);
        return dir;
    }

    function _buildNode(node) {
        // Document with sub-documents.
        if (node.path && node.children) {
            var link = _makeLink(node.path, node.subtitle, "docs-link-parent");
            return _makeDir(_makeToggle(false), link,
                node.children.map(_buildNode), false);
        }
        // Plain leaf.
        if (node.path) return _makeLink(node.path, node.subtitle);
        // Directory.
        var dirKids = (node.children || []).map(_buildNode);
        return _makeDir(_makeToggle(node.collapsed),
            (function () { var s = document.createElement("span"); s.textContent = node.label; return s; })(),
            dirKids, node.collapsed);
    }

    /** Expand every ancestor of a path so it's visible. */
    function _expandAncestors(path) {
        // EVERY FOLDER THE DOCUMENT SITS IN, climbed by the page's own
        // nesting from its link -- the top folder too.
        var a = _linksByPath[path];
        var dir = a ? a.closest(".docs-tree-dir") : null;
        while (dir) {
            var body = dir.querySelector(":scope > .docs-tree-body");
            var tgl = dir.querySelector(
                ":scope > .docs-tree-header > .docs-tree-toggle");
            if (body && tgl) _setOpen(body, tgl, true);
            dir = dir.parentElement
                ? dir.parentElement.closest(".docs-tree-dir") : null;
        }
    }

    /* ---- init --------------------------------------------------- */

    function renderTree(tree, readme) {
        listEl.innerHTML = "";
        if (readme) listEl.appendChild(_makeLink(readme.path, readme.subtitle, "docs-link-root", readme.label));
        tree.forEach(function (n) { listEl.appendChild(_buildNode(n)); });
    }

    async function init() {
        // THE ONE READER (ui-contract.md § 12) is a module: asked once it
        // has run.
        var server = await window.molbuilder.runtime.whenReady("server");
        var a = await server.ask("/api/docs/toc");
        if (!a.ok) {
            _setStatus("Failed to load the document list: " + a.error); return;
        }
        var r = a.body;
        if ((!r.tree || !r.tree.length) && !r.readme) {
            _setStatus("No documents found."); return;
        }
        statusEl.remove();
        renderTree(r.tree, r.readme);

        var initial = null;
        try { initial = new URL(window.location.href).searchParams.get("doc"); }
        catch (_) {}
        if (initial && _linksByPath[initial]) {
            _expandAncestors(initial);
            openDoc(initial, _linksByPath[initial]);
        }
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else { init(); }
})();