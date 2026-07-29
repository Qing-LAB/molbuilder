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
    var _parents = {};           // child-path -> parent DOM element

    /* ---- theme toggle ------------------------------------------- */

    function _applyTheme(dark) {
        renderEl.classList.toggle("docs-render-dark", dark);
        if (themeBtn) themeBtn.textContent = dark ? "\u25D1" : "\u25D0";
        try { sessionStorage.setItem("docs-theme", dark ? "dark" : "light"); }
        catch (_) {}
    }
    if (themeBtn) {
        var _saved = null;
        try { _saved = sessionStorage.getItem("docs-theme"); } catch (_) {}
        _applyTheme(_saved === "dark");
        themeBtn.addEventListener("click", function () {
            _applyTheme(!renderEl.classList.contains("docs-render-dark"));
        });
    }

    /* ---- rendering helpers -------------------------------------- */

    function _setStatus(msg) { if (statusEl) statusEl.textContent = msg; }

    function _showMessage(cls, msg) {
        renderEl.textContent = "";
        var p = document.createElement("p");
        p.className = cls;  p.textContent = msg;
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

        var r;
        try {
            r = await fetch("/api/docs/read?path=" + encodeURIComponent(path))
                    .then(function (x) { return x.json(); });
        } catch (e) {
            _showMessage("docs-error", "Network error: " + (e && e.message || ""));
            return;
        }
        if (!r || !r.ok) {
            _showMessage("docs-error", (r && r.error) || "Could not load document.");
            return;
        }
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

    function _makeToggle(collapsed) {
        var t = document.createElement("span");
        t.className = "docs-tree-toggle";
        t.textContent = collapsed ? "\u25B8" : "\u25BE";
        return t;
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

        hdr.addEventListener("click", function (ev) {
            if (ev.target.tagName === "A") return;  // don't toggle on link click
            var isCollapsed = body.classList.toggle("is-collapsed");
            toggle.textContent = isCollapsed ? "\u25B8" : "\u25BE";
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
            var dir = _makeDir(_makeToggle(false), link,
                node.children.map(_buildNode), false);
            // Record parent for ancestor expansion.
            node.children.forEach(function (c) {
                if (c.path) _parents[c.path] = dir;
            });
            return dir;
        }
        // Plain leaf.
        if (node.path) return _makeLink(node.path, node.subtitle);
        // Directory.
        var dirKids = (node.children || []).map(_buildNode);
        var dir = _makeDir(_makeToggle(node.collapsed),
            (function () { var s = document.createElement("span"); s.textContent = node.label; return s; })(),
            dirKids, node.collapsed);
        (node.children || []).forEach(function (c) {
            if (c.path) _parents[c.path] = dir;
        });
        return dir;
    }

    /** Expand every ancestor of a path so it's visible. */
    function _expandAncestors(path) {
        var seen = {};
        var el = _parents[path];
        while (el && !seen[path]) {
            seen[path] = true;
            var body = el.querySelector(".docs-tree-body");
            if (body) body.classList.remove("is-collapsed");
            var tgl = el.querySelector(".docs-tree-toggle");
            if (tgl) tgl.textContent = "\u25BE";
            // Find the parent path that maps to this dir element.
            var prev = path;
            path = null;
            for (var p in _parents) {
                if (_parents[p] === el && p !== prev) { path = p; break; }
            }
            el = path ? _parents[path] : null;
        }
    }

    /* ---- init --------------------------------------------------- */

    function renderTree(tree, readme) {
        listEl.innerHTML = "";
        if (readme) listEl.appendChild(_makeLink(readme.path, readme.subtitle, "docs-link-root", readme.label));
        tree.forEach(function (n) { listEl.appendChild(_buildNode(n)); });
    }

    async function init() {
        var r;
        try {
            r = await fetch("/api/docs/toc").then(function (x) { return x.json(); });
        } catch (e) { _setStatus("Failed to load the document list."); return; }
        if (!r || !r.ok || (!r.tree || !r.tree.length) && !r.readme) {
            _setStatus((r && r.error) || "No documents found."); return;
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