/* Shared Markdown → sanitised-HTML render (the ONE render + sanitise policy).
 *
 * Promoted out of lib/inspectors/markdown.js so the security-relevant
 * sanitise allow-list lives in exactly ONE place -- both the Results-tab
 * markdown inspector (edit + preview) and the Documents tab (read-only) render
 * through here, so they can never drift on what HTML is allowed.
 *
 * Surface (on window.molbuilder.markdownRender):
 *   loadRenderLibs() -> Promise   lazy-load marked + DOMPurify once (cached).
 *   render(text) -> string        marked.parse + DOMPurify.sanitize -> safe HTML.
 *
 * Renderers that ALSO need the CodeMirror editor (the inspector) load that
 * separately; this module owns only the render path (marked + DOMPurify).
 */
(function (root) {
    "use strict";

    let _libsPromise = null;

    function _loadScript(src) {
        return new Promise((ok, ko) => {
            const t = document.createElement("script");
            t.src = src;
            t.onload = () => ok();
            t.onerror = () => ko(new Error("failed to load " + src));
            document.head.appendChild(t);
        });
    }

    /** Lazy-load marked + DOMPurify once.  Cached promise -- the first call
     *  kicks off the fetch, later calls await the same promise.  Guarded so a
     *  page that already has either library doesn't re-fetch it. */
    function loadRenderLibs() {
        if (_libsPromise) return _libsPromise;
        _libsPromise = (async () => {
            if (!root.marked) {
                await _loadScript("/static/vendor/marked/marked.min.js");
            }
            if (!root.DOMPurify) {
                await _loadScript("/static/vendor/dompurify/purify.min.js");
            }
        })();
        return _libsPromise;
    }

    /** Render Markdown text -> sanitised HTML string.  Single render path so a
     *  future switch of sanitiser / marked options has one site to update.
     *  DOMPurify defaults strip <script>, on* attributes, javascript: URLs,
     *  and iframes; we additionally keep ``target`` so links can open in a new
     *  tab.  The GFM tables / lists / code blocks the docs use need nothing
     *  beyond the default allow-list.  ```mermaid fences survive as
     *  ``<pre><code class="language-mermaid">`` for :func:`renderMermaidIn`. */
    function render(text) {
        const raw = root.marked.parse(text || "", {
            breaks: false,
            gfm:    true,
        });
        return root.DOMPurify.sanitize(raw, {
            ADD_ATTR: ["target"],
        });
    }

    // ---- mermaid (lazy; ~3 MB, loaded only when a doc has a diagram) ---- //

    let _mermaidPromise = null;
    let _mmdSeq = 0;

    function _loadMermaid() {
        if (_mermaidPromise) return _mermaidPromise;
        _mermaidPromise = (async () => {
            if (!root.mermaid) {
                await _loadScript("/static/vendor/mermaid/mermaid.min.js");
            }
            // startOnLoad:false -> WE drive render() explicitly (the doc HTML is
            // injected after mermaid loads, so auto-scan would miss it).
            // securityLevel 'strict' sandboxes label HTML (mermaid's own XSS
            // guard) on top of the app-shipped, trusted doc source.
            root.mermaid.initialize({
                startOnLoad: false,
                securityLevel: "strict",
                theme: "neutral",
            });
        })();
        return _mermaidPromise;
    }

    /** Render every ```mermaid code block inside ``rootEl`` into an SVG figure,
     *  in place.  No-op (and no mermaid load) when the element has no diagram.
     *  Async: loads mermaid lazily on first diagram.  A block that fails to
     *  parse is left as its code + a small error note, so one bad diagram
     *  doesn't blank the doc. */
    async function renderMermaidIn(rootEl) {
        if (!rootEl) return;
        const blocks = rootEl.querySelectorAll(
            "code.language-mermaid, code.lang-mermaid");
        if (!blocks.length) return;
        await _loadMermaid();
        for (const code of blocks) {
            const src = code.textContent || "";
            const host = code.closest("pre") || code;
            try {
                const out = await root.mermaid.render(
                    "mmd-" + (_mmdSeq++), src);
                const fig = document.createElement("figure");
                fig.className = "mermaid-figure";
                // mermaid-generated SVG from app-shipped docs, rendered with
                // securityLevel 'strict'.  The ONE innerHTML here; allowlisted
                // in tests/test_xss_audit.py.
                fig.innerHTML = out.svg;
                host.replaceWith(fig);
            } catch (e) {
                const note = document.createElement("div");
                note.className = "mermaid-error";
                note.textContent = "Diagram could not be rendered: "
                    + (e && e.message ? e.message : String(e));
                host.parentNode && host.parentNode.insertBefore(note, host);
            }
        }
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.markdownRender = {
        loadRenderLibs: loadRenderLibs,
        render: render,
        renderMermaidIn: renderMermaidIn,
    };
})(window);
