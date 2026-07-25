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
     *  beyond the default allow-list. */
    function render(text) {
        const raw = root.marked.parse(text || "", {
            breaks: false,
            gfm:    true,
        });
        return root.DOMPurify.sanitize(raw, {
            ADD_ATTR: ["target"],
        });
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.markdownRender = { loadRenderLibs: loadRenderLibs, render: render };
})(window);
