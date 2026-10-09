/* Shared Markdown → sanitised-HTML render (the ONE render + sanitise policy).
 *
 * The security-relevant sanitise allow-list lives in exactly ONE place --
 * both the Results-tab
 * markdown inspector (edit + preview) and the Documents tab (read-only) render
 * through here, so they can never drift on what HTML is allowed.
 *
 * Surface (on window.molbuilder.markdownRender):
 *   loadRenderLibs() -> Promise   lazy-load marked + DOMPurify once (cached).
 *   render(text) -> string        marked.parse + DOMPurify.sanitize -> safe HTML,
 *                                 its LaTeX set aside as placeholders.
 *   renderMathIn(el) -> Promise   draw those placeholders with KaTeX.
 *   renderMermaidIn(el) -> Promise  draw ```mermaid blocks.
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
        const raw = root.marked.parse(_setMathAside(text || ""), {
            breaks: false,
            gfm:    true,
        });
        return root.DOMPurify.sanitize(raw, {
            ADD_ATTR: ["target"],
        });
    }

    // ---- math (KaTeX; lazy, loaded only when a doc has a formula) ---- //

    /* MATH IS SET ASIDE BEFORE THE MARKDOWN PASS.  marked reads the `_` and
     * `*` of a formula as emphasis, so each formula is cut out first -- never
     * inside code, where a `$` stays literal -- and left as an empty
     * placeholder holding its TeX in an attribute, which marked and DOMPurify
     * pass through untouched; `renderMathIn` draws each with KaTeX.
     *
     * `$$...$$` is display math.  `$...$` is inline math when the opening
     * `$` touches its formula and the closing one is not followed by a
     * digit -- so a shell prompt (`$ ls`) and a price (`$5`) stay text, the
     * same rule GitHub and pandoc read. */
    const _DISPLAY = /\$\$([\s\S]+?)\$\$/g;
    const _INLINE = /(^|[^\\$])\$(?=\S)((?:\\\$|[^$\n])+?)(?<=\S)\$(?!\d)/g;
    const _FENCE = /^\s*(```|~~~)/;
    const _CODE_SPAN = /(`+)([\s\S]*?[^`])\1(?!`)/g;

    function _attr(tex) {
        return tex.replace(/&/g, "&amp;").replace(/"/g, "&quot;")
                  .replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }
    function _placeholder(tex, display) {
        return '<span class="mb-math" data-display="' + (display ? "1" : "0")
            + '" data-tex="' + _attr(tex.trim()) + '"></span>';
    }
    function _mathInProse(text) {
        const spans = [];
        const held = text.replace(_CODE_SPAN, (m) => {
            spans.push(m);
            return "" + (spans.length - 1) + "";
        });
        return held
            .replace(_DISPLAY, (_m, tex) => _placeholder(tex, true))
            .replace(_INLINE, (_m, before, tex) => before + _placeholder(tex, false))
            .replace(/(\d+)/g, (_m, i) => spans[Number(i)]);
    }
    function _setMathAside(text) {
        if (text.indexOf("$") < 0) return text;
        const out = [];
        let prose = [];
        let fence = null;
        for (const line of text.split("\n")) {
            const f = _FENCE.exec(line);
            if (fence) {
                out.push(line);
                if (f && f[1] === fence) fence = null;
            } else if (f) {
                if (prose.length) { out.push(_mathInProse(prose.join("\n"))); prose = []; }
                out.push(line);
                fence = f[1];
            } else {
                prose.push(line);
            }
        }
        if (prose.length) out.push(_mathInProse(prose.join("\n")));
        return out.join("\n");
    }

    let _katexPromise = null;

    function _loadKatex() {
        if (_katexPromise) return _katexPromise;
        _katexPromise = (async () => {
            if (!document.querySelector('link[data-vendor="katex"]')) {
                const css = document.createElement("link");
                css.rel = "stylesheet";
                css.href = "/static/vendor/katex/katex.min.css";
                css.setAttribute("data-vendor", "katex");
                document.head.appendChild(css);
            }
            if (!root.katex) {
                await _loadScript("/static/vendor/katex/katex.min.js");
            }
        })();
        return _katexPromise;
    }

    /** Draw every formula `render` set aside inside ``rootEl``.  No-op (and no
     *  KaTeX load) when there is none.  A formula KaTeX cannot read is drawn
     *  as its source in the error colour, so one bad formula does not blank
     *  the doc. */
    async function renderMathIn(rootEl) {
        if (!rootEl) return;
        const spans = rootEl.querySelectorAll("span.mb-math[data-tex]");
        if (!spans.length) return;
        await _loadKatex();
        for (const el of spans) {
            root.katex.render(el.getAttribute("data-tex"), el, {
                displayMode: el.getAttribute("data-display") === "1",
                throwOnError: false,
                output: "htmlAndMathml",
                trust: false,
            });
        }
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
                // securityLevel 'strict'.  The ONE innerHTML here, and one of
                // the counted producers in tests/test_xss_audit.py.
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
        renderMathIn: renderMathIn,
        renderMermaidIn: renderMermaidIn,
    };
})(window);
