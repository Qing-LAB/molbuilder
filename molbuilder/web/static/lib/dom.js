/* dom.js — the one element builder the Results tab's record viewers and the
 * chemistry card (lib/chemistry.js, on the three form tabs) share.
 *
 * `el(tag, cls, text)`: a new element with a class and, when given, its
 * text -- written as textContent, never as HTML, so a record's strings
 * cannot become markup.  The benchmark-sweep, transport and displacement-
 * sweep viewers and the Run panel use this one copy.
 *
 * `codeWords(text)`: the server's words, a command in them kept in backticks
 * (`jobset init`), with that command shown as code and not as a backtick.
 *
 * Exports (on window.molbuilder.dom):
 *   el(tag, cls, text) -> Element
 *   codeWords(text)    -> DocumentFragment
 */
(function (root) {
    "use strict";

    function el(tag, cls, text) {
        const e = root.document.createElement(tag);
        if (cls) e.className = cls;
        if (text !== undefined && text !== null) e.textContent = String(text);
        return e;
    }

    function codeWords(text) {
        const out = root.document.createDocumentFragment();
        String(text == null ? "" : text).split("`").forEach((part, i) => {
            if (!part) return;
            out.appendChild(i % 2 ? el("code", "", part)
                                  : root.document.createTextNode(part));
        });
        return out;
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.dom = Object.assign(root.molbuilder.dom || {},
                                        { el: el, codeWords: codeWords });
})(typeof window !== "undefined" ? window : this);
