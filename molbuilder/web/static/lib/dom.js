/* dom.js — the one element builder the Results tab's record viewers and the
 * chemistry card (lib/chemistry.js, on the three form tabs) share.
 *
 * `el(tag, cls, text)`: a new element with a class and, when given, its
 * text -- written as textContent, never as HTML, so a record's strings
 * cannot become markup.  The benchmark-sweep, transport and displacement-
 * sweep viewers and the Run panel use this one copy.
 *
 * `codeWords(text)`: the server's words, drawn for a page
 * (`ui-contract.md` § 12.1) -- written once for the terminal and the page
 * alike, so a command in them kept in backticks (`jobset init`) is shown as
 * code and not as a backtick, and the terminal's dash, ` -- `, as a dash.
 * Loaded once for every page, by the shared header.
 *
 * Exports (on window.molbuilder.dom):
 *   el(tag, cls, text) -> Element
 *   codeWords(text)    -> DocumentFragment
 *   wordsText(text)    -> string   the text `codeWords` draws
 *   wordsEl(tag, cls, text) -> Element  `el`, its text through `codeWords`
 */
(function (root) {
    "use strict";

    function el(tag, cls, text) {
        const e = root.document.createElement(tag);
        if (cls) e.className = cls;
        if (text !== undefined && text !== null) e.textContent = String(text);
        return e;
    }

    function wordsEl(tag, cls, text) {
        const e = el(tag, cls);
        e.appendChild(codeWords(text));
        return e;
    }

    /* THE SERVER'S WORDS, CUT INTO WHAT IS SAID AND WHAT IS CODE: the
     * parts at odd places are commands, kept as typed; the others are
     * sentences, the terminal's dash drawn as a dash. */
    function _parts(text) {
        // ``a`` as the catalogue's help writes it is `a`.
        return String(text == null ? "" : text).replace(/``/g, "`").split("`")
            .map((part, i) => (i % 2 ? { code: part }
                                     : { said: part.replace(/ -- /g, " \u2014 ") }))
            .filter((p) => p.code || p.said);
    }

    function codeWords(text) {
        const out = root.document.createDocumentFragment();
        for (const p of _parts(text)) {
            out.appendChild(p.code !== undefined ? el("code", "", p.code)
                                                 : root.document.createTextNode(p.said));
        }
        return out;
    }

    /** The text `codeWords(text)` draws -- what a reader of the page, or
     *  one comparing what is drawn, reads. */
    function wordsText(text) {
        return _parts(text).map((p) => (p.code !== undefined ? p.code : p.said)).join("");
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.dom = Object.assign(root.molbuilder.dom || {},
                                        { el: el, codeWords: codeWords,
                                          wordsText: wordsText, wordsEl: wordsEl });
})(typeof window !== "undefined" ? window : this);
