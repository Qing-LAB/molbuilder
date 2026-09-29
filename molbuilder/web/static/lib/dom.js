/* dom.js — the one element builder the Results tab's record viewers and the
 * chemistry card (lib/chemistry.js, on the three form tabs) share.
 *
 * `el(tag, cls, text)`: a new element with a class and, when given, its
 * text -- written as textContent, never as HTML, so a record's strings
 * cannot become markup.  The benchmark-sweep, transport and displacement-
 * sweep viewers and the Run panel each wrote it, two of them byte for byte
 * (`tests/test_no_duplicated_ui_components.py` found that pair on
 * 2026-09-28; the Run panel's copy it could not see): four places for one
 * fix to miss.
 *
 * Exports (on window.molbuilder.dom):
 *   el(tag, cls, text) -> Element
 */
(function (root) {
    "use strict";

    function el(tag, cls, text) {
        const e = root.document.createElement(tag);
        if (cls) e.className = cls;
        if (text !== undefined && text !== null) e.textContent = String(text);
        return e;
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.dom = Object.assign(root.molbuilder.dom || {}, { el: el });
})(typeof window !== "undefined" ? window : this);
