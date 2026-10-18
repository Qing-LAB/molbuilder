/* The frame-set card — what a citation's frames are, for every page that
 * shows a citation (the Transport tab's junction card, Task setup's echo).
 *
 * Contract: docs/engines/transport.md § 3.8.1 ("A frame set says what each
 *           frame is"); docs/model/structure.md § 2.2f (the rows as written).
 * Data:     the server's `frame_set` — `{n_frames, definition}`, the
 *           frames' post-process's answer (`frame_average.answer`): the
 *           structure's rows and the frames' columns, each with its unit and
 *           kind -- the vocabulary's words for a row `frame_average.ROWS`
 *           knows, any other's name as written -- and what is said of the
 *           weights. The card keeps no list of names: what the answer holds
 *           is what it draws.
 * Owns:     lib/frameset/card.css.
 *
 * NEVER:
 *   - work out a number. Every value is the answer's, as written.
 *   - number a frame on its own: the frame bar's numbers, through MolView's
 *     one translation (`toDisplay`, model/overview.md § 2).
 */
import { toDisplay } from "/static/lib/molview/index.js";

/** A value as it reads on the card: a number to six significant figures,
 *  anything else as written, and a dash where a frame states none. */
function _fmt(v) {
    if (v === null || v === undefined) return "—";
    return typeof v === "number" ? String(Number(v.toPrecision(6))) : String(v);
}

/** A row as a page says it -- its words, with its unit -- and, on hover,
 *  the name the file stores it under and what kind of number it is. */
function _label(row) {
    return row.page_unit ? row.words + " (" + row.page_unit + ")" : row.words;
}

function _stored(row) {
    return row.name + " — " + row.kind;
}

function _el(doc, tag, cls, text) {
    const e = doc.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
}

/**
 * Draw ``frameSet`` into ``host``: the structure's rows on one line -- a mode's
 * set's as its mode line -- and a table: a row per frame, numbered as the
 * frame bar numbers them, a column per row the frames state, and what is said
 * of the weights beneath; for frames stating no rows, one line; for one
 * structure, nothing (the host hidden).
 */
export function renderFrameSet(host, frameSet) {
    if (!host) return;
    const doc = host.ownerDocument;
    host.replaceChildren();
    const n = frameSet ? Number(frameSet.n_frames) : 1;
    host.hidden = !(n > 1);
    if (host.hidden) return;
    const def = frameSet.definition;
    if (!def) {
        host.appendChild(_el(doc, "p", "frameset-line",
            n + " frames — a family: each frame's curve its own, no average."));
        return;
    }
    // THE MODE LINE: each row its words, its value and its unit -- "Mode 6
    // · Frequency 2548.37 cm⁻¹ · …" -- the stored name on hover.
    const ofMode = def.structure.some((r) => r.name === "mode_index_1based");
    const mode = _el(doc, "p", "frameset-line",
                     n + (ofMode ? " frames of one mode — " : " frames — "));
    def.structure.forEach((r, i) => {
        if (i) mode.appendChild(doc.createTextNode(" · "));
        const bit = _el(doc, "span", null, r.words + " " + _fmt(r.value)
                        + (r.page_unit ? " " + r.page_unit : ""));
        bit.title = _stored(r);
        mode.appendChild(bit);
    });
    host.appendChild(mode);
    const table = _el(doc, "table", "num-table frameset-table");
    const head = _el(doc, "tr");
    head.appendChild(_el(doc, "th", null, "Frame"));
    for (const c of def.columns) {
        const th = _el(doc, "th", null, _label(c));
        th.title = _stored(c);
        head.appendChild(th);
    }
    table.appendChild(_el(doc, "thead")).appendChild(head);
    const body = table.appendChild(_el(doc, "tbody"));
    def.frames.forEach((values, f) => {
        const tr = _el(doc, "tr");
        tr.appendChild(_el(doc, "td", "frameset-frame", String(toDisplay(f))));
        values.forEach((v) => tr.appendChild(_el(doc, "td", null, _fmt(v))));
        body.appendChild(tr);
    });
    host.appendChild(table);
    // WHAT IS SAID OF THE WEIGHTS -- a frame with none, a sum off 1.
    for (const w of def.warnings || []) {
        host.appendChild(_el(doc, "p", "frameset-line", "Warning: " + w));
    }
}
