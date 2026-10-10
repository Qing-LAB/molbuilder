/* The frame-set card — what a citation's frames are, for every page that
 * shows a citation (the Transport tab's junction card, Task setup's echo).
 *
 * Contract: docs/engines/transport.md § 3.8.1 ("A frame set says what each
 *           frame is"); docs/model/structure.md § 2.2f (the definition).
 * Data:     the server's `frame_set` — `{n_frames, definition}`, the
 *           definition `frameset.ModeFrameSet.to_wire()`: the structure's
 *           rows and the frames' columns, each with its unit and kind, built
 *           from `frameset.ROWS`. The card keeps no list of names: what the
 *           definition holds is what it draws.
 * Owns:     lib/frameset/card.css.
 *
 * NEVER:
 *   - work out a number. Every value is the definition's, as stated.
 *   - number a frame on its own: the frame bar's numbers, through MolView's
 *     one translation (`toDisplay`, model/overview.md § 2).
 */
import { toDisplay } from "/static/lib/molview/index.js";

/** A number as it reads on the card: six significant figures. */
function _fmt(v) {
    return typeof v === "number" ? String(Number(v.toPrecision(6))) : String(v);
}

/** A row's name with its unit, as the definition states both. */
function _label(row) {
    return row.unit ? row.name + " (" + row.unit + ")" : row.name;
}

function _el(doc, tag, cls, text) {
    const e = doc.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
}

/**
 * Draw ``frameSet`` into ``host``: for a mode's set, its mode line and a table
 * — a row per frame, numbered as the frame bar numbers them, a column per
 * frame row of the definition; for a set stating none of it, one line; for one
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
            n + " frames — a family: each frame's curve its own, no average "
            + "(it states no mode's definition, model/structure.md § 2.2f)."));
        return;
    }
    const mode = _el(doc, "p", "frameset-line");
    mode.textContent = n + " frames of one mode — " + def.structure
        .map((r) => _label(r) + " " + _fmt(r.value)).join(" · ");
    host.appendChild(mode);
    const table = _el(doc, "table", "num-table frameset-table");
    const head = _el(doc, "tr");
    head.appendChild(_el(doc, "th", null, "Frame"));
    for (const c of def.columns) {
        const th = _el(doc, "th", null, _label(c));
        th.title = c.kind;
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
}
