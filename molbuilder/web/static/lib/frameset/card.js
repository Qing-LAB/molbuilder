/* The frame-set card — a citation's frames as the set stores them, for
 * every page that shows a citation (the Transport tab's junction card, Task
 * setup's echo, the Results tab's report).
 *
 * Contract: docs/engines/transport.md § 3.8.1 ("A frame set's rows, as it
 *           stores them"); docs/model/structure.md § 2.2f (the rows as
 *           written).
 * Data:     the server's `frame_set` — `{n_frames, definition, said}`,
 *           transport's answer (`frame_average.answer`): the structure's
 *           rows and the frames' columns as stored, each its name, value and
 *           unit, what is said of the weights, and `said`, transport's words
 *           of the citation (a calculation's pair no longer as pinned).
 *           The card keeps no list of names and interprets no row: what a
 *           row means is the set's own (the user, 2026-11-26).
 * Owns:     lib/frameset/card.css.
 *
 * NEVER:
 *   - work out a number. Every value is the answer's, as stored.
 *   - put a row in other words, give it a meaning, or say what made the
 *     frames or where they came from — transport does the transport
 *     calculation, and that is all.
 *   - number a frame on its own: the frame bar's numbers, through MolView's
 *     one translation (`toDisplay`, model/overview.md § 2).
 */
import { toDisplay } from "/static/lib/molview/index.js";

/** A value as it reads on the card: a whole number whole, any other number
 *  to six significant figures, anything else as written, and a dash where a
 *  frame states none. */
function _fmt(v) {
    if (v === null || v === undefined) return "—";
    if (typeof v !== "number") return String(v);
    return Number.isInteger(v) ? String(v) : String(Number(v.toPrecision(6)));
}

/** A unit as a page shows it: none for the dimensionless one. */
function _unit(u) {
    return u && u !== "1" ? u : "";
}

function _el(doc, tag, cls, text) {
    const e = doc.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
}

/** A warning line, in the warning tone every surface gives a warning
 *  (`web/ui-contract.md` § 5, `status.tone`). */
function _warning(doc, words) {
    const p = _el(doc, "p", "frameset-line", "Warning: " + words);
    const status = doc.defaultView && doc.defaultView.molbuilder
        && doc.defaultView.molbuilder.status;
    if (status) status.tone(p, "warn");
    return p;
}

/** A row's name with its unit, as stored: ``name (unit)``, or the name
 *  alone where it states none or the dimensionless one. */
function _named(name, unit) {
    const u = _unit(unit);
    return u ? name + " (" + u + ")" : name;
}

/**
 * Draw ``frameSet`` into ``host``: its count, the structure's rows as the
 * set stores them -- a table of each row's name, value and unit -- and a
 * table, a row per frame, numbered as the frame bar numbers them, a column
 * per row the frames state, under its stored name and unit; what is said
 * of the weights and of the citation beneath. For frames stating no rows,
 * one line; for one structure, nothing (the host hidden).
 */
export function renderFrameSet(host, frameSet) {
    if (!host) return;
    const doc = host.ownerDocument;
    host.replaceChildren();
    // A CITATION THAT DOES NOT READ draws nothing here: the page that shows
    // the citation says why, once (`task-setup.md` § 3, its facts' row).
    const n = frameSet && frameSet.n_frames != null ? Number(frameSet.n_frames) : 1;
    host.hidden = !(n > 1);
    if (host.hidden) return;
    const def = frameSet.definition;
    // WHAT TRANSPORT SAYS OF THE CITATION -- a calculation's pair no longer
    // as its record pinned it -- under whatever the card shows.
    const citationSaid = () => {
        for (const w of frameSet.said || []) {
            host.appendChild(_warning(doc, w));
        }
    };
    if (!def) {
        host.appendChild(_el(doc, "p", "frameset-line",
            n + " frames — a family: each frame's curve its own, no average."));
        citationSaid();
        return;
    }
    host.appendChild(_el(doc, "p", "frameset-line", n + " frames"));
    // THE SET'S ROWS, AS STORED: name, value, unit -- in the order the set
    // writes them, none put in other words.
    if (def.structure.length) {
        const rows = _el(doc, "table", "num-table frameset-rows");
        const head = _el(doc, "tr");
        for (const h of ["Name", "Value", "Unit"]) {
            head.appendChild(_el(doc, "th", null, h));
        }
        rows.appendChild(_el(doc, "thead")).appendChild(head);
        const body = rows.appendChild(_el(doc, "tbody"));
        for (const r of def.structure) {
            const tr = _el(doc, "tr");
            tr.appendChild(_el(doc, "td", "frameset-name", r.name));
            tr.appendChild(_el(doc, "td", "frameset-value", _fmt(r.value)));
            tr.appendChild(_el(doc, "td", "frameset-unit", _unit(r.unit)));
            body.appendChild(tr);
        }
        host.appendChild(rows);
    }
    const table = _el(doc, "table", "num-table frameset-table");
    const head = _el(doc, "tr");
    head.appendChild(_el(doc, "th", null, "Frame"));
    for (const c of def.columns) {
        head.appendChild(_el(doc, "th", null, _named(c.name, c.unit)));
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
    // WHAT IS SAID OF THE WEIGHTS -- a frame with none, a sum off 1 -- and
    // OF THE CITATION.
    for (const w of def.warnings || []) {
        host.appendChild(_warning(doc, w));
    }
    citationSaid();
}
