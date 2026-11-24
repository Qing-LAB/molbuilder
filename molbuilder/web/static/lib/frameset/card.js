/* The frame-set card — what a citation's frames are, for every page that
 * shows a citation (the Transport tab's junction card, Task setup's echo,
 * the Results tab's report).
 *
 * Contract: docs/engines/transport.md § 3.8.1 ("A frame set says what each
 *           frame is"); docs/model/structure.md § 2.2f (the rows as written);
 *           docs/engines/vibration-frames.md § 5.1, § 6 (the words).
 * Data:     the server's `frame_set` — `{n_frames, definition}`, transport's
 *           answer (`frame_average.answer`): the structure's rows and the
 *           frames' columns as written, each with its unit and meaning
 *           (`note`), and what is said of the weights — and beside it
 *           `frame_set_said`, the vibration frame sets' words for the rows
 *           they speak for (a prose row among them), what the set is, where
 *           it came from and the rows that say it (`provenance.rows`), an
 *           older mode set's assumptions, and what is said of the pair
 *           (`vibration_frames.said_of`). A row the words do not name is
 *           shown by its name and unit as written. The card keeps no list of
 *           names: what the answers hold is what it draws.
 * Owns:     lib/frameset/card.css.
 *
 * NEVER:
 *   - work out a number. Every value is the answer's, as written.
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

/** A row as the page says it: the words the vibration frame sets give it
 *  where they speak for it, else its name and unit as written -- and, on
 *  hover, the name the file stores it under, what kind of number it is and
 *  what it means (the words' meaning, the set's own note folded in by the
 *  server; else the row's note); ``prose``, a row the words say is
 *  sentences. */
function _said(row, words) {
    const w = words && words[row.name];
    const means = w ? w.means : row.note;
    return {
        words: w ? w.words : row.name,
        unit: w ? (w.page_unit || "") : _unit(row.unit),
        prose: !!(w && w.prose),
        stored: row.name + " — " + (w ? w.kind : "as written")
                + (means ? "\n" + means : ""),
    };
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

/**
 * Draw ``frameSet`` into ``host``, in the words of ``said`` (the server's
 * `frame_set_said`, or null): the structure's rows on one line, prose rows
 * beneath it, where the set came from, and a table -- a row per frame,
 * numbered as the frame bar numbers them, a column per row the frames
 * state -- and what is said of the weights and of the pair beneath; for
 * frames stating no rows, one line; for one structure, nothing (the host
 * hidden).
 */
export function renderFrameSet(host, frameSet, said) {
    if (!host) return;
    const doc = host.ownerDocument;
    host.replaceChildren();
    // A CITATION THAT DOES NOT READ SAYS WHY (`task-setup.md` § 3): the
    // answer's own sentence, and the folder still opens.
    if (frameSet && frameSet.why) {
        host.hidden = false;
        host.appendChild(_el(doc, "p", "frameset-line",
                             "The cited frames could not be read: " + frameSet.why));
        return;
    }
    const n = frameSet && frameSet.n_frames != null ? Number(frameSet.n_frames) : 1;
    host.hidden = !(n > 1);
    if (host.hidden) return;
    const def = frameSet.definition;
    const words = said || {};
    // WHAT IS SAID OF THE PAIR -- changed since it was cited, not read now,
    // a row written without its meaning -- under whatever the card shows.
    const pairSaid = () => {
        for (const w of words.said || []) {
            host.appendChild(_warning(doc, w));
        }
    };
    if (!def) {
        host.appendChild(_el(doc, "p", "frameset-line",
            n + " frames — a family: each frame's curve its own, no average."));
        pairSaid();
        return;
    }
    // THE SET'S LINE: what it is, in the words' own term -- "frames of one
    // mode", "frames, the relaxed geometry and its snapshots" -- then each
    // row its words, its value and its unit, the stored name on hover; a
    // prose row on a line of its own beneath, and the rows the provenance
    // line says (`provenance.rows`) said there once.
    const from = words.provenance;
    const onProvenance = new Set((from && from.rows) || []);
    const line = _el(doc, "p", "frameset-line",
                     n + " " + (words.what || "frames") + " — ");
    const prose = [];
    let shown = 0;
    def.structure.forEach((r) => {
        if (onProvenance.has(r.name)) return;
        const s = _said(r, words.structure);
        if (s.prose) {
            prose.push([r, s]);
            return;
        }
        if (shown++) line.appendChild(doc.createTextNode(" · "));
        const bit = _el(doc, "span", null, s.words + " " + _fmt(r.value)
                        + (s.unit ? " " + s.unit : ""));
        bit.title = s.stored;
        line.appendChild(bit);
    });
    host.appendChild(line);
    for (const [r, s] of prose) {
        const p = _el(doc, "p", "frameset-line", s.words + ": " + r.value);
        p.title = s.stored;
        host.appendChild(p);
    }
    // A MODE'S OWN SET WRITTEN BEFORE 2.81.0 states no assumptions row: what
    // its harmonic picture assumes, in the words the server gives it.
    if (words.assumptions) {
        const p = _el(doc, "p", "frameset-line",
                      words.assumptions.words + ": " + words.assumptions.value);
        p.title = "molbuilder's words for a mode's own set that states none "
                  + "of its own (one an earlier molbuilder wrote)";
        host.appendChild(p);
    }
    // WHERE A VIBRATION'S SET CAME FROM, as the pair states it.
    if (from) {
        const sha = from.result_sha256 ? String(from.result_sha256) : "";
        const p = _el(doc, "p", "frameset-line",
            "From " + (from.run ? "the vibration run " + from.run
                                : "a vibration result read outside a run")
            + (sha ? " · its result's sha256 " + sha.slice(0, 12) + "…" : ""));
        p.title = "as the pair states it: " + from.stated_as
                  + (sha ? "\nresult sha256 " + sha : "");
        host.appendChild(p);
    }
    const table = _el(doc, "table", "num-table frameset-table");
    const head = _el(doc, "tr");
    head.appendChild(_el(doc, "th", null, "Frame"));
    for (const c of def.columns) {
        const s = _said(c, words.frames);
        const th = _el(doc, "th", null, s.unit ? s.words + " (" + s.unit + ")"
                                                : s.words);
        th.title = s.stored;
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
    // WHAT IS SAID OF THE WEIGHTS -- a frame with none, a sum off 1 -- and
    // OF THE PAIR.
    for (const w of def.warnings || []) {
        host.appendChild(_warning(doc, w));
    }
    pairSaid();
}
