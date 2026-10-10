/* point-words.js — a transport point as the UI names it, everywhere it is
 * named: the report's legend and points table, the ladder's rows.
 *
 * Contract: docs/web/results.md § 2.5 ("one wording"); a point is a frame
 *           and a voltage (docs/engines/transport.md § 2a.9).
 *
 * The UI's own display, so a frame reads as the frame bar counts — through
 * MolView's one translation (`toDisplay`, model/overview.md § 2) — and the
 * voltage as the record states it.
 */
import { toDisplay } from "/static/lib/molview/index.js";

/** ``Frame 2 of 3``, for the frame index ``frame`` of ``of`` frames. */
export function frameWords(frame, of) {
    return "Frame " + toDisplay(Number(frame))
        + (of != null ? " of " + Number(of) : "");
}

/**
 * A point: ``Frame 2 of 3 · 0.2 V``, ``0.2 V`` for a calculation of one
 * structure (``frame`` null or ``of`` < 2), ``Frame 2 of 3`` with no voltage.
 */
export function pointWords(frame, of, biasV) {
    const parts = [];
    if (frame != null && !(of != null && Number(of) < 2)) {
        parts.push(frameWords(frame, of));
    }
    if (biasV != null) parts.push(String(Number(biasV)) + " V");
    return parts.join(" · ");
}
