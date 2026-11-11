/* point-words.js — a point of a run as the UI names it, everywhere it is
 * named: the report's legend and points table, the ladder's rows.
 *
 * Contract: docs/web/results.md § 2.5 ("one wording"); a transport point is
 *           a frame and a voltage (docs/engines/transport.md § 2a.9); a
 *           SIESTA vibration's call is its phase, and an optical call its
 *           mode and side (docs/engines/vibration.md § 5.11 § 7).
 *
 * The UI's own display, so a frame reads as the frame bar counts — through
 * MolView's one translation (`toDisplay`, model/overview.md § 2) — and the
 * voltage as the record states it; a call as its wire form words it, the
 * server's one wording of a call (`Call.words`).
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

/**
 * Any point of a run, from its wire form: a SIESTA vibration's call by the
 * words its wire form carries (`vibration_phases.Call.to_wire`: *the
 * relaxation*, *the force constants*, *the equilibrium*, *mode 7
 * (4271 cm⁻¹), +δQ*), a transport point by its frame and voltage; ``""``
 * for a run of one.
 */
export function runPointWords(r) {
    if (r.phase) return r.words || "";
    return r.bias_v != null ? pointWords(r.frame, r.of, r.bias_v) : "";
}
