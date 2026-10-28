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

/**
 * A SIESTA vibration's call, from its one wire form
 * (`vibration_phases.Call.to_wire`): *the relaxation*, *the force
 * constants*, *the equilibrium*, *mode 7 (4271 cm⁻¹), +δQ*.
 */
export function callWords(c) {
    if (c.phase === "relax") return "the relaxation";
    if (c.phase === "fc") return "the force constants";
    if (c.mode == null) return "the equilibrium";
    const freq = c.frequency_cm1 != null
        ? " (" + Math.round(Number(c.frequency_cm1)) + " cm\u207b\u00b9)" : "";
    return "mode " + c.mode + freq + ", "
        + (Number(c.side) > 0 ? "+\u03b4Q" : "\u2212\u03b4Q");
}

/**
 * Any point of a run, from its wire form: a vibration's call by its phase,
 * a transport point by its frame and voltage; ``""`` for a run of one.
 */
export function runPointWords(r) {
    if (r.phase) return callWords(r);
    return r.bias_v != null ? pointWords(r.frame, r.of, r.bias_v) : "";
}
