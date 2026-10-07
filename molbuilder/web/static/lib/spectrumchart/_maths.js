/**
 * MODULE: spectrumchart — the pure arithmetic behind the picture.
 * CALLERS: spectrumchart/index.js (the handle) only. Internal to the module:
 *          nothing outside lib/spectrumchart/ may import this file.
 *
 * Two questions, answered with numbers and nothing else — no DOM, no drawing
 * library, no colours. Contract: docs/web/spectrumchart.md § 6.3 (the bands)
 * and § 9 (the envelope).
 */

/** How far from a peak still counts as clicking it, when the lines are drawn narrow. */
const BAND_FLOOR_CM1 = 8;

/** "A peak is smooth": samples across the full width of the narrowest peak (§ 9). */
const SAMPLES_ACROSS_PEAK = 8;

/** "The curve ends": the grid runs until the envelope is under this share of the tallest peak (§ 9). */
const TAIL_FRACTION = 0.01;

const isFiniteNumber = (v) => typeof v === "number" && Number.isFinite(v);

/**
 * How close to a mode still counts as clicking it, in cm⁻¹ — the broadening
 * width, or the floor if that is narrower (§ 6.3).
 *
 * ONE width, for every mode. Bands overlap wherever modes are closer together
 * than that, and the handle resolves an overlap by taking the NEAREST mode,
 * which is an answer that never needs a tie-break.
 */
export function bandHalfWidth(broadening) {
    const width = isFiniteNumber(broadening) && broadening > 0 ? broadening : 0;
    return Math.max(width, BAND_FLOOR_CM1);
}

/**
 * Which modes the envelope sums over, and at what height — § 9, following § 6.2.
 *
 * The sum runs over the modes that have a strength, and a mode without one
 * contributes nothing: it is missing, not weak. With none known there is no
 * curve at all — a sum over heights nobody computed would be a picture of
 * nothing (§ 6.2).
 * No imaginary mode is ever in the sum.
 */
/* WHICH NUMBER IS THIS CURVE'S HEIGHT is the caller's to say.
 *
 * One Hessian yields several property derivatives on one set of
 * eigenvectors, so "the activity" is not a property of a mode -- it is a
 * property of a mode IN A CHANNEL.  Taking the height as a parameter is
 * what lets one envelope serve Raman, IR and whatever is added next
 * (the plan's spectral density) without this file learning their names
 * or their units. */
const defaultHeight = (m) => m.activity;

function summedPeaks(modes, heightOf = defaultHeight) {
    const real = modes.filter((m) => !m.imaginary);
    return real.filter((m) => isFiniteNumber(heightOf(m)))
               .map((m) => ({ freq: m.freq, height: heightOf(m) }));
}

const lorentzianAt = (x, peaks, gamma) => {
    const g2 = gamma * gamma;
    let y = 0;
    for (const p of peaks) {
        const d = x - p.freq;
        y += p.height * (g2 / (d * d + g2));
    }
    return y;
};

/**
 * The broadened curve: {x, y} sampled arrays, or null when there is no curve to draw.
 *
 *              γ²
 *   y(x) = Σ A ────────────      γ = half the width asked for
 *            i (x−x_i)² + γ²
 *
 * Where it is sampled follows the width, never a fixed setting, and both
 * requirements are stated as accuracy so a test can check the result rather than
 * the arithmetic: at least SAMPLES_ACROSS_PEAK samples across a peak's full
 * width, and the grid runs out until the curve is under TAIL_FRACTION of the
 * tallest peak. A width of zero means bare sticks — no curve at all.
 */
export function envelope(modes, broadening, heightOf = defaultHeight) {
    if (!Array.isArray(modes) || modes.length === 0) return null;
    if (!isFiniteNumber(broadening) || broadening <= 0) return null;

    const peaks = summedPeaks(modes, heightOf);
    if (peaks.length === 0) return null;

    const gamma = broadening / 2;
    const step = broadening / SAMPLES_ACROSS_PEAK;

    const lo = Math.min(...peaks.map((p) => p.freq));
    const hi = Math.max(...peaks.map((p) => p.freq));
    const tallest = Math.max(...peaks.map((p) => Math.abs(p.height)));
    const cutoff = TAIL_FRACTION * tallest;

    // A single Lorentzian falls to TAIL_FRACTION of its height at γ√(1/f − 1);
    // where modes pile up the tails add, so extend until the curve itself is under.
    let pad = gamma * Math.sqrt(1 / TAIL_FRACTION - 1);
    for (let i = 0; i < 8; i += 1) {
        const ends = Math.max(
            lorentzianAt(lo - pad, peaks, gamma),
            lorentzianAt(hi + pad, peaks, gamma),
        );
        if (ends <= cutoff) break;
        pad *= 1.5;
    }

    /* The grid is a window around each mode rather than one ruler laid across the
     * whole spectrum. Both give the same curve where the curve is worth drawing,
     * but the work here scales with the NUMBER OF MODES instead of with how
     * narrow the lines are: one ruler at a broadening of 0.01 cm-1 across a
     * 3000 cm-1 spectrum is two million points, nearly all of them in empty
     * space, and the browser stops. Windowed, a mode costs about 80 points at
     * every width. */
    const reach = Math.ceil(pad / step);
    const marks = new Set();
    for (const p of peaks) {
        for (let k = -reach; k <= reach; k += 1) marks.add(p.freq + k * step);
    }

    const x = Array.from(marks).sort((a, b) => a - b);
    const y = x.map((v) => lorentzianAt(v, peaks, gamma));
    return { x, y };
}
