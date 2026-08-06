/**
 * MODULE: spectrumchart — the pure arithmetic behind the picture.
 * CALLERS: spectrumchart/index.js (the handle) only. Internal to the module:
 *          nothing outside lib/spectrumchart/ may import this file.
 *
 * Two questions, answered with numbers and nothing else — no DOM, no drawing
 * library, no colours. Contract: docs/web/spectrumchart.md § 6.3 (the bands)
 * and § 9 (the envelope).
 */

/** Below this, a click target is too fine to hit with a mouse (§ 6.3, step 2). */
export const BAND_FLOOR_CM1 = 8;

/** No clamp may take a band under this, or a mode becomes unreachable (§ 6.3, step 4). */
export const BAND_MIN_HALF_WIDTH_CM1 = 0.25;

/** "A peak is smooth": samples across the full width of the narrowest peak (§ 9). */
export const SAMPLES_ACROSS_PEAK = 8;

/** "The curve ends": the grid runs until the envelope is under this share of the tallest peak (§ 9). */
export const TAIL_FRACTION = 0.01;

const isFiniteNumber = (v) => typeof v === "number" && Number.isFinite(v);

/**
 * Half-width of every mode's click band, in cm⁻¹, one per mode in the order given.
 *
 * Four steps, in the order § 6.3 states them:
 *   1. start at the broadening width;
 *   2. raise it to the floor if it is below;
 *   3. clamp it to half the gap to the nearer neighbour, so no two bands overlap;
 *   4. raise it to the minimum if step 3 took it below.
 *
 * Steps 2 and 4 widen, step 3 narrows. None of them touches the drawn curve:
 * the band is about aiming, the curve is the claim about the science.
 */
export function bandHalfWidths(freqs, broadening) {
    if (!Array.isArray(freqs)) return [];
    const width = isFiniteNumber(broadening) && broadening > 0 ? broadening : 0;
    const start = Math.max(width, BAND_FLOOR_CM1);          // steps 1 and 2

    const sorted = freqs
        .map((freq, i) => ({ freq, i }))
        .sort((a, b) => a.freq - b.freq);

    const half = new Array(freqs.length);
    for (let k = 0; k < sorted.length; k += 1) {
        const { freq, i } = sorted[k];
        let gap = Infinity;                                  // step 3
        if (k > 0) gap = Math.min(gap, freq - sorted[k - 1].freq);
        if (k < sorted.length - 1) gap = Math.min(gap, sorted[k + 1].freq - freq);
        const clamped = Math.min(start, gap / 2);
        half[i] = Math.max(clamped, BAND_MIN_HALF_WIDTH_CM1); // step 4
    }
    return half;
}

/**
 * Which modes the envelope sums over, and at what height — § 9, following § 6.2.
 *
 * With strengths known anywhere, the sum runs over the modes that have one and a
 * mode without one contributes nothing: it is missing, not weak. With none known,
 * every mode contributes a peak of height one and the curve is a frequency
 * distribution. No imaginary mode is ever in the sum, in either picture.
 */
function summedPeaks(modes) {
    const real = modes.filter((m) => !m.imaginary);
    const withActivity = real.filter((m) => isFiniteNumber(m.activity));
    if (withActivity.length > 0) {
        return withActivity.map((m) => ({ freq: m.freq, height: m.activity }));
    }
    return real.map((m) => ({ freq: m.freq, height: 1 }));
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
export function envelope(modes, broadening) {
    if (!Array.isArray(modes) || modes.length === 0) return null;
    if (!isFiniteNumber(broadening) || broadening <= 0) return null;

    const peaks = summedPeaks(modes);
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
