/**
 * MODULE: spectrumchart — the handle, and the only way in.
 * CALLERS: any page that wants a vibrational spectrum. Today: the Spectrum tab.
 *
 * `mount` is the whole importable surface of this module. Everything else here
 * is reached through the object it hands back: five doors in, one callback out.
 * The modes, the selection and the broadening each live here once; the drawing
 * library is not visible from this file at all.
 *
 * Contract: docs/web/spectrumchart.md — § 7 (mount and dispose), § 8.2 (what
 * mount takes), § 8.3 (the doors), § 6 (the data), § 5.1 (what each door costs).
 */

import { bandHalfWidth, envelope } from "./_maths.js";
import { openSurface } from "./_seal.js";

/** A stick is a line, not a block: thin against whatever range is on screen. */
const STICK_WIDTH_FRACTION = 400;

/* HOW CLOSE COUNTS, AS A FLOOR IN PIXELS.
 *
 * The band is the Lorentz width, so what you aim at is what you see (§ 6.3) —
 * but that is a width in cm⁻¹, and what it is worth on screen depends entirely
 * on how much spectrum is in view. Twenty wavenumbers is a comfortable target
 * across a wide panel and about a pixel and a half in a narrow one, which is a
 * target nobody can hit. So a band is never narrower than this on screen,
 * however far out the axis is zoomed. */
const MIN_PICK_PX = 8;

const isFiniteNumber = (v) => typeof v === "number" && Number.isFinite(v);

/** § 7 — failure is a handle carrying three keys and no others. */
const failedMount = (error) => ({ ok: false, error, dispose() {} });

/**
 * Every reason a list is refused, or null if it is good (§ 6.1).
 *
 * A record that must be refused takes the whole call with it, so this answers
 * about the list rather than about a record: one bad row and nothing is drawn.
 */
function faultIn(list) {
    if (!Array.isArray(list)) return "setModes takes an array of modes";
    const seen = new Set();
    for (const mode of list) {
        if (!mode || typeof mode !== "object") return "each mode must be an object";
        if (!isFiniteNumber(mode.index)) return `a mode has no usable index: ${JSON.stringify(mode)}`;
        if (!isFiniteNumber(mode.freq)) return `mode ${mode.index} has no usable freq`;
        if (mode.activity !== undefined && mode.activity !== null
            && !isFiniteNumber(mode.activity)) {
            return `mode ${mode.index} has an activity that is not a number`;
        }
        if (seen.has(mode.index)) return `two modes share the index ${mode.index}`;
        seen.add(mode.index);
    }
    return null;
}

export async function mount(host, options = {}) {
    // § 8.2 — the one thing mount refuses: somewhere to draw that is not an element.
    if (!host || typeof host.appendChild !== "function") {
        return failedMount("SpectrumChart needs an element to mount into");
    }
    host.replaceChildren();          // § 7 — the module owns the inside of its host

    let surface;
    try {
        surface = await openSurface(host);
    } catch (err) {
        return failedMount(err && err.message ? err.message : String(err));
    }

    const onSelect = typeof options.onSelect === "function" ? options.onSelect : null;

    /* ONE HOME EACH (§ 5.2). The selection is recorded whether or not a list
     * holds it: what is DRAWN is derived from these three, never stored beside
     * them. */
    let modes = [];
    let selected = null;
    let hovered = null;      // which mode a click would pick right now
    let broadening = 0;
    let band = bandHalfWidth(0);     // how close counts, in cm⁻¹, from the maths
    let painted = [];                // the states last handed down, to skip repeats
    let readout = "";                // the line naming the mode nearest the pointer
    let shown = "";                  // ... and the one the chart is already showing
    let disposed = false;

    /* Which picture the data puts us in (§ 6.2), decided here and never set. */
    const strengthsKnown = () => modes.some((m) => isFiniteNumber(m.activity) && !m.imaginary);

    const stateOf = (mode, known) => (
        // Chosen outranks hovered: what you picked should not flicker away under
        // the pointer. Hovered outranks the rest, and only while the pointer is
        // there.
        mode.index === selected ? "chosen"
            : mode.index === hovered ? "hovered"
                : mode.imaginary ? "imaginary"
                    : (known && !isFiniteNumber(mode.activity)) ? "pending"
                        : "plain"
    );

    const pictureNow = () => {
        const known = strengthsKnown();
        const span = modes.length > 1
            ? Math.max(...modes.map((m) => m.freq)) - Math.min(...modes.map((m) => m.freq))
            : 100;
        const stickWidth = Math.max(span, 1) / STICK_WIDTH_FRACTION;
        const anyPending = known && modes.some((m) => !m.imaginary && !isFiniteNumber(m.activity));

        return {
            sticks: {
                x: modes.map((m) => m.freq),
                y: modes.map((m) => (known ? (isFiniteNumber(m.activity) ? m.activity : 0) : 1)),
                width: modes.map(() => stickWidth),
                state: modes.map((m) => stateOf(m, known)),
            },
            curve: envelope(modes, broadening),
            readout,
            xTitle: "frequency (cm⁻¹)",
            yTitle: known ? "Raman activity (Å⁴/amu)" : "modes",
            note: known
                ? (anyPending ? "× strengths not computed for these modes" : "")
                : "strengths not computed — height means nothing here",
        };
    };

    const redraw = () => {
        band = bandHalfWidth(broadening);
        const picture = pictureNow();
        painted = picture.sticks.state;
        surface.draw(picture);
    };

    /** § 5.1 — the cheap door: the same picture, with only its colours and its
     * readout changed.
     *
     * And nothing at all when neither would come out different. Hovering the mode
     * that is already chosen is the everyday case: the answer changes, the
     * picture does not, and the cheapest call is the one not made. */
    const recolour = () => {
        const known = strengthsKnown();
        const states = modes.map((m) => stateOf(m, known));
        const same = states.length === painted.length && states.every((s, i) => s === painted[i]);
        if (same && readout === shown) return;      // nothing on screen would differ
        painted = states;
        shown = readout;
        // `null` for the colours means "unchanged": moving the pointer inside one
        // band changes the words alone, and repainting the same colours to say so
        // would be work with nothing to show for it.
        surface.recolour(same ? null : states, readout);
    };

    /** The mode closest to a frequency, and how far off it was.
     *
     * One walk answers both questions the pointer asks — "what am I near", which
     * has an answer everywhere, and "what would a click take", which has one only
     * inside a band. They differ by the test applied to `away`, not by the search.
     */
    const nearest = (x) => {
        let found = null;
        let away = Infinity;
        for (const mode of modes) {
            const d = Math.abs(x - mode.freq);
            if (d < away) { away = d; found = mode; }
        }
        return found && { mode: found, away };
    };

    /** A position becomes a mode here, never below (§ 8.4).
     *
     * `perPixel` comes up with the position because the band has to be a target
     * on a screen as well as a width in the science: whichever is wider, the
     * Lorentz width or MIN_PICK_PX, is how close counts. */
    const modeAt = (at) => {
        const near = nearest(at.x);
        if (!near) return null;
        const reach = Math.max(band, MIN_PICK_PX * (at.perPixel || 0));
        return near.away <= reach ? near.mode.index : null;
    };

    /** The words: what the pointer is near, whether or not a click would take it. */
    const readoutFor = (at) => {
        const near = at === null ? null : nearest(at.x);
        if (!near) return "";
        const { mode } = near;
        const strength = isFiniteNumber(mode.activity)
            ? `  ·  ${mode.activity.toFixed(2)} Å⁴/amu` : "";
        return `mode ${mode.index}  ·  ${mode.freq.toFixed(1)} cm⁻¹${strength}`;
    };

    /* A band is invisible, so the mode a click would pick is a guess until the
     * chart says so: the one under the pointer lights up as you move.
     *
     * A pointer crossing the plot fires hundreds of times a second, so this
     * redraws only when the ANSWER changes -- and then only colours (§ 5.1's
     * cheap door). Sliding along inside one band costs nothing at all. */
    surface.onHover((at) => {
        if (disposed) return;
        hovered = at === null ? null : modeAt(at);
        readout = readoutFor(at);
        recolour();          // does nothing if neither the colours nor the words moved
    });

    surface.onClick((at) => {
        if (disposed || !onSelect) return;
        const index = modeAt(at);
        // § 8.3 — a click in no band reports nothing at all, not null.
        if (index !== null) onSelect(index);
    });

    /* § 5.4 — the module watches its own box, because a panel opening beside it
     * changes the box while the window sits still. */
    let watcher = null;
    if (typeof globalThis.ResizeObserver === "function") {
        watcher = new globalThis.ResizeObserver(() => {
            if (!disposed) surface.resize();
        });
        watcher.observe(host);
    }

    const handle = {
        ok: true,

        setModes(list) {
            if (disposed) return;
            const fault = faultIn(list);
            if (fault) {
                // § 6.1 — the whole call, and the chart empties rather than leaving
                // the last spectrum standing as though this had worked.
                modes = [];
                redraw();
                console.warn(`SpectrumChart refused a mode list: ${fault}`);
                return;
            }
            modes = list.map((m) => ({
                index: m.index,
                freq: m.freq,
                activity: isFiniteNumber(m.activity) ? m.activity : null,
                imaginary: Boolean(m.imaginary),
            }));
            redraw();
        },

        setSelected(index) {
            if (disposed) return;
            // Recorded whether or not the current list holds it; the highlight is
            // derived, so a later setModes brings it back without a second call.
            selected = index === null || index === undefined ? null : index;
            recolour();
        },

        setBroadening(width) {
            if (disposed) return;
            // § 8.3 — a bad width leaves the one already set standing, because a
            // substituted default would hide the caller's bug.
            if (!isFiniteNumber(width) || width < 0) return;
            broadening = width;
            redraw();
        },

        refit() {
            if (disposed) return;
            surface.resize();
        },

        dispose() {
            if (disposed) return;
            disposed = true;
            hovered = null;
            if (watcher) watcher.disconnect();
            surface.purge();
            host.replaceChildren();
        },
    };

    // § 8.2 — a mount option is the first write through the door of the same name.
    if (options.modes !== undefined) handle.setModes(options.modes);
    if (options.selected !== undefined) handle.setSelected(options.selected);
    if (options.broadening !== undefined) handle.setBroadening(options.broadening);
    if (options.modes === undefined && options.broadening === undefined) redraw();

    return handle;
}
