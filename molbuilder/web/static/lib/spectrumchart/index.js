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

/* A MODE IS NOT A HEIGHT; IT HAS HEIGHTS, ONE PER CHANNEL.
 *
 * One Hessian, several property derivatives, one set of eigenvectors --
 * so activity is an attribute of a mode IN A CHANNEL, and a chart that
 * knew only "the activity" could draw only one of them.  Channels are
 * declared at mount and each mode carries `values[key]`; this file never
 * learns what Raman or IR are, which is what leaves room for the next
 * one (the plan's spectral density) without editing it.
 *
 * `direction` is the only thing a channel says about LAYOUT: "up" and
 * "down" put it above or below the axis.  Both channels peak at the same
 * frequencies, so sharing one half-plane makes them collide -- and
 * absorption drawn downward reads the way absorption does. */
const normaliseChannel = (c, i) => ({
    key: String(c && c.key ? c.key : `channel${i}`),
    label: String(c && c.label ? c.label : "activity"),
    unit: String(c && c.unit ? c.unit : ""),
    direction: (c && c.direction === "down") ? "down" : "up",
    /* NORMALISE THIS CHANNEL'S DRAWN HEIGHTS to its own strongest band.
     *
     * Purely about heights.  What the axis is CALLED still comes from
     * `unit`, because this module names no quantity -- a caller that
     * normalises is the one that knows whether the result is a
     * percentage, an arbitrary scale, or something else again.  The
     * measured value is untouched: `valueIn` keeps returning it, so the
     * floor, the readout and the modes table stay in the units the run
     * produced. */
    relative: Boolean(c && c.relative),
});

const valueIn = (mode, key) => {
    const v = mode && mode.values ? mode.values[key] : undefined;
    return isFiniteNumber(v) ? v : null;
};

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
        if (mode.values !== undefined && mode.values !== null) {
            if (typeof mode.values !== "object") {
                return `mode ${mode.index} has values that are not an object`;
            }
            for (const [key, v] of Object.entries(mode.values)) {
                if (v !== null && v !== undefined && !isFiniteNumber(v)) {
                    return `mode ${mode.index} has a ${key} that is not a number`;
                }
            }
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

    /* Declared once, at mount: the channels this chart draws, in order.
     * A chart with no channels declared still works -- it draws the one
     * generic lane the module had before channels existed, which is what
     * a caller that only has "an activity" still means. */
    const channels = (Array.isArray(options.channels) && options.channels.length
        ? options.channels
        : [{ key: "activity", label: "activity", unit: "", direction: "up" }]
    ).map(normaliseChannel);

    /* ONE HOME EACH (§ 5.2). The selection is recorded whether or not a list
     * holds it: what is DRAWN is derived from these three, never stored beside
     * them. */
    let modes = [];
    let selected = null;
    let hovered = null;      // which mode a click would pick right now
    let broadening = 0;
    /* The display floor, as a PERCENTAGE of the strongest peak in each
     * channel.  Relative because one control serves two incommensurate
     * units; see the Results panel's own note.  It decides what is
     * DRAWN and nothing else -- no mode leaves the list, so the rug
     * still carries every one of them. */
    let floorPct = 0;
    let band = bandHalfWidth(0);     // how close counts, in cm⁻¹, from the maths
    let painted = [];                // the states last handed down, to skip repeats
    let readout = "";                // the line naming the mode nearest the pointer
    let shown = "";                  // ... and the one the chart is already showing
    let disposed = false;

    /* Which picture the data puts us in (§ 6.2), decided here and never set. */
    const laneKnown = (key) => modes.some(
        (m) => !m.imaginary && valueIn(m, key) !== null);
    /* The chart is in its "no heights" picture only when NO channel has a
     * number -- one channel still pending is not the same as a run that
     * computed nothing, and saying otherwise would blank a half-finished
     * spectrum that has real Raman in it. */
    const strengthsKnown = () => channels.some((c) => laneKnown(c.key));

    const stateOf = (mode, known, key) => (
        // Chosen outranks hovered: what you picked should not flicker away under
        // the pointer. Hovered outranks the rest, and only while the pointer is
        // there.
        mode.index === selected ? "chosen"
            : mode.index === hovered ? "hovered"
                : mode.imaginary ? "imaginary"
                    : (known && valueIn(mode, key) === null) ? "pending"
                        : "plain"
    );

    const pictureNow = () => {
        const anyKnown = strengthsKnown();
        const span = modes.length > 1
            ? Math.max(...modes.map((m) => m.freq)) - Math.min(...modes.map((m) => m.freq))
            : 100;
        const stickWidth = Math.max(span, 1) / STICK_WIDTH_FRACTION;
        const anyPending = channels.some(
            (c) => laneKnown(c.key)
                && modes.some((m) => !m.imaginary && valueIn(m, c.key) === null));

        /* THE "NO HEIGHTS" PICTURE BELONGS TO ONE LANE, NOT EVERY LANE.
         *
         * With nothing computed there is no height to draw, and § 6.2's
         * answer is a row of equal sticks saying "these modes exist".
         * Drawn once that is a mode list; drawn in BOTH half-planes it is
         * the same list mirrored, which reads as two spectra that happen
         * to agree exactly -- the most misleading thing the chart could
         * put on screen.  So the placeholder is the first lane's alone,
         * and the others stay empty until they have numbers. */
        /* Below the floor, a stick is drawn at ZERO height rather than
         * removed.  Two reasons, and the second is the load-bearing one:
         *   * the user asked for it hidden, so vanishing IS the intent
         *     -- unlike a mode whose strength was never computed, which
         *     § 6.2 refuses to draw at zero because that would read as a
         *     measured zero;
         *   * `recolour` restyles the stick trace by POSITION, so a
         *     trace whose points came and went with the slider would
         *     colour the wrong modes on the next hover. */
        const floorFor = (key) => {
            if (!(floorPct > 0)) return 0;
            const peak = Math.max(0, ...modes
                .filter((m) => !m.imaginary)
                .map((m) => Math.abs(valueIn(m, key) ?? 0)));
            return peak * (floorPct / 100);
        };

        const lanes = channels.map((c, i) => {
            const known = laneKnown(c.key);
            const cut = floorFor(c.key);
            /* A relative lane divides by its own peak on the way to the
             * screen only.  `valueIn` keeps returning the measured
             * number, so the floor, the readout and the table all stay
             * in the units the run actually produced. */
            const lanePeak = Math.max(0, ...modes
                .filter((m) => !m.imaginary)
                .map((m) => Math.abs(valueIn(m, c.key) ?? 0)));
            const toScreen = (v) => (
                c.relative && lanePeak > 0 ? (v / lanePeak) * 100 : v);
            const shown = (m) => {
                const v = valueIn(m, c.key);
                return v !== null && Math.abs(v) >= cut;
            };
            const placeholder = !anyKnown && i === 0;
            const silent = !anyKnown && i !== 0;
            return {
                key: c.key,
                direction: c.direction,
                /* The unit belongs to the CHANNEL, so it is named by
                 * whoever declared it.  The y-title used to be the
                 * string "Raman activity (Å⁴/amu)" hardcoded here --
                 * which is how a chart that was otherwise
                 * quantity-agnostic came to be a Raman viewer.  A lane
                 * with no numbers yet still names itself: an anonymous
                 * axis beside a named one reads as a second unit nobody
                 * declared. */
                title: known
                    ? `${c.label} (${c.unit})`.replace(" ()", "")
                    : (anyKnown ? `${c.label} — not computed` : "modes"),
                known,
                sticks: silent
                    ? { x: [], y: [], width: [], state: [] }
                    : {
                        x: modes.map((m) => m.freq),
                        y: modes.map((m) => {
                            if (placeholder) return 1;
                            const v = valueIn(m, c.key);
                            if (v === null) return 0;
                            return shown(m) ? toScreen(v) : 0;
                        }),
                        width: modes.map(() => stickWidth),
                        state: modes.map((m) => stateOf(m, known, c.key)),
                    },
                /* The envelope is a sum over the bands that are drawn,
                 * so a filtered band must not keep contributing to it --
                 * otherwise raising the floor would empty the sticks
                 * while the curve stayed exactly where it was. */
                curve: silent
                    ? null
                    : envelope(modes, broadening,
                               (m) => (shown(m)
                                   ? toScreen(valueIn(m, c.key)) : null)),
            };
        });

        /* THE RUG CARRIES EVERY MODE, AND ONLY ITS POSITION.
         *
         * A mode active in neither channel has no height in either, so
         * the sticks cannot show it at all -- and giving it one would be
         * a lie in whichever direction it pointed.  Ticks on the axis
         * are the only honest place it can appear, which is why the rug
         * exists and why it never varies in height.  `cls` is the
         * classification decided server-side (spectra/activity.py), not
         * a threshold re-invented here. */
        const rug = {
            x: modes.map((m) => m.freq),
            cls: modes.map((m) => (typeof m.cls === "string" ? m.cls : "partial")),
            index: modes.map((m) => m.index),
        };

        return {
            lanes,
            rug,
            readout,
            xTitle: "frequency (cm⁻¹)",
            note: anyKnown
                ? (anyPending ? "× strengths not computed for these modes" : "")
                : "strengths not computed — height means nothing here",
        };
    };

    const redraw = () => {
        band = bandHalfWidth(broadening);
        const picture = pictureNow();
        painted = picture.lanes.length ? picture.lanes[0].sticks.state : [];
        surface.draw(picture);
    };

    /** § 5.1 — the cheap door: the same picture, with only its colours and its
     * readout changed.
     *
     * And nothing at all when neither would come out different. Hovering the mode
     * that is already chosen is the everyday case: the answer changes, the
     * picture does not, and the cheapest call is the one not made. */
    const recolour = () => {
        const key = channels.length ? channels[0].key : "activity";
        const states = modes.map((m) => stateOf(m, laneKnown(key), key));
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
        /* EVERY channel that has a number for this mode, each with its
         * own unit.  The readout used to name one quantity in one
         * hardcoded unit, which on a mirror plot would describe half of
         * what is on screen and silently mislabel the other half. */
        const strengths = channels
            .map((c) => {
                const v = valueIn(mode, c.key);
                if (v === null) return null;
                const unit = c.unit ? ` ${c.unit}` : "";
                return `${v.toFixed(2)}${unit}`;
            })
            .filter((t) => t !== null)
            .map((t) => `  ·  ${t}`)
            .join("");
        return `mode ${mode.index}  ·  ${mode.freq.toFixed(1)} cm⁻¹${strengths}`;
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
            /* Copied field by field on the way in, so the caller's object
             * cannot change under the chart afterwards.  That is also why
             * the copy has to know about `values` and `cls`: a projection
             * that forgets a field silently drops it, and the chart then
             * draws a spectrum with no heights while the caller can see
             * the numbers it passed. */
            modes = list.map((m) => ({
                index: m.index,
                freq: m.freq,
                values: channels.reduce((acc, c) => {
                    acc[c.key] = valueIn(m, c.key);
                    return acc;
                }, {}),
                cls: typeof m.cls === "string" ? m.cls : "partial",
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

        /** § 8.3 — what to hide, as a percentage of each channel's peak.
         *
         * Out-of-range values leave the current floor standing, for the
         * same reason a bad broadening does: a substituted default would
         * hide the caller's bug. */
        setDisplayFloor(pct) {
            if (disposed) return;
            if (!isFiniteNumber(pct) || pct < 0 || pct > 100) return;
            floorPct = pct;
            redraw();
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
