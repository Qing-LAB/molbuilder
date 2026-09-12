/**
 * MODULE: spectrumchart — the sealed layer.
 * CALLERS: spectrumchart/index.js (the handle) only. Internal to the module:
 *          nothing outside lib/spectrumchart/ may import this file.
 *
 * THE ONLY FILE IN THIS MODULE THAT NAMES PLOTLY. Commands go down, one number
 * comes up. It answers no question about what is drawn, and it does not know
 * what a mode is: a click leaves here as a position on the frequency axis, and
 * working out which mode that means is the handle's job.
 *
 * Contract: docs/web/spectrumchart.md § 8.4, § 4 (it brings the library and the
 * stylesheet itself) and § 11 (it reads the module's own values for colour).
 */

/** Served by the app from the installed plotly package — app.py `vendor_plotly_js`. */
const PLOTLY_URL = "/vendor/plotly.min.js";
const STYLESHEET_URL = "/static/lib/spectrumchart/_style.css";

const FRAME_CLASS = "spectrumchart";
const SURFACE_CLASS = "spectrumchart-surface";

/* One page, one link and one library, however many charts are mounted. Each
 * promise is kept so a second mount waits on the first fetch instead of
 * starting another. */
let stylesheetOnce = null;
let libraryOnce = null;

function loadStylesheet(doc) {
    if (stylesheetOnce) return stylesheetOnce;
    const already = doc.querySelector(`link[data-spectrumchart="style"]`);
    if (already) {
        stylesheetOnce = Promise.resolve();
        return stylesheetOnce;
    }
    stylesheetOnce = new Promise((resolve, reject) => {
        const link = doc.createElement("link");
        link.rel = "stylesheet";
        link.href = STYLESHEET_URL;
        link.setAttribute("data-spectrumchart", "style");
        link.onload = () => resolve();
        link.onerror = () => {
            stylesheetOnce = null;   // a failure is not an answer to keep
            reject(new Error("the chart stylesheet could not be loaded"));
        };
        (doc.head || doc.documentElement).appendChild(link);
    });
    return stylesheetOnce;
}

function loadLibrary(win) {
    // A page may already carry it — results.html does. Use what is there.
    if (win.Plotly) return Promise.resolve(win.Plotly);
    if (libraryOnce) return libraryOnce;
    const doc = win.document;
    libraryOnce = new Promise((resolve, reject) => {
        const script = doc.createElement("script");
        script.src = PLOTLY_URL;
        script.setAttribute("data-spectrumchart", "library");
        script.onload = () => (win.Plotly
            ? resolve(win.Plotly)
            : reject(new Error("the plotting library loaded but published nothing")));
        script.onerror = () => {
            libraryOnce = null;      // ditto: a later mount may reach the server
            reject(new Error("the plotting library could not be loaded"));
        };
        (doc.head || doc.documentElement).appendChild(script);
    });
    return libraryOnce;
}

/** Every colour the library is handed comes from a value on the frame (§ 11). */
function paletteOf(win, frame) {
    const styles = win.getComputedStyle(frame);
    const read = (name) => String(styles.getPropertyValue(name) || "").trim();
    return {
        bg: read("--spectrumchart-bg"),
        ink: read("--spectrumchart-ink"),
        grid: read("--spectrumchart-grid"),
        stick: read("--spectrumchart-stick"),
        chosen: read("--spectrumchart-chosen"),
        curve: read("--spectrumchart-curve"),
        pending: read("--spectrumchart-pending"),
        imaginary: read("--spectrumchart-imaginary"),
        hovered: read("--spectrumchart-hovered"),
        /* One colour per lane and one per activity class.  A mirror plot
         * is only readable if each axis is tied to its own curve by
         * colour, and the rug is only readable if its classes differ. */
        laneUp: read("--spectrumchart-lane-up"),
        laneDown: read("--spectrumchart-lane-down"),
        rugBoth: read("--spectrumchart-rug-both"),
        rugIr: read("--spectrumchart-rug-ir"),
        rugRaman: read("--spectrumchart-rug-raman"),
        rugSilent: read("--spectrumchart-rug-silent"),
        rugPartial: read("--spectrumchart-rug-partial"),
    };
}

/* A mark's STATE arrives from above; what that state looks like is decided
 * here and nowhere else. The layer above never names a colour. */
const rugColourFor = (cls, palette) => (
    cls === "both" ? palette.rugBoth
        : cls === "ir-only" ? palette.rugIr
            : cls === "raman-only" ? palette.rugRaman
                : cls === "silent" ? palette.rugSilent
                    : palette.rugPartial
);

const laneColourFor = (direction, palette) => (
    direction === "down" ? palette.laneDown : palette.laneUp
);

const colourFor = (state, palette) => (
    state === "chosen" ? palette.chosen
        : state === "hovered" ? palette.hovered
            : state === "imaginary" ? palette.imaginary
                : state === "pending" ? palette.pending
                    : palette.stick
);

/**
 * Build the drawing surface inside `host` and return the five doors.
 *
 * Resolves once the stylesheet and the library are both there — the mount above
 * is asynchronous anyway, so waiting costs a caller nothing and the palette is
 * readable by the time anything is drawn. Rejects with a message a tab can show.
 */
export async function openSurface(host) {
    const doc = host.ownerDocument || globalThis.document;
    const win = (doc && doc.defaultView) || globalThis;

    await loadStylesheet(doc);
    const Plotly = await loadLibrary(win);

    const frame = doc.createElement("div");
    frame.className = FRAME_CLASS;
    const surface = doc.createElement("div");
    surface.className = SURFACE_CLASS;
    frame.appendChild(surface);
    host.appendChild(frame);

    const palette = paletteOf(win, frame);

    /* Everything the user reads is drawn IN the plot (§ 6.2): the pending note on
     * the left, the pointer's readout on the right. Neither is a text node beside
     * the chart, which is what keeps a chart one drawing surface. */
    const annotationsFor = (note, readout) => {
        const at = (text, x, anchor, colour) => ({
            text, showarrow: false,
            xref: "paper", yref: "paper", x, y: 1.04,
            xanchor: anchor, font: { color: colour, size: 11 },
        });
        const out = [];
        if (note) out.push(at(note, 0, "left", palette.pending));
        out.push(at(readout || "", 1, "right", palette.ink));   // always present, may be empty
        return out;
    };

    const layoutFor = (picture) => ({
        paper_bgcolor: palette.bg,
        plot_bgcolor: palette.bg,
        font: { color: palette.ink, size: 11 },
        /* `t` is room above the plot for the two lines that live there:
         * the note on the left, the pointer's readout on the right
         * (§ 6.2).  At the 12px it used to be, both sat in the margin's
         * edge and were clipped -- which is why the pending note had
         * never once been seen.
         *
         * `r` is bare, because both axes are on the LEFT.  It briefly
         * needed 56px for a right-hand axis, whose ticks rendered "60"
         * as "6C" at 12px -- kept as the reason this number is a
         * decision rather than a default. */
        margin: { l: 58, r: 14, t: 30, b: 40 },
        showlegend: false,
        /* No hover labels from the library. The chart carries its own readout,
         * which names the nearest mode wherever the pointer is (§ 6.3.1) -- a
         * tooltip that appears only when you are exactly on a stick would say
         * less, and would need the aim the bands exist to make unnecessary. */
        hovermode: false,
        /* Every axis comes from `axesFor` -- including x, which it must
         * own because the anchor depends on how many panels there are.
         * A second `xaxis` literal here was dead for exactly that
         * reason: the spread below overrode it silently, so editing it
         * changed nothing. */
        ...axesFor(picture),
        shapes: darkModeLines(picture),
        annotations: annotationsFor(picture.note, picture.readout),
    });

    /* A MODE WITH NO INTENSITY STILL EXISTS.
     *
     * Spectroscopically dark is not motionless: a mode forbidden in both
     * channels is a real vibration of the molecule, and drawing it only
     * as a tick on the axis says the opposite -- that it is a footnote
     * to the spectra rather than one of the modes they are spectra OF.
     *
     * So it gets a thin line across the FULL HEIGHT of every panel.
     * Full height is what makes it honest: a line that spans the axis
     * asserts a FREQUENCY and no intensity at all, where any finite bar
     * height would be a claim about a quantity that is zero.
     *
     * Drawn `below` the traces, so it marks the position without ever
     * competing with a band for the reader's eye. */
    const darkModeLines = (picture) => {
        const rug = picture.rug;
        const lanes = picture.lanes || [];
        if (!rug || !rug.x || !rug.x.length) return [];
        const dark = rug.x.filter((_, i) => (rug.cls || [])[i] === "silent");
        if (!dark.length) return [];
        const panels = Math.max(1, lanes.length);
        const out = [];
        for (let panel = 0; panel < panels; panel += 1) {
            const ref = axisRefFor(panel);
            for (const x of dark) {
                out.push({
                    type: "line",
                    layer: "below",
                    xref: "x",
                    yref: `${ref} domain`,
                    x0: x, x1: x, y0: 0, y1: 1,
                    line: { color: palette.rugSilent, width: 1, dash: "dot" },
                });
            }
        }
        return out;
    };

    /* TWO PANELS, ONE FREQUENCY AXIS.
     *
     * Both channels peak at the same frequencies, so they cannot share a
     * half-plane -- but they do not need to share a FRAME either.  Each
     * lane gets its own stacked panel with its own axis on the LEFT,
     * against one shared x underneath: the quantities are incommensurate
     * (Å⁴/amu against a normalised absorption scale), and two left axes
     * read as two measurements of one sample, where a left/right pair
     * reads as one plot with a trick in it.
     *
     * The panels are stacked in lane order, so a lane declared "down"
     * sits below -- the direction still decides position, it is just no
     * longer a sign on the numbers.
     *
     * Plotly's domain runs bottom-to-top, so the FIRST lane takes the
     * upper band and the last takes the lower one. */
    const PANEL_GAP = 0.10;

    const domainsFor = (n) => {
        if (n <= 1) return [[0, 1]];
        const each = (1 - PANEL_GAP * (n - 1)) / n;
        const out = [];
        for (let i = 0; i < n; i += 1) {
            const top = 1 - i * (each + PANEL_GAP);
            out.push([Math.max(0, top - each), top]);
        }
        return out;
    };

    const axisKeyFor = (i) => (i === 0 ? "yaxis" : `yaxis${i + 1}`);
    const axisRefFor = (i) => (i === 0 ? "y" : `y${i + 1}`);

    const axisRange = (lane) => {
        const vals = (lane.sticks.y || []).map(Math.abs).filter(Number.isFinite);
        const curve = (lane.curve && lane.curve.y) || [];
        const peak = Math.max(0, ...vals, ...curve.map(Math.abs));
        return peak > 0 ? peak * 1.08 : 1;
    };

    const axesFor = (picture) => {
        const lanes = picture.lanes || [];
        const domains = domainsFor(Math.max(1, lanes.length));
        const out = {};
        lanes.forEach((lane, i) => {
            out[axisKeyFor(i)] = {
                title: { text: lane.title,
                         font: { color: laneColourFor(lane.direction, palette) } },
                domain: domains[i],
                gridcolor: palette.grid,
                zeroline: false,
                rangemode: "tozero",
                range: [0, axisRange(lane)],
                tickfont: { color: laneColourFor(lane.direction, palette) },
            };
        });
        /* One x-axis, anchored under the LAST panel and shared by the
         * rest: the whole point of stacking is that a frequency lines up
         * vertically across every channel. */
        out.xaxis = {
            title: { text: picture.xTitle || "" },
            gridcolor: palette.grid,
            zeroline: false,
            anchor: axisRefFor(Math.max(0, lanes.length - 1)),
        };
        lanes.forEach((_, i) => {
            if (i < lanes.length - 1) {
                out[axisKeyFor(i)].matches = undefined;
            }
        });
        return out;
    };

    /* Which trace holds the selectable sticks; set by `tracesFor`,
     * read by `recolour`, which must reach the same one. */
    let stickTraceIndex = 0;

    const tracesFor = (picture) => {
        const lanes = picture.lanes || [];
        const traces = [];
        /* Trace order is FIXED, because `recolour` reaches back into the
         * sticks by index and a layout that reordered them would recolour
         * the wrong thing.  Per lane: curve, then sticks, then the marks
         * for modes with no height.  The rug goes last, on top. */
        let firstSticks = -1;

        lanes.forEach((lane, laneIndex) => {
            /* Each lane draws in its own panel, upright.  The mirror's
             * sign is gone with the mirror: a channel is placed by which
             * panel it is in, not by pointing its numbers downward, and
             * an axis that reads 0 upward is one less thing between a
             * reader and the value. */
            const axis = axisRefFor(laneIndex);
            const sign = 1;
            const laneColour = laneColourFor(lane.direction, palette);

            if (lane.curve) {
                traces.push({
                    type: "scatter",
                    mode: "lines",
                    x: lane.curve.x,
                    y: lane.curve.y.map((v) => v * sign),
                    yaxis: axis,
                    line: { color: laneColour, width: 1.5 },
                    /* A filled envelope is what lets two channels overlap
                     * and still be read apart -- the plan's "transparency
                     * fill".  Against the axis, not against each other,
                     * because they live in opposite half-planes. */
                    fill: "tozeroy",
                    fillcolor: withAlpha(laneColour, 0.16),
                    hoverinfo: "skip",
                });
            }

            const pending = [];
            const drawn = { x: [], y: [], width: [], colour: [] };
            lane.sticks.x.forEach((x, i) => {
                if (lane.sticks.state[i] === "pending") { pending.push(x); return; }
                drawn.x.push(x);
                drawn.y.push(lane.sticks.y[i] * sign);
                drawn.width.push(lane.sticks.width[i]);
                /* The FIRST lane keeps the selection colours -- it is the
                 * one `recolour` can reach, and the one the mode table is
                 * ordered by.  A second lane is drawn in its channel's own
                 * colour so the mirror stays legible; selection is shown
                 * once, not twice. */
                drawn.colour.push(firstSticks < 0
                    ? colourFor(lane.sticks.state[i], palette)
                    : laneColour);
            });

            if (firstSticks < 0) firstSticks = traces.length;
            traces.push({
                type: "bar",
                x: drawn.x,
                y: drawn.y,
                width: drawn.width,
                yaxis: axis,
                marker: { color: drawn.colour },
                hoverinfo: "skip",
            });

            if (pending.length) {
                traces.push({
                    type: "scatter",
                    mode: "markers",
                    x: pending,
                    y: pending.map(() => 0),
                    yaxis: axis,
                    marker: { color: palette.pending, symbol: "x", size: 7 },
                    hoverinfo: "skip",
                });
            }
        });

        /* EVERY MODE, POSITION ONLY.  Ticks on the zero line, one per
         * mode, coloured by activity class -- the only place a mode
         * active in NEITHER channel can honestly appear, since it has no
         * height to draw in either.  Height is fixed, never scaled: a
         * silent mode given a height would be a lie in either direction. */
        const rug = picture.rug;
        /* THE RUG REPEATS IN EVERY PANEL.  A mode active in neither
         * channel belongs to the whole measurement, not to one of its
         * halves -- and a reader comparing the panels has to see the
         * same frequency marked in both, or the rug would look like a
         * property of whichever panel happened to carry it. */
        const rugPanels = Math.max(1, lanes.length);
        if (rug && rug.x && rug.x.length) {
            for (let panel = 0; panel < rugPanels; panel += 1) traces.push({
                type: "scatter",
                mode: "markers",
                x: rug.x,
                y: rug.x.map(() => 0),
                yaxis: axisRefFor(panel),
                /* PER-POINT COLOUR GOES ON `marker.color`, NOT
                 * `marker.line.color`.  An open symbol is stroked, so
                 * `marker.line` looks like the right home -- but Plotly
                 * ignores a per-point ARRAY there and falls back to the
                 * trace's automatic colour, which painted all four CO2
                 * ticks the same violet while the data underneath was
                 * correct.  Caught in the browser; no test asserts a
                 * rendered stroke. */
                marker: {
                    symbol: "line-ns-open",
                    size: 11,
                    color: (rug.cls || []).map((c) => rugColourFor(c, palette)),
                    line: {
                        width: 2,
                        color: (rug.cls || []).map((c) => rugColourFor(c, palette)),
                    },
                },
                hoverinfo: "skip",
            });
        }
        stickTraceIndex = firstSticks < 0 ? 0 : firstSticks;
        return traces;
    };

    /* Plotly takes a fill colour as its own value, not as an opacity on
     * the line, so the lane's colour has to be re-expressed with alpha.
     * Handles the three forms a CSS custom property actually arrives in. */
    const withAlpha = (colour, alpha) => {
        const c = String(colour || "").trim();
        let m = c.match(/^#([0-9a-f]{6})$/i);
        if (m) {
            const n = parseInt(m[1], 16);
            return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
        }
        m = c.match(/^#([0-9a-f]{3})$/i);
        if (m) {
            const [r, g, b] = m[1].split("").map((h) => parseInt(h + h, 16));
            return `rgba(${r}, ${g}, ${b}, ${alpha})`;
        }
        m = c.match(/^rgba?\(([^)]+)\)$/i);
        if (m) {
            const parts = m[1].split(",").map((v) => v.trim());
            return `rgba(${parts[0]}, ${parts[1]}, ${parts[2]}, ${alpha})`;
        }
        return c;   // a named colour or something exotic: leave it opaque
    };

    let lastReadout = null;
    let readoutIndex = 0;   // where the readout sits: after the note, when there is one
    let disposed = false;
    let clicked = null;   // the handle's callback

    const plotArea = () => surface.querySelector && surface.querySelector(".nsewdrag");

    /* WHERE THE POINTER IS, AND WHAT A PIXEL IS WORTH.
     *
     * The library reports a click only when the pointer is over one of its own
     * points, so nothing it offers can hear a click in the empty space beside a
     * peak — and that space is the whole purpose of the bands (§ 6.3). So the
     * position is taken from the surface itself and converted here, against the
     * axis the library is currently drawing.
     *
     * The SCALE goes up with it, because a band is a width in cm⁻¹ and the
     * layer above has no way to know what that is worth on screen: twenty
     * wavenumbers is a comfortable target across a wide panel and a pixel and a
     * half in a narrow one. Both numbers are about where the pointer is; neither
     * says anything about what is drawn.
     *
     * This is the one place in the module that reads inside the library, and it
     * is the file whose job is to know it (§ 8.4): what goes up is two numbers,
     * and which mode they mean is decided above.
     *
     * VERTICAL POSITION IS NOT CONSULTED. A spectrum is picked by frequency, so
     * anywhere in the chart at that frequency means the same thing — clicking
     * level with a peak's tip and clicking down near the axis are the same
     * request. What is excluded is the library's own toolbar, and that is
     * excluded by WHAT was clicked rather than by where: a button is a button
     * wherever it sits.
     */
    const onToolbar = (event) => {
        let node = event.target;
        while (node && node !== surface) {
            /* A class name is not one kind of value: HTML gives a string, SVG
             * gives an object holding one, and either can be reached through the
             * attribute instead. A plot is drawn in SVG inside an HTML box, so
             * both turn up here. */
            const raw = (node.getAttribute && node.getAttribute("class")) || node.className;
            const cls = typeof raw === "string" ? raw : (raw && raw.baseVal) || "";
            if (/\bmodebar/.test(cls)) return true;
            node = node.parentNode;
        }
        return false;
    };

    const positionOf = (event) => {
        if (onToolbar(event)) return null;
        const area = plotArea();
        const axis = surface._fullLayout && surface._fullLayout.xaxis;
        if (!area || !axis || !Array.isArray(axis.range)) return null;
        const box = area.getBoundingClientRect();
        if (!box.width) return null;
        const [from, to] = axis.range;
        const perPixel = (to - from) / box.width;
        return { x: from + (event.clientX - box.left) * perPixel, perPixel };
    };

    let hovered = null;   // the handle's hover callback, if it asked for one
    let pressedAt = null;

    surface.addEventListener("mousemove", (event) => {
        if (disposed || !hovered) return;
        const at = positionOf(event);
        hovered(at && Number.isFinite(at.x) ? at : null);
    });
    surface.addEventListener("mouseleave", () => {
        if (!disposed && hovered) hovered(null);
    });

    /* A DRAG IS NOT A PICK.
     *
     * The browser fires `click` whenever a press and a release share an element,
     * however far the pointer travelled between them — so the library's own
     * drag-to-zoom and drag-to-pan each end in a click, and the chart would
     * select whatever mode happened to sit under the release. Looking at a peak
     * closely would keep changing the selection out from under the user.
     *
     * A few pixels of travel is a shaky hand, not a gesture. */
    const DRAG_SLOP_PX = 4;
    surface.addEventListener("mousedown", (event) => {
        pressedAt = { x: event.clientX, y: event.clientY };
    });

    surface.addEventListener("click", (event) => {
        const from = pressedAt;
        pressedAt = null;
        if (disposed || !clicked) return;
        if (from) {
            const travelled = Math.hypot(event.clientX - from.x, event.clientY - from.y);
            if (travelled > DRAG_SLOP_PX) return;       // a gesture, not a pick
        }
        const at = positionOf(event);
        if (at && Number.isFinite(at.x)) clicked(at);
    });

    return {
        draw(picture) {
            if (disposed) return;
            // `tracesFor` records which trace holds the selectable
            // sticks as it builds them -- with lanes, a rug and optional
            // curves the position is no longer guessable from outside.
            const traces = tracesFor(picture);
            readoutIndex = picture.note ? 1 : 0;
            lastReadout = picture.readout || "";
            Plotly.react(surface, traces, layoutFor(picture), {
                displaylogo: false,
                responsive: false,
                modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"],
            });
        },

        /** The cheap door: what the pointer changed, and nothing else (§ 5.1).
         *
         * Either half may be left out — `null` colours when only the words moved,
         * an unchanged readout when only the colours did. No rebuild, no axis. */
        recolour(states, readout) {
            if (disposed) return;
            if (states) Plotly.restyle(
                surface,
                { "marker.color": [states.map((s) => colourFor(s, palette))] },
                [stickTraceIndex],
            );

            /* The words move with the colours: both belong to the pointer, and
             * both are cheap. Only the text of one annotation changes, so this
             * is a touch of the layout rather than a rebuild of it. */
            if (readout !== undefined && readout !== lastReadout) {
                lastReadout = readout;
                Plotly.relayout(surface, { [`annotations[${readoutIndex}].text`]: readout || "" });
            }
        },

        resize() {
            if (disposed) return;
            Plotly.Plots.resize(surface);
        },

        /** One number goes up: where the click landed on the frequency axis. */
        onClick(cb) {
            clicked = cb;
        },

        /** The same number, continuously, or null over nowhere in particular. */
        onHover(cb) {
            hovered = cb;
        },

        purge() {
            if (disposed) return;
            disposed = true;
            Plotly.purge(surface);
            frame.remove();
        },
    };
}
