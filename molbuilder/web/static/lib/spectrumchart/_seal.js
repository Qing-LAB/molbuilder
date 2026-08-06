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
    };
}

/* A mark's STATE arrives from above; what that state looks like is decided
 * here and nowhere else. The layer above never names a colour. */
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

    const layoutFor = (picture) => ({
        paper_bgcolor: palette.bg,
        plot_bgcolor: palette.bg,
        font: { color: palette.ink, size: 11 },
        margin: { l: 52, r: 12, t: 12, b: 40 },
        showlegend: false,
        hovermode: "closest",
        xaxis: {
            title: { text: picture.xTitle || "" },
            gridcolor: palette.grid,
            zeroline: false,
        },
        yaxis: {
            title: { text: picture.yTitle || "" },
            gridcolor: palette.grid,
            zeroline: false,
            rangemode: "tozero",
        },
        annotations: picture.note
            ? [{
                text: picture.note,
                showarrow: false,
                xref: "paper", yref: "paper", x: 0, y: 1.04,
                xanchor: "left", font: { color: palette.pending, size: 11 },
            }]
            : [],
    });

    const tracesFor = (picture) => {
        const sticks = picture.sticks || { x: [], y: [], state: [] };
        const traces = [];
        if (picture.curve) {
            traces.push({
                type: "scatter",
                mode: "lines",
                x: picture.curve.x,
                y: picture.curve.y,
                line: { color: palette.curve, width: 1.5 },
                hoverinfo: "skip",
            });
        }
        /* A mode whose strength was never computed has no height to draw, and a
         * bar of height zero is a mode that vanished. What "not computed" LOOKS
         * like is this layer's to decide (§ 8.4), so it is drawn as a mark on the
         * axis instead of a bar of nothing. */
        const pending = [];
        const drawn = { x: [], y: [], width: [], colour: [] };
        sticks.x.forEach((x, i) => {
            if (sticks.state[i] === "pending") { pending.push(x); return; }
            drawn.x.push(x);
            drawn.y.push(sticks.y[i]);
            drawn.width.push(sticks.width[i]);
            drawn.colour.push(colourFor(sticks.state[i], palette));
        });

        traces.push({
            type: "bar",
            x: drawn.x,
            y: drawn.y,
            width: drawn.width,
            marker: { color: drawn.colour },
            hovertemplate: `%{x:.1f} ${picture.xUnit || ""}<extra></extra>`,
        });

        if (pending.length) {
            traces.push({
                type: "scatter",
                mode: "markers",
                x: pending,
                y: pending.map(() => 0),
                marker: { color: palette.pending, symbol: "x", size: 7 },
                hovertemplate: `%{x:.1f} ${picture.xUnit || ""}<extra></extra>`,
            });
        }
        return traces;
    };

    // Which trace the sticks are in depends on whether a curve is drawn, and
    // `recolour` has to reach the same one `draw` built.
    let stickTraceIndex = 0;
    let disposed = false;
    let clicked = null;   // the handle's callback

    /* WHERE A CLICK CAME FROM, IN THE UNITS OF THE PICTURE.
     *
     * The library reports a click only when the pointer is over one of its own
     * points, so nothing it offers can hear a click in the empty space beside a
     * peak — and that space is the whole purpose of the bands (§ 6.3). So the
     * click is taken from the surface itself and converted here: where the
     * pointer was, across the plot area, read against the axis range.
     *
     * This is the one place in the module that reads inside the library, and it
     * is the file whose job is to know it (§ 8.4). Nothing above sees anything
     * but a number, and which mode that number means is decided up there.
     */
    const plotArea = () => surface.querySelector && surface.querySelector(".nsewdrag");

    const positionOf = (event) => {
        const area = plotArea();
        const axis = surface._fullLayout && surface._fullLayout.xaxis;
        if (!area || !axis || !Array.isArray(axis.range)) return null;
        const box = area.getBoundingClientRect();
        if (!box.width || !box.height) return null;
        /* BOTH axes, not just the one the answer comes from. The library's own
         * toolbar sits above the plot and the axis labels below it, and both
         * share the plot's horizontal span — so a check on x alone turns
         * "zoom in" or a click on a tick label into "select the mode behind
         * that button". */
        const across = (event.clientX - box.left) / box.width;
        const down = (event.clientY - box.top) / box.height;
        if (across < 0 || across > 1 || down < 0 || down > 1) return null;
        const [from, to] = axis.range;
        return from + across * (to - from);
    };

    /* A DRAG IS NOT A PICK.
     *
     * The browser fires `click` whenever a press and a release share an element,
     * however far the pointer travelled between them — so the library's own
     * drag-to-zoom and drag-to-pan each end in a click, and the chart would
     * select whatever mode happened to sit under the release. Looking at a peak
     * closely would keep changing the selection out from under the user.
     *
     * A few pixels of travel is a shaky hand, not a gesture, so a click counts
     * as a pick only if the pointer stayed within DRAG_SLOP of where it went
     * down. */
    const DRAG_SLOP_PX = 4;
    let hovered = null;   // the handle's hover callback, if it asked for one
    let pressedAt = null;

    /* WHERE THE POINTER IS, continuously — the same conversion as a click.
     *
     * A band is invisible, so which mode a click would pick is a guess until
     * something says so. Reporting the position under the pointer lets the layer
     * above light up the mode it would choose, which is the band made visible
     * without drawing it. `null` means the pointer is somewhere that would pick
     * nothing. */
    surface.addEventListener("mousemove", (event) => {
        if (disposed || !hovered) return;
        const x = positionOf(event);
        hovered(x !== null && Number.isFinite(x) ? x : null);
    });
    surface.addEventListener("mouseleave", () => {
        if (!disposed && hovered) hovered(null);
    });

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
        const x = positionOf(event);
        if (x !== null && Number.isFinite(x)) clicked(x);
    });

    return {
        draw(picture) {
            if (disposed) return;
            const traces = tracesFor(picture);
            stickTraceIndex = picture.curve ? 1 : 0;   // after the curve, before the pending marks
            Plotly.react(surface, traces, layoutFor(picture), {
                displaylogo: false,
                responsive: false,
                modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"],
            });
        },

        /** The cheap door: colours only, no rebuild, no axis change (§ 5.1). */
        recolour(states) {
            if (disposed) return;
            Plotly.restyle(
                surface,
                { "marker.color": [states.map((s) => colourFor(s, palette))] },
                [stickTraceIndex],
            );
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
