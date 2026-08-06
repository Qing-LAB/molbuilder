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
        link.onerror = () => reject(new Error("the chart stylesheet could not be loaded"));
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
        script.onerror = () => reject(new Error("the plotting library could not be loaded"));
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
    };
}

/* A mark's STATE arrives from above; what that state looks like is decided
 * here and nowhere else. The layer above never names a colour. */
const colourFor = (state, palette) => (
    state === "chosen" ? palette.chosen
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
        traces.push({
            type: "bar",
            x: sticks.x,
            y: sticks.y,
            width: sticks.width,
            marker: { color: sticks.state.map((s) => colourFor(s, palette)) },
            hovertemplate: "%{x:.1f} cm⁻¹<extra></extra>",
        });
        return traces;
    };

    // Which trace the sticks are in depends on whether a curve is drawn, and
    // `recolour` has to reach the same one `draw` built.
    let stickTraceIndex = 0;
    let disposed = false;
    let clicked = null;   // the handle's callback, kept until there is a plot
    let wired = false;

    /* The library grows its event emitter on the element only once something has
     * been plotted on it, so a caller that asks for clicks before the first draw
     * would otherwise wire nothing at all and never hear one. */
    const wireClicks = () => {
        if (wired || !clicked || typeof surface.on !== "function") return;
        wired = true;
        surface.on("plotly_click", (ev) => {
            const point = ev && ev.points && ev.points[0];
            if (!point) return;
            const x = typeof point.x === "number" ? point.x : Number(point.x);
            if (Number.isFinite(x)) clicked(x);
        });
    };

    return {
        draw(picture) {
            const traces = tracesFor(picture);
            stickTraceIndex = traces.length - 1;
            Plotly.react(surface, traces, layoutFor(picture), {
                displaylogo: false,
                responsive: false,
                modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"],
            });
            wireClicks();
        },

        /** The cheap door: colours only, no rebuild, no axis change (§ 5.1). */
        recolour(states) {
            Plotly.restyle(
                surface,
                { "marker.color": [states.map((s) => colourFor(s, palette))] },
                [stickTraceIndex],
            );
        },

        resize() {
            Plotly.Plots.resize(surface);
        },

        /** One number goes up: where the click landed on the frequency axis. */
        onClick(cb) {
            clicked = cb;
            wireClicks();
        },

        purge() {
            if (disposed) return;
            disposed = true;
            Plotly.purge(surface);
            frame.remove();
        },
    };
}
