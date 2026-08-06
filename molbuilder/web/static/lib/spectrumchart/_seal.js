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
            hovertemplate: `%{x:.1f} ${picture.xUnit || ""}<extra></extra>`,
        });
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
        if (!box.width) return null;
        const across = (event.clientX - box.left) / box.width;
        if (across < 0 || across > 1) return null;          // outside the plot area
        const [from, to] = axis.range;
        return from + across * (to - from);
    };

    surface.addEventListener("click", (event) => {
        if (disposed || !clicked) return;
        const x = positionOf(event);
        if (x !== null && Number.isFinite(x)) clicked(x);
    });

    return {
        draw(picture) {
            if (disposed) return;
            const traces = tracesFor(picture);
            stickTraceIndex = traces.length - 1;
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

        purge() {
            if (disposed) return;
            disposed = true;
            Plotly.purge(surface);
            frame.remove();
        },
    };
}
