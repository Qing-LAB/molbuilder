/* plot-theme.js — the charts' colours, read from the tokens.
 *
 * ONE PLACE a Plotly chart asks what its grid, axis and ink are, which colour
 * its n-th trace takes and which marks a target (`web/results.md` § 2.5):
 * every value a token of `lib/tokens.css`, read when the chart is drawn, so
 * a chart follows the page's palette instead of keeping one of its own -- the
 * app's font among them (`--font-sans`).  No second copy of any value: a page
 * that draws a chart loads the tokens.
 * SpectrumChart keeps its own sealed palette (`web/spectrumchart.md` § 11).
 */
(function (root) {
    "use strict";

    function plotTheme() {
        const cs = root.getComputedStyle(root.document.documentElement);
        /* A TOKEN, OR THE PAGE IS WRONG: every page that draws a chart loads
         * lib/tokens.css first (ui-contract.md § 1), so a token that is not
         * there is said, never covered by a second copy of its value. */
        const get = (name) => {
            const v = (cs.getPropertyValue(name) || "").trim();
            if (!v) throw new Error("plot-theme: the token " + name + " is not "
                                    + "defined -- the page loads no lib/tokens.css");
            return v;
        };
        const t = {
            grid:    get("--border-subtle"),
            axis:    get("--border-soft"),
            ink:     get("--text-secondary"),
            muted:   get("--text-muted"),
            success: get("--success"),       // a target line, met or not
            page:    get("--bg-page"),       // a hover label's ground
            edge:    get("--border-strong"), // a hover label's edge
            text:    get("--text-primary"),  // a hover label's words
            font:    get("--font-sans"),
        };
        /* THE SERIES PALETTE, in order (`--series-*`, colour-vision safe):
         * the n-th curve of a chart takes the n-th colour, wrapping.  A
         * severity never colours one curve among others. */
        t.palette = ["--series-1", "--series-2", "--series-3", "--series-4",
                     "--series-5"].map(get);
        return t;
    }

    /* A layout's colours from the theme, merged under what the chart sets
     * itself (`Object.assign` order: the chart's own keys win). */
    function themedLayout(layout, theme) {
        const t = theme || plotTheme();
        const axis = (a) => Object.assign({
            gridcolor: t.grid, linecolor: t.axis, zerolinecolor: t.axis,
            color: t.ink,
        }, a || {});
        return Object.assign({}, layout, {
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor:  "rgba(0,0,0,0)",
            font: Object.assign({ family: t.font, size: 10,
                                  color: t.ink }, (layout || {}).font),
            xaxis: axis((layout || {}).xaxis),
            yaxis: axis((layout || {}).yaxis),
        });
    }

    /* AN ENERGY AXIS, for every chart of an energy: its ticks as plain
     * numbers, with the decimals their spacing needs -- Plotly's own rule --
     * never a fixed count of figures.  A run's last steps differ anywhere
     * from the first to the sixth decimal of a total of thousands to
     * millions of eV, and a fixed count labelled every tick alike (six
     * figures on the SCF chart, nine on a relaxation of 312 atoms that
     * moved 3 meV); Plotly's default would write them "-1.211221373M". */
    const ENERGY_AXIS = Object.freeze({ exponentformat: "none" });

    /* A SMALL CHART'S LEGEND, under its x-axis title -- the trajectory
     * viewer's max force and its SCF energy and residuals, the transport
     * report's SCF (`web/trajectory.md` § 2): drawn only when ``shown``
     * (more than one trace to tell apart), and the bottom margin grown to
     * hold it, so it never lies on the axis title.  ``{showlegend, legend,
     * bottom}`` -- ``bottom`` the chart's margin below. */
    function legendBelow(shown) {
        return {
            showlegend: !!shown,
            legend: { orientation: "h", x: 0.5, xanchor: "center",
                      y: -0.22, yanchor: "top", font: { size: 9 },
                      bgcolor: "rgba(0,0,0,0)" },
            bottom: shown ? 58 : 32,
        };
    }

    /* A THRESHOLD a chart is judged against -- a dashed line across it at
     * ``value``, its label at the right end above it -- as ``{shape,
     * note}``, for the layout's ``shapes`` and ``annotations``.  ``log``:
     * the y-axis is logarithmic, where Plotly places a shape by the value
     * and an annotation by its log10 -- so the label is put on its line,
     * not at 10^value (`web/trajectory.md` § 3). */
    function thresholdMark(value, text, colour, log) {
        return {
            shape: { type: "line", xref: "paper", x0: 0, x1: 1,
                     yref: "y", y0: value, y1: value,
                     line: { color: colour, width: 1.5, dash: "dash" } },
            note: { xref: "paper", x: 1, xanchor: "right", yref: "y",
                    y: log ? Math.log10(value) : value, yanchor: "bottom",
                    text: text, font: { size: 9, color: colour },
                    showarrow: false },
        };
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.plotTheme = plotTheme;
    root.molbuilder.themedLayout = themedLayout;
    root.molbuilder.energyAxis = ENERGY_AXIS;
    root.molbuilder.legendBelow = legendBelow;
    root.molbuilder.thresholdMark = thresholdMark;
})(typeof window !== "undefined" ? window : this);
