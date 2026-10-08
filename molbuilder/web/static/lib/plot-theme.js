/* plot-theme.js — the charts' colours, read from the tokens.
 *
 * ONE PLACE a Plotly chart asks what its paper, plot, grid, axis and ink are,
 * and which colour its n-th trace takes (`web/results.md` § 2.5): every value
 * a token of `lib/tokens.css`, read when the chart is drawn, so a chart
 * follows the page's palette instead of keeping one of its own.  The literal
 * after each name is what a page that has not loaded the tokens draws with.
 * SpectrumChart keeps its own sealed palette (`web/spectrumchart.md` § 11).
 */
(function (root) {
    "use strict";

    function plotTheme() {
        const cs = root.getComputedStyle(root.document.documentElement);
        const get = (name, fb) => (cs.getPropertyValue(name) || "").trim() || fb;
        const t = {
            paper:   get("--bg-card",        "#1b1f27"),
            grid:    get("--border-subtle",  "#2a2f3a"),
            axis:    get("--border-soft",    "#3a4050"),
            ink:     get("--text-secondary", "#c3c8d2"),
            muted:   get("--text-muted",     "#959ba7"),
            accent:  get("--accent",         "#6ba6ff"),
            success: get("--success",        "#4ade80"),
            warning: get("--warning",        "#f5b942"),
            error:   get("--error",          "#f87171"),
        };
        /* THE TRACE PALETTE, in order: the n-th curve of a chart takes the
         * n-th colour, wrapping. */
        t.palette = [t.accent, t.warning, t.success, t.error,
                     get("--warn-soft", "#d8a64b"), t.muted];
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
            font: Object.assign({ family: "system-ui, sans-serif", size: 10,
                                  color: t.ink }, (layout || {}).font),
            xaxis: axis((layout || {}).xaxis),
            yaxis: axis((layout || {}).yaxis),
        });
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.plotTheme = plotTheme;
    root.molbuilder.themedLayout = themedLayout;
})(typeof window !== "undefined" ? window : this);
