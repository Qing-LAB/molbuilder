/* plot-theme.js — the charts' colours, read from the tokens.
 *
 * ONE PLACE a Plotly chart asks what its paper, plot, grid, axis and ink are,
 * and which colour its n-th trace takes (`web/results.md` § 2.5): every value
 * a token of `lib/tokens.css`, read when the chart is drawn, so a chart
 * follows the page's palette instead of keeping one of its own -- the app's
 * font among them (`--font-sans`).  No second copy of any value: a page that
 * draws a chart loads the tokens.
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
            paper:   get("--bg-card"),
            grid:    get("--border-subtle"),
            axis:    get("--border-soft"),
            ink:     get("--text-secondary"),
            muted:   get("--text-muted"),
            accent:  get("--accent"),
            success: get("--success"),
            warning: get("--warning"),
            error:   get("--error"),
            font:    get("--font-sans"),
        };
        /* THE TRACE PALETTE, in order: the n-th curve of a chart takes the
         * n-th colour, wrapping. */
        t.palette = [t.accent, t.warning, t.success, t.error,
                     get("--warn-soft"), t.muted];
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

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.plotTheme = plotTheme;
    root.molbuilder.themedLayout = themedLayout;
})(typeof window !== "undefined" ? window : this);
