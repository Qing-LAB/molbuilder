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

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.plotTheme = plotTheme;
    root.molbuilder.themedLayout = themedLayout;
})(typeof window !== "undefined" ? window : this);
