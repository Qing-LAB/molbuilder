/* plot-box.js — a Plotly chart drawn into its box, and kept to it.
 *
 * THE ONE DOOR every Plotly chart of the Results page draws through
 * (`web/results.md` § 2.5, *Shared, not copied*); SpectrumChart keeps its own
 * sealed door to the same rules (`web/spectrumchart.md` § 11).
 *
 *   * The box states the size and the drawing fills it: the module's markup
 *     makes the element a box -- the `plot-box` class beside its own, so the
 *     box stands at its size before anything is drawn or after it is let go
 *     (`lib/plot-box.css`) -- its own class gives the height as
 *     `--plot-box-h`, and no layout carries a `width` or a `height`.  `draw`
 *     refuses an element that is not a box.
 *   * The drawing follows its BOX, not the window: a sidebar folding, a tab
 *     shown, a grid reflowing each change the box while the window sits still.
 *     So each box is watched and the chart redrawn at the box's size, and
 *     Plotly's own `responsive` -- the window's resize alone -- is off.
 *   * A box with no size -- in a hidden tab -- is not fitted: Plotly refuses a
 *     hidden plot, and the box's size comes back when it is shown, which the
 *     same watch sees.
 */
(function (root) {
    "use strict";

    const BOX = "plot-box";
    const watching = new WeakMap();      // element -> its ResizeObserver
    const live = new WeakSet();          // the boxes a chart is drawn in

    /* Plotly's resize answers with a promise, rejected for a hidden plot. */
    function _fit(el) {
        try { Promise.resolve(root.Plotly.Plots.resize(el)).catch(() => {}); }
        catch (_) { /* not drawn yet */ }
    }

    /* NO requestAnimationFrame COALESCING: rAF does not run in a tab that is
     * not rendering, so a pending flag set while the tab is hidden would
     * swallow every later resize.  Plotly debounces its own resize. */
    function _watch(el) {
        if (watching.has(el) || typeof root.ResizeObserver !== "function") return;
        // The size just drawn at: Plotly reads the computed size, which the
        // box's content-box sizing makes its content's -- the observer's too.
        const cs = root.getComputedStyle(el);
        let w = parseFloat(cs.width) || 0;
        let h = parseFloat(cs.height) || 0;
        const obs = new root.ResizeObserver((entries) => {
            const r = entries[entries.length - 1].contentRect;
            if (!r.width || !r.height) return;
            if (Math.abs(r.width - w) < 1 && Math.abs(r.height - h) < 1) return;
            w = r.width;
            h = r.height;
            _fit(el);
        });
        obs.observe(el);
        watching.set(el, obs);
    }

    /** Draw ``traces`` with ``layout`` into ``el`` and keep it to its box.
     *  ``config`` is Plotly's, less ``responsive``, which this door owns. */
    function draw(el, traces, layout, config) {
        if (!el.classList.contains(BOX)) {
            throw new Error("plotBox.draw: #" + (el.id || "?") + " is not a"
                + " plot box -- its markup carries no '" + BOX + "' class");
        }
        // A BOX WITH NO CHART IN IT holds only words standing in for one --
        // `say`'s, or a module's notice -- and they go before it is drawn.
        if (!live.has(el)) el.replaceChildren();
        const drawn = root.Plotly.react(el, traces, layout,
            Object.assign({}, config, { responsive: false }));
        live.add(el);
        _watch(el);
        return drawn;
    }

    /** A sentence in place of the chart -- why there is none: the chart let
     *  go first, so a later draw builds it afresh (Plotly's state left on
     *  an element whose drawing was overwritten updates a drawing that is
     *  no longer there), then the words, in the box at its size. */
    function say(el, text) {
        release(el);
        el.textContent = text;
    }

    /** The chart gone from ``el``: Plotly's state purged, the box let go. */
    function release(el) {
        const obs = watching.get(el);
        if (obs) { obs.disconnect(); watching.delete(el); }
        live.delete(el);
        try { root.Plotly.purge(el); } catch (_) { /* never drawn */ }
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.plotBox = { draw, say, release };
})(typeof window !== "undefined" ? window : this);
