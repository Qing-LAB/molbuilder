/* chart-export.js — a chart's own Export: its picture, and what it draws.
 *
 * Contract: docs/web/results.md § 1.2; the file, model/datafiles.md § 5b,
 *           § 9.7.
 * Owns:     the Export button in a chart's corner and what it does: the
 *           picture -- the chart as it stands, Plotly's own image of it --
 *           and the data -- every curve and threshold the chart carries a
 *           tag on, read from the chart as Plotly holds it and sent to the
 *           one route (`POST /api/export/chart`), which lays it out by the
 *           chart table.  It knows no chart: a chart's name, its path and
 *           its choices are handed in (`install`), its curves say what they
 *           are (`tag`).
 * Called by: the plot box (`lib/plot-box.js`), for a chart drawn with a name.
 * NEVER:    name a curve's quantity, its unit or its words (the chart
 *           table's); hand a file to the browser itself (`projects`).
 *
 * Exports (on window.molbuilder.chartExport):
 *   install(el, {chart, path, chosen}) -- the button in ``el``, the chart's
 *       box; ``path()`` the selection it was drawn from, ``chosen()`` what
 *       the person had chosen, each read as a button is pressed.
 */
(function (root) {
    "use strict";

    const MENU = "chart-export";
    const asked = new WeakMap();        // a box -> what its last draw handed in

    /* A value as the route takes it: a number, or a gap. */
    function _num(v) {
        return typeof v === "number" && isFinite(v) ? v : null;
    }

    /* EVERY CURVE THE CHART CARRIES A TAG ON, as Plotly holds it: its
     * values, its legend name, whether the legend hides it. */
    function curvesOf(el) {
        return (el.data || []).filter((t) => t && t.tag).map((t) => ({
            tag: t.tag, name: t.name || "", hidden: t.visible === "legendonly",
            x: Array.from(t.x || [], _num), y: Array.from(t.y || [], _num),
        }));
    }

    /* EVERY THRESHOLD LINE WITH A TAG (`thresholdMark`), its value. */
    function marksOf(el) {
        return ((el.layout && el.layout.shapes) || [])
            .filter((s) => s && s.tag)
            .map((s) => ({ tag: s.tag, value: _num(s.y0) }));
    }

    /* A PICTURE'S NAME, as its data's: the calculation's label, which its
     * Export's offer states (`GET /api/export/offer`), then the chart's --
     * `T.transmission.png`, beside `T.transmission.nc`. */
    async function _pictureName(how, fmt) {
        const a = await root.molbuilder.server.ask(
            "/api/export/offer?path=" + encodeURIComponent(how.path()));
        const label = a.ok && a.body && a.body.offer ? a.body.offer.label : "";
        if (!label) throw new Error("the chart is of no calculation that exports"
                                    + (a.ok ? "" : ": " + a.error));
        return label + "." + how.chart + "." + fmt;
    }

    /* Plotly's image, a data URL, as the file's bytes -- decoded here: the
     * page fetches nothing but its own server (`connect-src 'self'`,
     * web-api.md § 2). */
    function _blobOf(url) {
        const comma = url.indexOf(",");
        const head = url.slice(0, comma);
        const body = url.slice(comma + 1);
        const type = head.slice(5).split(";")[0];
        if (!/;base64$/.test(head)) {
            return new root.Blob([decodeURIComponent(body)], { type: type });
        }
        const raw = root.atob(body);
        const bytes = new Uint8Array(raw.length);
        for (let i = 0; i < raw.length; i += 1) bytes[i] = raw.charCodeAt(i);
        return new root.Blob([bytes], { type: type });
    }

    /* THE CHART AS IT STANDS, at the size it stands at. */
    async function _picture(el, how, fmt) {
        const url = await root.Plotly.toImage(el, {
            format: fmt, width: el.clientWidth, height: el.clientHeight });
        root.molbuilder.projects.download(await _pictureName(how, fmt),
                                          _blobOf(url));
        return { ok: true, said: "" };
    }

    /* WHAT THE CHART DRAWS, composed by the server and named by it. */
    function _data(el, how, fmt) {
        return root.molbuilder.projects.downloadExport(
            "/api/export/chart?format=" + encodeURIComponent(fmt), {
                path: how.path(), chart: how.chart,
                chosen: how.chosen ? how.chosen() : {},
                curves: curvesOf(el), marks: marksOf(el) });
    }

    function _build(el) {
        const dom = root.molbuilder.dom;
        const menu = dom.el("details", MENU);
        menu.appendChild(dom.el("summary", MENU + "-open", "Export ▾"));
        const items = dom.el("div", MENU + "-items");
        const said = dom.el("p", MENU + "-said status");
        const say = (text, kind) => root.molbuilder.status.set(said, text, kind);
        [["Picture", [["PNG", "png"], ["SVG", "svg"]], _picture],
         ["Data", [["NetCDF", "netcdf"], ["CSV", "csv"]], _data]]
            .forEach(([what, forms, act]) => {
                const row = dom.el("div", MENU + "-row");
                row.appendChild(dom.el("span", MENU + "-what", what));
                forms.forEach(([words, fmt]) => {
                    const b = dom.el("button", MENU + "-form", words);
                    b.type = "button";
                    b.addEventListener("click", async () => {
                        say("", "");
                        b.disabled = true;
                        try {
                            const got = await act(el, asked.get(el), fmt);
                            say(got.ok ? got.said : got.error,
                                got.ok ? "" : "warn");
                            if (got.ok && !got.said) menu.open = false;
                        } catch (exc) {
                            say(String(exc && exc.message || exc), "warn");
                        } finally {
                            b.disabled = false;
                        }
                    });
                    row.appendChild(b);
                });
                items.appendChild(row);
            });
        items.appendChild(said);
        menu.appendChild(items);
        return menu;
    }

    /** The Export button in ``el``'s corner, for what ``how`` names --
     *  ``{chart, path, chosen}`` -- kept to the box's last draw. */
    function install(el, how) {
        asked.set(el, how);
        if (!el.querySelector(":scope > ." + MENU)) el.appendChild(_build(el));
    }

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.chartExport = { install };
})(typeof window !== "undefined" ? window : this);
