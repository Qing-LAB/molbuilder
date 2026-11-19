/* projects/downloads.js — a file handed to the person, as a browser download.
 *
 * Contract: docs/web/projects.md § 5 (`projects.download`,
 *           `projects.downloadExport`, `projects.exportBar`).
 * Owns:     the browser download — bytes a page holds, or a file a server
 *           route composed and named.  Where files may go is the projects
 *           module's knowledge, so every page that hands a person a file
 *           asks here.
 * Called by: the MolView files door (`molview-doors.js`), the spectra
 *           viewer's CSV, the Results tab's one Export bar, and each chart's
 *           own Export (`lib/chart-export.js`): its picture and its data.
 *
 * NEVER: name a file the server did not name; throw — an export answers
 *        `{ok, ...}` like every public method of the module (projects.md
 *        § 5, the files layer's result).
 */
"use strict";

import { ask } from "../server.js";

/* A browser download of in-memory bytes.  Four lines of DOM, and they live
 * HERE — the projects module — because § 11.2a forbids exactly these four
 * lines inside the viewer: where files may go is one module's knowledge.
 * Every page that hands a person a file asks it (`projects.download`). */
export function download(filename, blob) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

/* The name a server route gave the file it sends, from its
 * Content-Disposition, or "" where it gave none. */
function _sentName(headers) {
    const said = headers.get("Content-Disposition") || "";
    const utf = said.match(/filename\*=UTF-8''([^;]+)/i);
    const plain = said.match(/filename="?([^";]+)"?/i);
    return utf ? decodeURIComponent(utf[1]) : plain ? plain[1] : "";
}

/* AN EXPORT THE SERVER COMPOSES, handed to the person
 * (`projects.downloadExport`): ``url`` a route answering the file itself,
 * named -- asked with GET, or POSTed ``body`` where one is given (a chart's
 * drawn data, `POST /api/export/chart`).  Resolves ``{ok: true, said}`` -- ``said`` what the server
 * said beside it (``X-Molbuilder-Said``), or "" -- once the download is
 * handed to the browser; ``{ok: false, error}`` in the words of the one
 * reader of an answer (ui-contract.md § 12) -- the route's own refusal, or
 * why it could not be read.  Never throws. */
export async function downloadExport(url, body) {
    const a = await ask(url, body === undefined ? { as: "file" }
                                                : { as: "file", method: "POST", body });
    if (!a.ok) return { ok: false, error: a.error, aborted: a.aborted };
    const name = _sentName(a.headers);
    if (!name) return { ok: false, error: url + " sent a file it did "
                                          + "not name" };
    download(name, a.body);
    return { ok: true, said: a.headers.get("X-Molbuilder-Said") || "" };
}

/* THE CALCULATION'S ONE EXPORT BAR (`projects.exportBar`, projects.md § 5;
 * model/datafiles.md § 5a): the offer for ``path`` (`GET /api/export/offer`)
 * drawn as a button for each form, in the words it is served with, and one
 * for each flag its kind declares beside its form's -- the flag's words its
 * row's ``button``, asking the one form its row names --
 * the server's sentence or refusal beside them through the one status
 * writer, every button disabled while one runs, and the kind's *How to read
 * these in Python*.  ``asks()`` -- the values the page shows that the export
 * takes (a vibration's laser and width) -- is read as a button is pressed,
 * and only the offer's numbers are sent; ``at()`` -- the selection on screen,
 * which a controller that keeps one bar across its calculation's selections
 * hands over -- is read so too, ``path`` where none is handed.  Resolves the bar; a line saying
 * why where the offer did not read -- never taken for none; or ``null``
 * where the path's calculation exports nothing.  Never throws. */
export async function exportBar(path, { asks, at } = {}) {
    const a = await ask("/api/export/offer?"
                        + new URLSearchParams({ path }).toString());
    // THE ONE ELEMENT BUILDER (`lib/dom.js`).
    const el = window.molbuilder.dom.el;
    if (!a.ok) {
        const line = el("span", "status");
        window.molbuilder.status.set(
            line, "The Export could not be offered: " + a.error, "warn");
        return line;
    }
    const offer = a.body && a.body.offer;
    if (!offer) return null;
    const bar = el("div", "fdf-actions export-bar");
    // WHOSE IT IS -- the calculation's folder -- so a page keeps one bar
    // across the selections in one calculation.
    bar.dataset.calculation = offer.calculation;
    bar.appendChild(el("span", "hint", "Export every result as it stands:"));
    const msg = el("span", "status");
    msg.setAttribute("role", "status");
    const say = (text, kind) => window.molbuilder.status.set(msg, text, kind);
    const numbers = offer.asks.filter((r) => r.type === "number")
                              .map((r) => r.name);
    let buttons = [];
    const run = async (choice) => {
        const shown = typeof asks === "function" ? (asks() || {}) : {};
        const where = typeof at === "function" ? at() : path;
        const q = Object.assign({ path: where }, choice);
        numbers.forEach((n) => { if (shown[n] != null) q[n] = String(shown[n]); });
        buttons.forEach((b) => { b.disabled = true; });
        say("Composing the export.", "muted");
        const got = await downloadExport(
            "/api/export?" + new URLSearchParams(q).toString());
        buttons.forEach((b) => { b.disabled = false; });
        say(got.ok ? got.said : got.error, got.ok ? "muted" : "warn");
    };
    // EACH FORM'S BUTTON, the buttons of the flags that go with it beside it.
    const flags = offer.asks.filter((r) => r.type === "flag");
    const choices = offer.forms.flatMap((form) => [
        [form.words, { format: form.name }],
        ...flags.filter((r) => r.form === form.name)
            .map((r) => [r.button, { format: form.name, [r.name]: "1" }]),
    ]);
    buttons = choices.map(([words, choice]) => {
        const b = el("button", null, words);
        b.type = "button";
        b.addEventListener("click", () => run(choice));
        bar.appendChild(b);
        return b;
    });
    bar.appendChild(msg);
    if (offer.doc) {
        const how = el("a", null, "How to read these in Python");
        how.href = "/documents?doc=" + offer.doc;
        how.target = "_blank";
        how.rel = "noopener";
        bar.appendChild(how);
    }
    return bar;
}
