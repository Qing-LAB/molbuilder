/* projects/downloads.js — a file handed to the person, as a browser download.
 *
 * Contract: docs/web/projects.md § 5 (`projects.download`,
 *           `projects.downloadExport`).
 * Owns:     the browser download — bytes a page holds, or a file a server
 *           route composed and named.  Where files may go is the projects
 *           module's knowledge, so every page that hands a person a file
 *           asks here.
 * Called by: the MolView files door (`molview-doors.js`), the spectra and
 *           trajectory viewers' CSVs, and every report's Export.
 *
 * NEVER: name a file the server did not name; throw — an export answers
 *        `{ok, ...}` like every public method of the module (projects.md,
 *        Principle 6).
 */
"use strict";

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
function _sentName(response) {
    const said = response.headers.get("Content-Disposition") || "";
    const utf = said.match(/filename\*=UTF-8''([^;]+)/i);
    const plain = said.match(/filename="?([^";]+)"?/i);
    return utf ? decodeURIComponent(utf[1]) : plain ? plain[1] : "";
}

/* AN EXPORT THE SERVER COMPOSES, handed to the person
 * (`projects.downloadExport`): ``url`` a GET route answering the file
 * itself, named.  Resolves ``{ok: true, said}`` -- ``said`` what the server
 * said beside it (``X-Molbuilder-Said``), or "" -- once the download is
 * handed to the browser; ``{ok: false, error}`` with the route's own refusal,
 * the HTTP status where it said none, or why the request failed.  Never
 * throws. */
export async function downloadExport(url) {
    try {
        const r = await fetch(url, { cache: "no-store" });
        if (!r.ok) {
            let error = "HTTP " + r.status;
            try { error = (await r.json()).error || error; }
            catch (_) { /* the status is the answer */ }
            return { ok: false, error };
        }
        const name = _sentName(r);
        if (!name) return { ok: false, error: url + " sent a file it did "
                                              + "not name" };
        download(name, await r.blob());
        return { ok: true, said: r.headers.get("X-Molbuilder-Said") || "" };
    } catch (e) {
        return { ok: false, error: String((e && e.message) || e) };
    }
}
