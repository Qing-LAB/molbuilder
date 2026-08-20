/* projects/molview-doors.js — the `files` door every MolView mount hands in.
 *
 * Contract: docs/web/projects.md § 5 (the MolView files door) and
 *           docs/web/molview.md § 11.3–§ 11.4 (what an export is; MolView
 *           names a destination and hands over bytes, never touching a file
 *           route itself).
 * Owns:     turning a viewer's export into files — the `.xyz` +
 *           `.molstruct.json` pair for the truth, one binary for a picture —
 *           at either destination: a browser download, or a dialog-chosen
 *           place in the project tree.
 * Called by: MolView's Export menu, through `opts.files` at mount.  ONE
 *           implementation; until 2026-08-19 no production page passed a
 *           door at all, so every Export row was a silent no-op.
 *
 * NEVER: parse a structure (the server's codec writes the pair, § 11.7);
 *        invent a sidecar in the browser (the save-then-reload trap,
 *        projects.md § 3); expose MolView to the sidebar's internals — the
 *        menu sees exactly `{save, saveBinary}`.
 */
"use strict";

import { chooseDestinationDir, chooseName, confirmDestructive }
    from "./dialogs.js";
import { apiUpload } from "./api.js";

async function _postJSON(url, body) {
    const r = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    let payload = null;
    try { payload = await r.json(); } catch (_) { /* empty body */ }
    return { http: r.status, body: payload };
}

/* A browser download of in-memory bytes.  Four lines of DOM, and they live
 * HERE — the projects module — because § 11.2a forbids exactly these four
 * lines inside the viewer: where files may go is one module's knowledge. */
function _download(filename, blob) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

/* Ask the user WHERE and AS WHAT — § 11.3: a project save is named by the
 * user, in a dialog.  Returns "<dir>/<name>" or null on cancel. */
async function _askProjectPath(stem, what) {
    const dir = await chooseDestinationDir({
        title: "Save " + what + " where?",
    });
    if (!dir) return null;
    const name = await chooseName({
        title: "Save " + what + " as",
        label: "Name",
        initial: stem,
        hint: "The extension is added for you.",
    });
    if (!name) return null;
    return String(dir).replace(/\/+$/, "") + "/" + name;
}

export const molviewFiles = {
    /**
     * The TRUTH leaving the viewer (§ 11.3's Data): `payload` is
     * `model.exportFile(range)`'s answer — `{structure, frames?}` — and the
     * server's one generator turns it into the `.xyz` + `.molstruct.json`
     * pair, so a project save and a download cannot produce different bytes.
     *
     * @returns {ok, cancelled?, error?, files?|path?}
     */
    async save(destination, stem, payload) {
        if (!payload || !payload.structure) {
            return { ok: false, error: "nothing to export" };
        }
        try {
            if (destination === "download") {
                const res = await _postJSON("/api/structure/export", {
                    structure: payload.structure,
                    frames: payload.frames || undefined,
                    name: stem,
                });
                if (!res.body || !res.body.ok || !Array.isArray(res.body.files)) {
                    return { ok: false, error: (res.body && res.body.error)
                        || ("HTTP " + res.http) };
                }
                // BOTH files — the coordinates and the metadata that has to
                // travel with them.  One without the other is a structure
                // whose labels were quietly dropped (§ 11.3).
                for (const f of res.body.files) {
                    _download(f.name,
                              new Blob([f.text], { type: "text/plain" }));
                }
                return { ok: true, files: res.body.files.map((f) => f.name) };
            }
            if (destination === "project") {
                const path = await _askProjectPath(stem, "structure");
                if (!path) return { ok: false, cancelled: true };
                const body = {
                    path: path + ".xyz",
                    structure: payload.structure,
                    frames: payload.frames || undefined,
                };
                let res = await _postJSON("/api/structure/save", body);
                if (res.body && res.body.needsOverwrite) {
                    // The module's own dialog, never a native confirm() --
                    // one dialog system, and a native modal freezes any
                    // automation driving the page.
                    const sure = await confirmDestructive({
                        title: "Overwrite?",
                        body: path + ".xyz exists.  Overwriting replaces it "
                            + "and its metadata sidecar together.",
                        confirmLabel: "Overwrite",
                    });
                    if (!sure) return { ok: false, cancelled: true };
                    res = await _postJSON("/api/structure/save",
                                          Object.assign({ overwrite: true },
                                                        body));
                }
                if (!res.body || !res.body.ok) {
                    return { ok: false, error: (res.body && res.body.error)
                        || ("HTTP " + res.http) };
                }
                _refreshSidebar();
                return { ok: true, path: path + ".xyz" };
            }
            return { ok: false, error: "unknown destination " + destination };
        } catch (e) {
            return { ok: false, error: String((e && e.message) || e) };
        }
    },

    /**
     * A picture or a movie (§ 11.3's Image): already bytes, already named —
     * the door only decides where they land.
     */
    async saveBinary(destination, filename, blob) {
        if (!blob) return { ok: false, error: "nothing to export" };
        try {
            if (destination === "download") {
                _download(filename, blob);
                return { ok: true, files: [filename] };
            }
            if (destination === "project") {
                const dot = filename.lastIndexOf(".");
                const stem = dot > 0 ? filename.slice(0, dot) : filename;
                const ext = dot > 0 ? filename.slice(dot) : "";
                const picked = await _askProjectPath(stem, "image");
                if (!picked) return { ok: false, cancelled: true };
                const dir = picked.slice(0, picked.lastIndexOf("/"));
                const name = picked.slice(picked.lastIndexOf("/") + 1) + ext;
                // Binary bytes ride the upload route (apiWrite is JSON text);
                // auto_rename keeps a collision from silently clobbering.
                const out = await apiUpload(dir, blob,
                                            { filename: name,
                                              auto_rename: true });
                if (!out || out.ok === false) {
                    return { ok: false,
                             error: (out && out.error) || "write failed" };
                }
                _refreshSidebar();
                return { ok: true, path: dir + "/" + name };
            }
            return { ok: false, error: "unknown destination " + destination };
        } catch (e) {
            return { ok: false, error: String((e && e.message) || e) };
        }
    },
};

/* The sidebar lists what is on disk; a save just changed that.  Reached
 * through the namespace's own refresh — and tolerated absent, because the
 * door must work on a page whose sidebar has not finished mounting. */
function _refreshSidebar() {
    try {
        const p = window.molbuilder && window.molbuilder.projects;
        if (p && typeof p.refresh === "function") p.refresh();
    } catch (_) { /* the save stands either way */ }
}
