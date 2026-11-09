/* projects/molview-doors.js — the doors every MolView mount hands in.
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
 *           implementation.  Every page that mounts a MolView hands it
 *           `molviewDoors` -- this door and the server's (`opts.server`).
 *
 * NEVER: parse a structure (the server's codec writes the pair, § 11.7);
 *        invent a sidecar in the browser (the save-then-reload trap,
 *        projects.md § 3); expose MolView to the sidebar's internals — the
 *        menu sees exactly `{save, saveBinary}`.
 */
"use strict";

import { chooseSavePath, confirmDestructive } from "./dialogs.js";
import { projects } from "./state.js";
import { storeZip } from "../zip-store.js";
import { download } from "./downloads.js";
// THE ONE READER of the server's answers (ui-contract.md § 12).
import { ask } from "../server.js";

/* WHERE and AS WHAT is the unified question (`chooseSavePath`,
 * projects.md § 5) — this door phrases it for an export, the export's
 * extension ``ext`` added by the question itself.  Resolves the full path,
 * or null on cancel. */
async function _askProjectPath(stem, what, ext) {
    return chooseSavePath({
        title: "Save " + what + " where?",
        nameTitle: "Save " + what + " as",
        initial: stem,
        ext: ext,
    });
}

export const molviewFiles = {
    /**
     * The TRUTH leaving the viewer (§ 11.3's Data): `payload` is
     * `model.exportFile(range)`'s answer — `{structure}`, one envelope, a
     * range's frames inside it — and the server's one generator turns it
     * into the `.xyz` + `.molstruct.json` pair, so a project save and a
     * download cannot produce different bytes.
     *
     * @returns {ok, cancelled?, error?, files?|path?}
     */
    async save(destination, stem, payload) {
        if (!payload || !payload.structure) {
            return { ok: false, error: "nothing to export" };
        }
        try {
            if (destination === "download") {
                const res = await ask("/api/structure/export", {
                    method: "POST",
                    body: { structure: payload.structure, name: stem },
                });
                if (!res.ok) return { ok: false, error: res.error };
                // BOTH files — the coordinates and the metadata that has to
                // travel with them (§ 11.3).  As ONE artifact when there are
                // two: a second programmatic download is exactly what
                // browsers silently swallow (Chrome's multiple-download
                // policy).  A store-zip named by the stem cannot lose its
                // half.
                const files = res.body.files;
                // The gate's verdicts ride the result: the server judged
                // what is leaving, and its sentence is said.
                const notices = Array.isArray(res.body.notices)
                    ? res.body.notices : null;
                if (files.length === 1) {
                    download(files[0].name,
                              new Blob([files[0].text],
                                       { type: "text/plain" }));
                    return { ok: true, files: [files[0].name],
                             notices: notices };
                }
                const enc = new TextEncoder();
                const archive = storeZip(files.map((f) => ({
                    name: f.name, bytes: enc.encode(f.text) })));
                download(stem + ".zip", archive);
                return { ok: true,
                         files: [stem + ".zip ("
                                 + files.map((f) => f.name).join(" + ")
                                 + ")"],
                         notices: notices };
            }
            if (destination === "project") {
                const path = await _askProjectPath(stem, "structure", ".xyz");
                if (!path) return { ok: false, cancelled: true };
                const body = {
                    path: path,
                    structure: payload.structure,
                };
                let res = await ask("/api/structure/save",
                                    { method: "POST", body: body });
                if (res.body && res.body.needsOverwrite) {
                    // The module's own dialog, never a native confirm() --
                    // one dialog system, and a native modal freezes any
                    // automation driving the page.
                    const sure = await confirmDestructive({
                        title: "Overwrite?",
                        body: path + " exists.  Overwriting replaces it "
                            + "and its metadata sidecar together.",
                        confirmLabel: "Overwrite",
                    });
                    if (!sure) return { ok: false, cancelled: true };
                    res = await ask("/api/structure/save", { method: "POST",
                        body: Object.assign({ overwrite: true }, body) });
                }
                if (!res.ok) {
                    return { ok: false, error: res.error };
                }
                _announce(path.slice(0, path.lastIndexOf("/")));
                return { ok: true, path: path,
                         notices: Array.isArray(res.body.notices)
                             ? res.body.notices : null };
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
                download(filename, blob);
                return { ok: true, files: [filename] };
            }
            if (destination === "project") {
                const dot = filename.lastIndexOf(".");
                const stem = dot > 0 ? filename.slice(0, dot) : filename;
                const ext = dot > 0 ? filename.slice(dot) : "";
                const picked = await _askProjectPath(stem, "image", ext);
                if (!picked) return { ok: false, cancelled: true };
                const dir = picked.slice(0, picked.lastIndexOf("/"));
                const name = picked.slice(picked.lastIndexOf("/") + 1);
                // THROUGH THE WRITE DOOR (web/projects.md § 2), which
                // announces the folder: binary bytes ride its upload route,
                // and ``autoRename`` keeps a collision from silently
                // clobbering.
                const out = await projects.writeFile(
                    dir + "/" + name, blob, { autoRename: true });
                if (!out || out.ok === false) {
                    return { ok: false,
                             error: (out && out.error) || "write failed" };
                }
                // The NAME comes from the server's answer: auto_rename may
                // have written `<stem>-2<ext>`, and reporting the name we
                // ASKED for would confirm a file that does not exist.
                const wrote = String(out.path || "");
                const finalName = wrote
                    ? wrote.slice(wrote.lastIndexOf("/") + 1) : name;
                return { ok: true, path: dir + "/" + finalName };
            }
            return { ok: false, error: "unknown destination " + destination };
        } catch (e) {
            return { ok: false, error: String((e && e.message) || e) };
        }
    },
};

/* A STRUCTURE SAVED THROUGH ITS OWN SERVER DOOR CHANGED A FOLDER: announced
 * (web/projects.md § 2) -- whoever shows it follows it, the sidebar
 * re-listing it.  Tolerated absent: the door works on a page whose
 * projects namespace has not finished mounting. */
function _announce(dir) {
    const p = window.molbuilder && window.molbuilder.projects;
    if (p && typeof p.publishFolderChanged === "function") {
        p.publishFolderChanged(dir);
    }
}

/* THE APP'S DOORS FOR A MOLVIEW, ONE BUNDLE: where its bytes leave
 * (`files`, above) and how its questions reach the server (`server`, the
 * app's one reader -- ui-contract.md § 12).  MolView reaches no module of the
 * app by name (molview.md § 4), so each page that mounts one hands them in;
 * handed as one, no page can hand one and forget the other. */
export const molviewDoors = Object.freeze({
    files: molviewFiles,
    server: Object.freeze({ ask }),
});
