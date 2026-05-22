/* Source-file inspector: read-only listing for .fdf / .py / .json /
 * .out / .log files the user wants to peek at without leaving the
 * page.  Reaches the existing GET /api/files/read endpoint via the
 * registry's ctx.readFile helper -- no new wire format.
 */
(function (root) {
    "use strict";

    const inspector = {
        name:        "source",
        displayName: "Source file",
        // Source inspector intentionally does NOT claim ``.out`` --
        // the trajectory inspector takes those (SIESTA's redirected
        // stdout parses to a frame trajectory; a raw text view of a
        // multi-MB SIESTA .out is hostile UX).  Users who need to
        // peek at the raw text can use the future range-read path
        // (task #119) or open the file in their editor.
        match:       (file) => {
            const lower = file.toLowerCase();
            return lower.endsWith(".fdf")
                || lower.endsWith(".py")
                || lower.endsWith(".log")
                || lower.endsWith(".json")
                || lower.endsWith(".txt")
                || lower.endsWith(".md");
        },

        mount(host, file, ctx) {
            host.innerHTML = "";

            const card = document.createElement("section");
            card.className = "inspector-card source-card";

            const header = document.createElement("header");
            header.className = "inspector-card-header";
            const title = document.createElement("h2");
            title.className = "inspector-card-title";
            title.textContent = "Source — " + _basename(file);
            header.appendChild(title);
            const note = document.createElement("p");
            note.className = "inspector-card-note";
            note.textContent = (
                "Read-only.  Edit via your editor + the sidebar's "
                + "Save (coming) or re-generate via the Build / "
                + "Spectra tab."
            );
            header.appendChild(note);
            card.appendChild(header);

            const pre = document.createElement("pre");
            pre.className = "source-body";
            pre.textContent = "Loading…";
            card.appendChild(pre);

            host.appendChild(card);

            let disposed = false;

            // SIESTA .out and engine .log files routinely exceed
            // the server-side 1 MB default cap; bump the per-call
            // budget to the server's hard ceiling (16 MB) so the
            // source inspector shows real content instead of a 413
            // "exceeds max_bytes" error for any non-trivial run.
            // Bigger-than-16 MB files still fail loudly with the
            // size + ceiling in the message; that's a corner case.
            ctx.readFile(file, { maxBytes: 16 * 1024 * 1024 }).then((r) => {
                if (disposed) return;
                if (!r.ok) {
                    // Backend returns 413 for oversize (with
                    // {ok:false, error:...}); the user sees the
                    // error verbatim including the file size and
                    // suggested max_bytes.  No separate "truncated"
                    // path -- the read endpoint is all-or-nothing.
                    pre.textContent = "Error: " + (r.error || "unknown");
                    pre.classList.add("inspector-inline-error");
                    return;
                }
                pre.textContent = r.text || "";
            });

            return {
                dispose() {
                    disposed = true;
                    host.innerHTML = "";
                },
            };
        },
    };

    const _basename = (window.molbuilder
                      && window.molbuilder.path
                      && window.molbuilder.path.basename)
                   || ((p) => p || "");

    root.molbuilder = root.molbuilder || {};
    root.molbuilder.inspectors = root.molbuilder.inspectors || {};
    root.molbuilder.inspectors.sourceInspector = inspector;
    if (root.molbuilder.inspectors.register) {
        root.molbuilder.inspectors.register(inspector);
    }
})(typeof window !== "undefined" ? window : this);
