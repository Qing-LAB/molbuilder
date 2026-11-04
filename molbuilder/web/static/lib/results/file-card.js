/* The file card -- what a file is, and who wrote it (`web/results.md` § 3b).
 *
 * One line about the file in hand: the dropdown's, or a file single-clicked
 * in the sidebar inside the folder this panel shows.  A file molbuilder
 * writes says what it holds and who writes it; a file of a calculation
 * whose description does not read says so, in the reader's words; anything
 * else says it is not written by molbuilder (user, 2026-10-04).
 *
 * THE WORDS ARE THE CATALOGUE'S, AND THIS COMPOSES NONE.  Every file of
 * `/api/results/dir`'s answer carries `about` -- the run door's reading of
 * `runfiles.WRITTEN`, read back with the run's label -- and the picker
 * carries the folder's files in its selection event.  So the card asks for
 * nothing: it shows what the scan already holds.
 *
 * A CLICK CHANGES THE CARD AND NOTHING ELSE: the mounted viewer stays
 * (§ 2.1), and a file clicked in another folder leaves the card as it was.
 */

const NS = window.molbuilder || {};
const C = NS.constants || {};

// ``dir`` is the panel's scope; ``files`` the rows the card holds -- the
// bound folder's, or a run picked off the ladder's -- each with its whole
// ``path``; ``path`` the file in hand.  A file is found among them by its
// path, never by a name cut from it (`web/results.md` § 0.3).
const shown = { dir: "", files: [], path: "" };

const { codeWords } = NS.dom;

function render() {
    const host = document.getElementById("results-file-card");
    if (!host) return;
    host.textContent = "";
    const entry = shown.files.find((f) => f.path === shown.path);
    const about = entry && entry.about;
    if (!about) { host.hidden = true; return; }
    const line = document.createElement("p");
    line.className = "fc-line";
    const name = document.createElement("code");
    name.className = "fc-name";
    name.textContent = entry.name;
    line.appendChild(name);
    if (about.ours) {
        line.appendChild(document.createTextNode(" — "));
        line.appendChild(codeWords(about.what));
        line.appendChild(document.createTextNode(". Written by "));
        line.appendChild(codeWords(about.writer));
        line.appendChild(document.createTextNode("."));
    } else if (about.unread) {
        // A CALCULATION'S FOLDER WHOSE DESCRIPTION DOES NOT READ: whose file
        // this is waits on it, and the reader's words say why and the way
        // out (`web/results.md` § 3b).
        line.appendChild(document.createTextNode(
            " — whose file this is cannot be said here: "));
        line.appendChild(codeWords(about.unread));
    } else {
        line.appendChild(document.createTextNode(
            " — not written by molbuilder."));
    }
    host.appendChild(line);
    host.hidden = false;
}

if (C.EVENT_FILE_SELECTED) {
    document.addEventListener(C.EVENT_FILE_SELECTED, (evt) => {
        const d = (evt && evt.detail) || {};
        shown.dir = d.dir || "";
        shown.files = d.files || [];
        shown.path = d.file || "";
        render();
    });
}
if (C.EVENT_SCOPE_CHANGED) {
    document.addEventListener(C.EVENT_SCOPE_CHANGED, (evt) => {
        const dir = ((evt && evt.detail) || {}).dir || "";
        if (dir !== shown.dir) {
            shown.dir = dir;
            shown.files = [];
            shown.path = "";
            render();
        }
    });
}
/* The sidebar's single click: a preview, and the card follows it -- only
 * to a file the card holds a row for, which the scan has already answered
 * for: after a run is picked off the ladder, that run's folder's, not the
 * root's.
 *
 * SUBSCRIBED AT DOMContentLoaded, reading the namespace then: the sidebar's
 * module comes after this one on the page, so `molbuilder.projects` does not
 * exist while this module runs -- the picker mounts at the same moment for the
 * same reason. */
function followTheSidebar() {
    const projects = (window.molbuilder || {}).projects;
    if (!projects || typeof projects.onChange !== "function") return;
    projects.onChange((sel) => {
        const file = sel && sel.file;
        if (!file || !shown.files.some((f) => f.path === file)) return;
        shown.path = file;
        render();
    });
}
if (document.readyState === "complete") {
    followTheSidebar();
} else {
    document.addEventListener("DOMContentLoaded", followTheSidebar, { once: true });
}
