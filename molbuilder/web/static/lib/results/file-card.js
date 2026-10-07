/* The file card -- what a file is, and who wrote it (`web/results.md` § 3b).
 *
 * One line about the file in hand: the dropdown's, or a file single-clicked
 * in the sidebar inside the folder this panel shows.  A file molbuilder
 * writes says what it holds and who writes it; anything else says it is not
 * written by molbuilder (user, 2026-10-04).
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

const shown = { dir: "", files: [], name: "" };

function baseName(path) {
    return String(path || "").replace(/\/+$/, "").split("/").pop();
}

function dirName(path) {
    const s = String(path || "").replace(/\/+$/, "");
    const i = s.lastIndexOf("/");
    return i >= 0 ? s.slice(0, i) : "";
}

/* The catalogue's text keeps a command in backticks -- `jobset init` --
 * so it reads as one; shown as code, not as a backtick. */
function words(text) {
    const out = document.createDocumentFragment();
    String(text || "").split("`").forEach((part, i) => {
        if (!part) return;
        if (i % 2) {
            const c = document.createElement("code");
            c.textContent = part;
            out.appendChild(c);
        } else {
            out.appendChild(document.createTextNode(part));
        }
    });
    return out;
}

function render() {
    const host = document.getElementById("results-file-card");
    if (!host) return;
    host.textContent = "";
    const entry = shown.files.find((f) => f.name === shown.name);
    const about = entry && entry.about;
    if (!about) { host.hidden = true; return; }
    const line = document.createElement("p");
    line.className = "fc-line";
    const name = document.createElement("code");
    name.className = "fc-name";
    name.textContent = shown.name;
    line.appendChild(name);
    if (about.ours) {
        line.appendChild(document.createTextNode(" — "));
        line.appendChild(words(about.what));
        line.appendChild(document.createTextNode(". Written by "));
        line.appendChild(words(about.writer));
        line.appendChild(document.createTextNode("."));
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
        shown.name = baseName(d.file);
        render();
    });
}
if (C.EVENT_SCOPE_CHANGED) {
    document.addEventListener(C.EVENT_SCOPE_CHANGED, (evt) => {
        const dir = ((evt && evt.detail) || {}).dir || "";
        if (dir !== shown.dir) {
            shown.dir = dir;
            shown.files = [];
            shown.name = "";
            render();
        }
    });
}
/* The sidebar's single click: a preview, and the card follows it -- only
 * inside the folder the panel shows, whose files the scan has already
 * answered for.
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
        if (!file || !shown.dir) return;
        if (dirName(file).replace(/\/+$/, "") !== shown.dir.replace(/\/+$/, "")) return;
        shown.name = baseName(file);
        render();
    });
}
if (document.readyState === "complete") {
    followTheSidebar();
} else {
    document.addEventListener("DOMContentLoaded", followTheSidebar, { once: true });
}
