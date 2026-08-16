/* codemirror-load.js — the ONE loader for the vendored CodeMirror 5 bundle.
 *
 * Extracted from ``lib/projects/preview.js`` on 2026-08-16, when the Task Setup
 * tab needed the same editor.  Two copies of a lazy-loader is two places for
 * the asset list to drift, and the vendor-integrity test pins that list — so
 * it has one home and both callers import it.
 *
 * WHAT IS VENDORED, and what follows from it (`static/vendor/README.md`):
 * CodeMirror **5.65.16**, MIT, served locally — the project ships browser
 * assets itself for offline use and a strict CSP, so there is no CDN path and
 * adding one is a reviewed decision, not a convenience.
 *
 * The bundle carries the core, the dialog/search/jump addons, and **eight
 * modes** (§ MODES below).  Highlighting is chosen from the FILE SUFFIX and the
 * mode file is fetched only when a file of that kind is first opened, so a page
 * that never shows Python never pays for the Python mode.
 *
 * A suffix with no mode is not a failure: ``mode: null`` is a real mode in
 * CodeMirror (plain text) and still gives line numbers, editing, undo and the
 * search addons.  That is what `.fdf`, `.xyz`, `.out` and the logs get, because
 * CodeMirror has no mode for them — asking for one it lacks yields plain text
 * anyway, with a misleading line of code left behind to explain later.
 *
 * Load order matters and is not an accident:
 *   1. CSS (parallel with the core — no execution dependency)
 *   2. the core, which must finish before any addon parses: addons register
 *      themselves on ``CodeMirror.commands`` / ``CodeMirror.defineOption`` at
 *      parse time, against a global that has to exist already
 *   3. dialog (search + jumpToLine both call its ``cm.openDialog``)
 *   4. searchcursor (search needs it)
 *   5. search + jumpToLine (both depend on the above; parallel with each other)
 */

export const CM_VENDOR_BASE = "/static/vendor/codemirror/";

/** Every file the bundle needs, in the order above.  The vendor-integrity
 *  test reads THIS list, so an asset added here is automatically required to
 *  exist on disk (`tests/test_codemirror_vendor_bundle.py`). */
export const CM_ASSETS = [
    "codemirror.min.css",
    "dialog.min.css",
    "codemirror.min.js",
    "dialog.min.js",
    "searchcursor.min.js",
    "search.min.js",
    "jump-to-line.min.js",
];

/* ---------- modes: chosen by suffix, fetched on demand ---------- */

/** Every vendored mode, and what each one needs loaded first.
 *
 * `markdown` genuinely depends on `xml` — its module head is
 * ``require("../xml/xml")`` — so opening a `.md` file without xml present left
 * the mode half-registered.  The dependency is declared here rather than
 * remembered, because the failure is silent: you get plain text and no error.
 */
export const CM_MODES = {
    javascript: { file: "javascript.min.js" },
    python:     { file: "python.min.js" },
    toml:       { file: "toml.min.js" },
    shell:      { file: "shell.min.js" },
    xml:        { file: "xml.min.js" },
    css:        { file: "css.min.js" },
    yaml:       { file: "yaml.min.js" },
    markdown:   { file: "markdown.min.js", needs: ["xml"] },
};

/** Suffix → the CodeMirror mode spec to open the file with.
 *
 * Longest suffix wins, so `.run.sh` resolves through `.sh` and a future
 * `.molwatch.log` could get its own entry without disturbing `.log`.
 *
 * JSON is the javascript mode with ``json: true`` — CodeMirror ships no
 * separate json mode, and the spec object is how you ask for the JSON dialect
 * rather than guessing at a mime string.
 *
 * A suffix absent here is deliberate, not missing: molbuilder's own formats
 * (`.fdf`, `.xyz`, `.out`, `.log`, `.molwatch.log`, `.STRUCT_OUT`) have no
 * upstream CodeMirror mode.
 */
const SUFFIX_MODE = {
    ".json":     { name: "javascript", json: true },
    ".js":       "javascript",
    ".py":       "python",
    ".toml":     "toml",
    ".sh":       "shell",
    ".bash":     "shell",
    ".sbatch":   "shell",
    ".md":       "markdown",
    ".markdown": "markdown",
    ".xml":      "xml",
    ".css":      "css",
    ".yaml":     "yaml",
    ".yml":      "yaml",
};

/**
 * The mode spec for a path, or ``null`` for plain text.
 *
 * Pure and synchronous — call it to decide, then `ensureMode` to load.
 */
export function modeForPath(path) {
    const name = String(path || "").toLowerCase();
    let best = null;
    for (const suffix of Object.keys(SUFFIX_MODE)) {
        if (name.endsWith(suffix)
            && (best === null || suffix.length > best.length)) {
            best = suffix;
        }
    }
    return best === null ? null : SUFFIX_MODE[best];
}

/** The mode's registration name, given what `modeForPath` returned. */
export function modeName(spec) {
    if (!spec) return null;
    return typeof spec === "string" ? spec : spec.name;
}

const _modePromises = new Map();

/**
 * Load the mode a spec needs, with its dependencies, once.
 *
 * Resolves to ``true`` when the mode is registered and ``false`` when the spec
 * asks for plain text — so a caller can pass the result straight through
 * without branching on null.
 */
export async function ensureMode(spec) {
    const name = modeName(spec);
    if (!name) return false;
    const entry = CM_MODES[name];
    if (!entry) return false;                 // not vendored: plain text
    if (_modePromises.has(name)) { await _modePromises.get(name); return true; }

    const load = (async () => {
        await loadCodeMirror();               // the core must exist first
        for (const dep of entry.needs || []) await ensureMode(dep);
        await injectScript(CM_VENDOR_BASE + entry.file);
    })();
    _modePromises.set(name, load);
    await load;
    return true;
}

/**
 * The whole job in one call: load the core, work out the mode from the path,
 * load that mode, and hand back the spec to pass to CodeMirror.
 *
 *     const mode = await modeFor(path);
 *     CodeMirror(host, { mode, ... });
 */
export async function modeFor(path) {
    const spec = modeForPath(path);
    await ensureMode(spec);
    return spec;                              // null == plain text
}

let _cmLoaderPromise = null;

export function injectStylesheet(href) {
    return new Promise((resolve, reject) => {
        const link = document.createElement("link");
        link.rel  = "stylesheet";
        link.href = href;
        link.onload  = () => resolve();
        link.onerror = () => reject(
            new Error("Could not load stylesheet: " + href));
        document.head.appendChild(link);
    });
}

export function injectScript(src) {
    return new Promise((resolve, reject) => {
        const s = document.createElement("script");
        s.src = src;
        s.async = false;  // preserve evaluation order vs. previous calls
        s.onload  = () => resolve();
        s.onerror = () => reject(
            new Error("Could not load script: " + src));
        document.head.appendChild(s);
    });
}

/**
 * Lazy-load the bundle.  The promise is cached, so several callers on one page
 * — the projects-sidebar preview modal and the Task Setup editor, say — share a
 * single fetch instead of racing to inject the same <script> twice.
 *
 * Returns ``window.CodeMirror``.
 */
export async function loadCodeMirror() {
    if (window.CodeMirror) return window.CodeMirror;
    if (_cmLoaderPromise)  return _cmLoaderPromise;
    _cmLoaderPromise = (async () => {
        await Promise.all([
            injectStylesheet(CM_VENDOR_BASE + "codemirror.min.css"),
            injectStylesheet(CM_VENDOR_BASE + "dialog.min.css"),
            injectScript(CM_VENDOR_BASE + "codemirror.min.js"),
        ]);
        await injectScript(CM_VENDOR_BASE + "dialog.min.js");
        await injectScript(CM_VENDOR_BASE + "searchcursor.min.js");
        await Promise.all([
            injectScript(CM_VENDOR_BASE + "search.min.js"),
            injectScript(CM_VENDOR_BASE + "jump-to-line.min.js"),
        ]);
        if (!window.CodeMirror) {
            throw new Error(
                "CodeMirror bundle loaded but the global is missing — "
                + "vendored bundle may be corrupt"
            );
        }
        return window.CodeMirror;
    })();
    return _cmLoaderPromise;
}
