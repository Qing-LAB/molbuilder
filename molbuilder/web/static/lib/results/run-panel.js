/* The Run panel -- what ran, with what, and how it went
 * (`web/results.md` § 3a).
 *
 * A viewer shows a FILE; this shows the RUN the folder's files came from:
 * the record `/api/results/dir` composes for a run folder (`model/parse.md`
 * § 5d), which the picker carries in its selection event.  It is not a
 * presenter -- no file picks it -- and `results/viewer.js` does not know it
 * (§ 1: the controller picks, disposes and mounts).
 *
 * GENERIC, SO A NEW FACT IS A LABEL.  The panel knows no engine and no kind
 * of calculation: it walks the record's parts and renders each field through
 * FIELDS, one table of labels and formatters keyed by the field's name.  A
 * field the record leaves out leaves no row (§ 5d.1a: not stated, so not
 * shown); a field the table has no label for shows under its own name, so
 * nothing the record states is hidden, and its label is the whole change.
 *
 * RE-READ WITH THE DIRECTORY, never on a timer: the picker's scan is the one
 * source (§ 2.1).  Hidden where there is no record -- a container, a folder
 * with no run in it -- and from the moment the panel is bound to another
 * folder until that folder's scan lands: a scan that fails announces
 * nothing, and the previous folder's record must not stand under the new
 * one.
 */

const NS = window.molbuilder || {};
const C = NS.constants || {};

/* ---- formatters ------------------------------------------------------ */

const num = (v) => (v === null || v === undefined || v === "" ? NaN : Number(v));

/** A length of time, as every page says one (`lib/time-ago.js`,
 *  `ui-contract.md` § 4). */
function duration(v) {
    return NS.timeSpan(v) || String(v);
}

function seconds(v) {
    const n = num(v);
    if (!isFinite(n)) return String(v);
    return (n >= 100 ? String(Math.round(n)) : String(+n.toPrecision(3))) + " s";
}

function gb(v) {
    const n = num(v);
    return isFinite(n) ? `${+n.toPrecision(3)} GB` : String(v);
}

function pct(v) {
    const n = num(v);
    return isFinite(n) ? `${Math.round(n * 10) / 10} %` : String(v);
}

/** An ISO time with a zone, stated in UTC as it was written. */
function utc(v) {
    const s = String(v);
    const z = /(Z|[+-]00:?00)$/.test(s);
    return s.replace("T", " ").replace(/(Z|[+-]00:?00)$/, "") + (z ? " UTC" : "");
}

/** The node's own clock, which the output writes with no zone. */
function local(v) {
    return String(v).replace("T", " ");
}

function yesno(v) {
    return v === true ? "yes" : v === false ? "no" : String(v);
}

/** An engine's printed logical: SIESTA's `T` / `F`. */
function logical(v) {
    const s = String(v).trim().toLowerCase().replace(/^\.|\.$/g, "");
    if (["t", "true", "yes"].includes(s)) return "yes";
    if (["f", "false", "no"].includes(s)) return "no";
    return String(v);
}

function words(v) {
    if (v === null || v === undefined) return "";
    if (Array.isArray(v)) return v.map(words).join(", ");
    if (typeof v === "object") {
        return Object.keys(v).map((k) => `${k} ${words(v[k])}`).join("; ");
    }
    return String(v);
}

function build(v) {
    if (!v || typeof v !== "object") return words(v);
    const on = [], off = [], said = [];
    Object.keys(v).forEach((k) => {
        if (v[k] === true) on.push(k);
        else if (v[k] === false) off.push(k);
        else said.push(words(v[k]));
    });
    return [said.join(" · "),
            on.length ? "with " + on.join(", ") : "",
            off.length ? "without " + off.join(", ") : ""]
        .filter(Boolean).join("; ");
}

function machine(v) {
    if (!v || typeof v !== "object") return words(v);
    return [v.node,
            v.cores !== undefined ? `${v.cores} cores` : "",
            v.mem_gb !== undefined ? gb(v.mem_gb) : "",
            v.gpu].filter(Boolean).join(" · ");
}

function gathered(v) {
    if (!Array.isArray(v)) return words(v);
    return v.map((g) => (g && typeof g === "object"
                         ? `${g.file} from ${g.from}` : String(g))).join("\n");
}

function force(v) {
    const n = num(v);
    return isFinite(n) ? `${+n.toPrecision(4)} eV/Å` : String(v);
}

/** The stress in the (x, y) plane from SIESTA's Voigt order
 *  [xx, yy, zz, yz, xz, xy], as it printed them, in kbar. */
function inPlane(voigt) {
    return `xx ${voigt[0]} · yy ${voigt[1]} · xy ${voigt[5]} kbar`;
}

const MONO = "mono";

/* ---- the one table ---------------------------------------------------- *
 * Keyed by the field's NAME, in the order a person asks: its position here
 * is the row's position.  `mono` marks a value that is a path, a command or
 * a hash.  A field `computation` groups by (engine, solver, ...) is a
 * heading.  `PHASED` names the fields a phase's name is appended to. */
const FIELDS = {
    // computation's groups
    engine:   { label: "Engine" },
    solver:   { label: "Solver" },
    host:     { label: "Host and environment" },
    launch:   { label: "Launch" },
    time:     { label: "Time" },
    memory:   { label: "Memory and use" },
    exit:     { label: "Exit" },
    // engine
    program:  { label: "Program" },
    version:  { label: "Version" },
    build:    { label: "Build", fmt: build },
    binary:   { label: "Binary", style: MONO },
    // solver
    algorithm:       { label: "Algorithm" },
    elpa_gpu:        { label: "ELPA on the GPU", fmt: logical },
    diag_blocksize:  { label: "Block size" },
    distribution:    { label: "Process grid" },
    parallel_over_k: { label: "Parallel over k-points", fmt: logical },
    // host
    hostname:              { label: "Host" },
    user:                  { label: "User" },
    machine:               { label: "Machine", fmt: machine },
    node_phys_cores:       { label: "Physical cores" },
    node_sockets:          { label: "Sockets" },
    node_cores_per_socket: { label: "Cores per socket" },
    conda_env:             { label: "Environment" },
    python:                { label: "Python", style: MONO },  // a version, or an interpreter
    cwd:                   { label: "Working directory", style: MONO },
    // launch
    mode:           { label: "Launched" },
    // A NULL IS AN ANSWER here (`job-contracts.md` § 6.1): a run here has
    // no job id and no queue, and a run that continued from nothing started
    // from the structure.
    job_id:         { label: "Job id", none: "none -- run here" },
    placed_on:      { label: "Placed on", none: "no queue -- run here" },
    // A RUN A SEQUENCE OF GROUPS' JOB SENT names that sequence -- its job
    // the sequence's (execution/job-system.md § 6.0); null for a run its
    // launch sent itself.
    sequence:       { label: "Sent by the sequence", style: MONO,
                      none: "none -- its launch sent it" },
    ranks_asked:    { label: "Ranks asked" },
    ranks:          { label: "Ranks the engine ran on" },
    threads:        { label: "Threads per rank" },
    threads_engine: { label: "Threads the engine used" },
    command:        { label: "Command", fmt: (v) => (Array.isArray(v) ? v.join(" ") : String(v)), style: MONO },
    launched_at:    { label: "Launched at", fmt: utc },
    continued_from: { label: "Continued from", style: MONO,
                      none: "nothing -- it started from the structure" },
    // A POINT'S START: the point it took its files from, stated only where
    // its marker is (execution/runs-of-points.md § 1.2 *start*).
    started_from:   { label: "Started from", style: MONO },
    // time
    run_start_local:  { label: "Started (the node's clock)", fmt: local },
    run_end_local:    { label: "Ended (the node's clock)", fmt: local },
    engine_elapsed_s: { label: "Engine wall time", fmt: duration },
    s_per_iter:       { label: "Seconds per SCF iteration", fmt: seconds },
    iters_measured:   { label: "SCF iterations timed" },
    rows:             { label: "SCF iterations logged" },
    // TBtrans's loop in its one sentence (a transmission), and when its
    // estimate was made while one stands (model/parse.md § 5d.1b, § 5d.5)
    progress:         { label: "TBtrans's loop" },
    estimated_at:     { label: "Estimated at", fmt: utc },
    // memory
    mem_peak_gb:     { label: "Peak memory", fmt: gb },
    mem_limit_gb:    { label: "Memory limit", fmt: gb },
    mem_peak_from:   { label: "Peak read from" },
    mem_basis:       { label: "Memory measured on" },
    cpu_mean_pct:    { label: "Mean CPU use", fmt: pct },
    gpu_sm_mean_pct: { label: "Mean GPU use", fmt: pct },
    util_basis:      { label: "Use measured from" },
    // exit
    code: { label: "Exit code" },
    at:   { label: "Exited at" },
    said: { label: "What the exit record says" },
    // deck
    path:          { label: "File", style: MONO },
    sha256:        { label: "SHA-256", style: MONO },
    current:       { label: "Still the stage's deck",
                     fmt: (v) => (v === false
                                  ? "no — the stage's deck has changed since this ran"
                                  : yesno(v)) },
    gathered_from: { label: "Gathered from", fmt: gathered },
};

/** Fields stated per phase as `<name>_<phase>` (`time.s_per_iter_negf`). */
const PHASED = ["s_per_iter", "iters_measured", "rows"];

const RANK = Object.keys(FIELDS);

/** `{label, fmt, style, rank}` for a field's name -- a phase's field under
 *  its base's label, an unknown field under its own name. */
function describe(name) {
    if (FIELDS[name]) return { ...FIELDS[name], rank: RANK.indexOf(name) };
    for (const base of PHASED) {
        if (name.startsWith(base + "_") && FIELDS[base]) {
            const phase = name.slice(base.length + 1);
            return { ...FIELDS[base], label: `${FIELDS[base].label} (${phase})`,
                     rank: RANK.indexOf(base) + 0.5 };
        }
    }
    return { label: name, rank: RANK.length };
}

function ordered(part) {
    return Object.keys(part).sort((a, b) => {
        const ra = describe(a).rank, rb = describe(b).rank;
        return ra !== rb ? ra - rb : a.localeCompare(b);
    });
}

/* ---- DOM ---------------------------------------------------------------- */

/* The one element builder the Results tab shares, `lib/dom.js` -- a classic
 * script, so it has run before this module does. */
function el(tag, cls, text) {
    return NS.dom.el(tag, cls, text);
}

/* THE ONE STATE CHIP (`lib/state-chip.js`). */
function chip(state) {
    return NS.stateChip(state);
}

/** One field as a row: its label, its value through its formatter. */
function row(name, value) {
    const d = describe(name);
    const r = el("div", "rp-row");
    r.appendChild(el("span", "rp-key", d.label));
    // A FIELD WHOSE NULL IS AN ANSWER says it (`none`); any other null is
    // an empty cell.
    const text = (value === null && d.none) ? d.none
        : (d.fmt ? d.fmt(value) : words(value));
    r.appendChild(el("span", "rp-val" + (d.style === MONO ? " rp-mono" : ""), text));
    return r;
}

/** A part of the record as rows -- a nested part with no formatter of its
 *  own as a group under its heading. */
function rows(part) {
    const box = el("div", "rp-rows");
    ordered(part).forEach((name) => {
        const v = part[name];
        if (v && typeof v === "object" && !Array.isArray(v) && !describe(name).fmt) {
            const g = el("div", "rp-group");
            g.appendChild(el("h4", "rp-group-title", describe(name).label));
            g.appendChild(rows(v));
            box.appendChild(g);
        } else {
            box.appendChild(row(name, v));
        }
    });
    return box;
}

function section(title) {
    const s = el("section", "rp-section");
    s.appendChild(el("h3", "rp-section-title", title));
    return s;
}

/* ---- the closed line ----------------------------------------------------- */

function phaseWords(phase, ok) {
    return ok === true ? `${phase} SCF converged`
         : ok === false ? `${phase} SCF did not converge`
         : `${phase} SCF: no verdict yet`;
}

function summary(rec) {
    const comp = rec.computation || {};
    const eng = comp.engine || {}, launch = comp.launch || {}, time = comp.time || {};
    const verdict = rec.verdict || {};
    const parts = [];
    const name = [eng.program, eng.version].filter(Boolean).join(" ");
    if (name) parts.push(name);
    const ranks = launch.ranks !== undefined ? launch.ranks : launch.ranks_asked;
    const threads = launch.threads_engine !== undefined ? launch.threads_engine : launch.threads;
    if (ranks !== undefined) {
        parts.push(`${ranks} rank${Number(ranks) === 1 ? "" : "s"}`
                   + (Number(threads) > 1 ? ` × ${threads} threads` : ""));
    } else if (threads !== undefined) {
        parts.push(`${threads} thread${Number(threads) === 1 ? "" : "s"}`);
    }
    if (time.engine_elapsed_s !== undefined) parts.push(duration(time.engine_elapsed_s));
    if (verdict.ended) {
        parts.push(rec.run !== undefined ? `try ${rec.run} ${verdict.ended}` : verdict.ended);
    }
    Object.keys(verdict.converged || {}).forEach((p) => {
        parts.push(phaseWords(p, verdict.converged[p]));
    });
    // "No findings" only where a row was compared: a run that has not read
    // its deck yet has asked values and nothing to hold them against.
    const findings = (verdict.findings || []).length;
    const compared = ((rec.setup || {}).rows || []).some((r) => "differs" in r);
    if (findings) parts.push(`${findings} finding${findings === 1 ? "" : "s"}`);
    else if (compared) parts.push("no findings");
    return parts;
}

/* ---- the four sections -------------------------------------------------- */

function verdictSection(rec) {
    const v = rec.verdict || {};
    const s = section("Verdict");
    const head = el("p", "rp-verdict");
    if (v.state) head.appendChild(chip(v.state));
    if (v.detail && v.detail !== v.state) head.appendChild(el("span", "rp-detail", v.detail));
    s.appendChild(head);
    const box = el("div", "rp-rows");
    const line = (label, text, cls) => {
        const r = el("div", "rp-row");
        r.appendChild(el("span", "rp-key", label));
        r.appendChild(el("span", "rp-val" + (cls ? " " + cls : ""), text));
        box.appendChild(r);
    };
    if (v.ended) line("Latest try", rec.run !== undefined ? `try ${rec.run}: ${v.ended}` : v.ended);
    Object.keys(v.converged || {}).forEach((p) => line("Converged", phaseWords(p, v.converged[p])));
    // THE LAST STEP'S FORCES AND STRESS, or why a TranSIESTA device has
    // none (`model/parse.md` § 5d.6).
    if (v.forces_not_computed) line("Forces", v.forces_not_computed);
    if (v.max_force_free_ev_ang !== undefined) {
        line("Max force", [`${force(v.max_force_free_ev_ang)} on the free atoms`,
                           v.max_force_held_ev_ang !== undefined
                               ? `${force(v.max_force_held_ev_ang)} on the held atoms` : ""]
            .filter(Boolean).join("; "));
    }
    if (Array.isArray(v.stress_voigt_kbar)) line("Stress in the (x, y) plane", inPlane(v.stress_voigt_kbar));
    (v.findings || []).forEach((f) => line("Finding", f.text || words(f), "rp-finding"));
    (rec.earlier || []).forEach((e) => line(
        "Earlier try",
        `try ${e.run}: ` + (e.ended || "its output states no ending")));
    if (box.childNodes.length) s.appendChild(box);
    return s;
}

/** One cell of the setup table: a value, a block's rows, one answer per
 *  keyword, or every reading of a key read to several values. */
function cell(v) {
    if (v === null || v === undefined) return "";
    if (Array.isArray(v)) {
        return v.map((r) => (Array.isArray(r) ? r.join(" ") : String(cell(r)).trim())).join("\n");
    }
    if (typeof v === "object") {
        if (Array.isArray(v.readings)) return v.readings.map(cell).join("\n");
        if ("value" in v) return String(v.value);
        return Object.keys(v).map((k) => `${k} ${cell(v[k])}`).join("\n");
    }
    return String(v);
}

function setupSection(rec) {
    const st = rec.setup || {};
    const s = section("Setup");
    if (st.rows) {
        const t = el("table", "rp-table rp-setup");
        const hr = el("tr");
        ["Parameter", "Default", "Asked", "Used"].forEach((h) => hr.appendChild(el("th", null, h)));
        t.appendChild(el("thead")).appendChild(hr);
        const tb = t.appendChild(el("tbody"));
        st.rows.forEach((r) => {
            const tr = el("tr", r.differs ? "is-differs" : null);
            const p = el("td", "rp-param");
            p.appendChild(el("span", "rp-item", (r.items || [r.item]).join(" + ")));
            if (r.differs) p.appendChild(el("span", "rp-tag", "differs"));
            if (r.keys && r.keys.length) p.appendChild(el("span", "rp-keys rp-mono", r.keys.join(" ")));
            tr.appendChild(p);
            tr.appendChild(el("td", "rp-mono", cell(r.default)));
            // AN ITEM THE DECK DOES NOT SET reads "engine default" -- which
            // only a deck that was read can say.  With none (an output read
            // alone), what was asked is not stated, and the cell is empty.
            tr.appendChild("asked" in r ? el("td", "rp-mono", cell(r.asked))
                         : rec.deck ? el("td", "rp-default", "engine default")
                         : el("td", null, ""));
            const used = el("td", "rp-mono", cell(r.used));
            if (r.echo !== undefined) used.appendChild(el("span", "rp-echo", "engine says: " + cell(r.echo)));
            tr.appendChild(used);
            tb.appendChild(tr);
        });
        s.appendChild(t);
    }
    if (st.engine_only && st.engine_only.length) {
        const d = el("details", "rp-fold");
        d.appendChild(el("summary", null,
            `Keys the engine read that no parameter above sets (${st.engine_only.length})`));
        const t = el("table", "rp-table");
        const hr = el("tr");
        ["Key", "Read as", "In the deck"].forEach((h) => hr.appendChild(el("th", null, h)));
        t.appendChild(el("thead")).appendChild(hr);
        const tb = t.appendChild(el("tbody"));
        st.engine_only.forEach((e) => {
            const tr = el("tr");
            tr.appendChild(el("td", "rp-mono", e.key));
            tr.appendChild(el("td", "rp-mono", cell(e.readings ? { readings: e.readings } : e.value)));
            tr.appendChild(el("td", null, yesno(e.in_deck)));
            tb.appendChild(tr);
        });
        d.appendChild(t);
        s.appendChild(d);
    }
    if (st.pseudopotentials && st.pseudopotentials.length) {
        s.appendChild(el("h4", "rp-group-title", "Pseudopotentials"));
        const t = el("table", "rp-table");
        const hr = el("tr");
        ["Species", "File", "Functional", "Relativistic", "Generator", "Identity"]
            .forEach((h) => hr.appendChild(el("th", null, h)));
        t.appendChild(el("thead")).appendChild(hr);
        const tb = t.appendChild(el("tbody"));
        st.pseudopotentials.forEach((p) => {
            const tr = el("tr");
            tr.appendChild(el("td", null, p.species));
            tr.appendChild(el("td", "rp-mono", p.file));
            tr.appendChild(el("td", null, [p.xc_family, p.xc_authors].filter(Boolean).join(" ")));
            tr.appendChild(el("td", null, p.relativistic || ""));
            tr.appendChild(el("td", null, p.generator || ""));
            tr.appendChild(el("td", "rp-mono rp-small",
                [p.uuid ? "uuid " + p.uuid : "", p.sha256 ? "sha256 " + p.sha256 : ""]
                    .filter(Boolean).join("\n")));
            tb.appendChild(tr);
        });
        s.appendChild(t);
    }
    return s;
}

function computationSection(rec) {
    const s = section("Computation");
    s.appendChild(rows(rec.computation));
    return s;
}

function deckSection(rec, recordDir) {
    const s = section("Deck");
    s.appendChild(rows(rec.deck));
    const proj = NS.projects;
    if (rec.deck.path && proj && typeof proj.showPreview === "function") {
        const full = rec.deck.path.startsWith("/") ? rec.deck.path
                   : String(recordDir || "").replace(/\/$/, "") + "/" + rec.deck.path;
        const b = el("button", "rp-view", "View");
        b.type = "button";
        b.title = "Open the deck in the sidebar's text viewer";
        b.addEventListener("click", () => proj.showPreview(full));
        s.appendChild(b);
    }
    return s;
}

/* ---- the panel ------------------------------------------------------------ */

/* `dir` is the folder the page is bound to (the scope); `recordDir` the one
 * the record was read from, whose file names the record gives -- a run picked
 * off a calculation's ladder is read in its own folder while the page stays
 * bound to the calculation (results.md § 2.4). */
const shown = { record: null, dir: "", recordDir: "" };
let open = false;

function render() {
    const host = document.getElementById("results-run-panel");
    if (!host) return;
    host.textContent = "";
    const rec = shown.record;
    if (!rec) { host.hidden = true; return; }

    const head = el("div", "rp-head");
    const toggle = el("button", "rp-toggle", "Run");
    toggle.type = "button";
    toggle.setAttribute("aria-expanded", open ? "true" : "false");
    toggle.title = open ? "Hide what ran" : "Show what ran, with what, and how it went";
    head.appendChild(toggle);
    const line = el("span", "rp-summary");
    if ((rec.verdict || {}).state) line.appendChild(chip(rec.verdict.state));
    summary(rec).forEach((p) => line.appendChild(el("span", "rp-part", p)));
    head.appendChild(line);
    head.addEventListener("click", () => {
        open = !open;
        render();
        // THE PANEL IS DRAWN AGAIN, its toggle with it: focus goes back to
        // the toggle, so a keyboard user stays where they pressed
        // (ui-contract.md § 4.1).
        const again = host.querySelector(".rp-toggle");
        if (again) again.focus();
    });
    host.appendChild(head);

    if (open) {
        const body = el("div", "rp-body");
        if (rec.verdict || rec.earlier) body.appendChild(verdictSection(rec));
        if (rec.setup) body.appendChild(setupSection(rec));
        if (rec.computation) body.appendChild(computationSection(rec));
        if (rec.deck) body.appendChild(deckSection(rec, shown.recordDir));
        host.appendChild(body);
    }
    host.hidden = false;
}

if (C.EVENT_FILE_SELECTED) {
    document.addEventListener(C.EVENT_FILE_SELECTED, (evt) => {
        const d = (evt && evt.detail) || {};
        shown.record = d.record || null;
        shown.dir = d.dir || "";
        shown.recordDir = d.record_dir || "";
        render();
    });
}
if (C.EVENT_SCOPE_CHANGED) {
    document.addEventListener(C.EVENT_SCOPE_CHANGED, (evt) => {
        const dir = ((evt && evt.detail) || {}).dir || "";
        if (shown.record && dir !== shown.dir) {
            shown.record = null;
            shown.dir = dir;
            shown.recordDir = "";
            render();
        }
    });
}
