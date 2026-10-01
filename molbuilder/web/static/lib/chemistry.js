/**
 * chemistry.js — the ONE chemistry card: the charge and spin this calculation
 * will carry, and why (docs/science/chemistry-correctness.md § 2a).
 *
 * A blank charge or spin item is the instruction "work it out", and the
 * electronic-state class works it out once, for the form, the checks and the
 * deck alike.  This module asks /api/structure/analyze for the state of
 * EXACTLY what the page's forms say, about EXACTLY the structure the page
 * would hand over -- the one its viewer holds, sent as the envelope the
 * preflight and the hand-over send -- and shows each value with where it came
 * from: on the card, and as each form's chip.  It fills nothing in: a blank
 * is already the instruction, and the card is its answer.
 *
 * (Until 2026-09-28 this was `auto-detect.js`, and an Auto-detect button copied
 * a per-engine SUGGESTION into the forms -- overwriting them.  Until the M6
 * review the card was handed a FILE PATH and the server re-read the file,
 * so after a restore, or with the file changed on disk, it answered for a
 * structure the deck would not carry.)
 *
 * The markup is `templates/_chemistry_card.html`; elements are built by
 * `lib/dom.js`.
 *
 * Exports (on window.molbuilder.chemistry):
 *   attach(opts)        -> { refresh() } — the page's card
 *   analyze(structure, opts) -> Promise<result> — the supersede protocol
 *   renderPanel(resp, hosts)
 *
 * ``attach`` owns everything a tab used to hand-write three times (the
 * analyze call, the re-analysis on an edit, the status line):
 *
 *   opts.kind        "optimization" | "vibration" | "transport"
 *   opts.forms()     {engine: {host, schema}} -- the rendered forms whose
 *                    state items the answer is for
 *   opts.structure() the structure envelope the page would hand over, or
 *                    null when it holds none
 *
 * Both are read at every ``refresh()``, and a change to one of the four
 * items in a form re-asks by itself.  The page calls ``refresh()`` when its
 * structure or its forms change -- a load, a restore, a citation, a form
 * rendered.  With no structure, or when the server cannot answer, the card
 * and the chips are hidden: an answer for another structure is not left on
 * screen.
 *
 * ``analyze`` returns an envelope rather than throwing:
 *   { ok: true, body } · { ok: false, superseded: true } · { ok: false, error }
 */
(function () {
    "use strict";

    var root = (typeof globalThis !== "undefined") ? globalThis
            : (typeof window !== "undefined") ? window : this;

    //: The electronic state's four items (science/chemistry-correctness.md
    //: § 2a.1) -- the fields whose change re-asks for the answer.
    var STATE_ITEMS = ["net_charge", "spin_treatment", "unpaired_electrons",
                       "method"];

    var ENGINE_NAME = { siesta: "SIESTA", pyscf: "PySCF" };

    function _$(id) { return root.document.getElementById(id); }

    function _el(tag, cls, text) {
        return root.molbuilder.dom.el(tag, cls, text);
    }

    /** The one fetch-failure sentence (lib/fetch-error.js), degrading to the
     *  bare message when the formatter is not on the page. */
    function _formatFetchError(e) {
        var f = root.molbuilder && root.molbuilder.fetchError;
        if (f && typeof f.format === "function") return f.format(e);
        return (e && e.message) ? String(e.message) : "request failed";
    }

    function _setStatus(text, severity) {
        var st = root.molbuilder && root.molbuilder.status;
        if (st && typeof st.set === "function") {
            st.set("chemistry-status", text || "", severity || null);
            return;
        }
        var el = _$("chemistry-status");
        if (el) el.textContent = text || "";
    }

    /** A form's four state items, as the form holds them -- through
     *  form-schema's one collector, so a blank is null here exactly as it
     *  is on the hand-over.  `null` when one will not read as its type:
     *  the field's own caption says why, and the card is not asked about a
     *  value nobody can send. */
    function _stateItems(host, schema) {
        var fs = root.molbuilder && root.molbuilder.formSchema;
        if (!host || !schema || !fs || typeof fs.collectForm !== "function") {
            return {};
        }
        try {
            return fs.collectForm(host, schema, STATE_ITEMS);
        } catch (_) {
            return null;
        }
    }

    // ------------------------------------------------------------ render

    function _spinText(st) {
        var t = st.spin_treatment.value, c = st.unpaired_electrons.value;
        if (t === "restricted") return "restricted (closed shell, 2S = 0)";
        return t + (c === "free" ? ", the moment floats (free)"
                                 : ", 2S = " + c);
    }

    /** One engine's state as the card's lines.  Each reason is the server's
     *  own phrasing (`Resolved.said`), the one a deck comment carries. */
    function _stateBlock(engine, st) {
        var box = _el("div", "chemistry-engine");
        box.appendChild(_el("h3", "chemistry-engine-name",
                            ENGINE_NAME[engine] || engine));
        var dl = _el("dl", "chemistry-items");
        function row(label, value, why) {
            dl.appendChild(_el("dt", null, label));
            var dd = _el("dd");
            dd.appendChild(_el("span", "chemistry-value", value));
            if (why) dd.appendChild(_el("span", "chemistry-why", " — " + why));
            dl.appendChild(dd);
        }
        var q = st.net_charge.value;
        row("Charge", (q > 0 ? "+" : "") + q, st.net_charge.said);
        var t = st.spin_treatment.said, c = st.unpaired_electrons.said;
        row("Spin", _spinText(st), t === c ? t : t + "; the count: " + c);
        if (st.method && st.method.source !== "rule") {
            row("Method", st.method.value, st.method.said);
        }
        row("Electrons", String(st.n_electrons),
            st.finite ? "a finite system"
                      : "a repeating cell -- the count per cell is not a spin");
        box.appendChild(dl);
        return box;
    }

    function _chips(resp, hosts) {
        var chip = root.molbuilder && root.molbuilder.detectionChip;
        if (chip && typeof chip.render === "function") chip.render(resp, hosts);
    }

    /**
     * Fill the card from an analyze response: each engine's state, the
     * metals' common spins, and each form's chip (``hosts``: {engine: the
     * element holding that engine's form}).  A page with no card has
     * nothing to draw on.
     */
    function renderPanel(resp, hosts) {
        var panel = _$("chemistry-panel");
        if (!panel) return;
        panel.hidden = false;
        var stateEl = _$("chemistry-state");
        if (stateEl) {
            stateEl.textContent = "";
            var states = (resp && resp.state) || {};
            Object.keys(states).forEach(function (engine) {
                stateEl.appendChild(_stateBlock(engine, states[engine]));
            });
        }
        var metBox = _$("chemistry-metals-box");
        var metEl = _$("chemistry-metals");
        if (metEl) {
            metEl.textContent = "";
            var hs = (resp && resp.metal_hints) || [];
            hs.forEach(function (h) {
                metEl.appendChild(_el("dt", null, h.element));
                (h.common_spins || []).forEach(function (cs) {
                    metEl.appendChild(_el("dd", null,
                        "2S = " + cs.spin + " — " + cs.label));
                });
            });
            if (metBox) metBox.hidden = hs.length === 0;
        }
        _chips(resp, hosts);
    }

    /** Nothing to answer, or no answer: the card and the chips go, so the
     *  last structure's answer is not read as this one's. */
    function _hide(hosts) {
        var panel = _$("chemistry-panel");
        if (panel) panel.hidden = true;
        var stateEl = _$("chemistry-state");
        if (stateEl) stateEl.textContent = "";
        _chips(null, hosts);
    }

    // ----------------------------------------------------------- protocol

    //: Bumped by every analyze; a call whose number is no longer the latest
    //: has been superseded and must not touch the DOM.
    var _seq = 0;
    //: Shared, so a newer request kills the older one ON THE WIRE.
    var _abort = null;

    function analyze(structure, opts) {
        if (!structure || typeof structure !== "object") {
            return Promise.resolve(
                { ok: false, error: "No structure to analyze." });
        }
        opts = opts || {};
        var body = { structure: structure };
        if (opts.kind) body.kind = opts.kind;
        if (opts.forms) body.forms = opts.forms;
        var mySeq = ++_seq;
        if (_abort) _abort.abort();
        _abort = new (root.AbortController)();
        var mySignal = _abort.signal;

        function superseded() { return mySeq !== _seq; }

        return root.fetch("/api/structure/analyze", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify(body),
            signal:  mySignal,
        }).then(function (r) {
            return r.json().then(function (b) {
                if (superseded()) return { ok: false, superseded: true };
                if (!r.ok || !b || !b.ok) {
                    return { ok: false,
                             error: (b && b.error) ? b.error
                                 : "Analyze failed (HTTP " + r.status + ")." };
                }
                return { ok: true, body: b };
            });
        }).catch(function (e) {
            // AbortError IS the supersede signal.
            if (e && e.name === "AbortError") {
                return { ok: false, superseded: true };
            }
            if (superseded()) return { ok: false, superseded: true };
            return { ok: false, error: _formatFetchError(e) };
        });
    }

    // ------------------------------------------------------------- attach

    function attach(opts) {
        opts = opts || {};
        var timer = null;
        var watched = [];         // hosts with a listener on them

        function forms() {
            return (typeof opts.forms === "function") ? (opts.forms() || {})
                                                      : {};
        }

        function refresh() {
            var fs = forms(), items = {}, hosts = {}, unreadable = false;
            Object.keys(fs).forEach(function (engine) {
                var f = fs[engine];
                if (!f || !f.host || !f.schema) return;
                var held = _stateItems(f.host, f.schema);
                if (held === null) unreadable = true;
                else items[engine] = held;
                hosts[engine] = f.host;
                _watch(f.host);
            });
            // A STATE FIELD THAT WILL NOT READ is said beside it, by its own
            // caption; the card keeps what it showed until it reads.
            if (unreadable) return Promise.resolve({ ok: false });
            var structure = (typeof opts.structure === "function")
                ? opts.structure() : null;
            if (!structure) {
                // A newer "nothing" supersedes an answer still on its way.
                ++_seq;
                if (_abort) _abort.abort();
                _hide(hosts);
                _setStatus("", null);
                return Promise.resolve({ ok: false });
            }
            _setStatus("Reading the structure…", null);
            return analyze(structure, {
                kind: opts.kind,
                forms: Object.keys(items).length ? items : undefined,
            }).then(function (res) {
                if (res.superseded) return res;
                if (!res.ok) {
                    _hide(hosts);
                    _setStatus(res.error, "error");
                    return res;
                }
                renderPanel(res.body, hosts);
                _setStatus("What this calculation will carry, and why -- "
                           + "change a charge or spin field and it follows.",
                           null);
                return res;
            });
        }

        /** One delegated listener per host: a change to one of the four
         *  items re-asks, after the typing settles.  Which fields those are
         *  is read from the form as it stands when the change happens -- a
         *  form rendered again after this listener went on is still read. */
        function _watch(host) {
            if (watched.indexOf(host) !== -1) return;
            watched.push(host);
            host.addEventListener("change", function (e) {
                var id = e.target && e.target.id;
                if (!id) return;
                var fs = forms();
                var hit = Object.keys(fs).some(function (engine) {
                    var f = fs[engine];
                    if (!f || f.host !== host || !f.schema) return false;
                    return (f.schema.sections || []).some(function (sec) {
                        return (sec.fields || []).some(function (fd) {
                            return fd.id === id
                                && STATE_ITEMS.indexOf(fd.name) !== -1;
                        });
                    });
                });
                if (!hit) return;
                if (timer) root.clearTimeout(timer);
                timer = root.setTimeout(refresh, 150);
            });
        }

        return { refresh: refresh };
    }

    var api = { attach: attach, analyze: analyze, renderPanel: renderPanel };
    if (typeof module !== "undefined" && module.exports) {
        module.exports = api;
    }
    root.molbuilder = root.molbuilder || {};
    root.molbuilder.chemistry = api;
})();
