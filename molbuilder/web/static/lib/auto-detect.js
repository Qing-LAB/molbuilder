/**
 * auto-detect.js — the ONE auto-detect chemistry surface.
 *
 * Every tab that asks the analyzer about the loaded structure runs
 * the SAME two halves: a POST to /api/structure/analyze under a
 * supersede protocol, and a render of the returned ChemistryAnalysis
 * into the "Analyze chemistry" card.  Until this module existed
 * (2026-08-22) both halves were hand-pasted per tab —
 * ``_renderAutoDetectPanel`` three times, byte-identical for 40
 * lines, over five hand-rolled fetch call sites each with its own
 * AbortController and sequence counter (`docs/web/audit-2026-08-05-
 * tab-ui.md` §§ C1, C2).  That is how the Spectrum tab came to
 * render the rationale panel but no detection chip (§ A2): a fix
 * applied to one copy simply did not exist in the others.
 *
 * The markup those ids live in is `templates/_analyze_chemistry_card.html`.
 *
 * Exports (on window.molbuilder.autoDetect):
 *   renderPanel(resp)     → boolean — did the panel exist to fill
 *   analyze(path, opts)   → Promise<result>
 *
 * ``analyze`` owns the CONCURRENCY protocol and nothing else; the
 * caller keeps every policy decision (status wording, whether a
 * failure is loud, what to do with the values).  It returns an
 * envelope rather than throwing, so a caller states its policy in
 * one line instead of re-deriving the protocol:
 *
 *   { ok: true,  body }               — success, and still current
 *   { ok: false, superseded: true }   — a newer analyze or load won;
 *                                       the DOM belongs to that one
 *   { ok: false, error: "<message>" } — the server said no, or the
 *                                       network died
 *
 * A background fire that must stay silent writes ``if (!res.ok)
 * return;``.  A user-initiated click that must speak writes
 * ``if (res.superseded) return;`` and then reports ``res.error``.
 *
 * opts:
 *   isStale()  — optional predicate, re-checked AFTER every await.
 *                The module knows when a newer ANALYZE superseded
 *                this one; only the page knows when a newer
 *                STRUCTURE LOAD did.  Returning true drops the
 *                response as superseded.
 *
 * The sequence counter and AbortController are module-scoped
 * because a page has exactly one auto-detect surface — which is
 * what each hand-rolled copy already assumed by keeping one
 * ``_autoDetectSeq`` per page shared across its call sites.
 *
 * HOLD-OUT: `lib/transport/core.js` still carries its own copy.
 * Transport's UI is designed in its own round (user ruling); it
 * joins there.  A shared module with one recorded hold-out beats
 * three copies drifting.
 */
(function () {
    "use strict";

    var root = (typeof globalThis !== "undefined") ? globalThis
            : (typeof window !== "undefined") ? window : this;

    function _$(id) { return root.document.getElementById(id); }

    /** The one fetch-failure sentence (lib/fetch-error.js). */
    function _formatFetchError(e) {
        return root.molbuilder.fetchError.format(e);
    }

    /**
     * Fill the "Analyze chemistry" card from an analyze response.
     *
     * Renders the rationale, the warnings list and the metal-spin
     * hints, hiding each sub-element that has nothing to say, then
     * refreshes the workflow-group detection chips — the chip pass
     * belongs HERE, not in the callers, because a tab that renders
     * the panel and forgets the chips is precisely § A2.
     *
     * Returns false when the card is absent (a page that never
     * included the partial), so a caller can tell "nothing to draw
     * on" from "drew nothing".
     */
    function renderPanel(resp) {
        var panel = _$("auto-detect-panel");
        if (!panel) return false;
        panel.hidden = false;
        panel.open = true;
        var ratEl  = _$("auto-detect-rationale");
        var warnEl = _$("auto-detect-warnings");
        var metEl  = _$("auto-detect-metals");
        var suggested = (resp && resp.suggested) || {};
        if (ratEl) {
            // The rationale is engine-AGNOSTIC — the analyzer writes
            // it once and every adapter echoes the same string, so
            // any present engine answers.  Falling back rather than
            // reading `.pyscf` alone is what lets a SIESTA-only
            // response still show its reasoning.
            var sug = suggested.pyscf || suggested.siesta || {};
            ratEl.textContent = sug.rationale || "";
        }
        if (warnEl) {
            warnEl.textContent = "";
            var ws = (resp && resp.warnings) || [];
            for (var i = 0; i < ws.length; i++) {
                var li = root.document.createElement("li");
                li.textContent = ws[i];
                warnEl.appendChild(li);
            }
            warnEl.hidden = ws.length === 0;
        }
        if (metEl) {
            metEl.textContent = "";
            var hs = (resp && resp.metal_hints) || [];
            for (var j = 0; j < hs.length; j++) {
                var h = hs[j];
                var dt = root.document.createElement("dt");
                dt.textContent = h.element;
                metEl.appendChild(dt);
                var cs = h.common_spins || [];
                for (var k = 0; k < cs.length; k++) {
                    var dd = root.document.createElement("dd");
                    dd.textContent =
                        "spin=" + cs[k].spin + " — " + cs[k].label;
                    metEl.appendChild(dd);
                }
            }
            metEl.hidden = hs.length === 0;
        }
        // Inject / refresh the .workflow-detection-chip on every
        // .workflow-group--profile + .workflow-group--budget card
        // header, so the analyzer's key conclusion sits on the card
        // where the user will act on it.  Stage-tagged fields get no
        // chip — the staged-relaxation recipe is system-agnostic.
        var chip = root.molbuilder && root.molbuilder.detectionChip;
        if (chip && typeof chip.render === "function") chip.render(resp);
        return true;
    }

    //: Bumped by every analyze; a call whose snapshot no longer
    //: matches has been superseded and must not touch the DOM.
    var _seq = 0;
    //: Shared across call sites so a spam-click, or a background
    //: fire racing a manual one, kills the older request ON THE
    //: WIRE rather than letting the server finish parsing a
    //: response nobody will read.
    var _abort = null;

    function analyze(path, opts) {
        if (!path) {
            return Promise.resolve(
                { ok: false, error: "No structure to analyze." });
        }
        var isStale = (opts && opts.isStale) || function () { return false; };
        var mySeq = ++_seq;
        if (_abort) _abort.abort();
        _abort = new (root.AbortController)();
        var mySignal = _abort.signal;

        function superseded() {
            return mySeq !== _seq || isStale();
        }

        return root.fetch("/api/structure/analyze", {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ structure_path: path }),
            signal:  mySignal,
        }).then(function (r) {
            return r.json().then(function (body) {
                if (superseded()) return { ok: false, superseded: true };
                if (!r.ok || !body || !body.ok) {
                    return {
                        ok: false,
                        error: (body && body.error)
                            ? body.error
                            : "Analyze failed (HTTP " + r.status + ").",
                    };
                }
                return { ok: true, body: body };
            });
        }).catch(function (e) {
            // AbortError IS the supersede signal — the newer request
            // owns the UI, and this one has nothing to report.
            if (e && e.name === "AbortError") {
                return { ok: false, superseded: true };
            }
            if (superseded()) return { ok: false, superseded: true };
            return { ok: false, error: _formatFetchError(e) };
        });
    }

    /**
     * The background fire: analyse on load, show the reasoning, touch no
     * form.  Both describing tabs spelled this out identically -- fourteen
     * lines differing in the name of a sequence counter and the word
     * "form"/"forms" -- which is what a shared module is for and what the
     * first extraction (2026-08-22) missed by taking the renderer and the
     * protocol but leaving the caller.
     *
     * Silent on EVERY failure, superseded included: the user did not ask for
     * this fire, so it must not flash an error at them.  The button is there
     * to retry.  Form-fill stays with the explicit click, so a background
     * analyse never silently rewrites the user's numbers.
     *
     * `opts.isStale` is the page's "a newer structure arrived" test;
     * `opts.say` writes the status line in the caller's own words.
     */
    function analyzeOnLoad(path, opts) {
        opts = opts || {};
        return analyze(path, { isStale: opts.isStale }).then(function (res) {
            if (!res.ok) return res;
            renderPanel(res.body);
            if (typeof opts.say === "function") opts.say();
            return res;
        });
    }

    var api = { renderPanel: renderPanel, analyze: analyze,
                analyzeOnLoad: analyzeOnLoad };
    if (typeof module !== "undefined" && module.exports) {
        module.exports = api;
    }
    root.molbuilder = root.molbuilder || {};
    root.molbuilder.autoDetect = api;
})();
