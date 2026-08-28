/**
 * Page busy fence — the ONE full-window cover for heavy user-triggered
 * operations.  Contract: docs/web/ui-contract.md § 10.
 *
 * Replaces the sidebar-scoped lock of 2026-05-27 (state.js), which built
 * the banner, the Cancel and the three-layer recovery — and never gained
 * a production caller.  The semantics here are that lock's, verbatim;
 * only the coverage grew from one sidebar to the whole window, because
 * covering the window blocks tab switching, sidebar clicks and every
 * form at once — nothing is guarded per-control (user, 2026-08-28:
 * "blocking the window is going to block switching so make it simple").
 *
 * The recovery contract (all three layers required at every call site):
 *   A. ``release()`` in a ``finally`` — a thrown operation still releases;
 *   B. the cancelers abort the in-flight request (AbortController);
 *   C. the Cancel button is the human override — always clickable, runs
 *      the cancelers, does NOT release (the finally does, after the
 *      abort unwinds).  No silent stuck state.
 *
 * ``claim()`` while claimed THROWS: one heavy operation at a time is the
 * design.  Periodic/automatic work (GPU sampler, polling) never claims —
 * there is no click to block, and freezing the page on a timer would be
 * the bug; that class is fenced server-side instead.
 *
 * DOM is best-effort: state (claimed / reason / cancelers) is enforced
 * even where no document exists (the node test harness), so a consumer
 * like ``navigateTo`` can rely on ``isClaimed()`` unconditionally.
 */

let _claim = null;          // { reason: string, cancelers: Function[] } | null
let _els = null;            // { cover, msg } once built

function _canPaint() {
    // Partial DOM stubs (the node test harnesses) fall back to the
    // state-only path.
    return (typeof document === "object" && document
            && typeof document.createElement === "function"
            && document.body && typeof document.body.appendChild === "function");
}

function _ensureDom() {
    if (_els || !_canPaint()) return;
    // Styles live in lib/page-shell.css (§ 2.1, beside the spinner):
    // a widget used on every page belongs to the SHARED sheet, and that
    // sheet loads in every page's <head> -- so the cover's geometry
    // exists before this first claim.  This module owns behavior + DOM
    // only; visibility is the ``hidden`` attribute (ui-contract § 6).
    const cover = document.createElement("div");
    cover.className = "page-busy-cover";
    cover.hidden = true;
    cover.setAttribute("role", "alert");
    cover.setAttribute("aria-live", "assertive");

    const panel = document.createElement("div");
    panel.className = "page-busy-panel";

    const spin = document.createElement("span");
    spin.className = "spinner";

    const msg = document.createElement("span");
    msg.className = "page-busy-msg";

    const cancel = document.createElement("button");
    cancel.type = "button";
    cancel.className = "page-busy-cancel";
    cancel.textContent = "Cancel";
    cancel.addEventListener("click", _runCancelers);

    panel.appendChild(spin);
    panel.appendChild(msg);
    panel.appendChild(cancel);
    cover.appendChild(panel);
    document.body.appendChild(cover);
    _els = { cover, msg };
}

/** Run the registered cancelers (Cancel button / tests).  Does NOT
 *  release — the operation's own ``finally`` does, after its abort
 *  path has unwound.  One bad canceler cannot break the rest. */
function _runCancelers() {
    if (_claim === null) return;
    for (const fn of _claim.cancelers.slice()) {
        try { fn(); } catch (_) { /* isolated */ }
    }
}

export const pageBusy = {
    /** Cover the window.  ``reason`` is shown to the user beside the
     *  spinner and the Cancel button; ``cancelers`` are zero-arg
     *  callables Cancel invokes (typically ``[() => ctl.abort()]``).
     *  Throws when already claimed — compose two operations in one
     *  claim, or release between them. */
    claim(reason, cancelers) {
        if (_claim !== null) {
            throw new Error(
                "pageBusy.claim(): already claimed -- previous reason: "
                + _claim.reason + ", new: " + reason);
        }
        _claim = {
            reason: String(reason || "Working…"),
            cancelers: Array.isArray(cancelers) ? cancelers.slice() : [],
        };
        _ensureDom();
        if (_els) {
            _els.msg.textContent = _claim.reason;
            _els.cover.hidden = false;
            if (typeof _els.cover.setAttribute === "function") {
                _els.cover.setAttribute("aria-busy", "true");
            }
        }
        return _claim;
    },

    /** Uncover.  Idempotent — always call from a ``finally`` so a thrown
     *  operation cannot leave the window covered. */
    release() {
        if (_claim === null) return;
        _claim = null;
        if (_els) _els.cover.hidden = true;
    },

    /** Is the fence held?  Programmatic movers of shared state
     *  (``projects.navigateTo``) refuse while this is true. */
    isClaimed() { return _claim !== null; },

    /** The current reason, or null — for refusal messages. */
    reason() { return _claim === null ? null : _claim.reason; },

    /** Test seam: what the Cancel button runs. */
    _runCancelers,
};

// Classic-script door: the modify/* panels are non-module scripts and
// cannot ``import``; the projects sidebar (a module, present on every
// page) pulls this file in via state.js, so the global exists before
// any panel's click handler can run.
if (typeof window === "object" && window) {
    window.molbuilder = window.molbuilder || {};
    window.molbuilder.pageBusy = pageBusy;
}
