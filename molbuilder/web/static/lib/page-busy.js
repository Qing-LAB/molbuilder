/**
 * Page busy fence — the ONE full-window cover for heavy user-triggered
 * operations.  Contract: docs/web/ui-contract.md § 10.
 *
 * It covers the whole window, because
 * covering the window blocks tab switching, sidebar clicks and every
 * form at once — nothing is guarded per-control (user, 2026-08-28:
 * "blocking the window is going to block switching so make it simple").
 *
 * The recovery contract:
 *   A. ``release()`` in a ``finally`` — a thrown operation still releases;
 *   B. the cancelers abort the in-flight request (AbortController);
 *   C. the Cancel button is the human override — shown when there are
 *      cancelers to run, always clickable, runs them, does NOT release (the
 *      finally does, after the abort unwinds).  No silent stuck state.
 *
 * THE KEYBOARD IS FENCED TOO: the cover stops the mouse, and while claimed
 * a key pressed anywhere but on the cover, or in a dialog the operation
 * itself opened, does nothing -- Enter on the button just clicked does not
 * send again.  Focus moves to the cover and comes back at release.
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
let _els = null;            // { cover, panel, msg, cancel } once built
let _before = null;         // the element focused when the fence went up

/** A key pressed outside the cover, or outside a dialog the operation
 *  opened, while claimed: nothing happens. */
function _fenceKey(e) {
    if (_claim === null || !_els) return;
    const t = e.target;
    if (t && (_els.cover.contains(t)
              || (typeof t.closest === "function" && t.closest("dialog[open]")))) {
        return;
    }
    e.preventDefault();
    e.stopPropagation();
}

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
    panel.tabIndex = -1;            // where focus rests when there is no Cancel

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
    _els = { cover, panel, msg, cancel };
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
            _els.cancel.hidden = _claim.cancelers.length === 0;
            _els.cover.hidden = false;
            if (typeof _els.cover.setAttribute === "function") {
                _els.cover.setAttribute("aria-busy", "true");
            }
            _before = document.activeElement || null;
            if (typeof document.addEventListener === "function") {
                document.addEventListener("keydown", _fenceKey, true);
            }
            // THE FENCE NEVER MOVES THE VIEW: focus goes to the cover and
            // back without scrolling, so what the operation brought into
            // view -- a refusal beside its field -- stays there.
            const rest = _els.cancel.hidden ? _els.panel : _els.cancel;
            if (typeof rest.focus === "function") {
                rest.focus({ preventScroll: true });
            }
        }
        return _claim;
    },

    /** Uncover.  Idempotent — always call from a ``finally`` so a thrown
     *  operation cannot leave the window covered. */
    release() {
        if (_claim === null) return;
        _claim = null;
        if (_els) {
            _els.cover.hidden = true;
            if (typeof document.removeEventListener === "function") {
                document.removeEventListener("keydown", _fenceKey, true);
            }
            if (_before && _before.isConnected
                    && typeof _before.focus === "function") {
                _before.focus({ preventScroll: true });
            }
            _before = null;
        }
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
