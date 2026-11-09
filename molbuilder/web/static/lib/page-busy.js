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
 * send again.  FOCUS COMES BACK WHERE THE PERSON LEFT IT: a control that
 * held focus as the fence went up gives it to the cover and gets it back at
 * release; a fence that went up while focus was moving (a field's change runs
 * as Tab or a click leaves it) leaves the move to finish, and focus comes
 * back to where it went -- a control the operation redrew found again
 * (``_foundAgain``).
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
let _before = null;         // the spot focused as the fence went up, or null

/* A CONTROL'S SPOT: the control, and the address it is found again by once
 * an operation has redrawn it -- its id; or, with none, its name (its
 * `aria-label`, a button's text) inside the nearest container that has an
 * id.  Taken while the control is in the page: a redraw detaches it from
 * its container. */
function _nameOf(node) {
    const said = typeof node.getAttribute === "function"
        ? node.getAttribute("aria-label") : null;
    if (said) return said;
    return node.tagName === "BUTTON" ? String(node.textContent || "").trim() : "";
}

function _spotOf(node) {
    if (!node) return null;
    if (node.id) return { node, id: node.id };
    let box = node.parentElement || null;
    while (box && !box.id) box = box.parentElement || null;
    const name = _nameOf(node);
    return { node, within: box ? box.id : null, tag: node.tagName, name };
}

/* ...FOUND AGAIN: the control itself while it is in the page; else the one
 * now at its address -- by its id, or the ONE control of its kind and name
 * in its container.  Two answering to the name: none is guessed, so no
 * keystroke lands in another stage's field. */
function _foundAgain(spot) {
    if (!spot) return null;
    if (spot.node && spot.node.isConnected) return spot.node;
    if (typeof document.getElementById !== "function") return null;
    if (spot.id) return document.getElementById(spot.id);
    if (!spot.within || !spot.name) return null;
    const box = document.getElementById(spot.within);
    if (!box || typeof box.querySelectorAll !== "function") return null;
    const hits = Array.from(box.querySelectorAll(spot.tag))
        .filter((n) => _nameOf(n) === spot.name);
    return hits.length === 1 ? hits[0] : null;
}

/* WHERE FOCUS IS, by the fence's own record (ui-contract.md § 10): the spot
 * of the control outside the cover that last took it, cleared when focus
 * goes to nothing (a click on bare page).  While a field's change runs as
 * Tab leaves it, the browser names no control focused, and focus lands on
 * the next field only after the fence went up: this record is how the
 * release knows where it went.
 *
 * NOT EVERY "FOCUS WENT TO NOTHING" IS A CLICK ON BARE PAGE.  A browser
 * says so too when the focused control is taken out of the page (the
 * control still in it as it says so) and when the window loses focus: both
 * keep the record.  A control an operation's redraw took out is found
 * again at its release; one already out of the page when an operation
 * begins is let go there (`claim`) -- never brought back. */
let _held = null;
if (typeof document === "object" && document
        && typeof document.addEventListener === "function") {
    document.addEventListener("focusin", (e) => {
        if (_els && _els.cover.contains(e.target)) return;
        _held = _spotOf(e.target);
    }, true);
    document.addEventListener("focusout", (e) => {
        if (!_held || e.target !== _held.node || e.relatedTarget) return;
        const left = _held;
        queueMicrotask(() => {
            if (_held !== left || !left.node.isConnected) return;
            if (typeof document.hasFocus === "function" && !document.hasFocus()) {
                return;                         // the window left
            }
            _held = null;
        });
    }, true);
}

/* FOCUS GIVEN BACK at the release (ui-contract.md § 10).  A control
 * outside the cover that has focus keeps it: where the person's move
 * landed, or where the operation put it.  Else the control that held focus
 * as the fence went up, or the one focus moved to, found again. */
function _giveFocusBack(before) {
    const now = document.activeElement || null;
    if (now && now !== document.body && !(_els && _els.cover.contains(now))) {
        return;
    }
    const back = _foundAgain(before || _held);
    if (back && back.isConnected && typeof back.focus === "function") {
        back.focus({ preventScroll: true });
    }
}

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
            // FOCUS MOVING -- no control named focused: the move finishes.
            const active = document.activeElement || null;
            const moving = !active || active === document.body;
            _before = moving ? null : _spotOf(active);
            // A CONTROL ALREADY OUT OF THE PAGE is not where focus is.
            if (_held && !_held.node.isConnected) _held = null;
            if (typeof document.addEventListener === "function") {
                document.addEventListener("keydown", _fenceKey, true);
            }
            // THE FENCE NEVER MOVES THE VIEW: focus goes to the cover and
            // back without scrolling, so what the operation brought into
            // view -- a refusal beside its field -- stays there.
            const rest = _els.cancel.hidden ? _els.panel : _els.cancel;
            if (!moving && typeof rest.focus === "function") {
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
            // ...ONCE THE CALLER HAS ENDED ITS OPERATION -- a button it
            // disabled enabled again -- and a move the browser is making has
            // finished: an operation can end inside the change of the field
            // Tab is leaving.
            setTimeout(_giveFocusBack, 0, _before);
            _before = null;
        }
    },

    /** Run ``fn`` under the fence -- THE way an operation takes it (§ 10):
     *  claim, run, release in a ``finally``.  When the fence is already
     *  held, ``fn`` runs inside that claim: with the mouse and the keyboard
     *  fenced, only the operation holding it can reach another ``run``, so
     *  the caller is one of its steps, and two steps compose in one claim;
     *  the step runs under that claim's Cancel, its own ``cancelers``
     *  unused.  ``claim`` and ``release`` are called through this object,
     *  so a spy on them sees every claim. */
    async run(reason, cancelers, fn) {
        if (_claim !== null) return await fn();
        pageBusy.claim(reason, cancelers);
        try {
            return await fn();
        } finally {
            pageBusy.release();
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
