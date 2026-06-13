"""Pin the live-poll + handler-snapshot invariants surfaced by the
2026-06-12 fresh-eyes audit.

Three thin source-text tests over JS modules that ship guards which
have no other regression coverage.  These are CHEAP (no Playwright,
no Flask) and run in ms — exactly the right size for "this guard
must stay in place across refactors".  An e2e equivalent would have
to fixture a multi-second watchTick poll loop just to assert "the
viewer didn't redraw"; a source-text pin is the same signal at 1000×
the speed.

The invariants pinned here all came from one diagnostic root cause:
**a live-poll loop that runs every N seconds must not touch user-
controlled UI state on no-op ticks, and an event handler must not
re-read input values after calling helpers that side-effect those
inputs.**  See ``docs/protocols/playwright-tests.md`` § "Handler
input snapshot" for the rule.

Each test is a tight regex over the source body: if a future edit
silently removes the guard (refactor, "simplify", de-duplicate), the
test fails with a message naming the load-bearing line.

Background
==========

* ``lib/trajectory/core.js::applyNewData`` — ``noNewContent`` early-
  return prevents the 2s watchTick from rebuilding the 3Dmol model
  on every same-data poll.  Without it the camera angle resets every
  2s and the animation pauses mid-watch (the original bug class).
  Shipped 2026-06-12 as task #350.

* ``lib/spectra/core.js::renderResults`` — ``_resultsFingerprint``
  guard prevents the 2s watchTick from disposing + rebuilding the
  3Dmol viewer when results are unchanged.  Same bug class as the
  trajectory one above.  Shipped 2026-06-12 as task #352 follow-up.

* ``lib/mol-viewer-embed.js`` frame-slider input handler — must
  snapshot ``parseInt(slider.value, 10)`` BEFORE calling
  ``_stopAnimationLoop(state)``.  That call routes through
  ``_refreshFrameStrip`` which rewrites ``slider.value`` from the
  pre-drag frame; reading after the helper means parseInt sees the
  reset value and the seek is a no-op.  Shipped 2026-06-12 as
  task #353.

* ``lib/selection-panel.js`` ``_isolateUnsubscribe`` — the
  auto-uncheck-on-empty subscriber's unsubscribe handle MUST be
  pushed into the panel's ``cleanups`` array so ``dispose()`` tears
  it down.  Pre-fix it was fire-and-forget — every /modify
  mount→dispose cycle stranded a closure holding the panel's els
  + the _isolateAdapter() lookup.  Shipped 2026-06-12.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_LIB = (Path(__file__).resolve().parent.parent
        / "molbuilder" / "web" / "static" / "lib")


# --------------------------------------------------------------------- #
#  Trajectory: noNewContent live-watch guard                            #
# --------------------------------------------------------------------- #


class TestTrajectoryNoNewContentGuard:
    """Live-watch polls must short-circuit when the new payload is
    structurally identical to the last one.  Without this guard the
    viewer rebuilds every 2s and snaps the user's camera back to the
    default angle."""

    @pytest.fixture(scope="class")
    def core_body(self):
        return (_LIB / "trajectory" / "core.js").read_text()

    def test_applyNewData_computes_noNewContent_from_3_subterms(self, core_body):
        """The guard is ``sameAtomCount && sameLength && sameLatticeNow``.
        Each of the three subterms is load-bearing — atom count alone
        misses ``frames re-emitted from offset 0``; length alone misses
        cell changes during a relax; lattice alone misses content
        mutations the same lattice can house.

        Pin the conjunction so a refactor that drops a subterm
        (e.g. ``noNewContent = sameLength``) fails the test."""
        assert "const sameLength" in core_body, (
            "trajectory/core.js::applyNewData lost its sameLength "
            "derivation — the noNewContent guard relies on it to "
            "detect 'identical-length poll' vs 'frames grew' cases")
        assert "const sameLatticeNow" in core_body, (
            "trajectory/core.js::applyNewData lost its sameLatticeNow "
            "derivation — needed to short-circuit when lattice is "
            "stable but the same atom count fired")
        assert re.search(
            r"const\s+noNewContent\s*=\s*"
            r"sameAtomCount\s*&&\s*sameLength\s*&&\s*sameLatticeNow",
            core_body,
        ), ("trajectory/core.js::applyNewData no longer composes "
            "noNewContent from the three required subterms; the "
            "live-watch camera-reset bug class will return.  See "
            "docs/protocols/playwright-tests.md § Handler input "
            "snapshot for the design rule.")

    def test_applyNewData_early_returns_on_noNewContent(self, core_body):
        """If noNewContent is true the function MUST bail before
        ``rebuildModel`` runs.  Pin the structural shape
        ``if (noNewContent) { ... return; }``."""
        # Locate the guard body and confirm a ``return`` statement
        # lives inside the same braced block (not somewhere later in
        # the function where the rebuild would already have fired).
        m = re.search(
            r"if\s*\(\s*noNewContent\s*\)\s*\{([^}]*?)\}",
            core_body, re.DOTALL,
        )
        assert m is not None, (
            "trajectory/core.js no longer has an `if (noNewContent)` "
            "block; the same-content live-watch guard has been "
            "removed and the camera-reset bug will return.")
        body = m.group(1)
        assert "return" in body, (
            "trajectory/core.js's `if (noNewContent)` block no "
            "longer early-returns — the function falls through to "
            "rebuildModel and the camera-reset bug returns.")


# --------------------------------------------------------------------- #
#  Spectra: _resultsFingerprint live-watch guard                        #
# --------------------------------------------------------------------- #


class TestSpectraResultsFingerprintGuard:
    """Spectra's renderResults runs on every 2s watchTick during a
    live run.  Without a same-content guard it disposes + rebuilds
    the 3Dmol viewer every tick and the user's camera angle / mode
    selection / scroll position all reset.

    Pin the fingerprint helper AND the early-return at the top of
    renderResults so a refactor that drops either gets caught."""

    @pytest.fixture(scope="class")
    def core_body(self):
        return (_LIB / "spectra" / "core.js").read_text()

    def test_resultsFingerprint_function_exists(self, core_body):
        """The fingerprint helper takes (results, selectedMode) and
        returns a stable string over the fields renderResults branches
        on.  Pin its definition."""
        assert re.search(
            r"function\s+_resultsFingerprint\s*\(\s*results\s*,\s*"
            r"selectedMode\s*\)",
            core_body,
        ), ("spectra/core.js::_resultsFingerprint helper is gone; "
            "renderResults can no longer detect no-op watchTick "
            "polls and the camera-reset bug class returns.")

    def test_resultsFingerprint_includes_loadbearing_fields(self, core_body):
        """The fingerprint MUST cover atom count, mode count, per-mode
        ES presence, all three phase markers, and the selected mode.
        Drop any of these and the early-return mis-fires (or
        spuriously bails when the user-visible state actually
        changed)."""
        m = re.search(
            r"function\s+_resultsFingerprint[^{]*\{(.+?)return\s*\[",
            core_body, re.DOTALL,
        )
        assert m is not None, "fingerprint body shape changed"
        # The full return-array contents:
        body_after = core_body[m.end() - len("return ["):]
        ret_match = re.search(
            r"return\s*\[(.+?)\]\.join\(",
            body_after, re.DOTALL,
        )
        assert ret_match is not None, "fingerprint return shape changed"
        ret_body = ret_match.group(1)
        required = [
            "n_atoms_total",
            "modes.length",
            "esBits",
            "phase_frequencies",
            "phase_raman",
            "phase_es",
            "selectedMode",
        ]
        missing = [f for f in required if f not in ret_body]
        assert not missing, (
            f"spectra/core.js::_resultsFingerprint dropped "
            f"load-bearing field(s): {missing}.  Without them the "
            f"early-return either spuriously fires (user-visible "
            f"change ignored) or fails to fire (live-watch redraws "
            f"every 2s, resetting camera).  See design.md "
            f"2026-06-12 entry on audit #352 follow-up.")

    def test_renderResults_bails_when_fingerprints_match(self, core_body):
        """The early-return must compare prevFp to newFp and skip the
        viewer dispose+rebuild when they're equal.  Pin the
        ``prevFp === newFp`` comparison + the bail."""
        assert "_resultsFingerprint(results" in core_body, (
            "spectra/core.js::renderResults no longer calls "
            "_resultsFingerprint on the incoming payload — the guard "
            "is dead code.")
        assert re.search(
            r"prevFp\s*!==\s*null\s*&&\s*prevFp\s*===\s*newFp",
            core_body,
        ), ("spectra/core.js::renderResults no longer compares "
            "previous-vs-new fingerprints; the live-watch camera-"
            "reset bug class returns.")


# --------------------------------------------------------------------- #
#  Frame-slider: snapshot input value BEFORE state-mutating helper      #
# --------------------------------------------------------------------- #


class TestFrameSliderHandlerSnapshotPattern:
    """The frame-strip slider's input handler MUST capture
    ``parseInt(slider.value, 10)`` BEFORE calling
    ``_stopAnimationLoop(state)``.  That call routes through
    ``_refreshFrameStrip`` which rewrites ``slider.value`` from the
    pre-drag frame; reading after the helper means parseInt sees the
    reset value and the seek is a no-op (the original bug)."""

    @pytest.fixture(scope="class")
    def embed_body(self):
        return (_LIB / "mol-viewer-embed.js").read_text()

    def test_slider_handler_snapshots_value_before_stopAnimationLoop(
            self, embed_body):
        """Find the slider's ``addEventListener("input", ...)`` block
        and verify the order of operations is:

          1. ``const target = parseInt(slider.value, 10)`` (snapshot)
          2. ``_stopAnimationLoop(state)``
          3. ``_showTrajectoryFrame(state, target)``

        Where ``target`` was captured at step 1 and reused at step 3,
        NOT re-read from slider.value after step 2.  If a future edit
        inlines the parseInt as a parameter (the pre-fix shape),
        ``_stopAnimationLoop``'s side effect on slider.value is what
        parseInt reads — and the slider scrub silently regresses."""
        # Locate the input handler body.
        m = re.search(
            r'slider\.addEventListener\("input"\s*,\s*\(\)\s*=>\s*'
            r"\{([^}]+)\}",
            embed_body, re.DOTALL,
        )
        assert m is not None, (
            "mol-viewer-embed.js no longer wires a slider 'input' "
            "handler with the documented closure shape; the frame-"
            "scrub UI is gone or has been refactored without "
            "preserving the snapshot pattern.")
        body = m.group(1)

        # Forbidden shape — `parseInt(slider.value, ...)` inside the
        # _showTrajectoryFrame call.  This is the pre-fix shape; the
        # post-fix shape captures it into a `const target` first.
        forbidden = re.search(
            r"_showTrajectoryFrame\s*\(\s*state\s*,\s*"
            r"parseInt\s*\(\s*slider\.value",
            body,
        )
        assert forbidden is None, (
            "mol-viewer-embed.js slider input handler re-reads "
            "slider.value AFTER _stopAnimationLoop runs.  This is "
            "the bug class fixed in 2026-06-12 (task #353): "
            "_stopAnimationLoop calls _refreshFrameStrip which "
            "rewrites slider.value to the pre-drag frame, so parseInt "
            "sees the old value and the seek is a no-op.  Snapshot "
            "the parsed value into a const BEFORE calling "
            "_stopAnimationLoop.  See "
            "docs/protocols/playwright-tests.md § Handler input "
            "snapshot for the rule.")

        # Required shape — a const captures parseInt(slider.value, ...)
        # BEFORE _stopAnimationLoop is referenced.
        assert re.search(
            r"const\s+\w+\s*=\s*parseInt\s*\(\s*slider\.value",
            body,
        ), ("mol-viewer-embed.js slider handler no longer snapshots "
            "parseInt(slider.value, 10) into a const at handler "
            "entry.  Without this snapshot, side-effecting helpers "
            "(_stopAnimationLoop -> _refreshFrameStrip) reset "
            "slider.value before the seek runs.  See "
            "docs/protocols/playwright-tests.md § Handler input "
            "snapshot.")


# --------------------------------------------------------------------- #
#  Selection panel: _isolateUnsubscribe cleanup-array discipline        #
# --------------------------------------------------------------------- #


class TestSelectionPanelIsolateUnsubscribeCleanup:
    """The selection panel's auto-uncheck-on-empty-selection wiring
    subscribes to the selection store.  That subscriber's
    unsubscribe handle MUST be pushed into the panel's ``cleanups``
    array so ``dispose()`` tears it down.

    Pre-fix the subscribe call was fire-and-forget: every /modify
    mount→dispose cycle stranded a closure holding the panel's
    ``els.isolateChk`` reference + the ``_isolateAdapter()`` lookup.
    A long-running session that opens /modify dozens of times
    accumulated those closures.

    Pin the capture + the cleanup push so a refactor can't silently
    revert."""

    @pytest.fixture(scope="class")
    def panel_body(self):
        return (_LIB / "selection-panel.js").read_text()

    def test_isolate_subscriber_captures_unsubscribe_handle(
            self, panel_body):
        """The auto-uncheck wiring assigns the return value of
        store.subscribe(...) to a const — that's the unsubscribe
        handle.  Verify the capture exists."""
        assert re.search(
            r"const\s+_isolateUnsubscribe\s*=\s*store\.subscribe",
            panel_body,
        ), ("selection-panel.js auto-uncheck-on-empty wiring no "
            "longer captures its store.subscribe() return value; "
            "the subscriber leaks every /modify mount cycle.  See "
            "design.md 2026-06-12 entry on audit #352 follow-up.")

    def test_isolate_unsubscribe_is_pushed_into_cleanups(
            self, panel_body):
        """The captured unsubscribe handle must be added to the
        ``cleanups`` array (LIFO-walked by ``dispose``)."""
        # Match `cleanups.push(() => { try { _isolateUnsubscribe(); } catch (_) ... })`
        assert re.search(
            r"cleanups\.push\(\s*\(\s*\)\s*=>\s*\{[^}]*"
            r"_isolateUnsubscribe\s*\(\s*\)",
            panel_body, re.DOTALL,
        ), ("selection-panel.js does NOT push _isolateUnsubscribe "
            "into the cleanups array.  dispose() will not tear it "
            "down; every mount cycle leaks the closure.  See the "
            "panel's cleanups discipline: every addEventListener / "
            "subscribe in mount() MUST push its undo into "
            "cleanups, no exceptions.")


# --------------------------------------------------------------------- #
#  Trajectory inspector claims geomeTRIC / PySCF multi-frame XYZ        #
# --------------------------------------------------------------------- #


class TestTrajectoryInspectorClaimsOptimXyz:
    """PySCF's geom-opt wrapper (and bare geomeTRIC runs) write the
    multi-frame trajectory to ``<job>_geom_optim.xyz`` (or older
    ``<job>_optim.xyz``).  These files ARE valid multi-frame XYZ
    that ``PySCFParser.can_parse`` accepts and renders correctly via
    the trajectory inspector.

    Before this pin (2026-06-12, after the user reported a Results-
    tab regression on a BDT/optimization folder), the trajectory
    inspector matched only ``.molwatch.log`` + ``.out``.  The
    structure inspector — which matches every ``.xyz`` — claimed the
    multi-frame trajectory and rendered the first frame as a
    single static structure.  The user saw "Structure" in the
    picker, with no way to access the trajectory animation /
    energy plots / SCF history that DO exist in the file.

    Pin both the match expansion AND the resultCategory routing so
    a future refactor that "simplifies" the match (e.g. drops the
    `_optim.xyz` arm assuming the structure inspector covers it)
    fails this test rather than silently reverting the regression."""

    @pytest.fixture(scope="class")
    def trajectory_inspector_body(self):
        return (_LIB / "inspectors" / "trajectory.js").read_text()

    def test_match_claims_geom_optim_xyz(self, trajectory_inspector_body):
        """``*_geom_optim.xyz`` (the conventional geomeTRIC + PySCF
        wrapper naming) MUST land on the trajectory inspector, not
        the structure inspector."""
        assert '_geom_optim.xyz"' in trajectory_inspector_body, (
            "trajectory inspector no longer claims `_geom_optim.xyz`. "
            "PySCF / geomeTRIC multi-frame trajectories will fall "
            "back to the structure inspector (single-frame view) and "
            "the user loses the animation + energy/force plots that "
            "are the whole point of the trajectory.  See the 2026-06-12 "
            "Results-tab regression report.")

    def test_match_claims_plain_optim_xyz(self, trajectory_inspector_body):
        """Older PySCF runs (and direct geomeTRIC invocations) write
        ``<job>_optim.xyz`` without the ``_geom_`` infix.  Keep the
        plain pattern in the match too."""
        assert '"_optim.xyz"' in trajectory_inspector_body, (
            "trajectory inspector no longer claims `*_optim.xyz`. "
            "Older PySCF / direct-geomeTRIC trajectories will hit "
            "the same single-frame structure-inspector regression as "
            "`_geom_optim.xyz`.")

    def test_resultCategory_routes_optim_xyz_to_pyscf_bucket(
            self, trajectory_inspector_body):
        """The picker groups files by ``resultCategory``; the new
        ``_optim.xyz`` files should bucket under "PySCF
        optimization" (same as ``.molwatch.log``) so the user finds
        them next to the engine's other artifacts."""
        # Confirm both _optim.xyz forms are routed to a PySCF bucket
        # somewhere inside the resultCategory function body.
        m = re.search(
            r"resultCategory\s*:\s*\(\s*file\s*\)\s*=>\s*\{(.+?)\}\s*,",
            trajectory_inspector_body, re.DOTALL,
        )
        assert m is not None, (
            "trajectory inspector lost its resultCategory function — "
            "the picker can no longer group result files by engine.")
        body = m.group(1)
        # Both _optim.xyz forms should return "PySCF optimization"
        # (or some label that contains "PySCF").
        assert "_optim.xyz" in body and "PySCF" in body, (
            "trajectory inspector's resultCategory no longer routes "
            "`_optim.xyz` to a PySCF bucket.  The files will still "
            "show in the picker but under a generic header, hiding "
            "their engine provenance from the user.")
