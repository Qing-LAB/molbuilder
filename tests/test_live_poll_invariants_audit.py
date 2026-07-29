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
inputs.**  See ``docs/process/testing.md`` § "Handler
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

* ``lib/viewer/mol-viewer-embed.js`` frame-slider input handler — must
  snapshot ``parseInt(slider.value, 10)`` BEFORE calling
  ``_stopAnimationLoop(state)``.  That call routes through
  ``_refreshFrameStrip`` which rewrites ``slider.value`` from the
  pre-drag frame; reading after the helper means parseInt sees the
  reset value and the seek is a no-op.  Shipped 2026-06-12 as
  task #353.

* ``lib/molview/selection/panel.js`` ``_isolateUnsubscribe`` — the
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
            "docs/process/testing.md § Handler input "
            "snapshot for the design rule.")

    def test_applyNewData_early_returns_on_noNewContent(self, core_body):
        """If noNewContent is true the function MUST bail before
        ``rebuildModel`` runs.  Pin the structural shape
        ``if (noNewContent) { ... return; }``.

        Updated 2026-06-17 (PR 2.3): the noNewContent body now
        contains nested ``{...}`` (e.g. ``transition('APPLY',
        {mtime, data})``).  The naive [^}]*? regex stopped at the
        first ``}`` and broke.  Use a brace-balancing approach:
        find the opening brace, then walk to the matching close."""
        m = re.search(
            r"if\s*\(\s*noNewContent\s*\)\s*\{", core_body,
        )
        assert m is not None, (
            "trajectory/core.js no longer has an `if (noNewContent)` "
            "block; the same-content live-watch guard has been "
            "removed and the camera-reset bug will return.")
        # Walk from the position right after the opening brace,
        # counting depth until we find the matching close.
        start = m.end()
        depth = 1
        i = start
        while i < len(core_body) and depth > 0:
            if core_body[i] == "{":
                depth += 1
            elif core_body[i] == "}":
                depth -= 1
            i += 1
        assert depth == 0, "noNewContent body has unbalanced braces"
        body = core_body[start:i - 1]
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
        ES presence, all three phase markers, the selected mode, AND
        the per-mode activity values + frequencies (Fix C, 2026-06-17).
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
            # Fix C 2026-06-17: activity + frequency sums must be in
            # the fingerprint so chart bar heights refresh when
            # activities populate mid-phase (same mode count, same
            # phase markers).  Pre-fix, Raman activities landed
            # without firing renderResults -> bars stayed at 0.
            "_freqSum",
            'raman_activity_a4_amu',
            'ir_intensity_km_mol',
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
            f"2026-06-12 entry on audit #352 follow-up + "
            f"2026-06-17 Fix C (Raman/IR activities).")

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
        return (_LIB / "viewer" / "mol-viewer-embed.js").read_text()

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
            "docs/process/testing.md § Handler input "
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
            "docs/process/testing.md § Handler input "
            "snapshot.")


# --------------------------------------------------------------------- #
#  Selection panel: _isolateUnsubscribe cleanup-array discipline        #
# --------------------------------------------------------------------- #


class TestInjectedToggleUnsubscribeCleanup:
    """The isolate ("Show selected only") toggle subscribes to the
    selection store to keep its button + menu entry in sync.  That
    subscriber's unsubscribe handle MUST be torn down when the toggle
    is disposed so a mount→dispose cycle doesn't strand it.

    History: the auto-uncheck-on-empty behaviour lived in
    ``selection-panel.js``, then ``molview/view-controls.js``.  Both
    are gone.  Isolate is now view toggle #5, injected through the
    embed's ONE view-toggle path (``handle.addViewToggle`` in
    ``mol-viewer-embed.js``): the embed renders its button + menu entry
    and owns the store subscription that keeps them synced.  The
    subscription's unsubscribe handle is captured in ``addViewToggle``
    and invoked from the returned ``dispose()``.  The auto-clear rule
    itself moved into the store (``_afterSelectionChange``), so this
    test now pins only the SUBSCRIBER-LEAK invariant at its new home.

    Pin the capture + the dispose teardown so a refactor can't
    silently revert."""

    @pytest.fixture(scope="class")
    def embed_body(self):
        return (_LIB / "viewer" / "mol-viewer-embed.js").read_text()

    def test_injected_toggle_captures_unsubscribe_handle(self, embed_body):
        """addViewToggle assigns the return value of
        spec.subscribe(...) to a variable — the unsubscribe handle."""
        assert re.search(
            r"unsub\s*=\s*spec\.subscribe",
            embed_body,
        ), ("addViewToggle no longer captures its spec.subscribe() "
            "return value; the injected-toggle (isolate) subscriber "
            "leaks every viewer mount cycle.")

    def test_injected_toggle_unsubscribe_is_called_in_dispose(
            self, embed_body):
        """The captured unsubscribe handle must be invoked from the
        toggle's returned ``dispose()`` so teardown removes it."""
        # Match `dispose: function () { ... if (unsub) unsub(); ... }`
        assert re.search(
            r"dispose\s*:\s*function\s*\([^)]*\)\s*\{[^}]*"
            r"unsub\s*\(\s*\)",
            embed_body, re.DOTALL,
        ), ("addViewToggle's dispose() does NOT call unsub(); every "
            "mount cycle leaks the injected-toggle store subscriber.")


# --------------------------------------------------------------------- #
#  Trajectory inspector claims geomeTRIC / PySCF multi-frame XYZ        #
# --------------------------------------------------------------------- #


class TestConvergenceTargetsAndPlotColors:
    """Two invariants for the 2026-06-13 convergence-summary work:

    1. SIESTA parser populates ``runtime_info["convergence_targets"]``
       with the values from the input echo block.  Without this, the
       Results-tab summary block + threshold line have nothing to
       render and silently fall back to the "targets unknown" hint.

    2. ``trajectory/core.js::makePlots`` reads its trace colours via
       a ``_themeColors()`` helper rather than hardcoded hex.  This
       protects the recolor (free-atoms trace blue, SCF gnorm orange,
       threshold line green) from a future "simplification" that
       pastes literal hex back into the trace specs and silently
       reintroduces the green-on-green threshold-vs-trace conflict
       that motivated the 2026-06-13 recolor."""

    def test_siesta_parser_extracts_convergence_targets(self):
        """The hemeC stage-2 fixture's input echo carries
        MD.MaxForceTol=0.02 eV/Å (tightened from the default 0.04 for
        a stage-2 run).  Pin that the parser actually reads it."""
        from molbuilder.parse.engines.siesta import SiestaParser
        path = (Path(__file__).resolve().parent
                / "watch" / "fixtures" / "siesta_frozen"
                / "hemeC-stage2-run3-finished-42fr.out")
        traj = SiestaParser.parse(str(path))
        ct = traj.runtime_info.get("convergence_targets")
        assert ct is not None, (
            "SIESTA parser dropped its convergence_targets extraction. "
            "The Results-tab threshold line + summary block now have "
            "nothing to render.  Check that _SIESTA_FORCE_TOL_RE et al. "
            "are still matching 'redata: Force tolerance = ...'.")
        assert ct.get("source") == "siesta_input_echo"
        assert ct.get("max_force_tol_eV_per_A") == 0.02
        assert ct.get("dm_tolerance") == 1e-4
        assert ct.get("max_scf_iter") == 500

    def test_themeColors_helper_present(self):
        """``_themeColors()`` reads CSS tokens via getComputedStyle
        instead of hardcoded hex.  This is what lets the plot colours
        track a future theme retune.  If a refactor inlines the
        accent/success colours, this test fails and points at the
        regression."""
        core = (_LIB / "trajectory" / "core.js").read_text()
        assert "function _themeColors" in core, (
            "trajectory/core.js lost its _themeColors() helper.  Trace "
            "colours can no longer follow lib/tokens.css; the 2026-06-13 "
            "recolor (free atoms blue, SCF gnorm orange, threshold "
            "lines green) is at risk of silent regression.")
        # The function must actually read at least --accent + --success
        # — those drive the convergence-signal trace colour + the
        # threshold-line colour.
        body_match = re.search(
            r"function\s+_themeColors\s*\(\s*\)\s*\{(.+?)\n\s*\}",
            core, re.DOTALL,
        )
        assert body_match is not None, (
            "_themeColors() body shape changed; refactor needs to "
            "update this test alongside.")
        body = body_match.group(1)
        assert "--accent" in body, (
            "_themeColors no longer reads --accent.  The convergence-"
            "signal force trace falls back to hardcoded hex.")
        assert "--success" in body, (
            "_themeColors no longer reads --success.  The threshold "
            "line falls back to hardcoded hex.")

    def test_force_trace_uses_themed_accent_not_green(self):
        """The 'free atoms' convergence-gating trace MUST use the
        theme accent (blue), not literal green (#1f9d55), or the
        green threshold line collides with it.  Pin the absence of
        the pre-recolor literal so a copy-paste regression is
        caught at the regex level."""
        core = (_LIB / "trajectory" / "core.js").read_text()
        # The free-atoms branch should reference theme.accent for
        # its line color.
        assert re.search(
            r"name:\s*\"free atoms\"|line:\s*\{[^}]*theme\.accent",
            core, re.DOTALL,
        ), ("free-atoms force trace is no longer wired to theme.accent. "
            "If you've changed how the line colour is set, update this "
            "regex; if you've reintroduced the old green hex, the "
            "threshold-line-vs-trace collision is back.")
        # The pre-recolor green literal must NOT reappear in the
        # force-plot trace spec (it can still legitimately appear
        # in a comment).  Carve out comments by stripping them
        # before the search.
        body_no_comments = re.sub(r"/\*.*?\*/|//[^\n]*", "", core, flags=re.S)
        assert "#1f9d55" not in body_no_comments, (
            "Hardcoded green #1f9d55 reintroduced into trajectory/"
            "core.js outside comments.  This was the pre-2026-06-13 "
            "free-atoms force trace colour; using it again puts a "
            "green trace under the green threshold line.")

    def test_convergence_summary_rendered_via_textContent(self):
        """The new convergence-summary block uses textContent +
        createElement everywhere — no innerHTML interpolation.  Per
        the XSS audit's no-unsafe-innerHTML rule + the data values
        flow through float.toFixed / .toExponential before reaching
        the DOM, but pinning textContent here is the safer guard."""
        core = (_LIB / "trajectory" / "core.js").read_text()
        m = re.search(
            r"function\s+_renderConvergenceSummary\s*\([^)]*\)\s*\{(.+?)\n\s{4}\}",
            core, re.DOTALL,
        )
        assert m is not None, "_renderConvergenceSummary is missing"
        body = m.group(1)
        # Inner DOM writes go through textContent / createElement / appendChild.
        assert "innerHTML" not in body, (
            "_renderConvergenceSummary uses innerHTML — switch to "
            "textContent + createElement to keep the XSS audit clean.")


class TestWorkflowGroupSchemaConsistency:
    """The 2026-06-13 form restructure tags each schema field with
    ``workflow_group``: one of ``"system"`` / ``"stage"`` / ``"budget"``.
    Three invariants that protect the user-visible contract
    (switching the stage selector mutates ONLY stage-tagged fields):

    1. Every field ID in viewer.js's STAGE_PRESETS dict must have a
       matching SIESTA dataclass field tagged
       ``workflow_group: "stage"``.  A field that's in the preset
       but NOT stage-tagged would write to it AND trigger the
       allowlist's defensive warn() — half-done state.

    2. viewer.js's ``_STAGE_PRESET_KEYS_SIESTA`` allowlist must
       contain EXACTLY the keys in STAGE_PRESETS' three stage dicts
       — otherwise a stage write gets silently refused.

    3. No stage-preset key may reach a SIESTA field tagged
       ``workflow_group: "budget"`` or ``"system"``.  This is the
       bug class that bit the user on Au-BDT-Au (relax-steps was
       in the preset but is a budget cap; mixing-weight was in the
       preset but is a system characteristic)."""

    def test_stage_preset_keys_match_stage_tagged_fields(self):
        """For every ``p-<id>`` in STAGE_PRESETS, the matching
        SiestaConfig field MUST carry
        ``metadata['workflow_group'] == 'stage'``.  This catches
        the half-done case where a stage field gets dropped from
        the preset (or vice versa)."""
        import re
        import dataclasses
        from molbuilder.config.siesta import SiestaConfig

        viewer = (_LIB / ".." / "structure-optimization" / "viewer.js").resolve().read_text()
        # Extract the union of all stage-preset keys across the
        # three sub-presets (coarse / medium / tight).
        block = re.search(
            r"const\s+STAGE_PRESETS\s*=\s*\{(.+?)\n\s*\};",
            viewer, re.DOTALL,
        )
        assert block is not None, (
            "viewer.js: STAGE_PRESETS dict not found.  If the symbol "
            "was renamed, update this test to match.")
        keys = set(re.findall(r'"(p-[a-z0-9-]+)"', block.group(1)))
        assert keys, "no stage-preset keys extracted; regex too strict"

        # Map p-<id> back to the SiestaConfig field name.  IDs use
        # the dataclass field name with ``_`` → ``-``, OR an
        # explicit ``id_suffix`` metadata override.
        siesta_id_to_field = {}
        for f in dataclasses.fields(SiestaConfig):
            id_suffix = f.metadata.get("id_suffix",
                                       f.name.replace("_", "-"))
            siesta_id_to_field["p-" + id_suffix] = f
        unmatched = [k for k in keys if k not in siesta_id_to_field]
        assert not unmatched, (
            f"STAGE_PRESETS contains keys with no matching "
            f"SiestaConfig field: {unmatched}.  If the field was "
            f"renamed, update the preset.  If the field was "
            f"deleted, drop it from the preset too.")

        wrong_tag = []
        for k in keys:
            f = siesta_id_to_field[k]
            tag = f.metadata.get("workflow_group")
            if tag != "stage":
                wrong_tag.append((k, f.name, tag))
        assert not wrong_tag, (
            "STAGE_PRESETS writes to fields NOT tagged "
            "workflow_group='stage'.  This is the 2026-06-13 bug "
            "class — switching the stage selector silently mutates "
            "a budget / system / untagged field.  Either tag the "
            "field 'stage' in molbuilder/config/siesta.py or drop "
            f"it from STAGE_PRESETS: {wrong_tag}")

    def test_stage_preset_allowlist_matches_dict_keys(self):
        """The defensive allowlist (``_STAGE_PRESET_KEYS_SIESTA``)
        in viewer.js MUST match the union of stage-preset dict
        keys exactly.  An ID in the dict but missing from the
        allowlist is a silent no-op; an ID in the allowlist but
        not in any preset is dead config."""
        import re
        viewer = (_LIB / ".." / "structure-optimization" / "viewer.js").resolve().read_text()

        dict_block = re.search(
            r"const\s+STAGE_PRESETS\s*=\s*\{(.+?)\n\s*\};",
            viewer, re.DOTALL,
        )
        dict_keys = set(re.findall(r'"(p-[a-z0-9-]+)"',
                                   dict_block.group(1)))

        allowlist_block = re.search(
            r"_STAGE_PRESET_KEYS_SIESTA\s*=\s*new\s+Set\(\[(.+?)\]\)",
            viewer, re.DOTALL,
        )
        assert allowlist_block is not None, (
            "viewer.js: _STAGE_PRESET_KEYS_SIESTA Set literal not "
            "found; the defensive allowlist was the load-bearing "
            "part of the 2026-06-13 restructure.")
        allowlist = set(re.findall(r'"(p-[a-z0-9-]+)"',
                                   allowlist_block.group(1)))

        extra_in_dict     = dict_keys - allowlist
        extra_in_allowlist = allowlist - dict_keys
        assert not extra_in_dict, (
            f"STAGE_PRESETS keys missing from _STAGE_PRESET_KEYS_"
            f"SIESTA allowlist: {extra_in_dict}.  Those preset "
            f"writes will be silently rejected at runtime.")
        assert not extra_in_allowlist, (
            f"_STAGE_PRESET_KEYS_SIESTA allows IDs not used by any "
            f"preset: {extra_in_allowlist}.  Either remove from the "
            f"allowlist or add to the presets.")

    def test_only_canonical_workflow_group_values(self):
        """Smoke check across ALL FOUR engine configs (Siesta, PySCF,
        Transport, Spectra): no field ever ends up tagged with a
        contradictory or unknown workflow_group.  Only the three
        documented values (profile/stage/budget) are valid.  The
        old "system" value was renamed to "profile" on 2026-06-13
        to avoid the collision with the existing "System" schema
        section name + the OS/file-system + physics-system
        meanings; a stale "system" tag is a sign the rename was
        half-applied.

        Pre-2026-06-17 this test inspected only SiestaConfig; the
        other three engines drifted (the round-7 audit found 7+
        misalignments).  Now it walks all four."""
        import dataclasses
        from molbuilder.config.siesta    import SiestaConfig
        from molbuilder.config.pyscf     import PySCFConfig
        from molbuilder.config.transport import TransportConfig
        from molbuilder.config.spectra   import SpectraConfig
        valid = {"profile", "stage", "budget"}
        bad = []
        for cls in (SiestaConfig, PySCFConfig, TransportConfig, SpectraConfig):
            for f in dataclasses.fields(cls):
                tag = f.metadata.get("workflow_group")
                if tag is not None and tag not in valid:
                    bad.append((cls.__name__, f.name, tag))
        assert not bad, (
            f"Config fields tagged with non-canonical "
            f"workflow_group values: {bad}.  Allowed values: "
            f"{sorted(valid)}.")

    def test_every_form_shown_field_has_workflow_group(self):
        """Across ALL FOUR engine configs, every field that surfaces
        on the form (carries a ``section`` metadata key) MUST also
        carry ``workflow_group`` so the corresponding validator
        Issues route to the Profile / Stage / Budget cards per
        web-ui-coherence.md Rule 2.

        Pre-2026-06-17 17 fields drifted into the residual ("Other")
        panel because the metadata fan-out from sections to
        workflow groups was half-done.  Each new ``section``-tagged
        field landing without ``workflow_group`` re-introduces the
        same residual-panel drift; this test catches it the moment
        the field is added rather than waiting for a UI audit."""
        import dataclasses
        from molbuilder.config.siesta    import SiestaConfig
        from molbuilder.config.pyscf     import PySCFConfig
        from molbuilder.config.transport import TransportConfig
        from molbuilder.config.spectra   import SpectraConfig
        untagged = []
        for cls in (SiestaConfig, PySCFConfig, TransportConfig, SpectraConfig):
            for f in dataclasses.fields(cls):
                if "section" in f.metadata and "workflow_group" not in f.metadata:
                    untagged.append((cls.__name__, f.name,
                                     f.metadata.get("section")))
        assert not untagged, (
            f"Form-shown fields missing ``workflow_group`` "
            f"metadata: {untagged}.  Each lands in the 'Other' "
            f"residual panel and its validator Issues can't route "
            f"to the Profile / Stage / Budget card.  Tag the field "
            f"with one of: ``profile`` (system identity, set once), "
            f"``stage`` (convergence target that tightens), "
            f"``budget`` (resource cap).")

    def test_detection_chip_renderer_present(self):
        """The .workflow-detection-chip in the Profile + Budget
        workflow-group headers is populated by
        ``window.molbuilder.detectionChip.render(resp)``, called from
        _renderAutoDetectPanel (viewer.js) and the Transport tab's
        auto-analyze handler (transport/core.js).  Pin the wiring:

          * The shared helper lives in lib/detection-chip.js (one
            implementation, not a per-tab duplicate — web-ui-coherence
            Rule 1).
          * The text builder reads resp.n_atoms / resp.metals
            (data-driven, not hardcoded).
          * SIESTA viewer.js calls render() from inside the auto-
            detect panel renderer so the chip refreshes whenever the
            analyzer fires (including the silent auto-fire on load).
          * Transport core.js calls render() from its auto-analyze
            handler — extending coverage to the highest-value chip
            use case (Au-thiol junctions).
        """
        import re
        chip = (_LIB / "detection-chip.js").resolve().read_text()
        viewer = (_LIB / ".." / "structure-optimization" / "viewer.js").resolve().read_text()
        transport = (_LIB / "transport" / "core.js").resolve().read_text()

        # The lib defines BOTH helpers.
        assert "function buildText(resp)" in chip, (
            "lib/detection-chip.js lost buildText — the chip text "
            "builder is gone.")
        assert "function render(resp" in chip, (
            "lib/detection-chip.js lost render — the chip injector "
            "is gone.")

        # The chip text must reflect resp.n_atoms and resp.metals; a
        # refactor that hardcodes "232 atoms" instead of pulling from
        # the analyzer response is exactly the bug class this tests.
        builder = re.search(
            r"function\s+buildText\(resp\)(.+?)\n\s{4}\}",
            chip, re.DOTALL,
        )
        assert builder is not None, (
            "lib/detection-chip.js: buildText function shape changed; "
            "update this test.")
        builder_body = builder.group(1)
        assert "resp.n_atoms" in builder_body, (
            "Chip builder no longer reads resp.n_atoms — must be "
            "data-driven, not hardcoded.")
        assert "resp.metals" in builder_body, (
            "Chip builder no longer reads resp.metals — must be "
            "data-driven, not hardcoded.")

        # SIESTA viewer must delegate to the shared lib AND invoke
        # it from the auto-detect panel render path.
        assert "molbuilder.detectionChip" in viewer, (
            "viewer.js stopped delegating to lib/detection-chip.js — "
            "the chip helpers are at risk of drifting back into a "
            "per-tab duplicate.")
        panel_match = re.search(
            r"function\s+_renderAutoDetectPanel\b(.+?)\n\s{8}\}",
            viewer, re.DOTALL,
        )
        assert panel_match is not None, (
            "viewer.js: _renderAutoDetectPanel function shape changed; "
            "update this test.")
        panel_body = panel_match.group(1)
        assert "_renderWorkflowGroupChips" in panel_body, (
            "_renderAutoDetectPanel no longer calls "
            "_renderWorkflowGroupChips — the detection chip stops "
            "updating after a structure load.")

        # Transport tab must also delegate — this is the Au-junction
        # chip drift the 2026-06-13 audit caught.
        assert "molbuilder.detectionChip" in transport, (
            "transport/core.js does not call the shared chip helper "
            "— Au junctions silently skip the chip again "
            "(see web-ui-coherence.md Rule 1).")


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


# --------------------------------------------------------------------- #
#  Trajectory: SCF fingerprint augments noNewContent guard (Fix A)      #
# --------------------------------------------------------------------- #


class TestTrajectoryScfFingerprintGuard:
    """The noNewContent geometry-only guard prevents camera reset on
    same-data ticks (the 2026-06-12 fix), but pre-2026-06-17 it ALSO
    suppressed makePlots() when SCF iterations were appended to the
    in-flight CG step.  Fix A: add ``_scfFingerprint`` helper and
    include ``scfChanged`` in the makePlots trigger condition so
    SCF energy / dDmax / residual plots refresh mid-step.

    Pin both the helper definition AND its use in the noNewContent
    branch.  A future refactor that removes either re-introduces the
    plot-starvation bug class."""

    @pytest.fixture(scope="class")
    def core_body(self):
        return (_LIB / "trajectory" / "core.js").read_text()

    def test_scfFingerprint_helper_exists(self, core_body):
        assert re.search(
            r"function\s+_scfFingerprint\s*\(\s*data\s*\)",
            core_body,
        ), ("trajectory/core.js::_scfFingerprint helper is gone; "
            "the noNewContent branch can no longer detect new SCF "
            "iterations and the plot-starvation bug returns.")

    def test_scfFingerprint_reads_scf_history(self, core_body):
        """The fingerprint MUST inspect data.scf_history (not
        data.frames or data.runtime_info).  Without it the helper
        can't see the in-step iter growth that's the bug's signal."""
        m = re.search(
            r"function\s+_scfFingerprint\s*\([^)]*\)\s*\{(.+?)^\s*\}",
            core_body, re.DOTALL | re.MULTILINE,
        )
        assert m is not None, "fingerprint body shape changed"
        body = m.group(1)
        assert "scf_history" in body, (
            "trajectory/core.js::_scfFingerprint no longer reads "
            "data.scf_history; it cannot detect intra-step SCF "
            "growth and the plot-starvation bug returns.")

    def test_noNewContent_triggers_makePlots_on_scfChanged(self, core_body):
        """In the noNewContent branch the makePlots() call MUST fire
        on either runStateChanged OR scfChanged.  Drop scfChanged and
        the plot-starvation bug returns."""
        # Locate the noNewContent branch body.
        m = re.search(
            r"if\s*\(\s*noNewContent\s*\)\s*\{(.+?)return\s*;\s*\}",
            core_body, re.DOTALL,
        )
        assert m is not None, "noNewContent block shape changed"
        body = m.group(1)
        assert "scfChanged" in body, (
            "trajectory/core.js::applyNewData noNewContent branch "
            "no longer derives scfChanged; SCF plots will starve on "
            "same-geometry ticks during a live run.")
        assert re.search(
            r"if\s*\(\s*runStateChanged\s*\|\|\s*scfChanged\s*\)"
            r"\s*makePlots\s*\(\s*\)",
            body,
        ), ("trajectory/core.js noNewContent branch no longer calls "
            "makePlots when scfChanged is true; the SCF plots will "
            "stay frozen mid-CG-step (the original 2026-06-17 Fix "
            "A bug).")


# --------------------------------------------------------------------- #
#  Trajectory: POLL_MS cadence pin (Fix B)                              #
# --------------------------------------------------------------------- #


class TestTrajectoryPollCadence:
    """POLL_MS sets the live-watch tick rate.  History: 15 s -> 60 s
    on 2026-06-14 to silence SSL noise, dropped back to 15 s on
    2026-06-17 (Fix B) because the in-flight guard alone is
    sufficient -- and 60 s lagged live SCF runs whose iters complete
    in 13-24 s.  Pin 15 s so a future "raise it back to 60 to silence
    log noise" PR is forced to confront the trade-off."""

    @pytest.fixture(scope="class")
    def core_body(self):
        return (_LIB / "trajectory" / "core.js").read_text()

    def test_poll_ms_is_15_seconds(self, core_body):
        assert re.search(
            r"const\s+POLL_MS\s*=\s*15000\s*;",
            core_body,
        ), ("trajectory/core.js::POLL_MS is no longer 15000 ms.  "
            "Bumping it back to 60 s makes live SCF plots lag the "
            "on-disk state by 1-4 iters on a typical workload.  "
            "See 2026-06-17 Fix B for the cadence/in-flight-guard "
            "trade-off; the in-flight guard handles the original "
            "SSL noise without a cadence bump.")


# --------------------------------------------------------------------- #
#  Spectra: preserve user's mode selection across re-renders (Fix D)    #
# --------------------------------------------------------------------- #


class TestSpectraPreserveSelectedMode:
    """Pre-Fix-D every renderResults call unconditionally re-ran
    _pickDefaultMode, silently overwriting the user's mode selection
    every watchTick.  Fix D: only auto-pick when the prior selection
    is null or no longer present in the new modes list.

    Pin the priorStillValid gate so a future "simplify" refactor that
    drops it re-introduces the user-mode-clobbered bug."""

    @pytest.fixture(scope="class")
    def core_body(self):
        return (_LIB / "spectra" / "core.js").read_text()

    def test_renderResults_checks_priorStillValid(self, core_body):
        """The renderResults function MUST gate the _pickDefaultMode
        call on a priorStillValid check, not call it unconditionally."""
        # Locate the renderResults body's auto-pick region.
        assert "priorStillValid" in core_body, (
            "spectra/core.js::renderResults no longer derives "
            "priorStillValid; every render now clobbers the user's "
            "selectedMode (the 2026-06-17 Fix D bug returns).")
        # The actual gate: only call _pickDefaultMode when the prior
        # selection is invalid.
        assert re.search(
            r"if\s*\(\s*!priorStillValid\s*\)\s*\{[^}]*?_pickDefaultMode",
            core_body, re.DOTALL,
        ), ("spectra/core.js::renderResults no longer gates "
            "_pickDefaultMode on !priorStillValid -- the user's "
            "mode selection is overwritten on every watchTick.")

    def test_priorStillValid_checks_index_1based_membership(self, core_body):
        """priorStillValid must verify the prior selection's
        index_1based is still present in results.modes, not just
        that prior is non-null (a stale index from a previous file
        would pass that weaker test)."""
        m = re.search(
            r"const\s+priorStillValid\s*=(.+?);",
            core_body, re.DOTALL,
        )
        assert m is not None, (
            "spectra/core.js no longer defines priorStillValid as a "
            "single expression; the test can't pin its semantics.")
        body = m.group(1)
        assert "index_1based" in body and "prior" in body, (
            "spectra/core.js::priorStillValid no longer compares "
            "the prior index to results.modes[].index_1based; a "
            "stale index from a previous file may pass the gate "
            "and the user sees garbled ES data.")
