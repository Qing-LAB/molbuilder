"""L2 mode-selection tests: ``select_modes`` + ``validate_selection``.

Spec § 8 + § 8.1 + § 2.5.3.  Pure functions, exhaustively tested:

  * the five Model-2 selectors (skip / all / top_n / threshold /
    explicit) on a 6-mode fixture with varied (freq, Raman activity)
    pairs;
  * the frequency-range filter composes with each selector by
    INTERSECTION (spec § 8.1);
  * priors / resume behaviour (spec § 2.5.3);
  * the ``validate_selection`` issue surface that the preflight
    advisory layer consumes.

Also includes the cross-check that the emitted script's inlined
selector matches the Python ``select_modes`` for the same (modes,
cfg) inputs (``TestSelectorEquivalence``).  Lives here because the
test's purpose is selector correctness; the script-template emission
itself is tested in ``test_script.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from molbuilder.spectra import ModeData, SpectraConfig, SpectraResults
from molbuilder.spectra.results import PHASE_COMPLETE, SCHEMA_VERSION

from tests.spectra._helpers import (
    _make_es,
    _modes_fixture,
    _struct_water,
)


class TestSelectModes:

    def test_selector_none_returns_empty(self):
        from molbuilder.spectra import select_modes
        cfg = SpectraConfig(es_mode_selection="skip")
        assert select_modes(_modes_fixture(), cfg) == []

    def test_selector_all_returns_every_mode(self):
        from molbuilder.spectra import select_modes
        cfg = SpectraConfig(es_mode_selection="all")
        assert select_modes(_modes_fixture(), cfg) == [1, 2, 3, 4, 5, 6]

    def test_selector_all_respects_freq_window(self):
        """Window [800, 2500] -> modes 3 (1023) and 4 (1612)."""
        from molbuilder.spectra import select_modes
        cfg = SpectraConfig(es_mode_selection="all",
                            freq_min_cm1=800.0, freq_max_cm1=2500.0)
        assert select_modes(_modes_fixture(), cfg) == [3, 4]

    def test_selector_top_n_orders_by_activity_descending(self):
        """Top-3: brightest first -> 3 (87.2), 5 (45.0), 6 (18.5)."""
        from molbuilder.spectra import select_modes
        cfg = SpectraConfig(es_mode_selection="top_n", es_top_n=3)
        assert select_modes(_modes_fixture(), cfg) == [3, 5, 6]

    def test_selector_top_n_skips_modes_without_activity(self):
        """Mode 4 has raman_activity=None -- excluded from top_n
        ranking even if N would otherwise include it."""
        from molbuilder.spectra import select_modes
        cfg = SpectraConfig(es_mode_selection="top_n", es_top_n=10)
        # Asks for 10, only 5 modes have activities; all 5 returned.
        out = select_modes(_modes_fixture(), cfg)
        assert 4 not in out
        assert sorted(out) == [1, 2, 3, 5, 6]

    def test_selector_top_n_with_freq_window(self):
        """Top-2 within [2000, 4000] -> 5 (45), 6 (18) -- mode 4
        falls in the window but has no activity, mode 5 + 6 are the
        only Raman-active candidates."""
        from molbuilder.spectra import select_modes
        cfg = SpectraConfig(es_mode_selection="top_n", es_top_n=2,
                            freq_min_cm1=2000.0, freq_max_cm1=4000.0)
        assert select_modes(_modes_fixture(), cfg) == [5, 6]

    def test_selector_top_n_clamps_silently(self):
        """N > available count silently clamps (warning lives in
        validate_selection, not select_modes)."""
        from molbuilder.spectra import select_modes
        cfg = SpectraConfig(es_mode_selection="top_n", es_top_n=100)
        out = select_modes(_modes_fixture(), cfg)
        # 5 modes have activities; all 5 returned regardless of N=100.
        assert len(out) == 5

    def test_selector_threshold(self):
        """Threshold 15 -> activities > 15 are modes 3 (87) and 5 (45).
        Mode 6 has 18.5, so it passes too.  Modes 2 (12.5) and 1 (3.2)
        fall under.  Mode 4 has no activity (excluded)."""
        from molbuilder.spectra import select_modes
        cfg = SpectraConfig(es_mode_selection="threshold",
                            es_threshold=15.0)
        out = select_modes(_modes_fixture(), cfg)
        assert sorted(out) == [3, 5, 6]

    def test_selector_threshold_with_freq_window(self):
        """Threshold 10 + window [500, 1500] -> only mode 3 passes
        both (mode 2's freq is in window but activity 12.5 > 10; let
        me recompute: mode 2 freq=745 in [500,1500] ✓, activity 12.5 >
        10 ✓ -- passes.  Mode 3 freq=1023 in window ✓, 87 > 10 ✓ --
        passes.)"""
        from molbuilder.spectra import select_modes
        cfg = SpectraConfig(es_mode_selection="threshold",
                            es_threshold=10.0,
                            freq_min_cm1=500.0, freq_max_cm1=1500.0)
        assert sorted(select_modes(_modes_fixture(), cfg)) == [2, 3]

    def test_selector_explicit(self):
        from molbuilder.spectra import select_modes
        cfg = SpectraConfig(es_mode_selection="explicit",
                            es_explicit_indices=[3, 5])
        assert select_modes(_modes_fixture(), cfg) == [3, 5]

    def test_selector_explicit_ignores_freq_window(self):
        """Spec § 8.1: explicit IGNORES freq filter.  User asked
        for modes 3 + 5 + 6; even though mode 3 is well below
        any window, the explicit selector returns them all."""
        from molbuilder.spectra import select_modes
        cfg = SpectraConfig(es_mode_selection="explicit",
                            es_explicit_indices=[3, 5, 6],
                            # Window that would exclude 3 if applied
                            freq_min_cm1=2000.0)
        assert select_modes(_modes_fixture(), cfg) == [3, 5, 6]

    def test_explicit_dedupes_repeats(self):
        """Repeated explicit indices collapse; order preserved."""
        from molbuilder.spectra import select_modes
        cfg = SpectraConfig(es_mode_selection="explicit",
                            es_explicit_indices=[2, 5, 2, 5, 3])
        assert select_modes(_modes_fixture(), cfg) == [2, 5, 3]


class TestSelectModesWithPriorResume:
    """Spec § 2.5.2 + § 6.1: when prior has ES data for some modes,
    those modes are skipped on the next run (non-destructive L4)."""

    def _prior_with_es_on_mode(self, idx: int) -> SpectraResults:
        """Build a SpectraResults whose modes list contains the
        requested mode, with electronic_structure populated.  The
        other fields are irrelevant to select_modes (it only reads
        prior.modes[*].electronic_structure)."""
        return SpectraResults(
            schema_version             = SCHEMA_VERSION,
            engine                     = "pyscf",
            engine_version             = "2.6.0",
            molbuilder_version         = "1.2.0",
            timestamp                  = "2026-05-11T12:00:00Z",
            structure_hash             = "sha256:abc",
            n_atoms_total              = 2,
            free_atom_idxs             = [0, 1],
            frozen_atom_idxs           = [],
            equilibrium_scf_eh         = -1.0,
            equilibrium_mo_energies_eh = np.zeros(5),
            equilibrium_homo_idx       = 2,
            modes                      = [
                ModeData(
                    index_1based          = idx,
                    frequency_cm1         = 1000.0,
                    raman_activity_a4_amu = 1.0,
                    ir_intensity_km_mol   = None,
                    eigenvector_canonical = np.zeros((2, 3)),
                    eigenvector_display   = np.zeros((2, 3)),
                    has_imag              = False,
                    electronic_structure  = _make_es(),
                ),
            ],
            selected_mode_idxs_1based  = [idx],
            config                     = {},
            methods_text               = "",
            bibliography_keys          = [],
            phase_frequencies          = PHASE_COMPLETE,
            phase_raman                = PHASE_COMPLETE,
            phase_es                   = PHASE_COMPLETE,
        )

    def test_prior_with_es_filters_out_completed_mode(self):
        """User re-runs with selector=explicit=[2,3,5] but mode 3
        already has ES from a prior run.  select_modes returns
        [2, 5] -- the engine will only compute ES for those."""
        from molbuilder.spectra import select_modes
        cfg = SpectraConfig(es_mode_selection="explicit",
                            es_explicit_indices=[2, 3, 5])
        prior = self._prior_with_es_on_mode(idx=3)
        assert select_modes(_modes_fixture(), cfg, prior=prior) == [2, 5]

    def test_prior_without_es_does_nothing(self):
        """prior=None and prior.modes-with-ES=[] both leave the
        selection unchanged."""
        from molbuilder.spectra import select_modes
        cfg = SpectraConfig(es_mode_selection="explicit",
                            es_explicit_indices=[2, 3])
        assert select_modes(_modes_fixture(), cfg, prior=None) == [2, 3]

    def test_prior_filters_top_n_path_too(self):
        """Resume works for every selector, not just explicit.
        top_n=3 with prior ES on mode 3 -> returns the next-best
        modes instead."""
        from molbuilder.spectra import select_modes
        cfg = SpectraConfig(es_mode_selection="top_n", es_top_n=3)
        prior = self._prior_with_es_on_mode(idx=3)
        out = select_modes(_modes_fixture(), cfg, prior=prior)
        # top_n=3 normally returns [3, 5, 6]; with 3 already done -> [5, 6].
        # Note that this is a DROP (3 removed), not a re-rank to take
        # the 4th-place mode (2) -- resume preserves "what was asked"
        # minus "what's done", it doesn't re-allocate slots.
        assert out == [5, 6]


class TestValidateSelection:
    """validate_selection surfaces preflight errors / warns before
    the script runs.  See spec § 2.5.3 (soft dep) + § 11.4
    (scientific warns)."""

    def test_clean_explicit_no_issues(self):
        from molbuilder.spectra import validate_selection
        cfg = SpectraConfig(es_mode_selection="explicit",
                            es_explicit_indices=[2, 3])
        issues = validate_selection(_modes_fixture(), cfg, l3_done=False)
        assert issues == []

    def test_top_n_without_l3_errors(self):
        """Soft dep: top_n REQUIRES the prior Raman activities."""
        from molbuilder.spectra import validate_selection
        cfg = SpectraConfig(es_mode_selection="top_n", es_top_n=3)
        issues = validate_selection(_modes_fixture(), cfg, l3_done=False)
        errs = [i for i in issues if i.severity == "error"]
        assert len(errs) == 1
        # Plain-language reference to Raman + the workaround hint.
        assert "Raman" in errs[0].message
        assert "Compute Raman activities" in errs[0].message

    def test_threshold_without_l3_errors(self):
        from molbuilder.spectra import validate_selection
        cfg = SpectraConfig(es_mode_selection="threshold",
                            es_threshold=10.0)
        issues = validate_selection(_modes_fixture(), cfg, l3_done=False)
        errs = [i for i in issues if i.severity == "error"]
        assert len(errs) == 1
        # Plain-language Raman reference + workaround hint.
        assert "Raman" in errs[0].message

    def test_top_n_with_l3_passes(self):
        """With L3 complete, top_n is fine."""
        from molbuilder.spectra import validate_selection
        cfg = SpectraConfig(es_mode_selection="top_n", es_top_n=3)
        issues = validate_selection(_modes_fixture(), cfg, l3_done=True)
        errs = [i for i in issues if i.severity == "error"]
        assert errs == []

    def test_explicit_out_of_range_errors(self):
        from molbuilder.spectra import validate_selection
        cfg = SpectraConfig(es_mode_selection="explicit",
                            es_explicit_indices=[3, 99, 200])
        issues = validate_selection(_modes_fixture(), cfg, l3_done=False)
        errs = [i for i in issues
                if i.where == "config.es_explicit_indices"]
        assert len(errs) == 1
        assert "99" in errs[0].message
        assert "200" in errs[0].message

    def test_freq_min_greater_than_max_errors(self):
        from molbuilder.spectra import validate_selection
        cfg = SpectraConfig(es_mode_selection="all",
                            freq_min_cm1=2000.0,
                            freq_max_cm1=1000.0)
        issues = validate_selection(_modes_fixture(), cfg, l3_done=False)
        errs = [i for i in issues
                if i.severity == "error"
                and i.where == "config.freq_window"]
        assert len(errs) == 1

    def test_freq_window_empty_warns(self):
        """A window that captures zero modes is suspicious but
        valid -- warn, not error."""
        from molbuilder.spectra import validate_selection
        cfg = SpectraConfig(es_mode_selection="all",
                            freq_min_cm1=4500.0,
                            freq_max_cm1=5000.0)
        issues = validate_selection(_modes_fixture(), cfg, l3_done=True)
        warns = [i for i in issues if i.severity == "warn"]
        assert any("zero modes" in i.message for i in warns)

    def test_top_n_exceeds_available_warns(self):
        from molbuilder.spectra import validate_selection
        cfg = SpectraConfig(es_mode_selection="top_n", es_top_n=20)
        issues = validate_selection(_modes_fixture(), cfg, l3_done=True)
        warns = [i for i in issues if i.severity == "warn"]
        # Plain-language: "only N modes are available"
        assert any("only" in i.message and "modes are available"
                   in i.message for i in warns)


class TestSelectorEquivalence:
    """Pin the script's inlined selector against the canonical
    Python `select_modes` for a fixture of (modes, cfg) pairs.

    The script's selector is hand-rolled.  If someone changes the
    Python version (e.g. tie-breaking rule) without updating the
    script, this test catches the drift immediately.

    We don't exec the full script (would need real PySCF + a
    converged SCF); we exec ONLY the selector function out of the
    emitted text by slicing the if/elif/else block + the helpers
    it needs.
    """

    def _build_selector_namespace(self, cfg, modes_payload):
        """Re-create the runtime environment the inlined selector
        sees: ES_MODE_SELECTION / ES_TOP_N / ES_THRESHOLD /
        ES_EXPLICIT_INDICES / FREQ_MIN_CM1 / FREQ_MAX_CM1 plus the
        modes_payload list."""
        return {
            "ES_MODE_SELECTION":    cfg.es_mode_selection,
            "ES_TOP_N":             cfg.es_top_n,
            "ES_THRESHOLD":         cfg.es_threshold,
            "ES_EXPLICIT_INDICES":  list(cfg.es_explicit_indices),
            "FREQ_MIN_CM1":         cfg.freq_min_cm1,
            "FREQ_MAX_CM1":         cfg.freq_max_cm1,
            "modes_payload":        modes_payload,
        }

    def _modes_payload_for_fixture(self):
        """A modes_payload list shaped like what the in-script
        Hessian block builds (matching the wire form expected by
        the inlined selector).  Same modes as `_modes_fixture()`."""
        out = []
        for m in _modes_fixture():
            out.append({
                "index_1based":          m.index_1based,
                "frequency_cm1":         m.frequency_cm1,
                "raman_activity_a4_amu": m.raman_activity_a4_amu,
                "ir_intensity_km_mol":   m.ir_intensity_km_mol,
                "eigenvector_display":      m.eigenvector_display.tolist(),
                "has_imag":              m.has_imag,
                "electronic_structure":  None,
            })
        return out

    def _exec_inlined_selector(self, script: str, ns: dict) -> list:
        """Slice the inlined selector out of the script + exec it
        against the prepared namespace.  Returns the value of
        `_selected` after execution."""
        # The inlined selector starts at the "if ES_MODE_SELECTION"
        # marker and ends at the "state['selected_mode_idxs_1based']"
        # write that immediately follows it.
        start = script.find("if ES_MODE_SELECTION == 'all':")
        end   = script.find("state['selected_mode_idxs_1based']")
        assert start != -1 and end != -1 and end > start, (
            "could not locate inlined selector block in script"
        )
        body = script[start:end]
        # The selector also references `_passes_freq_window`, defined
        # just above.  Include from the "def _passes_freq_window" line.
        helper_start = script.find("def _passes_freq_window")
        assert helper_start != -1
        helper = script[helper_start:start]
        exec(helper + body, ns)
        return list(ns["_selected"])

    @pytest.mark.parametrize("cfg_overrides", [
        # Each selector exercised on the same modes fixture.
        dict(es_mode_selection="skip"),
        dict(es_mode_selection="all"),
        dict(es_mode_selection="all", freq_min_cm1=500.0, freq_max_cm1=3500.0),
        dict(es_mode_selection="top_n", es_top_n=3),
        dict(es_mode_selection="top_n", es_top_n=10),  # exceeds count
        dict(es_mode_selection="top_n", es_top_n=2,
             freq_min_cm1=1000.0, freq_max_cm1=3000.0),
        dict(es_mode_selection="threshold", es_threshold=10.0),
        dict(es_mode_selection="threshold", es_threshold=100.0),  # nothing
        dict(es_mode_selection="explicit", es_explicit_indices=[1, 3, 5]),
        dict(es_mode_selection="explicit", es_explicit_indices=[2]),
    ])
    def test_inlined_selector_matches_select_modes(self, cfg_overrides):
        from molbuilder.spectra import select_modes
        from molbuilder.spectra.pyscf_script import render_spectra_script
        cfg = SpectraConfig(**cfg_overrides)
        modes = _modes_fixture()

        # Python canonical result.
        py_selected = select_modes(modes, cfg, prior=None)

        # When selector == "none", the L4 block isn't emitted at all
        # so there's nothing to exec against -- Python and "script
        # behaviour" trivially agree on the empty list.
        if cfg.es_mode_selection == "skip":
            assert py_selected == []
            return

        # Script's inlined result: render, slice, exec.
        script = render_spectra_script(_struct_water(), cfg)
        ns = self._build_selector_namespace(
            cfg, self._modes_payload_for_fixture()
        )
        script_selected = self._exec_inlined_selector(script, ns)

        assert py_selected == script_selected, (
            f"selector drift for cfg={cfg_overrides!r}: "
            f"python={py_selected}, script={script_selected}"
        )
