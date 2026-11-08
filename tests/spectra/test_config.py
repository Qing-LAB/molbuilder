"""The spectra config surface -- defaults and validation metadata.

The config a spectra calculation is described by is `PySCFConfig`,
seen through the vibration deck's view (`_spectra_cfg`).

Pinned here: its defaults and its validation-facing metadata.
"""

from __future__ import annotations


import pytest

from tests.spectra._helpers import _spectra_cfg



# --------------------------------------------------------------------- #
#  the spectra config surface -- defaults, metadata                    #
# --------------------------------------------------------------------- #


class TestSpectraDefaults:
    """The dataclass instantiates with all-defaults and the values
    match the v1 spec defaults so a user's first-pass run is
    cheap (es_mode_selection=skip) and uses production-defensible
    method/basis choices (B3LYP/def2-SVP/D3BJ, grid 4)."""

    def test_a_spectra_calculations_science_defaults(self):
        """What a vibration run gets when the user chooses nothing:
        B3LYP / def2-SVP / D3BJ, Kohn-Sham DFT, density-fitted -- and the
        charge and spin left blank, for the electronic state to work out
        from the structure (`science/chemistry-correctness.md` § 2a).

        Pinned HERE and nowhere else.  The catalogue-agreement gate
        mirrors help / range / unit / choices / label / engine_key
        between the class and the catalogue -- **not defaults** -- so
        without this a silent change to the functional or the basis
        would reach a user's spectrum with nothing failing.
        """
        cfg = _spectra_cfg()
        assert cfg.job_name   == "pyscf_relax"
        assert cfg.method     == "DFT"
        assert (cfg.net_charge, cfg.spin_treatment,
                cfg.unpaired_electrons) == (None, None, None)
        assert cfg.functional == "B3LYP"
        assert cfg.basis      == "def2-SVP"
        assert cfg.dispersion == "d3bj"
        assert cfg.density_fit is True

    def test_atom_freeze_list_is_empty_by_default(self):
        """Frozen atoms are INDICES and come from the structure's region
        store -- empty until the user freezes something."""
        cfg = _spectra_cfg()
        assert cfg.frozen_indices == []

    def test_es_off_by_default(self):
        """First-pass run should be cheap -- spectrum only, no
        displaced SCFs.  User opts in to all / explicit after
        seeing the spectrum."""
        cfg = _spectra_cfg()
        assert cfg.es_mode_selection == "skip"

    def test_infrared_on_and_raman_off_by_default(self):
        """The strengths' defaults, one on both engines
        (`engines/vibration.md` § 3.1): infrared on -- the cheap one --
        Raman off -- the expensive one."""
        cfg = _spectra_cfg()
        assert (cfg.compute_ir, cfg.compute_raman) == (True, False)

    def test_displacement_amplitude_production_default(self):
        """0.02 Å keeps ES probes inside the linear-response regime
        (ΔE_orbital ∝ displacement) and well below the threshold
        where Mills 1972 §2.4 anharmonic mixing becomes meaningful."""
        cfg = _spectra_cfg()
        assert cfg.displacement_amplitude_ang == pytest.approx(0.02)


class TestSpectraFieldMetadata:
    """The metadata here is the validation vocabulary
    (range / validate / choices / pattern) -- the form keys live in
    the catalogue with the vibration items."""


    def test_choices_are_enforced_where_a_user_supplies_one(
            self, tmp_path, monkeypatch):
        """A value outside a field's declared ``choices`` is refused --
        checked at the DESCRIPTION layer, which is where a user actually
        supplies one: a stage's override, written through Task setup's Save
        on a PySCF vibration `molbuilder describe` made.

        The guard that protects a real user reads `task.json`'s
        stage overrides against the catalogue
        (`validation/task.py::preflight`), and it refuses at ERROR --
        the Save answering 400 with its finding.
        """
        from molbuilder.task import of as task_of
        from support.road import (WATER, described,
                                  description_with_overrides, read_request,
                                  road_tree, save_both)

        bundle = described(road_tree(tmp_path, monkeypatch, WATER,
                                     engine="pyscf"),
                           name="W", calculation="vibration", engine="pyscf")
        stage = task_of(bundle).stages[0].name
        r = save_both(bundle, description_with_overrides(
            bundle, {stage: {"dispersion": "d5"}}), read_request(bundle))
        assert r.status_code == 400, r.get_json()
        bad = [f for f in r.get_json()["findings"]
               if f["severity"] == "error"
               and "dispersion" in (f["message"] or "")]
        assert bad, r.get_json()["findings"]
        assert "not one of" in bad[0]["message"]
        # All declared choices remain accepted.
        for v in ("skip", "all", "explicit"):
            _spectra_cfg(es_mode_selection=v)


class TestFreqRangeFilter:
    """The freq_min_cm1 / freq_max_cm1 fields default to no filter."""

    def test_default_no_filter(self):
        cfg = _spectra_cfg()
        assert cfg.freq_min_cm1 is None
        assert cfg.freq_max_cm1 is None
