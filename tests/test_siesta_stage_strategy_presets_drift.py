"""Drift gate: the two engines' stage-strategy preset tables agree.

Two sources carry the same data (#542 / C1.6):

  * ``molbuilder/config/siesta.py::SIESTA_STAGE_STRATEGY_PRESETS``
    (read by ``siesta/stages.py::default_siesta_stages``)
  * ``molbuilder/config/pyscf.py::STAGE_STRATEGY_PRESETS``
    (the PySCF-side driver)

If they drift, ``--stage-strategy publishable`` means a different
enable mask per engine — silently.  Update both in the same commit
when adding / renaming a preset.

(A THIRD source — the JS ``STAGE_STRATEGY_PRESETS`` behind the
form-schema stage-table dropdown — retired 2026-08-22 with that
widget; the five tests that regex-parsed it out of form-schema.js
went with it.  The tier presets a user picks TODAY come from
``/api/task-setup/presets``, which reads the SIESTA table through
``default_siesta_stages`` — one source, no copy to drift.)
"""
from __future__ import annotations


def test_siesta_and_pyscf_presets_match_value_for_value():
    """SIESTA + PySCF must agree name-for-name and mask-for-mask —
    the whole gate since the JS third source retired (module
    docstring)."""
    from molbuilder.config.siesta import SIESTA_STAGE_STRATEGY_PRESETS
    from molbuilder.config.pyscf import STAGE_STRATEGY_PRESETS as PYSCF
    assert set(SIESTA_STAGE_STRATEGY_PRESETS) == set(PYSCF), (
        f"preset name set drifted between SIESTA + PySCF:\n"
        f"  siesta: {sorted(SIESTA_STAGE_STRATEGY_PRESETS)}\n"
        f"  pyscf:  {sorted(PYSCF)}"
    )
    for name in SIESTA_STAGE_STRATEGY_PRESETS:
        s_val = SIESTA_STAGE_STRATEGY_PRESETS[name]
        p_val = PYSCF[name]
        assert s_val == p_val, (
            f"preset {name!r}: enable pattern drifted between "
            f"SIESTA + PySCF.\n  siesta: {s_val}\n  pyscf:  {p_val}"
        )
