"""Regression-prevention tests for known SIESTA / PySCF science gaps.
"""

from __future__ import annotations


from molbuilder.config.siesta import SiestaConfig


# --------------------------------------------------------------------- #
#  Gap 5: PAO.EnergyShift default is too loose                          #
# --------------------------------------------------------------------- #


def test_gap_5_siesta_pao_energy_shift_default_is_tight():
    """The default PAO.EnergyShift should be 0.01 Ry or tighter.
    0.02 Ry produces under-converged PAO basis tails for most
    production work."""
    assert SiestaConfig().pao_energy_shift <= 0.01, (
        "SiestaConfig.pao_energy_shift default is too loose for "
        "production work; should be <= 0.01 Ry."
    )


# --------------------------------------------------------------------- #
#  Gap 8: a def2 basis on a heavy atom names its core potential         #
# --------------------------------------------------------------------- #


def test_a_def2_basis_on_gold_asks_for_its_core_potential(tmp_path,
                                                         monkeypatch):
    """Through `molbuilder describe` and `prep`: PySCF applies a core potential only
    when the script names it (PySCF 2.14 `gto/mole.py`, `build`), so gold on
    def2-SVP with none declared is every electron in a valence basis -- and
    the settings check says so, naming the basis's own (`config.ecp`).
    Declared as it says, the hint goes and the potential reaches `gto.M`.
    The calculation is the one maker's (`support.road.described`), its run
    card the road's; the declaration is the person's, in the template
    (`support.road.set_by_the_person`)."""
    from molbuilder.task import of as task_of
    from support.road import (described, jobset, molbuilder, road_tree,
                              set_by_the_person, stage_file)
    gold_hydride = {"elements": ["Au", "H"],
                    "positions": [[5.0, 5.0, 5.0], [5.0, 5.0, 6.524]],
                    "cell": [10.0, 10.0, 10.0]}
    bundle = described(
        road_tree(tmp_path, monkeypatch, gold_hydride, engine="pyscf"),
        "P/spectrum/A", name="A", calculation="vibration", engine="pyscf")
    stage = task_of(bundle).stages[0].name

    def prep(answers=None):
        r = jobset("prep", "task", "--group", "1", "--bundle", bundle,
                   input=answers)
        assert r.exit_code == 0, r.output
        return r.output, stage_file(bundle, stage, ".py").read_text()

    out, deck = prep("y\n\n")
    assert "[config.ecp]" in out and "ecp = 'def2-SVP'" in out, out
    assert "ecp        =" not in deck
    # DECLARED AS IT SAYS, and prepared anew by going back (`job-system.md`
    # § 5.0): the state saved before the prep, restored, then the template.
    from molbuilder.checkpoint import Repo
    before = Repo(str(bundle)).states()[0]
    back = molbuilder("checkpoint", "restore", before.id, "-p", bundle,
                      "--force")
    assert back.exit_code == 0, back.output
    set_by_the_person(bundle, {"ecp": "def2-SVP", "ecp_atoms": ["Au"]})
    out, deck = prep()
    assert "[config.ecp]" not in out, out
    assert "ecp        = {'Au': 'def2-SVP'}," in deck
