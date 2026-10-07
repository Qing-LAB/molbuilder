"""Regression-prevention tests for known SIESTA / PySCF science gaps.
"""

from __future__ import annotations

import re

import pytest

from molbuilder.siesta import SiestaConfig


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
#  Gap 7: installation guide pins the supported SIESTA release           #
# --------------------------------------------------------------------- #


def test_gap_7_installation_documents_siesta_version():
    """The installation guide names the SIESTA release the RECIPES pin — read
    from `recipes.py`, not retyped here.

    THE FAILURE THIS CATCHES.  A person following the guide builds a different
    SIESTA from the one molbuilder was measured against.  `.fdf` input format
    and TranSiesta output format are version-sensitive, and `recipes.py` pins a
    version deliberately so the source build and the packaged env stay
    identical -- "a paper-citable, reproducible default", in its own words.

    Contract: `ops/installation.md`, and `envs/recipes.py` as the pin's one
    home.  The version is taken from the packaged conda spec rather than from
    `_SIESTA_REF`, because that one honours `MOLBUILDER_SIESTA_TAG` and would
    make this test fail on a developer machine that has legitimately overridden
    it -- the packaged pin is a literal and is what the guide describes.
    """
    from pathlib import Path
    from molbuilder.envs.recipes import builtin_recipes

    pinned = set()
    for rec in builtin_recipes():
        for spec in rec.conda_specs:
            m = re.match(r"siesta=([0-9][0-9.]*)=", spec)
            if m:
                pinned.add(m.group(1))
    assert pinned, (
        "no recipe pins a `siesta=<version>=` conda spec any more -- repoint "
        "this test at wherever the pin moved, do not delete it")

    guide = (Path(__file__).parent.parent / "docs" / "ops" / "installation.md"
             ).read_text(encoding="utf-8")
    assert "siesta" in guide.lower()
    missing = sorted(v for v in pinned if v not in guide)
    assert not missing, (
        f"`envs/recipes.py` pins SIESTA {missing} and `ops/installation.md` "
        f"never names it, so the guide builds a different SIESTA from the one "
        f"molbuilder was measured against. Pinned: {sorted(pinned)}")


# --------------------------------------------------------------------- #
#  Gap 8: no ECP support for non-def2 bases                             #
# --------------------------------------------------------------------- #


def test_a_def2_basis_on_gold_asks_for_its_core_potential(tmp_path,
                                                         monkeypatch):
    """Through `jobset init` and `prep`: PySCF applies a core potential only
    when the script names it (PySCF 2.14 `gto/mole.py`, `build`), so gold on
    def2-SVP with none declared is every electron in a valence basis -- and
    the settings check says so, naming the basis's own (`config.ecp`).
    Declared as it says, the hint goes and the potential reaches `gto.M`."""
    import dataclasses
    import json
    from click.testing import CliRunner

    from molbuilder.jobset._cli import jobset_group
    from molbuilder.projects import PROJECTS_ROOT_ENV
    from molbuilder.task import read_task
    from molbuilder.template import _emit, find_template, read_template
    tree = tmp_path / "projects"
    (tree / "P" / "structure").mkdir(parents=True)
    (tree / "P" / "structure" / "a.xyz").write_text(
        "2\ngold hydride\nAu 0.0 0.0 0.0\nH 0.0 0.0 1.524\n")
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tree))
    monkeypatch.chdir(tmp_path)
    run = CliRunner()
    r = run.invoke(jobset_group, [
        "init", "--structure", "P/structure/a.xyz", "--bundle",
        "P/spectrum/A", "--engine", "pyscf", "--shape", "flat",
        "--calculation", "vibration", "--name", "A"])
    assert r.exit_code == 0, r.output
    bundle = tree / "P" / "spectrum" / "A"
    # THE RUN CARD STATES THE THREADS, as a described calculation does
    # (`architecture.md` § 5.2); how a shell enters an environment is the
    # machine record's (`configuration.md` § 5 M-1).
    task = json.loads((bundle / "task.json").read_text())
    task["execution"] = {"threads": 1}
    (bundle / "task.json").write_text(json.dumps(task, indent=2))

    def prep(answers=None):
        r = run.invoke(jobset_group, ["prep", "run", "freq", "--bundle",
                                      str(bundle)], input=answers)
        assert r.exit_code == 0, r.output
        return r.output, next(bundle.rglob("A*.py")).read_text()

    out, deck = prep("y\n\n")                 # the save prep offers, taken
    assert "[config.ecp]" in out and "ecp = 'def2-SVP'" in out, out
    assert "ecp        =" not in deck
    # DECLARED AS IT SAYS, and prepped anew by going back (`job-system.md`
    # § 5.0): the state saved before the prep, restored, then the template.
    from molbuilder.checkpoint import Repo
    from molbuilder.cli import cli
    before = Repo(str(bundle)).states()[0]
    back = run.invoke(cli, ["checkpoint", "restore", before.id, "-p",
                            str(bundle), "--force"])
    assert back.exit_code == 0, back.output
    tmpl = find_template(bundle, read_task(bundle / "task.json").label)
    tmpl.write_text(_emit(
        [dataclasses.replace(i, value={"ecp": "def2-SVP",
                                       "ecp_atoms": ["Au"]}[i.name])
         if i.name in ("ecp", "ecp_atoms") else i
         for i in read_template(tmpl.read_text()).items],
        engines=("pyscf",)))
    out, deck = prep()
    assert "[config.ecp]" not in out, out
    assert "ecp        = {'Au': 'def2-SVP'}," in deck
