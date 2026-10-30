"""Regression-prevention tests for known SIESTA / PySCF science gaps.
"""

from __future__ import annotations


from molbuilder.siesta import SiestaConfig
from molbuilder.jobset._cli import describe_cmd


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
    Declared as it says, the hint goes and the potential reaches `gto.M`."""
    import dataclasses
    from click.testing import CliRunner

    from molbuilder.jobset._cli import jobset_group
    from molbuilder.projects import PROJECTS_ROOT_ENV
    from molbuilder.task import of as task_of
    from molbuilder.template import _emit, read_template
    from molbuilder.template_file import of as template_of
    tree = tmp_path / "projects"
    (tree / "P" / "structure").mkdir(parents=True)
    (tree / "P" / "structure" / "a.xyz").write_text(
        "2\ngold hydride\nAu 0.0 0.0 0.0\nH 0.0 0.0 1.524\n")
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tree))
    monkeypatch.chdir(tmp_path)
    run = CliRunner()
    r = run.invoke(describe_cmd, [
        "--structure", "P/structure/a.xyz", "--bundle",
        "P/spectrum/A", "--engine", "pyscf",
        "--calculation", "vibration", "--name", "A"])
    assert r.exit_code == 0, r.output
    bundle = tree / "P" / "spectrum" / "A"
    # THE RUN CARD STATES THE THREADS, as a described calculation does
    # (`architecture.md` § 5.2); how a shell enters an environment is the
    # machine record's (`configuration.md` § 5 M-1).
    from support.road import read_request, stage_file, write_request
    request = read_request(bundle)
    request["execution"] = {"threads": 1}
    write_request(bundle, request)
    stage = task_of(bundle).stages[0].name

    def prep(answers=None):
        r = run.invoke(jobset_group, ["prep", "task", "--group", "1", "--bundle",
                                      str(bundle)], input=answers)
        assert r.exit_code == 0, r.output
        return r.output, stage_file(bundle, stage, ".py").read_text()

    out, deck = prep("y\n\n")
    assert "[config.ecp]" in out and "ecp = 'def2-SVP'" in out, out
    assert "ecp        =" not in deck
    # DECLARED AS IT SAYS, and prepared anew by going back (`job-system.md`
    # § 5.0): the state saved before the prep, restored, then the template.
    from molbuilder.checkpoint import Repo
    from molbuilder.cli import cli
    before = Repo(str(bundle)).states()[0]
    back = run.invoke(cli, ["checkpoint", "restore", before.id, "-p",
                            str(bundle), "--force"])
    assert back.exit_code == 0, back.output
    tmpl = template_of(bundle, task_of(bundle).label)
    tmpl.path.write_text(_emit(
        [dataclasses.replace(i, value={"ecp": "def2-SVP",
                                       "ecp_atoms": ["Au"]}[i.name])
         if i.name in ("ecp", "ecp_atoms") else i
         for i in read_template(tmpl.text).items],
        engines=("pyscf",)))
    out, deck = prep()
    assert "[config.ecp]" not in out, out
    assert "ecp        = {'Au': 'def2-SVP'}," in deck
