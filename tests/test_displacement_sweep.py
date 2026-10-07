"""The displacement sweep's question that needs no run (`engines/vibration.md`
§ 5.9): which calculations have no displacement to sweep at all.  Which mode
of one stage is which mode of another is asked of free water made on the
road, `tests/test_vibration_e2e.py`.
"""
from __future__ import annotations


def test_a_pyscf_vibration_has_no_displacement_to_sweep(tmp_path, monkeypatch):
    """`jobset init` a PySCF vibration, then `summarize run`: refused by
    name, before anything is read -- its second derivatives are analytic,
    so no stage takes a finite difference whose step could be swept
    (`engines/vibration.md` § 5.9).  Nothing runs, so no engine is needed."""
    from click.testing import CliRunner

    from molbuilder.jobset._cli import jobset_group
    from molbuilder.projects import PROJECTS_ROOT_ENV
    tree = tmp_path / "projects"
    (tree / "P" / "structure").mkdir(parents=True)
    (tree / "P" / "structure" / "w.xyz").write_text(
        "3\nwater\nO 0.0 0.0 0.0\nH 0.757 0.586 0.0\nH -0.757 0.586 0.0\n")
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tree))
    monkeypatch.chdir(tmp_path)
    run = CliRunner()
    r = run.invoke(jobset_group, [
        "init", "--structure", "P/structure/w.xyz", "--bundle",
        "P/frequency/V", "--engine", "pyscf", "--shape", "hierarchical",
        "--calculation", "vibration", "--name", "W"])
    assert r.exit_code == 0, r.output
    r = run.invoke(jobset_group, ["summarize", "run", "--bundle",
                                  str(tree / "P" / "frequency" / "V")])
    assert r.exit_code != 0 and "analytic" in r.output, r.output
    assert not list((tree / "P" / "frequency" / "V").glob("*.fc-sweep.json"))
