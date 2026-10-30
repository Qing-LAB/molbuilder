"""The displacement sweep's question that needs no run (`engines/vibration.md`
§ 5.9): which calculations have no displacement to sweep at all.  Which mode
of one stage is which mode of another is asked of free water made on the
road, `tests/test_vibration_e2e.py`.
"""
from __future__ import annotations
from molbuilder.jobset._cli import describe_cmd


def test_a_pyscf_vibration_has_no_displacement_to_sweep(tmp_path, monkeypatch):
    """`molbuilder describe` a PySCF vibration, then `summarize task`: refused
    before anything is read or written -- the pair names nothing to add up
    in the kind table (`engines/vibration.md` § 5.9: the sweep is a SIESTA
    vibration's), and the refusal lists the pairs that do, from the table.
    Nothing runs, so no engine is needed.

    THE SILENT FAILURE: the refusal telling the person a PySCF vibration
    is among the pairs that add something up.  MUTATION THIS MUST FAIL
    AGAINST: the pair given a `sum` again in the kind table."""
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
    r = run.invoke(describe_cmd, [
        "--structure", "P/structure/w.xyz", "--bundle",
        "P/frequency/V", "--engine", "pyscf", "--calculation", "vibration", "--name", "W"])
    assert r.exit_code == 0, r.output
    r = run.invoke(jobset_group, ["summarize", "task", "--bundle",
                                  str(tree / "P" / "frequency" / "V")])
    from molbuilder.engine_table import engine
    from molbuilder.kind_table import kind
    asked = f"{kind('vibration').with_article} on {engine('pyscf').words}"
    assert r.exit_code != 0, r.output
    # Named once: as the pair asked about, never among those that add up.
    assert r.output.count(asked) == 1, r.output
    assert not list((tree / "P" / "frequency" / "V").glob("*.fc-sweep.json"))
