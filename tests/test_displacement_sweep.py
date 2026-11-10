"""The displacement sweep's question that needs no run (`engines/vibration.md`
§ 5.9): which calculations have no displacement to sweep at all.  Which mode
of one stage is which mode of another is asked of free water made on the
road, `tests/test_vibration_e2e.py`.
"""
from __future__ import annotations


def test_a_pyscf_vibration_has_no_displacement_to_sweep(tmp_path, monkeypatch):
    """`molbuilder describe` a PySCF vibration, then `summarize task`: refused
    before anything is read or written -- the pair names nothing to add up
    in the kind table (`engines/vibration.md` § 5.9: the sweep is a SIESTA
    vibration's), and the refusal lists the pairs that do, from the table.
    Nothing runs, so no engine is needed.  The calculation is the one
    maker's (`support.road.described`), of the road's water.

    THE SILENT FAILURE: the refusal telling the person a PySCF vibration
    is among the pairs that add something up.  MUTATION THIS MUST FAIL
    AGAINST: the pair given a `sum` again in the kind table."""
    from support.road import WATER, described, jobset, road_tree
    bundle = described(road_tree(tmp_path, monkeypatch, WATER, engine="pyscf"),
                       "P/spectrum/V", name="W", calculation="vibration",
                       engine="pyscf")
    r = jobset("summarize", "task", "--bundle", bundle)
    from molbuilder.engine_table import engine
    from molbuilder.kind_table import kind
    asked = f"{kind('vibration').with_article} on {engine('pyscf').words}"
    assert r.exit_code != 0, r.output
    # Named once: as the pair asked about, never among those that add up.
    assert r.output.count(asked) == 1, r.output
    assert not list(bundle.glob("*.fc-sweep.json"))
