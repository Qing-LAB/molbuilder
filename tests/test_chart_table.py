"""The chart table (`model/datafiles.md` § 5b, § 9.7): what each chart on the
Results page draws, as a data file holds it."""
from __future__ import annotations


def test_every_residual_a_run_converges_on_is_a_curve_and_a_threshold_of_its_chart():
    """Every residual an engine declares a criterion for -- the SCF residuals
    chart draws each, and each one's tolerance line (`web/trajectory.md`
    § 3.0) -- is a curve and a threshold the residuals chart may export,
    and the table loads whole (`chartdata.table`).  The silent failure: a
    residual an engine adds, drawn on the chart and refused as a person
    presses its Export."""
    from molbuilder.chartdata import table
    from molbuilder.parse.engines import molwatch_grammar, siesta_grammar
    chart = table().charts["scf-residuals"]
    drawn = {r["key"] for g in (siesta_grammar, molwatch_grammar)
             for r in g.SCF_RESIDUALS if r["criterion"]}
    assert drawn <= set(chart.tags), drawn - set(chart.tags)
    assert drawn <= set(chart.marks), drawn - set(chart.marks)
