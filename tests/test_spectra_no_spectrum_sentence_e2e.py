"""Whether a spectrum is still coming, and what to say if not -- on the files
real runs wrote.

`web/spectra.md` § 2: with no strength in the file the chart draws the mode
positions, and one sentence under them says why there are no heights, naming
one of its cases, read from the file's own record, the same on both engines
(`engines/vibration.md` § 5.11 § 8) --

* the run asked for none: its infrared not asked -- the load door's one
  reading (`infrared_asked`) -- and `phase_raman` *not requested*;
* they are still being computed: a run whose Raman or infrared phase is
  still to finish;
* the run asked and recorded none -- a SIESTA Raman range that kept no
  mode said as that.

And § 7: the viewer stops watching only when every phase the description asked
for is done -- infrared's own flag, `phase_ir`, included (`engines/vibration.md`
§ 4.9).

Each row is the load door's answer (`POST /api/spectra/load`) for a result a run
wrote, and both layers are judged: the door's one reading of whether the run
asked for its infrared and why a result carries none (`infrared.asked`,
`infrared_absent`), and the page that reads it, never `phase_ir` -- the sentence
and the run summary (`_irRouteLabel`).  A run part-way through its dipole sweep,
which writes nothing until it ends, is the version of its file kept while the
sweep ran (`support.real_runs._result_kept_as_it_grows`).

What no run writes goes unseen here: the sentence for an infrared asked of the
charges that could not be given (`infrared_said`) -- a run's charges state both
partitions, and the door's other refusal needs a file from before schema 12.

MUTATIONS THIS MUST FAIL AGAINST: the sentence's running test without
`phase_ir` (a run mid-sweep is told it recorded none); the done-rule without
`phase_ir` (the viewer stops watching mid-sweep); the absent words
entered while the infrared is still running; the range that kept no
mode said only beside charges (with Compute IR off it is told its log says
why).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from support.pages import spectra_served
from tests._node_esm import lifted, run_node

from _road import conda_hook, env_available
from molbuilder.runfiles import ROLE_SPECTRA
from molbuilder.runs import on_label_at

pytestmark = [
    pytest.mark.engine,
    pytest.mark.skipif(
        not (conda_hook().is_file() and env_available("molbuilder-pySCF")
             and env_available("molbuilder-siesta")),
        reason="needs the molbuilder-pySCF and molbuilder-siesta envs + a "
               "detectable conda hook"),
]


def _h2_with_compute_ir_off(tmp_path_factory, name: str, **settings):
    """The road's H2, its first atom held, as a SIESTA vibration stated relaxed
    with Compute IR off -- one force-constant call of seven single points, the
    smallest a SIESTA result is -- ``settings`` the person's beside: its result
    on the label."""
    from support.real_runs import CO2_SETTINGS, RealRun, on_the_road, taken
    from support.road import H2, described, set_by_the_person, structure_saved
    with on_the_road(tmp_path_factory, name) as tree:
        bundle = described(tree, f"P/freq/{name}",
                           structure=structure_saved(tree, H2), name="H2",
                           calculation="vibration",
                           execution={"mpi_np": 1, "omp_threads": 1})
        set_by_the_person(bundle, {**CO2_SETTINGS, "already_relaxed": True,
                                   "compute_ir": False, **settings})
        run = RealRun(tree=tree, bundle=bundle)
        taken(run, "prep freq", "prep", "task", "--group", "1",
              "--target", "this")
        taken(run, "launch freq", "launch", "task", "--stage", "freq",
              "--mode", "direct", "--foreground", "--yes")
    return on_label_at(run.attempt("freq"), ROLE_SPECTRA)


@pytest.fixture(scope="module")
def siesta_asked_nothing(tmp_path_factory):
    return _h2_with_compute_ir_off(tmp_path_factory, "asked-nothing")


@pytest.fixture(scope="module")
def siesta_range_kept_no_mode(tmp_path_factory):
    """Compute Raman on, over a range above H2's one mode: the phase ends with
    no optical call (`engines/vibration.md` § 5.11 § 8)."""
    return _h2_with_compute_ir_off(tmp_path_factory, "kept-none",
                                   compute_raman=True, raman_modes="range",
                                   raman_freq_min_cm1=9000.0,
                                   optical_mesh=[1, 1, 1])


#: The spectra viewer the sentence and the done-rule are lifted from.
_CORE = (Path(__file__).resolve().parents[1]
         / "molbuilder/web/static/lib/spectra/core.js")


def _judge(files):
    """What the viewer's own sentence, summary and done-rule, lifted from
    the shipped core (`_node_esm.lifted`), say of each of ``files`` -- run
    in Node (`_node_esm.run_node`)."""
    src = "\n".join(lifted(_CORE, f"    function {n}(", "\n    function ")
                    for n in
                    ("_ramanOptical", "_asked", "_ramanAsked", "_irAsked",
                     "_running",
                     "_ramanNoneComputed", "_irFromCharges",
                     "_noSpectrumSentence", "_irRouteLabel",
                     "allPhasesComplete"))
    return run_node([], f"{src}\nconst files = {json.dumps(files)};\n"
                        "console.log(JSON.stringify(files.map(f => ({"
                        "sentence: _noSpectrumSentence(f), "
                        "summary: _irRouteLabel(f), "
                        "done: allPhasesComplete(f)}))));")


def test_the_sentence_names_the_case_and_the_viewer_waits_for_infrared(
        siesta_asked_nothing, siesta_range_kept_no_mode, real_water_linear,
        real_water_held, web_client, monkeypatch):
    # ...THE ONE PYSCF RUN ASKING NONE; AND A RUN ASKING ITS INFRARED ALONE:
    # mid-sweep, and before its first SCF.
    off = on_label_at(real_water_linear.attempt("freq"), ROLE_SPECTRA)
    mid_sweep = real_water_held.version_where(
        lambda d: d["phase_ir"] == "running")
    before_sweep = real_water_held.version_where(
        lambda d: d["phase_relaxation"] == "running")
    files = [spectra_served(web_client, monkeypatch, path) for path in (
        siesta_asked_nothing, off, mid_sweep, before_sweep,
        siesta_range_kept_no_mode)]
    # THE ONE READING, as the door answers it: whether the run asked, and
    # why a result carries none -- the page reads these.
    assert [f["infrared_asked"] for f in files] == [
        False, False, True, True, False], files
    assert "Compute IR was off" in files[1]["infrared_absent"]
    assert "not computed its infrared yet" in files[2]["infrared_absent"]
    (siesta_off, off, mid_sweep, before_sweep, kept_none) = _judge(files)
    # the run asked for none, whichever engine: the same words
    for asked_none in (siesta_off, off):
        assert "not requested in this run" in asked_none["sentence"], \
            asked_none
    for mid in (mid_sweep, before_sweep):
        assert "still being computed" in mid["sentence"], mid
        assert mid["done"] is False, mid
    for asked_none in (siesta_off, off):
        assert asked_none["summary"] == "not requested", asked_none
    # mid-sweep the summary says what the sentence says (one reading)
    for mid in (mid_sweep, before_sweep):
        assert mid["summary"] == "being computed", mid
    # Compute IR off, Raman over a range that kept no mode: the range, and
    # the Methods paragraph, said -- not the log
    assert "no infrared was requested in this run" in kept_none["sentence"]
    assert "no mode lay in the range Raman was asked for" \
        in kept_none["sentence"], kept_none
    assert (siesta_off["done"], off["done"], kept_none["done"]) == (True,) * 3
