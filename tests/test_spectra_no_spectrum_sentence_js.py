"""L2 Node test: whether a spectrum is still coming, and what to say if not.

`web/spectra.md` § 2: with no strength in the file the chart draws the mode
positions, and one sentence under them says why there are no heights, naming
one of FOUR cases, read from the file's own record, the same on both engines
(`engines/vibration.md` § 5.11 § 8) --

* the run asked for none: `phase_ir` and `phase_raman` both *not requested*;
* a strength waits for you: a SIESTA result stating its charges whose
  infrared the load door could not give, told where its reason is (§ 3.2);
* they are still being computed: a run whose Raman or infrared phase is
  still to finish;
* the run asked and recorded none -- a SIESTA Raman range that kept no
  mode said as that, with or without infrared.

And § 7: the viewer stops watching only when every phase the description asked
for is done -- infrared's own flag, `phase_ir`, included -- while a file
written before that flag (`""`) is not waited on for it
(`engines/vibration.md` § 4.9).

API-level, on the writers' own shapes rather than a run: the case this exists
for -- a run asking for infrared alone, part-way through its dipole sweep,
which writes nothing until it ends -- lasts only while that sweep runs, so no
road test can hold the page there.

MUTATIONS THIS MUST FAIL AGAINST: the sentence's running test without
`phase_ir` (a run mid-sweep is told it recorded none); the done-rule without
`phase_ir` (the viewer stops watching mid-sweep); the done-rule waiting on
`""` (a finished file from before the flag is never done); the asked test
read from the config's switches (a SIESTA run with both off is told it
carries no intensities, not that it asked for none); the range that kept no
mode said only beside charges (with Compute IR off it is told its log says
why).
"""
from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from test_spectra_phase_indicator_js import _extract_fn_source

pytestmark = pytest.mark.module

#: a PySCF file's config carries every switch; SIESTA's names only the run
_PYSCF = {"compute_raman": False, "compute_ir": True,
          "es_mode_selection": "skip"}
_SIESTA = {"engine": "siesta", "calculation": "vibration", "stage": "freq"}


def _file(config, **phases):
    base = {"config": config, "modes": [{"index_1based": 1,
                                         "frequency_cm1": 1600.0,
                                         "raman_activity_a4_amu": None,
                                         "ir_intensity_km_mol": None}],
            "phase_frequencies": "complete", "phase_raman": "not requested",
            "phase_ir": "not requested", "phase_es": "not requested"}
    base.update(phases)
    return base


def _judge(files):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    src = "\n".join(_extract_fn_source(n) for n in
                    ("_ramanOptical", "_asked", "_ramanAsked", "_irAsked",
                     "_ramanNoneComputed", "_irFromCharges",
                     "_noSpectrumSentence",
                     "allPhasesComplete"))
    script = (f"{src}\nconst files = {json.dumps(files)};\n"
              "console.log(JSON.stringify(files.map(f => ({"
              "sentence: _noSpectrumSentence(f), "
              "done: allPhasesComplete(f)}))));")
    proc = subprocess.run([node, "--input-type=commonjs", "-e", script],
                          capture_output=True, text=True, timeout=10)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_the_sentence_names_the_case_and_the_viewer_waits_for_infrared():
    (siesta_off, charged, off, mid_sweep, before_sweep, none, old,
     kept_none) = _judge([
        _file(_SIESTA),
        # ...whose infrared source the load door serves (`infrared_source`,
        # `infrared.source_of`): the page reads that, never the file's
        # charges.
        _file(_SIESTA, phase_ir="complete", ir_route="charges",
              infrared_source="charges",
              infrared_said="the result states ['hirshfeld'] charges, not "
                            "'voronoi'"),
        _file({**_PYSCF, "compute_ir": False}),
        _file(_PYSCF, phase_ir="running"),
        _file(_PYSCF, phase_ir="empty"),
        _file(_PYSCF, phase_ir="complete"),
        _file(_PYSCF, phase_ir=""),
        _file(_SIESTA, phase_raman="complete", raman_route="none"),
    ])
    # the run asked for none, whichever engine: the same words
    for asked_none in (siesta_off, off):
        assert "not requested in this run" in asked_none["sentence"], \
            asked_none
    assert "the infrared asked of the run's charges could not be given " \
        "(why: in the bar above the chart)" in charged["sentence"], charged
    assert "no Raman was requested in this run" in charged["sentence"]
    for mid in (mid_sweep, before_sweep):
        assert "still being computed" in mid["sentence"], mid
        assert mid["done"] is False, mid
    assert "recorded none" in none["sentence"]
    # Compute IR off, Raman over a range that kept no mode: the range, and
    # the Methods paragraph, said -- not the log
    assert "no infrared was requested in this run" in kept_none["sentence"]
    assert "no mode lay in the range Raman was asked for" \
        in kept_none["sentence"], kept_none
    assert (siesta_off["done"], charged["done"], off["done"],
            none["done"]) == (True,) * 4
    # a file from before the flag carries "" -- no record, nothing to wait on
    assert old["done"] is True
