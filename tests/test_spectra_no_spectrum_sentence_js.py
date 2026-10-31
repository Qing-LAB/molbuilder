"""L2 Node test: whether a spectrum is still coming, and what to say if not.

`web/spectra.md` § 2: with no strength in the file the chart draws the mode
positions, and one sentence under them says why there are no heights, naming
one of FOUR cases, read from the file's own record, the same on both engines
(`engines/vibration.md` § 5.11 § 8) --

* the run asked for none: its infrared not asked -- the load door's one
  reading (`infrared_asked`) -- and `phase_raman` *not requested*;
* a strength waits for you, or says why it is not there: a SIESTA result
  stating its charges whose infrared the load door could not give, told
  where its reason is (§ 3.2);
* they are still being computed: a run whose Raman or infrared phase is
  still to finish;
* the run asked and recorded none -- a SIESTA Raman range that kept no
  mode said as that, with or without infrared.

And § 7: the viewer stops watching only when every phase the description asked
for is done -- infrared's own flag, `phase_ir`, included -- while a file
written before that flag (`""`) is not waited on for it
(`engines/vibration.md` § 4.9).

Each row is the load door's answer (`POST /api/spectra/load`) for a result
as this release's writers write it -- its engine, config and phases -- and
both layers are judged: the door's one reading of whether the run asked for
its infrared and why a result carries none (`infrared.asked`,
`infrared_absent`, `engines/vibration.md` § 5.11 § 8), and the page that
reads it, never `phase_ir` -- the sentence and the run summary
(`_irRouteLabel`).

API-level, on the writers' own shapes rather than a run: the case this exists
for -- a run asking for infrared alone, part-way through its dipole sweep,
which writes nothing until it ends -- lasts only while that sweep runs, so no
road test can hold the page there.

MUTATIONS THIS MUST FAIL AGAINST: the sentence's running test without
`phase_ir` (a run mid-sweep is told it recorded none); the done-rule without
`phase_ir` (the viewer stops watching mid-sweep); the absent words
entered while the infrared is still running; the range that kept no
mode said only beside charges (with Compute IR off it is told its log says
why).
"""
from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from test_spectra_phase_indicator_js import _extract_fn_source

from molbuilder.spectra.results import SCHEMA_VERSION

pytestmark = pytest.mark.module

#: a PySCF file's config carries every switch; SIESTA's names only the run
_PYSCF = {"compute_raman": False, "compute_ir": True,
          "es_mode_selection": "skip"}
_SIESTA = {"engine": "siesta", "calculation": "vibration", "stage": "freq"}


def _charges(n_atoms: int) -> dict:
    """A force-constant run's charges as a SIESTA finish keeps them: the
    reference step, then each free atom along x, y, z, side -1 before +1
    (`results.FC_STEP_PATTERN`) -- Hirshfeld's alone."""
    from molbuilder.spectra.results import FC_STEP_PATTERN
    steps = [{"atom": None, "axis": None, "side": 0}] + [
        {"atom": a, "axis": axis, "side": side}
        for a in range(n_atoms) for axis, side in FC_STEP_PATTERN]
    return {"displacement_ang": 0.02, "steps": steps,
            "charges_e": {"hirshfeld": [[0.1, -0.1]] * len(steps)}}


def _answers(web_client, tree, rows):
    """Each row -- ``(engine, schema, config, fields, infrared ask)`` -- a
    result written as that release wrote it (no mode carrying a strength),
    and the load door's answer for it."""
    import dataclasses
    from molbuilder.sidecars.spectra import dump_spectra_json
    from tests.spectra.test_blueprint import _make_minimal_results
    base = _make_minimal_results()
    modes = [dataclasses.replace(m, raman_activity_a4_amu=None,
                                 ir_intensity_km_mol=None) for m in base.modes]
    out = []
    for k, (engine, schema, config, fields, ask) in enumerate(rows):
        fields = {"phase_raman": "not requested", "phase_ir": "not requested",
                  "phase_es": "not requested", **fields}
        if fields.pop("charges", False):
            fields["partial_charges"] = _charges(base.n_atoms_total)
        made = dataclasses.replace(base, engine=engine, schema_version=schema,
                                   config=dict(config), modes=modes, **fields)
        path = tree / f"r{k}.spectra.json"
        dump_spectra_json(made, path)
        r = web_client.post("/api/spectra/load", json={
            "path": str(path), **({"infrared": ask} if ask else {})})
        assert r.status_code == 200, r.get_json()
        out.append(r.get_json()["results"])
    return out


def _judge(files):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not available")
    src = "\n".join(_extract_fn_source(n) for n in
                    ("_ramanOptical", "_asked", "_ramanAsked", "_irAsked",
                     "_running",
                     "_ramanNoneComputed", "_irFromCharges",
                     "_noSpectrumSentence", "_irRouteLabel",
                     "allPhasesComplete"))
    script = (f"{src}\nconst files = {json.dumps(files)};\n"
              "console.log(JSON.stringify(files.map(f => ({"
              "sentence: _noSpectrumSentence(f), "
              "summary: _irRouteLabel(f), "
              "done: allPhasesComplete(f)}))));")
    proc = subprocess.run([node, "--input-type=commonjs", "-e", script],
                          capture_output=True, text=True, timeout=10)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_the_sentence_names_the_case_and_the_viewer_waits_for_infrared(
        web_client, tmp_path, monkeypatch):
    from conftest import _point_the_one_door_at
    tree = _point_the_one_door_at(tmp_path, monkeypatch.setenv)
    files = _answers(web_client, tree, [
        ("siesta", SCHEMA_VERSION, _SIESTA, {}, None),
        # ...whose charges state no Voronoi partition: the door's own
        # refusal of the view asked (`infrared_said`).
        ("siesta", SCHEMA_VERSION, _SIESTA, {"phase_ir": "complete",
                                 "ir_route": "charges", "charges": True},
         {"partition": "voronoi"}),
        ("pyscf", SCHEMA_VERSION, {**_PYSCF, "compute_ir": False}, {}, None),
        ("pyscf", SCHEMA_VERSION, _PYSCF, {"phase_ir": "running"}, None),
        ("pyscf", SCHEMA_VERSION, _PYSCF, {"phase_ir": "empty"}, None),
        ("siesta", SCHEMA_VERSION, _SIESTA, {"phase_raman": "complete",
                                 "raman_route": "none"}, None),
    ])
    # THE ONE READING, as the door answers it: whether the run asked, and
    # why a result carries none -- the page reads these.
    assert [f["infrared_asked"] for f in files] == [
        False, True, False, True, True, False], files
    assert "Compute IR was off" in files[2]["infrared_absent"]
    assert "not computed its infrared yet" in files[3]["infrared_absent"]
    (siesta_off, charged, off, mid_sweep, before_sweep,
     kept_none) = _judge(files)
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
    assert (siesta_off["done"], charged["done"], off["done"],
            kept_none["done"]) == (True,) * 4
