"""`openable_in` -- what a viewer opens in a folder read WITHOUT its run
(`model/parse.md` § 5.2): a folder no calculation claims, by dotted role,
each candidate vetted by the registry.

A run folder of OURS is answered one level up, by the run door
(`molbuilder.runs.folder_answer`), with its run's label from the description;
its cases run on the road (`tests/test_results_blueprint.py`).  *(This file
held `JobDirParser`'s tests until 2026-10-04 -- among them three on run
folders a test laid by hand, a deck and outputs written beside it, and one on
reading a label off a deck: the door moved up and the rule went, plan B11,
B14; `process/testing.md` § 6: a run is made on the road, or not at all.)*
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from molbuilder.parse.dirs import openable_in


def _spectrum_dir(tmp_path, *, say_calculation=True, shape="flat"):
    """A finished vibration run: the sidecar, and a molwatch log that is a stub.

    The stub is the real shape, not a simplification -- measured on a CO2
    spectrum run driven through the UI 2026-09-18: header, ONE
    `kind: initial_preview` block, `# concluded:` footer.  A spectrum has no
    geometry sequence to log, so the progress channel every run seeds says
    nothing about this one.

    ``shape`` puts the run where that shape puts it, and `task.json` where
    `project-layout.md` § 1.0 puts it -- in the PARENT, always.  Flat's run
    files sit in the calculation root, so the two land in one directory;
    hierarchical's sit two levels down, so they do not.  Returns the RUN
    directory, which is what a viewer is handed either way.
    """
    import json
    from molbuilder import calcdirs
    run = tmp_path
    if shape == "hierarchical":
        run = tmp_path / "01_freq" / "run-0"
        run.mkdir(parents=True)
        # STAMPED, because that is what prep produces (`project-layout.md`
        # § 1.4a, invariant 6b).  A hand-made tree with no records is a
        # different case with its own test below -- it is read ALONE.
        calcdirs.write(run.parent, role=calcdirs.CONTAINER, root=tmp_path)
        calcdirs.write(run, role=calcdirs.RUN, root=tmp_path)
    (run / "co2spec.spectra.json").write_text(json.dumps({
        "schema_version": 1, "engine": "pyscf", "engine_version": "x",
        "molbuilder_version": "x", "timestamp": "2026-09-18T00:00:00Z",
        "structure_hash": "sha256:0", "n_atoms_total": 3,
        "free_atom_idxs": [0, 1, 2], "frozen_atom_idxs": [],
        "equilibrium_scf_eh": -188.4, "equilibrium_mo_energies_eh": [-1.0],
        "equilibrium_homo_idx": 0, "modes": [], "selected_mode_idxs_1based": [],
        "config": {}, "methods_text": "", "bibliography_keys": [],
        "phase_frequencies": "complete", "phase_raman": "complete",
        "phase_es": "complete", "phase_relaxation": "complete",
    }), encoding="utf-8")
    _mw_log(run, "co2spec_01_freq.molwatch.log", concluded=True)
    if say_calculation:
        (tmp_path / "task.json").write_text(json.dumps({
            "schema": "molbuilder/task@1", "engine": {"name": "pyscf"},
            "shape": shape, "calculation": "vibration",
            "run": {"name": "co2spec", "id": "co2spec_CO2"},
            "structure": {"source": "co2spec.source.xyz",
                          "formula": "CO2", "atoms": 3},
            "stages": [{"name": "freq", "enabled": True, "overrides": {}}],
        }), encoding="utf-8")
    return run


def _mw_log(dirpath, name, *, concluded):
    body = ("# molwatch trajectory log v1\n# engine: pyscf\n# job: co2spec\n"
            "# units: energy=eV, force=eV/Ang, coords=Ang\n\n"
            "==== molwatch step 0 begin ====\nstep_index: 0\n"
            "kind: initial_preview\nn_atoms: 1\ncoordinates (Ang):\n"
            "   H       0.0 0.0 0.0\n==== molwatch step 0 end ====\n")
    if concluded:
        body += "\n# concluded: 2026-09-18T00:00:00\n"
    p = pathlib.Path(dirpath) / name
    p.write_text(body, encoding="utf-8")
    return p


def test_an_unmarked_directory_is_read_alone(tmp_path):
    """No record ⇒ the directory answers for itself and claims nothing above.

    `project-layout.md` § 1.4a: *absence narrows the answer; it does not
    refuse the directory*.  A tree written before that rule, or an attempt
    copied out of its calculation, still reads — its files, which one to open
    — but it does not get to say which calculation it belongs to, because
    nothing here knows.

    The scenario is the one that matters: a `task.json` DOES sit above this
    directory, and the directory must not adopt it.  Proximity is not
    membership; the record is.  Adopting it would let a description anywhere
    up the tree decide what an unrelated folder opens, which is exactly what
    the walk this replaced could do.

    MUTATION THIS MUST FAIL AGAINST: search upward for a `task.json` instead
    of reading `calcdir.json`.  Then this directory adopts `vibration` and
    opens the spectrum.
    """
    loose = tmp_path / "01_freq" / "run-0"        # the shape, none of the record
    loose.mkdir(parents=True)
    _spectrum_dir(loose)                          # its own task.json, then:
    (loose / "task.json").unlink()                # ...say it only at the TOP
    _spectrum_dir(tmp_path, say_calculation=True)

    got, attempts = openable_in(str(loose))

    assert got is not None and pathlib.Path(got).name.endswith(
        ".molwatch.log"), (
        f"adopted {tmp_path}'s task.json without a record and opened "
        f"{pathlib.Path(got).name if got else None!r}; the trail was:\n  "
        + "\n  ".join(attempts))


def test_an_optimization_still_opens_its_trajectory(tmp_path):
    """The other half of the same rule: a run whose calculation names no
    product of its own opens the progress channel, which for an optimization
    is the trajectory that grows per step.  `task.json` omits `calculation`
    for the default kind, so this is also the not-stated path."""
    _mw_log(tmp_path, "co2flat_01_coarse.molwatch.log", concluded=True)
    got, attempts = openable_in(str(tmp_path))
    assert got is not None and pathlib.Path(got).name.endswith(".molwatch.log"), (
        attempts)


def test_the_door_never_offers_a_file_the_registry_refuses(tmp_path):
    """*What is a run's output* and *what can a person open* are different
    questions with different owners (§ 5.5), and this is the second one.

    Measured 2026-09-18 on a finished CO2 spectrum run with its molwatch log
    removed -- the exact shape of `projects/BDT/spectrum/BDT-only`: the chain
    returned `<job>_<stage>.log`, PySCF's own verbose logger, and the
    caller's very next step was `detect()`, which refused it.  A directory
    holding nothing openable must answer None, so the refusal names the
    directory instead of a file that cannot be read.

    MUTATION THIS MUST FAIL AGAINST: drop the `_claimed` filter.
    """
    (tmp_path / "co2spec_01_freq.log").write_text("PySCF verbose log\n" * 20,
                                                  encoding="utf-8")
    (tmp_path / "co2spec_01_freq-run0.pyscf.log").write_text("Job complete in 1.0 s\n",
                                                             encoding="utf-8")
    (tmp_path / "co2spec_01_freq.py").write_text('JOB = "co2spec"\n',
                                                 encoding="utf-8")
    got, attempts = openable_in(str(tmp_path))
    assert got is None, (
        f"offered {pathlib.Path(got).name!r}, which no parser claims:\n  "
        + "\n  ".join(attempts))


