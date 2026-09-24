"""The vibration kind on SIESTA, through the whole described road.

``jobset init --engine siesta --calculation vibration`` → ``prep run`` →
``launch run --mode direct`` → ``summarize run`` on this workstation: a
two-atom molecule with one atom held, the force-constant run nudging the
free atom only, and the modes derived on the host from ``<label>.FC``
through the one path both engines share.  The held atom is LAST in the
input, so the held-first sort really reorders the copy the deck is written
from and the answer has to come back through the recorded permutation.

Two roads, one per state of the person's one box (`engines/vibration.md`
§ 2.2): unticked, the ladder `init` writes relaxes first -- a `relax` stage
whose geometry the `freq` stage is written at (§ 5.2a) -- and the read-back
judges the reference-step forces by the template's own tolerance; ticked,
`freq` alone measures at the geometry as given.

Needs the ``molbuilder-siesta`` env + conda hook on this machine; skipped
cleanly anywhere else.  Wall cost ~40 s (a short Broyden relaxation plus
seven single points of H2, then seven more).
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from molbuilder import diagnostics

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _conda_hook() -> Path:
    binary = diagnostics.detect().conda_binary
    if not binary:
        return Path("/nonexistent/conda.sh")
    return Path(binary).parent.parent / "etc" / "profile.d" / "conda.sh"


CONDA_SH = _conda_hook()


def _env_bin() -> Path:
    """The ``molbuilder-siesta`` env's own ``bin``, through the product's
    resolver -- never a guess at the layout and never the host's PATH
    (`test_siesta_keyword_smoke.py` says why)."""
    from molbuilder.envs.install import _env_prefix
    caps = diagnostics.detect()
    if not caps.env_available("molbuilder-siesta"):
        return Path("/nonexistent")
    prefix = _env_prefix("molbuilder-siesta", caps.conda_binary)
    return Path(prefix) / "bin" if prefix else Path("/nonexistent")

pytestmark = pytest.mark.skipif(
    not (CONDA_SH.is_file()
         and diagnostics.detect().env_available("molbuilder-siesta")),
    reason="needs the molbuilder-siesta env + a detectable conda hook")


def _describe(tree, monkeypatch, positions):
    """The calculation the way a person makes it: the structure in the
    projects tree, the free atom FIRST and the held atom second (so the
    held-first sort really reorders), `init` on the SIESTA vibration kind."""
    from click.testing import CliRunner

    from molbuilder.jobset._cli import jobset_group
    from molbuilder.projects import PROJECTS_ROOT_ENV
    from molbuilder.structure import Structure
    from molbuilder.workingcopy_structure import StructureCodec

    (tree / "P" / "structure").mkdir(parents=True)
    (tree / "pseudopotential").mkdir()
    shutil.copy(FIXTURES / "psml" / "H.psml", tree / "pseudopotential" / "H.psml")
    s = Structure(elements=["H", "H"], positions=np.asarray(positions, float),
                  cell=np.diag([10.0, 10.0, 10.0]),
                  axis_kind=("isolated",) * 3)
    s.frozen_atoms = [1]
    StructureCodec().write(s, tree / "P" / "structure" / "h2.xyz")
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tree))
    monkeypatch.chdir(tree.parent)
    # The wrapper looks the engine up BY NAME after activating the env --
    # the road itself.  The suite's stub toolchain (conftest) sits ahead of
    # the env's bin so that no test reaches the host's engine; this test
    # wants the env's REAL engine on that same road, so the env's own bin
    # goes in front of the stubs -- the binary the product finds when the
    # person launches, and never /usr/local/bin's.
    env_bin = _env_bin()
    assert (env_bin / "siesta").is_file(), env_bin
    monkeypatch.setenv("PATH", f"{env_bin}{os.pathsep}{os.environ['PATH']}")

    r = CliRunner().invoke(jobset_group, [
        "init", "--structure", "P/structure/h2.xyz",
        "--bundle", "P/frequency/F", "--engine", "siesta",
        "--shape", "hierarchical", "--calculation", "vibration",
        "--name", "H2", "--psml-lib", "pseudopotential"])
    assert r.exit_code == 0, r.output
    bundle = tree / "P" / "frequency" / "F"
    (bundle / ".molbuilder.json").write_text(json.dumps(
        {"script_generation": {"activation": "conda activate",
                               "preamble": f"source {CONDA_SH}"}}))
    task = json.loads((bundle / "task.json").read_text())
    # Two atoms: the person's own door for the rank count, not the
    # target's sizing (prep says so when it sizes from the target).
    task["execution"] = {**task.get("execution", {}), "mpi_np": 2}
    (bundle / "task.json").write_text(json.dumps(task, indent=2))
    return bundle


def _jobset(*args):
    from click.testing import CliRunner

    from molbuilder.jobset._cli import jobset_group
    return CliRunner().invoke(jobset_group, list(args))


def _kw(deck, key):
    """The value tokens of the first line of ``deck`` that sets ``key``."""
    for line in deck.splitlines():
        parts = line.split()
        if parts and parts[0] == key:
            return parts[1:]
    return []


def _tick_already_relaxed(bundle):
    tmpl = bundle / "H2.template.toml"
    text = tmpl.read_text()
    head, _, tail = text.partition("[item.already_relaxed]")
    assert tail, text
    tail = tail.replace("value = false", "value = true", 1)
    tmpl.write_text(head + "[item.already_relaxed]" + tail)


def _common_assertions(d, *, converged_expected):
    assert d["engine"] == "siesta" and d["schema_version"] >= 6
    # The input order: the free atom is atom 0, the held one atom 1.
    assert d["free_atom_idxs"] == [0] and d["frozen_atom_idxs"] == [1]
    assert d["removed_motions"]["count"] == 2
    assert d["hessian_scope"] == "free" and d["n_atoms_in_hessian"] == 1
    assert d["equilibrium"]["mo_energies_eh"] is None
    assert d["modes"][0]["raman_activity_a4_amu"] is None
    assert d["thermo"]["regime"] == "vibrational-only"
    # the frequencies are this route's whole answer; the rest was never asked
    # of it (vibration.md § 4.9)
    assert d["phase_frequencies"] == "complete"
    assert (d["phase_raman"], d["phase_es"], d["phase_relaxation"]) \
        == ("not requested",) * 3
    assert "Head1997" in d["bibliography_keys"]
    # R5 on this route: the reference-step forces read back and judged by
    # the template's own tolerance -- the kind's recommended 0.01 eV/A
    rx = d["relaxation"]
    assert d["engine_metadata"]["reference_force_criterion_ev_ang"] == 0.01
    assert rx["converged"] is converged_expected
    # the mass-calibrated displacement a transport step displaces along:
    # one hydrogen near 3000 cm-1 swings about 0.075 A at its zero point
    m0 = d["modes"][0]
    assert 0.06 < m0["zero_point_amplitude_amu12_ang"] < 0.09


def test_unticked_the_ladder_relaxes_first_and_freq_measures_at_the_relaxed_bond(
        tmp_path, monkeypatch):
    """The box unticked (the default `init` writes): the ladder is `relax`
    then `freq`; `freq` is refused before `relax` has concluded; after it,
    `freq` is written at the relaxed geometry and the modes come out at
    the relaxed bond's frequency with the reference forces within the
    template's own tolerance (vibration.md § 2.2, § 5.2a, § 5.5)."""
    from molbuilder.transport.sort import read_permutation

    tree = tmp_path / "projects"
    # The experimental bond, 0.741 A -- NOT the stationary point at this
    # level of theory (fixtures/siesta_fc/README.md: 1.27 eV/A there, and
    # 3358 cm-1 where the relaxed bond gives 3022).
    bundle = _describe(tree, monkeypatch, [[5.0, 5.0, 5.741], [5.0, 5.0, 5.0]])
    task = json.loads((bundle / "task.json").read_text())
    assert [s["name"] for s in task["stages"]] == ["relax", "freq"]
    tmpl = (bundle / "H2.template.toml").read_text()
    # the kind's recommendation IS the template's value (template.md § 6.3a)
    block = tmpl[tmpl.index("[item.relax_force_tol]"):]
    block = block[:block.index("[item.", 1)]
    assert "value = 0.01" in block and "default = 0.01" in block, block

    # THE JOB SET'S OWN ORDER: freq before relax has concluded is refused,
    # naming the stage to run first.
    r = _jobset("prep", "run", "freq", "--bundle", str(bundle), "--target", "this")
    assert r.exit_code != 0 and "relax" in r.output and "concluded" in r.output, r.output

    r = _jobset("prep", "run", "relax", "--bundle", str(bundle), "--target", "this")
    assert r.exit_code == 0, r.output
    relax_deck = next((bundle / "01_relax").glob("*.fdf")).read_text()
    # the ordinary relaxation deck, from the same sorted copy: Broyden to
    # the kind's recommended 0.01 eV/A, the held atom (first after the
    # sort) constrained
    assert _kw(relax_deck, "MD.TypeOfRun") == ["Broyden"], relax_deck
    assert _kw(relax_deck, "MD.MaxForceTol")[:1] == ["0.01"], relax_deck
    assert "position 1" in relax_deck
    r = _jobset("launch", "run", "relax", "--bundle", str(bundle),
                "--mode", "direct", "--yes")
    assert r.exit_code == 0, r.output

    # WHAT THE RUN SAYS ABOUT THE GEOMETRY IT LEFT (model/parse.md § 5b.1):
    # the record the Results tab records onto an exported pair
    from molbuilder.parse.dirs.run_info import run_info_for_dir
    rec = run_info_for_dir(bundle / "01_relax" / "run-0")["relaxation"]
    assert rec["engine"] == "siesta" and rec["converged"] is True
    assert rec["force_tolerance_ev_ang"] == 0.01
    assert rec["max_force_free_ev_ang"] <= 0.01 and rec["held_atom_idxs"] == [0]

    r = _jobset("prep", "run", "freq", "--bundle", str(bundle), "--target", "this")
    assert r.exit_code == 0, r.output
    perm = read_permutation(bundle)
    assert perm.key == "held-first" and perm.sorted_to_original == (1, 0)
    fc_deck = next((bundle / "02_freq").glob("*.fdf")).read_text()
    assert "MD.TypeOfRun      FC" in fc_deck and "FC.First          2" in fc_deck \
        and "FC.Last           2" in fc_deck and "position 1" in fc_deck
    r = _jobset("launch", "run", "freq", "--bundle", str(bundle),
                "--mode", "direct", "--yes")
    assert r.exit_code == 0, r.output
    attempt = bundle / "02_freq" / "run-0"
    assert (attempt / "H2.FC").is_file(), "the force-constant run must leave H2.FC"

    r = _jobset("summarize", "run", "freq", "--bundle", str(bundle))
    assert r.exit_code == 0, r.output
    d = json.loads((attempt / "H2.spectra.json").read_text())
    _common_assertions(d, converged_expected=True)
    assert d["relaxation"]["already_relaxed"] is False
    freqs = [m["frequency_cm1"] for m in d["modes"]]
    # the relaxed bond's frequency (fixtures/siesta_fc/README.md: 3022-3024
    # across the measured tolerances), not the unrelaxed 3358
    assert len(freqs) == 1 and 2950.0 < freqs[0] < 3100.0, freqs
    # the geometry in the file is the one the force constants were taken
    # at -- the RELAXED bond, read from the run's own reference step
    pos = np.asarray(d["equilibrium"]["positions_ang"])
    bond = abs(pos[0, 2] - pos[1, 2])
    assert 0.76 < bond < 0.79, bond


def test_ticked_freq_alone_measures_at_the_geometry_as_given(tmp_path,
                                                             monkeypatch):
    """The box ticked: no relaxation, the force constants at the geometry as
    given, the reference forces judged by the template's tolerance.  And
    the contradiction refused first: unticked with no relax stage in the
    ladder is neither state, and prep names both ways out (§ 5.2a)."""
    tree = tmp_path / "projects"
    # the relaxed bond (fixtures/siesta_fc/README.md)
    bundle = _describe(tree, monkeypatch, [[5.0, 5.0, 5.77446], [5.0, 5.0, 5.0]])
    task = json.loads((bundle / "task.json").read_text())
    task["stages"] = [s for s in task["stages"] if s["name"] == "freq"]
    (bundle / "task.json").write_text(json.dumps(task, indent=2))

    r = _jobset("prep", "run", "freq", "--bundle", str(bundle), "--target", "this")
    assert r.exit_code != 0 and "already_relaxed" in r.output \
        and "relax" in r.output, r.output
    _tick_already_relaxed(bundle)
    r = _jobset("prep", "run", "freq", "--bundle", str(bundle), "--target", "this")
    assert r.exit_code == 0, r.output
    r = _jobset("launch", "run", "freq", "--bundle", str(bundle),
                "--mode", "direct", "--yes")
    assert r.exit_code == 0, r.output
    attempt = bundle / "01_freq" / "run-0"
    r = _jobset("summarize", "run", "freq", "--bundle", str(bundle))
    assert r.exit_code == 0, r.output
    d = json.loads((attempt / "H2.spectra.json").read_text())
    _common_assertions(d, converged_expected=True)
    assert d["relaxation"]["already_relaxed"] is True
    freqs = [m["frequency_cm1"] for m in d["modes"]]
    assert len(freqs) == 1 and 2950.0 < freqs[0] < 3100.0, freqs
    from molbuilder.constants import HARTREE_BOHR_EV_ANGSTROM_ASE
    assert d["relaxation"]["max_force_eh_bohr"] < 0.01 / HARTREE_BOHR_EV_ANGSTROM_ASE
