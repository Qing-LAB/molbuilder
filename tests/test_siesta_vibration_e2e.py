"""The vibration kind on SIESTA, through the whole described road.

``jobset init --engine siesta --calculation vibration`` → ``prep run`` →
``launch run --mode direct`` → ``summarize run`` on this workstation: a
two-atom molecule with one atom held, the force-constant run nudging the
free atom only, and the modes derived on the host from ``<label>.FC``
through the one path both engines share.  The held atom is LAST in the
input, so the held-first sort really reorders the copy the deck is written
from and the answer has to come back through the recorded permutation.

Needs the ``molbuilder-siesta`` env + conda hook on this machine; skipped
cleanly anywhere else.  Wall cost ~15 s (seven single points of H2).
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


def test_h2_with_one_atom_held_runs_the_force_constant_road(tmp_path,
                                                            monkeypatch):
    from click.testing import CliRunner

    from molbuilder.jobset._cli import jobset_group
    from molbuilder.projects import PROJECTS_ROOT_ENV
    from molbuilder.structure import Structure
    from molbuilder.transport.sort import read_permutation
    from molbuilder.workingcopy_structure import StructureCodec

    tree = tmp_path / "projects"
    (tree / "P" / "structure").mkdir(parents=True)
    (tree / "pseudopotential").mkdir()
    shutil.copy(FIXTURES / "psml" / "H.psml", tree / "pseudopotential" / "H.psml")
    # Free atom first, held atom second: the input order is NOT the order
    # the deck needs, so the sort and its record are exercised for real.
    s = Structure(elements=["H", "H"],
                  # the relaxed bond (fixtures/siesta_fc/README.md): a
                  # harmonic analysis is asked of a stationary point, and
                  # the road now judges that (vibration.md § 5.5)
                  positions=np.array([[5.0, 5.0, 5.77446], [5.0, 5.0, 5.0]]),
                  cell=np.diag([10.0, 10.0, 10.0]),
                  axis_kind=("isolated",) * 3)
    s.frozen_atoms = [1]
    StructureCodec().write(s, tree / "P" / "structure" / "h2.xyz")
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tree))
    monkeypatch.chdir(tmp_path)
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
    stage = task["stages"][0]["name"]
    # Two atoms: the person's own door for the rank count, not the
    # target's sizing (prep says so when it sizes from the target).
    task["execution"] = {**task.get("execution", {}), "mpi_np": 2}
    (bundle / "task.json").write_text(json.dumps(task, indent=2))

    # THE GATE ASKS FOR THE ASSERTION (vibration.md § 2.2): unmade, prep
    # refuses and names the two ways out; made -- in the template, the
    # person's own file -- the road goes on.
    r = CliRunner().invoke(jobset_group, ["prep", "run", stage,
                                          "--bundle", str(bundle),
                                          "--target", "this"])
    assert r.exit_code != 0 and "already_relaxed" in r.output, r.output
    tmpl = bundle / "H2.template.toml"
    text = tmpl.read_text()
    head, _, tail = text.partition("[item.already_relaxed]")
    assert tail, text
    tail = tail.replace("value = false", "value = true", 1)
    tmpl.write_text(head + "[item.already_relaxed]" + tail)
    r = CliRunner().invoke(jobset_group, ["prep", "run", stage,
                                          "--bundle", str(bundle),
                                          "--target", "this"])
    assert r.exit_code == 0, r.output
    # The record, beside the calculation, under the key that made the copy.
    perm = read_permutation(bundle)
    assert perm.key == "held-first" and perm.sorted_to_original == (1, 0)
    deck = next((bundle / f"01_{stage}").glob("*.fdf"))
    text = deck.read_text()
    assert "MD.TypeOfRun      FC" in text and "FC.First          2" in text \
        and "FC.Last           2" in text and "position 1" in text

    r = CliRunner().invoke(jobset_group, ["launch", "run", stage,
                                          "--bundle", str(bundle),
                                          "--mode", "direct", "--yes"])
    assert r.exit_code == 0, r.output
    attempt = bundle / f"01_{stage}" / "run-0"
    assert (attempt / "H2.FC").is_file(), "the force-constant run must leave H2.FC"

    r = CliRunner().invoke(jobset_group, ["summarize", "run", stage,
                                          "--bundle", str(bundle)])
    assert r.exit_code == 0, r.output
    d = json.loads((attempt / "H2.spectra.json").read_text())
    assert d["engine"] == "siesta" and d["schema_version"] >= 6
    # The input order: the free atom is atom 0, the held one atom 1.
    assert d["free_atom_idxs"] == [0] and d["frozen_atom_idxs"] == [1]
    freqs = [m["frequency_cm1"] for m in d["modes"]]
    assert len(freqs) == 1 and 2500.0 < freqs[0] < 4500.0, freqs
    assert d["removed_motions"]["count"] == 2
    assert d["hessian_scope"] == "free" and d["n_atoms_in_hessian"] == 1
    assert d["equilibrium"]["mo_energies_eh"] is None
    assert d["modes"][0]["raman_activity_a4_amu"] is None
    assert d["thermo"]["regime"] == "vibrational-only"
    assert "Head1997" in d["bibliography_keys"]
    # R5 on this route: the reference-step forces read back and judged
    rx = d["relaxation"]
    assert rx["already_relaxed"] is True and rx["converged"] is True
    assert rx["max_force_eh_bohr"] is not None and rx["max_force_eh_bohr"] < 0.02 / 51.42
    assert "within the relaxation criterion" in rx["warning"]
    assert d["engine_metadata"]["fc_asymmetry_max_ev_ang2"] >= 0.0
    # the mass-calibrated displacement a transport step displaces along
    m0 = d["modes"][0]
    assert m0["zero_point_amplitude_amu12_ang"] > 0
    assert np.asarray(m0["zero_point_displacement_ang"]).shape == (1, 3)
