"""The vibration calculation's E2E — the spectra-migration plan's P1 bar.

Water, on this workstation, through the WHOLE framework loop:
``describe --calculation vibration`` → ``prep run freq`` →
``submit --mode direct`` → a schema-5 ``.spectra.json`` in the attempt
directory, every phase complete, loadable through the Results door.  Two
runs: the default (Raman) and the DECOUPLED IR-only run the 2026-08-20
ruling asked for — whose intensities are held to water's literature
windows at B3LYP/def2-SVP, which is what resolves the IR prefactor's
NOT-VALIDATED flag at the band level (pattern + magnitudes; an external
cross-code digit match can harden it later, and the item's help says so).

Needs the ``molbuilder-pySCF`` env + conda hook on this machine; skipped
cleanly anywhere else.  Wall cost ~1 min total (water is 17 s a run).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import subprocess

import pytest

def _conda_hook() -> Path:
    """conda's activation hook, from the manager the PACKAGE detects.

    Was ``Path.home() / "miniconda3/etc/profile.d/conda.sh"`` until
    2026-09-12 -- one developer's install layout, hard-coded.  A test pinned
    to that runs on exactly one machine and SILENTLY SKIPS everywhere else,
    which is worse than no test: it reads as coverage from the outside.
    ``diagnostics.detect()`` is how the product itself finds the manager
    (conda / mamba / micromamba, any prefix), so this follows it instead.
    """
    from molbuilder import diagnostics
    binary = diagnostics.detect().conda_binary
    if not binary:
        return Path("/nonexistent/conda.sh")
    # <root>/condabin/conda or <root>/bin/conda -> <root>/etc/profile.d/
    return Path(binary).parent.parent / "etc" / "profile.d" / "conda.sh"


def _env_present(name: str) -> bool:
    """Through ``Capabilities.env_available`` -- the door that knows the
    manager's own env list, rather than guessing a directory under $HOME."""
    from molbuilder import diagnostics
    return diagnostics.detect().env_available(name)


CONDA_SH = _conda_hook()

pytestmark = pytest.mark.skipif(
    not (CONDA_SH.is_file() and _env_present("molbuilder-pySCF")),
    reason="needs the molbuilder-pySCF env + a detectable conda hook")

WATER = "3\nwater\nO 0.0 0.0 0.119\nH 0.0 0.757 -0.477\nH 0.0 -0.757 -0.477\n"


def _describe(tmp_path, monkeypatch):
    """Create the calculation the way a user does: both citations read from
    the projects root (`job-contracts.md` § 2.5b)."""
    from click.testing import CliRunner

    from molbuilder.jobset._cli import jobset_group
    from molbuilder.projects import PROJECTS_ROOT_ENV
    tree = tmp_path / "projects"
    (tree / "P" / "structure").mkdir(parents=True)
    (tree / "P" / "structure" / "w.xyz").write_text(WATER)
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tree))
    monkeypatch.chdir(tmp_path)
    r = CliRunner().invoke(jobset_group, [
        "init", "--structure", "P/structure/w.xyz",
        "--bundle", "P/frequency/V",
        "--engine", "pyscf", "--shape", "hierarchical",
        "--calculation", "vibration", "--name", "W"])
    assert r.exit_code == 0, r.output
    bundle = tree / "P" / "frequency" / "V"
    (bundle / ".molbuilder.json").write_text(json.dumps(
        {"script_generation": {"activation": "conda activate",
                               "preamble": f"source {CONDA_SH}"}}))
    return bundle


def _prep_and_run(bundle):
    from click.testing import CliRunner

    from molbuilder.jobset._cli import jobset_group
    r = CliRunner().invoke(jobset_group,
                           ["prep", "run", "freq", "--bundle", str(bundle)])
    assert r.exit_code == 0, r.output
    r = CliRunner().invoke(jobset_group,
                           ["launch", "run", "freq", "--bundle", str(bundle),
                            "--mode", "direct", "--yes"])
    assert r.exit_code == 0, r.output
    art = bundle / "01_freq" / "run-0" / "W.spectra.json"
    assert art.is_file(), "the artifact must land IN THE ATTEMPT DIR"
    return json.loads(art.read_text())


def test_water_runs_the_whole_loop_and_the_viewer_can_load_it(
        tmp_path, monkeypatch):
    bundle = _describe(tmp_path, monkeypatch)
    d = _prep_and_run(bundle)

    assert d["schema_version"] == 5
    for k in ("phase_relaxation", "phase_frequencies",
              "phase_raman", "phase_es"):
        assert d[k] == "complete", (k, d[k])

    # D3, live: the relaxation ran, was tracked, and converged.
    rel = d["relaxation"]
    assert rel["enabled"] and rel["converged"] and rel["n_steps"] >= 1
    assert rel["max_force_eh_a"] < 1e-3

    # Water's three modes at B3LYP/def2-SVP (harmonic): bend ~1639,
    # stretches ~3791/3886.  Windows generous enough for BLAS-level
    # variation, tight enough that a broken Hessian cannot pass.
    freqs = sorted(m["frequency_cm1"] for m in d["modes"])
    assert len(freqs) == 3
    assert 1550 < freqs[0] < 1750
    assert 3600 < freqs[1] < 3950
    assert 3700 < freqs[2] < 4050

    # D2 + the plots' data: the thermo block, full RRHO for a free
    # molecule, ZPE ~13.3 kcal/mol (water's known value), the T grid.
    th = d["thermo"]
    assert th["regime"] == "rrho"
    assert 0.019 < th["zpe_eh"] < 0.023
    assert len(th["grid"]["temperatures_K"]) == 30
    assert th["g_eh"] < d["equilibrium"]["scf_energy_eh"] + 0.05

    # The Results door loads it -- the tab's half of the bar.
    from molbuilder.sidecars.spectra import parse_spectra_json
    r = parse_spectra_json(str(bundle / "01_freq" / "run-0"
                               / "W.spectra.json"))
    assert r.engine == "pyscf" and len(r.modes) == 3


def test_ir_alone_runs_decoupled_and_lands_in_waters_windows(
        tmp_path, monkeypatch):
    """The 2026-08-20 ruling executed: IR without Raman (a dipole read per
    displacement), and the intensities in water's literature windows at
    B3LYP/def2-SVP -- bend strongest (~55 km/mol), asym stretch middle
    (~27), sym stretch weakest (~5).  This is the band-level validation
    that retires the prefactor's NOT-VALIDATED flag; the windows are wide
    enough for method/BLAS wiggle and narrow enough that a wrong
    prefactor (off by any structural factor) cannot pass."""
    bundle = _describe(tmp_path, monkeypatch)
    tpl = bundle / "W.template.toml"
    t = tpl.read_text()
    i = t.index("[item.compute_ir]"); j = t.index("[item.", i + 1)
    t = t[:i] + t[i:j].replace("value = false", "value = true", 1) + t[j:]
    i = t.index("[item.compute_raman]"); j = t.index("[item.", i + 1)
    t = t[:i] + t[i:j].replace("value = true", "value = false", 1) + t[j:]
    tpl.write_text(t)

    d = _prep_and_run(bundle)
    modes = sorted(d["modes"], key=lambda m: m["frequency_cm1"])
    ir = [m["ir_intensity_km_mol"] for m in modes]
    assert all(v is not None for v in ir), "IR-only must fill every mode"
    bend, sym, asym = ir
    assert 30.0 < bend < 90.0, f"bend {bend} outside water's window"
    assert 0.5 < sym < 15.0, f"sym stretch {sym} outside water's window"
    assert 10.0 < asym < 60.0, f"asym stretch {asym} outside water's window"
    assert bend > asym > sym, "water's IR ordering is bend > asym > sym"
    # WHICH ROUTE produced dmu/dR is recorded, because the two cost
    # wildly different amounts and a reader looking at a slow run
    # deserves to know which one they got.  Without this assertion the
    # test passes identically whether the analytic path fired or
    # silently never did -- and "silently never did" is a 6N-SCF
    # regression that looks like success.
    assert d["ir_route"] in ("analytic", "finite-difference"), (
        f"IR ran but recorded no route: {d.get('ir_route')!r}")

    # Raman was NOT requested: its phase closes as complete-with-nothing.
    assert d["phase_raman"] == "complete"
    assert all(m["raman_activity_a4_amu"] in (None, 0.0) for m in modes)


def test_water_in_water_runs_the_solvated_chain_end_to_end(
        tmp_path, monkeypatch):
    """Category 2's live bar (integration plan, 2026-08-21): PCM water,
    the WHOLE chain under one solvated Hamiltonian -- relaxation,
    Hessian, IR and Raman -- because pyscf 2.13's PCM carries every
    analytic derivative (probed; the polarizability response includes
    the solvent).  The physics pins are deliberately loose: PCM shifts
    water's bands by tens of cm^-1, so the gas windows widen; what is
    being proven is the CONSISTENT solvated run completing with real
    numbers, plus the solvated deck actually differing from gas
    (the eps line is in the deck; the energy is the solvated one)."""
    bundle = _describe(tmp_path, monkeypatch)
    tpl = bundle / "W.template.toml"
    t = tpl.read_text()
    # An optional-empty item emits NO value line at all -- the edit
    # INSERTS one (an escape-hatch assert here let a silent no-op
    # through on the first landing; now the write is verified).
    anchor = '[item.solvent]\n'
    assert t.count(anchor) == 1
    t = t.replace(anchor, anchor + 'value = "water"\n', 1)
    assert 'value = "water"' in t
    # BOTH lanes on: the solvated proof covers IR and Raman under one
    # Hamiltonian (compute_ir defaults false; the first landing of
    # this test asserted IR without enabling it).
    i = t.index("[item.compute_ir]"); j = t.index("[item.", i + 1)
    t = t[:i] + t[i:j].replace("value = false", "value = true", 1) + t[j:]
    assert t[t.index("[item.compute_ir]"):t.index("[item.", t.index("[item.compute_ir]") + 1)].count("value = true") == 1
    tpl.write_text(t)

    d = _prep_and_run(bundle)
    # The deck is born in its STAGE directory (L1, roadmap 7.10, layout
    # repair 2026-08-24) -- it sat at the bundle root until then.  Bound
    # AFTER the prep for the same reason: the file does not exist before.
    deck = bundle / "01_freq" / "W_01_freq.py"
    text = deck.read_text()
    assert "mf = mf.PCM()" in text or "_mb_apply_solvent" in text
    assert "78.3553" in text, "the water dielectric never reached the deck"

    assert d["phase_relaxation"] == "complete"
    assert d["phase_frequencies"] == "complete"
    modes = sorted(d["modes"], key=lambda m: m["frequency_cm1"])
    freqs = [m["frequency_cm1"] for m in modes]
    assert 1500.0 < freqs[0] < 1800.0, f"solvated bend {freqs[0]}"
    assert 3400.0 < freqs[1] < 4100.0 and 3400.0 < freqs[2] < 4100.0, freqs
    assert all(m["raman_activity_a4_amu"] is not None for m in modes)
    assert all(m["ir_intensity_km_mol"] is not None for m in modes)
    assert d["thermo"]["grid"]["temperatures_K"], "thermo grid missing"


def test_asking_for_ir_does_not_move_the_frequencies(tmp_path):
    """The analytic route must return the Hessian the NO-IR path returns.

    ``pyscf.prop.infrared`` computes a Hessian on its way to dmu/dR, and
    two of its choices differ from ``Hessian.kernel()``:

      * ``proc_hessian_`` is ``hess_elec + hess_nuc`` and STOPS.
        ``kernel()`` adds ``get_dispersion()`` when the functional has a
        dispersion correction -- measured on B3LYP-D3BJ as 7.2e-4
        Hartree/Bohr^2, a 3.7 cm^-1 shift on every frequency.
      * upstream hardcodes ``hess_cls`` to the NON-DF Hessian class, so
        on a density-fitted SCF -- molbuilder's default -- it builds a
        non-DF Hessian of a DF density (0.11 cm^-1).

    Both are invisible on a functional without dispersion, which is how
    the first version of this shipped.  So the grid below crosses
    dispersion WITH density fitting: ticking the IR box is a request for
    an extra column, never for different frequencies.

    Runs inside ``molbuilder-pySCF`` like everything else in this file --
    pyscf lives only there.
    """
    script = tmp_path / "hess_identity.py"
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parent.parent)!r})\n"
        "import numpy as np\n"
        "from pyscf import gto, dft\n"
        "from molbuilder.pyscf.vibration_emitters import dipole_derivatives\n"
        "mol = gto.Mole(atom='N 0 0 0; H 0.8 0 0; H 0 1 0; H 0 0 1.2',\n"
        "               basis='6-31G', verbose=0).build()\n"
        "for xc in ('PBE0', 'B3LYP-D3BJ'):\n"
        "    for df in (True, False):\n"
        "        mf = dft.RKS(mol, xc=xc)\n"
        "        if df: mf = mf.density_fit()\n"
        "        mf.run()\n"
        "        ref = np.asarray(mf.Hessian().kernel())\n"
        "        h, dmu, route = dipole_derivatives(mf, [0,1,2,3], True)\n"
        "        print('RESULT', xc, df, route,\n"
        "              float(np.max(np.abs(h - ref))),\n"
        "              None if dmu is None else list(dmu.shape))\n",
        encoding="utf-8")
    out = subprocess.run(
        ["bash", "-lc",
         f"source {CONDA_SH} && conda activate molbuilder-pySCF "
         f"&& python {script}"],
        capture_output=True, text=True, timeout=1800)
    rows = [ln.split() for ln in out.stdout.splitlines()
            if ln.startswith("RESULT")]
    assert rows, f"probe produced nothing:\n{out.stdout}\n{out.stderr[-2000:]}"
    for _, xc, df, route, drift, *shape in rows:
        if route != "analytic":
            pytest.skip("pyscf.prop.infrared not installed in this env")
        assert float(drift) < 1e-8, (
            f"{xc} density_fit={df}: asking for IR moved the Hessian by "
            f"{float(drift):.2e} Hartree/Bohr^2")
    assert len(rows) == 4, "every functional x density-fitting combination"
