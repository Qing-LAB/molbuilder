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
import subprocess

import numpy as np
import pytest

from _road import conda_hook, env_available

CONDA_SH = conda_hook()

pytestmark = [
    pytest.mark.engine,
    pytest.mark.skipif(
        not (CONDA_SH.is_file() and env_available("molbuilder-pySCF")),
        reason="needs the molbuilder-pySCF env + a detectable conda hook"),
]

WATER = "3\nwater\nO 0.0 0.0 0.119\nH 0.0 0.757 -0.477\nH 0.0 -0.757 -0.477\n"


def _bundle_head(folder, *objects) -> str:
    """A probe's first lines, the way a PySCF script starts
    (`engines/pyscf.md` § 3): ``mb_pyscf.pyz`` written into ``folder``, then
    the script's own head and an import of each of ``objects`` from the
    bundle, bound as ``_mb_<its name>`` -- so a probe runs the code a run
    executes, under the env that runs it, where molbuilder is not
    installed."""
    from molbuilder.pyscf.input import emit_bundle_imports, emit_script_head
    from molbuilder.runwrap import PYSCF_BUNDLE, pyscf_bundle
    (folder / PYSCF_BUNDLE).write_bytes(pyscf_bundle())
    return "\n".join([*emit_script_head(None),
                      *emit_bundle_imports(*objects)])


def _describe(tmp_path, monkeypatch, *, frozen=()):
    """Create the calculation the way a user does: both citations read from
    the projects root (`job-contracts.md` § 2.5b).  ``frozen`` holds atoms
    still the way the viewer does -- as the structure's own region, written
    into its pair by the codec -- never as a form field."""
    from click.testing import CliRunner

    from molbuilder.jobset._cli import jobset_group
    from molbuilder.projects import PROJECTS_ROOT_ENV
    tree = tmp_path / "projects"
    (tree / "P" / "structure").mkdir(parents=True)
    if frozen:
        from molbuilder.structure import Structure
        from molbuilder.workingcopy_structure import StructureCodec
        s = Structure.from_xyz(WATER)
        s.frozen_atoms = list(frozen)
        StructureCodec().write(s, tree / "P" / "structure" / "w.xyz")
    else:
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
    # How a shell enters conda HERE is this machine's record's to say -- and
    # the run states its threads (`architecture.md` § 5.2).
    from conftest import write_machine_record
    write_machine_record(env_init={
        "activation": "conda activate", "preamble": f"source {CONDA_SH}"})
    task = json.loads((bundle / "task.json").read_text())
    task["execution"] = {**task.get("execution", {}), "threads": 1}
    (bundle / "task.json").write_text(json.dumps(task, indent=2))
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
    # A converged SCF is worth a message (`stages.md` § 6.9's neighbour,
    # `notify.on_scf_converged`): asked for here the way a person asks, in
    # the description, so the monitor beside the run is told to send them.
    task = json.loads((bundle / "task.json").read_text())
    task["notify"] = {"on_scf_converged": True}
    (bundle / "task.json").write_text(json.dumps(task, indent=2))
    d = _prep_and_run(bundle)

    from molbuilder.spectra.results import SCHEMA_VERSION
    assert d["schema_version"] == SCHEMA_VERSION
    for k in ("phase_relaxation", "phase_frequencies", "phase_raman"):
        assert d[k] == "complete", (k, d[k])
    # the probe was never asked for (`es_mode_selection` left at `skip`):
    # the flag says so from the first write (vibration.md § 4.9) -- and so
    # does infrared's, which this description does not ask for either
    assert d["phase_es"] == "not requested"
    assert d["config"]["compute_ir"] is False
    assert d["phase_ir"] == "not requested"

    # D3, live: the relaxation ran, was tracked, and converged.
    rel = d["relaxation"]
    assert rel["enabled"] and rel["converged"] and rel["n_steps"] >= 1
    assert rel["max_force_eh_bohr"] < 1e-3
    # The geometry in the file is the one the Hessian was taken at -- the
    # RELAXED one, not the input (design § 15.4): the relaxation moved
    # atoms, so the two must differ.
    from molbuilder.structure import Structure
    _moved = np.abs(np.asarray(d["equilibrium"]["positions_ang"])
                    - Structure.from_xyz(WATER).positions).max()
    assert _moved > 1e-4, f"positions_ang is still the input geometry ({_moved})"

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
    assert th["g_eh"] < d["equilibrium"]["scf_energy_eh"] + 0.05
    # ONE quantity under one label (design § 15.3): the headline T is on
    # the grid and the grid's row there IS the headline -- full RRHO both.
    tg = th["grid"]["temperatures_K"]
    assert len(tg) == 31 and th["temperature_K"] in tg
    k = tg.index(th["temperature_K"])
    assert th["g_eh"] == th["grid"]["g_eh"][k]
    assert th["s_eh_k"] == th["grid"]["s_eh_k"][k]
    assert "the headline and the grid alike" in th["note"]

    # THE MONITOR BESIDE IT read the run with the framework's readers
    # (`run-reports.md` § 2.3): how it ended in the Results tab's words,
    # the progress log's last step -- and the JOB's basis, a run started
    # directly being its process tree, a CPU run judged on no GPU (§ 2.1a).
    from molbuilder.runfiles import compose
    log = (bundle / "01_freq" / "run-0"
           / compose("W", ".monitor.log", "01_freq", run=0)).read_text()
    closing = [ln for ln in log.splitlines() if "[STATUS]" in ln][-1]
    assert "finished" in closing and "| step " in closing, closing
    # the force beside the stage's own tolerance, from the header the deck
    # wrote over prep's seed -- read afresh, not from the seed's old offset
    assert "max force" in closing and "(tol 0.0102" in closing, closing
    # PySCF's own SCF, as its progress log states it: the last cycle and
    # its residuals (`molwatch_reader.MolwatchReader.now`)
    assert "SCF iteration" in closing and "dE " in closing, closing
    # ...each beside its criterion, which the deck read back off its own
    # solver, in eV (`molwatch_grammar.scf_criteria`, web/trajectory.md § 3)
    import re
    assert re.search(r"dE \S+ eV \(tol \S+\)", closing), closing
    assert re.search(r"\|g\| \S+ eV \(tol \S+\)", closing), closing
    # every finished step is a converged SCF -- the first one too: a PySCF
    # block is written when its step ENDS, and the rule once needed a step
    # before it, so a run whose steps all ended between two wakes sent none
    said = [ln for ln in log.splitlines() if "[NOTIFY]" in ln]
    assert any("scf_converged" in ln for ln in said), said
    basis = next(ln for ln in log.splitlines() if "[UTIL-BASIS]" in ln)
    assert ("[launched on]" in basis and "cpu time [process tree]" in basis
            and "mem [process tree" in basis), basis
    summary = next(ln for ln in log.splitlines() if "[UTIL-SUMMARY]" in ln)
    assert "gpu" not in summary.lower(), summary
    import re
    mean = float(re.search(r"cpu mean=(\d+)%", summary).group(1))
    assert 0.0 < mean <= 120.0, summary

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

    # Raman was NOT requested: its phase says so (vibration.md § 4.9),
    # and the run says so as a route, not as a zero.  Infrared's own flag
    # is closed by the dipole sweep -- the one a reader waits on while that
    # sweep, which writes nothing until it ends, runs (§ 4.9).
    assert d["phase_ir"] == "complete"
    assert d["phase_raman"] == "not requested"
    assert all(m["raman_activity_a4_amu"] in (None, 0.0) for m in modes)
    assert d["raman_route"] == "none" and d["raman_fd_step_ang"] is None


def test_water_in_water_runs_the_solvated_chain_end_to_end(
        tmp_path, monkeypatch):
    """Category 2's live bar (integration plan, 2026-08-21), on the route
    PCM reaches (`engines/vibration.md` § 4.6): PCM water, the relaxation
    and the Hessian under one solvated Hamiltonian -- PySCF's analytic
    Hessian adds the solvent's own response (`with_solvent.hess`).  IR and
    Raman are OFF: their routes are built without that term, and the
    settings gate refuses them with a solvent until they carry it and are
    measured (ruled 2026-09-29; the whole chain ran here until 2026-09-30,
    under an info saying it all carried the solvent).  The physics pins are
    deliberately loose: PCM shifts water's bands by tens of cm^-1, so the
    gas windows widen; what is being proven is the solvated run completing
    with real numbers, plus the solvated deck actually differing from gas
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
    # Raman OFF (it defaults on); IR stays off, its default.
    i = t.index("[item.compute_raman]"); j = t.index("[item.", i + 1)
    t = t[:i] + t[i:j].replace("value = true", "value = false", 1) + t[j:]
    assert t[t.index("[item.compute_raman]"):t.index("[item.", t.index("[item.compute_raman]") + 1)].count("value = false") == 1
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
    assert all(m["raman_activity_a4_amu"] is None for m in modes)
    assert all(m["ir_intensity_km_mol"] is None for m in modes)
    assert d["thermo"]["grid"]["temperatures_K"], "thermo grid missing"
    assert d["raman_route"] == "none"


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
    from molbuilder.spectra.pyscf_vibration import dipole_derivatives
    script = tmp_path / "hess_identity.py"
    script.write_text(
        _bundle_head(tmp_path, dipole_derivatives)
        + "import numpy as np\n"
        "from pyscf import gto, dft\n"
        "mol = gto.Mole(atom='N 0 0 0; H 0.8 0 0; H 0 1 0; H 0 0 1.2',\n"
        "               basis='6-31G', verbose=0).build()\n"
        "for xc in ('PBE0', 'B3LYP-D3BJ'):\n"
        "    for df in (True, False):\n"
        "        mf = dft.RKS(mol, xc=xc)\n"
        "        if df: mf = mf.density_fit()\n"
        "        mf.run()\n"
        "        ref = np.asarray(mf.Hessian().kernel())\n"
        "        h, dmu, route = _mb_dipole_derivatives(mf, [0,1,2,3], True)\n"
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


def test_the_rank_rule_reproduces_pyscf_on_free_molecules(tmp_path):
    """The gate of the unification design (§ 7.5 step 2): before the deck
    is touched, the one path -- `spectra.normal_modes.vibrational_modes`,
    nothing held -- must reproduce PySCF's own `harmonic_analysis` on
    free molecules to numerical noise.  Four systems, each for a reason:
    water (the ordinary case), CO2 (five motions removed, not six, with
    no linearity flag anywhere), HF (one mode, nothing to hide behind),
    methane (degenerate modes -- eigenvalues must agree, eigenvectors
    need not).

    The probe imports the path from ``mb_pyscf.pyz`` beside it, through
    the script's own head and import lines, the way the vibration script
    does (`engines/pyscf.md` § 3) -- so what is checked is the code a run
    executes, under the env that runs it.
    """
    from molbuilder.spectra.normal_modes import (frequencies_cm1,
                                                 vibrational_modes)
    script = tmp_path / "rank_gate.py"
    script.write_text(
        _bundle_head(tmp_path, vibrational_modes)
        + "import json\n"
        "import numpy as np\n"
        "from pyscf import gto, scf\n"
        "from pyscf.hessian import thermo\n"
        "SYSTEMS = {\n"
        " 'water': 'O 0 0 0.119; H 0 0.757 -0.477; H 0 -0.757 -0.477',\n"
        " 'co2': 'O 0 0 -1.16; C 0 0 0; O 0 0 1.16',\n"
        " 'hf': 'H 0 0 0; F 0 0 0.92',\n"
        " 'methane': ('C 0 0 0; H 0.629 0.629 0.629; H -0.629 -0.629 0.629;'\n"
        "             ' H -0.629 0.629 -0.629; H 0.629 -0.629 -0.629'),\n"
        "}\n"
        "for name, atom in SYSTEMS.items():\n"
        "    mol = gto.M(atom=atom, basis='sto-3g', verbose=0)\n"
        "    mf = scf.RHF(mol).run()\n"
        "    hess = np.asarray(mf.Hessian().kernel())\n"
        "    mass = np.asarray(mol.atom_mass_list(isotope_avg=True), dtype=float)\n"
        "    ref = thermo.harmonic_analysis(mol, hess, mass=mass)\n"
        "    lam, modes, pats = _mb_vibrational_modes(\n"
        "        hess, mass, mol.atom_coords(unit='Angstrom'), [],\n"
        "        ('isolated', 'isolated', 'isolated'))\n"
        "    ref_lam = np.asarray(ref['force_const_au'], dtype=float)\n"
        "    ref_w = ref['freq_wavenumber']\n"
        "    ref_cm1 = [float(w.real) if abs(getattr(w, 'imag', 0.0)) == 0\n"
        "               else -abs(float(w.imag)) for w in ref_w]\n"
        "    # mode overlap in the mass metric, ref mode i against ours, for\n"
        "    # the non-degenerate systems\n"
        "    ref_modes = np.asarray(ref['norm_mode'], dtype=float)\n"
        "    overlaps = []\n"
        "    for i in range(len(ref_lam)):\n"
        "        best = max(abs(float((mass[:, None] * ref_modes[i] * L).sum()))\n"
        "                   for L in modes)\n"
        "        overlaps.append(best)\n"
        "    print('RESULT', json.dumps({'name': name, 'n_ref': int(len(ref_lam)),\n"
        "          'n_ours': int(len(lam)), 'n_rigid': int(len(pats)),\n"
        "          'ref': sorted(ref_lam.tolist()), 'ours': sorted(lam.tolist()),\n"
        "          'ref_cm1': sorted(ref_cm1), 'overlaps': overlaps}))\n",
        encoding="utf-8")
    out = subprocess.run(
        ["bash", "-lc",
         f"source {CONDA_SH} && conda activate molbuilder-pySCF "
         f"&& python {script}"],
        capture_output=True, text=True, timeout=1800)
    rows = [json.loads(ln[len("RESULT "):]) for ln in out.stdout.splitlines()
            if ln.startswith("RESULT ")]
    assert len(rows) == 4, f"probe produced {len(rows)} rows:\n{out.stdout}\n{out.stderr[-3000:]}"
    expected_modes = {"water": 3, "co2": 4, "hf": 1, "methane": 9}
    import numpy as np
    for r in rows:
        name = r["name"]
        assert r["n_ref"] == r["n_ours"] == expected_modes[name], (name, r["n_ref"], r["n_ours"])
        assert r["n_rigid"] == 3 * {"water": 3, "co2": 3, "hf": 2, "methane": 5}[name] - expected_modes[name]
        ref, ours = np.asarray(r["ref"]), np.asarray(r["ours"])
        assert np.allclose(ours, ref, rtol=1e-8, atol=1e-13), (name, ours - ref)
        ours_cm1 = frequencies_cm1(ours)
        assert np.allclose(ours_cm1, np.asarray(r["ref_cm1"]), atol=1e-4), (name, ours_cm1, r["ref_cm1"])
        if name in ("water", "hf"):
            assert all(abs(o - 1.0) < 1e-6 for o in r["overlaps"]), (name, r["overlaps"])


def test_water_with_its_oxygen_held_reports_three_vibrations(tmp_path,
                                                             monkeypatch):
    """The unification's demonstrator (design § 7.6): hold water's oxygen
    and half of the six numbers the old path reported were not vibrations
    -- the two hydrogens swinging about a nailed-down atom, at 15-24 cm-1
    with the two loudest infrared bands of the run.  Through the whole
    described road, the run must now report the three vibrations only,
    say what it removed, and keep the free atoms' own stationarity test
    (R5) rather than the constraint force on the oxygen."""
    bundle = _describe(tmp_path, monkeypatch, frozen=(0,))
    d = _prep_and_run(bundle)

    assert d["frozen_atom_idxs"] == [0] and d["free_atom_idxs"] == [1, 2]
    freqs = sorted(m["frequency_cm1"] for m in d["modes"])
    assert len(freqs) == 3, freqs
    assert d["removed_motions"]["count"] == 3
    assert len(d["removed_motions"]["patterns"]) == 3
    # Nothing under 100 cm-1 survives: the leftovers are gone, and the three
    # that remain are the bend and the two stretches.
    assert freqs[0] > 1000.0, freqs
    assert 1400.0 < freqs[0] < 1900.0 and 3300.0 < freqs[2] < 4100.0, freqs
    th = d["thermo"]
    assert th["regime"] == "vibrational-only"
    # no pressure enters the vibrational sums, on this engine as on SIESTA,
    # and the note gives the reason the answer is vibrational-only (§ 4.7)
    assert th["pressure_atm"] is None
    assert "atoms are held" in th["note"], th["note"]
    # ONE quantity under one label here too: the vibrational sums above the
    # electronic energy, the headline a row of the grid, no gas-phase term.
    tg = th["grid"]["temperatures_K"]
    k = tg.index(th["temperature_K"])
    assert th["g_eh"] == th["grid"]["g_eh"][k]
    assert th["h_eh"] == th["grid"]["h_eh"][k]
    assert th["n_modes"] == 3 and th["n_rigid_removed"] == 3
    assert th["n_imag_excluded"] == 0
    # The reduced calculation, stated: second derivatives for the two free
    # atoms only.
    assert d["hessian_scope"] == "free" and d["n_atoms_in_hessian"] == 2
    assert d["hessian_density_fit"] is False
    # The run wrapper states the constraint in its own header, read back
    # off the deck: one fact, one spelling across the process boundary.
    attempt = bundle / "01_freq" / "run-0"
    wrap_logs = sorted(attempt.glob("*.runwrap-*.log"))
    assert wrap_logs, "the wrapper writes its session log beside the run"
    header = wrap_logs[-1].read_text(errors="replace")
    assert "frozen_atoms: 1 listed indices" in header, header[:1500]
    # Every mode is orthogonal to every removed motion in the mass metric
    # (science/normal-modes.md § 7 point 3), read off the artifact itself.
    import numpy as np
    from molbuilder.chemistry import atomic_mass
    masses = np.array([atomic_mass(e) for e in d["equilibrium"]["elements"]])
    free = d["free_atom_idxs"]
    sqm = np.sqrt(masses[free])
    for m in d["modes"]:
        L = np.asarray(m["eigenvector_canonical"]) * sqm[:, None]
        for pat in d["removed_motions"]["patterns"]:
            v = np.asarray(pat) * sqm[:, None]
            v /= np.linalg.norm(v)
            assert abs(float((L * v).sum())) < 1e-6


def test_the_free_atom_hessian_is_the_free_block_of_the_full_one(tmp_path):
    """§ 10's owed check.  The reduced calculation asks PySCF for second
    derivatives of the free atoms only.  It must give exactly the free-free
    block of the compute-everything Hessian -- with and without a
    dispersion correction, whose Hessian term is full-size and has to be
    cut to the free atoms by hand -- and it must come back numbered by
    position in the list, which is why the deck places it by index."""
    from molbuilder.spectra.pyscf_vibration import dipole_derivatives
    script = tmp_path / "partial_hessian.py"
    script.write_text(
        _bundle_head(tmp_path, dipole_derivatives)
        + "import numpy as np\n"
        "from pyscf import gto, scf, dft\n"
        "CASES = [\n"
        "  ('water-rhf', 'O 0 0 0.119; H 0 0.757 -0.477; H 0 -0.757 -0.477', 'sto-3g', None, [1, 2]),\n"
        "  ('nh3-b3lyp-d3bj', 'N 0 0 0; H 0.8 0 0; H 0 1 0; H 0 0 1.2', '6-31G', 'B3LYP-D3BJ', [0, 1]),\n"
        "]\n"
        "for name, atom, basis, xc, free in CASES:\n"
        "    mol = gto.M(atom=atom, basis=basis, verbose=0)\n"
        "    mf = (dft.RKS(mol, xc=xc) if xc else scf.RHF(mol)).run()\n"
        "    full = np.asarray(mf.Hessian().kernel())\n"
        "    reduced, dmu, route = _mb_dipole_derivatives(mf, free, True)\n"
        "    held = [i for i in range(mol.natm) if i not in free]\n"
        "    drift = float(np.max(np.abs(reduced[np.ix_(free, free)] - full[np.ix_(free, free)])))\n"
        "    zeros = float(np.max(np.abs(reduced[held])))\n"
        "    print('RESULT', name, route, drift, zeros, dmu is None)\n",
        encoding="utf-8")
    out = subprocess.run(
        ["bash", "-lc",
         f"source {CONDA_SH} && conda activate molbuilder-pySCF "
         f"&& python {script}"],
        capture_output=True, text=True, timeout=1800)
    rows = [ln.split() for ln in out.stdout.splitlines() if ln.startswith("RESULT")]
    assert len(rows) == 2, f"probe produced {len(rows)} rows:\n{out.stdout}\n{out.stderr[-3000:]}"
    for _, name, route, drift, zeros, dmu_none in rows:
        # HF: the block IS the block.  DFT: the partial list omits the held
        # atoms' grid-weight response -- measured 1.45e-5 Hartree/Bohr^2 on
        # this system, about 0.05 cm^-1 on a stretch; pinned at that scale
        # so a real disagreement (1e-3 and up) cannot hide behind it.
        assert float(drift) < (1e-7 if name.endswith("rhf") else 5e-5), (name, drift)
        assert float(zeros) == 0.0, (name, zeros)
        assert route == "finite-difference" and dmu_none == "True", (name, route)


def test_a_setting_that_enters_nothing_is_said_to_at_prep(tmp_path,
                                                          monkeypatch):
    """`engines/vibration.md` § 3.1 (user, 2026-09-28): a value the person set
    that the run cannot use is warned about at prep -- and only then.

    * The PRESSURE: with atoms held the thermochemistry is the vibrational
      contributions alone and no pressure enters it; on the free molecule
      it enters the gas-phase translation, and nothing is said.
    * The FREQUENCY WINDOW: it filters the modes `all` selects and nothing
      else (§ 4.8); set under `skip` it is said to do nothing, under `all`
      it is not.

    Prep alone; nothing is launched.

    MUTATIONS THIS MUST FAIL AGAINST: the pressure warning keyed on the
    pressure alone (it would speak for the free molecule too); the window
    warning keyed on the window alone (it would speak under `all`).
    """
    from click.testing import CliRunner

    from molbuilder.jobset._cli import jobset_group
    said = {}
    for held, selector in (((0,), '"skip"'), ((), '"all"')):
        sub = tmp_path / ("held" if held else "free")
        sub.mkdir()
        bundle = _describe(sub, monkeypatch, frozen=held)
        tpl = bundle / "W.template.toml"
        t = tpl.read_text()
        for item, old, new in (("pressure_atm", "value = 1.0", "value = 10.0"),
                               ("es_mode_selection", 'value = "skip"',
                                f"value = {selector}")):
            i = t.index(f"[item.{item}]"); j = t.index("[item.", i + 1)
            assert old in t[i:j], t[i:j]
            t = t[:i] + t[i:j].replace(old, new, 1) + t[j:]
        # the window is valueless until set: its value line goes in first
        i = t.index("[item.freq_min_cm1]"); j = t.index("[item.", i + 1)
        assert "\nvalue = " not in t[i:j], t[i:j]
        t = (t[:i] + t[i:j].replace('\ntype = "float"\n',
                                    '\ntype = "float"\nvalue = 500.0\n', 1)
             + t[j:])
        tpl.write_text(t)
        r = CliRunner().invoke(jobset_group, ["prep", "run", "freq",
                                              "--bundle", str(bundle)])
        assert r.exit_code == 0, r.output
        said[bool(held)] = (
            "pressure_atm = 10 atm has no effect here" in r.output,
            "freq_min_cm1 = 500 cm⁻¹ has no effect here" in r.output)
    assert said == {True: (True, True), False: (False, False)}, said


def test_a_hartree_fock_deck_names_no_functional(tmp_path, monkeypatch):
    """`engines/vibration.md` § 4.10 (V1.8): the level of theory is one
    answer, `is_dft`.  Described as Hartree-Fock with the functional changed
    to the hybrid PBE0 and the grid to 3, the prepped deck's header and
    Methods paragraph name Hartree-Fock and no functional, its constants
    carry no functional and no grid, prep says the changed functional enters
    nothing, and the grid advisory -- which a hybrid DFT run at level 3
    draws -- says nothing, since HF has no grid.

    THE DISPERSION CORRECTION IS NOT A DFT QUESTION (`engines/pyscf.md`
    § 7a; user, 2026-09-28: "scientifically correct decision applied"): HF
    takes it like any method, so d4 reaches the deck as `mf.disp = "d4"` in
    the dresser every construction calls, is named in the header and in the
    Methods paragraph with its own paper, and draws no warning; "none" is
    plain Hartree-Fock.  Prep alone; nothing is launched.

    MUTATIONS THIS MUST FAIL AGAINST: the paragraph reading `cfg.functional`
    whatever the method (its write-up named B3LYP under RHF until
    2026-09-28); the grid advisory asking the functional without asking the
    method; the dispersion dropped under Hartree-Fock (as both decks did
    until the same day).
    """
    from click.testing import CliRunner

    from molbuilder.jobset._cli import jobset_group
    for disp in ("d4", "none"):
        sub = tmp_path / disp
        sub.mkdir()
        bundle = _describe(sub, monkeypatch)
        tpl = bundle / "W.template.toml"
        t = tpl.read_text()
        for item, old, new in (("method", '"DFT"', '"HF"'),
                               ("functional", '"B3LYP"', '"PBE0"'),
                               ("dispersion", '"d3bj"', f'"{disp}"'),
                               ("grid_level", "4", "3")):
            i = t.index(f"[item.{item}]"); j = t.index("[item.", i + 1)
            assert f"value = {old}" in t[i:j], t[i:j]
            t = (t[:i] + t[i:j].replace(f"value = {old}", f"value = {new}", 1)
                 + t[j:])
        tpl.write_text(t)
        r = CliRunner().invoke(jobset_group, ["prep", "run", "freq",
                                              "--bundle", str(bundle)])
        assert r.exit_code == 0, r.output
        assert "functional = 'PBE0' has no effect here" in r.output, r.output
        assert "dispersion =" not in r.output, r.output
        assert "Grid level" not in r.output, r.output
        deck = next((bundle / "01_freq").glob("*.py")).read_text()
        head = deck[:deck.index('"""', 3)]
        assert "Method    : RHF (Hartree-Fock)" in head, head
        start = deck.index('METHODS_TEXT = """') + len('METHODS_TEXT = """')
        methods = deck[start:deck.index('"""', start)]
        assert "Hartree-Fock (RHF)" in methods, methods[:600]
        for dft_word in ("PBE", "B3LYP"):
            assert dft_word not in methods, (dft_word, methods[:600])
        assert "FUNCTIONAL                 = None" in deck
        assert "GRID_LEVEL                 = None" in deck
        assert f"DISPERSION                 = '{disp}'" in deck
        fn = deck[deck.index("def _mb_configure_theory(mf):"):]
        fn = fn[:fn.index("return mf")]
        if disp == "d4":
            assert "Dispersion: d4" in head, head
            assert ("with the D4 dispersion correction [Caldeweyher2019]"
                    in methods), methods[:600]
            assert 'mf.disp = "d4"' in fn, fn
        else:
            assert "Dispersion:" not in head, head
            assert "dispersion correction" not in methods, methods[:600]
            assert "mf.disp" not in fn, fn


def test_the_listed_modes_get_the_probe_and_no_other(tmp_path, monkeypatch):
    """`engines/vibration.md` § 4.8: `explicit` gives the per-mode probe to
    the modes its list names -- TEXT, "1, 3", read by the one index-list
    reader (`PySCFConfig.explicit_modes`).  Until 2026-09-28 the deck wrote
    `list()` of that text, its characters, and the inlined selector's
    `int(',')` stopped the probe; no run through the road had ever selected
    a mode, so nothing saw it.  A list the reader cannot take -- "0, 2", the
    modes count from 1 -- is refused at prep, before a deck exists.

    MUTATION THIS MUST FAIL AGAINST: the deck's constant written as
    `list(cfg.es_explicit_indices)`.
    """
    from click.testing import CliRunner

    from molbuilder.jobset._cli import jobset_group
    bundle = _describe(tmp_path, monkeypatch)
    tpl = bundle / "W.template.toml"
    original = tpl.read_text()

    def _answer(listed):
        t = original
        for item, old, new in (("es_mode_selection", '"skip"', '"explicit"'),
                               ("es_explicit_indices", '""', f'"{listed}"'),
                               # the probe is the subject; Raman's 6N SCFs
                               # are not
                               ("compute_raman", "true", "false")):
            i = t.index(f"[item.{item}]"); j = t.index("[item.", i + 1)
            assert f"value = {old}" in t[i:j], t[i:j]
            t = (t[:i] + t[i:j].replace(f"value = {old}", f"value = {new}", 1)
                 + t[j:])
        tpl.write_text(t)

    _answer("0, 2")
    r = CliRunner().invoke(jobset_group,
                           ["prep", "run", "freq", "--bundle", str(bundle)])
    assert r.exit_code != 0, r.output
    assert "can't be read: 0 is below 1" in r.output, r.output
    assert not list((bundle / "01_freq").glob("*.py")), "a deck was written"

    _answer("1, 3")
    d = _prep_and_run(bundle)
    assert d["phase_es"] == "complete", d["phase_es"]
    assert d["phase_raman"] == "not requested"
    probed = sorted(m["index_1based"] for m in d["modes"]
                    if m["electronic_structure"] is not None)
    assert probed == [1, 3], probed
    assert d["selected_mode_idxs_1based"] == [1, 3]
    for m in d["modes"]:
        es = m["electronic_structure"]
        if es is None:
            continue
        # the two displaced SCFs and their orbital windows, both sides
        assert np.isfinite(es["scf_energy_plus_eh"]), es
        assert np.isfinite(es["scf_energy_minus_eh"]), es
        assert es["mo_energies_plus_eh"] and es["mo_energies_minus_eh"], es
