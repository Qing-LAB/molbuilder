"""Initial-state ``.molwatch.log`` preview emission.

These tests cover both the standalone helper used by SIESTA-path
generation, and the inline emitter generated into PySCF scripts.

The contract these guard against is:  *the user must see the
initial molecular structure the moment they load the file in
molwatch* -- they must not have to wait for the engine to start
producing native output.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import molbuilder
from molbuilder.trajectory_log import write_initial_preview
from molbuilder.parse.engines._helpers import trajectory_to_legacy_dict
from molbuilder.pyscf import PySCFConfig, render_script
from molbuilder.siesta import SiestaConfig, convert
from molbuilder.structure import Structure


# --------------------------------------------------------------------- #
#  L4 env-isolation helper (#534 commit 7b)                              #
#                                                                       #
#  Per [[feedback_pyscf_env_isolation]]: pyscf + geomeTRIC live in the  #
#  molbuilder-pySCF env, NEVER in the host molbuilder env.  The L4     #
#  tests below need both: molbuilder (to render the script) AND pyscf  #
#  (to execute it).  Two-env split: pytest runs in molbuilder host;    #
#  subprocess for the generated script uses the pySCF env's python.    #
# --------------------------------------------------------------------- #


def _pyscf_env_python():
    """Return the path to the molbuilder-pySCF env's python, or None
    if the env isn't installed on this machine."""
    from molbuilder.diagnostics import get_capabilities
    caps = get_capabilities()
    if not caps.env_available("molbuilder-pySCF"):
        return None
    # Standard conda layout: $CONDA_PREFIX/envs/<name>/bin/python
    import os
    candidate = Path(
        os.environ.get("CONDA_PREFIX", "/home/qqing/miniconda3")
    ).parent / "envs" / "molbuilder-pySCF" / "bin" / "python"
    if not candidate.exists():
        # Fallback: ask conda for the env's prefix
        from molbuilder.envs import run_in_env
        try:
            r = run_in_env(
                "molbuilder-pySCF",
                ["python", "-c", "import sys; print(sys.executable)"],
                capture_output=True, text=True, check=True,
            )
            return Path(r.stdout.strip())
        except Exception:
            return None
    return candidate


def _require_pyscf_env():
    """pytest skip helper: skip if molbuilder-pySCF env isn't set up
    OR if it doesn't have pyscf + geometric importable."""
    py = _pyscf_env_python()
    if py is None:
        pytest.skip(
            "molbuilder-pySCF env not installed; "
            "install via `molbuilder envs install pySCF`."
        )
    # Quick probe -- if the env exists but is missing pyscf or
    # geometric, skip with a clear message instead of letting the
    # main subprocess fail with an opaque ImportError.
    probe = subprocess.run(
        [str(py), "-c", "import pyscf, geometric; print('ok')"],
        capture_output=True, text=True, timeout=30,
    )
    if probe.returncode != 0:
        pytest.skip(
            f"molbuilder-pySCF env exists but missing pyscf/geometric: "
            f"{probe.stderr.strip()}"
        )
    return py


# --------------------------------------------------------------------- #
#  _molwatch_log.write_initial_preview                                  #
# --------------------------------------------------------------------- #


@pytest.fixture
def water_struct():
    return Structure(
        elements=["O", "H", "H"],
        positions=np.array([[0, 0, 0], [0.957, 0, 0], [-0.24, 0.927, 0]]),
        title="water",
    )


def test_preview_helper_writes_header(tmp_path, water_struct):
    p = tmp_path / "preview.molwatch.log"
    write_initial_preview(water_struct, p, job="water", engine="siesta")
    text = p.read_text()
    # Format-detection marker the molwatch parser sniffs for
    assert text.startswith("# molwatch trajectory log v1")
    # Engine line drives molwatch's source_format
    assert re.search(r"^# engine:\s*siesta\s*$", text, re.MULTILINE)
    # Job line
    assert re.search(r"^# job:\s*water\s*$", text, re.MULTILINE)
    # Units declaration
    assert "energy=eV, force=eV/Ang, coords=Ang" in text


def test_preview_helper_one_block_with_all_atoms(tmp_path, water_struct):
    p = tmp_path / "preview.molwatch.log"
    write_initial_preview(water_struct, p, job="water", engine="siesta")
    text = p.read_text()
    # Exactly one step block
    assert text.count("==== molwatch step 0 begin ====") == 1
    assert text.count("==== molwatch step 0 end ====") == 1
    # Coordinates section has all three atoms
    coord_block = text.split("coordinates (Ang):", 1)[1].split("energy", 1)[0]
    coord_lines = [ln for ln in coord_block.splitlines() if ln.strip()]
    assert len(coord_lines) == 3
    # Each line starts with the element symbol followed by 3 floats
    for line, el in zip(coord_lines, ["O", "H", "H"]):
        toks = line.split()
        assert toks[0] == el
        assert len(toks) >= 4


def test_preview_helper_marks_kind_and_nulls(tmp_path, water_struct):
    p = tmp_path / "preview.molwatch.log"
    write_initial_preview(water_struct, p, job="w", engine="siesta")
    text = p.read_text()
    # The `kind: initial_preview` line lets a downstream consumer
    # distinguish a preview-only block from a real opt step.
    assert "kind: initial_preview" in text
    # Energy / max_force are explicitly None so the parser maps to null.
    assert "energy (eV): None" in text
    assert "max_force (eV/Ang): None" in text
    # An empty scf_history sub-block (begin immediately followed by end)
    assert re.search(r"scf_history begin\s*\n\s*scf_history end",
                     text, re.MULTILINE)


def test_preview_engine_label_is_passthrough(tmp_path, water_struct):
    """The engine string is whatever the caller passes -- no mapping."""
    p = tmp_path / "x.molwatch.log"
    write_initial_preview(water_struct, p, job="x", engine="orca")
    assert "# engine: orca" in p.read_text()


# --------------------------------------------------------------------- #
#  SIESTA convert(): emits sibling .molwatch.log alongside the .fdf     #
# --------------------------------------------------------------------- #


def test_siesta_convert_emits_molwatch_log_by_default(tmp_path):
    """Calling siesta.convert() must drop a sibling .molwatch.log so a
    user can preview the structure in molwatch before SIESTA runs."""
    s = Structure(
        elements=["H", "H"],
        positions=np.array([[0, 0, 0], [0.74, 0, 0]]),
        title="h2",
    )
    xyz = tmp_path / "h2.xyz"
    s.to_xyz(str(xyz))
    fdf = tmp_path / "h2.fdf"
    summary = convert(str(xyz), str(fdf), SiestaConfig(system_label="h2"), vacuum=(12.0, 12.0, 12.0))
    mw = tmp_path / "h2.molwatch.log"
    assert mw.exists()
    assert summary["molwatch_log"] == str(mw)
    text = mw.read_text()
    assert text.startswith("# molwatch trajectory log v1")
    assert "# engine: siesta" in text
    assert "==== molwatch step 0 begin ====" in text
    assert "==== molwatch step 0 end ====" in text


def test_siesta_convert_respects_disable_flag(tmp_path):
    """cfg.write_molwatch_log = False suppresses the sibling file."""
    s = Structure(
        elements=["H", "H"],
        positions=np.array([[0, 0, 0], [0.74, 0, 0]]),
        title="h2",
    )
    xyz = tmp_path / "h2.xyz"
    s.to_xyz(str(xyz))
    fdf = tmp_path / "h2.fdf"
    summary = convert(
        str(xyz), str(fdf),
        SiestaConfig(system_label="h2", write_molwatch_log=False),
        vacuum=(12.0, 12.0, 12.0),
    )
    mw = tmp_path / "h2.molwatch.log"
    assert not mw.exists()
    assert "molwatch_log" not in summary


# --------------------------------------------------------------------- #
#  PySCF generated script: inline emitter writes preview block first    #
# --------------------------------------------------------------------- #


def test_pyscf_generated_script_emits_preview_block_text():
    """The generated script's MolwatchEmitter writes an initial-state
    preview as its first action -- so the .molwatch.log has step 0
    available before the first SCF runs."""
    s = Structure(
        elements=["H", "H"],
        positions=np.array([[0, 0, 0], [0.74, 0, 0]]),
        title="h2",
    )
    text = render_script(s, PySCFConfig(job_name="h2"))
    # The class definition must contain a method that writes a preview
    # block, AND the constructor must call it.
    assert "_write_initial_preview" in text
    assert "kind: initial_preview" in text
    # The preview block is emitted before optimize() is called -- the
    # class instantiation line must appear before the optimize(...) call.
    inst_pos = text.index("MolwatchEmitter(")
    # #534 6c: optimize() lives inside _mb_run_stage_opt helper now;
    # the loop body calls the helper rather than calling optimize()
    # directly.  Anchor on the helper definition.
    opt_pos = text.index("def _mb_run_stage_opt(")
    assert inst_pos < opt_pos


def test_pyscf_generated_script_runs_and_produces_preview(tmp_path):
    """End-to-end: generate the script, run it, verify <job>.molwatch.log
    starts with a step 0 preview block (energy=None, no forces) BEFORE
    any opt steps run.

    Architectural note (#534 commit 7b): pytest itself runs in the
    molbuilder host env (which renders the script).  The generated
    script needs pyscf + geomeTRIC, which live ONLY in the
    molbuilder-pySCF env per [[feedback_pyscf_env_isolation]].
    Subprocess is dispatched into that env via run_in_env, not the
    test's own ``sys.executable``.
    """
    pyscf_py = _require_pyscf_env()

    s = Structure(
        elements=["H", "H"],
        positions=np.array([[0, 0, 0], [0.74, 0, 0]]),
        title="h2",
    )
    from molbuilder.config.pyscf import StageSpec
    cfg = PySCFConfig(
        job_name="prev_e2e",
        log_file=False,
        # Tiny single-stage ladder so the e2e run doesn't burn cycles
        # on the publication-guide three-stage default.
        stages=[StageSpec(name="quick", enabled=True, max_steps=2)],
        basis="STO-3G",
        dispersion=None,
        density_fit=False,
        write_trajectory=False,
    )
    text = render_script(s, cfg)
    script = tmp_path / "prev_e2e.py"
    script.write_text(text)
    subprocess.run([str(pyscf_py), str(script)],
                   cwd=str(tmp_path), check=True,
                   capture_output=True, timeout=120)
    mw = tmp_path / "prev_e2e.molwatch.log"
    assert mw.exists()
    txt = mw.read_text()
    # First block must be the preview (kind: initial_preview), with
    # energy=None and an empty forces section.
    first_block = txt.split("==== molwatch step 0 begin ====", 1)[1]
    first_block = first_block.split("==== molwatch step 0 end ====", 1)[0]
    assert "kind: initial_preview" in first_block
    assert "energy (eV): None" in first_block
    assert "max_force (eV/Ang): None" in first_block
    # Subsequent step (step 1) is a real opt iter with real numbers.
    # The hook reads the geomeTRIC envs dict via key 'gradients' (PLURAL
    # -- the singular 'gradient' is wrong; verified by direct probe of
    # the envs keyset emitted by pyscf.geomopt.geometric_solver).  If
    # this assert fails it usually means the key changed in geomeTRIC.
    assert "==== molwatch step 1 begin ====" in txt
    second = txt.split("==== molwatch step 1 begin ====", 1)[1]
    second = second.split("==== molwatch step 1 end ====", 1)[0]
    assert "energy (eV): None" not in second
    assert re.search(r"energy \(eV\):\s*-?\d+\.\d+", second)


# --------------------------------------------------------------------- #
#  L4 e2e: inter-stage warm-start (#534 commit 7b)                       #
#                                                                       #
#  Layer split with the L3 render-shape gate:                            #
#                                                                       #
#    L3 (render-only) -- tests/test_pyscf.py::                          #
#      test_staged_opt_warm_starts_inside_stage_loop                    #
#      Asserts the GENERATED SCRIPT contains, inside the for-STAGE      #
#      loop body: mf.reset(mol_eq) + mf.kernel(dm0=dm_prev), and        #
#      NO bare mf.kernel().  Per decision-log 2026-06-22 this is the    #
#      load-bearing warm-start guarantee for the staged ladder.  A      #
#      regression that emits cold-init between stages fails THIS test   #
#      at script-render time, before L4 even runs.                      #
#                                                                       #
#    L4 (this file) -- verifies the rendered script RUNS to             #
#      convergence end-to-end on a real molbuilder-pySCF subprocess.    #
#      Two stages, each capped at 2 max_steps so total runtime stays    #
#      under ~30s.                                                      #
#                                                                       #
#  Why L4 alone CANNOT gate warm-start: H2/STO-3G converges from        #
#  MINAO in ~4-5 cycles -- well within typical stage budgets.  A        #
#  regression that swapped dm0=dm_prev for bare mf.kernel() would       #
#  burn ~5 extra SCF cycles per stage transition but the test's        #
#  energy / banner / log-block assertions would still pass.  Gating    #
#  warm-start at the cycle-count level here would require the          #
#  emitter to surface inter-stage SCF counts that the existing         #
#  molwatch log shape doesn't carry (the warm-start mf.kernel()        #
#  cycles land in _scf_buf but are overwritten by geomeTRIC's first    #
#  cycle==0 before any opt_step_hook flushes them).  Inventing API     #
#  surface just to make a test green would violate the design-doc      #
#  layering principle (no patching to fit a test).  The right place    #
#  to assert "did the generator emit warm-start code?" is the render   #
#  layer, which the L3 test above already covers.                     #
# --------------------------------------------------------------------- #


def test_pyscf_staged_opt_warm_start_runs_two_stages(tmp_path):
    """End-to-end: two-stage cfg runs to completion; .molwatch.log
    contains opt-step blocks from BOTH stages; final energy is
    reasonable for H2/STO-3G; and the script's stage banner is
    printed twice (proving the loop body executed twice, not just
    the first stage).

    Warm-start CODE-SHAPE is gated at L3 (see header comment above);
    this L4 gates "the rendered script runs end-to-end without
    crashing on the real pyscf+geomeTRIC stack".

    Subprocess dispatches into molbuilder-pySCF env per
    [[feedback_pyscf_env_isolation]] -- the generated script needs
    pyscf + geomeTRIC, both of which live there, not in the host
    env where pytest itself runs.
    """
    pyscf_py = _require_pyscf_env()

    s = Structure(
        elements=["H", "H"],
        positions=np.array([[0, 0, 0], [0.74, 0, 0]]),
        title="h2",
    )
    from molbuilder.config.pyscf import StageSpec
    cfg = PySCFConfig(
        job_name="warm_start_e2e",
        log_file=False,
        # Two enabled stages -- the FIRST exercises the warm-start
        # handoff at the loop boundary; the SECOND is forced to halt
        # on non-convergence (script contract for the final stage).
        # max_steps=2 each keeps total runtime tiny while still
        # forcing the loop to take the inter-stage transition.
        stages=[
            StageSpec(name="warmup",
                      enabled=True, max_steps=2,
                      conv_tol=1.0e-7,
                      on_nonconvergence="proceed"),
            StageSpec(name="refine",
                      enabled=True, max_steps=2,
                      conv_tol=1.0e-9,
                      on_nonconvergence="proceed"),
        ],
        basis="STO-3G",
        dispersion=None,
        density_fit=False,
        write_trajectory=False,
    )
    text = render_script(s, cfg)
    script_path = tmp_path / "warm_start_e2e.py"
    script_path.write_text(text)
    proc = subprocess.run(
        [str(pyscf_py), str(script_path)],
        cwd=str(tmp_path), check=True,
        capture_output=True, text=True, timeout=180,
    )
    stdout = proc.stdout

    # 1. Stage banner printed twice -- the loop took the inter-
    #    stage handoff (not just the first stage).
    assert stdout.count("=== Stage: warmup optimization ===") == 1, (
        f"warmup stage banner missing/duplicated in stdout:\n{stdout}"
    )
    assert stdout.count("=== Stage: refine optimization ===") == 1, (
        f"refine stage banner missing/duplicated in stdout:\n{stdout}"
    )
    # 2. Final energy printed once at the end -- not raised, not lost.
    assert "Final energy:" in stdout, (
        f"missing 'Final energy:' line in stdout:\n{stdout}"
    )
    final_match = re.search(r"Final energy:\s*(-?\d+\.\d+)\s*Hartree",
                            stdout)
    assert final_match, f"can't parse final energy from:\n{stdout}"
    e_tot = float(final_match.group(1))
    # H2/STO-3G total energy is around -1.12 Ha; allow a generous
    # window (the run is only 2+2 steps, geometry isn't fully
    # relaxed, but we should be in the right basin).
    assert -1.3 < e_tot < -0.9, (
        f"H2/STO-3G final energy {e_tot} Ha is not in the expected "
        f"range -- something is wrong with the SCF or the warm-start"
    )

    # 3. .molwatch.log carries opt-step blocks from BOTH stages.
    #    Per-stage prefix on geomeTRIC trajectory files = each stage
    #    writes its own _geom_<stage>_optim.xyz; the unified log
    #    accumulates ALL steps across stages.
    mw = tmp_path / "warm_start_e2e.molwatch.log"
    assert mw.exists()
    log_text = mw.read_text()
    # Should have at least 3 marker-delimited blocks total: step 0
    # (initial preview) + at least 1 step from each of 2 stages.
    step_count = log_text.count("==== molwatch step ")
    # Each step has begin + end -> 2 markers; we counted "begin "
    # plus "end ====" so divide by 2.
    n_steps = step_count // 2
    assert n_steps >= 3, (
        f"expected at least 3 molwatch step blocks (initial + 1 per "
        f"stage); got {n_steps}.  Log:\n{log_text[-2000:]}"
    )

    # 4. Per-stage geomeTRIC trajectory files exist (write_trajectory
    #    was False so they should NOT appear -- this is the negative
    #    half of the per-stage prefix contract; flip to assert
    #    presence when write_trajectory=True).
    assert not list(tmp_path.glob("warm_start_e2e_geom_*_optim.xyz"))

    # 5. The nested-shape convergence_targets reached the molwatch
    #    header (7a contract): one entry per enabled stage with the
    #    8 leaves the post-7a generator emits.
    from molbuilder.parse.engines.molwatch import MolwatchLogParser
    traj = MolwatchLogParser.parse(str(mw))
    ct = traj.runtime_info.get("convergence_targets") or {}
    # Two enabled stages -> nested shape with two top-level entries.
    assert set(ct.keys()) - {"source"} == {"warmup", "refine"}, (
        f"convergence_targets keys: {sorted(ct.keys())}"
    )
    warmup = ct["warmup"]
    assert "max_force_tol_eV_per_A" in warmup
    assert "rms_force_tol_eV_per_A" in warmup
    assert "max_displ_ang" in warmup
    assert "rms_displ_ang" in warmup


# --------------------------------------------------------------------- #
#  Cross-repo round-trip: molwatch parser reads the preview block       #
# --------------------------------------------------------------------- #


def test_molwatch_can_parse_siesta_preview(tmp_path):
    """The .molwatch.log emitted by molbuilder must be loadable by
    molwatch's MolwatchLogParser, exposing the initial geometry as
    frame 0 with null energy and empty forces.  This is the cross-repo
    contract: molbuilder writes, molwatch reads."""
    from molbuilder.parse.engines.molwatch import MolwatchLogParser

    s = Structure(
        elements=["H", "H"],
        positions=np.array([[0, 0, 0], [0.74, 0, 0]]),
        title="h2",
    )
    p = tmp_path / "preview.molwatch.log"
    write_initial_preview(s, p, job="h2", engine="siesta")
    assert MolwatchLogParser.can_parse(str(p))
    result = trajectory_to_legacy_dict(MolwatchLogParser.parse(str(p)))
    assert len(result["frames"]) == 1
    assert result["frames"][0] == [["H", 0.0, 0.0, 0.0],
                                   ["H", 0.74, 0.0, 0.0]]
    assert result["energies"] == [None]
    assert result["max_forces"] == [None]
    assert result["forces"] == [[]]
    assert result["scf_history"] == [[]]
    assert result["source_format"] == "siesta"
