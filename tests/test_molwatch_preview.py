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
from molbuilder._molwatch_log import write_initial_preview
from molbuilder.pyscf_input import PySCFConfig, render_script
from molbuilder.siesta import SiestaConfig, convert
from molbuilder.structure import Structure


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
    summary = convert(str(xyz), str(fdf), SiestaConfig(system_label="h2"))
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
    )
    mw = tmp_path / "h2.molwatch.log"
    assert not mw.exists()
    assert "molwatch_log" not in summary


# --------------------------------------------------------------------- #
#  PySCF generated script: inline emitter writes preview block first    #
# --------------------------------------------------------------------- #


def test_pyscf_generated_script_emits_preview_block_text():
    """The generated script's _MolwatchEmitter writes an initial-state
    preview as its first action -- so the .molwatch.log has step 0
    available before the first SCF runs."""
    s = Structure(
        elements=["H", "H"],
        positions=np.array([[0, 0, 0], [0.74, 0, 0]]),
        title="h2",
    )
    text = render_script(s, PySCFConfig(job_name="h2", preopt=False))
    # The class definition must contain a method that writes a preview
    # block, AND the constructor must call it.
    assert "_write_initial_preview" in text
    assert "kind: initial_preview" in text
    # The preview block is emitted before optimize() is called -- the
    # class instantiation line must appear before the optimize(...) call.
    inst_pos = text.index("_MolwatchEmitter(")
    opt_pos = text.index("mol_eq = optimize(")
    assert inst_pos < opt_pos


def test_pyscf_generated_script_runs_and_produces_preview(tmp_path):
    """End-to-end: generate the script, run it, verify <job>.molwatch.log
    starts with a step 0 preview block (energy=None, no forces) BEFORE
    any opt steps run.

    Skipped if PySCF or geomeTRIC isn't installed in the test env.
    """
    pytest.importorskip("pyscf")
    pytest.importorskip("geometric")

    s = Structure(
        elements=["H", "H"],
        positions=np.array([[0, 0, 0], [0.74, 0, 0]]),
        title="h2",
    )
    cfg = PySCFConfig(
        job_name="prev_e2e",
        preopt=False,
        log_file=False,
        geom_max_steps=2,
        basis="STO-3G",
        dispersion=None,
        density_fit=False,
        write_trajectory=False,
    )
    text = render_script(s, cfg)
    script = tmp_path / "prev_e2e.py"
    script.write_text(text)
    subprocess.run([sys.executable, str(script)],
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
    assert "==== molwatch step 1 begin ====" in txt
    second = txt.split("==== molwatch step 1 begin ====", 1)[1]
    second = second.split("==== molwatch step 1 end ====", 1)[0]
    assert "energy (eV): None" not in second
    assert re.search(r"energy \(eV\):\s*-?\d+\.\d+", second)


# --------------------------------------------------------------------- #
#  Cross-repo round-trip: molwatch parser reads the preview block       #
# --------------------------------------------------------------------- #


def test_molwatch_can_parse_siesta_preview(tmp_path):
    """The .molwatch.log emitted by molbuilder must be loadable by
    molwatch's MolwatchLogParser, exposing the initial geometry as
    frame 0 with null energy and empty forces.  This is the cross-repo
    contract: molbuilder writes, molwatch reads."""
    pytest.importorskip("parsers.molwatch_log",
                        reason="molwatch package not on PYTHONPATH")
    from parsers.molwatch_log import MolwatchLogParser  # type: ignore

    s = Structure(
        elements=["H", "H"],
        positions=np.array([[0, 0, 0], [0.74, 0, 0]]),
        title="h2",
    )
    p = tmp_path / "preview.molwatch.log"
    write_initial_preview(s, p, job="h2", engine="siesta")
    assert MolwatchLogParser.can_parse(str(p))
    result = MolwatchLogParser.parse(str(p))
    assert len(result["frames"]) == 1
    assert result["frames"][0] == [["H", 0.0, 0.0, 0.0],
                                   ["H", 0.74, 0.0, 0.0]]
    assert result["energies"] == [None]
    assert result["max_forces"] == [None]
    assert result["forces"] == [[]]
    assert result["scf_history"] == [[]]
    assert result["source_format"] == "siesta"
