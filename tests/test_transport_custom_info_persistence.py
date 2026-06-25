"""Pin the transport-pipeline custom-info persistence contract.

Audit 2026-06-25 (#31) found three BLOCKER/IMPORTANT drift cases
where the transport .fdf + .transport.json lost information the
sibling SIESTA pipeline already preserves:

  1. Transport .fdf emitted no ``# Sidecar v3`` ATOM-METADATA block,
     so a parsed-back transport .fdf had no recoverable
     regions / frozen_atoms.
  2. Transport .fdf emitted no ``# runtime.<key>`` echo lines, so
     the .out parser's runtime-info probe could not recover
     max_memory_mb / num_threads.
  3. ``TransportResults`` had no first-class regions / frozen_atoms
     fields, so .transport.json was divorced from the boundary
     conditions of the calculation.

Each test below pins one leg of the fix.
"""
from __future__ import annotations

import json
from textwrap import dedent

import numpy as np
import pytest

from molbuilder.config.transport import TransportConfig
from molbuilder.structure import Structure
from molbuilder.transport.results import TransportResults, SCHEMA_VERSION
from molbuilder.transport.transiesta import TransiestaEngine


def _mk_device_struct() -> Structure:
    """Minimal 2-terminal device: 2 atoms per electrode + 2 bridge.
    Enough for render_script to fire all electrode + bridge branches."""
    return Structure(
        elements      = ["Au", "Au", "C", "S", "S", "Au", "Au"],
        positions     = np.array([
            [0, 0, 0], [0, 0, 1],          # left electrode
            [0, 0, 3], [0, 0, 4], [0, 0, 5],  # bridge
            [0, 0, 7], [0, 0, 8],          # right electrode
        ], dtype=float),
        atom_names    = ["Au1", "Au2", "C1", "S1", "S2", "Au3", "Au4"],
        residue_ids   = [1, 1, 2, 2, 2, 3, 3],
        residue_names = ["LEL", "LEL", "BRG", "BRG", "BRG", "REL", "REL"],
        chain_ids     = ["A"] * 7,
        regions       = {
            "L-electrode": [0, 1],
            "M-bridge":    [2, 3, 4],
            "R-electrode": [5, 6],
        },
        frozen_atoms  = {0, 1, 5, 6},  # electrodes pinned
    )


# ----------------------------------------------------------------- #
#  Fix 1: ATOM-METADATA block in transport .fdf                      #
# ----------------------------------------------------------------- #


def test_render_emits_atom_metadata_block_when_regions_present():
    struct = _mk_device_struct()
    cfg = TransportConfig(job_name="dev")
    text = TransiestaEngine.render_script(struct, cfg)
    # The block uses the canonical molbuilder marker syntax
    # (script_emit.begin_marker / end_marker).
    assert "# === molbuilder atom-metadata BEGIN ===" in text
    assert "# === molbuilder atom-metadata END ==="   in text
    # The Python region labels survive verbatim in the JSON payload
    # (electrode block names get sanitized in TS.Elec.<name> blocks,
    # but the source-of-truth label is in the metadata block).
    assert "L-electrode" in text
    assert "M-bridge"    in text
    assert "R-electrode" in text


def test_render_atom_metadata_block_carries_frozen_atoms():
    struct = _mk_device_struct()
    cfg = TransportConfig(job_name="dev")
    text = TransiestaEngine.render_script(struct, cfg)
    # frozen_atoms appears as a JSON list in the metadata payload.
    # The block is indented with "# " comment prefixes per line.
    assert '"frozen_atoms"' in text


def test_render_omits_atom_metadata_when_no_labels():
    """A structure with no regions + no frozen_atoms must NOT carry
    an empty metadata block -- the contract says emit only when there
    is something to declare."""
    struct = Structure(
        elements      = ["Au", "Au"],
        positions     = np.array([[0, 0, 0], [0, 0, 2]], dtype=float),
        atom_names    = ["Au1", "Au2"],
        residue_ids   = [1, 1],
        residue_names = ["UNL", "UNL"],
        chain_ids     = ["A", "A"],
    )
    # No regions => preflight refuses to render a transport device.
    # Exercise just the emitter directly.
    from molbuilder.transport.transiesta import (
        _emit_header, _emit_geometry,
    )
    cfg = TransportConfig(job_name="dev")
    text = "\n".join(_emit_header(cfg, struct) + _emit_geometry(struct))
    assert "atom-metadata BEGIN" not in text


# ----------------------------------------------------------------- #
#  Fix 2: # runtime.* echo lines in transport header                 #
# ----------------------------------------------------------------- #


def test_header_echoes_max_memory_mb_when_set():
    struct = _mk_device_struct()
    cfg = TransportConfig(job_name="dev", max_memory_mb=16000)
    text = TransiestaEngine.render_script(struct, cfg)
    assert "# runtime.max_memory_mb: 16000" in text


def test_header_echoes_num_threads_as_omp_threads_requested():
    """Key name MUST match SIESTA's convention (omp_threads_requested)
    so the shared SIESTA-side parser regex at
    parse/engines/siesta.py:1407 reads both engines uniformly."""
    struct = _mk_device_struct()
    cfg = TransportConfig(job_name="dev", num_threads=8)
    text = TransiestaEngine.render_script(struct, cfg)
    assert "# runtime.omp_threads_requested: 8" in text


def test_header_runtime_lines_match_siesta_parser_regex():
    """End-to-end: feed the rendered .fdf through the SIESTA parser's
    runtime probe and verify the keys round-trip."""
    from molbuilder.parse.engines.siesta import _SIESTA_RUNTIME_RE
    struct = _mk_device_struct()
    cfg = TransportConfig(job_name="dev", num_threads=4,
                          max_memory_mb=12000)
    text = TransiestaEngine.render_script(struct, cfg)
    captured = {}
    for line in text.splitlines():
        m = _SIESTA_RUNTIME_RE.match(line)
        if m:
            captured[m.group(1)] = m.group(2).strip()
    assert captured.get("omp_threads_requested") == "4"
    assert captured.get("max_memory_mb")         == "12000"


# ----------------------------------------------------------------- #
#  Fix 3: TransportResults.regions + .frozen_atoms                   #
# ----------------------------------------------------------------- #


def test_schema_version_bumped_to_2():
    assert SCHEMA_VERSION == "2"


def test_results_default_init_empty_boundary():
    r = TransportResults()
    assert r.regions      == {}
    assert r.frozen_atoms == []


def test_results_roundtrip_with_boundary_conditions():
    r = TransportResults(
        energy_grid_eV=np.array([-1.0, 0.0, 1.0]),
        transmission=np.array([0.1, 0.5, 0.9]),
        regions={"L-electrode": [0, 1], "M-bridge": [2, 3, 4],
                 "R-electrode": [5, 6]},
        frozen_atoms=[0, 1, 5, 6],
    )
    d = r.to_dict()
    assert d["schema_version"]  == "2"
    assert d["regions"]["L-electrode"]   == [0, 1]
    assert d["regions"]["M-bridge"]      == [2, 3, 4]
    assert d["regions"]["R-electrode"]   == [5, 6]
    assert d["frozen_atoms"]             == [0, 1, 5, 6]
    # Round-trip back.
    r2 = TransportResults.from_dict(d)
    assert r2.regions      == r.regions
    assert r2.frozen_atoms == r.frozen_atoms


def test_v1_sidecar_reads_as_empty_boundary():
    """Back-compat: a pre-2026-06-25 .transport.json with schema_version
    == '1' must decode without exception; missing boundary fields
    default to empty (NOT raise)."""
    v1_payload = {
        "schema_version":     "1",
        "metadata":           {"engine": "transiesta"},
        "energy_grid_eV":     [-1.0, 0.0, 1.0],
        "transmission":       [0.1, 0.5, 0.9],
        "fermi_energy_eV":    0.0,
        "conductance_G0":     0.5,
        "pdos":               {},
        "bias_grid_V":        None,
        "current_uA":         None,
        "methods_text":       "Transport via TranSIESTA",
        "bibliography_keys":  ["transiesta_brandbyge_2002"],
        "complete":           True,
    }
    r = TransportResults.from_dict(v1_payload)
    assert r.regions      == {}
    assert r.frozen_atoms == []
    assert r.complete is True
    # Re-emitting bumps to v2.
    assert r.to_dict()["schema_version"] == "2"


def test_unknown_schema_version_still_raises():
    """Forward-compat: a future v3+ payload must NOT silently degrade."""
    with pytest.raises(ValueError, match="unknown schema_version"):
        TransportResults.from_dict({"schema_version": "999"})


def test_regions_serialise_sorted_for_determinism():
    r = TransportResults(
        regions={"L-electrode": [3, 1, 2], "R-electrode": [9, 7]},
    )
    d = r.to_dict()
    # Each region's indices are sorted ascending in the wire form so
    # diff'ing two .transport.json files is robust against the
    # in-memory ordering.
    assert d["regions"]["L-electrode"] == [1, 2, 3]
    assert d["regions"]["R-electrode"] == [7, 9]
