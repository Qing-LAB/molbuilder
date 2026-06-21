"""Sanity check for the SIESTA output parser.

Builds a tiny synthetic SIESTA-style output that contains the same
markers as the real thing -- two complete CG steps plus a truncated
third step that is mid-writing its outcoor block -- and verifies that
the parser keeps two complete steps with the right numbers.
"""

from __future__ import annotations

import json
import math

import pytest

from molbuilder.parse.engines._helpers import trajectory_to_legacy_dict
from molbuilder.parse.engines.siesta import SiestaParser


SAMPLE = """\
Welcome to SIESTA -- some header noise
redata: Max. number of TDED Iter        =        1
redata: Max. number of SCF Iter                     =      500
redata: Maximum number of optimization moves        =      200

                     ====================================
                        Begin CG opt. move =      0
                     ====================================

outcoor: Atomic coordinates (Ang):
   1.00000000    2.00000000    3.00000000   1       1  C
   4.00000000    5.00000000    6.00000000   2       2  H

outcell: Unit cell vectors (Ang):
       10.000000    0.000000    0.000000
        0.000000   10.000000    0.000000
        0.000000    0.000000   10.000000

siesta: Eharris =   -289239.010387

   scf:    1  -100.0  -100.0  -100.0  0.001 -1.0 0.5
SCF Convergence by DM+H criterion

siesta: E_KS(eV) =          -100.1234

siesta: Atomic forces (eV/Ang):
     1    0.10    0.20    0.30
     2    0.40    0.50    0.60
----------------------------------------
   Tot    0.50    0.70    0.90
----------------------------------------
   Max    1.234567
   Res    0.987654    sqrt( Sum f_i^2 / 3N )
----------------------------------------
   Max    1.234567    constrained



                     ====================================
                        Begin CG opt. move =      1
                     ====================================

outcoor: Atomic coordinates (Ang):
   1.10000000    2.10000000    3.10000000   1       1  C
   4.10000000    5.10000000    6.10000000   2       2  H

   scf:    1  -101.0  -101.0  -101.0  0.001 -1.0 0.5
SCF Convergence by DM+H criterion

siesta: E_KS(eV) =          -101.5678

siesta: Atomic forces (eV/Ang):
     1    0.05    0.06    0.07
     2    0.08    0.09    0.10
----------------------------------------
   Tot    0.13    0.15    0.17
----------------------------------------
   Max    0.987654
   Res    0.123456    sqrt( Sum f_i^2 / 3N )
----------------------------------------
   Max    0.987654    constrained

                     ====================================
                        Begin CG opt. move =      2
                     ====================================

outcoor: Atomic coordinates (Ang):
   1.20000000    2.20000000    3.20000000   1       1  C
"""


@pytest.fixture
def siesta_path(tmp_path):
    p = tmp_path / "run.out"
    p.write_text(SAMPLE)
    return str(p)


def test_can_parse(siesta_path):
    assert SiestaParser.can_parse(siesta_path) is True


def test_can_parse_rejects_non_siesta(tmp_path):
    p = tmp_path / "garbage.txt"
    p.write_text("just some random text\nhello world\n")
    assert SiestaParser.can_parse(str(p)) is False


def test_can_parse_siesta_v5_banner(tmp_path):
    """SIESTA 5.x changed the banner: line 1 is `Executable      : siesta`
    and the welcome banner reads `*  WELCOME TO SIESTA  *` (uppercase,
    surrounded by asterisks).  The mixed-case `Welcome to SIESTA` marker
    that worked for v4.x doesn't match.  Either v5 marker must be
    sufficient on its own -- on a real v5 run, the later-but-stable
    markers (`siesta: System type`, `redata:`) only appear hundreds of
    lines into the file, past the parser's 80-line scan window."""
    v5_head = (
        "Executable      : siesta\n"
        "Version         : 5.4.2-11-g4e9a46060\n"
        "Architecture    : x86_64\n"
        "Compiler version: GNU-13.3.0\n"
        "Compiler flags  : -O3\n"
        "Parallelisations: MPI\n"
        "Lua support\n"
        "DFT-D3 support\n"
        "\n"
        "Runtime information:\n"
        "* Directory : /tmp/run\n"
        "* Running in serial mode.\n"
        ">> Start of run:  28-APR-2026  20:17:06\n"
        "\n"
        "                           *********************** \n"
        "                           *  WELCOME TO SIESTA  * \n"
        "                           *********************** \n"
        "\n"
        "(...lots of output before any `redata:` line ...)\n"
    )
    p = tmp_path / "v5.out"
    p.write_text(v5_head)
    assert SiestaParser.can_parse(str(p)) is True


def test_torn_frame_dropped_at_eof(siesta_path):
    result = trajectory_to_legacy_dict(SiestaParser.parse(siesta_path))
    assert len(result["frames"]) == 2


def test_frame_coordinates(siesta_path):
    result = trajectory_to_legacy_dict(SiestaParser.parse(siesta_path))
    assert result["frames"][0] == [
        ["C", 1.0, 2.0, 3.0],
        ["H", 4.0, 5.0, 6.0],
    ]
    assert result["frames"][1] == [
        ["C", 1.1, 2.1, 3.1],
        ["H", 4.1, 5.1, 6.1],
    ]


def test_energies(siesta_path):
    result = trajectory_to_legacy_dict(SiestaParser.parse(siesta_path))
    assert math.isclose(result["energies"][0], -100.1234)
    assert math.isclose(result["energies"][1], -101.5678)


def test_max_forces_captures_both_unconstrained_and_constrained(siesta_path):
    """2026-06-12: SIESTA emits TWO ``Max <val>`` lines per relax step
    when at least one atom is constrained — the second carries the
    ``constrained`` suffix and is what SIESTA actually compares
    against ``MD.MaxForceTol`` for convergence.  Capture both so the
    Results plot can show both traces (and the user can see which one
    is the meaningful "did we converge?" signal).
    """
    result = trajectory_to_legacy_dict(SiestaParser.parse(siesta_path))
    # Unconstrained (over ALL atoms).
    assert math.isclose(result["max_forces"][0], 1.234567)
    assert math.isclose(result["max_forces"][1], 0.987654)
    # Constrained (excludes frozen atoms) — fixture has the same
    # values in both forms (the max-force atom isn't frozen) but the
    # parser must still surface BOTH fields.
    assert math.isclose(result["max_forces_constrained"][0], 1.234567)
    assert math.isclose(result["max_forces_constrained"][1], 0.987654)


def test_max_forces_constrained_empty_when_no_constrained_line(tmp_path):
    """Negative path: a run with NO constrained-line entries (no
    frozen atoms) collapses ``max_forces_constrained`` to a top-level
    empty list — saves JSON bytes AND signals "no constraints in
    this run" to downstream consumers without scanning for null
    entries.
    """
    # Hand-rolled minimal output: 1 frame, NO constrained line.
    path = tmp_path / "norest.out"
    path.write_text(
        "Siesta Version: 5.4.2\n"
        "outcoor: Atomic coordinates (Ang):\n"
        "    0.0000    0.0000    0.0000   1\n"
        "    0.7400    0.0000    0.0000   1\n"
        "%block ChemicalSpeciesLabel\n1 1 H\n%endblock\n"
        "siesta: Atomic forces (eV/Ang):\n"
        "siesta:    1    0.1234    0.0000    0.0000\n"
        "siesta:    2   -0.1234    0.0000    0.0000\n"
        "   Max    0.1234\n"
        "   Res    0.1234    sqrt( Sum f_i^2 / 3N )\n"
        "siesta: E_KS(eV) = -1.0\n"
        "siesta: program end\n"
    )
    result = trajectory_to_legacy_dict(SiestaParser.parse(str(path)))
    # JSON collapse: no constrained values anywhere → []
    assert result["max_forces_constrained"] == [], (
        f"expected empty list (no constraints), got "
        f"{result['max_forces_constrained']!r}"
    )


def test_per_atom_forces(siesta_path):
    result = trajectory_to_legacy_dict(SiestaParser.parse(siesta_path))
    assert result["forces"][0] == [[0.10, 0.20, 0.30], [0.40, 0.50, 0.60]]
    assert result["forces"][1] == [[0.05, 0.06, 0.07], [0.08, 0.09, 0.10]]


def test_lattice_captured(siesta_path):
    result = trajectory_to_legacy_dict(SiestaParser.parse(siesta_path))
    assert result["lattice"] == [
        [10.0,  0.0,  0.0],
        [ 0.0, 10.0,  0.0],
        [ 0.0,  0.0, 10.0],
    ]


def test_iterations(siesta_path):
    result = trajectory_to_legacy_dict(SiestaParser.parse(siesta_path))
    assert result["iterations"] == [0, 1]


def test_source_format_tag(siesta_path):
    result = trajectory_to_legacy_dict(SiestaParser.parse(siesta_path))
    assert result["source_format"] == "siesta"


def test_scf_history_default_empty(tmp_path):
    """A SIESTA log with no scf: lines (header noise only) should
    yield scf_history=[]."""
    p = tmp_path / "noisy.out"
    p.write_text("Welcome to SIESTA\nredata: blah\n")
    result = trajectory_to_legacy_dict(SiestaParser.parse(str(p)))
    assert result["scf_history"] == []


def test_scf_history_collects_per_cycle(siesta_path):
    """The SAMPLE has one scf: line per CG step, so each step's
    history list has length 1."""
    result = trajectory_to_legacy_dict(SiestaParser.parse(siesta_path))
    runs = result["scf_history"]
    # Two CG steps in SAMPLE; each has exactly one scf: line.
    assert len(runs) == 2
    assert all(len(r) == 1 for r in runs)


def test_scf_history_per_cycle_keys(siesta_path):
    """Each per-cycle entry must have the SIESTA key set."""
    runs = trajectory_to_legacy_dict(SiestaParser.parse(siesta_path))["scf_history"]
    expected = {"cycle", "energy", "delta_E", "dHmax", "dDmax"}
    for run in runs:
        for entry in run:
            assert set(entry.keys()) == expected


def test_scf_history_real_multi_cycle_run(tmp_path):
    """A SIESTA-style run with multiple SCF iterations within one CG
    step splits correctly: iscf=1 marks each new run boundary.

    Each CG step needs its own outcoor block to emit a Frame; the
    parser attaches scf_history per-Frame (no Frame -> no scf
    history).  The realistic SIESTA stream is "SCF -> outcoor -> SCF
    -> outcoor -> ..."; we follow that here.
    """
    # Real SIESTA stream: outcoor (geometry for the step) -> SCF
    # iterations on that geometry -> next outcoor (post-CG-move
    # geometry) -> next SCF -> ...  We follow that order here so
    # commit() at the next outcoor attaches the just-finished SCF
    # to the correct Frame.
    sample = (
        "Welcome to SIESTA\n"
        "redata: prelude\n"
        # First CG step: outcoor first (the geometry), then 3 SCF
        # iterations that converge on that geometry.
        "outcoor: Atomic coordinates (Ang):\n"
        "   1.0  2.0  3.0   1   1  C\n"
        "\n"
        "   scf:    1   -100.0   -100.5   -100.5   0.10  -1.0   0.5\n"
        "   scf:    2   -100.4   -100.7   -100.7   0.05  -1.0   0.1\n"
        "   scf:    3   -100.45  -100.71  -100.71  0.01  -1.0   0.01\n"
        "SCF Convergence by DM+H criterion\n"
        # Second CG step: outcoor + 2 SCF iterations.  The next
        # outcoor (or EOF) is what commits step 0; iscf==1 of run 2
        # starts a fresh SCF that lands on Frame 1 at commit time.
        "outcoor: Atomic coordinates (Ang):\n"
        "   1.1  2.1  3.1   1   1  C\n"
        "\n"
        "   scf:    1   -101.0   -101.2   -101.2   0.08  -1.0   0.4\n"
        "   scf:    2   -101.1   -101.3   -101.3   0.02  -1.0   0.05\n"
        "SCF Convergence by DM+H criterion\n"
    )
    p = tmp_path / "multi.out"
    p.write_text(sample)
    runs = trajectory_to_legacy_dict(SiestaParser.parse(str(p)))["scf_history"]
    assert len(runs) == 2
    assert len(runs[0]) == 3
    assert len(runs[1]) == 2
    # Energy column is E_KS (eV); cycle 1 of run 1 has E_KS = -100.5
    import math as _math
    assert _math.isclose(runs[0][0]["energy"], -100.5)
    # delta_E for first cycle is 0; subsequent are differences.
    assert runs[0][0]["delta_E"] == 0.0
    assert _math.isclose(runs[0][1]["delta_E"], -100.7 - (-100.5))
    # dHmax column comes through:
    assert _math.isclose(runs[0][2]["dHmax"], 0.01)


def test_stray_max_line_outside_force_block_ignored(tmp_path):
    """Regression: a 'Max <num>' line that appears OUTSIDE a force
    block (e.g. in a header) must not be mis-attributed to the next
    step's max-force.  The gate is `step_forces` non-empty: only after
    the per-atom force block do we accept a Max line."""
    sample = (
        "Welcome to SIESTA -- v4.1\n"
        "redata: prelude\n"
        # Stray 'Max' line BEFORE any force block -- must be ignored.
        "   Max    9.999999\n"
        "\n"
        "outcoor: Atomic coordinates (Ang):\n"
        "   1.00000000    2.00000000    3.00000000   1       1  C\n"
        "\n"
        "siesta: E_KS(eV) =          -50.0000\n"
        # No 'siesta: Atomic forces' block, no real Max line.
    )
    p = tmp_path / "stray.out"
    p.write_text(sample)
    result = trajectory_to_legacy_dict(SiestaParser.parse(str(p)))
    # One frame, no max-force -- the stray 9.999 mustn't have been
    # attributed to it.
    assert len(result["frames"]) == 1
    assert result["max_forces"][0] is None


# --------------------------------------------------------------------- #
# Level-3 parser contract (2026-05-28)                                  #
#                                                                       #
# 1. Header-driven SCF column mapping (closed-shell vs spin-polarized   #
#    vs future SIESTA layouts).                                         #
# 2. Tight-packed fixed-width columns (Ef_dn + dHmax glued with no      #
#    separator) parse correctly -- the LAST column is dHmax.            #
# 3. Fail-soft on malformed lines: ParseWarning recorded, parse        #
#    continues, partial trajectory returned.                            #
# 4. Warnings round-trip through trajectory_to_legacy_dict.             #
# --------------------------------------------------------------------- #


def test_scf_header_drives_column_mapping_spin_polarized(tmp_path):
    """v5 spin-polarized SIESTA emits an 8-column SCF block.  The
    parser's header line tells us where dHmax lives; the data row's
    LAST column is dHmax (not 7th column).  Pin this so we never
    again confuse Ef_dn with dHmax (2026-05-28 sighting)."""
    sample = (
        "Welcome to SIESTA -- v5\n"
        "redata: prelude\n"
        "outcoor: Atomic coordinates (Ang):\n"
        "   1.0  2.0  3.0   1   1  C\n"
        "\n"
        # Real v5 spin-polarized header.
        "        iscf     Eharris(eV)        E_KS(eV)     "
        "FreeEng(eV)     dDmax     Ef_up Ef_dn(eV) dHmax(eV)\n"
        # iscf 1: 8 columns after the scf: prefix.
        "   scf:    1  -100.0  -100.5  -100.5  0.10  -1.0  -1.5  4.2\n"
        # iscf 2: same shape; dHmax is the LAST column.
        "   scf:    2  -100.4  -100.7  -100.7  0.05  -1.0  -1.5  0.1\n"
        "outcoor: Atomic coordinates (Ang):\n"
        "   1.1  2.1  3.1   1   1  C\n"
    )
    p = tmp_path / "spin.out"
    p.write_text(sample)
    legacy = trajectory_to_legacy_dict(SiestaParser.parse(str(p)))
    runs = legacy["scf_history"]
    assert len(runs) == 1
    assert len(runs[0]) == 2
    # dHmax must be the LAST column (4.2, 0.1) -- NOT Ef_dn (-1.5).
    assert math.isclose(runs[0][0]["dHmax"], 4.2)
    assert math.isclose(runs[0][1]["dHmax"], 0.1)
    # E_KS comes through as 'energy'.
    assert math.isclose(runs[0][0]["energy"], -100.5)
    # dDmax comes through.
    assert math.isclose(runs[0][0]["dDmax"], 0.10)


def test_scf_header_drives_column_mapping_closed_shell(tmp_path):
    """Closed-shell run: 7-column header (Ef instead of Ef_up/Ef_dn);
    dHmax still the last column."""
    sample = (
        "Welcome to SIESTA\n"
        "redata: prelude\n"
        "outcoor: Atomic coordinates (Ang):\n"
        "   1.0  2.0  3.0   1   1  C\n"
        "\n"
        "        iscf     Eharris(eV)        E_KS(eV)     "
        "FreeEng(eV)     dDmax     Ef(eV) dHmax(eV)\n"
        "   scf:    1  -100.0  -100.5  -100.5  0.10  -1.0  3.5\n"
        "outcoor: Atomic coordinates (Ang):\n"
        "   1.1  2.1  3.1   1   1  C\n"
    )
    p = tmp_path / "closed.out"
    p.write_text(sample)
    legacy = trajectory_to_legacy_dict(SiestaParser.parse(str(p)))
    cycle = legacy["scf_history"][0][0]
    assert math.isclose(cycle["dHmax"], 3.5)
    assert math.isclose(cycle["energy"], -100.5)


def test_scf_tight_packed_columns_handled(tmp_path):
    """SIESTA's fixed-width f10.6 format can leave NO whitespace
    between adjacent columns when both values fill their fields --
    the dHmax/Ef_dn pair in spin-polarized output.  Parser must
    insert the separator + still extract dHmax correctly (the LAST
    column, which gets glued onto its neighbour's tail)."""
    sample = (
        "Welcome to SIESTA -- v5\n"
        "redata: prelude\n"
        "outcoor: Atomic coordinates (Ang):\n"
        "   1.0  2.0  3.0   1   1  C\n"
        "\n"
        "        iscf     Eharris(eV)        E_KS(eV)     "
        "FreeEng(eV)     dDmax     Ef_up Ef_dn(eV) dHmax(eV)\n"
        # The crucial line: Ef_dn=-1.929956 and dHmax=131.029438 with
        # NO space between them (the exact format SIESTA produces).
        "   scf:    1   -12063.872757   -12728.905077   "
        "-12728.908102  0.982890 -3.184575 -1.929956131.029438\n"
        "outcoor: Atomic coordinates (Ang):\n"
        "   1.1  2.1  3.1   1   1  C\n"
    )
    p = tmp_path / "tightpack.out"
    p.write_text(sample)
    result = SiestaParser.parse(str(p))
    legacy = trajectory_to_legacy_dict(result)
    cycle = legacy["scf_history"][0][0]
    # dHmax (the legitimate 131.029438) — NOT Ef_dn (-1.929956).
    assert math.isclose(cycle["dHmax"], 131.029438)
    # And no spurious warnings -- the tight-pack normalization is
    # silent (the fix is invisible to the user).
    assert legacy["parse_warnings"] == []


def test_parse_warnings_on_malformed_scf_line(tmp_path):
    """A single bad SCF line must not abort the parse: it becomes a
    ParseWarning + parse continues.  Good lines before AND after the
    bad one show up in the trajectory."""
    sample = (
        "Welcome to SIESTA\n"
        "redata: prelude\n"
        "outcoor: Atomic coordinates (Ang):\n"
        "   1.0  2.0  3.0   1   1  C\n"
        "\n"
        "        iscf     Eharris(eV)        E_KS(eV)     "
        "FreeEng(eV)     dDmax     Ef(eV) dHmax(eV)\n"
        "   scf:    1  -100.0  -100.5  -100.5  0.10  -1.0  3.5\n"
        # Bad line: non-numeric value.
        "   scf:    2  BANANA  -100.7\n"
        "   scf:    3  -100.45 -100.71 -100.71 0.01 -1.0 0.01\n"
        "outcoor: Atomic coordinates (Ang):\n"
        "   1.1  2.1  3.1   1   1  C\n"
    )
    p = tmp_path / "malformed.out"
    p.write_text(sample)
    result = SiestaParser.parse(str(p))
    legacy = trajectory_to_legacy_dict(result)
    # Bad cycle 2 dropped; cycles 1 and 3 preserved.
    cycles = [c["cycle"] for c in legacy["scf_history"][0]]
    assert cycles == [1, 3]
    # Exactly one ParseWarning, in the "scf" category, with the bad
    # line's snippet quoted.
    assert len(legacy["parse_warnings"]) == 1
    w = legacy["parse_warnings"][0]
    assert w["category"] == "scf"
    assert "BANANA" in w["snippet"]
    assert w["line_no"] > 0


def test_scf_header_is_case_insensitive_and_unit_tolerant(tmp_path):
    """Parser robustness: SCF column header tokens match regardless
    of capitalisation and regardless of whether the ``(unit)``
    suffix is present.  ``DHMAX``, ``dhmax``, ``dHmax(eV)`` all
    resolve to the canonical key.

    This is the "names should be immune to capitalisation, small
    spelling differences" rule (2026-05-28 user signal).
    """
    sample = (
        "Welcome to SIESTA\n"
        "redata: prelude\n"
        "outcoor: Atomic coordinates (Ang):\n"
        "   1.0  2.0  3.0   1   1  C\n"
        "\n"
        # Header in unusual capitalisation + missing units:
        "  ISCF  EHARRIS  E_KS  FREEENG  DDMAX  EF  DHMAX\n"
        "   scf:    1  -100.0  -100.5  -100.5  0.10  -1.0  3.5\n"
        "outcoor: Atomic coordinates (Ang):\n"
        "   1.1  2.1  3.1   1   1  C\n"
    )
    p = tmp_path / "case.out"
    p.write_text(sample)
    legacy = trajectory_to_legacy_dict(SiestaParser.parse(str(p)))
    cycle = legacy["scf_history"][0][0]
    # The canonical keys still produce dHmax = last column (3.5)
    # and energy = E_KS (column 2 value of the row, = -100.5).
    assert math.isclose(cycle["dHmax"], 3.5)
    assert math.isclose(cycle["energy"], -100.5)
    assert math.isclose(cycle["dDmax"], 0.10)
    # No warnings — the case-insensitive lookup is transparent.
    assert legacy["parse_warnings"] == []


def test_parse_warnings_round_trip_through_legacy_dict(tmp_path):
    """Every ParseWarning must serialise into the legacy dict so the
    Results-tab UI can render them.  Schema: line_no/snippet/error/
    category, all simple types."""
    sample = (
        "Welcome to SIESTA\n"
        "redata: prelude\n"
        "outcoor: Atomic coordinates (Ang):\n"
        "   1.0  2.0  3.0   1   1  C\n"
        "\n"
        "        iscf     Eharris(eV)        E_KS(eV)     "
        "FreeEng(eV)     dDmax     Ef(eV) dHmax(eV)\n"
        # Bad: not enough columns.
        "   scf:    1  -100.0\n"
        "outcoor: Atomic coordinates (Ang):\n"
        "   1.1  2.1  3.1   1   1  C\n"
    )
    p = tmp_path / "warn.out"
    p.write_text(sample)
    legacy = trajectory_to_legacy_dict(SiestaParser.parse(str(p)))
    warnings = legacy["parse_warnings"]
    assert len(warnings) == 1
    for k in ("line_no", "snippet", "error", "category"):
        assert k in warnings[0], f"missing key {k!r} in warning"
    assert isinstance(warnings[0]["line_no"], int)
    # The entire dict round-trips through JSON cleanly.
    s = json.dumps(legacy)
    assert "parse_warnings" in s


def test_json_safe_no_nan(siesta_path):
    """Result must serialise with strict JSON (no NaN)."""
    result = trajectory_to_legacy_dict(SiestaParser.parse(siesta_path))
    json.dumps(result, allow_nan=False)
