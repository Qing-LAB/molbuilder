"""SIESTA parser: case-insensitive + alias-tolerant section markers.

These tests pin the 2026-05-29 rule-engine refactor that brought
case-insensitivity (the user's standing 2026-05-28 directive: "the
detection of names should be immune to capitalization, small spelling
differences etc.").

What we test:

  * Uppercase section headers parse identically to lowercase.
  * Mixed-case section headers parse identically.
  * Indented section headers parse identically (whitespace stripped).

What we DO NOT test (deliberate scope limit, locked 2026-05-29):

  * Levenshtein / fuzzy-typo tolerance.  We accept case drift, not
    spelling drift -- the realistic SIESTA version-skew case is
    capitalisation; tolerating typos would invite false positives.
"""
from __future__ import annotations

import math
import tempfile

import numpy as np
import pytest

from molbuilder.parsers.siesta import SiestaParser


def _parse(text: str):
    with tempfile.NamedTemporaryFile("w", suffix=".out", delete=False) as fh:
        fh.write(text)
        path = fh.name
    return SiestaParser.parse(path)


# A minimal SIESTA-style fragment with all the section markers we
# care about.  Two complete steps with coords + cell + E_KS + forces
# + max-force; one SCF cycle in step 0.  The CASE of the section
# markers is what each test mutates.
TEMPLATE = """\
Welcome to SIESTA
{outcoor_marker} Atomic coordinates (Ang):
   1.00000000    2.00000000    3.00000000   1       1  C
   4.00000000    5.00000000    6.00000000   2       2  H

{outcell_marker} Unit cell vectors (Ang):
       10.000000    0.000000    0.000000
        0.000000   10.000000    0.000000
        0.000000    0.000000   10.000000

   {scf_header_marker}    Eharris(eV)    E_KS(eV)   FreeEng(eV)   dDmax  Ef(eV) dHmax(eV)
   scf:    1   -100.0       -100.0       -100.0     0.001  -1.0   0.5

{ekv_prefix} E_KS(eV) =        -100.1234

{forces_prefix} Atomic forces (eV/Ang):
     1    0.10    0.20    0.30
     2    0.40    0.50    0.60
   Max    1.234567

{end_marker}: Tue Jan 14 01:23:45 2026
"""


@pytest.fixture
def baseline_text():
    """The canonical lower-case (real SIESTA) form.  Every variant
    in the tests below should yield identical Trajectory output."""
    return TEMPLATE.format(
        outcoor_marker="outcoor:",
        outcell_marker="outcell:",
        scf_header_marker="iscf",
        ekv_prefix="siesta:",
        forces_prefix="siesta:",
        end_marker=">> End of run",
    )


def _assert_full_parse(traj):
    """Helper: a complete one-frame parse with the expected fields."""
    assert len(traj.frames) == 1
    f = traj.frames[0]
    assert f.energy is not None
    assert math.isclose(f.energy, -100.1234, abs_tol=1e-6)
    assert f.max_force is not None
    assert math.isclose(f.max_force, 1.234567, abs_tol=1e-6)
    assert f.forces is not None
    assert f.forces.shape == (2, 3)
    assert f.scf_history is not None
    assert len(f.scf_history) == 1
    # Header was honoured: dHmax is the LAST column (0.5).
    assert math.isclose(f.scf_history[0]["dHmax"], 0.5, abs_tol=1e-6)
    # And the lattice was captured from outcell.
    assert traj.lattice is not None
    assert np.allclose(traj.lattice,
                       [[10.0, 0.0, 0.0],
                        [0.0, 10.0, 0.0],
                        [0.0, 0.0, 10.0]])
    # And run-state captured from ">> End of run".
    assert traj.run_state == "finished"


def test_baseline_lower_case(baseline_text):
    """Sanity: the canonical real-SIESTA form parses end to end."""
    _assert_full_parse(_parse(baseline_text))


def test_outcoor_uppercase():
    """``OUTCOOR:`` parses identically to ``outcoor:``."""
    text = TEMPLATE.format(
        outcoor_marker="OUTCOOR:",
        outcell_marker="outcell:",
        scf_header_marker="iscf",
        ekv_prefix="siesta:",
        forces_prefix="siesta:",
        end_marker=">> End of run",
    )
    _assert_full_parse(_parse(text))


def test_outcell_uppercase():
    """``OUTCELL: Unit cell vectors`` parses identically."""
    text = TEMPLATE.format(
        outcoor_marker="outcoor:",
        outcell_marker="OUTCELL:",
        scf_header_marker="iscf",
        ekv_prefix="siesta:",
        forces_prefix="siesta:",
        end_marker=">> End of run",
    )
    _assert_full_parse(_parse(text))


def test_outcoor_outcell_mixed_case():
    """A SIESTA build that capitalises FIRST letters parses fine."""
    text = TEMPLATE.format(
        outcoor_marker="Outcoor:",
        outcell_marker="Outcell:",
        scf_header_marker="iscf",
        ekv_prefix="siesta:",
        forces_prefix="siesta:",
        end_marker=">> End of run",
    )
    _assert_full_parse(_parse(text))


def test_scf_header_iscf_uppercase():
    """``ISCF`` and ``IsCf`` parse identically to ``iscf``.  Already
    covered by the column-detection regex test but worth repeating
    here at end-to-end scale."""
    text = TEMPLATE.format(
        outcoor_marker="outcoor:",
        outcell_marker="outcell:",
        scf_header_marker="ISCF",
        ekv_prefix="siesta:",
        forces_prefix="siesta:",
        end_marker=">> End of run",
    )
    _assert_full_parse(_parse(text))


def test_e_ks_marker_uppercase():
    """``SIESTA: E_KS(eV) = ...`` parses identically (substring matcher
    is case-insensitive)."""
    # The line containing E_KS is *inside* the template -- we tweak
    # the literal substring rather than going through ekv_prefix
    # because the prefix matcher captures the whole "siesta: E_KS(eV)"
    # sub-string.
    text = TEMPLATE.format(
        outcoor_marker="outcoor:",
        outcell_marker="outcell:",
        scf_header_marker="iscf",
        ekv_prefix="siesta:",
        forces_prefix="siesta:",
        end_marker=">> End of run",
    )
    text = text.replace("siesta: E_KS(eV)", "SIESTA: E_KS(eV)")
    _assert_full_parse(_parse(text))


def test_forces_marker_mixed_case():
    """``Siesta: Atomic Forces`` parses identically."""
    text = TEMPLATE.format(
        outcoor_marker="outcoor:",
        outcell_marker="outcell:",
        scf_header_marker="iscf",
        ekv_prefix="siesta:",
        forces_prefix="siesta:",
        end_marker=">> End of run",
    )
    text = text.replace("siesta: Atomic forces", "Siesta: Atomic Forces")
    _assert_full_parse(_parse(text))


def test_indented_section_markers():
    """SIESTA v5 may indent section headers by a few spaces.  The
    matchers strip leading whitespace before comparing."""
    text = TEMPLATE.format(
        outcoor_marker="  outcoor:",
        outcell_marker="  outcell:",
        scf_header_marker="iscf",
        ekv_prefix="  siesta:",
        forces_prefix="  siesta:",
        end_marker=">> End of run",
    )
    _assert_full_parse(_parse(text))


def test_end_of_run_uppercase():
    """``>> END OF RUN`` (hypothetical version) still flips run_state."""
    text = TEMPLATE.format(
        outcoor_marker="outcoor:",
        outcell_marker="outcell:",
        scf_header_marker="iscf",
        ekv_prefix="siesta:",
        forces_prefix="siesta:",
        end_marker=">> END OF RUN",
    )
    traj = _parse(text)
    assert traj.run_state == "finished"


def test_end_of_run_mid_coords_still_captured():
    """The END_BUBBLE escape hatch: if SIESTA aborts mid-coords-write
    and the next line is the run-end marker (no blank-line
    terminator), the rule engine still captures run_state.  This
    pins the 2026-05-29 fix where coords' malformed-line path
    re-feeds the line through scan rules instead of dropping it."""
    torn = (
        "Welcome to SIESTA\n"
        "outcoor: Atomic coordinates (Ang):\n"
        "   1.00    2.00    3.00   1       1  C\n"   # one valid atom
        ">> End of run: Tue Jan 14 01:23:45 2026\n"  # no blank-line gap
    )
    traj = _parse(torn)
    # End-of-run was captured even though it appeared mid-coords.
    assert traj.run_state == "finished"


def test_outcell_immediately_after_coords_no_blank_line():
    """Defensive: if a hypothetical future SIESTA emits outcell:
    immediately after outcoor: (no blank line), the END_BUBBLE
    semantics ensure both sections parse."""
    no_blank = (
        "Welcome to SIESTA\n"
        "outcoor: Atomic coordinates (Ang):\n"
        "   1.00    2.00    3.00   1       1  C\n"
        "outcell: Unit cell vectors (Ang):\n"   # no blank line above
        "       10.000000    0.000000    0.000000\n"
        "        0.000000   10.000000    0.000000\n"
        "        0.000000    0.000000   10.000000\n"
        ">> End of run: now\n"
    )
    traj = _parse(no_blank)
    # outcoor was captured...
    assert len(traj.frames) == 1
    assert traj.frames[0].structure.elements == ["C"]
    # ...AND outcell was captured (the line that ended outcoor was
    # re-fed through scan rules where the outcell rule matched it).
    assert traj.lattice is not None
    assert np.allclose(traj.lattice,
                       [[10.0, 0.0, 0.0],
                        [0.0, 10.0, 0.0],
                        [0.0, 0.0, 10.0]])


def test_comment_line_with_outcoor_does_not_enter_section():
    """A line that mentions 'outcoor:' in a comment (preceded by '#'
    or 'the ') must NOT start the coords section.  Pin the
    starts_with_ci contract."""
    sneaky = (
        "Welcome to SIESTA\n"
        "# outcoor: this is a comment, not a real section\n"
        "the outcoor: pattern is mentioned mid-line; ignore it\n"
        "outcoor: Atomic coordinates (Ang):\n"
        "   1.0    2.0    3.0   1       1  C\n"
        "\n"
        ">> End of run: now\n"
    )
    traj = _parse(sneaky)
    # Exactly one frame from the *real* outcoor block -- the comment
    # lines were ignored.
    assert len(traj.frames) == 1
    assert traj.frames[0].structure.elements == ["C"]
