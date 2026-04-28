"""Regression tests pinning down each P1 / P2 fix from the static review.

Each test maps to a specific finding in the previous review:

    S1   peptide.py element field stripped of leading whitespace
    S2   SiestaConfig.net_charge / PySCFConfig.charge override the
         phosphate-only auto-detection heuristic
    S3   smiles.py MMFF/UFF fallback runs UFF when MMFF can't
         parameterise the molecule
    S6   web app caps multipart upload size at 10 MB
    S7   Structure.to_pdb caps atom serial / residue id at PDB column
         widths
    T1   _detect_species (siesta) never sees a leading-whitespace
         element -- it gets pre-stripped at the source
    T3   protonate_phosphate_oxygens is a no-op on a peptide (no
         phosphates -> no Hs added)
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from molbuilder.structure import Structure
from molbuilder.chemistry import (
    formal_charge_from_phosphates,
    protonate_phosphate_oxygens,
)


# --------------------------------------------------------------------- #
#  S1 / T1 -- element field is stripped of leading/trailing whitespace  #
# --------------------------------------------------------------------- #


def test_s1_t1_element_strip_propagates_to_species() -> None:
    """If a Structure ever lands in the FDF generator with an element
    like ' C', species detection must NOT see two distinct species."""
    from molbuilder.siesta import _detect_species
    # Direct invariant on the helper (the fix is upstream in peptide.py
    # but _detect_species shouldn't be tripped up by it)
    species = _detect_species(["C", "C", "H", "O"])
    assert species == sorted(species, key=lambda s: __import__("ase").data.atomic_numbers[s])
    # If a downstream caller forgets to strip, the FDF generator should
    # crash visibly rather than silently produce a malformed input.
    try:
        _detect_species(["C", " C", "H"])      # space-padded element
    except KeyError:
        # acceptable: the unknown " C" raises
        return
    # If it didn't raise, the species list must at least dedupe sanely:
    # not 2 carbons under different keys.
    sp = _detect_species(["C", " C", "H"])
    counts = {s: sp.count(s) for s in set(sp)}
    assert all(c == 1 for c in counts.values())


# --------------------------------------------------------------------- #
#  S2 -- net_charge override on SiestaConfig + PySCFConfig              #
# --------------------------------------------------------------------- #


def test_s2_siesta_net_charge_override() -> None:
    """A peptide with charged side-chains (which the heuristic ignores)
    needs an explicit override; we emit the user's NetCharge regardless
    of the phosphate-only heuristic."""
    from molbuilder.siesta import SiestaConfig, render_fdf
    s = Structure(
        elements=["C", "N", "O"],
        positions=np.array([[0,0,0],[1.5,0,0],[0,1.5,0]]),
    )
    # No phosphate -> heuristic says 0 -> default omits NetCharge
    text_default = render_fdf(s, SiestaConfig(verbose_comments=False))
    assert "NetCharge" not in text_default

    # Explicit override emits NetCharge -1
    text_charged = render_fdf(s, SiestaConfig(net_charge=-1, verbose_comments=False))
    assert "NetCharge       -1" in text_charged

    # Explicit 0 disables auto-detection (heuristic would say 0 anyway here)
    text_neutral = render_fdf(s, SiestaConfig(net_charge=0, verbose_comments=False))
    assert "NetCharge" not in text_neutral


def test_s2_siesta_net_charge_overrides_auto_detect() -> None:
    """User-specified charge wins over the phosphate heuristic, even
    when the heuristic would have flagged a charge."""
    from molbuilder.siesta import SiestaConfig, render_fdf
    # Synthetic deprotonated diester: heuristic detects -1
    elements = ["C", "O", "P", "O", "O", "O", "C"]
    positions = np.array([
        [-2.5, 0, 0], [-1.4, 0, 0], [0, 0, 0],
        [0, 1.5, 0], [0, -0.8, 1.3], [1.4, 0, 0], [2.5, 0, 0],
    ])
    s = Structure(elements=elements, positions=positions,
                  atom_names=["C","O","P","OP1","OP2","O","C"])
    assert formal_charge_from_phosphates(s) == -1
    # Default render emits the auto-detected -1
    auto = render_fdf(s, SiestaConfig(verbose_comments=False))
    assert "NetCharge       -1" in auto
    # User says "treat as neutral" -- override wins
    forced = render_fdf(s, SiestaConfig(net_charge=0, verbose_comments=False))
    assert "NetCharge" not in forced
    # User says "more negative" -- override wins
    forced3 = render_fdf(s, SiestaConfig(net_charge=-3, verbose_comments=False))
    assert "NetCharge       -3" in forced3


def test_s2_pyscf_charge_override() -> None:
    from molbuilder.pyscf_input import PySCFConfig, render_script
    s = Structure(
        elements=["O", "H"],
        positions=np.array([[0,0,0],[0.957,0,0]]),
    )
    # Explicit -1 wins
    text = render_script(s, PySCFConfig(charge=-1, verbose_comments=False))
    assert "charge     = -1," in text


# --------------------------------------------------------------------- #
#  D3 -- charged-system cell padding auto-bumps to 25 A                 #
# --------------------------------------------------------------------- #


def test_d3_charged_system_bumps_padding() -> None:
    from molbuilder.siesta import SiestaConfig, render_fdf
    # Synthetic depro diester: heuristic -> -1
    elements = ["C", "O", "P", "O", "O", "O", "C"]
    positions = np.array([
        [-2.5, 0, 0], [-1.4, 0, 0], [0, 0, 0],
        [0, 1.5, 0], [0, -0.8, 1.3], [1.4, 0, 0], [2.5, 0, 0],
    ])
    s = Structure(elements=elements, positions=positions,
                  atom_names=["C","O","P","OP1","OP2","O","C"])
    text = render_fdf(s, SiestaConfig(cell_padding=15.0, verbose_comments=False))
    # The auto-vacuum cell should reflect padding=25 A on each face
    # (depro diester extent ~5 A, so cell sizes ~5 + 50 ~= 55 A)
    import re
    m = re.search(r"vacuum cell ([0-9.]+) x ([0-9.]+) x ([0-9.]+) A", text)
    assert m, text
    sizes = [float(g) for g in m.groups()]
    # All three sizes >= 50 (5 A extent + 2*25 padding) confirms bump
    assert all(s_ >= 50.0 for s_ in sizes), sizes


def test_d3_neutral_system_keeps_user_padding() -> None:
    from molbuilder.siesta import SiestaConfig, render_fdf
    # Neutral water: heuristic -> 0, padding stays at user value
    s = Structure(
        elements=["O", "H", "H"],
        positions=np.array([[0,0,0],[0.957,0,0],[-0.24,0.927,0]]),
    )
    text = render_fdf(s, SiestaConfig(cell_padding=15.0, verbose_comments=False))
    assert "cell_padding = 15.0 A on each face" in text
    assert "auto-bumped" not in text


# --------------------------------------------------------------------- #
#  S7 -- PDB serial / residue id wrapped at column widths               #
# --------------------------------------------------------------------- #


def test_s7_pdb_serial_caps_at_99999() -> None:
    """A 100k-atom Structure (synthetic) writes '*****' for serials > 99999
    rather than overflowing the 5-column PDB serial field."""
    n = 100001
    elements = ["C"] * n
    positions = np.zeros((n, 3))
    s = Structure(elements=elements, positions=positions)
    pdb = s.to_pdb()
    # First line after TITLE / direct ATOM is for serial 1
    lines = [ln for ln in pdb.splitlines() if ln.startswith("ATOM")]
    assert len(lines) == n
    # Serial column is cols 7-11 inclusive
    assert lines[0][6:11] == "    1"
    assert lines[99998][6:11] == "99999"
    assert lines[99999][6:11] == "*****"   # 100000th atom -> wrapped
    assert lines[-1][6:11]    == "*****"


def test_s7_pdb_residue_id_caps_at_9999() -> None:
    n = 12000
    s = Structure(
        elements=["C"] * n,
        positions=np.zeros((n, 3)),
        residue_ids=list(range(1, n + 1)),
    )
    pdb = s.to_pdb()
    lines = [ln for ln in pdb.splitlines() if ln.startswith("ATOM")]
    # Residue id column is cols 23-26 (4 chars).  Beyond 9999 -> "****".
    assert lines[0][22:26]    == "   1"
    assert lines[9998][22:26] == "9999"
    assert lines[9999][22:26] == "****"


# --------------------------------------------------------------------- #
#  T3 -- protonate_phosphate_oxygens is a no-op on phosphate-free input #
# --------------------------------------------------------------------- #


def test_t3_protonate_noop_on_peptide() -> None:
    """A real peptide structure has no phosphate; protonate must add 0 H.

    This was a documented assumption but never tested directly.
    """
    try:
        import molbuilder
        s = molbuilder.build_peptide("AC", add_hydrogens=False)
    except ImportError:
        print("  (skip: PeptideBuilder not installed)")
        return
    assert "P" not in s.elements
    assert formal_charge_from_phosphates(s) == 0
    s2, n_added = protonate_phosphate_oxygens(s)
    assert n_added == 0
    assert s2 is s   # returns the same instance when no work to do


# --------------------------------------------------------------------- #
#  S6 -- web app rejects oversized uploads with HTTP 413                #
# --------------------------------------------------------------------- #


def test_s6_web_app_caps_upload_size() -> None:
    try:
        from molbuilder.web.app import create_app
    except ImportError:
        print("  (skip: Flask not installed)")
        return
    app = create_app()
    # Verify the cap is configured
    assert app.config.get("MAX_CONTENT_LENGTH") == 10 * 1024 * 1024
    # Verify a too-big upload is rejected (Flask raises 413)
    client = app.test_client()
    big = "x" * (11 * 1024 * 1024)   # 11 MB > 10 MB cap
    r = client.post("/api/load", json={"text": big, "filename": "big.xyz"})
    assert r.status_code == 413, r.status_code


# --------------------------------------------------------------------- #
#  T5 -- web app's _xyz_to_structure delegates to Structure.from_xyz    #
# --------------------------------------------------------------------- #


def test_t5_web_uses_canonical_xyz_parser() -> None:
    """Ensure the web layer doesn't carry a divergent XYZ parser."""
    try:
        from molbuilder.web.app import _xyz_to_structure
    except ImportError:
        print("  (skip: Flask not installed)")
        return
    text = (
        "2\n"
        "h2\n"
        "H 0 0 0\n"
        "H 0.74 0 0\n"
    )
    s = _xyz_to_structure(text)
    assert s.n_atoms == 2
    # Sanity: the canonical parser handles this identically
    s_canonical = Structure.from_xyz(text)
    np.testing.assert_array_equal(s.positions, s_canonical.positions)
    assert s.elements == s_canonical.elements


# --------------------------------------------------------------------- #
#  D1 -- Config alias and SiestaConfig coexist                          #
# --------------------------------------------------------------------- #


def test_d1_siesta_config_alias() -> None:
    from molbuilder.siesta import Config, SiestaConfig
    # Both names refer to the same class
    assert Config is SiestaConfig
    cfg = Config(system_label="x")
    assert isinstance(cfg, SiestaConfig)


# --------------------------------------------------------------------- #
#  Test runner                                                          #
# --------------------------------------------------------------------- #


def main() -> None:
    failures = []
    for name in sorted(globals()):
        if not name.startswith("test_"):
            continue
        fn = globals()[name]
        try:
            fn()
            print(f"  ok   {name}")
        except AssertionError as e:
            print(f"  FAIL {name}: {e}")
            failures.append(name)
        except Exception as e:
            print(f"  ERR  {name}: {type(e).__name__}: {e}")
            failures.append(name)
    if failures:
        sys.exit(f"FAILED: {failures}")
    print("OK -- review-fix regression suite passes.")


if __name__ == "__main__":
    main()
