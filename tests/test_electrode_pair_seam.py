"""The electrode pair meets itself on the wrong registry — measured.

`science/junction-cell.md` § 3.1 tables the layer counts whose seam CONTINUES
the fcc crystal.  None of them reproduces through the CLI's own placement:
`--electrode` passes `sequence` but leaves `add_slab`'s `start_registry` at 0
on BOTH slabs, and its spec grammar has no field for it — so every junction the
CLI builds meets itself on the same registry.  Measured across three planes and
three layer counts: **eclipsed** on (100)/(110), **twin** on (111).  A twin
carries the RIGHT bond length, which § 3.1 says is exactly why a distance check
misses it.

The second test is the other half of the measurement and passes today:
`start_registry` already IS the control, so this is a wiring defect and not a
missing feature — `+z = 1`, `−z = 0` continues the crystal on both surfaces,
with the across-seam distance at the bulk `a/√2`.

*(Extracted 2026-09-15 from a walkthrough file that was withdrawn: its other
cases asserted an API-presence on a private name, pinned a claim no document
states, and used strict-xfail as a bug tracker.  These two assert a stated
contract's own table through `classify_seam`, which is why they survive.)*
"""
from __future__ import annotations

import numpy as np
import pytest


# --------------------------------------------------------------------- #
#  F9 — the CLI's electrode builder never continues the crystal          #
# --------------------------------------------------------------------- #

#: Interlayer spacing `d` per surface for Au at the builder's own default
#: lattice constant (a_experimental = 4.0782 Å): (111) a/√3, (100) a/2,
#: (110) a/(2√2).  `junction-cell.md` § 2.
_D = {"111": 2.35470, "100": 2.03910, "110": 1.44186}

#: `junction-cell.md` § 3.1's table: layers per side that CONTINUE the fcc
#: crystal across the cell boundary, measured "across the seam of a
#: translated slab, with c = z_span + d".
_CONTINUES = [("111", 3), ("111", 6), ("100", 4), ("100", 6),
              ("110", 4), ("110", 6)]


def _junction(plane, layers, *, registry_plus=0, registry_minus=0):
    """Two slabs facing each other, built the way `--electrode` builds them.

    Mirrors `cli.py`'s own call site: one `add_slab` per side, the side
    mapped to the walk through `_WALK_ALONG_GROWTH`, and `c` set to
    `z_span + d` as § 6.1 instructs the user to.  The only thing varied is
    `start_registry`, which the CLI leaves at its default.
    """
    from molbuilder.cell import classify_seam
    from molbuilder.modify import add_slab
    from molbuilder.structure import Structure

    ortho = plane != "111"          # FCC_ORTHOGONAL_CHOICES: only (111) is free
    struct = Structure(elements=["X"], positions=np.array([[0.0, 0.0, 0.0]]))
    for side, sign, seq, reg in (("+z", 1.0, "ABC", registry_plus),
                                 ("-z", -1.0, "ACB", registry_minus)):
        struct = add_slab(struct, element="Au", plane=plane,
                          size=(2, 2, layers), start_z=sign * 2.4,
                          grow=side, sequence=seq, start_registry=reg,
                          orthogonal=ortho)
    pos = np.asarray(struct.positions, dtype=float)
    metal = np.array([p for e, p in zip(struct.elements, pos) if e == "Au"])
    cell = np.array(struct.cell, dtype=float).copy()
    cell[2, 2] = float(metal[:, 2].max() - metal[:, 2].min()) + _D[plane]
    return classify_seam(metal, cell)


@pytest.mark.xfail(strict=True, reason=(
    "THE DEFAULT, and it is a decision now rather than a gap (2026-09-22). "
    "`--electrode` leaves `start_registry` at 0 on both slabs unless told "
    "otherwise, so a junction built without stating it meets itself on the "
    "same registry -- eclipsed on (100)/(110), TWIN on (111), where "
    "`junction-cell.md` 3.1 says all six of these continue the crystal. "
    "What CHANGED: the reason used to add 'and its spec grammar has no "
    "field for it', which was the actual defect and is fixed -- the spec "
    "takes `registry=A|B|C` beside `contact=`, per flag because the two "
    "sides need different values, and `test_cli.py::test_electrode_registry"
    "_is_per_slab_and_reaches_the_builder` drives the real CLI to prove it "
    "reaches the builder. Omitting it still means 0, so existing command "
    "lines are unchanged (user ruling), which is what keeps this xfail "
    "true. It is no longer a Fix: it is the documented default."))
@pytest.mark.parametrize("plane,layers", _CONTINUES)
def test_the_cli_electrode_pair_continues_the_crystal(plane, layers):
    """§ 3.1's table, as an assertion, at the CLI's own registry default."""
    seam = _junction(plane, layers)
    assert seam.verdict == "continues", (
        f"Au({plane}), {layers} layers/side: junction-cell.md 3.1 says this "
        f"continues the crystal; the CLI's placement gives {seam.verdict!r} "
        f"with seam step {tuple(round(float(x), 3) for x in seam.seam_step)} "
        f"against an in-slab step of "
        f"{tuple(round(float(x), 3) for x in seam.slab_step)}.\n{seam.message}")


@pytest.mark.parametrize("plane,layers", [("100", 4), ("111", 3)])
def test_start_registry_is_the_control_that_fixes_it(plane, layers):
    """AND THE FIX EXISTS ALREADY, which is what makes F9 a wiring defect
    rather than a missing feature: `add_slab` continues the crystal when the
    two slabs are given DIFFERENT registries.  Not xfail -- this passes today.
    """
    seam = _junction(plane, layers, registry_plus=1, registry_minus=0)
    assert seam.verdict == "continues", seam.message
    # And the nearest metal across the boundary is the BULK nearest-neighbour
    # distance a/√2, which is § 3.1's own column for a continuing seam.
    # (Not step equality: on a period-2 stacking -s and +s are the same step
    # modulo the lattice -- 2 x 1.442 = a/√2 -- which is why § 3.1 compares
    # "allowing for ... equivalent directions" and leaves the judgement to
    # `classify_seam`.  Asserting the vectors equal fails on (100) for that
    # reason alone, which is how this comment came to exist.)
    a_exp = 4.0782
    assert seam.gap == pytest.approx(a_exp / np.sqrt(2), abs=1e-3), (
        f"continues, but the across-seam distance is {seam.gap:.4f} Å where "
        f"the bulk nearest-neighbour distance is {a_exp / np.sqrt(2):.4f} Å")
