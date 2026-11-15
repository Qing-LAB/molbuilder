"""The electrode pair continues the crystal when its two sides state their
registries (`science/junction-cell.md` § 3.2).

`--electrode` leaves `add_slab`'s `start_registry` at 0 on both slabs unless
told otherwise -- the documented default (the user, 2026-09-22), under which
a pair meets itself on the same registry.  `start_registry` is the control:
`+z = 1`, `−z = 0` continues the crystal on both surfaces, with the
across-seam distance at the bulk `a/√2`.
"""
from __future__ import annotations

import numpy as np
import pytest


#: Interlayer spacing `d` per surface for Au at the builder's own default
#: lattice constant (a_experimental = 4.0782 Å): (111) a/√3, (100) a/2,
#: (110) a/(2√2).  `junction-cell.md` § 2.
_D = {"111": 2.35470, "100": 2.03910, "110": 1.44186}


def _junction(tmp_path, plane, layers, *, registry_plus="", registry_minus=""):
    """Two slabs facing each other, built by `molbuilder modify --electrode`,
    one flag per side, around one atom at the origin; `c` set to `z_span + d`
    as § 6.1 instructs the user to.  The only thing varied is ``registry=``,
    which the flag leaves at its default when it is not written.

    The command's own refusal is a failure.
    """
    from click.testing import CliRunner

    from molbuilder.cell import classify_seam
    from molbuilder.cli import cli
    from molbuilder.workingcopy_structure import StructureCodec

    src, out = tmp_path / "in.xyz", tmp_path / "out.xyz"
    src.write_text("1\none atom\nH 0 0 0\n")
    argv = ["modify", str(src), str(out), "--yes"]
    for side, reg in (("+z", registry_plus), ("-z", registry_minus)):
        argv += ["--electrode",
                 f"Au:{plane}:2x2x{layers}@contact=2.4{reg}:{side}=0"]
    if plane != "111":              # FCC_ORTHOGONAL_CHOICES: only (111) is free
        argv.append("--orthogonal")
    res = CliRunner().invoke(cli, argv)
    if res.exit_code != 0:
        pytest.fail(f"modify refused the pair: {res.output}")
    struct = StructureCodec().load(out)
    pos = np.asarray(struct.positions, dtype=float)
    metal = np.array([p for e, p in zip(struct.elements, pos) if e == "Au"])
    cell = np.array(struct.cell, dtype=float).copy()
    cell[2, 2] = float(metal[:, 2].max() - metal[:, 2].min()) + _D[plane]
    return classify_seam(metal, cell)


@pytest.mark.parametrize("plane,layers", [("100", 4), ("111", 3)])
def test_start_registry_is_the_control_that_fixes_it(tmp_path, plane, layers):
    """The pair continues the crystal when the two flags state DIFFERENT
    registries.
    """
    seam = _junction(tmp_path, plane, layers,
                     registry_plus=":registry=B", registry_minus=":registry=A")
    assert seam.verdict == "continues", seam.message
    # And the nearest metal across the boundary is the BULK nearest-neighbour
    # distance a/√2, which is § 3.1's own column for a continuing seam.
    # (Not step equality: on a period-2 stacking -s and +s are the same step
    # modulo the lattice -- 2 x 1.442 = a/√2 -- which is why § 3.1 compares
    # "allowing for ... equivalent directions" and leaves the judgement to
    # `classify_seam`.  Asserting the vectors equal fails on (100) for that
    # reason alone.)
    a_exp = 4.0782
    assert seam.gap == pytest.approx(a_exp / np.sqrt(2), abs=1e-3), (
        f"continues, but the across-seam distance is {seam.gap:.4f} Å where "
        f"the bulk nearest-neighbour distance is {a_exp / np.sqrt(2):.4f} Å")
