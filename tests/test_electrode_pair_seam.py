"""The electrode pair meets itself on the wrong registry — measured.

`science/junction-cell.md` § 3.1 tables the layer counts whose seam CONTINUES
the fcc crystal.  None of them reproduces at the CLI's registry default:
`--electrode` passes `sequence` but leaves `add_slab`'s `start_registry` at 0
on BOTH slabs unless told otherwise — so such a junction meets itself on the
same registry.  Measured across three planes and three layer counts:
**eclipsed** on (100)/(110), **twin** on (111).  A twin carries the RIGHT bond
length, which § 3.1 says is exactly why a distance check misses it.

The second test is the other half of the measurement: `start_registry` IS the
control — `+z = 1`, `−z = 0` continues the crystal on both surfaces, with the
across-seam distance at the bulk `a/√2`.
"""
from __future__ import annotations

import numpy as np
import pytest


# --------------------------------------------------------------------- #
#  F9 — the CLI's electrode default does not continue the crystal        #
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


def _junction(tmp_path, plane, layers, *, registry_plus="", registry_minus=""):
    """Two slabs facing each other, built by `molbuilder modify --electrode`,
    one flag per side, around one atom at the origin; `c` set to `z_span + d`
    as § 6.1 instructs the user to.  The only thing varied is ``registry=``,
    which the flag leaves at its default when it is not written.

    The command's own refusal is a failure, never the `xfail` below's.
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


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
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
def test_the_cli_electrode_pair_continues_the_crystal(tmp_path, plane, layers):
    """§ 3.1's table, as an assertion, at the CLI's own registry default."""
    seam = _junction(tmp_path, plane, layers)
    assert seam.verdict == "continues", (
        f"Au({plane}), {layers} layers/side: junction-cell.md 3.1 says this "
        f"continues the crystal; the CLI's placement gives {seam.verdict!r} "
        f"with seam step {tuple(round(float(x), 3) for x in seam.seam_step)} "
        f"against an in-slab step of "
        f"{tuple(round(float(x), 3) for x in seam.slab_step)}.\n{seam.message}")


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
