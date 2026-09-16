"""The junction workflow, walked end to end — the defects it found, pinned.

Built 2026-09-15 on the user's instruction: *"build a test case … simulate the
whole calculation by allowing high tolerance in SCF calculation so that the
calculation can conclude quickly … this is not to validate correctness, but
rather to test the workflow … do not hack and be honest to detect the missing
gaps."*

The walkthrough itself is `docs/execution/walkthrough-2026-09-15-junction.md`:
a 37-atom Au(100)/S-CH2-S/Au(100) junction built, relaxed (41 s, concluded),
and described for transport, through the real doors — Molbuilder tab, CLI,
Structure-optimization tab, Task setup, Transport tab. Nineteen findings.

**This file pins the ones a test can hold**, and only those: each assertion is
a measurement the walkthrough made, phrased so it FAILS while the defect is
present and passes when it is fixed. That is the opposite of the usual
direction, so every test here is marked ``xfail(strict=True)`` — when someone
fixes the defect the test turns XPASS and the suite says so, which is the
signal to delete the xfail rather than the test.

What is deliberately NOT here: the browser halves (the Molbuilder tab's
append-on-load, the missing seam verdict on the cell-apply path, Send-to-Task-
setup's silent no-op) — those are `*_e2e.py`'s territory; and the two prose
findings (a doc saying `bridge` is implicit, `junction-cell.md` § 3.2 naming
the wrong mechanism), which are contract edits with nothing to assert.
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
    mapped to the walk through `_CONTINUES_THE_CRYSTAL`, and `c` set to
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
    "F9: `--electrode` passes `sequence` but leaves `start_registry` at 0 on "
    "BOTH slabs and its spec grammar has no field for it, so every junction "
    "the CLI builds meets itself on the same registry -- eclipsed on "
    "(100)/(110), TWIN on (111). `junction-cell.md` 3.1 says all six of these "
    "continue the crystal. Fix: drive start_registry from the side."))
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


# --------------------------------------------------------------------- #
#  F13 — a set k-grid is reported as "not in the deck"                   #
# --------------------------------------------------------------------- #

_DECK_WITH_A_KGRID = """\
SystemLabel probe
MeshCutoff 100.0 Ry
PAO.BasisSize SZ
%block kgrid_Monkhorst_Pack
2 0 0 0.0
0 2 0 0.0
0 0 1 0.0
%endblock kgrid_Monkhorst_Pack
"""


def test_scalar_parameters_read_back_out_of_a_deck():
    """The baseline, so the next test cannot pass vacuously."""
    from molbuilder import script_emit as sc
    assert sc.parameter("mesh_cutoff", "siesta",
                        deck_text=_DECK_WITH_A_KGRID).value == "100.0"
    assert sc.parameter("basis_size", "siesta",
                        deck_text=_DECK_WITH_A_KGRID).value == "SZ"


@pytest.mark.xfail(strict=True, reason=(
    "F13: `script_emit.parameter(..., deck_text=)` cannot read a %block back "
    "out of a deck, so it answers value=None. `runwrap.py:1771` lists a "
    "parameter as absent when `writes and value is None`, so every "
    "block-valued parameter is printed under '-- not in the deck; the engine "
    "default applies --' in the run log -- including a k-grid that IS set. "
    "The provenance block then says the opposite of the truth, and the "
    "transport composite reads the transverse k off this very deck."))
@pytest.mark.parametrize("name", ["kgrid", "kgrid_displacement"])
def test_a_block_valued_parameter_reads_back_out_of_a_deck(name):
    from molbuilder import script_emit as sc
    assert "%block kgrid_Monkhorst_Pack" in _DECK_WITH_A_KGRID   # not vacuous
    param = sc.parameter(name, "siesta", deck_text=_DECK_WITH_A_KGRID)
    assert param.value is not None, (
        f"{name} writes {param.writes} and the deck carries that block with "
        f"2 2 1, yet the read-back answers None -- so the wrapper reports it "
        f"as taking the engine default")


# --------------------------------------------------------------------- #
#  F17 — the task validator has no transport branch                     #
# --------------------------------------------------------------------- #

@pytest.mark.xfail(strict=True, reason=(
    "F17: `validation/task.py::config_class_for` returns "
    "known.get(task.engine) over {'siesta','pyscf'}. A transport task "
    "correctly carries engine 'siesta' -- TranSIESTA has no separate binary "
    "-- so the transport vocabulary is validated against SiestaConfig and "
    "every transmission/contour field is refused. `task.calculation == "
    "'transport'` is in the file and never consulted, so `jobset prep run "
    "seed` cannot prep what the Transport tab writes, while the web prep "
    "door (which does not run this validator) preps it fine."))
def test_a_transport_description_validates_against_transportconfig():
    from molbuilder.config.transport import TransportConfig
    from molbuilder.validation.task import config_class_for

    class _Task:
        engine = "siesta"
        calculation = "transport"
        varies = ("tbt_k_grid", "transmission_n_points")
        stages = ()

    cls = config_class_for(_Task())
    assert cls is TransportConfig, (
        f"a task whose `calculation` is 'transport' resolved to "
        f"{getattr(cls, '__name__', cls)!r}; its own fields are then reported "
        f"as 'not a field of SiestaConfig' and prep refuses the description")


# --------------------------------------------------------------------- #
#  F4 — a docstring names a guard that was never written                 #
# --------------------------------------------------------------------- #

@pytest.mark.xfail(strict=True, reason=(
    "F4: modify.py's `_build_ase_slab` docstring says per-(plane, orthogonal) "
    "compatibility 'is enforced by :func:`_validate_orthogonal_compat` before "
    "the builder is called'. That function does not exist -- one reference, "
    "zero definitions -- so ASE's own exception is the only guard, which is "
    "why the CLI's `--orthogonal` default (False) reaches ASE and dies on "
    "(100). Fix: write it, or stop naming it."))
def test_the_orthogonal_compat_guard_the_docstring_promises_exists():
    import molbuilder.modify as m
    assert hasattr(m, "_validate_orthogonal_compat"), (
        "modify.py:742 promises this function enforces FCC_ORTHOGONAL_CHOICES "
        "before the builder is called")
