"""The k-point mesh -- every axis's sampling decided in one place -- through
the road: `molbuilder describe` -> `jobset prep task` -> the deck and its validation
report, or the refusal.

PINS: ``docs/engines/siesta.md`` § 6.1 (the mesh: each axis's role, the one
writer, what a kind fixes, the checks and their severities);
``docs/engines/template.md`` § 5.3 (`above`, ``template.why_not``: one value,
one refusal, on every door); ``docs/plans/plan.md`` § 5w K3.

WHAT THIS FILE HOLDS: one case, a relaxation's mesh -- each axis sampled by
its kind, the template's counts and offsets written, an over-sampled isolated
axis warned in the deck's validation report.

The transport axis -- its one point on every rung, the lead's own count, the
cited offset, the transmission's grid (`engines/transport.md` § 0.3a, § 5
I7-I9) -- is refused on every door and has no road case here: a transport calculation cites a finished relaxation, which only a real run makes.

Nothing here launches an engine: prep writes the decks and stops.
"""
from __future__ import annotations

from support.road import describe_calculation, jobset, set_by_the_person


def _mesh(deck: str, block: str = "kgrid_Monkhorst_Pack"):
    """``(counts, offsets)`` of the ``%block <block>`` in ``deck``."""
    lines = [ln.strip() for ln in deck.splitlines()]
    at = lines.index(f"%block {block}")
    assert lines[at + 4] == f"%endblock {block}", lines[at:at + 5]
    rows = [lines[at + 1 + i].split() for i in range(3)]
    return (tuple(int(rows[i][i]) for i in range(3)),
            tuple(float(rows[i][3]) for i in range(3)))


def test_a_relaxation_samples_every_axis_by_its_kind(tmp_path, monkeypatch):
    """Outside a transport calculation each axis is sampled by its kind: a
    periodic axis, and one declared ``transport`` -- relaxing a junction,
    whose deck is periodic along it (user, 2026-09-30) -- write the
    template's counts and offsets; an isolated axis sampled more than once is
    written as stated and warned in the deck's validation report.  And an
    offset of 1.0 on an axis sampled once is Gamma again -- both engines read
    the offset modulo 1 -- so it is not warned.

    MUTATIONS THIS MUST FAIL AGAINST: the relaxation's mesh derived as a
    transport rung's (prep refuses the relaxation as a stripe junction's);
    the deck's meshes not handed to the settings gate (the report says
    nothing); the offset compared with 0 rather than read modulo 1 (1.0 is
    warned)."""
    dest = describe_calculation(tmp_path, monkeypatch, stage_strategy="", structure={
        "elements": ["H"] * 6,
        "positions": [[7.0, 7.0, 3.0 + 1.5 * i] for i in range(6)],
        "cell": [14.0, 14.0, 14.0],
        "axis_kind": ["periodic", "isolated", "transport"]})
    set_by_the_person(dest, {"kgrid": [2, 3, 1],
                             "kgrid_displacement": [0.0, 0.0, 1.0]})
    r = jobset("prep", "task", "--group", "1", "--bundle", dest,
               "--target", "this")
    assert r.exit_code == 0, r.output
    assert _mesh(next(dest.rglob("*.fdf")).read_text()) == (
        (2, 3, 1), (0.0, 0.0, 1.0))
    report = next(dest.rglob("*.validation.txt")).read_text()
    assert "K-point grid along y = 3 on an isolated axis" in report, report
    assert "The k-point offset along z" not in report, report


def test_the_room_along_a_lattice_vector_is_measured_along_it():
    """The hint that k-points are sampling a direction with vacuum in it
    (`engines/siesta.md` § 6.1) asks how much of the lattice vector the atoms
    leave empty.  On a slanted cell the atoms' span across the x, y and z axes
    does not answer that: atoms filling a slanted vector span little of y.

    API-LEVEL: the shape is the point, and a slanted cell is one number away
    from an upright one, so the check is asked of the two directly.

    THE SILENT FAILURE: a filled, slanted slab warned of vacuum it does not
    have, and told to use one k-point, which would be wrong for it."""
    from types import SimpleNamespace

    import numpy as np

    from molbuilder import kmesh
    from molbuilder.structure import Structure

    a, b, c = np.array([10., 0, 0]), np.array([9., 4.4, 0]), np.array([0, 0, 30.])
    kinds = ("periodic", "periodic", "isolated")

    def said(share):
        """The findings on atoms spread over ``share`` of a and of b."""
        cell = np.array([a, b, c])
        spread = [f * share for f in (0.0, 0.5, 1.0)]
        struct = Structure(elements=["Au"] * 3,
                           positions=np.array([f * a + f * b for f in spread]),
                           cell=cell, axis_kind=kinds)
        cfg = SimpleNamespace(kgrid=(4, 4, 1), kgrid_displacement=(0., 0., 0.))
        mesh = kmesh.mesh_for(cfg, kinds, kind="optimization")
        return [i.where for i in kmesh.check([mesh], struct, cell=cell)]

    assert said(0.9) == []
    assert said(0.3) == ["config.kgrid"] * 2


def test_a_transport_meshs_start_and_its_one_point_warning():
    """`engines/transport.md` § 0.3b at the two doors that hold it --
    API-level, because a transport calculation cites a finished relaxation,
    which only a real run makes (this module's head):

    * the transmission's mesh, left blank, is three times the SCF's count
      along each in-plane direction the SCF samples, a count of one kept and
      the transport axis one (`kmesh.transmission_start`), worked out by the
      one derivation (`kmesh.mesh_for`) so it FOLLOWS the SCF's mesh; a mesh
      the person set is theirs, as given (§ 0.3b, 2026-10-13);
    * one point along a direction the electrodes repeat in is warned on a
      transport mesh -- the shared SCF mesh on every SCF rung, the
      transmission's own on the transmission when the person set it -- and
      never on a direction that does not repeat, nor on a count above one;
      a transmission mesh the person set coarser than the SCF's, above one,
      is warned too; and a transmission mesh the person set above the Gamma
      point while the SCF's is the Gamma point alone -- where tbtrans stops
      in its set-up, the device's Hamiltonian carrying no supercell -- is
      warned, never refused: tbtrans says it (`kmesh.check`).

    MUTATIONS THIS MUST FAIL AGAINST: the transmission starting at the SCF's
    counts; a one kept multiplied; a blank mesh not following the SCF's (or
    left at Gamma); the warnings dropped, or said on an isolated direction;
    the coarser warning said of a blank mesh or beside the one-point one;
    the Gamma device's warning dropped, or said of a transmission at Gamma."""
    from types import SimpleNamespace

    from molbuilder import kmesh

    assert kmesh.transmission_start((4, 4, 1)) == (12, 12, 1)
    assert kmesh.transmission_start((1, 4, 7)) == (1, 12, 1)

    slab = ("periodic", "periodic", "transport")
    chain = ("isolated", "isolated", "transport")

    def said(kgrid, tbt, kinds, shape):
        cfg = SimpleNamespace(kgrid=kgrid, tbt_k_grid=tbt, electrode_kz=100,
                              kgrid_displacement=(0.0, 0.0, 0.0))
        meshes = [kmesh.mesh_for(cfg, kinds, kind="transport", rung=shape)]
        if shape == "transmission":
            meshes.append(kmesh.mesh_for(cfg, kinds, kind="transport",
                                         rung=shape, program="tbtrans"))
        return [(i.severity, i.where) for i in kmesh.check(meshes, None)]

    one_each = [("warn", "config.kgrid")] * 2              # x, then y
    for shape in ("seed", "electrode_L", "device"):
        assert said((1, 1, 1), (12, 12, 1), slab, shape) == one_each, shape
        assert said((4, 4, 1), (12, 12, 1), slab, shape) == [], shape
        assert said((1, 1, 1), (1, 1, 1), chain, shape) == [], shape
    assert said((4, 4, 1), (1, 1, 1), slab, "transmission") == \
        [("warn", "config.tbt_k_grid")] * 2
    assert said((4, 4, 1), (12, 12, 1), slab, "transmission") == []
    # A mesh the person set COARSER than the SCF's, above one: warned, x
    # then y; at the SCF's own counts, not.
    assert said((4, 4, 1), (3, 3, 1), slab, "transmission") == \
        [("warn", "config.tbt_k_grid")] * 2
    assert said((4, 4, 1), (4, 4, 1), slab, "transmission") == []
    # A GAMMA DEVICE, the transmission asked for k-points: the SCF's own
    # one-point finding, x then y, and the Gamma device's -- once; with the
    # transmission at Gamma too, the two grids' one-point findings alone.
    assert said((1, 1, 1), (4, 4, 1), slab, "transmission") == \
        [("warn", "config.kgrid")] * 2 + [("warn", "config.tbt_k_grid")]
    assert said((1, 1, 1), (1, 1, 1), slab, "transmission") == \
        [("warn", "config.kgrid")] * 2 + [("warn", "config.tbt_k_grid")] * 2
    # BLANK IS THE RULE'S: three times the SCF's mesh, following it, never
    # the person's -- so neither warning names it; the SCF's own one-point
    # finding speaks for a Gamma SCF mesh.
    for kgrid, want in (((4, 4, 1), (12, 12, 1)), ((5, 5, 1), (15, 15, 1)),
                        ((1, 4, 1), (1, 12, 1))):
        cfg = SimpleNamespace(kgrid=kgrid, tbt_k_grid=None, electrode_kz=100,
                              kgrid_displacement=(0.0, 0.0, 0.0))
        tbt = kmesh.mesh_for(cfg, slab, kind="transport", rung="transmission",
                             program="tbtrans")
        assert tbt.counts == want and tbt.derived, (kgrid, tbt)
        assert kmesh.write(tbt)[0] == "%block TBT.k"
    assert said((4, 4, 1), None, slab, "transmission") == []
    assert said((1, 1, 1), None, slab, "transmission") == \
        [("warn", "config.kgrid")] * 2
