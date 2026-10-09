"""The k-point mesh -- every axis's sampling decided in one place -- through
the road: `jobset init` -> `jobset prep task` -> the deck and its validation
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


from click.testing import CliRunner

from molbuilder.jobset._cli import jobset_group

from support.junction import _isolated, _junction_struct  # noqa: F401


def _prep(dest, rung, *, refused=False):
    """`jobset prep task <rung>`; what it said when ``refused``."""
    r = CliRunner().invoke(jobset_group, ["prep", "task", "--stage", rung, "--bundle",
                                          str(dest), "--no-sbatch"])
    if refused:
        assert r.exit_code != 0, r.output
        return r.output
    assert r.exit_code == 0, r.output
    return None


def _mesh(deck: str, block: str = "kgrid_Monkhorst_Pack"):
    """``(counts, offsets)`` of the ``%block <block>`` in ``deck``."""
    lines = [ln.strip() for ln in deck.splitlines()]
    at = lines.index(f"%block {block}")
    assert lines[at + 4] == f"%endblock {block}", lines[at:at + 5]
    rows = [lines[at + 1 + i].split() for i in range(3)]
    return (tuple(int(rows[i][i]) for i in range(3)),
            tuple(float(rows[i][3]) for i in range(3)))


def test_a_relaxation_samples_every_axis_by_its_kind(isolated_projects_root):
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
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.task import Stage
    from test_engine_offset_reaches_every_deck import _prep as _describe_prep
    dest, _stage, deck = _describe_prep(
        isolated_projects_root, _junction_struct(across=("periodic",
                                                         "isolated")),
        SiestaConfig(system_label="JOB", kgrid=(2, 3, 1),
                     kgrid_displacement=(0.0, 0.0, 1.0)),
        (Stage(name="relax", overrides={}),), "siesta")
    assert _mesh(deck) == ((2, 3, 1), (0.0, 0.0, 1.0))
    report = next(dest.rglob("*.validation.txt")).read_text()
    assert "kgrid[1] = 3 on an isolated axis" in report, report
    assert "kgrid_displacement[2]" not in report, report


def test_a_transport_meshs_start_and_its_one_point_warning():
    """`engines/transport.md` § 0.3b at the two doors that hold it --
    API-level, because a transport calculation cites a finished relaxation,
    which only a real run makes (this module's head):

    * the transmission's mesh starts at three times the SCF's count along
      each in-plane direction the SCF samples, a count of one kept and the
      transport axis one (`kmesh.transmission_start`);
    * one point along a direction the electrodes repeat in is warned on a
      transport mesh -- the shared SCF mesh on every SCF rung, the
      transmission's own on the transmission -- and never on a direction
      that does not repeat, nor on a count above one (`kmesh.check`).

    MUTATIONS THIS MUST FAIL AGAINST: the transmission starting at the SCF's
    counts; a one kept multiplied; the warning dropped, or said on an
    isolated direction."""
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
    for shape in ("seed", "electrode", "device"):
        assert said((1, 1, 1), (12, 12, 1), slab, shape) == one_each, shape
        assert said((4, 4, 1), (12, 12, 1), slab, shape) == [], shape
        assert said((1, 1, 1), (1, 1, 1), chain, shape) == [], shape
    assert said((4, 4, 1), (1, 1, 1), slab, "transmission") == \
        [("warn", "config.tbt_k_grid")] * 2
    assert said((4, 4, 1), (12, 12, 1), slab, "transmission") == []
