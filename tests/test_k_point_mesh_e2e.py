"""The k-point mesh -- every rung's sampling decided in one place -- through
the road: `jobset init` (and the Transport tab's describe door) -> `jobset
prep run` -> each rung's deck and its validation report, or the refusal.

PINS: ``docs/engines/siesta.md`` § 6.1 (the mesh: each axis's role, the one
writer, what a kind fixes, the checks and their severities);
``docs/engines/template.md`` § 5.3 (`above`, ``template.why_not``: one value,
one refusal, on every door); ``docs/engines/transport.md`` § 0.3a (the offset
on every rung) and § 5 I7-I9; ``docs/plans/plan.md`` § 5w K3.

PREVENTS, each read in the code before 2026-09-30:

* the transport axis forced to 1 in six places, so a value the gate refused
  was also overwritten when the deck was written;
* a lead's ``Diag.ParallelOverK`` decided from the template's ``kx ky 1``
  while its deck wrote ``kx ky 40``;
* the cited offset dropped -- every transport rung wrote ``0.0`` and ``TBT.k``
  took its list form, which carries none;
* one fact, two severities -- a warning in the SIESTA validator and an error
  in the transport kind's -- and the transmission's grid never held to the
  isolated-axis rule the SCF grid was; the shared offset warned once per mesh;
* ``electrode_kz = 1`` refused only at prep, its range admitting the value;
* a stripe junction sampled across its vacuum stopped by TranSIESTA alone,
  at the device, after the seed and both leads had run (the K3 review).

Nothing here launches an engine: prep writes the decks and stops.  The form's
locked component is the browser's to show (`test_transport_tab_e2e.py`).
"""
from __future__ import annotations


from click.testing import CliRunner

from molbuilder.jobset._cli import jobset_group

from test_transport_prep import _isolated, _junction_struct  # noqa: F401


def _prep(dest, rung, *, refused=False):
    """`jobset prep run <rung>`; what it said when ``refused``."""
    r = CliRunner().invoke(jobset_group, ["prep", "run", rung, "--bundle",
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
        (Stage(name="relax", enabled=True, overrides={}),), "siesta")
    assert _mesh(deck) == ((2, 3, 1), (0.0, 0.0, 1.0))
    report = next(dest.rglob("*.validation.txt")).read_text()
    assert "kgrid[1] = 3 on an isolated axis" in report, report
    assert "kgrid_displacement[2]" not in report, report
