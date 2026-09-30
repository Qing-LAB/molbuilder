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
* ``electrode_kz = 1`` refused only at prep, its range admitting the value.

Nothing here launches an engine: prep writes the decks and stops.  The form's
locked component is the browser's to show (`test_transport_tab_e2e.py`).
"""
from __future__ import annotations

import json

from click.testing import CliRunner

from molbuilder.jobset._cli import jobset_group
from molbuilder.template import one, read_template

from test_transport_prep import (_CITE, _conclude, _isolated,  # noqa: F401
                                 _junction_struct, _says, _write_junction)

_DECK = "J/optimization/Relax/01_coarse/run-0/Relax_01_coarse.fdf"
_TOKENS = {"seed": "01_seed", "electrode_L": "02_electrode_L",
           "electrode_R": "03_electrode_R", "device": "04_device",
           "transmission": "05_transmission"}
#: What each rung's run leaves for the rungs after it (`engines/transport.md`
#: § 4.2's DAG) -- faked as a concluded attempt, since nothing here launches.
_PRODUCTS = {"seed": ["T.DM"], "electrode_L": ["T_L-electrode.TSHS"],
             "electrode_R": ["T_R-electrode.TSHS"], "device": ["T.TS.HSX"],
             "transmission": []}


def _cite_mesh(root, rows):
    """The cited relaxation ran THIS mesh: its deck's k block, rewritten --
    ``rows`` three ``"<counts> <offset>"`` lines -- before anything reads
    it."""
    deck = root / _DECK
    text = deck.read_text()
    old = "  4 0 0 0.0\n  0 4 0 0.0\n  0 0 2 0.0\n"
    assert old in text
    deck.write_text(text.replace(old, "".join(f"  {r}\n" for r in rows)))


def _init(root):
    """`jobset init` for a transport calculation citing the junction -- the
    CLI the person runs -- and the folder's own runtime settings beside it."""
    r = CliRunner().invoke(jobset_group, [
        "init", "--calculation", "transport", "--shape", "hierarchical",
        "--bundle", "J/transport/T", "--slot", f"junction={_CITE}",
        "--bias", "0.0"])
    assert r.exit_code == 0, r.output
    dest = root / "J" / "transport" / "T"
    (dest / ".molbuilder.json").write_text(json.dumps(
        {"script_generation": {"activation": "conda activate"}}))
    return dest


def _prep(dest, rung, *, refused=False):
    """`jobset prep run <rung>`; what it said when ``refused``."""
    r = CliRunner().invoke(jobset_group, ["prep", "run", rung, "--bundle",
                                          str(dest), "--no-sbatch"])
    if refused:
        assert r.exit_code != 0, r.output
        return r.output
    assert r.exit_code == 0, r.output
    return None


def _ladder(dest):
    """Every rung, in order: `jobset prep run`, then the run's conclusion --
    the attempt and what it leaves -- which the next rung's prep gathers."""
    for rung in _TOKENS:
        _prep(dest, rung)
        _conclude(dest, rung, _PRODUCTS[rung])


def _mesh(deck: str, block: str = "kgrid_Monkhorst_Pack"):
    """``(counts, offsets)`` of the ``%block <block>`` in ``deck``."""
    lines = [ln.strip() for ln in deck.splitlines()]
    at = lines.index(f"%block {block}")
    assert lines[at + 4] == f"%endblock {block}", lines[at:at + 5]
    rows = [lines[at + 1 + i].split() for i in range(3)]
    return (tuple(int(rows[i][i]) for i in range(3)),
            tuple(float(rows[i][3]) for i in range(3)))


def _deck(dest, rung, suffix=".fdf"):
    tok = _TOKENS[rung]
    return (dest / tok / f"T_{tok}{suffix}").read_text()


def test_every_rung_writes_its_own_mesh_with_the_cited_offset(
        isolated_projects_root):
    """One shared transverse grid and offset, cited from the relaxation; the
    transport axis one point on the open rungs and ``electrode_kz`` on the
    leads, offset 0 there; the transmission its own ``TBT.k``, in the block
    form that carries the offset.  And the transmission's grid is held to the
    same isolated-axis rule as the SCF's: the fixture's leads are a chain,
    isolated across, so four points there sample images of vacuum.

    MUTATIONS THIS MUST FAIL AGAINST: the cited offset not carried
    (`_apply_kgrid` ignoring it); a lead given the open role (its deck writes
    1 along transport); the one writer dropping the offset column; the
    transmission's mesh not handed to the settings gate (no warning)."""
    root = isolated_projects_root
    _write_junction(root, _junction_struct())
    _cite_mesh(root, ["4 0 0 0.5", "0 4 0 0.5", "0 0 2 0.25"])
    dest = _init(root)
    tmpl = read_template((dest / "T.template.toml").read_text())
    # The cited run's grid and offset, the transport axis laid on by the rule.
    assert one(tmpl, "kgrid").value == (4, 4, 1)
    assert one(tmpl, "tbt_k_grid").value == (4, 4, 1)
    assert one(tmpl, "kgrid_displacement").value == (0.5, 0.5, 0.0)

    _ladder(dest)
    open_axis = ((4, 4, 1), (0.5, 0.5, 0.0))
    lead_axis = ((4, 4, 40), (0.5, 0.5, 0.0))
    for rung, want in (("seed", open_axis), ("electrode_L", lead_axis),
                       ("electrode_R", lead_axis), ("device", open_axis),
                       ("transmission", open_axis)):
        assert _mesh(_deck(dest, rung)) == want, rung
    transmission = _deck(dest, "transmission")
    assert _mesh(transmission, "TBT.k") == open_axis
    report = _deck(dest, "transmission", ".validation.txt")
    assert "tbt_k_grid[0] = 4 on an isolated axis" in report, report


def test_a_gamma_only_junctions_lead_counts_the_points_it_writes(
        isolated_projects_root):
    """A lead of a Gamma-only junction writes forty points along transport,
    so ``Diag.ParallelOverK``'s automatic answer -- *more than one point*,
    counted on the mesh the deck writes -- splits its diagonaliser over k,
    while the seed's one point splits over orbitals.  And the offset, one
    value every mesh of a deck shares, is judged once per deck: shifted on
    an axis sampled at one point, it moves that point off Gamma -- said once
    on the transmission deck, whose two meshes both sample it so.

    MUTATIONS THIS MUST FAIL AGAINST: the split read from the template's
    ``kgrid`` (the lead says ``.false.`` over forty points); the offset
    judged per mesh (two lines for one value)."""
    root = isolated_projects_root
    _write_junction(root, _junction_struct())
    _cite_mesh(root, ["1 0 0 0.5", "0 1 0 0.0", "0 0 2 0.0"])
    dest = _init(root)
    _ladder(dest)
    assert _says(_deck(dest, "seed"), "Diag.ParallelOverK", ".false.")
    lead = _deck(dest, "electrode_L")
    assert _mesh(lead) == ((1, 1, 40), (0.5, 0.0, 0.0))
    assert _says(lead, "Diag.ParallelOverK", ".true."), (
        "a lead writing forty k-points splits its diagonaliser over them")
    report = _deck(dest, "transmission", ".validation.txt")
    said = [ln for ln in report.splitlines()
            if "kgrid_displacement[0] = 0.5 shifts an axis" in ln]
    assert len(said) == 1, said
    assert "kgrid[0] = 1, tbt_k_grid[0] = 1" in said[0], said


def test_the_transport_axis_is_fixed_on_every_door(web_client,
                                                   isolated_projects_root):
    """No transport rung reads the third component of ``kgrid``,
    ``tbt_k_grid`` or ``kgrid_displacement``, so each door treats it as the
    kind's: the Transport tab's Send refuses another value on a rung, and
    prep refuses one the template states -- with the one clause every door
    gives.  (The form draws it locked: the browser's test.)

    MUTATION THIS MUST FAIL AGAINST: `kmesh.fixed` answering nothing (the
    Send describes, and prep writes 1 over the refused 2 in silence)."""
    root = isolated_projects_root
    _write_junction(root, _junction_struct())

    said = web_client.post("/api/transport/describe", json=dict(
        engine="siesta", name="T", junction=_CITE, bias=[0.0],
        stages={"transmission": {"tbt_k_grid": [4, 4, 2]}}))
    assert said.status_code == 400, said.get_json()
    error = said.get_json()["error"]
    assert ("stage 'transmission' sets tbt_k_grid = [4, 4, 2], whose z "
            "component a transport calculation fixes at 1") in error, error
    assert "the open boundary" in error, error

    dest = _init(root)
    from test_fixed_and_shared_items_e2e import _template_says
    _template_says(dest, "kgrid", (4, 4, 2))
    refused = _prep(dest, "seed", refused=True)
    assert ("the template sets kgrid = [4, 4, 2], whose z component a "
            "transport calculation fixes at 1") in refused, refused


def test_a_lead_sampled_once_is_refused_and_a_thin_one_warned(
        web_client, isolated_projects_root):
    """``electrode_kz``'s limit (above 1) and its recommended range (from
    20) -- the item's own, read on every door: the Transport tab's Send
    refuses 1 with the reason and describes 10 with a notice.

    MUTATION THIS MUST FAIL AGAINST: the item's `above` at 0 (1 describes)."""
    _write_junction(isolated_projects_root, _junction_struct())

    def send(kz):
        return web_client.post("/api/transport/describe", json=dict(
            engine="siesta", name="T", junction=_CITE, bias=[0.0],
            stages={"electrode_L": {"electrode_kz": kz},
                    "electrode_R": {"electrode_kz": kz}}))

    once = send(1)
    assert once.status_code == 400, once.get_json()
    assert ("electrode_kz = 1: it must be greater than 1 -- a lead is "
            "periodic bulk along transport") in once.get_json()["error"]
    thin = send(10)
    assert thin.status_code == 200, thin.get_json()
    notices = [n["message"] for n in thin.get_json()["notices"]]
    assert any("electrode_kz = 10, outside the recommended range [20, 200]"
               in m for m in notices), notices


def test_one_value_draws_one_refusal():
    """API-LEVEL, because the road cannot reach it: `resolve` refuses first,
    so a value reaches the settings gate unrefused only in a render that
    skipped it (a library call or a test).  There, one value still draws one
    finding: a transport calculation's ``kgrid = (4, 4, 0)`` breaks both the
    fixed transport axis and the count floor, and says the first -- and the
    fixture's chain is isolated across, so the mesh built from it would earn
    two isolated-axis warnings, which stand aside for the refusal; a count of
    0 on an optimization is past the floor and outside the range, and says
    the limit alone; and a lead's ``electrode_kz = 1`` is past its limit and
    outside its recommended range, and draws the refusal alone.

    MUTATIONS THIS MUST FAIL AGAINST: the range warning not standing aside
    for a refused value (a second finding); the mesh check judging a mesh
    built from a refused value (two more); the floor asked before the fixed
    component (the wrong reason)."""
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.validation import validate

    def at(name, cfg, kind):
        return [i for i in validate(_junction_struct(), cfg, calculation=kind)
                if i.where == f"config.{name}"]

    [found] = at("kgrid", SiestaConfig(system_label="j", kgrid=(4, 4, 0)),
                 "transport")
    assert found.severity == "error", found
    assert "whose z component a transport calculation fixes at 1" in \
        found.message, found.message
    [found] = at("kgrid", SiestaConfig(system_label="j", kgrid=(0, 1, 1)),
                 "optimization")
    assert found.severity == "error", found
    assert "each component must be greater than 0" in found.message
    [found] = at("electrode_kz", SiestaConfig(system_label="j",
                                              electrode_kz=1), "transport")
    assert found.severity == "error", found
    assert "it must be greater than 1" in found.message, found.message
