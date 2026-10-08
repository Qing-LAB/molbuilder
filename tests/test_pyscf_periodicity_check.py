"""A periodic structure must not reach a gas-phase PySCF script in silence.

THE DEFECT, 2026-08-03.  The PySCF renderer builds a molecular ``gto.M()``: no
lattice, no k-points, no way to express one -- a periodic calculation is PySCF's
``pbc`` module, a different builder entirely.  So a structure with a repeating
axis produced a script that dropped the cell and computed an ISOLATED CLUSTER,
and nothing said so.  A three-axis-periodic NaCl cell with a 5.6 A lattice
generated a two-atom gas-phase script, the lattice appeared nowhere in it, and
no check mentioned the difference.

That is not a rough version of what was asked for.  It is a different
calculation, and a plausible-looking one.

WARN, NOT ERROR (user decision).  An isolated-cluster calculation of a periodic
input is legal and occasionally deliberate, and the project blocks only what is
physically impossible -- ``report()`` raises on error severity, so an error here
would mean no script at all.  The user decides; the user is told first.

WHY IT KEYS ON ``axis_kind``.  That field is authoritative and is never None
after construction; ``pbc`` is its derived view and collapses `transport` into
the same ``True`` as `periodic`.  Both are wrong for a gas-phase script, so both
are caught -- but a check written on ``pbc`` alone could not tell a lead from a
crystal axis, which other checks (the k-grid one) depend on.
"""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder.config.pyscf import PySCFConfig
from molbuilder.structure import Structure
from molbuilder.validation import validate


WHERE = "cell.periodic_in_gas_phase"


def _struct(kinds):
    return Structure(
        elements=["Na", "Cl"],
        positions=np.array([[0.0, 0.0, 0.0], [2.8, 0.0, 0.0]]),
        cell=np.eye(3) * 5.6,
        axis_kind=kinds,
    )


def _finding(struct):
    hits = [i for i in validate(struct, PySCFConfig()) if i.where == WHERE]
    return hits[0] if hits else None


@pytest.mark.parametrize("label, kinds", [
    ("a crystal",       ("periodic", "periodic", "periodic")),
    ("a slab",          ("periodic", "periodic", "isolated")),
    ("a transport lead", ("transport", "isolated", "isolated")),
])
def test_a_repeating_axis_is_reported(label, kinds):
    """Any axis that is not isolated is wrong for this emitter."""
    found = _finding(_struct(kinds))
    assert found is not None, (
        f"{label} generated a gas-phase PySCF script with no warning: the "
        f"cell is dropped and the result is an isolated cluster"
    )
    assert found.severity == "warn", (
        f"severity is {found.severity!r}; an error would stop the script being "
        f"written at all, and an isolated-cluster run of a periodic input is a "
        f"legal thing to ask for"
    )


def test_an_isolated_molecule_says_nothing():
    """The check must be quiet for what this emitter is actually for."""
    assert _finding(_struct(("isolated",) * 3)) is None


def test_the_message_names_what_is_dropped():
    """A warning that does not say WHICH cell is being ignored cannot be acted
    on -- the user has to know it is theirs."""
    found = _finding(_struct(("periodic",) * 3))
    msg = found.message
    assert "5.6" in msg, f"the lattice being dropped is not named: {msg}"
    assert "gas-phase" in msg.lower() or "gas phase" in msg.lower()
    assert "isolated cluster" in msg.lower(), (
        f"the message does not say what you WILL get instead: {msg}"
    )


def test_it_reaches_the_tab_before_generate():
    """A finding nobody sees is not a check.

    The structure-optimization tab previews findings through
    /api/build/preflight, so the warning has to arrive there -- before the
    click, not after.
    """
    pytest.importorskip("flask")
    from molbuilder.web.app import create_app

    client = create_app(config={}).test_client()
    # The structure's own canonical dict -- the envelope the browser sends.
    envelope = _struct(("periodic",) * 3).to_dict()
    r = client.post("/api/build/preflight",
                    json={"structure": envelope, "engine": "pyscf",
                          "calculation": "optimization", "params": {}})
    assert r.status_code == 200, r.get_json()
    wheres = [i["where"] for i in (r.get_json().get("issues") or [])]
    assert WHERE in wheres, (
        f"the periodicity warning did not reach the tab's panel: {wheres}"
    )


def test_a_periodic_vibration_preps_as_a_cluster_and_says_so(tmp_path,
                                                            monkeypatch):
    """Through `jobset init` and `prep`: a PySCF vibration of a structure
    that repeats is computed as an isolated cluster, and the check says so --
    a note, not a refusal (user, 2026-09-29: "just note that periodicity will
    not be respected in pySCF"; `engines/vibration.md` § 3).  The atoms the
    structure holds still reach the script: water in a periodic 10 Å cell,
    its oxygen held, preps with the note and writes the held atom in.

    AND ONE COUNT OF WHAT THE HOLD LEAVES (`model/structure-periodicity.md`
    § 2.1, plan § 5w K8): the deck removes a cluster's motions -- the three
    turns about the held oxygen -- and its Methods paragraph and the
    settings check's note say so.  They counted the structure's periodic
    axes, which permit no turn, and told this water six modes, none removed
    (the M11 review's PS-C4).

    MUTATIONS THIS MUST FAIL AGAINST: the door answering the structure's
    kinds for PySCF (the deck counts periodic axes); the Methods count or the
    note reading the structure's kinds (six modes; no note)."""
    import json
    from click.testing import CliRunner

    from molbuilder.jobset._cli import jobset_group
    from molbuilder.projects import PROJECTS_ROOT_ENV
    from molbuilder.workingcopy_structure import StructureCodec
    tree = tmp_path / "projects"
    (tree / "P" / "structure").mkdir(parents=True)
    water = Structure(
        elements=["O", "H", "H"],
        positions=np.array([[5.0, 5.0, 5.119], [5.0, 5.757, 4.523],
                            [5.0, 4.243, 4.523]]),
        cell=np.eye(3) * 10.0, axis_kind=("periodic",) * 3,
        frozen_atoms=[0])
    StructureCodec().write(water, tree / "P" / "structure" / "w.xyz")
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tree))
    monkeypatch.chdir(tmp_path)
    run = CliRunner()
    r = run.invoke(jobset_group, [
        "init", "--structure", "P/structure/w.xyz", "--bundle",
        "P/spectrum/V", "--engine", "pyscf", "--shape", "flat",
        "--calculation", "vibration", "--name", "W"])
    assert r.exit_code == 0, r.output
    bundle = tree / "P" / "spectrum" / "V"
    # THE RUN CARD STATES THE THREADS, as a described calculation does
    # (`architecture.md` § 5.2); how a shell enters an environment is the
    # machine record's (`configuration.md` § 5 M-1).
    task = json.loads((bundle / "task.json").read_text())
    task["execution"] = {"threads": 1}
    (bundle / "task.json").write_text(json.dumps(task, indent=2))
    r = run.invoke(jobset_group, ["prep", "task", "--stage", "freq", "--bundle",
                                  str(bundle)])
    assert r.exit_code == 0, r.output
    assert f"[{WHERE}]" in r.output, r.output
    decks = list(bundle.rglob("W*.py"))
    assert decks, r.output
    deck = decks[0].read_text()
    assert "FROZEN_INDICES_USER        = [0]" in deck
    assert ("AXIS_KIND                  = ('isolated', 'isolated', "
            "'isolated')") in deck
    assert ("giving 3 non-translational / non-rotational vibrational modes "
            "after projecting out the 3 whole-body motion(s)") in deck
    assert "Holding 1 atom(s) leaves 3 whole-body motion(s)" in r.output
    assert "so 3 modes will be reported, not 6" in r.output


def test_a_gas_phase_script_hears_no_advice_about_a_box_it_does_not_use():
    """`model/structure-periodicity.md` § 2.1 (plan § 5w K8): the box's advice
    -- its vacuum, its images -- is for an engine that computes in a cell.
    Water in a tight box draws SIESTA's finding about its images; the PySCF
    script, a molecule in free space, hears none of it, the same structure
    through the same live check (the M11 review's PO-C13).  What binds every
    engine it still hears: an impossible box is refused on every engine (§ 8.2;
    `test_periodicity_gate.py`), and an atom an assigned origin leaves outside
    is forecast as it is to SIESTA -- the hand-off places PySCF's atoms in the
    box too, and refuses that one at prep
    (`test_engine_offset_reaches_every_deck.py`; the K8 review found the
    forecast dropped with the advice).

    MUTATIONS THIS MUST FAIL AGAINST: the gate giving every engine the box's
    advice; a free-space engine hearing only the box's errors."""
    pytest.importorskip("flask")
    from molbuilder.web.app import create_app

    client = create_app(config={}).test_client()
    atoms = dict(elements=["O", "H", "H"],
                 positions=np.array([[0.0, 0.0, 0.119], [0.0, 0.757, -0.477],
                                     [0.0, -0.757, -0.477]]))
    from test_engine_offset_reaches_every_deck import _assigned
    _typed, outside = _assigned([-0.5, -1.0, -1.0])
    # A tight box (the geometry's measure of the images), a typed box beside
    # a vacuum it makes inert (the one checker's note) -- advice, SIESTA's
    # alone -- and an assigned origin that leaves an atom outside, the
    # forecast every engine hears.
    for water, advice, heard in (
            (Structure(**atoms, vacuum=(1.0, 1.0, 1.0)),
             "cell.image_distance", set()),
            (Structure(**atoms, cell=np.eye(3) * 10.0,
                       axis_kind=("isolated",) * 3, vacuum=(3.0, 3.0, 3.0)),
             "cell.vacuum_ignored", set()),
            (outside, "cell.atoms_outside", {"cell.atoms_outside"})):
        said = {}
        for engine in ("siesta", "pyscf"):
            r = client.post("/api/build/preflight",
                            json={"structure": water.to_dict(),
                                  "engine": engine,
                                  "calculation": "optimization",
                                  "params": {}})
            assert r.status_code == 200, r.get_json()
            said[engine] = {i["where"] for i in r.get_json()["issues"]
                            if i["where"].startswith("cell.")}
        assert advice in said["siesta"], said
        assert said["pyscf"] == heard, said
