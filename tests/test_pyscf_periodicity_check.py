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


def test_the_script_is_still_written():
    """The other half of 'warn, not error'."""
    from molbuilder.pyscf import render_script
    script = render_script(_struct(("periodic",) * 3), PySCFConfig())
    assert len(script.splitlines()) > 100, "the script was not generated"
    # ...and it really is the gas-phase builder, which is why the warning
    # exists.  If this ever becomes a periodic emitter, this test is the
    # record of what changed.
    assert "gto.M(" in script
    assert "pbc.gto" not in script and "Cell(" not in script


def test_it_reaches_the_tab_before_generate():
    """A finding nobody sees is not a check.

    The structure-optimization tab previews findings through
    /api/build/preflight, so the warning has to arrive there -- before the
    click, not after.
    """
    pytest.importorskip("flask")
    from molbuilder.web.app import create_app

    client = create_app(config={}).test_client()
    envelope = {
        "elements": ["Na", "Cl"],
        "positions": [[0, 0, 0], [2.8, 0, 0]],
        "metadata": {"regions": {},
                     "cell": [[5.6, 0, 0], [0, 5.6, 0], [0, 0, 5.6]],
                     "cell_origin": None,
                     "axis_kind": ["periodic"] * 3,
                     "vacuum": None},
    }
    r = client.post("/api/build/preflight",
                    json={"structure": envelope, "engine": "pyscf",
                          "params": {}})
    assert r.status_code == 200, r.get_json()
    wheres = [i["where"] for i in (r.get_json().get("issues") or [])]
    assert WHERE in wheres, (
        f"the periodicity warning did not reach the tab's panel: {wheres}"
    )
