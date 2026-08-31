"""`POST /api/modify/slab` — the new slab op's door (redesign plan § 3).

Beside `/api/modify/electrode`, not replacing it: the old panel stays until
this one is proven (§ 3.4 lists what goes when it is).

**The door reads no selection**, and that is asserted here rather than assumed,
because it is the difference between the two ops: the old one places relative
to a picked group, this one from absolute coordinates.  Sending `indices`
must change nothing.
"""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture()
def client():
    from molbuilder.web.app import create_app
    return create_app(config={}).test_client()


_H2 = "2\nh2\nH 0 0 0\nH 0 0 0.74\n"


def _env(xyz):
    from support.envelope import from_xyz
    return from_xyz(xyz)


def _post(client, **kw):
    body = {"structure": _env(_H2), "element": "Au", "plane": "111",
            "m": 2, "n": 2, "layers": 3, "start_z": 5.0}
    body.update(kw)
    return client.post("/api/modify/slab", json=body)


def _coords(xyz_text):
    rows = [ln.split() for ln in xyz_text.strip().splitlines()[2:] if ln.strip()]
    return np.array([[float(v) for v in r[1:4]] for r in rows])


class TestItBuildsWhatItWasTold:

    def test_a_slab_is_appended_and_the_molecule_survives(self, client):
        d = _post(client).get_json()
        assert d["ok"] is True, d
        assert d["n_atoms"] == 2 + 2 * 2 * 3, d["n_atoms"]
        assert d["elements"][:2] == ["H", "H"], "the molecule was disturbed"
        assert set(d["elements"][2:]) == {"Au"}

    def test_the_starting_layer_lands_on_start_z(self, client):
        z = _coords(_post(client, start_z=7.5).get_json()["xyz"])[2:, 2]
        assert abs(z.min() - 7.5) < 1e-6, z.min()

    def test_growth_direction_puts_the_slab_on_the_stated_side(self, client):
        up = _coords(_post(client, grow="+z", start_z=0.0).get_json()["xyz"])[2:, 2]
        down = _coords(_post(client, grow="-z", start_z=0.0).get_json()["xyz"])[2:, 2]
        assert up.max() > 0 and abs(up.min()) < 1e-6
        assert down.min() < 0 and abs(down.max()) < 1e-6

    def test_dx_dy_are_absolute(self, client):
        """Moving `dx, dy` moves the slab by exactly that, and nothing else.

        This asserted the SLICE's centroid landed on `(dx, dy)`, which pinned
        the reference that made `start_registry` wrong at any layer count that
        is not a whole stacking period (`tests/test_add_slab.py`,
        `TestTheRegistryIsNotContaminatedByTheTrim`).  Which point realises
        the placement is the builder's business; that the numbers translate
        it rigidly is the contract.
        """
        here = _coords(_post(client, dx=0.0, dy=0.0).get_json()["xyz"])[2:, :2]
        there = _coords(_post(client, dx=3.0, dy=-2.0).get_json()["xyz"])[2:, :2]
        assert np.allclose(there - here, [3.0, -2.0], atol=1e-6)

    def test_a_lattice_constant_override_reaches_the_builder(self, client):
        a = _coords(_post(client, lattice_constant=4.0).get_json()["xyz"])[2:]
        b = _coords(_post(client, lattice_constant=4.4).get_json()["xyz"])[2:]
        span = lambda p: float(p[:, 2].max() - p[:, 2].min())
        assert span(b) > span(a) * 1.05, (span(a), span(b))


class TestItReadsNoSelection:

    def test_sending_indices_changes_nothing(self, client):
        """The difference between this op and the old electrode one, asserted
        as behaviour: placement is absolute, so a selection is not merely
        ignored by the client — it cannot reach the answer even if sent."""
        plain = _post(client).get_json()["xyz"]
        with_sel = _post(client, indices=[0], center_indices=[0, 1]).get_json()["xyz"]
        assert plain == with_sel

    def test_the_client_table_says_so_too(self):
        """One rule, two places it must hold: the route ignores a selection,
        and `OPERATIONS` never sends one.  A route that ignored it while the
        table sent it would work by luck."""
        import re
        from pathlib import Path
        src = Path("molbuilder/web/static/lib/molview/model-jobs.js").read_text()
        row = re.search(r"slab:\s*\{[^}]*\}", src, re.S)
        assert row, "no `slab` row in OPERATIONS"
        assert "wholeStructure: true" in row.group(0)
        assert "group: null" in row.group(0)


class TestWhatItRefuses:

    @pytest.mark.parametrize("bad", ["112", "", "0001", None])
    def test_a_plane_it_does_not_support(self, client, bad):
        r = _post(client, plane=bad)
        assert r.status_code == 400
        assert "plane" in r.get_json()["error"]

    def test_a_missing_element(self, client):
        assert _post(client, element=None).status_code == 400

    @pytest.mark.parametrize("field,value", [("m", 0), ("n", -1), ("layers", -1)])
    def test_a_count_below_its_floor(self, client, field, value):
        r = _post(client, **{field: value})
        assert r.status_code == 400 and field in r.get_json()["error"]

    def test_a_non_numeric_count(self, client):
        r = _post(client, m="two")
        assert r.status_code == 400 and "whole number" in r.get_json()["error"]

    def test_a_growth_direction_it_does_not_know(self, client):
        r = _post(client, grow="sideways")
        assert r.status_code == 400 and "grow must be" in r.get_json()["error"]

    def test_a_stacking_it_does_not_know(self, client):
        r = _post(client, stacking="flip")
        assert r.status_code == 400 and "stacking must be" in r.get_json()["error"]

    def test_a_lattice_constant_that_is_not_a_length(self, client):
        assert _post(client, lattice_constant=0).status_code == 400
        assert _post(client, lattice_constant=-4.0).status_code == 400

    def test_zero_layers_is_a_no_op_not_an_error(self, client):
        d = _post(client, layers=0).get_json()
        assert d["ok"] is True and d["n_atoms"] == 2


# ---------------------------------------------------------------------------
# "From a bulk run…" reaches its route — the door it calls must exist
# ---------------------------------------------------------------------------

def test_the_bulk_run_button_calls_a_door_that_exists():
    """**The bug this exists for.**

    The button called `projects.shared().file` — a method that has never been
    on that module.  `&&`-guarded, so it was `undefined` on every press and
    every press took the "pick a file first" branch: the feature was
    unreachable, and no action a person could take made it work.  The route
    behind it was correct and tested the whole time.

    A name that does not exist cannot be caught by a JS unit test that stubs
    the module — the stub would define it.  So this reads the source and holds
    it to the surface `lib/projects/state.js` actually publishes, plus the
    tree picker's own entry point.
    """
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "molbuilder/web/static"
    panel = (root / "modify/slab-panel.js").read_text(encoding="utf-8")
    picker = (root / "lib/tree-picker.js").read_text(encoding="utf-8")

    # CODE, NOT PROSE.  The comment above the fix names the dead call on
    # purpose, and a substring search flags it -- the third time today a pin
    # punished the documentation it wanted.  Comment and block-comment lines
    # are dropped before looking.
    code = "\n".join(
        line for line in panel.splitlines()
        if not line.lstrip().startswith(("*", "//", "/*")))
    assert "projects.shared" not in code, (
        "`projects.shared()` does not exist; the button is unreachable again")
    assert "tree-picker.js" in panel and "pickPath" in panel, (
        "the button must ASK for the file rather than read the sidebar's "
        "current selection (user, 2026-08-31)")
    assert "export async function pickPath" in picker, (
        "slab-panel imports `pickPath` from the tree picker; it is gone")
    assert 'mode: "file"' in panel, (
        "the picker defaults to directories -- a lattice is measured from a "
        "FILE")
