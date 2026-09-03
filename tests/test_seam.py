"""`cell.classify_seam` — what the periodic boundary does to the crystal.

`science/junction-cell.md` § 4.1 (the verdicts) and § 3.1 (why a distance
check is not enough).  **The distance alone is not the test**: a twin has the correct bulk bond length, so a distance check passes it;
only the registry separates continuation from a twin.

And the registry is not tested by arithmetic.  Two earlier versions compared
the seam's lateral step against the in-slab step -- directly, then reduced
modulo the cell -- and both called a perfectly continuous 3-layer Au(111)
boundary `unknown`, because consecutive layers step by `period` different
vectors that agree only modulo the PRIMITIVE lattice while the cell on hand is
the supercell's.  So the tests below drive the real builders across every
layer count rather than checking one hand-picked slab.
"""
from __future__ import annotations

import numpy as np
import pytest

from molbuilder.cell import STACKING_PERIOD, classify_seam
from molbuilder.modify import _build_ase_slab

_A = 4.078
#: (plane, orthogonal) -- (100) and (110) have no non-orthogonal cell at all
#: (junction-cell.md § 2b).
_SURFACES = [("111", False), ("100", True), ("110", True)]


def _padded(plane, orthogonal, n_layers, mn=(2, 2)):
    """An ASE slab in a box padded by exactly one interlayer spacing.

    That is the cell `add_slab` captures, and the only one where "does the
    crystal continue across the boundary" is a question with an answer.
    """
    slab = _build_ase_slab("Au", plane, (mn[0], mn[1], n_layers), orthogonal, _A)
    pos = np.asarray(slab.positions, dtype=float)
    zs = sorted({round(float(z), 6) for z in pos[:, 2]})
    d = zs[1] - zs[0]
    cell = np.asarray(slab.get_cell(), dtype=float)
    cell[2] = [0.0, 0.0, (zs[-1] - zs[0]) + d]
    return pos, cell


class TestAWholeNumberOfPeriodsContinues:
    """§ 3.1's layer-count condition, driven across every count."""

    @pytest.mark.parametrize("plane,orthogonal", _SURFACES)
    @pytest.mark.parametrize("n_layers", range(2, 10))
    def test_it_continues_exactly_when_the_layers_make_whole_periods(
            self, plane, orthogonal, n_layers):
        period = STACKING_PERIOD[plane]
        verdict = classify_seam(*_padded(plane, orthogonal, n_layers))
        continues = verdict.verdict == "continues" and verdict.period == period
        assert continues == (n_layers % period == 0), (
            f"fcc({plane}) x{n_layers} (period {period}) came back "
            f"{verdict.verdict!r} / period {verdict.period}")

    def test_a_slab_too_thin_to_have_a_stacking_says_so(self):
        """Two layers of (111) are `A,B` -- fcc and hcp alike.

        The boundary genuinely continues them, as a 2-period stack.  Reporting
        `continues` alone would be read as "this is the fcc(111) you asked
        for", so the MEASURED period is what makes it checkable.
        """
        verdict = classify_seam(*_padded("111", False, 2))
        assert verdict.verdict == "continues"
        assert verdict.period == 2 != STACKING_PERIOD["111"]


class TestTheFailuresAreToldApart:

    def test_an_unpadded_box_is_a_collision_not_a_seam(self):
        pos, cell = _padded("111", False, 6)
        cell[2, 2] -= 2.3545          # take the padding back out
        verdict = classify_seam(pos, cell)
        assert verdict.verdict == "collision"
        assert verdict.z_room == pytest.approx(0.0, abs=1e-3)
        assert verdict.gap > 1.0, (
            "and this is why room is not the closest approach: the faces are "
            "at the same z with no room at all, yet the nearest atoms are "
            "1.66 Å apart because the layers sit laterally offset")

    def test_room_for_a_molecule_is_vacuum_not_a_seam(self):
        """Registry agreement across empty space is a coincidence.

        A slab built beside a molecule gets a box tall enough for both, and
        its two faces came back `continues` across 7.5 Å of nothing because
        the layers happened to land on matching sites.
        """
        pos, cell = _padded("111", False, 6)
        cell[2, 2] += 6.0
        verdict = classify_seam(pos, cell)
        assert verdict.verdict == "vacuum"
        assert verdict.z_room == pytest.approx(2.3545 + 6.0, abs=0.01)
        assert "free surface" in verdict.message

    def test_a_mirrored_junction_is_caught_though_its_bond_length_is_right(self):
        """The real `Au-BDT-Au` topology: 6 layers a side, mirrored.

        Measured on the shipped junction: seam 2.4008 Å at step (0.000,
        0.000) -- a correct bulk bond length, so a distance check passes it.
        Only the registry catches it.
        """
        pos, cell = _padded("111", False, 6)
        d = 2.3545
        top = pos[:, 2].max()
        upper = pos.copy()
        upper[:, 2] = (top + d) + (top - pos[:, 2])
        both = np.vstack([pos, upper])
        cell = cell.copy()
        cell[2, 2] = (both[:, 2].max() - both[:, 2].min()) + d

        verdict = classify_seam(both, cell)
        assert verdict.verdict == "eclipsed"
        assert verdict.gap == pytest.approx(d, abs=0.01), (
            "the bond length across a mirror seam is CORRECT -- that is why "
            "the distance cannot be the test")
        assert verdict.z_room == pytest.approx(d, abs=0.01)
        assert np.allclose(verdict.seam_step, (0.0, 0.0), atol=1e-6)

    def test_the_message_names_which_condition_failed(self):
        """§ 4.1: name which condition failed -- layer count or placement.

        And it does not follow from the verdict -- the real junction is
        `eclipsed` with 6 layers a side, a whole number of periods, so § 3.1
        HOLDS there and only the mirror is wrong.  Reading the condition off
        the verdict called that one backwards.
        """
        pos, cell = _padded("111", False, 6)
        d = 2.3545
        top = pos[:, 2].max()
        upper = pos.copy()
        upper[:, 2] = (top + d) + (top - pos[:, 2])
        both = np.vstack([pos, upper])
        cell = cell.copy()
        cell[2, 2] = (both[:, 2].max() - both[:, 2].min()) + d
        mirrored = classify_seam(both, cell)
        assert "§ 3.2" in mirrored.message and "§ 3.1" not in mirrored.message

        short = classify_seam(*_padded("111", False, 4))
        assert "§ 3.1" in short.message and "§ 3.2" not in short.message


class TestItRefusesToGuess:

    def test_one_layer_is_not_a_seam(self):
        pos, cell = _padded("111", False, 2)
        one = pos[pos[:, 2] < pos[:, 2].min() + 0.1]
        verdict = classify_seam(one, cell)
        assert verdict.verdict == "unknown"
        assert "one atomic layer" in verdict.message

    def test_no_cell_no_verdict(self):
        verdict = classify_seam(np.zeros((1, 3)), np.eye(3))
        assert verdict.verdict == "unknown"


class TestTheWarningReachesTheUser:
    """A measurement nobody is told about is not a warning.

    It travels in the RECEIPTS slot -- `ok_structure_response`'s `notices`,
    "what the edit did first, what is now true after it".  That channel had no
    caller until this one, and using it is what puts the message on screen
    without any display code: `applyOp` already hands `payload.notices` to the
    viewer.  A private `notes` key, which is what this shipped with first, is
    a second door onto the same fact and the panel dropped it on the floor.
    """

    @pytest.fixture()
    def client(self):
        from molbuilder.web.app import create_app
        return create_app(config={}).test_client()

    def _slab(self, client, **kw):
        from support.envelope import from_xyz
        body = {"structure": from_xyz("1\nx\nAu 0 0 0\n"), "element": "Au",
                "plane": "111", "m": 2, "n": 2, "layers": 6, "start_z": 0.0}
        body.update(kw)
        return client.post("/api/modify/slab", json=body).get_json()

    def test_a_fresh_slab_reports_the_c_nobody_has_set_yet(self, client):
        """`junction-cell.md` § 6: the builder leaves `c` as the extent, so a
        fresh slab collides with its own image -- and is TOLD SO, on the build,
        rather than the person discovering it at the engine.

        This asserted `eclipsed` on a bad layer count until 2026-08-31.  With
        no padding the boundary is a collision first, and the stacking question
        cannot be asked of a box whose faces are on top of each other."""
        j = self._slab(client, layers=4)
        assert j["ok"] is True, "a warning, never a refusal (§ 4.1)"
        warns = [n for n in j["notices"] if n["level"] == "warn"]
        assert warns, "a slab with an unset c must say so"
        assert "collision" in warns[0]["message"]
        assert "not padded" in warns[0]["message"]

    def test_it_says_the_same_thing_whatever_the_layer_count(self, client):
        """The layer-count question belongs to the seam, and there is no seam
        until `c` is set.  A good layer count does not buy a quiet build --
        it cannot, because nothing has been decided about the box yet."""
        for layers in (4, 6):
            j = self._slab(client, layers=layers)
            assert [n["level"] for n in j["notices"]] == ["warn"], layers

    def test_it_uses_the_one_channel_and_not_a_second_door(self, client):
        j = self._slab(client, layers=4)
        assert "notes" not in j, (
            "a private key beside `notices` is a second door onto the same "
            "fact, and nothing on the client reads it")

    def test_the_receipt_carries_the_shape_every_notice_carries(self, client):
        """`level` + `message` + a stable `where` + the subject it is about.

        `where` is the id, so a finding is identifiable without parsing its
        prose.  `about` decides WHERE it is shown -- and it is deliberately
        not "cell", which routes to the Cell page: a seam is fixed by changing
        the layer count or the placement, both of which are in the Slab panel.
        """
        seen = self._slab(client, layers=4)["notices"]
        assert {"level", "message", "where", "about"} <= set(seen[0])
        assert seen[0]["where"] == "slab.seam_collision"
        assert seen[0]["about"] != "cell"

    def test_the_periodicity_receipts_are_not_trampled(self, client):
        """The merge, not an assignment.

        `ok_structure_response` computes the cell's own conditions and
        prepends the caller's receipts.  Handing it a list must not cost the
        gate its findings, which is the failure that helper exists to make
        impossible.
        """
        from support.envelope import from_xyz
        body = {"structure": from_xyz("1\nx\nAu 0 0 0\n"), "element": "Au",
                "plane": "111", "m": 2, "n": 2, "layers": 4, "start_z": 0.0}
        got = client.post("/api/modify/slab", json=body).get_json()["notices"]
        assert got[0]["about"] == "slab", "receipts come first (molview.md § 6.8)"
        assert all("where" in n for n in got)
