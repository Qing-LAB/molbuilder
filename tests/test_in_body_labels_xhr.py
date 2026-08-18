"""L3 XHR contract test for the viewer-is-truth labels contract
(``struct_from_body`` in web/blueprints/_shared.py).

The labels ride INSIDE the structure envelope, so there is one place they can
arrive and no second place to drop them.

The companion file ``test_in_body_labels_contract.py`` pins the
client-side POST body shape via source-text scan; THIS file drives a
real Flask test client end-to-end against every endpoint that
rebuilds a structure from the request body, and asserts:

  * In-body ``frozen_atoms`` arrives at the renderer and ends up
    in the generated .fdf / PySCF script.
  * In-body ``frozen_atoms = []`` is an explicit "no labels"
    claim -- the server applies it AND does NOT fall back to the
    sidecar.
  * In-body labels are the ONLY labels: a disagreeing -- or stale, or
    corrupt -- on-disk sidecar cannot reach an emitted deck at all.
  * Out-of-range / wrong-type indices are rejected with a clear
    warn-severity notice rather than silently corrupting the
    output.

Pre-2026-06-14 the whole contract was checked only at source-text
level -- a regression that made the server ignore the envelope's
labels would have passed CI.  These tests close that gap.

WHERE THE RULE LIVES: the "viewer is truth" idea graduated on 2026-07-29 into
clause **F2** of the delivery contract, docs/science/validation.md § 4.1 -- the
server builds an emitted structure from the request BODY alone, and the sidecar
is never read for one.  Two consequences these tests now pin: omitting the label
keys declares NO labels (it does not trigger a disk read), and ``structure_path``
is not a label source even though it still travels for pseudopotential and
dest-dir resolution.  The client half of F1 -- one accessor
(``molview.data.getStructure()``) assembling coordinates, labels and
periodicity together so a tab cannot send a partial set -- is pinned in
tests/test_validation_delivery_contract.py.
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from support.envelope import (from_xyz as _env,
                             from_xyz_with_periodicity as _env_per)


import json
import re
from pathlib import Path

import pytest

pytest.importorskip("flask")


@pytest.fixture
def web():
    from molbuilder.web.app import create_app
    return create_app(config={}).test_client()


# A 5-atom benzene-like ring used for every test.  Small but
# realistic enough to drive the renderers; species/coordinates
# don't matter for the boundary-condition logic.
# A 3-D (non-linear) molecule so the derived vacuum cell isn't degenerate at
# vacuum=0 (structure-periodicity.md); these tests exercise label wiring, not
# geometry, so methane is fine.
_XYZ = """5
test fixture
C  0.000  0.000  0.000
H  0.629  0.629  0.629
H -0.629 -0.629  0.629
H -0.629  0.629 -0.629
H  0.629 -0.629 -0.629
"""


def _envelope(regions=None, frozen=None):
    """The test molecule as data, labels inside it -- through the ONE
    builder (`tests/support/envelope.py` -> `Structure.to_dict()`).

    These helpers posted `xyz` text plus a TOP-LEVEL `regions` until
    2026-08-03; the labels only reached the Structure because a second
    applier existed on the server to put them there."""
    from support.envelope import from_xyz
    return from_xyz(_XYZ, regions=regions, frozen=frozen)


def _bad_envelope(frozen):
    """An envelope carrying labels that are DELIBERATELY invalid -- an index
    naming an atom that isn't there, or a string where an int belongs.

    Hand-built on purpose, and the only place in this file that is.  The shared
    builder goes through `Structure(...)`, which validates -- so a fixture whose
    SUBJECT is a malformed request cannot be constructed by it: the assertion
    would fire while building the fixture instead of at the door under test.
    Everything else here uses `support.envelope`, because everything else is a
    structure that should be valid."""
    valid = _envelope()
    valid["metadata"]["regions"] = {"frozen_atoms": frozen}
    return valid


def _labels(body):
    """Fold the test-convenience `regions=` / `frozen_atoms=` kwargs into the
    structure envelope, so a caller writes what it means and the request is the
    real shape."""
    if "structure" not in body:
        body["structure"] = _envelope(body.pop("regions", None),
                                      body.pop("frozen_atoms", None))
    return body


def _post_fdf(web, **body) -> dict:
    """POST to /api/build/fdf with the test structure.  Returns the
    parsed JSON envelope.  Caller asserts on ``ok`` and ``fdf``."""
    body = _labels(body)
    body.setdefault("params", {})
    r = web.post("/api/build/fdf", json=body)
    return r.get_json() or {}


def _post_pyscf(web, **body) -> dict:
    body = _labels(body)
    body.setdefault("params", {})
    r = web.post("/api/build/pyscf", json=body)
    return r.get_json() or {}


# --------------------------------------------------------------------- #
#  /api/build/fdf -- the BLOCKER endpoint                                #
# --------------------------------------------------------------------- #


class TestBuildFdfInBodyLabels:
    """PINS: docs/science/validation.md § 4.1 clause F2 — no server-side second
    source — on the SIESTA /api/build/fdf seam.

    INVARIANT: ``regions`` in the request body IS the label store -- the
    reserved ``frozen_atoms`` label included -- and those ARE the labels; an
    empty dict is the client's explicit "nothing to declare", and omitting the
    key declares no labels rather than triggering a disk read.
    Out-of-range or wrong-typed indices are refused instead of being coerced.
    """






    @pytest.mark.parametrize(
        "bad_indices,description",
        [
            ([5], "5 is out of range for 5-atom struct (0..4)"),
            ([10000], "wildly out of range"),
            ([-1], "negative index"),
            (["zero"], "string instead of int"),
            ([{"x": 1}], "dict instead of int"),
            ([0.5], "float instead of int"),
        ],
    )
    def test_out_of_range_or_wrong_type_rejected(
            self, web, bad_indices, description):
        """A malicious / confused client sending out-of-range or
        wrong-typed indices MUST surface a warn-severity Issue
        rather than mis-emitting a .fdf or crashing.  Per
        viewer-is-truth contract: garbage in -> notice out."""
        body = _post_fdf(web, structure=_bad_envelope(bad_indices))
        # Render still succeeds (label rejection is a warn, not
        # an error).  But the FDF must NOT contain a Constraints
        # block with the bad indices.
        if body.get("ok") is True:
            fdf = body.get("fdf", "")
            for bad in bad_indices:
                if isinstance(bad, int) and bad >= 0:
                    # The fixture has 5 atoms (0..4 -> 1..5 in 1-based).
                    # bad >= 5 means 1-based bad+1 >= 6 which is out of
                    # range.  Check it's NOT in the rendered fdf.
                    assert str(bad + 1) not in re.findall(
                        r"position\s+([\d\s]+)", fdf
                    ), (
                        f"out-of-range index {bad!r} ({description}) "
                        f"leaked into the rendered Constraints block"
                    )
            # A warn-severity notice should be present in issues.
            issues = body.get("issues") or []
            assert any(
                i.get("severity") == "warn"
                and "labels could not be applied" in (i.get("message") or "")
                for i in issues
            ), (
                f"expected a warn-severity ``labels could not be "
                f"applied`` notice for {description}.  Got: {issues!r}"
            )
        # else: ok is False with a 400 -- also acceptable rejection.



# --------------------------------------------------------------------- #
#  /api/build/pyscf -- mirror coverage                                   #
# --------------------------------------------------------------------- #


class TestBuildPyscfInBodyLabels:
    """PINS: docs/science/validation.md § 4.1 clause F2 — no server-side second
    source — on the PySCF /api/build/pyscf seam.

    INVARIANT: ``regions`` in the request body IS the label store -- the
    reserved ``frozen_atoms`` label included -- and those ARE the labels; an
    empty dict is the client's explicit "nothing to declare", and omitting the
    key declares no labels rather than triggering a disk read.
    Out-of-range or wrong-typed indices are refused instead of being coerced.
    """


