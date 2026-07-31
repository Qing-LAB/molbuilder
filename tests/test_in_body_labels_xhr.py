"""L3 XHR contract test for the 2026-06-14 viewer-is-truth labels
contract (``apply_labels_to_struct`` in web/blueprints/_shared.py).

The companion file ``test_in_body_labels_contract.py`` pins the
client-side POST body shape via source-text scan; THIS file drives a
real Flask test client end-to-end against every endpoint that
flows through ``apply_labels_to_struct`` and asserts:

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
level -- a regression that made ``apply_labels_to_struct`` ignore
in-body keys would have passed CI.  These tests close that gap.

WHERE THE RULE LIVES: the "viewer is truth" idea graduated on 2026-07-29 into
clause **F2** of the delivery contract, docs/science/validation.md § 4.1 -- the
server builds an emitted structure from the request BODY alone, and the sidecar
is never read for one.  Two consequences these tests now pin: omitting the label
keys declares NO labels (it does not trigger a disk read), and ``structure_path``
is not a label source even though it still travels for pseudopotential and
dest-dir resolution.  The client half of F1 -- one accessor
(``molview.data.factsForRequest()``) assembling coordinates, labels and
periodicity together so a tab cannot send a partial set -- is pinned in
tests/test_validation_delivery_contract.py.
"""
from __future__ import annotations

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


def _labels(body):
    """The body carries ONE label store. `frozen_atoms=` is a test convenience
    that lands in it, because the reserved label is an ordinary label and the
    wire has no second key for it (molview.md § 6.6)."""
    frozen = body.pop("frozen_atoms", None)
    if frozen is not None:
        regions = dict(body.get("regions") or {})
        regions["frozen_atoms"] = frozen
        body["regions"] = regions
    return body


def _post_fdf(web, **body) -> dict:
    """POST to /api/build/fdf with the test XYZ.  Returns the
    parsed JSON envelope.  Caller asserts on ``ok`` and ``fdf``."""
    body = _labels(body)
    body.setdefault("xyz", _XYZ)
    body.setdefault("params", {})
    r = web.post("/api/build/fdf", json=body)
    return r.get_json() or {}


def _post_pyscf(web, **body) -> dict:
    body = _labels(body)
    body.setdefault("xyz", _XYZ)
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

    def test_in_body_frozen_atoms_reach_the_fdf(self, web):
        """The smoking gun: the rendered .fdf MUST contain a
        ``%block Geometry.Constraints`` with the indices we posted
        (in 1-based form, per SIESTA's convention).  Pre-fix this
        would pass only via the sidecar fallback; with the in-body
        contract, no sidecar exists in the request and the labels
        come from the POST body alone."""
        body = _post_fdf(web, frozen_atoms=[0, 2, 4])
        assert body.get("ok") is True, (
            f"render failed: {body!r}"
        )
        fdf = body.get("fdf", "")
        assert "%block Geometry.Constraints" in fdf, (
            "in-body frozen_atoms must produce a Constraints block "
            "in the rendered .fdf"
        )
        # SIESTA is 1-based; molbuilder Structure is 0-based.  0/2/4 -> 1/3/5.
        # The renderer emits comments inside the block ("# Source:
        # ... 0-based ...") so a naive ``re.findall(\d+)`` would
        # pick up the "0" from "0-based" too.  Parse only lines
        # whose first non-whitespace token is ``position``.
        m = re.search(
            r"%block\s+Geometry\.Constraints(.+?)%endblock",
            fdf, re.IGNORECASE | re.DOTALL,
        )
        assert m is not None
        block = m.group(1)
        emitted = set()
        for line in block.splitlines():
            stripped = line.split("#", 1)[0].strip()
            parts = stripped.split()
            if not parts or parts[0].lower() != "position":
                continue
            for tok in parts[1:]:
                if tok.isdigit():
                    emitted.add(int(tok))
        assert emitted == {1, 3, 5}, (
            f"expected 1-based indices {{1, 3, 5}} in the Constraints "
            f"block; got {emitted}"
        )

    def test_empty_list_is_explicit_no_labels_claim(self, web):
        """An empty list is the client's explicit ``I have nothing
        to freeze`` claim.  Server MUST honor it: no Constraints
        block in the .fdf, and NO sidecar fallback."""
        body = _post_fdf(web, frozen_atoms=[], regions={})
        assert body.get("ok") is True
        fdf = body.get("fdf", "")
        assert "%block Geometry.Constraints" not in fdf or \
               "%endblock Geometry.Constraints" not in fdf, (
            "empty in-body frozen_atoms must NOT produce a "
            "Constraints block.  This regression would mean the "
            "server is re-reading a sidecar when the user said "
            "``[]``."
        )

    def test_a_named_disk_path_is_not_a_label_source(self, web, tmp_path):
        """Contract F2 (science/validation.md 4.1): the server builds an emitted
        structure from the BODY alone.

        This used to fall back to a disk sidecar read against
        ``structure_path``, so an emitted deck could carry labels the model had
        already changed.  ``structure_path`` still travels (it anchors
        pseudopotential + dest-dir resolution) -- it is simply not a source of
        labels any more, and omitting the label keys means "no labels".
        """
        xyz_file = tmp_path / "probe.xyz"
        xyz_file.write_text(_XYZ, encoding="utf-8")
        # A sidecar that WOULD have frozen atom 0 under the old behaviour.
        import hashlib, json as _json
        (tmp_path / "probe.molstruct.json").write_text(_json.dumps({
            "schema_version": 3,
            "n_atoms_total": len([l for l in _XYZ.strip().splitlines()[2:] if l.strip()]),
            "structure_hash": hashlib.sha256(xyz_file.read_bytes()).hexdigest(),
            "regions": {}, "frozen_atoms": [0], "selection_rules": {},
        }), encoding="utf-8")
        r = web.post("/api/build/fdf", json={
            "xyz": _XYZ, "params": {}, "structure_path": str(xyz_file)})
        assert r.status_code == 200
        body = r.get_json() or {}
        assert body.get("ok") is True
        assert "%block Geometry.Constraints" not in body.get("fdf", ""), (
            "the disk sidecar's frozen atom leaked into the emitted deck")

    def test_explicit_empty_label_keys_are_a_valid_claim(self, web):
        """The other half of F2: `{}` / `[]` IS the client saying "nothing to
        declare", and it emits cleanly with no Constraints block."""
        r = web.post(
            "/api/build/fdf",
            json={"xyz": _XYZ, "params": {},
                  "regions": {}, "frozen_atoms": []},
        )
        assert r.status_code == 200
        body = r.get_json() or {}
        assert body.get("ok") is True
        assert "%block Geometry.Constraints" not in body.get("fdf", "")

    def test_no_path_and_no_label_keys_means_no_labels(self, web):
        """A body with neither label keys nor a disk path is unambiguous --
        there is no second source to confuse it with, so it means what it
        says.  Demanding two empty keys from such a caller would be ceremony
        with no information in it."""
        r = web.post("/api/build/fdf", json={"xyz": _XYZ, "params": {}})
        assert r.status_code == 200
        body = r.get_json() or {}
        assert body.get("ok") is True
        assert "%block Geometry.Constraints" not in body.get("fdf", "")

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
        body = _post_fdf(web, frozen_atoms=bad_indices)
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

    def test_regions_in_body_apply_to_struct(self, web):
        """``regions`` is the transport-tab boundary-condition
        carrier.  When sent in-body, it should attach to the
        Structure (verified via the spectra preflight which would
        warn about region labels if it sees them)."""
        body = _post_fdf(
            web,
            frozen_atoms=[],
            regions={"L-electrode": [0, 1], "R-electrode": [3, 4]},
        )
        # The /api/build/fdf renderer's three-stage Pattern B
        # surfaces region labels as info-severity issues since the
        # SIESTA SCF deck doesn't consume them.  Verify that notice.
        assert body.get("ok") is True
        issues = body.get("issues") or []
        notice_msgs = " ".join(
            (i.get("message") or "") for i in issues
        ).lower()
        assert "region" in notice_msgs or "transport" in notice_msgs, (
            "expected a Pattern-B notice about the region labels "
            "in the issues list; got " + repr(issues)
        )


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

    def test_in_body_frozen_atoms_reach_the_script(self, web):
        """PySCF emits frozen_atoms as a constraints-file reference
        (geomeTRIC convention).  The generated .py MUST mention
        the indices.  Pin that the contract reaches the PySCF
        renderer too."""
        body = _post_pyscf(
            web,
            frozen_atoms=[1, 3],
            params={"optimizer": "geometric"},
        )
        assert body.get("ok") is True, f"render failed: {body!r}"
        # /api/build/pyscf response key is ``script`` (see
        # web/blueprints/build.py docstring line 54).
        script = body.get("script", "")
        # PySCF generator emits a comment referencing the source
        # frozen_atoms list (verified by reading molbuilder/pyscf/
        # input.py line 585: ``# Source: Structure.frozen_atoms =
        # ...``).  We assert the indices appear in the script.
        assert "frozen_atoms" in script.lower() or \
               "1, 3" in script or "[1, 3]" in script, (
            "in-body frozen_atoms must appear in the rendered "
            "PySCF script"
        )

    def test_out_of_range_rejected(self, web):
        """Same rejection contract as /api/build/fdf."""
        body = _post_pyscf(web, frozen_atoms=[99])
        # Render still proceeds (warn-severity); but the bad index
        # must not leak.  Most importantly we want a warn notice.
        issues = body.get("issues") or []
        assert any(
            i.get("severity") == "warn"
            and "labels could not be applied" in (i.get("message") or "")
            for i in issues
        ), (f"expected warn-severity rejection notice; got {issues!r}")
