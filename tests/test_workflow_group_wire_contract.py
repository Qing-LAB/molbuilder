"""Wire-shape contract test: every issue-emitting render endpoint
MUST enrich the issues response with ``workflow_group`` when the
field carries that metadata.

History
=======

2026-06-13 task #373 shipped per-card issue routing for SIESTA +
PySCF Build forms.  The renderer iterates Issue objects + reads
``issue.workflow_group`` to pick which ``.card-issues[data-
workflow-group="<role>"]`` UL to write to.  This works because
``web/blueprints/build.py`` passes ``cfg=cfg`` to
``_issues_to_json``, which calls ``resolve_workflow_group(where,
cfg)`` to look up the field metadata and enrich each Issue.

2026-06-14 batch F4b extended the client renderer to spectra +
transport tabs (added the same ``.card-issues`` fan-out in
``lib/spectra/core.js::renderIssues`` and
``lib/transport/core.js::_renderIssues``).

The audit-driven follow-up (this commit) caught a BLOCKER: the
SERVER side never received the same treatment.  Spectra +
transport's render endpoints called
``_issues_to_json(issues)`` *without* the ``cfg`` kwarg, so
``resolve_workflow_group`` short-circuited to ``None`` and every
issue went out the wire un-tagged.  The client fanout had no
data to route — every issue landed in the residual panel, the
per-card UL stayed empty, and the entire F4b feature was DEAD
on those two tabs.

The 6 e2e tests we already had (``test_build_form_live_preflight_
fires_on_field_edit`` etc.) only covered SIESTA, where the
server WAS correct.  No test asserted on the wire shape for
spectra / transport, so the drift slipped through.

What this file pins
===================

For every issue-emitting render endpoint:

  1. POST a payload that triggers at least one validator warning
     on a field that DOES carry ``workflow_group`` metadata
     (mesh_cutoff < 100 Ry for SIESTA, scf_max_iter > 5000 for
     PySCF, low-N integration mesh for spectra, vibration mode
     for transport).

  2. Assert the response shape:
     - ``ok`` is True OR there are non-empty ``issues``
     - At least one ``issue.workflow_group`` field is present + non-null.
     - The field equals one of the three valid roles
       (profile / stage / budget) — confirms the enrichment path
       resolved the metadata correctly.

A regression that drops ``cfg=cfg`` from any future
``_issues_to_json`` call fails this test loudly.

This is L3 (web client + dataclass + validator) — not L5 — so it
runs in <0.5 s and doesn't depend on a browser.  The L5 layer is
covered by ``test_build_form_live_preflight_fires_on_field_edit``
for SIESTA and would be parametrised over spectra + transport in
a future pass.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest


pytest.importorskip("flask")


# --------------------------------------------------------------------- #
#  Fixtures                                                              #
# --------------------------------------------------------------------- #


# A tiny benzene structure -- enough atoms that the spectra +
# transport engines accept it without "system too small" failures
# but small enough that the render is fast.
_BENZENE_XYZ = """\
12

C    0.000    1.396    0.000
C    1.209    0.698    0.000
C    1.209   -0.698    0.000
C    0.000   -1.396    0.000
C   -1.209   -0.698    0.000
C   -1.209    0.698    0.000
H    0.000    2.481    0.000
H    2.149    1.240    0.000
H    2.149   -1.240    0.000
H    0.000   -2.481    0.000
H   -2.149   -1.240    0.000
H   -2.149    1.240    0.000
"""


_AU_BDT_AU_XYZ = """\
4

Au   0.000   0.000   0.000
Au   2.886   0.000   0.000
Au   1.443   2.500   0.000
Au   1.443   0.833   2.357
"""


_VALID_ROLES = {"profile", "stage", "budget"}


# --------------------------------------------------------------------- #
#  Helpers                                                               #
# --------------------------------------------------------------------- #


def _post(web, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """POST + assert 2XX/4XX and return JSON.  Lets callers test
    against either a successful response with warnings OR a
    preflight-failed response with errors — the workflow_group
    contract applies to both."""
    r = web.post(path, json=body)
    assert r.status_code in (200, 400), (
        f"unexpected status {r.status_code} from {path}: "
        f"{r.get_data(as_text=True)[:200]}"
    )
    return r.get_json() or {}


def _assert_workflow_group_enrichment(
    issues: List[Dict[str, Any]], endpoint: str,
) -> None:
    """Pin that at least one issue carries a valid ``workflow_group``
    field.  This is the load-bearing contract — if every issue has
    ``workflow_group: None`` or the field is missing entirely, the
    server forgot to pass ``cfg`` into ``_issues_to_json`` and the
    client-side card-issues fan-out is dead.
    """
    assert issues, (
        f"{endpoint} returned zero issues; pick a payload that "
        f"reliably triggers at least one validator warning."
    )
    tagged = [
        i for i in issues
        if i.get("workflow_group") is not None
    ]
    assert tagged, (
        f"{endpoint} returned issues but NONE carry a "
        f"``workflow_group`` field.  The server forgot to pass "
        f"``cfg=cfg`` to ``_issues_to_json``; client-side card-"
        f"issues fan-out is dead.  Issues: "
        f"{[(i.get('where'), i.get('message', '')[:40]) for i in issues]!r}"
    )
    for i in tagged:
        g = i["workflow_group"]
        assert g in _VALID_ROLES, (
            f"{endpoint} returned issue with workflow_group={g!r}; "
            f"expected one of {sorted(_VALID_ROLES)}."
        )


# --------------------------------------------------------------------- #
#  SIESTA build  (already correct; regression-pin)                       #
# --------------------------------------------------------------------- #


def test_siesta_build_issues_carry_workflow_group(web_client):
    """SIESTA's build.py is the reference implementation -- it
    passes ``cfg=cfg`` everywhere.  This pin would catch a future
    regression that drops the kwarg from any of build.py:726, 731,
    747."""
    body = _post(web_client, "/api/build/fdf", {
        "xyz": _BENZENE_XYZ,
        # mesh_cutoff = 30 is < 100 Ry, outside the declared range;
        # config-metadata validator emits a warn whose ``where`` is
        # ``config.mesh_cutoff``, which carries workflow_group="stage".
        "params": {"mesh_cutoff": 30},
    })
    assert body.get("ok") is True, body.get("error")
    _assert_workflow_group_enrichment(
        body.get("issues") or [], "/api/build/fdf",
    )


def test_pyscf_build_issues_carry_workflow_group(web_client):
    """PySCF mirror of the SIESTA pin.  ``build.py:793,796,812`` all
    pass cfg=cfg."""
    body = _post(web_client, "/api/build/pyscf", {
        "xyz": _BENZENE_XYZ,
        # Same OOB approach: pick a PySCF field with workflow_group +
        # range.  ``scf_max_cycle`` is range=(10, 1000),
        # workflow_group="budget" (see config/pyscf.py:172).
        "params": {"scf_max_cycle": 5},
    })
    assert body.get("ok") is True, body.get("error")
    _assert_workflow_group_enrichment(
        body.get("issues") or [], "/api/build/pyscf",
    )


# --------------------------------------------------------------------- #
#  /api/build/preflight                                                  #
# --------------------------------------------------------------------- #


def test_siesta_preflight_issues_carry_workflow_group(web_client):
    body = _post(web_client, "/api/build/preflight", {
        "xyz": _BENZENE_XYZ,
        "engine": "siesta",
        "params": {"mesh_cutoff": 30},
    })
    assert body.get("ok") is True, body.get("error")
    _assert_workflow_group_enrichment(
        body.get("issues") or [],
        "/api/build/preflight (siesta)",
    )


def test_pyscf_preflight_issues_carry_workflow_group(web_client):
    body = _post(web_client, "/api/build/preflight", {
        "xyz": _BENZENE_XYZ,
        "engine": "pyscf",
        "params": {"scf_max_cycle": 5},
    })
    assert body.get("ok") is True, body.get("error")
    _assert_workflow_group_enrichment(
        body.get("issues") or [],
        "/api/build/preflight (pyscf)",
    )


# --------------------------------------------------------------------- #
#  /api/spectra/render  (the BLOCKER that motivated this file)           #
# --------------------------------------------------------------------- #


def test_spectra_render_issues_carry_workflow_group(web_client):
    """Pre-fix this test FAILED — spectra.py:434 called
    ``_issues_to_json(issues)`` without ``cfg=cfg``.  Every issue
    went out with ``workflow_group: None`` and the F4b client-side
    fan-out routed zero issues to the per-card panels.

    Post-fix it passes because spectra.py now mirrors build.py's
    enrichment.  A regression that drops the kwarg again fails
    this loudly.
    """
    body = _post(web_client, "/api/spectra/render", {
        "structure_text": _BENZENE_XYZ,
        # SpectraConfig.scf_max_cycle range=(10, 1000),
        # workflow_group="budget" (config/spectra.py:581).  Value 5
        # is below the floor -> reliable warn.
        "params": {"scf_max_cycle": 5},
    })
    # spectra preflight either succeeds-with-warnings or fails-with-errors;
    # both branches must carry the workflow_group on warns.
    issues = body.get("issues") or []
    _assert_workflow_group_enrichment(issues, "/api/spectra/render")


# --------------------------------------------------------------------- #
#  /api/transport/render  (the OTHER BLOCKER)                            #
# --------------------------------------------------------------------- #


def test_transport_render_issues_carry_workflow_group(
        web_client, tmp_path, monkeypatch):
    """Pre-fix this test FAILED — transport.py:249,250,284 called
    ``_issues_to_json(issues)`` without ``cfg=cfg``.  Same dead-
    code symptom as spectra.

    Post-fix transport's render passes cfg into every _issues_to_json
    site, so workflow_group enrichment runs and the client's
    per-card panels populate.

    Transport requires (a) a relaxed geometry .xyz on disk and (b)
    a valid sidecar with ``electrode_1`` / ``channel`` / ``electrode_2``
    regions — the Au-BDT-Au validation fixture (#337) provides both.
    Copy them into tmp_path + register tmp_path as a picker root so
    the endpoint accepts the path.
    """
    import shutil
    from pathlib import Path

    from molbuilder import diagnostics
    caps = diagnostics.Capabilities(
        runtime_config={}, conda_binary=None, conda_envs=frozenset(),
    )
    cls = type(caps)
    monkeypatch.setattr(
        cls, "file_picker_roots",
        lambda self: ((tmp_path.resolve(), "projects"),),
    )
    diagnostics.set_capabilities(caps)

    fixture_dir = Path(__file__).parent / "data"
    xyz_path = tmp_path / "au_bdt_au.xyz"
    sidecar_path = tmp_path / "au_bdt_au.molstruct.json"
    shutil.copy(fixture_dir / "au_bdt_au.xyz", xyz_path)
    shutil.copy(fixture_dir / "au_bdt_au.molstruct.json", sidecar_path)

    body = _post(web_client, "/api/transport/render", {
        "structure_path": str(xyz_path.resolve()),
        "params": {
            "engine": "transiesta",
            # TransportConfig.electronic_temperature_k range=(10, 2000),
            # workflow_group="profile" (config/transport.py:230).
            # Value 5 K is below floor → reliable metadata-range warn.
            "electronic_temperature_k": 5.0,
        },
    })
    # The clean Au-BDT-Au fixture passes structural preflight, so
    # the response is 200 OK with non-empty ``issues`` carrying the
    # metadata-range warn.  Both ``issues`` and (preflight-error
    # only) ``errors_only`` must carry workflow_group enrichment.
    if "errors_only" in body and body["errors_only"]:
        _assert_workflow_group_enrichment(
            body["errors_only"], "/api/transport/render (errors_only)",
        )
    else:
        issues = body.get("issues") or []
        _assert_workflow_group_enrichment(
            issues, "/api/transport/render",
        )
