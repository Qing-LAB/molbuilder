"""Wire-shape contract test: every issue-emitting endpoint
MUST enrich the issues response with ``workflow_group`` when the
field carries that metadata.

Called *without* the ``cfg`` kwarg, ``_issues_to_json`` short-circuits
``resolve_workflow_group`` to ``None``: every issue goes out the wire
un-tagged and lands in the residual panel.  A regression that drops
``cfg=cfg`` from any ``_issues_to_json`` call fails this test loudly.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pytest

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from support.envelope import from_xyz as _env


pytest.importorskip("flask")


# --------------------------------------------------------------------- #
#  Fixtures                                                              #
# --------------------------------------------------------------------- #


# Benzene puckered slightly out of plane (alternating C/H z) so the derived
# vacuum cell isn't degenerate at vacuum=0 (a perfectly planar molecule has a
# zero-thickness axis -- structure-periodicity.md).  These tests exercise the
# workflow_group issue-enrichment wiring, not geometry.
_BENZENE_XYZ = """\
12

C    0.000    1.396    0.200
C    1.209    0.698   -0.200
C    1.209   -0.698    0.200
C    0.000   -1.396   -0.200
C   -1.209   -0.698    0.200
C   -1.209    0.698   -0.200
H    0.000    2.481    0.400
H    2.149    1.240   -0.400
H    2.149   -1.240    0.400
H    0.000   -2.481   -0.400
H   -2.149   -1.240    0.400
H   -2.149    1.240   -0.400
"""


_AU_BDT_AU_XYZ = """\
4

Au   0.000   0.000   0.000
Au   2.886   0.000   0.000
Au   1.443   2.500   0.000
Au   1.443   0.833   2.357
"""


#: DERIVED from the closed vocabulary, never listed.  The contract here is
#: *"a role on the wire is one the vocabulary knows"*, and the vocabulary is
#: `template.GROUPS` -- so ask it.
from molbuilder.template import GROUPS as _GROUPS      # noqa: E402

_VALID_ROLES = set(_GROUPS)


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
#  /api/build/preflight                                                  #
# --------------------------------------------------------------------- #


def test_siesta_preflight_issues_carry_workflow_group(web_client):
    body = _post(web_client, "/api/build/preflight", {
        "structure": _env(_BENZENE_XYZ),
        "engine": "siesta", "calculation": "optimization",
        "params": {"mesh_cutoff": 30},
    })
    assert body.get("ok") is True, body.get("error")
    _assert_workflow_group_enrichment(
        body.get("issues") or [],
        "/api/build/preflight (siesta)",
    )


def test_pyscf_preflight_issues_carry_workflow_group(web_client):
    body = _post(web_client, "/api/build/preflight", {
        "structure": _env(_BENZENE_XYZ),
        "engine": "pyscf", "calculation": "optimization",
        "params": {"scf_max_cycle": 5},
    })
    assert body.get("ok") is True, body.get("error")
    _assert_workflow_group_enrichment(
        body.get("issues") or [],
        "/api/build/preflight (pyscf)",
    )


def test_cfg_none_path_correctly_omits_workflow_group():
    """The other half of the contract: when ``_issues_to_json`` is
    called WITHOUT a cfg (e.g. ``workspace_payload`` in _shared.py
    that emits geometry-only warns), the resolver must SHORT-CIRCUIT
    rather than guess a workflow_group from the where string.

    Pin so a future refactor that tries to "be helpful" by inferring
    the group from the field name (``config.mesh_cutoff`` -> stage)
    silently rewires a non-engine surface to per-card routing and
    breaks the data-independent residual UI on every untagged
    endpoint.

    L1 / L2 contract — no web client needed.
    """
    from molbuilder.issues import Issue
    from molbuilder.web.blueprints._shared import issues_to_json

    # Two issues: one with a ``where`` that LOOKS like a config
    # field (would tempt an inference-by-name heuristic) and one
    # that's genuinely struct-level.
    issues = [
        Issue("warn", "tempting to infer", "config.mesh_cutoff"),
        Issue("error", "real struct issue",  "struct.regions"),
    ]
    serialized = issues_to_json(issues, cfg=None)
    assert len(serialized) == 2
    for i in serialized:
        assert i.get("workflow_group") in (None, ""), (
            f"cfg=None must short-circuit workflow_group resolution; "
            f"got {i.get('workflow_group')!r} on issue {i!r}.  A "
            f"future refactor that infers the group from ``where`` "
            f"is what this test forbids."
        )


def test_cfg_present_does_resolve_workflow_group():
    """Complement: with cfg actually passed, the resolver looks up
    the field metadata and propagates ``workflow_group`` onto the
    Issue dict.  Pin the producer side directly (no Flask client)
    so a regression in ``resolve_workflow_group`` doesn't require
    re-running the endpoint tests above to surface.
    """
    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.issues import Issue
    from molbuilder.web.blueprints._shared import issues_to_json

    cfg = SiestaConfig()
    # mesh_cutoff carries workflow_group="stage" per config/siesta.py.
    issues = [Issue("warn", "out of range", "config.mesh_cutoff")]
    serialized = issues_to_json(issues, cfg=cfg)
    assert len(serialized) == 1
    assert serialized[0].get("workflow_group") == "stage", (
        f"cfg=SiestaConfig + where='config.mesh_cutoff' should "
        f"resolve workflow_group='stage'; got "
        f"{serialized[0].get('workflow_group')!r}."
    )


# ── #64: the task-setup save ────────────────────────────────────────────────

def _cfg_field_with_a_group():
    """A SiestaConfig field that declares a `workflow_group`, and a value of
    the wrong type for it -- so the preflight raises a `config.<key>` finding.
    """
    import dataclasses
    from molbuilder.config.siesta import SiestaConfig
    for f in dataclasses.fields(SiestaConfig):
        if f.metadata.get("workflow_group") and f.type in ("int", int):
            return f.name, f.metadata["workflow_group"]
    raise AssertionError("no int field carries a workflow_group any more")


def test_the_task_setup_save_findings_carry_their_workflow_group(
        web_client, isolated_projects_root, tmp_path):
    """A `config.<key>` finding from the task-setup SAVE route reaches the page
    with its `workflow_group`, so the card can hold it.

    `web/ui-contract.md` Rule 2. MEASURED DEFECT (#64, 2026-09-09):
    `build.py:1760` and `:1805` called `_issues_to_json(_pf)` with no `cfg`, so
    `workflow_group` was omitted and the page had nowhere to put the finding.
    """
    import json as _json
    from support.envelope import envelope
    key, group = _cfg_field_with_a_group()
    d = isolated_projects_root / "wg" / "optimization" / "probe"
    d.mkdir(parents=True, exist_ok=True)
    # The description is built by the HAND-OVER door, not hand-listed, so a
    # field the schema grows lands here instead of being forgotten.
    over = web_client.post("/api/task-setup/handover", json=dict(
        structure=envelope(["H", "H"], [[0, 0, 0], [0, 0, 0.74]]),
        engine="siesta", calculation="optimization", name="probe",
        params={"system_label": "probe"})).get_json()
    assert over and over.get("ok"), over
    (d / over["handover_name"]).write_text(over["handover_text"])
    h = _json.loads(over["handover_text"])
    proposed = {"schema": "molbuilder/task@1", "engine": h["engine"],
                "calculation": h["calculation"],
                "shape": "flat", "run": h["run"], "structure": h["structure"],
                # the field must be PROMOTED before a stage may override it
                # (`engines/stages.md` § 6.2), which the codec enforces before
                # the preflight ever runs.
                "varies": [key],
                # a STRING where the field is an int -> `config.<key>`
                "stages": [{"name": "coarse", "enabled": True,
                            "overrides": {key: "not-an-int"}}]}
    r = web_client.post("/api/task-setup/save",
                        json={"dest": str(d), "text": _json.dumps(proposed)})
    body = r.get_json()
    findings = body.get("findings") or []
    ours = [f for f in findings if f.get("where") == f"config.{key}"]
    assert ours, (
        f"no `config.{key}` finding on the wire; got "
        f"{[f.get('where') for f in findings]} (status {r.status_code})")
    for f in ours:
        assert f.get("workflow_group") == group, (
            "the finding reached the page with no workflow_group, so the card "
            f"cannot hold it: {f}")
