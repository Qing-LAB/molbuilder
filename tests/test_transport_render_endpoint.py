"""Tests for POST /api/transport/render (Transport B.3 step 2).

Pins the contract:

* The endpoint dispatches via :func:`molbuilder.transport.get_engine`
  so a new engine drops in without endpoint code changes.
* Wire shape mirrors Spectra's render endpoint:
  ``{ok, engine, script, filename, issues, errors_only}`` on
  success; ``{ok=False, error, issues, errors_only}`` when preflight
  blocks emission.  ``errors_only`` is the pre-filtered
  error-severity subset of ``issues`` — see transport.py's
  field-meaning comment block for the full envelope shape.
* The structure path goes through the picker-root allowlist so a
  client can't read arbitrary files.
* Sidecar (.molstruct.json) is applied if reachable — that's where
  the region labels come from.
* Errors in preflight (missing regions, empty electrodes) return
  HTTP 400 with the issues attached, NOT a runtime crash.
* Unknown engine = HTTP 400 with the registered-list reported.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest


pytest.importorskip("flask")


# --------------------------------------------------------------------- #
#  Fixtures                                                             #
# --------------------------------------------------------------------- #


@pytest.fixture
def web(monkeypatch, tmp_path):
    """Flask test client with the tmp_path registered as a picker root.

    Mirrors the pattern from test_preview_modal_edit_save_e2e.py +
    test_auto_detect_button.py.  Restoration of the
    Capabilities.file_picker_roots class attribute is handled by an
    autouse fixture so a leaky test can't poison the next.
    """
    from molbuilder.web.app import create_app
    from molbuilder import diagnostics

    caps = diagnostics.Capabilities(
        runtime_config={}, conda_binary=None, conda_envs=frozenset(),
    )
    monkeypatch.setattr(
        type(caps), "file_picker_roots",
        lambda self: ((tmp_path.resolve(), "projects"),),
    )
    diagnostics.set_capabilities(caps)
    return create_app(config={}).test_client(), tmp_path


def _write_xyz(dir_: Path, name: str, atoms: list) -> Path:
    """Write a minimal XYZ and a matching .molstruct.json sidecar
    with the region labels needed for transiesta preflight to pass.
    """
    xyz_path = dir_ / name
    n = len(atoms)
    lines = [str(n), f"transport test fixture: {name}"]
    for el, (x, y, z) in atoms:
        lines.append(f"{el}  {x:.4f}  {y:.4f}  {z:.4f}")
    xyz_path.write_text("\n".join(lines) + "\n")
    return xyz_path


def _write_sidecar(xyz_path: Path, regions: dict) -> Path:
    """Write a .molstruct.json sidecar next to the XYZ with the
    given region map.  Uses the canonical sidecar shape per
    molstruct_json.py.
    """
    from molbuilder.sidecars.molstruct import (
        sidecar_path_for, sha256_of_file, to_dict,
    )
    sidecar_path = sidecar_path_for(xyz_path)
    payload = to_dict(
        {"regions": regions},
        n_atoms_total=len(xyz_path.read_text().splitlines()) - 2,
        structure_hash=sha256_of_file(xyz_path),
    )
    sidecar_path.write_text(json.dumps(payload))
    return sidecar_path


# --------------------------------------------------------------------- #
#  Happy path                                                           #
# --------------------------------------------------------------------- #


def test_render_returns_fdf_for_labeled_au_s_junction(web):
    """End-to-end: labeled Au-S-Au structure + sidecar → engine
    emits a runnable .fdf via the registry dispatch.
    """
    client, tmp = web
    proj = tmp / "auaud"
    proj.mkdir()
    xyz = _write_xyz(proj, "au_s_au.xyz", [
        ("Au", (0, 0, 0)), ("Au", (2, 0, 0)),
        ("S",  (4, 0, 0)), ("C",  (6, 0, 0)),
        ("Au", (8, 0, 0)), ("Au", (10, 0, 0)),
    ])
    _write_sidecar(xyz, {
        "L-electrode":  [0, 1],
        "bridge":       [2, 3],
        "R-electrode":  [4, 5],
    })
    r = client.post("/api/transport/render", json={
        "structure_path": str(xyz),
        "params": {"engine": "transiesta", "job_name": "ausau"},
    })
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body["ok"] is True
    assert body["engine"] == "transiesta"
    assert body["filename"] == "ausau.fdf"
    # Script carries the load-bearing keywords (already pinned per-
    # keyword in test_transport_transiesta.py; here we just confirm
    # the endpoint returned a non-empty .fdf).
    assert "TS.SolutionMethod" in body["script"]
    assert "SystemLabel            ausau" in body["script"]


# --------------------------------------------------------------------- #
#  Preflight error blocks emission                                      #
# --------------------------------------------------------------------- #


def test_render_blocks_when_regions_missing(web):
    """No sidecar (= no region labels) → preflight returns error →
    endpoint returns 400 with ``errors_only`` populated and NO
    script.  The user sees the issues panel before a runtime
    failure.
    """
    client, tmp = web
    proj = tmp / "unlabeled"
    proj.mkdir()
    xyz = _write_xyz(proj, "raw.xyz", [
        ("C", (0, 0, 0)), ("H", (1, 0, 0)),
    ])
    # NO sidecar written → struct.regions stays empty
    r = client.post("/api/transport/render", json={
        "structure_path": str(xyz),
        "params": {"engine": "transiesta", "job_name": "unlabeled"},
    })
    # web-api.md § 1.6 (b): validator hard-fail is scientific
    # advisory — HTTP 200 + ok:false, not 4xx.
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body["ok"] is False
    assert body.get("script") is None
    assert any(
        "region" in e["message"].lower() for e in body["errors_only"]
    )


# --------------------------------------------------------------------- #
#  Registry dispatch                                                    #
# --------------------------------------------------------------------- #


def test_render_unknown_engine_returns_400(web):
    """A bogus engine name surfaces as a clean 400 with the list
    of registered engines (so the user can correct a typo).  This
    pins the on-ramp claim: the endpoint never hardcodes the
    engine list.
    """
    client, tmp = web
    proj = tmp / "engine_test"
    proj.mkdir()
    xyz = _write_xyz(proj, "x.xyz", [("C", (0, 0, 0))])
    r = client.post("/api/transport/render", json={
        "structure_path": str(xyz),
        "params": {"engine": "no_such_engine"},
    })
    assert r.status_code == 400
    body = r.get_json()
    assert "no_such_engine" in body["error"]
    assert "transiesta" in body["error"]   # listed as registered


# --------------------------------------------------------------------- #
#  Path-traversal protection                                            #
# --------------------------------------------------------------------- #


def test_render_rejects_path_outside_picker_root(web):
    """Path validation goes through ``_resolve_path_within_roots``.
    A path outside the picker root must be rejected as 400/403.
    Pin so an endpoint refactor that drops the gate gets caught.
    """
    client, tmp = web
    r = client.post("/api/transport/render", json={
        "structure_path": "/etc/passwd",
        "params": {"engine": "transiesta"},
    })
    assert r.status_code in (400, 403), r.get_data(as_text=True)
    body = r.get_json()
    assert body["ok"] is False


def test_render_missing_structure_path_returns_400(web):
    """No structure path → 400 with a clean error message."""
    client, _ = web
    r = client.post("/api/transport/render", json={
        "params": {"engine": "transiesta"},
    })
    assert r.status_code == 400
    assert "structure_path" in r.get_json()["error"]


# --------------------------------------------------------------------- #
#  Wire-shape stability                                                 #
# --------------------------------------------------------------------- #


def test_render_response_has_documented_shape(web):
    """The response keys are part of the endpoint contract; the JS
    consumer reads ``script``, ``filename``, ``issues``,
    ``engine``, ``ok``.  Pin all five on the happy path.
    """
    client, tmp = web
    proj = tmp / "shape_test"
    proj.mkdir()
    xyz = _write_xyz(proj, "j.xyz", [
        ("Au", (0, 0, 0)), ("Au", (2, 0, 0)),
        ("S",  (4, 0, 0)), ("C",  (6, 0, 0)),
        ("Au", (8, 0, 0)), ("Au", (10, 0, 0)),
    ])
    _write_sidecar(xyz, {
        "L-electrode":  [0, 1],
        "bridge":       [2, 3],
        "R-electrode":  [4, 5],
    })
    r = client.post("/api/transport/render", json={
        "structure_path": str(xyz),
        "params": {"engine": "transiesta", "job_name": "shape"},
    })
    assert r.status_code == 200
    body = r.get_json()
    for key in (
        "ok", "engine", "script", "filename", "issues", "errors_only",
    ):
        assert key in body, f"response missing documented key {key!r}"


def test_render_coerces_bias_voltages_v_from_comma_string(web):
    """Regression for the 2026-06-11 review: ``bias_voltages_v`` is
    ``List[float]``.  The form renders it as a comma-floats text
    input that sends the value as a string ("0.0, 0.5, 1.0").  The
    server-side coercer MUST parse that into ``List[float]`` before
    the dataclass sees it — without coercion the dataclass stored
    the raw string and the engine then crashed slicing ``bias[0]``.
    """
    client, tmp = web
    proj = tmp / "bias_coerce"
    proj.mkdir()
    xyz = _write_xyz(proj, "j.xyz", [
        ("Au", (0, 0, 0)), ("Au", (2, 0, 0)),
        ("S",  (4, 0, 0)), ("C",  (6, 0, 0)),
        ("Au", (8, 0, 0)), ("Au", (10, 0, 0)),
    ])
    _write_sidecar(xyz, {
        "L-electrode":  [0, 1],
        "bridge":       [2, 3],
        "R-electrode":  [4, 5],
    })
    r = client.post("/api/transport/render", json={
        "structure_path": str(xyz),
        "params": {
            "engine":          "transiesta",
            "job_name":        "biasc",
            # Comma-floats text input shape — the bug was that this
            # string used to reach the dataclass unchanged.
            "bias_voltages_v": "0.0, 0.5",
        },
    })
    assert r.status_code == 200, r.get_data(as_text=True)
    body = r.get_json()
    assert body["ok"] is True
    # Engine emits only the first bias today (single-V .fdf per
    # render).  The 0.0 value lands on the TS.Voltage line.
    assert "TS.Voltage" in body["script"]


def test_render_preflight_error_envelope_carries_top_level_error(web):
    """Regression for the 2026-06-11 review: Spectra's preflight
    error envelope carries a top-level ``error`` field
    (``"preflight failed; see issues"``) but Transport's previously
    omitted it.  An envelope-handler that gates on ``error`` (the
    banner string, not the ``errors_only`` list) had no message to
    surface.  Pin parity with Spectra.
    """
    client, tmp = web
    proj = tmp / "envelope"
    proj.mkdir()
    xyz = _write_xyz(proj, "raw.xyz", [
        ("C", (0, 0, 0)), ("H", (1, 0, 0)),
    ])
    # No sidecar → struct.regions empty → preflight raises error
    r = client.post("/api/transport/render", json={
        "structure_path": str(xyz),
        "params": {"engine": "transiesta", "job_name": "envelope"},
    })
    # web-api.md § 1.6 (b): scientific advisory at HTTP 200.
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is False
    assert isinstance(body.get("error"), str) and body["error"], (
        "Transport preflight-error envelope MUST carry a top-level "
        "``error`` field (string) to match Spectra's contract."
    )
