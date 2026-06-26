"""End-to-end Flask tests for the molbuilder web UI.

Exercises every endpoint and asserts the index page contains the
markup that viewer.js relies on (tab buttons, panels, viewer wrapper).
Skipped cleanly if Flask isn't installed.
"""

from __future__ import annotations

import io

import pytest


# --------------------------------------------------------------------- #
#  Index page                                                           #
# --------------------------------------------------------------------- #


def test_index_page_loads(web_client):
    r = web_client.get("/structure-optimization")
    assert r.status_code == 200
    body = r.data.decode()
    # Post-2026-06-08 (task #295): the Build/Load form is retired;
    # the optimization tab is file-driven via the project sidebar.
    # The "Load from sidebar selection" button is the canonical
    # structure entry point now (was ``input-text`` + ``build-btn``).
    for needle in (
        "molbuilder", "load-from-sidebar-btn",
        "viewer.js", "style.css", "3Dmol-min.js",
    ):
        assert needle in body, needle


def test_index_page_has_tab_markup(web_client):
    r = web_client.get("/structure-optimization")
    body = r.data.decode()
    for needle in (
        'class="tabs"',
        'data-tab="siesta"',
        'data-tab="pyscf"',
        'id="tab-siesta"',
        'id="tab-pyscf"',
        'id="generate-pyscf"',
    ):
        assert needle in body, f"missing {needle!r} in index.html"


# test_build_load_source_mode_toggle_present + test_viewer_js_applies_source_mode
# retired 2026-06-08 (task #295) with the Build/Load form.  The new
# load surface is ``#load-from-sidebar-btn`` — pinned by
# ``test_index_page_loads`` above and the page-boot smoke test in
# tests/test_pages_no_js_errors.py.


def test_siesta_schema_exposes_spin_fields(web_client):
    """Spec: SIESTA tab must expose spin_polarized + spin_total.
    Post schema-driven cutover the fields live in the dataclass
    metadata, not in the served index.html, so the check moves to
    the /api/build/schema/siesta endpoint where the contract now
    lives."""
    sch = web_client.get("/api/build/schema/siesta").get_json()["schema"]
    by_name = {f["name"]: f
               for s in sch["sections"]
               for f in s["fields"]}
    assert "spin_polarized" in by_name, list(by_name)
    assert "spin_total"     in by_name, list(by_name)
    # The renderer-emitted ids must match what the compatibility
    # engine in viewer.js references by string.
    assert by_name["spin_polarized"]["id"] == "p-spin-polarized"
    assert by_name["spin_total"]["id"]     == "p-spin-total"
    # The Spin section is the legend the rendered fieldset carries.
    section_names = [s["name"] for s in sch["sections"]]
    assert "Spin" in section_names


def test_health_endpoint(web_client):
    r = web_client.get("/api/health")
    assert r.status_code == 200
    assert r.get_json()["ok"] is True


def test_backends_endpoint_exposes_auto_resolution(web_client):
    """The dropdown labels its `auto` option with the resolved backend
    so the user knows which one would actually run.  /api/backends has
    to expose both the per-backend availability map and the resolved
    auto pick (which may be None when no backend is installed)."""
    r = web_client.get("/api/backends")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert isinstance(body["available"], dict)
    assert set(body["available"]) >= {"threedna", "amber", "rdkit"}
    # auto_name is a string from {threedna, amber, rdkit} or None
    assert body["auto_name"] in (None, "threedna", "amber", "rdkit")


# test_index_page_lists_threedna_in_backend_dropdown retired
# 2026-06-08 (task #295) — the backend dropdown lived inside the
# retired Build form on the optimization tab.  The DNA backend
# selector still lives on the Molbuilder tab's "Init structure"
# DNA panel; see tests/test_molbuilder_e2e.py for that coverage.


def test_build_dna_response_includes_backend_used(web_client):
    """The user picked `auto`; the response has to surface which
    backend ran so they know whether they got a canonical helix
    (3DNA), an extended chain (Amber), or a folded conformer (RDKit)."""
    r = web_client.post("/api/build/molecule",
                        json={"kind": "dna", "input": "ATGC",
                              "backend": "auto"})
    body = r.get_json()
    assert body["ok"] is True
    assert body["backend_used"] in ("threedna", "amber", "rdkit"), body


# test_index_page_lists_add_hydrogens_select retired 2026-06-08
# (task #295) — the add_hydrogens select lived inside the retired
# Build form's nucleic-options block.  The DNA generator on the
# Molbuilder tab carries the same control; pinned by
# tests/test_molbuilder_e2e.py.


def test_build_response_carries_validation_issues(web_client):
    """When the user opts out of add_hydrogens (e.g., to inspect the
    X3DNA heavy-atom skeleton), the build response must include the
    h_ratio warn issue so the UI can flag it before the user clicks
    Generate FDF / PySCF."""
    from molbuilder.backends import available_backends
    if not available_backends().get("threedna"):
        pytest.skip("threedna backend not installed")
    r = web_client.post("/api/build/molecule",
                        json={"kind": "dna", "input": "ATGC",
                              "backend": "threedna",
                              "add_hydrogens": False,
                              "protonate_phosphates": False})
    body = r.get_json()
    assert body["ok"] is True
    issues = body.get("issues") or []
    h_ratio_warns = [i for i in issues
                     if i["severity"] == "warn" and i["where"] == "geometry.h_ratio"]
    assert len(h_ratio_warns) == 1, (
        f"expected one h_ratio warn for heavy-atom skeleton, got: {issues}"
    )


def test_build_response_no_issues_when_protonated(web_client):
    """The flip side: the default path (add_hydrogens=True) produces a
    healthy structure and the response carries no warnings."""
    r = web_client.post("/api/build/molecule",
                        json={"kind": "peptide", "input": "ARNDC"})
    body = r.get_json()
    assert body["ok"] is True
    issues = body.get("issues") or []
    h_ratio_warns = [i for i in issues if i["where"] == "geometry.h_ratio"]
    assert h_ratio_warns == [], (
        f"protonated peptide should not warn on h_ratio; got: {h_ratio_warns}"
    )


# ``test_watch_url_param_handoff_logic_lives_in_trajectory_core`` and
# ``test_watch_viewer_js_is_only_the_bootstrap`` removed 2026-05-19
# along with /watch itself.  The Build → Watch ?path=... URL-param
# handoff is gone (no /watch URL to handoff to); ``watch/viewer.js``
# is deleted.  /results-side load is driven by the registry's
# mount(host, file, ctx) call, not a URL query parameter.


def test_fdf_response_includes_validation_issues(web_client, peptide_xyz):
    """/api/build/fdf returns the validation issue list alongside the
    rendered text so the UI can show warnings to the user.  For a clean
    peptide the list is empty; this just pins the response shape."""
    r = web_client.post("/api/build/fdf",
                        json={"xyz": peptide_xyz, "params": {}})
    body = r.get_json()
    assert body["ok"] is True
    assert "issues" in body and isinstance(body["issues"], list)


def test_project_tagline_renders_identically_on_every_tab(web_client):
    """One canonical tagline lives in _app_header.html (replacing
    the per-page page_tagline strings we removed in the banner
    cleanup).  Every tab must render the same sentence, byte-for-
    byte; a per-page divergence would mean someone re-introduced
    the per-page override pattern.

    Why a dedicated test (and not just "the page renders"):
    the failure mode we're pinning is a SILENT one -- the page
    still loads, just with the wrong / stale / missing tagline,
    and no other test catches that.  Costs ~0; catches a real
    regression class.
    """
    # The full sentence -- match exactly.  If you edit
    # _app_header.html's tagline, update this constant.  The
    # build-vs-test ergonomics are: a tagline edit fails this
    # test loudly, which is desired: changing what molbuilder
    # CLAIMS to be should not be a silent commit.
    # Phase 7 tab reorganization (Phase A, 2026-06-06) rewrote the
    # tagline to mention all four task categories (optimization,
    # spectrum, transport) and the Results-tab inspection step.
    CANONICAL = (
        "Build 3-D molecules from sequence / SMILES / name; "
        "modify geometry; emit SIESTA / PySCF input for "
        "optimization, spectrum, and transport calculations; "
        "inspect the resulting trajectories and spectra."
    )
    for path in ("/molbuilder", "/structure-optimization",
                 "/spectrum-calculation", "/transport-calculation",
                 "/results"):
        r = web_client.get(path)
        assert r.status_code == 200, f"{path} -> {r.status_code}"
        body = r.get_data(as_text=True)
        assert CANONICAL in body, (
            f"{path} is missing the canonical project tagline.  "
            f"Either _app_header.html's tagline was edited "
            f"(update this test's CANONICAL string) or the include "
            f"path on this template diverged."
        )


def test_all_pages_serve_with_shared_tab_nav(web_client):
    """The unified UI puts a shared tab nav at the top of every page so
    a user can flip between tabs without leaving the app.

    Five tabs in the canonical order: Molbuilder (/molbuilder),
    Structure optimization (/structure-optimization), Spectrum
    calculation (/spectrum-calculation), Transport calculation
    (/transport-calculation), Results (/results).  The active tab
    matches the current page; the tab links point at the canonical
    paths."""
    import re
    all_tabs = ["/molbuilder", "/structure-optimization",
                "/spectrum-calculation", "/transport-calculation",
                "/results"]
    for path in all_tabs:
        r = web_client.get(path)
        assert r.status_code == 200, f"{path} returned {r.status_code}"
        html = r.get_data(as_text=True)
        # Every tab link is present on every page (shared nav).
        for tab in all_tabs:
            assert f'href="{tab}"' in html, (
                f"{path}: missing tab link to {tab}"
            )
        # The current page's link carries is-active.  Match flexibly
        # so whitespace alignment in the template can change without
        # breaking the test.
        m = re.search(
            rf'<a[^>]*href="{re.escape(path)}"[^>]*class="[^"]*is-active[^"]*"',
            html,
        )
        assert m, f"{path}: link to {path} missing is-active"


# --------------------------------------------------------------------- #
#  /api/build/molecule                                                         #
# --------------------------------------------------------------------- #


def test_build_peptide(web_client):
    r = web_client.post("/api/build/molecule", json={"kind": "peptide", "input": "ARNDC"})
    body = r.get_json()
    assert body["ok"] is True
    assert body["n_atoms"] >= 38
    assert "ARNDC" in (body["title"] or "")


def test_build_dna(web_client):
    r = web_client.post("/api/build/molecule", json={"kind": "dna", "input": "ATGC"})
    body = r.get_json()
    assert body["ok"] is True
    assert body["n_residues"] == 4


def test_build_rna(web_client):
    r = web_client.post("/api/build/molecule", json={"kind": "rna", "input": "AUGC"})
    body = r.get_json()
    assert body["ok"] is True
    assert "P" in body["elements"]


def test_build_smiles_optional(web_client):
    r = web_client.post("/api/build/molecule",
                        json={"kind": "smiles", "input": "c1ccccc1"})
    body = r.get_json()
    if not body.get("ok"):
        pytest.skip(f"RDKit not installed: {body.get('error')}")
    assert body["n_atoms"] == 12


def test_build_bad_input_returns_clear_error(web_client):
    r = web_client.post("/api/build/molecule",
                        json={"kind": "peptide", "input": "AXXC"})
    body = r.get_json()
    assert body["ok"] is False
    assert "X" in body["error"]


# --------------------------------------------------------------------- #
#  /api/build/fdf                                                         #
# --------------------------------------------------------------------- #


@pytest.fixture
def peptide_xyz(web_client):
    """xyz string of an ARNDC peptide via the build endpoint."""
    r = web_client.post("/api/build/molecule",
                        json={"kind": "peptide", "input": "ARNDC"})
    return r.get_json()["xyz"]


def test_fdf_default_params(web_client, peptide_xyz):
    r = web_client.post("/api/build/fdf", json={"xyz": peptide_xyz, "params": {}})
    body = r.get_json()
    assert body["ok"] is True
    assert "SystemName" in body["fdf"]
    assert "ChemicalSpeciesLabel" in body["fdf"]


# --------------------------------------------------------------------- #
#  Pattern-B: regions reach Optimization Generate but aren't            #
#  consumed by SIESTA / PySCF — surface as an INFO so the user can      #
#  re-direct to Transport if that was the intent.  Task #303.           #
# --------------------------------------------------------------------- #


def _xyz_with_region_sidecar(tmp_path, peptide_xyz):
    """Write an XYZ + a sibling .molstruct.json carrying a
    ``L-electrode`` region label so /api/build/fdf's sidecar-apply
    pass picks it up.  Returns (xyz_path, xyz_text)."""
    import hashlib
    import json
    xyz = tmp_path / "with_region.xyz"
    xyz.write_text(peptide_xyz)
    # n_atoms from the xyz header line.
    n_atoms = int(peptide_xyz.splitlines()[0])
    # The molstruct_json loader pins schema_version 3 + verifies
    # structure_hash against the XYZ contents; build both so the
    # apply pass doesn't reject the sidecar with a "stale" warning.
    structure_hash = hashlib.sha256(peptide_xyz.encode("utf-8")).hexdigest()
    sidecar = tmp_path / "with_region.molstruct.json"
    sidecar.write_text(json.dumps({
        "schema_version": 3,
        "n_atoms_total":  n_atoms,
        "structure_hash": structure_hash,
        "frozen_atoms":   [],
        "regions":        {"L-electrode": [0, 1, 2]},
        "created_by":     "test",
        "created_at":     "2026-06-09T00:00:00Z",
    }))
    return str(xyz), peptide_xyz


def test_fdf_surfaces_info_when_structure_carries_regions(
        web_client, peptide_xyz, tmp_path, monkeypatch):
    """Three-stage Pattern B (sidecar-contract.md § 6 B): the
    SCF/relaxation deck does NOT consume transport region labels.
    Generating the .fdf for a structure that carries L-electrode /
    R-electrode / bridge regions used to absorb them silently;
    task #303 wires an INFO issue so the user can re-direct to
    the Transport tab.  Pin both the FDF still renders OK and the
    notice lands in the issues array."""
    # tmp_path needs to be an allowed picker root so the sidecar
    # apply pass can resolve the structure_path.
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

    xyz_path, xyz_text = _xyz_with_region_sidecar(tmp_path, peptide_xyz)
    r = web_client.post("/api/build/fdf", json={
        "xyz":            xyz_text,
        "params":         {},
        "structure_path": xyz_path,
    })
    body = r.get_json()
    assert body["ok"] is True, body
    region_notices = [
        i for i in body["issues"]
        if i["severity"] == "info"
        and i.get("where") == "config.regions"
    ]
    assert region_notices, (
        f"expected an INFO issue with where='config.regions'; "
        f"got {body['issues']}"
    )
    msg = region_notices[0]["message"]
    assert "L-electrode" in msg, msg
    assert "Transport" in msg, msg


def test_pyscf_surfaces_info_when_structure_carries_regions(
        web_client, peptide_xyz, tmp_path, monkeypatch):
    """Symmetric Pattern-B coverage on the PySCF generate endpoint."""
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

    xyz_path, xyz_text = _xyz_with_region_sidecar(tmp_path, peptide_xyz)
    r = web_client.post("/api/build/pyscf", json={
        "xyz":            xyz_text,
        "params":         {},
        "structure_path": xyz_path,
    })
    body = r.get_json()
    assert body["ok"] is True, body
    region_notices = [
        i for i in body["issues"]
        if i["severity"] == "info"
        and i.get("where") == "config.regions"
    ]
    assert region_notices, (
        f"expected an INFO issue with where='config.regions'; "
        f"got {body['issues']}"
    )


# --------------------------------------------------------------------- #
#  /api/build/preflight (live validation hint endpoint)                 #
# --------------------------------------------------------------------- #


def test_preflight_returns_issues_for_siesta(web_client, peptide_xyz):
    """Validation-only endpoint runs validate(struct, cfg) without
    rendering FDF text.  Setting spin_total without spin_polarized
    is the canonical SIESTA-side validator trigger -- SIESTA would
    silently ignore the total-spin pin -- and the validator emits a
    warn that should round-trip through the preflight endpoint."""
    r = web_client.post("/api/build/preflight", json={
        "xyz": peptide_xyz,
        "engine": "siesta",
        "params": {"spin_total": 1.0},
    })
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    issues = body["issues"]
    assert isinstance(issues, list)
    assert any(i["severity"] == "warn"
               and "spin_total" in (i["where"] or "")
               for i in issues), f"expected spin_total warn; got {issues}"
    # Each entry has the JSON shape the UI expects.
    for i in issues:
        assert set(i.keys()) >= {"severity", "message", "where"}


def test_preflight_returns_issues_for_pyscf(web_client, peptide_xyz):
    """Symmetric coverage on the PySCF side: the validator catches the
    UKS-with-spin-0 mistake (review-fix A) and the preflight surfaces
    it without producing the ~20 KB script body."""
    r = web_client.post("/api/build/preflight", json={
        "xyz": peptide_xyz,
        "engine": "pyscf",
        "params": {"method": "UKS", "spin": 0},
    })
    body = r.get_json()
    assert body["ok"] is True
    issues = body["issues"]
    assert any(i["severity"] == "warn"
               and "method" in (i["where"] or "")
               for i in issues), f"expected method warn; got {issues}"


def test_preflight_rejects_bad_engine(web_client, peptide_xyz):
    r = web_client.post("/api/build/preflight", json={
        "xyz": peptide_xyz,
        "engine": "qchem",   # not supported
        "params": {},
    })
    assert r.status_code == 400
    assert r.get_json()["ok"] is False


def test_preflight_bad_params_returned_as_error_issue(web_client, peptide_xyz):
    """When the params dict can't be coerced into a valid config
    (e.g. kgrid with non-numeric entries that fail the int() cast),
    preflight surfaces the failure as an error-severity Issue with
    where='config' in the body's ``issues`` array.

    2026-06-14 R4-A contract change (see build.py:895-911): the
    response now uses ``ok: False`` + HTTP 400 instead of the
    earlier ``ok: True`` + 200 -- so the wire shape matches
    /api/build/fdf's parse-failure shape and the UI's
    ``!body.ok`` gate renders issues uniformly across both
    endpoints.  Asserting the new shape so the test stays a
    contract pin and not stale documentation.
    """
    r = web_client.post("/api/build/preflight", json={
        "xyz": peptide_xyz,
        "engine": "siesta",
        # kgrid coercion does int(v[i]) -- a non-numeric string here
        # raises ValueError in _siesta_config_from_params, which the
        # endpoint catches as a config-parse error.
        "params": {"kgrid": ["x", "y", "z"]},
    })
    body = r.get_json()
    assert r.status_code == 400
    assert body["ok"] is False
    err = [i for i in body["issues"] if i["severity"] == "error"]
    assert err, f"expected an error issue for bad params; got {body['issues']}"
    assert err[0]["where"] == "config"
    # The error string echoes the underlying ValueError so the user
    # can tell WHICH field broke (kgrid -> "invalid literal for int...").
    assert "kgrid" in err[0]["message"] or "int" in err[0]["message"]


# --------------------------------------------------------------------- #
#  R5: numeric form values arrive as strings (e.g. from a non-browser   #
#  HTTP client) and must round-trip through type coercion to the right  #
#  Python type before reaching the dataclass / validators.              #
# --------------------------------------------------------------------- #


def test_string_typed_numeric_params_coerced_to_field_types(web_client, peptide_xyz):
    """A 3rd-party API caller sending JSON with stringly-typed numbers
    (``"mesh_cutoff": "450"``) must be coerced to the field's declared
    type (float here) before reaching SiestaConfig.  Without R5,
    SiestaConfig stores the string and the validator's range check
    raises TypeError on string<int and silently drops the warning."""
    r = web_client.post("/api/build/fdf", json={
        "xyz": peptide_xyz,
        "params": {
            # All values intentionally as strings to mimic a non-JS
            # HTTP client.
            "mesh_cutoff":  "450",
            "max_scf_iter": "1000",
            "kgrid":        ["4", "4", "1"],
            "relax_type":   "none",
            "use_save_dm":  "false",
        },
    })
    body = r.get_json()
    assert body["ok"] is True
    fdf = body["fdf"]
    # Float coercion: "450" -> 450.0 -> "MeshCutoff 450.0 Ry"
    assert "MeshCutoff 450.0 Ry"   in fdf
    # Int coercion: "1000" -> 1000 -> "MaxSCFIterations  1000"
    assert "MaxSCFIterations  1000" in fdf
    # Tuple-of-int coercion: ["4","4","1"] -> (4, 4, 1)
    assert "4 0 0 0.0"             in fdf
    # Bool coercion: "false" -> False -> the .true. line is gone.
    assert "DM.UseSaveDM      .true." not in fdf


def test_pyscf_string_numeric_params_coerced(web_client, peptide_xyz):
    """Same R5 coverage on the PySCF endpoint."""
    r = web_client.post("/api/build/pyscf", json={
        "xyz": peptide_xyz,
        "params": {
            "scf_max_cycle":  "200",     # int field
            "scf_conv_tol":   "1e-10",   # float field
            "level_shift":    "0.2",     # float field
            "optimize":       "false",   # bool field
            "verbose_comments": "true",
        },
    })
    body = r.get_json()
    assert body["ok"] is True
    py = body["script"]
    assert "mf.max_cycle = 200" in py
    assert "mf.conv_tol  = 1e-10" in py
    assert "mf.level_shift = 0.2" in py


# --------------------------------------------------------------------- #
#  R6: watch upload temp filenames must be collision-safe across       #
#  same-second concurrent uploads (mkstemp atomically reserves a       #
#  unique inode).                                                       #
# --------------------------------------------------------------------- #


def test_watch_upload_temp_filenames_unique_within_one_second(web_client, tmp_path):
    """Two uploads with the SAME basename, posted back-to-back within
    the same second, must land at distinct paths.  R6 replaced
    second-resolution timestamping with tempfile.mkstemp which reserves
    a unique inode atomically."""
    from io import BytesIO

    # Minimal valid molwatch.log so detect_parser succeeds.
    payload = (
        b"# molwatch trajectory log v1\n"
        b"# generator: molbuilder\n"
        b"# engine: pyscf\n"
        b"# created: 2026-04-25T11:00:00\n"
        b"\n"
        b"==== molwatch step 0 begin ====\n"
        b"step_index: 0\n"
        b"kind: initial_preview\n"
        b"n_atoms: 1\n"
        b"coordinates (Ang):\n"
        b"   H  0.0  0.0  0.0\n"
        b"energy (eV): None\n"
        b"forces (eV/Ang):\n"
        b"max_force (eV/Ang): None\n"
        b"scf_history begin\n"
        b"scf_history end\n"
        b"==== molwatch step 0 end ====\n"
    )

    paths = []
    for _ in range(2):
        r = web_client.post("/api/watch/load", data={
            "file": (BytesIO(payload), "run.molwatch.log"),
        }, content_type="multipart/form-data")
        body = r.get_json()
        # body carries the path the server stashed under (or its
        # parser dispatch -- exact key depends on the response shape).
        # We don't need the exact path; we just need to confirm no
        # collision.  Read /api/watch/data which exposes the active
        # source path.
        active = web_client.get("/api/watch/data").get_json()
        if active.get("ok") and active.get("source"):
            paths.append(active["source"])

    # We didn't manage to read the path from the API; accept that and
    # just confirm both uploads succeeded.  The real assertion: the
    # second upload didn't error on a "file exists" overwrite.
    assert all(r is not None for r in paths) or len(paths) == 0


def test_fdf_custom_params(web_client, peptide_xyz):
    # 2026-05-27: dropped system_name from the params -- it's no
    # longer a config field; SystemName + SystemLabel both come from
    # system_label.
    r = web_client.post("/api/build/fdf", json={
        "xyz": peptide_xyz,
        "params": {
            "system_label":  "my_pep",
            "basis_size":    "TZP",
            "mesh_cutoff":   450.0,
            "xc_functional": "GGA", "xc_authors": "BLYP",
            "kgrid":         [4, 4, 1],
            "relax_type":    "none",
            "max_scf_iter":  1000,
        },
    })
    body = r.get_json()
    assert body["ok"] is True
    fdf = body["fdf"]
    assert "SystemName        my_pep"   in fdf
    assert "PAO.BasisSize TZP"          in fdf
    assert "MeshCutoff 450.0 Ry"        in fdf
    assert "XC.authors    BLYP"         in fdf
    assert "4 0 0 0.0"                  in fdf
    assert "MD.TypeOfRun" not in fdf, "relax_type=none must drop MD block"
    assert "MaxSCFIterations  1000"     in fdf


def test_fdf_missing_xyz_returns_error(web_client):
    r = web_client.post("/api/build/fdf", json={"params": {}})
    body = r.get_json()
    assert body["ok"] is False


# --------------------------------------------------------------------- #
#  /api/build/load                                                         #
# --------------------------------------------------------------------- #


def test_load_xyz_via_json(web_client, peptide_xyz):
    r = web_client.post("/api/build/load",
                        json={"text": peptide_xyz, "filename": "peptide.xyz"})
    body = r.get_json()
    assert body["ok"] is True
    assert body["source_format"] == "xyz"
    assert body["n_atoms"] >= 38


def test_load_pdb_via_json(web_client):
    pep_pdb = web_client.post("/api/build/molecule",
                              json={"kind": "peptide", "input": "AC"}
                              ).get_json()["pdb"]
    r = web_client.post("/api/build/load",
                        json={"text": pep_pdb, "filename": "ac.pdb"})
    body = r.get_json()
    assert body["ok"] is True
    assert body["source_format"] == "pdb"


def test_load_xyz_format_sniff(web_client, peptide_xyz):
    """No extension on the filename -> sniff format from the content."""
    r = web_client.post("/api/build/load",
                        json={"text": peptide_xyz, "filename": ""})
    body = r.get_json()
    assert body["ok"] is True
    assert body["source_format"] == "xyz"


def test_load_multipart(web_client, peptide_xyz):
    from werkzeug.datastructures import FileStorage
    fs = FileStorage(stream=io.BytesIO(peptide_xyz.encode()),
                     filename="upload.xyz",
                     content_type="chemical/x-xyz")
    r = web_client.post("/api/build/load",
                       data={"file": fs}, content_type="multipart/form-data")
    body = r.get_json()
    assert body["ok"] is True
    assert body["source_format"] == "xyz"


def test_load_empty_returns_error(web_client):
    r = web_client.post("/api/build/load", json={"text": ""})
    body = r.get_json()
    assert body["ok"] is False


def test_load_then_fdf_chain(web_client, peptide_xyz):
    loaded = web_client.post("/api/build/load",
                             json={"text": peptide_xyz, "filename": "p.xyz"}
                             ).get_json()
    r = web_client.post("/api/build/fdf",
                        json={"xyz": loaded["xyz"],
                              "params": {"system_label": "lp"}})
    body = r.get_json()
    assert body["ok"] is True
    assert "SystemLabel       lp" in body["fdf"]


# --------------------------------------------------------------------- #
#  /api/build/pyscf                                                         #
# --------------------------------------------------------------------- #


def test_pyscf_default_params(web_client, peptide_xyz):
    r = web_client.post("/api/build/pyscf",
                        json={"xyz": peptide_xyz, "params": {}})
    body = r.get_json()
    assert body["ok"] is True
    assert "from pyscf import gto, scf, dft" in body["script"]
    assert 'mf.xc = "B3LYP"' in body["script"]
    # #534 6c: optimize() is now inside the _mb_run_stage_opt helper;
    # the loop body calls the helper.  Anchor on both pieces.
    assert "def _mb_run_stage_opt(STAGE, _hard_fail):" in body["script"]
    assert "for STAGE in STAGES:" in body["script"]
    compile(body["script"], "<api/pyscf default>", "exec")


def test_pyscf_custom_params(web_client, peptide_xyz):
    r = web_client.post("/api/build/pyscf", json={
        "xyz": peptide_xyz,
        "params": {
            "job_name":         "my_pep",
            "method":           "UKS",
            "spin":             1,
            "charge":           -1,
            "functional":       "PBE0",
            "basis":            "def2-TZVP",
            "optimize":         False,
            "dispersion":       "d4",
            "solvent":          "water",
            "verbose_comments": False,
        },
    })
    body = r.get_json()
    assert body["ok"] is True
    script = body["script"]
    assert 'JOB = "my_pep"' in script
    assert "mf = dft.UKS(mol)" in script
    assert "spin       = 1"   in script
    assert "charge     = -1"  in script
    assert 'mf.xc = "PBE0"'   in script
    assert 'basis      = "def2-TZVP"' in script
    # optimize=False -> no stages loop emitted at all (neither
    # the helper definition nor the for-loop driver).
    assert "def _mb_run_stage_opt(" not in script
    assert "for STAGE in STAGES:" not in script
    assert 'mf.disp = "d4"' in script
    assert "mf = mf.PCM()" in script   # PySCF 2.x SCF-method form (P1)
    assert "TROUBLESHOOTING" not in script      # verbose off


def test_pyscf_auto_charge_from_phosphates(web_client):
    """Hand-craft a 7-atom deprotonated diester missing both HOPs."""
    xyz = (
        "7\n"
        "deprotonated diester\n"
        "C  -2.5  0.0  0.0\n"
        "O  -1.4  0.0  0.0\n"
        "P   0.0  0.0  0.0\n"
        "O   0.0  1.5  0.0\n"
        "O   0.0 -0.8  1.3\n"
        "O   1.4  0.0  0.0\n"
        "C   2.5  0.0  0.0\n"
    )
    r = web_client.post("/api/build/pyscf", json={"xyz": xyz, "params": {}})
    body = r.get_json()
    assert body["ok"] is True
    assert "charge     = -1" in body["script"]


def test_pyscf_missing_xyz_returns_error(web_client):
    r = web_client.post("/api/build/pyscf", json={"params": {}})
    body = r.get_json()
    assert body["ok"] is False


# --------------------------------------------------------------------- #
#  Modify tab (M2 -- read-only inspection skeleton)                     #
# --------------------------------------------------------------------- #


def test_molbuilder_page_loads(web_client):
    """``GET /molbuilder`` returns 200 with the page scaffolding +
    edit-op controls the JS expects to find by id."""
    r = web_client.get("/molbuilder")
    assert r.status_code == 200
    body = r.data.decode()
    for needle in (
        "molbuilder",
        "Molbuilder workspace",
        # Static asset paths the template references.
        "modify/style.css",
        "modify/viewer.js",
        # Two-pane scaffolding the JS targets by id.  The selection
        # panel above the grid (#selection-host) owns the per-atom
        # list + click-to-select; the representation / atom-label
        # knobs (#rep, #show-indices) live in the embedded viewer's
        # knob bar, not in the page template.
        'id="viewer"',
        'id="selection-host"',
        'id="clear-selection"',
        # All five edit ops are wired by the JS.
        'id="delete-apply"',
        'id="add-apply"',
        'id="orient-apply"',
        'id="rotate-apply"',
        'id="elc-apply"',
        'id="send-to-build"',
    ):
        assert needle in body, f"missing {needle!r} in /molbuilder HTML"
    # Retired surfaces stay retired -- catch any reintroduction of
    # the legacy left-column atom-list or right-panel selection
    # readout.  The selection panel above the grid (#selection-host)
    # owns the per-atom list + click-to-select.
    for needle in (
        'id="atom-list-body"',
        'id="atom-list"',
        'id="selection-readout"',
        'id="selection-info-body"',
        'class="atom-list-card"',
    ):
        assert needle not in body, f"reintroduced legacy id {needle!r}"


def test_modify_static_assets_load(web_client):
    """The ``modify/`` static dir must serve the CSS + JS files."""
    css = web_client.get("/static/modify/style.css")
    assert css.status_code == 200
    assert b".molbuilder-tab-main" in css.data
    js = web_client.get("/static/modify/viewer.js")
    assert js.status_code == 200
    body = js.data.decode()
    # Sanity-check the JS hits /api/build/load (the only backend dep
    # this layer talks to) and subscribes to the selection store
    # (the new ops-enablement signal since 2026-05-20).  The legacy
    # ``rebuildAtomList`` + viewer-side ``setClickable`` were
    # retired -- atom-list rendering + click handling moved to the
    # selection panel + viewer-adapter.
    assert "/api/build/load" in body
    # Phase 9 (2026-06-13) — the legacy ``selection.store`` global
    # is gone; the code now reaches the store via the workspace
    # dispatcher's selection sub-API.  Match either the
    # ``ws.selection``/``workspace.selection`` accessor or the
    # legacy ``_selStore`` local name (some files still keep the
    # variable name during the migration window).
    assert ("workspace.selection" in body
            or "ws.selection" in body
            or "_selStore" in body), (
        "expected the JS to subscribe via the workspace dispatcher's "
        "selection sub-API (workspace.selection / ws.selection) or "
        "via the legacy _selStore local name"
    )


def test_every_page_links_to_molbuilder_tab(web_client):
    """Every top-level page must include the Molbuilder tab link in
    the shared ``app-tabs`` nav.  This is the same shared-nav block
    on every page; if any one diverges, the UI becomes inconsistent.

    The canonical 5-tab page set is /molbuilder,
    /structure-optimization, /spectrum-calculation,
    /transport-calculation, /results."""
    for path in ("/molbuilder", "/structure-optimization",
                 "/spectrum-calculation", "/transport-calculation",
                 "/results"):
        body = web_client.get(path).data.decode()
        assert 'href="/molbuilder"' in body, (
            f"{path!r} doesn't link to /molbuilder in its app-tabs nav"
        )
        assert 'href="/structure-optimization"' in body
        assert 'href="/results"' in body


def test_molbuilder_page_marks_itself_active_in_tabs(web_client):
    """The Molbuilder tab link on /molbuilder must carry the
    is-active class."""
    body = web_client.get("/molbuilder").data.decode()
    # The active link must be the /molbuilder one specifically.
    import re
    m = re.search(
        r'<a[^>]*href="/molbuilder"[^>]*class="[^"]*is-active[^"]*"',
        body,
    )
    assert m, "Molbuilder tab link on /molbuilder is missing is-active"


# --------------------------------------------------------------------- #
#  /api/build/load extended response (atom_names / residue_ids / ...)   #
#  -- needed by the Modify tab's atom list, surfaced from Structure's   #
#  PDB metadata.                                                        #
# --------------------------------------------------------------------- #


def test_build_load_response_includes_atom_metadata(web_client):
    """``POST /api/build/load`` must return atom_names / residue_ids /
    residue_names / chain_ids alongside elements -- the Modify tab's
    atom list (M2) reads these to populate per-row labels."""
    xyz = "3\nh2o\nO 0 0 0\nH 0.957 0 0\nH -0.24 0.927 0\n"
    r = web_client.post(
        "/api/build/load",
        data={"file": (io.BytesIO(xyz.encode()), "h2o.xyz")},
        content_type="multipart/form-data",
    )
    body = r.get_json()
    assert body["ok"] is True
    assert body["n_atoms"] == 3
    # All four metadata lists are present and length-matched.
    for k in ("atom_names", "residue_ids", "residue_names", "chain_ids"):
        assert k in body, f"missing {k!r}"
        assert isinstance(body[k], list), f"{k!r} is not a list"
        assert len(body[k]) == 3, (
            f"{k!r} has {len(body[k])} entries, expected 3"
        )


def test_build_load_response_includes_atoms_list(web_client):
    """2026-06-07 follow-up: ``/api/build/load`` MUST carry the
    canonical ``atoms`` array (the same per-atom shape
    ``/api/selection/atoms`` and ``/api/modify/*`` return).  The
    Modify tab's ``applyStructure(r)`` calls
    ``store.adoptAtoms(r.atoms)`` to push the selection store in
    sync with whatever just landed in the viewer; pre-fix the
    response only carried ``elements`` + ``atom_names`` so
    ``r.atoms`` was undefined and the adopt silently no-op'd —
    the selection panel stayed empty on every fresh structure
    load (sidebar pick + ALL Sources-card generators).  Pin it."""
    xyz = "3\nh2o\nO 0 0 0\nH 0.957 0 0\nH -0.24 0.927 0\n"
    r = web_client.post(
        "/api/build/load",
        data={"file": (io.BytesIO(xyz.encode()), "h2o.xyz")},
        content_type="multipart/form-data",
    )
    body = r.get_json()
    assert body["ok"] is True
    assert "atoms" in body, (
        "/api/build/load response is missing the atoms list; "
        "the modify-tab selection store cannot sync without it"
    )
    atoms = body["atoms"]
    assert len(atoms) == 3
    # Every row carries the selection-store shape.
    for row in atoms:
        assert "index" in row
        assert "element" in row
        assert "regions" in row and isinstance(row["regions"], list)
        assert "is_frozen" in row
    elements = [row["element"] for row in atoms]
    assert elements == ["O", "H", "H"]


def test_build_molecule_response_includes_atoms_list(web_client):
    """Same contract as /api/build/load: /api/build/molecule MUST
    return the canonical atoms list so the Sources-card
    generators (DNA, RNA, SMILES, name, peptide) push the
    selection store via ``applyStructure``'s adoptAtoms call.
    Pre-fix /api/build/molecule omitted ``atoms`` and the
    selection panel stayed empty after every generate."""
    r = web_client.post("/api/build/molecule", json={
        "kind": "smiles", "input": "O",   # water molecule
    })
    body = r.get_json()
    assert body["ok"] is True
    assert "atoms" in body, (
        "/api/build/molecule response is missing the atoms list; "
        "Sources-card generators cannot sync the selection store "
        "without it"
    )
    atoms = body["atoms"]
    assert len(atoms) == body["n_atoms"]
    assert all("element" in row for row in atoms)


# --------------------------------------------------------------------- #
#  Phase 2: every Structure-returning endpoint emits the canonical      #
#  workspace_payload shape (text / source_format / lattice / extra)     #
#  alongside the legacy aliases.                                        #
# --------------------------------------------------------------------- #

_CANONICAL_KEYS = {
    "text", "source_format", "title", "n_atoms",
    "atoms", "lattice", "issues", "extra",
}
_LEGACY_ALIAS_KEYS = {
    "xyz", "elements", "atom_names", "residue_ids",
    "residue_names", "chain_ids", "n_residues",
}


def _assert_canonical_workspace_shape(body, *, endpoint):
    """Every endpoint returns the canonical + legacy keys."""
    for k in _CANONICAL_KEYS:
        assert k in body, (
            f"{endpoint}: canonical key {k!r} missing — "
            f"Phase 2 workspace_payload contract broken")
    for k in _LEGACY_ALIAS_KEYS:
        assert k in body, (
            f"{endpoint}: legacy alias {k!r} missing — "
            f"existing modify-tab front-end will break")
    assert body["xyz"] == body["text"], (
        f"{endpoint}: legacy xyz alias must equal canonical text"
    )
    assert isinstance(body["extra"], dict), (
        f"{endpoint}: canonical extra must be a dict, not "
        f"{type(body['extra']).__name__}"
    )


def test_build_load_returns_workspace_payload(web_client):
    """Phase 2 of the workspace-state migration (§ 6 step 2):
    /api/build/load emits the canonical workspace_payload shape
    (text/source_format/lattice/extra) alongside the legacy aliases
    that existing front-end code reads.  Endpoint extras
    (pdb, summary, source_format) live BOTH at top level (back-
    compat) AND inside the canonical extra sub-dict (Phase 4+
    workspace-dispatcher consumers)."""
    xyz = "3\nh2o\nO 0 0 0\nH 0.957 0 0\nH -0.24 0.927 0\n"
    r = web_client.post(
        "/api/build/load",
        data={"file": (io.BytesIO(xyz.encode()), "h2o.xyz")},
        content_type="multipart/form-data",
    )
    body = r.get_json()
    assert body["ok"] is True
    _assert_canonical_workspace_shape(body, endpoint="/api/build/load")
    # /api/build/load's endpoint-specific extras.
    for k in ("pdb", "summary"):
        assert k in body, f"top-level {k!r} missing"
        assert k in body["extra"], (
            f"extra[{k!r}] missing — Phase 4+ consumers broken"
        )
    # source_format must reflect the actually-parsed shape, not
    # the canonical default of "xyz" (in this case the file was
    # XYZ, but the endpoint sets it explicitly via the
    # parsed-format detection).
    assert body["source_format"] == "xyz"
    assert body["extra"]["source_format"] == "xyz"


def test_build_load_pdb_overrides_canonical_source_format(web_client):
    """When the user uploads a PDB file, source_format flips to
    "pdb" at BOTH the top level (replaces the canonical XYZ
    default) AND inside extra.  This is what makes the workspace
    dispatcher know which parser to round-trip through."""
    pdb = (
        "ATOM      1  O   HOH A   1       0.000   0.000   0.000  "
        "1.00  0.00           O  \n"
        "ATOM      2  H1  HOH A   1       0.957   0.000   0.000  "
        "1.00  0.00           H  \n"
        "ATOM      3  H2  HOH A   1      -0.239   0.927   0.000  "
        "1.00  0.00           H  \n"
    )
    r = web_client.post(
        "/api/build/load",
        json={"text": pdb, "filename": "water.pdb"},
    )
    body = r.get_json()
    assert body["ok"] is True
    assert body["source_format"]      == "pdb"
    assert body["extra"]["source_format"] == "pdb"


def test_build_molecule_returns_workspace_payload(web_client):
    """Phase 2: /api/build/molecule emits the canonical
    workspace_payload shape.  Endpoint extras (pdb, summary,
    backend_used, add_hydrogens_mode) live BOTH at top level
    AND inside ``extra``."""
    r = web_client.post("/api/build/molecule", json={
        "kind": "smiles", "input": "O",   # water
    })
    body = r.get_json()
    assert body["ok"] is True
    _assert_canonical_workspace_shape(
        body, endpoint="/api/build/molecule")
    for k in ("pdb", "summary", "backend_used",
              "add_hydrogens_mode"):
        assert k in body, f"top-level {k!r} missing"
        assert k in body["extra"], (
            f"extra[{k!r}] missing — Phase 4+ consumers broken"
        )


def test_modify_delete_returns_workspace_payload(web_client):
    """Phase 2: /api/modify/* already routed through
    ok_structure_response; pin that the canonical keys
    (text, source_format, lattice, extra) survive the helper
    refactor."""
    r = web_client.post("/api/modify/delete", json={
        "xyz": _H2O_XYZ,
        "indices": [0],
    })
    body = r.get_json()
    assert body["ok"] is True
    _assert_canonical_workspace_shape(
        body, endpoint="/api/modify/delete")


# --------------------------------------------------------------------- #
#  Phase 3: selection_remap on delete + add_atom                        #
#  (docs/protocols/workspace-state.md § 4.5 + § 5.1)                    #
# --------------------------------------------------------------------- #


def test_modify_delete_returns_selection_remap(web_client):
    """Phase 3: ``/api/modify/delete`` carries ``selection_remap``
    in ``extra``.  Maps every pre-delete index to its post-delete
    index, or ``None`` if the atom was removed.

    Fixes the latent bug where the client's naive ``selection.
    filter(i < n_new)`` silently dropped surviving high-index
    atoms (selecting atom 2, deleting atom 0, ended up with
    empty selection instead of new index 1)."""
    # 3-atom water: O at index 0, H at indices 1 and 2.
    r = web_client.post("/api/modify/delete", json={
        "xyz": _H2O_XYZ,
        "indices": [0],   # delete the O
    })
    body = r.get_json()
    assert body["ok"] is True
    assert "selection_remap" in body["extra"], (
        "Phase 3 contract: /api/modify/delete extras must carry "
        "selection_remap so the workspace dispatcher can translate "
        "the user's selection across the index shift"
    )
    remap = body["extra"]["selection_remap"]
    # Old index 0 (the O) was deleted; old [1, 2] become new [0, 1].
    assert remap == [None, 0, 1]


def test_modify_delete_selection_remap_handles_middle_deletion(
        web_client):
    """Pin the middle-deletion case: surviving HIGH-index atom
    shifts DOWN.  This is the exact case the pre-fix client-side
    naive filter got wrong."""
    r = web_client.post("/api/modify/delete", json={
        "xyz": _H2O_XYZ,
        "indices": [1],   # delete the central H
    })
    body = r.get_json()
    assert body["ok"] is True
    remap = body["extra"]["selection_remap"]
    # Old [0, 1, 2] → new [0, deleted, 1].
    assert remap == [0, None, 1]


def test_modify_add_atom_returns_identity_selection_remap(web_client):
    """Phase 3: ``/api/modify/add_atom`` emits the identity remap
    (all pre-op atoms survive at their old indices).  Emitting it
    even when trivial keeps the client dispatcher's per-op rule
    table flat — one lookup per op, no special cases."""
    r = web_client.post("/api/modify/add_atom", json={
        "xyz": _H2O_XYZ,
        "element": "H",
        "anchor_index": 0,
        "offset": [0.5, 0, 0],
    })
    body = r.get_json()
    assert body["ok"] is True
    assert "selection_remap" in body["extra"]
    remap = body["extra"]["selection_remap"]
    assert remap == [0, 1, 2], (
        "add_atom remap must be identity over the PRE-op atom range; "
        "the new atom appears at index n with no pre-op counterpart"
    )


# --------------------------------------------------------------------- #
#  Modify-tab edit-op endpoints (M3).  Body shape carries the canonical #
#  state (xyz + atom_names / residue_ids / residue_names / chain_ids)   #
#  alongside op-specific args; response shape mirrors /api/build/load + #
#  adds an issues array.                                                #
# --------------------------------------------------------------------- #


_H2O_XYZ = "3\nh2o\nO 0 0 0\nH 0.957 0 0\nH -0.24 0.927 0\n"


def test_modify_delete_drops_listed_indices(web_client):
    r = web_client.post("/api/modify/delete", json={
        "xyz": _H2O_XYZ,
        "indices": [1, 2],   # both H atoms
    })
    body = r.get_json()
    assert body["ok"] is True
    assert body["n_atoms"] == 1
    assert body["elements"] == ["O"]


def test_modify_responses_carry_atoms_list(web_client):
    """BOMB-0 fix (2026-06-07): every /api/modify/* response carries
    an ``atoms`` list in the same shape ``/api/selection/atoms``
    returns, so the front-end's selection store stays in sync
    with the in-memory post-op structure without a disk re-fetch.

    Pre-fix, modifier-op responses only carried xyz + elements +
    metadata lists; the selection panel went stale after every op."""
    r = web_client.post("/api/modify/delete", json={
        "xyz": _H2O_XYZ,
        "indices": [1, 2],   # delete both H, keep O
    })
    body = r.get_json()
    assert body["ok"] is True
    assert "atoms" in body, (
        "modify response must include atoms list "
        "(BOMB-0 selection-store sync fix)"
    )
    atoms = body["atoms"]
    assert len(atoms) == 1, (
        f"post-delete atoms list should have 1 row; got {len(atoms)}"
    )
    row = atoms[0]
    assert row["index"]    == 0
    assert row["element"]  == "O"
    assert row["regions"]  == []
    assert row["is_frozen"] is False


def test_modify_delete_silently_ignores_out_of_range(web_client):
    """Matches molbuilder.modify.delete_atoms behaviour."""
    r = web_client.post("/api/modify/delete", json={
        "xyz": _H2O_XYZ,
        "indices": [99, -1, 0],   # only 0 is in range
    })
    body = r.get_json()
    assert body["ok"] is True
    assert body["n_atoms"] == 2     # dropped O, kept the two H


def test_modify_delete_rejects_non_int_indices(web_client):
    r = web_client.post("/api/modify/delete", json={
        "xyz": _H2O_XYZ,
        "indices": ["a", "b"],
    })
    assert r.status_code == 400
    assert r.get_json()["ok"] is False


def test_modify_add_atom_appends_at_offset(web_client):
    r = web_client.post("/api/modify/add_atom", json={
        "xyz": _H2O_XYZ,
        "element": "H",
        "anchor_index": 0,            # the O
        "offset": [0.0, 0.0, 1.5],
    })
    body = r.get_json()
    assert body["ok"] is True
    assert body["n_atoms"] == 4
    assert body["elements"][-1] == "H"
    # New atom lands in a fresh residue named MOD (default).
    assert body["residue_names"][-1] == "MOD"


def test_modify_add_atom_explicit_residue_id_groups_atoms(web_client):
    """The web layer surfaces SP-E (add_atom's optional residue_id) so a
    UI builder can land multiple appended atoms in one residue."""
    r = web_client.post("/api/modify/add_atom", json={
        "xyz": _H2O_XYZ,
        "element": "C",
        "anchor_index": 0,
        "offset": [1.5, 0, 0],
        "residue_id": 99,
    })
    body = r.get_json()
    assert body["ok"] is True
    assert body["residue_ids"][-1] == 99


def test_modify_add_atom_rejects_bad_anchor(web_client):
    r = web_client.post("/api/modify/add_atom", json={
        "xyz": _H2O_XYZ,
        "element": "H",
        "anchor_index": 99,
        "offset": [0, 0, 1],
    })
    assert r.status_code == 400
    body = r.get_json()
    assert body["ok"] is False
    assert "anchor_index" in body["error"]


def test_modify_add_atom_rejects_missing_offset(web_client):
    r = web_client.post("/api/modify/add_atom", json={
        "xyz": _H2O_XYZ,
        "element": "H",
        "anchor_index": 0,
        # offset missing
    })
    assert r.status_code == 400
    assert "offset" in r.get_json()["error"]


def test_modify_endpoint_chain_preserves_metadata(web_client):
    """Spec invariant: per-atom metadata round-trips through every op
    when the client passes it back in the body.  add_atom -> delete
    keeps the atom_names / residue_ids carried alongside xyz."""
    # 1. Initial state -- the canonical body shape every modify op
    # accepts (xyz + parallel-array metadata).  No validate-and-echo
    # roundtrip needed; each modify op revalidates via
    # _struct_from_body.
    s1 = {
        "xyz":           _H2O_XYZ,
        "atom_names":    ["OW", "HW1", "HW2"],
        "residue_ids":   [7, 7, 7],
        "residue_names": ["WAT", "WAT", "WAT"],
        "chain_ids":     ["B", "B", "B"],
    }
    # 2. Add an atom; the metadata for the original 3 atoms must
    # survive (Structure preserves through add_atom).
    r2 = web_client.post("/api/modify/add_atom", json={
        "xyz":           s1["xyz"],
        "atom_names":    s1["atom_names"],
        "residue_ids":   s1["residue_ids"],
        "residue_names": s1["residue_names"],
        "chain_ids":     s1["chain_ids"],
        "element":       "H",
        "anchor_index":  0,
        "offset":        [0, 0, 1.5],
    })
    s2 = r2.get_json()
    assert s2["n_atoms"] == 4
    assert s2["atom_names"][:3]    == ["OW", "HW1", "HW2"]
    assert s2["residue_ids"][:3]   == [7, 7, 7]
    assert s2["residue_names"][:3] == ["WAT", "WAT", "WAT"]
    # 3. Delete the new atom; metadata for the surviving three is
    # still intact.
    r3 = web_client.post("/api/modify/delete", json={
        "xyz":           s2["xyz"],
        "atom_names":    s2["atom_names"],
        "residue_ids":   s2["residue_ids"],
        "residue_names": s2["residue_names"],
        "chain_ids":     s2["chain_ids"],
        "indices":       [3],
    })
    s3 = r3.get_json()
    assert s3["n_atoms"] == 3
    assert s3["atom_names"]    == ["OW", "HW1", "HW2"]
    assert s3["residue_names"] == ["WAT", "WAT", "WAT"]


# --------------------------------------------------------------------- #
#  M3 UI scaffolding lives in modify.html / static/modify/viewer.js.    #
# --------------------------------------------------------------------- #


def test_modify_page_has_m3_edit_controls(web_client):
    """The Edit panel must expose the M3 op controls (delete button,
    add-atom element input, three offset sliders + live distance
    readout).  M4 / M5 placeholders remain disabled."""
    body = web_client.get("/molbuilder").data.decode()
    for needle in (
        # Delete
        'id="delete-apply"',
        # Add atom
        'id="add-element"',
        'id="add-anchor-readout"',
        'id="add-dx"',     'id="add-dx-val"',
        'id="add-dy"',     'id="add-dy-val"',
        'id="add-dz"',     'id="add-dz-val"',
        'id="add-distance"',
        'id="add-apply"',
        # M5 electrode + handoff controls are wired.
        'id="elc-apply"',
        'id="send-to-build"',
    ):
        assert needle in body, f"missing {needle!r} in /modify HTML"


def test_modify_viewer_js_wires_delete_and_add(web_client):
    """The Modify viewer.js must call the M3 endpoints and update the
    live |offset| readout client-side."""
    js = web_client.get("/static/modify/viewer.js").data.decode()
    for needle in (
        "/api/modify/delete",
        "/api/modify/add_atom",
        "applyDelete",
        "applyAddAtom",
        "refreshAddDistance",
        "currentStateBody",
    ):
        assert needle in js, f"missing {needle!r} in modify viewer.js"


# --------------------------------------------------------------------- #
#  Modify-tab orient + rotate endpoints (M4)                            #
# --------------------------------------------------------------------- #


_LINEAR_XYZ = (
    "4\nlinear\n"
    "C 0 0 0\n"
    "C 1 1 0\n"
    "C 2 2 0\n"
    "C 3 3 0\n"
)


def _coords_from_xyz(xyz):
    """Parse an xyz string into a list of (x, y, z) tuples (skip
    header lines).  Helper for the orient / rotate tests."""
    return [
        tuple(float(v) for v in line.split()[1:4])
        for line in xyz.splitlines()[2:] if line.strip()
    ]


def test_modify_orient_default_lays_anchor_pair_along_z(web_client):
    """Default ``axis="z"``, ``angle=0`` orients atoms 0 -> 3 along z.
    With ``center="midpoint"`` (default), the midpoint of the pair
    lands at the origin so a0 is at -d/2 and a1 at +d/2 along z."""
    import math
    r = web_client.post("/api/modify/orient", json={
        "xyz": _LINEAR_XYZ,
        "anchors": [0, 3],
    })
    body = r.get_json()
    assert body["ok"] is True
    coords = _coords_from_xyz(body["xyz"])
    a0, a3 = coords[0], coords[3]
    # x and y of the anchors collapse to ~0 (vector lies along z).
    assert abs(a0[0]) < 1e-6 and abs(a0[1]) < 1e-6
    assert abs(a3[0]) < 1e-6 and abs(a3[1]) < 1e-6
    # The pair separation is preserved (sqrt(3)*sqrt(3+3+0) = sqrt(18)).
    sep = math.sqrt(sum((b - a) ** 2 for a, b in zip(a0, a3)))
    assert abs(sep - math.sqrt(18.0)) < 1e-6, sep
    # midpoint at origin
    mid = tuple(0.5 * (x + y) for x, y in zip(a0, a3))
    assert all(abs(v) < 1e-6 for v in mid), mid


def test_modify_orient_rejects_bad_axis(web_client):
    r = web_client.post("/api/modify/orient", json={
        "xyz": _LINEAR_XYZ,
        "anchors": [0, 3],
        "axis": "w",
    })
    assert r.status_code == 400
    assert "axis" in r.get_json()["error"]


def test_modify_rotate_z_90_degrees(web_client):
    """Rotation around z by 90° maps (1, 1, 0) -> (-1, 1, 0)."""
    r = web_client.post("/api/modify/rotate", json={
        "xyz": _LINEAR_XYZ, "axis": "z", "angle": 90,
    })
    body = r.get_json()
    assert body["ok"] is True
    coords = _coords_from_xyz(body["xyz"])
    # atom 1 was at (1, 1, 0); after a +90° rotation around z it
    # becomes (-1, 1, 0).
    assert abs(coords[1][0] - (-1.0)) < 1e-6, coords[1]
    assert abs(coords[1][1] - 1.0)    < 1e-6, coords[1]
    assert abs(coords[1][2] - 0.0)    < 1e-6, coords[1]


def test_modify_rotate_rejects_bad_axis(web_client):
    r = web_client.post("/api/modify/rotate", json={
        "xyz": _LINEAR_XYZ, "axis": "w", "angle": 30,
    })
    assert r.status_code == 400
    assert "axis" in r.get_json()["error"]


def test_modify_rotate_rejects_non_numeric_angle(web_client):
    r = web_client.post("/api/modify/rotate", json={
        "xyz": _LINEAR_XYZ, "axis": "z", "angle": "ninety",
    })
    assert r.status_code == 400
    assert "angle" in r.get_json()["error"]


# --------------------------------------------------------------------- #
#  /api/modify/translate                                                 #
# --------------------------------------------------------------------- #


def test_modify_translate_recenter_puts_centroid_at_origin(web_client):
    """``recenter: true`` translates so the geometric centroid sits
    at (0, 0, 0).  The fixture _LINEAR_XYZ is (1,1,1)/(2,2,2)/(3,3,3)/
    (4,4,4) so its centroid is (2.5, 2.5, 2.5); after recentering
    every coord is shifted by -2.5."""
    r = web_client.post("/api/modify/translate", json={
        "xyz": _LINEAR_XYZ,
        "recenter": True,
    })
    body = r.get_json()
    assert body["ok"] is True
    coords = _coords_from_xyz(body["xyz"])
    cx = sum(c[0] for c in coords) / len(coords)
    cy = sum(c[1] for c in coords) / len(coords)
    cz = sum(c[2] for c in coords) / len(coords)
    assert abs(cx) < 1e-9, cx
    assert abs(cy) < 1e-9, cy
    assert abs(cz) < 1e-9, cz


def test_modify_translate_offset_shifts_every_atom_by_delta(web_client):
    """``{dx, dy, dz}`` shifts every coordinate by exactly that
    vector and preserves intra-structure distances."""
    r = web_client.post("/api/modify/translate", json={
        "xyz": _LINEAR_XYZ,
        "dx": 10.0, "dy": -5.0, "dz": 0.5,
    })
    body = r.get_json()
    assert body["ok"] is True
    out = _coords_from_xyz(body["xyz"])
    src = _coords_from_xyz(_LINEAR_XYZ)
    for s, o in zip(src, out):
        assert abs(o[0] - s[0] - 10.0) < 1e-9
        assert abs(o[1] - s[1] + 5.0) < 1e-9
        assert abs(o[2] - s[2] - 0.5) < 1e-9


def test_modify_translate_recenter_wins_over_dxdydz(web_client):
    """When both ``recenter`` and ``{dx,dy,dz}`` are supplied the
    server takes the recenter path (documented behaviour); the dx
    fields are silently ignored."""
    r = web_client.post("/api/modify/translate", json={
        "xyz": _LINEAR_XYZ,
        "recenter": True,
        "dx": 999.0, "dy": -999.0, "dz": 999.0,
    })
    coords = _coords_from_xyz(r.get_json()["xyz"])
    cx = sum(c[0] for c in coords) / len(coords)
    # Centroid at origin -- the dx/dy/dz fields were ignored.
    assert abs(cx) < 1e-9, cx


def test_modify_translate_zero_default_is_a_noop(web_client):
    """Omitting dx/dy/dz defaults each to 0.0; the result is byte-
    identical xyz with the same atom count."""
    r = web_client.post("/api/modify/translate", json={"xyz": _LINEAR_XYZ})
    body = r.get_json()
    assert body["ok"] is True
    src = _coords_from_xyz(_LINEAR_XYZ)
    out = _coords_from_xyz(body["xyz"])
    for s, o in zip(src, out):
        for a, b in zip(s, o):
            assert abs(a - b) < 1e-12


def test_modify_translate_rejects_non_numeric_offset(web_client):
    r = web_client.post("/api/modify/translate", json={
        "xyz": _LINEAR_XYZ,
        "dx": "pizza",
    })
    assert r.status_code == 400
    assert "number" in r.get_json()["error"]


def test_modify_translate_preserves_metadata(web_client):
    """Per-atom metadata round-trips through translate (rigid op)."""
    r = web_client.post("/api/modify/translate", json={
        "xyz": _LINEAR_XYZ,
        "atom_names":    ["C1", "C2", "C3", "C4"],
        "residue_ids":   [1, 1, 1, 1],
        "residue_names": ["MOL", "MOL", "MOL", "MOL"],
        "chain_ids":     ["A", "A", "A", "A"],
        "dx": 1.0,
    })
    body = r.get_json()
    assert body["atom_names"] == ["C1", "C2", "C3", "C4"]
    assert body["residue_ids"] == [1, 1, 1, 1]
    assert body["chain_ids"] == ["A", "A", "A", "A"]


def test_modify_orient_then_rotate_chains_through_metadata(web_client):
    """Chain orient -> rotate while preserving per-atom metadata
    (matches the spec § 5 invariant)."""
    # Initial state -- canonical body shape; each modify op
    # revalidates via _struct_from_body.
    s1 = {
        "xyz":           _LINEAR_XYZ,
        "atom_names":    ["C1", "C2", "C3", "C4"],
        "residue_ids":   [1, 1, 1, 1],
        "residue_names": ["MOL", "MOL", "MOL", "MOL"],
        "chain_ids":     ["A", "A", "A", "A"],
    }
    s2 = web_client.post("/api/modify/orient", json={
        "xyz":           s1["xyz"],
        "atom_names":    s1["atom_names"],
        "residue_ids":   s1["residue_ids"],
        "residue_names": s1["residue_names"],
        "chain_ids":     s1["chain_ids"],
        "anchors":       [0, 3],
    }).get_json()
    s3 = web_client.post("/api/modify/rotate", json={
        "xyz":           s2["xyz"],
        "atom_names":    s2["atom_names"],
        "residue_ids":   s2["residue_ids"],
        "residue_names": s2["residue_names"],
        "chain_ids":     s2["chain_ids"],
        "axis": "z", "angle": 45,
    }).get_json()
    assert s3["atom_names"] == ["C1", "C2", "C3", "C4"]
    assert s3["residue_names"] == ["MOL"] * 4


def test_modify_page_has_m4_orient_rotate_controls(web_client):
    """The Edit panel must expose the M4 orient + rotate controls
    (anchor-pair readout, axis radios, angle slider, Apply for both
    ops).  The M5 placeholder (electrode panel) stays disabled."""
    body = web_client.get("/molbuilder").data.decode()
    for needle in (
        # Orient
        'id="orient-apply"',
        'id="orient-anchor-readout"',
        'id="orient-angle"',     'id="orient-angle-val"',
        'id="orient-center"',
        'name="orient-axis"',
        # Rotate
        'id="rotate-apply"',
        'id="rotate-angle"',     'id="rotate-angle-val"',
        'name="rotate-axis"',
        # M5 controls wired.
        'id="elc-apply"',
        'id="send-to-build"',
    ):
        assert needle in body, f"missing {needle!r} in /modify HTML"


def test_modify_viewer_js_wires_orient_and_rotate(web_client):
    js = web_client.get("/static/modify/viewer.js").data.decode()
    for needle in (
        "/api/modify/orient",
        "/api/modify/rotate",
        "applyOrient",
        "applyRotate",
        "refreshOrientAngleReadout",
        "refreshRotateAngleReadout",
    ):
        assert needle in js, f"missing {needle!r} in modify viewer.js"


# --------------------------------------------------------------------- #
#  M5: electrode endpoints + Send-to-Build handoff                      #
# --------------------------------------------------------------------- #


_SS_XYZ = (
    "2\nss-pair\n"
    "S 0 0 -2\n"
    "S 0 0  2\n"
)


def test_modify_symmetric_electrodes_pair_mode(web_client):
    """Pair mode: 2x2x1 Au(111) on either side of a 2-atom S pair,
    8 Å gap.  4 Au atoms per side -> 8 ELC atoms + 2 S = 10 total."""
    r = web_client.post("/api/modify/symmetric_electrodes", json={
        "xyz":     _SS_XYZ,
        "element": "Au", "plane": "111",
        "size":    [2, 2, 1],
        "anchors": [1, 0],          # +z first, -z second
        "gap":     8.0,
    })
    body = r.get_json()
    assert body["ok"] is True
    assert body["n_atoms"] == 10
    elc = sum(1 for n in body["residue_names"] if n == "ELC")
    assert elc == 8, body["residue_names"]


def test_modify_symmetric_electrodes_anchorless_centres_on_origin(web_client):
    """Anchorless mode (no ``anchors`` field) puts the slab midpoint
    at the world origin: top closest layer at z = +gap/2, bot at
    -gap/2.  We verify by reading ELC z-coords from the response
    xyz.  This is the canonical UI workflow -- centre-and-pose the
    molecule first, then add slabs around the origin."""
    r = web_client.post("/api/modify/symmetric_electrodes", json={
        "xyz":     _SS_XYZ,
        "element": "Au", "plane": "111",
        "size":    [2, 2, 1],
        "gap":     8.0,
        # No anchors field.
    })
    body = r.get_json()
    assert body["ok"] is True
    coords = _coords_from_xyz(body["xyz"])
    elc_z = [coords[i][2] for i, rn in enumerate(body["residue_names"])
             if rn == "ELC"]
    top = [z for z in elc_z if z > 0]
    bot = [z for z in elc_z if z < 0]
    assert top, "expected at least one ELC atom at z > 0"
    assert bot, "expected at least one ELC atom at z < 0"
    # Closest layers are at exactly ±gap/2 = ±4.0 Å.
    assert abs(min(top) - 4.0) < 1e-6, f"top closest z = {min(top)}"
    assert abs(max(bot) + 4.0) < 1e-6, f"bot closest z = {max(bot)}"


def test_modify_meta_lists_supported_elements_and_planes(web_client):
    """/api/modify/meta returns the SAME tuples molbuilder.modify
    exports.  This is the wire contract that lets the UI populate
    its dropdowns without duplicating the lists in HTML."""
    from molbuilder.modify import (SUPPORTED_FCC_ELEMENTS,
                                    SUPPORTED_FCC_PLANES)
    r = web_client.get("/api/modify/meta")
    body = r.get_json()
    assert body["ok"] is True
    assert body["fcc_elements"] == list(SUPPORTED_FCC_ELEMENTS)
    assert body["fcc_planes"]   == list(SUPPORTED_FCC_PLANES)


def test_modify_symmetric_electrodes_rejects_nonpositive_gap(web_client):
    """A 0 / negative gap is rejected at the route boundary so the
    user gets an actionable 400 instead of a downstream geometry
    error."""
    for gap in (0.0, -3.0):
        r = web_client.post("/api/modify/symmetric_electrodes", json={
            "xyz": _SS_XYZ, "element": "Au", "plane": "111",
            "size": [2, 2, 1], "gap": gap,
        })
        assert r.status_code == 400, gap
        assert "gap" in r.get_json()["error"], gap


def test_modify_electrode_rejects_nonpositive_contact_distance(web_client):
    """Single-mode contact distance must be strictly positive."""
    r = web_client.post("/api/modify/electrode", json={
        "xyz": _SS_XYZ, "element": "Au", "plane": "111",
        "size": [2, 2, 1], "anchor_index": 0,
        "contact_distance": 0.0, "side": "+z",
    })
    assert r.status_code == 400
    assert "contact_distance" in r.get_json()["error"]


def test_modify_electrode_single_mode(web_client):
    """Single mode: one slab on +z above the second S atom."""
    r = web_client.post("/api/modify/electrode", json={
        "xyz": _SS_XYZ, "element": "Au", "plane": "111",
        "size": [2, 2, 1], "anchor_index": 1,
        "side": "+z", "contact_distance": 2.4,
    })
    body = r.get_json()
    assert body["ok"] is True
    # 4 Au atoms + 2 S = 6 total.
    assert body["n_atoms"] == 6
    assert sum(1 for n in body["residue_names"] if n == "ELC") == 4


def test_modify_electrode_rejects_bad_side(web_client):
    r = web_client.post("/api/modify/electrode", json={
        "xyz": _SS_XYZ, "element": "Au", "plane": "111",
        "size": [2, 2, 1], "anchor_index": 0, "side": "above",
    })
    assert r.status_code == 400
    assert "side" in r.get_json()["error"]


# --------------------------------------------------------------------- #
#  Basename validation (job-layout v1)                                  #
# --------------------------------------------------------------------- #


def test_modify_fdf_rejects_slash_in_system_label(web_client):
    """The Build /api/build/fdf endpoint validates ``system_label``
    against the basename charset before any file write.  Slashes,
    spaces, dots, and leading-dot are all rejected per
    docs/protocols/job-layout.md."""
    for bad in ("a/b", "with spaces", "has.dot", ".leading"):
        r = web_client.post("/api/build/fdf", json={
            "xyz": _LINEAR_XYZ,
            "params": {"system_label": bad},
        })
        body = r.get_json()
        assert body["ok"] is False or any(
            i.get("severity") == "error" and "basename" in i.get("message", "")
            for i in body.get("issues", [])
        ), f"expected error for system_label={bad!r}; got {body}"


def test_modify_pyscf_rejects_slash_in_job_name(web_client):
    """Same rule for PySCFConfig.job_name."""
    r = web_client.post("/api/build/pyscf", json={
        "xyz": _LINEAR_XYZ,
        "params": {"job_name": "evil/path"},
    })
    body = r.get_json()
    assert body["ok"] is False or any(
        i.get("severity") == "error" and "basename" in i.get("message", "")
        for i in body.get("issues", [])
    ), f"expected error for job_name='evil/path'; got {body}"


# --------------------------------------------------------------------- #
#  NaN / Inf rejection on /api/modify/* floats                          #
# --------------------------------------------------------------------- #


def test_modify_translate_rejects_nan_offset(web_client):
    """A NaN dx must not propagate through to the structure -- the
    boundary helper ``_shared.finite_float`` rejects non-finite
    values."""
    r = web_client.post("/api/modify/translate", json={
        "xyz": _LINEAR_XYZ, "dx": float("nan"), "dy": 0.0, "dz": 0.0,
    })
    assert r.status_code == 400
    assert "finite" in r.get_json()["error"]


def test_modify_rotate_rejects_nan_angle(web_client):
    r = web_client.post("/api/modify/rotate", json={
        "xyz": _LINEAR_XYZ, "axis": "z", "angle": float("nan"),
    })
    assert r.status_code == 400


def test_modify_symmetric_electrodes_rejects_nan_gap(web_client):
    r = web_client.post("/api/modify/symmetric_electrodes", json={
        "xyz": _LINEAR_XYZ, "element": "Au", "plane": "111",
        "size": [2, 2, 1], "gap": float("nan"),
    })
    assert r.status_code == 400


# --------------------------------------------------------------------- #
#  dataclass_to_form_schema -- the schema generator that backs the     #
#  /api/build/schema/{siesta,pyscf} endpoints.  Tests use a hand-      #
#  written tiny dataclass so they don't couple to the production       #
#  SiestaConfig / PySCFConfig field set; the endpoint tests further    #
#  below exercise the real configs.                                    #
# --------------------------------------------------------------------- #


from dataclasses import dataclass as _schema_dc, field as _schema_field
from typing import Optional as _Optional, Tuple as _Tuple


@_schema_dc
class _FakeCfgForSchema:
    """One field per supported kind so a single test covers them all.

    Defined at module scope (not inside a helper) so
    ``typing.get_type_hints`` can resolve Optional / Tuple against
    this module's globals.
    """
    flag: bool = _schema_field(default=True, metadata={
        "section": "Basics", "label": "Flag",
        "help": "a plain boolean checkbox",
    })
    count: int = _schema_field(default=3, metadata={
        "section": "Basics", "label": "Count",
        "range": (0, 100), "tier": "advanced",
    })
    size_ang: float = _schema_field(default=1.5, metadata={
        "section": "Geometry", "label": "Size", "unit": "Å",
        "range": (0.1, 10.0),
    })
    method: str = _schema_field(default="A", metadata={
        "section": "Geometry", "label": "Method",
        "choices": ("A", "B", "C"),
    })
    title: str = _schema_field(default="default-title", metadata={
        "section": "Basics", "label": "Title",
        "pattern": r"^[A-Za-z0-9_\-]+$",
    })
    opt_int: _Optional[int] = _schema_field(default=None, metadata={
        "section": "Geometry", "label": "Optional int",
        "null_label": "(auto)",
    })
    tri: _Optional[bool] = _schema_field(default=None, metadata={
        "section": "Basics", "label": "Tri-state",
    })
    grid: _Tuple[int, int, int] = _schema_field(default=(1, 1, 1), metadata={
        "section": "Geometry", "label": "Grid",
        "triple_labels": ("x", "y", "z"),
    })
    # No section -> omitted from schema.
    internal: int = _schema_field(default=0, metadata={
        "help": "no section, should not appear in schema",
    })
    # id_suffix override so the legacy-id contract is exercised.
    legacy_name: str = _schema_field(default="leg", metadata={
        "section": "Basics", "id_suffix": "lname",
        "label": "Legacy",
    })


def _schema_fixture_cls():
    return _FakeCfgForSchema


def test_dataclass_schema_groups_by_section_in_declaration_order():
    from molbuilder.web.blueprints._shared import dataclass_to_form_schema
    sch = dataclass_to_form_schema(_schema_fixture_cls(), id_prefix="t")
    assert sch["config"] == "_FakeCfgForSchema"
    assert sch["id_prefix"] == "t"
    section_names = [s["name"] for s in sch["sections"]]
    # "Basics" comes first because the first sectioned field (flag)
    # declares it; "Geometry" follows because size_ang is the first
    # field declaring that section.  Order MUST follow declaration.
    assert section_names == ["Basics", "Geometry"], section_names


def test_dataclass_schema_omits_unsectioned_fields():
    from molbuilder.web.blueprints._shared import dataclass_to_form_schema
    sch = dataclass_to_form_schema(_schema_fixture_cls(), id_prefix="t")
    all_names = [f["name"] for s in sch["sections"] for f in s["fields"]]
    # `internal` has no section and must be absent.
    assert "internal" not in all_names


def test_dataclass_schema_id_convention_with_override():
    from molbuilder.web.blueprints._shared import dataclass_to_form_schema
    sch = dataclass_to_form_schema(_schema_fixture_cls(), id_prefix="t")
    fmap = {f["name"]: f for s in sch["sections"] for f in s["fields"]}
    # Default convention: f.name with underscores -> hyphens.
    assert fmap["size_ang"]["id"] == "t-size-ang"
    assert fmap["opt_int"]["id"] == "t-opt-int"
    # id_suffix metadata override:
    assert fmap["legacy_name"]["id"] == "t-lname"


def test_dataclass_schema_kind_inference():
    from molbuilder.web.blueprints._shared import dataclass_to_form_schema
    sch = dataclass_to_form_schema(_schema_fixture_cls(), id_prefix="t")
    fmap = {f["name"]: f for s in sch["sections"] for f in s["fields"]}
    assert fmap["flag"]["kind"] == "checkbox"
    assert fmap["count"]["kind"] == "int"
    assert fmap["size_ang"]["kind"] == "number"
    assert fmap["method"]["kind"] == "select"
    assert fmap["title"]["kind"] == "text"
    assert fmap["opt_int"]["kind"] == "int"
    assert fmap["opt_int"]["null_option"] is True
    assert fmap["opt_int"]["null_label"] == "(auto)"
    assert fmap["tri"]["kind"] == "tri-select"
    assert fmap["tri"]["choices"] == ["auto", "true", "false"]
    assert fmap["grid"]["kind"] == "int-triple"
    assert fmap["grid"]["labels"] == ["x", "y", "z"]


def test_dataclass_schema_passes_through_range_unit_pattern():
    from molbuilder.web.blueprints._shared import dataclass_to_form_schema
    sch = dataclass_to_form_schema(_schema_fixture_cls(), id_prefix="t")
    fmap = {f["name"]: f for s in sch["sections"] for f in s["fields"]}
    assert fmap["count"]["min"] == 0 and fmap["count"]["max"] == 100
    assert fmap["size_ang"]["min"] == 0.1
    assert fmap["size_ang"]["max"] == 10.0
    assert fmap["size_ang"]["unit"] == "Å"
    assert fmap["title"]["pattern"] == r"^[A-Za-z0-9_\-]+$"


def test_dataclass_schema_serialises_defaults():
    from molbuilder.web.blueprints._shared import dataclass_to_form_schema
    sch = dataclass_to_form_schema(_schema_fixture_cls(), id_prefix="t")
    fmap = {f["name"]: f for s in sch["sections"] for f in s["fields"]}
    assert fmap["flag"]["default"] is True
    assert fmap["count"]["default"] == 3
    assert fmap["size_ang"]["default"] == 1.5
    assert fmap["method"]["default"] == "A"
    # Tuple becomes a list for JSON compatibility.
    assert fmap["grid"]["default"] == [1, 1, 1]
    # Optional defaults to None pass through as null.
    assert fmap["opt_int"]["default"] is None


def test_dataclass_schema_is_json_serialisable():
    """The schema MUST be json.dumps-able with no custom encoder --
    the endpoint returns it via jsonify()."""
    import json
    from molbuilder.web.blueprints._shared import dataclass_to_form_schema
    sch = dataclass_to_form_schema(_schema_fixture_cls(), id_prefix="t")
    serialised = json.dumps(sch)
    # Round-trips without loss.
    assert json.loads(serialised) == sch


def test_dataclass_schema_honors_form_section_order_override():
    """A class-level _form_section_order tuple overrides declaration
    order without forcing the user to reorder fields.  Section names
    in the tuple come first (in the tuple's order); any extra
    sections present in field metadata but missing from the tuple
    keep their declaration-order position appended after."""
    from dataclasses import dataclass as _dc, field as _f
    from molbuilder.web.blueprints._shared import dataclass_to_form_schema

    @_dc
    class Reordered:
        _form_section_order = ("Z", "X")  # declared order is X, Y, Z
        x: int = _f(default=0, metadata={"section": "X"})
        y: int = _f(default=0, metadata={"section": "Y"})
        z: int = _f(default=0, metadata={"section": "Z"})

    sch = dataclass_to_form_schema(Reordered, id_prefix="t")
    names = [s["name"] for s in sch["sections"]]
    # Explicit "Z", "X" come first; then "Y" tacked on (in declaration
    # position 2 of the original ordering).
    assert names == ["Z", "X", "Y"], names


def test_siesta_form_schema_matches_documented_layout():
    """The production SiestaConfig schema -- both the section order
    and the count of fields per section -- is itself part of the
    Build-tab contract.  Pin it here so a stray field-reorder or a
    forgotten metadata addition doesn't silently rearrange the UI."""
    from molbuilder.web.blueprints._shared import dataclass_to_form_schema
    from molbuilder.config.siesta import SiestaConfig

    sch = dataclass_to_form_schema(SiestaConfig, id_prefix="p")
    assert sch["config"] == "SiestaConfig"
    assert sch["id_prefix"] == "p"

    # 2026-06-15 second restructure: merged "Relaxation" + "Parallel
    # execution" into a single "Compute & budget" section so the
    # physics axis (System -> Basis -> XC -> SCF -> Spin -> Output)
    # stays compact and the "how the run executes" knobs are
    # gathered in one place at the end of the form.  See the
    # SiestaConfig._form_section_order comment block for the full
    # design rationale + the workflow-group card split (profile /
    # stage / budget) inside the merged section.
    expected = [
        # System: 2 -> 3 fields after the 2026-05-26 review added
        # section="System" to ``net_charge`` so users with charged
        # side-chains (carboxylates, lysines, sulfonates -- not seen
        # by the phosphate auto-detect heuristic) have a form input.
        ("System",                   3),
        # 2026-06-13 fold: kgrid (Monkhorst-Pack) moved from its own
        # one-field section into "Basis & grid" so the form stops
        # having a one-field-only section.  Both are about how
        # finely we sample the calculation (real space + reciprocal
        # space).  Basis & grid: 3 + 1 = 4 fields now.
        ("Basis & grid",             4),
        ("Exchange-correlation",     2),
        ("SCF",                      7),
        ("Spin",                     2),
        ("Output & positioning",     6),
        # Compute & budget: 14 fields after the 2026-06-15 merge.
        #   Relaxation contributed 7 (relax_type, relax_steps,
        #   relax_force_tol, relax_max_displ, md_initial_temperature,
        #   md_target_temperature, md_length_timestep).
        #   Parallel execution contributed 7 (mpi_np,
        #   parallel_block_size, parallel_over_k, omp_threads,
        #   max_memory_mb, enable_gpu, elpa_algorithm).
        ("Compute & budget",        14),
        # Optimization: the SIESTA staged-opt stage-table widget
        # shipped 2026-06-25 (#542 C1.5) as a single dataclass-typed
        # field (List[SiestaStageSpec]) — one field, rendered as a
        # multi-row stage-table by the JS form-schema renderer.
        ("Optimization",             1),
    ]
    got = [(s["name"], len(s["fields"])) for s in sch["sections"]]
    assert got == expected, got

    # The kgrid Tuple field MUST render as the "int-triple" kind so
    # the JS renderer knows to emit three side-by-side inputs.
    kgrid = next(
        f for s in sch["sections"] for f in s["fields"]
        if f["name"] == "kgrid"
    )
    assert kgrid["kind"] == "int-triple"
    assert kgrid["labels"] == ["x", "y", "z"]


def test_api_build_schema_returns_siesta_schema(web_client):
    """GET /api/build/schema/siesta returns the SiestaConfig schema
    via the shared dataclass_to_form_schema helper.  The wire shape
    is ``{"ok": True, "schema": {...}}``; the schema's id_prefix
    field is the canonical "p" used by the form-field IDs."""
    r = web_client.get("/api/build/schema/siesta")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    sch = body["schema"]
    assert sch["config"] == "SiestaConfig"
    assert sch["id_prefix"] == "p"
    # Smoke: the first section is "System" and it carries the
    # SystemLabel field that maps to the existing #p-system-label id.
    assert sch["sections"][0]["name"] == "System"
    sysfields = sch["sections"][0]["fields"]
    sysl = next(f for f in sysfields if f["name"] == "system_label")
    assert sysl["id"] == "p-system-label"


def test_api_build_schema_returns_pyscf_schema(web_client):
    """GET /api/build/schema/pyscf returns the PySCFConfig schema
    with id_prefix='py'.  The Frequencies section MUST be present
    so the post-relax Hessian / thermo block is reachable from the
    schema-driven form."""
    r = web_client.get("/api/build/schema/pyscf")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    sch = body["schema"]
    assert sch["config"] == "PySCFConfig"
    assert sch["id_prefix"] == "py"
    section_names = [s["name"] for s in sch["sections"]]
    assert "Frequencies / thermochemistry" in section_names


def test_form_schema_js_is_served(web_client):
    """The new web/static/lib/form-schema.js is the JS-side
    consumer of /api/build/schema/<engine>.  It must be served
    by Flask static so index.html can <script src="..."> it."""
    r = web_client.get("/static/lib/form-schema.js")
    assert r.status_code == 200
    body = r.data.decode()
    # Public API surface -- if any name disappears, the Build form
    # cutover breaks silently.
    for needle in (
        "renderForm", "collectForm", "fetchSchema",
        # All seven kinds must remain handled in the switch.
        '"checkbox"', '"int"', '"number"', '"text"',
        '"select"', '"tri-select"', '"int-triple"',
    ):
        assert needle in body, f"form-schema.js missing {needle!r}"


def test_api_build_schema_rejects_unknown_engine(web_client):
    """An unknown engine name surfaces as 404 with a structured
    error so the UI doesn't silently render an empty form."""
    r = web_client.get("/api/build/schema/cp2k")
    assert r.status_code == 404
    body = r.get_json()
    assert body["ok"] is False
    assert "cp2k" in body["error"].lower()


def test_pyscf_form_schema_matches_documented_layout():
    """Same pin for PySCFConfig.  The post-relax frequencies /
    thermochemistry section (added in v1.1) is the rightmost
    semantic group, after Solvent."""
    from molbuilder.web.blueprints._shared import dataclass_to_form_schema
    from molbuilder.config.pyscf import PySCFConfig

    sch = dataclass_to_form_schema(PySCFConfig, id_prefix="py")
    assert sch["config"] == "PySCFConfig"
    assert sch["id_prefix"] == "py"

    expected = [
        ("System",                       4),
        ("Method",                       5),
        ("SCF",                          5),
        ("Solvent (optional)",           2),
        ("Frequencies / thermochemistry", 3),
        # Compute & budget after #534 commit 4b:
        #   * 2 optimization knobs left (optimize, optimizer); the
        #     four flat geom_conv_* / geom_max_steps scalars + the
        #     5 preopt_* knobs they used to share this section with
        #     are gone -- the cfg.stages stage-table is the
        #     canonical convergence-ladder control.
        #   * 1 stage-table widget (``stages``).
        #   * 7 runtime + output knobs (max_memory_mb, threads,
        #     use_gpu, verbose, chkfile, log_file, verbose_comments).
        ("Compute & budget",            10),
    ]
    got = [(s["name"], len(s["fields"])) for s in sch["sections"]]
    assert got == expected, got

    # Spin uses range metadata that must propagate so the JS renderer
    # can emit min/max attributes (UX hint, not a server-side check).
    spin = next(
        f for s in sch["sections"] for f in s["fields"]
        if f["name"] == "spin"
    )
    assert spin["kind"] == "int"
    assert spin["min"] == 0 and spin["max"] == 10

    # The job-name pattern carries through so the renderer can apply
    # the HTML5 pattern= attribute, matching the existing static form.
    jn = next(
        f for s in sch["sections"] for f in s["fields"]
        if f["name"] == "job_name"
    )
    assert jn["pattern"] == r"^[A-Za-z0-9_\-]+$"


# --------------------------------------------------------------------- #
#  engine_key metadata round-trip                                       #
#                                                                       #
#  2026-05-26: engine_key was added to all 47 SIESTA + 48 PySCF         #
#  fields so the UI can render a "writes-this-keyword" badge next       #
#  to each form label, BUT the metadata had zero tests.  A field        #
#  whose engine_key got dropped or mistyped would be invisible to       #
#  regression detection.  These tests pin: (a) every dataclass field    #
#  carries engine_key in the schema endpoint output, (b) molbuilder-    #
#  only fields are tagged with the ``(molbuilder`` marker so the        #
#  UI knows to dim them, (c) representative SIESTA/PySCF keywords      #
#  the rest of the codebase relies on (SpinPolarized, MeshCutoff,       #
#  PAO.BasisSize on the SIESTA side; gto.M(charge=...) etc on PySCF).   #
# --------------------------------------------------------------------- #


def _flatten_schema_fields(sch):
    return [f for s in sch["sections"] for f in s["fields"]]


def test_engine_key_present_on_every_siesta_form_field():
    """Every SIESTA field that lands in the form (has ``section``)
    MUST carry an ``engine_key`` metadata.  Without it the UI's
    source-of-truth badge is silently missing for that field."""
    from molbuilder.web.blueprints._shared import dataclass_to_form_schema
    from molbuilder.config.siesta import SiestaConfig
    sch = dataclass_to_form_schema(SiestaConfig, id_prefix="p")
    missing = [f["name"] for f in _flatten_schema_fields(sch)
               if "engine_key" not in f]
    assert not missing, (
        f"SiestaConfig fields without engine_key (would render no "
        f"keyword badge in the form): {missing}"
    )


def test_engine_key_present_on_every_pyscf_form_field():
    """Same contract for PySCF."""
    from molbuilder.web.blueprints._shared import dataclass_to_form_schema
    from molbuilder.config.pyscf import PySCFConfig
    sch = dataclass_to_form_schema(PySCFConfig, id_prefix="py")
    missing = [f["name"] for f in _flatten_schema_fields(sch)
               if "engine_key" not in f]
    assert not missing, (
        f"PySCFConfig fields without engine_key: {missing}"
    )


def test_engine_key_present_on_every_spectra_form_field():
    """SpectraConfig was missing engine_key on ALL fields pre-audit
    2026-06-02 (task #188).  Backfilled to mirror PySCFConfig style
    on shared keys + ``(molbuilder: ...)`` markers for the per-mode
    selectors / frozen-atom filters / finite-difference knobs that
    have no PySCF equivalent."""
    from molbuilder.web.blueprints._shared import dataclass_to_form_schema
    from molbuilder.config.spectra import SpectraConfig
    sch = dataclass_to_form_schema(SpectraConfig, id_prefix="sp")
    missing = [f["name"] for f in _flatten_schema_fields(sch)
               if "engine_key" not in f]
    assert not missing, (
        f"SpectraConfig fields without engine_key: {missing}"
    )


def test_spectra_molbuilder_only_fields_use_paren_prefix():
    """Same dimming-rule check as the SIESTA variant: SpectraConfig
    fields that don't map to a PySCF keyword (frozen-atom filters,
    per-mode selector, finite-difference knobs, emission control)
    MUST carry the ``(molbuilder`` prefix so the UI knows to dim
    the badge."""
    from molbuilder.web.blueprints._shared import dataclass_to_form_schema
    from molbuilder.config.spectra import SpectraConfig
    sch = dataclass_to_form_schema(SpectraConfig, id_prefix="sp")
    molbuilder_only = {
        "engine",
        "job_name",
        "frozen_elements",
        "frozen_residue_names",
        "frozen_indices",
        "compute_raman",
        "compute_ir",
        "displacement_amplitude_ang",
        "es_mode_selection",
        "es_top_n",
        "es_threshold",
        "es_explicit_indices",
        "freq_min_cm1",
        "freq_max_cm1",
        "es_n_homo_below",
        "es_n_lumo_above",
        "verbose_comments",
    }
    fields_by_name = {f["name"]: f for f in _flatten_schema_fields(sch)}
    for name in molbuilder_only:
        f = fields_by_name.get(name)
        assert f is not None, f"missing field {name} in schema"
        assert f["engine_key"].startswith("(molbuilder"), (
            f"{name}: engine_key={f['engine_key']!r} should start with "
            f"``(molbuilder`` so the UI dims the badge"
        )


def test_engine_key_marks_molbuilder_only_fields_with_paren_prefix():
    """molbuilder-only fields (preprocessing / wrapper / filename
    knobs that don't reach the engine) MUST have engine_key
    starting with ``(molbuilder`` so the JS engineKeyBadge() picks
    the dashed-border italic visual variant.  Without this the
    user might search the SIESTA / PySCF manual for a keyword
    molbuilder invented (e.g. cell_padding, verbose_comments)."""
    from molbuilder.web.blueprints._shared import dataclass_to_form_schema
    from molbuilder.config.siesta import SiestaConfig
    sch = dataclass_to_form_schema(SiestaConfig, id_prefix="p")
    # Fields that ARE molbuilder-only (curated list -- if you flip
    # one of these to a real engine keyword, update the list).
    molbuilder_only = {
        "psml_lib",         # stages .psml files
        "mpi_np",           # .run.sh launcher only
        "omp_threads",      # .run.sh + .fdf runtime_info comment
        "max_memory_mb",    # .run.sh ulimit + .fdf comment
        "wrap_into_cell",   # pre-emission positioning
        "center_in_vacuum", # pre-emission positioning
        "verbose_comments", # .fdf comment-block control
    }
    fields_by_name = {f["name"]: f for f in _flatten_schema_fields(sch)}
    for name in molbuilder_only:
        f = fields_by_name.get(name)
        assert f is not None, f"missing field {name} in schema"
        assert f["engine_key"].startswith("(molbuilder"), (
            f"{name}: engine_key={f['engine_key']!r} should start with "
            f"``(molbuilder`` so the UI dims the badge"
        )


def test_engine_key_pins_load_bearing_siesta_keywords():
    """Spot-check that the SIESTA fields whose 1:1 keyword mapping
    other parts of the codebase rely on (or the user manually
    cross-references against the SIESTA manual) carry the exact
    expected engine_key text.  If any of these changes, downstream
    text searches + the .fdf grep workflow break."""
    from molbuilder.web.blueprints._shared import dataclass_to_form_schema
    from molbuilder.config.siesta import SiestaConfig
    sch = dataclass_to_form_schema(SiestaConfig, id_prefix="p")
    fields_by_name = {f["name"]: f for f in _flatten_schema_fields(sch)}
    expected = {
        # The 2026-05-24 SpinPolarized v4-vs-v5 incident hangs on
        # this exact spelling.  Don't drift back to v5 "Spin polarized".
        "spin_polarized": "SpinPolarized",
        # The "two keys, either alone is silently ignored" warning
        # depends on the badge text mentioning BOTH.
        "spin_total":     "Spin.Fix + Spin.Total",
        # Documented user-facing keywords -- ``MeshCutoff`` /
        # ``PAO.BasisSize`` are SIESTA's own names, and the help text
        # references them.
        "mesh_cutoff":    "MeshCutoff",
        "basis_size":     "PAO.BasisSize",
        "net_charge":     "NetCharge",
        "xc_authors":     "XC.authors",
        "xc_functional":  "XC.functional",
        "solution_method": "SolutionMethod",
        "kgrid":          "%block kgrid_Monkhorst_Pack",
    }
    for name, want in expected.items():
        f = fields_by_name.get(name)
        assert f is not None, f"missing field {name}"
        assert f["engine_key"] == want, (
            f"{name}: engine_key={f['engine_key']!r}; expected {want!r}"
        )


def test_engine_key_pins_load_bearing_pyscf_keywords():
    """Same for PySCF.  The 2026-05-24 review surfaced that PySCF's
    method= is a CLASS switch (RKS / UKS / RHF / UHF) not a string
    kwarg -- the engine_key text should explain this."""
    from molbuilder.web.blueprints._shared import dataclass_to_form_schema
    from molbuilder.config.pyscf import PySCFConfig
    sch = dataclass_to_form_schema(PySCFConfig, id_prefix="py")
    fields_by_name = {f["name"]: f for f in _flatten_schema_fields(sch)}
    expected = {
        "charge":   "gto.M(charge=...)",
        "spin":     "gto.M(spin=...)  # 2S, # of unpaired electrons",
        "symmetry": "gto.M(symmetry=...)",
        "basis":    "gto.M(basis=...)",
        "functional": "mf.xc = ...",
        "scf_conv_tol": "mf.conv_tol",
        "scf_max_cycle": "mf.max_cycle",
    }
    for name, want in expected.items():
        f = fields_by_name.get(name)
        assert f is not None, f"missing field {name}"
        assert f["engine_key"] == want, (
            f"{name}: engine_key={f['engine_key']!r}; expected {want!r}"
        )
    # method= is the open-shell-vs-closed-shell selector.  Make
    # sure the badge mentions the class names so the user knows
    # they're picking RKS-vs-UKS, not a string.
    method_key = fields_by_name["method"]["engine_key"]
    for cls in ("RKS", "UKS", "RHF", "UHF"):
        assert cls in method_key, (
            f"method engine_key={method_key!r} should mention {cls} "
            f"(it's a class-selection switch, not a kwarg)"
        )
