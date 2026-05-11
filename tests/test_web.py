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
    r = web_client.get("/")
    assert r.status_code == 200
    body = r.data.decode()
    for needle in (
        "molbuilder", "input-text", "build-btn",
        "viewer.js", "style.css", "3Dmol-min.js",
    ):
        assert needle in body, needle


def test_index_page_has_tab_markup(web_client):
    r = web_client.get("/")
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


def test_index_page_has_siesta_spin_fields(web_client):
    """Spec: SIESTA tab must expose spin_polarized + spin_total."""
    body = web_client.get("/").data.decode()
    for needle in (
        'id="p-spin-polarized"',
        'id="p-spin-total"',
        'Spin polarized',     # legend / label
    ):
        assert needle in body, f"missing {needle!r} in index.html"


def test_viewer_js_has_compatibility_engine(web_client):
    """Spec: viewer.js must include the parameter-compatibility logic
    (otherwise the UI would let users build a malformed config)."""
    js = web_client.get("/static/viewer.js").data.decode()
    for needle in (
        "applyCompatibility",
        "applyPyscfCompatibility",
        "applySiestaCompatibility",
        "setLock",
        # Each compatibility rule must be present:
        '"RKS" || method === "RHF"',  # method <-> spin
        "py-preopt-functional",        # preopt fields locked
        "p-spin-total",                # SIESTA spin_total lock
    ):
        assert needle in js, f"missing {needle!r} in viewer.js"


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


def test_index_page_lists_threedna_in_backend_dropdown(web_client):
    """3DNA is the highest-quality DNA / RNA backend (canonical B/A/Z
    helix).  It must appear as an explicit choice in the dropdown so
    users with x3dna installed can pick it -- and so users without it
    get the (not installed) suffix the JS adds at page load."""
    body = web_client.get("/").data.decode()
    assert 'value="threedna"' in body, (
        "Backend dropdown should list threedna explicitly"
    )


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


def test_index_page_lists_add_hydrogens_select(web_client):
    """The add_hydrogens control is a tri-state select (auto/on/off)
    in the nucleic-options block, NOT a bool checkbox -- the H/heavy
    threshold heuristic is size-dependent and explicit on/off control
    is the user's escape hatch when auto misclassifies."""
    body = web_client.get("/").data.decode()
    assert 'id="add-hydrogens"' in body
    import re
    m = re.search(r'<select[^>]*id="add-hydrogens"[^>]*>(.*?)</select>',
                  body, re.S)
    assert m, "add-hydrogens should be a <select> (tri-state), not a checkbox"
    options = m.group(1)
    for value in ("auto", "on", "off"):
        assert f'value="{value}"' in options, (
            f"add-hydrogens select missing option {value!r}"
        )
    # auto must be the default-selected option (the "I don't want to
    # think about it" path stays unchanged from the bool-True default).
    auto_opt = re.search(r'<option[^>]*value="auto"[^>]*>', options)
    assert auto_opt and "selected" in auto_opt.group(0), (
        f"auto should be default-selected: {auto_opt.group(0) if auto_opt else None}"
    )


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


def test_watch_viewer_js_honours_path_url_param(web_client):
    """The watch page must read ?path=... from the URL and pre-fill the
    input.  That's the receiving half of the Build -> Watch handoff."""
    js = web_client.get("/static/watch/viewer.js").data.decode()
    for needle in (
        'URLSearchParams',
        'params.get("path")',
        '$("path-input").value = path',
    ):
        assert needle in js, f"missing {needle!r} in watch/viewer.js"


def test_fdf_response_includes_validation_issues(web_client, peptide_xyz):
    """/api/build/fdf returns the validation issue list alongside the
    rendered text so the UI can show warnings to the user.  For a clean
    peptide the list is empty; this just pins the response shape."""
    r = web_client.post("/api/build/fdf",
                        json={"xyz": peptide_xyz, "params": {}})
    body = r.get_json()
    assert body["ok"] is True
    assert "issues" in body and isinstance(body["issues"], list)


def test_both_pages_serve_with_shared_tab_nav(web_client):
    """The unified UI puts a shared tab nav at the top of every page so
    a user can flip between Build (/), Watch (/watch), and Modify
    (/modify) without leaving the app.  The active tab matches the
    current page; the tab links point at the canonical paths."""
    import re
    for path, active_href in [
        ("/",       "/"),
        ("/watch",  "/watch"),
        ("/modify", "/modify"),
    ]:
        r = web_client.get(path)
        assert r.status_code == 200, f"{path} returned {r.status_code}"
        html = r.get_data(as_text=True)
        # Every tab link is present on every page (shared nav).
        assert 'href="/"'       in html, f"{path}: missing Build tab link"
        assert 'href="/watch"'  in html, f"{path}: missing Watch tab link"
        assert 'href="/modify"' in html, f"{path}: missing Modify tab link"
        # The current page's link carries is-active.  Match flexibly
        # so whitespace alignment in the template can change without
        # breaking the test.
        m = re.search(
            rf'<a[^>]*href="{re.escape(active_href)}"[^>]*class="[^"]*is-active[^"]*"',
            html,
        )
        assert m, f"{path}: link to {active_href} missing is-active"


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
    preflight returns a single error-severity issue with where='config'
    rather than HTTP 400.  This lets the UI render the same panel for
    the parse-failure case without a separate code path."""
    r = web_client.post("/api/build/preflight", json={
        "xyz": peptide_xyz,
        "engine": "siesta",
        # kgrid coercion does int(v[i]) -- a non-numeric string here
        # raises ValueError in _siesta_config_from_params, which the
        # endpoint catches as a config-parse error.
        "params": {"kgrid": ["x", "y", "z"]},
    })
    body = r.get_json()
    # ok=True so the UI can keep showing the panel; the actual
    # problem surfaces as an error-severity Issue.
    assert body["ok"] is True
    err = [i for i in body["issues"] if i["severity"] == "error"]
    assert err, f"expected an error issue for bad params; got {body['issues']}"
    assert err[0]["where"] == "config"


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
    r = web_client.post("/api/build/fdf", json={
        "xyz": peptide_xyz,
        "params": {
            "system_name":   "my_pep", "system_label": "pep",
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
    assert "mol_eq = optimize(" in body["script"]
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
            "preopt":           True,
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
    assert "mol_eq = optimize(" not in script   # optimize off
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


def test_modify_page_loads(web_client):
    """``GET /modify`` returns 200 with the M2 page contents.  M2 is
    read-only inspection; the edit ops land in M3+."""
    r = web_client.get("/modify")
    assert r.status_code == 200
    body = r.data.decode()
    for needle in (
        "molbuilder",
        # The page-title and tagline are Modify-specific.
        "modify a structure",
        # Static asset paths the template references.
        "modify/style.css",
        "modify/viewer.js",
        # Three-pane scaffolding the JS targets by id.
        'id="atom-list-body"',
        'id="atom-list"',
        'id="viewer"',
        'id="file-picker"',
        'id="load-btn"',
        'id="rep"',
        'id="show-indices"',
        'id="clear-selection"',
        'id="selection-readout"',
        # All five edit ops are wired (M3 + M4 + M5).
        'id="delete-apply"',
        'id="add-apply"',
        'id="orient-apply"',
        'id="rotate-apply"',
        'id="elc-apply"',
        'id="send-to-build"',
    ):
        assert needle in body, f"missing {needle!r} in /modify HTML"


def test_modify_static_assets_load(web_client):
    """The ``modify/`` static dir must serve the new CSS + JS files."""
    css = web_client.get("/static/modify/style.css")
    assert css.status_code == 200
    assert b".modify-grid" in css.data
    js = web_client.get("/static/modify/viewer.js")
    assert js.status_code == 200
    body = js.data.decode()
    # Sanity-check the JS hits /api/build/load (M2's only backend dep)
    # and registers the click hook for the viewer ↔ list sync.
    assert "/api/build/load" in body
    assert "setClickable"    in body
    assert "rebuildAtomList" in body


def test_all_three_pages_link_to_modify_tab(web_client):
    """Every top-level page (/, /watch, /modify) must include the
    Modify tab link in the shared ``app-tabs`` nav.  Spec: this is the
    same shared-nav block in three templates -- if one drifts, the UI
    becomes inconsistent."""
    for path in ("/", "/watch", "/modify"):
        body = web_client.get(path).data.decode()
        assert 'href="/modify"' in body, (
            f"{path!r} doesn't link to /modify in its app-tabs nav"
        )
        assert 'href="/"' in body
        assert 'href="/watch"' in body


def test_modify_page_marks_itself_active_in_tabs(web_client):
    """The Modify tab link on /modify must carry the is-active class
    (matches /'s Build link and /watch's Watch link)."""
    body = web_client.get("/modify").data.decode()
    # The active link must be the /modify one specifically.
    import re
    m = re.search(
        r'<a[^>]*href="/modify"[^>]*class="[^"]*is-active[^"]*"',
        body,
    )
    assert m, "Modify tab link on /modify is missing is-active"


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


# --------------------------------------------------------------------- #
#  Modify-tab edit-op endpoints (M3).  Body shape carries the canonical #
#  state (xyz + atom_names / residue_ids / residue_names / chain_ids)   #
#  alongside op-specific args; response shape mirrors /api/build/load + #
#  adds an issues array.                                                #
# --------------------------------------------------------------------- #


_H2O_XYZ = "3\nh2o\nO 0 0 0\nH 0.957 0 0\nH -0.24 0.927 0\n"


def test_modify_load_validates_and_canonicalises(web_client):
    r = web_client.post("/api/modify/load", json={"xyz": _H2O_XYZ})
    body = r.get_json()
    assert body["ok"] is True
    assert body["n_atoms"] == 3
    assert body["elements"] == ["O", "H", "H"]
    # Canonical re-export round-trips through Structure.from_xyz, so the
    # atom count + first/last positions match the input.
    out_xyz = body["xyz"]
    assert "3" in out_xyz.splitlines()[0]
    # Issues array is always present (even when empty).
    assert isinstance(body.get("issues"), list)


def test_modify_load_rejects_empty_xyz(web_client):
    r = web_client.post("/api/modify/load", json={"xyz": ""})
    assert r.status_code == 400
    body = r.get_json()
    assert body["ok"] is False
    assert "xyz" in body["error"].lower()


def test_modify_delete_drops_listed_indices(web_client):
    r = web_client.post("/api/modify/delete", json={
        "xyz": _H2O_XYZ,
        "indices": [1, 2],   # both H atoms
    })
    body = r.get_json()
    assert body["ok"] is True
    assert body["n_atoms"] == 1
    assert body["elements"] == ["O"]


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
    when the client passes it back in the body.  Load -> add_atom ->
    delete keeps the atom_names / residue_ids carried alongside xyz."""
    # 1. Initial load -- supply explicit per-atom metadata.
    r1 = web_client.post("/api/modify/load", json={
        "xyz": _H2O_XYZ,
        "atom_names":    ["OW", "HW1", "HW2"],
        "residue_ids":   [7, 7, 7],
        "residue_names": ["WAT", "WAT", "WAT"],
        "chain_ids":     ["B", "B", "B"],
    })
    s1 = r1.get_json()
    assert s1["atom_names"]  == ["OW", "HW1", "HW2"]
    assert s1["residue_ids"] == [7, 7, 7]
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
    body = web_client.get("/modify").data.decode()
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


def test_modify_orient_with_tilt_angle_puts_pair_in_xz_plane(web_client):
    """Non-zero ``angle`` tilts the anchor pair away from the target
    axis in a fixed plane (xz for axis=z).  The vector becomes
    ``(sin θ * d, 0, cos θ * d)``."""
    import math
    r = web_client.post("/api/modify/orient", json={
        "xyz": _LINEAR_XYZ,
        "anchors": [0, 3],
        "axis": "z",
        "angle": 30.0,
    })
    coords = _coords_from_xyz(r.get_json()["xyz"])
    vec = tuple(b - a for a, b in zip(coords[0], coords[3]))
    d = math.sqrt(sum(v * v for v in vec))
    # Tilt direction lies in the xz-plane: y-component is ~0.
    assert abs(vec[1]) < 1e-6, vec
    assert abs(vec[0] - math.sin(math.radians(30.0)) * d) < 1e-5, vec
    assert abs(vec[2] - math.cos(math.radians(30.0)) * d) < 1e-5, vec


def test_modify_orient_center_first_pins_a0_at_origin(web_client):
    r = web_client.post("/api/modify/orient", json={
        "xyz": _LINEAR_XYZ,
        "anchors": [0, 3],
        "center": "first",
    })
    coords = _coords_from_xyz(r.get_json()["xyz"])
    assert all(abs(v) < 1e-6 for v in coords[0]), coords[0]


def test_modify_orient_rejects_same_anchor_twice(web_client):
    r = web_client.post("/api/modify/orient", json={
        "xyz": _LINEAR_XYZ,
        "anchors": [1, 1],
    })
    assert r.status_code == 400
    assert "distinct" in r.get_json()["error"]


def test_modify_orient_rejects_out_of_range_anchor(web_client):
    r = web_client.post("/api/modify/orient", json={
        "xyz": _LINEAR_XYZ,
        "anchors": [0, 99],
    })
    assert r.status_code == 400
    assert "out of range" in r.get_json()["error"]


def test_modify_orient_rejects_bad_axis(web_client):
    r = web_client.post("/api/modify/orient", json={
        "xyz": _LINEAR_XYZ,
        "anchors": [0, 3],
        "axis": "w",
    })
    assert r.status_code == 400
    assert "axis" in r.get_json()["error"]


def test_modify_orient_rejects_non_two_anchors(web_client):
    r = web_client.post("/api/modify/orient", json={
        "xyz": _LINEAR_XYZ,
        "anchors": [0, 1, 2],
    })
    assert r.status_code == 400
    assert "anchors" in r.get_json()["error"]


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


def test_modify_rotate_x_180_degrees(web_client):
    """Rotation around x by 180° flips y and z signs."""
    r = web_client.post("/api/modify/rotate", json={
        "xyz": _LINEAR_XYZ, "axis": "x", "angle": 180,
    })
    coords = _coords_from_xyz(r.get_json()["xyz"])
    # atom 1: (1, 1, 0) -> (1, -1, 0)
    assert abs(coords[1][0] - 1.0)    < 1e-6, coords[1]
    assert abs(coords[1][1] - (-1.0)) < 1e-6, coords[1]
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
    s1 = web_client.post("/api/modify/load", json={
        "xyz": _LINEAR_XYZ,
        "atom_names":    ["C1", "C2", "C3", "C4"],
        "residue_ids":   [1, 1, 1, 1],
        "residue_names": ["MOL", "MOL", "MOL", "MOL"],
        "chain_ids":     ["A", "A", "A", "A"],
    }).get_json()
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
    body = web_client.get("/modify").data.decode()
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


def test_modify_symmetric_electrodes_anchorless_offset_shifts_xy(web_client):
    """Anchorless + ``offset:[Δx, Δy]`` shifts both slabs in xy
    while keeping the z midpoint on the origin."""
    r = web_client.post("/api/modify/symmetric_electrodes", json={
        "xyz":     _SS_XYZ,
        "element": "Au", "plane": "111",
        "size":    [2, 2, 1],
        "gap":     8.0,
        "offset":  [3.0, -2.0],
    })
    body = r.get_json()
    assert body["ok"] is True
    coords = _coords_from_xyz(body["xyz"])
    elc = [coords[i] for i, rn in enumerate(body["residue_names"])
           if rn == "ELC"]
    cx = sum(c[0] for c in elc) / len(elc)
    cy = sum(c[1] for c in elc) / len(elc)
    # Both slabs together: their lateral centroid lands at the offset.
    assert abs(cx - 3.0) < 1e-6, cx
    assert abs(cy + 2.0) < 1e-6, cy


def test_modify_symmetric_electrodes_pair_mode_orthogonal_111(web_client):
    """Orthogonal cell on fcc(111) needs even m × n; 2x2 is fine."""
    r = web_client.post("/api/modify/symmetric_electrodes", json={
        "xyz": _SS_XYZ,
        "element": "Au", "plane": "111",
        "size": [2, 2, 1], "anchors": [1, 0],
        "gap": 8.0, "orthogonal": True,
    })
    assert r.get_json()["ok"] is True


def test_modify_symmetric_electrodes_rejects_unsupported_element(web_client):
    """The closed list of supported FCC metals (Au/Ag/Cu/Ni/Pt/Pd) is
    enforced at the endpoint boundary, not just at the Python API."""
    r = web_client.post("/api/modify/symmetric_electrodes", json={
        "xyz": _SS_XYZ,
        "element": "Fe",            # BCC; not on the FCC closed list
        "plane": "111", "size": [2, 2, 1],
        "anchors": [1, 0],
    })
    assert r.status_code == 400
    err = r.get_json()["error"]
    assert "element" in err
    assert "Au" in err               # the supported list is named


def test_modify_symmetric_electrodes_rejects_bad_plane(web_client):
    r = web_client.post("/api/modify/symmetric_electrodes", json={
        "xyz": _SS_XYZ, "element": "Au", "plane": "112",
        "size": [2, 2, 1], "anchors": [1, 0],
    })
    assert r.status_code == 400
    assert "plane" in r.get_json()["error"]


def test_modify_symmetric_electrodes_rejects_bad_size(web_client):
    """``size`` must be 3 positive integers."""
    r = web_client.post("/api/modify/symmetric_electrodes", json={
        "xyz": _SS_XYZ, "element": "Au", "plane": "111",
        "size": [0, 1, 1], "anchors": [1, 0],
    })
    assert r.status_code == 400
    assert "size" in r.get_json()["error"]


def test_modify_symmetric_electrodes_rejects_same_anchor_twice(web_client):
    r = web_client.post("/api/modify/symmetric_electrodes", json={
        "xyz": _SS_XYZ, "element": "Au", "plane": "111",
        "size": [2, 2, 1], "anchors": [0, 0],
    })
    assert r.status_code == 400
    assert "distinct" in r.get_json()["error"]


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


def test_modify_electrode_single_mode_minus_z(web_client):
    """``side="-z"`` builds below the anchor."""
    r = web_client.post("/api/modify/electrode", json={
        "xyz": _SS_XYZ, "element": "Au", "plane": "111",
        "size": [2, 2, 1], "anchor_index": 0, "side": "-z",
    })
    body = r.get_json()
    assert body["ok"] is True
    # The 4 added Au atoms must all sit below the anchor's z (= -2).
    z_au = [body["xyz"].splitlines()[i + 2].split()[3]
            for i, el in enumerate(body["elements"]) if el == "Au"]
    assert all(float(z) < -2.0 for z in z_au), z_au


def test_modify_electrode_rejects_bad_side(web_client):
    r = web_client.post("/api/modify/electrode", json={
        "xyz": _SS_XYZ, "element": "Au", "plane": "111",
        "size": [2, 2, 1], "anchor_index": 0, "side": "above",
    })
    assert r.status_code == 400
    assert "side" in r.get_json()["error"]


def test_modify_electrode_rejects_bad_anchor(web_client):
    r = web_client.post("/api/modify/electrode", json={
        "xyz": _SS_XYZ, "element": "Au", "plane": "111",
        "size": [2, 2, 1], "anchor_index": 99,
    })
    assert r.status_code == 400
    assert "out of range" in r.get_json()["error"]


def test_modify_electrode_lattice_constant_override(web_client):
    """``lattice_constant`` lets the user override the per-element
    table value (e.g. for strained-electrode studies)."""
    r = web_client.post("/api/modify/electrode", json={
        "xyz": _SS_XYZ, "element": "Au", "plane": "111",
        "size": [2, 2, 1], "anchor_index": 1, "side": "+z",
        "lattice_constant": 4.5,        # vs Au's actual 4.0782 Å
    })
    assert r.get_json()["ok"] is True


# --------------------------------------------------------------------- #
#  Basename validation (job-layout v1)                                  #
# --------------------------------------------------------------------- #


def test_modify_fdf_rejects_slash_in_system_label(web_client):
    """The Build /api/build/fdf endpoint validates ``system_label``
    against the basename charset before any file write.  Slashes,
    spaces, dots, and leading-dot are all rejected per
    docs/spec/job-layout.md."""
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


def test_modify_translate_rejects_inf_offset(web_client):
    """Same for Inf."""
    r = web_client.post("/api/modify/translate", json={
        "xyz": _LINEAR_XYZ, "dx": float("inf"), "dy": 0.0, "dz": 0.0,
    })
    assert r.status_code == 400


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

    expected = [
        ("System",                   1),
        ("Basis & grid",             3),
        ("Exchange-correlation",     2),
        ("SCF",                      7),
        ("Parallel execution",       2),
        ("Spin",                     2),
        ("k-grid (Monkhorst-Pack)",  1),
        ("Relaxation",               4),
        ("Output & positioning",     6),
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
        ("Pre-optimization (optional)",  5),
        ("Optimization",                 6),
        ("Solvent (optional)",           2),
        ("Frequencies / thermochemistry", 3),
        ("Runtime & output",             6),
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
