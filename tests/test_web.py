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


def test_index_page_lists_add_hydrogens_checkbox(web_client):
    """The Add hydrogens opt-out lives next to Neutralise phosphates
    in the nucleic-options block.  Default ON for simulation-readiness."""
    body = web_client.get("/").data.decode()
    assert 'id="add-hydrogens"' in body
    # Default checked: the input element should carry the `checked`
    # attribute (HTML allows it as a bare boolean attr).
    import re
    m = re.search(r'<input[^>]*id="add-hydrogens"[^>]*>', body)
    assert m and "checked" in m.group(0), (
        f"add-hydrogens checkbox should be default-checked: {m.group(0) if m else None}"
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


def test_index_page_has_watch_handoff_buttons(web_client):
    """Phase 3d: every Generate flow gets a 'Watch this run' button
    that opens /watch with a pre-filled molwatch.log path.  The button
    is disabled at page load (Generate hasn't run yet) and enabled by
    viewer.js after a successful render."""
    body = web_client.get("/").data.decode()
    for needle in ('id="watch-fdf"', 'id="watch-pyscf"'):
        assert needle in body, f"missing {needle!r} on Build page"
    # Both should be `disabled` on initial render -- we don't want a
    # dead-link click before Generate has produced a label.
    import re
    for btn_id in ("watch-fdf", "watch-pyscf"):
        m = re.search(rf'<button[^>]*id="{btn_id}"[^>]*>', body)
        assert m and "disabled" in m.group(0), (
            f"{btn_id} should be disabled on initial load: "
            f"{m.group(0) if m else None}"
        )


def test_viewer_js_wires_watch_handoff(web_client):
    """The viewer.js bundle must (a) enable each watch button after
    a successful Generate, (b) navigate to /watch?path=<label>.molwatch.log
    on click, and (c) read system_label / job_name from the response
    into the state object so the click handler has a label to use."""
    js = web_client.get("/static/viewer.js").data.decode()
    for needle in (
        'state.fdf_label = r.system_label',
        'state.pyscf_label = r.job_name',
        '$("watch-fdf").disabled = false',
        '$("watch-pyscf").disabled = false',
        # Use raw double-quotes -- JS uses double-quoted /watch URL.
        '/watch?path=',
    ):
        assert needle in js, f"missing {needle!r} in viewer.js"


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
    a user can flip between Build (/) and Watch (/watch) without
    leaving the app.  The active tab matches the current page; both
    tab links point at the canonical paths."""
    for path, active in [("/", "Build"), ("/watch", "Watch")]:
        r = web_client.get(path)
        assert r.status_code == 200, f"{path} returned {r.status_code}"
        html = r.get_data(as_text=True)
        # Both tab links present
        assert 'href="/"' in html, f"{path}: missing Build tab link"
        assert 'href="/watch"' in html, f"{path}: missing Watch tab link"
        # Active state on the current page
        if active == "Build":
            assert 'href="/" class="app-tab is-active"' in html, (
                "/ should mark Build active"
            )
        else:
            assert 'href="/watch" class="app-tab is-active"' in html, (
                "/watch should mark Watch active"
            )


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
    assert "mf = pcm.PCM(mf)" in script
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
