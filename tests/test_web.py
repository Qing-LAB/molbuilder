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
