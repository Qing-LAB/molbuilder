"""End-to-end Flask test for the molbuilder web UI.

Exercises both /api/build (every kind we ship) and /api/fdf, plus the
/ index page.  Skipped cleanly if Flask isn't installed.
"""

from __future__ import annotations

import json
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    try:
        from molbuilder.web.app import create_app
    except ImportError as e:
        print(f"SKIP -- Flask not installed ({e})")
        return

    app = create_app()
    client = app.test_client()

    # ---- Index page -------------------------------------------------
    r = client.get("/")
    assert r.status_code == 200, r.status_code
    body = r.data.decode()
    for needle in (
        "molbuilder", "input-text", "build-btn",
        "viewer.js", "style.css", "3Dmol-min.js",
    ):
        assert needle in body, needle
    print("  / OK")

    # ---- /api/health ------------------------------------------------
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.get_json()["ok"] is True

    # ---- /api/build : peptide --------------------------------------
    r = client.post("/api/build", json={"kind": "peptide", "input": "ARNDC"})
    assert r.status_code == 200, r.status_code
    body = r.get_json()
    assert body["ok"] is True
    assert body["n_atoms"] >= 38      # heavy-atom minimum; H optional
    assert "ARNDC" in (body["title"] or "")
    print(f"  /api/build peptide: {body['n_atoms']} atoms")

    xyz_pep = body["xyz"]

    # ---- /api/build : DNA -------------------------------------------
    r = client.post("/api/build", json={"kind": "dna", "input": "ATGC"})
    body = r.get_json()
    assert body["ok"] is True
    assert body["n_residues"] == 4
    print(f"  /api/build dna: {body['n_atoms']} atoms")

    # ---- /api/build : RNA -------------------------------------------
    r = client.post("/api/build", json={"kind": "rna", "input": "AUGC"})
    body = r.get_json()
    assert body["ok"] is True
    assert "P" in body["elements"]
    print(f"  /api/build rna: {body['n_atoms']} atoms")

    # ---- /api/build : SMILES (optional, needs RDKit) ---------------
    r = client.post("/api/build",
                    json={"kind": "smiles", "input": "c1ccccc1"})
    body = r.get_json()
    if body.get("ok"):
        assert body["n_atoms"] == 12
        print(f"  /api/build smiles benzene: {body['n_atoms']} atoms")
    else:
        print(f"  /api/build smiles: skipped ({body.get('error')})")

    # ---- /api/build : bad input gives helpful error ----------------
    r = client.post("/api/build", json={"kind": "peptide", "input": "AXXC"})
    body = r.get_json()
    assert body["ok"] is False
    assert "X" in body["error"], body
    print("  bad input rejected with clear error")

    # ---- /api/fdf : default params ---------------------------------
    r = client.post("/api/fdf", json={"xyz": xyz_pep, "params": {}})
    body = r.get_json()
    assert body["ok"] is True
    assert "SystemName" in body["fdf"]
    assert "ChemicalSpeciesLabel" in body["fdf"]
    print(f"  /api/fdf default: {body['fdf'].count(chr(10))} lines")

    # ---- /api/fdf : custom params ----------------------------------
    r = client.post("/api/fdf", json={
        "xyz": xyz_pep,
        "params": {
            "system_name":  "my_pep", "system_label": "pep",
            "basis_size":   "TZP",
            "mesh_cutoff":  450.0,
            "xc_functional": "GGA", "xc_authors": "BLYP",
            "kgrid": [4, 4, 1],
            "relax_type": "none",
            "max_scf_iter": 1000,
        }
    })
    body = r.get_json()
    assert body["ok"] is True
    fdf = body["fdf"]
    assert "SystemName        my_pep" in fdf
    assert "PAO.BasisSize TZP"        in fdf
    assert "MeshCutoff 450.0 Ry"      in fdf
    assert "XC.authors    BLYP"       in fdf
    assert "4 0 0 0.0"                in fdf
    assert "MD.TypeOfRun"         not in fdf, "relax_type=none must drop MD block"
    assert "MaxSCFIterations  1000"   in fdf
    print(f"  /api/fdf custom: TZP / 450 Ry / BLYP / kgrid 4x4x1 / no relax")

    # ---- /api/fdf : missing xyz -----------------------------------
    r = client.post("/api/fdf", json={"params": {}})
    body = r.get_json()
    assert body["ok"] is False
    print("  /api/fdf without xyz -> error (correct)")

    # ---- /api/load : XYZ via JSON body -----------------------------
    r = client.post("/api/load",
                    json={"text": xyz_pep, "filename": "peptide.xyz"})
    body = r.get_json()
    assert body["ok"] is True
    assert body["source_format"] == "xyz"
    assert body["n_atoms"] >= 38
    print(f"  /api/load xyz (json): {body['n_atoms']} atoms")

    # ---- /api/load : PDB via JSON body -----------------------------
    pep_pdb = client.post("/api/build",
                          json={"kind": "peptide", "input": "AC"}
                          ).get_json()["pdb"]
    r = client.post("/api/load",
                    json={"text": pep_pdb, "filename": "ac.pdb"})
    body = r.get_json()
    assert body["ok"] is True
    assert body["source_format"] == "pdb"
    print(f"  /api/load pdb (json): {body['n_atoms']} atoms")

    # ---- /api/load : sniffing without extension --------------------
    r = client.post("/api/load", json={"text": xyz_pep, "filename": ""})
    body = r.get_json()
    assert body["ok"] is True
    assert body["source_format"] == "xyz"
    print("  /api/load xyz (sniffed without extension): ok")

    # ---- /api/load : multipart upload ------------------------------
    import io
    from werkzeug.datastructures import FileStorage
    fs = FileStorage(stream=io.BytesIO(xyz_pep.encode()),
                     filename="upload.xyz",
                     content_type="chemical/x-xyz")
    r = client.post("/api/load",
                    data={"file": fs}, content_type="multipart/form-data")
    body = r.get_json()
    assert body["ok"] is True, body
    assert body["source_format"] == "xyz"
    print(f"  /api/load xyz (multipart): {body['n_atoms']} atoms")

    # ---- /api/load : empty -> error -------------------------------
    r = client.post("/api/load", json={"text": ""})
    body = r.get_json()
    assert body["ok"] is False
    print("  /api/load with empty body -> error (correct)")

    # ---- /api/load -> /api/fdf chain ------------------------------
    r = client.post("/api/load",
                    json={"text": xyz_pep, "filename": "p.xyz"})
    loaded = r.get_json()
    r = client.post("/api/fdf",
                    json={"xyz": loaded["xyz"], "params": {"system_label": "lp"}})
    body = r.get_json()
    assert body["ok"] is True
    assert "SystemLabel       lp" in body["fdf"]
    print("  /api/load -> /api/fdf chain works end-to-end")

    # ---- index page must include the tab markup -----------------
    r = client.get("/")
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
    print("  / includes SIESTA/PySCF tab markup")

    # ---- /api/pyscf : default params ----------------------------
    r = client.post("/api/pyscf", json={"xyz": xyz_pep, "params": {}})
    body = r.get_json()
    assert body["ok"] is True, body
    assert "from pyscf import gto, scf, dft, lib" in body["script"]
    assert 'mf.xc = "B3LYP"' in body["script"]
    assert "mol_eq = optimize(" in body["script"]
    print(f"  /api/pyscf default: {body['script'].count(chr(10))} lines, "
          f"job={body['job_name']!r}")

    # Verify the script we got back is syntactically valid Python
    compile(body["script"], "<api/pyscf default>", "exec")

    # ---- /api/pyscf : custom params ----------------------------
    r = client.post("/api/pyscf", json={
        "xyz": xyz_pep,
        "params": {
            "job_name":  "my_pep",
            "method":    "UKS",
            "spin":      1,
            "charge":    -1,
            "functional": "PBE0",
            "basis":     "def2-TZVP",
            "preopt":    True,
            "optimize":  False,
            "dispersion": "d4",
            "solvent":   "water",
            "verbose_comments": False,
        }
    })
    body = r.get_json()
    assert body["ok"] is True, body
    script = body["script"]
    assert 'JOB = "my_pep"' in script
    assert "mf = dft.UKS(mol)" in script
    assert "spin       = 1" in script
    assert "charge     = -1" in script
    assert 'mf.xc = "PBE0"' in script
    assert 'basis      = "def2-TZVP"' in script
    assert "mol_eq = optimize(" not in script           # opt off
    assert 'mf.disp = "d4"' in script
    assert "mf = pcm.PCM(mf)" in script
    assert "TROUBLESHOOTING" not in script              # verbose off
    print("  /api/pyscf custom: UKS / PBE0 / def2-TZVP / D4 / water / no-opt")

    # ---- /api/pyscf : auto-detect charge from phosphates -------
    # Strip the H-on-OP atoms from the DNA xyz so detection sees a charge.
    # Easier path: hand-craft a 7-atom diester missing both HOPs.
    dna_xyz_strip = (
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
    r = client.post("/api/pyscf", json={"xyz": dna_xyz_strip, "params": {}})
    body = r.get_json()
    assert body["ok"] is True
    assert "charge     = -1" in body["script"]
    print("  /api/pyscf auto-charge: detects -1 from deprotonated phosphate")

    # ---- /api/pyscf : missing xyz -------------------------------
    r = client.post("/api/pyscf", json={"params": {}})
    body = r.get_json()
    assert body["ok"] is False
    print("  /api/pyscf without xyz -> error (correct)")

    print("OK -- all endpoints exercised.")


if __name__ == "__main__":
    main()
