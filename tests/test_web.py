"""Flask tests for the build/describe surface of the molbuilder web UI.

Covers the build blueprint's doors (molecule / load / preflight /
schema), the modify routes, and the form-schema plumbing.  NOT every endpoint: spectra,
watch, checkpoint, projects and transport are tested in their own
files.  Skipped cleanly if Flask isn't installed.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from support.envelope import (from_xyz as _env,
                             from_xyz_with_periodicity as _env_per)

import pytest


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


def test_build_response_carries_validation_issues(web_client):
    """The build response carries the structure's findings, so the page
    shows them as soon as the structure exists -- here the H/heavy
    reference note of an X3DNA heavy-atom skeleton (add_hydrogens off)."""
    from molbuilder.builders.backends import available_backends
    if not available_backends().get("threedna"):
        pytest.skip("threedna backend not installed")
    r = web_client.post("/api/build/molecule",
                        json={"kind": "dna", "input": "ATGC",
                              "backend": "threedna",
                              "add_hydrogens": "off",
                              "protonate_phosphates": False})
    body = r.get_json()
    assert body["ok"] is True
    issues = body.get("issues") or []
    h_ratio_notes = [i for i in issues
                     if i["severity"] == "info" and i["where"] == "geometry.h_ratio"]
    assert len(h_ratio_notes) == 1, (
        f"expected one h_ratio note for the skeleton, got: {issues}"
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
    assert len(set(body["residue_ids"])) == 4


def test_build_rna(web_client):
    r = web_client.post("/api/build/molecule", json={"kind": "rna", "input": "AUGC"})
    body = r.get_json()
    assert body["ok"] is True
    assert "P" in body["structure"]["elements"]


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


@pytest.fixture
def peptide_xyz(web_client):
    """xyz string of an ARNDC peptide via the build endpoint."""
    r = web_client.post("/api/build/molecule",
                        json={"kind": "peptide", "input": "ARNDC"})
    return r.get_json()["text"]


_PATTERN_B_REGIONS = {"L-electrode": [0, 1, 2]}


# --------------------------------------------------------------------- #
#  /api/build/preflight (live validation hint endpoint)                 #
# --------------------------------------------------------------------- #


def test_preflight_returns_issues_for_siesta(web_client, peptide_xyz):
    """Validation-only endpoint runs validate(struct, cfg) without
    rendering FDF text.  Unrestricted at 2S = 0 on a closed-shell peptide
    -- a constrained singlet, the electronic state's own warning (ES9) --
    round-trips through the preflight endpoint, keyed to the field."""
    r = web_client.post("/api/build/preflight", json={
        "structure": _env(peptide_xyz),
        "engine": "siesta", "calculation": "optimization",
        "params": {"spin_treatment": "unrestricted",
                   "unpaired_electrons": 0},
    })
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    issues = body["issues"]
    assert isinstance(issues, list)
    assert any(i["severity"] == "warn"
               and "spin_treatment" in (i["where"] or "")
               for i in issues), f"expected spin_treatment warn; got {issues}"
    # Each entry has the JSON shape the UI expects.
    for i in issues:
        assert set(i.keys()) >= {"severity", "message", "where"}


def test_preflight_rejects_bad_engine(web_client, peptide_xyz):
    r = web_client.post("/api/build/preflight", json={
        "structure": _env(peptide_xyz),
        "engine": "qchem",   # not supported
        "calculation": "optimization",
        "params": {},
    })
    assert r.status_code == 400
    assert r.get_json()["ok"] is False


def test_preflight_bad_params_returned_as_error_issue(web_client, peptide_xyz):
    """When the params dict can't be coerced into a valid config
    (e.g. kgrid with non-numeric entries that fail the int() cast),
    preflight surfaces the failure as an error-severity Issue with
    where='config' in the body's ``issues`` array.

    The response uses ``ok: False`` + HTTP 400 (R4-A), so the UI's
    ``!body.ok`` gate renders issues uniformly.
    """
    r = web_client.post("/api/build/preflight", json={
        "structure": _env(peptide_xyz),
        "engine": "siesta", "calculation": "optimization",
        # a non-numeric k-grid will not read as three whole numbers: the
        # params door (`_shared.stated_params`, `template.read_text`)
        # refuses it, which the endpoint catches as a config-parse error.
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


def test_a_blank_is_not_chosen_and_a_value_that_will_not_read_is_refused_by_name(
        web_client, peptide_xyz):
    """`web/form-schema.md` § 1.1 (plan § 5w K7), at the one door every form
    goes through (`_shared.stated_params`, then `template.stated_config`).

    A BLANK IS NOT CHOSEN, for every field: the field keeps what lies under
    it.  An empty number reached ``float(None)`` and answered a raw
    TypeError naming no field (the M11 review's SS-C16 / SO-N5).  A VALUE
    THAT WILL NOT READ AS ITS TYPE is refused, naming its field -- a
    fractional count among them, never rounded: ``int(4.5)`` gave 4, a k-point
    mesh of four where 4.5 was typed (the K3 review)."""
    def preflight(params):
        return web_client.post("/api/build/preflight", json={
            "structure": _env(peptide_xyz), "engine": "siesta",
            "calculation": "optimization", "params": params})

    r = preflight({"mesh_cutoff": None, "max_scf_iter": "",
                   "basis_size": None, "kgrid": None})
    assert r.status_code == 200, r.get_json()
    for name, value in (("kgrid", [4.5, 4, 1]), ("kgrid", "4,4"),
                        ("max_scf_iter", "2.5"), ("mesh_cutoff", "abc")):
        r = preflight({name: value})
        assert r.status_code == 400, (name, r.get_json())
        assert name in r.get_json()["error"], r.get_json()


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


def test_load_empty_returns_error(web_client):
    r = web_client.post("/api/build/load", json={"text": ""})
    body = r.get_json()
    assert body["ok"] is False


def test_modify_op_round_trips_the_periodic_cell(web_client):
    """Regression (2026-07 fresh-eyes review): a modify op must carry the
    periodicity (cell / axis_kind / vacuum) through the round-trip.  Before the
    fix ``struct_from_body`` rebuilt an isolated Structure and the response reset
    the client's cell to defaults -> the next SIESTA FDF silently dropped
    LatticeVectors.  (k-grid is NOT geometry -- a stray ``kgrid`` in the payload
    is ignored and never echoed back; structure-periodicity.md.)"""
    import numpy as np
    periodicity = {
        "cell": [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 20.0]],
        "axis_kind": ["periodic", "periodic", "transport"],
        "vacuum": [5.0, 5.0, 0.0],
    }
    # A count-changing op (delete) carries the lattice VERBATIM.
    body = web_client.post("/api/modify/delete", json={
        "structure": _env_per(_H2O_XYZ, periodicity), "indices": [1]}).get_json()
    assert body["ok"] is True, body
    per = body["periodicity"]
    assert per["cell"] == periodicity["cell"], "delete dropped/changed the cell"
    assert per["axis_kind"] == periodicity["axis_kind"], "delete dropped axis_kind"
    assert per["vacuum"] == periodicity["vacuum"], "delete dropped vacuum"
    assert "kgrid" not in per, "delete should not echo k-grid (not geometry)"

    # A rigid ROTATION turns the atoms and leaves the box alone (user,
    # 2026-09-25: "moving atoms only moves atoms"); it must not DROP the cell
    # either.  axis_kind / vacuum are non-geometric -> verbatim.
    body = web_client.post("/api/modify/rotate", json={
        "structure": _env_per(_H2O_XYZ, periodicity), "axis": "z", "angle": 30}).get_json()
    assert body["ok"] is True, body
    per = body["periodicity"]
    assert per["cell"] is not None, "rotate dropped the cell"
    assert np.allclose(per["cell"], periodicity["cell"]), \
        "rotate moved the cell: moving atoms only moves atoms"
    assert per["axis_kind"] == periodicity["axis_kind"], "rotate dropped axis_kind"
    assert per["vacuum"] == periodicity["vacuum"], "rotate dropped vacuum"
    assert "kgrid" not in per, "rotate should not echo k-grid (not geometry)"
    # Nobody assigned an origin, and a move does not assign one: the rule
    # places the box around the atoms where they now are
    # (`model/structure-periodicity.md` § 6.0).
    assert per["engine_offset"] is None, (
        "rotate stored an origin nobody assigned", per["engine_offset"])


def test_modify_op_keeps_an_assigned_origin(web_client):
    """A rigid whole-structure ROTATION leaves an assigned origin as it was:
    moving atoms only moves atoms (user, 2026-09-25).  It must not DROP it
    either -- the box would jump to wherever the rule puts it."""
    import numpy as np
    periodicity = {
        "cell": [[4.0, 0, 0], [0, 4.0, 0], [0, 0, 12.0]],
        "engine_offset": [2.0, 2.0, 6.0],
        "axis_kind": ["periodic", "periodic", "transport"],
    }
    r = web_client.post("/api/modify/rotate", json={
        "structure": _env_per(_H2O_XYZ, periodicity), "axis": "z", "angle": 15})
    per = r.get_json()["periodicity"]
    assert per["engine_offset"] is not None, "rotate dropped the assigned origin"
    assert np.allclose(per["engine_offset"], periodicity["engine_offset"]), \
        "rotate moved the assigned origin: moving atoms only moves atoms"
    assert np.allclose(per["box_corner"], [-2.0, -2.0, -6.0]), per["box_corner"]


# --------------------------------------------------------------------- #
#  Modify-tab edit-op endpoints (M3).  Body shape carries the canonical #
#  state (xyz + atom_names / residue_ids / residue_names / chain_ids)   #
#  alongside op-specific args; response shape mirrors /api/build/load + #
#  adds an issues array.                                                #
# --------------------------------------------------------------------- #


_H2O_XYZ = "3\nh2o\nO 0 0 0\nH 0.957 0 0\nH -0.24 0.927 0\n"


def test_modify_delete_drops_listed_indices(web_client):
    r = web_client.post("/api/modify/delete", json={
        "structure": _env(_H2O_XYZ),
        "indices": [1, 2],   # both H atoms
    })
    body = r.get_json()
    assert body["ok"] is True
    assert body["n_atoms"] == 1
    assert body["structure"]["elements"] == ["O"]


def test_modify_responses_carry_atoms_list(web_client):
    """BOMB-0 fix (2026-06-07): every /api/modify/* response carries
    an ``atoms`` list in the same shape ``/api/selection/atoms``
    returns, so the front-end's selection store stays in sync
    with the in-memory post-op structure without a disk re-fetch.

    Pre-fix, modifier-op responses only carried xyz + elements +
    metadata lists; the selection panel went stale after every op."""
    r = web_client.post("/api/modify/delete", json={
        "structure": _env(_H2O_XYZ),
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


def test_modify_delete_silently_ignores_out_of_range(web_client):
    """Matches molbuilder.modify.delete_atoms behaviour."""
    r = web_client.post("/api/modify/delete", json={
        "structure": _env(_H2O_XYZ),
        "indices": [99, -1, 0],   # only 0 is in range
    })
    body = r.get_json()
    assert body["ok"] is True
    assert body["n_atoms"] == 2     # dropped O, kept the two H


def test_modify_delete_rejects_non_int_indices(web_client):
    r = web_client.post("/api/modify/delete", json={
        "structure": _env(_H2O_XYZ),
        "indices": ["a", "b"],
    })
    assert r.status_code == 400
    assert r.get_json()["ok"] is False


def test_modify_add_atom_appends_at_offset(web_client):
    r = web_client.post("/api/modify/add_atom", json={
        "structure": _env(_H2O_XYZ),
        "element": "H",
        "anchor_index": 0,            # the O
        "offset": [0.0, 0.0, 1.5],
    })
    body = r.get_json()
    assert body["ok"] is True
    assert body["n_atoms"] == 4
    assert body["structure"]["elements"][-1] == "H"
    # New atom lands in a fresh residue named MOD (default).
    assert body["residue_names"][-1] == "MOD"


def test_modify_add_atom_without_anchor_measures_from_the_origin(web_client):
    """An OMITTED ``anchor_index`` is the wire form of "nothing is selected",
    and the route reads it as the world origin (molview.md § 11.1).

    The client omits the group key at empty selection for every op; for this
    one that absence has to mean something, and (0, 0, 0) is it.
    """
    r = web_client.post("/api/modify/add_atom", json={
        "structure": _env(_H2O_XYZ),
        "element": "S",
        "offset": [1.25, -0.5, 2.0],
    })
    body = r.get_json()
    assert body["ok"] is True, body
    assert body["n_atoms"] == 4
    assert body["structure"]["elements"][-1] == "S"
    # The offset IS the position -- read it back off the xyz the route returns.
    x, y, z = [float(v) for v in body["text"].splitlines()[-1].split()[1:4]]
    assert (round(x, 6), round(y, 6), round(z, 6)) == (1.25, -0.5, 2.0)


def test_modify_add_atom_rejects_bad_anchor(web_client):
    r = web_client.post("/api/modify/add_atom", json={
        "structure": _env(_H2O_XYZ),
        "element": "H",
        "anchor_index": 99,
        "offset": [0, 0, 1],
    })
    assert r.status_code == 400
    body = r.get_json()
    assert body["ok"] is False
    assert "anchor_index" in body["error"]


def test_modify_add_atom_rejects_unknown_element(web_client):
    """Scientific guard: a non-periodic-table symbol is rejected at the op
    boundary (400), not silently accepted to detonate later in a generator."""
    r = web_client.post("/api/modify/add_atom", json={
        "structure": _env(_H2O_XYZ),
        "element": "Xx",
        "anchor_index": 0,
        "offset": [0, 0, 1],
    })
    assert r.status_code == 400
    body = r.get_json()
    assert body["ok"] is False
    assert "unknown element symbol" in body["error"]


def test_modify_add_atom_zero_offset_is_advisory_not_blocked(web_client):
    """Advisory-not-enforcing (validation contract): a zero offset places the new
    atom on top of the anchor, but /api/modify/add_atom does NOT reject it -- it
    returns the structure (ok, 200) and surfaces the coincident atoms as a
    non-blocking ``geometry.min_distance`` issue.  The editing stage advises; the
    generation gate enforces."""
    r = web_client.post("/api/modify/add_atom", json={
        "structure": _env(_H2O_XYZ),
        "element": "H",
        "anchor_index": 0,
        "offset": [0, 0, 0],
    })
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    assert body["n_atoms"] == 4
    assert any(i.get("where") == "geometry.min_distance"
               for i in (body.get("issues") or []))


def test_modify_add_atom_rejects_missing_offset(web_client):
    r = web_client.post("/api/modify/add_atom", json={
        "structure": _env(_H2O_XYZ),
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
        "structure": _env(_H2O_XYZ, atom_names=["OW", "HW1", "HW2"], residue_ids=[7, 7, 7], residue_names=["WAT", "WAT", "WAT"], chain_ids=["B", "B", "B"]),
    }
    # 2. Add an atom; the metadata for the original 3 atoms must
    # survive (Structure preserves through add_atom).
    r2 = web_client.post("/api/modify/add_atom", json={
        "structure": s1["structure"],
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
        "structure": s2["structure"],
        "indices":       [3],
    })
    s3 = r3.get_json()
    assert s3["n_atoms"] == 3
    assert s3["atom_names"]    == ["OW", "HW1", "HW2"]
    assert s3["residue_names"] == ["WAT", "WAT", "WAT"]


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
        "structure": _env(_LINEAR_XYZ),
        "anchors": [0, 3],
    })
    body = r.get_json()
    assert body["ok"] is True
    coords = _coords_from_xyz(body["text"])
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
        "structure": _env(_LINEAR_XYZ),
        "anchors": [0, 3],
        "axis": "w",
    })
    assert r.status_code == 400
    assert "axis" in r.get_json()["error"]


def test_modify_rotate_z_90_degrees(web_client):
    """Rotation around z by 90° maps (1, 1, 0) -> (-1, 1, 0)."""
    r = web_client.post("/api/modify/rotate", json={
        "structure": _env(_LINEAR_XYZ), "axis": "z", "angle": 90,
    })
    body = r.get_json()
    assert body["ok"] is True
    coords = _coords_from_xyz(body["text"])
    # atom 1 was at (1, 1, 0); after a +90° rotation around z it
    # becomes (-1, 1, 0).
    assert abs(coords[1][0] - (-1.0)) < 1e-6, coords[1]
    assert abs(coords[1][1] - 1.0)    < 1e-6, coords[1]
    assert abs(coords[1][2] - 0.0)    < 1e-6, coords[1]


def test_modify_rotate_rejects_bad_axis(web_client):
    r = web_client.post("/api/modify/rotate", json={
        "structure": _env(_LINEAR_XYZ), "axis": "w", "angle": 30,
    })
    assert r.status_code == 400
    assert "axis" in r.get_json()["error"]


def test_modify_rotate_rejects_non_numeric_angle(web_client):
    r = web_client.post("/api/modify/rotate", json={
        "structure": _env(_LINEAR_XYZ), "axis": "z", "angle": "ninety",
    })
    assert r.status_code == 400
    assert "angle" in r.get_json()["error"]


# --------------------------------------------------------------------- #
#  /api/modify/translate                                                 #
# --------------------------------------------------------------------- #


def test_modify_translate_recenter_puts_centroid_at_origin(web_client):
    """``recenter: true`` with no group translates so the WHOLE
    structure's centroid sits at (0, 0, 0).

    _LINEAR_XYZ is (0,0,0)/(1,1,0)/(2,2,0)/(3,3,0), centroid
    (1.5, 1.5, 0)."""
    r = web_client.post("/api/modify/translate", json={
        "structure": _env(_LINEAR_XYZ),
        "recenter": True,
    })
    body = r.get_json()
    assert body["ok"] is True
    coords = _coords_from_xyz(body["text"])
    cx = sum(c[0] for c in coords) / len(coords)
    cy = sum(c[1] for c in coords) / len(coords)
    cz = sum(c[2] for c in coords) / len(coords)
    assert abs(cx) < 1e-9, cx
    assert abs(cy) < 1e-9, cy
    assert abs(cz) < 1e-9, cz


def test_modify_translate_recenter_centres_THE_GROUP_and_moves_only_it(web_client):
    """The bug this route shipped with: Center ignored the selection.

    User, 2026-08-30: *"when we have selected a group of atoms, the idea
    is that the center operation would be about this selected group, not
    the whole structure"* -- and *"the group moves without anything else
    moved.  the group is the rigid part."*

    The browser was never at fault: `applyOp` injects the selection from
    `OPERATIONS.translate.group` for Center exactly as it does for
    Translate, so both bodies carry `indices`.  The route returned from
    its `recenter` branch BEFORE that key was read.

    _LINEAR_XYZ is (0,0,0)/(1,1,0)/(2,2,0)/(3,3,0).  Centring atoms 0-1
    moves their own centroid (0.5, 0.5, 0) to the origin, so they land
    on (-0.5,-0.5,0) and (0.5,0.5,0) -- and atoms 2-3 must not move at
    all, which is the half a whole-structure centring would also get
    wrong in the other direction.
    """
    r = web_client.post("/api/modify/translate", json={
        "structure": _env(_LINEAR_XYZ),
        "recenter": True,
        "indices": [0, 1],
    })
    body = r.get_json()
    assert body["ok"] is True, body
    out = _coords_from_xyz(body["text"])
    assert abs(out[0][0] + 0.5) < 1e-9 and abs(out[0][1] + 0.5) < 1e-9, out[0]
    assert abs(out[1][0] - 0.5) < 1e-9 and abs(out[1][1] - 0.5) < 1e-9, out[1]
    # The atoms outside the group are the point: they stay put.
    src = _coords_from_xyz(_LINEAR_XYZ)
    for i in (2, 3):
        assert all(abs(out[i][k] - src[i][k]) < 1e-9 for k in range(3)), \
            f"atom {i} moved; only the selected group may move"


def test_modify_translate_recenter_with_no_group_is_one_rigid_move(web_client):
    """User: *"when nothing is selected, it act as if all are selected as
    one rigid move."*  An EMPTY list says the same thing as no key at all
    -- the browser omits `indices` when the selection is empty, but a
    caller that sends `[]` must not get a different structure back."""
    both = []
    for body in ({"structure": _env(_LINEAR_XYZ), "recenter": True},
                 {"structure": _env(_LINEAR_XYZ), "recenter": True,
                  "indices": []}):
        r = web_client.post("/api/modify/translate", json=body).get_json()
        assert r["ok"] is True, r
        both.append(r["text"])
    assert both[0] == both[1], "an empty group must mean the whole structure"


def test_modify_translate_recenter_of_a_group_leaves_the_box(web_client):
    """A group is not the whole structure, so the box does not travel:
    only part of what it contains moved.  That rule is
    ``modify.translate``'s, and Center now reaches it by BEING a
    translate rather than by re-deciding (`archive/2026-09-01-modify-redesign-plan.md`
    § 2.3)."""
    periodicity = {
        "cell": [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 20.0]],
        "engine_offset": [5.0, 5.0, 10.0],
        "axis_kind": ["periodic", "periodic", "transport"],
    }
    r = web_client.post("/api/modify/translate", json={
        "structure": _env_per(_LINEAR_XYZ, periodicity),
        "recenter": True, "indices": [0, 1]}).get_json()
    assert r["ok"] is True, r
    assert r["periodicity"]["engine_offset"] == periodicity["engine_offset"], \
        "centring a GROUP must leave the box where it is"


def test_modify_translate_recenter_of_everything_leaves_the_box(web_client):
    """The other half of the same rule, on the WHOLE-structure path (nothing
    selected, through ``Structure.affine``): the atoms move and the box stays
    where it is -- *"leave the cell alone, moving atoms only moves atoms"*
    (user, 2026-09-25)."""
    import numpy as np
    periodicity = {
        "cell": [[10.0, 0, 0], [0, 10.0, 0], [0, 0, 20.0]],
        "engine_offset": [5.0, 5.0, 10.0],
        "axis_kind": ["periodic", "periodic", "transport"],
    }
    r = web_client.post("/api/modify/translate", json={
        "structure": _env_per(_LINEAR_XYZ, periodicity),
        "recenter": True}).get_json()
    assert r["ok"] is True, r
    assert np.allclose(r["periodicity"]["engine_offset"],
                       periodicity["engine_offset"]), \
        r["periodicity"]["engine_offset"]


def test_modify_translate_offset_shifts_every_atom_by_delta(web_client):
    """``{dx, dy, dz}`` shifts every coordinate by exactly that
    vector and preserves intra-structure distances."""
    r = web_client.post("/api/modify/translate", json={
        "structure": _env(_LINEAR_XYZ),
        "dx": 10.0, "dy": -5.0, "dz": 0.5,
    })
    body = r.get_json()
    assert body["ok"] is True
    out = _coords_from_xyz(body["text"])
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
        "structure": _env(_LINEAR_XYZ),
        "recenter": True,
        "dx": 999.0, "dy": -999.0, "dz": 999.0,
    })
    coords = _coords_from_xyz(r.get_json()["text"])
    cx = sum(c[0] for c in coords) / len(coords)
    # Centroid at origin -- the dx/dy/dz fields were ignored.
    assert abs(cx) < 1e-9, cx


def test_modify_translate_zero_default_is_a_noop(web_client):
    """Omitting dx/dy/dz defaults each to 0.0; the result is byte-
    identical xyz with the same atom count."""
    r = web_client.post("/api/modify/translate", json={"structure": _env(_LINEAR_XYZ)})
    body = r.get_json()
    assert body["ok"] is True
    src = _coords_from_xyz(_LINEAR_XYZ)
    out = _coords_from_xyz(body["text"])
    for s, o in zip(src, out):
        for a, b in zip(s, o):
            assert abs(a - b) < 1e-12


def test_modify_translate_rejects_non_numeric_offset(web_client):
    r = web_client.post("/api/modify/translate", json={
        "structure": _env(_LINEAR_XYZ),
        "dx": "pizza",
    })
    assert r.status_code == 400
    assert "number" in r.get_json()["error"]


def test_modify_translate_preserves_metadata(web_client):
    """Per-atom metadata round-trips through translate (rigid op)."""
    r = web_client.post("/api/modify/translate", json={
        "structure": _env(_LINEAR_XYZ, atom_names=["C1", "C2", "C3", "C4"], residue_ids=[1, 1, 1, 1], residue_names=["MOL", "MOL", "MOL", "MOL"], chain_ids=["A", "A", "A", "A"]),
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
        "structure": _env(_LINEAR_XYZ, atom_names=["C1", "C2", "C3", "C4"], residue_ids=[1, 1, 1, 1], residue_names=["MOL", "MOL", "MOL", "MOL"], chain_ids=["A", "A", "A", "A"]),
    }
    s2 = web_client.post("/api/modify/orient", json={
        "structure": s1["structure"],
        "anchors":       [0, 3],
    }).get_json()
    s3 = web_client.post("/api/modify/rotate", json={
        "structure": s2["structure"],
        "axis": "z", "angle": 45,
    }).get_json()
    assert s3["atom_names"] == ["C1", "C2", "C3", "C4"]
    assert s3["residue_names"] == ["MOL"] * 4


# --------------------------------------------------------------------- #
#  NaN / Inf rejection on /api/modify/* floats                          #
# --------------------------------------------------------------------- #


def test_modify_translate_rejects_nan_offset(web_client):
    """A NaN dx must not propagate through to the structure -- the
    boundary helper ``_shared.finite_float`` rejects non-finite
    values."""
    # THE BODY AS RAW TEXT: the app's JSON provider writes a NaN as null
    # (`persist.json_safe`), and the test client encodes `json=` with it --
    # a NaN literal on the wire is what another client can still send.
    import json as _json
    r = web_client.post("/api/modify/translate", data=_json.dumps({
        "structure": _env(_LINEAR_XYZ), "dx": float("nan"), "dy": 0.0, "dz": 0.0,
    }), content_type="application/json")
    assert r.status_code == 400
    assert "finite" in r.get_json()["error"]


def test_api_build_schema_returns_siesta_schema(web_client):
    """GET /api/build/schema/siesta returns the schema built from the
    CATALOGUE (`web/form-schema.md` § 1).  The wire shape is
    ``{"ok": True, "schema": {...}}``; the schema's id_prefix field is the
    canonical "p" used by the form-field IDs."""
    r = web_client.get("/api/build/schema/siesta?calculation=optimization")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    sch = body["schema"]
    assert sch["config"] == "siesta"
    assert sch["id_prefix"] == "p"
    # The first panel is `system` -- § 6.2's reading order starts there -- and
    # it carries SystemLabel.
    assert sch["sections"][0]["name"] == "system"
    sysl = next(f for s in sch["sections"] for f in s["fields"]
                if f["name"] == "system_label")
    assert sysl["id"] == "p-system-label"


def test_api_build_schema_rejects_unknown_engine(web_client):
    """An unknown engine name surfaces as 404 with a structured
    error so the UI doesn't silently render an empty form."""
    r = web_client.get("/api/build/schema/cp2k")
    assert r.status_code == 404
    body = r.get_json()
    assert body["ok"] is False
    assert "cp2k" in body["error"].lower()


@pytest.mark.parametrize("engine,prefix", [("siesta", "p"), ("pyscf", "py")])
def test_the_form_reads_in_the_categories_shared_by_both_engines(engine, prefix):
    """The sections of a form are the shared categories, in their reading
    order, so both engines show the same headings (`engines/template.md`
    § 6.2) -- whatever they hold."""
    from molbuilder.web.blueprints._shared import catalogue_to_form_schema
    sch = catalogue_to_form_schema(engine, prefix, "optimization")
    assert [s["name"] for s in sch["sections"]] == [
        "system", "method", "accuracy", "convergence", "procedure"]


def _flatten_schema_fields(sch):
    return [f for s in sch["sections"] for f in s["fields"]]


@pytest.mark.parametrize("engine,calculation", [
    ("siesta", "optimization"), ("siesta", "vibration"),
    ("siesta", "transport"),
    ("pyscf", "optimization"), ("pyscf", "vibration"),
])
def test_engine_key_present_on_every_form_field(engine, calculation):
    """Every item a form renders MUST carry an ``engine_key``
    (`web/form-schema.md` § 1a, rule 4).  Without it the UI's
    source-of-truth badge is silently missing for that field.

    EVERY CALCULATION, not the default one: the optimization form alone
    leaves out the transport-only and vibration-only items (W54 S12)."""
    from molbuilder.web.blueprints._shared import catalogue_to_form_schema
    sch = catalogue_to_form_schema(engine, "p", calculation=calculation)
    fields = _flatten_schema_fields(sch)
    assert fields, f"no {engine} {calculation} form was built at all"
    missing = [f["name"] for f in fields if "engine_key" not in f]
    assert not missing, (
        f"{engine} {calculation} fields without engine_key (would render no "
        f"keyword badge in the form): {missing}"
    )


def test_engine_key_marks_molbuilder_only_fields_with_paren_prefix():
    """molbuilder-only fields (preprocessing / wrapper / filename
    knobs that don't reach the engine) MUST have engine_key
    starting with ``(molbuilder`` so the JS engineKeyBadge() picks
    the dashed-border italic visual variant.  Without this the
    user might search the SIESTA / PySCF manual for a keyword
    molbuilder invented (e.g. verbose_comments)."""
    from molbuilder.web.blueprints._shared import catalogue_to_form_schema
    from molbuilder import template as T

    # WHICH fields are molbuilder-only is DERIVED, not curated.  A `kind`
    # of `wrapper` or `produce` IS the
    # statement "this never becomes an engine keyword" (template.md § 6),
    # so ask the catalogue and the list cannot go stale.
    items = {i.name: i for i in T.read_template(T.load_catalogue()).items}
    for engine, prefix in (("siesta", "p"), ("pyscf", "py")):
        sch = catalogue_to_form_schema(engine, prefix, "optimization")
        checked = 0
        for f in _flatten_schema_fields(sch):
            item = items.get(f["name"])
            if item is None or item.kind not in ("wrapper", "produce"):
                continue
            checked += 1
            assert f["engine_key"].startswith("(molbuilder"), (
                f"{engine}:{f['name']}: kind={item.kind!r} never reaches the "
                f"deck, but engine_key={f['engine_key']!r} does not start "
                f"with ``(molbuilder`` -- the UI cannot dim a badge that "
                f"looks like a real keyword, and a user will search the "
                f"manual for a word molbuilder invented")
        assert checked >= 3, (
            f"{engine}: only {checked} molbuilder-only field(s) on the form -- "
            f"too few for this to be proving anything")


def test_a_siesta_keyword_badge_leads_with_the_keyword_the_deck_writes():
    """The badge (`engine_key`) is what a person looks up in the manual, and
    the `anchor` is what the deck writes: for a SIESTA `engine` item the first
    is led by the second (`engines/template.md`, *The shape of `engine_key`*).

    THE SILENT FAILURE: a typo in a badge, which the deck never reads, sends a
    person to the manual for a keyword that does not exist."""
    from molbuilder import template as T

    def leading(key):
        words = key.split(" (SIESTA)")[0].split()
        return " ".join(words[:2]) if words[:1] == ["%block"] else words[0]

    items = [it for it in T.select(T.catalogue(), engine="siesta", kind="engine")
             if it.anchor and not it.engine_key.startswith("(molbuilder")]
    assert items
    wrong = [(it.name, it.anchor, it.engine_key) for it in items
             if leading(it.engine_key) != it.anchor]
    assert not wrong, wrong
