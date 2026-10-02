"""Flask tests for the build/describe surface of the molbuilder web UI.

Covers the build blueprint's doors (molecule / load / preflight /
schema), the structure-optimization page's markup contract with its
viewer.js, and the form-schema plumbing.  NOT every endpoint: spectra,
watch, checkpoint, projects and transport are tested in their own
files.  Skipped cleanly if Flask isn't installed.
"""

from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from support.envelope import (from_xyz as _env,
                             from_xyz_with_periodicity as _env_per)


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
        # The two engine panels are each a schema-driven form container.
        # ``id="generate-pyscf"`` stood here until 2026-08-15: the tab
        # generated the script itself, so the Generate button was the
        # thing that proved the PySCF panel was wired.  The tab now
        # COLLECTS PARAMETERS and hands them on -- *the browser describes;
        # a machine's own facts render* (`web/task-setup.md` § 1) -- so the
        # container is what proves it, and asserting a button that is
        # deliberately gone would pin the retired shape.
        'id="pyscf-form-container"',
        'id="siesta-form-container"',
    ):
        assert needle in body, f"missing {needle!r} in index.html"


def test_the_tab_neither_generates_nor_saves(web_client):
    """The tab collects parameters; it does not produce artefacts.

    `web/task-setup.md` § 1 — *the browser describes; a machine's own
    facts render*.  A deck carries values that depend on how it will be
    launched, so a browser that "finished" one would be guessing -- the
    parameter tab hands over, and a deck is only ever rendered from a
    machine's record, by the one prep entry.  This
    is the guard on that: the buttons are not merely unwired, they are
    absent, and a future edit that re-adds one fails here rather than
    quietly reintroducing the split.
    """
    r = web_client.get("/structure-optimization")
    assert r.status_code == 200
    body = r.data.decode()
    for gone in ('id="generate-fdf"', 'id="generate-pyscf"',
                 'id="save-fdf"', 'id="save-pyscf"',
                 'id="dl-fdf"', 'id="dl-pyscf"',
                 'id="fdf-output"', 'id="pyscf-output"',
                 'id="p-stage-preset"'):
        assert gone not in body, f"{gone} is back in index.html"


# test_build_load_source_mode_toggle_present + test_viewer_js_applies_source_mode
# retired 2026-06-08 (task #295) with the Build/Load form.  The new
# load surface is ``#load-from-sidebar-btn`` — pinned by
# ``test_index_page_loads`` above and the page-boot smoke test in
# tests/test_pages_no_js_errors.py.


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
    from molbuilder.builders.backends import available_backends
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


@pytest.fixture
def peptide_xyz(web_client):
    """xyz string of an ARNDC peptide via the build endpoint."""
    r = web_client.post("/api/build/molecule",
                        json={"kind": "peptide", "input": "ARNDC"})
    return r.get_json()["xyz"]


# --------------------------------------------------------------------- #
#  Pattern-B: regions reach Optimization Generate but aren't            #
#  consumed by SIESTA / PySCF — surface as an INFO so the user can      #
#  re-direct to Transport if that was the intent.  Task #303.           #
# --------------------------------------------------------------------- #


#: The region map the Pattern-B tests below hand to the generate endpoints.
#: Named once so the on-disk sidecar and the request body cannot drift.
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
        "engine": "siesta",
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
    earlier ``ok: True`` + 200, so the UI's ``!body.ok`` gate
    renders issues uniformly.  (It compared this to
    ``/api/build/fdf``'s parse-failure shape until that route was
    deleted on 2026-08-17; the shape it describes is this one.)  Asserting the new shape so the test stays a
    contract pin and not stale documentation.
    """
    r = web_client.post("/api/build/preflight", json={
        "structure": _env(peptide_xyz),
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


def test_a_comma_string_coerces_for_every_sequence_shape():
    """Four sequence shapes reach this function and until 2026-08-25 only
    three of them parsed the text a person types.

    ``Sequence[str]`` (``species_order``), ``Sequence[int]`` and
    ``Sequence[float]`` each split a comma string.  The TUPLE branch beside
    them did not -- it read ``if not isinstance(value, (list, tuple)):
    return value`` and handed the string straight back, while the
    docstring above it claimed tuples "fall through to per-element int
    coercion".  So a POST carrying ``kgrid: "4,4,1"`` stored a ``str`` in a
    field declaring ``Tuple[int, int, int]``, and the range check
    downstream could only report it as a programmer bug.

    Three spellings are accepted because a person writes a k-grid all three
    ways and the field means the same thing each time.  They came from
    ``--kgrid`` (`cli.KGridParam`), which went with `molbuilder fdf` --
    a k-grid is a PARAMETER, said in the description, so there is no
    terminal half of this any more and the browser's parser is the one
    that has to be right.
    """
    import dataclasses
    import typing

    from molbuilder.config.siesta import SiestaConfig
    from molbuilder.web.blueprints._shared import coerce_to_field_type

    hints = typing.get_type_hints(SiestaConfig)
    fields = {f.name: f for f in dataclasses.fields(SiestaConfig)}

    def coerce(name, value):
        return coerce_to_field_type(fields[name], value, hints)

    for spelling in ("4,4,1", "4x4x1", "4 4 1"):
        assert coerce("kgrid", spelling) == (4, 4, 1), spelling
    assert coerce("kgrid", [4, 4, 1]) == (4, 4, 1)
    assert coerce("kgrid_displacement", "0.5,0.5,0.0") == (0.5, 0.5, 0.0)
    assert coerce("species_order", "Au,C,H,S") == ["Au", "C", "H", "S"]
    assert coerce("mesh_cutoff", "300") == 300.0


def test_a_blank_is_not_chosen_and_a_value_that_will_not_read_is_refused_by_name(
        web_client, peptide_xyz):
    """`web/form-schema.md` § 1.1 (plan § 5w K7), at the one door every form
    goes through (`_shared.config_from_params`).

    A BLANK IS NOT CHOSEN, for every field: the field keeps what lies under
    it.  An empty number reached ``float(None)`` and answered a raw
    TypeError naming no field (the M11 review's SS-C16 / SO-N5).  A VALUE
    THAT WILL NOT READ AS ITS TYPE is refused, naming its field -- a
    fractional count among them, never rounded: ``int(4.5)`` gave 4, a k-point
    mesh of four where 4.5 was typed (the K3 review)."""
    def preflight(params):
        return web_client.post("/api/build/preflight", json={
            "structure": _env(peptide_xyz), "engine": "siesta",
            "params": params})

    r = preflight({"mesh_cutoff": None, "max_scf_iter": "",
                   "basis_size": None, "kgrid": None})
    assert r.status_code == 200, r.get_json()
    for name, value in (("kgrid", [4.5, 4, 1]), ("max_scf_iter", "2.5"),
                        ("mesh_cutoff", "abc")):
        r = preflight({name: value})
        assert r.status_code == 400, (name, r.get_json())
        assert name in r.get_json()["error"], r.get_json()


def test_a_kgrid_that_is_not_three_numbers_is_still_refused(web_client,
                                                            peptide_xyz):
    """Parsing the text must not swallow a value that is not one.  The
    length is deliberately NOT checked in the coercion -- ``_validate_kgrid``
    already says it better -- so this pins that the refusal still arrives."""
    r = web_client.post("/api/build/preflight", json={
        "structure": _env(peptide_xyz),
        "engine": "siesta",
        "params": {"kgrid": "4,4"},
    })
    body = r.get_json()
    # The config now PARSES -- that is the fix -- so the envelope is the
    # ordinary one and the complaint arrives where a wrong value belongs:
    # an error Issue naming the field, rather than a 400 naming a cast.
    errs = [i for i in body["issues"]
            if i["severity"] == "error" and i["where"] == "config.kgrid"]
    assert errs, body["issues"]
    assert "3-tuple" in errs[0]["message"], errs[0]


# --------------------------------------------------------------------- #
#  R6: watch upload temp filenames must be collision-safe across       #
#  same-second concurrent uploads (mkstemp atomically reserves a       #
#  unique inode).                                                       #
# --------------------------------------------------------------------- #


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


# `test_load_multipart` STOOD HERE and is retired (2026-09-07).  Its whole
# subject was `/api/build/load`'s multipart branch, which is deleted: nothing
# ever posted a `FormData` to that route -- the only one in the entire front
# end targets `/api/files/upload` -- so the branch was reachable from this
# test and from nowhere else.  A test is the last thing that should keep a
# door open.


def test_load_empty_returns_error(web_client):
    r = web_client.post("/api/build/load", json={"text": ""})
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
        # THE PAGE'S OWNER, and the only modify/ module with a <script> tag.
        # `viewer.js` and `periodicity.js` are IMPORTED by it -- they used to
        # load before the file that mounts, so nothing could have handed them a
        # viewer even if one had existed (molview.md § 8 — making and tearing down a viewer).
        # Asserting a tag for viewer.js pinned that broken load order.
        "modify/selection-bootstrap.js",
        # Scaffolding the JS targets by id.  Post-Track-B the template
        # exposes only the EMPTY host (#molview-host); molview.mount
        # builds the whole fused card (the .viewer div, the selection
        # panel, the View-menu knobs) into it client-side, so those are
        # NOT in the static HTML.
        'id="molview-host"',
        # All five edit ops are wired by the JS.
        'id="delete-apply"',
        'id="add-apply"',
        'id="orient-apply"',
        'id="rotate-apply"',
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
        # JSON, not multipart: the multipart branch of this route was deleted
        # 2026-09-07 with no caller, ever -- the only FormData in the front end
        # targets /api/files/upload.  These three tests are about the RESPONSE
        # SHAPE, so they moved to the transport a caller actually uses.
        json={"text": xyz, "filename": "h2o.xyz"},
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
        # JSON, not multipart: the multipart branch of this route was deleted
        # 2026-09-07 with no caller, ever -- the only FormData in the front end
        # targets /api/files/upload.  These three tests are about the RESPONSE
        # SHAPE, so they moved to the transport a caller actually uses.
        json={"text": xyz, "filename": "h2o.xyz"},
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


# --------------------------------------------------------------------- #
#  Atom-count-changing ops emit NO selection_remap (retired)           #
#  (web/molview.md § 11 -- the client clears the selection)      #
# --------------------------------------------------------------------- #


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


# `test_modify_calibrate_moves_atoms_into_the_cell` RETIRED 2026-09-25, with
# the route (`model/structure-periodicity.md` § 6.0: calibrate is retired).


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
    assert body["elements"] == ["O"]


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
    assert body["elements"][-1] == "H"
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
    assert body["elements"][-1] == "S"
    # The offset IS the position -- read it back off the xyz the route returns.
    x, y, z = [float(v) for v in body["xyz"].splitlines()[-1].split()[1:4]]
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
#  M3 UI scaffolding lives in modify.html / static/modify/viewer.js.    #
# --------------------------------------------------------------------- #


def test_modify_page_has_m3_edit_controls(web_client):
    """The Edit panel must expose the M3 op controls (delete button,
    add-atom element input, three offset sliders + live distance
    readout)."""
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
    ):
        assert needle in body, f"missing {needle!r} in /modify HTML"


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
    coords = _coords_from_xyz(body["xyz"])
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
    (1.5, 1.5, 0).  (The docstring said (1,1,1)..(4,4,4) until
    2026-08-30 -- a fixture it has not matched for some time, and
    nothing failed, because the assertion only reads the centroid.)"""
    r = web_client.post("/api/modify/translate", json={
        "structure": _env(_LINEAR_XYZ),
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
    out = _coords_from_xyz(body["xyz"])
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
        both.append(r["xyz"])
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
    (user, 2026-09-25).  Until then this path took the box along."""
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
        "structure": _env(_LINEAR_XYZ),
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
    r = web_client.post("/api/modify/translate", json={"structure": _env(_LINEAR_XYZ)})
    body = r.get_json()
    assert body["ok"] is True
    src = _coords_from_xyz(_LINEAR_XYZ)
    out = _coords_from_xyz(body["xyz"])
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


def test_modify_page_has_m4_orient_rotate_controls(web_client):
    """The Edit panel must expose the M4 orient + rotate controls
    (anchor-pair readout, axis radios, angle slider, Apply for both
    ops)."""
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
    ):
        assert needle in body, f"missing {needle!r} in /modify HTML"


# --------------------------------------------------------------------- #
#  Send-to-Build handoff                                                #
# --------------------------------------------------------------------- #
#
# The electrode endpoints that shared this banner are gone: the pair went
# with the Junction panel, and the per-side one went on 2026-09-01 once
# `/api/modify/slab` had replaced it (`archive/2026-09-01-modify-redesign-plan.md` 3.4a).
# Their three tests went with them -- what they pinned (a positive contact
# distance, a `+z`/`-z` side) were that route's arguments, and no other
# route takes either.


# --------------------------------------------------------------------- #
#  Basename validation (job-layout v1)                                  #
# --------------------------------------------------------------------- #


# --------------------------------------------------------------------- #
#  NaN / Inf rejection on /api/modify/* floats                          #
# --------------------------------------------------------------------- #


def test_modify_translate_rejects_nan_offset(web_client):
    """A NaN dx must not propagate through to the structure -- the
    boundary helper ``_shared.finite_float`` rejects non-finite
    values."""
    r = web_client.post("/api/modify/translate", json={
        "structure": _env(_LINEAR_XYZ), "dx": float("nan"), "dy": 0.0, "dz": 0.0,
    })
    assert r.status_code == 400
    assert "finite" in r.get_json()["error"]


# (The tests of `dataclass_to_form_schema` -- eight, on a hand-written
#  fixture dataclass -- were retired with the function, 2026-10-02, M5 step
#  3: no route had called it since the forms moved to the catalogue.)


def test_siesta_form_schema_matches_documented_layout():
    """The layout the STRUCTURE-OPTIMIZATION TAB actually renders.

    Repointed 2026-08-15 from ``dataclass_to_form_schema`` to
    ``catalogue_to_form_schema``.  The tab is served by the catalogue
    builder (``build.py`` /api/build/schema); the dataclass builder was
    deleted on 2026-10-02.  Asked the old builder, this test could not see
    the tab:
    it passed UNCHANGED on the day ``restart`` and ``continue_retries``
    left the form, which is precisely the regression a layout test exists
    to catch.

    THE SECTIONS ARE THE SIX SHARED CATEGORIES, not per-engine fieldset
    names.  ``category`` replaced ``section`` for exactly this reason
    (`engines/template.md` § 6.2): ``section`` was free text chosen per
    engine, so SIESTA's *"Basis & grid"* and PySCF's *"Method"* were
    unrelated words and no surface could group across engines.  The six
    are shared, in `template.CATEGORIES` order, so both engines show the
    same inner headings.

    The counts are a fact about the catalogue, and they move when a
    parameter is added or re-filed -- which is the point: a diff here is
    a prompt to check the form still reads well, not a nuisance.
    """
    from molbuilder.web.blueprints._shared import catalogue_to_form_schema
    sch = catalogue_to_form_schema("siesta", "p")
    # The catalogue builder names the ENGINE, not the translator class --
    # which is the point: the form is built from the catalogue, and the
    # config class is a translator on the way out to that engine.
    assert sch["config"] == "siesta"
    assert sch["id_prefix"] == "p"

    got = [(s["name"], len(s["fields"])) for s in sch["sections"]]
    assert got == [
        ("system",      6),
        ("method",      6),
        ("accuracy",    7),
        ("convergence", 4),
        # 16 -> 15 on 2026-09-25: `wrap_into_cell` retired.  Where the atoms
        # sit is one rule, the engine offset (`model/structure-periodicity.md`
        # § 6.0), and a per-atom wrap can cut a device at its widest gap.
        # 15 -> 13 on 2026-09-29: `write_forces` and `write_coor_step` are
        # fixed by the rung on every SIESTA kind -- not a choice, so not a
        # control (`engines/template.md` § 6.4).  13 -> 15 on 2026-09-30:
        # shown again, read-only at that answer (`locked`, § 6.6 obligation
        # 3; plan § 5w K7) -- still never a control.
        ("procedure",  15),
        # 7 -> 3 on 2026-08-15: mpi_np, omp_threads, max_memory_mb and
        # use_gpu moved to the staging surface.  They are bench axes
        # measured on the machine, not parameters typed beside the physics.
        ("execution",   3),
    ], got

    # `staging` items are declared parameters this surface does not ask
    # (user, 2026-08-15).  `restart` and `continue_retries` are the two
    # today; a regression that put them back would land here.
    ids = [f["id"] for s in sch["sections"] for f in s["fields"]]
    assert not [i for i in ids if "restart" in i or "retr" in i], ids


def test_api_build_schema_returns_siesta_schema(web_client):
    """GET /api/build/schema/siesta returns the schema built from the
    CATALOGUE (`web/form-schema.md` § 1).  The wire shape is
    ``{"ok": True, "schema": {...}}``; the schema's id_prefix field is the
    canonical "p" used by the form-field IDs."""
    r = web_client.get("/api/build/schema/siesta")
    assert r.status_code == 200
    body = r.get_json()
    assert body["ok"] is True
    sch = body["schema"]
    assert sch["config"] == "siesta"
    assert sch["id_prefix"] == "p"
    # The first panel is `system` -- § 6.2's reading order starts there -- and
    # it carries SystemLabel, whose id the compatibility engine in viewer.js
    # references by string.
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


def test_pyscf_form_schema_matches_documented_layout():
    """PySCF's half of the same contract -- the six shared categories,
    from the same builder the tab uses.  See the SIESTA test above for
    why this moved off ``dataclass_to_form_schema``."""
    from molbuilder.web.blueprints._shared import catalogue_to_form_schema
    sch = catalogue_to_form_schema("pyscf", "py")
    assert sch["config"] == "pyscf"
    assert sch["id_prefix"] == "py"

    got = [(s["name"], len(s["fields"])) for s in sch["sections"]]
    assert got == [
        # `system` gained job_name (it leads the Setup card now, the
        # same treatment system_label got); `execution` is gone entirely --
        # threads and use_gpu were its only members and both are bench axes.
        # +1 on 2026-09-28: the electronic state's `spin_treatment` beside
        # the count that replaced `spin` (`unpaired_electrons`).
        ("system",      7),
        ("method",      8),
        # +5 on 2026-08-17 with P1's stage ladder: the FIVE geomeTRIC criteria
        # (`geom_gmax`/`_grms`/`_dmax`/`_drms`/`_etol`) are one family
        # (`tuning.md` § 2.4) and they are THRESHOLDS -- *what answer you will
        # accept* -- which is what `accuracy` means in § 6.2, and what puts
        # `scf_conv_tol` there too.  `_dmax`/`_drms` landed under `procedure`
        # and split the family across two panels until this was found.
        ("accuracy",    8),
        # `convergence` is *how do I reach it when it fights* -- the § 7.2
        # escalation ladder (diis_space, level_shift, damp, soscf) and the SCF
        # iteration budget.  Loosening a threshold to "fix" a stubborn run is
        # the substitution § 6.2 splits these two categories to discourage.
        ("convergence", 6),
        # `geom_max_steps` is the OUTER geometry budget, so it sits with
        # `relax_steps`, the SIESTA knob `tuning.md` § 3.1 pairs it with --
        # not with the inner SCF budget.  It was under `convergence`, which
        # put one cross-engine concept on two different panels.
        # 16 -> 13 at P3 (2026-08-21): the in-deck compute_frequencies
        # trio left the OPTIMIZATION form -- the item retired outright,
        # and temperature_K / pressure_atm re-homed to the vibration
        # kind (calculations = ["vibration"]).
        # 13 -> 12 on 2026-09-29: the one-choice `optimizer` retired with
        # berny (`engines/pyscf.md` § 3).
        ("procedure",  12),
    ], got

    # The stage table is NOT here.  PySCFConfig still has a `stages`
    # field -- the ladder is real -- but the staging surface owns it, so
    # this form never asks.  Three tests asserting the opposite were
    # retired the same day (tests/test_pyscf_stages.py).
    ids = [f["id"] for s in sch["sections"] for f in s["fields"]]
    assert not [i for i in ids if "stage" in i], ids


def _flatten_schema_fields(sch):
    return [f for s in sch["sections"] for f in s["fields"]]


def test_engine_key_present_on_every_siesta_form_field():
    """Every SIESTA item the form renders MUST carry an
    ``engine_key``.  Without it the UI's
    source-of-truth badge is silently missing for that field."""
    from molbuilder.web.blueprints._shared import catalogue_to_form_schema
    from molbuilder.config.siesta import SiestaConfig
    sch = catalogue_to_form_schema("siesta", "p")
    missing = [f["name"] for f in _flatten_schema_fields(sch)
               if "engine_key" not in f]
    assert not missing, (
        f"SiestaConfig fields without engine_key (would render no "
        f"keyword badge in the form): {missing}"
    )


def test_engine_key_present_on_every_pyscf_form_field():
    """Same contract for PySCF."""
    from molbuilder.web.blueprints._shared import catalogue_to_form_schema
    from molbuilder.config.pyscf import PySCFConfig
    sch = catalogue_to_form_schema("pyscf", "py")
    missing = [f["name"] for f in _flatten_schema_fields(sch)
               if "engine_key" not in f]
    assert not missing, (
        f"PySCFConfig fields without engine_key: {missing}"
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

    # WHICH fields are molbuilder-only is DERIVED, not curated.  The
    # curated set here listed six names and three of them -- mpi_np,
    # omp_threads, max_memory_mb -- left this form on 2026-08-15 for the
    # staging surface, so the test failed for a move it should not have
    # had an opinion about.  A `kind` of `wrapper` or `produce` IS the
    # statement "this never becomes an engine keyword" (template.md § 6),
    # so ask the catalogue and the list cannot go stale.
    items = {i.name: i for i in T.read_template(T.load_catalogue()).items}
    for engine, prefix in (("siesta", "p"), ("pyscf", "py")):
        sch = catalogue_to_form_schema(engine, prefix)
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


def test_engine_key_pins_load_bearing_siesta_keywords():
    """Spot-check that the SIESTA fields whose 1:1 keyword mapping
    other parts of the codebase rely on (or the user manually
    cross-references against the SIESTA manual) carry the exact
    expected engine_key text.  If any of these changes, downstream
    text searches + the .fdf grep workflow break."""
    from molbuilder.web.blueprints._shared import catalogue_to_form_schema
    from molbuilder.config.siesta import SiestaConfig
    sch = catalogue_to_form_schema("siesta", "p")
    fields_by_name = {f["name"]: f for f in _flatten_schema_fields(sch)}
    expected = {
        # `Spin`, not `SpinPolarized`: the manual deprecates all three old
        # spin booleans in favour of the one four-valued keyword.  A MERGED
        # item since 2026-09-28 (the electronic state), so the badge names
        # both engines' spelling.
        "spin_treatment": "Spin (SIESTA) | the SCF class's R / RO / U (PySCF)",
        # The "two keys, either alone is silently ignored" rule depends on
        # the badge naming BOTH of SIESTA's.
        "unpaired_electrons": ("Spin.Fix + Spin.Total (SIESTA) | "
                               "gto.M(spin=...) (PySCF)"),
        # Documented user-facing keywords -- ``MeshCutoff`` /
        # ``PAO.BasisSize`` are SIESTA's own names, and the help text
        # references them.
        "mesh_cutoff":    "MeshCutoff",
        "basis_size":     "PAO.BasisSize",
        # MERGED with PySCF's `charge` 2026-08-19: one question, one
        # item, and the spelling names both engines because neither is
        # THE answer (`template.md` § 6.3).
        "net_charge":     "NetCharge (SIESTA) | gto.M(charge=...) (PySCF)",
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
    method= is a CLASS switch, not a string kwarg -- and the class is
    composed from the method and the spin treatment since 2026-09-28, so
    the two badges between them say which class (the treatment's is the
    merged item's, pinned in the SIESTA test)."""
    from molbuilder.web.blueprints._shared import catalogue_to_form_schema
    from molbuilder.config.pyscf import PySCFConfig
    sch = catalogue_to_form_schema("pyscf", "py")
    fields_by_name = {f["name"]: f for f in _flatten_schema_fields(sch)}
    # (The merged items -- the charge and the spin -- are ONE declaration
    # for both engines, pinned once, in the SIESTA test above.)
    expected = {
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
    # `method` says which theory, and the class is composed from it and the
    # spin treatment (2026-09-28): the method's badge names the class
    # MODULE it picks, the treatment's the R / RO / U it contributes -- so
    # the person knows they are picking a class, not a string.
    method_key = fields_by_name["method"]["engine_key"]
    assert "dft.<class>" in method_key and "scf.<class>" in method_key, \
        method_key
