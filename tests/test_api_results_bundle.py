"""L2 + L3 tests for ``POST /api/results/bundle`` (PR-E, task #492).

Endpoint contract: ``docs/protocols/bundle-contract.md`` § 5.2 + § 7.

L2 = the endpoint shape + error envelope.
L3 = end-to-end: write a fake run dir, POST the endpoint, assert the
     materialised .xyz + .molstruct.json pair lands at the target and
     the response carries the right summary fields.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from molbuilder import diagnostics


_BOHR = 0.5291772108


def _set_picker_root(monkeypatch, tmp_path):
    """Make ``tmp_path`` the only allowed picker root so the
    endpoint's ``_resolve_within_roots`` accepts paths inside it."""
    caps = diagnostics.Capabilities(
        runtime_config={}, conda_binary=None, conda_envs=frozenset(),
    )
    cls = type(caps)
    monkeypatch.setattr(
        cls, "file_picker_roots",
        lambda self: ((tmp_path.resolve(), "projects"),),
    )
    diagnostics.set_capabilities(caps)


def _atom_md_block(*, regions, frozen, n_atoms_total):
    """Emit an ATOM-METADATA block via the canonical contract emitter."""
    from molbuilder import script_contract as sc
    block = sc.emit_atom_metadata(
        regions=regions or {},
        frozen_atoms=frozen or [],
        n_atoms_total=n_atoms_total,
    )
    return ("" if block is None else "\n" + block + "\n")


def _siesta_run_dir(tmp_path):
    """Build a minimal SIESTA run dir under ``tmp_path``: a .fdf
    with ATOM-METADATA + a SystemLabel-keyed .XV."""
    run_dir = tmp_path / "src-run"
    run_dir.mkdir()
    (run_dir / "h2.fdf").write_text(
        "SystemLabel h2\n"
        "%block ChemicalSpeciesLabel\n"
        "    1    1    H\n"
        "%endblock ChemicalSpeciesLabel\n"
        "AtomicCoordinatesFormat Ang\n"
        "%block AtomicCoordinatesAndAtomicSpecies\n"
        "    0.000   0.000   0.000   1\n"
        "    0.740   0.000   0.000   1\n"
        "%endblock AtomicCoordinatesAndAtomicSpecies\n"
        + _atom_md_block(
            regions={"L-electrode": [0]},
            frozen=[1],
            n_atoms_total=2,
        )
    )
    # SIESTA writes ``<SystemLabel>.XV``; minimal valid .XV for H2.
    (run_dir / "h2.XV").write_text(
        "  10.0   0.0   0.0   0.0 0.0 0.0\n"
        "   0.0  10.0   0.0   0.0 0.0 0.0\n"
        "   0.0   0.0  10.0   0.0 0.0 0.0\n"
        "  2\n"
        "  1   1   0.000   0.000   0.000   0.0 0.0 0.0\n"
        "  1   1   1.500   0.000   0.000   0.0 0.0 0.0\n"
    )
    return run_dir


# --------------------------------------------------------------------- #
#  L2: input-shape rejections                                            #
# --------------------------------------------------------------------- #


def test_bundle_endpoint_rejects_missing_run_dir(web_client):
    r = web_client.post("/api/results/bundle", json={
        "target_dir": "/tmp/anywhere",
        "stem": "h2",
    })
    assert r.status_code == 400
    body = r.get_json()
    assert body["ok"] is False
    assert "run_dir" in body["error"]


def test_bundle_endpoint_rejects_missing_target_dir(web_client):
    r = web_client.post("/api/results/bundle", json={
        "run_dir": "/tmp/anywhere",
        "stem": "h2",
    })
    assert r.status_code == 400
    body = r.get_json()
    assert body["ok"] is False
    assert "target_dir" in body["error"]


def test_bundle_endpoint_rejects_missing_stem(web_client):
    r = web_client.post("/api/results/bundle", json={
        "run_dir": "/tmp/x",
        "target_dir": "/tmp/y",
    })
    assert r.status_code == 400
    body = r.get_json()
    assert body["ok"] is False
    assert "stem" in body["error"]


def test_bundle_endpoint_rejects_bad_stem_charset(web_client, tmp_path, monkeypatch):
    """Stem MUST start with [A-Za-z0-9] and contain only
    [A-Za-z0-9._-]; the endpoint refuses whitespace, shell-special,
    leading dot, leading dash, all-dots stems, and NUL."""
    _set_picker_root(monkeypatch, tmp_path)
    run_dir = _siesta_run_dir(tmp_path)
    for bad in (
        "h2 bundle",     # whitespace
        "h2/bundle",     # slash
        "h2;rm -rf /",   # shell-special
        "h2$VAR",        # shell-var
        ".hidden",       # leading dot -> dotfile, sidebar filters out
        ".",             # bare dot -> ..xyz
        "..",            # all dots -> ...xyz
        "...",           # all dots -> ....xyz
        "-flag",         # leading dash -> CLI consumer reads as flag
        "h2\x00null",    # NUL byte -- filename-injection class
        "ñame",          # non-ASCII -- NFKC spoofing class per memory
    ):
        r = web_client.post("/api/results/bundle", json={
            "run_dir":    str(run_dir),
            "target_dir": str(tmp_path / "dst"),
            "stem":       bad,
        })
        assert r.status_code == 400, bad
        body = r.get_json()
        assert body["ok"] is False
        assert "stem" in body["error"]


def test_bundle_endpoint_accepts_canonical_stems(web_client, tmp_path, monkeypatch):
    """Boundary check on the regex.  Stems like ``h2``, ``h2-bundle``,
    ``a_b.c``, single-char ``h``, and exact-64-char are accepted."""
    _set_picker_root(monkeypatch, tmp_path)
    run_dir = _siesta_run_dir(tmp_path)
    for good in ("h", "h2", "h2-bundle", "a_b.c", "x" * 64):
        r = web_client.post("/api/results/bundle", json={
            "run_dir":    str(run_dir),
            "target_dir": str(tmp_path / good),
            "stem":       good,
        })
        # We don't assert on the bundle content here; just that the
        # stem is accepted (so the endpoint reaches assemble + write).
        assert r.status_code == 200, (good, r.get_json())


def test_bundle_endpoint_rejects_non_dict_body(web_client, tmp_path, monkeypatch):
    """Audit BLOCKER: ``request.get_json(silent=True) or {}`` lets a
    JSON list / number / string sail past the falsy check (they're
    all truthy non-dicts), then ``body.get(...)`` raised AttributeError
    -> uncaught 500.  Tighten to require an object."""
    _set_picker_root(monkeypatch, tmp_path)
    for bad in ([1, 2, 3], "just a string", 42, True):
        r = web_client.post(
            "/api/results/bundle",
            data=__import__("json").dumps(bad),
            content_type="application/json",
        )
        assert r.status_code == 400, bad
        body = r.get_json()
        assert body["ok"] is False
        assert "JSON object" in body["error"]


def test_bundle_endpoint_rejects_target_dir_pointing_at_file(web_client, tmp_path, monkeypatch):
    """If target_dir is an existing FILE (inside picker roots), the
    materialiser would crash inside mkdir.  Pre-check rejects it
    with an actionable 400."""
    _set_picker_root(monkeypatch, tmp_path)
    run_dir = _siesta_run_dir(tmp_path)
    not_a_dir = tmp_path / "this-is-a-file"
    not_a_dir.write_text("not a directory\n")
    r = web_client.post("/api/results/bundle", json={
        "run_dir":    str(run_dir),
        "target_dir": str(not_a_dir),
        "stem":       "h2",
    })
    assert r.status_code == 400
    body = r.get_json()
    assert body["ok"] is False
    assert "not a directory" in body["error"].lower()


def test_bundle_endpoint_rejects_path_outside_picker_roots(web_client, tmp_path, monkeypatch):
    """Picker-roots security boundary per bundle-contract.md § 7.1.
    ``_resolve_within_roots`` rejects ``/etc/passwd`` style paths."""
    _set_picker_root(monkeypatch, tmp_path)
    r = web_client.post("/api/results/bundle", json={
        "run_dir":    "/etc",      # outside the only allowed root
        "target_dir": str(tmp_path / "dst"),
        "stem":       "evil",
    })
    assert r.status_code in (400, 404)
    body = r.get_json()
    assert body["ok"] is False


def test_bundle_endpoint_404s_on_missing_run_dir(web_client, tmp_path, monkeypatch):
    _set_picker_root(monkeypatch, tmp_path)
    r = web_client.post("/api/results/bundle", json={
        "run_dir":    str(tmp_path / "does-not-exist"),
        "target_dir": str(tmp_path / "dst"),
        "stem":       "h2",
    })
    assert r.status_code == 404
    body = r.get_json()
    assert body["ok"] is False


def test_bundle_endpoint_400s_on_empty_run_dir(web_client, tmp_path, monkeypatch):
    """Empty dir -> assembler raises BundleError -> endpoint surfaces 400."""
    _set_picker_root(monkeypatch, tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    r = web_client.post("/api/results/bundle", json={
        "run_dir":    str(empty),
        "target_dir": str(tmp_path / "dst"),
        "stem":       "h2",
    })
    assert r.status_code == 400
    body = r.get_json()
    assert body["ok"] is False
    assert "no engine script" in body["error"].lower()


# --------------------------------------------------------------------- #
#  L3: end-to-end round-trip                                             #
# --------------------------------------------------------------------- #


def test_bundle_endpoint_writes_pair_and_returns_summary(web_client, tmp_path, monkeypatch):
    """The happy path: bundle a SIESTA run dir into a fresh target,
    verify the .xyz + .molstruct.json land + the JSON envelope
    carries the summary fields the UI needs (final_coords_from,
    notes, regions, n_atoms)."""
    _set_picker_root(monkeypatch, tmp_path)
    run_dir = _siesta_run_dir(tmp_path)
    dst_dir = tmp_path / "handoff"
    r = web_client.post("/api/results/bundle", json={
        "run_dir":    str(run_dir),
        "target_dir": str(dst_dir),
        "stem":       "h2-bundle",
    })
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert body["ok"] is True
    assert body["source_engine"]     == "siesta"
    assert body["final_coords_from"] == "xv"
    assert body["n_atoms"]           == 2
    assert body["regions"]           == ["L-electrode"]
    assert body["frozen_atoms_count"] == 1
    # Files land at the claimed paths.
    xyz_path     = Path(body["xyz_path"])
    sidecar_path = Path(body["sidecar_path"])
    assert xyz_path     == dst_dir / "h2-bundle.xyz"
    assert sidecar_path == dst_dir / "h2-bundle.molstruct.json"
    assert xyz_path.exists() and sidecar_path.exists()
    # XYZ provenance comment carries the source basename.
    assert "bundled from h2.fdf" in xyz_path.read_text()
    # Sidecar regions match what the .fdf carried.
    sidecar = json.loads(sidecar_path.read_text())
    assert sidecar["regions"] == {"L-electrode": [0]}
    assert sidecar["frozen_atoms"] == [1]


def test_bundle_endpoint_refuses_overwrite_by_default(web_client, tmp_path, monkeypatch):
    """Second call to the same stem without overwrite=True -> 400
    (mirrors write_bundle_as_handoff's BundleError)."""
    _set_picker_root(monkeypatch, tmp_path)
    run_dir = _siesta_run_dir(tmp_path)
    dst_dir = tmp_path / "handoff"
    # First call succeeds.
    r1 = web_client.post("/api/results/bundle", json={
        "run_dir":    str(run_dir),
        "target_dir": str(dst_dir),
        "stem":       "h2-bundle",
    })
    assert r1.status_code == 200
    # Second call to same stem -> refused.
    r2 = web_client.post("/api/results/bundle", json={
        "run_dir":    str(run_dir),
        "target_dir": str(dst_dir),
        "stem":       "h2-bundle",
    })
    assert r2.status_code == 400
    body = r2.get_json()
    assert body["ok"] is False
    assert "exists" in body["error"].lower()


def test_bundle_endpoint_overwrite_true_replaces(web_client, tmp_path, monkeypatch):
    _set_picker_root(monkeypatch, tmp_path)
    run_dir = _siesta_run_dir(tmp_path)
    dst_dir = tmp_path / "handoff"
    dst_dir.mkdir()
    (dst_dir / "h2-bundle.xyz").write_text("stale placeholder\n")
    (dst_dir / "h2-bundle.molstruct.json").write_text("{}\n")
    r = web_client.post("/api/results/bundle", json={
        "run_dir":    str(run_dir),
        "target_dir": str(dst_dir),
        "stem":       "h2-bundle",
        "overwrite":  True,
    })
    assert r.status_code == 200, r.get_json()
    assert "stale placeholder" not in (dst_dir / "h2-bundle.xyz").read_text()


def test_results_page_renders_bundle_panel(web_client):
    """L2: the /results page includes ``_bundle_handoff_panel.html``
    so the form is on the DOM and the JS module can mount.  Pins the
    template-wiring so a future template refactor that drops the
    include is caught by a test, not by the user."""
    r = web_client.get("/results")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    # Form element + the canonical input names + the submit button.
    assert 'id="bundle-handoff-form"' in html
    assert 'name="run_dir"' in html
    assert 'name="target_dir"' in html
    assert 'name="stem"' in html
    assert 'name="overwrite"' in html
    assert 'bundle-handoff-submit' in html


def test_bundle_handoff_js_uses_correct_sidebar_api(tmp_path):
    """Audit BLOCKER: the prior JS read ``getState().cursor`` and
    called ``ws.refresh(dir)``.  Neither exists on the projects-
    sidebar public surface (``state.js::projects`` exports
    ``getCurrentDir`` / ``getCurrentFile`` + ``navigateTo``; the
    ``refresh()`` it exports takes no dir argument).  Pin the
    correct call sites so a future regression here is caught
    without needing a Playwright run."""
    import re
    from pathlib import Path
    js_raw = Path("molbuilder/web/static/lib/results/bundle-handoff.js").read_text()
    # Strip block + line comments before asserting on call shapes so
    # historical-context mentions ("the prior JS used ws.getState()")
    # don't false-positive the regression check.
    no_block_comments = re.sub(r"/\*.*?\*/", "", js_raw, flags=re.DOTALL)
    js = "\n".join(
        line for line in no_block_comments.splitlines()
        if not line.lstrip().startswith("//")
    )
    # Correct public-API calls present.
    assert "getCurrentDir" in js
    assert "getCurrentFile" in js
    assert "navigateTo" in js
    # Phantom calls absent (matched as actual call sites).
    assert not re.search(r"\bws\.getState\s*\(", js), (
        "bundle-handoff.js calls ws.getState() — the projects-sidebar "
        "public API never exposed that.  Switch to getCurrentDir() + "
        "getCurrentFile().")
    # refresh(<arg>) is the wrong shape; the public refresh() takes
    # only an opts object.  navigateTo is the correct way to move
    # the sidebar to a specific dir.
    assert not re.search(r"\bws\.refresh\s*\(\s*[A-Za-z_]", js), (
        "bundle-handoff.js calls ws.refresh(<some-identifier>) — "
        "refresh() takes no directory argument; it re-lists wherever "
        "the sidebar already points.  Use navigateTo(dir) so the "
        "sidebar moves to and lists the new bundle dir.")


def test_bundle_handoff_default_target_is_subdir(tmp_path):
    """Audit IMPORTANT: pre-fix the JS defaulted target_dir to the
    same path as run_dir.  Bundling into the run dir meant the new
    .molstruct.json sat next to the run's source .fdf, which can
    then shadow the bundle's labels via apply_companion_labels_if_present
    on next load.  The default is now ``<run_dir>/handoff``."""
    from pathlib import Path
    js = Path("molbuilder/web/static/lib/results/bundle-handoff.js").read_text()
    assert "/handoff" in js, (
        "bundle-handoff.js no longer appends '/handoff' to the default "
        "target_dir.  This re-introduces the data-loss trap where a "
        "sibling .fdf shadows the bundle's sidecar.")


def test_bundle_endpoint_carries_xv_unreadable_warn_note(web_client, tmp_path, monkeypatch):
    """When .XV exists but is malformed, the endpoint must surface
    the 'NOT converged geometry' note in the response so the UI can
    show it to the user.  Pins the diagnostic-flow-through fix."""
    _set_picker_root(monkeypatch, tmp_path)
    run_dir = _siesta_run_dir(tmp_path)
    # Replace the good .XV with garbage that read_xv will reject.
    (run_dir / "h2.XV").write_text("not a valid XV\n")
    r = web_client.post("/api/results/bundle", json={
        "run_dir":    str(run_dir),
        "target_dir": str(tmp_path / "handoff"),
        "stem":       "h2-bundle",
    })
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert body["final_coords_from"] == "fdf-initial"
    combined = "\n".join(body["notes"])
    assert "NOT converged geometry" in combined
