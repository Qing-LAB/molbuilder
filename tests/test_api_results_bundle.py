"""L2 + L3 tests for ``POST /api/results/bundle`` (PR-E, task #492).

Endpoint contract: ``docs/execution/handoff-bundle.md`` § 3 (the object) and
§ 6 (the surfaces); the change process is ``docs/execution/job-contracts.md``
§ 7.

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
    """Emit an ATOM-METADATA block via the canonical contract emitter.

    ``frozen`` is written where it belongs -- as the reserved label in the ONE
    label store. The emitter has no second parameter for it."""
    from molbuilder import script_emit as sc
    from molbuilder.structure import FROZEN_LABEL
    labels = dict(regions or {})
    if frozen:
        labels[FROZEN_LABEL] = list(frozen)
    block = sc.emit_atom_metadata(
        regions=labels,
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
    """Picker-roots security boundary per web/web-api.md (the picker-roots sandbox) and handoff-bundle.md § 6.
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


def test_bundle_handoff_js_gates_timer_and_chain_on_settled_flag(tmp_path):
    """Audit follow-up race fix: the timer macrotask could fire
    AFTER fetch resolved but BEFORE the body-parse promise resolved;
    abort() then rejected the body parse with AbortError and the
    user saw 'timed out' for a request that completed.  Pin the
    ``settled`` flag pattern so a future refactor can't drop the
    gate.

    Counts only CODE occurrences -- comments are stripped first so
    the historical-context block at the top doesn't inflate the
    count past the threshold even after a future refactor removes
    every actual guard.  Pre-fix the test counted comment mentions
    toward the 5-line threshold.
    """
    import re
    from pathlib import Path
    js_raw = Path("molbuilder/web/static/lib/results/bundle-handoff.js").read_text()
    no_block = re.sub(r"/\*.*?\*/", "", js_raw, flags=re.DOTALL)
    js = "\n".join(
        line for line in no_block.splitlines()
        if not line.lstrip().startswith("//")
    )
    assert "settled" in js, (
        "bundle-handoff.js no longer carries a ``settled`` flag in "
        "code (excluding comments); the timer/promise race window "
        "re-opens.")
    # Require three concrete patterns: the declaration + at least
    # one write (``settled = true``) + at least two early-return
    # guards (``if (settled) return``).  Each is a distinct site
    # that the race fix needs.  Loose enough to survive ident
    # rename refactors via the ``\bsettled\b`` boundary, strict
    # enough that "settled" appearing in a string literal can't
    # satisfy the test.
    assert re.search(r"\bvar\s+settled\s*=\s*false\b", js), (
        "bundle-handoff.js no longer declares ``var settled = false``; "
        "the race-gate is missing its initial state.")
    assert len(re.findall(r"\bsettled\s*=\s*true\b", js)) >= 1, (
        "bundle-handoff.js no longer assigns ``settled = true`` "
        "anywhere; the race gate never closes.")
    guards = re.findall(r"\bif\s*\(\s*settled\s*\)\s*return\b", js)
    assert len(guards) >= 2, (
        f"bundle-handoff.js has only {len(guards)} ``if (settled) "
        f"return`` guards (need ≥ 2: at least one each on the "
        f"timer / .then / .catch path).  The race-gate may have "
        f"been simplified into a single check, leaving paths "
        f"where the timer can clobber the chain or vice-versa.")


def test_bundle_handoff_js_guards_double_submit(tmp_path):
    """Audit follow-up: Enter-in-input can dispatch a second submit
    event before the browser observes ``btn.disabled``.  Pre-fix two
    concurrent fetches shared the status/spinner/result DOM and
    clobbered each other's render.  Pin the early-return guard."""
    import re
    from pathlib import Path
    js = Path("molbuilder/web/static/lib/results/bundle-handoff.js").read_text()
    # Strip block comments so the historical-context mention doesn't
    # false-positive.
    cleaned = re.sub(r"/\*.*?\*/", "", js, flags=re.DOTALL)
    cleaned = "\n".join(
        line for line in cleaned.splitlines()
        if not line.lstrip().startswith("//")
    )
    # The guard reads ``btn.disabled`` early in the submit handler.
    # Match the call shape (identifier . identifier) so a future
    # rename to ``submitBtn.disabled`` is still picked up.
    assert re.search(r"\bif\s*\(\s*\w+\.disabled\s*\)\s*return", cleaned), (
        "bundle-handoff.js no longer guards the submit handler with "
        "an early-return on btn.disabled.  Enter-in-input can fire "
        "two submit events before the browser disables the button; "
        "without this guard two in-flight fetches clobber each "
        "other's render.")


def test_bundle_handoff_js_has_fetch_timeout(tmp_path):
    """Audit follow-up deferral #1: pre-fix the fetch had no
    timeout, so a hung server left the button disabled forever.
    Pin both the AbortController wiring and the timeout constant."""
    from pathlib import Path
    js = Path("molbuilder/web/static/lib/results/bundle-handoff.js").read_text()
    assert "AbortController" in js, (
        "bundle-handoff.js no longer uses AbortController; the fetch "
        "can hang indefinitely without surfacing a timeout.")
    assert "FETCH_TIMEOUT_MS" in js, (
        "bundle-handoff.js no longer defines a FETCH_TIMEOUT_MS "
        "constant; timeout regression risk.")
    assert "timedOut" in js, (
        "bundle-handoff.js no longer distinguishes timeout from other "
        "AbortError sources; the error message would be ambiguous.")


def test_bundle_handoff_js_handles_non_json_response(tmp_path):
    """Audit follow-up deferral #2: a Flask default 500 returns HTML,
    not JSON.  Pre-fix r.json() rejected with SyntaxError and the
    user saw 'Unexpected token <'.  Pin the content-type guard."""
    from pathlib import Path
    js = Path("molbuilder/web/static/lib/results/bundle-handoff.js").read_text()
    assert "content-type" in js.lower(), (
        "bundle-handoff.js no longer checks the response's "
        "content-type; a non-JSON 500 will surface as SyntaxError.")


def test_bundle_handoff_html_has_spinner_element(tmp_path):
    """Audit follow-up deferral #1: visual feedback that a request
    is in flight.  Pin the spinner element + data-spinner hook so
    the JS's element lookup can never silently miss."""
    from pathlib import Path
    html = Path("molbuilder/web/templates/_bundle_handoff_panel.html").read_text()
    assert "data-spinner" in html, (
        "_bundle_handoff_panel.html no longer carries a [data-spinner] "
        "element; the JS spinner toggle becomes a no-op (bundle-handoff.js "
        "looks it up via the [data-spinner] attribute hook).")


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


# --------------------------------------------------------------------- #
#  One paired-file writer                                               #
# --------------------------------------------------------------------- #

def test_a_bundled_structure_keeps_every_metadata_field(tmp_path):
    """THE defect this closes. The bundle path used to write the pair itself and
    passed only `regions` / `cell` / `pbc` into the payload, so every bundled
    result silently lost its `cell_origin`, `axis_kind`, `vacuum` and every
    annotation channel — absent keys reset to defaults at the metadata gate.

    It goes through `StructureCodec.write` now: the one paired-file door, whose
    own docstring says no caller may re-derive the sidecar path or re-implement
    the write order. This asserts the consequence — nothing is dropped — rather
    than the call.
    """
    from molbuilder.bundle_writer import write_bundle_as_handoff
    from molbuilder.parse.types import BundleResult
    from molbuilder.sidecars import molstruct
    from molbuilder.structure import AtomChannel, Structure

    struct = Structure.from_xyz("2\n\nC 0 0 0\nO 1 0 0\n")
    struct.cell = [[9.0, 0, 0], [0, 9.0, 0], [0, 0, 9.0]]
    struct.cell_origin = [-1.0, -2.0, -3.0]
    struct.axis_kind = ("periodic", "periodic", "isolated")
    struct.vacuum = (0.0, 0.0, 12.0)
    struct.set_channel("charge", AtomChannel("value", {0: -0.5}))
    struct.__post_init__()

    bundle = BundleResult(
        schema_version=1, parsed_at="2026-01-01T00:00:00Z", parser_name="t",
        source="run", result_kind="bundle", structure=struct,
        cell=struct.cell, regions={"L-electrode": [0]}, frozen_atoms=[1],
        notes=[], block_schema_versions={}, source_engine="siesta",
        final_coords_from="xv", source_script="h2.fdf",
    )
    xyz_path, sidecar_path = write_bundle_as_handoff(
        bundle, tmp_path, stem="handoff")

    saved = molstruct.load(sidecar_path)
    assert saved["cell_origin"] == [-1.0, -2.0, -3.0], "the origin was dropped"
    assert saved["axis_kind"] == ["periodic", "periodic", "isolated"], (
        "the axis kinds were dropped"
    )
    assert saved["vacuum"] == [0.0, 0.0, 12.0], "the vacuum was dropped"
    assert saved["annotations"]["charge"]["kind"] == "value", (
        "the annotation channels were dropped"
    )
    # The labels the RUN recorded land in the one label store, reserved ones too.
    assert saved["regions"] == {"L-electrode": [0], "frozen_atoms": [1]}
    # The provenance line still reaches the .xyz, carried by the structure's
    # own title rather than by a second writer's comment argument.
    assert "bundled from h2.fdf" in xyz_path.read_text(encoding="utf-8")


def test_a_bundled_structure_with_nothing_but_its_title_says_only_that(tmp_path):
    """The codec's rule — a sidecar exists only when there is something to say —
    still holds here, but since schema 8 the bundle's own provenance title IS
    something said: the writer stamps it deliberately ("bundled from …"), the
    identity columns persist when real (the 2026-08-20 ruling), so the pair
    keeps it and a reloaded handoff still says where it came from.  What must
    NOT come back is anything the bundle did not state: no labels, no cell, no
    channels, no per-atom identity — the sidecar says the title and nothing
    else.  (Until schema 8 this asserted "no sidecar at all"; the premise
    moved with the format, the spirit — no empty statements — did not.)"""
    import json

    from molbuilder.bundle_writer import write_bundle_as_handoff
    from molbuilder.parse.types import BundleResult
    from molbuilder.structure import IDENTITY_FIELDS, Structure

    bundle = BundleResult(
        schema_version=1, parsed_at="2026-01-01T00:00:00Z", parser_name="t",
        source="run", result_kind="bundle",
        structure=Structure.from_xyz("2\n\nC 0 0 0\nO 1 0 0\n"),
        cell=None, regions={}, frozen_atoms=[], notes=[],
        block_schema_versions={}, source_engine="pyscf",
        final_coords_from="log", source_script="opt.py",
    )
    xyz_path, sidecar_path = write_bundle_as_handoff(
        bundle, tmp_path, stem="plain")

    assert xyz_path.exists()
    assert sidecar_path is not None, (
        "the bundle stamped a provenance title -- a statement the pair keeps")
    side = json.loads(sidecar_path.read_text())
    assert side["title"] == "bundled from opt.py (pyscf/log)"
    assert [k for k in IDENTITY_FIELDS if k in side] == ["title"], (
        "per-atom identity was never stated and must not appear")
    assert side["regions"] == {} and side["annotations"] == {}
    assert side["cell"] is None
