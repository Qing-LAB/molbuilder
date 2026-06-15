"""End-to-end: the "Hide frozen atoms" toggle survives a real
script-tag load + a real DOM mount + a real click.

L2 source-text tests already pin the function-level contract
(``test_ui_presence_data_independent_js.py`` ensures the row is
never written to ``.hidden = true`` after no-data;
``test_hide_frozen_visibility_transition_js.py`` exercises the row
visibility under JSDOM).  Neither catches:

  * the partial HTML failing to inject under real /results script
    order,
  * the embed handle being created before the click listener is
    wired (or the reverse),
  * a future refactor moving ``#hide-frozen`` out of the partial,
  * the click handler reaching the wrong embed instance because
    the dispatcher was clobbered (the 2026-06-13 BLOCKER class).

What the test does:

  1. Goto ``/results`` and wait for the inspector registry.
  2. Patch ``window.molbuilder.viewer.embed`` to capture the inner
     handle when the trajectory inspector creates it.
  3. Write a real SIESTA-style ``.out`` fixture WITH the
     ``siesta: Constraints applied in the following order:`` echo
     for atoms 0 and 1 (the v5 BDT case) so the parser populates
     ``runtime_info.frozen_atoms`` end-to-end.
  4. Mount the trajectory inspector with that fixture path.
  5. Verify ``#hide-frozen-row`` is computed-visible BEFORE the
     parser surfaces frozen atoms (data-independent affordance).
  6. After load, click the checkbox + assert the captured embed
     handle's ``getOverlays()`` returns a single entry whose
     ``indices`` match the parsed frozen set AND whose
     ``style.hidden === true``.
  7. Click again to clear; assert the overlays are empty.

Pin the BOTH halves of the hide-frozen contract at the browser
tier: data-independent UI presence + data-dependent effect.
"""
from __future__ import annotations

import threading
from textwrap import dedent

import pytest


pytestmark = pytest.mark.e2e

pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")


# --------------------------------------------------------------------- #
#  Server fixture                                                       #
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def flask_server():
    from werkzeug.serving import make_server
    from molbuilder.web.app import create_app
    app = create_app(config={})
    server = make_server("127.0.0.1", 0, app, threaded=True)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


# --------------------------------------------------------------------- #
#  SIESTA fixture builder                                               #
# --------------------------------------------------------------------- #


def _siesta_out_with_constraints(frozen_one_based=(1, 2)) -> str:
    """Minimal SIESTA-style .out with the v5 constraints echo + one
    optimization step so the parser populates a frame + frozen_atoms.

    Coordinate / SCF / forces / Etot sections lifted from
    ``tests/watch/test_siesta_parser.py``'s SAMPLE and pared to one
    step.  ``frozen_one_based`` becomes the ``[ A -- B ]`` range
    inside a single ``Constraint (N): pos`` block (SIESTA's actual
    format -- inclusive 1-based).
    """
    lo, hi = min(frozen_one_based), max(frozen_one_based)
    range_str = f"[ {lo} -- {hi} ]"
    n_atoms = 4
    coord_block = "\n".join(
        f"   {i+1}.00000000    {i+1}.00000000    {i+1}.00000000   "
        f"{(i % 2) + 1}       {i+1}  {('C', 'H')[i % 2]}"
        for i in range(n_atoms)
    )
    return dedent(f"""\
        Executable      : siesta
        *  WELCOME TO SIESTA  *

        siesta: Constraints applied in the following order:
        siesta: Constraint ({hi - lo + 1}): pos
          {range_str}


                             ====================================
                                Begin CG opt. move =      0
                             ====================================

        outcoor: Atomic coordinates (Ang):
        {coord_block}

        outcell: Unit cell vectors (Ang):
               10.000000    0.000000    0.000000
                0.000000   10.000000    0.000000
                0.000000    0.000000   10.000000

        siesta: Eharris =   -289239.010387

           scf:    1  -100.0  -100.0  -100.0  0.001 -1.0 0.5
        SCF Convergence by DM+H criterion

        siesta: E_KS(eV) =          -100.1234

        siesta: Atomic forces (eV/Ang):
             1    0.10    0.20    0.30
             2    0.40    0.50    0.60
             3    0.05    0.06    0.07
             4    0.08    0.09    0.10
        ----------------------------------------
           Tot    0.63    0.85    1.07
        ----------------------------------------
           Max    1.234567
           Res    0.987654    sqrt( Sum f_i^2 / 3N )
        ----------------------------------------
           Max    1.234567    constrained
        """)


def _register_tmp_as_picker_root(tmp_path, monkeypatch):
    """Same shim ``test_results_folder_dispatch_e2e.py`` uses --
    pin the file picker's root to tmp_path so the /api/build endpoints
    (which the trajectory loader calls) resolve the fixture path."""
    from molbuilder import diagnostics
    caps = diagnostics.Capabilities(
        runtime_config={}, conda_binary=None,
        conda_envs=frozenset(),
    )
    cls = type(caps)
    monkeypatch.setattr(
        cls, "file_picker_roots",
        lambda self: ((tmp_path.resolve(), "projects"),),
    )
    diagnostics.set_capabilities(caps)


# --------------------------------------------------------------------- #
#  Capture helper                                                       #
# --------------------------------------------------------------------- #


_CAPTURE_EMBED_JS = """
() => {
  // Wrap window.molbuilder.viewer.embed so every handle the
  // trajectory inspector creates lands on window.__capturedHandle.
  // Original is preserved + delegated to so the embed actually
  // mounts (we want the real WebGL handle, not a stub).
  if (window.__embedWrapped) return;
  window.__embedWrapped = true;
  const orig = window.molbuilder.viewer.embed;
  window.molbuilder.viewer.embed = function (el, opts) {
    const h = orig.apply(this, arguments);
    window.__capturedHandle = h;
    return h;
  };
}
"""


# --------------------------------------------------------------------- #
#  Test                                                                 #
# --------------------------------------------------------------------- #


def test_hide_frozen_toggle_drives_overlay_payload(
        page, flask_server, tmp_path, monkeypatch):
    """The full chain: write a SIESTA .out with the constraints
    echo, mount the trajectory inspector, verify the checkbox is
    reachable + visible regardless of data shape, then assert that
    clicking it flips the embed's overlay state to a single
    ``style.hidden:true`` entry covering the parsed indices.

    Regression-pin for the 2026-06-13 hide-frozen BLOCKER class
    (row was tied to data shape, checkbox handler reached the
    wrong embed via a clobbered dispatcher, parser never surfaced
    constraints from ``-runN.out`` files).
    """
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    out_path = tmp_path / "BDT_stage2-run0.out"
    out_path.write_text(_siesta_out_with_constraints((1, 2)))

    page.goto(f"{flask_server}/results")
    # Inspectors register on /results page bootstrap.
    page.wait_for_function(
        "() => window.molbuilder "
        "      && window.molbuilder.inspectors "
        "      && window.molbuilder.inspectors.list "
        "      && window.molbuilder.inspectors.list().length >= 4"
    )
    page.wait_for_selector("#inspector-host", timeout=5000)

    # Patch embed factory BEFORE the inspector mounts.  Order
    # matters -- after mount, window.molbuilder.viewer.embed has
    # already been called and we'd miss the handle.
    page.evaluate(_CAPTURE_EMBED_JS)

    # Drive the registry: pick() returns the trajectory inspector
    # for an ``.out`` path; mount() fetches the partial + invokes
    # the core's mount, which in turn calls viewer.embed (captured
    # above) and starts the file load.
    page.evaluate(
        "(path) => {"
        "  const reg = window.molbuilder.inspectors;"
        "  const insp = reg.pick(path);"
        "  if (!insp) throw new Error('no inspector for ' + path);"
        "  const host = document.getElementById('inspector-host');"
        "  window.__handle = insp.mount(host, path);"
        "}",
        str(out_path),
    )

    # Partial fetch + DOM injection is async.  Wait for the
    # checkbox row to land.
    page.wait_for_selector("#hide-frozen-row", timeout=8000)

    # Half 1 of the contract: the row is computed-visible BEFORE
    # the parser has finished (no data dependency).
    row_visible_pre_data = page.evaluate(
        "() => {"
        "  const row = document.getElementById('hide-frozen-row');"
        "  return !!row && getComputedStyle(row).display !== 'none';"
        "}"
    )
    assert row_visible_pre_data, (
        "#hide-frozen-row must be computed-visible after mount, "
        "regardless of whether the parser has surfaced frozen "
        "atoms yet.  Pre-2026-06-14 the row carried [hidden] until "
        "runtime_info.frozen_atoms.size > 0; that contract is "
        "retired."
    )

    # Wait for the file load to populate state.data AND the parser
    # to surface frozen_atoms via the constraints echo.  The
    # trajectory core's runtime_info.frozen_atoms is the slot.
    page.wait_for_function(
        "() => {"
        "  const h = window.__capturedHandle;"
        "  return !!h && typeof h.getOverlays === 'function';"
        "}",
        timeout=8000,
    )

    # Sanity: the embed handle exists + overlays start empty (the
    # core's mount() does NOT pre-apply a hide-frozen entry; the
    # checkbox starts unchecked).
    overlays_before = page.evaluate(
        "() => window.__capturedHandle.getOverlays()"
    )
    # ``getOverlays`` returns null when no overlays have been set
    # OR ``{atoms: []}`` after an explicit empty-set.  Both are OK
    # for "no hide entry"; assert neither has a hidden:true atom.
    assert not _has_hidden_entry(overlays_before), (
        f"pre-click overlays already carry a hidden entry: "
        f"{overlays_before!r} -- mount must not auto-apply the "
        f"hide-frozen filter."
    )

    cb_starts_unchecked = page.evaluate(
        "() => {"
        "  const cb = document.getElementById('hide-frozen');"
        "  return !!cb && cb.checked === false;"
        "}"
    )
    assert cb_starts_unchecked, (
        "#hide-frozen must start unchecked: a stale 'checked' "
        "state would hide atoms the user expects to see."
    )

    # Wait for the parser to surface frozen atoms via the loaded
    # .out (state.data.runtime_info.frozen_atoms).  The
    # trajectory core's state is private, but its applyHideFrozen
    # is wired to the checkbox + reads from state.data, so the
    # observable IS the click effect: once we click, getOverlays()
    # should reflect the frozen indices.  We bound the wait via
    # the network response on /api/trajectory/load.
    page.wait_for_function(
        # Force-status text shows up after data lands; cheap proxy
        # for "trajectory loaded into state.data".  If neither
        # forces nor energies are surfaced the status stays blank.
        "() => {"
        "  const txt = document.body && document.body.innerText || '';"
        "  return /frame/i.test(txt) || /scf/i.test(txt) || "
        "         /step/i.test(txt);"
        "}",
        timeout=8000,
    )

    # Click the checkbox -- state.current.overlays must gain a
    # single ``style.hidden: true`` entry covering atoms 0 + 1.
    page.evaluate(
        "() => { document.getElementById('hide-frozen').click(); }"
    )
    # Allow the change-event handler to dispatch + the embed to
    # update its overlays state.  Polling is cheap.
    page.wait_for_function(
        "() => {"
        "  const h = window.__capturedHandle;"
        "  const o = h && h.getOverlays();"
        "  if (!o || !Array.isArray(o.atoms)) return false;"
        "  const e = o.atoms[0];"
        "  return !!e && e.style && e.style.hidden === true;"
        "}",
        timeout=4000,
    )

    overlays_after = page.evaluate(
        "() => window.__capturedHandle.getOverlays()"
    )
    # ``_normaliseOverlays`` transforms the public ``indices`` slot
    # into ``selectorKind: "indices", selectorValue: [...]`` — that's
    # the shape getOverlays() returns.
    entry = (overlays_after.get("atoms") or [{}])[0]
    assert entry.get("selectorKind") == "indices", (
        f"hide-frozen overlay must have selectorKind='indices', "
        f"got: {entry!r}."
    )
    indices = entry.get("selectorValue", [])
    assert sorted(int(i) for i in indices) == [0, 1], (
        f"hide-frozen overlay indices must match the 0-based set "
        f"the parser surfaced from the constraints echo.  Got: "
        f"{indices!r}.  Echo had ``[ 1 -- 2 ]`` (1-based) -> "
        f"expected 0-based [0, 1]."
    )

    # Final check: un-click clears.
    page.evaluate(
        "() => { document.getElementById('hide-frozen').click(); }"
    )
    page.wait_for_function(
        "() => {"
        "  const h = window.__capturedHandle;"
        "  const o = h && h.getOverlays();"
        "  if (!o) return true;"  # null is "no overlays"
        "  if (!Array.isArray(o.atoms)) return true;"
        "  return o.atoms.length === 0 || !o.atoms[0];"
        "}",
        timeout=4000,
    )


# --------------------------------------------------------------------- #
#  Helpers                                                              #
# --------------------------------------------------------------------- #


def _has_hidden_entry(overlays):
    """True iff ``overlays`` carries a single atoms[] entry whose
    ``style.hidden === True``.  Tolerant of the ``null`` / empty
    shapes the embed returns for "no overlays set"."""
    if not overlays:
        return False
    atoms = overlays.get("atoms")
    if not isinstance(atoms, list) or not atoms:
        return False
    style = (atoms[0] or {}).get("style") or {}
    return style.get("hidden") is True
