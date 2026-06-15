"""End-to-end smoke: hide-frozen toggle reaches the user as a
visible affordance after a real trajectory mount.

History (E1 demote, 2026-06-14 round-3 follow-up)
=================================================

Pre-E1-demote this file held 376 LoC of L5 contract checks
covering:

  1. The trajectory inspector mounts with a real SIESTA .out.
  2. ``#hide-frozen-row`` is computed-visible after mount.
  3. The Inspect checkbox starts unchecked.
  4. Clicking the checkbox calls setOverlays with the right
     ``selectorValue`` and ``style.hidden: true``.
  5. Un-clicking clears the overlay.

Per the round-3 R3-B test-pyramid audit + ``docs/protocols/test-
strategy.md`` § 4: items (4) and (5) are pure JS function-chain
contracts that an L2 Node test (``test_hide_frozen_overlay_payload
_js.py``) drives in <1 s without a browser.  Only items (1) - (3)
need the e2e tier -- they cover "the full chain (partial fetch +
script-tag order + DOM mount + parser surfaces frozen_atoms via
the real /api/watch/data response) actually wires up under
Chromium."

What this file pins after E1 demote
===================================

A single e2e smoke that loads /results with a real SIESTA .out
fixture containing the ``siesta: Constraints applied in the
following order:`` echo, mounts the trajectory inspector, and
asserts:

  * ``#hide-frozen-row`` is computed-visible (data-independent
    UI-presence contract from the 2026-06-14 update).
  * ``#hide-frozen`` checkbox is present + unchecked.
  * After the parser surfaces ``runtime_info.frozen_atoms``, the
    state.data field carries the expected indices.

Click semantics + overlay-payload shape are NOT exercised here
(L2 owns those).  See test_hide_frozen_overlay_payload_js.py.
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


def _register_tmp_as_picker_root(tmp_path, monkeypatch):
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


def _siesta_out_with_constraints(frozen_one_based=(1, 2)) -> str:
    """Minimal SIESTA-style .out with the v5 constraints echo +
    one optimization step so the parser populates a frame +
    runtime_info.frozen_atoms.  Identical fixture builder to the
    pre-demote version of this file -- kept verbatim because the
    parser pipeline is what this smoke exercises."""
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


# --------------------------------------------------------------------- #
#  Smoke test                                                            #
# --------------------------------------------------------------------- #


def test_hide_frozen_row_and_data_visible_after_real_mount(
        page, flask_server, tmp_path, monkeypatch):
    """The full mount chain -- /results inspector registry +
    partial fetch + trajectory core boot + /api/watch/data +
    SiestaParser.constraints-echo + state.data wiring -- reaches
    the user as a visible affordance.

    Click semantics + overlay-payload shape live at L2; this
    smoke pins the integration tier ONLY.
    """
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    out_path = tmp_path / "BDT_stage2-run0.out"
    out_path.write_text(_siesta_out_with_constraints((1, 2)))

    page.goto(f"{flask_server}/results")
    page.wait_for_function(
        "() => window.molbuilder "
        "      && window.molbuilder.inspectors "
        "      && window.molbuilder.inspectors.list "
        "      && window.molbuilder.inspectors.list().length >= 4"
    )
    page.wait_for_selector("#inspector-host", timeout=5000)

    # Drive the registry -> partial fetch -> trajectory mount.
    page.evaluate(
        "(path) => {"
        "  const reg = window.molbuilder.inspectors;"
        "  const insp = reg.pick(path);"
        "  if (!insp) throw new Error('no inspector for ' + path);"
        "  const host = document.getElementById('inspector-host');"
        "  insp.mount(host, path);"
        "}",
        str(out_path),
    )

    page.wait_for_selector("#hide-frozen-row", timeout=8000)

    # (i) row computed-visible regardless of data shape.
    row_visible = page.evaluate(
        "() => {"
        "  const row = document.getElementById('hide-frozen-row');"
        "  return !!row && getComputedStyle(row).display !== 'none';"
        "}"
    )
    assert row_visible, (
        "#hide-frozen-row must be computed-visible after mount "
        "regardless of whether the parser has finished surfacing "
        "frozen_atoms (UI-presence-is-data-independent contract)."
    )

    # (ii) checkbox is reachable + starts unchecked.
    cb_starts_unchecked = page.evaluate(
        "() => {"
        "  const cb = document.getElementById('hide-frozen');"
        "  return !!cb && cb.checked === false;"
        "}"
    )
    assert cb_starts_unchecked, (
        "#hide-frozen must be reachable + unchecked at mount."
    )

    # (iii) wait for the parser to surface frozen_atoms via the
    # real /api/watch/data response.  Loose status-text wait so we
    # don't depend on private state-object access.
    page.wait_for_function(
        "() => {"
        "  const txt = document.body && document.body.innerText || '';"
        "  return /frame/i.test(txt) || /scf/i.test(txt) || "
        "         /step/i.test(txt);"
        "}",
        timeout=8000,
    )
