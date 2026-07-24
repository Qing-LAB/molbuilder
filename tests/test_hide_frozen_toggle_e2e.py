"""End-to-end smoke: hide-frozen toggle reaches the user as a
visible affordance after a real trajectory mount.

Post-MolView migration (task #34)
=================================

The trajectory inspector now mounts the concealed MolView module
into an empty ``#viewer-host`` and is a DATA FEEDER; the only
trajectory-specific control left is the force-vector producer,
and ``#hide-frozen`` is a PURE force-arrow filter (MolView owns
atom hiding in the viewer via its selection/isolate pipeline).
The old ``#hide-frozen-row`` id + the setOverlays viewer-hide
payload are gone, so the click-semantics / overlay-payload L2
test (``test_hide_frozen_overlay_payload_js.py``) was retired.

What this file pins now
=======================

A single e2e smoke that loads /results with a real SIESTA .out
fixture containing the ``siesta: Constraints applied in the
following order:`` echo, mounts the trajectory inspector via the
registry, and asserts:

  * the ``#hide-frozen`` toggle is computed-visible after mount
    (data-independent UI-presence contract) — it lives in the flat
    force-controls block now, no ``#hide-frozen-row`` wrapper id.
  * the ``#hide-frozen`` checkbox is present + starts unchecked.
  * the full chain (partial fetch + script-tag order + MolView
    mount + parser surfaces frame/scf/step content via the real
    /api/watch/data response) actually wires up under Chromium.
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

    page.wait_for_selector("#hide-frozen", timeout=8000)

    # (i) the toggle's enclosing row is computed-visible regardless of
    # data shape.  Post-task-#34 the row is a plain <label> in the flat
    # force-controls block (no #hide-frozen-row id); check the checkbox's
    # closest label is displayed + laid out (offsetParent non-null).
    row_visible = page.evaluate(
        "() => {"
        "  const cb = document.getElementById('hide-frozen');"
        "  const row = cb && cb.closest('label');"
        "  return !!row && getComputedStyle(row).display !== 'none'"
        "         && cb.offsetParent !== null;"
        "}"
    )
    assert row_visible, (
        "the #hide-frozen toggle must be computed-visible after mount "
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


# --------------------------------------------------------------------- #
#  Multi-frame regression (2026-07)                                      #
# --------------------------------------------------------------------- #

def _siesta_out_two_moves(with_forces=True):
    """A SIESTA .out with TWO CG moves (two outcoor blocks) -> the parser
    must surface TWO frames.  Regression fixture for the one-frame bug
    below.

    ``with_forces=False`` drops the ``siesta: Atomic forces`` block (and
    its Max/Res footer) from every move: the parser commits frames on the
    outcoor block alone (forces are optional), and the inspector then
    takes the OTHER load branch -- ``buildForcesPerFrame()`` returns null
    and ``reloadFrames(coordFrames, {forces: null})`` runs with no arrow
    bake.  Frame loading must work identically on both branches.
    """
    def move(n):
        coord_block = "\n".join(
            f"   {i+1 + n}.00000000    {i+1}.00000000    {i+1}.00000000   "
            f"{(i % 2) + 1}       {i+1}  {('C', 'H')[i % 2]}"
            for i in range(4)
        )
        text = dedent(f"""\

                                 ====================================
                                    Begin CG opt. move =      {n}
                                 ====================================

            outcoor: Atomic coordinates (Ang):
            {coord_block}

            outcell: Unit cell vectors (Ang):
                   10.000000    0.000000    0.000000
                    0.000000   10.000000    0.000000
                    0.000000    0.000000   10.000000

               scf:    1  -100.0  -100.0  -100.0  0.001 -1.0 0.5
            SCF Convergence by DM+H criterion

            siesta: E_KS(eV) =          -10{n}.1234
            """)
        if with_forces:
            text += dedent(f"""\

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
        return text
    return ("Executable      : siesta\n*  WELCOME TO SIESTA  *\n"
            + move(0) + move(1))


@pytest.mark.parametrize("with_forces", [True, False],
                         ids=["with-forces", "no-forces"])
def test_multiframe_trajectory_loads_all_frames_with_frame_bar(
        page, flask_server, tmp_path, monkeypatch, with_forces):
    """REGRESSION (2026-07): a multi-frame trajectory result must load EVERY
    frame into MolView and show the frame bar.  The bug: rebuildModel called
    ``molview.data.setViewFlag`` (the flag lives on the SELECTION surface,
    ``data.selection.setViewFlag``) OUTSIDE its try block -- the TypeError
    rejected the promise before ``reloadFrames`` ran, so a trajectory showed
    ONE frame (the installMolecule frame 0) with no frame bar and NO error.

    Parametrized over BOTH force branches: with per-atom force blocks the
    load path sets the forceScale flag + hands forces for the arrow bake;
    without them ``buildForcesPerFrame()`` returns null and reloadFrames
    runs arrow-free.  Frame loading must survive either way -- a bug in
    the force plumbing must never take the trajectory down with it.

    Asserts through molview.data (frameCount) + molview's own frame-controls
    DOM -- never the 3Dmol render target.
    """
    _register_tmp_as_picker_root(tmp_path, monkeypatch)
    out_path = tmp_path / "RELAX_stage2-run0.out"
    out_path.write_text(_siesta_out_two_moves(with_forces=with_forces))

    page.goto(f"{flask_server}/results")
    page.wait_for_function(
        "() => window.molbuilder "
        "      && window.molbuilder.inspectors "
        "      && window.molbuilder.inspectors.list "
        "      && window.molbuilder.inspectors.list().length >= 4"
    )
    page.wait_for_selector("#inspector-host", timeout=5000)
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
    page.wait_for_selector("#hide-frozen", timeout=8000)

    # THE regression assertion: every frame reached the model.
    page.wait_for_function(
        "() => {"
        "  const d = window.molbuilder.molview && window.molbuilder.molview.data;"
        "  return !!d && typeof d.frameCount === 'function' && d.frameCount() >= 2;"
        "}",
        timeout=8000,
    )

    # And MolView surfaced its trajectory UI: the frame-controls bar is
    # un-hidden (mountFrameControls shows it iff frameCount > 1).
    bar_shown = page.evaluate(
        "() => {"
        "  const fc = document.querySelector('.molview-frame-controls');"
        "  return !!fc && !fc.hidden && getComputedStyle(fc).display !== 'none';"
        "}"
    )
    assert bar_shown, (
        "frameCount >= 2 but the molview frame-controls bar is hidden -- "
        "the trajectory UI (slider/play) never surfaced."
    )
