"""Shared pytest fixtures for the molbuilder test suite.

Pytest auto-discovers this file and makes the fixtures it defines
available to every test module under ``tests/``.  Add a fixture here
when more than one test file needs the same setup.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from molbuilder import diagnostics
from molbuilder.structure import Structure


@pytest.fixture
def checkpoint_config(tmp_path, monkeypatch):
    """Set the checkpoint classification for one test, where it really lives.

    **S1c: the classification has one home, and it is not beside the folder
    being saved.**  A test cannot set it by dropping a config into the
    calculation directory -- ``_read_project`` refuses a ``checkpoint`` section
    there outright, which is the rule, not a limitation to work around.

    The one home is the server-wide scope, and its first candidate is
    ``./molbuilder.json``.  So this gives the test its own working directory
    and writes the real file into it: the same resolution path production
    takes, with nothing mocked (checkpointing.md § 4, S1c).

    Returns a setter, because several tests change the classification *during*
    a test -- narrowing it between a save and a restore is I2c's own scenario.
    """
    home = tmp_path / "config-home"
    home.mkdir()
    monkeypatch.chdir(home)
    # The OTHER two-thirds of the sandbox (H-5, 2026-08-13): read_config's
    # default lookup falls back to $XDG_CONFIG_HOME / ~/.config when the
    # cwd file is absent (deployment.md § 5), so a chdir alone leaves the
    # default-read tests floating on the developer's own per-user config.
    monkeypatch.setenv("HOME", str(tmp_path / "user-home"))
    (tmp_path / "user-home").mkdir()
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)

    def _set(**section) -> Path:
        target = home / "molbuilder.json"
        target.write_text(json.dumps({"checkpoint": section}))
        return target
    return _set


@pytest.fixture
def isolated_projects_root(tmp_path, monkeypatch):
    """Point the ONE door at a per-test tree — **opt in, not autouse**.

    ``projects_root`` resolves from the molbuilder root now, not the
    working directory (2026-08-22), so the default a test sees is the
    developer's REAL tree — which is exactly what the web-route tests that
    build ``projects/_t_handover`` have always used, and what cwd-anchoring
    gave them when they ran from the repo root.

    A blanket autouse override broke 13 of those: they construct a tree and
    then ask a route to serve it, and the route's root guard was pointed
    somewhere else entirely.  So isolation is requested by the tests that
    build their own tmp tree, and everything else keeps the real default.
    """
    from molbuilder.projects import PROJECTS_ROOT_ENV, PROJECTS_ROOT_NAME
    root = tmp_path / PROJECTS_ROOT_NAME
    root.mkdir(exist_ok=True)
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(root))
    return root


# --------------------------------------------------------------------- #
#  No test resolves the PRODUCT's toolchain from the host  (2026-08-25)  #
# --------------------------------------------------------------------- #

#: The binaries molbuilder resolves from the environment it ENTERS, never
#: from whatever the developer's box happens to have on PATH.  A wrapper
#: activates its conda env and then looks up ``siesta`` there -- its own
#: failure text says so: *"not on PATH after activating '<env>'"*.
_PRODUCT_TOOLCHAIN = ("siesta", "conda", "mamba", "mpirun")

#: `conda activate` / `mamba activate` succeed silently -- that is all a
#: wrapper running in a bare shell needs them to do, and it is the call
#: that reached the host.  EVERYTHING ELSE IS DELEGATED to the real binary
#: (2026-08-25): `molbuilder.diagnostics` asks `conda env list --json` which
#: envs exist, and a stub answering that with silence made `env_available`
#: report none -- so `test_siesta_keyword_smoke.py` and twelve others
#: SKIPPED instead of running.  A guard that turns real tests into skips is
#: worse than the hole it closes, because a skip is quiet.
_DELEGATING = ("conda", "mamba")

_STUB_BODIES = {
    # Answers the build probe the way the parser downstream expects.  NOT
    # delegated: reaching the host's engine is the hole this closes, and a
    # suite that wants a real SIESTA addresses the env's binary by absolute
    # path (`test_siesta_keyword_smoke.py` does, and says so).
    "siesta": (
        'if [ "${1:-}" = "--version" ]; then\n'
        '    echo "Version         : 5.4.2-stub"\n'
        '    echo "Parallelisations: MPI"\n'
        "    exit 0\n"
        "fi\n"
        'echo "stub siesta: $*"\n'
    ),
    # Drops its own flags and runs what it was asked to launch, so a
    # `mpirun -np 4 siesta ...` still reaches the siesta stub above.
    "mpirun": (
        "while [ $# -gt 0 ]; do\n"
        '    case "$1" in\n'
        "        -np|-n|--np|--n) shift 2 ;;\n"
        "        -*) shift ;;\n"
        "        *) break ;;\n"
        "    esac\n"
        "done\n"
        'if [ $# -gt 0 ]; then exec "$@"; fi\n'
        "exit 0\n"
    ),
}


def _delegating_body(real: str) -> str:
    """Swallow `activate`; hand everything else to the real binary."""
    return (
        'case "${1:-}" in\n'
        "    activate|deactivate) exit 0 ;;\n"
        "esac\n"
        f'_real="{real}"\n'
        'if [ -n "$_real" ] && [ -x "$_real" ]; then exec "$_real" "$@"; fi\n'
        "exit 0\n"
    )


@pytest.fixture(scope="session")
def _product_toolchain_stubs(tmp_path_factory) -> Path:
    """Built once per session; the PATH entry is per-test (below)."""
    d = tmp_path_factory.mktemp("product-toolchain")
    # Resolved BEFORE the stub dir is on PATH, so a delegating stub cannot
    # find itself.
    real = {n: (shutil.which(n) or "") for n in _DELEGATING}
    for name in _PRODUCT_TOOLCHAIN:
        if name in _DELEGATING:
            # A GUARD MUST NOT INVENT A TOOL THE MACHINE DOES NOT HAVE
            # (2026-08-25).  `mamba` is absent on qlabsrv, and stubbing it
            # anyway put one on PATH -- where `_locate_env_manager` PREFERS
            # it to conda.  The invented mamba shadowed the real conda,
            # answered `env list --json` with silence, and detection
            # reported zero envs; thirteen tests that need
            # `molbuilder-siesta` skipped instead of running, quietly.
            # Absent stays absent: there is no host binary to reach.
            if not real[name]:
                continue
            body = _delegating_body(real[name])
        else:
            body = _STUB_BODIES[name]
        f = d / name
        f.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
        f.chmod(0o755)
    return d


@pytest.fixture(autouse=True)
def product_toolchain_is_the_suites_own(_product_toolchain_stubs, monkeypatch):
    """**No test may reach the host's engine or conda.**

    Several suites render a run wrapper and EXECUTE it -- which is the right
    thing to test, because that bash is the artifact the cluster runs.  To
    run it in a bare shell they strip the bootstrap out of it, and that is
    where the hole was: with no activation, every lookup inside the wrapper
    (``command -v siesta``, ``conda activate``, ``mpirun``) fell through to
    the SYSTEM path.  The suite's result then depended on what the developer
    happened to have installed, which is the opposite of a test.

    Found live on qlabsrv 2026-08-25.  A root-owned 2023
    ``/usr/local/bin/siesta`` was on PATH; it does not know ``--version``,
    and SIESTA reads its deck from stdin, so it did not fail -- it waited
    for a deck.  Eight tests in ``test_runwrap_cold_restart.py`` failed as
    20-second timeouts, each leaving a blocked process behind (28 had
    accumulated).  A hostile-toolchain sweep then found the same hole in
    ``test_launch_door_gate.py`` (conda), ``test_wrapper_preamble_preflight``
    and ``test_runwrap.py``.

    So the suite brings its own, and they behave.  **A test that needs a
    HOSTILE binary builds its own stub and prepends it** -- it lands ahead of
    this one and wins (``test_runwrap_engine_probe.py`` does exactly that,
    and must, because a guard against hanging cannot be proven by a stub
    that never hangs).  What no test gets is the host's.

    This does NOT stub the test harness's own tools -- ``bash``, ``node``,
    ``git``, ``timeout``.  Those are how the tests RUN; the product does not
    resolve them from an env it activates, and a suite that shadowed them
    would be unable to execute at all.
    """
    monkeypatch.setenv(
        "PATH", f"{_product_toolchain_stubs}{os.pathsep}{os.environ['PATH']}")


@pytest.fixture(autouse=True)
def _reset_diagnostics_singleton():
    """Every test starts AND ends with no bound Capabilities snapshot.

    Without this, a test that calls ``cli.main()`` or otherwise
    triggers ``diagnostics.initialize()`` would leak its snapshot into
    whatever runs next, creating order-dependent failures.  Tests that
    want a specific snapshot inject it via
    :func:`molbuilder.diagnostics.set_capabilities`; that injection is
    cleaned up by this fixture afterwards.
    """
    diagnostics.reset_capabilities()
    yield
    diagnostics.reset_capabilities()


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Absolute path to ``tests/data/`` (PDB / XYZ fixtures live here)."""
    return Path(__file__).parent / "data"


@pytest.fixture
def water_structure() -> Structure:
    """Tiny three-atom Structure (H2O) used by several modules."""
    return Structure(
        elements=["O", "H", "H"],
        positions=np.array([
            [0.000, 0.000, 0.000],
            [0.957, 0.000, 0.000],
            [-0.240, 0.927, 0.000],
        ]),
        atom_names=["O", "H1", "H2"],
        residue_ids=[1, 1, 1],
        residue_names=["HOH", "HOH", "HOH"],
        chain_ids=["A", "A", "A"],
        title="water",
        vacuum=(12.0, 12.0, 12.0),
    )


@pytest.fixture
def deprotonated_diester() -> Structure:
    """Synthetic R-O-P(=O)(O-)-O-R' phosphate diester with no Hs on
    either non-bridging oxygen.  Heuristic charge -> -1.

    Used by the chemistry / siesta / pyscf charge-detection tests.
    """
    elements  = ["C", "O", "P", "O", "O", "O", "C"]
    positions = np.array([
        [-2.5, 0.0, 0.0],   # C5'
        [-1.4, 0.0, 0.0],   # O5' (bridge)
        [ 0.0, 0.0, 0.0],   # P
        [ 0.0, 1.5, 0.0],   # OP1 (non-bridging)
        [ 0.0, -0.8, 1.3],  # OP2 (non-bridging)
        [ 1.4, 0.0, 0.0],   # O3' (bridge)
        [ 2.5, 0.0, 0.0],   # C3'
    ])
    return Structure(
        elements=elements, positions=positions,
        atom_names=["C5'", "O5'", "P", "OP1", "OP2", "O3'", "C3'"],
        residue_ids=[1] * 7, residue_names=["DA"] * 7, chain_ids=["A"] * 7,
        vacuum=(12.0, 12.0, 12.0),
    )


@pytest.fixture
def web_client():
    """Flask test client; skips the test if Flask isn't installed.

    Passes ``config={}`` explicitly so ``create_app`` does NOT read
    the repo-root ``molbuilder.json`` (which may have auth/TLS
    enabled in a developer's working tree).  Tests against the
    no-auth/no-TLS default isolate the page-render + API surface
    from per-machine config state.  Tests that need an auth-enabled
    or otherwise non-default app build their own fixture and pass
    the matching ``config`` dict.
    """
    pytest.importorskip("flask")
    from molbuilder.web.app import create_app
    # rate_limit.enabled=false in the default test client because
    # several test files fire 60+ requests within the rolling
    # window and would otherwise hit the total-burst threshold.
    # tests/test_rate_limit.py builds its own client with the
    # limiter enabled to exercise the module directly.
    app = create_app(config={"rate_limit": {"enabled": False}})
    return app.test_client()


# --------------------------------------------------------------------- #
#  Marker auto-application by file pattern (2026-06-24)                 #
#  Implements the marker discipline documented in                       #
#  docs/process/testing.md  Tests can override at the     #
#  function level by carrying an explicit @pytest.mark.<X>.            #
# --------------------------------------------------------------------- #

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Stash each phase's report on the item so fixtures can tell, at
    teardown, whether the test failed.  Used by the ``capture_on_fail``
    diagnostic harness (a test flagged with @pytest.mark.capture_on_fail
    dumps browser state + console to a PERSISTENT ``test-artifacts/`` dir
    when it fails -- so a rare intermittent E2E failure is inspectable
    after the fact instead of needing a re-repro)."""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, "rep_" + rep.when, rep)


@pytest.fixture(autouse=True)
def _capture_on_fail(request):
    """Persistent failure diagnostics for ``@pytest.mark.capture_on_fail``
    tests (any file).  On failure of a flagged test, dump the browser state
    (viewer/store atom counts, source_file, dirty, mountRestoreTarget, the
    persisted snapshot) + the console to a PERSISTENT ``test-artifacts/``
    dir -- so a rare intermittent E2E is inspectable after the fact without
    a re-repro.  No-op (and no ``page`` dependency) for unflagged tests."""
    marked = request.node.get_closest_marker("capture_on_fail") is not None
    page = None
    msgs: list = []
    if marked:
        try:
            page = request.getfixturevalue("page")
            page.on("console", lambda m: msgs.append(m.text))
        except Exception:
            page = None
    yield
    if not marked:
        return
    rep = getattr(request.node, "rep_call", None)
    if not (rep and rep.failed):
        return
    import json as _json
    import datetime as _dt
    import pathlib as _pl
    art = _pl.Path(__file__).resolve().parent.parent / "test-artifacts"
    art.mkdir(exist_ok=True)
    snap = {"note": "no page for this test"}
    if page is not None:
        try:
            # ASK THE VIEWER, NOT THE WORKSPACE.  This used to read five doors
            # off ``window.molbuilder.workspace`` -- readPersistedSnapshot,
            # getState, getSourceFile, isDirty, mountRestoreTarget -- none of
            # which exist.  The workspace stores opaque bytes and never opens
            # them (workspace.md § 4), so it cannot answer "how many atoms" at
            # all; every field came back null and a failing e2e wrote a
            # diagnostic that said nothing.
            #
            # The structure lives in MolView.  The Results pages stash their
            # handle on the viewer host (``__molview_results_handle``) so the
            # inspector and the trajectory can find each other's viewer; the
            # Modify page exposes its own small read-only hook instead.  "Is
            # there unsaved work" is ``uncommitted``, a value the viewer holds
            # (molview.md § 11.2).
            snap = page.evaluate("""() => {
                let h = null;
                // The two hosts that carry a handle: the trajectory's
                // #viewer-host and the structure inspector's generated slot.
                for (const el of document.querySelectorAll(
                        "#viewer-host, .structure-viewer-slot")) {
                    if (el.__molview_results_handle) {
                        h = el.__molview_results_handle;
                        break;
                    }
                }
                const d = h && h.ok !== false ? h.data : null;
                let s = null;
                try { s = d && d.getStructure ? d.getStructure() : null; } catch (e) {}
                const t = window.__molbuilder_modify_test;
                return {
                    url: location.href,
                    viewer_found: !!d,
                    n_atoms: s ? (s.elements || []).length : null,
                    has_cell: !!(s && s.periodicity),
                    uncommitted: d ? !!d.uncommitted : null,
                    state_index: d ? d.state_index : null,
                    // The Modify tab's own hook, when this is that page.
                    modify_n_atoms: t && t.getNAtoms ? t.getNAtoms() : null,
                    modify_selected: t && t.getSelected ? t.getSelected().length : null,
                };
            }""")
        except Exception as e:  # noqa: BLE001
            snap = {"evaluate_error": str(e)}
    ts = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    out = art / f"fail-{request.node.name}-{ts}.log"
    out.write_text(
        "test:  " + request.node.nodeid + "\n\n"
        "PAGE STATE:\n" + _json.dumps(snap, indent=2) + "\n\n"
        "ERROR:\n" + str(rep.longrepr)[:3000] + "\n\n"
        "CONSOLE (last 300):\n" + "\n".join(msgs[-300:]) + "\n",
        encoding="utf-8")
    print(f"[capture_on_fail] diagnostic written -> {out}")


def pytest_collection_modifyitems(config, items):
    """Auto-apply ``e2e`` to ``*_e2e.py`` files + ``integration`` to
    files that subprocess-run a real engine binary (siesta /
    transiesta / pyscf).

    The pyproject.toml markers list registered ``unit / module /
    interface / integration / smoke / e2e / slow`` but as of the
    2026-06-24 audit zero tests carried any of them, so
    ``pytest -m integration`` returned nothing.  This hook gives
    every existing test file the right baseline marker so the doc-
    promised selectors work, without requiring per-test annotation.
    File-name pattern is a coarse pre-classifier; finer-grained
    decisions still belong on individual tests via explicit
    decorators.
    """
    import pytest as _pt
    for item in items:
        fn = item.fspath.basename
        if fn.endswith("_e2e.py") or "_e2e_" in fn:
            item.add_marker(_pt.mark.e2e)
        if "_smoke" in fn or "_smoke_l4" in fn:
            item.add_marker(_pt.mark.integration)
            item.add_marker(_pt.mark.smoke)


# --------------------------------------------------------------------- #
#  Pseudopotentials for a calculation under test                        #
# --------------------------------------------------------------------- #

#: The smallest PSML ``pseudos.parse_psml_header`` accepts: a real element and
#: a scalar-relativistic spec, and no projector block -- so no null channel,
#: and ``check_coverage`` returns ``ok``.
_PSML_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<psml version="1.1" energy_unit="hartree" length_unit="bohr"
      xmlns="http://launchpad.net/psml">
  <pseudo-atom-spec atomic-label="{el}" atomic-number="{z}"
                    z-pseudo="1" core-corrections="no"
                    relativity="scalar" spin-dft="no" flavor="test-fixture">
    <valence-configuration total-valence-charge="1"/>
  </pseudo-atom-spec>
</psml>
"""

_PSML_Z = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16, "Fe": 26, "Au": 79}


def write_pseudos(dest, elements) -> None:
    """Put the ``.psml`` files a SIESTA calculation needs into *dest*.

    Since 2026-08-18 `prep` REFUSES by element name when a species has no
    pseudopotential in the calculation and none in the library
    (`project-layout.md` § 2.6): SIESTA opens ``<element>.psml`` in the
    directory it runs from and has no search path, so a bundle without them is
    one that cannot start, and laying out a tree for it helps nobody.  A test
    that preps therefore supplies them, exactly as a person does.

    **Real PSML, not touch-files.**  `prep` also runs the screening in
    `science/pseudopotentials.md` § 1 over whatever is in the folder -- the
    checks that exist because a dead-p-channel ``S.psml`` once shipped into a
    real run, giving wrong sulfur bonding and a `propor: ERROR: IMAX=0` that
    appeared only at high rank counts.  An unparseable file reads as
    ``missing``, which blocks.  These parse, name their element and declare no
    null channel; the physics is not what the tests using this are about.

    One home because four test files needed it within an hour of the refusal
    landing.
    """
    from pathlib import Path as _P
    for el in elements:
        (_P(dest) / f"{el}.psml").write_text(
            _PSML_FIXTURE.format(el=el, z=_PSML_Z[el]))


@pytest.fixture(autouse=True)
def _isolated_machine_scope(tmp_path_factory, monkeypatch):
    """Every test reads its OWN machine record, never the developer's.

    ``machine_scope_path()`` resolves ``$XDG_CONFIG_HOME/molbuilder/`` (else
    ``~/.config/molbuilder/``), and ``environments_dir()`` is the
    ``environments/`` beside it -- the USER's real probed machines.  Nothing
    isolated them, so the suite has been reading whatever this box happens to
    have, and passing because most boxes have nothing.

    **How it surfaced.**  On 2026-08-23 a real ``sol.json`` was probed on Sol
    and saved to ``~/.config/molbuilder/environments/``.  Two tests in
    ``test_cheapest_ceiling_that_fits`` went red at once -- not wrongly: with a
    named target present, `resolve` is RIGHT to refuse and ask which machine is
    meant (C1).  The tests were asserting placement against found state, and
    the found state changed.  **A suite whose colour depends on the developer's
    home directory is not a suite** -- and the failure lands on whoever did the
    correct thing, which is the worst possible messenger.

    Sibling of :func:`_isolated_workspace_store` and for the same reason: a
    per-user store that production reads by default needs a test-time home, or
    the tests and the user share one.

    ``tmp_path_factory`` rather than ``tmp_path`` so the directory does not
    appear inside the tree a test builds and then walks.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME",
                       str(tmp_path_factory.mktemp("machine-scope")))


@pytest.fixture(autouse=True)
def _isolated_workspace_store(tmp_path, monkeypatch):
    """Every test reads and writes its OWN workspace-state store.

    The store's home is ``projects_root()/.molbuilder_workspace`` -- the
    USER's real saved workspaces.  Without this, any test that drives a
    page writes its panel state into that real store on teardown
    (pagehide -> persist), and any later test whose page mounts on an
    empty canvas RESTORES it: on 2026-08-21 the auto-detect e2e left a
    one-atom "Fe atom for auto-analyze test" panel there, and the
    build tab's "setShared must NOT auto-load" e2e failed against a
    structure it never loaded -- proven with found state, twice over
    (the tests polluted each other AND the user's own store).

    ``workspace_storage``'s own docstring names this exact patch point:
    the name is bound at import, so it is rebound on THAT module.  A
    test that wants the store elsewhere (test_workspace_storage_api)
    applies its own later patch, which wins.
    """
    try:
        from molbuilder.web.blueprints import workspace_storage as _ws
    except Exception:
        yield          # no flask in this env; nothing to isolate
        return
    monkeypatch.setattr(_ws, "projects_root",
                        lambda: tmp_path / "_workspace_store_root")
    yield
