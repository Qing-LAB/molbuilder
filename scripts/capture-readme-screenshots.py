"""Capture the 10 PNG screenshots referenced by the top-level README.

Usage -- the host env active, on its own Python (`ops/installation.md` M6):

    conda activate molbuilder
    "${CONDA_PREFIX:?}/bin/python" scripts/capture-readme-screenshots.py

The script drives Chromium via Playwright against a fresh
``molbuilder serve foreground`` process, started on the host env's own
Python by its path, and writes the 10 PNGs to ``docs/img/``.
Demo data is the BDT project under ``projects/``.  Re-runnable; PNGs
are overwritten in place.

The 10 filenames + intended routes are hardcoded in ``_capture_each``
below.  ``docs/process/screenshots.md`` carries the human-facing capture
guide (URL, viewport, zoom region per shot) for manual re-shoots when
Playwright is unavailable; that doc and this script are two
independent paths to the same 10-PNG set.  Any time a route is added,
renamed, or split, BOTH this script and ``SCREENSHOTS.md`` must be
updated.

The script obeys three load-bearing constraints documented in the
molbuilder code; each is cited inline at the implementation site:

  (1) Package discovery. ``-m molbuilder`` is run from outside
      the repo root, so ``PYTHONPATH`` must include the repo root --
      molbuilder is not pip-installed (see ``feedback_no_pip_install_e``
      in memory + ``docs/ops/installation.md``).
  (2) Its own folders.  The server reads its config from its config
      folder (``$MOLBUILDER_CONFIG_DIR``, ``config_dir.machine_config_dir``)
      and keeps its state and runtime files under ``$XDG_STATE_HOME`` and
      ``$XDG_RUNTIME_DIR``: the child is given folders of its own in the
      temp folder, so the person's config (TLS, sign-in) is never loaded
      and nothing of theirs is written; ``--no-auth`` says the capture
      needs no sign-in.
  (3) Projects discovery.  ``projects.projects_root`` reads
      ``$MOLBUILDER_PROJECTS``, set to the repo's ``projects/`` so the demo
      data resolves.

Playwright runs Chromium at a fixed 1440x900 viewport with
device-scale-factor 2 (retina-style sharpness) so the captures form
a consistent family.  Server output is redirected to a temp file --
not ``stdout=PIPE`` -- so a chatty startup cannot fill the buffer and
block the server.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

REPO_ROOT      = Path(__file__).resolve().parent.parent
IMG_DIR        = REPO_ROOT / "docs" / "img"
DEMO_PROJECT   = "BDT"
DEMO_STRUCTURE = "BDT-AuJunction_siestaStage1_optimized.xyz"

VIEWPORT_W = 1440
VIEWPORT_H = 900

# Idle wait after navigation, in seconds.  Lets 3Dmol's WebGL paint
# the molecule + camera-fit before the screenshot fires.  Tuned by
# experiment on the BDT project (~700 ms suffices for most shots,
# 1.2 s leaves margin for the largest -- the Au-junction transport
# render).
IDLE_AFTER_NAV_S = 1.2


def _free_port() -> int:
    """Bind a temporary socket on 127.0.0.1:0 to discover an unused port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def _serve(port: int, capture_cwd: Path) -> Iterator[subprocess.Popen]:
    """Spawn ``molbuilder serve foreground`` and yield its Popen handle.

    Terminates the child on context exit (SIGTERM, then SIGKILL after
    a 5 s grace).  Raises ``RuntimeError`` if the server fails to bind
    the port within 20 s and surfaces the server log in the message.

    ``capture_cwd`` is the temp folder the child runs in and keeps its
    own config, state and runtime folders under (constraint (2)).
    """
    env = dict(os.environ)
    # Constraint (1): PYTHONPATH lets ``-m molbuilder`` find the
    # package from the temp cwd.  Prepend so REPO_ROOT wins over any
    # parent-shell PYTHONPATH.
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(REPO_ROOT) + (os.pathsep + existing_pp if existing_pp else "")
    )
    # Constraints (2) and (3): its own folders, the repo's demo projects.
    for variable, folder in (("MOLBUILDER_CONFIG_DIR", "config"),
                             ("XDG_STATE_HOME", "state"),
                             ("XDG_RUNTIME_DIR", "run")):
        where = capture_cwd / folder
        where.mkdir(mode=0o700, exist_ok=True)
        env[variable] = str(where)
    env["MOLBUILDER_PROJECTS"] = str(REPO_ROOT / "projects")

    # THE HOST ENV'S OWN PYTHON: the one this script runs on, by its usage
    # line -- the server child is that one, as `serve start`'s is
    # (`ops/installation.md` M6); never a `python` a PATH would find.
    cmd = [
        sys.executable, "-m", "molbuilder", "serve", "foreground",
        "--host", "127.0.0.1", "--no-auth",
        "--port", str(port),
        # ONE process, so the handle below really owns it.  Supervision became
        # the default on 2026-08-04; under it this Popen would hold the PARENT,
        # and terminate() would kill the supervisor while the child kept the
        # port -- an orphan the next run collides with.  Nothing here needs a
        # restart button.
        "--no-supervise",
    ]

    # Redirect server output to a file so the PIPE buffer never blocks
    # the server mid-startup; surface the log on launch failure.
    server_log = tempfile.NamedTemporaryFile(
        prefix="molbuilder-serve-", suffix=".log",
        delete=False, mode="w",
    )
    log_path = server_log.name
    server_log.close()

    proc = subprocess.Popen(
        cmd, env=env, cwd=str(capture_cwd),
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Wait for the bind to land (~3 s is typical; 20 s gives plenty of
    # head-room).  Poll with 0.2 s sleep; abort if the deadline passes.
    deadline = time.time() + 20.0
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.2)
    else:
        try:
            output = Path(log_path).read_text()
        except OSError:
            output = "(server log unreadable)"
        proc.terminate()
        raise RuntimeError(
            f"molbuilder serve did not bind to 127.0.0.1:{port} "
            f"within 20 s.  Server output:\n{output}"
        )

    try:
        yield proc
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        try:
            os.unlink(log_path)
        except OSError:
            pass


def _prepare_capture_cwd(td_path: Path) -> Path:
    """The temp folder the server runs in, its own folders under it
    (constraint (2)); a note saying why it exists, for anyone who opens
    the log."""
    (td_path / "README.txt").write_text(
        "Working directory for capture-readme-screenshots.py: the server\n"
        "it starts keeps its config, state and runtime folders here, so\n"
        "the person's own (TLS, sign-in) are never loaded or written.\n"
    )
    return td_path


def _capture_each(page, base_url: str) -> None:
    """Capture the 10 README screenshots.

    Each shot maps to a row in ``docs/process/screenshots.md``.  Three
    capture modes:

      * ``locator.screenshot()`` -- exact element bounding box, for
        targeted slices (tab strip, sidebar column).
      * ``page.screenshot(clip=...)`` -- fixed-rectangle fallback
        when the locator misses (UI rename safety net).
      * ``page.screenshot(full_page=True)`` -- captures the entire
        scrollable document height beyond the viewport.  Used for
        the form-heavy tabs (Structure-optimization / Spectrum /
        Transport) so all workflow-group cards + the viewer + the
        issues panel fit in one image even when they stack past the
        viewport edge.  Per user 2026-06-24: "see if you can allow
        the web page to extend its vertical size such that you can
        include all the contents inside that frame."
    """
    def goto(path: str) -> None:
        page.goto(f"{base_url}{path}",
                  wait_until="networkidle", timeout=30_000)
        time.sleep(IDLE_AFTER_NAV_S)

    def shot(filename: str, locator=None,
             clip: dict | None = None,
             full_page: bool = False) -> None:
        out = IMG_DIR / filename
        if locator is not None and locator.count():
            locator.screenshot(path=str(out), type="png")
        elif clip is not None:
            page.screenshot(path=str(out), type="png", clip=clip)
        else:
            page.screenshot(path=str(out), type="png",
                            full_page=full_page)
        size_kb = out.stat().st_size / 1024
        print(f"  wrote {out.relative_to(REPO_ROOT)} ({size_kb:.0f} KB)")

    print("[1/10] hero-molbuilder.png")
    goto(f"/molbuilder?project={DEMO_PROJECT}&file={DEMO_STRUCTURE}")
    shot("hero-molbuilder.png")

    print("[2/10] tab-bar.png")
    goto("/molbuilder")
    nav = page.locator("nav, .tab-bar, .app-tabs, header").first
    shot("tab-bar.png", locator=nav,
         clip={"x": 0, "y": 0, "width": VIEWPORT_W, "height": 80})

    print("[3/10] sidebar-projects.png")
    # Show the BDT project with its structure/ subdir expanded so the
    # sidebar carries real content (post-cleanup state).
    goto(f"/molbuilder?project={DEMO_PROJECT}&"
         f"path=structure")
    sidebar = page.locator(
        ".projects-sidebar, #projects-sidebar, aside.sidebar, "
        "[data-projects-sidebar]"
    ).first
    shot("sidebar-projects.png", locator=sidebar,
         clip={"x": 0, "y": 0, "width": 360, "height": VIEWPORT_H})

    print("[4/10] molbuilder-workspace.png")
    # Workspace tab with the Au-BDT-Au junction loaded from BDT/
    # structure/.  Full-page so the right-hand commands stack
    # (Sources / Atom / Pose / Geom / Junction / Save) renders end-
    # to-end even when it overflows the viewport.
    goto(f"/molbuilder?project={DEMO_PROJECT}&file={DEMO_STRUCTURE}")
    shot("molbuilder-workspace.png", full_page=True)

    print("[5/10] structure-optimization-form.png")
    # Structure-optimization tab loaded with the BDT/structure/
    # geometry.  Full-page captures the workflow-group cards
    # (Profile / Stage / Budget) all the way down past viewport.
    goto(f"/structure-optimization?project={DEMO_PROJECT}&"
         f"file={DEMO_STRUCTURE}")
    shot("structure-optimization-form.png", full_page=True)

    print("[6/10] spectrum-form.png")
    # Spectrum tab loaded from BDT/spectrum/BDT-only/ -- a real
    # spectra.spectra.py example.  Full-page so the form's vertical
    # stack (Profile / Stage / Budget) is fully visible.
    goto(f"/spectrum-calculation?project={DEMO_PROJECT}&"
         f"path=spectrum/BDT-only/spectra.spectra.py")
    shot("spectrum-form.png", full_page=True)

    print("[7/10] transport-form.png")
    # Transport tab loaded with the Au-BDT-Au junction.  Full-page
    # so electrode region labels + the form + the viewer all fit.
    goto(f"/transport-calculation?project={DEMO_PROJECT}&"
         f"file={DEMO_STRUCTURE}")
    shot("transport-form.png", full_page=True)

    print("[8/10] results-trajectory.png")
    # Results-tab trajectory inspector pointed at TJ-BDT-Au111
    # (the real multi-stage optimization in the cleaned-up project).
    goto(f"/results?project={DEMO_PROJECT}&"
         f"path=optimization/TJ-BDT-Au111")
    shot("results-trajectory.png", full_page=True)

    print("[9/10] results-spectra.png")
    goto(f"/results?project={DEMO_PROJECT}&"
         f"path=spectrum/BDT-only/spectra.spectra.json")
    shot("results-spectra.png", full_page=True)

    print("[10/10] results-bundle-card.png")
    goto(f"/results?project={DEMO_PROJECT}&"
         f"path=optimization/BDT-withAuJunction")
    bundle = page.locator(
        ".bundle-card, [data-bundle-card], section.bundle"
    ).first
    shot("results-bundle-card.png", locator=bundle,
         clip={"x": VIEWPORT_W - 820, "y": VIEWPORT_H - 380,
               "width": 800, "height": 360})


def main() -> int:
    if not (REPO_ROOT / "molbuilder").is_dir():
        print(f"ERROR: {REPO_ROOT}/molbuilder/ not found; "
              f"is the script at the repo root?", file=sys.stderr)
        return 2
    if not (REPO_ROOT / "projects" / DEMO_PROJECT / "structure"
            / DEMO_STRUCTURE).is_file():
        print(f"ERROR: demo structure missing: "
              f"projects/{DEMO_PROJECT}/structure/{DEMO_STRUCTURE}",
              file=sys.stderr)
        return 2

    IMG_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        print(f"ERROR: playwright not importable ({e}).  Re-run on the "
              f"host env's own Python, the env active:", file=sys.stderr)
        print(f'  "${{CONDA_PREFIX:?}}/bin/python" '
              f"{Path(__file__).relative_to(REPO_ROOT)}",
              file=sys.stderr)
        return 2

    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    print(f"using port {port} -> {base_url}")

    with tempfile.TemporaryDirectory(prefix="molbuilder-capture-") as td:
        capture_cwd = _prepare_capture_cwd(Path(td))
        print(f"capture cwd: {capture_cwd}")

        with _serve(port, capture_cwd):
            print("molbuilder serve up; opening Chromium...")
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True)
                ctx = browser.new_context(
                    viewport={"width": VIEWPORT_W, "height": VIEWPORT_H},
                    device_scale_factor=2,
                )
                page = ctx.new_page()
                try:
                    _capture_each(page, base_url)
                finally:
                    ctx.close()
                    browser.close()

    n = len(list(IMG_DIR.glob("*.png")))
    print(f"\ndone.  {n} PNG(s) under {IMG_DIR.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
