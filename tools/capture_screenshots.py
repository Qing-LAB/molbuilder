"""Capture the three Watch / Modify / Build screenshots used in the
top-level README.

Run from the repo root::

    python tools/capture_screenshots.py

Writes PNGs to ``docs/img/``.  Requires playwright + chromium
(already installed for the pytest-playwright E2E suite).

The screenshots are deliberately staged with small, recognisable
inputs so a reader on GitHub gets the gist in one glance:

* Build tab  -- a peptide built from the sequence ``ARNDC`` with
                the SIESTA panel populated at the production-stage
                defaults.
* Modify tab -- water loaded so the atom list is non-empty and the
                viewer renders something; the Atom subtab is the
                landing view.
* Watch tab  -- a tiny synthetic .molwatch.log so the energy /
                force plots have a couple of points and the
                Inspect panel is reachable.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

from werkzeug.serving import make_server

from molbuilder.web import create_app


def _start_server(port: int):
    app = create_app()
    server = make_server("127.0.0.1", port, app)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    # Give Werkzeug a moment to bind.
    time.sleep(0.4)
    return server


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    img_dir   = repo_root / "docs" / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

    port = 5050
    server = _start_server(port)
    base = f"http://127.0.0.1:{port}"

    # Small, deterministic fixtures.
    fixture_dir = repo_root / "docs" / "img" / "_fixtures"
    fixture_dir.mkdir(exist_ok=True)
    water_xyz = fixture_dir / "water.xyz"
    water_xyz.write_text(
        "3\nh2o\n"
        "O 0.000  0.000 0.000\n"
        "H 0.957  0.000 0.000\n"
        "H -0.240 0.927 0.000\n"
    )
    # A two-step molwatch.log so the energy plot has a slope.
    sample_log = fixture_dir / "demo.molwatch.log"
    sample_log.write_text(
        "# molwatch trajectory log v1\n"
        "# engine: pyscf\n"
        "==== molwatch step 0 begin ====\n"
        "step_index: 0\n"
        "n_atoms: 3\n"
        "coordinates (Ang):\n"
        "   O   0.00000000   0.00000000   0.00000000\n"
        "   H   0.95700000   0.00000000   0.00000000\n"
        "   H  -0.23900000   0.92700000   0.00000000\n"
        "energy (eV): -2073.10000000\n"
        "max_force (eV/Ang): 0.04800000\n"
        "forces (eV/Ang):\n"
        "   O  -0.01200000   0.00500000   0.00000000\n"
        "   H   0.00600000  -0.00250000   0.00000000\n"
        "   H   0.00600000  -0.00250000   0.00000000\n"
        "==== molwatch step 0 end ====\n"
        "==== molwatch step 1 begin ====\n"
        "step_index: 1\n"
        "n_atoms: 3\n"
        "coordinates (Ang):\n"
        "   O   0.00000000   0.00000000   0.00000000\n"
        "   H   0.95700000   0.00000000   0.00000000\n"
        "   H  -0.23900000   0.92700000   0.00000000\n"
        "energy (eV): -2073.18000000\n"
        "max_force (eV/Ang): 0.01200000\n"
        "forces (eV/Ang):\n"
        "   O  -0.00300000   0.00100000   0.00000000\n"
        "   H   0.00150000  -0.00050000   0.00000000\n"
        "   H   0.00150000  -0.00050000   0.00000000\n"
        "==== molwatch step 1 end ====\n"
    )

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright not installed; pip install pytest-playwright", file=sys.stderr)
        return 1

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1500, "height": 950},
                                  device_scale_factor=2)
        page = ctx.new_page()

        # ---- Build tab -----------------------------------------------
        page.goto(f"{base}/", wait_until="domcontentloaded")
        page.wait_for_selector("#input-text")
        page.fill("#input-text", "ARNDC")
        page.click("#build-btn")
        # Wait for the n-atom readout to flip from "—" to a number.
        page.wait_for_function(
            "() => /^[0-9]/.test(document.getElementById('info-atoms').textContent)",
            timeout=10000,
        )
        page.wait_for_timeout(400)        # let 3Dmol settle
        page.screenshot(path=str(img_dir / "build-tab.png"))
        print("  ✓ build-tab.png")

        # ---- Modify tab ----------------------------------------------
        page.goto(f"{base}/modify", wait_until="domcontentloaded")
        page.set_input_files("#file-picker", str(water_xyz))
        page.wait_for_function(
            "() => document.querySelectorAll('#atom-list-body tr').length === 3",
            timeout=5000,
        )
        # Click the oxygen to demonstrate selection + halo.
        page.locator("#atom-list-body tr").nth(0).click()
        page.wait_for_timeout(300)
        page.screenshot(path=str(img_dir / "modify-tab.png"))
        print("  ✓ modify-tab.png")

        # ---- Watch tab ------------------------------------------------
        page.goto(f"{base}/watch", wait_until="domcontentloaded")
        page.fill("#path-input", str(sample_log))
        page.click("#load-btn")
        page.wait_for_function(
            "() => document.querySelector('#frame-tot') && "
            "      document.querySelector('#frame-tot').textContent === '1'",
            timeout=5000,
        )
        # Open the Inspect tab so the panel is visible.
        page.locator(".ctab[data-tab='inspect']").click()
        page.wait_for_timeout(400)
        page.screenshot(path=str(img_dir / "watch-tab.png"))
        print("  ✓ watch-tab.png")

        browser.close()

    server.shutdown()

    # Clean up the temporary fixtures dir (we don't want it in git).
    for f in fixture_dir.iterdir():
        f.unlink()
    fixture_dir.rmdir()

    print(f"\nScreenshots saved to {img_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
