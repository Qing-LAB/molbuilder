"""Third-party browser assets must retain notices in source and wheels."""
from __future__ import annotations

from pathlib import Path
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "molbuilder" / "web" / "static" / "vendor"


def test_vendor_inventory_lists_every_shipped_library():
    inventory = (VENDOR / "README.md").read_text(encoding="utf-8")
    for name in (
        "3Dmol.js", "gif.js", "CodeMirror", "DOMPurify", "GitGraph",
        "Marked", "Mermaid", "Plotly.js",
    ):
        assert name in inventory


def test_vendor_notices_include_complete_license_texts():
    notices = {
        "LICENSE-3Dmol.txt": "Redistribution and use in source and binary forms",
        "gif.min.js.LICENSE.txt": "Permission is hereby granted",
        "codemirror/LICENSE": "Permission is hereby granted",
        "dompurify/LICENSE": "Apache License",
        "gitgraph/LICENSE": "Permission is hereby granted",
        "marked/LICENSE": "Permission is hereby granted",
        "mermaid/LICENSE": "Permission is hereby granted",
        "LICENSE-plotly.txt": "Permission is hereby granted",
    }
    for rel, required_text in notices.items():
        assert required_text in (VENDOR / rel).read_text(encoding="utf-8")


@pytest.mark.slow
def test_vendor_assets_and_notices_are_packaged(tmp_path):
    """**Build the wheel and look inside it.**

    Rewritten 2026-09-04, and the old version is the reason.  It asserted
    two literal strings in `pyproject.toml`::

        assert "web/static/vendor/*"   in patterns
        assert "web/static/vendor/*/*" in patterns

    On 2026-09-02 those two were replaced by one recursive
    `web/static/**/*` -- which packages strictly MORE (it cannot miss a
    vendor subdirectory somebody adds next year) -- and this test went red
    on a change that made the thing it guards better.  It then sat red,
    because a test that fails for the wrong reason teaches nobody to look.

    The property is not "pyproject contains these two globs".  It is
    **third-party code ships with its notice**, which is a licensing
    obligation and is a fact about the ARTIFACT.  So this builds one.

    ~7 s, hence `slow`.  Worth it: the sibling test in
    `test_wheel_ships_the_front_end.py` simulates packaging by matching
    patterns with its own matcher, and a simulation is what let a wheel
    ship 90 of 141 static files (2026-09-02).  This is the only test in
    the suite that opens the real thing.

    **A correction, because this docstring carried a false measurement
    for part of 2026-09-04.**  It claimed that on setuptools 82.0.1 the
    `package-data` patterns "DO NOT DECIDE what ships" -- that the wheel
    carried all 141 static files with every pattern deleted, so the
    pattern-matching tests next door were "guarding a lever that is not
    connected".

    That was wrong, and the way it was wrong is worth more than the
    claim.  The experiment ran in the WORKING TREE, which holds a stale
    `molbuilder.egg-info/SOURCES.txt` listing every file from an earlier
    build; setuptools reads it and ships what it names, so deleting the
    patterns changed nothing.  Re-run on a clean tree
    (`git archive HEAD | tar -x -C tmp`, no egg-info, no build/):

    * with the patterns:     141 static files, 17 templates
    * patterns deleted:      **0 static files, 0 templates**

    So the patterns decide everything, and the sibling pattern tests in
    `test_wheel_ships_the_front_end.py` guard a lever that is very much
    connected.  This test still earns its ~7 s -- a simulation of
    packaging is what let a wheel ship 90 of 141 static files
    (2026-09-02), and this is the only test in the suite that opens the
    real artifact -- but it is a SECOND line of defence, not a
    replacement for them.

    The general rule, which cost a day to relearn: a packaging
    measurement taken in a dirty tree measures the tree's history, not
    the build.
    """
    import subprocess
    import sys
    import zipfile

    out = tmp_path / "dist"
    out.mkdir()
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "-q",
         "-w", str(out), str(ROOT)],
        capture_output=True, text=True, timeout=600)
    wheels = sorted(out.glob("*.whl"))
    if not wheels:
        pytest.skip(f"could not build a wheel here: {proc.stderr[-400:]}")

    shipped = set(zipfile.ZipFile(wheels[0]).namelist())
    on_disk = sorted(p.relative_to(ROOT / "molbuilder")
                     for p in VENDOR.rglob("*") if p.is_file())
    missing = [str(r) for r in on_disk
               if f"molbuilder/{r.as_posix()}" not in shipped]
    assert not missing, (
        f"{len(missing)} vendored file(s) are not in the wheel: {missing}.  "
        f"Third-party code must ship with its notice; a package-data "
        f"pattern that stops matching them is a licensing defect, not a "
        f"packaging nit.")

    notices = [r for r in on_disk
               if "LICENSE" in r.name.upper() or "README" in r.name.upper()]
    assert notices, (
        "no LICENSE/README files found under web/static/vendor, so the "
        "check above proved nothing about notices")
    assert all(f"molbuilder/{r.as_posix()}" in shipped for r in notices), (
        f"a vendored notice is missing from the wheel: "
        f"{[str(r) for r in notices if f'molbuilder/{r.as_posix()}' not in shipped]}")
