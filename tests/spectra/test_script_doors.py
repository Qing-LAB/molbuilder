"""The door a person's script is told to read a vibration result by loads
as its FIRST import (`engines/vibration.md` § 6.8): in a fresh python,
nothing of molbuilder loaded before it.

THE SILENT FAILURE: an import cycle among molbuilder's own modules that only
shows when one of them is loaded first -- the CLI, the web app and the test
suite all load the parse package before it, so none of them meets it -- and
a person's script stopping at its first line, in words about a module it
never named (2.77.26's round 1: the result's module loading the parse
package, whose registered spectra parser imported the reader still
half-loaded).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_the_results_reader_loads_as_a_scripts_first_import(tmp_path):
    """§ 6.8: molbuilder's own read of the result,
    `molbuilder.sidecars.spectra.parse_spectra_json`."""
    proc = subprocess.run(
        [sys.executable, "-I", "-c",
         f"import sys; sys.path.insert(0, {str(ROOT)!r}); "
         "from molbuilder.sidecars.spectra import parse_spectra_json; "
         "print('loaded')"],
        cwd=tmp_path, capture_output=True, text=True, timeout=120,
        env={"HOME": str(tmp_path), "XDG_CONFIG_HOME": str(tmp_path / "c"),
             "XDG_STATE_HOME": str(tmp_path / "s"), "PATH": "/usr/bin:/bin"})
    assert proc.returncode == 0 and "loaded" in proc.stdout, \
        proc.stderr[-2000:]
