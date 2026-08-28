"""L4 smoke test: run real SIESTA 5.4.2 on the molbuilder-emitted
TranSIESTA .fdf.

This is the "fact-check the agent" test the 2026-06-18 TranSIESTA
modernization wanted: it generates an Au-BDT-Au transport .fdf via
:class:`molbuilder.transport.transiesta.TransiestaEngine.render_script`,
hands it to the actual SIESTA binary, and asserts SIESTA's parser
got through the device .fdf without complaining about unknown
keywords / unrecognised values.

The run is EXPECTED TO FAIL — we use empty placeholder .TSHS files
that SIESTA correctly rejects at the TSHS-version check.  That
failure proves parse-clean: SIESTA had to parse every
``%block TS.Elec.<name>`` / ``%block TS.ChemPot.<name>`` block,
match the chem-pot bindings, and walk the device geometry before
it ever opened a .TSHS file.

Skips when the ``molbuilder-siesta`` conda env's siesta binary
isn't available (CI / dev machines without the SIESTA
installation).  Pseudopotentials are taken from
``projects/pseudopotential/`` (the in-repo PSML set).

Why L4: this is a long-tail integration test — uses a real
external binary, takes ~10 seconds wall-clock, and gives a
high-value durable guard against ANY future TranSIESTA syntax
drift that the unit tests would miss.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest


_SIESTA_BIN = (
    Path.home() / "miniconda3" / "envs" / "molbuilder-siesta" / "bin" / "siesta"
)
_PSML_DIR = Path(__file__).parent.parent / "projects" / "pseudopotential"
_AU_BDT_XYZ = Path(__file__).parent / "data" / "au_bdt_au.xyz"
_AU_BDT_SIDECAR = Path(__file__).parent / "data" / "au_bdt_au.molstruct.json"


def _siesta_available() -> bool:
    """True iff SIESTA binary + required pseudos are present.

    Required PSML files: Au, C, H, S (the Au-BDT-Au fixture).
    """
    if not _SIESTA_BIN.is_file():
        return False
    if not os.access(_SIESTA_BIN, os.X_OK):
        return False
    for elem in ("Au", "C", "H", "S"):
        if not (_PSML_DIR / f"{elem}.psml").is_file():
            return False
    return True


@pytest.mark.skipif(
    not _siesta_available(),
    reason=(
        f"SIESTA binary or PSML pseudos not available "
        f"({_SIESTA_BIN} / {_PSML_DIR}); skipping L4 SIESTA smoke."
    ),
)
def test_emitted_fdf_parses_cleanly_in_siesta_5_4_2(tmp_path):
    """Generate Au-BDT-Au transport .fdf via molbuilder; SIESTA must
    parse every keyword / block without "Unrecognized" / "does not
    exist" errors; failure must be at TSHS-read (the only step our
    empty placeholders block).
    """
    from molbuilder.config.transport import TransportConfig
    from molbuilder.structure import Structure
    from molbuilder.transport.transiesta import TransiestaEngine

    # 1. Load the Au-BDT-Au fixture.
    xyz_lines = _AU_BDT_XYZ.read_text().splitlines()
    n = int(xyz_lines[0])
    elements, positions = [], []
    for line in xyz_lines[2 : 2 + n]:
        parts = line.split()
        elements.append(parts[0])
        positions.append([float(x) for x in parts[1:4]])
    regions = json.loads(_AU_BDT_SIDECAR.read_text())["regions"]
    struct = Structure(
        elements=elements,
        positions=np.array(positions),
        regions=regions,
    )

    # 2. Emit the device .fdf.
    cfg = TransportConfig(job_name="au_bdt_au_smoke")
    fdf = TransiestaEngine.render_script(struct, cfg)
    (tmp_path / "device.fdf").write_text(fdf)

    # 3. Stage pseudos (symlinks to the in-repo PSML set).
    for elem in ("Au", "C", "H", "S"):
        (tmp_path / f"{elem}.psml").symlink_to(_PSML_DIR / f"{elem}.psml")

    # 4. Empty placeholder .TSHS files.  Names match what the
    #    emitter wrote into ``TS.Elec.<name>.HS`` lines:
    #    ``<job_name>_<region-label>.TSHS``.  Au-BDT-Au has
    #    L-electrode + R-electrode regions.
    (tmp_path / f"{cfg.job_name}_L-electrode.TSHS").touch()
    (tmp_path / f"{cfg.job_name}_R-electrode.TSHS").touch()

    # 5. Run SIESTA.  Expect non-zero exit; capture output.
    result = subprocess.run(
        [str(_SIESTA_BIN), "device.fdf"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    combined = result.stdout + result.stderr

    # 6. Assertions.
    #    The .fdf is INTENTIONALLY broken (empty .TSHS files), so
    #    SIESTA MUST exit non-zero — but at a SPECIFIC failure mode.
    assert result.returncode != 0, (
        "SIESTA returned 0 — the empty .TSHS files should have "
        "triggered failure.  The test setup is wrong if this "
        "happens; investigate."
    )

    #    Parse-level failures we MUST NOT see:
    parse_errors = [
        "Unrecognized",
        "does not exist",
        "Unknown",
        "syntax error",
        "Could not interpret",
    ]
    for needle in parse_errors:
        assert needle not in combined, (
            f"SIESTA flagged a parse-level error ({needle!r}) in "
            f"the molbuilder-emitted .fdf.  This is the bug class "
            f"the test exists to catch: the TranSIESTA syntax has "
            f"drifted from what SIESTA 5.4.2 accepts.  Full output "
            f"snippet:\n"
            + "\n".join(combined.splitlines()[-30:])
        )

    #    Expected failure mode: TSHS read failed.
    assert "TSHS" in combined, (
        "SIESTA neither parsed-cleanly nor failed at the TSHS "
        "step.  Something unexpected went wrong; investigate the "
        "full output."
    )
