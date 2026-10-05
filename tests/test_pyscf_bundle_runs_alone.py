"""The code a PySCF script imports runs beside the job with molbuilder absent.

`engines/pyscf.md` § 3: the script imports everything of molbuilder's it
runs from `mb_pyscf.pyz` beside it (`runwrap.PYSCF_COMPANIONS`), under
`molbuilder-pySCF`, where molbuilder is not installed.  This reproduces that -- the bundle in a folder with nothing
else, a python that cannot import molbuilder, the script's own head and import
lines (`pyscf.input.emit_script_head`, `emit_bundle_imports`) -- imports every
member, and writes a geometry the way the script does, which the package's own
codec reads back.

API-level, because the road's run of this needs PySCF: every PySCF run
through `jobset launch` proves it too (`test_pyscf_relaxation_outcome_e2e.py`,
the engine test in `test_molwatch_preview.py`).  This one runs in every
batch, and that is what catches a member growing an import of something that
does not travel -- `structure.py` and `sidecars/molstruct.py` are edited
often, and the cost would be every PySCF run stopping on its first lines.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np


def test_the_bundle_loads_alone_and_writes_a_pair_the_codec_reads(tmp_path):
    """MUTATION THIS MUST FAIL AGAINST: a member importing, at load, a
    molbuilder module the bundle does not carry -- ``from .runfiles import
    compose`` at the top of `structure.py`, which fails in the bundle with
    *attempted relative import with no known parent package*."""
    from molbuilder.pyscf.input import (_sidecar_for, emit_bundle_imports,
                                        emit_script_head)
    from molbuilder.runwrap import (PYSCF_BUNDLE, PYSCF_COMPANIONS,
                                    pyscf_bundle)
    from molbuilder.spectra.selection import select_modes
    from molbuilder.structure import FROZEN_LABEL, Structure
    from molbuilder.workingcopy_structure import StructureCodec

    (tmp_path / PYSCF_BUNDLE).write_bytes(pyscf_bundle())
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    # The condition has to be real, or this proves nothing.
    control = subprocess.run([sys.executable, "-c", "import molbuilder"],
                             cwd=tmp_path, env=env, capture_output=True,
                             text=True, timeout=120)
    assert control.returncode != 0, (
        "molbuilder is importable from the job's folder, so this test cannot "
        "reproduce a compute node.  Is it pip-installed?")

    # The structure as a calculation holds it -- the O held -- and the
    # payload the script carries for it, made at compose time.
    water = Structure(elements=["O", "H", "H"],
                      positions=[[0.0, 0.0, 0.119], [0.0, 0.757, -0.477],
                                 [0.0, -0.757, -0.477]],
                      regions={FROZEN_LABEL: [0]})
    relaxed = [[0.0, 0.0, 0.1173], [0.0, 0.7612, -0.4713],
               [0.0, -0.7612, -0.4713]]
    # The script's own head -- its anchor, the bundle on the path, the core
    # count -- then every member the bundle holds, each found where the
    # script's imports find it.
    members = sorted(name[:-len(".py")] for name in PYSCF_COMPANIONS)
    script = "\n".join([
        *emit_script_head(),
        *emit_bundle_imports(StructureCodec, select_modes),
        "import importlib, json, os",
        f"_found = {{m: os.path.basename(os.path.dirname("
        f"importlib.import_module(m).__file__)) for m in {members!r}}}",
        f"_mb_StructureCodec().write_moved(_mb_outfile('w_optimized.xyz'), "
        f"['O', 'H', 'H'], {relaxed!r}, {_sidecar_for(water)!r}, "
        f"comment='Optimized geometry (PySCF)')",
        "print(json.dumps({",
        "    'from': sorted(set(_found.values())),",
        "    'cores': _mb_physical_core_count() >= 1,",
        "    'selected': _mb_select_modes([412.3, 1023.4, 3656.0], 'all',",
        "                                 freq_min_cm1=800.0)}))",
    ])
    (tmp_path / "w.py").write_text(script)
    done = subprocess.run([sys.executable, "w.py"], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, (
        f"the PySCF bundle cannot be imported, or cannot write, without "
        f"molbuilder:\n{done.stdout}{done.stderr}")
    got = json.loads(done.stdout.strip().splitlines()[-1])
    assert got == {"from": [PYSCF_BUNDLE], "cores": True,
                   "selected": [2, 3]}, got

    # THE PACKAGE'S CODEC READS WHAT THE SHIPPED ONE WROTE: the moved
    # coordinates, the held atom, and the engine's origin.
    back = StructureCodec().load(tmp_path / "w_optimized.xyz")
    assert back.elements == ["O", "H", "H"]
    assert np.allclose(back.positions, relaxed, atol=1e-6)
    assert back.frozen_atoms == [0]
    assert np.array_equal(back.engine_offset, np.zeros(3))
