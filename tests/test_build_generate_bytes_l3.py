"""L3 interface tests: SIESTA + PySCF Generate POST bytes contracts.

History (J6 round-3, 2026-06-14)
================================

Pre-J6 this file's predecessor was named
``test_build_generate_bytes_e2e.py`` and drove a real Playwright
browser to call ``fetch('/api/build/fdf')`` from JS.  Every
assertion is on the SERVER RESPONSE body (``r.json()``); the
browser was used only to dispatch the POST.  Per the round-3 test-
pyramid audit + ``docs/process/testing.md`` § 4, that's L5
when an L3 (Flask test client) would pin the exact same contract
at ~10x the speed.

Demoted shape:

  * Flask test_client() POST instead of Playwright page.evaluate fetch.
  * No module-level Flask server fixture; web_client (conftest.py) is
    enough.
  * Same JSON body, same response-shape assertions.
  * No browser → no Chromium startup, no headless flake.

The L5 coverage that was lost (browser-CORS / auth-middleware
regressions on the build endpoints) is covered by the smoke e2e in
test_workspace_dispatcher_mount_e2e.py (loads /molbuilder + verifies
mount) + the actually-load-bearing e2es (spectrum + transport
Generate flows).  This file is pure wire-shape verification.

What this file pins
===================

* SIESTA: ``/api/build/fdf`` returns ``%block ChemicalSpeciesLabel``,
  ``%block AtomicCoordinatesAndAtomicSpecies`` with the right atom
  count, ``SystemLabel`` from params reaches the rendered FDF.

* SIESTA viewer-is-truth: in-body ``frozen_atoms: [0]`` lands in
  ``%block Geometry.Constraints`` with 1-based-index ``1``.

* PySCF: ``/api/build/pyscf`` returns a non-empty script containing
  ``from pyscf``, a ``gto`` mol-construction call, and an SCF
  kernel construction.  Input atoms (O, H) appear.
"""
from __future__ import annotations

import json
import re

import pytest

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from support.envelope import (from_xyz as _env,
                             from_xyz_with_periodicity as _env_per)



pytest.importorskip("flask")


# Water, made slightly NON-planar (one H off the z=0 plane) so the derived vacuum
# cell isn't degenerate at vacuum=0 (a planar molecule has a zero-thickness axis --
# structure-periodicity.md).  These tests exercise the generate-bytes path, not geometry.
_H2O_XYZ = """3
water
O   0.000   0.000   0.000
H   0.757   0.586   0.300
H  -0.757   0.586  -0.300
"""


# --------------------------------------------------------------------- #
#  SIESTA bytes assertion                                                #
# --------------------------------------------------------------------- #


class TestSiestaGenerateBytes:

    def test_fdf_contains_systemlabel_and_atom_block(self, web_client):
        """A Generate POST returns an FDF text that contains every
        load-bearing block.  A schema regression that drops
        ``%block AtomicCoordinatesAndAtomicSpecies`` would silently
        produce an unrunnable .fdf; this pins it at the wire tier.
        """
        r = web_client.post(
            "/api/build/fdf",
            json={
                # NO LABEL KEYS.  They were `frozen_atoms: []` + `regions: {}`,
                # which read as "I declare nothing is labelled" and in fact
                # declared nothing at all: nothing has read a top-level label
                # key since `apply_labels_to_struct` was deleted.  This test is
                # about the EMITTED BYTES for an unlabelled structure, and it
                # says that better by not pretending to send a declaration.
                "structure": _env(_H2O_XYZ),
                "params": {"system_label": "h2o_test"},
            },
        )
        assert r.status_code == 200, r.get_data(as_text=True)
        body = r.get_json()
        assert body.get("ok") is True, f"render failed: {body!r}"
        fdf = body.get("fdf", "")

        # Load-bearing blocks.  Each is what a SIESTA user would
        # check first if the wrapper crashed.
        assert "SystemLabel" in fdf, "FDF must declare SystemLabel"
        assert "h2o_test" in fdf, (
            "user-provided SystemLabel must reach the FDF"
        )
        assert "%block ChemicalSpeciesLabel" in fdf, (
            "FDF must define species via ChemicalSpeciesLabel"
        )
        assert "%block AtomicCoordinatesAndAtomicSpecies" in fdf, (
            "FDF must include the atom block"
        )
        # 3-atom water -> 3 coordinate lines inside the atom block.
        m = re.search(
            r"%block\s+AtomicCoordinatesAndAtomicSpecies\s*\n"
            r"(.*?)%endblock\s+AtomicCoordinatesAndAtomicSpecies",
            fdf, re.IGNORECASE | re.DOTALL,
        )
        assert m is not None
        coord_lines = [
            ln for ln in m.group(1).splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
        ]
        assert len(coord_lines) == 3, (
            f"3-atom water must produce 3 atom rows; got "
            f"{len(coord_lines)}"
        )

    def test_fdf_frozen_atoms_in_body_reach_constraints_block(
            self, web_client):
        """Viewer-is-truth: in-body ``frozen_atoms`` in a Generate
        POST surfaces in the rendered FDF's
        ``%block Geometry.Constraints``.  The L2 source-text test
        (test_in_body_labels_contract.py) pins that the helper is
        wired into every emitter; this L3 pins the actual byte
        output of one emitter.
        """
        # ONE LABEL STORE, INSIDE THE STRUCTURE.  `regions` IS the whole store
        # and frozen atoms are a label in it; the store lives in the structure's
        # own `metadata`, which is where `Structure.from_dict` reads it.
        #
        # Two shapes preceded this.  A TOP-LEVEL `frozen_atoms` beside
        # `regions: {}` said "freeze atom 0" and "nothing is labelled" in the
        # same breath (retired 2026-07-31).  Then a top-level `regions` beside
        # an `xyz` string, which only worked because a second applier existed on
        # the server to move it onto the structure -- deleted 2026-08-03.
        lines = [ln.split() for ln in _H2O_XYZ.strip().splitlines()[2:]
                 if len(ln.split()) == 4]
        r = web_client.post(
            "/api/build/fdf",
            json={
                "structure": {
                    "elements":  [p[0] for p in lines],
                    "positions": [[float(p[1]), float(p[2]), float(p[3])]
                                  for p in lines],
                    "metadata":  {"regions": {"frozen_atoms": [0]}},  # the O
                },
                "params": {},
            },
        )
        assert r.status_code == 200, r.get_data(as_text=True)
        body = r.get_json()
        assert body.get("ok") is True, f"render failed: {body!r}"
        fdf = body.get("fdf", "")
        # 0-based 0 -> 1-based 1.
        assert "%block Geometry.Constraints" in fdf, (
            "in-body frozen_atoms must produce a Constraints block"
        )
        m = re.search(
            r"%block\s+Geometry\.Constraints(.+?)%endblock",
            fdf, re.IGNORECASE | re.DOTALL,
        )
        assert m is not None
        # Parse only position-line tokens.
        emitted = set()
        for line in m.group(1).splitlines():
            stripped = line.split("#", 1)[0].strip()
            if stripped.lower().startswith("position"):
                for tok in stripped.split()[1:]:
                    if tok.isdigit():
                        emitted.add(int(tok))
        assert emitted == {1}, (
            f"in-body [0] should map to 1-based {{1}}; got {emitted}"
        )


# --------------------------------------------------------------------- #
#  PySCF bytes assertion                                                 #
# --------------------------------------------------------------------- #


class TestPyscfGenerateBytes:

    def test_script_contains_mol_block(self, web_client):
        """A Generate POST to /api/build/pyscf returns a runnable
        .py.  Pin that the mol-build code with ``gto.M`` (or the
        equivalent) and the SCF kernel construction are both
        present.
        """
        r = web_client.post(
            "/api/build/pyscf",
            json={
                "structure": _env(_H2O_XYZ),          # unlabelled; see the note above
                "params": {},
            },
        )
        assert r.status_code == 200, r.get_data(as_text=True)
        body = r.get_json()
        assert body.get("ok") is True, f"render failed: {body!r}"
        script = body.get("script", "")
        assert script, "rendered PySCF script is empty"

        # Load-bearing pieces.
        assert "from pyscf" in script, "PySCF imports missing"
        assert (
            "gto.M" in script
            or "mol = gto.Mole" in script
            or "mol.build" in script
        ), "PySCF mol construction missing"
        # SCF kernel call (one of several forms).
        assert (
            "scf.RKS" in script or "scf.UKS" in script
            or "scf.RHF" in script or "scf.UHF" in script
            or "dft.RKS" in script or "dft.UKS" in script
        ), "PySCF SCF kernel construction missing"
        # The input atoms must appear (at least element symbols).
        # ``O   0.0`` style or ``"O"`` style depending on emitter.
        assert "O " in script or '"O"' in script or "'O'" in script
        assert "H " in script or '"H"' in script or "'H'" in script
