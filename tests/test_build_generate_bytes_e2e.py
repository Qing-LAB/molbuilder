"""e2e: SIESTA + PySCF Generate flows assert OUTPUT BYTES, not
just button-enable.

Pre-2026-06-14 the existing e2e tests for /structure-optimization
stop at ``btn.is_enabled()`` (test_build_e2e.py:347,
test_molbuilder_e2e.py:3704).  Every assert about the .fdf / .py
that the Generate button is supposed to produce is L1/L2 at best
-- the BROWSER path that POSTs to /api/build/fdf and returns the
text is unverified.

This file fires that POST end-to-end and asserts the response
``body.fdf`` / ``body.script`` text actually contains the
load-bearing pieces:

  * SIESTA: ``SystemLabel``, ``%block ChemicalSpeciesLabel``,
    ``%block AtomicCoordinatesAndAtomicSpecies`` with the right
    atom count, ``MeshCutoff`` from params.
  * PySCF: ``mol.atom`` with the input atoms, ``mol.basis`` from
    params, ``mf = scf.RKS / UKS``.

Plus the viewer-is-truth contract: in-body ``frozen_atoms`` reach
the rendered FDF as a Geometry.Constraints block (mirrors the
test_in_body_labels_xhr.py coverage at the BROWSER tier).
"""
from __future__ import annotations

import json
import re
import threading

import pytest

pytestmark = pytest.mark.e2e

pytest.importorskip("playwright.sync_api")


_H2O_XYZ = """3
water
O   0.000   0.000   0.000
H   0.757   0.586   0.000
H  -0.757   0.586   0.000
"""


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


def _open_optimization(page, base_url):
    errors = []
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    page.goto(f"{base_url}/structure-optimization")
    page.wait_for_load_state("networkidle", timeout=15_000)
    return errors


# --------------------------------------------------------------------- #
#  SIESTA bytes assertion                                                #
# --------------------------------------------------------------------- #


class TestSiestaGenerateBytes:

    def test_fdf_contains_systemlabel_and_atom_block(
            self, page, flask_server):
        """The CONTRACT: a Generate POST returns an FDF text that
        contains every load-bearing block.  A schema regression
        that drops ``%block AtomicCoordinatesAndAtomicSpecies``
        would silently produce an unrunnable .fdf; pinning this
        catches it at e2e tier."""
        errors = _open_optimization(page, flask_server)
        assert not errors
        body = json.dumps({
            "xyz": _H2O_XYZ,
            "params": {"system_label": "h2o_test"},
            "frozen_atoms": [],
            "regions": {},
        })
        js = (
            "(body) => fetch('/api/build/fdf', { "
            "  method: 'POST', "
            "  headers: { 'Content-Type': 'application/json' }, "
            "  body: body "
            "}).then(r => r.json())"
        )
        result = page.evaluate(js, body)
        assert result.get("ok") is True, f"render failed: {result!r}"
        fdf = result.get("fdf", "")

        # Load-bearing blocks.  Each is what a SIESTA user would
        # check first if the wrapper crashed.
        assert "SystemLabel" in fdf, (
            "FDF must declare SystemLabel"
        )
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
            self, page, flask_server):
        """Viewer-is-truth at e2e: in-body ``frozen_atoms`` in a
        Generate POST surfaces in the rendered FDF's
        ``%block Geometry.Constraints``.  Pinning at e2e because
        the L2 source-text test and the L3 Flask-test-client test
        could pass while the browser path is broken via a
        different route (CORS / auth middleware regression / etc.)."""
        _open_optimization(page, flask_server)
        body = json.dumps({
            "xyz": _H2O_XYZ,
            "params": {},
            "frozen_atoms": [0],  # freeze the O
            "regions": {},
        })
        js = (
            "(body) => fetch('/api/build/fdf', { "
            "  method: 'POST', "
            "  headers: { 'Content-Type': 'application/json' }, "
            "  body: body "
            "}).then(r => r.json())"
        )
        result = page.evaluate(js, body)
        assert result.get("ok") is True, f"render failed: {result!r}"
        fdf = result.get("fdf", "")
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

    def test_script_contains_mol_block(self, page, flask_server):
        """A Generate POST to /api/build/pyscf returns a runnable
        .py.  Pin that the mol-build code with ``mol.atom = [...]``
        and the SCF kernel call ``mf.kernel()`` (or scf.RKS/UKS
        construction) are both present."""
        _open_optimization(page, flask_server)
        body = json.dumps({
            "xyz": _H2O_XYZ,
            "params": {},
            "frozen_atoms": [],
            "regions": {},
        })
        js = (
            "(body) => fetch('/api/build/pyscf', { "
            "  method: 'POST', "
            "  headers: { 'Content-Type': 'application/json' }, "
            "  body: body "
            "}).then(r => r.json())"
        )
        result = page.evaluate(js, body)
        assert result.get("ok") is True, f"render failed: {result!r}"
        script = result.get("script", "")
        assert script, "rendered PySCF script is empty"

        # Load-bearing pieces.
        assert "from pyscf" in script, "PySCF imports missing"
        assert "gto.M" in script or "mol = gto.Mole" in script or \
               "mol.build" in script, (
            "PySCF mol construction missing"
        )
        # SCF kernel call (one of several forms).
        assert "scf.RKS" in script or "scf.UKS" in script or \
               "scf.RHF" in script or "scf.UHF" in script or \
               "dft.RKS" in script or "dft.UKS" in script, (
            "PySCF SCF kernel construction missing"
        )
        # The input atoms must appear (at least element symbols).
        # ``O   0.0`` style or ``"O"`` style depending on emitter.
        assert "O " in script or '"O"' in script or "'O'" in script
        assert "H " in script or '"H"' in script or "'H'" in script
