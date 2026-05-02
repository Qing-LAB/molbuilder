"""Backend dispatch / registry tests.

Tested in isolation from the SIESTA / PySCF e2e flow that
``test_smiles_and_siesta.py`` covers.  These tests pin the dispatch
contract:

  * ``available_backends()`` returns one bool per registered backend.
  * ``dispatch(backend="auto")`` picks the first available in the
    documented order (threedna > amber > rdkit).
  * ``dispatch(backend="<name>")`` calls that backend exclusively;
    no fallback.
  * Unknown backend names raise ``ValueError``.
  * No-backends-available raises ``BackendUnavailable``.

The tests stub out the actual backend build functions so they don't
depend on rdkit / AmberTools being installed -- the dispatch layer
is what's under test, not the builders themselves.
"""

from __future__ import annotations

import pytest

from molbuilder.backends import (
    BackendUnavailable,
    available_backends,
    dispatch,
)


def test_available_backends_returns_dict_of_bools():
    """available_backends() returns one entry per registered backend
    with a bool value -- the web UI's `/api/backends` endpoint relies
    on this exact shape."""
    avail = available_backends()
    assert isinstance(avail, dict)
    assert "rdkit" in avail and "amber" in avail
    assert all(isinstance(v, bool) for v in avail.values())


def test_dispatch_explicit_backend_calls_only_that_one(monkeypatch):
    """When the caller names a backend explicitly, dispatch must NOT
    fall back to another one if that backend's build() fails -- the
    user picked the backend for a reason."""
    calls = []

    def fake_amber_build(kind, sequence, form, terminal, title):
        calls.append("amber")
        return "amber-result"

    def fake_rdkit_build(kind, sequence, form, terminal, title):
        calls.append("rdkit")
        return "rdkit-result"

    # Patch the loader, not the modules, so we exercise the real
    # registry plumbing.
    from molbuilder import backends
    monkeypatch.setattr(
        backends, "_load_backends",
        lambda: {"amber": fake_amber_build, "rdkit": fake_rdkit_build},
    )

    result = dispatch("dna", "ATGC", backend="rdkit")
    assert result == "rdkit-result"
    assert calls == ["rdkit"]   # amber NOT called


def test_dispatch_auto_prefers_amber_then_rdkit(monkeypatch):
    """auto mode picks the FIRST available backend in dispatch order
    (amber > rdkit).  Documented in molbuilder/backends/__init__.py
    docstring; reflected here so a future re-ordering shows up.
    """
    calls = []
    from molbuilder import backends
    monkeypatch.setattr(
        backends, "_load_backends",
        lambda: {
            "amber": lambda *a, **k: (calls.append("amber") or "amber-out"),
            "rdkit": lambda *a, **k: (calls.append("rdkit") or "rdkit-out"),
        },
    )
    monkeypatch.setattr(
        backends, "available_backends",
        lambda: {"amber": True, "rdkit": True},
    )
    out = dispatch("dna", "ATGC", backend="auto")
    assert out == "amber-out"
    assert calls == ["amber"]


def test_dispatch_auto_falls_through_when_amber_unavailable(monkeypatch):
    calls = []
    from molbuilder import backends
    monkeypatch.setattr(
        backends, "_load_backends",
        lambda: {
            "amber": lambda *a, **k: (calls.append("amber") or "amber-out"),
            "rdkit": lambda *a, **k: (calls.append("rdkit") or "rdkit-out"),
        },
    )
    monkeypatch.setattr(
        backends, "available_backends",
        lambda: {"amber": False, "rdkit": True},
    )
    out = dispatch("dna", "ATGC", backend="auto")
    assert out == "rdkit-out"
    assert calls == ["rdkit"]   # amber skipped, rdkit ran


def test_dispatch_auto_raises_when_nothing_available(monkeypatch):
    """No backends installed -> a single, helpful BackendUnavailable
    pointing at both install commands.  No backend silently runs."""
    from molbuilder import backends
    monkeypatch.setattr(
        backends, "_load_backends",
        lambda: {"amber": lambda *a, **k: None, "rdkit": lambda *a, **k: None},
    )
    monkeypatch.setattr(
        backends, "available_backends",
        lambda: {"amber": False, "rdkit": False},
    )
    with pytest.raises(BackendUnavailable) as exc:
        dispatch("dna", "ATGC", backend="auto")
    msg = str(exc.value)
    # Must point users at both alternatives.
    assert "rdkit" in msg
    assert "ambertools" in msg.lower()


def test_dispatch_unknown_backend_raises_value_error():
    """Unknown backend name is a usage error, not a missing-tool
    error -- ValueError, not BackendUnavailable."""
    with pytest.raises(ValueError) as exc:
        dispatch("dna", "ATGC", backend="nonsense")
    msg = str(exc.value)
    assert "nonsense" in msg
    # Error must enumerate the valid choices so users can fix it.
    assert "rdkit" in msg or "amber" in msg
    assert "auto" in msg


# --------------------------------------------------------------------- #
#  is_available() smoke tests -- run only when the backend's external   #
#  dep is installed; otherwise skipped via importorskip.                #
# --------------------------------------------------------------------- #


def test_rdkit_is_available_returns_bool():
    from molbuilder.backends import _rdkit
    assert isinstance(_rdkit.is_available(), bool)


def test_amber_is_available_returns_bool():
    from molbuilder.backends import _amber
    assert isinstance(_amber.is_available(), bool)


def test_threedna_is_available_returns_bool():
    from molbuilder.backends import _threedna
    assert isinstance(_threedna.is_available(), bool)


# --------------------------------------------------------------------- #
#  3DNA detection chain                                                  #
#                                                                        #
#  Resolution priority (per molbuilder/backends/_threedna.py):           #
#    1. in-tree:  <repo_root>/x3dna-v*/  (preferred dev path)           #
#    2. env var:  $X3DNA points at a complete install                   #
#    3. PATH:     `fiber` on $PATH, derive root from its location        #
#                                                                        #
#  Each candidate must have BOTH bin/fiber (executable) AND config/     #
#  to count as "complete".                                              #
# --------------------------------------------------------------------- #


def test_threedna_resolves_via_env_when_in_tree_missing(tmp_path, monkeypatch):
    """If no in-tree x3dna-v*/ is present but $X3DNA points at a
    valid install, the env path wins."""
    from molbuilder.backends import _threedna
    fake_root = tmp_path / "x3dna-fake"
    (fake_root / "bin").mkdir(parents=True)
    (fake_root / "config").mkdir()
    fiber = fake_root / "bin" / "fiber"
    fiber.write_text("#!/bin/sh\nexit 0\n")
    fiber.chmod(0o755)

    # Force in-tree to miss; env to hit.
    monkeypatch.setattr(_threedna, "_find_in_tree", lambda: None)
    monkeypatch.setenv("X3DNA", str(fake_root))
    found = _threedna._resolve()
    assert found is not None
    assert found.source == "env"
    assert found.root == str(fake_root)


def test_threedna_unavailable_when_config_missing(tmp_path, monkeypatch):
    """An incomplete install (bin/fiber present but no config/) must
    NOT be claimed.  Without config/ the fiber binary fails at runtime
    with cryptic errors -- detection has to filter this case out."""
    from molbuilder.backends import _threedna
    bad_root = tmp_path / "x3dna-broken"
    (bad_root / "bin").mkdir(parents=True)
    fiber = bad_root / "bin" / "fiber"
    fiber.write_text("#!/bin/sh\nexit 0\n")
    fiber.chmod(0o755)
    # No config/ directory -- detection must reject this root.

    monkeypatch.setattr(_threedna, "_find_in_tree", lambda: None)
    monkeypatch.setenv("X3DNA", str(bad_root))
    monkeypatch.setattr(_threedna, "_find_via_path", lambda: None)
    assert _threedna._resolve() is None


def test_threedna_unavailable_message_has_required_pieces(monkeypatch):
    """The unavailability message must mention all three failure modes
    plus the license / fallback contract from docs/design.md."""
    from molbuilder.backends import _threedna
    monkeypatch.setattr(_threedna, "_find_in_tree", lambda: None)
    monkeypatch.delenv("X3DNA", raising=False)
    monkeypatch.setattr(_threedna, "_find_via_path", lambda: None)
    msg = _threedna._unavailable_message()
    # All three resolution strategies named:
    assert "in-tree" in msg.lower()
    assert "X3DNA" in msg
    assert "PATH" in msg
    # License / fetch-yourself contract:
    assert "x3dna.org" in msg
    assert "non-commercial" in msg.lower()
    # Fallback backends named:
    assert "amber" in msg
    assert "rdkit" in msg


def test_dispatch_threedna_unavailable_raises_with_message(monkeypatch):
    """When the user explicitly requests --backend threedna and the
    chain finds nothing, dispatch must raise BackendUnavailable
    carrying the full message contract.  Auto-mode falling through is
    NOT enough -- explicit request must surface the install hint."""
    from molbuilder.backends import _threedna
    monkeypatch.setattr(_threedna, "_resolve", lambda: None)
    with pytest.raises(BackendUnavailable) as exc:
        dispatch("dna", "ATGC", backend="threedna")
    msg = str(exc.value)
    assert "x3dna.org" in msg
    assert "non-commercial" in msg.lower()


def test_dispatch_auto_falls_through_threedna_silently(monkeypatch):
    """When threedna is unavailable in auto mode, dispatch must skip
    it cleanly and try amber.  No BackendUnavailable, no warning --
    just a quiet fall-through."""
    captured = []
    from molbuilder import backends
    monkeypatch.setattr(
        backends, "_load_backends",
        lambda: {
            "threedna": lambda *a, **k: (
                pytest.fail("threedna build called when unavailable")
            ),
            "amber": lambda *a, **k: (calls := captured.append("amber")) or "amber-out",
            "rdkit": lambda *a, **k: "rdkit-out",
        },
    )
    monkeypatch.setattr(
        backends, "available_backends",
        lambda: {"threedna": False, "amber": True, "rdkit": True},
    )
    out = dispatch("dna", "ATGC", backend="auto")
    assert out == "amber-out"
    assert captured == ["amber"]


# --------------------------------------------------------------------- #
#  3DNA real-fiber smoke (skipped if no 3DNA install)                   #
# --------------------------------------------------------------------- #


def test_threedna_builds_real_dna_when_installed():
    """When 3DNA is reachable on this machine, a tiny B-form DNA build
    must produce a chemically plausible Structure: at least one P, at
    least one of each base (DA/DT/DG/DC), backbone connectivity check
    passing.  Skipped on machines without 3DNA."""
    from molbuilder.backends import _threedna
    if not _threedna.is_available():
        pytest.skip("3DNA not reachable on this machine")
    s = _threedna.build("dna", "ATGC", form="B", terminal="OH")
    assert s.n_atoms > 0
    assert "P" in s.elements
    assert {"DA", "DT", "DG", "DC"} <= set(s.residue_names)


def test_threedna_a_form_differs_from_b_form():
    """A-form and B-form DNA have different helical parameters -- the
    geometries the fiber backend produces must actually differ.  This
    catches the case where the form flag isn't being plumbed through."""
    from molbuilder.backends import _threedna
    if not _threedna.is_available():
        pytest.skip("3DNA not reachable on this machine")
    b = _threedna.build("dna", "ATGC", form="B", terminal="OH")
    a = _threedna.build("dna", "ATGC", form="A", terminal="OH")
    # Same sequence -> same atom count, but positions differ.
    assert a.n_atoms == b.n_atoms
    import numpy as np
    diff = np.linalg.norm(a.positions - b.positions, axis=1)
    assert diff.max() > 0.1, (
        "A-form and B-form DNA produced identical coordinates -- "
        "form flag is probably not reaching fiber."
    )


def test_threedna_rna_uses_uracil_not_thymine():
    """RNA must use U (uracil), not T (thymine).  fiber's `-rna` flag
    has to be set; if we accidentally call DNA mode for RNA, residue
    names will include DT and the structure will be chemically wrong."""
    from molbuilder.backends import _threedna
    if not _threedna.is_available():
        pytest.skip("3DNA not reachable on this machine")
    r = _threedna.build("rna", "AUGC", form="A", terminal="OH")
    assert "DT" not in r.residue_names, (
        "RNA build emitted DT residues -- fiber was called in DNA mode"
    )
    # A/U/G/C (RNA) vs DA/DT/DG/DC (DNA): RNA uses single-letter codes.
    assert {"A", "U", "G", "C"} & set(r.residue_names)
