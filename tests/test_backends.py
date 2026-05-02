"""Backend dispatch / registry tests.

Tested in isolation from the SIESTA / PySCF e2e flow that
``test_smiles_and_siesta.py`` covers.  These tests pin the dispatch
contract:

  * ``available_backends()`` returns one bool per registered backend.
  * ``dispatch(backend="auto")`` picks the first available in the
    documented order (amber > rdkit; threedna will land first in
    Phase 2.5).
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
