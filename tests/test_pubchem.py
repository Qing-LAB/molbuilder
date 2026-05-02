"""PubChem name-lookup tests with mocked network.

The real ``pubchempy.get_compounds`` hits the PubChem REST API; CI
must never depend on network reachability or upstream availability.
These tests stub ``pubchempy`` at the module-import boundary so the
lookup logic, error handling, and timeout plumbing can be exercised
in isolation.

What is NOT tested here: that the SMILES we get back from a real
PubChem lookup is correct.  That's PubChem's job, and an
integration test that runs only when the user opts in (e.g. with a
``-m network`` marker) is the right fix for that, not this file.
"""

from __future__ import annotations

import socket
import sys
import types

import pytest

from molbuilder import pubchem


class _FakeCompound:
    """Stand-in for pubchempy.Compound with the two fields our code
    actually reads."""
    def __init__(self, canonical=None, isomeric=None):
        self.canonical_smiles = canonical
        self.isomeric_smiles = isomeric


def _install_fake_pubchempy(monkeypatch, get_compounds):
    """Inject a stub `pubchempy` module so pubchem._import_pubchempy()
    returns it.  Caller supplies the get_compounds callable."""
    fake = types.SimpleNamespace(get_compounds=get_compounds)
    monkeypatch.setitem(sys.modules, "pubchempy", fake)
    return fake


# --------------------------------------------------------------------- #
#  smiles_for_name happy path                                           #
# --------------------------------------------------------------------- #


def test_smiles_for_name_returns_canonical(monkeypatch):
    _install_fake_pubchempy(
        monkeypatch,
        lambda name, kind: [_FakeCompound(canonical="c1ccccc1")],
    )
    smi = pubchem.smiles_for_name("benzene")
    assert smi == "c1ccccc1"


def test_smiles_for_name_falls_back_to_isomeric(monkeypatch):
    """If canonical_smiles is empty / None but isomeric_smiles is set
    (rare but happens for stereo-rich molecules), use the isomeric."""
    _install_fake_pubchempy(
        monkeypatch,
        lambda name, kind: [
            _FakeCompound(canonical=None, isomeric="C[C@H](N)C(=O)O"),
        ],
    )
    smi = pubchem.smiles_for_name("L-alanine")
    assert smi == "C[C@H](N)C(=O)O"


# --------------------------------------------------------------------- #
#  smiles_for_name error paths                                          #
# --------------------------------------------------------------------- #


def test_smiles_for_name_unknown_compound(monkeypatch):
    """Unknown name -> ValueError, not a silent None or a network
    error masquerading as a usage error."""
    _install_fake_pubchempy(monkeypatch, lambda name, kind: [])
    with pytest.raises(ValueError, match="no compounds"):
        pubchem.smiles_for_name("definitely-not-a-real-molecule")


def test_smiles_for_name_compound_with_no_smiles(monkeypatch):
    """PubChem can return a compound entry with no SMILES at all
    (rare, but it happens for some salts).  We must surface that as
    a clear ValueError, not crash on attribute access."""
    _install_fake_pubchempy(
        monkeypatch,
        lambda name, kind: [_FakeCompound(canonical=None, isomeric=None)],
    )
    with pytest.raises(ValueError, match="no SMILES"):
        pubchem.smiles_for_name("weird-salt")


def test_smiles_for_name_socket_timeout_becomes_runtime_error(monkeypatch):
    """A network timeout is a transient external-system failure, NOT
    a usage error -- it must surface as RuntimeError with a hint
    about the SMILES escape hatch."""
    def raise_timeout(name, kind):
        raise socket.timeout("simulated timeout")
    _install_fake_pubchempy(monkeypatch, raise_timeout)
    with pytest.raises(RuntimeError) as exc:
        pubchem.smiles_for_name("anything", timeout=0.1)
    msg = str(exc.value)
    assert "did not complete" in msg
    assert "build_from_smiles" in msg


def test_smiles_for_name_oserror_becomes_runtime_error(monkeypatch):
    """A generic OSError (DNS failure, connection refused) is also a
    network problem; same RuntimeError path."""
    def raise_oserror(name, kind):
        raise OSError("connection refused")
    _install_fake_pubchempy(monkeypatch, raise_oserror)
    with pytest.raises(RuntimeError, match="did not complete"):
        pubchem.smiles_for_name("anything", timeout=0.1)


def test_smiles_for_name_restores_socket_timeout(monkeypatch):
    """The lookup sets the global socket timeout for the duration of
    its call.  It MUST restore the previous value, even on error --
    otherwise a transient PubChem hiccup would silently break every
    subsequent network call in the same process."""
    sentinel = 12.34
    socket.setdefaulttimeout(sentinel)
    _install_fake_pubchempy(
        monkeypatch,
        lambda name, kind: [_FakeCompound(canonical="O")],
    )
    pubchem.smiles_for_name("water", timeout=99.0)
    assert socket.getdefaulttimeout() == sentinel
    # Reset for any later tests in the same session.
    socket.setdefaulttimeout(None)


# --------------------------------------------------------------------- #
#  build_from_name composition                                          #
# --------------------------------------------------------------------- #


def test_build_from_name_threads_through_build_from_smiles(monkeypatch):
    """build_from_name must call smiles_for_name() then hand the
    result + a default title to build_from_smiles().  This pins the
    composition contract -- a refactor that drops kwargs or swaps
    title precedence shows up here."""
    _install_fake_pubchempy(
        monkeypatch,
        lambda name, kind: [_FakeCompound(canonical="O")],
    )
    captured = {}

    def fake_build_from_smiles(smi, **kwargs):
        captured["smi"] = smi
        captured["kwargs"] = kwargs
        from molbuilder.structure import Structure
        import numpy as np
        return Structure(elements=["O"],
                         positions=np.array([[0.0, 0.0, 0.0]]),
                         title=kwargs.get("title", ""))
    monkeypatch.setattr("molbuilder.pubchem.build_from_smiles",
                        fake_build_from_smiles)
    s = pubchem.build_from_name("water", optimize=False)
    assert captured["smi"] == "O"
    # Default title is the name (not None, not empty).
    assert captured["kwargs"]["title"] == "water"
    # Caller's smiles_kwargs forward through.
    assert captured["kwargs"]["optimize"] is False
    assert s.title == "water"


def test_build_from_name_explicit_title_overrides_default(monkeypatch):
    _install_fake_pubchempy(
        monkeypatch,
        lambda name, kind: [_FakeCompound(canonical="O")],
    )
    captured = {}

    def fake_build_from_smiles(smi, **kwargs):
        captured["title"] = kwargs.get("title")
        from molbuilder.structure import Structure
        import numpy as np
        return Structure(elements=["O"],
                         positions=np.array([[0.0, 0.0, 0.0]]),
                         title=kwargs.get("title", ""))
    monkeypatch.setattr("molbuilder.pubchem.build_from_smiles",
                        fake_build_from_smiles)
    pubchem.build_from_name("water", title="H2O")
    assert captured["title"] == "H2O"
