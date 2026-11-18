"""PubChem name lookup (`engines/builders.md` § 6), with the network call
stubbed.

The real ``pubchempy.get_compounds`` reaches the PubChem REST API; no test
may depend on the network.  Each test replaces that ONE function of the real
library (``monkeypatch.setattr``) -- every other part, its record reader and
its error types, is pubchempy's own, so the code is tested against what the
library answers and raises.

What is NOT tested here: that a real PubChem answer is right.  That is
PubChem's job.
"""

from __future__ import annotations

import json
import socket
from urllib.error import URLError

import pytest

from molbuilder import pubchem
from molbuilder.chemistry import BackendUnavailable, BuildRefused

#: THE REAL LIBRARY, its network call replaced test by test.
_PCP = pytest.importorskip("pubchempy")


def _compound(absolute=None, connectivity=None):
    """A REAL ``pubchempy.Compound``, built offline from a record holding the
    two SMILES PubChem serves -- the library's own reading of them (F12's
    review, 2026-11-18: a stand-in with the deprecated attributes hid that
    ``canonical_smiles`` is the connectivity-only one)."""
    props = [{"urn": {"label": "SMILES", "name": name}, "value": {"sval": s}}
             for name, s in (("Absolute", absolute),
                             ("Connectivity", connectivity)) if s]
    return _PCP.Compound({"atoms": {"aid": [], "element": []},
                         "props": props})


def _not_json() -> json.JSONDecodeError:
    """What the library's reader raises on a page that is not PubChem's JSON
    -- a proxy's or a network login's -- the real error, from json itself."""
    try:
        json.loads("<html><body>Sign in to the network</body></html>")
    except json.JSONDecodeError as exc:
        return exc


def _answers(monkeypatch, outcome):
    """PubChem's answer to every lookup: ``outcome`` returned, or raised
    when it is an exception."""
    def get_compounds(name, namespace):
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome
    monkeypatch.setattr(_PCP, "get_compounds", get_compounds)


def test_the_absolute_smiles_is_the_one_built(monkeypatch):
    """The absolute SMILES -- stereochemistry included -- never the
    connectivity-only one, which builds "D-alanine" as whichever isomer the
    embedder lands on."""
    d_alanine = "C[C@@H](C(=O)O)N"
    _answers(monkeypatch, [_compound(absolute=d_alanine,
                                     connectivity="CC(C(=O)O)N")])
    assert pubchem.smiles_for_name("D-alanine") == d_alanine


#: EACH FAILURE SAID ONCE, BY WHOSE IT IS: (PubChem's answer, the type the
#: lookup raises, words it must say, words it must not).  The request's is a
#: refusal; PubChem out of reach is advice naming PubChem, each cause in its
#: own words -- a timeout never worded as a network error or the reverse.
_FAILURES = [
    pytest.param([], BuildRefused, "no compounds", None, id="unknown_name"),
    pytest.param([_compound()], BuildRefused, "no SMILES", None,
                 id="record_without_smiles"),
    pytest.param(_PCP.BadRequestError(400, "PUGREST.BadRequest",
                                      ["Invalid name"]),
                 BuildRefused, "refused the name", None, id="name_refused"),
    pytest.param(socket.timeout("timed out"), BackendUnavailable,
                 "did not complete within 0.1 s", "could not reach",
                 id="read_out_of_time"),
    pytest.param(URLError(socket.timeout("timed out")), BackendUnavailable,
                 "did not complete within 0.1 s", "could not reach",
                 id="connect_out_of_time"),
    pytest.param(URLError("Name or service not known"), BackendUnavailable,
                 "could not reach PubChem (Name or service not known)",
                 "did not complete", id="no_network"),
    pytest.param(_PCP.ServerBusyError(503, "PUGREST.ServerBusy",
                                      ["Too many requests"]),
                 BackendUnavailable, "answered by PubChem with an error",
                 "did not complete", id="pubchem_busy"),
    # A proxy's page in place of PubChem's JSON: a 500 in a parser's words
    # until F12's second review (2026-11-18).
    pytest.param(_not_json(), BackendUnavailable, "does not read",
                 "did not complete", id="answer_that_does_not_read"),
    # A record with no properties at all -- a broken proxy's or cache's: its
    # SMILES read outside the sorting was a 500 (F12's third review,
    # 2026-11-18).
    pytest.param([_PCP.Compound({"atoms": {"aid": [], "element": []}})],
                 BackendUnavailable, "does not read", "did not complete",
                 id="record_without_properties"),
]


@pytest.mark.parametrize("answer, raised, says, never", _FAILURES)
def test_a_failed_lookup_is_said_by_whose_it_is(monkeypatch, answer, raised,
                                                says, never):
    _answers(monkeypatch, answer)
    with pytest.raises(raised) as exc:
        pubchem.smiles_for_name("anything", timeout=0.1)
    assert says in str(exc.value)
    if never:
        assert never not in str(exc.value)
    if raised is BackendUnavailable:
        assert exc.value.missing == "pubchem"


def test_the_socket_timeout_is_restored_after_a_lookup(monkeypatch):
    """The lookup sets the global socket timeout for its call and MUST put
    the previous one back, a failure included -- else one PubChem hiccup
    changes every later network call in the process."""
    before = socket.getdefaulttimeout()
    socket.setdefaulttimeout(12.34)
    try:
        _answers(monkeypatch, socket.timeout("timed out"))
        with pytest.raises(BackendUnavailable):
            pubchem.smiles_for_name("water", timeout=99.0)
        assert socket.getdefaulttimeout() == 12.34
    finally:
        socket.setdefaulttimeout(before)


def test_an_empty_name_is_refused_before_any_lookup(monkeypatch):
    """No name is the request's to fix, refused before PubChem is asked --
    never PubChemPy's own error from its insides (a traceback on the command
    line, F12's second review, 2026-11-18)."""
    asked = []
    monkeypatch.setattr(_PCP, "get_compounds",
                        lambda *a: asked.append(a) or [])
    with pytest.raises(BuildRefused):
        pubchem.smiles_for_name("  ")
    assert asked == []

