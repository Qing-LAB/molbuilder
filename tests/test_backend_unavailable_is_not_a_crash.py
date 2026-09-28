"""A backend the person did not install is not a server fault.

`web-api.md` § 1 defines the four buckets, and 5xx is *"an I/O error, an
engine that fell over, a bug"*.  A missing optional backend is none of
those: the request was well-formed and molbuilder is refusing it, which is
the contract's ADVISORY case -- 200 with ``ok: false``.

Without a handler, `BackendUnavailable` falls into the route's generic
``except Exception`` and a duplex requested without X3DNA reports as a
crash, naming a tool the person may never have heard of.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

import molbuilder.builders.backends._threedna as _threedna


@pytest.fixture
def no_threedna():
    """3DNA absent, however it is looked for."""
    with patch.object(_threedna, "_resolve", return_value=None), \
         patch.object(_threedna, "is_available", return_value=False):
        yield


# `backend` NAMES WHAT IS MISSING (`web-api.md`, the route's row): a duplex
# asked of "auto" needs X3DNA, so X3DNA is what the page can disable.  It
# echoed the request until 2026-09-28.
@pytest.mark.parametrize("body,backend", [
    ({"kind": "dna", "input": "ATGC", "backend": "threedna"}, "threedna"),
    ({"kind": "dna", "input": "ds,ATGCATGC"},                 "threedna"),
])
def test_a_missing_backend_answers_advisory_not_server_fault(
        web_client, no_threedna, body, backend):
    r = web_client.post("/api/build/molecule", json=body)

    assert r.status_code == 200, (
        f"a backend the person did not install answered HTTP {r.status_code}; "
        f"`web-api.md` § 1 reserves 5xx for a fault on our side")
    j = r.get_json()
    assert j["ok"] is False
    # MACHINE-READABLE, so the page can disable what this box cannot do
    # rather than letting the person find out by pressing the button.
    assert j["reason"] == "backend_unavailable"
    assert j["backend"] == backend
    # And the message still earns its place (`design.md`): it names the
    # tool, where to get it, and what still works without it.
    said = j["error"]
    assert "X3DNA" in said or "3DNA" in said
    assert "x3dna.org" in said


def test_a_build_that_can_succeed_still_does(web_client, no_threedna):
    """The guard must not turn a working fallback into a refusal."""
    r = web_client.post("/api/build/molecule",
                        json={"kind": "dna", "input": "ATGCATGC"})
    assert r.status_code == 200
    assert r.get_json()["ok"] is True


def test_hydrogens_nothing_here_can_add_are_refused_as_advice(
        web_client, monkeypatch):
    """A peptide asks for its hydrogens and neither OpenBabel nor RDKit is
    installed: refused through the same missing-backend door, as advice.

    It returned the chain WITHOUT hydrogens and a Python warning that no page
    showed until 2026-09-28 -- and DFT then ran on the wrong electron count
    (plan W36 ⑨; `science/validation.md` § 4.1 R5: a finding never travels as
    a warning).  The road cannot reach this -- both engines are in the host
    env's recipe -- so the two imports are made to fail.

    MUTATION THIS MUST FAIL AGAINST: `chemistry.add_hydrogens` warning and
    returning the heavy-atom structure.
    """
    import sys
    for missing in ("openbabel", "rdkit"):
        monkeypatch.setitem(sys.modules, missing, None)
    r = web_client.post("/api/build/molecule",
                        json={"kind": "peptide", "input": "GG"})
    assert r.status_code == 200, r.get_json()
    j = r.get_json()
    assert j["ok"] is False and j["reason"] == "backend_unavailable", j
    assert j["backend"] == "hydrogens", j
    assert "OpenBabel" in j["error"] and "RDKit" in j["error"], j["error"]


def test_the_cli_says_the_same_refusal_in_one_line(monkeypatch, capsys):
    """The same refusal on the CLI: one ``Error: ...`` line and exit 1, as a
    ``ClickException`` is said -- not a traceback (`model/chemistry.md`,
    ``add_hydrogens``).  The engines are made to fail as above.

    MUTATION THIS MUST FAIL AGAINST: `cli.main` without its
    `BackendUnavailable` door, which ended in a Python traceback until
    2026-09-28 (M2b's review).
    """
    import sys

    from molbuilder.cli import main
    for missing in ("openbabel", "rdkit"):
        monkeypatch.setitem(sys.modules, missing, None)
    with pytest.raises(SystemExit) as ended:
        main(["peptide", "GG"])
    assert ended.value.code == 1
    said = capsys.readouterr().err
    assert said.startswith("Error: Cannot add hydrogens"), said
    assert "Traceback" not in said, said
