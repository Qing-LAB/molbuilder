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

import pytest

from support import tooling


def test_a_build_that_can_succeed_still_does(web_client, monkeypatch):
    """With 3DNA absent, a single strand on auto is built by the next
    backend installed: the guard must not turn a working fallback into a
    refusal."""
    tooling.make_absent(monkeypatch, "threedna")
    r = web_client.post("/api/build/molecule",
                        json={"kind": "dna", "input": "ATGCATGC"})
    assert r.status_code == 200
    assert r.get_json()["ok"] is True


#: EVERY TOOL A BUILD NEEDS, absent: the request, the ``missing`` name the
#: answer carries -- what is made absent (`support.tooling.make_absent`) --
#: and the words it must say (`engines/builders.md` § 1): a package of the
#: host env with the command that installs it there (`repair molbuilder`).
_REPAIR = "repair molbuilder"
_TOOLS = [
    # 3DNA asked by name, and a duplex on auto, which only 3DNA builds: the
    # page can disable what is named (`web-api.md`, the route's row).
    pytest.param({"kind": "dna", "input": "ATGC", "backend": "threedna"},
                 "threedna", ("3DNA", "x3dna.org"), id="3dna-asked-by-name"),
    pytest.param({"kind": "dna", "input": "ds,ATGCATGC"}, "threedna",
                 ("3DNA", "x3dna.org"), id="a-duplex-on-auto-without-3dna"),
    # A backend asked by name is refused by the registry, its own words and
    # install line -- the path any backend takes (`engines/builders.md`
    # § 4: a backend checks nothing of its own absence).
    pytest.param({"kind": "dna", "input": "ATGC", "backend": "amber"},
                 "amber", ("AmberTools", "molbuilder-MDtools"),
                 id="amber-asked-by-name"),
    pytest.param({"kind": "smiles", "input": "CCO"}, "rdkit",
                 ("RDKit", _REPAIR), id="a-smiles-without-rdkit"),
    # RDKit rejects a five-membered all-aromatic carbon ring; OpenBabel is
    # the fallback that reads what RDKit cannot.
    pytest.param({"kind": "smiles", "input": "c1cccc1"}, "openbabel",
                 ("OpenBabel", _REPAIR),
                 id="a-smiles-rdkit-rejects-without-openbabel"),
    pytest.param({"kind": "peptide", "input": "GG"}, "peptidebuilder",
                 ("PeptideBuilder", _REPAIR),
                 id="a-peptide-without-peptidebuilder"),
    pytest.param({"kind": "name", "input": "ethanol"}, "pubchempy",
                 ("PubChemPy", _REPAIR), id="a-name-without-pubchempy"),
    # A peptide's hydrogens with neither engine: it returned the chain
    # WITHOUT hydrogens and a Python warning no page showed until
    # 2026-09-28, and DFT ran on the wrong electron count (plan W36 ⑨).
    pytest.param({"kind": "peptide", "input": "GG"}, "hydrogens",
                 ("OpenBabel", "RDKit", _REPAIR),
                 id="hydrogens-nothing-here-can-add"),
]


@pytest.mark.parametrize("body,missing,names", _TOOLS)
def test_every_missing_tool_is_advice(web_client, monkeypatch, body, missing,
                                      names):
    """A tool any build needs, absent, is the one missing-tool answer --
    advice (200, ``ok: false``) naming it -- never a fault.  The silent
    failures: a duplex asked without 3DNA reported as a crash; the same
    absence answered three ways -- PeptideBuilder, RDKit and pubchempy as a
    500 "missing dependency", OpenBabel's fallback as a 400 bad request
    (F12's review, 2026-11-18) -- and the hydrogens' row above."""
    tooling.make_absent(monkeypatch, missing)
    r = web_client.post("/api/build/molecule", json=body)
    j = r.get_json()
    assert r.status_code == 200, j
    assert j["ok"] is False and j["reason"] == "backend_unavailable", j
    assert j["backend"] == missing, j
    for name in names:
        assert name in j["error"], j["error"]
