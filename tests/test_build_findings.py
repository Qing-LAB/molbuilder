"""What a build says of what it built -- through the build route.

`science/validation.md` § 4.1 R5 *(decided with the user, 2026-10-31,
(b))*: a builder returns the structure, and one check after the build
(`validation.builders.check_built`, run by the build door `builders.build`)
says what it found -- the structure's own (a peptide's aldehyde end, a
nucleic acid's steric clash on every DNA or RNA build), read off what was
built, and what the backend that
built it does not build of what was asked (a form, a terminal), in each
backend's own answer (`builders.backends.built`).  `POST /api/build/molecule` answers them
as notices beside the structure, each a finding by its `where`; none travels
as a Python warning, which no web user sees.

THE SILENT FAILURES: the peptide's aldehyde warning was lost on the route
(its builder warned where nothing caught it); a backend's limit -- a helical
form it does not shape, a terminal it does not write -- reached the server
log alone.

A row is a build as a person asks it: the kind, the input, what it asks of
the backend; the findings the answer carries, by `where`; and words a
finding must say, where the row names some (what the backend built instead).
A row whose backend is not installed is skipped.
"""
from __future__ import annotations

import warnings

import pytest

ROWS = [
    pytest.param("peptide", "ARNDC", {}, ["build.peptide_aldehyde"], [],
                 id="a-peptide's-c-terminus-as-the-kit-writes-it"),
    pytest.param("dna", "AAAA,TTTG", {"backend": "threedna"},
                 ["build.steric_clash"], ["mismatched", "A1:H on N6"],
                 id="a-mismatched-duplex-clashes"),
    pytest.param("dna", "ATGC,GCAT", {"backend": "threedna"}, [], [],
                 id="a-complementary-duplex-is-clash-free"),
    pytest.param("rna", "ACGU", {"backend": "threedna", "form": "B"},
                 ["build.form"], ["was built A-form"],
                 id="3dna-builds-a-form-rna-only"),
    pytest.param("dna", "ATGC", {"backend": "threedna", "terminal": "PP"},
                 ["build.terminal"], ["PP asked was built 5P"],
                 id="3dna-adds-no-3'-phosphate"),
    pytest.param("dna", "ATGC", {"backend": "threedna", "terminal": "3P"},
                 ["build.terminal"], ["was built OH"],
                 id="3dna-builds-a-3'-phosphate-ask-with-both-ends-oh"),
    pytest.param("dna", "ATGC", {"backend": "amber", "terminal": "5P"},
                 ["build.steric_clash", "build.form", "build.terminal"],
                 ["was built OH", "H3'", "H5'", "as built on the amber",
                  "B-form (the default)"],
                 id="tleap-builds-an-extended-chain-with-oh-termini"),
    pytest.param("rna", "ACGU", {"backend": "amber"},
                 ["build.steric_clash", "build.form"], ["H3'", "H5'"],
                 id="tleap-rna-links-clash-said"),
    pytest.param("dna", "ATGC", {"backend": "rdkit"}, ["build.form"], [],
                 id="rdkit-embeds-a-conformer-not-a-helix"),
    pytest.param("dna", "ATGC", {"backend": "rdkit", "terminal": "3P"},
                 ["build.form", "build.terminal"], ["was built OH"],
                 id="rdkit-dna-has-no-3'-phosphate"),
    pytest.param("dna", "ATGC", {"backend": "rdkit", "terminal": "PP"},
                 ["build.form", "build.terminal"], ["was built 5P"],
                 id="rdkit-dna-pp-is-built-5p"),
]


@pytest.mark.parametrize("kind, text, ask, want, says", ROWS)
def test_what_a_build_found_is_answered_beside_its_structure(
        web_client, kind, text, ask, want, says):
    from molbuilder.builders.backends import available_backends
    backend = ask.get("backend")
    if backend and not available_backends().get(backend):
        pytest.skip(f"{backend} is not installed here")
    with warnings.catch_warnings(record=True) as raised:
        warnings.simplefilter("always")
        r = web_client.post("/api/build/molecule",
                            json={"kind": kind, "input": text, **ask})
    body = r.get_json()
    assert r.status_code == 200 and body["ok"], body
    found = [n["where"] for n in body.get("notices", [])
             if n["where"].startswith("build.")]
    assert found == want, body.get("notices")
    said = " ".join(n["message"] for n in body.get("notices", []))
    for words in says:
        assert words in said, said
    if backend:
        assert body["backend_used"] == backend, body["backend_used"]
    # ONE CHANNEL (R5): nothing the build found was a Python warning.
    assert not [w for w in raised
                if issubclass(w.category, RuntimeWarning)], \
        [str(w.message) for w in raised]
