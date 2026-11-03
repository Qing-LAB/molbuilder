"""``/api/structure/analyze`` -- the chemistry card's one door: its REFUSALS,
and whose an answer is where only the form knows it (the method's: the one
item a form's blank leaves to the catalogue).

PINS: ``docs/web/web-api.md`` § 5's row for this route and
``docs/science/chemistry-correctness.md`` § 2a.3 (which engine runs which
kind).  API-level, for two things.  The REFUSALS the pages cannot reach:
every page sends the envelope its viewer holds, with a form for each engine
the kind runs, so a body with no structure, an unreadable one, a symbol that
names no element or a form for an engine that does not run the kind comes only
from a caller outside the pages -- and must be a 400 that says why, never a
500.  And the METHOD's source, which only the route knows: a form left blank
is not a form stated, and every form a page can send is a row.

The rest of the ANSWER is pinned where it is made and where it is read: the
class through prep in ``tests/test_electronic_state.py``, and the card, driven
the way a person drives it, in ``tests/test_chemistry_card_e2e.py``.
"""
from __future__ import annotations

import pytest


pytest.importorskip("flask")


@pytest.fixture
def web():
    from molbuilder.web.app import create_app
    return create_app(config={}).test_client()


def _post_analyze(web, body):
    r = web.post("/api/structure/analyze", json=body)
    return r, r.get_json()


_FORMATE = {"structure": {
    "elements": ["C", "O", "O", "H"],
    "positions": [[0, 0, 0], [1.26, 0, 0], [-0.63, 1.09, 0], [-0.55, -0.95, 0]],
    "metadata": {}}}


def test_a_form_for_an_engine_the_kind_does_not_run_is_refused(web):
    """Transport runs on SIESTA alone (its parts in the kind table, § 2a.3):
    unasked, only SIESTA is answered, and a PySCF form for a transport
    calculation is refused naming who runs it -- the route's engine list is
    the kind table's, not a second copy of it."""
    r, body = _post_analyze(web, {**_FORMATE, "kind": "transport"})
    assert r.status_code == 200, body
    assert set(body["state"]) == {"siesta"}
    r, body = _post_analyze(web, {**_FORMATE, "kind": "transport",
                                  "forms": {"pyscf": {}}})
    assert r.status_code == 400, body
    assert "transport runs on siesta" in body["error"], body


@pytest.mark.parametrize("form, source", [
    ({}, "default"),
    ({"method": ""}, "default"),
    ({"method": "  "}, "default"),
    ({"method": "HF"}, "stated"),
], ids=["a method nobody chose is not chosen",
        "a blank method is not chosen",
        "a method of white space is blank, as the config reads it",
        "a method the form states is stated"])
def test_the_route_says_whose_method_it_is(web, form, source):
    """`chemistry-correctness.md` § 2a.1a, `form-schema.md` § 1.1: the
    route answers for a FORM, and a method the person left blank -- blank
    as the config door reads it -- is the catalogue's, said in the words
    every surface says a value nobody chose (`template.SOURCE_WORDS`),
    never *stated*, which would present a value nobody chose as chosen.
    SIESTA's method is its rule, not a form item, so only PySCF's form
    says it."""
    from molbuilder.template import SOURCE_WORDS
    r, body = _post_analyze(web, {**_FORMATE, "kind": "optimization",
                                  "forms": {"pyscf": form}})
    assert r.status_code == 200, body
    method = body["state"]["pyscf"]["method"]
    assert method["source"] == source, method
    assert (SOURCE_WORDS["default"] in method["said"]) == (
        source == "default"), method
    assert method["value"] == ((form.get("method") or "").strip()
                               or "DFT"), method


def test_missing_body_returns_400(web):
    """An empty body is a 400 that says what to send: the structure the page
    holds, in the envelope -- the one way in."""
    r, body = _post_analyze(web, {})
    assert r.status_code == 400
    assert body["ok"] is False
    said = body["error"].lower()
    assert "structure" in said and "envelope" in said, said


def test_an_UNREADABLE_ENVELOPE_returns_400_not_500(web):
    """Garbage in, a refusal out -- never a stack trace: a body this route
    cannot turn into a structure is the CALLER's error."""
    r, body = _post_analyze(web, {"structure": {"elements": ["C"],
                                                "positions": "not a list"}})
    assert r.status_code == 400, body
    assert body["ok"] is False
    assert "could not restore structure" in body["error"].lower(), body


def test_unknown_element_returns_400_not_500(web):
    """A symbol that names no element is a clean 400 with the parser's own
    words: the answer is an electron count, and a count with an atom left
    out is a wrong one (`chemistry.resolve_element`)."""
    r, body = _post_analyze(web, {"kind": "optimization", "structure": {
        "elements": ["Xx"], "positions": [[0.0, 0.0, 0.0]], "metadata": {}}})
    assert r.status_code == 400
    assert body["ok"] is False
    assert "Xx" in body["error"] or "unknown" in body["error"].lower()
