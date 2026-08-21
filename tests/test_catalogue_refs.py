"""Every citation the catalogue names resolves in the one bibliography.

The user's contract (2026-08-21): parameter guidance argues from
CONFIRMED references -- the catalogue's ``refs`` keys resolve against
``docs/science/references.bib``, so an invented or misremembered
citation fails HERE instead of reaching a user's help expander or a
manuscript's bibliography.
"""
from __future__ import annotations

from molbuilder import template as T
from molbuilder.references import known_keys, citation_for, BIB_PATH


def _all_items():
    return T.read_template(T.load_catalogue()).items


def test_every_catalogue_ref_resolves_in_the_bib():
    keys = known_keys()
    bad = [(it.name, r) for it in _all_items()
           for r in it.refs if r not in keys]
    assert not bad, (
        f"catalogue items cite keys the bibliography does not define: "
        f"{bad}.  Add the verified BibTeX entry to {BIB_PATH} "
        f"(see its header for the @verified protocol) or fix the key.")


def test_a_resolved_citation_carries_what_a_person_needs():
    """The form shows title + a findable locator -- pin the resolver's
    shape on a stable entry."""
    c = citation_for("Sun2020")
    assert c and c["title"] and c["doi"], c
    assert "PySCF" in c["title"]


def test_the_resolver_answers_none_for_an_unknown_key():
    """The FORM omits an unknown key (this test is where it fails);
    the resolver must not invent."""
    assert citation_for("Fabricated2099") is None
