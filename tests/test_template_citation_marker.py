"""`citation: True` — the answerer that is neither the person nor the machine.

`template.md` § 6.4 already had the STATE a transport calculation's electronic
contract needs: *declared, valueless, filled by a later step*. What it lacked
was the **source marker** — a way for the item to say *who* fills it. The
scheduler had one (`allocation`); the cited run did not, so the same rule was
carried by two hand-maintained frozensets in `transport/stages.py` plus a
predicate spelled twice, in two files, with disagreeing formulations.

*Ruled 2026-09-15 by the user — "use the citation marker, not a view".*

**What is under test is the RULE, not the plumbing**: a template may declare
such an item and must never answer it, and the refusal must say which answerer
it belongs to. A wrong-answerer message is the whole cost of not having the
marker — *"the scheduler answers it"* over a basis set sends a reader to the
queue configuration.

These run on a synthetic config class, not on `SiestaConfig`: the rule is the
framework's and must hold before any engine adopts it.
"""
from __future__ import annotations

import dataclasses

import pytest

from molbuilder import template as T


@dataclasses.dataclass
class _Cited:
    """Three fields, one per answerer, so each assertion has a control."""

    #: `_engine_name` reads this, else it strips a `Config` suffix off the
    #: class name -- and `_Cited` would resolve to the engine `_cited`.
    ENGINE = "siesta"

    #: the person answers this one
    mesh: int = dataclasses.field(default=300, metadata={
        "kind": "engine", "engine_key": "MeshCutoff",
        "category": ("accuracy",), "help": "the person's own answer",
        "group": "stage"})
    #: the CITED RUN answers this one
    basis: str = dataclasses.field(default="DZP", metadata={
        "citation": ("transport",), "kind": "engine",
        "engine_key": "PAO.BasisSize",
        "category": ("method",), "help": "the basis the citation used",
        "group": "profile"})
    #: the SCHEDULER answers this one
    ranks: int = dataclasses.field(default=4, metadata={
        "allocation": True, "kind": "wrapper",
        "category": ("execution",), "help": "ranks", "group": "budget"})


def _decl(name):
    # `from __future__ import annotations` makes `f.type` a STRING, and
    # `declaration_for` reads a real annotation -- resolve them once.
    import typing
    hints = typing.get_type_hints(_Cited)
    f = {x.name: x for x in dataclasses.fields(_Cited)}[name]
    return T.declaration_for(f, hints[name])


def test_the_marker_survives_the_round_trip():
    """Declared from the dataclass, emitted, and parsed back as itself."""
    item = _decl("basis")
    assert item is not None and item.citation == ("transport",)
    back = T.read_template(T._emit([item], engines=("siesta",)))
    assert back.items[0].citation == ("transport",), (
        "the marker did not survive emit → read, so a template cannot carry "
        "which answerer owns the item")


def test_an_ordinary_item_is_not_marked():
    """The control: without the metadata key nothing is claimed."""
    assert _decl("mesh").citation == ()


def test_the_template_writer_CARRIES_the_cited_value():
    """§ 6.4's state. The person's field keeps its value; the citation's
    does not, however much the config object happens to hold."""
    # `template_with_values` narrows the CATALOGUE, so it is handed one:
    # the master has no synthetic rows, and the rule under test is the
    # writer's, not the catalogue's.
    cat = T._emit([_decl(n) for n in ("mesh", "basis", "ranks")],
                  engines=("siesta",))
    text = T.template_with_values(_Cited(mesh=400, basis="TZP"),
                                  engine="siesta", catalogue=cat,
                                  calculation="transport")
    got = {i.name: i.value for i in T.read_template(text).items}
    assert got["mesh"] == 400, "the person's answer must be written"
    assert got["basis"] == "TZP", (
        "the cited item must be written CARRYING its value.  It was emitted "
        "valueless until 2026-09-16 -- the sealed reading -- and "
        "engines/transport.md 2a.7 reversed that: the cited run DEFAULTS "
        "these and the person may change them, so the template has to show "
        "the number they would be changing")
    assert got["ranks"] is None, (
        "the allocation control still holds -- it is a different answerer "
        "and nothing about it changed")


def test_a_typed_value_is_ACCEPTED_because_the_citation_only_DEFAULTS():
    """The reversal, asserted rather than assumed *(2026-09-16)*.

    This test said the opposite until today: a template answering a cited
    item was refused, because the write side emitted them valueless and so a
    value could only be a hand edit -- "the one edit that can make a device
    disagree with its own leads".

    `engines/transport.md` § 2a.7 ruled the other way, and the invariant
    survives intact: electrode and device cannot disagree because there is
    **one** value shared by every stage, not because it came from the cited
    run.  What is withdrawn is only the claim about its source.  So the
    worked case is now legal -- relax with DZP because it is cheap and
    adequate for geometry, then transport with TZP because the longer
    orbital tails carry the metal-molecule coupling.
    """
    text = T._emit([dataclasses.replace(_decl("basis"), value="TZP")],
                   engines=("siesta",))
    cfg = T.config_from_template(text, _Cited, calculation="transport")
    assert cfg.basis == "TZP", (
        "a transport template answering a cited item is now the ORDINARY "
        "state: it is what `jobset init` writes, defaulted from the cited "
        "run, and what a person edits afterwards")


def test_the_SAME_value_is_accepted_for_a_kind_the_citation_does_not_answer():
    """The half that makes it a per-KIND marker rather than a seal.

    `basis_size` is a shared row: an optimization's person answers it.  If
    the refusal fired on the name alone, tagging the row for transport would
    break every optimization that sets a basis.
    """
    text = T._emit([dataclasses.replace(_decl("basis"), value="TZP")],
                   engines=("siesta",))
    cfg = T.config_from_template(text, _Cited, calculation="optimization")
    assert cfg.basis == "TZP"


def test_the_allocation_exclusion_is_untouched():
    """`template_fields` still strips machine facts — and NOT citation ones,
    because that exclusion is kind-dependent and this function has no kind.
    The transport seal is enforced by `config_from_template`, above."""
    fields = T.template_fields(_Cited)
    assert "mesh" in fields
    assert "basis" in fields, (
        "excluded by name -- an optimization could then not override its "
        "own basis, because the row is shared")
    assert "ranks" not in fields, "the allocation control still holds"


def test_select_can_ask_which_items_the_citation_answers():
    """So `prep` asks the template instead of carrying its own name list —
    which is what the frozensets were."""
    items = [_decl(n) for n in ("mesh", "basis", "ranks")]
    t = T.read_template(T._emit(items, engines=("siesta",)))
    assert [i.name for i in T.select(t, citation=True)] == ["basis"]
    assert [i.name for i in T.select(t, citation=False)] == ["mesh", "ranks"]
