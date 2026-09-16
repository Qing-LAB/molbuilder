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
    #: the person answers this one
    mesh: int = dataclasses.field(default=300, metadata={
        "kind": "engine", "engine_key": "MeshCutoff",
        "category": ("accuracy",), "help": "the person's own answer",
        "group": "stage"})
    #: the CITED RUN answers this one
    basis: str = dataclasses.field(default="DZP", metadata={
        "citation": True, "kind": "engine", "engine_key": "PAO.BasisSize",
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
    assert item is not None and item.citation is True
    back = T.read_template(T._emit([item], engines=("siesta",)))
    assert back.items[0].citation is True, (
        "the marker did not survive emit → read, so a template cannot carry "
        "which answerer owns the item")


def test_an_ordinary_item_is_not_marked():
    """The control: without the metadata key nothing is claimed."""
    assert _decl("mesh").citation is False


def test_the_template_writer_leaves_it_VALUELESS():
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
    assert got["basis"] is None, (
        "the citation's item was emitted carrying a value -- floor 2 would "
        "then assert an electronic contract it does not own")
    assert got["ranks"] is None, "the allocation control still holds"


def test_a_typed_value_is_REFUSED_and_named_to_the_right_answerer():
    """THE ASSERTION THE TWO FROZENSETS WERE STANDING IN FOR.

    A template is a file people edit (§ 4.1), so the rule is checked on
    READ. And the message must name the CITATION: routed to the scheduler's
    story it would send a reader to the queue configuration over a basis set.
    """
    text = T._emit([dataclasses.replace(_decl("basis"), value="TZP")],
                   engines=("siesta",))
    with pytest.raises(Exception) as exc:
        T.read_template(text)
    msg = str(exc.value)
    assert "CITATION" in msg, f"refused, but not as the citation's: {msg}"
    assert "SCHEDULER" not in msg, (
        f"refused with the WRONG answerer's story -- this is the confusion "
        f"the marker exists to remove: {msg}")


def test_it_is_not_an_override_a_pin_or_a_sweep_axis():
    """`template_fields` is the one membership rule. A value the citation
    owns is no more a stage's to override than a rank count is."""
    fields = T.template_fields(_Cited)
    assert "mesh" in fields
    assert "basis" not in fields, (
        "a citation-answered field is offered as an override/pin/sweep name")
    assert "ranks" not in fields, "the allocation control still holds"


def test_select_can_ask_which_items_the_citation_answers():
    """So `prep` asks the template instead of carrying its own name list —
    which is what the frozensets were."""
    items = [_decl(n) for n in ("mesh", "basis", "ranks")]
    t = T.read_template(T._emit(items, engines=("siesta",)))
    assert [i.name for i in T.select(t, citation=True)] == ["basis"]
    assert [i.name for i in T.select(t, citation=False)] == ["mesh", "ranks"]
