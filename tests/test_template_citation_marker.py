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

    Checked where the KIND is known, not in `read_template`: the master
    catalogue carries both the marker and a value (86 rows of 103 carry
    one), and `basis_size` is the person's answer for an optimization and
    the citation's for transport.  A reader with no kind cannot tell.

    And the message must name the CITATION: routed to the scheduler's story
    it would send a reader to the queue configuration over a basis set.
    """
    text = T._emit([dataclasses.replace(_decl("basis"), value="TZP")],
                   engines=("siesta",))
    with pytest.raises(ValueError) as exc:
        T.config_from_template(text, _Cited, calculation="transport")
    msg = str(exc.value)
    assert "CITATION" in msg, f"refused, but not as the citation's: {msg}"
    assert "SCHEDULER" not in msg, (
        f"refused with the WRONG answerer's story -- this is the confusion "
        f"the marker exists to remove: {msg}")


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
