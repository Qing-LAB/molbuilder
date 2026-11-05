"""`citation: True` — the answerer that is neither the person nor the machine.

`template.md` § 6.4 already had the STATE a transport calculation's electronic
contract needs: *declared, valueless, filled by a later step*. `citation` is
the **source marker** — a way for the item to say *who* fills it, as
`allocation` says the scheduler does.

*Ruled 2026-09-15 by the user — "use the citation marker, not a view".*

**What is under test is the RULE, not the plumbing**: a template may declare
such an item and must never answer it, and the refusal must say which answerer
it belongs to. A wrong-answerer message is the whole cost of not having the
marker — *"the scheduler answers it"* over a basis set sends a reader to the
queue configuration.

These run on a synthetic config class, not on `SiestaConfig`: the rule is the
framework's and must hold before any engine adopts it -- all but the reading of
a template into a config, which is an engine's: a config is an engine's when
its record names its class (`engine_table.engine_of_config`), so that test
reads into `SiestaConfig` from the catalogue's own item.
"""
from __future__ import annotations

import dataclasses


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
        "citation": ("transport",), "kind": "engine",
        "engine_key": "PAO.BasisSize",
        "category": ("method",), "help": "the basis the citation used",
        "group": "profile"})
    #: the SCHEDULER answers this one
    ranks: int = dataclasses.field(default=4, metadata={
        "allocation": True, "kind": "wrapper",
        "category": ("execution",), "help": "ranks", "group": "budget"})


def _decl(name):
    """The synthetic catalogue ITEM for one of `_Cited`'s fields -- built as
    the catalogue builds one, from the facts the field's metadata states
    (`T.Item`; `declaration_for`, which read them off a dataclass field,
    went with the class copies in 2.15.4)."""
    import typing
    hints = typing.get_type_hints(_Cited)
    f = {x.name: x for x in dataclasses.fields(_Cited)}[name]
    m = dict(f.metadata)
    return T.Item(name=name, kind=m["kind"],
                  type={int: "int", float: "float", str: "str",
                        bool: "bool"}[hints[name]],
                  help=m["help"], default=f.default,
                  engine_key=m.get("engine_key", ""),
                  anchor=m.get("engine_key", ""),
                  category=tuple(m.get("category", ())),
                  group=m.get("group"),
                  allocation=bool(m.get("allocation", False)),
                  citation=tuple(m.get("citation", ())))


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
    assert "ranks" not in got, (
        "the allocation control still holds -- a run setting is the job "
        "request's, and the template holds none (engines/template.md § 2)")


def test_a_typed_value_is_ACCEPTED_because_the_citation_only_DEFAULTS():
    """Asserted rather than assumed *(2026-09-16)*.

    `engines/transport.md` § 2a.7: electrode and device cannot disagree
    because there is **one** value shared by every stage, not because it
    came from the cited run.  So the worked case is legal -- relax with DZP because it is cheap and
    adequate for geometry, then transport with TZP because the longer
    orbital tails carry the metal-molecule coupling.
    """
    from molbuilder.config.siesta import SiestaConfig
    item = T.one(T.catalogue(), "basis_size", engine="siesta")
    assert "transport" in item.citation, "the worked case's item is cited"
    text = T._emit([dataclasses.replace(item, value="TZP")],
                   engines=("siesta",))
    cfg = T.config_from_template(text, SiestaConfig)
    assert cfg.basis_size == "TZP", (
        "a transport template answering a cited item is now the ORDINARY "
        "state: it is what `molbuilder describe` writes, defaulted from the cited "
        "run, and what a person edits afterwards")


def test_what_a_citation_takes_is_read_back():
    """`engines/template.md` § 5: the catalogue's `citation` tag is the one
    source of what a citation takes, and `parse/contract.py`'s three tuples
    say how each is read back.  An item tagged and listed in none of them is
    written to the deck record and read by nothing: the citation silently
    short, with every run still finishing."""
    from molbuilder.parse.contract import (K_MESH_RECORD_KEYS, RECORDED_FIELDS,
                                           SCF_RECORD_KEYS)
    tagged = {it.name for it in T.select(T.catalogue(), citation="transport")}
    read = {*RECORDED_FIELDS, *SCF_RECORD_KEYS, *K_MESH_RECORD_KEYS}
    assert tagged - read == set(), (
        "add each to RECORDED_FIELDS, SCF_RECORD_KEYS or K_MESH_RECORD_KEYS")


def test_select_can_ask_which_items_the_citation_answers():
    """So `prep` asks the template instead of carrying its own name list."""
    items = [_decl(n) for n in ("mesh", "basis", "ranks")]
    t = T.read_template(T._emit(items, engines=("siesta",)))
    assert [i.name for i in T.select(t, citation=True)] == ["basis"]
    assert [i.name for i in T.select(t, citation=False)] == ["mesh", "ranks"]
