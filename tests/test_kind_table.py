"""The kind table and the engine table -- the declarations every situation
is decided by (`execution/architecture.md` § 3.3, A18).

WHY THESE ROWS.  The two tables are the one place a kind's and an engine's
facts live, so a record that reads wrongly is wrong everywhere at once and
silently: a part naming a function that was renamed fails only at the prep
of that one pair, on a cluster; a mistyped key would drop a fact and the
code would take the absence for "this kind fixes nothing".  Reading the TOML
by eye cannot tell either.  So: every part each record names resolves, a
record that does not say what a record must is refused naming the file and
the key, and a part that does not give the floor that asks what it asks is
refused as it resolves, naming the record and the key.
"""
from __future__ import annotations

import pytest

from molbuilder import engine_table, kind_table


def _pairs():
    return [(k, e) for k in kind_table.kinds()
            for e in kind_table.kind(k).engines]


@pytest.mark.parametrize("calculation, engine", _pairs(),
                         ids=lambda x: str(x))
def test_every_part_a_pair_names_resolves(calculation, engine):
    """Every code part a pair names -- its run of points' declaration, its
    rung's builder, its rungs' inputs and their gather -- is importable:
    the floor that asks finds it."""
    got = kind_table.kind(calculation).on(engine)
    for key in kind_table.PART_REFS:
        if getattr(got, key) is not None:
            assert kind_table.part_of(calculation, engine, key) is not None


_DESCRIPTION_PARTS = [(k, key) for k in kind_table.kinds()
                      for key in kind_table.kind(k).description.parts]


@pytest.mark.parametrize("calculation, key", _DESCRIPTION_PARTS,
                         ids=[f"{k}-{key}" for k, key in _DESCRIPTION_PARTS])
def test_every_part_a_kind_s_description_names_resolves(calculation, key):
    """Every door a kind's description names for the shared doors that ask
    of a kind that cites or extends -- the extension's limits, a citation's
    frames, the order record, the reuse module with every door prep,
    describe and launch ask of it (`kind_table.REUSE_DOORS`) -- is
    importable: the floor that asks finds it (`execution/architecture.md`
    § 3.3)."""
    assert kind_table.description_part(calculation, key) is not None


#: (calculation, engine, stage, the order its deck lists atoms in) -- the
#: pair's word (`execution/architecture.md` § 3.3: a SIESTA vibration's held
#: atoms first, a transport calculation's TranSIESTA order), none on a lead
#: rung, whose electrode cell is a structure of its own, and none for a
#: pair that states none, the user order.
#: The last column: whether the deck is a structure of its own.
ORDER_ROWS = [
    ("vibration", "siesta", "freq", "held-first", False),
    ("transport", "siesta", "device", "transport", False),
    ("transport", "siesta", "seed", "transport", False),
    ("transport", "siesta", "electrode_L", None, True),
    ("transport", "siesta", "03_electrode_R", None, True),
    ("optimization", "siesta", "coarse", None, False),
    ("vibration", "pyscf", "freq", None, False),
]


@pytest.mark.parametrize("calculation, engine, stage, order, own", ORDER_ROWS,
                         ids=[f"{c}-{e}-{s}" for c, e, s, _, _ in ORDER_ROWS])
def test_a_deck_lists_its_atoms_in_its_pair_s_order(calculation, engine,
                                                     stage, order, own):
    """The order the run door, the working copy and prep each ask of the
    table (`kind_table.atom_order`) for a deck of a stage, by its name or
    its token.  THE SILENT FAILURE: a lead rung's electrode cell sorted as
    the whole junction is, or a pair that states no order sorted anyway --
    a deck's atoms in an order its order record does not say."""
    assert kind_table.atom_order(calculation, engine, stage) == order
    assert (kind_table.own_structure(calculation, stage) is not None) is own


@pytest.mark.parametrize("engine", engine_table.engines())
def test_every_part_an_engine_names_resolves(engine):
    """Its config class and its seam resolve -- and the environments its
    runs enter are categories an environment is registered for
    (`configuration.md` § 4), so its prep finds one."""
    from molbuilder.diagnostics import DEFAULT_ENV_NAMES
    from molbuilder.jobset.engines import EngineSeam, engine_seam
    assert isinstance(engine_table.config_class(engine), type)
    assert isinstance(engine_seam(engine), EngineSeam)
    eng = engine_table.engine(engine)
    for category in filter(None, (eng.environment, eng.gpu_environment)):
        assert category in DEFAULT_ENV_NAMES, category


# A record, broken one way each -- the refusal names the record and the key.
_KIND_ROWS = [
    ("an unknown key",
     '[k]\nnoun = "k"\ntopic = "k"\ncolour = "red"\n[k.on.siesta]\n', "unknown key(s) colour"),
    ("no part on any engine",
     '[k]\nnoun = "k"\ntopic = "k"\n', "no part on any engine"),
    ("a part on an engine the engine table does not hold",
     '[k]\nnoun = "k"\ntopic = "k"\n[k.on.orca]\n', "a part on orca"),
    ("roles with no stage_roles on a part",
     '[k]\nnoun = "k"\ntopic = "k"\nroles = [{name = "a", computes = "scf"}]\n'
     '[k.on.siesta]\n', "no stage_roles"),
    ("stage_roles naming a role the kind has not",
     '[k]\nnoun = "k"\ntopic = "k"\nroles = [{name = "a", computes = "scf"}]\n'
     '[k.on.siesta]\nstage_roles = ["b"]\n', "stage_roles names ['b']"),
    ("a role computing what no role computes",
     '[k]\nnoun = "k"\ntopic = "k"\nroles = [{name = "a", computes = "dft"}]\n'
     '[k.on.siesta]\nstage_roles = "own"\n', "computes one of"),
    ("a role named twice",
     '[k]\nnoun = "k"\ntopic = "k"\nroles = [{name = "a", computes = "scf"}, '
     '{name = "a", computes = "scf"}]\n[k.on.siesta]\nstage_roles = "own"\n',
     "roles names a twice"),
    ("roles that are not tables",
     '[k]\nnoun = "k"\ntopic = "k"\nroles = [1]\n[k.on.siesta]\n',
     "roles is a list of tables"),
    ("a rung's inputs without the gather that takes them",
     '[k]\nnoun = "k"\ntopic = "k"\n[k.on.siesta]\ninputs = "m:f"\n',
     "come together"),
    ("inputs on a kind without roles",
     '[k]\nnoun = "k"\ntopic = "k"\n[k.on.siesta]\n'
     'inputs = "m:f"\ngather = "m:g"\n', "inputs on a kind without roles"),
    # A lead is found by its stage's name alone (`kind_table.lead_of`), so
    # a name two kinds' roles share would put one kind's files on the other
    # kind's lead stem.
    ("a lead's name another kind's role has",
     '[k]\nnoun = "k"\ntopic = "k"\n'
     'roles = [{name = "a", computes = "scf", lead = "A-region", '
     'about = "what the A-region is"}]\n'
     '[k.on.siesta]\nstage_roles = "own"\n'
     '[j]\nnoun = "j"\ntopic = "j"\n'
     'roles = [{name = "a", computes = "scf"}]\n'
     '[j.on.siesta]\nstage_roles = "own"\n', "lead 'a' is a role of j"),
    ("a retired stage naming a role the kind has not",
     '[k]\nnoun = "k"\ntopic = "k"\n'
     'roles = [{name = "a", computes = "scf"}]\n[k.on.siesta]\n'
     'stage_roles = "own"\n'
     'retired = { stage = "s", role = "b", since = "1", why = "w" }\n',
     "retired names role 'b'"),
    # The order a pair's deck lists its atoms in is one of the sort rules'
    # words (`atom_permutation.SORT_KEY_NAMES`), so the order record prep
    # writes names a rule its reader knows.
    ("an atom order that is no sort rule's word",
     '[k]\nnoun = "k"\ntopic = "k"\n[k.on.siesta]\natom_order = "by_colour"\n',
     "atom_order is 'by_colour'"),
    ("a scratch status that names no part",
     '[k]\nnoun = "k"\ntopic = "k"\n[k.on.siesta]\n'
     'scratch_status = "a sentence"\n', "scratch_status is 'a sentence'"),
    # A lead's label is offered on the label palette with what it is for
    # (`web/molview.md`), so a lead says it.
    ("a lead that does not say what its label is for",
     '[k]\nnoun = "k"\ntopic = "k"\n'
     'roles = [{name = "a", computes = "scf", lead = "A-region"}]\n'
     '[k.on.siesta]\nstage_roles = "own"\n', "what its label is for"),
    ("a region that is not a label with what it is for",
     '[k]\nnoun = "k"\ntopic = "k"\nregions = [{name = "x"}]\n'
     '[k.on.siesta]\n', "a region is {name, about}"),
    ("a citation of a kind the table does not hold",
     '[k]\nnoun = "k"\ntopic = "k"\n'
     'description = {cites = "c", cited_kind = "nope", '
     'cited_frames = "m:f", describe = "m"}\n[k.on.siesta]\n',
     "cited_kind 'nope' is no kind of the table"),
    # A door asked of a kind that cites or extends comes with what makes it
    # asked: a cited slot, a carried `reuses`.
    ("an order record on a kind that cites nothing",
     '[k]\nnoun = "k"\ntopic = "k"\n'
     'description = {order_record = "m:f"}\n[k.on.siesta]\n',
     "description.order_record on a kind whose description does not state "
     "cites"),
    # ...and a kind that cites names what reads its citation's frames, and
    # one that carries `reuses` what decides it.
    ("a kind that cites and names no cited frames",
     '[k]\nnoun = "k"\ntopic = "k"\n'
     'description = {cites = "c"}\n[k.on.siesta]\n',
     "cites and names no cited_frames part"),
    ("a kind that carries reuses and names no reuse",
     '[k]\nnoun = "k"\ntopic = "k"\n'
     'description = {carries = ["reuses"], extension = "m:f"}\n'
     '[k.on.siesta]\n', "carries reuses and names no reuse part"),
    ("a reuse on a kind that carries no reuses",
     '[k]\nnoun = "k"\ntopic = "k"\n'
     'description = {reuse = "m"}\n[k.on.siesta]\n',
     "description.reuse on a kind whose description does not state carries "
     "reuses"),
    # A part is named as its door asks -- `module:NAME`, or a module whole
    # where its doors are asked of it -- refused as the table loads, before
    # any command asks it.
    ("an extension named as a module",
     '[k]\nnoun = "k"\ntopic = "k"\n'
     'description = {carries = ["reuses"], extension = "m", '
     'reuse = "m", describe = "m"}\n[k.on.siesta]\n',
     "description.extension is 'm' -- a part is \"module:NAME\""),
    # ...and a kind that cites or extends names how it is described: its
    # own describe, which the command line asks (a kind naming none would
    # end `molbuilder describe` in a traceback).
    ("a kind that cites and names no describe",
     '[k]\nnoun = "k"\ntopic = "k"\n'
     'description = {cites = "c", cited_kind = "optimization", '
     'cited_frames = "m:f"}\n[k.on.siesta]\n',
     "cites and names no describe part"),
    ("a kind that carries reuses and names no describe",
     '[k]\nnoun = "k"\ntopic = "k"\n'
     'description = {carries = ["reuses"], extension = "m:f", '
     'reuse = "m"}\n[k.on.siesta]\n',
     "carries reuses and names no describe part"),
    ("a describe on a kind that neither cites nor extends",
     '[k]\nnoun = "k"\ntopic = "k"\n'
     'description = {describe = "m"}\n[k.on.siesta]\n',
     "description.describe on a kind whose description does not state "
     "cites or carries reuses"),
    ("a describe named as one function",
     '[k]\nnoun = "k"\ntopic = "k"\n'
     'description = {carries = ["reuses"], extension = "m:f", '
     'reuse = "m", describe = "m:f"}\n[k.on.siesta]\n',
     "description.describe is 'm:f' -- a describe part is a module, named "
     "whole"),
    ("a reuse named as one function",
     '[k]\nnoun = "k"\ntopic = "k"\n'
     'description = {carries = ["reuses"], extension = "m:f", '
     'reuse = "m:f", describe = "m"}\n[k.on.siesta]\n',
     "description.reuse is 'm:f' -- a reuse part is a module, named whole"),
]


@pytest.mark.parametrize("what, text, words", _KIND_ROWS,
                         ids=[r[0] for r in _KIND_ROWS])
def test_a_record_that_does_not_say_what_it_must_is_refused(
        tmp_path, monkeypatch, what, text, words):
    table = tmp_path / "kind_table.toml"
    table.write_text(text)
    monkeypatch.setattr(kind_table, "TABLE_FILE", table)
    kind_table._table.cache_clear()
    try:
        with pytest.raises(kind_table.KindTableError) as got:
            kind_table.kinds()
        assert "kind_table.toml" in str(got.value) and words in str(got.value)
    finally:
        kind_table._table.cache_clear()


def test_a_kind_on_an_engine_it_has_no_part_for_is_refused_naming_its_engines():
    with pytest.raises(kind_table.KindTableError) as got:
        kind_table.kind("transport").on("pyscf")
    assert "runs on siesta" in str(got.value)


#: A description part that does not give the floor that asks what it asks,
#: refused as it resolves, naming the kind and the key: a part naming no
#: function; the reuse module lacking one door launch asks -- found missing
#: otherwise at a first launch, on a cluster -- and the describe module
#: lacking one `molbuilder describe` asks.
_PART_ROWS = [
    ("a part naming no function", "extension",
     'extension = "molbuilder.transport.reuse:no_such_limits", '
     'reuse = "molbuilder.transport.reuse"', None,
     "k's description's extension, "
     "'molbuilder.transport.reuse:no_such_limits', does not resolve"),
    ("a reuse module without a door it is asked", "reuse",
     'extension = "molbuilder.transport.reuse:Extension", '
     'reuse = "molbuilder.transport.reuse"',
     ("molbuilder.transport.reuse", "carry_shared"),
     "k's description's reuse, 'molbuilder.transport.reuse', provides no "
     "carry_shared"),
    ("a describe module without a door it is asked", "describe",
     'extension = "molbuilder.transport.reuse:Extension", '
     'reuse = "molbuilder.transport.reuse"',
     ("molbuilder.transport.describe", "not_taken"),
     "k's description's describe, 'molbuilder.transport.describe', "
     "provides no not_taken"),
]


@pytest.mark.parametrize("what, key, parts, gone, words", _PART_ROWS,
                         ids=[r[0] for r in _PART_ROWS])
def test_a_part_that_does_not_resolve_is_refused_naming_record_and_key(
        tmp_path, monkeypatch, what, key, parts, gone, words):
    if gone:
        import importlib
        module, door = gone
        monkeypatch.delattr(importlib.import_module(module), door)
    table = tmp_path / "kind_table.toml"
    table.write_text('[k]\nnoun = "k"\ntopic = "k"\n'
                     f'description = {{carries = ["reuses"], {parts}, '
                     f'describe = "molbuilder.transport.describe"}}\n'
                     '[k.on.siesta]\n')
    monkeypatch.setattr(kind_table, "TABLE_FILE", table)
    kind_table._table.cache_clear()
    try:
        with pytest.raises(kind_table.KindTableError) as got:
            kind_table.description_part("k", key)
        assert words in str(got.value)
    finally:
        kind_table._table.cache_clear()


#: An engine record added to the shipped table, broken one way each -- the
#: refusal names what is wrong.
_ENGINE_ROWS = [
    # An engine the engine table holds whose files the file catalogue does
    # not name -- or the reverse -- is refused where the two lists meet.
    ("an engine whose files the file catalogue does not name",
     "one_process = true", "file catalogue"),
    # Ranks start a program: a record that runs ranks and names none would
    # give its run script nothing to launch on each rank.
    ("an engine running ranks that names no program",
     "one_process = false", "one_process = false names the programs"),
    # Every engine's config is checked: a record that names no validator
    # would let a deck of that engine pass a gate that asked nothing.
    ("an engine that names no validator",
     "one_process = true\n# names no validator", "missing validator"),
    # A benchmark trial's cold start is read from its deck's restart
    # keywords: a lane for an engine whose deck states none would refuse
    # every trial as carrying no restart group.
    ("an engine with a benchmark lane and no keyword reader",
     "one_process = true\nbenchmark = true", "needs a keyword_reader"),
]


@pytest.mark.parametrize("what, lines, words", _ENGINE_ROWS,
                         ids=[r[0] for r in _ENGINE_ROWS])
def test_an_engine_record_that_does_not_hold_together_is_refused(
        tmp_path, monkeypatch, what, lines, words):
    text = engine_table.TABLE_FILE.read_text() + (
        '\n[orca]\nwords = "ORCA"\nidentity_field = "x"\n'
        'config = "m:C"\nsame_calculation = "m:d"\nseam = "m:s"\n'
        'computes_in = "cell"\n'
        + (lines if "no validator" in lines
           else lines + '\nvalidator = "m:v"')
        + ("" if "benchmark" in lines else "\nbenchmark = false")
        + '\npresets = "m:P"\nenvironment = "pyscf"\n')
    table = tmp_path / "engine_table.toml"
    table.write_text(text)
    monkeypatch.setattr(engine_table, "TABLE_FILE", table)
    engine_table._table.cache_clear()
    try:
        with pytest.raises(engine_table.EngineTableError) as got:
            engine_table.engines()
        assert words in str(got.value)
    finally:
        engine_table._table.cache_clear()
