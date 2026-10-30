"""The kind table and the engine table -- the declarations every situation
is decided by (`execution/architecture.md` § 3.3, A18).

WHY THESE ROWS.  The two tables are the one place a kind's and an engine's
facts live, so a record that reads wrongly is wrong everywhere at once and
silently: a part naming a function that was renamed fails only at the prep
of that one pair, on a cluster; a mistyped key would drop a fact and the
code would take the absence for "this kind fixes nothing".  Reading the TOML
by eye cannot tell either.  So: every part each record names resolves, and a
record that does not say what a record must is refused naming the file and
the key.
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


@pytest.mark.parametrize("engine", engine_table.engines())
def test_every_part_an_engine_names_resolves(engine):
    """Its config class and its seam: the seam's factory builds a seam
    whose config class is the record's."""
    from molbuilder.jobset.engines import engine_seam
    cls = engine_table.config_class(engine)
    assert engine_seam(engine).config_cls is cls


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
     'roles = [{name = "a", computes = "scf", lead = "A-region"}]\n'
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


def test_a_part_that_does_not_resolve_is_refused_naming_record_and_key():
    with pytest.raises(kind_table.KindTableError) as got:
        kind_table.part("molbuilder.jobset.prep:no_such_rung",
                        record="vibration on siesta", key="rung")
    assert "vibration on siesta's rung" in str(got.value)


def test_the_engine_table_is_the_file_catalogue_s_list(tmp_path, monkeypatch):
    """An engine the engine table holds whose files the file catalogue does
    not name -- or the reverse -- is refused where the two lists meet."""
    text = engine_table.TABLE_FILE.read_text() + (
        '\n[orca]\nwords = "ORCA"\nidentity_field = "x"\n'
        'config = "m:C"\nseam = "m:s"\ncomputes_in = "cell"\n'
        'one_process = true\nbenchmark = false\npresets = "m:P"\n'
        'keyword_deck = false\n')
    table = tmp_path / "engine_table.toml"
    table.write_text(text)
    monkeypatch.setattr(engine_table, "TABLE_FILE", table)
    engine_table._table.cache_clear()
    try:
        with pytest.raises(engine_table.EngineTableError) as got:
            engine_table.engines()
        assert "file catalogue" in str(got.value)
    finally:
        engine_table._table.cache_clear()
