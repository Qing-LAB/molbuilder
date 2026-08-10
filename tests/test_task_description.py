"""``task.json`` -- the description on disk, and its one reader.

The contract is ``docs/engines/stages.md`` § 6.  Every test below names the
rule it serves, and the round-trip fixture is **the contract's own jsonc
block, parsed out of the document**, so the file format and the document
cannot drift apart without a test going red (the pattern
``docs/process/testing.md`` § 6 calls a source-text invariant, already used by
``checkpoint_message()`` against § 7.3).

WHAT P1 OWNS, AND WHAT IT DOES NOT.  § 6.6's preflight is eight rows, and they
do not all belong to the same layer:

  P1, here -- purely structural, answerable from the file alone:
    * the schema string's major (through ``persist.check_schema_major``)
    * ``shape`` present and one of the two legal values
    * stage names match ``[A-Za-z0-9_]+`` and are unique case-insensitively
    * no ``overrides`` key names a stage field (§ 2's three)
    * every stage's ``overrides`` holds exactly the keys in ``varies``
    * unknown keys refused, naming the key

  P2, when resolution lands -- needs the engine's schema, which an L1 codec
  must not import:
    * the engine is one this backend has a generator for
    * the schema fingerprint matches
    * every named field exists in the shared schema
    * every value is inside the schema's bounds
    * the § 6.6a warning for two stages that RESOLVE identically -- the
      comparison is over the resolved pair, and resolving is P2's verb

Splitting it this way is not a weakening: the rows P1 skips are the rows that
would force ``molbuilder/task.py`` to import an engine, which is what
``test_layering.py`` exists to prevent.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from molbuilder.identity import run_id
from molbuilder.task import (
    SCHEMA,
    FILENAME,
    Run,
    Stage,
    StructureRef,
    Task,
    derive_run,
    read_task,
    write_task,
)


REPO = Path(__file__).resolve().parents[1]
CONTRACT = REPO / "docs" / "engines" / "stages.md"


# --------------------------------------------------------------------- #
#  The fixture IS the contract                                          #
# --------------------------------------------------------------------- #

def _strip_jsonc(text: str) -> str:
    """Drop ``//`` comments that are outside string literals."""
    out = []
    for line in text.splitlines():
        in_str = False
        esc = False
        cut = len(line)
        for i, ch in enumerate(line):
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
            elif ch == '"':
                in_str = not in_str
            elif ch == "/" and not in_str and line[i:i + 2] == "//":
                cut = i
                break
        out.append(line[:cut].rstrip())
    # Trailing commas are legal in jsonc, not in json.
    return re.sub(r",(\s*[}\]])", r"\1", "\n".join(out))


def _contract_example() -> dict:
    """§ 6's worked description, read out of the document itself."""
    text = CONTRACT.read_text(encoding="utf-8")
    start = text.index("## 6. `task.json`")
    block = re.search(r"```jsonc\n(.*?)\n```", text[start:], re.S)
    assert block, "engines/stages.md § 6 no longer opens with a jsonc block"
    return json.loads(_strip_jsonc(block.group(1)))


@pytest.fixture
def example() -> dict:
    return _contract_example()


def test_the_contracts_own_example_is_a_valid_description(example):
    """If the document's worked example does not parse, one of them is
    wrong -- and this is how we find out which, on the same day."""
    task = Task.from_dict(example)
    assert task.engine == "siesta"
    assert task.shape == "hierarchical"
    # Derived, not retyped.  The id in § 6's example is what
    # `run-identity.md § 2` builds from the run's name and the structure's
    # formula, and asserting it as a literal is how this test went red on
    # 2026-08-08 when the id became case-preserving (§ 8 #14) -- a fixture
    # parsed from one document and compared to a constant copied out of
    # another is only half the pattern.
    assert task.run.id == run_id(task.run.name, task.structure.formula)
    assert [s.name for s in task.stages] == ["coarse", "tight"]
    assert task.varies == ("mesh_cutoff", "relax_force_tol",
                           "relax_type", "restart")


def test_round_trip_is_byte_identical(example, tmp_path):
    """M1.  read -> write -> read, and the bytes settle."""
    p = tmp_path / FILENAME
    first = write_task(p, Task.from_dict(example)).read_bytes()
    again = write_task(p, read_task(p)).read_bytes()
    assert first == again
    assert read_task(p) == Task.from_dict(example)


def test_written_file_is_named_and_versioned(example, tmp_path):
    p = tmp_path / FILENAME
    write_task(p, Task.from_dict(example))
    assert p.name == "task.json"
    assert json.loads(p.read_text())["schema"] == SCHEMA == "molbuilder/task@1"


# --------------------------------------------------------------------- #
#  The refusals -- each names its offender (§ 6.6, and decision 3:       #
#  a person edits this file, so the message owes them the same as the    #
#  browser gets)                                                         #
# --------------------------------------------------------------------- #

def test_unknown_top_level_key_is_refused_by_name(example):
    """§ 6.1 rule 1.  Reverses --stages-json's 'Unknown keys ignored'."""
    example["mesh_cutof"] = 300
    with pytest.raises(ValueError) as e:
        Task.from_dict(example)
    assert "mesh_cutof" in str(e.value)


def test_unknown_key_suggests_the_obvious_near_miss(example):
    """Decision 3 (2026-08-07): hand-editing is supported, so a typo one
    character away from a real key says so rather than only refusing."""
    example["structur"] = {}
    with pytest.raises(ValueError) as e:
        Task.from_dict(example)
    msg = str(e.value)
    assert "structur" in msg and "structure" in msg


def test_unknown_stage_key_is_refused_by_name(example):
    example["stages"][1]["enabledd"] = True
    with pytest.raises(ValueError) as e:
        Task.from_dict(example)
    assert "enabledd" in str(e.value)
    assert "tight" in str(e.value), "the message must name WHICH stage"


def test_repeated_stage_name_is_refused_naming_the_repeat(example):
    example["stages"][1]["name"] = "coarse"
    with pytest.raises(ValueError) as e:
        Task.from_dict(example)
    assert "coarse" in str(e.value)


def test_stage_names_collide_case_insensitively(example):
    """§ 6.6: 'compared case-insensitively' -- because the names become
    filenames, and two of those collide on a case-insensitive filesystem."""
    example["stages"][1]["name"] = "COARSE"
    with pytest.raises(ValueError) as e:
        Task.from_dict(example)
    assert "COARSE" in str(e.value) or "coarse" in str(e.value)


@pytest.mark.parametrize("bad", ["with space", "has-hyphen", "dot.ted", ""])
def test_stage_name_outside_the_set_is_refused(example, bad):
    """§ 6.6 refuses 'naming the stage and the rule'."""
    example["stages"][1]["name"] = bad
    with pytest.raises(ValueError) as e:
        Task.from_dict(example)
    assert "A-Za-z0-9_" in str(e.value)


def test_missing_shape_is_refused_saying_so(example):
    """§ 6.7: required, no default, never inferred."""
    del example["shape"]
    with pytest.raises(ValueError) as e:
        Task.from_dict(example)
    assert "shape" in str(e.value)


def test_shape_must_be_one_of_the_two(example):
    example["shape"] = "nested"
    with pytest.raises(ValueError) as e:
        Task.from_dict(example)
    assert "nested" in str(e.value)
    assert "flat" in str(e.value) and "hierarchical" in str(e.value)


def test_a_different_schema_major_is_refused(example):
    example["schema"] = "molbuilder/task@2"
    with pytest.raises(ValueError) as e:
        Task.from_dict(example)
    assert "task@2" in str(e.value)


def test_a_same_major_minor_bump_is_tolerated(example):
    """job-contracts § 6.1: check the MAJOR only."""
    example["schema"] = "molbuilder/task@1.4"
    Task.from_dict(example)


def test_overrides_may_not_name_a_stage_field(example):
    """§ 6.6.  A stage has three fields (§ 2); an override naming one of
    them would be a stage trying to redefine what a stage is."""
    example["stages"][0]["overrides"]["enabled"] = False
    with pytest.raises(ValueError) as e:
        Task.from_dict(example)
    assert "enabled" in str(e.value)


def test_overrides_must_hold_exactly_the_keys_in_varies(example):
    """§ 6.2: 'Every stage's overrides holds exactly the keys in varies --
    no more, so a demoted parameter cannot leave a value hiding in a stage
    nobody can see.'"""
    example["stages"][0]["overrides"]["relax_steps"] = 200
    with pytest.raises(ValueError) as e:
        Task.from_dict(example)
    assert "relax_steps" in str(e.value)


def test_a_stage_may_omit_a_varied_key_and_that_means_base(example):
    """§ 6.2, corrected 2026-08-07: `overrides` is a SUBSET of `varies`.

    An absent key means *this stage uses base's value* -- a real state, and
    the one § 6 of the tab plan draws as a quiet cell.  This replaces a test
    that required equality; equality both made `varies` redundant (derivable
    as any stage's key set) and forced every cell to be filled with a copy
    of base, which is the table design it was supposed to serve.
    """
    del example["stages"][0]["overrides"]["relax_type"]
    task = Task.from_dict(example)
    assert "relax_type" not in task.stages[0].overrides
    assert "relax_type" in task.varies          # still a column
    # What it resolves to is the TEMPLATE's value -- not a second copy in this
    # file.  There is no `base` key (§ 4, corrected 2026-08-07).


# --------------------------------------------------------------------- #
#  § 6.5 -- one stage is no stages                                      #
# --------------------------------------------------------------------- #

def _one_stage(example: dict) -> dict:
    example.pop("stages")
    example.pop("varies")
    return example


def test_absent_stages_means_one(example):
    task = Task.from_dict(_one_stage(example))
    assert task.stages is None
    assert task.varies is None


def test_a_one_stage_description_round_trips_without_the_keys(
        example, tmp_path):
    """'an empty list would be a second way to spell the same state' --
    so neither key may reappear on write."""
    p = tmp_path / FILENAME
    write_task(p, Task.from_dict(_one_stage(example)))
    written = json.loads(p.read_text())
    assert "stages" not in written
    assert "varies" not in written


def test_varies_without_stages_is_refused(example):
    example.pop("stages")
    with pytest.raises(ValueError) as e:
        Task.from_dict(example)
    assert "varies" in str(e.value)


def test_an_empty_stage_list_is_refused(example):
    """§ 6.5: 'A description WITH stages has at least one; removing the
    last is refused.'  Not silently equal to the one-stage case -- that
    would make an empty list the second spelling § 6.5 rules out."""
    example["stages"] = []
    with pytest.raises(ValueError) as e:
        Task.from_dict(example)
    assert "stages" in str(e.value)


# --------------------------------------------------------------------- #
#  § 6.3 -- structure is a reference plus a witness, never a copy        #
# --------------------------------------------------------------------- #

def test_structure_is_a_reference_and_a_witness(example):
    task = Task.from_dict(example)
    assert task.structure.source.endswith("bdt_au.xyz")
    # Alphabetical, per `run-identity.md § 2.0` (decided 2026-08-08): Au, C,
    # H, S.  The example used to read C6H4S2Au38 -- hand-grouped as "the
    # molecule, then the gold", which matched no rule and which no code could
    # therefore produce.
    assert task.structure.formula == "Au38C6H4S2"
    # DERIVED from the formula rather than retyped.  The example said 46 for
    # a formula that is 50 atoms, and this assertion held the wrong number in
    # place -- found by P3's Review 1 (§ 5c finding 5).  A witness whose two
    # halves disagree cannot detect anything, which is the one job it has.
    counts = re.findall(r"([A-Z][a-z]?)(\d*)", task.structure.formula)
    assert task.structure.atoms == sum(int(n or 1) for _, n in counts)


def test_structure_may_not_carry_coordinates(example):
    """§ 6.3: 'a description that embedded them would be a second copy of
    a file the tree already holds, drifting from it the moment either
    moved.'  The unknown-key rule is what enforces it; this pins the
    intent so a later widening cannot pass unnoticed."""
    example["structure"]["positions"] = [[0.0, 0.0, 0.0]]
    with pytest.raises(ValueError) as e:
        Task.from_dict(example)
    assert "positions" in str(e.value)


# --------------------------------------------------------------------- #
#  project-layout.md § 2.1 -- the description names no machine           #
# --------------------------------------------------------------------- #

def test_the_description_names_no_machine(example, tmp_path):
    """Not a refusal rule -- a shape rule.  There is no key for a host, a
    partition, a rank count or a queue, and this fails if one is added
    without the contract being changed first."""
    p = tmp_path / FILENAME
    write_task(p, Task.from_dict(example))
    text = p.read_text().lower()
    for machine_word in ("host", "partition", "queue", "slurm", "node",
                         "walltime", "account"):
        assert machine_word not in text, (
            f"{machine_word!r} reached the portable description -- "
            "project-layout.md § 2.1 says it names no machine")


# --------------------------------------------------------------------- #
#  § 6.4 -- ONE reader, for both surfaces                               #
# --------------------------------------------------------------------- #

def test_only_one_module_reads_or_writes_task_json():
    """§ 6.4: 'the two must produce the same bytes for the same
    description, and a single reader is how.'

    Asserted as a source-text invariant rather than by importing the
    module from both surfaces: an import nothing calls is dead code, and
    it would not stop a second codec appearing beside it.  This does.
    """
    owners = set()
    for path in sorted((REPO / "molbuilder").rglob("*.py")):
        src = path.read_text(encoding="utf-8", errors="replace")
        if '"task.json"' in src or "'task.json'" in src:
            owners.add(path.relative_to(REPO).as_posix())
    assert owners <= {"molbuilder/task.py", "molbuilder/checkpoint.py"}, (
        f"a second place knows the description's filename: {sorted(owners)}.\n"
        "checkpoint.py is allowed because it only DETECTS the file "
        "(_BUNDLE_DESCRIPTORS); anything that parses it must go through "
        "molbuilder/task.py."
    )


def test_checkpoint_recognises_a_calculation_by_its_description(tmp_path,
                                                                example):
    """P1 unit 4.  checkpoint.py's _is_bundle_root arm was dead because no
    producer wrote a description.  Writing one makes it live."""
    from molbuilder.checkpoint import _is_bundle_root
    assert not _is_bundle_root(tmp_path)
    write_task(tmp_path / FILENAME, Task.from_dict(example))
    assert _is_bundle_root(tmp_path)


# --------------------------------------------------------------------- #
#  run.id is derived, and the label is what is on disk                   #
#  (run-identity.md § 2.0a and § 3 rule 1, decision 26 -- 2026-08-09)    #
# --------------------------------------------------------------------- #

def test_the_id_must_be_what_the_name_and_the_formula_derive(example):
    """§ 3 rule 1 says the id is normalised ONCE and stored. Storing it only
    means something if the stored value is checked against its inputs --
    otherwise ``id`` is a free string and the file can say two things about
    which calculation it is."""
    example["run"]["id"] = "something_else"
    with pytest.raises(ValueError) as e:
        Task.from_dict(example)
    msg = str(e.value)
    assert "run.id" in msg and "something_else" in msg
    # It names what the inputs derive, so the reader can see which half moved.
    assert "BDT_Au_relax_Au38C6H4S2" in msg


def test_editing_the_name_alone_is_caught(example):
    """§ 1's second failure mode, at the description level. Hand-editing
    ``task.json`` is supported (decision 3), so the edit that orphans a
    geometry has to be caught by the reader rather than discovered days later
    in a wall-clock cost."""
    example["run"]["name"] = "BDT/Au relax v2"
    with pytest.raises(ValueError) as e:
        Task.from_dict(example)
    assert "BDT/Au relax v2" in str(e.value)


def test_editing_the_formula_alone_is_caught(example):
    """The other half of the same pin. The formula is § 2.1's one entry
    beyond the user's own name, so a description whose formula moved is a
    description of a different molecule."""
    example["structure"]["formula"] = "H2O"
    with pytest.raises(ValueError) as e:
        Task.from_dict(example)
    assert "H2O" in str(e.value)


def test_the_refusal_does_not_repair_it(example):
    """§ 3 rule 3 one layer up: never patch a name and carry on. Which of the
    three fields is the right one is not knowable from inside the file -- a
    corrected formula and a renamed calculation look identical."""
    example["run"]["id"] = "wrong"
    with pytest.raises(ValueError) as e:
        Task.from_dict(example)
    assert "guess" in str(e.value)


def test_the_label_is_the_stem_and_carries_no_formula(example):
    """Decision 26. `SystemLabel` is what every file is named from, and the
    formula is not in it -- that is the whole split, asserted on the
    contract's own worked example."""
    task = Task.from_dict(example)
    assert task.label == "BDT_Au_relax"
    assert task.structure.formula == "Au38C6H4S2"
    assert task.structure.formula not in task.label


def test_the_label_is_the_ids_first_half(example):
    """Deriving the label is only safe because the id is stored and checked;
    this is the relation that makes the two impossible to disagree."""
    task = Task.from_dict(example)
    assert task.run.id == f"{task.label}_{task.structure.formula}"


def test_a_description_with_no_formula_has_label_equal_to_id(example):
    """§ 2: the formula is optional, and a label alone is a legitimate id.
    The two collapse to one string rather than the id growing a dangling
    separator."""
    example["structure"]["formula"] = ""
    example["run"]["id"] = "BDT_Au_relax"
    task = Task.from_dict(example)
    assert task.label == task.run.id == "BDT_Au_relax"


def test_derive_run_is_how_a_producer_makes_one(tmp_path):
    """§ 3 rule 1 is *"there is one place it happens"*, which is only true if
    producers have somewhere to call rather than spelling the join."""
    run = derive_run("BDT/Au relax", "Au38C6H4S2")
    assert run.id == "BDT_Au_relax_Au38C6H4S2"
    assert run.name == "BDT/Au relax"       # kept verbatim, never normalised


def test_a_task_whose_stages_are_long_still_derives_the_same_id(example):
    """The ladder changes the CAP, not the string. § 3's cap is about what
    ``<label>_<stage>.<ext>`` occupies, so knowing the stages can only change
    whether a name is refused -- never what it normalises to."""
    for s in example["stages"]:
        s["name"] = s["name"] + "_" * 20
    example["varies"] = list(example["varies"])
    task = Task.from_dict(example)
    assert task.run.id == "BDT_Au_relax_Au38C6H4S2"


# --------------------------------------------------------------------- #
#  Constructed by hand, not only parsed                                  #
# --------------------------------------------------------------------- #

def test_a_task_built_in_code_writes_the_same_shape(tmp_path):
    task = Task(
        engine="siesta",
        shape="flat",
        # Derived, not retyped.  The literal here was ``h2_relax_h2``, which
        # had been wrong since the id became case-preserving on 2026-08-08
        # and was invisible while nothing checked it.
        run=derive_run("H2 relax", "H2",
                       created="2026-08-07T10:00:00-07:00"),
        structure=StructureRef(source="projects/x/structure/h2.xyz",
                               formula="H2", atoms=2),
        varies=("mesh_cutoff", "restart"),
        stages=(Stage(name="coarse", enabled=True,
                      overrides={"mesh_cutoff": 150, "restart": "clean"}),
                Stage(name="tight", enabled=True,
                      overrides={"mesh_cutoff": 300, "restart": "continue"})),
    )
    p = write_task(tmp_path / FILENAME, task)
    assert read_task(p) == task
    assert task.to_dict()["schema"] == SCHEMA


@pytest.mark.parametrize("key,value", [
    ("engine", "siesta"),
    ("run", "bdt-relax"),
    ("structure", ["projects/x/structure/h2.xyz"]),
])
def test_a_non_object_where_an_object_belongs_says_so(example, key, value):
    """Decision 3 again.  Before the type guard, ``"engine": "siesta"``
    was refused with ``unknown key 's'`` -- the key check had iterated the
    string's characters.  A refusal that teaches nothing is barely one."""
    example[key] = value
    with pytest.raises(ValueError) as e:
        Task.from_dict(example)
    msg = str(e.value)
    assert key in msg and "expected an object" in msg


def test_a_non_object_overrides_says_so(example):
    example["stages"][0]["overrides"] = ["mesh_cutoff"]
    with pytest.raises(ValueError) as e:
        Task.from_dict(example)
    assert "overrides" in str(e.value) and "expected an object" in str(e.value)


def test_a_task_cannot_hold_varies_without_stages():
    """§ 6.5 holds for the OBJECT, not only for the file.  Without this
    check ``to_dict`` drops the varies silently, losing the one thing
    varies exists to carry (§ 6.2)."""
    with pytest.raises(ValueError) as e:
        Task(engine="siesta", shape="flat",
             run=Run(name="x", id="x"), structure=StructureRef(source="x.xyz"),
             varies=("mesh_cutoff",), stages=None)
    assert "varies" in str(e.value) and "stages" in str(e.value)


def test_a_task_cannot_hold_an_empty_stage_tuple():
    with pytest.raises(ValueError) as e:
        Task(engine="siesta", shape="flat",
             run=Run(name="x", id="x"), structure=StructureRef(source="x.xyz"),
             varies=(), stages=())
    assert "empty" in str(e.value)


def test_stages_with_nothing_varying_is_legal(example, tmp_path):
    """§ 6.5 spells 'one parameter set' by omitting BOTH keys, so a
    present-but-empty ``varies`` is not that second spelling -- it is
    several stages that differ in nothing but their name, which § 6.6a
    (2026-08-07) explicitly allows.  The object invariant must not
    mistake one for the other."""
    example["varies"] = []
    for s in example["stages"]:
        s["overrides"] = {}
    task = Task.from_dict(example)
    assert task.varies == ()
    assert [s.name for s in task.stages] == ["coarse", "tight"]

    p = tmp_path / FILENAME
    write_task(p, task)
    assert read_task(p) == task
    assert json.loads(p.read_text())["varies"] == []
