"""The paths STANDARD: one address, three verbs — `molbuilder/ref.py`.

GOAL.  Catch the framework answering a question DIFFERENTLY in the two layouts.
That asymmetry is what every defect this standard exists to prevent was made
of: a caller composed a name in one place and globbed for it in another, the
two agreed in the hierarchy and disagreed in flat, and the flat calculation
reported the wrong stage or no stage at all without failing.

CONTRACT.  `plans/plan.md` § 5l — the standard; § 5l.1 the address, § 5l.1a how
⑤ renders in each shape (measured 2026-09-09), § 5l.2 the three verbs and why
there is no fourth.  `project-layout.md` § 2.6 owns the tree itself and § 1.5a
owns the attempt rule these tests assert.

WHY THESE ARE OVER THE ADDRESS AND NOT OVER CALL SITES.  § 5l.6 makes it the
condition for N3: *"with tests over the address, not over call sites."*  A test
per caller of a door observes the door's outcome through a wrapper and carries
one bit; the properties below are quantified over the catalogue and over both
shapes, so a new role or a new layout rule is covered the day it is declared.
"""
import pytest

from molbuilder import runfiles as rf
from molbuilder.paths import Shape
from molbuilder.ref import Ref, AddressError, compose, directory, find, parse

SHAPES = [Shape.named(n) for n in ("hierarchical", "flat")]
IDS = ["hierarchical", "flat"]


def _written(shape, root, refs):
    """Lay the addresses down on disk and hand back the paths."""
    out = []
    for r in refs:
        p = compose(root, shape, r)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
        out.append(p)
    return out


@pytest.mark.parametrize("shape", SHAPES, ids=IDS)
def test_every_composed_address_reads_back_as_itself(shape, tmp_path):
    """A name the framework composed must parse back to the SAME coordinates.

    `plans/plan.md` § 5l.2: the round trip is what makes a catalogue row a
    declaration rather than a label.  This is the property that fails when a
    layout rule and a name rule drift apart — the two-owner defect § 5l exists
    to end — and it is quantified over the catalogue, so a new role is covered
    the day it is declared rather than when someone writes its test.
    """
    roles = [a.role for a in rf.WRITTEN if not a.fields and a.staged]
    refs = [Ref("bdt", role, stage="01_coarse", attempt=1) for role in roles]
    refs += [Ref("bdt", role) for role in roles]
    for ref, path in zip(refs, _written(shape, tmp_path, refs)):
        assert parse(tmp_path, shape, path, "bdt") == ref, path


@pytest.mark.parametrize("shape", SHAPES, ids=IDS)
def test_a_field_survives_the_round_trip_and_is_not_a_wildcard(shape, tmp_path):
    """A role carrying a field is a FAMILY of names, and the field tells them
    apart — `.runwrap-{stamp}.log` is not a role with a star in it.

    `plans/plan.md` § 5l.3: a glob inside a role is a coordinate that escaped
    into the vocabulary, and `role_matches` was the machinery built to chase it
    (deleted in N2).  Two launches must compare EQUAL on role and differ here.
    """
    a = Ref("bdt", ".runwrap-{stamp}.log", stage="01_coarse",
            fields=(("stamp", "20260908-120000"),))
    b = Ref("bdt", ".runwrap-{stamp}.log", stage="01_coarse",
            fields=(("stamp", "20260909-235959"),))
    pa, pb = _written(shape, tmp_path, [a, b])
    assert pa != pb
    assert parse(tmp_path, shape, pa, "bdt") == a
    assert parse(tmp_path, shape, pb, "bdt") == b
    assert len(find(tmp_path, shape, "bdt", role=".runwrap-{stamp}.log")) == 2


@pytest.mark.parametrize("shape", SHAPES, ids=IDS)
def test_the_same_query_gives_the_same_answer_in_both_layouts(shape, tmp_path):
    """One question, one answer, whichever shape the calculation is in.

    `project-layout.md` § 1: the shapes differ in whether the stage boundary is
    a directory wall or a filename convention. Every layer built for the
    hierarchy assumes the distinction lives in the PATH — handed a flat
    calculation those layers do not fail, they answer about the wrong stage and
    say nothing. This test is that failure made visible: the counts below are
    written once and must hold for both.
    """
    refs = [Ref("bdt", ".out", stage="01_coarse", attempt=0),
            Ref("bdt", ".out", stage="01_coarse", attempt=1),
            Ref("bdt", ".fdf", stage="01_coarse", attempt=0),
            Ref("bdt", ".out", stage="02_tight", attempt=0),
            Ref("bdt", "_optimized.xyz"),
            Ref("bdt", ".out", stage="01_coarse", bench="G1K4C6", attempt=0)]
    _written(shape, tmp_path, refs)
    assert len(find(tmp_path, shape, "bdt")) == 6
    assert len(find(tmp_path, shape, "bdt", stage="01_coarse")) == 4
    assert len(find(tmp_path, shape, "bdt", role=".out")) == 4
    assert len(find(tmp_path, shape, "bdt", bench="G1K4C6")) == 1
    assert len(find(tmp_path, shape, "bdt", bench=None)) == 5
    assert len(find(tmp_path, shape, "bdt", attempt=1)) == 1


@pytest.mark.parametrize("shape", SHAPES, ids=IDS)
def test_an_absent_coordinate_is_a_statement_and_an_omitted_one_is_a_question(
        shape, tmp_path):
    """`stage=None` finds ONLY the files that cross rungs; omitting `stage`
    finds those and every rung's too.

    `plans/plan.md` § 5l.1: *an absent coordinate is a STATEMENT, never a
    default.* This is the distinction that makes `find` take keywords rather
    than a `Ref` — in an address `None` says "not stage-scoped", in a query it
    would have to mean "any", and one symbol cannot carry both.
    `process/code-audit.md` D1 is the same rule learned per-parameter.
    """
    _written(shape, tmp_path, [Ref("bdt", ".out", stage="01_coarse"),
                               Ref("bdt", "_optimized.xyz")])
    assert [p.name for p, _ in find(tmp_path, shape, "bdt", stage=None)] \
        == ["bdt_optimized.xyz"]
    assert len(find(tmp_path, shape, "bdt")) == 2


def test_the_hierarchy_keeps_the_attempt_and_the_wrappers_index_apart(tmp_path):
    """`run-2/bdt_01_coarse-run0.out` is attempt 2, wrapper index 0.

    `plans/plan.md` § 5l.1a, measured from `runwrap._run_index_resolver`, which
    says it outright: *"the hierarchy tells them apart by directory, and that is
    the layout layer's job, not the wrapper's"* — and the bash it emits starts
    again at `-run0` in every fresh attempt directory. An address that folded
    the two would have to report one of these numbers as the other.
    """
    sh = Shape.named("hierarchical")
    ref = Ref("bdt", ".out", stage="01_coarse", attempt=2,
              counters=(("run", 0),))
    path, = _written(sh, tmp_path, [ref])
    assert path.parent.name == "run-2"
    assert path.name == "bdt_01_coarse-run0.out"
    got = parse(tmp_path, sh, path, "bdt")
    assert (got.attempt, got.run) == (2, 0)


def test_flat_refuses_an_address_that_numbers_the_attempt_twice():
    """Flat has one slot for ⑤, so an address giving both is refused, not
    resolved.

    `project-layout.md` § 1.5a: flat tells attempts apart by *"the filename
    index the wrapper already writes"*. Two numbers claiming one coordinate is
    the `_stage_state(label, stage, out_glob)` fault `plans/plan.md` § 5l.3
    removes; silently preferring one would make the round trip lie about the
    file it came from.
    """
    ref = Ref("bdt", ".out", stage="01_coarse", attempt=1,
              counters=(("run", 3),))
    with pytest.raises(AddressError, match="two numbers for one coordinate"):
        compose("/c", Shape.named("flat"), ref)
    # The same address is legitimate in the hierarchy, where they are two
    # different quantities.
    assert compose("/c", Shape.named("hierarchical"), ref).parent.name == "run-1"


def test_a_role_outside_the_catalogue_is_refused_at_the_address():
    """Extensibility is a catalogue ROW, never a spelling at a call site.

    `plans/plan.md` § 5l.2. The standard's whole reason to exist is that § 5k's
    method — add a door for whatever question a call site asks — produced ~40
    public functions, five of them in one day. A role the catalogue does not
    declare is the same move one layer down, so it fails here rather than
    producing a name nothing can read back.
    """
    with pytest.raises(AddressError, match="not in the catalogue"):
        Ref("bdt", ".invented")
    with pytest.raises(AddressError, match="not a declared field"):
        Ref("bdt", ".out", fields=(("colour", "red"),))
    with pytest.raises(AddressError, match="not a declared counter"):
        Ref("bdt", ".out", counters=(("attempt", 1),))


@pytest.mark.parametrize("shape", SHAPES, ids=IDS)
def test_a_foreign_file_is_not_ours_and_says_so_quietly(shape, tmp_path):
    """A file somebody else put in the directory reads back as None, never as
    an exception and never as an attempt.

    `plans/plan.md` § 5l.3: a foreign stem is not in the address space, and the
    caller should be TOLD that rather than served by loosening the grammar.
    Measured defect (2026-09-08): moving `attempt_concluded` onto a composer
    that validates the label turned "no record" into a crash for a person's own
    `my.relaxation.fdf` sitting beside the run.
    """
    (tmp_path / "my.relaxation.fdf").write_text("x")
    (tmp_path / "notes.txt").write_text("x")
    assert parse(tmp_path, shape, tmp_path / "my.relaxation.fdf", "bdt") is None
    assert find(tmp_path, shape, "bdt") == []


@pytest.mark.parametrize("shape", SHAPES, ids=IDS)
def test_the_directory_verb_answers_without_a_file_in_it(shape, tmp_path):
    """`role=None` addresses the DIRECTORY, so no caller composes a file just
    to take its parent.

    `plans/plan.md` § 5l.2: `compose` returns a whole path deliberately,
    because the three-call detour — stage directory, attempt directory, then a
    name — is where a caller starts joining strings. The directory is the same
    verb with one coordinate absent, not a fourth function.
    """
    ref = Ref("bdt", None, stage="01_coarse", attempt=1)
    assert compose(tmp_path, shape, ref) == directory(tmp_path, shape, ref)
    assert compose(tmp_path, shape, Ref("bdt", ".out", stage="01_coarse",
                                        attempt=1)).parent \
        == directory(tmp_path, shape, ref)


def test_a_name_that_disagrees_with_its_directory_is_not_that_paths_address(
        tmp_path):
    """A file whose NAME says one rung and whose DIRECTORY says another is
    nobody's address — `parse` returns None rather than believing half of it.

    `plans/plan.md` § 5l.2: the round trip is what the standard rests on, so
    `parse` checks that the address it is about to return actually composes
    back to the path it came from. Without that check a layout rule that
    drifted from a name rule would be discovered at a CALL SITE, reported as
    the wrong stage — which is the two-owner defect the whole section exists to
    end. (Added 2026-09-09: a mutant that deleted the check left every other
    test in this file green.)
    """
    sh = Shape.named("hierarchical")
    stray = tmp_path / "02_tight" / "bdt_01_coarse.out"
    stray.parent.mkdir(parents=True)
    stray.write_text("x")
    assert parse(tmp_path, sh, stray, "bdt") is None
    assert find(tmp_path, sh, "bdt") == []


def test_two_flat_stages_benchmarks_do_not_share_one_container(tmp_path):
    """In flat, each stage's bench container carries its stage in its own name,
    so two stages' sweeps cannot land in one directory.

    `job-contracts.md` § 6.3 via `paths.bench_container`. MEASURED DEFECT
    (2026-08-12, plan A5): the flat container was unqualified `bench/`, so two
    flat stages' benchmarks shared it and each prep overwrote the other's
    job-set, plan and verdict. The hierarchy never had this — its container
    sits inside the stage directory — which is why a test that exercises one
    stage, or only the hierarchy, cannot see it. (Added 2026-09-09: a mutant
    that removed the qualifier left every other test in this file green.)
    """
    sh = Shape.named("flat")
    a = Ref("bdt", ".out", stage="01_coarse", bench="G1", attempt=0)
    b = Ref("bdt", ".out", stage="02_tight", bench="G1", attempt=0)
    pa, pb = _written(sh, tmp_path, [a, b])
    assert pa.parent != pb.parent
    assert parse(tmp_path, sh, pa, "bdt") == a
    assert parse(tmp_path, sh, pb, "bdt") == b
    assert len(find(tmp_path, sh, "bdt", bench="G1")) == 2
    assert len(find(tmp_path, sh, "bdt", stage="01_coarse", bench="G1")) == 1
