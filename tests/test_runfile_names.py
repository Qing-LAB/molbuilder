"""The run-file name generator — `job-contracts.md` § 2.2a.

WHAT THIS REPLACED.  The rule ("one basename"; "only per-stage files carry the
token") was written twice in the contract and enforced nowhere, so every writer
and reader built its own name out of strings.  Measured 2026-09-07: geomeTRIC's
trajectory had six spellings — the emitter's, the warm-file declaration's,
three in the parser, and the contract's own catalogue — and only the emitter's
was right.  On a staged run the parser's error message named a file that does
not exist, and the warm-file carry looked for one too, which fails silently
because a missing warm file is a legal state.

THESE TESTS ARE ABOUT THE GENERATOR AND NOTHING ELSE (user, 2026-09-07:
*"focus on api, and use api that is tested to generate confirmed structured
names"*).  Nothing here greps a rendered deck or scans the tree for call sites.
A name is correct because it came out of a generator whose whole parameter
space is checked — not because a regex failed to find something in some text.
That is also why this is one file rather than a check beside each writer
(*"stop scatter tests around all instances"*).
"""
from __future__ import annotations

import fnmatch
import itertools

import pytest

from molbuilder.runfiles import (QUALIFIERS, WRITTEN, RunFile, RunFileError,
                                 find, find_by_role, latest_run,
                                 compose, is_carried, manifest, parse,
                                 patterns, tail)


def expect(label, stage, role, run=None):
    """The `RunFile` a name with these segments must read back as.

    Written here rather than spelled at each assertion because the COUNTERS
    are a declared class now (`runfiles.QUALIFIERS`), not a field per keyword
    -- so a second counter is a line in that tuple and this helper, not an
    edit to every expectation in the file.
    """
    counters = () if run is None else (("run", run),)
    return RunFile(label, stage, role, counters)

LABEL = "my-job"

#: The four segments, each with the values that matter.  `compose` is total
#: over their product, so the product is what gets tested.
LABELS = ["my-job", "job_2", "A"]
#: A transport ladder's rungs carry `_` in the NAME
#: (`02_electrode_L`) and the ordinal is two-or-more digits --
#: `identity.STAGE_NAME_RE`.  A token pattern narrower than that
#: refused every transport deck (found 2026-09-07 by the suite).
STAGES = [None, "01_coarse", "02_medium", "02_electrode_L",
          "100_final"]
RUNS = [None, 0, 2, 17]
ROLES = [".chk", ".out", ".py", ".run.sh", ".molwatch.log", ".pyscf.log",
         "_optimized.xyz", "_initial.xyz", "_geom_optim.xyz", "_geom.log",
         ".XV", ".DM", ".BASIS_ENTHALPY"]


# --------------------------------------------------------------------- #
#  The generator over its whole parameter space                         #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("label,stage,run,role", list(itertools.product(
    LABELS, STAGES, RUNS, ROLES)))
def test_every_name_the_generator_makes_reads_back_to_its_segments(
        label, stage, run, role):
    """THE PROPERTY, over the whole product: what `compose` builds, `parse`
    takes apart into the same four segments.

    Round-tripping is exactly what the six hand-built spellings did not do, and
    it is the only claim that makes a name trustworthy without looking at it —
    a reader that can recover the segments never needs to know how the writer
    spelled them.
    """
    name = compose(label, role, stage, run)
    got = parse(name, label)
    assert got is not None, f"{name!r} did not parse"
    assert got == expect(label, stage, role, run)
    assert got.name == name


@pytest.mark.parametrize("stage,run,expected", [
    (None,        None, "my-job.chk"),
    (None,        2,    "my-job-run2.chk"),
    ("01_coarse", None, "my-job_01_coarse.chk"),
    ("01_coarse", 2,    "my-job_01_coarse-run2.chk"),
])
def test_the_two_separators_say_which_segment_is_which(stage, run, expected):
    """`job-contracts.md` § 6.3: *"a hyphen announces a counter follows... a
    stage is not a counter — it is a name"*.

    So `_` introduces the stage and `-run` the attempt, which is what lets
    `parse` tell them apart without being told which it is looking at.
    """
    assert compose(LABEL, ".chk", stage, run) == expected


def test_a_role_with_underscores_is_not_mistaken_for_a_stage():
    """`_geom_optim.xyz` is a ROLE; `01_coarse` is a stage.  Splitting on `_`
    cannot tell them apart, and the call sites that tried disagreed: one read
    `my-job_01_coarse_geom_optim.xyz` as stage `01`, another as
    `01_coarse_geom`.  A stage begins with two digits, and that is what ends it.
    """
    assert parse("my-job_01_coarse_geom_optim.xyz", LABEL) == \
        expect(LABEL, "01_coarse", "_geom_optim.xyz")
    # AND WITH A STAGE NAME THAT ITSELF CARRIES `_`, which is the case the
    # separator cannot decide and the role vocabulary can.
    assert parse("my-job_02_electrode_L_geom_optim.xyz", LABEL) == \
        expect(LABEL, "02_electrode_L", "_geom_optim.xyz")
    assert parse("my-job_02_electrode_L.fdf", LABEL) == \
        expect(LABEL, "02_electrode_L", ".fdf")
    # The same trailing role with no stage in front stays a role.
    assert parse("my-job_geom_optim.xyz", LABEL) == \
        expect(LABEL, None, "_geom_optim.xyz")


def test_carried_is_the_absence_of_a_stage():
    """SIESTA's .XV/.DM and PySCF's .chk/_optimized.xyz cross rungs.  A stage
    in them would make each rung hunt for a file only its own stage wrote."""
    assert is_carried(compose(LABEL, ".chk"), LABEL) is True
    assert is_carried(compose(LABEL, "_optimized.xyz"), LABEL) is True
    assert is_carried(compose(LABEL, ".molwatch.log", "01_coarse"), LABEL) is False
    # An ATTEMPT is not a stage: `-run2` still names a carried file.
    assert is_carried(compose(LABEL, ".chk", None, 2), LABEL) is True


def test_another_labels_file_is_not_claimed():
    assert parse("other-job.chk", LABEL) is None
    assert parse("my-jobbish.chk", LABEL) is None


# --------------------------------------------------------------------- #
#  What the generator refuses                                           #
# --------------------------------------------------------------------- #

def test_a_role_without_its_separator_is_refused():
    with pytest.raises(RunFileError, match="must begin with"):
        compose(LABEL, "chk")


@pytest.mark.parametrize("bad", ["my job", "my.job", "", "a/b"])
def test_a_label_that_cannot_be_read_back_is_refused(bad):
    """A dot, a space or a slash makes `parse` ambiguous, so it is refused at
    COMPOSE time — where the caller still knows what it meant."""
    with pytest.raises(RunFileError, match="single"):
        compose(bad, ".chk")


@pytest.mark.parametrize("bad", ["coarse", "1_coarse", "01coarse", "01_", "",
                                 "01_a-b", "01_a.b"])
def test_a_malformed_stage_is_refused_rather_than_embedded(bad):
    with pytest.raises(RunFileError, match="NN_name"):
        compose(LABEL, ".chk", bad)


@pytest.mark.parametrize("bad", [1, 2, 0])
def test_a_stage_POSITION_cannot_become_a_name(bad):
    """The old `-stage<N>` convention keyed names on a stage's POSITION, and a
    positional name silently reassigns outputs the moment the ladder grows a
    rung (§ 6.3).  Refused as a wrong TYPE, at the call."""
    with pytest.raises(RunFileError, match="POSITION is not a token"):
        compose(LABEL, ".molwatch.log", bad)


@pytest.mark.parametrize("bad", [-1, "2", 1.5, True])
def test_a_counter_that_is_not_a_count_is_refused(bad):
    with pytest.raises(RunFileError, match="non-negative counter"):
        compose(LABEL, ".out", None, bad)


# --------------------------------------------------------------------- #
#  The qualifier class -- one declaration, not a hand-written keyword    #
# --------------------------------------------------------------------- #
#
#  § 6.3 gives the hyphen ONE meaning: a counter follows, and a counter is a
#  keyword plus a number.  `parse` knew the keyword `run` from a literal
#  `-run(\d+)` written into it three times, so a second counter would have
#  been a parser edit.  It is a line in `QUALIFIERS` now.

def test_the_only_counter_today_is_the_attempt():
    """Stated so that adding one is a DECISION.  `stage` is not here and never
    will be -- it is a NAME and takes `_` -- and neither is a bench point,
    which is a whole label (`project-layout.md` § 2.3.2)."""
    assert QUALIFIERS == ("run",)


def test_a_keyword_the_grammar_does_not_declare_is_refused():
    """The refusal names the declaration, because that is where the fix is:
    a caller inventing `-seg2` at the call site is how the hyphen would come
    to mean two things."""
    with pytest.raises(RunFileError, match="QUALIFIERS"):
        compose(LABEL, ".out", None, None, seg=2)


@pytest.mark.parametrize("run", [None, 0, 7])
def test_the_declared_keyword_round_trips_through_every_door(run):
    """`compose`, `tail` and `parse` read the counter off ONE declaration, so
    the three cannot disagree about what a hyphen introduces."""
    name = compose(LABEL, ".out", "01_coarse", run)
    assert name.endswith(tail(".out", "01_coarse", run))
    got = parse(name, LABEL)
    assert got.run == run
    assert dict(got.counters).get("run") == run
    assert got.name == name


def test_a_hyphen_that_introduces_no_declared_keyword_is_not_a_counter():
    """`-v2` is not a counter, so it is part of the LABEL -- which is why a
    bench trial's `bdt-G1K4C6` reads as a label and not as a qualifier."""
    assert parse("my-job-v2.out", LABEL) is None
    assert parse("my-job-v2.out", "my-job-v2") == expect("my-job-v2", None,
                                                         ".out")


# --------------------------------------------------------------------- #
#  The catalogue, and the two views of it                               #
# --------------------------------------------------------------------- #
#
#  `WRITTEN` says what molbuilder writes; `patterns()` is the glob family
#  (`identity.OUR_FILE_PATTERNS`) and `manifest()` the concrete names (the
#  Task-setup card).  The two were one hand-written glob list with no name
#  view, and it had drifted: geomeTRIC's opt log was listed as
#  `{label}_geom_*.log` -- the stage token INSIDE the role, the spelling
#  § 2.2a retired -- so the file that IS written matched no row and our own
#  log was reported back to the user as the engine's warm state.

def _legal_runs(a):
    """The attempt values this artifact's name can legally carry."""
    return {"never": [None], "maybe": [None, 0, 3], "always": [0, 3]}[a.attempt]


def _fields_for(a):
    """A legal value for every field this role declares, from the catalogue.

    `runfiles.FIELDS` carries an ``example`` precisely so a walker like this one
    can fill a template it does not know the shape of -- and a test below pins
    each example to its own shape, so this cannot quietly start composing an
    illegal name.
    """
    from molbuilder.runfiles import FIELDS
    return {f: FIELDS[f].example for f in a.fields}


def test_every_declared_examples_matches_its_own_shape():
    """The example is load-bearing (see `_fields_for`), so it cannot rot."""
    import re
    from molbuilder.runfiles import FIELDS
    for name, f in FIELDS.items():
        assert re.fullmatch(f.shape, f.example), (
            f"field {name!r}: example {f.example!r} does not match its own "
            f"shape {f.shape!r}")


@pytest.mark.parametrize("art", WRITTEN, ids=lambda a: a.role)
@pytest.mark.parametrize("label", LABELS)
@pytest.mark.parametrize("stage", STAGES)
def test_every_name_the_catalogue_can_produce_is_matched_by_its_globs(
        art, label, stage):
    """THE PROPERTY THAT TIES THE TWO VIEWS, over the segment product.

    `patterns()` cannot go through `compose` -- `{label}_*` stands for every
    stage token at once and no single call produces it -- so what keeps the
    glob family honest is this: every name the grammar can build for a
    catalogued role is matched by one of the globs.

    This is the check the drifted `_geom.log` row would have failed, and it
    fails for any future row whose glob and whose name are spelled apart.
    """
    for run in _legal_runs(art):
        name = compose(label, art.role, stage if art.staged else None,
                       run, **_fields_for(art))
        assert any(fnmatch.fnmatchcase(name, p.format(label=label))
                   for p in patterns()), (
            f"{name!r} is a name molbuilder writes and no glob matches it")


@pytest.mark.parametrize("art", WRITTEN, ids=lambda a: a.role)
def test_a_file_that_carries_no_stage_never_grows_one(art):
    """The three that belong to the CALCULATION -- the template and the
    source pair -- are written once at the bundle root, so a stage token in
    one would claim a rung wrote it."""
    got = compose("my-job", art.role, "01_coarse" if art.staged else None,
                  **_fields_for(art))
    assert ("_01_coarse" in got) is art.staged


@pytest.mark.parametrize("engine,absent,present", [
    ("siesta", ".py", ".fdf"),
    ("pyscf", ".fdf", ".py"),
])
def test_the_manifest_tells_a_run_about_its_own_engine_only(
        engine, absent, present):
    rows = manifest(LABEL, "01_coarse", engine)
    names = [r["name"] for r in rows]
    assert any(n.endswith(present) for n in names)
    assert not any(n.endswith(absent) for n in names)
    # No engine named is a question, not a claim: answer for both.
    both = [r["name"] for r in manifest(LABEL, "01_coarse")]
    assert any(n.endswith(".py") for n in both)
    assert any(n.endswith(".fdf") for n in both)


def test_every_manifest_name_reads_back_to_the_stage_it_was_asked_for():
    """The card shows these to a person, so they have to be names the
    writers use -- which is the same round-trip the grammar promises."""
    for row in manifest(LABEL, "02_medium"):
        got = parse(row["name"], LABEL)
        assert got is not None, f"{row['name']!r} did not parse"
        assert got.stage in (None, "02_medium")
        assert row["what"], f"{row['name']} has no line saying what it is"


def test_the_first_attempt_is_a_real_name_and_not_a_placeholder():
    """`runwrap` opens at ``_run_n=0``, so the indexed rows show `-run0` --
    the name the first launch actually writes."""
    rows = {r["name"]: r for r in manifest(LABEL, "01_coarse", "pyscf")}
    assert f"{LABEL}_01_coarse-run0.concluded" in rows
    assert rows[f"{LABEL}_01_coarse-run0.concluded"]["carries_attempt"] is True
    assert rows[f"{LABEL}_01_coarse.run.sh"]["carries_attempt"] is False


def test_the_catalogue_and_the_warm_vocabulary_do_not_overlap():
    """WHAT MAKES THE SUBTRACTION WORK (`job-contracts.md` § 4.2): a file is
    the engine's restart state OR one molbuilder wrote, never both.  A role on
    both lists would make `--cold` walk past the file it exists to move, which
    is exactly what a widened `*.xyz` row once did.
    """
    from molbuilder.warmfiles import inventory
    ours = {a.role for a in WRITTEN}
    for engine in ("siesta", "pyscf"):
        warm = set(inventory(engine))
        assert not (ours & warm), (
            f"{engine}: {sorted(ours & warm)} is claimed by both the warm "
            f"vocabulary and the catalogue of what molbuilder writes")


def test_the_role_is_what_says_which_engine_wrote_a_file():
    """WHY THE NAME CARRIES NO ENGINE SEGMENT (user asked, 2026-09-07:
    *"engine name should be part of the name?"*).

    It already does, wherever a file is an engine's at all: the two engines'
    roles are DISJOINT, both for what they leave behind (`.XV`/`.DM` against
    `.chk`/`_optimized.xyz`) and for what molbuilder writes for them (`.fdf`
    against `.py` / `.pyscf.log` / `_geom.log`).  A fifth segment would restate
    what the suffix already says, and a name with two sources of truth for one
    fact is what § 2.2a exists to stop.

    The rest -- the wrapper, its logs, the trajectory -- is genuinely shared:
    ONE wrapper runs both engines, so an engine token on `.run.sh` would be a
    claim about the file that is not true of it.
    """
    from molbuilder.warmfiles import inventory
    assert not set(inventory("siesta")) & set(inventory("pyscf"))
    per_engine = {e: {a.role for a in WRITTEN if a.engine == e}
                  for e in ("siesta", "pyscf")}
    assert not per_engine["siesta"] & per_engine["pyscf"]
    assert per_engine["siesta"] and per_engine["pyscf"]


def test_a_carried_file_is_named_the_same_whichever_engine_reads_it_next():
    """AND WHY AN ENGINE SEGMENT WOULD BREAK SOMETHING REAL.  A carried file
    is how one rung hands the next its geometry, and the next rung may be the
    other engine (a SIESTA relaxation into a PySCF spectrum).  It is found BY
    NAME, so a name that said which engine wrote it would be invisible to the
    rung that wants it -- the same failure the stage token has, one exception
    further on.
    """
    from molbuilder.warmfiles import inventory
    for engine in ("siesta", "pyscf"):
        for suffix in inventory(engine):
            name = compose(LABEL, suffix)
            assert is_carried(name, LABEL), name
            assert LABEL + suffix == name, (
                f"{name!r} carries something between the label and the role; "
                f"the next rung looks for {LABEL + suffix!r}")


def test_a_rung_is_told_its_own_files_and_the_calculation_its_own():
    """A rung's list and the calculation's list PARTITION the catalogue.

    The template and the source pair are written once at the bundle root, so a
    per-rung card that repeated them would say each of them N times and imply
    N copies of a person's own input.
    """
    rung = {r["name"] for r in manifest(LABEL, "01_coarse")}
    calc = {r["name"] for r in manifest(LABEL)}
    assert rung and calc
    assert not rung & calc
    assert len(rung) + len(calc) == len(WRITTEN)
    # And the calculation's are exactly the ones with no rung in the name.
    assert all(parse(n, LABEL).stage is None for n in calc)
    assert all(parse(n, LABEL).stage == "01_coarse" for n in rung)


def test_a_relaxation_is_not_promised_a_spectrum():
    """A list that named a file the run will never write is the same fault as
    one that omits a file it does -- an answer a person cannot check against
    the folder.  `.spectra.json` is the vibration calculation's."""
    relax = {r["name"] for r in manifest(LABEL, None, "pyscf",
                                         calculation="optimization")}
    vib = {r["name"] for r in manifest(LABEL, None, "pyscf",
                                       calculation="vibration")}
    assert compose(LABEL, ".spectra.json") not in relax
    assert compose(LABEL, ".spectra.json") in vib
    # Everything a relaxation writes, a vibration writes too: the kind ADDS.
    assert relax < vib
    # And a caller that has not asked which kind is told about both, because
    # None here is "no question asked", not "the default kind".
    assert compose(LABEL, ".spectra.json") in {
        r["name"] for r in manifest(LABEL, None, "pyscf")}


def test_each_file_says_which_moment_it_appears_in():
    """A card listing a run's files has to say which of them exist yet, and
    the three moments are what it says: the description hand-over, the prep,
    and the launch.  Declared per file, so no page subtracts one list from
    another to find out."""
    moments = {a.when for a in WRITTEN}
    assert moments == {"setup", "prep", "run"}
    setup = {r["name"] for r in manifest(LABEL, None, when=("setup",))}
    assert setup == {compose(LABEL, r) for r in (".template.toml",
                                                 ".source.xyz",
                                                 ".source.molstruct.json")}
    # The deck and its wrapper are PREP's, and they are this rung's.
    prep = {r["name"] for r in manifest(LABEL, "01_coarse", "siesta",
                                        when=("prep",))}
    assert compose(LABEL, ".fdf", "01_coarse") in prep
    assert compose(LABEL, ".run.sh", "01_coarse") in prep
    assert compose(LABEL, ".molwatch.log", "01_coarse") not in prep


# ══ THE READER — find / latest_run ═════════════════════════════════════════
#
# Added with the doors (2026-09-08, plan.md § 5k M3).  Both properties below
# were load-bearing in the code that these doors REPLACED -- the inline scan
# in `attempt_concluded` and the hand-rolled candidate sort in the PySCF
# reader -- and neither was covered: mutating `max` to `min`, and the
# counterless-first order to counterless-last, left the whole suite green.
# One door with three callers makes that gap wider than it was.

class TestTheReader:
    """`find` and `latest_run` — `compose`'s counterpart (project-layout § 4.5)."""

    @staticmethod
    def _lay(d, names):
        for n in names:
            (d / n).write_text("")
        return d

    def test_only_our_files_come_back(self, tmp_path):
        self._lay(tmp_path, ["JOB-run0.out", "JOB.fdf",
                             "somebody-elses.log", "OTHER-run1.out"])
        got = [p.name for p, _ in find(tmp_path, "JOB")]
        assert got == ["JOB.fdf", "JOB-run0.out"], got

    def test_a_role_narrows_it(self, tmp_path):
        self._lay(tmp_path, ["JOB-run0.out", "JOB-run1.out",
                             "JOB-run1.concluded"])
        got = [p.name for p, _ in find(tmp_path, "JOB", role=".out")]
        assert got == ["JOB-run0.out", "JOB-run1.out"], got

    def test_the_counterless_name_sorts_FIRST(self, tmp_path):
        """The PySCF reader's fallback rule, and why it is the door's job.

        That reader walks its candidates REVERSED and wants the bare log read
        LAST, so a `-run<N>` always beats it.  Sorted the other way the
        fallback wins over every real attempt -- the 2026-08-13 bug, which its
        comment records and no test caught.  The order lives here now, so it
        is stated once and checked once.
        """
        self._lay(tmp_path, ["JOB-run2.pyscf.log", "JOB.pyscf.log",
                             "JOB-run10.pyscf.log", "JOB-run3.pyscf.log"])
        got = [p.name for p, _ in find(tmp_path, "JOB", role=".pyscf.log")]
        assert got[0] == "JOB.pyscf.log", f"the bare log must lead: {got}"
        assert got == ["JOB.pyscf.log", "JOB-run2.pyscf.log",
                       "JOB-run3.pyscf.log", "JOB-run10.pyscf.log"], got

    def test_run_10_is_not_run_1(self, tmp_path):
        """NUMERIC, not lexicographic -- the other half of the same 2026-08-13
        fix, which a string sort gets wrong at exactly ten attempts."""
        self._lay(tmp_path, ["JOB-run9.out", "JOB-run10.out"])
        assert latest_run(tmp_path, "JOB") == 10

    def test_the_latest_run_is_the_HIGHEST_across_every_role(self, tmp_path):
        """`attempt_concluded` asks this to find the newest attempt's marker.

        Lowest instead of highest reads an EARLIER attempt's goodbye as the
        latest word -- which is what that function's own comment says must not
        happen, for a warm-retry chain where only the final process concludes.
        And it ranges over roles: an engine that dies before printing leaves a
        `.concluded` and no output at all.
        """
        self._lay(tmp_path, ["JOB-run0.out", "JOB-run0.concluded",
                             "JOB-run1.pyscf.log"])
        assert latest_run(tmp_path, "JOB") == 1

    def test_no_counter_anywhere_is_None_not_zero(self, tmp_path):
        self._lay(tmp_path, ["JOB.fdf", "JOB.out"])
        assert latest_run(tmp_path, "JOB") is None

    def test_a_directory_that_is_not_there_is_empty_not_an_error(self, tmp_path):
        assert find(tmp_path / "nope", "JOB") == []


class TestFindingARoleWithNoLabel:
    """`find_by_role` — for the caller that has a folder and no label."""

    def test_it_returns_every_file_in_that_role(self, tmp_path):
        for n in ("JOB_01_c.fdf", "SOMEBODY_ELSE.fdf", "notes.txt"):
            (tmp_path / n).write_text("")
        assert [p.name for p in find_by_role(tmp_path, ".fdf")] == [
            "JOB_01_c.fdf", "SOMEBODY_ELSE.fdf"]

    def test_an_UNDERSCORE_role_is_refused_rather_than_answered(self, tmp_path):
        """`parse` states the reason: a stage NAME may contain `_` too, so an
        underscore role and a stage cannot be told apart without a label.
        Answering anyway would split `my-job_01_coarse_geom_optim.xyz` at the
        wrong place — the exact defect `parse` was written to end."""
        with pytest.raises(RunFileError) as exc:
            find_by_role(tmp_path, "_geom_optim.xyz")
        assert "without a label" in str(exc.value)
        assert "find(dir, label" in str(exc.value)

    def test_a_role_nothing_writes_is_refused_with_the_catalogue(self, tmp_path):
        """A typo is a refusal here, not an empty list at the call site —
        which reads as *there are none of those* and is a different answer."""
        with pytest.raises(RunFileError) as exc:
            find_by_role(tmp_path, ".fdff")
        assert ".fdf" in str(exc.value)

    def test_a_directory_that_is_not_there_is_empty(self, tmp_path):
        assert find_by_role(tmp_path / "nope", ".fdf") == []


def test_find_returns_files_and_not_directories(tmp_path):
    """`find`'s first line says *our FILES*, and it iterated everything.

    Found by reading the module end to end rather than by a failure: the two
    halves of one door disagreed, since `find_by_role` had always checked.
    Unreachable in today's layouts — a stage directory carries no label prefix
    — which is exactly why nothing caught it, and why the claim was worth
    making true rather than narrowing.
    """
    (tmp_path / "JOB-run0.out").write_text("")
    (tmp_path / "JOB-run1.out").mkdir()          # a directory that parses
    got = [p.name for p, _ in find(tmp_path, "JOB")]
    assert got == ["JOB-run0.out"], got
    assert latest_run(tmp_path, "JOB") == 0, "a directory is not an attempt"
