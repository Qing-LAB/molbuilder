"""The run-file name generator — `job-contracts.md` § 2.2a.

The rule ("one basename"; "only per-stage files carry the token") is held by
one generator that every writer and reader uses.

THESE TESTS ARE ABOUT THE GENERATOR AND NOTHING ELSE (user, 2026-09-07:
*"focus on api, and use api that is tested to generate confirmed structured
names"*).  Nothing here greps a rendered deck or scans the tree for call sites.
A name is correct because it came out of a generator whose whole parameter
space is checked — not because a regex failed to find something in some text.
That is also why this is one file rather than a check beside each writer
(*"stop scatter tests around all instances"*).
"""
from __future__ import annotations

import itertools

import pytest

from molbuilder.runfiles import (ON_THE_LABEL, RunFile, RunFileError,
                                 find, find_by_role, compose, manifest,
                                 parse)


def expect(label, stage, role):
    """The `RunFile` a name with these segments must read back as."""
    return RunFile(label, stage, role)

LABEL = "my-job"

#: The three segments, each with the values that matter.  `compose` is total
#: over their product, so the product is what gets tested.
LABELS = ["my-job", "job_2", "A"]
#: A transport ladder's rungs carry `_` in the NAME
#: (`02_electrode_L`) and the ordinal is two-or-more digits --
#: `identity.STAGE_NAME_RE`.  A token pattern narrower than that
#: refused every transport deck (found 2026-09-07 by the suite).
STAGES = [None, "01_coarse", "02_medium", "02_electrode_L",
          "100_final"]
ROLES = [".chk", ".out", ".py", ".run.sh", ".molwatch.log", ".pyscf.log",
         "_optimized.xyz", "_initial.xyz", "_geom_optim.xyz",
         ".XV", ".DM", ".BASIS_ENTHALPY"]


# --------------------------------------------------------------------- #
#  The generator over its whole parameter space                         #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("label,stage,role", list(itertools.product(
    LABELS, STAGES, ROLES)))
def test_every_name_the_generator_makes_reads_back_to_its_segments(
        label, stage, role):
    """THE PROPERTY, over the whole product: what `compose` builds, `parse`
    takes apart into the same three segments.

    Round-tripping is the only claim that makes a name trustworthy without looking at it —
    a reader that can recover the segments never needs to know how the writer
    spelled them.
    """
    name = compose(label, role, stage)
    got = parse(name, label)
    assert got is not None, f"{name!r} did not parse"
    assert got == expect(label, stage, role)
    assert got.name == name


def test_a_file_taken_over_from_another_stage_is_named_for_the_stage_it_lies_in():
    """`RunNames.restaged` over the whole product: a file of one stage's run,
    taken over into another stage's (`execution/runs-of-points.md` § 3,
    op. 3; a displacement sweep's later stage taking the first stage's
    relaxation, `engines/vibration.md` § 5.9), is named as the stage it now
    lies in names it -- its role and fields kept (a prep log's engine); a
    file on the label alone, or of a third stage, as it is.

    THE SILENT FAILURE: the call taken over keeping the first stage's
    names, so the later stage's readers find no output and no conclusion
    in it and read the call pending.  MUTATION THIS MUST FAIL AGAINST:
    ``restaged`` handing the name back as it is."""
    from molbuilder.runfiles import ROLE_PIPELINE_LOG, RunNames
    for label, stage, role in itertools.product(LABELS, STAGES, ROLES):
        name = compose(label, role, stage)
        here = RunNames(label, "03_freq_half")
        got = here.restaged(name, RunNames(label, stage or "01_freq"))
        assert got == (compose(label, role, "03_freq_half") if stage
                       else name), (name, got)
        # A THIRD STAGE'S FILE is not the source's: as it is.
        assert here.restaged(name, RunNames(label, "99_other")) == name
    src, here = RunNames(LABEL, "01_freq"), RunNames(LABEL, "03_freq_half")
    assert here.restaged(src.name(ROLE_PIPELINE_LOG, engine="siesta"),
                         src) == here.name(ROLE_PIPELINE_LOG, engine="siesta")


@pytest.mark.parametrize("stage,expected", [
    (None,        "my-job.chk"),
    ("01_coarse", "my-job_01_coarse.chk"),
])
def test_the_stage_follows_the_label_and_nothing_follows_the_stage(
        stage, expected):
    """`job-contracts.md` § 2.2a: ``<label>[_<stage>]<role>`` -- `_`
    introduces the stage, and nothing stands between it and the role: a run
    folder is one start, and the folder names the run
    (`project-layout.md` § 1.6.1)."""
    assert compose(LABEL, ".chk", stage) == expected


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
    assert parse(compose(LABEL, ".chk"), LABEL).stage is None
    assert parse(compose(LABEL, "_optimized.xyz"), LABEL).stage is None
    assert parse(compose(LABEL, ".molwatch.log", "01_coarse"),
                 LABEL).stage == "01_coarse"


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
    """A positional name silently reassigns outputs the moment the ladder grows a
    rung (§ 6.3).  Refused as a wrong TYPE, at the call."""
    with pytest.raises(RunFileError, match="POSITION is not a token"):
        compose(LABEL, ".molwatch.log", bad)


@pytest.mark.parametrize("field", [{"run": 0}, {"seg": 2}],
                         ids=["a-try-number", "an-invented-field"])
def test_a_field_the_role_does_not_carry_is_refused(field):
    """A name carries what its role declares and nothing more: a try's
    number is no part of a run's names (`project-layout.md` § 1.6.1), and
    a caller inventing a field at the call site would spell a name nothing
    reads back.  THE SILENT FAILURE: a ``run=`` left at a call site still
    composing a ``-run0`` name that no reader of the run finds."""
    with pytest.raises(RunFileError, match="carries no field"):
        compose(LABEL, ".out", "01_coarse", **field)


@pytest.mark.parametrize("name,label", [
    ("my-job-v2.out", LABEL),
    ("my-job_01_coarse-run2.out", LABEL),
    ("my-job_01_coarse-run2.out", "my-job_01_coarse"),
], ids=["a-longer-label", "a-try-number-on-the-label",
        "a-try-number-on-the-stem"])
def test_a_hyphen_after_the_label_or_the_stage_is_no_file_of_it(name, label):
    """`-v2` after the label is part of a longer LABEL -- which is why a
    bench trial's `bdt-G1N4C6` reads as a label of its own -- and a name an
    older molbuilder gave a try (``-run2``) is no file of the run: runs
    written before one start per run are not read (`project-layout.md`
    § 1.6.1)."""
    assert parse(name, label) is None
    assert parse("my-job-v2.out", "my-job-v2") == expect("my-job-v2", None,
                                                         ".out")


# --------------------------------------------------------------------- #
#  The catalogue, and the two views of it                               #
# --------------------------------------------------------------------- #
#
#  `WRITTEN` says what molbuilder writes; `manifest()` the concrete names
#  (the Task-setup card).


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


@pytest.mark.parametrize("art", ON_THE_LABEL, ids=lambda a: a.role)
def test_a_file_that_carries_no_stage_never_grows_one(art):
    """The three that belong to the CALCULATION -- the template and the
    source pair -- are written once at the bundle root, so a stage token in
    one would claim a rung wrote it."""
    got = compose("my-job", art.role, "01_coarse" if art.staged else None,
                  **_fields_for(art))
    assert ("_01_coarse" in got) is art.staged


@pytest.mark.parametrize("engine,absent,present", [
    # The DECK each engine runs, and the file each writes its stdout to.
    # `.out` is SIESTA's: PySCF under the wrapper writes `.pyscf.log` and
    # never `.out`, so a PySCF rung's card promised a file no run produces
    # (`model/parse.md` § 5.5).  So is the timing tee's file: it reads the
    # SIESTA family's rows.  The monitor's two are EVERY engine's -- the
    # wrapper starts it from its shared part
    # (`run-reports.md` § 2.3).
    ("pyscf", (".fdf", "_01_coarse.out", "_01_coarse.scf-timing.log"),
     (".py", "_01_coarse.pyscf.log", "_01_coarse.monitor.log",
      "_01_coarse.util.csv")),
])
def test_the_manifest_tells_a_run_about_its_own_engine_only(
        engine, absent, present):
    rows = manifest(LABEL, "01_coarse", engine=engine)
    names = [r["name"] for r in rows]
    for suffix in present:
        assert any(n.endswith(suffix) for n in names), f"{engine}: no {suffix}"
    for suffix in absent:
        assert not any(n.endswith(suffix) for n in names), (
            f"{engine}: named {suffix}, which its engine does not write")
    # No engine named is a question, not a claim: answer for both.
    both = [r["name"] for r in manifest(LABEL, "01_coarse")]
    assert any(n.endswith(".py") for n in both)
    assert any(n.endswith(".fdf") for n in both)
    assert any(n.endswith("_01_coarse.out") for n in both)


def test_the_catalogue_and_the_warm_vocabulary_do_not_overlap():
    """WHAT MAKES THE SUBTRACTION WORK (`job-contracts.md` § 4.2): a file is
    the engine's restart state OR one molbuilder wrote, never both.  A role on
    both lists would make *"did the engine leave something here"* count an
    engine's restart file as one molbuilder wrote.
    """
    from molbuilder.warmfiles import warm_list
    ours = {a.role for a in ON_THE_LABEL}
    for engine in ("siesta", "pyscf"):
        warm = set(warm_list(engine).suffixes)
        assert not (ours & warm), (
            f"{engine}: {sorted(ours & warm)} is claimed by both the warm "
            f"vocabulary and the catalogue of what molbuilder writes")


def test_the_role_is_what_says_which_engine_wrote_a_file():
    """WHY THE NAME CARRIES NO ENGINE SEGMENT (user asked, 2026-09-07:
    *"engine name should be part of the name?"*).

    It already does, wherever a file is an engine's at all: the two engines'
    roles are DISJOINT, both for what they leave behind (`.XV`/`.DM` against
    `.chk`/`_optimized.xyz`) and for what molbuilder writes for them (`.fdf`
    against `.py` / `.pyscf.log` / `_geom_optim.xyz`).  A fifth segment would restate
    what the suffix already says, and a name with two sources of truth for one
    fact is what § 2.2a exists to stop.

    The rest -- the wrapper, its logs, the trajectory -- is genuinely shared:
    ONE wrapper runs both engines, so an engine token on `.run.sh` would be a
    claim about the file that is not true of it.
    """
    from molbuilder.warmfiles import warm_list
    assert not set(warm_list("siesta").suffixes) & set(warm_list("pyscf").suffixes)
    per_engine = {e: {a.role for a in ON_THE_LABEL if a.engine == e}
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
    from molbuilder.warmfiles import warm_list
    for engine in ("siesta", "pyscf"):
        for suffix in warm_list(engine).suffixes:
            name = compose(LABEL, suffix)
            assert parse(name, LABEL).stage is None, name
            assert LABEL + suffix == name, (
                f"{name!r} carries something between the label and the role; "
                f"the next rung looks for {LABEL + suffix!r}")


def test_a_relaxation_is_not_promised_a_spectrum():
    """A list that named a file the run will never write is the same fault as
    one that omits a file it does -- an answer a person cannot check against
    the folder.  `.spectra.json` is the vibration calculation's."""
    relax = {r["name"] for r in manifest(LABEL, None, engine="pyscf",
                                         calculation="optimization")}
    vib = {r["name"] for r in manifest(LABEL, None, engine="pyscf",
                                       calculation="vibration")}
    assert compose(LABEL, ".spectra.json") not in relax
    assert compose(LABEL, ".spectra.json") in vib
    # Everything a relaxation writes, a vibration writes too: the kind ADDS.
    assert relax < vib
    # And a caller that has not asked which kind is told about both, because
    # None here is "no question asked", not "the default kind".
    assert compose(LABEL, ".spectra.json") in {
        r["name"] for r in manifest(LABEL, None, engine="pyscf")}


# ══ THE READER — find ═══════════════════════════════════════════════════════

class TestTheReader:
    """`find` — `compose`'s counterpart (project-layout § 4.5)."""

    @staticmethod
    def _lay(d, names):
        for n in names:
            (d / n).write_text("")
        return d

    def test_only_our_files_come_back(self, tmp_path):
        self._lay(tmp_path, ["JOB.out", "JOB.fdf",
                             "somebody-elses.log", "OTHER.out"])
        got = [p.name for p, _ in find(tmp_path, "JOB")]
        assert got == ["JOB.fdf", "JOB.out"], got

    def test_a_role_narrows_it(self, tmp_path):
        self._lay(tmp_path, ["JOB.out", "JOB.concluded", "JOB.fdf"])
        got = [p.name for p, _ in find(tmp_path, "JOB", role=".out")]
        assert got == ["JOB.out"], got

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
    (tmp_path / "JOB.out").write_text("")
    (tmp_path / "JOB.concluded").mkdir()         # a directory that parses
    got = [p.name for p, _ in find(tmp_path, "JOB")]
    assert got == ["JOB.out"], got
