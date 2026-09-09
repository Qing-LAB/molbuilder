"""What the new doors ANSWER — the migration's outcomes, not its text.

`tests/test_path_framework.py` proves nobody hand-spells a name any more.  That
is a claim about the source.  This file is the other half: each call site moved
onto a door, so each door must give the answer the glob gave — and in the three
places where it now gives a BETTER one, that difference is asserted rather than
hoped for.

Every test here builds files on disk and asks the function.  None of them reads
source text.
"""
from __future__ import annotations

import pytest

from molbuilder import runfiles
from molbuilder.runfiles import canonical_role, find, find_by_role


# `_stage_state(observed, launch, label, stage, out_glob)` -- the label and the
# stage come BEFORE the glob, and none of the three has a default.  These three
# tests caught the reorder by failing, which is the argument for them: an empty
# label matches nothing, so a positional slip reports every rung as unstarted.
def _touch(d, *names):
    for n in names:
        p = d / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x", encoding="utf-8")
    return d


# --------------------------------------------------------------------------- #
# 1.  The patterned role -- the one row in the catalogue that is a family.     #
# --------------------------------------------------------------------------- #

def test_no_role_in_the_catalogue_is_a_GLOB():
    """A wildcard in a role is a coordinate that escaped into the vocabulary.

    `.runwrap-*.log` was declared that way until 2026-09-08, and the module had
    to grow `role_matches` to compare one — a whole mechanism to chase a star.
    § 5l.3: the stamp is a **field**, so the role is a template and every
    comparison is an ordinary equality again. A new starred row would bring the
    mechanism back, so it fails here instead.
    """
    starred = sorted(a.role for a in runfiles.WRITTEN if "*" in a.role)
    assert not starred, (
        f"these rows declare a glob as a role: {starred}. If the name varies, "
        f"declare the varying part as a FIELD (`fields=(...)` on the row, its "
        f"shape in `runfiles.FIELDS`) — see § 5l.1.")


def test_every_declared_field_has_a_declared_shape():
    """A field whose shape is not written down cannot be read back out.

    The shapes live in one place for the reason `QUALIFIERS` does: `compose`
    validates against it and `parse` reads with it, so the two cannot disagree
    about what a stamp looks like.
    """
    for a in runfiles.WRITTEN:
        for f in a.fields:
            assert f in runfiles.FIELDS, (
                f"role {a.role!r} names the field {f!r} with no shape in "
                f"`FIELDS`")
        # and the template must actually mention what it declares
        assert set(a.fields) == set(runfiles._fields_in(a.role)), (
            f"role {a.role!r} declares fields {a.fields} but its template "
            f"names {runfiles._fields_in(a.role)}")


def test_canonical_role_maps_a_name_back_to_its_TEMPLATE():
    """The one reading that replaced `role_matches`.

    A plain role must come back unchanged and must NOT become fuzzy — ``.out``
    matching ``.pyscf.out``, or ``.log`` matching ``.pyscf.log``, is the failure
    mode that would make every comparison in the module quietly approximate.
    """
    assert canonical_role(".runwrap-20260908-120000.log") == (
        ".runwrap-{stamp}.log", {"stamp": "20260908-120000"})
    for plain in (".out", ".pyscf.log", ".log", "_geom.log"):
        assert canonical_role(plain) == (plain, {}), plain
    # a stamp-shaped thing that is not in the right place stays foreign
    assert canonical_role(".runwrap-nope.log") == (".runwrap-nope.log", {})


def test_a_field_round_trips_through_compose_and_parse():
    """`parse` then `.name` must give back the byte-identical filename.

    That is what makes the template a real declaration rather than a label: the
    grammar can rebuild the exact name it read.
    """
    written = "bdt_01_tight.runwrap-20260908-120000.log"
    rf = runfiles.parse(written, "bdt")
    assert rf is not None
    assert rf.role == ".runwrap-{stamp}.log"
    assert dict(rf.fields) == {"stamp": "20260908-120000"}
    assert rf.name == written


def test_a_template_cannot_be_composed_without_its_field():
    """An omission is refused, and so is a value of the wrong shape.

    Both would produce a name `parse` could not read back — which is exactly
    what the field's declared shape exists to prevent.
    """
    with pytest.raises(runfiles.RunFileError) as e:
        runfiles.compose("bdt", ".runwrap-{stamp}.log")
    assert "needs the field" in str(e.value)
    with pytest.raises(runfiles.RunFileError) as e:
        runfiles.compose("bdt", ".runwrap-{stamp}.log", stamp="not-a-stamp")
    assert "declared shape" in str(e.value)


def test_the_glob_family_is_unchanged_by_the_field():
    """`patterns()` must still emit `{label}.runwrap-*.log`.

    `identity.OUR_FILE_PATTERNS` and `runwrap`'s `--cold` sweep read that view,
    and a `--cold` run that stopped recognising the wrapper log would leave it to
    be appended to by the next launch. The star belongs in the GLOB view, which
    is the only place it is now produced.
    """
    from molbuilder.identity import OUR_FILE_PATTERNS
    fam = [p for p in runfiles.patterns() if "runwrap" in p]
    assert fam == ["{label}.runwrap-*.log", "{label}_*.runwrap-*.log"]
    assert [p for p in OUR_FILE_PATTERNS if "runwrap" in p] == fam


def test_find_answers_for_the_wrapper_log_family(tmp_path):
    """The question `summarize._wrapper_log` asks, through the door.

    Two launches leave two stamped logs; the newest is wanted, and the stamp is
    ``%Y%m%d-%H%M%S`` so name order IS time order.  A file belonging to another
    rung and one belonging to another label must not be in the answer — which
    the old ``glob(f"{basename}.runwrap-*.log")`` got right by luck of the
    prefix and this gets right by parsing.
    """
    d = _touch(tmp_path,
               "bdt_01_tight.runwrap-20260101-000000.log",
               "bdt_01_tight.runwrap-20260908-120000.log",
               "bdt_02_fine.runwrap-20260908-130000.log",
               "other.runwrap-20260908-140000.log")
    hits = [p.name for p, _rf in find(d, "bdt", role=".runwrap-{stamp}.log",
                                     stage="01_tight")]
    assert hits == ["bdt_01_tight.runwrap-20260101-000000.log",
                    "bdt_01_tight.runwrap-20260908-120000.log"]
    assert hits[-1].endswith("20260908-120000.log"), "newest is last"


def test_find_by_role_answers_for_the_family_without_a_label(tmp_path):
    """Label-less, because the catalogue's role is recognisable on its own."""
    d = _touch(tmp_path, "a.runwrap-20260101-000000.log",
               "b.runwrap-20260102-000000.log", "a.out")
    assert [p.name for p in find_by_role(d, ".runwrap-{stamp}.log")] == [
        "a.runwrap-20260101-000000.log", "b.runwrap-20260102-000000.log"]


def test_find_by_role_still_refuses_a_role_the_catalogue_does_not_declare():
    """A typo is a refusal, not an empty list — the reason the check is there."""
    with pytest.raises(runfiles.RunFileError) as e:
        find_by_role(".", ".runwrap.log")           # no star: not a row
    assert "not a role molbuilder writes" in str(e.value)


def test_wrapper_log_reports_absence_as_absence(tmp_path):
    """`summarize._wrapper_log` answers None, not a name nothing writes.

    It used to return ``<basename>.runwrap-none.log`` — a composed spelling for
    a file that cannot exist, which is the same handcraft pointing the other
    way, and which any caller that printed the path would have shown a person.
    """
    from molbuilder.jobset.summarize import _wrapper_log
    assert _wrapper_log(tmp_path, "bdt") is None
    _touch(tmp_path, "bdt.runwrap-20260908-090000.log",
           "bdt.runwrap-20260908-173000.log")
    got = _wrapper_log(tmp_path, "bdt")
    assert got is not None, "two launches are here"
    # THE NEWEST, not the first: the caller compares what the run was ASKED to
    # do against what it DID, and an older session's log describes a different
    # run.  Two files, so first-vs-last is a real distinction here.
    assert got.name.endswith("-20260908-173000.log")


# --------------------------------------------------------------------------- #
# 2.  runstatus: the narrowing must keep PySCF's own output visible.           #
# --------------------------------------------------------------------------- #

def test_a_pyscf_rung_that_only_wrote_pyscf_log_is_not_reported_queued(tmp_path):
    """§ 1.6's forbidden line, in the shape the migration could have caused.

    PySCF under the wrapper writes ``.pyscf.log`` — the catalogue says so
    explicitly: *"the same, for PySCF under the wrapper — it writes here and
    not to .out."*  So narrowing the existence check to the exact roles ``.out``
    and ``.log`` would have made a finished PySCF rung answer *"queued"*, which
    is the one answer status is not allowed to invent.  The check asks for the
    role FAMILIES for that reason.
    """
    from molbuilder.jobset.runstatus import _stage_state
    _touch(tmp_path, "bdt_01_relax.pyscf.log")
    state, _detail = _stage_state(tmp_path, {"job_id": 481923},
                                  "bdt", "01_relax", "bdt_01_relax*")
    assert state != "queued", (
        "a rung whose engine wrote .pyscf.log has produced output; reporting "
        "it queued is § 1.6's exact forbidden line")


def test_a_flat_rung_that_never_ran_does_not_read_its_siblings_output(tmp_path):
    """One directory, two rungs: the token in the filename is what selects.

    This is the defect the ``out_glob`` argument was added for, re-asserted
    against the grammar-based narrowing that replaced it — the failure is
    silent and reads as *success*, which is why it needs its own test.
    """
    from molbuilder.jobset.runstatus import _stage_state
    _touch(tmp_path, "bdt_01_coarse.out")            # only the FIRST rung ran
    state, detail = _stage_state(tmp_path, None, "bdt", "02_fine",
                                 "bdt_02_fine*")
    assert state == "pending", (
        f"rung 02_fine has written nothing; got {state!r} ({detail!r}) — it "
        f"has read its sibling's .out")


def test_a_hierarchical_rung_is_found_although_the_shape_says_star(tmp_path):
    """The hierarchy's glob is ``*`` and the grammar's answer is the token.

    Both are correct: in the hierarchy the DIRECTORY already selected the rung,
    and the deck still carries the token because `prep` composes the name the
    same way in either layout.  So passing the token narrows nothing there —
    and must not narrow it to nothing, which is what would happen if a
    hierarchical deck were named without its stage.
    """
    from molbuilder.jobset.runstatus import _stage_state
    _touch(tmp_path, "bdt_01_tight.out")
    state, _d = _stage_state(tmp_path, {"mode": "direct"}, "bdt",
                             "01_tight", "*")
    assert state != "queued" and state != "pending"


# --------------------------------------------------------------------------- #
# 3.  The sidecar gained a finder; its composer already existed.               #
# --------------------------------------------------------------------------- #

def test_sidecars_in_finds_what_sidecar_path_for_composes(tmp_path):
    """The § 4.5 pairing, asserted as a round trip rather than as two lists."""
    from molbuilder.sidecars.molstruct import (is_sidecar, sidecar_path_for,
                                               sidecars_in)
    for stem in ("relaxed", "bridge.spectra"):
        sidecar_path_for(tmp_path / f"{stem}.xyz").write_text("{}",
                                                             encoding="utf-8")
    found = sidecars_in(tmp_path)
    assert [p.name for p in found] == ["bridge.spectra.molstruct.json",
                                       "relaxed.molstruct.json"]
    assert all(is_sidecar(p) for p in found)
    assert not is_sidecar(tmp_path / "relaxed.xyz")


def test_the_citation_pair_uses_the_composer_not_a_suffix_slice(tmp_path):
    """`classify_citation` pairs each `.xyz` through `sidecar_path_for`.

    It sliced exactly ``.xyz`` off the name, where the composer strips the LAST
    suffix — the two agree only for names ending in ``.xyz``.  Asserted as the
    outcome: a compound-suffixed structure and its sidecar are recognised as a
    pair, which a caller with its own slice would also have got right only by
    matching the composer's rule by hand.
    """
    from molbuilder.sidecars.molstruct import sidecar_path_for
    from molbuilder.transport.compose import classify_citation
    xyz = tmp_path / "bridge.spectra.xyz"
    xyz.write_text("1\n\nAu 0 0 0\n", encoding="utf-8")
    sidecar_path_for(xyz).write_text("{}", encoding="utf-8")
    cited = classify_citation(tmp_path)
    assert cited.form == "structure"
    assert cited.xyz.name == "bridge.spectra.xyz"
    assert cited.sidecar.name == "bridge.spectra.molstruct.json"


def test_a_directory_holding_only_a_deck_is_still_refused_by_name(tmp_path):
    """The § 4.1b message keeps naming what is missing.

    `.fdf` now comes from `find_by_role`; the refusal is the behaviour that
    matters, and it must still say WHICH file the condition wants.
    """
    from molbuilder.transport.compose import ComposeError, classify_citation
    (tmp_path / "bdt.fdf").write_text("SystemLabel bdt\n", encoding="utf-8")
    with pytest.raises(ComposeError) as e:
        classify_citation(tmp_path)
    assert ".fdf but no .XV" in str(e.value)


# --------------------------------------------------------------------------- #
# 4.  The rest of the migrated callers, asked for their answers.               #
# --------------------------------------------------------------------------- #

def test_the_provenance_step_reads_the_wrapper_as_well_as_the_deck(tmp_path):
    """`parse.contract` asks the catalogue for `.run.sh`, `.fdf` and `.py`.

    The wrapper is the point: a TranSIESTA run has no deck PROVENANCE but
    always has a `.run.sh`, so a search that lost that role would silently stop
    seeing every transport run's engine declaration.  The block here declares
    `siesta` because `_declared_in_provenance` only reports a name the parse
    registry knows -- an unregistered spelling is dropped on purpose, which is
    what this test first tripped over.
    """
    from molbuilder import script_emit as se
    from molbuilder.parse.contract import _declared_in_provenance
    # THE BLOCK IS WRITTEN THROUGH THE EMITTER'S OWN MARKERS, so this fixture
    # cannot drift from what a real wrapper carries.
    (tmp_path / "j.run.sh").write_text(
        se.begin_marker(se.BLOCK_PROVENANCE) + "\n"
        "#  engine  siesta\n"
        + se.end_marker(se.BLOCK_PROVENANCE) + "\n", encoding="utf-8")
    assert _declared_in_provenance(tmp_path) == {"siesta"}
    # ...and the deck-less case is the point: nothing but the wrapper is here.
    assert not list(tmp_path.glob("*.fdf"))


def test_the_watch_resolver_finds_a_molwatch_log_first(tmp_path):
    """Step 1 of `_resolve_run_directory`, through `find_by_role`."""
    from molbuilder.web.blueprints.watch import _resolve_run_directory
    _touch(tmp_path, "bdt.molwatch.log", "bdt.out")
    chosen, attempts = _resolve_run_directory(str(tmp_path))
    assert chosen is not None and chosen.endswith("bdt.molwatch.log")
    assert any("molwatch" in a for a in attempts)


def test_the_watch_resolver_falls_through_to_engine_stdout(tmp_path):
    """Step 4's `*.out`, with no molwatch log and no readable deck."""
    from molbuilder.web.blueprints.watch import _resolve_run_directory
    _touch(tmp_path, "bdt.out")
    chosen, _attempts = _resolve_run_directory(str(tmp_path))
    assert chosen is not None and chosen.endswith("bdt.out")


def test_find_template_still_refuses_two_answers(tmp_path):
    """`template.find_template` raises rather than picking, and still does.

    Two templates in one bundle is ambiguity a person must resolve; a finder
    that silently returned the first would make the wrong run reproducible.
    """
    from molbuilder.template import SUFFIX, find_template
    assert find_template(tmp_path) is None, "nothing here is not I cannot tell"
    for stem in ("a", "b"):
        (tmp_path / f"{stem}{SUFFIX}").write_text("", encoding="utf-8")
    with pytest.raises(ValueError) as e:
        find_template(tmp_path)
    assert "holds 2 templates" in str(e.value)


def test_the_geometry_picker_takes_its_pyscf_spellings_from_their_home():
    """`projects._geom_output_patterns` asks `pyscf.input`, not a fourth copy.

    Asserted as an equality with the declared constants, because the failure is
    a spelling drifting apart — which is how ``_geom_optim.xyz`` came to have
    six of them.
    """
    from molbuilder.projects import _geom_output_patterns
    from molbuilder.pyscf.input import ROLE_GEOM_TRAJ, ROLE_OPTIMIZED
    pats = _geom_output_patterns()
    assert "*" + ROLE_OPTIMIZED in pats
    assert "*" + ROLE_GEOM_TRAJ in pats
    assert "*.STRUCT_OUT" in pats, "SIESTA's own name has no home of ours"


def test_stage_state_cannot_be_asked_without_a_label(tmp_path):
    """No default for the label, because an empty one matches NOTHING.

    `runfiles.parse` states the rule this rests on: *"``label`` is required, and
    that is the whole reason this can be exact."*  So a default of ``""`` would
    not degrade — it would report *"prepped, not launched"* for a rung that has
    finished, which is § 1.6's forbidden line reached by a signature rather than
    by a bug.  Asserted as the outcome pair: the call is refused without a
    label, and answers with one, on the same directory.
    """
    from molbuilder.jobset.runstatus import _stage_state
    _touch(tmp_path, "bdt_01_tight.out")
    with pytest.raises(TypeError):
        _stage_state(tmp_path, None)                       # no label, no stage
    state, _d = _stage_state(tmp_path, None, "bdt", "01_tight", "*")
    assert state != "pending", (
        "with the label given, this rung's .out is visible — which is exactly "
        "what a defaulted empty label would have hidden")


def test_read_system_degrades_on_a_missing_bundle():
    """It is a REPORTER: absence degrades rather than raises.

    Its own docstring promises that, and the M7 migration broke it — the
    `glob("*/*.fdf")` it replaced yields nothing for a directory that is not
    there, while a bare `iterdir()` raises `FileNotFoundError`.  Measured
    2026-09-08 by re-reading the diff, not by any failing test, which is why
    this one exists.
    """
    from pathlib import Path

    from molbuilder.jobset.summarize import _read_system
    assert _read_system(Path("/nonexistent/bundle/no-such-thing")) == {
        "engine": "siesta"}


def test_attempt_concluded_answers_for_a_persons_own_deck_name(tmp_path):
    """A cited directory holds whatever the person named their deck.

    `transport.classify_citation` calls this with `deck.stem`, and a cited
    relaxation is not necessarily one molbuilder prepped — `my.relaxation.fdf`
    is a legal thing to cite.  Composing the marker with `runfiles.compose`
    would **raise** there (§ 2.1: a label carrying a dot cannot be read back out
    of a filename, and that refusal is right), turning *"this directory has no
    record"* into a crash at somebody who used a dot.  `runfiles.tail` is the
    door for a caller holding a stem it did not choose: the grammar owns the
    ``-run<N>.concluded`` end, the caller owns the stem.

    Measured 2026-09-08: the first M8 spelling raised RunFileError here where
    the code it replaced returned the marker.
    """
    from molbuilder.jobset.materialize import attempt_concluded
    (tmp_path / "my.relaxation-run0.concluded").write_text("rc=0\n",
                                                           encoding="utf-8")
    assert attempt_concluded(tmp_path, "my.relaxation") == "rc=0"
    # ...and our own naming still answers, which is what makes the test above
    # a discrimination rather than a blanket loosening.
    (tmp_path / "bdt_01_tight-run2.concluded").write_text("rc=1 (walltime)\n",
                                                          encoding="utf-8")
    assert attempt_concluded(tmp_path, "bdt_01_tight") == "rc=1 (walltime)"
    assert attempt_concluded(tmp_path, "never_ran") is None


def test_the_watch_resolver_survives_a_dotted_script_filename(tmp_path):
    """A dot in a filename must not 500 the Watch tab.

    `_resolve_run_directory` reads `JOB` and `SystemLabel` through regexes
    bounded to ``[A-Za-z0-9_-]+`` — deliberately, so a malformed deck cannot
    inject a path. But ``py_stem`` is the deck's FILENAME stem, taken off disk
    and unbounded, and it goes into `runfiles.compose`: ``my.job.py`` gives
    ``my.job``, which § 2.1 refuses (rightly — a dotted label cannot be read
    back out of a filename).

    A resolver's contract is to try each step, SAY what it tried, and fall
    through. Raising there turns "I could not resolve this directory" into a
    server error at somebody who used a dot. Introduced by ``3dfa76c9`` when
    these names moved onto the composer; measured 2026-09-08.
    """
    from molbuilder.web.blueprints.watch import _resolve_run_directory
    (tmp_path / "my.job.py").write_text("JOB = 'myjob'\n", encoding="utf-8")
    (tmp_path / "run.out").write_text("done\n", encoding="utf-8")
    chosen, attempts = _resolve_run_directory(str(tmp_path))
    assert chosen is not None and chosen.endswith("run.out"), (
        "the resolver should fall through to its generic step, not raise")
    assert any("not a run-file label" in a for a in attempts), (
        f"and it should SAY why it skipped the stem: {attempts}")


# ── N4: the two sites that stopped spelling the layout ───────────────────────

def test_the_container_search_answers_without_a_declared_shape(tmp_path):
    """A caller with no description to read a shape from still gets the
    containers -- and gets only the DECLARED ones.

    `paths.bench_container` is the namer; `bench_containers_in` is its search
    half, and `shape=None` is the arm `jobset._cli`'s error path needs (it runs
    when there is no readable `job-set.json`, which is the whole reason it runs).
    A missing union arm returns nothing and the CLI silently stops listing
    sweeps, which is why this is a test and not a review note.

    `engines/stages.md` § 6.7 is not violated: that rule is a DESCRIPTION
    declaring its shape, and this is a search saying it does not know which tree
    it walks. The two layouts' containers cannot collide, so the union is exact.
    """
    from molbuilder.paths import Shape, bench_containers_in
    for d in ("bench", "bench_01_coarse", "01_coarse/bench", "02_tight/bench",
              "01_coarse/run-0", "not_a_container"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    either = dict(bench_containers_in(tmp_path))
    assert either == {"bench": None, "bench_01_coarse": "01_coarse",
                      "01_coarse/bench": "01_coarse", "02_tight/bench": "02_tight"}
    # and each layout alone is a subset of it, never something else
    for name in ("hierarchical", "flat"):
        one = dict(bench_containers_in(tmp_path, Shape.named(name)))
        assert set(one) <= set(either)


def test_the_sweep_search_finds_the_declared_containers_and_not_a_stage(tmp_path):
    """`sweep_set_paths` returns a `job-set.json` in a bench container, and NOT
    one sitting in a stage directory.

    Its two globs were `*/job-set.json` and `*/*/job-set.json` -- WIDER than the
    rule, because `*/` at depth 1 matches any directory. Its own docstring said
    it could not ask (*"one door to COMPOSE a path, none to FIND one -- is what
    the paths framework is for; when it lands, this is one of its callers"*),
    and N4 (2026-09-09) is that landing. The decoy below is what the globs would
    have returned and the rule never places.

    `job-contracts.md` § 6.3 via `paths.bench_container`: a sweep's record goes
    in the container, a ladder's goes at the bundle root.

    **It also returns only paths that EXIST.** Composing from the containers
    means composing a name whether or not the file is there, where the globs it
    replaced could only return what they found -- so a container prepared but
    never written would hand the caller a path to nothing, and `jobset._cli`
    swallows the failed load in a bare `except`. Silent, so it is asserted.
    (Added 2026-09-09: a mutant that dropped the existence check left the rest
    of this file green, because the first version seeded every container.)
    """
    from molbuilder.jobset.materialize import sweep_set_paths
    for d in ("bench", "bench_01_coarse", "01_coarse/bench", "01_coarse"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
        (tmp_path / d / "job-set.json").write_text("{}", encoding="utf-8")
    (tmp_path / "02_tight" / "bench").mkdir(parents=True)   # a container, no set
    got = {str(p.parent.relative_to(tmp_path)) for p in sweep_set_paths(tmp_path)}
    assert got == {"bench", "bench_01_coarse", "01_coarse/bench"}
    assert all(p.is_file() for p in sweep_set_paths(tmp_path))


def test_the_trial_search_reads_the_prefix_through_its_own_reader(tmp_path):
    """`materialize.trials_in` returns the trial directories and skips a
    directory that merely sits beside them.

    `paths.trial_name` composes `bench-<point>` and `paths.trial_point` reads it
    back; this door spelled `startswith(TRIAL_PREFIX)` itself until N4. The pair
    is § 4.5's rule, and a door that re-spells one half is how the two drift.
    """
    from molbuilder.jobset.materialize import trials_in
    c = tmp_path / "01_coarse" / "bench"
    for d in ("bench-G1", "bench-G2K4", "notatrial", "bench-"):
        (c / d).mkdir(parents=True, exist_ok=True)
    assert [p.name for p in trials_in(c)] == ["bench-G1", "bench-G2K4"]
