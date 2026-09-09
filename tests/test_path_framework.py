"""The guard for `project-layout.md` § 4.5 -- and the proof that it can fail.

**THE RULE.**  *For every name it composes, the framework owns the search.*  A
name molbuilder writes has one composer (`runfiles.compose`, `paths.attempt_dir`,
`materialize.trial_dir`, `sidecars.molstruct.sidecar_path_for`); § 4.5 says it
must also have exactly one FINDER, because a caller with a question and no
finder spells a glob, and every spelling is a place the rule can drift.  That
is not a hypothesis: the survey found 24 such sites on 2026-09-08, one of them
added by the very commit that closed the previous one.

**WHY A TEST AND NOT A REVIEW ITEM.**  A migration is provable only if the
finished state is checkable.  "17 is fewer than 24" is a progress report; a
check that fails the build when someone hand-spells the twenty-fifth is a
guarantee.  `tools/classify_path_finders.py` is the instrument -- an AST pass
over `molbuilder/`, not a text grep -- and this is the assertion.

**THE EXEMPTIONS ARE PART OF THE GUARANTEE, NOT A HOLE IN IT.**  Some searches
are for names molbuilder does not compose (SIESTA's `.XV`, conda-meta's
`*.json`), and § 4.5 gives those no door by design.  Each is an entry in
`_OVERRIDES` carrying the reason someone wrote after reading the site.  Two
things keep that from becoming a silencer: an exemption that matches no site is
itself a failure (:func:`stale_overrides`), and the buckets an exemption may
name are the non-failing ones only -- an override cannot mark a handcrafted
site as fine, it can only say which OTHER thing the site is.

**MUTATION-TESTED.**  A guard that asserts the tree is clean passes just as
happily when the classifier has stopped classifying, so the second test points
the same survey at a throwaway package holding one hand-spelled glob and
requires the verdict `owned`.  That is the failure the first test is claiming
to prevent, demonstrated rather than asserted.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "classify_path_finders.py"


def _tool():
    """Import the survey as a module -- it is a tool, not a package member."""
    spec = importlib.util.spec_from_file_location("_classify_path_finders", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# 1.  The guard.                                                              #
# --------------------------------------------------------------------------- #

def test_no_path_search_is_handcrafted():
    """Every search in `molbuilder/` is a door, door-fed, or recorded foreign.

    The failure message names the site and the door it should have asked, so a
    person who hits this in CI is told what to call rather than told off.
    """
    t = _tool()
    rows = t.survey()
    bad = [r for r in rows if r["verdict"].startswith(t.FAILING_VERDICTS)]
    assert not bad, (
        "these path searches spell a name the framework composes "
        "(`project-layout.md` § 4.5 -- for every name it composes, the "
        "framework owns the search):\n"
        + "\n".join(
            f'  {r["file"]}:{r["line"]}  {r["func"]}()  '
            f'{r["call"]}({r["pattern"]!r})'
            + (f'\n      -> ask {r["composer"]}' if r["composer"] else "")
            + (f'\n      ({r["names"]})' if r["names"] else "")
            for r in bad)
        + "\n\nIf the name is genuinely not ours, read the site and record it "
          "in `_OVERRIDES` in tools/classify_path_finders.py with the reason.")


def test_no_recorded_exemption_has_come_unanchored():
    """An override that matches no site is a failure, not a leftover.

    `classify_source_reads.py` keyed its reasons by LINE NUMBER; deleting tests
    around them displaced two and landed one on an unrelated assertion, and the
    tool then reported a number for work nobody had done (2026-09-08).  These
    keys are (file, function, pattern), which survives an edit above the site
    -- but not a rename, a move, or a fix.  A dead exemption is how a rule
    stops applying without anyone deciding that, so it fails here.
    """
    t = _tool()
    stale = t.stale_overrides(t.survey())
    assert not stale, (
        "these recorded exemptions match no site any more -- the site was "
        "renamed, moved, or fixed.  Delete the entry (if fixed) or re-anchor "
        "it (if moved):\n" + "\n".join(f"  {k}" for k in stale))


def test_an_exemption_cannot_excuse_a_handcrafted_site():
    """The buckets an override may name exclude the failing ones.

    Otherwise the guard has a back door: any site could be silenced by adding
    ``("owned - ...", "we'll do it later")`` beside it, and the check would go
    green while the codebase got worse.  An override says which OTHER thing a
    site is; it cannot say that being handcrafted is acceptable.
    """
    t = _tool()
    offenders = {key: verdict for key, (verdict, _why) in t._OVERRIDES.items()
                 if verdict.startswith(t.FAILING_VERDICTS)}
    assert not offenders, (
        "an override may not assign a failing verdict -- that would make the "
        "guard silenceable:\n"
        + "\n".join(f"  {k} -> {v!r}" for k, v in offenders.items()))


def test_every_exemption_carries_a_reason():
    """A bare verdict is not a decision anyone can re-check.

    The whole value of the exemption list is that someone READ the site; the
    reason is the evidence of that, and the next person needs it to know
    whether the judgement still holds.  **Non-empty, not long** -- a length
    floor would be a rule no document states, and some of these sites really
    are one clause ("conda-meta's own naming"); what must not happen is a
    verdict with nothing behind it.
    """
    t = _tool()
    thin = sorted(k for k, (_v, why) in t._OVERRIDES.items() if not why.strip())
    assert not thin, (
        "these exemptions state a verdict with no reason:\n"
        + "\n".join(f"  {k}" for k in thin))


# --------------------------------------------------------------------------- #
# 2.  The mutation test -- the guard must actually catch one.                  #
# --------------------------------------------------------------------------- #

_HANDCRAFTED = '''\
from pathlib import Path


def newest_output(d):
    """Spells `.out` -- a role `runfiles.WRITTEN` declares."""
    return sorted(Path(d).glob("*.out"))[-1]
'''

_HANDCRAFTED_ATTEMPT = '''\
from pathlib import Path


def attempts(d):
    """Spells the attempt-directory prefix `paths.ATTEMPT_PREFIX` owns."""
    return sorted(Path(d).glob("run-*"))
'''

_THROUGH_THE_DOOR = '''\
from molbuilder.runfiles import find_by_role


def newest_output(d):
    return find_by_role(d, ".out")[-1]
'''


@pytest.mark.parametrize("source,expect_owned", [
    (_HANDCRAFTED, True),
    (_HANDCRAFTED_ATTEMPT, True),
    (_THROUGH_THE_DOOR, False),
])
def test_the_guard_catches_a_new_handcrafted_search(tmp_path, source,
                                                   expect_owned):
    """Point the same survey at a package holding the defect.

    This is the test the first one needs in order to mean anything: it fails if
    the classifier stops recognising a role from the catalogue, or the attempt
    prefix, or starts calling a door a violation.  The third case is what keeps
    it from passing by simply flagging everything.
    """
    t = _tool()
    pkg = tmp_path / "molbuilder"
    pkg.mkdir()
    (pkg / "offender.py").write_text(source, encoding="utf-8")
    rows = t.survey(pkg=pkg, root=tmp_path)
    owned = [r for r in rows if r["verdict"].startswith("owned")]
    if expect_owned:
        assert owned, (
            "the guard did not notice a hand-spelled name the framework "
            f"composes.  Verdicts seen: {[r['verdict'] for r in rows]}")
        assert owned[0]["composer"], "and it did not name the door to ask"
        # ...AND THE VERDICT MUST BE ONE THE GUARD FAILS ON.  Recognising the
        # site is only half of it: emptying `FAILING_VERDICTS` leaves the
        # classifier working perfectly and every assertion in this file green,
        # which is a kill switch on the whole guard (measured 2026-09-08, this
        # mutation survived until this line was added).
        assert owned[0]["verdict"].startswith(t.FAILING_VERDICTS), (
            f"{owned[0]['verdict']!r} is recognised but not FAILING: the guard "
            f"would report this site and pass.  FAILING_VERDICTS is "
            f"{t.FAILING_VERDICTS!r}")
    else:
        assert not owned, (
            "a call THROUGH the door was reported as handcrafted: "
            f"{[(r['pattern'], r['verdict']) for r in rows]}")


def test_the_survey_sees_a_name_hidden_behind_a_constant(tmp_path):
    """An imported filename constant does not hide the site.

    The first pass matched literal patterns only, so `glob(f"*/{_JS}")` came
    back *unclassified* while `_JS` is `job-set.json` -- and a caller that
    knows the filename has a home and still assembles the search by hand is
    exactly the worst case, not an edge one.
    """
    t = _tool()
    pkg = tmp_path / "molbuilder"
    pkg.mkdir()
    (pkg / "offender.py").write_text(
        'from pathlib import Path\n'
        'ROLE = ".molwatch.log"\n\n\n'
        'def logs(d):\n'
        '    return sorted(Path(d).glob("*" + ROLE))\n'
        '    # noqa\n',
        encoding="utf-8")
    # `"*" + ROLE` is a BinOp, which `_pattern_of` unparses; the constant map
    # is what makes the f-string form resolvable.  Assert the f-string form,
    # which is the one that actually appeared in the codebase.
    (pkg / "offender2.py").write_text(
        'from pathlib import Path\n'
        'ROLE = ".molwatch.log"\n\n\n'
        'def logs(d):\n'
        '    return sorted(Path(d).glob(f"*{ROLE}"))\n',
        encoding="utf-8")
    rows = t.survey(pkg=pkg, root=tmp_path)
    hidden = [r for r in rows if r["file"].endswith("offender2.py")]
    assert hidden and hidden[0]["verdict"].startswith("owned"), (
        "a role reached through a module constant was not recognised: "
        f"{hidden}")


# --------------------------------------------------------------------------- #
# 3.  The vocabulary is the catalogue's, not the survey's own copy.            #
# --------------------------------------------------------------------------- #

def test_the_survey_derives_its_vocabulary_from_the_catalogue():
    """Every role in `runfiles.WRITTEN` is a name the survey recognises.

    The table was hand-copied beside the catalogue until 2026-09-08 and had
    already drifted -- it claimed `.XV` was ours, and `.XV` is SIESTA's own
    restart file, which `WRITTEN` deliberately excludes.  A survey with its own
    copy of the vocabulary is the very habit it exists to find, so the check is
    that adding a row to the catalogue is enough.
    """
    from molbuilder.runfiles import WRITTEN
    t = _tool()
    known = {needle for needle, _door, _what in t.OWNED}
    missing = sorted({a.role for a in WRITTEN} - known)
    assert not missing, (
        f"these catalogued roles are invisible to the survey: {missing}.  "
        f"`OWNED` must derive from `runfiles.WRITTEN`, never restate it.")


def test_the_survey_does_not_claim_an_engines_files():
    """`.XV` and `.STRUCT_OUT` are not in the survey's owned vocabulary.

    The other half of the same drift: a survey that claims an engine's output
    sends the next person looking for a door that cannot exist, because
    `job-contracts.md` § 4.2 is explicit that what an engine writes is not
    enumerable -- *"a snapshot pretending to be a rule."*
    """
    t = _tool()
    known = {needle for needle, _door, _what in t.OWNED}
    for engine_file in (".XV", ".DM", ".STRUCT_OUT", ".ANI", ".TSHS"):
        assert engine_file not in known, (
            f"{engine_file} is the ENGINE's; no door of ours composes it")


# --------------------------------------------------------------------------- #
# 4.  The COMPOSE half -- the population a search survey cannot see.          #
# --------------------------------------------------------------------------- #

def test_no_counter_keyed_name_is_built_by_hand():
    """`-run<N>` and `run-<N>` have one composer each, and nobody else spells them.

    The search guard above is blind to this by construction: a duplicate
    COMPOSER performs no search. `materialize.attempt_concluded` spelled
    ``f"{basename}-run{newest}.concluded"`` on the line *after* asking
    `runfiles.latest_run` for that counter, and `submit.py` built
    ``f"{names[j]}/run-{n}"`` and handed it to `prepare_attempt` as the attempt
    to continue FROM — a real path on the live continue-a-run route, not a
    message. Both were found by reading a diff, which is why they are a test now.
    """
    t = _tool()
    built = [r for r in t.compositions() if r["reason"] is None]
    assert not built, (
        "these build a counter-keyed name by hand instead of asking its one "
        "composer (`project-layout.md` § 4.5):\n"
        + "\n".join(f'  {r["file"]}:{r["line"]}  {r["func"]}()\n'
                    f'      {r["text"][:100]}\n'
                    f'      -> ask {r["door"]}'
                    for r in built)
        + "\n\nIf the number genuinely is not an attempt or a run index, read "
          "the site and record it in `_COMPOSE_OVERRIDES` with the reason.")


def test_no_compose_exemption_has_come_unanchored():
    """Same rule as the search side: a dead exemption is a rule silently lapsing."""
    t = _tool()
    stale = t.stale_compose_overrides(t.compositions())
    assert not stale, (
        "these compose-side exemptions match no site any more:\n"
        + "\n".join(f"  {k}" for k in stale))


_BUILDS_A_COUNTER = '''\
from pathlib import Path


def marker(d, basename, n):
    """Spells the wrapper's `-run<N>` counter, whose one composer is
    `runfiles.compose(run=N)`."""
    return Path(d) / f"{basename}-run{n}.concluded"
'''

_BUILDS_AN_ATTEMPT_PATH = '''\
def continue_from(stage, n):
    """Spells the attempt directory, whose composer is `paths.attempt_name`."""
    return f"{stage}/run-{n}"
'''

_ASKS_THE_COMPOSER = '''\
from pathlib import Path

from molbuilder.paths import attempt_name
from molbuilder.runfiles import compose


def marker(d, basename, n):
    return Path(d) / compose(basename, ".concluded", run=n)


def continue_from(stage, n):
    return f"{stage}/{attempt_name(n)}"
'''


@pytest.mark.parametrize("source,expect_built", [
    (_BUILDS_A_COUNTER, True),
    (_BUILDS_AN_ATTEMPT_PATH, True),
    (_ASKS_THE_COMPOSER, False),
])
def test_the_compose_guard_catches_a_new_hand_built_name(tmp_path, source,
                                                        expect_built):
    """Shown to fail, and shown NOT to fail on the correct spelling.

    The third case is what stops this passing by flagging every f-string that
    mentions a run: composing through the door still interpolates the number,
    just not next to the fragment.
    """
    t = _tool()
    pkg = tmp_path / "molbuilder"
    pkg.mkdir()
    (pkg / "offender.py").write_text(source, encoding="utf-8")
    rows = t.compositions(pkg=pkg, root=tmp_path)
    if expect_built:
        assert rows, "the guard did not notice a hand-built counter name"
        assert rows[0]["door"], "and it did not name the composer to ask"
        assert rows[0]["reason"] is None, "an unrecorded site must not read as excused"
    else:
        assert not rows, (
            f"a name built THROUGH the composer was flagged: "
            f"{[r['text'] for r in rows]}")


def test_the_grammar_modules_are_exempt_by_identity_not_by_override():
    """`runfiles` and `paths` compose these names for a living.

    Exempting them with an override would be a lie about why: they are not
    sites someone read and excused, they are the composers. So they are skipped
    by module path, and the list of them is short enough to be checkable.
    """
    t = _tool()
    assert t.GRAMMAR_MODULES == ("molbuilder/runfiles.py", "molbuilder/paths.py")
    for mod in t.GRAMMAR_MODULES:
        assert (ROOT / mod).is_file(), f"{mod} is exempted and does not exist"


# --------------------------------------------------------------------------- #
# 5.  The THIRD axis -- a directory of the hierarchy, spelled at a caller.    #
# --------------------------------------------------------------------------- #

def test_no_third_hierarchy_segment_is_invented():
    """`project-layout.md` § 2.6 is the authority on the tree.

    **What is guarded is the SET of undeclared segments, not the site count.**
    Ten sites build `launch/` and `pseudos/` today and § 5l.6 step N5 takes them
    to zero; what must not happen in the meantime is a *third* segment joining
    them silently — because that is a level of the tree created at a call site,
    and only § 2.6 may add one.

    Shrinking the set is the goal, so this asserts equality rather than
    membership: closing one is a deliberate edit here, not a quiet pass.
    """
    t = _tool()
    rows = t.segments()
    got = t.undeclared_segments(rows)
    assert got == sorted(t.GUARDED_UNDECLARED), (
        f"undeclared hierarchy segments are now {got!r}, guarded set is "
        f"{sorted(t.GUARDED_UNDECLARED)!r}.\n"
        + "\n".join(f'  {r["file"]}:{r["line"]}  {r["func"]}()  '
                    f'[{r["segment"]}]  {r["how"]}'
                    for r in rows if not r["declared"])
        + "\n\nIf a NEW segment appeared: declare it with a door "
          "(§ 5l.6 N5) or do not build it. If one was CLOSED: update "
          "GUARDED_UNDECLARED in tools/classify_path_finders.py.")


def test_no_segment_exemption_has_come_unanchored():
    """Third axis, same anchoring rule as the other two."""
    t = _tool()
    stale = t.stale_segment_overrides(t.segments())
    assert not stale, (
        "these segment exemptions match no site any more:\n"
        + "\n".join(f"  {k}" for k in stale))


def test_a_declared_segment_names_a_real_door():
    """A row saying "declared" must name something that exists.

    Otherwise the declaration is a comment: `bench` claims
    `materialize.bench_container` and `.binsnapshots` claims
    `checkpoint.ARCHIVE_DIR`, and a rename would leave the claim standing while
    the door moved.
    """
    import importlib
    t = _tool()
    for seg, door in t.CALC_CONTAINERS.items():
        if door is None:
            continue
        mod, _, attr = door.rpartition(".")
        obj = importlib.import_module(f"molbuilder.{mod}")
        assert hasattr(obj, attr), (
            f"segment {seg!r} claims the door molbuilder.{door}, "
            f"which does not exist")


_INVENTS_A_SEGMENT = '''\
from pathlib import Path


def stash(base):
    """Builds a container of the tree that nothing declares."""
    return Path(base) / "scratchpad"
'''

_INVENTS_IT_AS_A_STRING = '''\
def relative(name):
    return f"scratchpad/{name}.out"
'''

_A_FLASK_ROUTE = '''\
def register(bp):
    @bp.route("/api/scratchpad/summary")
    def summary():
        return {}
'''


@pytest.mark.parametrize("source,expect_hit", [
    (_INVENTS_A_SEGMENT, True),
    (_INVENTS_IT_AS_A_STRING, True),
    (_A_FLASK_ROUTE, False),
])
def test_the_segment_guard_catches_an_invented_container(tmp_path, source,
                                                        expect_hit):
    """Shown to fail — and shown NOT to fail on a URL.

    The third case is the one that matters: the throwaway version of this pass
    flagged `/api/structure/analyze`, `/api/bench/summary` and
    `/api/task-setup/attempts` as hierarchy segments. A route starts with `/`,
    so its first path component is empty, which is exactly what the rule uses to
    tell a path from a URL.
    """
    t = _tool()
    pkg = tmp_path / "molbuilder"
    pkg.mkdir()
    (pkg / "offender.py").write_text(source, encoding="utf-8")
    # Point the vocabulary at the invented name, since the real table lists only
    # the segments that really exist.
    saved = dict(t.CALC_CONTAINERS)
    t.CALC_CONTAINERS["scratchpad"] = None
    try:
        rows = t.segments(pkg=pkg, root=tmp_path)
    finally:
        t.CALC_CONTAINERS.clear()
        t.CALC_CONTAINERS.update(saved)
    hits = [r for r in rows if r["segment"] == "scratchpad"]
    if expect_hit:
        assert hits, f"the guard did not notice an invented container: {rows}"
        assert not hits[0]["declared"], "and it read as declared"
    else:
        assert not hits, (
            f"a Flask ROUTE was reported as a hierarchy segment: "
            f"{[(r['how'], r['line']) for r in hits]}")


def test_the_segment_pass_is_scoped_and_says_why():
    """The vocabulary is a calculation's containers, not every directory name.

    A broad net over every `X / "plain-name"` finds 67 segments across 128 sites
    at 11% precision — units (`Ha/`, `eV/`), a MIME type (`application/json`),
    conda's trees (`bin`, `conda-meta`, `opt`), git's (`refs/`). Measured
    2026-09-08. This asserts the scope stayed small, so nobody widens it back
    into a nag list without reading § 5l.4.
    """
    t = _tool()
    assert len(t.CALC_CONTAINERS) <= 6, (
        f"the segment vocabulary grew to {sorted(t.CALC_CONTAINERS)} — levels "
        f"① and ② are `projects.py`'s (CANONICAL_TOPICS is declared AND "
        f"validated), and a broad vocabulary is what makes this a nag list")
    # levels ①/② must NOT be in here: `transport` and `user` are topic names and
    # also ordinary words, and including them flagged a schema string and a
    # GitHub endpoint.
    from molbuilder.projects import CANONICAL_TOPICS
    assert not (set(CANONICAL_TOPICS) & set(t.CALC_CONTAINERS)), (
        "a canonical TOPIC is in the calculation-container vocabulary; topics "
        "are validated by `projects.validate_topic` and including them here "
        "measured 2 false positives out of 2")
