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
from molbuilder.runfiles import find, find_by_role, role_matches


def _touch(d, *names):
    for n in names:
        p = d / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x", encoding="utf-8")
    return d


# --------------------------------------------------------------------------- #
# 1.  The patterned role -- the one row in the catalogue that is a family.     #
# --------------------------------------------------------------------------- #

def test_the_catalogue_declares_exactly_one_patterned_role():
    """`.runwrap-*.log` is the only row with a ``*``, and that is why
    :func:`runfiles.role_matches` exists at all.

    If a second one appears, this fails — and the reader of the next role has
    to decide whether a pattern is really what the file's name is, rather than
    inheriting a mechanism silently.
    """
    patterned = sorted(a.role for a in runfiles.WRITTEN if "*" in a.role)
    assert patterned == [".runwrap-*.log"], (
        f"the patterned rows are now {patterned}; `role_matches` and "
        f"`find_by_role` treat a starred role as a family — check that is what "
        f"the new row means.")


def test_role_matches_is_equality_for_an_ordinary_role():
    """A plain role must not become a pattern by accident.

    ``.out`` matching ``.pyscf.out`` (or the reverse) is the failure mode that
    would make every role comparison in the module quietly fuzzy.
    """
    assert role_matches(".out", ".out")
    assert not role_matches(".pyscf.log", ".log")
    assert not role_matches(".log", ".pyscf.log")


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
    hits = [p.name for p, _rf in find(d, "bdt", role=".runwrap-*.log",
                                     stage="01_tight")]
    assert hits == ["bdt_01_tight.runwrap-20260101-000000.log",
                    "bdt_01_tight.runwrap-20260908-120000.log"]
    assert hits[-1].endswith("20260908-120000.log"), "newest is last"


def test_find_by_role_answers_for_the_family_without_a_label(tmp_path):
    """Label-less, because the catalogue's role is recognisable on its own."""
    d = _touch(tmp_path, "a.runwrap-20260101-000000.log",
               "b.runwrap-20260102-000000.log", "a.out")
    assert [p.name for p in find_by_role(d, ".runwrap-*.log")] == [
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
                                  "bdt_01_relax*", "bdt", "01_relax")
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
    state, detail = _stage_state(tmp_path, None, "bdt_02_fine*",
                                 "bdt", "02_fine")
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
    state, _d = _stage_state(tmp_path, {"mode": "direct"}, "*",
                             "bdt", "01_tight")
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
