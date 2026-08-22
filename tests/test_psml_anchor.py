"""**A10 — an anchor is declared, never discovered** (`architecture.md` § 7).

What this file is about
=======================

A user types a pseudopotential library path once, in a template, and that
template is then carried to another machine and run somewhere else.  The
question this rule settles is the boring-sounding one that broke a real run:
*relative to what?*

Until 2026-08-21 the answer was "whichever of three candidates happens to
exist" -- the calculation folder, then the ``projects/`` tree above it, then
``<cwd>/projects``.  Trying costs nothing, so nothing ever had to decide, and
two things followed.  The same string named a different folder on different
machines, because *which candidate exists* is a property of the machine.  And
when none existed, the refusal named the **last one tried** -- which is how
``prep bench`` on Sol came to refuse with
``…/optimization/Relax/projects/pseudopotential``, a folder assembled out of
where the user was standing that no user had ever chosen.

The rule now is `job-contracts.md` § 2.5a: **the spelling names the anchor.**
Absolute is itself; a leading dot means *from this calculation*; a bare name
means *the tree this calculation lives in*.  One anchor per spelling, no
fallback order, and a miss reported against the anchor that was asked for.

Every test below builds its own tree under ``tmp_path`` and passes the
calculation explicitly, so nothing here can pass or fail because of where
pytest was started -- which is the very coupling the rule removes.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from molbuilder.projects import find_projects_root
from molbuilder.pseudos import describe_psml_anchor, resolve_psml_lib


@pytest.fixture
def calc(tmp_path: Path) -> Path:
    """A calculation at its real depth: ``projects/<proj>/<topic>/<calc>``."""
    d = tmp_path / "projects" / "Au-BDT-Au" / "optimization" / "Relax"
    d.mkdir(parents=True)
    return d


class TestTheSpellingNamesTheAnchor:
    """The matrix.  One row per spelling the rule defines."""

    def test_absolute_is_itself(self, calc, tmp_path):
        assert resolve_psml_lib(str(tmp_path / "psml"), dest_dir=calc) == \
            tmp_path / "psml"

    def test_tilde_expands_and_stops_there(self, calc):
        got = resolve_psml_lib("~/psml", dest_dir=calc)
        assert got == Path.home() / "psml"

    def test_a_leading_dot_means_this_calculation(self, calc):
        assert resolve_psml_lib("./psml", dest_dir=calc) == calc / "psml"

    def test_dot_dot_walks_from_the_calculation(self, calc):
        assert resolve_psml_lib("../psml", dest_dir=calc) == calc / ".." / "psml"

    def test_a_bare_name_means_the_tree_the_calculation_lives_in(
            self, calc, tmp_path):
        """The Sol case.  Note what is NOT in the answer: the working
        directory.  This is the whole point of the rule -- the tree is found
        from the calculation's own position."""
        assert resolve_psml_lib("pseudopotential", dest_dir=calc) == \
            tmp_path / "projects" / "pseudopotential"

    def test_the_answer_does_not_depend_on_where_you_stand(
            self, calc, tmp_path, monkeypatch):
        """Same question, three working directories, one answer.

        Under the old cascade this was the failure: standing inside the
        calculation made ``<calc>/projects/pseudopotential`` the answer.
        """
        expected = tmp_path / "projects" / "pseudopotential"
        for cwd in (tmp_path, calc, Path("/")):
            monkeypatch.chdir(cwd)
            assert resolve_psml_lib("pseudopotential", dest_dir=calc) == expected


class TestTheCallerWithNoCalculation:
    """Server-side validation runs before anything exists on disk."""

    def test_it_anchors_at_the_servers_own_root(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert resolve_psml_lib("pseudopotential") == \
            tmp_path / "projects" / "pseudopotential"

    def test_base_overrides_that_root(self, tmp_path):
        assert resolve_psml_lib("pseudopotential", base=tmp_path / "elsewhere") \
            == tmp_path / "elsewhere" / "pseudopotential"


class TestACalculationOutsideAnyTree:
    """A bundle copied somewhere flat has no tree to name."""

    def test_the_bare_name_falls_to_the_folder_the_user_chose(self, tmp_path):
        loose = tmp_path / "scratch" / "run1"
        loose.mkdir(parents=True)
        assert find_projects_root(loose) is None
        assert resolve_psml_lib("pseudopotential", dest_dir=loose) == \
            loose / "pseudopotential"


class TestTheRefusalNamesWhatWasAskedFor:
    """A10's second half: the message a user acts on.

    A refusal is where most users learn the rule, so these assert the
    *content* -- which anchor, and where it landed -- not that a string is
    non-empty.
    """

    def test_it_names_the_tree_not_the_working_directory(
            self, calc, tmp_path, monkeypatch):
        monkeypatch.chdir(calc)
        msg = describe_psml_anchor("pseudopotential", dest_dir=calc)
        assert str(tmp_path / "projects" / "pseudopotential") in msg
        # The folder the old cascade would have named must NOT appear.
        assert str(calc / "projects") not in msg

    def test_a_dotted_spelling_says_from_this_calculation(self, calc):
        msg = describe_psml_anchor("./psml", dest_dir=calc)
        assert "from this calculation" in msg
        assert str(calc / "psml") in msg

    def test_the_doubled_prefix_is_explained_not_silently_fixed(self, calc):
        """``projects/pseudopotential`` is the one spelling that cannot work.

        It is NOT stripped for the user: a path is what they typed, and a
        resolver that quietly edits it is deciding on their behalf.  It is
        explained instead.
        """
        msg = describe_psml_anchor("projects/pseudopotential", dest_dir=calc)
        assert "doubled" in msg and "Drop the leading projects/" in msg
        assert resolve_psml_lib("projects/pseudopotential", dest_dir=calc).name \
            == "pseudopotential"

    def test_no_calculation_yet_is_said_plainly(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert "no calculation folder yet" in describe_psml_anchor("psml")


class TestPrepRefusesWithTheRealPlace:
    """End to end through `prep`'s own refusal -- the Sol reproduction."""

    def _struct(self):
        import numpy as np
        from molbuilder.structure import Structure
        return Structure(elements=["H", "C", "S", "Au"],
                         positions=np.array([[0, 0, 0], [1, 0, 0],
                                             [2, 0, 0], [3, 0, 0]], float))

    def test_the_missing_library_is_named_by_its_tree(
            self, calc, tmp_path, monkeypatch):
        from molbuilder.config.siesta import SiestaConfig
        from molbuilder.jobset.prep import PrepError, _siesta_provide_pseudos
        monkeypatch.chdir(calc)          # stand where the failure happened
        with pytest.raises(PrepError) as e:
            _siesta_provide_pseudos(
                self._struct(),
                SiestaConfig(system_label="x", psml_lib="pseudopotential"),
                calc)
        msg = str(e.value)
        assert str(tmp_path / "projects" / "pseudopotential") in msg
        assert "H, C, S, Au" in msg
        assert str(calc / "projects") not in msg

    def test_it_finds_the_library_that_is_really_there(
            self, calc, tmp_path, monkeypatch):
        """The other half: the rule must also SUCCEED from anywhere."""
        from molbuilder.config.siesta import SiestaConfig
        from molbuilder.siesta.input import copy_pseudopotentials
        lib = tmp_path / "projects" / "pseudopotential"
        lib.mkdir(parents=True)
        for el in ("H", "C", "S", "Au"):
            (lib / f"{el}.psml").write_text("<psml/>")
        monkeypatch.chdir(Path("/"))
        resolved = resolve_psml_lib("pseudopotential", dest_dir=calc)
        assert resolved.is_dir()
        assert not copy_pseudopotentials(["H", "C", "S", "Au"], resolved, calc)
        assert (calc / "Au.psml").is_file()
