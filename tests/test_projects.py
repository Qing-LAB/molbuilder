"""Project layout, validation, discovery (``molbuilder.projects``).

The module under test is filesystem-pure: every test runs against
``tmp_path`` (no real ``./projects/`` directory is touched).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from molbuilder import projects
from molbuilder.projects import (CANONICAL_TOPICS, InvalidName,
                                   PROJECTS_ROOT_NAME, ensure_structure_dir,
                                   find_geom_candidates, list_projects,
                                   list_structures, list_topics, project_dir,
                                   projects_root, structure_dir, topic_dir,
                                   validate_name, validate_topic)


# --------------------------------------------------------------------- #
#  Name / topic validation                                              #
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("good", ["job", "job-1", "job_1", "Au-S-12mer",
                                    "x", "0", "_under", "MixedCASE"])
def test_validate_name_accepts_good(good):
    assert validate_name(good) == good


@pytest.mark.parametrize("bad", ["", "has space", "has.dot", "has/slash",
                                   "has?punct", "has\nnewline"])
def test_validate_name_rejects_bad(bad):
    with pytest.raises(InvalidName):
        validate_name(bad)


def test_validate_name_kind_appears_in_message():
    """Error includes the field's role so the user sees which input failed."""
    with pytest.raises(InvalidName, match="structure"):
        validate_name("bad name", kind="structure")


def test_validate_name_rejects_non_string():
    with pytest.raises(InvalidName):
        validate_name(None)                # type: ignore[arg-type]
    with pytest.raises(InvalidName):
        validate_name(42)                  # type: ignore[arg-type]


def test_validate_topic_accepts_canonical():
    for t in CANONICAL_TOPICS:
        assert validate_topic(t) == t


def test_validate_topic_rejects_non_canonical():
    with pytest.raises(InvalidName, match="canonical"):
        validate_topic("uvvis")


def test_validate_topic_still_validates_chars():
    """Even canonical-name-shape but invalid chars rejected via name path."""
    with pytest.raises(InvalidName):
        validate_topic("with space")


# --------------------------------------------------------------------- #
#  Path resolution (no I/O)                                             #
# --------------------------------------------------------------------- #


def test_projects_root_comes_from_the_one_door(tmp_path, monkeypatch):
    """`projects_root` is the ONE door, and it is not the working
    directory (user, 2026-08-22).

    It used to return ``Path.cwd()/"projects"``, which made the tree a
    property of where you happened to stand.  It now resolves in a
    declared order -- ``base=``, ``$MOLBUILDER_PROJECTS``,
    ``molbuilder.json``'s ``paths.projects``, then ``repo_root()/projects``
    -- so one setting moves the tree for every surface at once.
    """
    from molbuilder.projects import PROJECTS_ROOT_ENV, projects_root
    import molbuilder

    # cwd does NOT decide it any more (this is the behaviour change).
    elsewhere = tmp_path / "somewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.delenv(PROJECTS_ROOT_ENV, raising=False)
    assert projects_root() != elsewhere / PROJECTS_ROOT_NAME
    assert projects_root() == molbuilder.repo_root() / PROJECTS_ROOT_NAME

    # the env override wins over the default
    monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tmp_path / "elsewhere-tree"))
    assert projects_root() == tmp_path / "elsewhere-tree"

    # an explicit base wins over everything
    assert projects_root(base=tmp_path) == tmp_path / PROJECTS_ROOT_NAME

def test_projects_root_with_explicit_base(tmp_path):
    assert projects_root(base=tmp_path) == tmp_path / PROJECTS_ROOT_NAME


def test_path_resolution_composes(tmp_path):
    p = project_dir("p1", base=tmp_path)
    t = topic_dir("p1", "spectrum", base=tmp_path)
    s = structure_dir("p1", "spectrum", "Au-S-12mer", base=tmp_path)
    assert p == tmp_path / "projects" / "p1"
    assert t == tmp_path / "projects" / "p1" / "spectrum"
    assert s == tmp_path / "projects" / "p1" / "spectrum" / "Au-S-12mer"


def test_path_resolution_validates_each_segment(tmp_path):
    """Bad project / topic / structure all raise individually."""
    with pytest.raises(InvalidName):
        project_dir("bad name", base=tmp_path)
    with pytest.raises(InvalidName):
        topic_dir("p", "uvvis", base=tmp_path)
    with pytest.raises(InvalidName):
        structure_dir("p", "spectrum", "bad/name", base=tmp_path)


def test_ensure_structure_dir_creates_full_chain(tmp_path):
    d = ensure_structure_dir("p", "spectrum", "s", base=tmp_path)
    assert d.is_dir()
    # And the parents:
    assert (tmp_path / "projects" / "p").is_dir()
    assert (tmp_path / "projects" / "p" / "spectrum").is_dir()


def test_ensure_structure_dir_idempotent(tmp_path):
    d1 = ensure_structure_dir("p", "spectrum", "s", base=tmp_path)
    d2 = ensure_structure_dir("p", "spectrum", "s", base=tmp_path)
    assert d1 == d2 and d1.is_dir()


# --------------------------------------------------------------------- #
#  Discovery                                                            #
# --------------------------------------------------------------------- #


def test_list_projects_empty_when_no_projects_dir(tmp_path):
    assert list_projects(base=tmp_path) == []


def test_list_projects_returns_sorted(tmp_path):
    for name in ("zeta", "alpha", "beta"):
        ensure_structure_dir(name, "spectrum", "s", base=tmp_path)
    assert list_projects(base=tmp_path) == ["alpha", "beta", "zeta"]


def test_list_projects_skips_invalid_names_with_warning(tmp_path, caplog):
    """Stray dirs that don't match the name regex are skipped + warned."""
    import logging
    (tmp_path / "projects").mkdir()
    (tmp_path / "projects" / "valid_name").mkdir()
    (tmp_path / "projects" / "stray dir with space").mkdir()
    (tmp_path / "projects" / "another.bad").mkdir()
    with caplog.at_level(logging.WARNING, logger="molbuilder.projects"):
        result = list_projects(base=tmp_path)
    assert result == ["valid_name"]
    # The user should be told *why* the stray dirs vanished from the list.
    skipped = " ".join(r.message for r in caplog.records)
    assert "stray dir with space" in skipped
    assert "another.bad"          in skipped


def test_list_topics_returns_canonical_order(tmp_path):
    """Topics come back in CANONICAL_TOPICS order, regardless of mkdir order."""
    ensure_structure_dir("p", "transport", "s", base=tmp_path)
    ensure_structure_dir("p", "optimization", "s", base=tmp_path)
    ensure_structure_dir("p", "spectrum", "s", base=tmp_path)
    # Canonical order is optimization, frequency, spectrum, transport, ...
    assert list_topics("p", base=tmp_path) == [
        "optimization", "spectrum", "transport",
    ]


def test_list_topics_ignores_non_canonical_dirs(tmp_path):
    pd = project_dir("p", base=tmp_path)
    pd.mkdir(parents=True)
    (pd / "spectrum").mkdir()
    (pd / "uvvis").mkdir()                 # not canonical
    (pd / "README.md").touch()             # not a dir
    assert list_topics("p", base=tmp_path) == ["spectrum"]


def test_list_structures(tmp_path):
    for s in ("zz", "aa", "mm"):
        ensure_structure_dir("p", "spectrum", s, base=tmp_path)
    assert list_structures("p", "spectrum", base=tmp_path) == ["aa", "mm", "zz"]


def test_list_structures_empty_when_topic_missing(tmp_path):
    """Topic dir absent -> empty list, not error.  Caller can distinguish
    via list_topics() if needed."""
    ensure_structure_dir("p", "optimization", "s", base=tmp_path)
    assert list_structures("p", "spectrum", base=tmp_path) == []


# --------------------------------------------------------------------- #
#  find_geom_candidates -- name conventions + mtime                     #
# --------------------------------------------------------------------- #


def _touch_with_mtime(path: Path, t: float) -> None:
    """Create the file and set its mtime to `t` (seconds since epoch)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    os.utime(path, (t, t))


def test_find_geom_candidates_empty_when_no_projects(tmp_path):
    assert find_geom_candidates(base=tmp_path) == []


def test_find_geom_candidates_picks_known_patterns(tmp_path):
    sd = ensure_structure_dir("p", "optimization", "Au-S", base=tmp_path)
    _touch_with_mtime(sd / "Au-S_optimized.xyz", 100.0)
    _touch_with_mtime(sd / "Au-S.STRUCT_OUT",   100.0)
    _touch_with_mtime(sd / "random.txt",        100.0)
    found = find_geom_candidates(base=tmp_path)
    names = {p.name for p in found}
    assert "Au-S_optimized.xyz" in names
    assert "Au-S.STRUCT_OUT"    in names
    assert "random.txt"         not in names


def test_find_geom_candidates_sorted_newest_first(tmp_path):
    sd1 = ensure_structure_dir("p", "optimization", "old", base=tmp_path)
    sd2 = ensure_structure_dir("p", "optimization", "new", base=tmp_path)
    _touch_with_mtime(sd1 / "old_optimized.xyz", 100.0)
    _touch_with_mtime(sd2 / "new_optimized.xyz", 200.0)
    found = find_geom_candidates(base=tmp_path)
    assert [p.name for p in found] == ["new_optimized.xyz", "old_optimized.xyz"]


def test_find_geom_candidates_ignores_generic_xyz_and_pdb(tmp_path):
    """Patterns are deliberately specific -- generic ``*.xyz`` /
    ``*.pdb`` would catch user inputs, intermediate frames, and other
    noise the picker shouldn't surface.  Only files matching one of
    the documented output conventions appear in the list."""
    sd = ensure_structure_dir("p", "optimization", "x", base=tmp_path)
    _touch_with_mtime(sd / "x_optimized.xyz", 100.0)
    _touch_with_mtime(sd / "starting_input.xyz", 100.0)
    _touch_with_mtime(sd / "reference.pdb", 100.0)
    names = {p.name for p in find_geom_candidates(base=tmp_path)}
    assert names == {"x_optimized.xyz"}


def test_find_geom_candidates_scoped_to_one_project(tmp_path):
    sda = ensure_structure_dir("alpha", "optimization", "x", base=tmp_path)
    sdb = ensure_structure_dir("beta",  "optimization", "y", base=tmp_path)
    _touch_with_mtime(sda / "x_optimized.xyz", 100.0)
    _touch_with_mtime(sdb / "y_optimized.xyz", 100.0)
    only_alpha = find_geom_candidates(base=tmp_path, project="alpha")
    names = {p.name for p in only_alpha}
    assert names == {"x_optimized.xyz"}


def test_find_geom_candidates_alphabetical_when_not_newest_first(tmp_path):
    sd = ensure_structure_dir("p", "optimization", "x", base=tmp_path)
    _touch_with_mtime(sd / "z_optimized.xyz", 100.0)
    _touch_with_mtime(sd / "a_optimized.xyz", 200.0)
    found = find_geom_candidates(base=tmp_path, newest_first=False)
    assert [p.name for p in found] == ["a_optimized.xyz", "z_optimized.xyz"]


class TestTheProjectsRootIsOneConfigurableDoor:
    """`projects_root` is the ONE door every surface resolves the tree
    through, and it is settable (user, 2026-08-22).

    The reason it must be settable is deployment, not taste: the default
    lives inside the checkout, and on a cluster that is often a quota'd
    home directory, a read-only shared install, or simply not where the
    data belongs.  Because every surface -- the sidebar backend, the
    `jobset` verbs' `--bundle`, the workspace store, the pseudopotential
    anchor -- goes through this function, one setting moves them all.
    """

    def _pj(self):
        """No reload needed: `runtime_config` is stateless by contract --
        it reads the file on every call.  (Reloading it also mints a NEW
        RuntimeConfigError class, which then fails to match `pytest.raises`
        against the one imported earlier -- a trap worth not re-setting.)"""
        import molbuilder.projects as pj
        return pj

    def test_the_config_key_moves_the_tree(self, tmp_path, monkeypatch):
        import json
        # tmp_path IS the config root: the machine scope has one location
        # and it is not the working directory (configuration.md § 2.1a).
        monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(tmp_path))
        monkeypatch.delenv("MOLBUILDER_PROJECTS", raising=False)
        (tmp_path / "molbuilder.json").write_text(
            json.dumps({"paths": {"projects": str(tmp_path / "elsewhere")}}))
        pj = self._pj()
        assert pj.projects_root() == tmp_path / "elsewhere"

    def test_a_relative_setting_is_read_from_the_molbuilder_root(
            self, tmp_path, monkeypatch):
        """So the setting means the same folder whatever directory you run
        from -- the same rule `--bundle` and `psml_lib` follow."""
        import json
        import molbuilder
        # tmp_path IS the config root: the machine scope has one location
        # and it is not the working directory (configuration.md § 2.1a).
        monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(tmp_path))
        monkeypatch.delenv("MOLBUILDER_PROJECTS", raising=False)
        (tmp_path / "molbuilder.json").write_text(
            json.dumps({"paths": {"projects": "shared-tree"}}))
        pj = self._pj()
        assert pj.projects_root() == molbuilder.repo_root() / "shared-tree"

    def test_the_env_override_beats_the_config(self, tmp_path, monkeypatch):
        import json
        # tmp_path IS the config root: the machine scope has one location
        # and it is not the working directory (configuration.md § 2.1a).
        monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(tmp_path))
        (tmp_path / "molbuilder.json").write_text(
            json.dumps({"paths": {"projects": str(tmp_path / "from-config")}}))
        monkeypatch.setenv("MOLBUILDER_PROJECTS", str(tmp_path / "from-env"))
        pj = self._pj()
        assert pj.projects_root() == tmp_path / "from-env"

    def test_a_malformed_setting_is_refused_not_ignored(
            self, tmp_path, monkeypatch):
        """A config the user wrote and molbuilder silently ignored is the
        worst of both.  A blanket `except Exception` did exactly that for
        one revision."""
        import json
        import pytest as _pytest
        from molbuilder.runtime_config import RuntimeConfigError
        # tmp_path IS the config root: the machine scope has one location
        # and it is not the working directory (configuration.md § 2.1a).
        monkeypatch.setenv("MOLBUILDER_CONFIG_DIR", str(tmp_path))
        monkeypatch.delenv("MOLBUILDER_PROJECTS", raising=False)
        (tmp_path / "molbuilder.json").write_text(
            json.dumps({"paths": {"porjects": "/typo"}}))
        pj = self._pj()
        with _pytest.raises(RuntimeConfigError) as e:
            pj.projects_root()
        assert "porjects" in str(e.value) and "projects" in str(e.value)


class TestABundleIsNamedFromTheRootAndStaysInside:
    """`--bundle` is uniform and fenced (`job-contracts.md` § 2.5b).

    Uniform: a supplied path is read from the projects root, full stop --
    no dotted escape hatch, because a calculation outside the tree is not a
    calculation molbuilder manages.  Fenced: `..` and absolute paths are
    resolved and then checked, so the rule cannot be spelled around.
    """

    def _tree(self, tmp_path, monkeypatch):
        from molbuilder.projects import PROJECTS_ROOT_ENV
        root = tmp_path / "projects"
        (root / "P" / "optimization" / "Relax").mkdir(parents=True)
        monkeypatch.setenv(PROJECTS_ROOT_ENV, str(root))
        monkeypatch.chdir(tmp_path)          # deliberately OUTSIDE the tree
        return root

    def _bundle(self, args):
        from click.testing import CliRunner
        from molbuilder.jobset._cli import jobset_group
        return CliRunner().invoke(jobset_group, ["status"] + args)

    def test_a_path_is_read_from_the_projects_root(self, tmp_path, monkeypatch):
        root = self._tree(tmp_path, monkeypatch)
        res = self._bundle(["--bundle", "P/optimization/Relax"])
        assert str(root / "P" / "optimization" / "Relax") in res.output

    def test_a_leading_dot_is_not_an_escape_hatch(self, tmp_path, monkeypatch):
        """`./x` means the same as `x` now -- one anchor, no exceptions."""
        root = self._tree(tmp_path, monkeypatch)
        res = self._bundle(["--bundle", "./P/optimization/Relax"])
        assert str(root / "P" / "optimization" / "Relax") in res.output

    def test_dot_dot_cannot_climb_out(self, tmp_path, monkeypatch):
        """Refused on the RAW spelling, before resolution -- the shared
        fence rejects `..` early so there is no "did they think it was
        harmless?" ambiguity (`projects.contain`)."""
        self._tree(tmp_path, monkeypatch)
        res = self._bundle(["--bundle", "../escaped"])
        assert res.exit_code == 2
        assert "may not contain '..'" in res.output
        assert "inside the projects tree" in res.output

    def test_an_absolute_path_outside_is_refused(self, tmp_path, monkeypatch):
        self._tree(tmp_path, monkeypatch)
        res = self._bundle(["--bundle", "/etc"])
        assert res.exit_code == 2
        assert "inside the projects tree" in res.output

    def test_an_absolute_path_inside_is_fine(self, tmp_path, monkeypatch):
        """The fence is about leaving the tree, not about spelling."""
        root = self._tree(tmp_path, monkeypatch)
        res = self._bundle(["--bundle", str(root / "P")])
        assert res.exit_code != 2, res.output

    def test_omitting_it_outside_the_tree_says_so(self, tmp_path, monkeypatch):
        """The default is the working directory -- which must itself be a
        place in the tree, or it names no calculation."""
        self._tree(tmp_path, monkeypatch)
        res = self._bundle([])
        assert res.exit_code == 2
        assert "is not inside the projects tree" in res.output
        assert "<project>/<topic>/<calculation>" in res.output
