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


def test_projects_root_default_is_cwd_relative(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert projects_root() == tmp_path / PROJECTS_ROOT_NAME


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
