"""The checkpoint classification lives in molbuilder's config (S1c, § 4).

**Contract:** [`execution/checkpointing.md`](?doc=execution/checkpointing.md)
§ 4 — the config is molbuilder-wide, holds the size limit and three engine
entries, and a caller may name its engine to get the matching one. S1b — the
store is chosen by measuring; S1c — the classification has one home.

Every assertion here is derived from that section, not from
`runtime_config.py`: what is asserted is the *behaviour the contract promises a
caller*, so a rename inside the module that keeps the promise does not fail
these, and a change that breaks the promise does.

Real files, real config scopes (`tmp_path`), no mocks — § 13.3 bans mocking the
layer under test, and a config reader whose file-reading is faked is a config
reader nothing has checked.
"""
from __future__ import annotations

import json

import pytest

from molbuilder.runtime_config import (
    PROJECT_CONFIG_FILENAME,
    RuntimeConfigError,
    get_checkpoint,
)


def _write_config(project_dir, section) -> None:
    """Put a ``checkpoint`` section in a project-scope molbuilder.json."""
    (project_dir / PROJECT_CONFIG_FILENAME).write_text(
        json.dumps({"checkpoint": section}), encoding="utf-8")


# ------------------------------------------------------------------ #
#  § 4 — the size limit is a stated number, not an invention          #
# ------------------------------------------------------------------ #


def test_size_limit_defaults_to_ten_megabytes(tmp_path):
    """§ 4 names the number so it is not left for the code to pick."""
    assert get_checkpoint(project_dir=tmp_path)["size_limit_bytes"] == 10 * 1024 * 1024


def test_size_limit_is_changeable(tmp_path):
    """"...and you can change it" -- § 4."""
    _write_config(tmp_path, {"size_limit_bytes": 4096})
    assert get_checkpoint(project_dir=tmp_path)["size_limit_bytes"] == 4096


# ------------------------------------------------------------------ #
#  § 4 — three entries, and an engine is only ever a hint             #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("engine", ["generic", "siesta", "pyscf"])
def test_the_three_named_entries_exist(tmp_path, engine):
    """§ 4: `generic` plus `siesta` and `pyscf`."""
    got = get_checkpoint(engine, project_dir=tmp_path)
    assert isinstance(got["always_large"], list)


def test_generic_names_nothing_so_everything_is_measured(tmp_path):
    """`generic` is 'always correct and merely measures more' (§ 4).

    It must therefore name NO always-large family: every file it sees goes to
    the size gate, which is the only thing that can be correct without knowing
    the engine.
    """
    assert get_checkpoint("generic", project_dir=tmp_path)["always_large"] == []
    assert get_checkpoint(project_dir=tmp_path)["always_large"] == []


def test_an_engine_hint_selects_that_engines_families(tmp_path):
    """"A caller may name its engine to get the matching entry" (§ 4)."""
    siesta = get_checkpoint("siesta", project_dir=tmp_path)["always_large"]
    pyscf = get_checkpoint("pyscf", project_dir=tmp_path)["always_large"]
    assert siesta and pyscf
    assert set(siesta) != set(pyscf), (
        "the two engine entries must differ, or naming an engine buys nothing")


def test_an_unknown_engine_falls_back_to_generic_not_to_an_error(tmp_path):
    """An engine nobody configured must measure MORE, never store less.

    § 4: with no name `generic` is used, "which is always correct and merely
    measures more".  An unknown name is the same situation, and the dangerous
    reading is the other one -- refusing, or silently naming no size limit,
    would make an unconfigured engine save wrongly rather than slowly.
    """
    assert get_checkpoint("no-such-engine", project_dir=tmp_path)["always_large"] == []
    assert get_checkpoint("no-such-engine", project_dir=tmp_path)["size_limit_bytes"] > 0


def test_a_configured_engine_overrides_the_built_in_one(tmp_path):
    _write_config(tmp_path, {"engines": {"siesta": ["*.MYBIN"]}})
    assert get_checkpoint("siesta", project_dir=tmp_path)["always_large"] == ["*.MYBIN"]


def test_a_new_engine_can_be_added_without_touching_code(tmp_path):
    """The classification is config, so a fourth engine is a config edit."""
    _write_config(tmp_path, {"engines": {"vasp": ["*.CHGCAR", "*.WAVECAR"]}})
    assert get_checkpoint("vasp", project_dir=tmp_path)["always_large"] == [
        "*.CHGCAR", "*.WAVECAR"]


# ------------------------------------------------------------------ #
#  A wrong classification is silent, so nothing is coerced or guessed #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("section, why", [
    ({"size_limit_bytes": 0},            "zero would archive everything"),
    ({"size_limit_bytes": -1},           "negative is not a size"),
    ({"size_limit_bytes": "10MB"},       "a string is not a byte count"),
    ({"size_limit_bytes": True},         "bool is an int subclass and is not a size"),
    ({"size_limit_bytes": 1.5},          "a float is not a byte count"),
    ({"engines": []},                    "engines maps a name to patterns"),
    ({"engines": {"siesta": "*.DM"}},    "a bare string is not a list of globs"),
    ({"engines": {"siesta": [""]}},      "an empty glob matches nothing and hides intent"),
    ({"engines": {"siesta": [7]}},       "a glob is a string"),
])
def test_a_malformed_section_is_refused_and_says_why(tmp_path, section, why):
    """Refused, not repaired.

    Every one of these would otherwise change where files are stored while
    looking healthy, which is S1's data-losing branch reached through a typo.
    """
    _write_config(tmp_path, section)
    with pytest.raises(RuntimeConfigError) as exc:
        get_checkpoint(project_dir=tmp_path)
    assert "checkpoint" in str(exc.value), why


def test_the_calculation_folder_holds_no_classification_file(tmp_path):
    """S1c: one home, and it is not the folder being saved.

    A per-folder classification is a file a person can edit between a save and
    a restore, which is the hazard I2c is about.  Reading the config must not
    consult, create, or require anything inside a calculation directory.
    """
    before = set(p.name for p in tmp_path.iterdir())
    get_checkpoint("siesta", project_dir=tmp_path)
    assert set(p.name for p in tmp_path.iterdir()) == before
    assert not (tmp_path / ".mbcheckpoint.json").exists()
