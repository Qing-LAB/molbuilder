"""The checkpoint classification lives in molbuilder's config (S1c, § 4).

**Contract:** [`execution/checkpointing.md`](?doc=execution/checkpointing.md)
§ 4 — the config is molbuilder-wide, holds the size limit and three engine
entries, and a caller may name its engine to get the matching one. S1b — the
store is chosen by measuring; S1c — the classification has **one home**.

Every assertion here is derived from that section, not from
`runtime_config.py`: what is asserted is the *behaviour the contract promises a
caller*, so a rename inside the module that keeps the promise does not fail
these, and a change that breaks the promise does.

**S1c is asserted as a closed door, not as an absent file.** The old test here
checked that reading the config created nothing in the calculation folder and
that a file under the *previous* name was absent — both true while a file under
the *current* name sat in that folder deciding where every byte was stored. A
rule about where something may live is only tested by putting it in the wrong
place and requiring a refusal.

Real files, real config scopes, no mocks — § 13.3 bans mocking the layer under
test, and a config reader whose file-reading is faked is a config reader nothing
has checked.
"""
from __future__ import annotations

import json

import pytest

from molbuilder.runtime_config import (
    CONFIG_FILENAME,
    PROJECT_CONFIG_FILENAME,
    RuntimeConfigError,
    get_checkpoint,
    read_effective_config,
)


@pytest.fixture(autouse=True)
def _own_scope(checkpoint_config):
    """Every test in this file gets its own server-wide config directory.

    Without it these read whatever `molbuilder.json` happens to sit in the
    directory the suite was launched from, which is how a config test comes to
    pass for a reason nobody chose.
    """
    return checkpoint_config


def _write_config(section) -> None:
    """Put a ``checkpoint`` section where the contract says it lives."""
    with open(CONFIG_FILENAME, "w", encoding="utf-8") as fh:
        json.dump({"checkpoint": section}, fh)


# ------------------------------------------------------------------ #
#  § 4 — the size limit is a stated number, not an invention          #
# ------------------------------------------------------------------ #


def test_size_limit_defaults_to_ten_megabytes():
    """§ 4 names the number so it is not left for the code to pick."""
    assert get_checkpoint()["size_limit_bytes"] == 10 * 1024 * 1024


def test_size_limit_is_changeable():
    """"...and you can change it" -- § 4."""
    _write_config({"size_limit_bytes": 4096})
    assert get_checkpoint()["size_limit_bytes"] == 4096


# ------------------------------------------------------------------ #
#  § 4 — three entries, and an engine is only ever a hint             #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("engine", ["generic", "siesta", "pyscf"])
def test_the_three_named_entries_exist(engine):
    """§ 4: `generic` plus `siesta` and `pyscf`."""
    assert isinstance(get_checkpoint(engine)["always_large"], list)


def test_generic_names_nothing_so_everything_is_measured():
    """`generic` is 'always correct and merely measures more' (§ 4).

    It must therefore name NO always-large family: every file it sees goes to
    the size gate, which is the only thing that can be correct without knowing
    the engine.
    """
    assert get_checkpoint("generic")["always_large"] == []
    assert get_checkpoint()["always_large"] == []


def test_an_engine_hint_selects_that_engines_families():
    """"A caller may name its engine to get the matching entry" (§ 4)."""
    siesta = get_checkpoint("siesta")["always_large"]
    pyscf = get_checkpoint("pyscf")["always_large"]
    assert siesta and pyscf
    assert set(siesta) != set(pyscf), (
        "the two engine entries must differ, or naming an engine buys nothing")


def test_an_unknown_engine_falls_back_to_generic_not_to_an_error():
    """An engine nobody configured must measure MORE, never store less.

    § 4: with no name `generic` is used, "which is always correct and merely
    measures more".  An unknown name is the same situation, and the dangerous
    reading is the other one -- refusing, or silently naming no size limit,
    would make an unconfigured engine save wrongly rather than slowly.
    """
    assert get_checkpoint("no-such-engine")["always_large"] == []
    assert get_checkpoint("no-such-engine")["size_limit_bytes"] > 0


def test_a_configured_engine_overrides_the_built_in_one():
    _write_config({"engines": {"siesta": ["*.MYBIN"]}})
    assert get_checkpoint("siesta")["always_large"] == ["*.MYBIN"]


def test_a_new_engine_can_be_added_without_touching_code():
    """The classification is config, so a fourth engine is a config edit."""
    _write_config({"engines": {"vasp": ["*.CHGCAR", "*.WAVECAR"]}})
    assert get_checkpoint("vasp")["always_large"] == ["*.CHGCAR", "*.WAVECAR"]


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
    ({"engines": {"siesta": ["runs/*.bin"]}},
     "a slash makes git and the size check disagree, and the file lands nowhere"),
    ({"engines": {"siesta": ["/abs/*.bin"]}}, "same, from the other end"),
])
def test_a_malformed_section_is_refused_and_says_why(section, why):
    """Refused, not repaired.

    Every one of these would otherwise change where files are stored while
    looking healthy, which is S1's data-losing branch reached through a typo.
    """
    _write_config(section)
    with pytest.raises(RuntimeConfigError) as exc:
        get_checkpoint()
    assert "checkpoint" in str(exc.value), why


# ------------------------------------------------------------------ #
#  S1c — one home, and the other doors are shut                       #
# ------------------------------------------------------------------ #


def test_a_classification_beside_a_calculation_is_refused(tmp_path):
    """S1c, asserted where it can actually fail.

    A project-scope file is read for other subsystems, and it deep-merges with
    the project winning.  So a `checkpoint` section dropped into a calculation
    folder was not merely tolerated -- it *won*, and set the size limit that
    decided where every file in that folder was stored.  That is the per-folder
    classification S1c forbids, wearing the project scope's filename.

    Refused rather than ignored: a section that is read and silently dropped
    looks effective, and the folder is then saved under rules nobody applied.
    """
    calc = tmp_path / "BDT_Au_relax"
    calc.mkdir()
    (calc / PROJECT_CONFIG_FILENAME).write_text(json.dumps(
        {"checkpoint": {"size_limit_bytes": 1}}), encoding="utf-8")

    with pytest.raises(RuntimeConfigError) as exc:
        read_effective_config(calc)
    message = str(exc.value)
    assert "checkpoint" in message and "S1c" in message, (
        "the refusal must name the section and the rule, or the person "
        "holding the file cannot tell why it is not allowed")


def test_the_classification_is_read_from_one_scope_only(tmp_path):
    """Two folders under one molbuilder must classify a file the same way.

    S1c's own stated test: "two folders under different engines produce the
    same store decision for the same file".  Nothing about *where* the folder
    sits may enter the answer, which is why the accessor takes no directory.
    """
    _write_config({"size_limit_bytes": 4096, "engines": {"generic": []}})
    here = tmp_path / "one"
    there = tmp_path / "two" / "deeper"
    there.mkdir(parents=True)
    here.mkdir()

    from molbuilder.checkpoint import Repo, is_big
    limit = get_checkpoint()["size_limit_bytes"]
    for folder in (here, there):
        sample = folder / "same.bin"
        sample.write_bytes(b"x" * 5000)
        assert is_big(sample, limit) is True
        assert Repo(str(folder)).classification()["size_limit_bytes"] == 4096


def test_the_accessor_takes_no_directory_at_all():
    """The absence of the parameter IS the rule (S1c).

    A `project_dir` argument is how the per-folder classification came back
    last time: every caller had a directory in hand and passing it looked
    harmless.  There is nothing to pass now.
    """
    import inspect
    assert "project_dir" not in inspect.signature(get_checkpoint).parameters
    from molbuilder.checkpoint import classification
    assert "project_dir" not in inspect.signature(classification).parameters
