"""A PySCF rung names its own artifacts, and starts where its `restart` says.

Contract: ``docs/engines/stages.md`` § 1.1a -- **a PySCF ladder is N decks and
N jobs, exactly as SIESTA's is** -- and its five numbered consequences, which
are what this file checks one by one:

  1. the stage token reaches the deck's name, the engine's log and the
     trajectory log, so two rungs cannot write to one file;
  2. the ``JOB`` literal stays unsuffixed, because that is how the engine finds
     the previous rung's state;
  3. ``restart`` is both engines' now, and it GATES the reads;
  4. the warm-file declaration carries the geometry as well as the checkpoint;
  5. the in-script ladder is gone.

**These are deck-side facts, so they are read off rendered text**, not off the
config that produced it.  Every one of them was invisible to the tests that
existed: those assert that a config value reached the deck, and each of these
is about a name or a branch the config does not carry.
"""
from __future__ import annotations

import ast
import dataclasses
import re

import numpy as np
import pytest

from molbuilder.config.pyscf import PySCFConfig
from molbuilder.pyscf.input import (GEOMETRIC_APPENDS, ROLE_GEOM_TRAJ,
                                    render_script)
from molbuilder.runfiles import tail as _rf_tail
from molbuilder.structure import Structure


def _water() -> Structure:
    return Structure(
        elements=["O", "H", "H"],
        positions=np.array([[0.0, 0.0, 0.0],
                            [0.757, 0.586, 0.0],
                            [-0.757, 0.586, 0.0]]),
        title="water")


def _deck(*, token=None, **overrides) -> str:
    cfg = dataclasses.replace(PySCFConfig(job_name="wat"), **overrides)
    return render_script(_water(), cfg, stage_token=token)


def _vib_deck(*, token=None, **overrides) -> str:
    """The VIBRATION deck's text, rendered the same way.

    It names the same three artifacts under the same rules, and nothing
    checked it -- which is how it kept the retired geomeTRIC prefix for the
    five weeks after the optimization deck's was fixed.
    """
    from molbuilder import script_emit as _sc
    from molbuilder.pyscf.input import spec_for
    cfg = dataclasses.replace(
        PySCFConfig(job_name="wat", optimize=True, write_trajectory=True),
        **overrides)
    spec = spec_for(_water(), cfg, stage_token=token,
                    calculation="vibration")
    return _sc.render_deck(spec, _water(), cfg,
                           verbose=cfg.verbose_comments)


# --------------------------------------------------------------------- #
#  Consequence 1 — every name the SCRIPT chooses carries the rung        #
# --------------------------------------------------------------------- #

#: ``what it names -> (the regex that finds it, the ROLE it is named from)``.
#: One row per name the script picks for itself -- the closed set consequence 1
#: is about.  A new artifact belongs here on the day it is added.
#:
#: THE ROLE, not a spelling.  These rows asserted only that the token appeared
#: SOMEWHERE in the name, which a wrong position passes: geomeTRIC's prefix was
#: `JOB + '_geom_01_coarse'` for five weeks, giving
#: `<JOB>_geom_01_coarse_optim.xyz` where the declared role is
#: `_geom_optim.xyz`, and this test was green throughout.  Each row now says
#: which role the name is built from, and the expected tail comes out of
#: `runfiles.tail` -- the generator the naming suite tests over its whole
#: parameter space.
_SELF_CHOSEN_NAMES = {
    "the engine's own log":
        (r"output\s*=\s*(?:str\()?_mb_outfile\(JOB \+ '([^']+)'\)", ".log"),
    "the molwatch trajectory":
        (r"MolwatchEmitter\(_mb_outfile\(JOB \+ '([^']+)'", ".molwatch.log"),
    "geomeTRIC's trajectory":
        (r"prefix\s*=\s*(?:str\()?_mb_outfile\(JOB \+ '([^']+)'\)",
         ROLE_GEOM_TRAJ),
}


def _expected(role: str, token) -> str:
    """What the name after ``JOB`` must be, from the grammar.

    geomeTRIC is handed a PREFIX and appends its own suffix, so what the deck
    emits for it is the trajectory's tail minus what geomeTRIC will add.
    """
    t = _rf_tail(role, token)
    return t[:-len(GEOMETRIC_APPENDS)] if role == ROLE_GEOM_TRAJ else t


#: The two decks that name their own artifacts.  Both write the same three
#: files under the same rules, and only the optimization deck was checked --
#: which is why the vibration deck still carried the retired geomeTRIC prefix
#: after the optimization deck was fixed (2026-09-07).
_DECKS = {"optimization": _deck, "vibration": _vib_deck}


@pytest.mark.parametrize("deck", sorted(_DECKS))
@pytest.mark.parametrize("what", sorted(_SELF_CHOSEN_NAMES))
def test_every_name_the_script_chooses_is_the_one_the_grammar_builds(
        what, deck):
    """Two rungs are two PROCESSES in one folder.  Anything they both name is
    a file the second overwrites -- silently, and after the first has run.

    Asserted as EQUALITY against `runfiles.tail`, not as "the token appears":
    a token in the wrong place is still a token, and it still collides with
    nothing while making the file unfindable by everything that looks for it.
    """
    rx, role = _SELF_CHOSEN_NAMES[what]
    render = _DECKS[deck]
    for token in ("01_coarse", "02_medium"):
        text = render(token=token)
        m = re.search(rx, text)
        assert m, f"{deck}/{what}: no match for /{rx}/"
        assert m.group(1) == _expected(role, token), (deck, what)


@pytest.mark.parametrize("deck", sorted(_DECKS))
@pytest.mark.parametrize("what", sorted(_SELF_CHOSEN_NAMES))
def test_a_deck_with_no_rung_takes_the_unsuffixed_name(what, deck):
    """A single-run workflow has one rung and no name for it, and the file
    names must not grow a placeholder for the absence."""
    rx, role = _SELF_CHOSEN_NAMES[what]
    text = _DECKS[deck](token=None)
    m = re.search(rx, text)
    assert m, f"{deck}/{what}"
    assert m.group(1) == _expected(role, None), (deck, what, m.group(1))


# --------------------------------------------------------------------- #
#  Consequence 2 — the JOB literal does NOT move                        #
# --------------------------------------------------------------------- #

def test_the_job_literal_is_the_same_in_every_rung():
    """§ 1.1a consequence 2, and it is the opposite rule to consequence 1 --
    which is exactly why both are asserted.  The engine finds the previous
    rung's ``.chk`` and ``_optimized.xyz`` by this name, so a name that
    changed per rung would hide them."""
    def _job(text):
        m = re.search(r'^JOB\s*=\s*"([^"]+)"', text, re.M)
        assert m, "no JOB literal"
        return m.group(1)
    assert (_job(_deck(token="01_coarse"))
            == _job(_deck(token="02_medium"))
            == _job(_deck(token=None))
            == "wat")


# --------------------------------------------------------------------- #
#  Consequence 3 — `restart` gates the reads, and nothing else does     #
# --------------------------------------------------------------------- #

#: The two halves of what a rung hands the next one: the geometry and the
#: converged density (`stages.md` § 1.1a; SIESTA carries the same pair as
#: ``.XV`` and ``.DM``).
_RESUME_READS = {
    "the previous rung's geometry": r"^_opt_path = ",
    "the previous rung's density":  r'^\s*mf\.init_guess = "chkfile"',
}


@pytest.mark.parametrize("what", sorted(_RESUME_READS))
def test_a_continuing_rung_reads_what_the_one_before_it_left(what):
    assert re.search(_RESUME_READS[what], _deck(restart="continue"), re.M), what


@pytest.mark.parametrize("what", sorted(_RESUME_READS))
def test_a_clean_rung_reads_neither(what):
    assert not re.search(_RESUME_READS[what], _deck(restart="clean"), re.M), what


def test_a_clean_rung_still_WRITES_its_checkpoint():
    """**The sentence this engine could not say until `restart` existed.**
    The resume branches were gated on ``chkfile`` and ``save_optimized_xyz``
    -- write flags doubling as read gates -- so *"write a checkpoint but do
    not resume from one"* had no expression (`run-identity.md` § 4 rule 2).
    Turning off the write to get a cold start threw away the checkpoint the
    NEXT rung would have wanted."""
    text = _deck(restart="clean")
    assert re.search(r"^mf\.chkfile = ", text, re.M)
    assert not re.search(r'^\s*mf\.init_guess = "chkfile"', text, re.M)


def test_the_write_flag_no_longer_decides_the_read():
    """The gates are separate in BOTH directions: a rung that continues reads
    the geometry whether or not it means to write one of its own."""
    text = _deck(restart="continue", save_optimized_xyz=False)
    assert re.search(r"^_opt_path = ", text, re.M)


# --------------------------------------------------------------------- #
#  Consequence 5 — one deck is one rung                                 #
# --------------------------------------------------------------------- #

def test_the_deck_runs_exactly_one_optimisation():
    """The in-script ladder is gone: no loop over rungs, and one
    ``optimize()`` call.  A deck that ran several would end once, at the end
    -- and a ladder exists so that somebody looks BETWEEN the rungs."""
    tree = ast.parse(_deck(token="01_coarse"))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "optimize"]
    assert len(calls) == 1, f"{len(calls)} optimize() calls"
    assert not re.search(r"^STAGES\s*=", ast.unparse(tree), re.M)


def test_the_convergence_record_describes_this_rung_and_no_other():
    """The molwatch log's ``_CONVERGENCE_TARGETS`` is one entry, keyed by the
    rung's own token -- the same string the log's filename carries, so a log
    and the targets inside it name the rung identically."""
    m = re.search(r"_CONVERGENCE_TARGETS = (\{.*?\n\})",
                  _deck(token="02_medium"), re.S)
    assert m, "no _CONVERGENCE_TARGETS literal"
    targets = ast.literal_eval(m.group(1))
    assert list(targets) == ["02_medium"]
    assert set(targets["02_medium"]) == {
        "max_force_tol_eV_per_A", "rms_force_tol_eV_per_A",
        "max_displ_ang", "rms_displ_ang", "energy_step_tol_eV",
        "scf_energy_tol", "max_scf_iter", "max_geom_iter"}


def test_the_recorded_targets_are_this_rungs_resolved_values():
    """They come from the config `prep` resolved for THIS rung, so a ladder
    whose rungs differ produces logs whose threshold lines differ.  Before
    2026-08-18 they came from a ladder inside the config and every rung's log
    carried every rung's targets."""
    HA_BOHR_TO_EV_ANG = 51.42208619
    text = _deck(token="03_tight", geom_gmax=2.0e-4, geom_grms=1.0e-4,
                 geom_max_steps=100, scf_conv_tol=1.0e-10)
    m = re.search(r"_CONVERGENCE_TARGETS = (\{.*?\n\})", text, re.S)
    row = ast.literal_eval(m.group(1))["03_tight"]
    assert row["max_force_tol_eV_per_A"] == pytest.approx(
        2.0e-4 * HA_BOHR_TO_EV_ANG)
    assert row["rms_force_tol_eV_per_A"] == pytest.approx(
        1.0e-4 * HA_BOHR_TO_EV_ANG)
    assert row["scf_energy_tol"] == 1.0e-10
    assert row["max_geom_iter"] == 100
