"""P3 unit 4 — one `restart` field, expanded into the engine's group.

Contract: ``docs/execution/run-identity.md`` § 4 — rule 1 (*an engine declares
its group*), rule 2 (*the user says one thing; the generator sets the group*),
rule 3 (*it is per-stage, because it is an ordinary field*) — and
``docs/execution/job-contracts.md`` § 4.2 (which files those parameters
govern).

**The failure this prevents is silent in two directions** (§ 4): *honoured
with nothing to load*, where the deck says resume and the engine cold-starts;
and *present but not honoured*, where the files are right there and the stage
starts from scratch. The second was live until 2026-08-08 — the renderer read
three booleans that defaulted to True and never read ``restart`` at all, so
``--restart clean`` emitted the whole group and a stage told to start clean
continued.
"""
from __future__ import annotations

import pytest

from molbuilder.config.pyscf import PYSCF_RESTART_GROUP, PySCFConfig
from molbuilder.config.siesta import SIESTA_RESTART_GROUP, SiestaConfig
from molbuilder.identity import RestartGroup
from molbuilder.siesta.input import render_fdf
from molbuilder.structure import Structure


def _h2o():
    return Structure(elements=["O", "H", "H"],
                     positions=[[0.0, 0.0, 0.0],
                                [0.76, 0.59, 0.0],
                                [-0.76, 0.59, 0.0]])


def _deck(**kw) -> str:
    return render_fdf(_h2o(), SiestaConfig(system_label="job", **kw))


def _group_lines(deck: str):
    """The group's members, as ``{key: ".true."/".false."}``.

    It returned a bare sorted list of KEYS until 2026-08-18, which was the
    right shape while `clean` was expressed by leaving them out: presence WAS
    the answer.  It is not any more -- both answers are written -- so a test
    that only asked which keys appeared could no longer tell a continuing deck
    from a clean one.
    """
    return {ln.split()[0]: ln.split()[1] for ln in deck.splitlines()
            if ln.split() and ln.split()[0] in SIESTA_RESTART_GROUP.keys}


def _group_keys(deck: str):
    """Which members the deck carries at all -- the run-mode question."""
    return sorted(_group_lines(deck))


def _string_literals(mod):
    """Every string a module BUILDS — literals and f-string pieces alike,
    with docstrings and comments left out.

    A grep for ``".XV"`` is not this test, and the difference is the whole
    point: the code being prevented spelled it ``f"{jobset.name}.XV"``, which
    contains no such substring. Reading the AST catches the interpolated form,
    and skipping docstrings is what lets the modules go on *quoting* the
    contract — which is where these suffixes belong.
    """
    import ast
    import inspect

    src = inspect.getsource(mod)
    tree = ast.parse(src)
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if (isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                docs.add(id(first.value))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in docs):
            yield node.value, node.lineno


# --------------------------------------------------------------------- #
#  Rule 1 — an engine declares its group                                #
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("group", [SIESTA_RESTART_GROUP, PYSCF_RESTART_GROUP],
                         ids=["siesta", "pyscf"])
def test_every_shipped_engine_declares_a_group(group):
    """§ 4 rule 1: *"A new engine that cannot fill this in is a new engine
    whose restart behaviour nobody has thought about yet."*

    Both halves are required. An identity literal with no bound parameters
    describes a name and not a resume; bound parameters with no literal
    describes flags with nothing to key on."""
    assert isinstance(group, RestartGroup)
    assert group.literal
    assert group.mechanism


def test_the_two_engines_mean_the_same_idea_by_different_mechanisms():
    """§ 4's table, read across. SIESTA declares keys; PySCF generates
    control flow. Collapsing that difference is how a design ends up
    describing only the filename."""
    assert SIESTA_RESTART_GROUP.keys
    assert not PYSCF_RESTART_GROUP.keys
    assert SIESTA_RESTART_GROUP.literal != PYSCF_RESTART_GROUP.literal


# --------------------------------------------------------------------- #
#  Rule 2 — the user says ONE thing                                     #
# --------------------------------------------------------------------- #

def test_the_members_cannot_be_set_individually_any_more():
    """§ 4 rule 2: *"no description can carry its members individually and
    disagree with itself"*.

    The three booleans are gone from the schema, which also removes the three
    generated ``--use-save-*`` CLI flags and the three web form controls,
    since both surfaces are built from the dataclass."""
    names = {f.name for f in SiestaConfig.__dataclass_fields__.values()}
    assert not (names & {"use_save_dm", "use_save_cg", "use_save_xv"})
    assert "restart" in names


def test_every_member_is_written_and_restart_decides_the_answer():
    """**M3, asserted together in one test on purpose** — the plan says the
    failure mode is that the two halves disagree, so checking them apart
    would let exactly that through.

    **`clean` writes `.false.`; it does not stay silent** (2026-08-18).  This
    read ``_group_lines(clean) == []`` and was named
    ``test_clean_renders_none_and_continue_renders_every_member``, pinning the
    premise that a key left out is a key not honoured.  Measured against
    SIESTA 5.4.2, it is not: a deck carrying none of these keys, with a `.DM`
    beside it, printed *"Attempting to read DM from file... Succeeded"* and
    opened its SCF at the previous run's converged density.  So a stage told
    to start clean continued, and a benchmark trial -- forced clean precisely
    so every point measures the same thing -- measured a warm run whenever the
    wrapper retried it.

    The direction that matters is `clean`: `continue` was always right, and
    the test that only checked `continue` would have stayed green forever."""
    clean = _group_lines(_deck(restart="clean", relax_type="CG"))
    cont = _group_lines(_deck(restart="continue", relax_type="CG"))
    assert sorted(clean) == sorted(cont) == sorted(SIESTA_RESTART_GROUP.keys)
    assert set(clean.values()) == {".false."}
    assert set(cont.values()) == {".true."}


def test_the_deck_never_carries_a_member_the_run_mode_ignores():
    """'Every member' means every *meaningful* member: a dynamics run has no
    relaxation history to reload, so ``MD.UseSaveCG`` is not emitted for it.

    **Pinned against what the code does, not what a comment claimed.** The
    condition is ``relax_kind in ("VERLET", "NOSE")``, so Broyden and FIRE do
    get ``MD.UseSaveCG`` — while the comment beside it said they ignore it.
    Unit 4 changed *when* the group is emitted and deliberately not *which*
    optimizers get this member; that is a SIESTA semantics question needing
    the manual and a science review. Asserting both halves here so whichever
    way that question is settled, the day it changes is loud."""
    assert _group_keys(_deck(restart="continue", relax_type="Verlet")) == \
        ["DM.UseSaveDM", "MD.UseSaveXV"]
    assert _group_keys(_deck(restart="continue", relax_type="Broyden")) == \
        sorted(SIESTA_RESTART_GROUP.keys)

    # WHICH members a run mode carries is independent of the ANSWER: a clean
    # Verlet deck carries the same two, saying .false.
    assert _group_keys(_deck(restart="clean", relax_type="Verlet")) == \
        ["DM.UseSaveDM", "MD.UseSaveXV"]

    # Every assertion here reads EMITTED KEYS rather than raw text, and the
    # reason is worth keeping even though the specific hazard has moved.  The
    # per-member explanatory blocks used to NAME `MD.UseSaveCG` whenever a run
    # continued, so on a Verlet deck the string was present while the key was
    # not, and a substring check passed for the wrong reason -- caught when
    # this test first ran.  Those per-member blocks were replaced on
    # 2026-08-18 by one block for the whole group, which names the FILES
    # (.XV / .DM / .CG) rather than the keys, so the trap is not armed today.
    # The keys-not-text discipline stays: what a deck says in a comment and
    # what it instructs the engine to do are different questions, and only one
    # of them is this test's.
    verlet = _deck(restart="continue", relax_type="Verlet")
    assert "MD.UseSaveCG" not in _group_keys(verlet)
    assert _group_lines(verlet)["MD.UseSaveXV"] == ".true."


def test_a_continuing_static_stage_still_reads_the_geometry():
    """A-6 (final review, 2026-08-13): a continuing stage with NO
    relaxation — the ladder's classic finisher, SCF-only at the relaxed
    geometry — declared and copied ``.XV`` while its deck emitted only
    ``DM.UseSaveDM``: the geometry sat beside the run unread and SIESTA
    computed at the DECK's coordinates — `run-identity.md` § 4's *present
    but not honoured*, silent in the worst direction (the run reports
    success at the wrong geometry).  ``MD.UseSaveXV`` is honoured at
    initialisation regardless of run mode, so it rides the same
    unconditional continue group as ``DM.UseSaveDM``; ``MD.UseSaveCG``
    stays a relaxation member."""
    assert _group_lines(_deck(restart="continue", relax_type="none")) == \
        {"DM.UseSaveDM": ".true.", "MD.UseSaveXV": ".true."}
    assert _group_lines(_deck(restart="clean", relax_type="none")) == \
        {"DM.UseSaveDM": ".false.", "MD.UseSaveXV": ".false."}


def test_a_missing_restart_field_reads_as_clean():
    """The safe reading of silence. The dangerous direction is resuming when
    nobody asked -- that discards nothing, but it silently changes what the
    run computed from."""
    class Bare:
        pass
    from molbuilder.identity import continues
    assert continues(Bare()) is False


def test_both_engines_read_restart_through_the_one_function():
    """§ 4 rule 2 says ONE field; this is the other half -- one READING of
    it.  SIESTA kept the predicate private in ``siesta/input.py`` while PySCF
    was about to need the same three lines, and two copies of *"does this run
    continue?"* are two things that can answer differently."""
    import molbuilder.identity as _id
    import molbuilder.siesta.input as _si
    import molbuilder.pyscf.input as _pi
    assert _si.continues is _id.continues
    assert _pi.continues is _id.continues
    assert not hasattr(_si, "_continues")


# --------------------------------------------------------------------- #
#  Rule 3 — it is per-stage, because it is an ordinary field            #
# --------------------------------------------------------------------- #

def test_a_stage_sets_restart_like_any_other_field():
    """§ 4 rule 3. *"A first stage is normally clean and everything after it
    continue. Nothing special is needed to say so."*"""
    from molbuilder.resolve import effective_config
    from molbuilder.task import Stage
    tpl = SiestaConfig(system_label="job", restart="clean")
    later = effective_config(tpl, {"restart": "continue"})
    assert later.restart == "continue"
    assert tpl.restart == "clean", "the template must not be mutated"


def test_prep_carries_state_only_into_a_stage_that_will_read_it():
    """The other face of *present but not honoured*: state placed beside a
    run that was told not to look at it.

    The carry used to key on the template's ``use_save_dm``, so a stage
    saying 'clean' still had the previous stage's ``.DM`` carried in."""
    # Repointed 2026-08-12 (u5): the producer this drove is deleted; the
    # declaration is built the LIVE way -- the resolved stage through the
    # one seam, exactly what `prep`'s `_job_for` hands each Job.
    from molbuilder.resolve import effective_config
    from molbuilder.siesta.stages import _warm_declaration
    from molbuilder.task import Stage
    tpl = SiestaConfig(system_label="job", relax_type="CG")

    def warm(overrides):
        eff = effective_config(tpl, overrides)
        return _warm_declaration("job", eff)

    # THE PROPERTY IS THE GATE, not the membership.  `clean` carries
    # nothing; `continue` carries whatever the rules file declares --
    # DERIVED here rather than listed, because a literal list is a fourth
    # copy of `siesta/warm-files.toml` and goes stale the moment the
    # vocabulary grows.  It did: the four accumulative records (.MD.nc,
    # .MD, .MDE, .ANI) were added 2026-08-15 and this line still said
    # [.XV, .DM, .CG].
    from molbuilder.warmfiles import rules_for
    declared = [f"job{r.suffix}"
                for r in rules_for("siesta", "optimization") if r.carry]
    assert warm({"restart": "clean"}) == []   # told to start clean
    assert [w.name for w in warm({"restart": "continue"})] == declared
    # ...and the gate is the whole point: the two answers differ.
    assert declared, "the rules file declares no carry rows at all"


# --------------------------------------------------------------------- #
#  Rule 4 — what `continue` implies is a short fixed set, and the        #
#  producer DECLARES it rather than the framework knowing it             #
# --------------------------------------------------------------------- #

def test_the_group_reaches_prep_as_a_declaration_not_as_engine_knowledge():
    """§ 4 rule 1: *"an engine declares its group ... a new engine that cannot
    fill this in is a new engine whose restart behaviour nobody has thought
    about yet."*  Rule 4 records what the alternative cost: *"the set used to
    be three suffixes written into the producer, which meant a TranSIESTA
    ladder could not express its `.TSHS` dependency without changing
    molbuilder's code."*

    So `prep` must not know SIESTA's suffixes. It reads what the job carries,
    which is why the declaration is on the JOB and travels in `job-set.json`
    to the machine that will run it.
    """
    import importlib

    from molbuilder.resolve import effective_config
    from molbuilder.siesta.stages import _warm_declaration
    from molbuilder.task import Stage

    # ``import_module``, not ``from molbuilder.jobset import materialize``:
    # `jobset/__init__.py` re-exports a FUNCTION under that name, and the
    # attribute wins over the submodule.  Written the obvious way, this check
    # read one 43-line function body and passed on a module it never opened --
    # found by a mutation that put the leak back and watched nothing happen.
    _materialize, _model, _prep = (
        importlib.import_module(f"molbuilder.jobset.{n}")
        for n in ("materialize", "model", "prep"))

    eff = effective_config(
        SiestaConfig(system_label="job", relax_type="CG"),
        {"restart": "continue"})
    from molbuilder.warmfiles import rules_for
    declared = [r.suffix for r in rules_for("siesta", "optimization")
                if r.carry]
    assert [w.name for w in _warm_declaration("job", eff)] == [
        f"job{s}" for s in declared]

    # ...and the framework that consumes it builds none of them.  Swept
    # over the DECLARED suffixes rather than a literal trio, so a suffix
    # added to the rules file is automatically checked for the same leak
    # instead of being exempt from the rule by omission.
    for mod in (_materialize, _model, _prep):
        for text, where in _string_literals(mod):
            for suffix in declared:
                assert suffix not in text, (
                    f"{mod.__name__}:{where} builds {suffix!r} -- the engine's "
                    f"group leaked back into the agnostic layer")


def test_only_the_optimizer_history_is_conditional():
    """§ 2.3.4's three rows: `.XV` *"always -- this is the point of
    continuing"*, `.DM` when the description says, `.CG` *"only if both stages
    use the same algorithm"*.

    Only the third needs a second stage, so only the third carries a
    condition — and a condition on the geometry would make a continuation
    silently lose the very thing it exists to move forward.
    """
    from molbuilder.resolve import effective_config
    from molbuilder.siesta.stages import _warm_declaration
    from molbuilder.task import Stage
    eff = effective_config(
        SiestaConfig(system_label="job", relax_type="CG"),
        {"restart": "continue"})
    warm = {w.name: w.requires_same for w in _warm_declaration("job", eff)}
    # THE PROPERTY: exactly one row is conditional, and it is the optimiser
    # history.  Asserted as a property rather than as a full dict, because
    # the membership grew on 2026-08-15 (the accumulative records) while
    # this rule did not change at all -- a dict literal made an unrelated
    # addition look like a violation of a rule about conditionality.
    conditional = {n: c for n, c in warm.items() if c is not None}
    assert conditional == {"job.CG": "optimizer"}, (
        f"expected only the optimiser history to be conditional, got "
        f"{conditional}")
    # And the geometry is unconditional -- a condition here would make a
    # continuation silently lose the very thing it exists to move forward.
    assert warm["job.XV"] is None
    assert warm["job.DM"] is None


# --------------------------------------------------------------------- #
#  The gap this phase knowingly leaves                                  #
# --------------------------------------------------------------------- #

def test_pyscf_also_says_it_with_one_field():
    """§ 4 rule 2 for the other engine.  This carried a strict xfail until
    2026-08-18: PySCF had no ``restart``, and its resume branches were gated
    on ``chkfile`` and ``save_optimized_xyz`` -- WRITE flags doubling as read
    gates, so *"write a checkpoint but do not resume from one"* was a
    sentence the engine could not say.

    `stages.md` § 1.1a consequence 3 is what closed it: a PySCF ladder is N
    decks and N jobs, so there is a real gap between rungs for the field to
    answer about, and rule 3's first-clean/rest-continue default has
    something to fill in."""
    names = {f.name for f in PySCFConfig.__dataclass_fields__.values()}
    assert "restart" in names
    fld = PySCFConfig.__dataclass_fields__["restart"]
    # The SAME two answers SIESTA gives.  A third value on one engine would
    # make `restart` mean different things in two descriptions.
    assert tuple(fld.metadata["choices"]) == ("clean", "continue")
    # And the same DEFAULT.  `continue` since 2026-08-18 (user): a run started
    # in a folder that already holds a result was started after somebody read
    # that result, so it continues from it.  `clean` is that person overriding,
    # and it overwrites (`run-identity.md` § 4 rule 3).
    assert fld.default == "continue"
