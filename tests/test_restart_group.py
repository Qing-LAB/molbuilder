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
    return sorted(ln.split()[0] for ln in deck.splitlines()
                  if ln.split() and ln.split()[0] in SIESTA_RESTART_GROUP.keys)


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


def test_clean_renders_none_and_continue_renders_every_member():
    """**M3, asserted together in one test on purpose** — the plan says the
    failure mode is that the two halves disagree, so checking them apart
    would let exactly that through."""
    assert _group_lines(_deck(restart="clean", relax_type="CG")) == []
    assert _group_lines(_deck(restart="continue", relax_type="CG")) == \
        sorted(SIESTA_RESTART_GROUP.keys)


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
    assert _group_lines(_deck(restart="continue", relax_type="Verlet")) == \
        ["DM.UseSaveDM", "MD.UseSaveXV"]
    assert _group_lines(_deck(restart="continue", relax_type="Broyden")) == \
        sorted(SIESTA_RESTART_GROUP.keys)

    # And the reason every assertion here reads EMITTED KEYS rather than raw
    # text, shown rather than asserted in a comment: the explanatory block is
    # emitted whenever the run continues and it *names* MD.UseSaveCG, so on a
    # Verlet deck the string is present while the key is not.  A substring
    # check would have passed for the wrong reason.  Caught when this test
    # first ran.
    verlet = _deck(restart="continue", relax_type="Verlet")
    assert "MD.UseSaveCG" in verlet
    assert "MD.UseSaveCG" not in _group_lines(verlet)


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
        ["DM.UseSaveDM", "MD.UseSaveXV"]
    assert _group_lines(_deck(restart="clean", relax_type="none")) == []


def test_a_missing_restart_field_reads_as_clean():
    """The safe reading of silence. The dangerous direction is resuming when
    nobody asked -- that discards nothing, but it silently changes what the
    run computed from."""
    class Bare:
        pass
    from molbuilder.siesta.input import _continues
    assert _continues(Bare()) is False


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

    assert warm({"restart": "clean"}) == []   # told to start clean
    assert [w.name for w in warm({"restart": "continue"})] == [
        "job.XV", "job.DM", "job.CG"]


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
    assert [w.name for w in _warm_declaration("job", eff)] == [
        "job.XV", "job.DM", "job.CG"]

    # ...and the framework that consumes it builds none of them.
    for mod in (_materialize, _model, _prep):
        for text, where in _string_literals(mod):
            for suffix in (".XV", ".DM", ".CG"):
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
    assert warm == {"job.XV": None, "job.DM": None, "job.CG": "optimizer"}


# --------------------------------------------------------------------- #
#  The gap this phase knowingly leaves                                  #
# --------------------------------------------------------------------- #

@pytest.mark.xfail(strict=True, reason=(
    "P4 -- PySCF has no `restart` field, so § 4 rule 2's 'the user says one "
    "thing' is unimplemented for it.  Its resume branches are gated on "
    "`chkfile` and `save_optimized_xyz`, which are WRITE flags doubling as "
    "read gates, so 'write a checkpoint but do not resume from one' cannot "
    "be said.  Separating them changes emitted-script behaviour and needs a "
    "science review of its own."))
def test_pyscf_also_says_it_with_one_field():
    names = {f.name for f in PySCFConfig.__dataclass_fields__.values()}
    assert "restart" in names
