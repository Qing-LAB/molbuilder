"""Validation across a ladder — the members, and the sequence.

**Module:** L3. Imports ``issues``, and ``template`` at call time; called by producers and
by the web Build path. Imported by nothing below it.

**Contract:** [`engines/stages.md`](?doc=engines/stages.md) § 4 — R2 (*a stage
is validated as a resolved **whole**, never as a diff*) and R3 (*the sequence is
checked as well as its members*) · [`science/validation.md`](?doc=science/validation.md)
§ 4.1 (one result type; ``where`` is the stable id) ·
[`engines/tuning.md`](?doc=engines/tuning.md) § 2 — **which** parameters must
not go backwards, and by how much, is that document's to say and not this one's.

Two questions, and they are genuinely different:

**R2 — is each stage sound?**  Two overrides can each be individually
reasonable and jointly wrong: a mesh cutoff that is fine, a basis that is fine,
and a pair that is under-converged together. So each stage is judged as a
*resolved whole* -- and the door that judges it is the RENDER gate
(`script_emit.render_deck`, step 3.3): every rung's deck passes the shipped
validator, with the calculation kind, before a line of it exists.

**R3 — is the ORDER sound?**  R2 makes every stage individually sound and says
nothing about the order they are in, yet the order is the whole point of having
several. A ladder that *loosens* — stage 2 coarser than stage 1 — passes R2
twice and throws away what the first stage paid for. Those findings carry **no**
stage label: a fact about the description is not a fact about a member of it.
"""
from __future__ import annotations

import dataclasses
from typing import Any, List, Optional, Sequence, Tuple

from ..issues import Issue


# --------------------------------------------------------------------- #
#  R3 — which parameters must not go backwards                          #
# --------------------------------------------------------------------- #

#: Values within this relative distance are "the same tier", not a loosening.
#: A ladder that holds a parameter steady is ordinary — most stages change one
#: or two things — and float arithmetic on a round-tripped value must not be
#: read as a step backwards.
_REL_TOL = 1e-9


def check_ladder_does_not_loosen(
    resolved: Sequence[Tuple[str, Any]], *, engine: str = "siesta",
    kind: str = "optimization",
) -> List[Issue]:
    """§ 4 R3. One ``warn`` per parameter that goes backwards along the ladder.

    ``resolved`` is ``[(stage name, effective config), ...]`` **in ladder
    order** — the order is the thing being checked, so a
    caller that sorts or filters differently is asking a different question.

    **Which parameters, and which way, is the catalogue's** — each item's
    ``tightens`` (`engines/template.md` § 5), set where `engines/tuning.md`
    § 2 gives a tier table and nowhere else: a direction invented without
    one would be a scientific claim with no source, and a false "your ladder
    loosens" is worse than a missing one — it teaches people to ignore the
    check.

    **Only rungs of one ROLE are compared** (`template.stage_role`): a ladder
    tightens one calculation as it goes, and rungs that are different
    programs on different cells — transport's five, a vibration's relaxation
    and its force constants — are not one calculation tuned twice.  So a
    transport rung's own tolerance never reads as a loosening (M11 T-F14),
    and an optimization, whose rungs have no roles, is compared whole.

    Severity is ``warn``, never ``error``, and that follows the rule
    `validation.md § 4.1` already states: *adequacy is advisory,
    representability is blocking*. A loosening ladder runs perfectly well; what
    is in question is whether it is what the user meant. Deliberately loosening
    is even legitimate — a cheap wide scan after an expensive refinement — so
    molbuilder says what it noticed and gets out of the way.

    **No stage label.** The finding is about the description: naming one of the
    two stages would put the blame on a member for a property of the pair.
    """
    from ..template import catalogue, reads, select, stage_role
    out: List[Issue] = []
    for it in select(catalogue(), engine=engine, calculation=kind):
        if not it.tightens:
            continue
        last: dict = {}                      # role -> (stage name, value)
        for name, cfg in resolved:
            if not reads(it, engine, kind, name):
                continue
            val = getattr(cfg, it.name, None)
            if val is None:
                continue
            val = float(val)
            role = stage_role(engine, kind, name)
            if role in last and _loosens(last[role][1], val, it.tightens):
                prev_name, prev_val = last[role]
                out.append(Issue(
                    "warn",
                    f"{it.name} loosens from {prev_val:g} in stage "
                    f"{prev_name!r} to {val:g} in stage {name!r}. A ladder "
                    f"tightens as it goes (engines/tuning.md 2, the tier "
                    f"tables); a later stage that is coarser discards what "
                    f"the earlier one paid for. Deliberate? Then nothing is "
                    f"wrong -- this is advice, not a refusal",
                    where=f"stages.loosens.{it.name}",
                ))
                # One finding per parameter, not one per adjacent pair: a
                # three-stage ladder built backwards would otherwise report
                # the same mistake twice and read as two problems.
                break
            last[role] = (name, val)
    return out


def check_a_relaxation_takes_a_step(
    resolved: Sequence[Tuple[str, Any]], *, engine: str = "siesta",
    kind: str = "optimization",
) -> List[Issue]:
    """A rung that relaxes takes at least one step (plan § 5w K4; M11 SS-C5).

    ``relax_steps = 0`` beside a ``relax_type`` that moves atoms is a single
    point (``MD.Steps 0``, `read_options.F90`): the rung relaxes nothing, and
    a vibration's force constants are then taken at a geometry nobody
    relaxed.  Refused where a DESCRIPTION states it — on the rungs that read
    ``relax_steps`` (`template.reads`), as the description resolves them —
    because the value itself is one SIESTA takes: ``prep bench`` pins exactly
    0 to time one SCF on a rung's own deck, and that pin is prep's own,
    never part of a description (`engines/template.md` § 5.3).
    """
    from ..template import catalogue, one, reads
    try:
        steps = one(catalogue(), "relax_steps", engine=engine)
    except KeyError:
        return []
    out: List[Issue] = []
    for name, cfg in resolved:
        moves = getattr(cfg, "relax_type", None)
        if (moves in (None, "none") or getattr(cfg, "relax_steps", None) != 0
                or not reads(steps, engine, kind, name)):
            continue
        out.append(Issue(
            "error",
            f"stage {name!r} relaxes ({moves}) in relax_steps = 0 steps -- "
            f"MD.Steps 0 is a single point (read_options.F90), so the rung "
            f"relaxes nothing.  Give it at least one step "
            f"(engines/template.md 5.3)",
            where="config.relax_steps", stage=name))
    return out


def _loosens(prev: float, cur: float, direction: str) -> bool:
    if abs(cur - prev) <= _REL_TOL * max(abs(prev), abs(cur), 1.0):
        return False
    return cur > prev if direction == "down" else cur < prev


# --------------------------------------------------------------------- #
#  § 6.6a — two stages that resolve to the same thing                   #
# --------------------------------------------------------------------- #

#: The field that says whether a stage starts from what is in the folder.
#: **Excluded from the equality test on purpose** — see below.
_RESTART = "restart"


def check_identical_stages(resolved: Sequence[Tuple[str, Any]]) -> List[Issue]:
    """§ 6.6a. Warn where a stage recomputes the one before it and discards it.

    **Two stages may resolve to identical settings, and that is
    allowed.** `tight` followed by `tight` where the second *continues* is
    simply *more steps at these settings* — the honest way to say *keep going*
    after a stage ran out of its step budget. Refusing it would make someone
    invent a token difference to get past the check, which is worse than the
    thing being prevented.

    **Exactly one case warns: the later stage resolves identically *and*
    starts `clean`.** Then it recomputes what the stage before it just
    produced and throws that result away.

    **Why ``restart`` is not part of the comparison.** § 6.6a says *"what
    separates them is `start from`, not the overrides"* — so it is the
    **discriminator**, and a field cannot both distinguish two stages and be
    part of the test for whether they are the same. Read the other way the
    second clause would be redundant (equal configs already agree about
    `restart`), and the case where an earlier stage *continues* and a later
    identical one *cleans* — a real recompute — would slip through.

    Adjacent pairs only: *"the stage before it"*. Two identical stages with a
    different one between them do not recompute each other's output.

    Comparison is over the **resolved** pair, never the overrides: comparing
    overrides would flag the legitimate case, which is how a warning becomes
    noise people learn to click through.
    """
    out: List[Issue] = []
    for (a_name, a_cfg), (b_name, b_cfg) in zip(resolved, resolved[1:]):
        if getattr(b_cfg, _RESTART, None) != "clean":
            continue
        if not _same_but_for_restart(a_cfg, b_cfg):
            continue
        out.append(Issue(
            "warn",
            f"stage {b_name!r} resolves to the same settings as "
            f"{a_name!r} and starts clean, so it recomputes what "
            f"{a_name!r} just produced and discards that result. If you "
            f"meant *more steps at these settings*, set this stage's "
            f"restart to 'continue'",
            where="stages.recomputes_previous",
        ))
    return out


def _same_but_for_restart(a, b) -> bool:
    fa = {f.name: getattr(a, f.name) for f in dataclasses.fields(a)
          if f.name != _RESTART}
    fb = {f.name: getattr(b, f.name) for f in dataclasses.fields(b)
          if f.name != _RESTART}
    return fa == fb


__all__ = ["check_ladder_does_not_loosen", "check_identical_stages"]
