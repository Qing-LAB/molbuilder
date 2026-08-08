"""Validation across a ladder — the members, and the sequence.

**Module:** L3. Imports ``issues``, ``template``-free; called by producers and
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
and a pair that is under-converged together. So the validator is handed the
*resolved config*, and the stage's name travels beside the finding as a label.
Nothing here re-implements a single-config check; :func:`validate_ladder` calls
the shipped one, once per stage.

**R3 — is the ORDER sound?**  R2 makes every stage individually sound and says
nothing about the order they are in, yet the order is the whole point of having
several. A ladder that *loosens* — stage 2 coarser than stage 1 — passes R2
twice and throws away what the first stage paid for. Those findings carry **no**
stage label: a fact about the description is not a fact about a member of it.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, List, Optional, Sequence, Tuple

from ..issues import Issue


# --------------------------------------------------------------------- #
#  R3 — which parameters must not go backwards                          #
# --------------------------------------------------------------------- #

#: field -> (direction, what the tier ladder does, where tuning.md says so).
#:
#: ``"down"`` means a *tighter* setting is a **smaller** number, so the ladder's
#: values must not increase; ``"up"`` means tighter is larger.
#:
#: **Only what `tuning.md` gives an explicit tier table for is here.** It has
#: none for ``basis_size`` or ``pao_energy_shift``, so neither is checked — a
#: monotonic direction invented here would be a scientific claim with no source,
#: and a false "your ladder loosens" is worse than a missing one: the first
#: teaches users to ignore the check.
_MONOTONIC: dict = {
    "relax_force_tol": (
        "down", "0.10 screening -> 0.04 publishable -> 0.01 tight (eV/A)",
        "tuning.md 2.3"),
    "relax_max_displ": (
        "down", "0.30 screening -> 0.05 publishable -> 0.02 tight (A)",
        "tuning.md 2.2"),
    "dm_tolerance": (
        "down", "1e-3 screening -> 1e-4 publishable -> 1e-5 tight",
        "tuning.md 2.4"),
    "mesh_cutoff": (
        "up", "150 screening -> 350 publishable -> 500 tight (Ry)",
        "tuning.md 2.5"),
}

#: Values within this relative distance are "the same tier", not a loosening.
#: A ladder that holds a parameter steady is ordinary — most stages change one
#: or two things — and float arithmetic on a round-tripped value must not be
#: read as a step backwards.
_REL_TOL = 1e-9


def check_ladder_does_not_loosen(
    resolved: Sequence[Tuple[str, Any]],
) -> List[Issue]:
    """§ 4 R3. One ``warn`` per parameter that goes backwards along the ladder.

    ``resolved`` is ``[(stage name, effective config), ...]`` **in ladder
    order**, enabled stages only — the order is the thing being checked, so a
    caller that sorts or filters differently is asking a different question.

    Severity is ``warn``, never ``error``, and that follows the rule
    `validation.md § 4.1` already states: *adequacy is advisory,
    representability is blocking*. A loosening ladder runs perfectly well; what
    is in question is whether it is what the user meant. Deliberately loosening
    is even legitimate — a cheap wide scan after an expensive refinement — so
    molbuilder says what it noticed and gets out of the way.

    **No stage label.** The finding is about the description: naming one of the
    two stages would put the blame on a member for a property of the pair.
    """
    out: List[Issue] = []
    for field, (direction, ladder, cite) in _MONOTONIC.items():
        prev_name: Optional[str] = None
        prev_val: Optional[float] = None
        for name, cfg in resolved:
            val = getattr(cfg, field, None)
            if val is None:
                continue
            val = float(val)
            if prev_val is not None and _loosens(prev_val, val, direction):
                out.append(Issue(
                    "warn",
                    f"{field} loosens from {prev_val:g} in stage "
                    f"{prev_name!r} to {val:g} in stage {name!r}. A ladder "
                    f"tightens as it goes ({ladder}, {cite}); a later stage "
                    f"that is coarser discards what the earlier one paid "
                    f"for. Deliberate? Then nothing is wrong -- this is "
                    f"advice, not a refusal",
                    where=f"stages.loosens.{field}",
                ))
                # One finding per parameter, not one per adjacent pair: a
                # three-stage ladder built backwards would otherwise report
                # the same mistake twice and read as two problems.
                break
            prev_name, prev_val = name, val
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

    **Two enabled stages may resolve to identical settings, and that is
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


# --------------------------------------------------------------------- #
#  R2 + R3 together                                                     #
# --------------------------------------------------------------------- #

def validate_ladder(
    struct,
    template,
    stages: Sequence[Any],
    *,
    cell=None,
    resolve: Optional[Callable[[Any, Any], Any]] = None,
    validate: Optional[Callable[..., List[Issue]]] = None,
) -> List[Issue]:
    """Every finding a staged calculation has: per stage, then per sequence.

    Each enabled stage is resolved into an ordinary config and handed to the
    **shipped** validator — the same function a single run goes through, so a
    stage cannot be checked by a weaker set of rules than a plain calculation
    (which is what a parallel per-stage validator drifted into being, and why
    the old one was deleted with ``SiestaStageSpec``).

    Findings come back in ladder order, each stamped with the stage it belongs
    to, followed by the sequence's own — which carry no stage.

    ``resolve`` and ``validate`` default to SIESTA's. They are arguments
    because the *shape* of this is engine-agnostic — resolve, validate each,
    then check the order — and only the two functions are SIESTA's.
    """
    # Imported here, not at module scope: ``validation`` is imported by the
    # emitters, and reaching back up to ``siesta.input`` at import time would
    # close a cycle.  Bound to local names that shadow the parameters, which
    # is the point -- the rest of the body does not care which it got.
    if resolve is None:
        from ..siesta.input import effective_config as _resolve
        resolve = _resolve
    if validate is None:
        from . import validate as _validate
        validate = _validate

    enabled = [s for s in stages if getattr(s, "enabled", True)]
    resolved = [(s.name, resolve(template, s)) for s in enabled]

    out: List[Issue] = []
    for name, cfg in resolved:
        for issue in validate(struct, cfg, cell=cell):
            # ``dataclasses.replace`` rather than a mutation: Issue is frozen,
            # and the stamp must not reach a caller that kept a reference to
            # the validator's list.
            out.append(_stamped(issue, name))
    out.extend(check_ladder_does_not_loosen(resolved))
    out.extend(check_identical_stages(resolved))
    return out


def _stamped(issue: Issue, stage: str) -> Issue:
    return dataclasses.replace(issue, stage=stage)


__all__ = ["check_ladder_does_not_loosen", "check_identical_stages",
           "validate_ladder"]
