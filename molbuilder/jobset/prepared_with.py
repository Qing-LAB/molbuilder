"""WHAT EACH STAGE WAS PREPARED WITH -- and whether its records still give
it (`job-system.md`, *The task*, *Changed since prepared*; the user,
2026-10-25).

One record, ``prepared-with.json`` at the calculation's root, merged per
stage as ``job-set.json`` is: for each stage a prep prepares, the values it
prepared the stage with -- every setting as resolved for it (the template ⊕
the stage's overrides, `resolve.resolved_ladder`), its points (a rung of
points') and its run card (the job request's values that reach it).
:func:`values_of` is the one function that reads them from a calculation's
records: prep records what it gives (:func:`record`), and `status`, Task
setup and the Results ladder ask it again of the records as they read now
(:func:`changes`) -- a read that renders and writes nothing, naming each
value that differs.  Launch compares nothing (`job-system.md` § 6.0).
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..runfiles import PREPARED_WITH_FILE as FILENAME

SCHEMA = "molbuilder/prepared-with@1"


def _plain(value: Any) -> Any:
    """``value`` as JSON holds it -- a tuple a list, a dataclass a dict --
    so a value recorded and one read now compare as JSON does."""
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _frames(calc, plan) -> Optional[int]:
    """The cited set's frame count as this prep leaves it -- the citation's
    record read through ``plan`` when the plan writes it (a calculation's
    first prep), else ``None``: the record on disk answers
    (`transport.stages.frames_of`)."""
    if plan is None:
        return None
    from ..transport.compose import PROVENANCE_FILE
    path = Path(calc.base) / PROVENANCE_FILE
    if not plan.decides(path) or not plan.is_file(path):
        return None
    return int(json.loads(plan.read_text(path)).get("frames") or 1)


def values_of(calc, stage: str, *, frames: Optional[int] = None
              ) -> Dict[str, Any]:
    """THE VALUES ``stage`` IS PREPARED WITH, from ``calc``'s records --
    ``settings``: every field of its configuration as resolved for it;
    ``points``: a rung of points' points, in words, else ``None``;
    ``run_card``: the job request's values that reach it (its run card,
    the calculation's allocation and notification), else ``None``.
    ``frames`` is the cited set's frame count where the caller holds it.
    Raises ``ValueError`` when the template does not resolve."""
    from ..resolve import resolved_ladder
    from ..identity import stage_key
    from ..transport.stages import frames_of, rung_points
    from .engines import engine_seam
    task = calc.task
    ladder = resolved_ladder(calc.template_text or "", task,
                             engine_seam(task.engine).config_cls)
    cfg = next(c for name, c in ladder if stage_key(name) == stage_key(stage))
    settings = {f.name: _plain(getattr(cfg, f.name))
                for f in dataclasses.fields(cfg)}
    points = None
    if task.calculation == "transport":
        pts = rung_points(task, stage, frames=(
            frames if frames is not None else frames_of(calc.base)))
        points = [pt.words() for pt in pts] or None
    request = calc.request
    run_card = None
    if request is not None:
        whole = request.to_dict()
        run_card = _plain({
            "execution": request.run_condition(stage),
            "allocation": dataclasses.asdict(request.allocation),
            "notify": whole.get("notify")})
    return {"settings": settings, "points": points, "run_card": run_card}


def read(base) -> Dict[str, Dict[str, Any]]:
    """Every stage's recorded values, by stage name -- ``{}`` before any prep
    recorded one."""
    from ..persist import read_json
    path = Path(base) / FILENAME
    if not path.is_file():
        return {}
    return dict(read_json(path).get("stages") or {})


def record(calc, stages, *, plan) -> Path:
    """PREP'S RECORD of ``stages``: each one's values from ``calc``'s records
    as this prep read them, merged over what the file holds, written
    through ``plan`` (nothing is written until the plan is carried out)."""
    from ..persist import json_text
    from ..identity import stage_key
    path = Path(calc.base) / FILENAME
    held = (json.loads(plan.read_text(path)).get("stages") or {}
            if plan.is_file(path) else {})
    frames = _frames(calc, plan)
    for st in stages:
        held = {k: v for k, v in held.items() if stage_key(k) != stage_key(st)}
        held[st] = values_of(calc, st, frames=frames)
    return plan.text(path, json_text({"schema": SCHEMA, "stages": held}))


def _words(value: Any) -> str:
    """A value as a person reads it in a difference."""
    if isinstance(value, str):
        return value
    return json.dumps(value)


#: A run card block in the words a person knows it by (`job-request.md`
#: § 3: the run card, the allocation, the notification).
_BLOCK_WORDS = {"execution": "run card", "allocation": "allocation",
                "notify": "notify"}

#: The one spelling of a stage whose prep recorded nothing (prepared before
#: 2.41.0) -- `status`, Task setup and the contract say it with this.
UNRECORDED_SAID = ("what it was prepared with is not recorded (prepared "
                   "before 2.41.0)")

#: What a difference is OF (`job-system.md`, *The task*, *Changed since
#: prepared*): the description's values -- ``settings``, ``points`` -- or
#: the job request's, ``run_card``.  A launched group changed in its run
#: card alone is prepared again for its request, its runs kept; one changed
#: in what it computes starts from the beginning.
SETTINGS, POINTS, RUN_CARD = "settings", "points", "run_card"


@dataclasses.dataclass(frozen=True)
class Difference:
    """One value that differs: ``kind`` (:data:`SETTINGS`, :data:`POINTS`,
    :data:`RUN_CARD`) and ``said``, ``name: was → now``."""
    kind: str
    said: str


def _differences(was: Dict[str, Any], now: Dict[str, Any]
                 ) -> List[Difference]:
    """Each value that differs, ``name: was → now`` -- the settings by
    their item's name, the run card's by its block and field, the points
    as their list.  **Only what the person changed is a difference**: a
    key one side holds and the other does not is the catalogue's or the
    request's change between two molbuilders -- an item added or retired
    since -- never the person's, and is not compared."""
    out: List[Difference] = []
    s_was, s_now = was.get("settings") or {}, now.get("settings") or {}
    for name in sorted(set(s_was) & set(s_now)):
        if s_was[name] != s_now[name]:
            out.append(Difference(SETTINGS, f"{name}: {_words(s_was[name])}"
                                            f" → {_words(s_now[name])}"))
    if was.get("points") != now.get("points"):
        out.append(Difference(
            POINTS, f"points: {', '.join(was.get('points') or ['none'])} → "
                    f"{', '.join(now.get('points') or ['none'])}"))
    rc_was, rc_now = was.get("run_card") or {}, now.get("run_card") or {}
    for block in sorted(set(rc_was) & set(rc_now)):
        a, b = rc_was.get(block) or {}, rc_now.get(block) or {}
        said = _BLOCK_WORDS.get(block, block)
        if not isinstance(a, dict) or not isinstance(b, dict):
            if a != b:
                out.append(Difference(
                    RUN_CARD, f"{said}: {_words(a)} → {_words(b)}"))
            continue
        for key in sorted(set(a) & set(b)):
            if a[key] != b[key]:
                out.append(Difference(
                    RUN_CARD, f"{said} {key}: {_words(a[key])} → "
                              f"{_words(b[key])}"))
    return out


def run_card_only(differences) -> bool:
    """Whether every difference is the job request's (:data:`RUN_CARD`) --
    a launched group so changed is prepared again for its request, its
    runs kept (`job-request.md` § 5).  False for none at all."""
    ds = list(differences)
    return bool(ds) and all(d.kind == RUN_CARD for d in ds)


def changes(calc, stage: str) -> Tuple[bool, List[Difference]]:
    """WHETHER ``stage`` IS STILL WHAT ITS RECORDS GIVE -- ``(recorded,
    differences)``: ``recorded`` false when its prep recorded no values
    (prepared before 2.41.0), and nothing is compared; else each value that
    differs (:class:`Difference`, ``mesh_cutoff: 300 → 400``), ``[]`` when
    none does.  A template that does not resolve now is said as the one
    difference, in its reader's words."""
    from ..identity import stage_key
    held = read(calc.base)
    was = next((v for k, v in held.items()
                if stage_key(k) == stage_key(stage)), None)
    if was is None:
        return False, []
    try:
        now = values_of(calc, stage)
    except ValueError as exc:
        return True, [Difference(SETTINGS,
                                 f"its template does not resolve now: {exc}")]
    return True, _differences(was, now)
