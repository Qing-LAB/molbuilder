"""The run record's setup part — `model/parse.md` § 5d.3.

For every parameter a run's engine read: **the catalogue default · what the
run asked for · what the engine used**, each as THE RUN recorded it, and one
rule for when the last two disagree.  The columns come from the files the run
left, the same way for every engine:

* the **effective-parameters block** (`script_emit.read_parameters_fence`) —
  the one block both engines write: SIESTA's wrapper states every item's
  default at launch, PySCF's deck states all three columns at run time;
* what the block leaves empty, the run's other records fill: SIESTA's asked
  column from the deck itself (`script_emit.parameter(..., deck_text=)`), its
  used column from SIESTA's own ``fdf.<stamp>.log``.

The rows are the catalogue items the engine declares for this calculation and
stage (`script_emit.declarations`); items that share one block are one row.
Then the keys the engine read that no catalogue item writes, and the finding
for each row the engine did not end up with.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:                                           # pragma: no cover
    from .record import RunFiles

#: fdf's and PySCF's spellings of a logical, one vocabulary.
_TRUE = {"t", "true", ".true.", "yes", "y", "1"}
_FALSE = {"f", "false", ".false.", "no", "n", "0"}
#: fdf prints a number to ten significant digits; two values that agree to
#: that are one value.
_REL_TOL = 1e-8


def _logical(text: str) -> Optional[bool]:
    t = text.strip().lower()
    return True if t in _TRUE else False if t in _FALSE else None


def _number_and_unit(text: str):
    toks = str(text).split()
    if not toks:
        return None, None
    try:
        num = float(toks[0])
    except ValueError:
        return None, None
    return num, (toks[1].lower() if len(toks) > 1 else None)


def _same(asked: Any, used: Any) -> Optional[bool]:
    """Whether two spellings are one value -- ``None`` when that cannot be
    told.  Numbers at fdf's printed precision (and the same unit when both
    state one), logicals in one vocabulary, blocks row by row, text blind to
    case and spacing."""
    if asked is None or used is None or used == "(absent)":
        return None
    if isinstance(asked, (list, tuple)) or isinstance(used, (list, tuple)):
        def rows(v):
            if isinstance(v, str):
                v = v.splitlines()
            return [" ".join(str(r if not isinstance(r, (list, tuple))
                                 else " ".join(map(str, r))).split()).lower()
                    for r in v if str(r).strip()]
        return rows(asked) == rows(used)
    if isinstance(asked, bool) or isinstance(used, bool):
        a = asked if isinstance(asked, bool) else _logical(str(asked))
        b = used if isinstance(used, bool) else _logical(str(used))
        return None if a is None or b is None else a == b
    na, ua = _number_and_unit(asked)
    nb, ub = _number_and_unit(used)
    if na is not None and nb is not None:
        if ua and ub and ua != ub:
            return None                    # converted: the log's `original` says
        return math.isclose(na, nb, rel_tol=_REL_TOL, abs_tol=0.0)
    la, lb = _logical(str(asked)), _logical(str(used))
    if la is not None and lb is not None:
        return la == lb
    return " ".join(str(asked).split()).lower() == " ".join(
        str(used).split()).lower()


def differs(asked: Any, used: Any) -> Optional[bool]:
    """THE rule (§ 5d.3): the run asked for ``asked`` and the engine did not
    end up with it.  ``used`` is a value, or SIESTA's fdf-log entry -- whose
    ``original`` is the spelling the deck gave, compared before the converted
    value, and whose ``readings`` all have to agree with what was asked.
    ``None`` is "cannot tell", which is never a finding."""
    if asked is None or used is None:
        return None
    if isinstance(used, dict):
        readings = used.get("readings") or [used]
        verdicts = [_same(asked, r.get("original", r.get("value")))
                    for r in readings]
        if any(v is False for v in verdicts):
            return True
        return False if verdicts and all(v is True for v in verdicts) else None
    same = _same(asked, used)
    return None if same is None else not same


def _stage_role(f: "RunFiles") -> Optional[str]:
    """The ROLE the run's rung plays (`template.stage_role`) -- what the
    catalogue's `stages` axis names, so the items a rung reads are the
    ones its deck was written with.  Its stage's name comes out of the token
    in the run's names, which the run door gave the record
    (`RunFiles.names`): the run's stage is known before any file of it is
    read."""
    from ...identity import parse_token
    from ...template import stage_role
    got = parse_token(f.names.stage)
    return stage_role(f.engine, f.calculation, got[1] if got else None)


def _used_from_fdf_log(keys, params, blocks) -> Any:
    """What SIESTA's log says it read for an item's keywords: the entry of
    the first keyword it read, or the block's rows."""
    from ..fdf import _norm
    for key in keys:
        if key.lower().startswith("%block"):
            name = _norm(key[len("%block"):].strip())
            if name in blocks:
                return blocks[name]
            continue
        entry = params.get(_norm(key))
        if entry is not None:
            return entry
    return None


def _engine_only(params: Dict[str, Any], written: set) -> List[Dict[str, Any]]:
    """The keys the engine read that no catalogue item writes (§ 5d.3), each
    with what it read and ``in_deck``: false only when EVERY reading was the
    engine's own default -- a key read several times is set by nobody when
    each time says ``# default value``."""
    return [
        {"key": e["key"],
         **({"readings": e["readings"]} if "readings" in e
            else {"value": e.get("value")}),
         "in_deck": not all(r.get("default", False)
                            for r in e.get("readings", [e]))}
        for label, e in params.items() if label not in written]


def setup_rows(f: "RunFiles") -> Dict[str, Any]:
    """``{"setup": {"rows", "engine_only"}, "verdict": {"findings"}}`` for the
    run ``f`` describes -- each part present only when a file states it."""
    from ... import script_emit as _sc
    from ..fdf import _norm
    from ..registry import parse as _parse
    from .record import _read

    if f.engine not in ("siesta", "pyscf"):
        return {}
    block = {r["item"]: r for r in
             _sc.read_parameters_fence(f.wrapper_text)
             + _sc.read_parameters_fence(_read(f.pyscf_log))}
    deck_text = _read(f.deck) if f.engine == "siesta" else ""
    params: Dict[str, Any] = {}
    blocks: Dict[str, Any] = {}
    if f.fdf_log is not None:
        try:
            res = _parse(f.fdf_log)
            params, blocks = dict(res.params), dict(res.blocks)
        except Exception:                                  # noqa: BLE001
            pass

    items = _sc.declarations(engine=f.engine, calculation=f.calculation,
                             stage=_stage_role(f))
    rows: List[Dict[str, Any]] = []
    by_keys: Dict[tuple, Dict[str, Any]] = {}
    written: set = set()
    for it in items:
        p = _sc.parameter(it.name, f.engine, deck_text=deck_text or None)
        keys = tuple(p.writes)
        if f.engine == "siesta" and not keys:
            continue                   # no keyword: the launch's, not the deck's
        written.update(_norm(k[len("%block"):].strip())
                       if k.lower().startswith("%block") else _norm(k)
                       for k in keys)
        if keys and keys in by_keys:           # items sharing one block
            by_keys[keys]["items"].append(it.name)
            continue
        b = block.get(it.name, {})
        row: Dict[str, Any] = {"item": it.name, "items": [it.name],
                               "keys": list(keys)}
        if b.get("default") is not None:
            row["default"] = b["default"]
        asked = b.get("asked")
        if asked is None and deck_text:
            # EACH keyword's answer when the item writes several (`Spin.Fix`
            # and `Spin.Total`), the one answer otherwise.
            per_key = _sc.deck_values(it.name, f.engine, deck_text)
            asked = (per_key if len(per_key) > 1
                     else next(iter(per_key.values()), None))
        if asked is not None:
            row["asked"] = asked
        used = b.get("used")
        if used is None and (params or blocks):
            if isinstance(asked, dict):
                used = {k: v for k in asked
                        for v in [_used_from_fdf_log((k,), params, blocks)]
                        if v is not None} or None
            else:
                used = _used_from_fdf_log(keys, params, blocks)
        if used is not None:
            row["used"] = used
        if isinstance(asked, dict):
            verdicts = [differs(v, (used or {}).get(k))
                        for k, v in asked.items()]
            d = (True if any(v is True for v in verdicts) else
                 False if verdicts and all(v is False for v in verdicts)
                 else None)
        else:
            d = differs(asked, used)
        if d is not None:
            row["differs"] = d
        rows.append(row)
        if keys:
            by_keys[keys] = row
    # A row no file states anything for says nothing (§ 5d.1a): a run from
    # before its engine printed the block, whose deck does not set the item.
    rows = [r for r in rows if any(k in r for k in ("default", "asked",
                                                     "used"))]
    for row in rows:
        if len(row["items"]) == 1:
            row.pop("items")
    # ASKED ≠ USED FIRST (§ 5d.3), then the catalogue's own order.
    rows.sort(key=lambda r: 0 if r.get("differs") else 1)

    out: Dict[str, Any] = {}
    setup: Dict[str, Any] = {}
    if rows:
        setup["rows"] = rows
    engine_only = _engine_only(params, written)
    if engine_only:
        setup["engine_only"] = engine_only
    if setup:
        out["setup"] = setup
    findings = [
        {"id": "asked-not-used",
         "text": (f"{r['item']}: the run asked for {r.get('asked')!s}, "
                  f"and the engine used "
                  f"{_used_text(r.get('used'))}")}
        for r in rows if r.get("differs")]
    if findings:
        out["verdict"] = {"findings": findings}
    return out


def _used_text(used: Any) -> str:
    if isinstance(used, dict):
        readings = used.get("readings") or [used]
        return " then ".join(str(r.get("value")) for r in readings)
    return str(used)
