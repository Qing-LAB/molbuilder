"""fdf emit-strategy registry for extensible atom-annotation channels
(atom-annotations.md § 4).

ADDITIVE, not a rewrite.  The two built-in channels keep their existing,
Sol-validated emission:
  * ``frozen`` flag  -> ``%block Geometry.Constraints`` (``siesta/input.py``)
  * region tags      -> transport/electrode blocks (``transport/transiesta.py``)

This module adds the EXTENSION POINT: an extensible channel in
``Structure.annotations`` may carry ``fdf = "<strategy-id>"``; a strategy
registered here emits its fdf lines during assembly.  A channel with no
registered strategy is *carried but not emitted* -- so a generic user tag
never accidentally becomes engine directives.  New per-atom metadata (e.g. a
``charge`` / ``initspin`` value channel) => register one strategy, no emitter
rewrite and no risk to the proven frozen/region paths.
"""

from __future__ import annotations

from typing import Callable, Dict, List

# strategy-id -> fn(channel, struct) -> list[str] of fdf lines
_STRATEGIES: Dict[str, Callable] = {}


def register_fdf_strategy(strategy_id: str, fn: Callable) -> None:
    """Register an emit strategy.  ``fn(channel, struct)`` returns the fdf
    lines for that channel (may be empty).  Re-registering replaces."""
    if not isinstance(strategy_id, str) or not strategy_id:
        raise ValueError("strategy_id must be a non-empty string")
    if not callable(fn):
        raise ValueError("fn must be callable (channel, struct) -> [str]")
    _STRATEGIES[strategy_id] = fn


def registered_strategies() -> List[str]:
    """Sorted snapshot of registered strategy ids (diagnostics/tests)."""
    return sorted(_STRATEGIES)


def emit_channels(struct) -> List[str]:
    """fdf lines for every EXTENSIBLE channel (``struct.annotations``) that
    carries a REGISTERED ``fdf`` strategy, in sorted-name order (stable
    output).  Channels with no ``fdf`` id, or an unregistered one, are
    silently carried-not-emitted.  Built-in channels (regions/frozen) are
    NOT handled here -- their existing emitters own them."""
    lines: List[str] = []
    ann = getattr(struct, "annotations", None) or {}
    for name in sorted(ann):
        ch = ann[name]
        strat = getattr(ch, "fdf", None)
        if strat and strat in _STRATEGIES:
            lines.extend(_STRATEGIES[strat](ch, struct))
    return lines


def unregistered_channels(struct) -> List[str]:
    """Names of extensible channels carrying an ``fdf`` id with NO registered
    strategy -- for a "channel present, no fdf consumer" validator warning."""
    ann = getattr(struct, "annotations", None) or {}
    return sorted(name for name, ch in ann.items()
                  if getattr(ch, "fdf", None) and ch.fdf not in _STRATEGIES)


__all__ = ["register_fdf_strategy", "registered_strategies",
           "emit_channels", "unregistered_channels"]
