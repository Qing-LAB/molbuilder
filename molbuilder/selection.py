"""Composable atom-selection rules.

A *selection rule* describes WHICH atoms of a :class:`Structure` are
"selected", in a form that is:

  * **composable** -- rules combine with set operators (union /
    intersection / difference / complement) so the user can express
    "all Au atoms except those tagged as bridge" in one expression;
  * **engine-agnostic** -- the rule tree is pure Python data with a
    JSON round-trip, so it serialises into the ``.molstruct.json``
    sidecar and survives a server round-trip / re-evaluation;
  * **UI-friendly** -- the JS selection panel can map UI rows
    one-to-one onto rule classes and round-trip the rule tree
    through ``POST /api/selection/eval`` for live atom-count
    feedback.

This module is the L1 (pure-Python) layer of the selection system.
The L2 web API (``/api/selection/eval``), L3 JS state holder
(``lib/selection/core.js``) and L4 UI card (``lib/selection-panel.js``)
all consume / produce the JSON form defined here.

Design notes
------------

Each rule is a frozen dataclass with a ``KIND`` class variable that
labels it in the JSON form.  Boolean ops carry sub-rules in their
fields, so a tree of rules is just a nested object.  Evaluation
walks the tree and returns a :class:`frozenset` of 0-based atom
indices; everything composes off this single shape.

The grammar is intentionally small.  The five "primitive" rules
(``All``, ``ByElement``, ``ByResidueName``, ``ByIndexRange``,
``ByRegion``, ``ByAtomName``, ``ByChainId``, ``ByClick``) cover every
selection scheme the modify / spectra / transport tabs need today;
the four ops (``Or`` / ``And`` / ``Minus`` / ``Not``) plus ``FirstN``
cover every composition pattern in our requirements (e.g. "first 12
Au atoms = left electrode" is ``FirstN(ByElement(['Au']), 12)``).

Adding a new primitive: add a frozen dataclass with a ``KIND`` and a
handler in :func:`evaluate`; register it in :data:`_KIND_TO_CLS`.
The JSON round-trip + the equality / hashing properties come for
free from the dataclass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, ClassVar, Dict, FrozenSet, List, Optional, Tuple


# --------------------------------------------------------------------- #
#  Errors                                                               #
# --------------------------------------------------------------------- #


class SelectionError(ValueError):
    """A rule is malformed, references unknown data (e.g. a region
    label that isn't on the Structure), or fails JSON validation."""


# --------------------------------------------------------------------- #
#  Primitive rules                                                      #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class All:
    """Every atom in the structure.  Useful as the LHS of a
    :class:`Minus` ("everything except ...")."""
    KIND: ClassVar[str] = "all"


@dataclass(frozen=True)
class ByElement:
    """Atoms whose element symbol is in ``elements`` (case-sensitive).

    Use ``("Au",)`` to select all gold atoms.  Wrap with
    :class:`FirstN` to pick the first N (e.g. the lead-side electrode
    in a centred junction)."""
    elements: Tuple[str, ...]
    KIND: ClassVar[str] = "by_element"


@dataclass(frozen=True)
class ByResidueName:
    """Atoms whose ``residue_name`` is in ``names``.  Only useful for
    PDB-derived structures; an XYZ-only structure has every residue
    named ``"MOL"`` and this rule degenerates to all-or-none."""
    names: Tuple[str, ...]
    KIND: ClassVar[str] = "by_residue_name"


@dataclass(frozen=True)
class ByAtomName:
    """Atoms whose ``atom_name`` is in ``names``.  PDB-style labels
    (``"CA"``, ``"OP1"``).  Same caveat as :class:`ByResidueName`."""
    names: Tuple[str, ...]
    KIND: ClassVar[str] = "by_atom_name"


@dataclass(frozen=True)
class ByChainId:
    """Atoms whose ``chain_id`` is in ``ids``."""
    ids: Tuple[str, ...]
    KIND: ClassVar[str] = "by_chain_id"


@dataclass(frozen=True)
class ByIndexRange:
    """Atoms whose 0-based index matches ``expression``.

    Expression grammar (matches the frozen-atom index list
    today): comma-separated tokens, each either a single integer
    (``42``) or a range (``0-35``).  Whitespace around tokens is
    tolerated.  Examples::

        "0-35"
        "0-35, 100, 150-200"
        "42"
    """
    expression: str
    KIND: ClassVar[str] = "by_index_range"


@dataclass(frozen=True)
class ByRegion:
    """Atoms tagged with region label ``name`` in ``Structure.regions``.

    Raises :class:`SelectionError` at evaluate-time if the structure
    has no such region (so the user catches typos like
    ``"L_electrode"`` vs ``"L-electrode"`` immediately)."""
    name: str
    KIND: ClassVar[str] = "by_region"


@dataclass(frozen=True)
class ByClick:
    """Atoms whose indices were collected by clicking in the 3Dmol
    viewer.  Stored verbatim so the UI can round-trip the rule
    through ``/api/selection/eval`` and have the click set survive."""
    indices: Tuple[int, ...]
    KIND: ClassVar[str] = "by_click"


# --------------------------------------------------------------------- #
#  Boolean operators                                                    #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class Or:
    """Set union of the operand rules.  An atom is selected if any
    operand selects it."""
    operands: Tuple["Rule", ...]
    KIND: ClassVar[str] = "or"


@dataclass(frozen=True)
class And:
    """Set intersection of the operand rules.  An atom is selected
    only if every operand selects it."""
    operands: Tuple["Rule", ...]
    KIND: ClassVar[str] = "and"


@dataclass(frozen=True)
class Minus:
    """``a`` minus ``b``: atoms selected by ``a`` but not by ``b``.
    Asymmetric, unlike Or / And."""
    a: "Rule"
    b: "Rule"
    KIND: ClassVar[str] = "minus"


@dataclass(frozen=True)
class Not:
    """Complement: every atom NOT selected by ``rule``.  Equivalent
    to ``Minus(All(), rule)`` but more compact."""
    rule: "Rule"
    KIND: ClassVar[str] = "not"


@dataclass(frozen=True)
class FirstN:
    """First ``n`` atoms selected by ``rule`` (in ascending index
    order).  Useful for "first 12 Au atoms = left electrode" where
    ``ByElement(("Au",))`` would also pick up gold bridges or tips."""
    rule: "Rule"
    n: int
    KIND: ClassVar[str] = "first_n"


# Type alias for any rule -- used purely for documentation /
# IDE-hovers, not enforced at runtime.
Rule = Any  # one of the dataclasses above (PEP 604 union would be
            # cleaner but its hash semantics don't combine well with
            # isinstance() chains).


# --------------------------------------------------------------------- #
#  Evaluation                                                           #
# --------------------------------------------------------------------- #


def evaluate(rule: Rule, struct) -> FrozenSet[int]:
    """Evaluate ``rule`` against ``struct`` and return the set of
    selected 0-based atom indices.

    Raises :class:`SelectionError` if the rule references something
    that doesn't exist on the structure (an unknown region label,
    an out-of-range click index, ...).
    """
    n = len(struct.elements)
    return _evaluate(rule, struct, n)


def _evaluate(rule: Rule, struct, n: int) -> FrozenSet[int]:
    # Dispatch on type.  An isinstance() chain rather than a dict
    # of handlers keeps the call-graph easy to read in profiles and
    # lets each branch's body access the rule's fields with full
    # type-safety (the dataclass attributes).

    if isinstance(rule, All):
        return frozenset(range(n))

    if isinstance(rule, ByElement):
        wanted = set(rule.elements)
        return frozenset(
            i for i, e in enumerate(struct.elements) if e in wanted
        )

    if isinstance(rule, ByResidueName):
        wanted = set(rule.names)
        # residue_names is a list; missing on synthetic Structures
        # built without PDB metadata, but the dataclass default is
        # an empty list per element (not None), so this is safe.
        return frozenset(
            i for i, r in enumerate(struct.residue_names or [])
            if r in wanted
        )

    if isinstance(rule, ByAtomName):
        wanted = set(rule.names)
        return frozenset(
            i for i, a in enumerate(struct.atom_names or [])
            if a in wanted
        )

    if isinstance(rule, ByChainId):
        wanted = set(rule.ids)
        return frozenset(
            i for i, c in enumerate(struct.chain_ids or [])
            if c in wanted
        )

    if isinstance(rule, ByIndexRange):
        return frozenset(_parse_index_range(rule.expression, n))

    if isinstance(rule, ByRegion):
        regions = getattr(struct, "regions", {}) or {}
        if rule.name not in regions:
            raise SelectionError(
                f"ByRegion: structure has no region labelled "
                f"{rule.name!r} (known: {sorted(regions)!r})"
            )
        return frozenset(regions[rule.name])

    if isinstance(rule, ByClick):
        for idx in rule.indices:
            if not 0 <= idx < n:
                raise SelectionError(
                    f"ByClick: index {idx} out of range [0, {n})"
                )
        return frozenset(rule.indices)

    if isinstance(rule, Or):
        result: set = set()
        for op in rule.operands:
            result |= _evaluate(op, struct, n)
        return frozenset(result)

    if isinstance(rule, And):
        if not rule.operands:
            # Empty AND = full set (the identity for intersection).
            # The empty OR is empty set; both follow the lattice
            # convention.  Surface the case explicitly so a
            # programmer reading this isn't surprised.
            return frozenset(range(n))
        it = iter(rule.operands)
        acc = _evaluate(next(it), struct, n)
        for op in it:
            acc = acc & _evaluate(op, struct, n)
            if not acc:
                # Early-exit: intersection with empty stays empty.
                break
        return frozenset(acc)

    if isinstance(rule, Minus):
        return _evaluate(rule.a, struct, n) - _evaluate(rule.b, struct, n)

    if isinstance(rule, Not):
        return frozenset(range(n)) - _evaluate(rule.rule, struct, n)

    if isinstance(rule, FirstN):
        if rule.n < 0:
            raise SelectionError(
                f"FirstN.n must be non-negative; got {rule.n}"
            )
        ordered = sorted(_evaluate(rule.rule, struct, n))
        return frozenset(ordered[: rule.n])

    raise SelectionError(
        f"unknown rule type: {type(rule).__name__}"
    )


# --------------------------------------------------------------------- #
#  Index-range parser                                                   #
# --------------------------------------------------------------------- #

_RANGE_TOKEN_RE = re.compile(r"^\s*(\d+)\s*(?:-\s*(\d+)\s*)?$")


def _parse_index_range(expression: str, n: int) -> List[int]:
    """Parse ``"0-35, 100, 150-200"`` into a sorted unique list of
    0-based indices in ``[0, n)``.  Empty expression -> empty list.

    Raises :class:`SelectionError` on malformed tokens, reversed
    ranges (``35-0``), or out-of-range values.
    """
    if not expression or not expression.strip():
        return []
    out: set = set()
    for tok in expression.split(","):
        m = _RANGE_TOKEN_RE.match(tok)
        if not m:
            raise SelectionError(
                f"ByIndexRange: invalid token {tok!r} in expression "
                f"{expression!r}; expected '<int>' or '<int>-<int>'"
            )
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2) is not None else lo
        if hi < lo:
            raise SelectionError(
                f"ByIndexRange: range {lo}-{hi} has hi < lo"
            )
        if lo < 0 or hi >= n:
            raise SelectionError(
                f"ByIndexRange: range {lo}-{hi} out of [0, {n})"
            )
        out.update(range(lo, hi + 1))
    return sorted(out)


# --------------------------------------------------------------------- #
#  JSON round-trip                                                      #
# --------------------------------------------------------------------- #
#
# The on-the-wire form is::
#
#     {"op": "<KIND>", "<field>": <value>, ...}
#
# with nested rules serialised the same way.  Tuples become lists in
# JSON (json.dumps handles that automatically); on the way back in we
# normalise lists -> tuples so the round-tripped rule equals the
# original (frozen dataclasses use tuple-of-fields hashing).

# Discovery happens here so adding a new rule class only requires
# editing this list:
_RULE_CLASSES: Tuple[type, ...] = (
    All, ByElement, ByResidueName, ByAtomName, ByChainId,
    ByIndexRange, ByRegion, ByClick,
    Or, And, Minus, Not, FirstN,
)

_KIND_TO_CLS: Dict[str, type] = {cls.KIND: cls for cls in _RULE_CLASSES}


# Sub-rule field names per class -- ``to_json`` recurses into these
# when building the JSON form.  Anything not listed here is treated
# as a leaf (tuples become lists, scalars pass through).
_SUBRULE_FIELDS: Dict[type, Tuple[str, ...]] = {
    Or:     ("operands",),
    And:    ("operands",),
    Minus:  ("a", "b"),
    Not:    ("rule",),
    FirstN: ("rule",),
}

# Within sub-rule fields, which are SEQUENCES of rules vs single
# rules.  Used to decide whether to map to_json over the field or
# call to_json on it directly.
_SUBRULE_LIST_FIELDS: Dict[type, Tuple[str, ...]] = {
    Or:  ("operands",),
    And: ("operands",),
}


def to_json(rule: Rule) -> Dict[str, Any]:
    """Serialise a rule (or rule tree) into the JSON-able dict form.

    Tuples are written as lists (json.dumps does this implicitly,
    but we keep the shape explicit so a non-JSON consumer reading
    this dict isn't surprised by tuples)."""
    if type(rule) not in _KIND_TO_CLS.values():
        raise SelectionError(
            f"to_json: unknown rule type {type(rule).__name__}"
        )
    out: Dict[str, Any] = {"op": rule.KIND}
    subrule_fields      = _SUBRULE_FIELDS.get(type(rule), ())
    subrule_list_fields = _SUBRULE_LIST_FIELDS.get(type(rule), ())
    for f in fields(rule):
        v = getattr(rule, f.name)
        if f.name in subrule_list_fields:
            out[f.name] = [to_json(r) for r in v]
        elif f.name in subrule_fields:
            out[f.name] = to_json(v)
        elif isinstance(v, tuple):
            out[f.name] = list(v)
        else:
            out[f.name] = v
    return out


def from_json(payload: Dict[str, Any]) -> Rule:
    """Deserialise a JSON-able dict back into a rule (or rule tree).

    Raises :class:`SelectionError` on unknown ``op``, missing required
    fields, or malformed nested rules.
    """
    if not isinstance(payload, dict):
        raise SelectionError(
            f"from_json: expected dict, got {type(payload).__name__}"
        )
    if "op" not in payload:
        raise SelectionError(
            f"from_json: missing 'op' key in {payload!r}"
        )
    op = payload["op"]
    cls = _KIND_TO_CLS.get(op)
    if cls is None:
        raise SelectionError(
            f"from_json: unknown op {op!r} "
            f"(known: {sorted(_KIND_TO_CLS)!r})"
        )

    kwargs: Dict[str, Any] = {}
    subrule_fields      = _SUBRULE_FIELDS.get(cls, ())
    subrule_list_fields = _SUBRULE_LIST_FIELDS.get(cls, ())
    for f in fields(cls):
        if f.name not in payload:
            # ``All`` has no fields; every other rule has at least
            # one required field.  Missing one means the caller
            # built a malformed rule.
            raise SelectionError(
                f"from_json[{op!r}]: missing required field {f.name!r}"
            )
        raw = payload[f.name]
        if f.name in subrule_list_fields:
            kwargs[f.name] = tuple(from_json(r) for r in raw)
        elif f.name in subrule_fields:
            kwargs[f.name] = from_json(raw)
        elif isinstance(raw, list):
            # Leaf-level lists (e.g. ByElement.elements) round-trip
            # back to tuples to keep the frozen dataclass hashable.
            kwargs[f.name] = tuple(raw)
        else:
            kwargs[f.name] = raw
    return cls(**kwargs)


__all__ = [
    "SelectionError",
    "Rule",
    # Primitives:
    "All", "ByElement", "ByResidueName", "ByAtomName", "ByChainId",
    "ByIndexRange", "ByRegion", "ByClick",
    # Operators:
    "Or", "And", "Minus", "Not", "FirstN",
    # Evaluator + round-trip:
    "evaluate", "to_json", "from_json",
]
