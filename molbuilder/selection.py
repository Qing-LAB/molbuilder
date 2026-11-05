"""Composable atom-selection rules.

A *selection rule* describes WHICH atoms of a :class:`Structure` are
"selected", in a form that is:

  * **composable** -- rules combine with set operators (union /
    intersection / difference / complement) so the user can express
    "all Au atoms except those tagged as bridge" in one expression;
  * **engine-agnostic** -- the rule tree is pure Python data with a
    JSON round-trip, so it survives a server round-trip;
  * **UI-friendly** -- the JS selection panel can map UI rows
    one-to-one onto rule classes and round-trip the rule tree
    through ``POST /api/selection/eval`` for live atom-count
    feedback.

This module is the L1 (pure-Python) layer of the selection system: the web API
(``POST /api/selection/eval``) reads the JSON form the page's filter rows build
(``lib/molview/stores.js``).

The grammar is intentionally small.  The eight "primitive" rules
(``All``, ``ByElement``, ``ByResidueName``, ``ByIndexRange``,
``ByRegion``, ``ByAtomName``, ``ByChainId``, ``ByClick``) cover every
selection scheme the modify / spectra / transport tabs need today;
the four ops (``Or`` / ``And`` / ``Minus`` / ``Not``) plus ``FirstN``
cover every composition pattern in our requirements (e.g. "first 12
Au atoms = left electrode" is ``FirstN(ByElement(['Au']), 12)``).

Adding a new primitive: add a frozen dataclass with a ``KIND`` and a
handler in :func:`evaluate`; register it in :data:`_RULE_CLASSES`, its fields
in :data:`_FIELD_KINDS`.
The reading from JSON + the equality / hashing properties come for
free from the dataclass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from typing import Any, ClassVar, Dict, FrozenSet, List, Optional, Tuple


# --------------------------------------------------------------------- #
#  Errors, and the index-list grammar                                   #
# --------------------------------------------------------------------- #


class SelectionError(ValueError):
    """A rule is malformed, references unknown data (e.g. a region
    label that isn't on the Structure), or fails JSON validation."""


#: ASCII digits alone (``re.ASCII``): the browser's one door shifts the
#: number a person types (`_atom.js` `shiftExpression`, which reads every
#: digit form as one), so a digit it did not see is refused, never read
#: unshifted.
_RANGE_TOKEN_RE = re.compile(r"^\s*(\d+)\s*(?:-\s*(\d+)\s*)?$", re.ASCII)


def parse_index_list(expression: str, *, first: int = 0,
                     count: Optional[int] = None) -> List[int]:
    """Parse ``"0-35, 100, 150-200"`` into a sorted unique list of indices.

    The one reader of the index-list grammar: comma-separated tokens, each
    ``<int>`` or ``<lo>-<hi>`` (inclusive), whitespace tolerated; an empty
    expression is an empty list.  ``first`` is where the numbering starts --
    0 for atoms (``model/overview.md`` § 2), 1 for vibrational modes
    (``engines/vibration.md`` § 4.8) -- and ``count``, when the number of
    items is known, bounds it above.

    Raises :class:`SelectionError` on a malformed token (a stray comma
    included), a reversed range (``35-0``), or an index outside
    ``[first, first + count)``.
    """
    if not expression or not expression.strip():
        return []
    out: set = set()
    for tok in expression.split(","):
        m = _RANGE_TOKEN_RE.fullmatch(tok)
        if not m:
            raise SelectionError(
                f"invalid token {tok.strip()!r} in {expression!r}; "
                f"expected '<int>' or '<int>-<int>'"
            )
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2) is not None else lo
        if hi < lo:
            raise SelectionError(f"range {lo}-{hi} has hi < lo")
        if lo < first:
            raise SelectionError(
                f"{lo} is below {first}, where the numbering starts")
        if count is not None and hi >= first + count:
            raise SelectionError(
                f"range {lo}-{hi} out of [{first}, {first + count})")
        out.update(range(lo, hi + 1))
    return sorted(out)


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

    Expression grammar (:func:`parse_index_list`, the one reader --
    the vibration's explicit mode list is read by it too, 1-based):
    comma-separated tokens, each either a single integer (``42``) or
    a range (``0-35``).  Whitespace around tokens is tolerated.
    Examples::

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
        return frozenset(
            i for i, r in enumerate(struct.residue_names)
            if r in wanted
        )

    if isinstance(rule, ByAtomName):
        wanted = set(rule.names)
        return frozenset(
            i for i, a in enumerate(struct.atom_names)
            if a in wanted
        )

    if isinstance(rule, ByChainId):
        wanted = set(rule.ids)
        return frozenset(
            i for i, c in enumerate(struct.chain_ids)
            if c in wanted
        )

    if isinstance(rule, ByIndexRange):
        return frozenset(parse_index_list(rule.expression, count=n))

    if isinstance(rule, ByRegion):
        if rule.name not in struct.regions:
            raise SelectionError(
                f"ByRegion: structure has no region labelled "
                f"{rule.name!r} (known: {sorted(struct.regions)!r})"
            )
        return frozenset(struct.regions[rule.name])

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
#  A rule as JSON, both ways                                            #
# --------------------------------------------------------------------- #
#
# The on-the-wire form is::
#
#     {"op": "<KIND>", "<field>": <value>, ...}
#
# with nested rules the same way.  JSON's lists become tuples, which keeps a
# frozen dataclass hashable.

_RULE_CLASSES: Tuple[type, ...] = (
    All, ByElement, ByResidueName, ByAtomName, ByChainId,
    ByIndexRange, ByRegion, ByClick,
    Or, And, Minus, Not, FirstN,
)

_KIND_TO_CLS: Dict[str, type] = {cls.KIND: cls for cls in _RULE_CLASSES}


#: What each field of a rule must be, declared rather than inferred: the
#: annotation on a frozen dataclass is documentation, and nothing was checking
#: it.
#:
#: WHY THIS EXISTS.  Without it ``from_json`` accepts any JSON value, and the
#: wrong type raises a bare :class:`TypeError` -- from ``set(rule.elements)``
#: deep inside :func:`evaluate`.  ``web/blueprints/selection.py`` catches
#: :class:`SelectionError` and only that, so each of those escapes as Flask's
#: HTML 500 page WITH A STACK TRACE where `web-api.md` § 1 requires a JSON 400
#: -- on a route the filter panel calls on every keystroke.  A malformed rule
#: is a bad REQUEST, and it is refused here, at the boundary where untrusted
#: JSON becomes a rule.
_FIELD_KINDS: Dict[str, str] = {
    "operands":   "rule-seq",
    "a":          "rule",
    "b":          "rule",
    "rule":       "rule",
    "elements":   "str-seq",
    "names":      "str-seq",
    "ids":        "str-seq",
    "indices":    "int-seq",
    "expression": "str",
    "name":       "str",
    "n":          "int",
}


def to_json(rule: Rule) -> Dict[str, Any]:
    """A rule (or rule tree) as the JSON-able dict :func:`from_json` reads back
    equal: ``{"op": <KIND>, <field>: <value>}``, nested rules the same way and
    tuples as lists -- a selection a script keeps in a file or sends to
    ``/api/selection/eval``."""
    out: Dict[str, Any] = {"op": rule.KIND}
    for f in fields(rule):
        kind = _FIELD_KINDS[f.name]
        value = getattr(rule, f.name)
        out[f.name] = (to_json(value) if kind == "rule"
                       else [to_json(r) for r in value] if kind == "rule-seq"
                       else list(value) if kind.endswith("-seq")
                       else value)
    return out


def _read_field(op: str, name: str, raw: Any) -> Any:
    """The value a rule holds for this field, or refused naming both."""
    kind = _FIELD_KINDS[name]
    if kind == "rule":
        return from_json(raw)
    if kind == "str":
        if not isinstance(raw, str):
            raise SelectionError(
                f"from_json[{op!r}]: {name!r} must be a str, got "
                f"{type(raw).__name__}")
        return raw
    if kind == "int":
        # bool is an int subclass; True as a count is a mistake.
        if not isinstance(raw, int) or isinstance(raw, bool):
            raise SelectionError(
                f"from_json[{op!r}]: {name!r} must be an int, got "
                f"{type(raw).__name__}")
        return raw
    if not isinstance(raw, (list, tuple)):
        what = "a list of rules" if kind == "rule-seq" else "a list"
        raise SelectionError(
            f"from_json[{op!r}]: {name!r} must be {what}, got "
            f"{type(raw).__name__}")
    if kind == "rule-seq":
        return tuple(from_json(r) for r in raw)
    want = str if kind == "str-seq" else int
    for i, v in enumerate(raw):
        # bool is an int subclass; a rule holding True as an index is a
        # mistake that would evaluate to atom 1.
        if not isinstance(v, want) or (want is int and isinstance(v, bool)):
            raise SelectionError(
                f"from_json[{op!r}]: {name}[{i}] must be "
                f"{want.__name__}, got {type(v).__name__}")
    return tuple(raw)


def from_json(payload: Dict[str, Any]) -> Rule:
    """A JSON-able dict read into a rule (or rule tree).

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
    for f in fields(cls):
        if f.name not in payload:
            # ``All`` has no fields; every other rule has at least
            # one required field.  Missing one means the caller
            # built a malformed rule.
            raise SelectionError(
                f"from_json[{op!r}]: missing required field {f.name!r}"
            )
        kwargs[f.name] = _read_field(op, f.name, payload[f.name])
    return cls(**kwargs)


__all__ = [
    "SelectionError",
    "Rule",
    # Primitives:
    "All", "ByElement", "ByResidueName", "ByAtomName", "ByChainId",
    "ByIndexRange", "ByRegion", "ByClick",
    # Operators:
    "Or", "And", "Minus", "Not", "FirstN",
    # Evaluator + the JSON form, both ways:
    "evaluate", "to_json", "from_json",
    # The index-list grammar (atoms here; the vibration's explicit modes):
    "parse_index_list",
]
