"""The warm-file rules loader — an engine's warm-state vocabulary, as data.

Contract: `docs/execution/job-contracts.md` § 4.2a (settled 2026-08-13).
Each engine ships ONE schema-stamped ``<engine>/warm-files.toml``:
``[base]`` — what every calculation of that engine shares — plus one
section per calculation type, extending it.  This module is the ONE
reader; every consumer (the declaration builder, the wrapper inventory,
validation, the guards) derives from what it returns, which is what
retires the three hard-coded copies § 4.2a's history records drifting.

LAYER.  L1: stdlib (``tomllib``) plus ``persist`` for the schema gate —
the same leaf posture as ``task``, and load-bearing the same way: the
jobset seam, the wrapper generator and validation must all reach ONE
loader or the vocabulary forks again.

TWO QUESTIONS, TWO FUNCTIONS (§ 4.2's three-questions table):

* :func:`rules_for` — *"what does THIS calculation carry, and what does
  its deck honour?"* — type-scoped: ``[base]`` + the calculation's own
  section, refusing an unknown type BY NAMING THE SECTIONS THAT EXIST.
* :func:`inventory` — *"what might warm-start an engine here at all?"* —
  type-blind: ``[base]`` plus every section, because the wrapper's
  banner and warm detection are a HINT about the directory (they run
  where no description need exist), and an over-inclusive hint is safe
  where an over-inclusive carry is not.

THE CLOSED VOCABULARY — three keys, and it stays three (§ 4.2a):
``carry`` (``"when-continuing"`` or absent = inventory-only),
``requires_same`` (a trait name the source/destination pair must agree
on), ``honoured_by`` (the deck keyword that reads the file).  Anything
this cannot express belongs in the ONE interpreter
(``jobset/model.py::warm_carry``), and reaching for a fourth key is the
signal to design, not to patch.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .persist import check_schema

SCHEMA = "molbuilder/warm-files@1"
FILENAME = "warm-files.toml"

#: The row keys, closed by contract.  ``suffix`` identifies the row.
_ROW_KEYS = ("suffix", "carry", "requires_same", "honoured_by")
_CARRY_VALUES = ("when-continuing",)


class WarmFilesError(Exception):
    """The rules file is malformed, missing, or names nothing usable."""


@dataclass(frozen=True)
class WarmRule:
    """One row of the vocabulary: a file the engine may warm-start from."""
    suffix: str
    carry: Optional[str] = None
    requires_same: Optional[str] = None
    honoured_by: Optional[str] = None


@dataclass(frozen=True)
class WarmFilesDoc:
    """A parsed rules file: the engine and its sections, in file order."""
    engine: str
    sections: Tuple[Tuple[str, Tuple[WarmRule, ...]], ...]

    def section_names(self) -> List[str]:
        return [name for name, _ in self.sections if name != "base"]


def _rules_path(engine: str) -> Path:
    return Path(__file__).parent / engine / FILENAME


def _parse_row(obj: dict, *, where: str) -> WarmRule:
    unknown = sorted(set(obj) - set(_ROW_KEYS))
    if unknown:
        raise WarmFilesError(
            f"{where}: unknown key(s) {', '.join(map(repr, unknown))}. "
            f"The vocabulary is closed (job-contracts.md 4.2a): "
            f"{', '.join(_ROW_KEYS)}.  A rule it cannot express belongs "
            f"in warm_carry, not in a new key.")
    suffix = obj.get("suffix")
    if not isinstance(suffix, str) or not suffix:
        raise WarmFilesError(f"{where}: 'suffix' must be a non-empty string.")
    carry = obj.get("carry")
    if carry is not None and carry not in _CARRY_VALUES:
        raise WarmFilesError(
            f"{where} ({suffix!r}): carry={carry!r} is not one of "
            f"{_CARRY_VALUES} -- omit the key for an inventory-only row.")
    req = obj.get("requires_same")
    if req is not None and (not isinstance(req, str) or not req):
        raise WarmFilesError(
            f"{where} ({suffix!r}): 'requires_same' must name a trait.")
    hon = obj.get("honoured_by")
    if hon is not None and (not isinstance(hon, str) or not hon):
        raise WarmFilesError(
            f"{where} ({suffix!r}): 'honoured_by' must name a deck keyword.")
    return WarmRule(suffix=suffix, carry=carry, requires_same=req,
                    honoured_by=hon)


def load_warm_files(engine: str) -> WarmFilesDoc:
    """Read and validate ``<engine>/warm-files.toml``, preserving file order.

    File order is load-bearing both ways: :func:`rules_for` hands the
    declaration builder its rows in it (so ``Job.warm`` is stable), and
    :func:`inventory` hands the wrapper its banner/test order from it.
    """
    path = _rules_path(engine)
    if not path.is_file():
        raise WarmFilesError(
            f"no {FILENAME} for engine {engine!r} (expected {path}). "
            f"An engine's warm-state vocabulary is this ONE file "
            f"(job-contracts.md 4.2a).")
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    check_schema(str(raw.pop("schema", "")), SCHEMA, label=str(path))
    file_engine = raw.pop("engine", None)
    if file_engine != engine:
        raise WarmFilesError(
            f"{path}: engine = {file_engine!r} but the file sits in "
            f"{engine!r}'s package -- the two must agree.")
    sections: List[Tuple[str, Tuple[WarmRule, ...]]] = []
    seen_suffixes: Dict[str, str] = {}
    for name, body in raw.items():
        if not isinstance(body, dict) or set(body) - {"file"}:
            raise WarmFilesError(
                f"{path}: section [{name}] must hold only [[{name}.file]] "
                f"rows.")
        rows = []
        for i, row in enumerate(body.get("file", ())):
            rule = _parse_row(row, where=f"{path} [{name}] row {i}")
            prior = seen_suffixes.get(rule.suffix)
            if prior is not None:
                raise WarmFilesError(
                    f"{path}: suffix {rule.suffix!r} appears in [{name}] "
                    f"and [{prior}] -- one row per file, one section per "
                    f"row.")
            seen_suffixes[rule.suffix] = name
            rows.append(rule)
        sections.append((name, tuple(rows)))
    if not any(name == "base" for name, _ in sections):
        raise WarmFilesError(
            f"{path}: no [base] section.  Every engine has one -- it may "
            f"be empty, but its absence reads as a truncated file.")
    return WarmFilesDoc(engine=engine, sections=tuple(sections))


def rules_for(engine: str, calculation: str) -> List[WarmRule]:
    """``[base]`` + the calculation's own section, in file order.

    The type-scoped answer: what THIS calculation carries between stages
    and what its deck honours.  An unknown type is refused **by naming
    the sections that exist** — the § 4.2a growth rule's other half: a
    new calculation type is a new section, and the refusal is the prompt
    that says where it goes.
    """
    doc = load_warm_files(engine)
    table = dict(doc.sections)
    if calculation not in table or calculation == "base":
        raise WarmFilesError(
            f"engine {engine!r} has no warm-file section for calculation "
            f"{calculation!r}.  Sections: "
            f"{', '.join(doc.section_names()) or '(none)'} "
            f"(job-contracts.md 4.2a: a new calculation type is a new "
            f"section in {engine}/{FILENAME}, never a branch).")
    return list(table.get("base", ())) + list(table[calculation])


def inventory(engine: str) -> Tuple[str, ...]:
    """Every suffix the engine may warm-start from — ``[base]`` plus all
    sections, in file order.  The type-blind answer, for the wrapper's
    banner and warm detection: a HINT about a directory, safe to
    over-include, required to under-include nothing."""
    doc = load_warm_files(engine)
    return tuple(rule.suffix
                 for _, rows in doc.sections
                 for rule in rows)
