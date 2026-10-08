"""The warm-file rules loader — an engine's warm-state vocabulary, as data.

Contract: `docs/execution/job-contracts.md` § 4.2a (settled 2026-08-13).
Each engine ships ONE schema-stamped ``<engine>/warm-files.toml``:
``[base]`` — what every calculation of that engine shares — plus one
section per calculation type, extending it.  This module is the ONE
reader; every consumer (the declaration builder, the wrapper inventory,
validation, the guards) derives from what it returns.

LAYER.  L1: stdlib (``tomllib``) plus ``persist`` for the schema gate —
the same leaf posture as ``task``, and load-bearing the same way: the
jobset seam, the wrapper generator and validation must all reach ONE
loader or the vocabulary forks again.

ONE DOOR, :func:`warm_list` (§ 4.2a, plan W36 ⑧) -- the list in effect
for a calculation: its own copy beside ``task.json`` first, else the
engine's file; for a kind's section over ``[base]`` (*"what does THIS
calculation carry, and what does its deck honour?"*, refusing an unknown
kind BY NAMING THE SECTIONS THAT EXIST), or every section (*"what might
warm-start here at all?"* -- a HINT, safe to over-include where a carry is
not).  Every reader asks it with the calculation's folder and takes one of
its two views, :attr:`WarmList.carry` and :attr:`WarmList.suffixes`.

THE CLOSED VOCABULARY — three keys, and it stays three (§ 4.2a):
``carry`` (``"when-continuing"`` or absent = inventory-only),
``requires_same`` (a trait name the source/destination pair must agree
on), ``honoured_by`` (the deck keyword that reads the file).  Anything
this cannot express belongs in the ONE interpreter
(``jobset/model.py::warm_carry``), and reaching for a fourth key is the
signal to design, not to patch.

AND ONE FACT ABOUT A WHOLE SECTION, ``resumes`` (2026-09-29, the design that
signal asked for): whether a re-run of that kind of run continues from what
the last one left -- a fact no row can say.  :func:`warm_list` answers it.
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
    """A parsed rules file: the engine and its sections, in file order.
    ``path`` is WHICH file answered -- the engine's own, or a
    calculation's fine-tuned copy (U6a) -- and it is what provenance
    prints, so a surprising carry is debuggable from the plan alone."""
    engine: str
    sections: Tuple[Tuple[str, Tuple[WarmRule, ...]], ...]
    path: str = ""
    #: The one section-level fact (`job-contracts.md` § 4.2a): the sections
    #: that STATE whether a re-run of their kind resumes.  A section that
    #: states nothing is absent here, and :func:`warm_list` answers with
    #: ``[base]``'s statement, else true.
    resumes: Tuple[Tuple[str, bool], ...] = ()
    #: The second section-level fact: what a SWEEP hands from point to
    #: point, by suffix -- each a row of the same section
    #: (`engines/transport.md` § 2a.11).  ``()`` where a section states none.
    along: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()
    #: The third: the ENGINE'S SCRATCH at a run of that kind, by suffix --
    #: written and re-read within one run, recomputed by the next: never
    #: carried, never taken over into a new run (`engines/transport.md`
    #: § 2a.11).  Not rows: a scratch file is no restart file.
    scratch: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()

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


@dataclass(frozen=True)
class WarmList:
    """THE RESTART FILES IN EFFECT (`job-contracts.md` § 4.2a): which file
    answered -- the calculation's own copy (``own``) or the engine's --
    and its rows for the kind asked, ``[base]`` first, in file order (the
    order is load-bearing: ``Job.warm``'s, and the banner's)."""
    engine: str
    #: The section asked over ``[base]``; ``None`` -- every section.
    kind: Optional[str]
    path: str
    own: bool
    rules: Tuple[WarmRule, ...]
    #: Whether a re-run of this kind of run continues from what the last
    #: one left -- the section's statement, else ``[base]``'s, else true.
    resumes: bool
    #: What a sweep of this kind hands from a done point to the next, by
    #: suffix -- the section's ``along`` (`engines/transport.md` § 2a.11).
    along: Tuple[str, ...] = ()
    #: The engine's scratch at a run of this kind, by suffix -- the
    #: section's ``scratch``: never carried, never taken over.
    scratch: Tuple[str, ...] = ()

    @property
    def carry_rules(self) -> Tuple[WarmRule, ...]:
        """The rows a continuing run takes -- ``carry`` set."""
        return tuple(r for r in self.rules if r.carry)

    @property
    def carry(self) -> Tuple[str, ...]:
        """WHAT CARRIES: the suffixes of :attr:`carry_rules`."""
        return tuple(r.suffix for r in self.carry_rules)

    @property
    def suffixes(self) -> Tuple[str, ...]:
        """EVERY RESTART FILE: every row's suffix."""
        return tuple(r.suffix for r in self.rules)


def warm_list(engine: str, kind: Optional[str] = None,
              base=None) -> WarmList:
    """THE ONE DOOR to the restart files (`job-contracts.md` § 4.2a,
    `execution/architecture.md` § 3.2): the list in effect for the
    calculation in ``base`` -- its own ``warm-files.toml`` beside
    ``task.json`` first, else the engine's -- for ``kind``'s section over
    ``[base]``, or every section when no kind is asked.  Asked with no
    folder it is the engine's file alone: the question a folder no
    calculation describes poses.  An unknown kind is refused by naming the
    sections that exist -- a new calculation type is a new section, never a
    branch."""
    doc = _load(engine, base)
    stated = dict(doc.resumes)
    if kind is None:
        rules = tuple(r for _, rows in doc.sections for r in rows)
        resumes = stated.get("base", True)
    else:
        table = dict(doc.sections)
        if kind not in table or kind == "base":
            raise WarmFilesError(
                f"engine {engine!r} has no warm-file section for calculation "
                f"{kind!r}.  Sections: "
                f"{', '.join(doc.section_names()) or '(none)'} "
                f"(job-contracts.md 4.2a: a new calculation type is a new "
                f"section in {engine}/{FILENAME}, never a branch).")
        rules = tuple(table.get("base", ())) + tuple(table[kind])
        resumes = stated.get(kind, stated.get("base", True))
    own = base is not None and Path(doc.path) == Path(base) / FILENAME
    return WarmList(engine=engine, kind=kind, path=doc.path, own=own,
                    rules=rules, resumes=resumes,
                    along=(dict(doc.along).get(kind, ()) if kind else ()),
                    scratch=(dict(doc.scratch).get(kind, ()) if kind else ()))


def _load(engine: str, base_dir=None) -> WarmFilesDoc:
    """Read and validate the rules file, preserving file order.

    NEAREST FILE WINS (U6a, § 4.2a's template mechanism): a calculation
    carrying its own ``warm-files.toml`` beside ``task.json`` is the
    fine-tuned state and answers for that calculation; without one, the
    engine's own file answers -- the default state, which is almost
    every calculation.

    File order is load-bearing both ways: :func:`warm_list` hands the
    declaration builder its rows in it (so ``Job.warm`` is stable), and the
    run script its banner/test order.
    """
    path = _rules_path(engine)
    if base_dir is not None:
        local = Path(base_dir) / FILENAME
        if local.is_file():
            path = local
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
    resumes: List[Tuple[str, bool]] = []
    along: List[Tuple[str, Tuple[str, ...]]] = []
    scratch: List[Tuple[str, Tuple[str, ...]]] = []
    seen_suffixes: Dict[str, str] = {}
    for name, body in raw.items():
        if (not isinstance(body, dict)
                or set(body) - {"file", "resumes", "along", "scratch"}):
            raise WarmFilesError(
                f"{path}: section [{name}] must hold only [[{name}.file]] "
                f"rows and the section-level facts, `resumes`, `along` and "
                f"`scratch` (job-contracts.md 4.2a).")
        if "scratch" in body:
            said = body["scratch"]
            if (not isinstance(said, list)
                    or not all(isinstance(x, str) and x.startswith(".")
                               for x in said)):
                raise WarmFilesError(
                    f"{path}: [{name}] scratch = {said!r} -- a list of "
                    f"suffixes, the engine's scratch at a run of this kind: "
                    f"never carried, never taken over "
                    f"(engines/transport.md 2a.11).")
            scratch.append((name, tuple(said)))
        if "resumes" in body:
            if not isinstance(body["resumes"], bool):
                raise WarmFilesError(
                    f"{path}: [{name}] resumes = {body['resumes']!r} -- a "
                    f"true or a false, whether a re-run of this kind of run "
                    f"continues from what the last one left.")
            resumes.append((name, body["resumes"]))
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
        if "along" in body:
            said = body["along"]
            own_rows = {r.suffix for r in rows}
            if (not isinstance(said, list)
                    or not all(isinstance(x, str) and x in own_rows
                               for x in said)):
                raise WarmFilesError(
                    f"{path}: [{name}] along = {said!r} -- a list of this "
                    f"section's own suffixes, what a sweep hands from point "
                    f"to point (engines/transport.md 2a.11).")
            along.append((name, tuple(said)))
        sections.append((name, tuple(rows)))
    if not any(name == "base" for name, _ in sections):
        raise WarmFilesError(
            f"{path}: no [base] section.  Every engine has one -- it may "
            f"be empty, but its absence reads as a truncated file.")
    return WarmFilesDoc(engine=engine, sections=tuple(sections),
                        path=str(path), resumes=tuple(resumes),
                        along=tuple(along), scratch=tuple(scratch))
