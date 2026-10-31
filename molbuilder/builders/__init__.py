"""molbuilder.builders -- structure-building verbs, and the one build door.

Hosts ``backends/`` (the per-tool nucleic-acid builders: ``_amber.py``,
``_rdkit.py``, ``_threedna.py``) and :func:`build`, the door every builder is
reached through -- the build route (`/api/build/molecule`) and the command
line alike: the builder its kind names, run on what was asked, and the check
after it (`validation.builders.check_built`; `science/validation.md` § 4.1
R5) -- so a builder's findings reach every surface, and no surface assembles
the build its own way.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Dict, List, Optional, Tuple

#: What a nucleic-acid builder takes beside its sequence.
_NUCLEIC = ("backend", "form", "terminal", "add_hydrogens",
            "protonate_phosphates")


@dataclass(frozen=True)
class Generator:
    """One builder kind: its builder, named ``module:function`` and looked
    up at each build (so this package imports no builder -- the nucleic one
    imports this package's backends); what it takes beside its input;
    whether it is built by a nucleic-acid backend -- a backend, a helical
    form and a terminal state -- and whether it answers which backend built
    it (``return_backend``)."""
    builder: str
    asks: Tuple[str, ...] = ()
    nucleic: bool = False
    names_backend: bool = False


#: THE BUILDER KINDS, the panels' five generators (`web/tabs.md`).
GENERATORS: Dict[str, Generator] = {
    "peptide": Generator("molbuilder.peptide:build_peptide"),
    "dna":     Generator("molbuilder.nucleic:build_dna",
                         _NUCLEIC + ("relax_clashes",), nucleic=True),
    "rna":     Generator("molbuilder.nucleic:build_rna", _NUCLEIC,
                         nucleic=True),
    "smiles":  Generator("molbuilder.smiles:build_from_smiles",
                         names_backend=True),
    "name":    Generator("molbuilder.pubchem:build_from_name",
                         names_backend=True),
}


@dataclass(frozen=True)
class Build:
    """A build's answer: the structure, the backend that built it (``None``
    where the kind names none), and what the build says of it."""
    structure: Any
    backend: Optional[str]
    findings: List[Any]


def build(kind: str, text: str, *, title: Optional[str] = None,
          **asked) -> Build:
    """Build ``kind`` from ``text``: ``auto`` resolved once, before the build
    (`backends.resolved_backend`), the builder run on what was ``asked`` --
    a value left ``None`` not passed, the builder's own default standing --
    and the check after it, told the form and terminal the build used (the
    kind's own, `backends.DEFAULT_FORM`, `DEFAULT_TERMINAL`, when none was
    asked).  An ask the kind does not take is refused, naming it."""
    from ..validation.builders import Asked, check_built
    from .backends import DEFAULT_FORM, DEFAULT_TERMINAL, resolved_backend
    gen = GENERATORS.get(kind)
    if gen is None:
        raise ValueError(f"unknown builder kind {kind!r}; valid: "
                         f"{sorted(GENERATORS)}")
    stray = sorted(set(asked) - set(gen.asks))
    if stray:
        raise ValueError(f"a {kind} build takes {list(gen.asks)}, not "
                         f"{stray}")
    asked = {k: v for k, v in asked.items() if v is not None}
    module, name = gen.builder.split(":")
    builder = getattr(import_module(module), name)
    used: Optional[str] = None
    if gen.nucleic:
        used = resolved_backend(asked.get("backend", "auto"))
        if used is not None:
            asked["backend"] = used
    if gen.names_backend:
        struct, used = builder(text, title=title, return_backend=True)
    else:
        struct = builder(text, title=title, **asked)
    check = (Asked(text=text, backend=used,
                   form=asked.get("form") or DEFAULT_FORM[kind],
                   terminal=asked.get("terminal") or DEFAULT_TERMINAL,
                   relax_clashes=bool(asked.get("relax_clashes", False)),
                   defaults=tuple(k for k in ("form", "terminal")
                                  if not asked.get(k)))
             if gen.nucleic else Asked(text=text, backend=used))
    return Build(struct, used, check_built(struct, kind=kind, asked=check))
