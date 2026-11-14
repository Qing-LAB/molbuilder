"""Pseudopotential files for SIESTA: validating them, and copying them in.

SIESTA needs one ``.psml`` file per species label in
the structure, named for it (``<label>.psml``).  These files carry
metadata in their header that MUST match the calculation config:

  * ``element``         -- which element this pseudo describes
  * XC functional       -- the same family the user picked in
                           ``cfg.xc_authors`` (a PBE pseudo on an
                           LDA calc gives silently-wrong bond lengths)
  * Relativistic level  -- SR (scalar) for most cases; FR (fully-
                           relativistic) only when spin-orbit matters
  * Generator           -- usually PseudoDojo / ATOM / Hamann's ONCVPSP

Functions here:

  * :func:`parse_psml_header` -- read the XML header of a .psml file
    and return a ``PsmlInfo`` dataclass with the canonical fields.
  * :func:`scan_psml_directory` -- walk a directory, return
    ``{file name: PsmlInfo}`` for all parseable .psml files.
  * :func:`labels_in_directory` -- the species labels a whole folder's
    files are for, as `molbuilder pseudo check DIR` asks them.
  * :func:`check_coverage` -- given the structure's species labels and
    the directory their files are read from, or each label's own, return a
    status entry per label: present / missing / misnamed / defective /
    mismatched-XC / mismatched-relativistic, and the set's generators.  The SIESTA
    validator (``molbuilder.validation.siesta``) calls this during
    preflight + render so the user sees a clear "no .psml file for Fe"
    or "Fe.psml is for LDA but you picked GGA" Issue message
    before they run SIESTA.
  * :func:`copy_pseudopotentials` -- copy the files a deck names beside it.

Format reference: PSML 1.1 spec
(https://siesta-project.org/SIESTA_MATERIAL/Pseudos/Code/psml-1.1.pdf).
PseudoDojo PSML files we tested against (2024 release).
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union
import re
import shutil
import sys
import xml.etree.ElementTree as ET


#: Where a calculation's staged pseudopotentials live, relative to it
#: (roadmap 7.10 M6: grouped in one folder rather than N loose `<label>.psml`).
from .runfiles import PSEUDO_DIR as PSEUDO_DIRNAME  # noqa: E402 -- the catalogue's name
from .runfiles import (PSEUDO_FILE, RunFileError, fixed_fields,  # noqa: E402
                       fixed_name)
from .structure import split_species_label  # noqa: E402 -- a label's name


#: THE FOLDER AS A PREP WILL LEAVE IT, while one is being planned -- a
#: `jobset.planned.Plan`, read through ``is_file``, ``glob`` and
#: ``source_of`` -- or ``None``, and the disk answers.  `prep` sets it for
#: the steps it plans (`job-system.md` § 5.0: nothing is written until the
#: whole plan stands), so every reader of a calculation's pseudopotentials --
#: the settings gate at each deck -- reads the files the plan will put
#: there.  A context variable, as the gate's report stream is
#: (`validation.REPORT_STREAM`): two preps at once each see their own.
PLANNED: "ContextVar" = ContextVar("molbuilder.pseudos.planned",
                                   default=None)


def _is_file(path) -> bool:
    view = PLANNED.get()
    return view.is_file(path) if view is not None else Path(path).is_file()


def where_bytes_are(path) -> Path:
    """The file that holds ``path``'s bytes now: the path itself, or --
    while a prep is planned (:data:`PLANNED`) -- what the plan will copy
    or move there."""
    view = PLANNED.get()
    src = view.source_of(path) if view is not None else None
    return Path(src) if src is not None else Path(path)


def psml_sources(labels, *, dest_dir=None,
                 library=None) -> "Dict[str, Optional[Path]]":
    """Which directory each species label's `.psml` is read from, or
    ``None``.

    **The calculation's own folder first, then ``library``** -- the rule prep
    applies when it puts the files in place (*the folder wins; only what is
    missing is fetched*, `jobset/prep.py`), and the rule every check asks, so
    that a check reads the files the run will open.  The library is the
    template's ``psml_lib`` on SIESTA and the cited run's folder on
    transport.  Keyed by LABEL, verbatim: SIESTA opens `<label>.psml`
    (`check_coverage`).

    **No side effects** -- which is why this is here and not
    `siesta.seam.pseudo_dir`, whose job is to CREATE the folder and migrate
    root strays into it.  A read-only check cannot call a function that
    writes, and copying its rule instead would let the two disagree about
    where a file is.

    In the folder, ``pseudos/`` before the root.  Both are real at different
    moments -- `molbuilder describe` and a travelled bundle leave `<label>.psml` at the root,
    and the first `prep` moves them into ``pseudos/``, keeping what is
    already there.  Asking only the root would refuse a calculation whose
    pseudopotentials are already in place.
    """
    places: List[Path] = []
    if dest_dir is not None:
        places += [Path(dest_dir) / PSEUDO_DIRNAME, Path(dest_dir)]
    if library is not None:
        places.append(Path(library))
    out: Dict[str, Optional[Path]] = {}
    for el in labels:
        key = str(el).strip()
        name = file_named_for(key)
        if key not in out:
            out[key] = (None if name is None else
                        next((d for d in places if _is_file(d / name)), None))
    return out


def file_named_for(label: str) -> Optional[str]:
    """The file SIESTA opens for species ``label`` -- ``<label>.psml``
    (:data:`runfiles.PSEUDO_FILE`) -- or ``None`` for a label no file can be
    named for: not an element and an optional index (``au``, ``Au_1``),
    which `check_coverage` says rather than raising."""
    try:
        return fixed_name(PSEUDO_FILE, species=label)
    except RunFileError:
        return None


#: The library's conventional name: the tree's own ``pseudopotential``
#: folder, measured from the tree root (`job-contracts.md` § 2.5a) -- what
#: an unset ``psml_lib`` is told to use, and named only when it covers the
#: structure (plan § 5w K20).
CONVENTIONAL_LIBRARY = "pseudopotential"


class PsmlLibError(ValueError):
    """A ``psml_lib`` spelling the rule cannot answer -- a ``./`` one, one
    with a ``..`` in it, or one that lands outside the projects tree,
    absolute or through a link.  The message teaches the rule; callers
    surface it verbatim in their own refusal type."""


def resolve_psml_lib(raw: str, *,
                     base: Optional[Path] = None,
                     dest_dir: Optional[Path] = None) -> Path:
    """Resolve ``cfg.psml_lib``: **a path inside the projects tree.**

    The rule is `job-contracts.md` § 2.5a, ONE line (user, 2026-08-28: the
    library always lives inside the project tree, so psml
    paths speak the same language as every sidebar path):

      * a RELATIVE path (``pseudopotential``, ``shared/psml``) is
        measured from the TREE ROOT -- the tree the calculation lives
        in, walked up from ``dest_dir``; the server's own
        ``projects_root()`` when there is no calculation yet (``base``
        overrides it, for tests and other trees);
      * an ABSOLUTE path is accepted as a convenience spelling of the
        same fact; either spelling must lie INSIDE that tree, and is
        refused otherwise -- and a ``..`` is refused in any spelling, inside
        or out, before anything resolves it;
      * the ``./`` / ``../`` spellings are refused: pseudopotentials
        already beside the calculation are used without this field (prep
        adopts them).

    Raises :class:`PsmlLibError` for a spelling the rule cannot answer.
    Returns the absolute path the fence resolved (`projects.at`), whether
    or not it exists: this function answers *"which folder does this name?"* and
    the caller reports a miss (``.is_dir()``) with
    :func:`describe_psml_anchor`.
    """
    from .projects import (OutsideRoot, PROJECTS_ROOT_NAME, at,
                           find_projects_root, projects_root)
    p = Path(raw).expanduser()

    if raw.startswith("./") or raw.startswith("../"):
        raise PsmlLibError(
            f"{raw!r}: the ./ spelling is retired -- `psml_lib` is a "
            f"path inside the {PROJECTS_ROOT_NAME}/ tree, measured from "
            f"the tree root (write `pseudopotential`, not "
            f"`./pseudopotential`).  Pseudopotentials already beside "
            f"the calculation are used without this field.")

    tree: Optional[Path] = None
    if dest_dir is not None:
        tree = find_projects_root(dest_dir)
    if tree is None:
        tree = base if base is not None else projects_root()

    # INSIDE THE TREE, either spelling -- an absolute one, or a relative one
    # measured from the root -- and no `..` in either: the one fence decides
    # (`projects.contain`).
    try:
        named = at(p, tree)
    except OutsideRoot as exc:
        # The fence refuses a `..` in any spelling, before it resolves one,
        # so a `..` that stays inside is refused too -- said as what it is.
        said = (f"{raw!r} is outside the {PROJECTS_ROOT_NAME}/ tree ({tree})"
                if exc.outside else f"{raw!r} is refused: {exc.reason}")
        raise PsmlLibError(
            f"{said}.  `psml_lib` always lives inside the tree -- "
            f"copy the library in (e.g. {tree}/pseudopotential) and "
            f"write the tree-relative name.") from exc
    return named


def describe_psml_anchor(raw: str, *, dest_dir: Optional[Path] = None) -> str:
    """One sentence for refusals: where this spelling looked (§ 2.5a).

    Lives beside the rule so every surface that refuses over a missing
    library -- ``prep``, deck emission, the browser preflight -- says
    the same thing, in the rule's own words.
    """
    from .projects import PROJECTS_ROOT_NAME
    try:
        landed = resolve_psml_lib(raw, dest_dir=dest_dir)
    except PsmlLibError as exc:
        return str(exc)
    p = Path(raw).expanduser()
    if p.is_absolute():
        return (f"{raw} is an absolute path inside the tree, so that is "
                f"where it looked: {landed}.")
    out = (f"{raw!r} is measured from the {PROJECTS_ROOT_NAME}/ tree "
           f"root, so it looked in {landed}.")
    if raw.split("/", 1)[0] == PROJECTS_ROOT_NAME:
        out += (f"  Note the doubled {PROJECTS_ROOT_NAME}/: paths are "
                f"measured from the tree root already.  Drop the "
                f"leading {PROJECTS_ROOT_NAME}/.")
    return out


@dataclass(frozen=True)
class PsmlInfo:
    """Canonical metadata extracted from a .psml file's header."""
    path:                Path
    element:             str             # "Fe", "C", "H", ...
    atomic_number:       int             # integer Z, the file's atomic-number
    xc_family:           str             # "GGA" | "LDA" | "VDW" | "unknown"
    xc_authors:          str             # "PBE" | "PBEsol" | "CA" | etc.
    relativistic:        str             # "no" | "scalar" | "spin-orbit"
    generator:           str             # "ONCVPSP" / "ATOM" / "Hamann" / etc.
    valence_config:      str             # its shells, "2s2 2p2"
    #: ---- what the file REQUIRES OF THE CALCULATION (layer 2) ----------
    #: `science/pseudopotentials.md` § 2a: a pseudo does not only have to be
    #: sound, it STATES things the calculation must satisfy.  These are read
    #: here and compared by `validation/siesta.py`, never by this module --
    #: parsing a requirement and judging a configuration are two jobs.
    suggested_mesh_ry:   Optional[float] # the `normal` hint; the threshold
    cutoff_hints_ry:     Dict[str, float] = field(default_factory=dict)
                                         # {"low","normal","high"} as stated,
                                         # empty when the file states none
                                         # (v0.4 files do not; the eleven
                                         # elements v0.5 re-generated do)
    null_channels:       List[str] = field(default_factory=list)
                                         # l-letters (s/p/d/f) whose ENTIRE
                                         # Kleinman-Bylander channel has
                                         # ekb~=0 -- a defective/incomplete
                                         # pseudo (the BDT S.psml had a dead
                                         # 'p' channel; triggers propor
                                         # IMAX=0 AND gives wrong physics).
    partial_projector_channels: List[str] = field(default_factory=list)
                                         # VALENCE l-channels where SOME but
                                         # not all projectors are zero -- the
                                         # channel survives at less than the
                                         # completeness the generator asked
                                         # for.  Degraded, not absent: WARN.
                                         # (v0.5: Bi, Pb, Po, Rn, Te, Tl all
                                         # have p1 real and p2 zeroed.)
    semilocal_only_channels: List[str] = field(default_factory=list)
                                         # VALENCE l-channels whose KB
                                         # projectors are all ~zero but which
                                         # an <slps> semilocal block carries.
                                         # Valid PSML -- and SIESTA was
                                         # observed not to rebuild the channel
                                         # from it (S, PseudoDojo v0.5,
                                         # 2026-06).  § 2a.2.
    parse_warnings:      List[str] = field(default_factory=list)


# PSML uses a namespace for all elements (``http://esl.cecam.org/PSML/ns/1.1``).
# ElementTree's strict namespace handling means we have to match on the local
# tag (the part after the }).
def _localtag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _findall_local(root, local_name: str):
    """Find all descendants with local tag == ``local_name``."""
    return [e for e in root.iter() if _localtag(e.tag) == local_name]


def _first_local(root, local_name: str):
    matches = _findall_local(root, local_name)
    return matches[0] if matches else None


# Map libxc functional ids (the integer codes) to (family, authors).
# Truncated to the families we'd realistically encounter; an unknown
# id leaves xc_family/xc_authors as "unknown" so the caller can
# warn explicitly rather than mis-classify.
_LIBXC_MAP = {
    # GGA exchange + correlation (separate components in libxc)
    101: ("GGA", "PBE"),    # XC_GGA_X_PBE
    130: ("GGA", "PBE"),    # XC_GGA_C_PBE
    116: ("GGA", "PBEsol"), # XC_GGA_X_PBE_SOL
    133: ("GGA", "PBEsol"), # XC_GGA_C_PBE_SOL
    106: ("GGA", "BLYP"),   # XC_GGA_X_B88
    131: ("GGA", "BLYP"),   # XC_GGA_C_LYP
    117: ("GGA", "RPBE"),   # XC_GGA_X_RPBE (Hammer 1999).  NOTE: 117 is RPBE,
                            # NOT revPBE (revPBE = XC_GGA_X_PBE_R = 102) -- these
                            # are physically distinct functionals.
    # LDA (both 9 = Perdew-Zunger and 12 = Perdew-Wang parameterise the same
    # Ceperley-Alder uniform-gas correlation data -> family author "CA").
    1:   ("LDA", "CA"),     # XC_LDA_X (Slater/Dirac exchange)
    9:   ("LDA", "CA"),     # XC_LDA_C_PZ (Perdew-Zunger)
    12:  ("LDA", "CA"),     # XC_LDA_C_PW (Perdew-Wang)
    11:  ("LDA", "CA"),     # XC_LDA_C_OB_PZ (Ortiz-Ballone form of PZ) -- an LDA
                            # correlation, NOT a vdW-DF exchange.
}


def parse_psml_header(path: Path) -> PsmlInfo:
    """Parse a single .psml file and return the canonical metadata.

    Reads what a PSML file holds.  A file that does not hold it is said so: no
    element, and a warning, never an exception or a value made up.  XC family
    and authors are ``"unknown"`` when the libxc id isn't in our table (rare
    functionals).
    """
    path = Path(path)
    warnings: List[str] = []
    try:
        # WHERE THE BYTES ARE NOW (:func:`where_bytes_are`): a file a planned
        # prep will copy here is read at its source and reported here.
        root = ET.parse(str(where_bytes_are(path))).getroot()
    except (ET.ParseError, OSError) as exc:
        return PsmlInfo(
            path=path, element="", atomic_number=0,
            xc_family="unknown", xc_authors="unknown",
            relativistic="unknown", generator="unknown",
            valence_config="", suggested_mesh_ry=None,
            cutoff_hints_ry={}, null_channels=[],
            semilocal_only_channels=[], partial_projector_channels=[],
            parse_warnings=[f"could not parse XML: {exc}"],
        )

    # ----- element + Z ----------------------------------------------
    # PSML states them on <pseudo-atom-spec>; a file with none reads as no
    # element, and is said so (C6).  ``_first_local`` returns an empty Element
    # for a tag with no children, which is falsy: hence ``is not None``.
    # `atomic-number` is the element's Z; `z-pseudo` beside it is the valence
    # count (Fe has Z=26 and z-pseudo=16).
    spec = _first_local(root, "pseudo-atom-spec")
    spec_says = spec.attrib if spec is not None else {}
    element = (spec_says.get("atomic-label") or "").strip().capitalize()
    z_str = spec_says.get("atomic-number") or ""
    try:
        atomic_number = int(float(z_str)) if z_str else 0
    except ValueError:
        atomic_number = 0
        warnings.append(f"unparseable atomic-number: {z_str!r}")
    if not element:
        warnings.append("no <pseudo-atom-spec atomic-label=...>: not a PSML "
                        "file this reader knows")

    # ----- XC functional -------------------------------------------
    xc_family, xc_authors = "unknown", "unknown"
    # PSML states it once:
    #   <libxc-info number-of-functionals="2">
    #     <functional name="..." type="exchange"    id="101"/>
    #     <functional name="..." type="correlation" id="130"/>
    #   </libxc-info>
    libxc_ids: List[int] = []
    for el in _findall_local(root, "functional"):
        v = el.attrib.get("id")
        if v and v.isdigit():
            libxc_ids.append(int(v))
    for lid in libxc_ids:
        if lid in _LIBXC_MAP:
            fam, auth = _LIBXC_MAP[lid]
            # Prefer the FIRST hit (exchange usually comes before
            # correlation; both give the same family/authors).
            if xc_family == "unknown":
                xc_family, xc_authors = fam, auth
                break
    # ----- relativistic --------------------------------------------
    # PSML's three values, on <pseudo-atom-spec>; anything else is unknown.
    relativistic = {"dirac": "spin-orbit", "scalar": "scalar",
                    "no": "no"}.get(
        (spec_says.get("relativity") or "").lower(), "unknown")

    # ----- generator -----------------------------------------------
    provenance = _first_local(root, "provenance")
    generator = ((provenance.attrib.get("creator") or "").strip()
                 if provenance is not None else "") or "unknown"

    # ----- valence configuration -----------------------------------
    vc_el = _first_local(root, "valence-configuration")
    valence_config = (" ".join(
        f'{sh.attrib.get("n", "")}{sh.attrib.get("l", "")}'
        f'{sh.attrib.get("occupation", "")}' for sh in vc_el)
        if vc_el is not None else "")

    # ----- nonlocal-projector value validation ---------------------
    # Each <proj l=".." ekb=".."> is a Kleinman-Bylander projector; the
    # KB energy ``ekb`` IS the projector's strength (V_nl = Σ ekb·|p><p|),
    # so ekb=0 means the KB projector contributes NOTHING for that l.
    #
    # BUT ekb=0 alone does NOT mean the channel is dead: ONCVPSP-4.0.1 /
    # psml-4.0.1 pseudos write the pre-computed nonlocal <proj> block as
    # zero PLACEHOLDERS for channels that are instead carried by the
    # <semilocal-potentials> <slps l=".."> block, from which SIESTA
    # rebuilds the KB projector at read time.  Standard, PseudoDojo-
    # validated pseudos for I, Xe, Rb, Ba do exactly this for their p
    # channel -- flagging them as "dead" would be a FALSE POSITIVE that
    # ERROR-blocks those common elements.
    #
    # A channel is genuinely null ONLY if EVERY <proj> for that l is
    # ~zero AND there is NO <slps l> semilocal potential for it (nothing
    # for SIESTA to rebuild from).  (A channel chosen as the LOCAL
    # potential has no <proj> at all -- absent, not present-but-zero --
    # and is likewise not flagged.)
    _EKB_NULL = 1e-6
    # l-letters that carry a semilocal potential (the authoritative
    # representation when the nonlocal block is a zero placeholder).
    semilocal_ls = {
        (s.attrib.get("l") or "").strip().lower()
        for s in _findall_local(root, "slps")
        if (s.attrib.get("l") or "").strip()
    }
    def _as_float(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    #: ---- the file's REQUIREMENTS (layer 2, § 2a.1) ---------------------
    #: PseudoDojo states its recommended mesh cutoffs on an <annotation> as
    #: `cutoff_hint_low` / `_normal` / `_high` (Ry).  Only the eleven elements
    #: v0.5 re-generated carry them; a v0.4 file states nothing, and stating
    #: nothing is a real answer -- it must not be read as "any cutoff will do"
    #: (§ 2a.1: silence abstains, it does not vote for a lower number).
    cutoff_hints: Dict[str, float] = {}
    for ann in _findall_local(root, "annotation"):
        for k, v in ann.attrib.items():
            if k.startswith("cutoff_hint_"):
                f = _as_float(v)
                if f is not None:
                    cutoff_hints[k[len("cutoff_hint_"):]] = f

    #: Which l-channels are OCCUPIED, from the file's own valence
    #: configuration (`<shell n=.. l=.. occupation=..>`).  Read here rather
    #: than assumed from the element, because the file is the authority on
    #: what IT put in valence.
    valence_ls = {
        (sh.attrib.get("l") or "").strip().lower()
        for sh in _findall_local(root, "shell")
        if (sh.attrib.get("l") or "").strip()
        and _as_float(sh.attrib.get("occupation")) not in (None, 0.0)
    }
    proj_by_l: Dict[str, List[float]] = {}
    for pr in _findall_local(root, "proj"):
        l_letter = (pr.attrib.get("l") or "").strip().lower()
        if not l_letter:
            continue
        try:
            ekb = abs(float(pr.attrib.get("ekb", "")))
        except (TypeError, ValueError):
            continue
        proj_by_l.setdefault(l_letter, []).append(ekb)
    #: Channels with nothing in their KB projectors, split by whether the
    #: file offers a fallback.  BOTH are read from one pass, because they are
    #: the same measurement asked two ways -- and keeping them apart is the
    #: § 2a.2 distinction: `null` is a file that is BROKEN, `semilocal_only`
    #: is a file that is VALID and may still be unusable.
    _dead = {l for l, eks in proj_by_l.items()
             if eks and all(e < _EKB_NULL for e in eks)}
    null_channels = sorted(_dead - semilocal_ls)
    #: Valence only.  A dead channel the semilocal block carries matters when
    #: the channel is OCCUPIED -- sulfur's p (3s2 3p4) is what made the run
    #: wrong; an unoccupied one is a representation choice with nothing
    #: riding on it.
    semilocal_only_channels = sorted(
        (_dead & semilocal_ls) & valence_ls)
    #: SOME projectors zero, not all -- the channel is still there, at less
    #: than the completeness the generator's own input asked for.  Every
    #: v0.5 element requests `nproj=2` for p; six of them shipped one real
    #: and one empty, which costs transferability rather than the channel.
    _partial = {l for l, eks in proj_by_l.items()
                if eks and any(e < _EKB_NULL for e in eks)
                and not all(e < _EKB_NULL for e in eks)}
    partial_projector_channels = sorted(_partial & valence_ls)

    return PsmlInfo(
        path=path, element=element, atomic_number=atomic_number,
        xc_family=xc_family, xc_authors=xc_authors,
        relativistic=relativistic, generator=generator,
        valence_config=valence_config,
        suggested_mesh_ry=cutoff_hints.get("normal"),
        cutoff_hints_ry=cutoff_hints,
        semilocal_only_channels=semilocal_only_channels,
        partial_projector_channels=partial_projector_channels,
        null_channels=null_channels,
        parse_warnings=warnings,
    )


def scan_psml_directory(directory: Path) -> Dict[str, PsmlInfo]:
    """Walk ``directory`` (non-recursive) and return ``{file name:
    PsmlInfo}`` for every .psml file that declares an element -- as a prep
    will leave it, while one is being planned (:data:`PLANNED`).

    BY FILE NAME, the name SIESTA opens (`science/pseudopotentials.md`
    § C1a): two files of one element -- ``Au1.psml`` and ``Au2.psml``, or
    ``Fe.psml`` and ``Fe_pbe.psml`` -- are two entries, and which one a
    species reads is its name's answer, never the element's.
    """
    out: Dict[str, PsmlInfo] = {}
    for p in _psml_paths(directory):
        info = parse_psml_header(p)
        if not info.element:
            continue
        out[p.name] = info
    return out


def _psml_paths(directory: Path) -> List[Path]:
    """Every ``.psml`` file in ``directory`` (non-recursive), read or not --
    as a prep will leave it, while one is being planned (:data:`PLANNED`)."""
    directory = Path(directory)
    view = PLANNED.get()
    if view is not None:
        listing = view.glob(directory, "*")
    elif directory.is_dir():
        listing = sorted(directory.iterdir())
    else:
        return []
    return [p for p in listing if p.suffix.lower() == ".psml"]


def labels_in_directory(directory: Path) -> List[str]:
    """The species labels a folder's pseudopotentials are for, as a check
    of the whole folder asks them (`molbuilder pseudo check DIR`): each
    file's label where it is named for one (``<label>.psml``,
    :data:`runfiles.PSEUDO_FILE`), read or not -- so one that does not parse
    is screened (C6) -- and the element it declares where it is not, so a
    file SIESTA never opens is said misnamed (C1a) where its element has no
    file of its own; one beside its element's own (``Fe_pbe.psml`` beside
    ``Fe.psml``) is opened by no species and not screened.  Sorted; ``[]``
    for a folder with none."""
    declared = scan_psml_directory(directory)
    out = set()
    for p in _psml_paths(directory):
        named = fixed_fields(PSEUDO_FILE, p.name)
        if named:
            out.add(named["species"])
        elif p.name in declared:
            out.add(declared[p.name].element)
    return sorted(out)


@dataclass
class CoverageEntry:
    """One row of a coverage report: a species label's pass / warn / fail
    (``element`` holds the label, verbatim -- ``Au1``)."""
    element:  str
    status:   str       # "ok" | "missing" | "misnamed" |
                        # "dead_projector" | "semilocal_only" |
                        # "partial_projectors" | "xc_family_mismatch" |
                        # "xc_mismatch" | "relativistic_mismatch" |
                        # "generator_mismatch" | "parse_warning"
    message:  str
    path:     Optional[Path] = None


#: The `CoverageEntry.status` values that BLOCK — a run with any of these
#: cannot be correct: a pseudo is absent (SIESTA won't start), a valence
#: channel is physically missing (wrong Hamiltonian), the XC *family* is wrong
#: (silently-wrong energies), or a valence channel is present and states a
#: strength of ZERO (``semilocal_only`` — the same silently-wrong-energies
#: case, user ruling 2026-09-03 after PseudoDojo v0.5's sulfur ran with no
#: p channel).  This is the SINGLE source of truth for "which statuses are
#: ERROR"; ``misnamed`` is one (the file for a species exists and is healthy
#: but is not called `<label>.psml`, so SIESTA never opens it -- a certain
#: start-up failure, said at configuration time -- or `<label>.psml`
#: declares another element, which SIESTA would read for it).
#: The SIESTA preflight
#: (``validation.siesta._check_siesta_pseudo_coverage``) and the CLI
#: (``cli.cmd_pseudo_check``) both consume it so the two surfaces cannot
#: drift.
#: Everything else — ``xc_mismatch`` (same-family author diff),
#: ``relativistic_mismatch``, ``generator_mismatch``, ``parse_warning`` — is
#: advisory (WARN); ``ok`` is a silent pass.
ERROR_STATUSES = frozenset({"missing", "misnamed", "dead_projector",
                            "xc_family_mismatch", "semilocal_only"})


#: XC authors -> the FAMILY a pseudopotential must belong to.
#:
#: One home, for the same reason ``ERROR_STATUSES`` has one: it decides an
#: ERROR-severity verdict (``xc_family_mismatch`` — *"never physically
#: correct, energies/forces silently wrong"*), so two copies of it are two
#: opinions about whether a run may proceed.
_XC_FAMILIES = {
    "GGA": ("pbe", "pbesol", "blyp", "revpbe", "rpbe"),
    "LDA": ("ca", "pz", "pw"),
    "VDW": ("drsll", "lmkll"),
}


def expected_xc_family(xc_authors: Optional[str]) -> Optional[str]:
    """The XC family ``xc_authors`` implies, or ``None`` if it names none.

    ``None`` is a real answer and not a failure: an unrecognised or empty
    authors string means *do not compare families*, which is what lets a
    curated set through rather than blocking on a name this table has not been
    taught yet.
    """
    a = (xc_authors or "").strip().lower()
    for family, authors in _XC_FAMILIES.items():
        if a in authors:
            return family
    return None


def check_coverage(labels: Iterable[str],
                    directory: Union[Path, Mapping[str, Path]],
                    *,
                    expected_xc_family: Optional[str] = None,
                    expected_xc_authors: Optional[str] = None,
                    expected_relativistic: str = "scalar",
                    ) -> List[CoverageEntry]:
    """Given the species labels the structure needs (no duplicates
    required; we de-duplicate) and the directory their pseudopotentials are
    read from -- or each label's own, for a structure whose files come from
    two places (the calculation's folder first, then the library,
    :func:`psml_sources`) -- return each label's status, read from the file
    SIESTA opens for it, ``<label>.psml`` (`science/pseudopotentials.md`
    § C1, C1a).  THE WHOLE STRUCTURE IN ONE CALL: a file another label of
    the structure opens is that label's, and the set's generators are
    compared across every file read.

    Status values:
      * ``"ok"`` -- pseudo present, XC + relativistic match (or
        expectations weren't supplied).
      * ``"missing"`` -- no .psml file for this label, or a label no file
        can be named for.  Hard fail for SIESTA: the run won't start.
      * ``"misnamed"`` -- the label's element is in a file no label of
        the structure opens, which SIESTA never reads for it; or
        ``<label>.psml`` declares another element, which SIESTA would read
        for it.
      * ``"xc_family_mismatch"`` -- pseudo's XC FAMILY differs from the
        calc's (e.g. an LDA pseudo in a GGA run).  ERROR-severity: never
        physically correct, energies/forces silently wrong.
      * ``"xc_mismatch"`` -- same family, different authors (PBE vs
        PBEsol); WARN-severity, a minor ~1-2 kcal/mol difference.
      * ``"relativistic_mismatch"`` -- e.g. user picked an SR pseudo
        for a fully-relativistic run; WARN-severity.
      * ``"parse_warning"`` -- the .psml file is present but missing
        metadata (C6); assume it's OK and let SIESTA report any issues
        at startup.
    """
    from .chemistry import is_atom
    # BY LABEL, deliberately: SIESTA reads `<label>.psml`, so a species
    # written `Au1` really does need `Au1.psml` and is not covered by
    # `Au.psml`.  Verbatim -- no folding (`model/chemistry.md` § 3).  The
    # file THIS label opens is the one read: two labels of one element are
    # two files (`science/pseudopotentials.md` § C1, C1a).
    labels = list(dict.fromkeys(str(el).strip() for el in labels))
    where = ({k: Path(directory[k]) for k in labels}
             if isinstance(directory, Mapping)
             else dict.fromkeys(labels, Path(directory)))
    opens = {key: file_named_for(key) for key in labels}
    named = {n for n in opens.values() if n is not None}
    folders: Dict[Path, Dict[str, PsmlInfo]] = {}     # each folder read once
    out: List[CoverageEntry] = []
    read: Dict[str, PsmlInfo] = {}
    for key in labels:
        if opens[key] is None:
            out.append(CoverageEntry(
                element=key, status="missing",
                message=(f"{key!r} is not a species label a pseudopotential "
                         f"can be named for -- SIESTA opens <label>.psml, "
                         f"and a label is an element and an optional index "
                         f"(Au, Au1).  Rename the "
                         f"species."),
                path=None))
            continue
        name = split_species_label(key)[0]
        d = where[key]
        if d not in folders:
            folders[d] = scan_psml_directory(d)
        folder = folders[d]
        opened = d / opens[key]
        info = folder.get(opened.name) or (
            parse_psml_header(opened) if _is_file(opened) else None)
        if info is None:
            # A FILE NO LABEL OPENS declaring this label's element -- a
            # correct gold pseudopotential saved as `gold.psml` -- is never
            # read for it: SIESTA opens `<label>.psml` and has no search
            # path, so it refuses to start.  A file named for another label
            # of the structure is that label's, wherever it is read from.
            others = sorted(f for f, i in folder.items()
                            if is_atom(key, i.element) and f not in named)
            if others:
                out.append(CoverageEntry(
                    element=key, status="misnamed",
                    message=(f"{' and '.join(others)} "
                             f"declare{'s' if len(others) == 1 else ''} "
                             f"element {name}, but SIESTA opens "
                             f"{opened.name} for species {key} and has no "
                             f"search path -- so it never reads "
                             f"{'that file' if len(others) == 1 else 'them'} "
                             f"for {key}, and the run will refuse to start.  "
                             f"Save {name}'s pseudopotential as "
                             f"{opened.name}."),
                    path=d / others[0],
                ))
            else:
                out.append(CoverageEntry(
                    element=key, status="missing",
                    message=(f"no .psml file for {key} in {d} -- SIESTA "
                             f"opens <label>.psml in the folder it runs "
                             f"from and has no search path, so it will "
                             f"refuse to start.  Put {key}.psml in {d}, or "
                             f"point `psml_lib` at a library that has it "
                             f"(http://www.pseudo-dojo.org, PSML format, "
                             f"the functional the run uses)."),
                    path=None,
                ))
            continue
        # THE RIGHT NAME, ANOTHER ELEMENT: SIESTA reads it for this species.
        # A file that declares none is C6's parse warning, below.
        if info.element and not is_atom(key, info.element):
            out.append(CoverageEntry(
                element=key, status="misnamed",
                message=(f"{opened.name} declares element "
                         f"{info.element}, but species {key} is "
                         f"{name} -- SIESTA opens {opened.name} for {key} "
                         f"and would read it as {name}'s.  Put {name}'s "
                         f"pseudopotential under {opened.name}."),
                path=opened,
            ))
            continue
        read[key] = info
        # Value validation: a defective pseudo with a dead KB channel.
        # ERROR-severity -- this is the failure that masqueraded as a
        # propor crash; it also silently corrupts the physics for that
        # element's bonding, so it must block BEFORE the run.
        if info.null_channels:
            chans = "/".join(info.null_channels)
            out.append(CoverageEntry(
                element=key, status="dead_projector",
                message=(f"{opened.name} has a NULL Kleinman-Bylander "
                         f"projector for the '{chans}' channel (ekb=0): "
                         f"the pseudopotential is defective/incomplete "
                         f"for a valence angular momentum.  It gives "
                         f"wrong {key} bonding AND can trip SIESTA's "
                         f"'propor: ERROR: IMAX=0'.  Replace it with a "
                         f"vetted pseudo (PseudoDojo) matching the rest "
                         f"of your set's generator version + XC."),
                path=info.path,
            ))
            continue
        # VALID, AND POSSIBLY UNUSABLE (§ 2a.2).  A valence channel whose
        # KB projectors are all zero, carried by the <slps> semilocal block
        # instead.  That is legitimate PSML -- which is exactly why C5 above
        # exempts it and is right to -- but SIESTA was observed not to
        # rebuild the channel from it: PseudoDojo v0.5's sulfur has no p
        # projectors, S's valence is 3s2 3p4, and the run was wrong until
        # the v0.4 file replaced it (2026-06).
        #
        # ERROR (user ruling, 2026-09-03).  The file states something FALSE
        # about a channel that carries valence electrons -- sulfur's p holds
        # four of its six -- and nothing in the file says to read the
        # semilocal block instead.  So a reader doing the normal thing is
        # reading the file correctly and gets wrong physics with no crash to
        # notice: the `xc_family_mismatch` case, and it blocks for the same
        # reason.  A user whose reader consumes the semilocal form points
        # `psml_lib` at a set that does not carry the claim (v0.4.1).
        if info.semilocal_only_channels:
            chans = "/".join(info.semilocal_only_channels)
            out.append(CoverageEntry(
                element=key, status="semilocal_only",
                message=(f"{opened.name} carries the VALENCE '{chans}' channel "
                         f"only as a semilocal <slps> block: its "
                         f"Kleinman-Bylander projectors are present and set "
                         f"to ZERO -- not omitted, which is how PSML says "
                         f"'nothing here', but written out as zeros, which "
                         f"claims the projector IS zero.  It is not, and "
                         f"NOTHING in the file says to read the semilocal "
                         f"block instead -- so a reader that takes the "
                         f"nonlocal block at its word is reading the file "
                         f"correctly and gets no '{chans}' for {key} at all: "
                         f"wrong bonding, not a crash you would notice.  "
                         f"PseudoDojo v0.5 re-generated eleven elements this "
                         f"way (Ba, Bi, I, Pb, Po, Rb, Rn, S, Te, Tl, Xe); "
                         f"v0.4.1 of the same element does not."),
                path=info.path,
            ))
            continue
        # DEGRADED, not absent (WARN).  The channel survives -- one real
        # projector -- but at less than the completeness the generator's own
        # input asked for.  Every v0.5 element requests `nproj=2` for p and
        # six shipped one real, one empty (Bi, Pb, Po, Rn, Te, Tl): a second
        # projector is what makes a channel transferable across bonding
        # environments, so losing it costs accuracy rather than the physics.
        # Advisory, because the run is not wrong -- it is less good, and only
        # the user knows whether that matters for what they are doing.
        if info.partial_projector_channels:
            chans = "/".join(info.partial_projector_channels)
            out.append(CoverageEntry(
                element=key, status="partial_projectors",
                message=(f"{opened.name} has a ZERO projector on the valence "
                         f"'{chans}' channel while another projector on the "
                         f"same channel is real.  The channel still works, "
                         f"at less than the completeness its own generator "
                         f"input asked for (`nproj`), which costs "
                         f"transferability rather than correctness.  "
                         f"PseudoDojo v0.5 ships six elements this way (Bi, "
                         f"Pb, Po, Rn, Te, Tl); v0.4.1 of the same element "
                         f"does not."),
                path=info.path,
            ))
            continue
        # Check XC family + authors when expectations were supplied.
        if expected_xc_family and info.xc_family != "unknown" \
                and info.xc_family != expected_xc_family:
            # FAMILY mismatch (e.g. an LDA pseudo in a GGA run): never
            # physically correct -- the pseudo was generated with a
            # different exchange-correlation than the calc uses, so the
            # frozen core is inconsistent and energies/forces are silently
            # wrong.  Distinct status so the caller can BLOCK (error), unlike
            # the same-family author mismatch below (a minor warn).
            out.append(CoverageEntry(
                element=key, status="xc_family_mismatch",
                message=(f"{opened.name} was generated for "
                         f"{info.xc_family} ({info.xc_authors}); calc "
                         f"requests {expected_xc_family} "
                         f"({expected_xc_authors or '?'}) -- XC-FAMILY "
                         f"mismatch: bond lengths and energies will be "
                         f"silently wrong.  Use a pseudo generated with "
                         f"the calc's XC family."),
                path=info.path,
            ))
            continue
        if expected_xc_authors and info.xc_authors != "unknown" \
                and info.xc_authors.lower() != expected_xc_authors.lower():
            # Same family but different authors (PBE vs PBEsol).
            out.append(CoverageEntry(
                element=key, status="xc_mismatch",
                message=(f"{opened.name} was generated for "
                         f"{info.xc_authors}; calc requests "
                         f"{expected_xc_authors} -- minor mismatch; "
                         f"results may differ from publication-grade "
                         f"by ~1-2 kcal/mol."),
                path=info.path,
            ))
            continue
        # Relativistic check.
        if expected_relativistic and info.relativistic != "unknown" \
                and info.relativistic != expected_relativistic:
            out.append(CoverageEntry(
                element=key, status="relativistic_mismatch",
                message=(f"{opened.name} is {info.relativistic}; calc "
                         f"expects {expected_relativistic}.  For "
                         f"most non-spin-orbit work, scalar (SR) is "
                         f"correct.  Fully-relativistic (FR) is needed "
                         f"only when spin-orbit coupling matters."),
                path=info.path,
            ))
            continue
        # Pass-through warnings from the parser.
        if info.parse_warnings:
            out.append(CoverageEntry(
                element=key, status="parse_warning",
                message=(f"{opened.name} present but header had issues: "
                         f"{'; '.join(info.parse_warnings)}"),
                path=info.path,
            ))
            continue
        out.append(CoverageEntry(
            element=key, status="ok",
            message=(f"{opened.name} OK ({info.xc_family} / "
                     f"{info.xc_authors}, {info.relativistic})"),
            path=info.path,
        ))

    # Set-level VERSION CONTROL: a coherent pseudopotential set comes from
    # ONE generator version.  A single stranger (e.g. an ONCVPSP-4.0.1
    # pseudo dropped into an otherwise ONCVPSP-3.3.0 set -- exactly how
    # the bad BDT S.psml entered) is a strong smell even when each file
    # individually parses.  WARN-severity: it MIGHT be intentional, but
    # the user should confirm the whole set is from one PseudoDojo
    # release.  Compares the files the labels open (``read``), each once.
    # Iterate in a STABLE order: the labels come in the structure's order,
    # and the group-insertion order and the majority tie-break
    # (``max(..., key=len)`` below) would follow it -- the named "stranger"
    # could change with the atom order.  Sorted labels make the C4 warning
    # reproducible.
    gen_keys: Dict[str, List[str]] = {}
    for key in sorted(read):
        info = read[key]
        if info.generator in ("", "unknown"):
            continue
        gen_keys.setdefault(_generator_key(info.generator), []).append(key)
    if len(gen_keys) > 1:
        summary = "; ".join(
            f"{gk} ({','.join(sorted(els))})"
            for gk, els in sorted(gen_keys.items())
        )
        # Name the minority version(s) as the likely stranger(s).
        majority = max(gen_keys.values(), key=len)
        strangers = sorted(
            el for gk, els in gen_keys.items()
            if els is not majority for el in els
        )
        out.append(CoverageEntry(
            element=",".join(strangers) or "*", status="generator_mismatch",
            message=(f"pseudopotential set mixes generator versions: "
                     f"{summary}.  A coherent set should come from ONE "
                     f"PseudoDojo / ONCVPSP release; confirm the "
                     f"odd-one-out ({', '.join(strangers)}) is intended "
                     f"-- a stray version is how a defective pseudo "
                     f"usually sneaks in."),
            path=None,
        ))
    return out


def _generator_key(generator: str) -> str:
    """Reduce a free-form creator string to a comparable version key.

    ``"ONCVPSP-4.0.1+psml-4.0.1-76 (scalar-relativistic)"`` ->
    ``"ONCVPSP-4"``; ``"ONCVPSP-3.3.0+psml-3.3.0-73 ..."`` ->
    ``"ONCVPSP-3"``.  Compares on generator name + MAJOR version so a
    3.3.0-vs-3.3.1 patch difference doesn't warn but a 3.x-vs-4.x set
    mix does.  Falls back to the raw (trimmed) string for unrecognized
    creators so two genuinely different generators still differ.
    """
    g = generator.strip()
    m = re.match(r"([A-Za-z][A-Za-z0-9_]*)[-\s]*v?(\d+)", g)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return g.split("+", 1)[0].strip() or "unknown"


def copy_pseudopotentials(species: Sequence[str], lib: Path,
                          dest_dir: Path, *, plan=None,
                          say: bool = True) -> List[str]:
    """Copy psml files for each species. Returns list of missing species.

    WHICH FILE is :func:`psml_sources`' rule, the one every check asks:
    ``<label>.psml`` in ``lib``, verbatim -- SIESTA opens that name and no
    other (`science/pseudopotentials.md` § C1a).

    ``plan`` (`jobset.planned.Plan`) receives the copies instead of the
    disk.  ``say`` false plans them silently: a question asked of the
    folder as a prep will leave it, which copies nothing itself."""
    def _said(line: str) -> None:
        if say:
            print(line, file=sys.stderr)
    found = psml_sources(species, library=lib)
    missing: List[str] = []
    for s in species:
        key = str(s).strip()
        name = file_named_for(key) or f"{key} (no file can be named for it)"
        if found[key] is None:
            missing.append(s)
            _said(f"  WARN: no {name} in {lib}")
            continue
        src = found[key] / name
        dst = dest_dir / name
        if src.resolve() == dst.resolve():
            _said(f"  ok:   {dst.name} (already present)")
            continue
        if plan is not None:
            # PLANNED, NOT COPIED: the plan is written after the save, or
            # never -- a preview, a refusal (`job-system.md` § 5.0).
            plan.copy(src, dst)
            _said(f"  ok:   {src} -> {dst} (to copy)")
        else:
            shutil.copyfile(src, dst)
            _said(f"  ok:   {src} -> {dst}")
    return missing


__all__ = [
    "copy_pseudopotentials",
    "PsmlInfo", "CoverageEntry",
    "parse_psml_header", "scan_psml_directory", "check_coverage",
]
