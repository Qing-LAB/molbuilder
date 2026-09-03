"""Pseudopotential file validation for SIESTA.

SIESTA needs one ``.psml`` file per element in
the structure.  These files carry metadata in their header that MUST
match the calculation config:

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
    ``{element: PsmlInfo}`` mapping all parseable .psml files.
  * :func:`check_coverage` -- given a Structure and a directory,
    return a list of per-element status entries: present / missing
    / mismatched-XC / mismatched-relativistic.  The SIESTA
    validator (``molbuilder.validation.siesta``) calls this during
    preflight + render so the user sees a clear "missing Fe.psml"
    or "Fe.psml is for LDA but you picked GGA" Issue message
    before they run SIESTA.

Format reference: PSML 1.1 spec
(https://siesta-project.org/SIESTA_MATERIAL/Pseudos/Code/psml-1.1.pdf).
PseudoDojo PSML files we tested against (2024 release).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Iterable
import re
import xml.etree.ElementTree as ET


class PsmlLibError(ValueError):
    """A ``psml_lib`` spelling the rule cannot answer -- dotted, or an
    absolute path outside the projects tree.  The message teaches the
    rule; callers surface it verbatim in their own refusal type."""


def resolve_psml_lib(raw: str, *,
                     base: Optional[Path] = None,
                     dest_dir: Optional[Path] = None) -> Path:
    """Resolve ``cfg.psml_lib``: **a path inside the projects tree.**

    The rule is `job-contracts.md` § 2.5a, ONE line since 2026-08-28
    (user: the library always lives inside the project tree, so psml
    paths speak the same language as every sidebar path):

      * a RELATIVE path (``pseudopotential``, ``shared/psml``) is
        measured from the TREE ROOT -- the tree the calculation lives
        in, walked up from ``dest_dir``; the server's own
        ``projects_root()`` when there is no calculation yet (``base``
        overrides it, for tests and other trees);
      * an ABSOLUTE path is accepted as a convenience spelling of the
        same fact -- it must lie INSIDE that tree, and is refused
        otherwise;
      * the ``./`` / ``../`` spellings are RETIRED with the old anchor
        cascade: pseudopotentials already beside the calculation are
        used without this field (prep adopts them), so the dotted
        anchor had no remaining job.

    Raises :class:`PsmlLibError` for a spelling the rule cannot answer.
    Returns an absolute, **not-resolved** Path whether or not it
    exists: this function answers *"which folder does this name?"* and
    the caller reports a miss (``.is_dir()``) with
    :func:`describe_psml_anchor`.
    """
    from .projects import (OutsideRoot, PROJECTS_ROOT_NAME, contain,
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

    if p.is_absolute():
        try:
            contain(p, tree)
        except OutsideRoot as exc:
            raise PsmlLibError(
                f"{raw!r} is outside the {PROJECTS_ROOT_NAME}/ tree "
                f"({tree}).  `psml_lib` always lives inside the tree -- "
                f"copy the library in (e.g. {tree}/pseudopotential) and "
                f"write the tree-relative name.") from exc
        return p
    return tree / p


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
    atomic_number:       int             # integer Z (decoded from z-pseudo)
    xc_family:           str             # "GGA" | "LDA" | "VDW" | "unknown"
    xc_authors:          str             # "PBE" | "PBEsol" | "CA" | etc.
    relativistic:        str             # "no" | "scalar" | "spin-orbit"
    generator:           str             # "ONCVPSP" / "ATOM" / "Hamann" / etc.
    valence_config:      str             # "[Ar] 3d6 4s2" etc. (free-form)
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
    semilocal_only_channels: List[str] = field(default_factory=list)
                                         # VALENCE l-channels whose KB
                                         # projectors are all ~zero but which
                                         # an <slps> semilocal block carries.
                                         # Valid PSML -- and SIESTA was
                                         # observed not to rebuild the channel
                                         # from it (S, PseudoDojo v0.5,
                                         # 2026-06).  § 2a.2.
    parse_warnings:      List[str] = field(default_factory=list)


# PSML uses the namespace ``http://launchpad.net/psml`` for all
# elements (sometimes versioned: psml-1.1).  ElementTree's strict
# namespace handling means we have to match on the local tag (the
# part after the }).
def _localtag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _findall_local(root, local_name: str):
    """Find all descendants with local tag == ``local_name`` regardless
    of namespace.  PseudoDojo PSML files vary in their xmlns URL."""
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
                            # correlation, NOT a vdW-DF exchange (the pre-2026-07
                            # ("VDW","DRSLL") label mis-classified its family).
}


def parse_psml_header(path: Path) -> PsmlInfo:
    """Parse a single .psml file and return the canonical metadata.

    Tolerant: missing fields produce empty strings + a parse warning,
    not an exception.  XC family / authors fall back to ``"unknown"``
    when the libxc id isn't in our table (rare functionals).  Returns
    a PsmlInfo with ``element=""`` only if the file is so malformed
    we can't even find the element symbol (extremely rare; PSML
    requires it).
    """
    path = Path(path)
    warnings: List[str] = []
    try:
        root = ET.parse(str(path)).getroot()
    except (ET.ParseError, OSError) as exc:
        return PsmlInfo(
            path=path, element="", atomic_number=0,
            xc_family="unknown", xc_authors="unknown",
            relativistic="unknown", generator="unknown",
            valence_config="", suggested_mesh_ry=None,
            cutoff_hints_ry={}, null_channels=[],
            semilocal_only_channels=[],
            parse_warnings=[f"could not parse XML: {exc}"],
        )

    # ----- element + Z ----------------------------------------------
    # Real PseudoDojo PSML files (the standard format SIESTA users
    # download) put element + Z + relativity on ``<pseudo-atom-spec>``;
    # the older / simpler PSML used ``<header>``.  Try both before
    # falling back to root (which won't have the attribute and yields
    # element="" + a parse warning).
    # ``_first_local`` returns an empty Element for tags with no
    # children -- in Python 3.13 that's "falsy" with a
    # DeprecationWarning, in older Python 3.x it's also "falsy"
    # silently.  Use ``is not None`` so the empty element isn't
    # mistakenly skipped.
    _hdr = _first_local(root, "pseudo-atom-spec")
    if _hdr is None:
        _hdr = _first_local(root, "header")
    hdr = _hdr if _hdr is not None else root
    element = (hdr.attrib.get("atomic-label")
               or hdr.attrib.get("element")
               or "").strip().capitalize()
    # Prefer atomic-number (the element's Z) over z-pseudo (the
    # valence electron count -- e.g. Fe has Z=26 but z-pseudo=16
    # because the pseudo treats 3s²3p⁶3d⁶4s² = 16 e⁻ as valence).
    # Previous order silently mis-reported Z=16 for Fe.
    z_str = (hdr.attrib.get("atomic-number")
             or hdr.attrib.get("z-pseudo")
             or "")
    try:
        atomic_number = int(float(z_str)) if z_str else 0
    except ValueError:
        atomic_number = 0
        warnings.append(f"unparseable z-pseudo: {z_str!r}")
    if not element:
        warnings.append("no element / atomic-label attribute in <header>")

    # ----- XC functional -------------------------------------------
    xc_family, xc_authors = "unknown", "unknown"
    # Real PseudoDojo PSML uses:
    #   <libxc-info number-of-functionals="2">
    #     <functional name="..." type="exchange"    id="101"/>
    #     <functional name="..." type="correlation" id="130"/>
    #   </libxc-info>
    # The id attribute is on <functional>, NOT on <libxc-info>.
    # Earlier code looked for id on <libxc-info> and missed every
    # real PSML.  Also support older / synthesized formats that put
    # id directly on <libxc-info> or on bare <exchange> / <correlation>.
    libxc_ids: List[int] = []
    for el in _findall_local(root, "functional"):
        v = el.attrib.get("id")
        if v and v.isdigit():
            libxc_ids.append(int(v))
    if not libxc_ids:
        for el in _findall_local(root, "libxc-info"):
            for attr in ("id", "code", "name"):
                v = el.attrib.get(attr)
                if v and v.isdigit():
                    libxc_ids.append(int(v))
                    break
    if not libxc_ids:
        for el in (_findall_local(root, "exchange")
                   + _findall_local(root, "correlation")):
            v = el.attrib.get("id") or el.attrib.get("libxc")
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
    # Fall back to the <header xc-id="..."> attribute used by some files.
    if xc_family == "unknown":
        xc_attr = (hdr.attrib.get("xc")
                   or hdr.attrib.get("xc-functional")
                   or hdr.attrib.get("xc-id")
                   or "").upper()
        if xc_attr:
            if "PBE" in xc_attr and "SOL" in xc_attr:
                xc_family, xc_authors = "GGA", "PBEsol"
            elif "PBE" in xc_attr:
                xc_family, xc_authors = "GGA", "PBE"
            elif "BLYP" in xc_attr:
                xc_family, xc_authors = "GGA", "BLYP"
            elif xc_attr.startswith("LDA") or "CA" in xc_attr:
                xc_family, xc_authors = "LDA", "CA"

    # ----- relativistic --------------------------------------------
    # PSML carries this either as a top-level header attribute
    # (relativity="scalar" | "no" | "dirac") or inside <provenance>.
    rel_raw = (hdr.attrib.get("relativity")
               or hdr.attrib.get("relativistic")
               or "").lower()
    if rel_raw in ("dirac", "fr", "fully", "fully-relativistic", "spin-orbit"):
        relativistic = "spin-orbit"
    elif rel_raw in ("scalar", "sr", "scalar-relativistic"):
        relativistic = "scalar"
    elif rel_raw in ("no", "non", "non-relativistic", "nr"):
        relativistic = "no"
    else:
        relativistic = "unknown"

    # ----- generator -----------------------------------------------
    provenance = _first_local(root, "provenance")
    generator = ""
    if provenance is not None:
        generator = (provenance.attrib.get("creator")
                     or provenance.attrib.get("generator")
                     or "").strip()
    if not generator:
        # Some PSMLs have a <generator-info> sibling.
        gi = _first_local(root, "generator")
        if gi is not None:
            generator = (gi.attrib.get("name")
                         or (gi.text or "").strip())
    generator = generator or "unknown"

    # ----- valence configuration -----------------------------------
    vc_el = _first_local(root, "valence-configuration")
    if vc_el is not None:
        # Some files put it as text; others as child <shell> tags.
        children = list(vc_el)
        if children:
            parts = []
            for sh in children:
                n = sh.attrib.get("n", "")
                l = sh.attrib.get("l", "")
                occ = sh.attrib.get("occupation", "")
                parts.append(f"{n}{l}{occ}")
            valence_config = " ".join(parts)
        else:
            valence_config = (vc_el.text or "").strip()
    else:
        valence_config = ""

    # ----- suggested mesh cutoff (PseudoDojo extension) ------------
    suggested_mesh = None
    pd = _first_local(root, "pseudo-dojo")
    if pd is not None:
        for key in ("rmax", "mesh-cutoff", "recommended-cutoff"):
            v = pd.attrib.get(key)
            if v:
                try:
                    suggested_mesh = float(v)
                    break
                except ValueError:
                    pass

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
    # channel -- flagging them as "dead" (the pre-2026-07 behaviour) was
    # a FALSE POSITIVE that ERROR-blocked those common elements.
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

    return PsmlInfo(
        path=path, element=element, atomic_number=atomic_number,
        xc_family=xc_family, xc_authors=xc_authors,
        relativistic=relativistic, generator=generator,
        valence_config=valence_config,
        suggested_mesh_ry=(cutoff_hints.get("normal")
                           if cutoff_hints else suggested_mesh),
        cutoff_hints_ry=cutoff_hints,
        semilocal_only_channels=semilocal_only_channels,
        null_channels=null_channels,
        parse_warnings=warnings,
    )


def scan_psml_directory(directory: Path) -> Dict[str, PsmlInfo]:
    """Walk ``directory`` (non-recursive) and return ``{element:
    PsmlInfo}`` for every parseable .psml file.

    When two files claim the same element (e.g. ``Fe.psml`` AND
    ``Fe_pbe.psml``), the FIRST one encountered wins.  The
    additional file is silently dropped from the mapping but the
    caller can call :func:`parse_psml_header` directly for full
    enumeration.
    """
    directory = Path(directory)
    out: Dict[str, PsmlInfo] = {}
    if not directory.is_dir():
        return out
    for p in sorted(directory.iterdir()):
        if p.suffix.lower() != ".psml":
            continue
        info = parse_psml_header(p)
        if not info.element:
            continue
        out.setdefault(info.element, info)
    return out


@dataclass
class CoverageEntry:
    """One row of a coverage report: per-element pass / warn / fail."""
    element:  str
    status:   str       # "ok" | "missing" | "dead_projector" |
                        # "xc_family_mismatch" | "xc_mismatch" |
                        # "relativistic_mismatch" | "generator_mismatch" |
                        # "parse_warning" | "semilocal_only"
    message:  str
    path:     Optional[Path] = None


#: The `CoverageEntry.status` values that BLOCK — a run with any of these
#: cannot be correct: a pseudo is absent (SIESTA won't start), a valence
#: channel is physically missing (wrong Hamiltonian), the XC *family* is wrong
#: (silently-wrong energies), or a valence channel is present and states a
#: strength of ZERO (``semilocal_only`` — the same silently-wrong-energies
#: case, added 2026-09-03 by user ruling after PseudoDojo v0.5's sulfur ran
#: with no p channel).  This is the SINGLE source of truth for "which statuses
#: are ERROR"; the SIESTA preflight
#: (``validation.siesta._check_siesta_pseudo_coverage``) and the CLI
#: (``cli.cmd_pseudo_check``) both consume it so the two surfaces cannot
#: drift (they did until 2026-07-26: the CLI omitted ``xc_family_mismatch``).
#: Everything else — ``xc_mismatch`` (same-family author diff),
#: ``relativistic_mismatch``, ``generator_mismatch``, ``parse_warning`` — is
#: advisory (WARN); ``ok`` is a silent pass.
ERROR_STATUSES = frozenset({"missing", "dead_projector", "xc_family_mismatch",
                            "semilocal_only"})


#: XC authors -> the FAMILY a pseudopotential must belong to.
#:
#: One home, for the same reason ``ERROR_STATUSES`` has one: it decides an
#: ERROR-severity verdict (``xc_family_mismatch`` — *"never physically
#: correct, energies/forces silently wrong"*), so two copies of it are two
#: opinions about whether a run may proceed.
#:
#: **It was written twice and they already disagreed** (found 2026-08-18):
#: ``validation/siesta.py`` mapped ``drsll`` / ``lmkll`` to ``VDW`` and
#: ``cli.py`` had no VDW arm at all, so a van-der-Waals run audited from the
#: command line compared its pseudos against *no* expected family and passed
#: a mismatch the preflight would have blocked.
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


def check_coverage(elements: Iterable[str],
                    directory: Path,
                    *,
                    expected_xc_family: Optional[str] = None,
                    expected_xc_authors: Optional[str] = None,
                    expected_relativistic: str = "scalar",
                    ) -> List[CoverageEntry]:
    """Given a list of element symbols the structure needs (no
    duplicates required; we de-duplicate) and a directory of
    pseudopotentials, return per-element status.

    Status values:
      * ``"ok"`` -- pseudo present, XC + relativistic match (or
        expectations weren't supplied).
      * ``"missing"`` -- no .psml file for this element.  Hard fail
        for SIESTA: the run won't start.
      * ``"xc_family_mismatch"`` -- pseudo's XC FAMILY differs from the
        calc's (e.g. an LDA pseudo in a GGA run).  ERROR-severity: never
        physically correct, energies/forces silently wrong.
      * ``"xc_mismatch"`` -- same family, different authors (PBE vs
        PBEsol); WARN-severity, a minor ~1-2 kcal/mol difference.
      * ``"relativistic_mismatch"`` -- e.g. user picked an SR pseudo
        for a fully-relativistic run; WARN-severity.
      * ``"parse_warning"`` -- the .psml file is present but missing
        metadata; assume it's OK and let SIESTA report any issues
        at startup.
    """
    info_map = scan_psml_directory(directory)
    out: List[CoverageEntry] = []
    seen: set = set()
    for el in elements:
        key = el.capitalize()
        if key in seen:
            continue
        seen.add(key)
        info = info_map.get(key)
        if info is None:
            out.append(CoverageEntry(
                element=key, status="missing",
                message=(f"no .psml file for {key} in {directory.name!r} -- "
                         f"SIESTA will refuse to start.  Download from "
                         f"http://www.pseudo-dojo.org (PSML format, "
                         f"functional matching cfg.xc_authors)."),
                path=None,
            ))
            continue
        # Value validation: a defective pseudo with a dead KB channel.
        # ERROR-severity -- this is the failure that masqueraded as a
        # propor crash; it also silently corrupts the physics for that
        # element's bonding, so it must block BEFORE the run.
        if info.null_channels:
            chans = "/".join(info.null_channels)
            out.append(CoverageEntry(
                element=key, status="dead_projector",
                message=(f"{key}.psml has a NULL Kleinman-Bylander "
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
                message=(f"{key}.psml carries the VALENCE '{chans}' channel "
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
                         f"v0.4.1 of the same element does not.  See "
                         f"science/pseudopotentials.md 2a.2."),
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
                message=(f"{key}.psml was generated for "
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
                message=(f"{key}.psml was generated for "
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
                message=(f"{key}.psml is {info.relativistic}; calc "
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
                message=(f"{key}.psml present but header had issues: "
                         f"{'; '.join(info.parse_warnings)}"),
                path=info.path,
            ))
            continue
        out.append(CoverageEntry(
            element=key, status="ok",
            message=(f"{key}.psml OK ({info.xc_family} / "
                     f"{info.xc_authors}, {info.relativistic})"),
            path=info.path,
        ))

    # Set-level VERSION CONTROL: a coherent pseudopotential set comes from
    # ONE generator version.  A single stranger (e.g. an ONCVPSP-4.0.1
    # pseudo dropped into an otherwise ONCVPSP-3.3.0 set -- exactly how
    # the bad BDT S.psml entered) is a strong smell even when each file
    # individually parses.  WARN-severity: it MIGHT be intentional, but
    # the user should confirm the whole set is from one PseudoDojo
    # release.  Only compares pseudos that are actually present.
    # Iterate in a STABLE order: ``seen`` is a set, and a hash-ordered walk
    # would make both the group-insertion order and the majority tie-break
    # (``max(..., key=len)`` below) non-deterministic -- the named "stranger"
    # could flip run to run.  Sorted keys make the C4 warning reproducible.
    gen_keys: Dict[str, List[str]] = {}
    for key in sorted(seen):
        info = info_map.get(key)
        if info is None or info.generator in ("", "unknown"):
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


__all__ = [
    "PsmlInfo", "CoverageEntry",
    "parse_psml_header", "scan_psml_directory", "check_coverage",
]
