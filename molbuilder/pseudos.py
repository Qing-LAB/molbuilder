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


def resolve_psml_lib(raw: str, *,
                      base: Optional[Path] = None,
                      dest_dir: Optional[Path] = None) -> Path:
    """Resolve ``cfg.psml_lib`` against the anchor its SPELLING names.

    The rule is `job-contracts.md` § 2.5a, and it is the whole rule:

      * absolute, or ``~/...``           -> itself.
      * starts with ``./`` or ``../``    -> the CALCULATION folder
                                            (``dest_dir``).
      * anything else (a bare name)      -> the ``projects/`` tree the
                                            calculation lives in, found by
                                            walking up from ``dest_dir``.

    **One anchor per spelling, and nothing is tried-and-discarded.**  Until
    2026-08-21 this walked a cascade -- ``dest_dir``, then the tree, then
    ``<cwd>/projects`` -- taking whichever candidate happened to exist.  Three
    consequences, all of them found in one Sol failure:

      * the same string named a different folder on different machines,
        because which candidate "happened to exist" is a property of the
        machine, not of what the user asked for;
      * a total miss reported the LAST candidate, so ``prep`` refused with
        ``.../optimization/Relax/projects/pseudopotential`` -- a folder
        assembled from where the user was standing, which no user had chosen
        and no amount of reading the message would explain;
      * and because trying was free, nothing ever had to decide what the
        spelling meant.

    Now the spelling decides, the answer is one path, and a miss is reported
    against the anchor that spelling asked for (`architecture.md` § 7, A10).

    **The caller with no calculation folder.**  Server-side validation runs
    before anything is written, so there is no *"here"* to be relative to and
    no tree to walk up from.  With ``dest_dir=None`` both relative forms
    anchor at ``projects_root()`` -- the server's own declared root, which is
    the one place a working directory is a legitimate anchor.  ``base``
    overrides that root (tests, and any caller serving a different tree).

    Returns an absolute, **not-resolved** Path: callers do ``.is_dir()`` and
    want the path to stay symlink-faithful in error messages.  The path is
    returned whether or not it exists -- this function answers *"which folder
    does this spelling name?"*, and the caller reports the miss.
    """
    from .projects import find_projects_root, projects_root
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p

    dotted = raw.startswith("./") or raw.startswith("../")

    if dest_dir is None:
        # No calculation to be relative to: the server's declared root.
        root = base if base is not None else projects_root()
        return root / p

    if dotted:
        return Path(dest_dir) / p

    tree = find_projects_root(dest_dir)
    if tree is not None:
        return tree / p
    # The calculation is not inside a projects/ tree -- a hand-made folder,
    # or a bundle copied somewhere flat.  There is no tree to name, so the
    # bare spelling has no anchor and the calculation folder is the only
    # place left that the user actually chose.
    return Path(dest_dir) / p


def describe_psml_anchor(raw: str, *, dest_dir: Optional[Path] = None) -> str:
    """One sentence: which anchor this spelling named, and where that landed.

    **Why the explanation lives beside the rule.**  Two surfaces refuse over a
    missing library -- ``prep`` on the target machine and the browser's
    preflight -- and each used to describe the resolution in its own words.
    Both descriptions were of the old cascade, so when the rule changed there
    were two places telling users about anchors that are no longer tried.  A
    refusal is the only place most users ever learn the rule, which makes it
    part of the rule, not decoration on top of it.

    The caller owns severity and formatting; this owns the fact.
    """
    landed = resolve_psml_lib(raw, dest_dir=dest_dir)
    p = Path(raw).expanduser()
    if p.is_absolute():
        return f"{raw} is an absolute path, so that is where it looked: {landed}."

    if dest_dir is None:
        return (f"{raw!r} is relative and there is no calculation folder yet, "
                f"so it was resolved against this server's projects tree: "
                f"{landed}.")

    if raw.startswith("./") or raw.startswith("../"):
        return (f"{raw!r} starts with a dot, which means "
                f"\"from this calculation\", so it looked in {landed}.")

    from .projects import PROJECTS_ROOT_NAME, find_projects_root
    tree = find_projects_root(dest_dir)
    if tree is None:
        return (f"{raw!r} is a bare name, which means the {PROJECTS_ROOT_NAME}/ "
                f"tree this calculation lives in -- but this calculation is "
                f"not inside one, so it looked beside the calculation: "
                f"{landed}.")
    lead = (f"{raw!r} is a bare name, which means the {PROJECTS_ROOT_NAME}/ "
            f"tree this calculation lives in ({tree}), so it looked in "
            f"{landed}.")
    if raw.split("/", 1)[0] == PROJECTS_ROOT_NAME:
        # The one spelling that cannot work: the walk-up already found the
        # tree, so naming it again nests it inside itself.  Say so -- do not
        # silently strip it, because a user who meant something else by that
        # first segment deserves to see what happened to it.
        lead += (f"  Note the doubled {PROJECTS_ROOT_NAME}/: the tree is what "
                 f"the walk-up finds, so the path you write is the part "
                 f"INSIDE it.  Drop the leading {PROJECTS_ROOT_NAME}/.")
    return lead


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
    suggested_mesh_ry:   Optional[float] # PseudoDojo's recommended cutoff
    null_channels:       List[str] = field(default_factory=list)
                                         # l-letters (s/p/d/f) whose ENTIRE
                                         # Kleinman-Bylander channel has
                                         # ekb~=0 -- a defective/incomplete
                                         # pseudo (the BDT S.psml had a dead
                                         # 'p' channel; triggers propor
                                         # IMAX=0 AND gives wrong physics).
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
            null_channels=[],
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
    null_channels = sorted(
        l for l, eks in proj_by_l.items()
        if eks and all(e < _EKB_NULL for e in eks)
        and l not in semilocal_ls          # semilocal block carries it
    )

    return PsmlInfo(
        path=path, element=element, atomic_number=atomic_number,
        xc_family=xc_family, xc_authors=xc_authors,
        relativistic=relativistic, generator=generator,
        valence_config=valence_config,
        suggested_mesh_ry=suggested_mesh,
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
                        # "parse_warning"
    message:  str
    path:     Optional[Path] = None


#: The `CoverageEntry.status` values that BLOCK — a run with any of these
#: cannot be correct: a pseudo is absent (SIESTA won't start), a valence
#: channel is physically missing (wrong Hamiltonian), or the XC *family* is
#: wrong (silently-wrong energies).  This is the SINGLE source of truth for
#: "which statuses are ERROR"; the SIESTA preflight
#: (``validation.siesta._check_siesta_pseudo_coverage``) and the CLI
#: (``cli.cmd_pseudo_check``) both consume it so the two surfaces cannot
#: drift (they did until 2026-07-26: the CLI omitted ``xc_family_mismatch``).
#: Everything else — ``xc_mismatch`` (same-family author diff),
#: ``relativistic_mismatch``, ``generator_mismatch``, ``parse_warning`` — is
#: advisory (WARN); ``ok`` is a silent pass.
ERROR_STATUSES = frozenset({"missing", "dead_projector", "xc_family_mismatch"})


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
