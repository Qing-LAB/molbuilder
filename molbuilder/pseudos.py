"""Pseudopotential file validation for SIESTA.

SIESTA needs one ``.psml`` (or legacy ``.psf``) file per element in
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
    / mismatched-XC / mismatched-relativistic.  The web layer's
    ``/api/siesta/check-pseudos`` endpoint surfaces these so the
    user sees a clear "missing Fe.psml" or "Fe.psml is for LDA but
    you picked GGA" message before they run SIESTA.

Format reference: PSML 1.1 spec
(https://siesta-project.org/SIESTA_MATERIAL/Pseudos/Code/psml-1.1.pdf).
PseudoDojo PSML files we tested against (2024 release).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Iterable
import xml.etree.ElementTree as ET


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
    117: ("GGA", "revPBE"), # XC_GGA_X_REVPBE (some IDs)
    # LDA
    1:   ("LDA", "PW"),     # XC_LDA_X
    9:   ("LDA", "CA"),     # XC_LDA_C_PW
    12:  ("LDA", "CA"),     # XC_LDA_C_PZ
    # vdW-DF
    11:  ("VDW", "DRSLL"),  # XC_GGA_X_DRSLL (approximate; nonlocal kernel handled separately)
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
            parse_warnings=[f"could not parse XML: {exc}"],
        )

    # ----- element + Z ----------------------------------------------
    # ``_first_local`` returns an empty Element for tags with no
    # children -- in Python 3.13 + lxml that's "falsy" with a
    # DeprecationWarning, in older Python 3.x it's also "falsy"
    # silently.  Use ``is not None`` so the empty <header> isn't
    # mistakenly skipped (header attributes carry the element symbol).
    _hdr = _first_local(root, "header")
    hdr = _hdr if _hdr is not None else root
    element = (hdr.attrib.get("atomic-label")
               or hdr.attrib.get("element")
               or "").strip().capitalize()
    z_str = hdr.attrib.get("z-pseudo") or hdr.attrib.get("atomic-number") or ""
    try:
        atomic_number = int(float(z_str)) if z_str else 0
    except ValueError:
        atomic_number = 0
        warnings.append(f"unparseable z-pseudo: {z_str!r}")
    if not element:
        warnings.append("no element / atomic-label attribute in <header>")

    # ----- XC functional -------------------------------------------
    xc_family, xc_authors = "unknown", "unknown"
    # Try the libxc-formatted entries first (PseudoDojo's preferred).
    libxc_ids: List[int] = []
    for el in _findall_local(root, "libxc-info"):
        for attr in ("id", "code", "name"):
            v = el.attrib.get(attr)
            if v and v.isdigit():
                libxc_ids.append(int(v))
                break
    # Also try <exchange> / <correlation> child id attributes.
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

    return PsmlInfo(
        path=path, element=element, atomic_number=atomic_number,
        xc_family=xc_family, xc_authors=xc_authors,
        relativistic=relativistic, generator=generator,
        valence_config=valence_config,
        suggested_mesh_ry=suggested_mesh,
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
    status:   str       # "ok" | "missing" | "xc_mismatch" | "relativistic_mismatch" | "parse_warning"
    message:  str
    path:     Optional[Path] = None


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
      * ``"xc_mismatch"`` -- pseudo's XC differs from the calc's;
        WARN-severity.  The pseudo will load and SIESTA won't object,
        but bond lengths / energies are silently wrong.
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
        # Check XC family + authors when expectations were supplied.
        if expected_xc_family and info.xc_family != "unknown" \
                and info.xc_family != expected_xc_family:
            out.append(CoverageEntry(
                element=key, status="xc_mismatch",
                message=(f"{key}.psml was generated for "
                         f"{info.xc_family} ({info.xc_authors}); calc "
                         f"requests {expected_xc_family} "
                         f"({expected_xc_authors or '?'}) -- bond "
                         f"lengths will be silently wrong unless the "
                         f"pseudo matches the calc's XC family."),
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
    return out


__all__ = [
    "PsmlInfo", "CoverageEntry",
    "parse_psml_header", "scan_psml_directory", "check_coverage",
]
