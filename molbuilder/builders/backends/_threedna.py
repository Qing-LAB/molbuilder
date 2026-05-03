"""3DNA backend (uses ``fiber``).

3DNA's ``fiber`` produces canonical B-form / A-form / Z-form helical
geometry from sequence -- the only thing the existing ``rdkit``
(folded conformer) and ``amber`` (extended chain) backends do not
provide.

Tool limitations (and how this module compensates)
--------------------------------------------------
``fiber`` is fast, well-validated for canonical helical geometry, and
the de-facto standard for fiber-diffraction-derived starting
coordinates.  It also has two long-standing quirks that the user
will hit:

  1. **Heavy-atom output.**  ``fiber`` writes deoxyribose, base, and
     phosphate heavy atoms only -- no H on bases, sugars, or phosphate
     oxygens.  H atoms are needed for any DFT / MD calculation: the
     electron count is wrong without them, and Watson-Crick H-bonding
     can't form.  We compensate by routing every X3DNA build through
     ``chemistry.add_hydrogens`` (OpenBabel-first, RDKit-fallback) at
     the ``nucleic.build_dna``/``build_rna`` layer.  See
     ``molbuilder/chemistry.py`` for the H-add tool comparison.

  2. **5'-terminal phosphate is mandatory.**  ``fiber`` always emits
     a 5'-phosphate group on the first residue, regardless of any
     flag we pass (``-single`` controls duplex vs single-strand, not
     terminal phosphorylation).  An "ATGC" oligo with the user's
     ``terminal="OH"`` request comes back with 4 phosphate groups
     instead of the chemically-correct 3 (the three internal A-T,
     T-G, G-C bridges).  ``_strip_5prime_phosphate`` post-processes
     the parsed Structure to remove the spurious P + OP1 + OP2 from
     the 5'-terminal residue when ``terminal in ('OH', '3P')``.  The
     bridging O5' atom stays; H is added later by chemistry.add_hydrogens.

  3. **3'-phosphate cannot be added.**  fiber's output has 5'-P / 3'-OH;
     we can strip the 5'-P, but adding a 3'-P would require re-running
     chemistry from scratch.  ``terminal in ('PP', '3P')`` warn that
     the request will be served as 5'-P / 3'-OH or 5'-OH / 3'-OH
     respectively.

  4. **Form constraints.**  fiber's ``-z`` flag (Z-DNA) only works
     for poly-d(GC) sequences; ``-rna`` only produces A-form RNA.
     Mismatches are warned by the dispatcher above (see ``build()``).

Detection chain (first hit wins)
--------------------------------

  1. **In-tree** -- look for ``<repo_root>/x3dna-v*/`` next to the
     molbuilder package.  This is the easiest path for a dev install:
     just unpack the 3DNA tarball at the repo root and the backend
     finds it automatically.  The ``x3dna-v*/`` directory is
     gitignored (see ``.gitignore`` -- both for hygiene and to make
     it structurally hard to redistribute 3DNA accidentally).

  2. **$X3DNA env var** -- the canonical 3DNA install convention.
     Set ``export X3DNA=$HOME/opt/x3dna-v2.4`` and we use it.

  3. **fiber on PATH** -- last resort; we derive the X3DNA root from
     ``shutil.which('fiber')`` (assumes the standard ``$X3DNA/bin/``
     layout).  Useful when the user has a system-wide install that
     doesn't bother setting the env var.

For each candidate we verify the install is *complete*: ``bin/fiber``
exists and is executable AND ``config/`` (the 3DNA atomic-parameter
files, ``Atomic_*.pdb`` and friends) is present.  Without ``config/``,
``fiber`` runs but emits cryptic errors at runtime.

Licensing
---------
3DNA is distributed under a **non-commercial-use** license through
http://x3dna.org/ behind a registration form.  molbuilder itself is
MIT; we **do not** auto-fetch, mirror, or bundle 3DNA, and the
``BackendUnavailable`` message tells the user to download it from
x3dna.org per their instructions.  See
``docs/design.md`` § "3DNA (canonical helix builder)" for the full
contract.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ...structure import Structure
from ._common import (parse_pdb_to_structure, select_chain,
                      verify_backbone_connectivity)


# Atom names that constitute the 5'-terminal phosphate group in fiber's
# PDB output.  The bridging O5' is part of the sugar and is NOT in this
# set -- after the strip it becomes a free hydroxyl oxygen, ready to
# accept an H from chemistry.add_hydrogens().  We accept both modern
# (OP1/OP2/OP3) and legacy (O1P/O2P/O3P) PDB naming conventions to be
# robust against fiber-version drift.
_PHOSPHATE_ATOM_NAMES = {
    "P", "OP1", "OP2", "OP3", "O1P", "O2P", "O3P",
    # Sometimes phosphate H's appear in protonated outputs:
    "HOP1", "HOP2", "HOP3", "H1P", "H2P", "H3P",
}


# --------------------------------------------------------------------- #
#  Resolution                                                           #
# --------------------------------------------------------------------- #


@dataclass
class _Threedna:
    """Resolved 3DNA install: path to the fiber executable and the
    X3DNA root (which we'll inject into the env when shelling out, so
    fiber's auxiliary scripts can find their config files)."""
    fiber: str
    root:  str
    source: str    # "in-tree" / "env" / "path" -- only for diagnostics


def _looks_complete(root: str) -> bool:
    """A 3DNA root is 'complete' if bin/fiber is executable AND
    config/ exists.  config/ holds the per-base PDB templates fiber
    needs at runtime; without it fiber dies cryptically."""
    fiber  = os.path.join(root, "bin", "fiber")
    config = os.path.join(root, "config")
    return (os.path.isfile(fiber)
            and os.access(fiber, os.X_OK)
            and os.path.isdir(config))


def _find_in_tree() -> Optional[_Threedna]:
    """Look for an x3dna-v*/ directory at the repo root (one level
    above the molbuilder package).  Works for dev / editable installs
    where the user has unpacked the tarball next to the source."""
    # _threedna.py -> repo_root/molbuilder/builders/backends/_threedna.py
    # parent.parent.parent.parent = repo_root
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    for candidate in sorted(repo_root.glob("x3dna-v*")):
        if candidate.is_dir() and _looks_complete(str(candidate)):
            return _Threedna(
                fiber  = str(candidate / "bin" / "fiber"),
                root   = str(candidate),
                source = "in-tree",
            )
    return None


def _find_via_env() -> Optional[_Threedna]:
    """Use $X3DNA if set and pointing at a complete install."""
    root = os.environ.get("X3DNA")
    if not root:
        return None
    root = os.path.abspath(os.path.expanduser(root))
    if not _looks_complete(root):
        return None
    return _Threedna(
        fiber  = os.path.join(root, "bin", "fiber"),
        root   = root,
        source = "env",
    )


def _find_via_path() -> Optional[_Threedna]:
    """Last resort: fiber is on PATH; derive X3DNA root as the parent
    of the bin/ directory."""
    fiber = shutil.which("fiber")
    if not fiber:
        return None
    fiber = os.path.abspath(fiber)
    root = os.path.dirname(os.path.dirname(fiber))
    if not _looks_complete(root):
        return None
    return _Threedna(fiber=fiber, root=root, source="path")


def _resolve() -> Optional[_Threedna]:
    """Walk the detection chain.  First hit wins."""
    return _find_in_tree() or _find_via_env() or _find_via_path()


def is_available() -> bool:
    """True iff some 3DNA install is reachable via the detection chain."""
    return _resolve() is not None


# --------------------------------------------------------------------- #
#  Build                                                                #
# --------------------------------------------------------------------- #


# fiber's form flags (from `fiber -h`):
#   -b   B-DNA (default)
#   -a   A-DNA
#   -z   Z-DNA  (only valid for poly d(GC))
#   -rna RNA (A-form duplex)
_FIBER_FLAGS = {
    ("dna", "B"): ["-b"],
    ("dna", "A"): ["-a"],
    ("dna", "Z"): ["-z"],
    ("rna", "A"): ["-rna"],
}


def build(kind: str, sequence: str, form: str, terminal: str,
          title: Optional[str] = None) -> Structure:
    if kind not in ("dna", "rna"):
        raise ValueError(
            f"3DNA backend supports kind in 'dna'|'rna'; got {kind!r}"
        )

    # RNA only supports A-form via fiber.
    if kind == "rna" and form not in ("A", None, ""):
        import warnings
        warnings.warn(
            f"3DNA fiber builds A-form RNA only; ignoring form={form!r}.",
            RuntimeWarning, stacklevel=4,
        )
        form = "A"
    if form is None or form == "":
        form = "B" if kind == "dna" else "A"

    flags_key = (kind, form)
    if flags_key not in _FIBER_FLAGS:
        raise ValueError(
            f"3DNA backend doesn't support kind={kind!r} form={form!r}; "
            f"valid: {sorted(_FIBER_FLAGS)}"
        )

    found = _resolve()
    if found is None:
        from . import BackendUnavailable
        raise BackendUnavailable(_unavailable_message())

    seq = "".join(c for c in sequence.upper() if c.isalpha())
    if not seq:
        raise ValueError("Empty sequence")

    # fiber's default output is 5'-phosphate / 3'-OH regardless of the
    # `-single` flag; we post-process below to honour the requested
    # terminal state.  We can strip the 5'-phosphate (turning fiber's
    # 5'-P into the requested 5'-OH) but cannot add a 3'-phosphate
    # without rerunning chemistry, so PP / 3P warn for the missing 3'.
    if terminal in ("PP", "3P"):
        import warnings
        warnings.warn(
            f"3DNA backend cannot add a 3'-terminal phosphate "
            f"(fiber emits 5'-P / 3'-OH).  Requested terminal={terminal!r} "
            f"will be served as 5'-P / 3'-OH.  For a 3'-phosphate, "
            f"post-process the structure with an external tool.",
            RuntimeWarning, stacklevel=4,
        )

    cmd = [found.fiber] + _FIBER_FLAGS[flags_key] + [
        f"-seq={seq}",
        # Single-stranded output -- molbuilder builders are
        # single-chain; the user can swap to duplex by post-processing.
        "-single",
    ]

    # fiber needs $X3DNA in the env at runtime so its auxiliary scripts
    # / config look-ups resolve.  Inject ours regardless of what the
    # caller's shell has set.
    env = os.environ.copy()
    env["X3DNA"] = found.root
    env["PATH"]  = os.path.join(found.root, "bin") + os.pathsep + env.get("PATH", "")

    with tempfile.TemporaryDirectory(prefix="molbuilder_3dna_") as workdir:
        pdb_path = os.path.join(workdir, "out.pdb")
        cmd_full = cmd + [pdb_path]
        try:
            result = subprocess.run(
                cmd_full,
                capture_output=True, text=True,
                cwd=workdir, env=env,
                timeout=60,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"fiber did not finish within 60 s -- likely hung.  "
                f"Command: {' '.join(cmd_full)}"
            ) from exc
        if result.returncode != 0 or not os.path.isfile(pdb_path):
            raise RuntimeError(
                f"fiber failed (exit {result.returncode}).\n"
                f"Command: {' '.join(cmd_full)}\n"
                f"--- stdout ---\n{result.stdout}\n"
                f"--- stderr ---\n{result.stderr}"
            )
        pdb_text = Path(pdb_path).read_text()

    if not pdb_text.strip():
        raise RuntimeError("fiber produced an empty PDB.")

    struct = parse_pdb_to_structure(
        pdb_text,
        title=title or f"{kind} {seq} (3DNA fiber, {form}-form)",
    )

    # fiber's PDB has chains 'A' (and sometimes 'B' for duplex output we
    # didn't ask for); pin to the first chain so downstream layers see
    # a single-chain Structure.
    chains = sorted(set(struct.chain_ids))
    if len(chains) > 1:
        struct = select_chain(struct, chains[0])

    # Strip the spurious 5'-terminal phosphate when the user asked for
    # a 5'-OH (terminal in OH / 3P).  Done AFTER select_chain so the
    # 5'-terminal residue is unambiguous.  See `_PHOSPHATE_ATOM_NAMES`
    # for the exact atom set we remove.
    if terminal in ("OH", "3P"):
        struct = _strip_5prime_phosphate(struct)

    err = verify_backbone_connectivity(struct, kind, max_O3_P=1.80)
    if err is not None:
        raise RuntimeError(
            f"fiber output failed connectivity check: {err}.  This is "
            f"unexpected from fiber and probably indicates a config-"
            f"file mismatch ($X3DNA={found.root})."
        )
    return struct


# --------------------------------------------------------------------- #
#  5'-terminal phosphate strip                                          #
# --------------------------------------------------------------------- #


def _strip_5prime_phosphate(struct: Structure) -> Structure:
    """Remove the 5'-terminal phosphate group from fiber's PDB output.

    fiber always emits a 5'-phosphate regardless of any flag we pass.
    For canonical 5'-OH ends (the dominant case for short oligos in
    DFT prep), strip the phosphate so the chain starts with a free
    O5'-H hydroxyl.  The bridging O5' stays -- it's part of the
    sugar; the H is added downstream by chemistry.add_hydrogens().

    The 5'-terminal residue is the first residue of the chain in
    PDB listing order.  fiber writes residues in 5'->3' order, so
    `residue_ids[0]` (paired with `chain_ids[0]`) identifies it.
    """
    if struct.residue_ids is None or struct.atom_names is None:
        # Defensive: fiber always populates these, but if a caller
        # ever feeds a stripped Structure, do nothing rather than
        # crash.
        return struct

    first_rid   = struct.residue_ids[0]
    first_chain = (struct.chain_ids[0]
                   if struct.chain_ids is not None else None)

    keep = np.ones(struct.n_atoms, dtype=bool)
    for i in range(struct.n_atoms):
        if struct.residue_ids[i] != first_rid:
            continue
        if first_chain is not None and struct.chain_ids[i] != first_chain:
            continue
        if struct.atom_names[i] in _PHOSPHATE_ATOM_NAMES:
            keep[i] = False

    if keep.all():
        return struct          # no 5'-phosphate found (already OH)

    return Structure(
        elements      = [e for k, e in zip(keep, struct.elements)      if k],
        positions     = struct.positions[keep],
        atom_names    = ([a for k, a in zip(keep, struct.atom_names)    if k]
                         if struct.atom_names    is not None else None),
        residue_ids   = ([r for k, r in zip(keep, struct.residue_ids)   if k]
                         if struct.residue_ids   is not None else None),
        residue_names = ([n for k, n in zip(keep, struct.residue_names) if k]
                         if struct.residue_names is not None else None),
        chain_ids     = ([c for k, c in zip(keep, struct.chain_ids)     if k]
                         if struct.chain_ids     is not None else None),
        title         = struct.title,
    )


# --------------------------------------------------------------------- #
#  Error message contract (see docs/design.md)                          #
# --------------------------------------------------------------------- #


def _unavailable_message() -> str:
    """Build the canonical BackendUnavailable message.

    Per docs/design.md, the message must include:
      * which preconditions were checked (in-tree / env / PATH);
      * the URL http://x3dna.org/ and an explicit "register and accept
        the license -- molbuilder cannot fetch this for you";
      * a one-line non-commercial-license reminder;
      * the names of the fallback backends (amber, rdkit).
    """
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    in_tree_glob = repo_root / "x3dna-v*"
    env_root = os.environ.get("X3DNA", "(unset)")
    fiber_path = shutil.which("fiber") or "(not on PATH)"

    return (
        "3DNA is not available.  Tried, in order:\n"
        f"  1. in-tree   : no match for {in_tree_glob}\n"
        f"                 (unpack the 3DNA tarball at the repo root and "
        f"this lights up automatically)\n"
        f"  2. $X3DNA    : {env_root}\n"
        f"                 (must point at a directory containing bin/fiber "
        f"+ config/)\n"
        f"  3. fiber on PATH: {fiber_path}\n"
        f"\n"
        "3DNA must be downloaded directly from http://x3dna.org/ after\n"
        "registering and accepting the license -- molbuilder cannot fetch\n"
        "it for you.  The license is non-commercial-use only; do not\n"
        "redistribute the archive.\n"
        "\n"
        "If you don't need a canonical helix, the `amber` (extended chain)\n"
        "and `rdkit` (folded conformer) backends remain available."
    )
