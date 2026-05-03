"""3DNA backend (uses ``fiber``).

3DNA's ``fiber`` produces canonical B-form / A-form / Z-form helical
geometry from sequence -- the only thing the existing ``rdkit``
(folded conformer) and ``amber`` (extended chain) backends do not
provide.

Detection chain (first hit wins):

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

from ...structure import Structure
from ._common import (parse_pdb_to_structure, select_chain,
                      verify_backbone_connectivity)


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

    if terminal not in ("OH",):
        import warnings
        warnings.warn(
            f"3DNA backend currently emits the fiber-default terminus "
            f"(5'-OH / 3'-OH); requested terminal={terminal!r} ignored.  "
            f"For phosphorylated termini, post-process the .pdb by hand.",
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

    err = verify_backbone_connectivity(struct, kind, max_O3_P=1.80)
    if err is not None:
        raise RuntimeError(
            f"fiber output failed connectivity check: {err}.  This is "
            f"unexpected from fiber and probably indicates a config-"
            f"file mismatch ($X3DNA={found.root})."
        )
    return struct


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
