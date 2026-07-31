"""molbuilder.bundle_writer — materialize a BundleResult to disk.

H2 of parse-module.md migration: absorbed from the legacy
:func:`molbuilder.script_bundle.write_bundle_as_handoff`.  The
read-side bundle assembler lives in
:mod:`molbuilder.parse.dirs.bundle` (returns a
:class:`molbuilder.parse.types.BundleResult`); this module is the
sole supported way to write that bundle back out as a handoff pair
the next workflow stage can pick up.

Output layout (per ``docs/execution/job-contracts.md``)::

    <target_dir>/<stem>.xyz             # geometry + element + free-text comment
    <target_dir>/<stem>.molstruct.json  # regions + frozen_atoms + structure_hash

The next stage's ``.xyz`` load path (build / spectra / transport)
picks the pair up unchanged: it reads the XYZ for structure, then
the sidecar loader picks up the labels.  No new load primitive
needed downstream.

Atomic writes (tmp + rename + fsync) for both files individually.
If the .xyz lands but the sidecar fails partway through, the caller
is left with a labeled-but-no-sidecar XYZ — recoverable + a
diagnostic note is added downstream.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Tuple

from molbuilder.parse.types import BundleResult
from molbuilder.sidecars import molstruct as _msj


class BundleWriteError(Exception):
    """Raised when materialization can't proceed (e.g. target file
    exists and ``overwrite=False``)."""


def write_bundle_as_handoff(bundle: BundleResult,
                            target_dir: Path,
                            *,
                            stem: str,
                            overwrite: bool = False
                            ) -> Tuple[Path, Path]:
    """Materialize ``bundle`` as ``<target_dir>/<stem>.xyz`` +
    ``<target_dir>/<stem>.molstruct.json``.

    Parameters
    ----------
    bundle
        :class:`BundleResult` from
        :meth:`molbuilder.parse.dirs.bundle.BundleDirParser.parse`.
    target_dir
        Destination directory; created if it doesn't exist.
    stem
        Output file basename.  ``<stem>.xyz`` + ``<stem>.molstruct.json``
        are written next to each other.
    overwrite
        Default False — raise :class:`BundleWriteError` when either
        destination file already exists.  Pass True to replace.

    Returns
    -------
    (xyz_path, sidecar_path)
        Both as resolved :class:`Path` objects.

    Raises
    ------
    BundleWriteError
        Either destination exists and ``overwrite=False``.
    OSError
        Filesystem write failure (permission, no such directory, disk
        full).  The .tmp file is best-effort cleaned up.
    """
    if bundle.structure is None:
        raise BundleWriteError(
            "bundle.structure is None; nothing to materialize.  The "
            "assembler should have populated structure from .XV / "
            ".fdf-initial / py-opt / py-initial — check the upstream "
            "BundleDirParser result.")
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    xyz_path     = target_dir / f"{stem}.xyz"
    sidecar_path = _msj.sidecar_path_for(xyz_path)
    if not overwrite:
        for p in (xyz_path, sidecar_path):
            if p.exists():
                raise BundleWriteError(
                    f"target exists: {p}.  Pass overwrite=True to "
                    f"replace, or pick a different stem/target_dir.")

    # Provenance line on the .xyz so a curious user reading the XYZ
    # comment can trace the bundle back to its source.  Falls back to
    # the bare source path when source_engine / final_coords_from
    # weren't recorded (older BundleResults pre-H2).
    # Prefer the source-script basename (h2.fdf, opt.py) when the
    # parser surfaced it; fall back to the run-dir name on older
    # BundleResults that pre-date the source_script field.
    if bundle.source_script:
        src_label = Path(bundle.source_script).name
    elif bundle.source:
        src_label = Path(bundle.source).name
    else:
        src_label = "<unknown>"
    if bundle.source_engine and bundle.final_coords_from:
        comment = (
            f"bundled from {src_label} "
            f"({bundle.source_engine}/{bundle.final_coords_from})"
        )
    else:
        comment = f"bundled from {src_label}"
    xyz_text = bundle.structure.to_xyz(comment=comment)
    _atomic_write_text(xyz_path, xyz_text)

    # structure_hash is the SHA-256 of the XYZ file BYTES (per
    # sidecars.molstruct.sha256_of_file); compute from the freshly-
    # written file so the invariant pin is what the next-stage
    # loader will check.
    structure_hash = _msj.sha256_of_file(xyz_path)
    # The parsed bundle keeps its own `frozen_atoms` list (it is a RESULTS
    # record of what a run held still, not the structure's label store), so it
    # is written where a label belongs: into `regions`, with the rest.
    from molbuilder.structure import FROZEN_LABEL
    labels = dict(bundle.regions or {})
    if bundle.frozen_atoms:
        labels[FROZEN_LABEL] = list(bundle.frozen_atoms)
    sidecar_payload = _msj.to_dict(
        {
            "regions":      labels or None,
            "cell":         bundle.structure.cell,
            "pbc":          bundle.structure.pbc,
        },
        n_atoms_total=len(bundle.structure.elements),
        structure_hash=structure_hash,
        created_by="molbuilder bundle_writer",
    )
    _msj.save(sidecar_path, sidecar_payload)
    return (xyz_path, sidecar_path)


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (tmp + rename) so a
    partial write never leaves a half-XYZ on disk.

    Tmp filename is unique per writer (PID + monotonic-ns) so two
    concurrent writers to the same final path don't collide on the
    same tmp file (a deterministic ``<path>.tmp`` would let writer B's
    ``open(...)`` truncate writer A's in-flight bytes before either
    finishes, and the ``os.replace`` would commit a mix).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(
        path.suffix + f".{os.getpid()}.{time.monotonic_ns()}.tmp"
    )
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                # Some filesystems (tmpfs on certain kernels) reject
                # fsync — the data is still in the OS buffer + will
                # land before the replace.  Don't let a quirky FS
                # block the write.
                pass
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


__all__ = [
    "BundleWriteError",
    "write_bundle_as_handoff",
]
