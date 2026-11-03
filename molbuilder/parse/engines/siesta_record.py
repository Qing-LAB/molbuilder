"""WHAT A SIESTA-FAMILY OUTPUT ANSWERS OF THE RUN RECORD -- its build,
launch, solver and pseudopotential lines, and TBtrans's loop, read through
the family's table: the reader the ``.out``'s catalogue row names for
``record`` (`runfiles.WRITTEN`; `model/parse.md` § 5d.1b), asked by the run
record's table (`parse.dirs.record.CONTRIBUTORS`), which names the fields it
answers.
"""
from __future__ import annotations

from typing import Any, Dict


def out_fields(f) -> Dict[str, Any]:
    """The ``.out``'s build, launch, solver and pseudopotential lines, through
    its family's table (`siesta_grammar`), and a program's loop through its
    reading pass (`tbtrans.loop_record`), said with the run's state -- over
    once the run is in a final state (``f.over``).  ``f`` is the run's files
    as the run record found them (`parse.dirs.record.RunFiles`)."""
    from . import siesta_grammar as _G
    head = f.stdout_head
    if not head:
        return {}
    build: Dict[str, Any] = {}
    launch: Dict[str, Any] = {}
    solver: Dict[str, Any] = {}
    lines = head.splitlines()
    for line in lines:
        (_G.read_build_line(line, build) or _G.read_launch_line(line, launch)
         or _G.read_diag_line(line, solver))
    tail = f.tail(f.stdout)
    for line in tail.splitlines():
        _G.read_launch_line(line, launch)
    out: Dict[str, Any] = {"computation": {}}
    comp = out["computation"]
    if build:
        exe = str(build.get("executable", "")).rsplit("/", 1)[-1]
        comp["engine"] = {**({"program": exe} if exe else {}),
                          **({"version": build["version"]}
                             if "version" in build else {}),
                          "build": {k: v for k, v in build.items()
                                    if k not in ("executable", "version")}}
    if solver:
        comp["solver"] = solver
    if "n_mpi_processes" in launch:
        comp["launch"] = {"ranks": launch["n_mpi_processes"]}
    time = {k: launch[k] for k in ("run_start_local", "run_end_local")
            if k in launch}
    from .tbtrans import loop_record
    time.update(loop_record(
        f.stdout, head=head,
        # OVER only when the run's state is final: an `unreadable` run's
        # loop is not said stopped -- its state cannot be told.
        over=f.over))
    if time:
        comp["time"] = time
    pseudos = []
    for sp in _G.read_psml_lines(lines):
        entry = dict(sp)
        path = f.directory / sp["file"]
        if path.is_file():
            from ...persist import sha256_of
            entry["sha256"] = sha256_of(path)
            from ...pseudos import parse_psml_header
            info = parse_psml_header(path)
            for key in ("xc_family", "xc_authors", "relativistic",
                        "generator"):
                val = getattr(info, key, None)
                if val and val != "unknown":
                    entry[key] = val
        pseudos.append(entry)
    if pseudos:
        out["setup"] = {"pseudopotentials": pseudos}
    return out
