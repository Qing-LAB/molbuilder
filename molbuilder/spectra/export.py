"""THE EXPORT OF A VIBRATION RESULT -- its data files as their writers wrote
them, and the Raman view at a laser composed now, as files or as their tables
(`engines/vibration.md` § 5.11 § 10, *The export of a vibration
calculation*).

MODULE  spectra.export (L4)
ROLE    :func:`export` -- the one composition the spectrum viewer's Export
        (`GET /api/spectra/export`) and `molbuilder jobset export` ask alike,
        on the calculation as the transport export takes it, in the export's
        one shape (`ncdata.exported`): ``<label>.spectra.nc`` and, on a
        SIESTA run with an optical phase, ``<label>.optical.nc``, each copied
        as it is and read first through its reader; when a laser is named,
        ``<label>.raman.nc``, the view (`spectra.raman`) composed now from
        the same bytes.  :func:`result_path` -- the result an export is of:
        the one in the folder the viewer shows, else the calculation's
        stage's newest run's.  **Nothing is written into the calculation**,
        and the ledger has no line of it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from ..ncdata import ExportError, Exported
from .raman import ASK_KEYS

#: What a vibration export asks beyond its form: the Raman view's own asks.
ASKS = ASK_KEYS


def _stem(calc) -> str:
    return f"{calc.task.label}.vibration"


def name_of(calc, fmt: str) -> str:
    """What an export of ``calc`` as ``fmt`` is named -- the folder its files
    lie in (`ncdata.export_name`), known before anything is composed."""
    from ..ncdata import export_name
    return export_name(_stem(calc), fmt)


def result_path(calc, folder=None) -> Path:
    """THE VIBRATION RESULT AN EXPORT OF ``calc`` IS OF -- the one in
    ``folder`` where it holds one (a run's own folder, or a run of points'
    root: the run the viewer shows, `runs.on_label_at`), else the
    calculation's, its stage's newest run's (`materialize.newest_on_label`).
    Refused by name: a calculation of another kind or not yet launched, a
    displacement sweep -- its several stages each hold a result -- and a
    run with no result yet."""
    from ..jobset.materialize import newest_on_label
    from ..jobset.naming import stage_home
    from ..runfiles import ROLE_SPECTRA
    from ..runs import on_label_at
    if folder is not None:
        got = on_label_at(Path(folder), ROLE_SPECTRA)
        if got is not None:
            return got
    task, name = calc.task, Path(calc.base).name
    if task.calculation != "vibration":
        raise ExportError(f"{name} is a {task.calculation} calculation, not "
                          f"a vibration one")
    if len(task.stages) != 1:
        raise ExportError(
            f"{name} sweeps the displacement over {len(task.stages)} stages, "
            f"each holding its own result: an export is of one result -- "
            f"export it from the spectrum viewer, which opens one stage's run")
    attempt, got = newest_on_label(
        stage_home(calc.base, task, task.stages[0].name).dir, ROLE_SPECTRA)
    if attempt is None:
        raise ExportError(f"{name} has no run yet -- nothing to export")
    if got is None:
        raise ExportError(f"{name}'s run holds no result yet "
                          f"(<label>{ROLE_SPECTRA}) -- its finish writes it")
    return got


def export(calc, fmt: str, *, folder=None, laser_nm: Optional[float] = None,
           broaden_ev: Optional[float] = None) -> Exported:
    """The vibration result of ``calc`` (:func:`result_path`, ``folder`` the
    run the viewer shows), as ``fmt``: its data files as their writers wrote
    them -- each read once, through its reader -- and, when ``laser_nm`` is
    named, the Raman view at it (`spectra.raman.raman_view_of`) from the
    same bytes, as ``<label>.raman.nc``; ``csv``: each file's tables and
    ``README.txt`` in a folder of its own.  On a SIESTA result with optical
    calls and no laser named, ``said`` says the view is not in it.
    :class:`ncdata.ExportError` names a form that is not one, a width named
    with no laser, a laser named for a result whose Raman is not the optical
    route, the view's own refusals, and a data file that does not read -- in
    its reader's words."""
    from ..ncdata import export_format, exported
    from ..runfiles import ROLE_SPECTRA_DATA, compose
    from ..sidecars.spectra import (SpectraJsonError,
                                    parse_spectra_json)
    from .datafile import read_spectra_data
    from .optical_data import read_optical_data
    from .raman import (RAMAN_FILE, has_optical_route, optical_data_of,
                        raman_data_bytes, raman_view_of, read_ask)
    export_format(fmt)
    asked = None
    if laser_nm is not None or broaden_ev is not None:
        try:
            asked = read_ask({"laser_nm": laser_nm, "broaden_ev": broaden_ev})
        except ValueError as exc:
            raise ExportError(str(exc)) from None
    path = result_path(calc, folder)
    label = calc.task.label
    try:
        result = parse_spectra_json(path)
    except SpectraJsonError as exc:
        raise ExportError(f"{path.name}: {exc}") from exc
    # THE DATA FILES AS THEIR WRITERS WROTE THEM, each read once: its bytes
    # are what is copied, what its reader reads, and what the view hashes.
    spectra_nc = path.with_name(compose(label, ROLE_SPECTRA_DATA))
    optical = has_optical_route(result)
    try:
        sources = [spectra_nc] + ([optical_data_of(path, result)]
                                  if optical else [])
    except ValueError as exc:
        raise ExportError(str(exc)) from exc
    files: Dict[str, bytes] = {}
    for source in sources:
        if not source.is_file():
            raise ExportError(f"{path.name}'s data file {source.name} is not "
                              f"beside it")
        try:
            files[source.name] = source.read_bytes()
        except OSError as exc:
            raise ExportError(f"{source.name}: {exc}") from exc
    try:
        read_spectra_data(spectra_nc, data=files[spectra_nc.name])
        if optical and asked is None:
            read_optical_data(sources[1], data=files[sources[1].name])
    except (OSError, ValueError) as exc:
        raise ExportError(str(exc)) from exc
    said = ""
    if asked is not None:
        try:
            view = raman_view_of(
                path, result=result, **asked,
                optical_bytes=files[sources[1].name] if optical else None)
        except (ValueError, OSError) as exc:
            raise ExportError(str(exc)) from exc
        name, data = raman_data_bytes(view, label=label)
        files[name] = data
    elif optical:
        said = (f"the Raman view is not in it: its strengths depend on the "
                f"laser -- name one (the page's laser box, `--laser-nm`) to "
                f"add {label}{RAMAN_FILE}")
    return exported(files, fmt, stem=_stem(calc), said=said)


__all__ = ["ASKS", "export", "name_of", "result_path"]
