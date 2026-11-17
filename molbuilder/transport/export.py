"""THE EXPORT OF A TRANSPORT CALCULATION -- every result it holds, as it
stands when asked, as its data file or as that file's tables
(`engines/transport.md` § 2a.12, *Export*).

MODULE  transport.export (L4)
ROLE    :func:`export` -- the one composition the Results tab's Export
        (`GET /api/export`) and `molbuilder jobset export` ask
        alike: the record composed now, as the Results tab reads a ladder
        in progress (`record.collect_record`, ``partial``), the data file's
        bytes from its own writer (`datafile.data_file_bytes`), and for CSV
        its tables and README (`ncdata.as_tables`) -- **what is already
        there and nothing more**: no data file is read and nothing is
        written into the calculation's folder; a point whose DOS per
        orbital is not kept exports without its DOS by region, and the
        export says how many and how to read them (:func:`missing_dos_said`;
        the user, 2026-10-23: "the export should only contain information
        that is already processed"); and, asked (``dos_per_orbital``), the
        NetCDF form with each point's kept digests beside the data file --
        their bytes copied, never read -- what `postprocess.pdos` reads a
        PDOS of chosen atoms from, anywhere.
"""
from __future__ import annotations

from typing import Dict

from ..ncdata import ExportError, Exported


def _data_file_name(calc) -> str:
    from .datafile import data_file_path
    return data_file_path(calc.base, calc.task.label).name


def _stem(calc, name: str) -> str:
    """An export's stem, ``<label>.<kind>``: the label the data file is
    named on, and the calculation's kind -- which `ncdata.export_name` gives
    each format's ending (``<label>.transport-csv``)."""
    from ..runfiles import ROLE_TRANSPORT_DATA, stem_before
    return f"{stem_before(name, ROLE_TRANSPORT_DATA)}.{calc.task.calculation}"


def name_of(calc, fmt: str, *, dos_per_orbital: bool = False) -> str:
    """What an export of ``calc`` as ``fmt`` is named -- the data file
    itself, or the folder its files lie in: its tables', or the data file's
    with each point's digest (``dos_per_orbital``; `ncdata.export_name`) --
    known before anything is composed."""
    from ..ncdata import export_name
    name = _data_file_name(calc)
    return export_name(_stem(calc, name), fmt,
                       None if dos_per_orbital else name)


def missing_dos_said(record: Dict) -> str:
    """How many finished points hold no DOS by region -- their DOS per
    orbital not kept (`model/parse.md` § 5e) -- and how to read them; ``""``
    when every one is kept."""
    from .record import readings_kept
    # A POINT WITH A DATA FILE whose reading is not whole -- one with none
    # has nothing to read.
    missing = [p for p in record.get("points") or ()
               if p.get("digest") and not readings_kept(p["digest"])]
    if not missing:
        return ""
    n = len(missing)
    from .tbtnc import READ_WAY
    return (f"{n} finished point{'s' if n != 1 else ''} export"
            f"{'' if n != 1 else 's'} without {'their' if n != 1 else 'its'} "
            f"DOS by region: {'their' if n != 1 else 'its'} DOS per orbital "
            f"is not read yet -- {READ_WAY}, then export again")


def export(calc, fmt: str, *, folder=None,
           dos_per_orbital: bool = False) -> Exported:
    """Every result of the transport calculation ``calc`` (`calculation.open`)
    as it stands, as ``fmt`` -- ``netcdf``: the data file, ``{name: bytes}``;
    ``csv``: its tables and ``README.txt`` -- from what is already there, no
    data file read (:func:`missing_dos_said` in ``said``).
    :class:`ExportError` names a form that is not one, and results that do
    not read or do not lay out on one grid -- in their owners' words, as
    the record's route says them; which kind's export runs is the kind
    table's (`kind_table.export_of`).
    ``dos_per_orbital``: the NetCDF form with each point's kept digest beside
    it, at the path the point names, its bytes copied and never read -- what
    `postprocess.pdos` reads a PDOS of chosen atoms from, anywhere; asked
    with that form alone, its row's (`kind_table.export_asks_of`)."""
    from ..ncdata import export_format, exported
    from .compose import ComposeError
    from .datafile import DataFileError, data_file_bytes
    from .record import RecordError, collect_record
    export_format(fmt)
    try:
        # WHAT EXISTS, as the Results tab reads a ladder in progress: a
        # point not done is the fill value (`done` 0).
        record = collect_record(calc, partial=True)
        name, data = data_file_bytes(calc, record)
    except (RecordError, ComposeError, DataFileError, OSError,
            ValueError) as exc:
        raise ExportError(str(exc)) from exc
    files = {name: data}
    if dos_per_orbital:
        # EACH POINT'S DIGESTS, at the paths the data file names -- the
        # record's one decision of which a point names (`collect_record`).
        from pathlib import Path
        try:
            for p in record.get("points") or ():
                for src in (p.get("dos") or {}).get("per_orbital") or ():
                    files[src["digest"]] = (
                        Path(calc.base) / src["digest"]).read_bytes()
        except OSError as exc:
            raise ExportError(str(exc)) from exc
    return exported(files, fmt, stem=_stem(calc, name),
                    one_file=not dos_per_orbital,
                    said=missing_dos_said(record))


__all__ = ["export", "missing_dos_said", "name_of"]
