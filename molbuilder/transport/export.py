"""THE EXPORT OF A TRANSPORT CALCULATION -- every result it holds, as it
stands when asked, as its data file or as that file's tables
(`engines/transport.md` § 2a.12, *Export*).

MODULE  transport.export (L4)
ROLE    :func:`export` -- the one composition the transport report's Export
        (`GET /api/transport/export`) and `molbuilder jobset export` ask
        alike: the record composed now, as the Results tab reads a ladder
        in progress (`record.collect_record`, ``partial``), the data file's
        bytes from its own writer (`datafile.data_file_bytes`), and for CSV
        its tables and README (`ncdata.as_tables`) -- **what is already
        there and nothing more**: no data file is read and nothing is
        written into the calculation's folder; a point whose DOS per
        orbital is not kept exports without its DOS by region, and the
        export says how many and how to read them (:func:`missing_dos_said`;
        the user, 2026-10-23: "the export should only contain information
        that is already processed").
"""
from __future__ import annotations

from typing import Dict

from ..ncdata import ExportError, Exported

#: What a transport export asks beyond its form: nothing.
ASKS: tuple = ()


def _data_file_name(calc) -> str:
    from .datafile import data_file_path
    return data_file_path(calc.base, calc.task.label).name


def name_of(calc, fmt: str) -> str:
    """What an export of ``calc`` as ``fmt`` is named -- the data file
    itself, or its tables' folder (`ncdata.export_name`) -- known before
    anything is composed."""
    from ..ncdata import export_name
    name = _data_file_name(calc)
    return export_name(name[:-len(".nc")], fmt, name)


def missing_dos_said(record: Dict) -> str:
    """How many finished points hold no DOS by region -- their DOS per
    orbital not kept (`model/parse.md` § 5e) -- and how to read them; ``""``
    when every one is kept."""
    missing = [p for p in record.get("points") or ()
               if not all(d.get("state") == "kept"
                          for d in p.get("digest") or ())]
    if not missing:
        return ""
    n = len(missing)
    return (f"{n} finished point{'s' if n != 1 else ''} export"
            f"{'' if n != 1 else 's'} without {'their' if n != 1 else 'its'} "
            f"DOS by region: {'their' if n != 1 else 'its'} DOS per orbital "
            f"is not read yet -- read it from the report's DOS tab (Read the "
            f"DOS per orbital), or `jobset summarize task`, then export again")


def export(calc, fmt: str) -> Exported:
    """Every result of the transport calculation ``calc`` (`calculation.open`)
    as it stands, as ``fmt`` -- ``netcdf``: the data file, ``{name: bytes}``;
    ``csv``: its tables and ``README.txt`` -- from what is already there, no
    data file read (:func:`missing_dos_said` in ``said``).
    :class:`ExportError` names a form that is not one, a calculation that is
    not transport, and results that do not read or do not lay out on one
    grid -- in their owners' words, as the record's route says them."""
    from ..ncdata import export_format, exported
    from .compose import ComposeError
    from .datafile import DataFileError, data_file_bytes
    from .record import RecordError, collect_record, not_transport
    export_format(fmt)
    other = not_transport(calc)
    if other:
        raise ExportError(other)
    try:
        # WHAT EXISTS, as the Results tab reads a ladder in progress: a
        # point not done is the fill value (`done` 0).
        record = collect_record(calc, partial=True)
        name, data = data_file_bytes(calc.base, calc.task, record)
    except (RecordError, ComposeError, DataFileError, OSError,
            ValueError) as exc:
        raise ExportError(str(exc)) from exc
    return exported({name: data}, fmt, stem=name[:-len(".nc")],
                     one_file=True, said=missing_dos_said(record))


__all__ = ["ASKS", "export", "missing_dos_said", "name_of"]
