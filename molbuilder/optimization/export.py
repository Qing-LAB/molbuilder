"""THE EXPORT OF AN OPTIMIZATION -- every step its runs finished, as it stands
when asked, as its data file or as that file's tables
(`engines/optimization-postprocessing.md` § 6).

MODULE  optimization.export (L4)
ROLE    :func:`export` -- what `molbuilder jobset export` asks (`kind_table`,
        ``export``): the data file composed now from what the runs hold
        (`datafile.data_file_bytes`), and for CSV its tables and README
        (`ncdata.as_tables`).  **Nothing is written into the calculation.**
"""
from __future__ import annotations

from ..ncdata import ExportError, Exported

#: What an optimization's export asks beyond its form: nothing.
ASKS: tuple = ()


def _stem(calc) -> str:
    """An export's stem, ``<label>.<kind>``, which `ncdata.export_name` gives
    each format's ending (``<label>.optimization-csv``)."""
    return f"{calc.task.label}.{calc.task.calculation}"


def name_of(calc, fmt: str) -> str:
    """What an export of ``calc`` as ``fmt`` is named -- the data file itself,
    or its tables' folder -- known before anything is composed."""
    from ..ncdata import export_name
    from .datafile import data_file_name
    return export_name(_stem(calc), fmt, data_file_name(calc))


def export(calc, fmt: str) -> Exported:
    """Every step the optimization ``calc`` (`calculation.open`) has finished,
    as ``fmt`` -- ``netcdf``: the data file, ``{name: bytes}``; ``csv``: its
    tables and ``README.txt`` -- the runs that give no step said, and why.
    :class:`ExportError` names a form that is not one, and what does not read
    or does not lay out on one grid, in its owner's words."""
    from ..ncdata import export_format, exported
    from .datafile import data_file_bytes
    export_format(fmt)
    try:
        name, data, said = data_file_bytes(calc)
    except (OSError, ValueError) as exc:
        raise ExportError(str(exc)) from exc
    return exported({name: data}, fmt, stem=_stem(calc), one_file=True,
                    said=said)


__all__ = ["ASKS", "export", "name_of"]
