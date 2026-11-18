"""WHAT ONE CHART DRAWS, AS A DATA FILE -- the chart table read once and checked
whole, the drawn curves a page sends laid out by it through the data-file
framework, and the file read back (`model/datafiles.md` § 5b, § 9.7;
`web/results.md` § 1.2).

MODULE  chartdata (L2; the data-file framework, the package's data folder, the
        kind table's one door to a part a record names)
ROLE    :func:`table` -- ``data/chart_table.toml`` read once: each chart's
        words and the tags and marks it may draw, each tag's curve (its ``y``
        along its ``x``) or threshold (its ``mark``) -- a variable its owner
        declares, the owner's statement whole (a data file's table, the
        calculation's engine's SCF columns, an axis's row), or, where no file
        declares one, the table's own -- refused as it loads where a row
        breaks the framework's rules; :func:`export` -- what a page sends of
        one chart (its curves by tag, its thresholds, the choices it was drawn
        with) composed now as ``<label>.<chart>.nc`` or its tables, in the
        export's one shape (`ncdata.exported`); :func:`read_data_file` -- the
        file back, refusing another schema by name.  It knows no chart: a
        page names the chart and the tags, the table says what they are.
        **Nothing is written into the calculation.**
USED-BY web.blueprints.results (``POST /api/export/chart``), postprocess
        (``read_chart_data``)
"""
from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import data_dir
from .ncdata import ExportError, Var, check_said

#: The table every chart's drawn data is laid out by.
TABLE_FILE = data_dir() / "chart_table.toml"

#: Which layout this is, at which version (`model/datafiles.md` § 2b).
DATA_SCHEMA = "molbuilder/chart-data@1"

#: THE ROOT'S ATTRIBUTES THAT HOLD RECORDS -- what decided the chart's
#: numbers -- written as JSON text and read back as the blocks they hold
#: (`ncdata.json_attrs`, `ncdata.read_file`).
RECORD_ATTRS = ("chosen",)

#: THE MOST CELLS ONE TAG'S CURVES TAKE, padded to the longest: a chart
#: draws thousands of points on tens of curves; a body that would pad past
#: this is no chart's, and is refused before the padded grid is made.
MAX_CELLS = 10_000_000

#: The keys each kind of row may hold (`persist.unknown_keys`).
_CHART_KEYS = ("words", "tags", "marks")
_TAG_KEYS = ("x", "y", "mark")
_REF_KEYS = ("of", "name")
_OWN_KEYS = ("name", "kind", "units", "long_name", "definition", "source")
_X_OWN_KEYS = ("units", "long_name", "definition")

#: The axes of one tag's group: its curves, and each curve's points.
_CURVE_VARS = (
    Var("curve", ("curve",), "coordinate", "1", "a curve",
        "from 0, in the order the chart drew it", dtype="i4"),
    Var("curve_point", ("curve_point",), "coordinate", "1", "a curve's point",
        "from 0, in the order the chart drew it; a curve of fewer points "
        "padded with the fill", dtype="i4"),
    Var("curve_name", ("curve",), "given", "", "the curve's legend name",
        "as the chart showed it", dtype=str, auxiliary=True),
    Var("curve_hidden", ("curve",), "state", "1",
        "whether the legend hid the curve", "1 hidden, 0 shown",
        dtype="i1", flag=True),
)
_CURVE = ("curve", "curve_point")


class ChartTableError(ValueError):
    """The chart table breaks a rule -- the message names the row."""


class ChartDataError(ValueError):
    """A file that is not a chart's drawn data of this schema."""


@dataclass(frozen=True)
class Tag:
    """A kind of curve -- ``y`` along ``x`` -- or of threshold line --
    ``mark``: each a row's statement, resolved into its variable as a file
    is composed (:func:`_variable`)."""
    y: Optional[Mapping[str, Any]]
    x: Optional[Mapping[str, Any]]
    mark: Optional[Mapping[str, Any]]


@dataclass(frozen=True)
class Chart:
    """A chart that exports: its words, the tags and marks it may draw."""
    words: str
    tags: Tuple[str, ...]
    marks: Tuple[str, ...]


@dataclass(frozen=True)
class Table:
    """The chart table: each chart, each tag."""
    charts: Mapping[str, Chart]
    tags: Mapping[str, Tag]


def _refuse(where: str, why: str):
    return ChartTableError(f"{TABLE_FILE.name} {where}: {why}")


def _keys(where: str, row: Mapping[str, Any], allowed) -> None:
    from .persist import unknown_keys
    said = unknown_keys(row, allowed)
    if said:
        raise _refuse(where, said)


def _variable(spec: Mapping[str, Any], dims: Tuple[str, ...], *,
              engine: Optional[str] = None, kind: Optional[str] = None,
              where: str = "") -> Var:
    """THE VARIABLE A ROW STATES, over ``dims``: its owner's declaration
    whole -- a data file's table (``of = "<module:NAME>"``), the
    calculation's ``engine``'s SCF column (``of = "scf"``), an axis's row
    (``of = "axis"``) -- or the row's own; ``kind`` an x's, ``given``."""
    from .kind_table import part
    from .ncdata import AXES, coordinate
    of = spec.get("of")
    name = str(spec.get("name", ""))
    if of is None:
        missing = [k for k in ("name", "units", "long_name", "definition")
                   + (() if kind else ("kind",)) if k not in spec]
        if missing:
            raise _refuse(where, f"it states no {', '.join(missing)}")
        # AN OWN VARIABLE IN UNITS "" is a category's labels: text.
        var = Var(name, dims, kind or str(spec["kind"]), str(spec["units"]),
                  str(spec["long_name"]), str(spec["definition"]),
                  source=str(spec.get("source", "")),
                  dtype=str if str(spec["units"]) == "" else "f8")
    elif of == "axis":
        if AXES.get(name) is None:
            raise _refuse(where, f"{name!r} is no axis of ncdata.AXES")
        var = coordinate(name)
    elif of == "scf":
        # THE ENGINE'S SCF COLUMN, through the framework's one door to an
        # SCF history's variables (`ncdata.scf_history_vars`).
        from .engine_table import engines, scf_columns
        from .ncdata import scf_history_vars
        found = [v for n in ([engine] if engine else engines())
                 for v in scf_history_vars(scf_columns(n), (), whose="",
                                           source="the run's own output")
                 if v.name == name]
        if not found:
            raise ExportError(f"{name} is no SCF column the engine "
                              f"{engine or 'of any engine'} declares")
        var = found[0]
    else:
        owned = [v for v in part(str(of), record=where, key="of")
                 if v.name == name]
        if not owned:
            raise _refuse(where, f"{of} declares no variable {name!r}")
        var = owned[0]
    var = replace(var, dims=dims, kind=kind or var.kind, earlier=())
    if var.kind == "coordinate":
        var = replace(var, kind="given")
    try:
        check_said(var)
    except ValueError as exc:
        raise _refuse(where, str(exc)) from exc
    return var


@lru_cache(maxsize=1)
def table() -> Table:
    """``data/chart_table.toml``, read once and checked whole: every key
    known, every chart's tags and marks rows of the right kind, every row's
    variable one the one writer takes, no two curves of one chart one
    variable."""
    raw = tomllib.loads(TABLE_FILE.read_text(encoding="utf-8"))
    _keys("", raw, ("charts", "xs", "tags"))
    xs = raw.get("xs", {})
    tags: Dict[str, Tag] = {}
    for key, row in raw.get("tags", {}).items():
        where = f"[tags.{key}]"
        _keys(where, row, _TAG_KEYS)
        y = x = mark = None
        if "y" in row:
            x_key = row.get("x")
            if x_key not in xs:
                raise _refuse(where, f"its x {x_key!r} is no [xs] row")
            x = {"name": x_key, **xs[x_key]}
            _keys(f"[xs.{x_key}]", xs[x_key],
                  _REF_KEYS if "of" in xs[x_key] else _X_OWN_KEYS)
            y = dict(row["y"])
            _keys(where + " y", y, _REF_KEYS if "of" in y else _OWN_KEYS)
            _variable(x, _CURVE, kind="given", where=f"[xs.{x_key}]")
            _variable(y, _CURVE, where=where + " y")
        if "mark" in row:
            mark = dict(row["mark"])
            _keys(where + " mark", mark,
                  _REF_KEYS if "of" in mark else _OWN_KEYS)
            _variable(mark, (), where=where + " mark")
        if y is None and mark is None:
            raise _refuse(where, "neither a curve (y) nor a threshold (mark)")
        tags[str(key)] = Tag(y=y, x=x, mark=mark)
    charts: Dict[str, Chart] = {}
    for key, row in raw.get("charts", {}).items():
        where = f"[charts.{key}]"
        _keys(where, row, _CHART_KEYS)
        if not row.get("words"):
            raise _refuse(where, "it states no words")
        chart = Chart(words=str(row["words"]),
                      tags=tuple(map(str, row.get("tags", ()))),
                      marks=tuple(map(str, row.get("marks", ()))))
        for t in chart.tags:
            if t not in tags or tags[t].y is None:
                raise _refuse(where, f"its tag {t!r} is no curve's row")
        for t in chart.marks:
            if t not in tags or tags[t].mark is None:
                raise _refuse(where, f"its mark {t!r} is no threshold's row")
        ys = [tags[t].y["name"] for t in chart.tags]
        if len(set(ys)) != len(ys):
            raise _refuse(where, "two of its tags are one variable")
        charts[str(key)] = chart
    return Table(charts=charts, tags=tags)


def _labels(values: Any, what: str) -> List[str]:
    """``values`` as labels -- a category axis's (a variable in units ``""``)
    -- a gap (``None``) the empty label."""
    if not isinstance(values, (list, tuple)):
        raise ExportError(f"{what} is not a list")
    return ["" if v is None else str(v) for v in values]


def _numbers(values: Any, what: str) -> List[float]:
    """``values`` as numbers, a gap (``None``) the fill -- refused as
    ``what`` where one is not a finite number's spelling."""
    if not isinstance(values, (list, tuple)):
        raise ExportError(f"{what} is not a list of numbers")
    out = []
    for v in values:
        if v is None:
            out.append(math.nan)
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ExportError(f"{what} holds {v!r}, which is not a number")
        try:
            out.append(float(v))
        except OverflowError:
            raise ExportError(f"{what} holds {v!r}, too large a number") from None
    return out


def _record(item: Any, what: str) -> Mapping[str, Any]:
    if not isinstance(item, Mapping):
        raise ExportError(f"{what} is not a record")
    return item


def data_file_bytes(calc, path: Path, *, chart: str,
                    chosen: Mapping[str, Any],
                    curves: Sequence[Mapping[str, Any]],
                    marks: Sequence[Mapping[str, Any]]) -> bytes:
    """``<label>.<chart>.nc`` of what the page drew of ``chart`` from
    ``path`` in the opened calculation ``calc`` (`model/datafiles.md`
    § 9.7): a group per tag, named by its variable, each curve of it a row;
    each threshold a variable on the root.  :class:`ncdata.ExportError`
    names a chart the table does not, a tag or mark the chart does not draw,
    a curve whose x and y do not pair, two thresholds of one tag, a chart
    that drew nothing, and curves too many to be a chart's."""
    from . import __version__
    from .ncdata import (NC_LOCK, as_bytes, empty, in_memory, json_attrs,
                         state, write_group)
    got = table().charts.get(chart)
    if got is None:
        raise ExportError(f"{chart!r} is no chart the chart table names "
                          f"(data/chart_table.toml): "
                          f"{', '.join(table().charts)}")
    engine = str(calc.task.engine)
    x_of: Dict[str, Var] = {}
    by_tag: Dict[str, List[Dict[str, Any]]] = {}
    for k, c in enumerate(curves):
        c = _record(c, f"curve {k}")
        tag = str(c.get("tag"))
        if tag not in got.tags:
            raise ExportError(f"the {chart} chart draws no curve tagged "
                              f"{tag!r}: its curves are "
                              f"{', '.join(got.tags)}")
        if tag not in x_of:
            row = table().tags[tag]
            x_of[tag] = _variable(row.x, _CURVE, engine=engine, kind="given",
                                  where=f"[xs.{row.x['name']}]")
        x = (_labels if x_of[tag].dtype is str else _numbers)(
            c.get("x"), f"curve {k}'s x")
        y = _numbers(c.get("y"), f"curve {k}'s y")
        if len(x) != len(y):
            raise ExportError(f"curve {k} ({c.get('name')!r}) has {len(x)} "
                              f"x values and {len(y)} y values")
        by_tag.setdefault(tag, []).append(
            {"name": str(c.get("name") or ""), "hidden": bool(c.get("hidden")),
             "x": x, "y": y})
    held: Dict[str, Var] = {}
    values: Dict[str, Any] = {}
    for k, m in enumerate(marks):
        m = _record(m, f"threshold {k}")
        tag = str(m.get("tag"))
        if tag not in got.marks:
            raise ExportError(f"the {chart} chart draws no threshold tagged "
                              f"{tag!r}")
        var = _variable(table().tags[tag].mark, (), engine=engine,
                        where=f"[tags.{tag}] mark")
        if var.name in held:
            raise ExportError(f"the {chart} chart drew two thresholds tagged "
                              f"{tag!r}")
        held[var.name] = var
        values[var.name] = np.float64(_numbers([m.get("value")],
                                               f"threshold {tag!r}")[0])
    if not by_tag and not held:
        raise ExportError(f"the {chart} chart drew no curve and no threshold: "
                          f"there is nothing to export")
    for tag, rows in by_tag.items():
        cells = len(rows) * max(len(r["x"]) for r in rows)
        if cells > MAX_CELLS:
            raise ExportError(f"the {tag} curves would take {cells} cells "
                              f"padded to the longest, more than a chart "
                              f"draws ({MAX_CELLS})")
    base = Path(calc.base)
    shown = Path(path).resolve().relative_to(base.resolve()).as_posix()
    label = str(calc.task.label)
    with NC_LOCK:
        ds = in_memory(f"{label}.{chart}.nc")
        state(ds, title=f"what the chart of {got.words} drew of {label}",
              schema=DATA_SCHEMA, molbuilder_version=__version__,
              derived_from=(f"the {chart} chart on the Results page, drawn "
                            f"from {shown} of the calculation {label}"))
        ds.setncatts(json_attrs({"label": label, "chart": chart,
                                 "shown": shown, "chosen": dict(chosen)},
                                RECORD_ATTRS))
        if held:
            write_group(ds, {}, tuple(held.values()), values)
        for tag, rows in by_tag.items():
            row = table().tags[tag]
            x = x_of[tag]
            y = _variable(row.y, _CURVE, engine=engine,
                          where=f"[tags.{tag}] y")
            dims = {"curve": len(rows),
                    "curve_point": max([1] + [len(r["x"]) for r in rows])}
            group_vars = (*_CURVE_VARS, x, y)
            v = {var.name: empty(var, dims) for var in group_vars}
            v["curve"][:] = np.arange(dims["curve"])
            v["curve_point"][:] = np.arange(dims["curve_point"])
            for r, curve in enumerate(rows):
                v["curve_name"][r] = curve["name"]
                v["curve_hidden"][r] = int(curve["hidden"])
                for var, got in ((x, curve["x"]), (y, curve["y"])):
                    if var.dtype is str:          # a category's labels
                        v[var.name][r, :len(got)] = got
                        continue
                    # A GAP ON A WHOLE-NUMBER VARIABLE is its fill.
                    cells = np.asarray(got, dtype=float)
                    if np.issubdtype(v[var.name].dtype, np.integer):
                        cells = np.where(np.isnan(cells), var.fill, cells)
                    v[var.name][r, :len(cells)] = cells
            write_group(ds.createGroup(y.name), dims, group_vars, v,
                        title=f"every curve of {y.long_name} the chart "
                              f"drew, along {x.long_name}")
        return as_bytes(ds)


def export(calc, fmt: str, *, path, chart: str, chosen: Mapping[str, Any],
           curves: Sequence[Mapping[str, Any]],
           marks: Sequence[Mapping[str, Any]] = ()):
    """What a page drew of ``chart`` as ``fmt`` -- ``<label>.<chart>.nc``,
    or its tables in ``<label>.<chart>-csv`` -- in the export's one shape
    (`ncdata.exported`), written nowhere."""
    from .ncdata import export_format, export_stem, exported
    export_format(fmt)
    if not (isinstance(chosen, Mapping) and isinstance(curves, (list, tuple))
            and isinstance(marks, (list, tuple))):
        raise ExportError("chosen is a record; curves and marks are lists")
    label = str(calc.task.label)
    data = data_file_bytes(calc, Path(path), chart=chart, chosen=chosen,
                           curves=curves, marks=marks)
    stem = export_stem(label, chart)
    return exported({f"{stem}.nc": data}, fmt, stem=stem, one_file=True)


def read_data_file(path) -> Dict[str, Any]:
    """The file back (`ncdata.read_file`): ``{"attrs", "groups"}`` -- the
    root's attributes, ``chosen`` the record it holds, and every group by its
    path, the root's thresholds under ``""``; another schema refused by
    name."""
    from .ncdata import read_file
    return read_file(path, schemas=(DATA_SCHEMA,), error=ChartDataError,
                     way="its chart's Export writes it again",
                     records=RECORD_ATTRS)


__all__ = ["Chart", "ChartDataError", "ChartTableError", "DATA_SCHEMA",
           "MAX_CELLS", "TABLE_FILE", "Table", "Tag", "data_file_bytes",
           "export", "read_data_file", "table"]
