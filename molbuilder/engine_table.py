"""The engines -- ``data/engine_table.toml`` read, once, into one
:class:`Engine` per table (`execution/architecture.md` § 3.3).

**What differs between engines is a field here, never a branch** (A18):
its name as a person reads it, the setting that names its files, its config
class and seam, what it computes the atoms in and how its runs run.  A fact
another declaration holds stays there -- an engine's files are the file
catalogue's (`runfiles`), the settings it offers the parameter catalogue's
(`template`) -- and the engines those declarations name are checked against
this table where each loads: the file catalogue's here, the parameter
catalogue's in `template.catalogue`, the warm-file vocabularies' sections in
`warmfiles`.

Floor 1 (`execution/architecture.md` § 2.1): data, read once; it imports no
engine.  A part that is code is named ``module:NAME`` and resolved by the
floor it lives on (`kind_table.part`).
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

#: The file, beside the parameter catalogue.
TABLE_FILE = Path(__file__).resolve().parent / "data" / "engine_table.toml"

#: What an engine computes the atoms in: the structure's own cell, or one
#: molecule in free space whatever the cell.
COMPUTES_IN = ("cell", "free space")

#: The keys an engine's table states, each required ...
_KEYS = ("words", "identity_field", "config", "seam", "computes_in",
         "one_process", "benchmark", "presets", "keyword_deck")
#: ... and those it states where it has one.
_OPTIONAL = ("method",)


class EngineTableError(ValueError):
    """The engine table does not say what an engine must, or an engine named
    that it does not hold -- said with the engine and the key."""


@dataclass(frozen=True)
class Engine:
    """One engine, as the engine table states it."""
    name: str
    #: Its name as a person reads it -- *SIESTA*.
    words: str
    #: The setting that names a calculation's files -- ``system_label``.
    identity_field: str
    #: ``module:NAME`` of the dataclass a template is rebuilt into (floor 1).
    config: str
    #: ``module:NAME`` of the factory of its seam (floor 3,
    #: `jobset.engines.EngineSeam`).
    seam: str
    #: What it computes the atoms in (:data:`COMPUTES_IN`).
    computes_in: str
    #: Whether a run is one process with threads, its request one task.
    one_process: bool
    #: Whether it has a benchmark lane -- a benchmark's measurement pins
    #: name its settings.
    benchmark: bool
    #: ``module:NAME`` of its stage presets -- ``{tier: {item: value}}``, the
    #: table its ladder's tiers are read from by the one ladder rule
    #: (`jobset.ladder.engine_ladder`; floor 1, `config/`).
    presets: str
    #: Whether its deck states its settings as keywords the engine reads --
    #: SIESTA's fdf, not PySCF's script -- which a run's setup rows read back
    #: from the deck (`parse.dirs.setup`).
    keyword_deck: bool
    #: Its method, where its catalogue offers no choice -- SIESTA's DFT --
    #: else ``None``: the config states it.
    method: Optional[str] = None

    @property
    def free_space(self) -> bool:
        """Whether it computes one molecule in free space, whatever the
        structure's cell -- its electron count a molecule's, no box to
        judge."""
        return self.computes_in == "free space"


def _engine(name: str, body: object) -> Engine:
    where = f"{TABLE_FILE.name} [{name}]"
    if not isinstance(body, dict):
        raise EngineTableError(f"{where} is not a table")
    unknown = sorted(set(body) - set(_KEYS) - set(_OPTIONAL))
    missing = [k for k in _KEYS if k not in body]
    if unknown or missing:
        raise EngineTableError(
            f"{where}: " + "; ".join(
                ([f"unknown key(s) {', '.join(unknown)}"] if unknown else [])
                + ([f"missing {', '.join(missing)}"] if missing else []))
            + f" -- an engine states {', '.join(_KEYS)}")
    if body["computes_in"] not in COMPUTES_IN:
        raise EngineTableError(
            f"{where}: computes_in {body['computes_in']!r} is not one of "
            f"{', '.join(COMPUTES_IN)}")
    for key in ("one_process", "benchmark", "keyword_deck"):
        if not isinstance(body[key], bool):
            raise EngineTableError(f"{where}: {key} is true or false")
    return Engine(name=name, words=str(body["words"]),
                  identity_field=str(body["identity_field"]),
                  config=str(body["config"]), seam=str(body["seam"]),
                  computes_in=str(body["computes_in"]),
                  one_process=bool(body["one_process"]),
                  benchmark=bool(body["benchmark"]),
                  presets=str(body["presets"]),
                  keyword_deck=bool(body["keyword_deck"]),
                  method=body.get("method"))


@lru_cache(maxsize=1)
def _table() -> Dict[str, Engine]:
    raw = tomllib.loads(TABLE_FILE.read_text(encoding="utf-8"))
    table = {name: _engine(name, body) for name, body in raw.items()}
    # THE FILE CATALOGUE'S ENGINES ARE THESE: a row naming another, or an
    # engine whose deck the catalogue does not name, refused here, where the
    # two lists meet (`execution/architecture.md` § 3.3).
    from .runfiles import engines as _catalogued
    named = set(_catalogued())
    if named != set(table):
        raise EngineTableError(
            f"{TABLE_FILE.name} holds {', '.join(sorted(table))}; the file "
            f"catalogue (runfiles.WRITTEN) names {', '.join(sorted(named))} "
            f"-- one list of engines, the engine table's")
    return table


def engines() -> Tuple[str, ...]:
    """Every engine, in the table's order."""
    return tuple(_table())


def engine(name: str) -> Engine:
    """The engine called ``name``, or :class:`EngineTableError` naming the
    engines there are."""
    table = _table()
    if name not in table:
        raise EngineTableError(
            f"no engine {name!r} -- the engines are "
            f"{', '.join(table)} ({TABLE_FILE.name})")
    return table[name]


def config_class(name: str) -> type:
    """The dataclass engine ``name``'s template is rebuilt into -- the part
    its record names (``config``, floor 1, `config/`), resolved here, the
    floor it lives on; an engine the table does not hold refused naming the
    engines there are."""
    from .kind_table import part
    rec = engine(name)
    return part(rec.config, record=f"engine {rec.name}", key="config")


def stage_presets(name: str) -> dict:
    """Engine ``name``'s stage presets, ``{tier: {item: value}}`` -- the part
    its record names (``presets``, floor 1, `config/`), resolved here."""
    from .kind_table import part
    rec = engine(name)
    return part(rec.presets, record=f"engine {rec.name}", key="presets")


__all__ = ["TABLE_FILE", "COMPUTES_IN", "EngineTableError", "Engine",
           "engines", "engine", "config_class", "stage_presets"]
