"""The engines -- ``data/engine_table.toml`` read, once, into one
:class:`Engine` per table (`execution/architecture.md` § 3.3).

**What differs between engines is a field here, never a branch** (A18):
its name as a person reads it, the setting that names its files, its config
class, its own settings checks, its reading of *the same calculation* and
its seam, the longest label it keeps whole, the reader of its keyword deck,
what it computes the atoms in and how its runs run -- the
environment they enter and the programs they start, which the one run
script reads (`runwrap`, `running-a-job.md` § 1).  A fact
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

from . import data_dir

#: The file, beside the parameter catalogue.
TABLE_FILE = data_dir() / "engine_table.toml"

#: What an engine computes the atoms in: the structure's own cell, or one
#: molecule in free space whatever the cell.
COMPUTES_IN = ("cell", "free space")

#: The keys an engine's table states, each required ...
_KEYS = ("words", "identity_field", "config", "validator",
         "same_calculation", "seam", "computes_in", "one_process",
         "benchmark", "presets", "environment")
#: ... and those it states where it has one.
_OPTIONAL = ("method", "gpu_environment", "programs", "longest_label",
             "keyword_reader")


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
    #: ``module:NAME`` of its own settings checks, resolved by the settings
    #: gate (`validation.validate`, `science/validation.md` § 7).
    validator: str
    #: ``module:NAME`` of its reading of *the same calculation* -- the
    #: differences between two of its decks (floor 1), resolved by
    #: `deck_record.same_calculation`.
    same_calculation: str
    #: ``module:NAME`` of the factory of its seam (floor 3, in its own
    #: package, building `jobset.engines.EngineSeam`).
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
    #: THE ENVIRONMENT ITS RUNS ENTER -- a category of `configuration.md`
    #: § 4 (`diagnostics.Capabilities.env_for_category`).
    environment: str = ""
    #: The category a run that uses a GPU enters, where its GPU capability
    #: is a build of its own -- SIESTA's -- else ``None``: its deck uses
    #: the device in its own environment (`execution/gpu.md` G8).
    gpu_environment: Optional[str] = None
    #: THE PROGRAMS ITS RUNS START, ``{program: its name as a run's log
    #: says it}``, the first the engine's own -- empty where its deck is a
    #: script the job's own python runs (PySCF).
    programs: Tuple[Tuple[str, str], ...] = ()
    #: Its method, where its catalogue offers no choice -- SIESTA's DFT --
    #: else ``None``: the config states it.
    method: Optional[str] = None
    #: THE LONGEST LABEL IT KEEPS WHOLE in the names of the files it writes
    #: -- SIESTA's 48 -- or ``None``: it cuts none, and only the cap every
    #: label has binds it (`execution/run-identity.md` § 3).
    longest_label: Optional[int] = None
    #: ``module:NAME`` of THE READER OF ITS KEYWORD DECK -- SIESTA's fdf,
    #: ``parse.fdf:KEYWORDS``: its key rule, its switches, its values and
    #: blocks, the deck with its includes (floor 1, it travels beside a job)
    #: -- or ``None``: its deck is a script whose values are its own
    #: (PySCF's).
    keyword_reader: Optional[str] = None

    @property
    def program(self) -> Optional[str]:
        """The program a run of it starts unless its job names another --
        the first of :attr:`programs` -- or ``None``: its deck is a script
        the job's python runs."""
        return self.programs[0][0] if self.programs else None

    def program_words(self, program: Optional[str]) -> str:
        """``program`` as a run's log names it -- or, for a deck that is a
        script (``None``), the engine's own words.  A program the record
        does not name is refused: a run script starts only what its engine
        runs."""
        if program is None and not self.programs:
            return self.words
        named = dict(self.programs)
        if program not in named:
            raise EngineTableError(
                f"engine {self.name!r} runs no program {program!r} -- its "
                f"programs are {', '.join(named) or 'none'} ({TABLE_FILE.name})")
        return named[program]

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
    from .persist import unknown_keys
    said = unknown_keys(body, (*_KEYS, *_OPTIONAL))
    missing = [k for k in _KEYS if k not in body]
    if said or missing:
        raise EngineTableError(
            f"{where}: " + "; ".join(
                ([said] if said else [])
                + ([f"missing {', '.join(missing)}"] if missing else []))
            + ("" if said else f" -- an engine states {', '.join(_KEYS)}"))
    if body["computes_in"] not in COMPUTES_IN:
        raise EngineTableError(
            f"{where}: computes_in {body['computes_in']!r} is not one of "
            f"{', '.join(COMPUTES_IN)}")
    for key in ("one_process", "benchmark"):
        if not isinstance(body[key], bool):
            raise EngineTableError(f"{where}: {key} is true or false")
    longest = body.get("longest_label")
    if longest is not None and (isinstance(longest, bool)
                                or not isinstance(longest, int)
                                or longest < 1):
        raise EngineTableError(
            f"{where}: longest_label is a whole number of characters")
    # EVERY PART IS NAMED ALIKE, ``module:NAME``, and resolved by the floor
    # that asks (`execution/architecture.md` § 3.3).
    for key in ("config", "validator", "same_calculation", "seam",
                "presets", "keyword_reader"):
        ref = body.get(key)
        if ref is not None and (not isinstance(ref, str) or ":" not in ref):
            raise EngineTableError(f"{where}: {key} is \"module:NAME\"")
    # A BENCHMARK TRIAL STARTS COLD, read from its deck's restart keywords
    # (`jobset.agreement`, `script_emit.restart_group`): a benchmark lane is
    # an engine's whose deck states its settings as keywords.
    if body["benchmark"] and body.get("keyword_reader") is None:
        raise EngineTableError(
            f"{where}: benchmark = true needs a keyword_reader -- a trial's "
            f"cold start is read from its deck's restart keywords")
    programs = body.get("programs", {})
    if not (isinstance(programs, dict) and all(
            isinstance(k, str) and isinstance(v, str) and v
            for k, v in programs.items())):
        raise EngineTableError(
            f"{where}: programs is a table of program = its name in words")
    # RANKS START A PROGRAM: a run script launches each rank on one, and a
    # deck that is a script runs once, on the job's python (`runwrap`).
    if not body["one_process"] and not programs:
        raise EngineTableError(
            f"{where}: one_process = false names the programs its ranks "
            f"start (programs) -- a deck that is a script runs as one "
            f"process")
    return Engine(name=name, words=str(body["words"]),
                  identity_field=str(body["identity_field"]),
                  config=str(body["config"]),
                  validator=str(body["validator"]),
                  same_calculation=str(body["same_calculation"]),
                  seam=str(body["seam"]),
                  computes_in=str(body["computes_in"]),
                  one_process=bool(body["one_process"]),
                  benchmark=bool(body["benchmark"]),
                  presets=str(body["presets"]),
                  environment=str(body["environment"]),
                  gpu_environment=body.get("gpu_environment"),
                  programs=tuple(programs.items()),
                  method=body.get("method"),
                  longest_label=longest,
                  keyword_reader=body.get("keyword_reader"))


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


def relabelled(name: str, cfg, label: str):
    """``cfg`` with ``label`` written into the setting engine ``name``'s
    record names as the one that names a calculation's files
    (``identity_field`` -- ``system_label``, ``job_name``)."""
    import dataclasses
    return dataclasses.replace(cfg, **{engine(name).identity_field: label})


def engine_of_config(cfg_or_cls) -> str:
    """THE ENGINE A CONFIG IS FOR -- the engine whose ``config`` part names
    its class (an instance's class asked, or the class itself), or
    :class:`EngineTableError` naming the engines there are.  The one answer,
    for a caller that holds a config and not the task; a caller holding the
    task asks its engine."""
    cls = cfg_or_cls if isinstance(cfg_or_cls, type) else type(cfg_or_cls)
    for name in engines():
        if cls is config_class(name):
            return name
    raise EngineTableError(
        f"{cls.__name__} is no engine's config -- the engines' are "
        f"{', '.join(config_class(n).__name__ for n in engines())} "
        f"({TABLE_FILE.name}, config)")


def keywords(name: str):
    """Engine ``name``'s keyword reader -- the part its record names
    (``keyword_reader``, floor 1), resolved here -- or ``None``: its deck
    is a script, with no keywords to read."""
    rec = engine(name)
    if rec.keyword_reader is None:
        return None
    from .kind_table import part
    return part(rec.keyword_reader, record=f"engine {rec.name}",
                key="keyword_reader")


def read_deck(name: str, path, *, read=None) -> str:
    """THE DECK AS ENGINE ``name`` READS IT -- through its keyword reader
    where its record names one (an fdf's ``< file`` blocks inlined), else
    the file's text: a script's lines are its own.  ``read`` reads a file
    by its path (a plan's ``read_text``, so a prep reads what it is about
    to write)."""
    kw = keywords(name)
    if kw is not None:
        return kw.read_deck(path, read=read)
    p = Path(path)
    return (read(p) if read is not None
            else p.read_text(encoding="utf-8-sig", errors="replace"))


def stage_presets(name: str) -> dict:
    """Engine ``name``'s stage presets, ``{tier: {item: value}}`` -- the part
    its record names (``presets``, floor 1, `config/`), resolved here."""
    from .kind_table import part
    rec = engine(name)
    return part(rec.presets, record=f"engine {rec.name}", key="presets")


__all__ = ["TABLE_FILE", "COMPUTES_IN", "EngineTableError", "Engine",
           "engines", "engine", "config_class", "relabelled", "engine_of_config",
           "keywords", "read_deck", "stage_presets"]
