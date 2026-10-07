"""TBtrans's own output -- the transmission rung's account of itself.

`model/parse.md` § 5d.2 and § 5d.5: its build, its ranks, its k-points,
how long each spin pass took, the voltage it applied and the current and
power it reports -- and which transmission files it wrote, per spin channel.

The patterns are ``siesta_grammar``'s, the family's one table.  Not a
registered viewer file: the transmission rung's result is the transport
record (`engines/transport.md` § 2a.12), whose builder moves onto these
readers in W35's P3 -- until then `transport/record.py` reads the
transmission itself.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from . import siesta_grammar as _G


def read_tbtrans_out(text: str) -> Dict[str, Any]:
    """The facts a TBtrans ``.out`` states, keyed as the record names them:
    ``build`` (the header SIESTA prints too), ``n_mpi_processes``,
    ``run_start_local`` / ``run_end_local``, ``k_points``, ``k_method``,
    ``completed_s`` -- one per spin pass, since TBtrans runs its whole loop
    once per channel -- and ``currents``: one row per electrode pair and pass,
    the bias TBtrans applied with the current and power it reports.  A row
    names its ``channel`` (``up`` / ``down``) when the run made two passes;
    one pass states no channel of its own.  A key is absent when the file
    does not state it."""
    facts: Dict[str, Any] = {}
    build: Dict[str, Any] = {}
    passes: List[float] = []
    currents: List[Dict[str, Any]] = []
    for line in text.splitlines():
        if (_G.read_build_line(line, build)
                or _G.read_launch_line(line, facts)):
            continue
        for key, pat, cast in (("k_points", _G.TBT_KPOINTS, int),
                               ("k_method", _G.TBT_KMETHOD, str)):
            m = pat.match(line)
            if m:
                try:
                    facts.setdefault(key, cast(m.group(1)))
                except ValueError:
                    pass
        m = _G.TBT_COMPLETED.match(line)
        if m:
            try:
                passes.append(float(m.group(1)))
            except ValueError:
                pass
            continue
        for pat, field in ((_G.TBT_CURRENT, "current_a"),
                           (_G.TBT_POWER, "power_w")):
            m = pat.match(line)
            if not m:
                continue
            try:
                bias, value = float(m.group(3)), float(m.group(4))
            except ValueError:
                continue
            # A pass prints its currents after its own "Completed in".
            key = (m.group(1), m.group(2), bias, len(passes))
            row = next((c for c in currents if c["_key"] == key), None)
            if row is None:
                row = {"_key": key, "from": key[0], "to": key[1],
                       "voltage_v": bias}
                currents.append(row)
            row[field] = value
    channels = ("up", "down") if len(passes) == 2 else ()
    for row in currents:
        n = row.pop("_key")[3]
        if 1 <= n <= len(channels):
            row["channel"] = channels[n - 1]
    if build:
        facts["build"] = build
    if passes:
        facts["completed_s"] = passes
    if currents:
        facts["currents"] = currents
    return facts


def transmission_files(directory, label: str) -> Dict[str, List[Path]]:
    """``{channel: [AVTRANS file, ...]}`` -- the unpolarized file, or the two
    channels of a polarized run (``siesta_grammar.TBT_CHANNELS``)."""
    out: Dict[str, List[Path]] = {}
    for channel, tag in _G.TBT_CHANNELS:
        files = sorted(Path(directory).glob(f"{label}{tag}AVTRANS_*"))
        if files:
            out[channel] = files
    return out
