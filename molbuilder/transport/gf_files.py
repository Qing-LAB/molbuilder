"""THE LEADS' SURFACE GREEN'S FUNCTION FILES, AS THE STATUS VERB SAYS THEM
-- `engines/transport.md` § 2a.11, *What a run's points share*: what a
device run's scratch folder (``tsgf/``) holds, and, before the launch, the
size each lead's file will have, read from the files on disk alone: the
leads' run outputs (their orbital count) and the device's prepared deck.
One door, asked by `jobset.runstatus` for the device stage's row.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..units import size_words

#: What the engine stores per matrix element: a complex double.
_BYTES = 16


def status_line(base, task, stage_dir, run_dir: Optional[Path],
                scratch_dir: str, *, going: bool = False) -> Optional[str]:
    """The device's one line (`engines/transport.md` § 2a.11, *Checked by
    the status verb*): for a launched run what its scratch folder holds;
    for a prepared stage the estimate, once the leads have run.
    ``scratch_dir`` is the job's declared folder (`Job.scratch_dir`, the
    vocabulary's name as the stage was prepared); ``None`` with none."""
    if not scratch_dir:
        return None
    if run_dir is not None:
        return _holds(Path(run_dir) / scratch_dir, scratch_dir, task.label,
                      Path(stage_dir), task, base, going=going)
    return _estimate_line(base, task, Path(stage_dir), scratch_dir)


def _holds(folder: Path, scratch_dir: str, label: str, stage_dir: Path,
           task, base, *, going: bool) -> str:
    if not folder.is_dir():
        return f"{scratch_dir}/: not there"
    from ..warmfiles import is_scratch
    files = sorted(p for p in folder.iterdir()
                   if p.is_file() and is_scratch(p.name))
    if not files:
        deck = _a_device_deck(base, task, stage_dir)
        try:
            in_memory = deck is not None and not _reads(deck).out_of_core
        except _Unread as why:
            return f"{scratch_dir}/: empty -- {why}"
        if in_memory:
            return f"{scratch_dir}/: none -- the self-energies are kept in memory"
        return f"{scratch_dir}/: empty"
    size = sum(p.stat().st_size for p in files)
    volts = sorted({_voltage_of(p.name, label) for p in files})
    if len(files) == 2 and len(volts) == 1:
        return (f"{scratch_dir}/: the {volts[0]} pair, 2 files, "
                f"{size_words(size)}")
    if len(files) == 1:
        # ONE FILE OF THE PAIR: the round's first point writing the second
        # while its run is going; a walk that stopped between the two
        # writes once it has ended.
        return (f"{scratch_dir}/: one file, {files[0].name}, {size_words(size)} "
                + ("-- the round's pair being written" if going else
                   "-- a walk that did not reach its end; yours to remove"))
    return (f"{scratch_dir}/: {len(files)} files, {size_words(size)} -- "
            f"{', '.join(p.name for p in files)}; the walk leaves one "
            f"pair at most, so these are an edit's")


def _voltage_of(name: str, label: str) -> str:
    """The voltage token in ``<label>.<voltage>.TSGF<lead>``."""
    rest = name[len(label) + 1:] if name.startswith(label + ".") else name
    return rest.split(".TSGF", 1)[0]


# ---- the estimate ------------------------------------------------------------

#: How many spin components each spin treatment's Hamiltonian holds -- the
#: factor of the engine's own file estimate (`m_ts_electrode.F90`) -- keyed
#: by the deck reader's words (`parse.fdf.FdfParams.spin_treatment`).
_SPIN_COMPONENTS = {"restricted": 1, "unrestricted": 2, "non-collinear": 4,
                    "spin-orbit": 8}


class _Deck:
    """What the estimate reads off a device deck, through the deck's own
    readers: the electronic temperature, the k-grid and its offset and the
    spin by `parse.fdf.parse_fdf_params`; TranSIESTA's keywords in their
    units (`units`).  A keyword the deck leaves out is the engine's own
    default as the catalogue states it (`siesta.engine_input.
    engine_defaults`); one with no default stated there is
    :class:`_Unread`, named."""
    def __init__(self, text: str):
        from ..parse.fdf import fdf_key, fdf_true, parse_fdf_params, read_fdf
        from ..siesta.engine_input import NEQ_LINE, NEQ_LINE_DELTA, \
            engine_defaults
        from ..units import energy_ev, temperature_k
        scalars, blocks = read_fdf(text)
        defaults = engine_defaults()
        where = "the device deck"

        def said(key):
            toks = scalars.get(fdf_key(key)) or defaults.get(fdf_key(key))
            if not toks:
                raise _Unread(f"{where} states no {key}")
            return toks

        def ev(toks, key):
            return energy_ev(float(toks[0]), toks[1] if len(toks) > 1
                             else None, what=key, source=where)
        p = parse_fdf_params(text, source=where)
        self.out_of_core = fdf_true(said("TS.Elecs.Out-of-core"))
        self.volts = abs(ev(said("TS.Voltage"), "TS.Voltage"))
        self.pole_ev = ev(said("TS.Contours.Eq.Pole"), "TS.Contours.Eq.Pole")
        toks = said("ElectronicTemperature")
        self.temperature_k = (p.electronic_temperature_k
                              if p.electronic_temperature_k is not None
                              else temperature_k(
                                  float(toks[0]),
                                  toks[1] if len(toks) > 1 else None,
                                  what="ElectronicTemperature",
                                  source=where))
        self.cutoff_kt = float(said("TS.Contours.nEq.Fermi.Cutoff")[0])
        if p.spin_treatment not in _SPIN_COMPONENTS:
            raise _Unread(f"{where}'s Spin is no word SIESTA reads")
        self.spins = _SPIN_COMPONENTS[p.spin_treatment]
        # THE BIAS WINDOW'S SPACING: the deck's line, or the one the engine
        # builds where it writes none (`engine_input.NEQ_LINE_DELTA`).
        row = next((r for r in blocks.get(NEQ_LINE) or []
                    if r and r[0].lower() == "delta"), None)
        self.delta_ev = ev(row[1:] if row else list(NEQ_LINE_DELTA),
                           "TS.Contour.nEq.line delta")
        self.grid = (list(zip(p.kgrid, p.kgrid_displacement or (0.0,) * 3))
                     if p.kgrid else [])


def _reads(deck_path: Path) -> _Deck:
    """The deck at ``deck_path``, read (:class:`_Deck`); one that does not
    read is :class:`_Unread`, in its reader's words."""
    from ..parse.fdf import read_deck
    try:
        return _Deck(read_deck(deck_path))
    except (OSError, ValueError) as exc:
        raise _Unread(f"{deck_path.name} does not read: {exc}") from None


def k_points_time_reversed(grid: List[Tuple[int, float]]) -> int:
    """How many k-points TranSIESTA keeps of a Monkhorst-Pack grid -- k and
    -k counted once, the only symmetry its grid uses (`kpoint_t.F90`,
    `ts_kpoint_scf.F90`): 5×5×1 → 13, 3×3×1 → 5.  ``grid`` is each axis's
    count and its shift in grid spacings; the transport axis holds 1."""
    import itertools
    axes = []
    for n, shift in grid:
        n = max(int(n), 1)
        # SIESTA's own placement (`find_kgrid.F`): (i + shift) / n for i in
        # -(n-1)/2 .. n/2, the division truncating -- an even grid holds
        # Gamma and the zone boundary when unshifted.
        axes.append([((i + shift) / n) % 1.0
                     for i in range(-((n - 1) // 2), n // 2 + 1)])
    points = [tuple(round(k, 9) for k in p) for p in itertools.product(*axes)]
    seen, kept = set(), 0
    for p in points:
        if p in seen:
            continue
        kept += 1
        seen.add(p)
        seen.add(tuple(round((-k) % 1.0, 9) for k in p))
    return kept


def energy_points(deck: _Deck) -> int:
    """The contour's energy points: the equilibrium poles, and under bias
    the second reservoir's poles and the bias window's line
    (`engines/transport.md` § 2a.11, *What it costs*)."""
    from ..siesta.engine_input import IS_VOLT_EV
    from .transiesta import pole_count
    poles = pole_count(deck.pole_ev, deck.temperature_k)
    if deck.volts <= IS_VOLT_EV:
        return poles
    return 2 * poles + line_points(
        bias_window_ev(deck.volts, deck.cutoff_kt, deck.temperature_k),
        deck.delta_ev)


def bias_window_ev(volts: float, cutoff_kt: float,
                   temperature_k: float) -> float:
    """The bias window's line's width as TranSIESTA builds it: |V| and the
    Fermi cutoff's ``cutoff_kt`` kT on each side (`m_ts_contour_neq.f90`;
    the device deck's line, `transport.deck._emit_neq_contour`)."""
    from ..constants import BOLTZMANN_EV_K
    return (abs(float(volts))
            + 2 * float(cutoff_kt) * BOLTZMANN_EV_K * float(temperature_k))


def line_points(window_ev: float, delta_ev: float) -> int:
    """The bias window's line points, as TranSIESTA counts them: the
    window over the spacing, rounded half away from zero (`nint`,
    `m_ts_io_ctype.f90`)."""
    return int(math.floor(window_ev / delta_ev + 0.5))


def file_bytes(n_orbitals: int, deck: _Deck) -> float:
    """The engine's own estimate (`m_ts_electrode.F90` `print_Elec_Green`):
    16 bytes × orbitals² × spins × k-points × (energy points + 2)."""
    return (_BYTES * n_orbitals ** 2 * deck.spins
            * k_points_time_reversed(deck.grid) * (energy_points(deck) + 2))


def estimate_from(deck_text: str, n_orbitals: int) -> float:
    """The bytes one lead's file will have for ``deck_text``, the device
    deck as prep wrote it, and the lead's orbital count."""
    return file_bytes(n_orbitals, _Deck(deck_text))


def _lead_orbitals(base, task) -> Dict[str, Optional[int]]:
    """Each lead's orbital count from its newest run's output (the reading
    pass's ``n_orbitals``), ``None`` where the lead has not run."""
    from ..jobset.materialize import latest_attempt
    from ..jobset.naming import stage_home
    from ..parse import detect
    from ..runs import run_of
    out: Dict[str, Optional[int]] = {}
    for stage, lead in (("electrode_L", "L"), ("electrode_R", "R")):
        attempt = latest_attempt(stage_home(base, task, stage).dir)
        run = run_of(attempt) if attempt is not None else None
        output = run.stdout if run is not None else None
        if output is None:
            out[lead] = None            # the lead has not run
            continue
        try:
            n = (detect(str(output)).parse(str(output)).runtime_info
                 or {}).get("n_orbitals")
        except Exception as exc:        # noqa: BLE001 -- an output that
            raise _Unread(f"{lead}: {output.name} does not read: {exc}")
        if not n:
            raise _Unread(f"{lead}: {output.name} states no orbital count")
        out[lead] = int(n)
    return out


class _Unread(Exception):
    """A lead's run output that could not be read for its orbital count."""


def _a_device_deck(base, task, stage_dir: Path) -> Optional[Path]:
    """One prepared device deck -- any point's, the first on disk."""
    from ..transport.stages import point_folders
    from ..jobset.naming import stage_home
    names = stage_home(base, task, "device").names(task.shape)
    folders = [d for d, _pt in point_folders(base, task, "device")] or [stage_dir]
    for d in folders:
        p = Path(d) / names.name(".fdf")
        if p.is_file():
            return p
    return None


def _decks_per_voltage(base, task, stage_dir: Path) -> List[Tuple[float, Path]]:
    """One prepared device deck per voltage -- any frame's, the decks being
    the same across frames."""
    from ..transport.stages import point_folders
    from ..jobset.naming import stage_home
    names = stage_home(base, task, "device").names(task.shape)
    seen: Dict[float, Path] = {}
    pts = point_folders(base, task, "device")
    for d, pt in pts:
        v = pt.bias_v
        p = Path(d) / names.name(".fdf")
        if v not in seen and p.is_file():
            seen[v] = p
    if not pts:
        p = stage_dir / names.name(".fdf")
        if p.is_file():
            seen[0.0] = p
    return sorted(seen.items())


def _estimate_line(base, task, stage_dir: Path, scratch_dir: str) -> str:
    decks = _decks_per_voltage(base, task, stage_dir)
    if not decks:
        return f"{scratch_dir}/: one pair at a time"
    try:
        read = [(v, _reads(path)) for v, path in decks]
        first = read[0][1]
        if not first.out_of_core:
            return (f"{scratch_dir}/: none -- the self-energies are kept "
                    f"in memory")
        orbitals = _lead_orbitals(base, task)
    except _Unread as why:
        return (f"{scratch_dir}/: one pair at a time; the size cannot be "
                f"said -- {why}")
    if any(n is None for n in orbitals.values()):
        return (f"{scratch_dir}/: one pair at a time; the size is known once "
                f"the leads have run")
    same = len(set(orbitals.values())) == 1
    sizes, energies = [], []
    for v, deck in read:
        energies.append(energy_points(deck))
        if same:
            sizes.append(f"{size_words(file_bytes(orbitals['L'], deck))} a lead at {v:g} V")
        else:
            sizes.append(", ".join(
                f"{lead} {size_words(file_bytes(n, deck))}"
                for lead, n in orbitals.items()) + f" a lead at {v:g} V")
    k = k_points_time_reversed(first.grid)
    orbs = (f"{orbitals['L']} orbitals" if same
            else f"L {orbitals['L']}, R {orbitals['R']} orbitals")
    return (f"{scratch_dir}/: one pair at a time, about "
            + " and ".join(sizes)
            + f" ({orbs} × {k} k-point{'s' if k != 1 else ''} × "
            + " and ".join(str(e) for e in energies)
            + " energies, with the lead's H and S)")


__all__ = ["status_line", "estimate_from", "k_points_time_reversed",
           "bias_window_ev", "line_points"]
