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

from ..units import gib_words

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
        return (f"{scratch_dir}/: not there (a run made before 2.22.0, "
                f"whose pair lies in the run's folder)")
    from ..warmfiles import is_scratch
    files = sorted(p for p in folder.iterdir()
                   if p.is_file() and is_scratch(p.name))
    if not files:
        deck = _a_device_deck(base, task, stage_dir)
        if deck is not None and not _reads(deck).out_of_core:
            return f"{scratch_dir}/: none -- the self-energies are kept in memory"
        return f"{scratch_dir}/: empty"
    size = sum(p.stat().st_size for p in files)
    volts = sorted({_voltage_of(p.name, label) for p in files})
    if len(files) == 2 and len(volts) == 1:
        return (f"{scratch_dir}/: the {volts[0]} pair, 2 files, "
                f"{gib_words(size)}")
    if len(files) == 1:
        # ONE FILE OF THE PAIR: the round's first point writing the second
        # while its run is going; a walk that stopped between the two
        # writes once it has ended.
        return (f"{scratch_dir}/: one file, {files[0].name}, {gib_words(size)} "
                + ("-- the round's pair being written" if going else
                   "-- a walk that did not reach its end; yours to remove"))
    return (f"{scratch_dir}/: {len(files)} files, {gib_words(size)} -- "
            f"{', '.join(p.name for p in files)}; the walk leaves one "
            f"pair at most, so these are an edit's")


def _voltage_of(name: str, label: str) -> str:
    """The voltage token in ``<label>.<voltage>.TSGF<lead>``."""
    rest = name[len(label) + 1:] if name.startswith(label + ".") else name
    return rest.split(".TSGF", 1)[0]


# ---- the estimate ------------------------------------------------------------

class _Deck:
    """What the estimate reads off a device deck."""
    def __init__(self, text: str):
        from ..parse.fdf import fdf_key, fdf_true, read_fdf
        scalars, blocks = read_fdf(text)

        def one(key, default=None):
            v = scalars.get(fdf_key(key))
            return v[0] if v else default
        self.out_of_core = fdf_true(scalars.get(fdf_key("TS.Elecs.Out-of-core")) or ["true"])
        self.volts = abs(float(one("TS.Voltage", "0")))
        self.pole_ev = float(one("TS.Contours.Eq.Pole", "10"))
        self.temperature_k = float(one("ElectronicTemperature", "300"))
        self.cutoff_kt = float(one("TS.Contours.nEq.Fermi.Cutoff", "5"))
        spin = str(one("Spin", "non-polarized")).lower()
        # SIESTA's own words (`siesta/layout.py` writes `non-colinear`).
        self.spins = {"non-polarized": 1, "polarized": 2, "non-colinear": 4,
                      "non-collinear": 4, "spin-orbit": 8}.get(spin, 1)
        line = blocks.get(fdf_key("TS.Contour.nEq.line")) or []
        self.delta_ev = next((float(r[1]) for r in line
                              if r and r[0].lower() == "delta"), 0.01)
        grid = blocks.get(fdf_key("kgrid_Monkhorst_Pack")) or []
        self.grid = [(int(r[i]), float(r[3]) if len(r) > 3 else 0.0)
                     for i, r in enumerate(grid[:3])]


def _reads(deck_path: Path) -> _Deck:
    return _Deck(Path(deck_path).read_text(errors="replace"))


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
    from .transiesta import pole_count
    poles = pole_count(deck.pole_ev, deck.temperature_k)
    if deck.volts <= 1e-5:
        return poles
    from ..constants import BOLTZMANN_EV_K
    window = deck.volts + 2 * deck.cutoff_kt * BOLTZMANN_EV_K * deck.temperature_k
    return 2 * poles + line_points(window, deck.delta_ev)


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
    first = _reads(decks[0][1])
    if not first.out_of_core:
        return f"{scratch_dir}/: none -- the self-energies are kept in memory"
    try:
        orbitals = _lead_orbitals(base, task)
    except _Unread as why:
        return (f"{scratch_dir}/: one pair at a time; the size cannot be "
                f"said -- {why}")
    if any(n is None for n in orbitals.values()):
        return (f"{scratch_dir}/: one pair at a time; the size is known once "
                f"the leads have run")
    same = len(set(orbitals.values())) == 1
    sizes, energies = [], []
    for v, path in decks:
        deck = _reads(path)
        energies.append(energy_points(deck) + 2)
        if same:
            sizes.append(f"{gib_words(file_bytes(orbitals['L'], deck))} a lead at {v:g} V")
        else:
            sizes.append(", ".join(
                f"{lead} {gib_words(file_bytes(n, deck))}"
                for lead, n in orbitals.items()) + f" a lead at {v:g} V")
    k = k_points_time_reversed(first.grid)
    orbs = (f"{orbitals['L']} orbitals" if same
            else f"L {orbitals['L']}, R {orbitals['R']} orbitals")
    return (f"{scratch_dir}/: one pair at a time, about "
            + " and ".join(sizes)
            + f" ({orbs} × {k} k-point{'s' if k != 1 else ''} × "
            + " and ".join(str(e) for e in energies) + " energies)")


__all__ = ["status_line", "estimate_from", "k_points_time_reversed"]
