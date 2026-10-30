"""THE RAMAN OF A SIESTA VIBRATION, AT A LASER -- derived on request from the
optical calls' eps2 (`engines/vibration.md` § 5.11 § 10;
`science/vibrational-optical-response.md` § 3, § 3.2a).

MODULE  spectra.raman (L2; numpy, the optical calls' data file, the variable
        record)
ROLE    :func:`raman_view` -- the one derivation: each frame's eps2, the spin
        channels summed, broadened to the width asked (:func:`spreader`),
        its real part by Kramers-Kronig at the laser and static
        (:func:`eps1_minus_one`), the polarizability -- complex at the laser,
        its imaginary part the spectrum itself there (:func:`eps2_at`) --
        and per picked mode dalpha/dQ and R(e) = |dalpha/dQ|^2, ranked; :func:`raman_view_of` -- the same from a
        result on disk and the data file it cites, the files named with
        their sha256; :func:`read_ask` -- what a page or a script asks;
        :func:`raman_data_bytes` / :func:`read_raman_data` -- the view as
        ``<label>.raman.nc``, an export's (`model/datafiles.md` § 1)
USED-BY the load door (`web/blueprints/spectra.py`), the export
        (`spectra.export`), `postprocess`

**Nothing derived is written into the calculation**: the result keeps a
SIESTA mode's Raman activity null, and the view is served beside it.

**The broadening is SIESTA's own Gaussian**, ``exp(-((e - w)/s)^2) /
(sqrt(pi) s)`` with ``s`` its ``Optical.Broaden`` (`Src/optical.F`; its
full width at half height is ``2 sqrt(ln 2) s``): two of them make one of
width ``sqrt(s1^2 + s2^2)``, so a wider view is the run's spectrum spread
once more by ``sqrt(s^2 - s_run^2)`` -- exact for a transition a few widths
inside the stored grid (§ 5.11 § 12).  Near zero it is not: SIESTA keeps
only the part of each Gaussian above zero, and that lost part is not spread.

**At the laser the polarizability is complex** -- its real part from
Kramers-Kronig, its imaginary part ``V/4pi eps2`` -- and the strength is the
squared modulus of its change, ``R = (dalpha1/dQ)^2 + (dalpha2/dQ)^2``
(Gillet, Giantomassi & Gonze, PRB 88, 094305 (2013), Eqs. 1, 4): off any
transition ``eps2`` is zero at the laser and ``R`` is the real part's alone.
The static value is the real part's, as ``eps2`` is zero at zero frequency.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..constants import HC_EV_NM
from ..runpoints import NOT_RUN
from .activity import CHANNEL_PRESENT_FLOOR_RAMAN_A4_AMU, relative_to_strongest

#: The view's data file, written only by an export (`model/datafiles.md` § 1).
RAMAN_SCHEMA = "molbuilder/raman-view@2"
READABLE_RAMAN_SCHEMAS = (RAMAN_SCHEMA,)

#: What a page or a script asks (:func:`read_ask`).
ASK_KEYS = ("laser_nm", "broaden_ev")

#: The exported view's file, after the label (`model/datafiles.md` § 1).
RAMAN_FILE = ".raman.nc"

#: A side with no result yet, in the run's words
#: (`execution/runs-of-points.md` § 1.3) -- a mode's state beside
#: ``finished`` and ``failed``.
NOT_ENDED = "not ended"

#: How far SIESTA spreads one transition, in its widths (`Src/optical.F`:
#: ``wi = ediff - 20*smooth``) -- the kernel's reach here too.
_KERNEL_REACH = 20.0

#: What the view is, in words -- the data file's title.
WHAT = ("the Raman strengths of a SIESTA vibration's picked modes at a "
        "laser, derived from its optical calls' eps2 (independent states, "
        "no local field): each frame's spectrum, its spin channels summed, "
        "broadened to the width asked -- SIESTA's Optical.Broaden "
        "convention, a Gaussian exp(-(E/s)^2) of s = broaden_ev, its full "
        "width at half height 2 sqrt(ln 2) s -- its real part by "
        "Kramers-Kronig at the laser's photon energy and at zero; the "
        "polarizability alpha = V/4pi (eps - 1) of the cell of volume "
        "cell_volume_a3, complex at the laser (its imaginary part V/4pi "
        "eps2); dalpha/dQ = [alpha(+dQ) - alpha(-dQ)] / 2dQ and the strength "
        "R = |dalpha/dQ|^2 for light along direction -- a strength, not an "
        "intensity: no laser-frequency or temperature factor "
        "(engines/vibration.md 5.11 10; "
        "science/vibrational-optical-response.md 3.2a)")


class RamanError(ValueError):
    """The view cannot be derived -- the message names what the result or
    the ask lacks, ready to surface verbatim."""


# --------------------------------------------------------------------- #
#  The numbers                                                           #
# --------------------------------------------------------------------- #

def photon_energy_ev(laser_nm: float) -> float:
    """A laser's photon energy in eV, ``hc / lambda``."""
    return HC_EV_NM / float(laser_nm)


def width_words(ev: float) -> str:
    """A width as it is shown, eV: six significant digits, rounded UP, so
    the run's own width typed back as shown is never narrower than the
    run's -- which :func:`spreader` refuses.  Served with every view
    (:meth:`RamanView.to_wire`), which the page shows as written."""
    x = float(ev)
    s = f"{x:.6g}"
    if float(s) < x:
        s = f"{float(s) + 10 ** (math.floor(math.log10(x)) - 5):.6g}"
    return s


def spreader(energies: Sequence[float], run_ev: float, width_ev: float):
    """The spread from ``run_ev`` to ``width_ev`` on the uniform grid
    ``energies``, built once for every frame of a view: a function of a
    spectrum -- the spectrum itself when the two widths are one, else its
    convolution with SIESTA's Gaussian of ``sqrt(width^2 - run^2)``.  A
    width below the run's is refused, naming the run's -- the narrowest
    there is."""
    if math.isclose(width_ev, run_ev, rel_tol=1e-9, abs_tol=1e-12):
        return lambda y: np.array(y, dtype=float)
    if width_ev < run_ev:
        raise RamanError(
            f"a broadening of {width_ev:g} eV is narrower than the run's own, "
            f"{width_words(run_ev)} eV (Optical.Broaden) -- the narrowest "
            f"there is")
    e = np.asarray(energies, dtype=float)
    s = math.sqrt(width_ev ** 2 - run_ev ** 2)
    de = float(e[1] - e[0])
    full = int(math.ceil(_KERNEL_REACH * s / de))
    # NO FURTHER THAN THE GRID: an offset past it never reaches the kept
    # slice, whatever width is typed.
    reach = min(full, e.size - 1)
    x = np.arange(-reach, reach + 1) * de
    kernel = np.exp(-(x / s) ** 2)
    # THE WHOLE GAUSSIAN SUMS TO ONE: uncut, its own sum on the grid -- a
    # width a hair above the run's, narrower than one step, is the spectrum
    # itself; cut by the grid, the whole Gaussian's, so the weight beyond
    # the grid stays beyond it.  What lies within a few widths of the
    # grid's ends is spread against nothing beyond them (the module's
    # docstring).
    kernel /= (kernel.sum() if reach == full
               else math.sqrt(math.pi) * s / de)
    from scipy.signal import fftconvolve

    def spread(y):
        # BY FFT: a direct sum is minutes on a junction's frames at a wide
        # width; the transform gives the same numbers to rounding.
        y = np.asarray(y, dtype=float)
        return fftconvolve(y, kernel, mode="full")[reach:reach + y.size]
    return spread


def _spline(energies: np.ndarray, eps2: np.ndarray):
    """The stored spectrum between its grid points: a cubic spline through
    them -- what the spectrum's value and slope at a laser are read from."""
    from scipy.interpolate import CubicSpline
    return CubicSpline(energies, eps2)


def _inside(energies: np.ndarray, omega_ev: float) -> None:
    a, b = float(energies[0]), float(energies[-1])
    if not a < omega_ev < b:
        raise RamanError(
            f"the laser's photon energy, {omega_ev:.4f} eV, is outside the "
            f"stored spectrum's {a:g}-{b:g} eV")


def eps2_at(energies: Sequence[float], eps2: Sequence[float],
            omega_ev: float) -> float:
    """``eps2(omega)``, the stored spectrum at a photon energy between its
    grid points (:func:`_spline`); an ``omega`` outside the grid refused."""
    e = np.asarray(energies, dtype=float)
    _inside(e, omega_ev)
    return float(_spline(e, np.asarray(eps2, dtype=float))(omega_ev))


def eps1_minus_one(energies: Sequence[float], eps2: Sequence[float],
                   omega_ev: float) -> float:
    """``eps1(omega) - 1`` by Kramers-Kronig over the stored spectrum,

    ``(2/pi) P int w' eps2(w') / (w'^2 - omega^2) dw'``,

    the trapezoid on the grid up to its last energy.  At ``omega`` 0 the
    static one, ``(2/pi) int eps2(w')/w' dw'`` from the grid's first energy
    above zero; else the principal value, split as ``(1/pi)[P int
    eps2/(w' - omega) + int eps2/(w' + omega)]`` and the pole taken out:
    ``int (eps2 - eps2(omega))/(w' - omega) + eps2(omega) ln((b - omega)/
    (omega - a))`` -- the value and the slope at ``omega`` the spline's
    (:func:`_spline`), so what is left has no pole for the trapezoid, at a
    grid point or between two.  An ``omega`` outside the grid is
    refused."""
    e = np.asarray(energies, dtype=float)
    y = np.asarray(eps2, dtype=float)
    if omega_ev == 0.0:
        keep = e > 0.0
        return float(2.0 / math.pi * np.trapezoid(y[keep] / e[keep], e[keep]))
    _inside(e, omega_ev)
    a, b = float(e[0]), float(e[-1])
    spline = _spline(e, y)
    at = float(spline(omega_ev))
    near = np.isclose(e, omega_ev, rtol=0.0, atol=1e-12)
    diff = np.where(near, 1.0, e - omega_ev)
    pole = np.where(near, float(spline(omega_ev, 1)), (y - at) / diff)
    principal = (float(np.trapezoid(pole, e))
                 + at * math.log((b - omega_ev) / (omega_ev - a)))
    regular = float(np.trapezoid(y / (e + omega_ev), e))
    return (principal + regular) / math.pi


def polarizability_a3(volume_a3: float, eps1m1: float) -> float:
    """``alpha = V/4pi (eps1 - 1)`` in Angstrom^3 -- the cell's
    polarizability volume along the light's direction."""
    return volume_a3 / (4.0 * math.pi) * eps1m1


# --------------------------------------------------------------------- #
#  The view                                                              #
# --------------------------------------------------------------------- #

@dataclass(frozen=True)
class RamanMode:
    """One picked mode's Raman, at the laser and static -- ``None`` where
    a side is not done, ``why`` saying which.  Keyed as the result keys its
    modes, and the infrared view its own: ``index_1based``."""
    index_1based: int
    frequency_cm1: float
    dq_amu12_ang: float
    dalpha_dq_static_a2_per_amu12: Optional[float]
    #: the change at the laser, complex: its real part and its imaginary
    #: part, the spectrum's own change there
    dalpha_dq_laser_real_a2_per_amu12: Optional[float]
    dalpha_dq_laser_imag_a2_per_amu12: Optional[float]
    strength_static_a4_amu: Optional[float]
    #: ``|dalpha/dQ|^2`` at the laser, the two parts' squares summed
    strength_laser_a4_amu: Optional[float]
    #: the laser's strength over the strongest real mode's; ``None`` when
    #: the strongest is at or below the Raman channel's band floor
    relative: Optional[float]
    state: str
    why: str


@dataclass(frozen=True)
class RamanView:
    """What :func:`raman_view` answers."""
    laser_nm: float
    photon_energy_ev: float
    broaden_ev: float
    run_broaden_ev: float
    direction: Tuple[float, float, float]
    #: the direction's axis by name, ``""`` for any other
    #: (`spectra.infrared.direction_name`)
    direction_name: str
    delta_ang: float
    alpha_static_a3: Optional[float]
    alpha_laser_real_a3: Optional[float]
    alpha_laser_imag_a3: Optional[float]
    modes: Tuple[RamanMode, ...]
    #: the reference structure's cell volume ``V`` in ``alpha = V/4pi (eps
    #: - 1)``, Angstrom^3
    cell_volume_a3: float
    #: the spin channels the spectrum was summed over
    spins: Tuple[str, ...]
    #: the run the optical calls were computed in, and SIESTA's version
    run: str
    siesta_version: str
    #: each file the view was derived from, ``{name, sha256}``
    derived_from: Tuple[Dict[str, str], ...]

    def to_wire(self) -> Dict[str, Any]:
        return {"laser_nm": self.laser_nm,
                "photon_energy_ev": self.photon_energy_ev,
                "broaden_ev": self.broaden_ev,
                "run_broaden_ev": self.run_broaden_ev,
                # THE WIDTHS AS SHOWN, by the one rule (:func:`width_words`).
                "broaden_words": width_words(self.broaden_ev),
                "run_broaden_words": width_words(self.run_broaden_ev),
                "direction": list(self.direction),
                "direction_name": self.direction_name,
                "delta_ang": self.delta_ang,
                "alpha_static_a3": self.alpha_static_a3,
                "alpha_laser_real_a3": self.alpha_laser_real_a3,
                "alpha_laser_imag_a3": self.alpha_laser_imag_a3,
                "cell_volume_a3": self.cell_volume_a3,
                "spins": list(self.spins),
                "run": self.run, "siesta_version": self.siesta_version,
                "derived_from": [dict(d) for d in self.derived_from],
                "modes": [dict(m.__dict__) for m in self.modes]}


def run_broaden_ev(optical: Mapping[str, Any]) -> float:
    """The run's own width, eV: the optical deck's ``Optical.Broaden``, as
    the data file's ``/settings`` keeps every optical keyword its writer read
    off the deck, read again by the deck's reader
    (`parse.fdf.parse_fdf_params`) with its rule for an energy's unit.  The
    deck's own text is not re-read: its coordinates come from a file beside
    it (``%block ... < coordinates.fdf``), which text alone cannot give.  A
    deck stating none is refused -- every optical deck molbuilder writes
    states it, and the width cannot be said otherwise."""
    from ..parse.fdf import parse_fdf_params
    from ..units import energy_ev
    settings = optical.get("settings") or {}
    stated = "\n".join(f"{key} {value}" for key, value
                       in (settings.get("keywords") or {}).items()
                       if isinstance(value, str))
    ry = parse_fdf_params(stated, source="the optical data file's "
                          "settings").optical_broaden_ry
    if ry is None:
        raise RamanError("the optical deck the data file keeps states no "
                         "Optical.Broaden -- the run's width cannot be said")
    return energy_ev(ry, "Ry")


def _cell_volume_a3(optical: Mapping[str, Any]) -> float:
    """The reference structure's cell volume, Angstrom^3, through the
    cell's one door (`cell.resolve`); a box with no volume is refused --
    every polarizability would be zero."""
    from ..cell import resolve
    from ..structure import Structure
    box = resolve(Structure.from_dict(optical["structure"]))
    if box.unresolvable or not box.has_volume:
        raise RamanError(f"the data file's structure has no cell volume "
                         f"({box.unresolvable or 'a box of zero volume'}) "
                         f"-- the polarizability needs it")
    return float(box.volume)


def has_optical_route(result) -> bool:
    """Whether ``result``'s Raman is SIESTA's optical route, its calls
    cited -- the only one with a spectrum to take a laser to."""
    return result.raman_route == "optical" and bool(result.optical)


def require_optical_route(result, where: str = "") -> None:
    """Refused unless :func:`has_optical_route`."""
    if not has_optical_route(result):
        raise RamanError(
            f"{where}this {result.engine} result's Raman is not SIESTA's "
            f"optical route ({result.raman_route!r}): there is no spectrum "
            f"to take a laser to")


def _positive(key: str, value: Any, what: str) -> float:
    """An asked number -- finite and positive -- or refused by name: the
    one rule :func:`read_ask` and :func:`raman_view` apply."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)) or float(value) <= 0.0:
        raise RamanError(f"{key} is {what}, a positive number -- not "
                         f"{value!r}")
    return float(value)


def raman_view(result, optical: Mapping[str, Any], *, laser_nm: float,
               broaden_ev: Optional[float] = None,
               derived_from: Sequence[Mapping[str, str]] = ()) -> RamanView:
    """THE RAMAN OF ``result`` (a `SpectraResults` whose Raman route is
    ``optical``) from its optical calls' data file read back
    (`spectra.optical_data.read_optical_data`): at ``laser_nm``, broadened
    to ``broaden_ev`` -- the run's own width when ``None``.  Refused by
    name: a result whose Raman is not the optical route, a data file with
    no optical call finished yet, whose deck states no width or whose
    structure no cell, a width below the run's, a laser outside the stored
    spectrum."""
    require_optical_route(result)
    laser = _positive("laser_nm", laser_nm, "the laser's wavelength in nm")
    run_ev = run_broaden_ev(optical)
    width = (run_ev if broaden_ev is None else
             _positive("broaden_ev", broaden_ev, "the broadening in eV"))
    omega = photon_energy_ev(laser)
    volume = _cell_volume_a3(optical)
    grid = np.asarray(optical["photon_energy_ev"], dtype=float)
    # NO SPECTRUM YET: the data file is written with every frame pending
    # before the first optical call ends (§ 8), its photon energies none.
    if grid.size < 2:
        raise RamanError("no optical call has finished yet -- the spectrum "
                         "the Raman is derived from is stored once the "
                         "first one does")
    spread = spreader(grid, run_ev, width)

    def alphas(frame) -> Optional[Tuple[float, complex]]:
        """A frame's polarizability, static and at the laser -- complex
        there, its imaginary part ``V/4pi eps2(omega)``."""
        if frame is None or frame.get("state") != "finished" \
                or frame.get("eps2") is None:
            return None
        y = spread(np.asarray(frame["eps2"]).sum(axis=0))
        return (polarizability_a3(volume, eps1_minus_one(grid, y, 0.0)),
                complex(polarizability_a3(volume,
                                          eps1_minus_one(grid, y, omega)),
                        polarizability_a3(volume, eps2_at(grid, y, omega))))

    frames = list(optical.get("frames") or ())
    reference = next((f for f in frames if f.get("mode") is None), None)
    eq = alphas(reference)
    real = {r.index_1based: not r.has_imag for r in result.modes}
    rows = []
    for m in result.optical["modes"]:
        sides = {f.get("side"): f for f in frames if f.get("mode") == m["mode"]}
        plus, minus = alphas(sides.get(1)), alphas(sides.get(-1))
        dq = float(m["dq_amu12_ang"])
        if plus is None or minus is None:
            # A SIDE NOT DONE: the row names it and its state -- never a
            # number from one side.  A side with no result is *not ended*
            # (`execution/runs-of-points.md` § 1.3); `pending` is the data
            # file's word for it, kept there.
            def shown(side):
                said = (sides.get(side) or {}).get("state")
                return NOT_ENDED if said in (None, "pending") else said
            states = {name: shown(s)
                      for name, s, a in (("+δQ", 1, plus), ("−δQ", -1, minus))
                      if a is None}
            said = set(states.values())
            rows.append((m, None, None,
                         next((w for w in ("failed", NOT_RUN, "not done")
                               if w in said), NOT_ENDED),
                         "; ".join(f"the {name} side {state}"
                                   for name, state in states.items())))
            continue
        d_static = (plus[0] - minus[0]) / (2.0 * dq)
        d_laser = (plus[1] - minus[1]) / (2.0 * dq)
        rows.append((m, d_static, d_laser, "finished", ""))
    relative = relative_to_strongest(
        [None if dl is None else abs(dl) ** 2 for _m, _s, dl, _st, _w in rows],
        [real.get(m["mode"], True) for m, *_ in rows],
        CHANNEL_PRESENT_FLOOR_RAMAN_A4_AMU)
    modes = tuple(RamanMode(
        index_1based=int(m["mode"]), frequency_cm1=float(m["frequency_cm1"]),
        dq_amu12_ang=float(m["dq_amu12_ang"]),
        dalpha_dq_static_a2_per_amu12=ds,
        dalpha_dq_laser_real_a2_per_amu12=None if dl is None else dl.real,
        dalpha_dq_laser_imag_a2_per_amu12=None if dl is None else dl.imag,
        strength_static_a4_amu=None if ds is None else ds * ds,
        strength_laser_a4_amu=None if dl is None else abs(dl) ** 2,
        relative=rel, state=st, why=why)
        for (m, ds, dl, st, why), rel in zip(rows, relative))
    from .infrared import direction_name
    direction = tuple(float(x) for x in result.optical["direction"])
    return RamanView(
        laser_nm=laser, photon_energy_ev=omega, broaden_ev=width,
        run_broaden_ev=run_ev, direction=direction,  # type: ignore[arg-type]
        direction_name=direction_name(direction),
        delta_ang=float(result.optical["displacement_ang"]),
        alpha_static_a3=None if eq is None else eq[0],
        alpha_laser_real_a3=None if eq is None else eq[1].real,
        alpha_laser_imag_a3=None if eq is None else eq[1].imag, modes=modes,
        cell_volume_a3=volume,
        spins=tuple(str(s) for s in optical.get("spin") or ()),
        run=str(optical.get("run") or ""),
        siesta_version=str(optical.get("siesta_version") or ""),
        derived_from=tuple(dict(d) for d in derived_from))


def optical_data_of(result_path, result) -> Path:
    """The optical calls' data file the result at ``result_path`` cites,
    beside it -- refused where the result's Raman is not the optical route
    or the file is not there."""
    path = Path(result_path)
    require_optical_route(result, f"{path.name}: ")
    data = path.parent / result.optical["data_file"]
    if not data.is_file():
        raise RamanError(f"{path.name} cites {data.name}, which is not "
                         f"beside it")
    return data


def raman_view_of(result_path, *, laser_nm: float,
                  broaden_ev: Optional[float] = None, result=None,
                  optical_bytes: Optional[bytes] = None) -> RamanView:
    """The view of the result at ``result_path`` (``<label>.spectra.json``)
    and the optical calls' data file it cites (:func:`optical_data_of`) --
    the data file read through its reader, or from ``optical_bytes`` where
    the caller holds them (an export, which copies what it read), and each
    file named with its sha256.  ``result`` is the result file already
    read, where the caller holds it.  The one door the load door, the
    export and `postprocess` ask."""
    from ..persist import sha256_of, sha256_of_bytes
    from ..sidecars.spectra import parse_spectra_json
    from .optical_data import read_optical_data
    path = Path(result_path)
    if result is None:
        result = parse_spectra_json(path)
    data = optical_data_of(path, result)
    if optical_bytes is None:
        optical, data_sha = read_optical_data(data), sha256_of(data)
    else:
        optical = read_optical_data(data, data=optical_bytes)
        data_sha = sha256_of_bytes(optical_bytes)
    return raman_view(result, optical,
                      laser_nm=laser_nm, broaden_ev=broaden_ev,
                      derived_from=({"name": path.name,
                                     "sha256": sha256_of(path)},
                                    {"name": data.name, "sha256": data_sha}))


def read_ask(raw: Any) -> Dict[str, Any]:
    """What a page or a script asks of a result's Raman --
    ``{laser_nm, broaden_ev?}`` -- read by the rule :func:`raman_view`
    applies (:func:`_positive`): the laser a positive number of nm, the
    width a positive number of eV or absent (the run's own); any other key,
    or a value of another kind, refused by name (:class:`RamanError`)."""
    if not isinstance(raw, Mapping):
        raise RamanError(f"raman is an object {{laser_nm, broaden_ev?}}, not "
                         f"{raw!r}")
    extra = sorted(k for k in raw if k not in ASK_KEYS)
    if extra:
        raise RamanError(f"raman asks {', '.join(ASK_KEYS)}; not "
                         f"{', '.join(extra)}")
    if raw.get("laser_nm") is None:
        raise RamanError("raman asks laser_nm: the laser's wavelength in nm")
    return {"laser_nm": _positive("laser_nm", raw["laser_nm"],
                                  "the laser's wavelength in nm"),
            "broaden_ev": (None if raw.get("broaden_ev") is None else
                           _positive("broaden_ev", raw["broaden_ev"],
                                     "the broadening in eV"))}


# --------------------------------------------------------------------- #
#  The view as a data file -- ``<label>.raman.nc``, an export's          #
# --------------------------------------------------------------------- #

def _mode_numbers() -> Tuple[str, ...]:
    """Each mode's numbers, as the file's own table names them -- the
    view's fields of the same names."""
    return tuple(v.name for v in _mode_vars() if v.dtype == "f8")


def _view_vars():
    """The view's own numbers, on the root -- the view's fields of the same
    names."""
    from ..ncdata import Var
    return (
        Var("laser_nm", (), "given", "nm", "the laser's wavelength",
            "as asked"),
        Var("photon_energy_ev", (), "derived", "eV",
            "the laser's photon energy", "hc / laser_nm: where the "
            "polarizability is read"),
        Var("broaden_ev", (), "given", "eV",
            "the width the spectrum was broadened to",
            "s of SIESTA's Gaussian exp(-(E/s)^2), its full width at half "
            "height 2 sqrt(ln 2) s: the run's own when not asked, never "
            "below it"),
        Var("run_broaden_ev", (), "given", "eV", "the run's own width",
            "the optical deck's Optical.Broaden, the spectrum as SIESTA "
            "computed it"),
        Var("delta_ang", (), "given", "angstrom",
            "the displacement of the atom that moves most",
            "each frame moves the mode's largest atom by delta: "
            "dQ = delta / max_k |L_k|, max_k |L_k| the length of the largest atom's displacement vector"),
        Var("cell_volume_a3", (), "derived", "angstrom^3",
            "the cell's volume V", "the reference structure's, in "
            "alpha = V/4pi (eps - 1)"),
        Var("band_floor_a4_amu", (), "given", "angstrom^4/amu",
            "the Raman channel's band floor",
            "at or below it the strongest mode sets no relative scale: "
            "relative is NaN -- set on the Placzek activity's scale, which "
            "runs above the strength R (engines/vibration.md 5.11 10)"),
    )


def _equilibrium_vars():
    """Frame 0's polarizability -- the view's fields of the same names."""
    from ..ncdata import Var
    return (
        Var("alpha_static_a3", (), "derived", "angstrom^3",
            "the polarizability, static", "V/4pi (eps1 - 1) at zero "
            "frequency, light along the direction; eps1 by Kramers-Kronig "
            "over the stored spectrum, cut at its highest energy"),
        Var("alpha_laser_real_a3", (), "derived", "angstrom^3",
            "the polarizability at the laser, its real part",
            "V/4pi (eps1 - 1) at the laser's frequency; eps1 by "
            "Kramers-Kronig over the stored spectrum, cut at its highest "
            "energy"),
        Var("alpha_laser_imag_a3", (), "derived", "angstrom^3",
            "the polarizability at the laser, its imaginary part",
            "V/4pi eps2 at the laser's frequency -- zero off any electronic "
            "transition"),
    )


def _mode_vars():
    from ..ncdata import Var
    return (
        Var("mode", ("mode",), "coordinate", "1", "the mode",
            "the mode's 1-based index in the result", dtype="i4"),
        Var("frequency_cm1", ("mode",), "given", "cm-1",
            "the mode's frequency", "the harmonic frequency"),
        Var("dq_amu12_ang", ("mode",), "given", "amu^1/2 angstrom",
            "the displacement along the mode",
            "dQ = delta / max_k |L_k|, max_k |L_k| the length of the largest atom's displacement vector: the atom that moves most "
            "moves by delta"),
        Var("dalpha_dq_static_a2_per_amu12", ("mode",), "derived",
            "angstrom^2/amu^1/2", "the polarizability's change, static",
            "[alpha(+dQ) - alpha(-dQ)] / 2dQ at zero frequency"),
        Var("dalpha_dq_laser_real_a2_per_amu12", ("mode",), "derived",
            "angstrom^2/amu^1/2",
            "the polarizability's change at the laser, its real part",
            "[alpha1(+dQ) - alpha1(-dQ)] / 2dQ at the laser's frequency, "
            "alpha1 = V/4pi (eps1 - 1)"),
        Var("dalpha_dq_laser_imag_a2_per_amu12", ("mode",), "derived",
            "angstrom^2/amu^1/2",
            "the polarizability's change at the laser, its imaginary part",
            "[alpha2(+dQ) - alpha2(-dQ)] / 2dQ at the laser's frequency, "
            "alpha2 = V/4pi eps2 -- zero off any electronic transition"),
        Var("strength_static_a4_amu", ("mode",), "derived",
            "angstrom^4/amu", "the Raman strength, static",
            "R = (dalpha/dQ)^2 at zero frequency, light along the direction"),
        Var("strength_laser_a4_amu", ("mode",), "derived",
            "angstrom^4/amu", "the Raman strength at the laser",
            "R = |dalpha/dQ|^2 = real^2 + imag^2 at the laser's frequency, "
            "light along the direction"),
        Var("relative", ("mode",), "derived", "1",
            "the strength over the strongest real mode's",
            "at the laser; NaN when the strongest is at or below the Raman "
            "channel's band floor, band_floor_a4_amu"),
        Var("state", ("mode",), "state", "", "the mode's state",
            "finished, or not ended/failed with a side not done", dtype=str),
        Var("why", ("mode",), "state", "", "why a value is missing",
            "the side not done and its state; empty when finished",
            dtype=str),
    )


def raman_data_bytes(view: RamanView, *, label: str) -> Tuple[str, bytes]:
    """``(name, bytes)`` -- ``<label>.raman.nc``, the view through the
    data-file framework (`model/datafiles.md`): ``/`` what it is and what it
    was derived from (§ 2b), the laser, the widths, ``delta``, the cell's
    volume and the band floor, the direction and the spin channels summed,
    the run and SIESTA's version; ``/equilibrium`` frame 0's alpha;
    ``/modes`` each mode's numbers."""
    from .. import __version__
    from ..ncdata import (NC_LOCK, as_bytes, empty, in_memory, json_attr,
                          number_attr, state, write_group)
    name = f"{label}{RAMAN_FILE}"
    numbers = {**view.__dict__,
               "band_floor_a4_amu": CHANNEL_PRESENT_FLOOR_RAMAN_A4_AMU}
    with NC_LOCK:
        ds = in_memory(name)
        state(ds, title=WHAT, schema=RAMAN_SCHEMA,
              molbuilder_version=__version__,
              source=(f"derived from "
                      f"{', '.join(d['name'] for d in view.derived_from)}: "
                      f"the optical calls SIESTA {view.siesta_version} "
                      f"computed in {view.run}"))
        ds.setncatts({
            "label": label,
            "direction": json_attr(list(view.direction)),
            "spins": json_attr(list(view.spins)),
            "run": view.run, "siesta_version": view.siesta_version,
            "derived_from": json_attr([dict(d) for d in view.derived_from])})
        for g, vars_, title in (
                (ds, _view_vars(), None),
                (ds.createGroup("equilibrium"), _equilibrium_vars(),
                 "frame 0's polarizability, the force-constant run's "
                 "reference geometry: static, and at the laser its real and "
                 "imaginary parts")):
            write_group(g, {}, vars_,
                        {v.name: np.asarray(number_attr(numbers[v.name]))
                         for v in vars_}, title=title)
        g = ds.createGroup("modes")
        vars_ = _mode_vars()
        sizes = {"mode": len(view.modes)}
        values = {v.name: empty(v, sizes) for v in vars_}
        for k, m in enumerate(view.modes):
            values["mode"][k] = m.index_1based
            for key in _mode_numbers():
                values[key][k] = number_attr(getattr(m, key))
            values["state"][k] = m.state
            values["why"][k] = m.why
        write_group(g, sizes, vars_, values,
                    title="each picked mode's change of the polarizability "
                          "and its Raman strength, at the laser and static")
        data = as_bytes(ds)
    return name, data


def read_raman_data(path) -> Dict[str, Any]:
    """``<label>.raman.nc`` back as data -- its attributes, its numbers,
    frame 0's alpha and each mode's row; a schema it does not read refused
    by name, with the way back: the view exported again."""
    import json
    from ..ncdata import read_variables, reading, unnumber_attr
    with reading(path) as ds:
        attrs = {k: ds.getncattr(k) for k in ds.ncattrs()}
        if attrs.get("schema") not in READABLE_RAMAN_SCHEMAS:
            raise RamanError(f"{Path(path).name} states the schema "
                             f"{attrs.get('schema')!r}; this molbuilder reads "
                             f"{', '.join(READABLE_RAMAN_SCHEMAS)} -- export "
                             f"the view again (jobset export)")
        out: Dict[str, Any] = {
            "label": str(attrs.get("label", "")),
            **{k: json.loads(str(attrs[k]))
               for k in ("direction", "spins", "derived_from")},
            **{k: str(attrs.get(k, "")) for k in (
                "run", "siesta_version", "molbuilder_version")}}
        for g, vars_ in ((ds, _view_vars()),
                         (ds["equilibrium"], _equilibrium_vars())):
            held = read_variables(g)
            out.update({v.name: unnumber_attr(held[v.name]["data"])
                        for v in vars_})
        mv = read_variables(ds["modes"])
        rows: List[Dict[str, Any]] = []
        for k in range(int(mv["mode"]["data"].shape[0])):
            rows.append({
                "mode": int(mv["mode"]["data"][k]),
                **{key: unnumber_attr(mv[key]["data"][k])
                   for key in _mode_numbers()},
                "state": str(mv["state"]["data"][k]),
                "why": str(mv["why"]["data"][k])})
        out["modes"] = rows
    return out


__all__ = ["ASK_KEYS", "RAMAN_SCHEMA", "READABLE_RAMAN_SCHEMAS", "RamanError",
           "RAMAN_FILE", "RamanMode", "RamanView", "WHAT",
           "width_words",
           "eps1_minus_one", "eps2_at", "has_optical_route",
           "optical_data_of", "spreader",
           "photon_energy_ev", "polarizability_a3", "raman_data_bytes",
           "require_optical_route",
           "raman_view", "raman_view_of", "read_ask", "read_raman_data",
           "run_broaden_ev"]
