"""SIESTA 5.4.2's own reading of the units a deck states -- asked of the binary.

molbuilder's deck reader (`parse/fdf.py`) answers what an omitted unit or
scale means exactly as SIESTA does, and each answer here is SIESTA's, read
off the binary in the molbuilder-siesta env: a physical value with no unit is
refused, coordinates with no format are Bohr, lattice vectors with no
constant are scaled by one.  Skipped cleanly where that env is not
installed.

Subprocess dispatch via the molbuilder-siesta env's siesta binary
(no host PATH siesta is permitted; the env's binary is the
authoritative one per docs/execution/job-contracts.md).
"""

from __future__ import annotations


import shutil
import subprocess
from pathlib import Path

import pytest

# The pseudopotential is an INPUT checked in beside the tests, because
# `conftest.write_pseudos`'s PSML is enough for prep's screening but SIESTA
# does not start on it (tests/fixtures/psml/README.md).
_H_PSML_SOURCE = Path(__file__).resolve().parent / "fixtures" / "psml" / "H.psml"

pytestmark = pytest.mark.engine


def _siesta_binary():
    """Return the path to the molbuilder-siesta env's siesta binary,
    or None when the env is not present on this machine.

    Asks the product where the env is (``_env_prefix``), which cannot return
    something outside the env -- never a host PATH siesta.
    """
    from molbuilder.diagnostics import get_capabilities
    from molbuilder.envs.install import _env_prefix
    caps = get_capabilities()
    if not caps.env_available("molbuilder-siesta"):
        return None
    prefix = _env_prefix("molbuilder-siesta", caps.conda_binary)
    if not prefix:
        return None
    candidate = Path(prefix) / "bin" / "siesta"
    return candidate if candidate.exists() else None


def _require_siesta_binary():
    """pytest skip helper: skip when the SIESTA binary is unreachable
    or refuses to start (broken env, missing libs)."""
    binary = _siesta_binary()
    if binary is None:
        pytest.skip(
            "molbuilder-siesta env not installed; install via "
            "`bash scripts/install-env.sh --bootstrap --yes` or "
            "`python -m molbuilder envs install molbuilder-siesta`."
        )
    # Fast self-test so a downstream subprocess failure has a clean
    # cause (env broken vs the keyword we are gating).
    probe = subprocess.run(
        [str(binary), "--version"],
        capture_output=True, text=True, timeout=15,
    )
    if probe.returncode != 0:
        pytest.skip(
            f"molbuilder-siesta env's siesta refused to start: "
            f"{probe.stderr.strip() or probe.stdout.strip()}"
        )
    return binary


def _run_siesta_on_fdf(binary: Path, fdf_path: Path, *, work_dir: Path,
                      timeout_s: float = 60.0
                      ) -> "subprocess.CompletedProcess":
    """Run SIESTA against ``fdf_path`` and return the full
    ``CompletedProcess`` so the caller can inspect ``returncode``
    along with stdout.

    Runs in ``work_dir`` so SIESTA's per-run side files (`.BASIS`,
    `.bib`, MESSAGES, …) land in the temp dir, not in cwd.  No MPI
    -- single-process.
    """
    return subprocess.run(
        [str(binary), str(fdf_path.name)],
        cwd=str(work_dir),
        capture_output=True, text=True, timeout=timeout_s,
    )


# ===================================================================== #
#  SIESTA'S OWN UNIT RULES, asked of the binary                         #
# ===================================================================== #
#
# `parse/fdf.py` has to decide what a keyword MEANS when the deck states
# no unit.  An invented rule is a rule nothing can check, so these ask the
# engine instead: whatever SIESTA does IS the requirement, and if a future
# SIESTA changes it these go red.


def _unit_probe_deck(tmp_path, *, extra_lines: str = "",
                     coord_format: str = None) -> Path:
    """A two-atom H deck, minimal enough that SIESTA reaches the reader."""
    shutil.copy(_H_PSML_SOURCE, tmp_path / "H.psml")
    fmt = f"AtomicCoordinatesFormat {coord_format}\n" if coord_format else ""
    path = tmp_path / "probe.fdf"
    path.write_text(
        "SystemLabel probe\nNumberOfAtoms 2\nNumberOfSpecies 1\n"
        "%block ChemicalSpeciesLabel\n 1 1 H\n%endblock ChemicalSpeciesLabel\n"
        + fmt + extra_lines +
        "%block AtomicCoordinatesAndAtomicSpecies\n"
        " 0.0 0.0 0.0 1\n 0.0 0.0 1.4 1\n"
        "%endblock AtomicCoordinatesAndAtomicSpecies\n")
    return path


@pytest.mark.parametrize("keyword,bare,with_unit", [
    ("MeshCutoff",           "MeshCutoff 250",          "MeshCutoff 250 Ry"),
    ("PAO.EnergyShift",      "PAO.EnergyShift 0.01",    "PAO.EnergyShift 0.01 Ry"),
    ("ElectronicTemperature", "ElectronicTemperature 300",
     "ElectronicTemperature 300 K"),
    ("LatticeConstant",      "LatticeConstant 10.0",    "LatticeConstant 10.0 Ang"),
])
def test_siesta_REFUSES_a_physical_value_with_no_unit(keyword, bare,
                                                      with_unit, tmp_path):
    """THE REQUIREMENT, from the engine: there is no default unit.

    `parse/fdf.py` therefore passes no `default=` for any of these, and
    a bare value is left unanswered rather than read as a guess.  A deck
    carrying one is a deck SIESTA would not have run.
    """
    binary = _require_siesta_binary()
    bad_dir = tmp_path / "bad"; bad_dir.mkdir(parents=True)
    out = _run_siesta_on_fdf(
        binary, _unit_probe_deck(bad_dir, extra_lines=bare + "\n"),
        work_dir=bad_dir)
    assert "no unit specified" in (out.stdout + out.stderr), (
        f"SIESTA accepted a bare {keyword}; if it has gained a default "
        f"unit, parse/fdf.py may adopt it -- but read it off THIS output, "
        f"not off a manual:\n{out.stdout[-800:]}")

    good_dir = tmp_path / "good"; good_dir.mkdir(parents=True)
    ok = _run_siesta_on_fdf(
        binary, _unit_probe_deck(good_dir, extra_lines=with_unit + "\n"),
        work_dir=good_dir)
    assert "no unit specified" not in (ok.stdout + ok.stderr), (
        f"the control failed: {keyword} WITH a unit was also refused")

    # AND OUR READER FOLLOWS IT.  Establishing the engine's rule is only
    # half a gate; this is the half that fails when we drift from it.
    from molbuilder.parse.fdf import parse_fdf_params
    from molbuilder.units import UnknownUnit
    probe = f"{bare}\n"
    if keyword == "LatticeConstant":
        probe += ("%block LatticeVectors\n 1 0 0\n 0 1 0\n 0 0 1\n"
                  "%endblock LatticeVectors\n")
    try:
        got = parse_fdf_params(probe, source="probe.fdf")
    except UnknownUnit:
        return                      # refused, which is the engine's answer
    field = {"MeshCutoff": "mesh_cutoff_ry",
             "PAO.EnergyShift": "energy_shift_ry",
             "ElectronicTemperature": "electronic_temperature_k",
             "LatticeConstant": "cell_ang"}[keyword]
    assert getattr(got, field) is None, (
        f"SIESTA refuses a bare {keyword}, so parse/fdf.py must not "
        f"answer one -- it returned {getattr(got, field)!r}")


def test_siesta_defaults_omitted_coordinates_to_BOHR(tmp_path):
    """The one keyword here that DOES have a default, and it is not Ang.

    `AtomicCoordinatesFormat` is read with `fdf_string(key, default)`,
    so omitting it is legal and means something.  `parse/fdf.py` read it
    as Ang, which is 1.89x out -- and `coords_ang` is the frozen gate's
    baseline, so a correct junction cited as a foreign deck was refused
    for atoms that had not moved.
    """
    binary = _require_siesta_binary()
    (tmp_path / "d").mkdir(parents=True, exist_ok=True)
    deck = _unit_probe_deck(tmp_path / "d")
    out = _run_siesta_on_fdf(binary, deck, work_dir=deck.parent)
    text = out.stdout + out.stderr
    assert "Bohr" in text and "coor:" in text, (
        f"could not read the coordinate-format banner:\n{text[-800:]}")
    assert "Angstrom" not in text.split("coor:")[1][:200], (
        "SIESTA now defaults omitted coordinates to Angstrom; "
        "parse/fdf.py's default must follow THIS, not a manual")

    # AND OUR READER FOLLOWS IT: 1.4 with no keyword is 1.4 BOHR.
    from molbuilder.constants import BOHR_ANGSTROM
    from molbuilder.parse.fdf import parse_fdf_params
    got = parse_fdf_params(deck.read_text(), source="probe.fdf")
    assert got.coords_ang is not None
    assert got.coords_ang[1][2] == pytest.approx(1.4 * BOHR_ANGSTROM), (
        f"SIESTA read this deck in Bohr; parse/fdf.py read "
        f"{got.coords_ang[1][2]} A, which is Angstrom")


def test_siesta_scales_lattice_vectors_by_ONE_when_no_constant_is_given(
        tmp_path):
    """The OTHER omitted keyword.

    A bare `LatticeConstant 10.0` is refused (above).  OMITTING it is a
    different question and `parse/fdf.py` answers it with 1 Ang.
    If SIESTA scales by anything else, every cell read from a deck
    without the keyword is wrong by that factor, silently.
    """
    binary = _require_siesta_binary()
    (tmp_path / "d").mkdir(parents=True, exist_ok=True)
    deck = _unit_probe_deck(
        tmp_path / "d", coord_format="Ang",
        extra_lines=("%block LatticeVectors\n"
                     " 4.0 0.0 0.0\n 0.0 4.0 0.0\n 0.0 0.0 4.0\n"
                     "%endblock LatticeVectors\n"))
    out = _run_siesta_on_fdf(binary, deck, work_dir=deck.parent)
    text = out.stdout + out.stderr

    # SIESTA echoes the cell it built, in Ang.  A 4.0 vector read with a
    # unit lattice constant stays 4.0; any other constant scales it.
    import re
    m = re.search(r"outcell: Unit cell vectors \(Ang\):\s*\n\s*"
                  r"([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", text)
    if m is None:
        m = re.search(r"outcell: Cell vector modules \(Ang\)\s*:\s*"
                      r"([-\d.]+)", text)
    assert m, (f"could not read the cell SIESTA built:\n{text[-1500:]}")
    assert float(m.group(1)) == pytest.approx(4.0, abs=1e-3), (
        f"SIESTA scaled a 4.0 lattice vector to {m.group(1)} with no "
        f"LatticeConstant, so its default is not 1 Ang; parse/fdf.py's "
        f"default must follow THIS, not a manual")

    # AND OUR READER FOLLOWS IT.
    from molbuilder.parse.fdf import parse_fdf_params
    got = parse_fdf_params(deck.read_text(), source="probe.fdf")
    assert got.cell_ang is not None
    assert got.cell_ang[0][0] == pytest.approx(4.0)
