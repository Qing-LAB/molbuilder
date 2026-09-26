"""The SIESTA family's output lines -- ONE table, every reader.

`model/parse.md` § 5d.5.  The parser reads these patterns; the wrapper's
SCF-timing tee and the monitor get theirs rendered from here -- neither can
import molbuilder, and molbuilder writes both; the cheap ending scan
(`_run_ending`), the benchmark reader and the TBtrans reader read them too.
Three hand-kept regexes for the one `scf:` row were how TranSIESTA's
`ts-scf:` loop went unseen by all three readers at once -- a device that ran
1000 NEGF iterations was timed as 7, and the monitor reported *"no SCF
progress"* for 7.6 hours (2026-09-25).

Every pattern is read off SIESTA 5.4.2's own writer, named beside it.  The
one older spelling kept, the 5.0 betas' ``Siesta Version``, is off that
build's own output (`tests/watch/fixtures/siesta_frozen/BDT_METAL-*`).
Stdlib only: the cheap ending scan imports it and must not pay for arrays.
"""
from __future__ import annotations

import re
from typing import List, NamedTuple, Optional, Tuple

# ---- The SCF row, both phases ------------------------------------------------
#: SIESTA's periodic SCF and TranSIESTA's NEGF loop print one row per
#: iteration with one format (``Src/write_subs.F``: ``(a8,i4,3f16.6,3f10.6)``,
#: ``4f10.6`` under ``Spin.Fix``): the tag -- ``scf:`` or ``ts-scf:`` -- the
#: iteration, then :data:`SCF_COLUMNS`.  Every pattern for the row is built
#: from these two strings, so the tee, the monitor, the parser and the timing
#: instrument cannot disagree about what a row is.
_NEGF_MARK = "ts-"
_ROW_TAG = "scf:"


def _either_case(word: str) -> str:
    """``scf:`` as ``[Ss][Cc][Ff]:`` -- case-blind in an ERE, which POSIX
    awk has no flag for, and the same set to Python.  The standing rule is
    that detection is immune to capitalisation (user, 2026-05-28)."""
    return "".join(f"[{c.upper()}{c.lower()}]" if c.isalpha() else c
                   for c in word)


#: The row's start as a POSIX ERE that is also a Python regex -- the wrapper's
#: awk takes them verbatim.
SCF_PREFIX_ERE = (rf"^[ \t]*({_either_case(_NEGF_MARK)})?"
                  rf"{_either_case(_ROW_TAG)}[ \t]*")
SCF_ROW_ERE = SCF_PREFIX_ERE + "[0-9]"
#: The row itself: group 1 the NEGF mark, 2 the iteration, 3 the columns.
#: :func:`scf_row` is how a reader asks it.
SCF_ROW = re.compile(SCF_PREFIX_ERE + r"([0-9]+)[ \t]+(.+)$")

#: The phases, as each cycle carries them.  The periodic SCF is SIESTA's own;
#: the NEGF loop is TranSIESTA's, which a device runs after the periodic
#: initialization (``transiesta: Initialization run using siesta``).
PHASE_PERIODIC = "periodic"
PHASE_NEGF = "negf"

#: The values after the iteration, in ``write_subs.F``'s order.  Under
#: ``Spin.Fix`` the Fermi column is two (``Ef_up``, ``Ef_dn``); the three
#: energies lead either way.
SCF_COLUMNS = ("Eharris", "E_KS", "FreeEng", "dDmax", "Ef", "dHmax")
#: The whitespace-split field holding E_KS: the tag, the iteration, Eharris,
#: then E_KS -- the energy the parser reports, so the monitor reports it too.
SCF_E_KS_FIELD = 2 + SCF_COLUMNS.index("E_KS")

#: The line opening each relaxation step (``Src/state_init.F``: ``Begin
#: <CG|Broyden|FIRE> opt. move = <N>``) -- how the monitor counts steps.
GEOM_MOVE = re.compile(r"Begin\b.*\bmove\b\s*=\s*([0-9]+)")

#: The names row SIESTA prints above a step's rows (``write_subs.F``:
#: ``iscf  Eharris(eV)  E_KS(eV) ...``), which the parser maps columns by.
SCF_HEADER = re.compile(r"^\s*iscf\s+\S")


class ScfRow(NamedTuple):
    """One SCF row: its phase, its iteration, and where its columns start --
    the origin of the fixed-width read (``3f16.6`` from there)."""
    phase: str
    iscf: int
    columns_at: int


def scf_row(line: str) -> Optional[ScfRow]:
    """The row ``line`` is, or ``None``."""
    m = SCF_ROW.match(line)
    if m is None:
        return None
    return ScfRow(PHASE_NEGF if m.group(1) else PHASE_PERIODIC,
                  int(m.group(2)), m.end(2))


# ---- How a run ends -----------------------------------------------------------
#: Markers that prove the run did NOT reach its own end, in the order a reader
#: should prefer them: ``(substring, run_state)``, matched case-insensitively
#: anywhere in the line -- SIESTA prefixes them with ``node 0:`` under MPI.
#: `_run_ending.scan_ending` reads them and the parser builds its fatal rules
#: from them (`model/parse.md` § 2b).
FATAL_MARKERS: Tuple[Tuple[str, str], ...] = (
    # Out of memory is called out from the generic aborts because it is the
    # most common cause and the most actionable: "you ran out of memory" is
    # the one sentence that tells a user what to change.
    ("out of memory",                "out_of_memory"),
    ("oom-kill",                     "out_of_memory"),
    ("killed process",               "out_of_memory"),
    ("cannot allocate memory",       "out_of_memory"),
    ("insufficient virtual memory",  "out_of_memory"),
    ("siesta: error",                "stopped"),
    ("propor: error",                "stopped"),
    ("stopping program from node",   "stopped"),
    ("siesta died",                  "stopped"),
    ("abnormal_termination",         "stopped"),
)

#: The SCF met its criterion (``Src/scfconvergence_test.F``: ``SCF
#: Convergence by <criteria>``), and the two ways it did not: ``SCF_NOT_CONV:``,
#: the fatal form, and the softer line a relaxation can recover from.
#: Matched case-insensitively anywhere in the line.
SCF_CONVERGED_MARKER = "scf convergence by"
SCF_NOT_CONV_MARKER = "scf_not_conv"
SCF_NOT_CONVERGED_MARKER = "scf did not converge"
#: SIESTA taking back a convergence it has just printed
#: (``Src/siesta_forces.F90``): TranSIESTA's charge still off
#: (``SCF cycle continued due to TranSiesta charge deviation``), or fewer
#: iterations than ``SCF.MinIterations``.  Until the next row, the phase has
#: not converged.
SCF_CONTINUED_MARKER = "scf cycle continued"


# ---- The launch lines, SIESTA and TBtrans alike -------------------------------
#: ``Src/runinfo_m.F90``, which TBtrans's ``tbt_init.F90`` calls too: the
#: ranks MPI gave the run, or serial mode -- one rank, whether the build has
#: MPI (``(only 1 MPI rank)``) or not.  Multiline, so a caller holding the
#: whole text can search it.
RUNNING_ON = re.compile(r"^\s*\*\s*Running on\s+(\d+)\s+nodes? in parallel",
                        re.IGNORECASE | re.MULTILINE)
RUNNING_SERIAL = re.compile(r"^\s*\*\s*Running in serial mode",
                            re.IGNORECASE | re.MULTILINE)
#: The time of day at both ends (``Src/timestamp.f90``:
#: ``>> <what>:  dd-MON-yyyy hh:mm:ss``, printed by SIESTA's ``siesta_init.F``
#: and ``siesta_end.F`` and by TBtrans's ``tbt_init.F90`` and ``tbt_end.F90``).
RUN_START = re.compile(r"^\s*>>\s*Start of run\b[:\s]*(.*?)\s*$",
                       re.IGNORECASE)
RUN_END = re.compile(r"^\s*>>\s*End of run\b[:\s]*(.*?)\s*$", re.IGNORECASE)


def local_time(stamp: str):
    """``24-SEP-2026   9:17:29`` as a naive ISO string, or ``None``.

    NAIVE on purpose: SIESTA prints the node's local time with no zone, and
    stamping one on would be a claim the file does not make.  The month is
    ``timestamp.f90``'s own English table, so no locale is involved.
    """
    from datetime import datetime
    try:
        return datetime.strptime(" ".join(stamp.split()),
                                 "%d-%b-%Y %H:%M:%S").isoformat()
    except ValueError:
        return None


def read_launch_line(line: str, facts: dict) -> Optional[str]:
    """Record one launch line into ``facts`` -- ``n_mpi_processes``,
    ``run_start_local``, ``run_end_local`` -- and return the key, or ``None``
    when the line is not one.  Each is stated once, so the first line wins."""
    m = RUNNING_ON.match(line)
    if m:
        facts.setdefault("n_mpi_processes", int(m.group(1)))
        return "n_mpi_processes"
    if RUNNING_SERIAL.match(line):
        facts.setdefault("n_mpi_processes", 1)
        return "n_mpi_processes"
    for key, pat in (("run_start_local", RUN_START),
                     ("run_end_local", RUN_END)):
        m = pat.match(line)
        if m:
            stamp = local_time(m.group(1))
            if stamp:
                facts.setdefault(key, stamp)
            return key
    return None


def mpi_ranks(text: str) -> Optional[int]:
    """The ranks a whole ``.out`` states, or ``None`` when it states none."""
    facts: dict = {}
    for line in text.splitlines():
        if read_launch_line(line, facts) == "n_mpi_processes":
            break
    return facts.get("n_mpi_processes")


# ---- The build header, SIESTA and TBtrans alike --------------------------------
#: Both programs open their ``.out`` with it (``Src/version-info-template.inc``),
#: so :func:`read_build_line` is the one reader of both.
EXECUTABLE = re.compile(r"^\s*Executable\s*:\s*(\S+)", re.IGNORECASE)
BUILD_VERSION = re.compile(r"^\s*(?:Siesta\s+)?Version\s*:\s*(\S+)",
                           re.IGNORECASE)
BUILD_ARCH = re.compile(r"^\s*Architecture\s*:\s*(.+?)\s*$", re.IGNORECASE)
BUILD_COMPILER = re.compile(r"^\s*Compiler version\s*:\s*(.+?)\s*$",
                            re.IGNORECASE)
#: ``MPI``, ``OpenMP``, both, or ``none``.
BUILD_PARALLEL = re.compile(r"^\s*Parallelisations?\s*:\s*(.+?)\s*$",
                            re.IGNORECASE)
#: One line per compiled-in component, and not always the bare name: 5.4.2
#: writes ``ELSI support. Solvers:`` and ``Native PEXSI support``, which a
#: bare-name pattern read as absent -- the packaged SIESTA's ELSI, the road
#: its ELPA takes, was recorded as missing until 2026-09-26.  ``ELPA`` and
#: ``FLOOK`` are older builds' spellings, kept.
BUILD_FEATURE = re.compile(
    r"^\s*(?:Native\s+)?(GEMM3M|NetCDF-4 MPI-IO|NetCDF-4|NetCDF|"
    r"METIS ordering|Lua|PEXSI|ELSI|ELPA|FLOOK|DFT-D3|Wannier90 wrapper)"
    r"\s+support\b", re.IGNORECASE)
#: Native ELPA prints no line of its own; built with GPU kernels it prints
#: `` --- ELPA GPU support: <kind>`` (measured: ``nvidia-gpu``).
BUILD_ELPA_GPU = re.compile(r"^\s*-+\s*ELPA GPU support:\s*(\S+)",
                            re.IGNORECASE)


def read_build_line(line: str, build: dict) -> Optional[str]:
    """Record one build-header line into ``build``; return the key it set,
    or ``None`` when the line is not one.  The header states each fact once,
    so the first line wins."""
    for key, pat in (("executable", EXECUTABLE), ("version", BUILD_VERSION),
                     ("architecture", BUILD_ARCH),
                     ("compiler", BUILD_COMPILER),
                     ("elpa_gpu", BUILD_ELPA_GPU)):
        m = pat.match(line)
        if m:
            build.setdefault(key, m.group(1).strip())
            return key
    m = BUILD_PARALLEL.match(line)
    if m:
        # "MPI, OpenMP" -> ["MPI", "OPENMP"]; "none" stays one token.
        build.setdefault("parallelisations",
                         [t.strip().upper()
                          for t in re.split(r"[,\s]+", m.group(1))
                          if t.strip()])
        return "parallelisations"
    m = BUILD_FEATURE.match(line)
    if m:
        # "NetCDF-4 MPI-IO" -> "netcdf4_mpiio"; "ELSI" -> "elsi".
        key = re.sub(r"\s+", "_", m.group(1).lower().replace("-", ""))
        build[key] = True
        return key
    return None


# ---- TranSIESTA's lines --------------------------------------------------------
#: The charge report TranSIESTA prints before each NEGF row
#: (``Src/ts_charge.F90``): a header -- ``D``, then per electrode ``E<i>`` and
#: ``C<i>``, then ``B`` when there is a buffer -- and a row of numbers, both
#: behind the ``ts-q:`` prefix.  The last one or two columns are no region's
#: charge: :data:`TS_Q_TOTALS`.
TS_Q_ROW = re.compile(r"^\s*ts-q:\s+(.+?)\s*$")
#: ``dQ`` -- the charge off its target -- and, spin-polarized, ``Qup-Qdn``.
TS_Q_TOTALS = ("dQ", "Qup-Qdn")

#: The Hartree-potential correction TranSIESTA applies each iteration
#: (``Src/m_ts_hartree.F90``: ``'ts-Vha: ',NEGF_Vha / eV,' eV'``).  In BOTH
#: phases: ``dhscf.F`` fixes the potential for the periodic initialization
#: too (*"We require that even the SIESTA potential is 'fixed'"*), so a
#: device's periodic rows carry it as well.
TS_VHA = re.compile(r"^\s*ts-Vha:\s*(\S+)\s*eV", re.IGNORECASE)

#: The start-up echo's frame and its lines (``Src/m_ts_options.F90``; the
#: contour part is ``Src/m_ts_contour_eq.f90``'s): a row of stars opens and
#: closes it; ``ts: <label> = <value>`` inside, sectioned by
#: ``>> Electrodes <<``, ``>> <electrode>`` and the contour banners.
TS_ECHO = re.compile(r"^\s*ts:\s")
TS_ECHO_FRAME = re.compile(r"^\s*ts:\s*\*{20,}\s*$")
TS_ECHO_LINE = re.compile(r"^\s*ts:\s+(.*?)\s+=\s+(.*?)\s*$")
TS_ECHO_SECTION = re.compile(r"^\s*ts:\s*>>\s*(.*?)\s*(?:<<)?\s*$")
TS_ECHO_CONTOUR = re.compile(r"^\s*ts:\s*-{5,}\s*Contour\s*-{5,}\s*$",
                             re.IGNORECASE)
#: A contour banner lists one SEGMENT per chemical potential or contour part,
#: each with the same labels (``print_contour_eq_options``): a segment starts
#: at the line naming its chemical potential, or at ``Contour name``.
TS_ECHO_SEGMENT = re.compile(r"(chemical potential|^Contour name)$",
                             re.IGNORECASE)

#: The charge distribution TranSIESTA reports when it takes over from the
#: periodic density -- the baseline every NEGF iteration's charge is judged
#: against: ``transiesta: Charge distribution, target = <N>`` and then
#: ``<name> [<tag>] : <value>...`` rows -- one value, or with spin one per
#: channel, the ``[Q]`` row adding the total (``Src/ts_charge.F90``).
TS_CHARGE_START = re.compile(
    r"^\s*transiesta:\s*Charge distribution,\s*target\s*=\s*(\S+)",
    re.IGNORECASE)
TS_CHARGE_ROW = re.compile(r"^\s*(.+?)\s*\[(\w+)\]\s*:\s*(.+?)\s*$")

#: The electrode checks.  The principal-cell test (``Src/ts_electrode.F90``)
#: says one of three things: ``perfect!``, ``extending out with <n> elements,
#: all being zero.``, or ``extending out with <n> elements:`` and the
#: offending coupling.  The first two pass -- a coupling past the cell whose
#: elements are all zero couples nothing -- and ``ts_init.F90`` stops the run
#: on the third.  Then the surface Green's function's recursion.
TS_PRINCIPAL_CELL = re.compile(
    r"^\s*(\S+)\s+principal cell is\s+(.+?)\s*[.!:]*\s*$", re.IGNORECASE)
TS_GF_FOR = re.compile(
    r"^\s*Calculating surface Green functions for:\s*(\S+)", re.IGNORECASE)
TS_GF_MEAN_STD = re.compile(
    r"Lopez Sancho.*Mean/std iterations:\s*(\S+)\s*/\s*(\S+)", re.IGNORECASE)
TS_GF_MIN_MAX = re.compile(
    r"Lopez Sancho.*Min/Max iterations\s*:\s*(\d+)\s*/\s*(\d+)",
    re.IGNORECASE)

#: SIESTA's own Makov-Payne monopole term (``Src/write_subs.F``):
#: ``siesta: Emadel  = <eV>`` -- zero unless SIESTA applied it (a molecule
#: in a simple, face- or body-centred cubic cell, ``Src/madelung.f``).
EMADEL = re.compile(r"^\s*siesta:\s+Emadel\s*=\s*(\S+)", re.IGNORECASE)


def ts_charge_values(text: str) -> List[float]:
    """The numbers on a charge-distribution row: one, or one per spin
    channel (and the total, on the ``[Q]`` row); ``[]`` if any is not one."""
    out: List[float] = []
    for tok in text.split():
        try:
            out.append(float(tok))
        except ValueError:
            return []
    return out


# ---- TBtrans (the transmission rung) -------------------------------------------
#: ``Util/TS/TBtrans/m_tbt_kpoint.F90`` states the transverse k-points and how
#: they were chosen; ``m_tbt_trik.F90`` the time each spin pass of the loop
#: took -- TBtrans runs its whole loop once per spin channel
#: (``m_tbtrans.F90``), printing ``tbt: Completed in`` and then that pass's
#: currents.
TBT_KPOINTS = re.compile(
    r"^\s*tbt:\s*Number of transport k-points\s*=\s*(\d+)", re.IGNORECASE)
TBT_KMETHOD = re.compile(r"^\s*tbt:\s*Method\s*=\s*(.+?)\.?\s*$",
                         re.IGNORECASE)
TBT_COMPLETED = re.compile(r"^\s*tbt:\s*Completed in\s+(\S+)\s*s",
                           re.IGNORECASE)
#: ``L -> R, V [V] / I [A]: 0.400000     V / 0.309835E-04 A`` and the ``P [W]``
#: line beside it (``m_tbt_save.F90``, one pair per electrode pair per pass):
#: the bias TBtrans itself applied -- the electrodes' chemical-potential
#: difference -- and what flowed.
TBT_CURRENT = re.compile(
    r"^\s*(\S+)\s*->\s*(\S+),\s*V \[V\] / I \[A\]:\s*(\S+)\s*V\s*/\s*(\S+)\s*A",
    re.IGNORECASE)
TBT_POWER = re.compile(
    r"^\s*(\S+)\s*->\s*(\S+),\s*V \[V\] / P \[W\]:\s*(\S+)\s*V\s*/\s*(\S+)\s*W",
    re.IGNORECASE)
#: The transmission files, per spin channel (``Util/TS/TBtrans/m_tbt_save.F90``
#: ``name_save``): ``<label>.TBT.AVTRANS_<E1>-<E2>`` unpolarized, and
#: ``<label>.TBT_UP.`` / ``<label>.TBT_DN.`` for the two channels of a
#: polarized run -- which a reader globbing ``.TBT.`` alone never finds.
#: Written beside the ``.TBT.nc`` by ``state_cdf2ascii`` when NetCDF-4 is
#: built in, as it is in the packaged SIESTA.  A deck setting
#: ``TBT.Directory`` would move them; molbuilder's decks set none.
TBT_CHANNELS = (("unpolarized", ".TBT."), ("up", ".TBT_UP."),
                ("down", ".TBT_DN."))
