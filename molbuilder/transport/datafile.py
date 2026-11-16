"""THE TRANSPORT DATA FILE -- ``<label>.transport.nc``, the record's results
as data on their grid (`engines/transport.md` § 2a.12, *the data file*).

MODULE  transport.datafile (L2; numpy, netCDF4 and molbuilder's own doors)
ROLE    one NetCDF-4 file per transport calculation, in groups -- the
        junction, each stage that runs once, each point and its device run,
        and a summary across the points (`engines/transport.md` § 2a.12,
        *The calculation as groups*): every point's raw outputs beside what
        molbuilder derives from them, each frame's rows beside its values
        and the frames' average -- that part the frames' post-process's
        (`frame_average`) -- each variable stating its units, its long name,
        its definition and, for a raw one, the file it was read from.  The
        variables are :data:`VARIABLES`, one table the writer fills and the
        reader reads, each placed by :func:`_home` and :data:`SUMMARY`;
        :func:`data_file_bytes` builds it from the composition `summarize
        task` writes the JSON record from, :func:`write_data_file` writes
        it; :func:`read_data_file` is its door
USED-BY `jobset summarize task`, beside `record.write_record`; an export
        (`transport.export`), which hands the bytes out

**One shape for every transport calculation** (user, 2026-10-10: "doing a
derivative is trivial if the result are saved with these parameter and
calculation results consistently aligned and logically organized"): one
structure at one voltage is a 1 x 1 grid, never another layout.  A point not
done is the fill value, with ``done`` 0 -- a gap, never filled in.  The JSON
record is the report -- `summarize` writes it for the command line, and the
Results tab composes the same record on read; this file is the same results
for analysis -- an average, a slope along a frame row, a comparison of
two calculations is a line of xarray on it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from ..ncdata import Var, json_attr
from ..parse.engines.tbtrans import PRINTED_HALF_DECIMAL_EV
from ..runfiles import (ROLE_TBT_DIGEST, TBT_CHANNEL_TAGS, TBTRANS_AVADOS,
                        TBTRANS_AVBDOS, TBTRANS_AVDOS, TBTRANS_AVTEIG,
                        TBTRANS_AVTRANS, TBTRANS_NC, shown)

#: The unpolarized channel's tag, which the variables' sources name.
_TBT = TBT_CHANNEL_TAGS["unpolarized"]
from .sort import REGION_BRIDGE, REGION_LEFT_ELECTRODE, REGION_RIGHT_ELECTRODE
from .stages import same_volts

#: The file's schema, stated in it.  ``@4``: every result the record holds
#: -- each spin's DOS, eigenchannels and currents, each point's transmission
#: k-points and loop time, what a point shares or is a copy of, every NEGF
#: row's charges, each run's phases, the calculation's own records and its
#: provenance (`engines/transport.md` § 2a.12, *Export*; the user,
#: 2026-10-23); ``@7``: the file states itself, its numbers variables
#: (`model/datafiles.md` § 2b); ``@8``: a mode's frame set's ``mean_quanta``,
#: and its outermost move measured (`model/structure.md` § 2.2f); ``@9``:
#: the calculation as groups -- the junction whole (its cell, each atom's
#: region and labels), each stage, each point and its device run, a summary
#: across the points from the same arrays -- each run's engine version, and
#: which atoms a point's DOS covers (the user, 2026-11-16).  An older file
#: is refused by its version, and `summarize task` writes it again.
DATA_SCHEMA = "molbuilder/transport-data@9"

#: THE ROOT ATTRIBUTES THIS FILE WRITES AS JSON TEXT (`ncdata.json_attr`,
#: :func:`data_file_bytes`; `engines/transport.md` § 2a.12 names each
#: "JSON") -- read back as the blocks they hold by its door
#: (:func:`read_data_file`, `ncdata.unjson_attr`).  The frames' part names
#: its own (`frame_average.JSON_ATTRS`).
JSON_ATTRS = ("current_means", "provenance_chain", "iv_notes",
              "fermi_frames", "scf_criteria")

#: Where every DOS variable is read from, and WHAT ITS SPIN IS: TBtrans's DOS
#: carries no spin factor (``m_tbt_tri_scat.F90``), so a non-polarized
#: point's is each spin's (↑ = ↓) and a polarized point's two channels are
#: summed (`tbtnc.point_dos`) -- ↑ + ↓, as the record's ``dos`` states it.
#: Each from the file the SIESTA family's parser reads it from
#: (`model/parse.md` § 5e): TBtrans's k-averaged text files, and the data
#: file's digest for the DOS by region.
_SPIN_SUMMED = ("each spin's, up = down, for a non-polarized point; a "
                "polarized point's TBT_UP and TBT_DN summed, up + down")


def _from(role: str) -> str:
    return f"<label>{role} -- {_SPIN_SUMMED}"


#: The eigenchannels are one channel's own -- never summed over spin: each
#: spin channel's set, a non-polarized point's one.
_EIG_SOURCE = ("<label>" + shown(TBTRANS_AVTEIG, tbt=_TBT, pair="<L>-<R>")
               + " -- each channel's own: TBT_UP and TBT_DN polarized")

#: The regions molbuilder owns in the device, in order along z.
_REGIONS = (REGION_LEFT_ELECTRODE, REGION_BRIDGE, REGION_RIGHT_ELECTRODE)

class DataFileError(ValueError):
    """The record does not lay out on one grid -- the message names what
    disagrees, ready to surface verbatim."""


#: THE SCF ROW, column by column -- the record's names (`record._SCF_FIELDS`,
#: the SIESTA reader's) and the file's, each with its unit, kind and words,
#: for every run's history alike: ``(key, suffix, kind, units, long name,
#: definition, dtype, fill)``.
_SCF_COLUMNS = (
    ("cycle", "iscf", "raw", "1", "the cycle's number",
     "SIESTA's own cycle number (iscf), from 1 in each phase", "i4", -1),
    ("energy", "energy_ev", "raw", "eV", "the cycle's energy",
     "E_KS, the Kohn-Sham energy", "f8", None),
    ("delta_E", "delta_e_ev", "derived", "eV", "the energy's change",
     "E_KS minus the cycle before's, in the same phase; 0 at a phase's "
     "first cycle (the SIESTA reader's convention)", "f8", None),
    ("dDmax", "ddmax", "raw", "1", "the density matrix's largest change",
     "dDmax, as SIESTA printed it", "f8", None),
    ("dHmax", "dhmax_ev", "raw", "eV", "the Hamiltonian's largest change",
     "dHmax, as SIESTA printed it", "f8", None),
    ("ef", "ef_ev", "raw", "eV", "the cycle's Fermi level",
     "Ef, as SIESTA printed it", "f8", None),
)
#: The columns a device's NEGF rows add -- TranSIESTA's unconserved charge
#: and boundary Hartree potential -- and each row's phase.
_NEGF_SCF_COLUMNS = (
    ("dq", "dq", "raw", "e", "the charge not conserved",
     "dQ of the ts-q: line (NEGF rows): every column summed less the "
     "neutral count -- the charge the open boundaries did not conserve, "
     "not a change between cycles", "f8", None),
    ("vha_ev", "vha_ev", "raw", "eV", "the boundary Hartree potential",
     "ts-Vha (NEGF rows)", "f8", None),
    ("phase", "phase", "state", "", "the cycle's phase",
     "periodic -- the start -- or negf", str, None),
)


def _scf_vars(prefix: str, dims: Tuple[str, ...],
              columns) -> Tuple[Var, ...]:
    """A run's SCF history over ``dims``, a variable per column."""
    return tuple(Var(f"{prefix}_scf_{suffix}", dims, kind, units, long_name,
                     definition, source="the run's SIESTA output",
                     dtype=dtype, fill=fill)
                 for (_key, suffix, kind, units, long_name, definition,
                      dtype, fill) in columns)


def _variables() -> Tuple[Var, ...]:
    from ..frame_average import data_file_variables
    S = str
    return (
        Var("frame", ("frame",), "coordinate", "1", "frame index",
            "its index, counted from 0 as MolView's API counts; a message and "
            "the page number it from 1 (model/structure.md 2.2e)",
            dtype="i4"),
        Var("frame_token", ("frame",), "coordinate", "", "the frame's folder",
            "its index: f000 is frame index 0 (engines/transport.md 2a.11); "
            "empty for one structure, which has no frame level",
            dtype=S, auxiliary=True),
        Var("bias_v", ("bias_v",), "coordinate", "V", "bias voltage",
            "the voltage the device and the transmission ran at -- the "
            "description's list, in its order"),
        Var("energy_ev", ("energy_ev",), "coordinate", "eV", "energy",
            "E - E_F: E_F the leads' Fermi level at 0 V; under a bias V the left lead's chemical potential sits at +V/2 on this axis and the right's at -V/2: the transmission's "
            "grid, at the full precision of each point's "
            f"{shown(TBTRANS_NC, tbt=_TBT)}, ascending; "
            "TBtrans's text files, printed to 5 decimals, put on it within "
            "half that decimal"),
        Var("spin", ("spin",), "coordinate", "", "spin channel",
            "the record's: the one channel of a non-polarized run, or up "
            "and down", dtype=S),
        Var("iv_bias_v", ("iv_bias_v",), "coordinate", "V",
            "the I-V's voltage",
            "the description's voltage list, in its order"),
        Var("atom", ("atom",), "coordinate", "1", "atom index",
            "the atom's user index, from 0 -- the file is listed by user "
            "index (atoms_listed_by), the composed junction put back in the "
            "user order through its order record; engine_atom_index beside "
            "it (model/overview.md 2.2b)", dtype="i4"),
        Var("xyz", ("xyz",), "coordinate", "", "Cartesian component",
            "x, y, z", dtype=S),
        Var("element", ("atom",), "given", "", "the atom's element",
            "as the composed junction states it", dtype=S, auxiliary=True),
        # THE JUNCTION WHOLE (the user, 2026-11-16: "junction's cell should
        # be saved together with structure"): its cell, its periodicity,
        # each atom's region and labels (`engines/transport.md` § 2a.12,
        # *The junction, whole*).
        Var("cell_vector", ("cell_vector",), "coordinate", "",
            "lattice vector", "a, b, c: the cell's rows", dtype=S),
        Var("cell_ang", ("cell_vector", "xyz"), "given", "angstrom",
            "the junction's cell",
            "its three lattice vectors as rows, as the composed junction "
            "states them -- the same for every frame"),
        Var("cell_axis_kind", ("cell_vector",), "given", "",
            "each cell vector's periodicity",
            "as the composed junction states it: periodic, isolated, or "
            "transport -- the electrodes' semi-infinite direction "
            "(model/structure-periodicity.md)", dtype=S),
        Var("atom_region", ("atom",), "given", "", "the atom's region",
            "the region molbuilder owns that it lies in -- L-electrode, "
            "bridge, R-electrode -- from the composed junction's labels "
            "(record.device_regions); empty for an atom in none.  dos_region "
            "sums the atoms of a region that a point's DOS covers "
            "(dos_atom); the region axis names the three",
            dtype=S, auxiliary=True),
        Var("labels", ("atom",), "given", "", "the atom's labels",
            "every label the composed junction states for it, as written, a "
            "JSON list -- the person's own among them, matched as written "
            "by a selection and never read for meaning", dtype=S,
            auxiliary=True),
        Var("region", ("region",), "coordinate", "", "device region",
            "the regions molbuilder owns: L-electrode, bridge, R-electrode",
            dtype=S),
        Var("lead", ("lead",), "coordinate", "", "lead",
            "the two leads, as TBtrans names them (transiesta.LEAD_NAMES)",
            dtype=S),
        Var("eigenchannel", ("eigenchannel",), "coordinate", "1",
            "eigenchannel", "counted from 0, as TBtrans orders them -- as "
            "many as its files hold, those it was asked for (TBT.T.Eig), "
            "each spin channel's; none where no file is read", dtype="i4"),
        # THE FRAMES' POST-PROCESS: their rows, their masses and their
        # average (`frame_average`).
        *data_file_variables(Var),
        Var("positions_ang", ("frame", "atom", "xyz"), "given", "angstrom",
            "atom positions", "every frame's coordinates, listed by user index "
            "-- the composed junction restored"),
        Var("transmission_by_spin", ("frame", "bias_v", "spin", "energy_ev"),
            "raw", "1", "transmission, by spin channel",
            "T(E), k-averaged, channel by channel",
            source=("<label>" + shown(TBTRANS_AVTRANS, tbt=_TBT,
                                                       pair="<L>-<R>")
                    + " (TBT_UP / TBT_DN polarized)")),
        Var("transmission", ("frame", "bias_v", "energy_ev"), "derived", "1",
            "transmission per spin channel",
            "the one channel, or (T_up + T_down) / 2 -- what G is computed "
            "from"),
        Var("conductance_g0", ("frame", "bias_v"), "derived", "G0",
            "conductance", "G / G0 = T at E - E_F = 0, linearly interpolated; "
            "G0 = 2 e^2 / h -- under a bias, T(0) of that bias's "
            "transmission: neither I / V nor dI / dV"),
        Var("current_by_spin_a", ("frame", "bias_v", "spin"), "raw", "A",
            "current, as TBtrans printed it, by spin channel",
            "each channel's own -- a non-polarized point's one channel, half "
            "the junction's total; the fill value for a point whose energies "
            "two runs computed, each printing its own part",
            source="the transmission point's TBtrans output"),
        Var("current_a", ("frame", "bias_v"), "derived", "A",
            "the junction's total current",
            "as current_from says: tbtrans -- twice the printed figure "
            "unpolarized, the channels' sum polarized (current_by_spin_a); "
            "joined -- a point that "
            "shares energies with a finished calculation's, the record's "
            "over its joined grid, its printed figure the fill value"),
        Var("current_from", ("frame", "bias_v"), "state", "",
            "which current current_a is", "tbtrans or joined "
            "(engines/transport.md 2a.16)", dtype=S),
        Var("iv_current_a", ("frame", "iv_bias_v"), "derived", "A",
            "the I-V's current",
            "as the file's iv_computed says: tbtrans -- each point's own "
            "total, current_a; low-bias-approximation "
            "-- I(V) = G0 * integral of T(E, 0) [f(E - V/2) - f(E + V/2)] dE "
            "over that frame's 0 V transmission, E in eV, f the Fermi "
            "function at iv_kt_ev; the fill value at a voltage whose window "
            "and Fermi tails reach past the transmission's (iv_notes)"),
        Var("iv_kt_ev", (), "given", "eV", "the I-V's k_B T",
            "k_B T of the leads' Fermi functions in the low-bias integral, "
            "the device run's electronic temperature; NaN where the I-V is "
            "TBtrans's own"),
        Var("iv_window_lowest_ev", (), "derived", "eV",
            "the low-bias integral's lowest energy",
            "the 0 V transmission's lowest energy, E - E_F; NaN where the "
            "I-V is TBtrans's own"),
        Var("iv_window_highest_ev", (), "derived", "eV",
            "the low-bias integral's highest energy",
            "the 0 V transmission's highest energy, E - E_F; NaN where the "
            "I-V is TBtrans's own"),
        Var("iv_current_by_spin_a", ("frame", "iv_bias_v", "spin"),
            "derived", "A", "the I-V's current, by spin channel",
            "a polarized calculation's, each spin's as the record states it; "
            "the fill value for a non-polarized one, whose spins carry half "
            "the total each"),
        Var("device_ef_ev", ("frame", "bias_v"), "raw", "eV",
            "the device point's Fermi level",
            "its NEGF phase's last cycle, in TranSIESTA's frame",
            source="the device point's SIESTA output"),
        Var("device_vha_ev", ("frame", "bias_v"), "raw", "eV",
            "the device point's boundary Hartree potential",
            "its NEGF phase's last cycle's ts-Vha, as TranSIESTA printed it "
            "-- a value, not the shift between the device's zero and the "
            "seed's, which is the periodic start's (`fermi_frames`)",
            source="the device point's SIESTA output"),
        Var("dos_total", ("frame", "bias_v", "energy_ev"), "raw", "1/eV",
            "the device's density of states", "its Green-function DOS, "
            "k-averaged: a non-polarized point's one spin's, a polarized "
            "point's up + down",
            source=_from(shown(TBTRANS_AVDOS, tbt=_TBT))),
        Var("dos_total_by_spin", ("frame", "bias_v", "spin", "energy_ev"),
            "raw", "1/eV", "the device's density of states, by spin channel",
            "each channel's Green-function DOS, k-averaged -- a "
            "non-polarized point's one channel",
            source=("<label>" + shown(TBTRANS_AVDOS, tbt=_TBT)
                    + " (TBT_UP / TBT_DN polarized)")),
        Var("dos_region", ("frame", "bias_v", "region", "energy_ev"), "raw",
            "1/eV", "the device DOS by region",
            "summed over the region's atoms in the device: a non-polarized "
            "point's one spin's, a polarized point's up + down",
            source=_from(shown(ROLE_TBT_DIGEST, tbt=_TBT)
                         + ", the data file's DOS per orbital")),
        Var("lead_spectral_dos", ("frame", "bias_v", "lead", "energy_ev"), "raw",
            "1/eV", "the device's DOS from a lead's states",
            "TBT.DOS.A, k-averaged: the device's spectral DOS of the states "
            "coming in from this lead, not the lead's own (lead_bulk_dos); "
            "a non-polarized point's one spin's, a polarized point's up + "
            "down",
            source=_from(shown(TBTRANS_AVADOS, tbt=_TBT, lead="<lead>"))),
        Var("lead_bulk_dos", ("frame", "bias_v", "lead", "energy_ev"), "raw",
            "1/eV", "a lead's bulk DOS", "TBT.DOS.Elecs, k-averaged: a "
            "non-polarized point's one spin's, a polarized point's up + "
            "down",
            source=_from(shown(TBTRANS_AVBDOS, tbt=_TBT, lead="<lead>"))),
        Var("eigenchannel_transmission",
            ("frame", "bias_v", "spin", "eigenchannel", "energy_ev"), "raw",
            "1", "eigenchannel transmission, by spin channel",
            "TBT.T.Eig, k-averaged", source=_EIG_SOURCE),
        Var("done", ("frame", "bias_v"), "state", "1", "the point is done",
            "1 when the point has its transmission; 0 otherwise, its "
            "transmission's values the fill value -- a gap, never filled "
            "in; the device's values are its device point's wherever that "
            "ran", dtype="i1", flag=True),
        Var("point_folder", ("frame", "bias_v"), "state", "",
            "the transmission point's folder",
            "relative to the calculation folder: where its raw values were "
            "read -- for a point not done, where its attempt is", dtype=S),
        Var("point_why", ("frame", "bias_v"), "state", "",
            "why a point is not done",
            "in its run's words: pending, or failed and what stopped it",
            dtype=S),
        Var("point_progress", ("frame", "bias_v"), "state", "",
            "how far a point not done got",
            "its TBtrans loop -- running, or where it stopped -- in its run "
            "record's sentence (model/parse.md 5d.1b)", dtype=S),
        Var("dos_why", ("frame", "bias_v"), "state", "",
            "why a done point has no DOS",
            "in its reader's words -- its DOS variables the fill value",
            dtype=S),
        Var("tbt_k_points", ("frame", "bias_v"), "raw", "1",
            "the transmission's transverse k-points",
            "as the point's TBtrans run states it", dtype="i4", fill=-1,
            source="the transmission point's TBtrans output"),
        Var("tbt_k_method", ("frame", "bias_v"), "raw", "",
            "how those k-points were chosen",
            "as the point's TBtrans run states it", dtype=S,
            source="the transmission point's TBtrans output"),
        Var("loop_elapsed_s", ("frame", "bias_v"), "raw", "s",
            "the transmission's loop time",
            "its passes' k-point x energy loops summed, as TBtrans measured "
            "them (model/parse.md 5c.1)",
            source="the transmission point's TBtrans output"),
        Var("copied_from", ("frame", "bias_v"), "state", "",
            "the run a point is a copy of",
            "a calculation that reuses a finished one "
            "(engines/transport.md 2a.16)", dtype=S),
        Var("shared_from", ("frame", "bias_v"), "state", "",
            "the finished point whose energies a point shares",
            "engines/transport.md 2a.16", dtype=S),
        Var("shared_held", ("frame", "bias_v"), "state", "1",
            "the energies a point shares", "how many the finished point "
            "holds", dtype="i4", fill=-1),
        Var("shared_computed", ("frame", "bias_v"), "state", "1",
            "the energies a point's own run computes",
            "beside those it shares", dtype="i4", fill=-1),
        Var("dos_atom", ("frame", "bias_v", "atom"), "state", "1",
            "the point's DOS covers the atom",
            "1 for each atom of TBtrans's device region -- the only atoms it "
            "computes a DOS for -- from the point's kept reading of its data "
            "file (model/parse.md 5e), 0 for every other; -1 where none is "
            "kept", dtype="i1", flag=True),
        Var("dos_per_orbital_reading", ("frame", "bias_v"), "state", "",
            "the point's data-file reading, the DOS by region's",
            "kept, or each data file's state in its words -- a reading just "
            "tried that failed, or that another reading holds, in that "
            "reading's -- dos_region the fill value where it is not kept "
            "(model/parse.md 5e)", dtype=S),
        # EVERY RUN'S STATE AND CONVERGENCE, on the grid it varies over.
        Var("rung", ("rung",), "coordinate", "", "stage",
            "the stages that run once, by their names", dtype=S),
        Var("scf_cycle", ("scf_cycle",), "coordinate", "1", "SCF row",
            "an SCF row's index within its run, from 0: every row of the "
            "run's last step, the periodic start and the NEGF loop alike",
            dtype="i4"),
        Var("charge_part", ("charge_part",), "coordinate", "",
            "TranSIESTA's charge column",
            "as its ts-q: header names them: D the device, E1 C1 ... each "
            "electrode and its coupling", dtype=S),
        Var("point_state", ("frame", "bias_v"), "state", "",
            "the transmission point's state",
            "the run door's word for its run (finished, failed, not run, "
            "pending ...)",
            dtype=S),
        Var("device_state", ("frame", "bias_v"), "state", "",
            "the device point's state", "the run door's word for its run",
            dtype=S),
        Var("device_detail", ("frame", "bias_v"), "state", "",
            "what the device point's state says",
            "the run door's sentence for its run", dtype=S),
        Var("device_scf_converged", ("frame", "bias_v"), "state", "1",
            "the device point's SCF converged",
            "1 converged, 0 not, -1 not stated", dtype="i1", flag=True),
        Var("device_energy_ev", ("frame", "bias_v"), "raw", "eV",
            "the NEGF phase's last energy",
            "never compared across points or calculations "
            "(engines/transport.md 1.1)",
            source="the device point's SIESTA output"),
        Var("device_negf_cycles", ("frame", "bias_v"), "raw", "1",
            "how many NEGF cycles ran",
            "the NEGF rows of the run's SCF history -- TranSIESTA's loop, "
            "after the periodic start", source="the device point's SIESTA "
            "output", dtype="i4", fill=-1),
        Var("device_dq", ("frame", "bias_v"), "raw", "e",
            "the last NEGF cycle's charge not conserved",
            "dQ of its ts-q: line: every column summed less the neutral "
            "count -- the charge the open boundaries did not conserve",
            source="the device point's SIESTA output"),
        Var("device_charge", ("frame", "bias_v", "charge_part"), "raw", "e",
            "the last NEGF cycle's Mulliken populations", "electrons, "
            "column by column, as TranSIESTA printed them: a net population "
            "for a region, an overlap population across a boundary -- "
            "positive, electrons counted, not a net charge",
            source="the device point's SIESTA output"),
        Var("device_d_change_from_0v", ("frame", "bias_v"), "derived", "e",
            "the bridge's population change from 0 V",
            "D(V) - D(0 V): the bridge's net Mulliken population (the D "
            "column of device_charge) at this voltage less the same "
            "frame's at 0 V; not stated where the frame has no finished "
            "0 V point; positive where the bridge gained electrons "
            "(engines/transport.md 2a.12)"),
        Var("device_contour_poles", ("frame", "bias_v"), "raw", "1",
            "the equilibrium contour's poles", "as TranSIESTA echoed it",
            source="the device point's SIESTA output", dtype="i4", fill=-1),
        Var("device_neq_emin_ev", ("frame", "bias_v"), "raw", "eV",
            "the non-equilibrium line's lower end", "under a bias",
            source="the device point's SIESTA output"),
        Var("device_neq_emax_ev", ("frame", "bias_v"), "raw", "eV",
            "the non-equilibrium line's upper end", "under a bias",
            source="the device point's SIESTA output"),
        Var("device_neq_points", ("frame", "bias_v"), "raw", "1",
            "the non-equilibrium line's points", "under a bias",
            source="the device point's SIESTA output", dtype="i4", fill=-1),
        Var("device_scf_cycles", ("frame", "bias_v"), "state", "1",
            "how many SCF rows the device point holds",
            "the cycles beyond it are the fill value; -1 where none ran",
            dtype="i4", fill=-1),
        *_scf_vars("device", ("frame", "bias_v", "scf_cycle"),
                   _SCF_COLUMNS + _NEGF_SCF_COLUMNS),
        Var("device_scf_charge", ("frame", "bias_v", "scf_cycle",
                                  "charge_part"), "raw", "e",
            "every NEGF row's Mulliken populations", "electrons, column by "
            "column, as TranSIESTA printed them",
            source="the device point's SIESTA output"),
        Var("device_scf_phases", ("frame", "bias_v"), "state", "",
            "the device run's phases", "periodic and negf: what each had to "
            "reach and how it ended, JSON", dtype=S),
        Var("rung_state", ("rung",), "state", "", "the stage's state",
            "the run door's word for its run", dtype=S),
        Var("rung_detail", ("rung",), "state", "",
            "what the stage's state says",
            "the run door's sentence for its run", dtype=S),
        Var("rung_scf_converged", ("rung",), "state", "1",
            "the stage's SCF converged", "1 converged, 0 not, -1 not stated",
            dtype="i1", flag=True),
        Var("rung_energy_ev", ("rung",), "raw", "eV", "the stage's energy",
            "its run's last energy", source="the stage's SIESTA output"),
        Var("rung_fermi_ev", ("rung",), "raw", "eV", "the stage's Fermi level",
            "its last SCF cycle's E_F, in its own run's frame -- how the "
            "frames relate, fermi_frames", source="the stage's SIESTA output"),
        Var("rung_scf_cycles", ("rung",), "state", "1",
            "how many SCF rows the stage holds",
            "the cycles beyond it are the fill value; -1 where none ran",
            dtype="i4", fill=-1),
        *_scf_vars("rung", ("rung", "scf_cycle"), _SCF_COLUMNS),
        Var("rung_scf_phases", ("rung",), "state", "", "the stage's phases",
            "what each had to reach and how it ended, JSON", dtype=S),
        Var("rung_copied_from", ("rung",), "state", "",
            "the run a stage's newest run is a copy of",
            "a calculation that reuses a finished one "
            "(engines/transport.md 2a.16)", dtype=S),
    )


#: THE FILE'S VARIABLES -- one table the writer fills and the reader reads.
VARIABLES: Tuple[Var, ...] = _variables()

#: WHAT THE SUMMARY GATHERS (`engines/transport.md` § 2a.12, *The
#: calculation as groups*; the user, 2026-11-16: "to access the IV curve,
#: the user could decide to take individual points from each calculation
#: group, or ... just go for the summary ... consistent and coming from the
#: same result"): what is compared across the points -- each point's
#: numbers and its T(E) over frame x voltage, written from the same arrays
#: as each point's own -- the I-V, and the frames' average.
SUMMARY = ("done", "point_state", "conductance_g0", "current_a",
           "current_by_spin_a", "current_from", "device_ef_ev",
           "device_vha_ev", "device_dq", "device_d_change_from_0v",
           "device_scf_converged", "transmission",
           "iv_current_a", "iv_current_by_spin_a", "iv_kt_ev",
           "iv_window_lowest_ev", "iv_window_highest_ev",
           "average_transmission", "average_delta_transmission",
           "average_conductance_g0", "average_conductance_change_percent",
           "average_why", "average_conductance_why",
           "weight_sum", "weight_tolerance")

#: THE PREFIX A VARIABLE DROPS in the group its path names
#: (`engines/transport.md` § 2a.12: ``rung_fermi_ev`` is a stage's
#: ``fermi_ev``, ``point_why`` a point's ``why``).
_PREFIX = {"stage": "rung_", "point": "point_", "device": "device_"}


def _is_axis(var: Var) -> bool:
    """An axis's own coordinate -- written in each group that runs over it."""
    return var.kind == "coordinate" and var.dims == (var.name,)


def _home(var: Var) -> str:
    """Where a variable lives, by the axes it runs over: a stage's over
    ``rung``; a point's over ``frame`` and ``bias_v`` -- its device run's
    where it is ``device_*``; the summary's what :data:`SUMMARY` names; the
    junction's everything else (`engines/transport.md` § 2a.12)."""
    if "rung" in var.dims:
        return "stage"
    if "frame" in var.dims and "bias_v" in var.dims:
        return "device" if var.name.startswith(_PREFIX["device"]) else "point"
    if var.name in SUMMARY:
        return "summary"
    return "structure"


def data_file_path(base_dir, label: str) -> Path:
    """The ONE spelling of the file's location, composed by the run-file
    catalogue (`runfiles.compose`), as `record.record_path` is."""
    from ..runfiles import ROLE_TRANSPORT_DATA, compose
    return Path(base_dir) / compose(label, ROLE_TRANSPORT_DATA)


def _voltages(volts) -> List[float]:
    """``volts`` in their order, each voltage once (`stages.same_volts`)."""
    out: List[float] = []
    for v in volts:
        if not any(same_volts(v, u) for u in out):
            out.append(float(v))
    return out


def _index(v: float, voltages: Sequence[float]) -> int:
    """Where ``v`` sits along ``voltages`` -- every voltage the record
    states is one of the file's, so one that is not is a
    :class:`DataFileError`."""
    k = next((k for k, u in enumerate(voltages) if same_volts(v, u)), None)
    if k is None:
        raise DataFileError(f"{float(v):g} V is not one of the file's "
                            f"voltages {list(voltages)}")
    return k


def write_data_file(calc, record: Dict, *,
                    readings: Sequence[Dict] = ()) -> Path:
    """Write ``<label>.transport.nc`` from ``record`` (:func:`data_file_bytes`,
    ``readings`` as it takes them) beside the calculation's record; returns
    its path -- `summarize task`'s deliverable."""
    from ..persist import write_bytes
    name, data = data_file_bytes(calc, record, readings=readings)
    out = Path(calc.base) / name
    write_bytes(out, data)
    return out


def data_file_bytes(calc, record: Dict, *,
                    readings: Sequence[Dict] = ()) -> "Tuple[str, bytes]":
    """``<label>.transport.nc`` of the opened calculation ``calc`` from
    ``record`` -- the composition
    `record.collect_record` answers -- and the calculation's composed
    junction, as ``(its name, its bytes)``, written nowhere: `summarize
    task` writes them (:func:`write_data_file`), an export hands them out
    (`transport.export`).  ``readings`` -- the outcomes of the data files'
    readings just made (`tbtnc.read_points`) -- say a file that failed to
    read, or that another reading holds, in that outcome's words.
    :class:`DataFileError` names what does not lay out on one grid."""
    from .. import __version__
    from ..frame_average import fill_data_file, with_meanings
    from ..ncdata import NC_LOCK, as_bytes, empty, in_memory, state
    from .compose import recorded_junction
    from .record import iv_notes_placed, order_unstated_said
    from ..paths import frame_token
    from .stages import (TRANSPORT_STAGES, frames_of,
                         per_point_rungs, rung_points)
    from .transiesta import LEAD_NAMES
    base, task = Path(calc.base), calc.task
    # A READING JUST TRIED AND NOT KEPT, in its outcome's words -- never
    # read as one not tried (`tbtnc.read_points`).
    tried = {o["file"]: o["said"] for o in readings
             if o.get("state") != "kept"}
    # THE ATOM AXIS IN THE USER ORDER (`model/overview.md` § 2.2a, I1): the
    # composed junction, in the engine's order, restored through its record
    # -- which a junction whose sidecar states none does not hold (I9).
    recorded = recorded_junction(base)
    unstated = order_unstated_said(base, recorded)
    if unstated:
        raise DataFileError(unstated)
    junction = (recorded.structure.restored() if recorded is not None
                else None)
    n_frames = frames_of(base)
    biases = _voltages(pt.bias_v for pt in rung_points(
        task, "transmission", frames=n_frames)) or [0.0]
    from .energies import same_grid
    done_points = list(record.get("points") or ())
    grid = done_points[0]["energy_ev"] if done_points else []
    for p in done_points:
        # ONE GRID, energy for energy (`energies.same_grid`): a point that
        # shares energies holds the finished point's (§ 2a.16).
        if not same_grid(p["energy_ev"], grid):
            raise DataFileError(
                f"{p['point']}'s transmission is on another "
                f"energy grid than the first point's -- every point's "
                f"transmission deck is the one template's")
    # THE SPIN CHANNELS, by the record's names: a polarized point's two
    # channels, or the one channel of the spin its deck states.
    polarized = any(p.get("spin") == "polarized" for p in done_points)
    spins = (list(next(p["channels"] for p in done_points
                       if p.get("channels"))) if polarized
             else list(dict.fromkeys(p["spin"] for p in done_points)))
    leads = list(LEAD_NAMES)

    def by_spin(p) -> List[Dict]:
        """Each spin channel's own block of a point's DOS, in ``spins``'
        order: a polarized point's ``by_spin``, a non-polarized one's its
        one channel's -- the block itself."""
        dos = p.get("dos") or {}
        if polarized:
            return [(dos.get("by_spin") or {}).get(s) or {} for s in spins]
        return [dos]
    n_eig = max((len(c.get("eigenchannels") or ())
                 for p in done_points for c in by_spin(p)), default=0)
    iv = record.get("iv") or {}
    iv_biases = _voltages(task.bias)
    n_atoms = junction.n_atoms if junction is not None else 0
    # THE ATOM AXIS, LISTED BY USER INDEX, with each atom's engine index --
    # the device run's, from the composed junction's order record
    # (`model/overview.md` § 2.2b, F2: a result for the user), so a script
    # asks sisl about a point's `.TBT.nc` straight from this file.
    from ..atom_permutation import AtomIndexMap
    atoms = AtomIndexMap.of_engine_record(
        recorded.structure.atom_order if recorded is not None else None,
        n_atoms).as_listed_by_user()
    # EVERY RUN'S CONVERGENCE: the rungs that run once, by the stage table;
    # the device's runs, one a point; one SCF-row axis long enough for the
    # longest; TranSIESTA's charge columns as its header named them.
    by_stage = {s.get("stage"): s for s in record.get("stages") or ()}
    once = [s for s in TRANSPORT_STAGES if s not in per_point_rungs()]
    device = by_stage.get("device") or {}
    device_runs = list(device.get("by_point") or ()) or (
        [device] if device else [])
    histories = ([by_stage.get(r) or {} for r in once] + device_runs)
    n_cycles = max((len(h.get("scf") or ()) for h in histories), default=0)
    # EVERY NEGF ROW'S charge columns, and the last cycle's.
    parts = list(dict.fromkeys(
        [k for h in device_runs
         for k in (((h.get("negf") or {}).get("charges")) or {})]
        + [k for h in device_runs for row in h.get("scf") or ()
           for k in (row.get("charges") or {})]))

    dims = {"frame": n_frames, "bias_v": len(biases), "energy_ev": len(grid),
            "spin": len(spins), "iv_bias_v": len(iv_biases), "atom": atoms,
            "xyz": 3, "cell_vector": 3, "region": len(_REGIONS),
            "lead": len(leads),
            "eigenchannel": n_eig, "rung": len(once), "scf_cycle": n_cycles,
            "charge_part": len(parts)}
    values: Dict[str, Any] = {v.name: empty(v, dims) for v in VARIABLES}
    # EVERY POINT STATES WHETHER IT IS DONE: 0 until its transmission says 1.
    values["done"][...] = 0
    values["frame"][:] = np.arange(n_frames)
    if n_frames > 1:
        for f in range(n_frames):
            values["frame_token"][f] = frame_token(f)
    values["bias_v"][:] = biases
    values["energy_ev"][:] = grid
    values["spin"][:] = spins
    values["iv_bias_v"][:] = iv_biases
    values["atom"][:] = np.arange(n_atoms)
    values["xyz"][:] = ["x", "y", "z"]
    values["region"][:] = list(_REGIONS)
    values["lead"][:] = leads
    values["eigenchannel"][:] = np.arange(n_eig)
    values["rung"][:] = once
    values["scf_cycle"][:] = np.arange(n_cycles)
    values["charge_part"][:] = parts
    values["cell_vector"][:] = ["a", "b", "c"]
    if junction is not None:
        # The element, never the species label (`Au1` is gold): the
        # variable says "the atom's element" (`Structure.chemical_symbols`).
        values["element"][:] = junction.chemical_symbols()
        values["positions_ang"][:] = np.asarray(
            junction.frames if n_frames > 1 else [junction.positions])
        # THE JUNCTION WHOLE: its cell and periodicity, each atom's region
        # molbuilder owns and every label it carries -- the user order's,
        # as the restored junction states them.
        if junction.cell is not None:
            values["cell_ang"][:] = np.asarray(junction.cell, dtype=float)
        values["cell_axis_kind"][:] = list(junction.axis_kind)
        from .record import device_regions
        for name, members in device_regions(junction).items():
            for a in members:
                values["atom_region"][a] = name
        carried: List[List[str]] = [[] for _ in range(n_atoms)]
        for label, members in (junction.regions or {}).items():
            for a in members:
                carried[int(a)].append(label)
        for a, labels in enumerate(carried):
            values["labels"][a] = json_attr(labels)

    def at(p) -> Tuple[int, int]:
        f = p.get("frame")
        return (0 if f is None else int(f)), _index(p["bias_v"], biases)
    # EACH RUN'S ENGINE, as its output states it: the stages', each point's
    # TBtrans, each point's device run (`engines/transport.md` § 2a.12).
    versions: Dict[str, Dict] = {"stage": {}, "point": {}, "device": {}}
    for p in done_points:
        f, b = at(p)
        values["done"][f, b] = 1
        values["point_state"][f, b] = p.get("state") or ""
        values["point_folder"][f, b] = p.get("attempt") or ""
        values["transmission"][f, b] = p["transmission"]
        chans = p.get("channels") or {}
        for s, name in enumerate(spins):
            values["transmission_by_spin"][f, b, s] = (
                chans[name] if polarized else p["transmission"])
        for name in ("conductance_g0", "current_a", "current_from"):
            if p.get(name) is not None:
                values[name][f, b] = p[name]
        # EACH CHANNEL'S PRINTED CURRENT: a polarized point's two, a
        # non-polarized one's its one -- none for a point whose energies two
        # runs computed, its channels' the record's over its joined grid.
        printed = ({} if p.get("current_from") == "joined"
                   else p.get("current_by_spin_a") or {} if polarized
                   else {spins[0]: p.get("current_a_printed")}
                   if spins else {})
        for s, name in enumerate(spins):
            if printed.get(name) is not None:
                values["current_by_spin_a"][f, b, s] = printed[name]
        # THE TRANSMISSION'S OWN SETTINGS AND TIME, what it shares and what
        # it is a copy of, and its DOS per orbital's reading.
        for key, name in (("k_points", "tbt_k_points"),
                          ("k_method", "tbt_k_method"),
                          ("loop_elapsed_s", "loop_elapsed_s"),
                          ("copied_from", "copied_from")):
            if p.get(key) is not None:
                values[name][f, b] = p[key]
        _shared_facts(p, (f, b), values)
        if p.get("dos_why"):
            values["dos_why"][f, b] = p["dos_why"]
        # WHICH ATOMS ITS DOS COVERS, where its reading is kept.
        covered = (p.get("dos") or {}).get("device_atoms")
        if covered is not None:
            values["dos_atom"][f, b, :] = 0
            values["dos_atom"][f, b, list(covered)] = 1
        if p.get("engine_version"):
            versions["point"][(f, b)] = str(p["engine_version"])
        kept = p.get("digest") or ()
        if kept:
            values["dos_per_orbital_reading"][f, b] = (
                "kept" if all(d.get("state") == "kept" for d in kept)
                else "; ".join(tried.get(d.get("file"))
                               or f"{d.get('file')}: {d.get('said')}"
                               for d in kept))
        dos = p.get("dos") or {}
        if dos.get("energy_ev") is not None and not (
                len(dos["energy_ev"]) == len(grid)
                and np.allclose(dos["energy_ev"], grid, rtol=0.0,
                                atol=PRINTED_HALF_DECIMAL_EV)):
            raise DataFileError(
                f"{p['point']}'s DOS is on another energy "
                f"grid than its transmission -- not the same points within "
                f"the {PRINTED_HALF_DECIMAL_EV:g} eV the transmission "
                f"file's five decimals allow")
        if dos.get("total") is not None:
            values["dos_total"][f, b] = dos["total"]
        for r, region in enumerate(_REGIONS):
            if region in (dos.get("by_label") or {}):
                values["dos_region"][f, b, r] = dos["by_label"][region]
        for key, name in (("lead_spectral", "lead_spectral_dos"),
                          ("lead_bulk", "lead_bulk_dos")):
            for k, lead in enumerate(leads):
                if lead in (dos.get(key) or {}):
                    values[name][f, b, k] = dos[key][lead]
        for s, channel in enumerate(by_spin(p)):
            if channel.get("total") is not None:
                values["dos_total_by_spin"][f, b, s] = channel["total"]
            for k, curve in enumerate(channel.get("eigenchannels") or ()):
                values["eigenchannel_transmission"][f, b, s, k] = curve
    for p in (list(record.get("pending") or ())
              + list(record.get("not_run") or ())
              + list(record.get("failed") or ())):
        f, b = at(p)
        values["point_state"][f, b] = p.get("state") or ""
        values["point_folder"][f, b] = p.get("attempt") or ""
        values["point_why"][f, b] = p.get("why") or ""
        values["point_progress"][f, b] = (p.get("progress") or {}).get(
            "said") or ""
        _shared_facts(p, (f, b), values)
    rows = list(zip(iv.get("frame") or [None] * len(iv.get("voltages_v")
                                                   or ()),
                    iv.get("voltages_v") or ()))
    for (f, v), i in zip(rows, iv.get("current_a") or ()):
        if i is not None:
            values["iv_current_a"][0 if f is None else int(f),
                                   _index(v, iv_biases)] = i
    # EACH SPIN'S I-V CURRENT, a polarized calculation's (the record's).
    for s, key in ((spins.index("up") if "up" in spins else None,
                    "current_up_a"),
                   (spins.index("down") if "down" in spins else None,
                    "current_down_a")):
        if s is None:
            continue
        for (f, v), i in zip(rows, iv.get(key) or ()):
            if i is not None:
                values["iv_current_by_spin_a"][0 if f is None else int(f),
                                               _index(v, iv_biases), s] = i
    _device_facts(device, device_runs, biases, parts, values)
    for p in device_runs:
        if p.get("engine_version"):
            cells = ([(int(p.get("frame") or 0), _index(p["bias_v"], biases))]
                     if device.get("by_point") else
                     [(f, b) for f in range(n_frames)
                      for b in range(len(biases))])
            for cell in cells:
                versions["device"][cell] = str(p["engine_version"])
    for r, name in enumerate(once):
        if (by_stage.get(name) or {}).get("engine_version"):
            versions["stage"][name] = str(by_stage[name]["engine_version"])
        _run_facts(by_stage.get(name) or {}, "rung", (r,), values,
                   _SCF_COLUMNS)
        if (by_stage.get(name) or {}).get("copied_from"):
            values["rung_copied_from"][r] = by_stage[name]["copied_from"]
    average = record.get("average") or {}

    attrs: Dict[str, Any] = {
        "label": str(record.get("label") or task.label),
        "treatment": str(record.get("treatment") or ""),
        # WHETHER THE LOW-BIAS APPROXIMATION WAS MADE, 1 or 0
        # (`engines/transport.md` § 2a.12, *The data file*).
        "low_bias_approximation": int(bool(
            record.get("low_bias_approximation"))),
        "treatment_note": str(record.get("treatment_note") or ""),
        "current_means": json_attr(record.get("current_means") or {}),
        "energies_relative_to_ef": 1,
        "caveat": str(record.get("caveat") or ""),
        "frames": int(n_frames),
        # WHERE EACH RESULT CAME FROM: each rung's run and what it was
        # gathered from, a run of points' points (`record._chain`).
        "provenance_chain": json_attr(
            (record.get("provenance") or {}).get("chain") or []),
        # THE CALCULATION'S OWN RECORDS, as they read when the file was
        # written -- its description, its template, its job request -- each
        # from its owner (`_records_of`).
        **_records_of(calc),
    }
    slot = ((record.get("provenance") or {}).get("slot") or {})
    for key in ("citation", "kind"):
        if slot.get(key) is not None:
            attrs[f"citation_{key}" if key != "citation" else "citation"] = (
                str(slot[key]))
    # THE FRAMES' PART -- their rows, masses and average, and what is said
    # of them: the post-process's (`frame_average`).
    fill_data_file(values, attrs, junction, average, biases, _index)
    # HOW THE I-V WAS COMPUTED, as the record states it (§ 2a.10).
    attrs["iv_computed"] = str(iv.get("computed") or "")
    if iv.get("kt_ev") is not None:
        values["iv_kt_ev"][...] = float(iv["kt_ev"])
        values["iv_window_lowest_ev"][...], \
            values["iv_window_highest_ev"][...] = iv["window_ev"]
    # EACH NOTE PLACED BY ITS FRAME AND VOLTAGE (`record.iv_notes_placed`):
    # the record keys them by its table's rows, an axis this file has not.
    attrs["iv_notes"] = json_attr([
        {**n, "frame": 0 if n["frame"] is None else int(n["frame"])}
        for n in iv_notes_placed(iv)])
    # THE FERMI FRAMES, EACH RUN'S TOLERANCES (§ 2a.12).
    attrs["fermi_frames"] = json_attr(record.get("fermi_frames") or {})
    attrs["scf_criteria"] = json_attr(
        {name: by_stage[name]["scf_criteria"] for name in TRANSPORT_STAGES
         if (by_stage.get(name) or {}).get("scf_criteria")})

    # THE ENGINE THAT COMPUTED IT, as the runs' outputs state it (the user,
    # 2026-11-16: "transport can also save siesta version where appropriate
    # in the meta data").
    found = sorted({v for by in versions.values() for v in by.values()})
    engine = (f"SIESTA {' and '.join(found)}" if found
              else "SIESTA, its version not stated by the runs' outputs")

    # THE FILE, through the variable record's one writing loop (`ncdata`),
    # group by group, stating itself (`model/datafiles.md` § 2b).
    with NC_LOCK:
        ds = in_memory("transport.nc")
        state(ds, title=(f"the transport calculation {attrs['label']}: "
                         f"every result in groups -- the junction, each "
                         f"stage that runs once, each point and its device "
                         f"run, and a summary across the points"),
              schema=DATA_SCHEMA, molbuilder_version=__version__,
              source=(f"computed by {engine} -- its TranSIESTA and TBtrans "
                      f"-- in the calculation's runs, each stage's run named "
                      f"in provenance_chain and each point's in its folder; "
                      f"each variable's source names its file"))
        # A FRAME ROW'S DEFINITION is its meaning as the set writes it
        # (`model/datafiles.md` § 2).
        _write_groups(ds, dims, with_meanings(VARIABLES, junction), values,
                      label=attrs["label"], once=once, biases=biases,
                      n_frames=n_frames, versions=versions)
        ds.setncatts(attrs)
        data = as_bytes(ds)
    return data_file_path(base, attrs["label"]).name, data


def _write_groups(ds, dims: Dict[str, Any], variables: Sequence[Var],
                  values: Dict[str, Any], *, label: str, once: Sequence[str],
                  biases: Sequence[float], n_frames: int,
                  versions: Dict[str, Dict]) -> None:
    """THE CALCULATION AS GROUPS (`engines/transport.md` § 2a.12): every
    variable in its home (:func:`_home`) -- a stage's and a point's sliced
    from the one array at its index, the prefix its group names dropped --
    and the summary's (:data:`SUMMARY`) whole, so a value two groups hold is
    one array's.  Each group holds the coordinate of every axis it runs
    over, so it is read alone."""
    from dataclasses import replace
    from ..ncdata import write_group
    from ..paths import frame_token
    from ..task import bias_token
    from .stages import Point
    axes = {v.name: v for v in variables if _is_axis(v)}
    homes = {v.name: _home(v) for v in variables if not _is_axis(v)}
    labels_of = [v for v in variables if v.auxiliary]

    def put(group, members, at: Dict[str, int], drop: str, title: str,
            extra: Sequence[Tuple[Var, Any]] = ()) -> None:
        """``members`` sliced at ``at``, ``drop`` taken off their names, and
        the coordinates of the axes they run over -- one ``write_group``."""
        out_vars: List[Var] = []
        out_values: Dict[str, Any] = {}
        for v in members:
            name = (v.name[len(drop):] if drop and v.name.startswith(drop)
                    else v.name)
            idx = tuple(at.get(d, slice(None)) for d in v.dims)
            out_vars.append(replace(v, name=name, dims=tuple(
                d for d in v.dims if d not in at)))
            # A CELL OF THE ONE ARRAY, kept an array of its own type: a
            # text cell indexes out as a bare string, which the writer
            # cannot size.
            whole = values[v.name]
            out_values[name] = (np.asarray(whole[idx], dtype=whole.dtype)
                                if idx else whole)
        for v, value in extra:
            out_vars.append(v)
            out_values[v.name] = np.asarray(value)
        used = list(dict.fromkeys(d for v in out_vars for d in v.dims))
        for d in used:
            out_vars.append(axes[d])
            out_values[d] = values[d]
        # THE LABELS OF AN AXIS'S POINTS GO WITH IT -- a frame's folder and
        # rows, an atom's element, region and labels -- into every group
        # that runs over it, so each is read alone (CF `coordinates`).
        for v in labels_of:
            if v.dims[0] in used and v.name not in out_values:
                out_vars.append(v)
                out_values[v.name] = values[v.name]
        write_group(group, {d: dims[d] for d in used}, out_vars, out_values,
                    title=title)

    def members(home: str) -> List[Var]:
        return [v for v in variables if homes.get(v.name) == home]

    put(ds.createGroup("structure"), members("structure"), {}, "",
        f"the junction of {label} as the calculation composed it, listed "
        f"by user index: its cell, each atom with its region and labels, "
        f"every frame")
    stages = ds.createGroup("stages")
    stages.setncattr("title", "each stage that runs once, a group each, by "
                              "its stage name")
    for r, name in enumerate(once):
        g = stages.createGroup(name)
        put(g, members("stage"), {"rung": r}, _PREFIX["stage"],
            f"the stage {name}: its run's state, energy, Fermi level and "
            f"SCF history")
        g.setncattr("engine_version", versions["stage"].get(name, ""))
    points = ds.createGroup("points")
    points.setncattr("title", "every point the description asks for, done "
                              "or not, a group each, named as its folders "
                              "are")
    for f in range(n_frames):
        under = points
        if n_frames > 1:
            under = points.createGroup(frame_token(f))
            under.setncattr("title", f"the points of frame {frame_token(f)}")
        for b, volts in enumerate(biases):
            said = Point(frame=f if n_frames > 1 else None, volts=volts,
                         of=n_frames if n_frames > 1 else None).words()
            g = under.createGroup(bias_token(volts))
            put(g, members("point"), {"frame": f, "bias_v": b},
                _PREFIX["point"],
                f"the point {said}: its transmission, its DOS, its current "
                f"and its state",
                extra=(
                    (Var("frame", (), "coordinate", "1", "frame index",
                         "the point's frame, counted from 0 -- the "
                         "summary's frame axis at this point", dtype="i4"),
                     f),
                    (Var("bias_v", (), "coordinate", "V", "bias voltage",
                         "the point's voltage -- the summary's bias_v axis "
                         "at this point"), volts)))
            g.setncattr("engine_version", versions["point"].get((f, b), ""))
            d = g.createGroup("device")
            put(d, members("device"), {"frame": f, "bias_v": b},
                _PREFIX["device"],
                f"the device run of the point {said} -- the one its "
                f"transmission gathered: its NEGF figures, charges and SCF "
                f"history")
            d.setncattr("engine_version", versions["device"].get((f, b), ""))
    put(ds.createGroup("summary"),
        [v for v in variables if v.name in SUMMARY], {}, "",
        "what is compared across the points, gathered from the same arrays "
        "as each point's own: each point's numbers and T(E) over frame x "
        "voltage, the I-V, and the frames' average")


def _records_of(calc) -> Dict[str, str]:
    """``{description, template, job_request}`` -- the opened calculation's
    own records as read (`architecture.md` A16): its description and its job
    request in the text their writers write (`persist.json_text`, as
    `task.write_task` and `job_request.write` write them), so a hand edit's
    spacing is not kept; its template's text."""
    from ..persist import json_text
    return {"description": json_text(calc.task.to_dict()),
            "template": calc.template.text,
            "job_request": json_text(calc.request.to_dict())}


def _device_facts(device: Dict, runs: Sequence[Dict],
                  biases: Sequence[float], parts: Sequence[str],
                  values: Dict[str, Any]) -> None:
    """Each point's device facts from the record's device rung -- its run
    state, convergence, NEGF figures, contour and SCF history: a rung that
    runs per point, point by point; one that runs once, its one run's, at
    every point."""
    for p in runs:
        if device.get("by_point"):
            f = 0 if p.get("frame") is None else int(p["frame"])
            cell = (f, _index(p["bias_v"], biases))
        else:
            cell = (Ellipsis,)
        negf = p.get("negf") or {}
        contour = negf.get("contour") or {}
        line = contour.get("neq_line") or {}
        for value, name in ((negf.get("ef"), "device_ef_ev"),
                            (negf.get("vha_ev"), "device_vha_ev"),
                            (negf.get("energy"), "device_energy_ev"),
                            (negf.get("cycles"), "device_negf_cycles"),
                            (negf.get("dq"), "device_dq"),
                            (negf.get("d_change_from_0v_e"),
                             "device_d_change_from_0v"),
                            (contour.get("poles"), "device_contour_poles"),
                            (line.get("emin_ev"), "device_neq_emin_ev"),
                            (line.get("emax_ev"), "device_neq_emax_ev"),
                            (line.get("points"), "device_neq_points")):
            if value is not None:
                values[name][cell] = value
        for k, part in enumerate(parts):
            if part in (negf.get("charges") or {}):
                values["device_charge"][cell + (k,)] = negf["charges"][part]
        _run_facts(p, "device", cell, values,
                   _SCF_COLUMNS + _NEGF_SCF_COLUMNS)
        # EVERY NEGF ROW'S CHARGES, column by column.
        for c, row in enumerate(p.get("scf") or ()):
            for k, part in enumerate(parts):
                if part in (row.get("charges") or {}):
                    values["device_scf_charge"][cell + (c, k)] = (
                        row["charges"][part])


def _shared_facts(point: Dict, cell: Tuple, values: Dict[str, Any]) -> None:
    """What a point shares with a finished calculation's (`engines/
    transport.md` § 2a.16), at ``cell`` -- done or not, as the record
    states it."""
    shared = point.get("shared") or {}
    for key, name in (("from", "shared_from"), ("held", "shared_held"),
                      ("computed", "shared_computed")):
        if shared.get(key) is not None:
            values[name][cell] = shared[key]


def _run_facts(run: Dict, prefix: str, cell: Tuple, values: Dict[str, Any],
               columns) -> None:
    """One run's state and its sentence, convergence and SCF history into
    ``prefix``'s variables at ``cell`` -- a rung's index, or a device
    point's (frame, voltage)."""
    for key in ("state", "detail"):
        if run.get(key):
            values[f"{prefix}_{key}"][cell] = run[key]
    if run.get("scf_converged") is not None:
        values[f"{prefix}_scf_converged"][cell] = int(bool(
            run["scf_converged"]))
    if prefix == "rung":
        for key, name in (("energy_ev", "rung_energy_ev"),
                          ("fermi_ev", "rung_fermi_ev")):
            if run.get(key) is not None:
                values[name][cell] = run[key]
    if run.get("scf_phases"):
        values[f"{prefix}_scf_phases"][cell] = json_attr(run["scf_phases"])
    # A RUN THAT DID NOT RUN states no history: its count keeps the fill,
    # -1, never 0, which is a count (§ 2a.12).
    if "scf" not in run:
        return
    rows = run["scf"] or ()
    values[f"{prefix}_scf_cycles"][cell] = len(rows)
    for c, row in enumerate(rows):
        for (key, suffix, *_rest) in columns:
            if row.get(key) is not None:
                values[f"{prefix}_scf_{suffix}"][cell + (c,)] = row[key]


def read_data_file(path) -> Dict[str, Any]:
    """``<label>.transport.nc`` as data: ``{"attrs": {...}, "groups":
    {path: {"attrs": {...}, "variables": {name: {"dims", "data",
    "attrs"}}}}}`` -- every group by its path (``structure``,
    ``stages/seed``, ``points/v0.2``, ``points/v0.2/device``, ``summary``),
    each variable's values as numpy arrays (a number's fill NaN, a flag's
    -1), its units, long name, definition and source; the root's attributes
    written as JSON text (:data:`JSON_ATTRS`, the frames' part's
    `frame_average.JSON_ATTRS`) as the block each holds, where the file
    holds it.  The file's door (`runfiles.WRITTEN`); refuses another schema
    by name."""
    from ..frame_average import JSON_ATTRS as FRAME_JSON_ATTRS
    from ..ncdata import read_variables, reading, unjson_attr
    groups: Dict[str, Dict[str, Any]] = {}

    def walk(group, at: str) -> None:
        for name, g in group.groups.items():
            here = f"{at}/{name}" if at else name
            groups[here] = {"attrs": {k: g.getncattr(k) for k in g.ncattrs()},
                            "variables": read_variables(g)}
            walk(g, here)
    with reading(path) as ds:
        attrs = {k: ds.getncattr(k) for k in ds.ncattrs()}
        if attrs.get("schema") != DATA_SCHEMA:
            raise DataFileError(f"{Path(path).name} states the schema "
                                f"{attrs.get('schema')!r}, not {DATA_SCHEMA} "
                                f"-- jobset summarize task writes it again")
        walk(ds, "")
    attrs.update({k: unjson_attr(attrs[k])
                  for k in JSON_ATTRS + FRAME_JSON_ATTRS if k in attrs})
    return {"attrs": attrs, "groups": groups}
