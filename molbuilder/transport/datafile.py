"""THE TRANSPORT DATA FILE -- ``<label>.transport.nc``, the record's results
as data on their grid (`engines/transport.md` § 2a.12, *the data file*;
every name it holds, `model/datafiles.md` § 9.1).

MODULE  transport.datafile (L2; numpy, netCDF4 and molbuilder's own doors)
ROLE    one NetCDF-4 file per transport calculation, in groups -- the
        junction, each stage that runs once, each point and its device run,
        and a summary across the points (`engines/transport.md` § 2a.12,
        *The calculation as groups*): every point's raw outputs beside what
        molbuilder derives from them, each frame's weight beside its values
        and the frames' average -- that part the frames' post-process's
        (`frame_average`), every other row of a frame set its maker's,
        carried in the junction's ``structure_json`` -- each variable stating its units, its long name,
        its definition and, for a raw one, the file it was read from.  The
        variables are :data:`VARIABLES`, one table the writer fills and the
        reader reads, each placed by :func:`_home` and :data:`SUMMARY`;
        :func:`data_file_bytes` builds it from the composition `summarize
        task` writes the JSON record from -- each value copied from the
        record at the record key its variable states (`ncdata.copy_stated`)
        -- :func:`write_data_file` writes it; :func:`read_data_file` is its
        door, :func:`point_groups` its points by folder
USED-BY `jobset summarize task`, beside `record.write_record`; an export
        (`transport.export`), which hands the bytes out; a person's script
        through `postprocess.read_transport_data`; the PDOS from the files
        (`tbtnc.pdos_from_files`), the frames averaged at weights given
        (`frame_average.averaged_at`) and a frame set's rows
        (`postprocess.frame_rows`) -- each reading through
        :func:`read_data_file`

**One shape for every transport calculation** (user, 2026-10-10: "doing a
derivative is trivial if the result are saved with these parameter and
calculation results consistently aligned and logically organized"): one
structure at one voltage is a 1 x 1 grid, never another layout.  A point not
done is the fill value, with ``done`` 0 -- a gap, never filled in.  The JSON
record is the report -- `summarize` writes it for the command line, and the
Results tab composes the same record on read; this file is the same results
for analysis -- an average, a slope along a frame's row (read from the
junction's structure, `postprocess.frame_rows`), a comparison of two
calculations is a line of xarray on it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..ncdata import (Var, coordinate, copy_stated, json_attrs,
                      scf_history_values, scf_history_vars, structure_attrs,
                      structure_values, structure_vars)
from ..runfiles import (CHANNEL_UNPOLARIZED, ROLE_TBT_DIGEST,
                        SPIN_POLARIZED, TBT_CHANNEL_TAGS, TBT_SPIN_CHANNELS,
                        TBTRANS_AVADOS, TBTRANS_AVBDOS, TBTRANS_AVDOS,
                        TBTRANS_AVTEIG, TBTRANS_AVTRANS, TBTRANS_NC, shown)

#: The unpolarized channel's tag, which the variables' sources name.
_TBT = TBT_CHANNEL_TAGS[CHANNEL_UNPOLARIZED]
from ..engine_table import engine as _engine
from ..kind_table import kind as _kind
from ..units import volts_words
from .sort import DEVICE_REGIONS
from .stages import DEVICE_RUNG, ENGINE, same_volts, voltages_once
from .window import window_rung

#: The file's schema, stated in it: an older file is refused by its
#: version, and `summarize task` writes it again.
DATA_SCHEMA = "molbuilder/transport-data@14"


#: THE FILE'S GROUPS by name (`engines/transport.md` § 2a.12, *The
#: calculation as groups*): the junction, the stages that run once, the
#: points -- each a group below ``points``, its device run's below it --
#: and the summary across them; read by name by every reader of the file.
STRUCTURE_GROUP, STAGES_GROUP, POINTS_GROUP, SUMMARY_GROUP, DEVICE_GROUP = (
    "structure", "stages", "points", "summary", "device")

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


class DataFileError(ValueError):
    """The record does not lay out on one grid -- the message names what
    disagrees, ready to surface verbatim."""


def _scf_columns() -> tuple:
    """THE SCF ROW, column by column: the engine's -- every stage and
    device run is one (`stages.ENGINE`) -- declared once, named by the
    engine table (``scf_columns``; `model/datafiles.md` § 9.0.6)."""
    from ..engine_table import scf_columns
    return scf_columns(ENGINE)


def _variables() -> Tuple[Var, ...]:
    from ..frame_average import data_file_variables
    S = str
    return (
        coordinate("frame", more=(
            "the frame set's, as MolView's API counts; a message and the "
            "page number it from 1 (model/structure.md 2.2e)")),
        Var("frame_token", ("frame",), "given", "", "the frame's folder",
            "the name of its folder: f000 is frame 0's "
            "(engines/transport.md 2a.11); empty for one structure, which "
            "has no frame level",
            dtype=S, auxiliary=True),
        Var("bias_v", ("bias_v",), "coordinate", "V", "bias voltage",
            "the voltage the device and the transmission ran at -- the "
            "description's list, in its order; 0 V alone under the "
            "low-bias approximation, whose transmission ran there "
            "(iv_bias_v holds the list)"),
        coordinate("energy_ev", more=(
            "the transmission's grid, at the full precision of each "
            f"point's {shown(TBTRANS_NC, tbt=_TBT)}, ascending; TBtrans's "
            "text files, printed to 5 decimals, put on it within half that "
            "decimal")),
        coordinate("spin", more="the channels TBtrans wrote for the "
                                "point"),
        Var("iv_bias_v", ("iv_bias_v",), "coordinate", "V",
            "the I-V's voltage",
            "the description's voltage list, in its order"),
        # THE JUNCTION WHOLE, in the framework's one structure layout
        # (`model/datafiles.md` § 3a; the user, 2026-11-16: "junction's cell
        # should be saved together with structure"): each atom's element,
        # every frame's positions, its mass and whether it was held, the
        # cell and each cell vector's periodicity, the structure whole in
        # `structure_json` -- its labels among it.
        *structure_vars(frames=True),
        # THE JUNCTION'S BASIS, every orbital in the device run's order --
        # what a PDOS of chosen atoms narrows by (`engines/transport.md`
        # § 2a.12, *A PDOS of the atoms you choose, from the files*).
        Var("orbital", ("orbital",), "coordinate", "1", "orbital index",
            "its index in the geometry, from 0, the device run's order -- "
            "a digest's device_orbital names a subset of them", dtype="i4"),
        Var("orbital_atom", ("orbital",), "given", "1", "the orbital's atom",
            "the atom it sits on, by the file's atom index map (user "
            "index)", dtype="i4", atom_index=True),
        *(Var(f"orbital_{q}", ("orbital",), "raw", "1", words,
              f"as the device run's ORB_INDX states it ({key})",
              source="the device run's <label>.ORB_INDX", dtype="i4")
          for q, words, key in (("n", "the orbital's principal number", "n"),
                                ("l", "the orbital's angular momentum", "l"),
                                ("m", "the orbital's magnetic number", "m"),
                                ("zeta", "the orbital's zeta", "z"))),
        Var("orbital_type", ("orbital",), "derived", "", "the orbital's type",
            "its l and m in the orbital menu's words -- s, px, py, pz, d "
            "(tbtnc.ORBITAL_TYPES) -- empty for an l the menu does not "
            "offer", dtype=S),
        # EACH ATOM'S REGION molbuilder owns (`engines/transport.md`
        # § 2a.12, *The junction, whole*).
        Var("atom_region", ("atom",), "given", "", "the atom's region",
            "the region molbuilder owns that it lies in -- L-electrode, "
            "bridge, R-electrode -- from the composed junction's labels "
            "(record.device_regions); empty for an atom in none.  dos_region_per_ev "
            "sums the atoms of a region that a point's DOS covers "
            "(dos_atom); the region axis names the three",
            dtype=S, auxiliary=True),
        Var("region", ("region",), "coordinate", "", "device region",
            "the regions molbuilder owns: L-electrode, bridge, R-electrode",
            dtype=S),
        coordinate("lead", more="transiesta.LEAD_NAMES"),
        coordinate("eigenchannel", more=(
            "as many as its files hold, those it was asked for (TBT.T.Eig), "
            "each spin channel's; none where no file is read")),
        # THE FRAMES' POST-PROCESS: each frame's weight and their average
        # (`frame_average`).
        *data_file_variables(),
        Var("transmission_by_spin", ("frame", "bias_v", "spin", "energy_ev"),
            "raw", "1", "transmission, by spin channel",
            "T(E), k-averaged, channel by channel",
            source=("<label>" + shown(TBTRANS_AVTRANS, tbt=_TBT,
                                                       pair="<L>-<R>")
                    + " (TBT_UP / TBT_DN polarized)")),
        Var("transmission", ("frame", "bias_v", "energy_ev"), "derived", "1",
            "the transmission of one spin channel",
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
            "the electronic temperature the 0 V transmission's deck states; "
            "NaN where the I-V is TBtrans's own"),
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
        Var("device_fermi_level_ev", ("frame", "bias_v"), "raw", "eV",
            "the device point's Fermi level",
            "its NEGF phase's last cycle, in TranSIESTA's frame",
            source="the device point's SIESTA output", key="negf.ef"),
        Var("device_vha_ev", ("frame", "bias_v"), "raw", "eV",
            "the device point's boundary Hartree potential",
            "its NEGF phase's last cycle's ts-Vha, as TranSIESTA printed it "
            "-- a value, not the shift between the device's zero and the "
            "seed's, which is the periodic start's (`fermi_frames`)",
            source="the device point's SIESTA output", key="negf.vha_ev"),
        Var("dos_total_per_ev", ("frame", "bias_v", "energy_ev"), "raw", "1/eV",
            "the device's density of states", "its Green-function DOS, "
            "k-averaged: a non-polarized point's one spin's, a polarized "
            "point's up + down",
            source=_from(shown(TBTRANS_AVDOS, tbt=_TBT)), key="dos.total"),
        Var("dos_total_by_spin_per_ev", ("frame", "bias_v", "spin", "energy_ev"),
            "raw", "1/eV", "the device's density of states, by spin channel",
            "each channel's Green-function DOS, k-averaged -- a "
            "non-polarized point's one channel", key="total",
            source=("<label>" + shown(TBTRANS_AVDOS, tbt=_TBT)
                    + " (TBT_UP / TBT_DN polarized)")),
        Var("dos_region_per_ev", ("frame", "bias_v", "region", "energy_ev"), "raw",
            "1/eV", "the device DOS by region",
            "the point's DOS per orbital summed over the orbitals of the "
            "atoms whose atom_region is the region and whose dos_atom is 1 "
            "-- the atoms of it the point's DOS covers: a non-polarized "
            "point's one spin's, a polarized point's up + down",
            source=_from(shown(ROLE_TBT_DIGEST, tbt=_TBT)
                         + ", the data file's DOS per orbital")),
        Var("lead_spectral_dos_per_ev", ("frame", "bias_v", "lead", "energy_ev"), "raw",
            "1/eV", "the device's DOS from a lead's states",
            "TBT.DOS.A, k-averaged: the device's spectral DOS of the states "
            "coming in from this lead, not the lead's own (lead_bulk_dos_per_ev); "
            "a non-polarized point's one spin's, a polarized point's up + "
            "down; TBtrans computes it for every lead but the last "
            "(TBT.DOS.A.All not set), whose row is the fill",
            source=_from(shown(TBTRANS_AVADOS, tbt=_TBT, lead="<lead>"))),
        Var("lead_bulk_dos_per_ev", ("frame", "bias_v", "lead", "energy_ev"), "raw",
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
            "read -- for a point not done, where its attempt is", dtype=S,
            key="attempt"),
        Var("point_why", ("frame", "bias_v"), "state", "",
            "why a point is not done",
            "in its run's words: pending, or failed and what stopped it",
            dtype=S, key="why"),
        Var("point_progress", ("frame", "bias_v"), "state", "",
            "how far a point not done got",
            "its TBtrans loop -- running, or where it stopped -- in its run "
            "record's sentence (model/parse.md 5d.1b)", dtype=S,
            key="progress.said"),
        Var("dos_why", ("frame", "bias_v"), "state", "",
            "why a done point has no DOS",
            "in its reader's words -- its DOS variables the fill value",
            dtype=S),
        Var("tbt_kpoints", ("frame", "bias_v"), "raw", "1",
            "the transmission's transverse k-points",
            "as the point's TBtrans run states it", dtype="i4", fill=-1,
            source="the transmission point's TBtrans output", key="k_points"),
        Var("tbt_kpoint_method", ("frame", "bias_v"), "raw", "",
            "how those k-points were chosen",
            "as the point's TBtrans run states it", dtype=S,
            source="the transmission point's TBtrans output", key="k_method"),
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
            "that point's folder, in the calculation it reuses "
            "(engines/transport.md 2a.16); empty for a point that shares "
            "none", dtype=S, key="shared.from"),
        Var("shared_held", ("frame", "bias_v"), "state", "1",
            "the energies a point shares",
            "how many energies of its grid the finished point and it hold "
            "in common -- those its own run did not compute",
            dtype="i4", fill=-1, key="shared.held"),
        Var("shared_computed", ("frame", "bias_v"), "state", "1",
            "the energies a point's own run computes",
            "beside those it shares", dtype="i4", fill=-1,
            key="shared.computed"),
        Var("dos_atom", ("frame", "bias_v", "atom"), "state", "1",
            "the point's DOS covers the atom",
            "1 for each atom of TBtrans's device region -- the only atoms it "
            "computes a DOS for -- from the point's kept reading of its data "
            "file (model/parse.md 5e), 0 for every other; -1 where none is "
            "kept", dtype="i1", flag=True),
        Var("dos_per_orbital_files", ("frame", "bias_v"), "state", "",
            "the point's digests, and the reading each is",
            "a JSON list: each digest's path from the calculation's folder "
            "-- where this file lies -- its channel, and the version of the "
            "data file it states, [its size in bytes, its modification time "
            "in ns] when it was read: the reading dos_region_per_ev is summed from, "
            "which postprocess.pdos reads back; empty where none is kept",
            dtype=S, json=True, key="dos.per_orbital"),
        Var("dos_per_orbital_reading", ("frame", "bias_v"), "state", "",
            "the state of the point's DOS-per-orbital reading",
            "kept -- read and kept beside its data file, which "
            "dos_region_per_ev is summed from -- or why not, in the "
            "reading's own words: not read yet, out of date, being read, "
            "or why a reading just tried failed; dos_region_per_ev is the "
            "fill where it is not kept (model/parse.md 5e)", dtype=S),
        # EVERY RUN'S STATE AND CONVERGENCE, on the grid it varies over --
        # a stage's at its group, sliced off the internal ``rung`` axis.
        coordinate("scf_cycle", more=(
            "every row of the run's last step, a device's periodic start "
            "and its NEGF loop alike")),
        Var("charge_part", ("charge_part",), "coordinate", "",
            "TranSIESTA's charge column",
            "a column of TranSIESTA's ts-q: line, as its header names it: "
            "D the bridge -- TranSIESTA's device region; E1, E2 each "
            "electrode block; C1, C2 the overlap population across each "
            "electrode's boundary with the bridge; B the buffer atoms, "
            "where there are some.  The line prints no column for the rest "
            "(O), the overlap populations it leaves out, which its dQ "
            "counts (engines/transport.md 2a.12)",
            dtype=S),
        Var("point_state", ("frame", "bias_v"), "state", "",
            "the transmission point's state",
            "the run door's word for its run (finished, failed, not run, "
            "pending ...)",
            dtype=S, key="state"),
        Var("device_state", ("frame", "bias_v"), "state", "",
            "the device point's state", "the run door's word for its run",
            dtype=S, key="state"),
        Var("device_detail", ("frame", "bias_v"), "state", "",
            "what the device point's state says",
            "the run door's sentence for its run", dtype=S, key="detail"),
        Var("device_scf_converged", ("frame", "bias_v"), "state", "1",
            "the device point's SCF converged",
            "1 converged, 0 not, -1 not stated", dtype="i1", flag=True,
            key="scf_converged"),
        Var("device_total_energy_ev", ("frame", "bias_v"), "raw", "eV",
            "the NEGF phase's total energy",
            "E_KS of its last SCF cycle, converged where its run's "
            "scf_converged says so; never compared across points or "
            "calculations (engines/transport.md 1.1)",
            source="the device point's SIESTA output", key="negf.energy"),
        Var("device_negf_cycles", ("frame", "bias_v"), "raw", "1",
            "how many NEGF cycles ran",
            "the NEGF rows of the run's SCF history -- TranSIESTA's loop, "
            "after the periodic start", source="the device point's SIESTA "
            "output", dtype="i4", fill=-1, key="negf.cycles"),
        Var("device_dq_e", ("frame", "bias_v"), "raw", "e",
            "the last NEGF cycle's charge not conserved",
            "dQ of its ts-q: line, as TranSIESTA printed it: the device "
            "cell's total population -- every region's, and the rest (O) "
            "the line prints no column for -- less the neutral count; the "
            "charge the open boundaries did not conserve",
            source="the device point's SIESTA output", key="negf.dq"),
        Var("device_charge_e", ("frame", "bias_v", "charge_part"), "raw", "e",
            "the last NEGF cycle's Mulliken populations", "electrons, "
            "column by column, as TranSIESTA printed them: a net population "
            "for a region, an overlap population across a boundary -- "
            "positive, electrons counted, not a net charge",
            source="the device point's SIESTA output"),
        Var("device_d_change_from_0v_e", ("frame", "bias_v"), "derived", "e",
            "the bridge's population change from 0 V",
            "D(V) - D(0 V): the bridge's net Mulliken population (the D "
            "column of the device's charge_e) at this voltage less the same "
            "frame's at 0 V -- 0 at 0 V itself; not stated where the frame "
            "has no 0 V point finished and converged; positive where the "
            "bridge gained electrons "
            "(engines/transport.md 2a.12)", key="negf.d_change_from_0v_e"),
        Var("device_contour_poles", ("frame", "bias_v"), "raw", "1",
            "the equilibrium contour's poles", "as TranSIESTA echoed it",
            source="the device point's SIESTA output", dtype="i4", fill=-1,
            key="negf.contour.poles"),
        Var("device_neq_emin_ev", ("frame", "bias_v"), "raw", "eV",
            "the non-equilibrium line's lower end", "under a bias",
            source="the device point's SIESTA output",
            key="negf.contour.neq_line.emin_ev"),
        Var("device_neq_emax_ev", ("frame", "bias_v"), "raw", "eV",
            "the non-equilibrium line's upper end", "under a bias",
            source="the device point's SIESTA output",
            key="negf.contour.neq_line.emax_ev"),
        Var("device_neq_points", ("frame", "bias_v"), "raw", "1",
            "the non-equilibrium line's points", "under a bias",
            source="the device point's SIESTA output", dtype="i4", fill=-1,
            key="negf.contour.neq_line.points"),
        *scf_history_vars(_scf_columns(), ("frame", "bias_v"),
                          whose="the device point",
                          source="the run's SIESTA output",
                          prefix="device_"),
        Var("device_scf_charge_e", ("frame", "bias_v", "scf_cycle",
                                  "charge_part"), "raw", "e",
            "every NEGF row's Mulliken populations", "electrons, column by "
            "column, as TranSIESTA printed them",
            source="the device point's SIESTA output"),
        Var("device_scf_phases", ("frame", "bias_v"), "state", "",
            "the device run's phases", "periodic and negf: what each had to "
            "reach and how it ended, JSON", dtype=S, json=True,
            key="scf_phases"),
        Var("rung_state", ("rung",), "state", "", "the stage's state",
            "the run door's word for its run", dtype=S, key="state"),
        Var("rung_detail", ("rung",), "state", "",
            "what the stage's state says",
            "the run door's sentence for its run", dtype=S, key="detail"),
        Var("rung_scf_converged", ("rung",), "state", "1",
            "the stage's SCF converged", "1 converged, 0 not, -1 not stated",
            dtype="i1", flag=True, key="scf_converged"),
        Var("rung_total_energy_ev", ("rung",), "raw", "eV",
            "the stage's total energy",
            "E_KS of its run's last SCF cycle, as SIESTA printed it -- "
            "converged where scf_converged says so",
            source="the stage's SIESTA output", key="energy_ev"),
        Var("rung_fermi_level_ev", ("rung",), "raw", "eV", "the stage's Fermi level",
            "its last SCF cycle's E_F, in its own run's frame -- how the "
            "frames relate, fermi_frames", source="the stage's SIESTA output",
            key="fermi_ev"),
        *scf_history_vars(_scf_columns(), ("rung",), whose="the stage",
                          source="the run's SIESTA output", prefix="rung_"),
        Var("rung_scf_phases", ("rung",), "state", "", "the stage's phases",
            "what each had to reach and how it ended, JSON", dtype=S,
            json=True, key="scf_phases"),
        Var("rung_copied_from", ("rung",), "state", "",
            "the run a stage's newest run is a copy of",
            "a calculation that reuses a finished one "
            "(engines/transport.md 2a.16)", dtype=S, key="copied_from"),
    )


#: THE FILE'S VARIABLES -- one table the writer fills and the reader reads.
VARIABLES: Tuple[Var, ...] = _variables()

#: WHAT THE SUMMARY GATHERS (`engines/transport.md` § 2a.12, *The
#: calculation as groups*; the user, 2026-11-16: "to access the IV curve,
#: the user could decide to take individual points from each calculation
#: group, or ... just go for the summary ... consistent and coming from the
#: same result"): what is compared across the points -- each point's
#: numbers and its T(E) over frame x voltage, written from the same arrays
#: as each point's own -- the I-V, and the frames' average, whose part is
#: its post-process's to name (`frame_average.SUMMARY_PART`).
def _summary() -> Tuple[str, ...]:
    from ..frame_average import SUMMARY_PART
    return ("done", "point_state", "conductance_g0", "current_a",
            "current_by_spin_a", "current_from", "device_fermi_level_ev",
            "device_vha_ev", "device_dq_e", "device_d_change_from_0v_e",
            "device_scf_converged", "transmission",
            "iv_current_a", "iv_current_by_spin_a", "iv_kt_ev",
            "iv_window_lowest_ev", "iv_window_highest_ev", *SUMMARY_PART)


SUMMARY: Tuple[str, ...] = _summary()


def _record_attrs() -> Tuple[str, ...]:
    from ..frame_average import RECORD_ATTRS as FRAMES
    return ("current_means", "provenance_chain", "iv_notes", "fermi_frames",
            "scf_criteria") + FRAMES


#: THE ATTRIBUTES THE FILE HOLDS AS RECORDS, on its root and the summary
#: (`model/datafiles.md` § 9.1 marks each **record**) -- its own and the
#: frames' part's (`frame_average.RECORD_ATTRS`): the one list they are
#: written as JSON text and read back as blocks by (`ncdata.json_attrs`,
#: `ncdata.read_file`); ``/structure``'s structure the framework's own.
RECORD_ATTRS: Tuple[str, ...] = _record_attrs()

#: THE PREFIX A VARIABLE DROPS in the group its path names
#: (`engines/transport.md` § 2a.12): ``rung_fermi_level_ev`` is a stage's
#: ``fermi_level_ev``, ``point_why`` a point's ``why`` -- the internal
#: ``rung`` axis is cut away before any group is written.
_PREFIX = {"stage": "rung_", "point": "point_", "device": "device_"}

#: THE JUNCTION'S AXES: a variable over these alone -- an atom, an orbital,
#: the cell, a frame's rows -- is the junction's.
_STRUCTURE_AXES = frozenset(("frame", "atom", "xyz", "cell_vector",
                             "orbital"))


def _is_axis(var: Var) -> bool:
    """An axis's own coordinate -- written in each group that runs over it."""
    return var.kind == "coordinate" and var.dims == (var.name,)


def _home(var: Var) -> str:
    """Where a variable lives (`engines/transport.md` § 2a.12): a stage's
    over ``rung``; a point's over ``frame`` and ``bias_v`` -- its device
    run's where it is ``device_*`` -- the summary gathering those
    :data:`SUMMARY` names besides; the summary's what it names of the rest;
    the junction's over its axes alone.  A variable no rule places is
    refused, naming it -- never laid in a group by default."""
    if "rung" in var.dims:
        return "stage"
    if "frame" in var.dims and "bias_v" in var.dims:
        return "device" if var.name.startswith(_PREFIX["device"]) else "point"
    if var.name in SUMMARY:
        return "summary"
    if var.dims and set(var.dims) <= _STRUCTURE_AXES:
        return "structure"
    raise ValueError(f"the data file's variable {var.name} has no group: "
                     f"engines/transport.md § 2a.12 names one first")


#: EACH VARIABLE'S GROUP, placed once as the module loads -- a variable no
#: rule places refuses the module, not a file.
_HOMES: Dict[str, str] = {v.name: _home(v) for v in VARIABLES
                          if not _is_axis(v)}

#: WHAT EACH RECORD FILLS AS IT STATES IT, by the variable's name -- a
#: point's entry its point's, a device run's and a stage's entry that run's,
#: a spin channel's DOS block its channel's -- each value at its record key
#: (`ncdata.copy_stated`, `ncdata.Var.key`).  Every other variable its
#: writer assembles: a spin's channels, a region's DOS, the SCF history row
#: by row (`ncdata.scf_history_values`).
_FROM_RECORD: Dict[str, Tuple[str, ...]] = {
    "point": ("point_state", "point_folder", "point_why", "point_progress",
              "transmission", "conductance_g0", "current_a", "current_from",
              "tbt_kpoints", "tbt_kpoint_method", "loop_elapsed_s",
              "copied_from", "shared_from", "shared_held", "shared_computed",
              "dos_why", "dos_total_per_ev", "dos_per_orbital_files"),
    "device": ("device_state", "device_detail", "device_scf_converged",
               "device_scf_phases", "device_fermi_level_ev", "device_vha_ev",
               "device_total_energy_ev", "device_negf_cycles", "device_dq_e",
               "device_d_change_from_0v_e", "device_contour_poles",
               "device_neq_emin_ev", "device_neq_emax_ev",
               "device_neq_points"),
    "stage": ("rung_state", "rung_detail", "rung_scf_converged",
              "rung_scf_phases", "rung_total_energy_ev",
              "rung_fermi_level_ev", "rung_copied_from"),
    "spin": ("dos_total_by_spin_per_ev",),
}
_BY_NAME: Dict[str, Var] = {v.name: v for v in VARIABLES}
_KEYED: Dict[str, Tuple[Var, ...]] = {
    record: tuple(_BY_NAME[n] for n in names)
    for record, names in _FROM_RECORD.items()}

#: A POINT'S OWN PLACE on the summary's axes, in its group: its frame and its
#: voltage, each the axis's value there (`engines/transport.md` § 2a.12).
_POINT_AT = (
    Var("frame", (), "given", "1", "frame index",
        "the point's frame, counted from 0 -- the summary's frame axis at "
        "this point", dtype="i4"),
    Var("bias_v", (), "given", "V", "bias voltage",
        "the point's voltage -- the summary's bias_v axis at this point"),
)


def data_file_path(base_dir, label: str) -> Path:
    """The ONE spelling of the file's location, composed by the run-file
    catalogue (`runfiles.compose`), as `record.record_path` is."""
    from ..runfiles import ROLE_TRANSPORT_DATA, compose
    return Path(base_dir) / compose(label, ROLE_TRANSPORT_DATA)


def _frame_index(frame: Optional[int]) -> int:
    """A frame's index on the ``frame`` axis -- 0 for one structure, which
    names none (`engines/transport.md` § 2a.12)."""
    return 0 if frame is None else int(frame)


def _cell(frame: Optional[int], v: float,
          voltages: Sequence[float]) -> Tuple[int, int]:
    """A point's cell on the grid, ``(frame, voltage's index)``."""
    return _frame_index(frame), _index(v, voltages)


def _index(v: float, voltages: Sequence[float]) -> int:
    """Where ``v`` sits along ``voltages`` -- every voltage the record
    states is one of the file's, so one that is not is a
    :class:`DataFileError`."""
    k = next((k for k, u in enumerate(voltages) if same_volts(v, u)), None)
    if k is None:
        raise DataFileError(f"{volts_words(v)} is not one of the file's "
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
    :class:`DataFileError` names what does not lay out on one grid, and a
    composed junction that does not read, in its reader's words."""
    from .. import __version__
    from ..frame_average import fill_data_file
    from ..ncdata import NC_LOCK, as_bytes, empty, in_memory, state
    from .compose import recorded_junction
    from .rung import compose_refusals
    from .record import iv_rows, order_unstated_said
    from ..paths import frame_token
    from .rung import frames_had
    from .stages import (TRANSPORT_STAGES,
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
    try:
        recorded = recorded_junction(base)
    except compose_refusals() as exc:
        # A JUNCTION THAT DOES NOT READ, refused in its reader's words
        # (F12's review, E-D1).
        raise DataFileError(str(exc)) from None
    unstated = order_unstated_said(base, recorded)
    if unstated:
        raise DataFileError(unstated)
    junction = (recorded.structure.restored() if recorded is not None
                else None)
    n_frames = frames_had(calc)
    biases = voltages_once(pt.bias_v for pt in rung_points(
        task, window_rung(), frames=n_frames)) or [0.0]
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
    # THE SPIN CHANNELS: a polarized calculation's two, by the record's
    # names, or the one channel of the spin its deck states -- the record's
    # one answer (`record.collect_record`).
    polarized = bool(record.get("polarized"))
    # EACH CHANNEL BY ITS NAME, the one table of a spin's channels
    # (`runfiles.TBT_SPIN_CHANNELS`): a non-polarized deck's one channel is
    # ``unpolarized``, as every data file names it -- never the spin
    # option's own word.
    spins = (list(next(p["channels"] for p in done_points
                       if p.get("channels"))) if polarized
             else list(dict.fromkeys(c for p in done_points
                                     for c in TBT_SPIN_CHANNELS[p["spin"]])))
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
    iv_biases = voltages_once(task.bias)
    # THE JUNCTION'S BASIS, each orbital's atom by user index -- the device
    # run's own order, the engine's (`record.device_orbital_index`).
    # A table that cannot be read is said, never the file refused for it:
    # an export holds what exists (§ 2a.12, *Export*).
    from .record import device_orbital_index
    from .tbtnc import TbtError, basis_table, orbital_word
    orb_indx = device_orbital_index(calc) if recorded is not None else None
    basis: List[Tuple] = []
    from ..runfiles import SIESTA_ORB_INDX
    orbitals_why = ("" if orb_indx is not None
                    else "no junction composed yet" if recorded is None
                    else f"no device run holds its {SIESTA_ORB_INDX} yet")
    if orb_indx is not None:
        try:
            rows = basis_table(orb_indx)
        except TbtError as exc:
            rows, orbitals_why = [], str(exc)
        engine = sorted({e for e, *_q in rows})
        user_of = dict(zip(engine,
                           recorded.structure.map_to_user_index(engine)))
        basis = [(int(user_of[e]), n, l, m, z, orbital_word(l, m))
                 for e, n, l, m, z in rows]
    # THE ATOM AXIS, LISTED BY USER INDEX, with each atom's engine index --
    # the map the device's deck writes (`runs.stage_atom_map`,
    # `model/overview.md` § 2.2b, F2: a result for the user), so a script
    # asks sisl about a point's `.TBT.nc` straight from this file.
    from ..atom_permutation import AtomIndexMap
    from ..runs import stage_atom_map
    atoms = (stage_atom_map(recorded.structure.n_atoms,
                            recorded.structure.atom_order, task.calculation,
                            DEVICE_RUNG)
             if recorded is not None
             else AtomIndexMap.of_engine_record(None, 0)).as_listed_by_user()
    # EVERY RUN'S CONVERGENCE: the rungs that run once, by the stage table;
    # the device's runs, one a point; one SCF-row axis long enough for the
    # longest; TranSIESTA's charge columns as its header named them.
    by_stage = {s.get("stage"): s for s in record.get("stages") or ()}
    once = [s for s in TRANSPORT_STAGES if s not in per_point_rungs()]
    device = by_stage.get(DEVICE_RUNG) or {}
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
            "xyz": 3, "cell_vector": 3, "orbital": len(basis),
            "region": len(DEVICE_REGIONS),
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
    values["region"][:] = list(DEVICE_REGIONS)
    values["lead"][:] = leads
    values["eigenchannel"][:] = np.arange(n_eig)
    values["scf_cycle"][:] = np.arange(n_cycles)
    values["charge_part"][:] = parts
    values["orbital"][:] = np.arange(len(basis))
    for k, name in enumerate(("orbital_atom", "orbital_n", "orbital_l",
                              "orbital_m", "orbital_zeta", "orbital_type")):
        values[name][:] = [row[k] for row in basis]
    # THE JUNCTION WHOLE through the framework's one structure door, the
    # user order's, as the restored junction states it -- no masses: a
    # transport calculation uses none, and a frame set's are its maker's,
    # in ``structure_json`` (`model/datafiles.md` § 9.1).
    values.update(structure_values(
        junction.to_dict() if junction is not None else None, atoms,
        frames=n_frames))
    if junction is not None:
        from .record import device_regions
        for name, members in device_regions(junction).items():
            for a in members:
                values["atom_region"][a] = name

    # EACH RUN'S ENGINE, as its output states it: the stages', each point's
    # TBtrans, each point's device run (`engines/transport.md` § 2a.12).
    versions: Dict[str, Dict] = {"stage": {}, "point": {}, "device": {}}
    # EVERY POINT, DONE OR NOT: what its entry states, each value at its
    # record key (:data:`_KEYED`), and its engine's version.
    for p in (done_points + list(record.get("pending") or ())
              + list(record.get("not_run") or ())
              + list(record.get("failed") or ())):
        at = _cell(p.get("frame"), p["bias_v"], biases)
        copy_stated(p, _KEYED["point"], at, values)
        if p.get("engine_version"):
            versions["point"][at] = str(p["engine_version"])
    for p in done_points:
        f, b = _cell(p.get("frame"), p["bias_v"], biases)
        values["done"][f, b] = 1
        chans = p.get("channels") or {}
        for s, name in enumerate(spins):
            values["transmission_by_spin"][f, b, s] = (
                chans[name] if polarized else p["transmission"])
        # EACH CHANNEL'S PRINTED CURRENT, as the record states it by channel
        # (`record.point_currents`): none for a point whose energies two
        # runs computed.
        printed = p.get("printed_a") or {}
        for s, name in enumerate(spins):
            if printed.get(name) is not None:
                values["current_by_spin_a"][f, b, s] = printed[name]
        # WHICH ATOMS ITS DOS COVERS, where its reading is kept.
        covered = (p.get("dos") or {}).get("device_atoms")
        if covered is not None:
            values["dos_atom"][f, b, :] = 0
            values["dos_atom"][f, b, list(covered)] = 1
        readings = p.get("digest") or ()
        if readings:
            # KEPT, or why not, by the record's one answer for the point.
            values["dos_per_orbital_reading"][f, b] = (
                "kept" if not p.get("per_orbital_unread")
                else "; ".join(tried.get(d.get("file"))
                               or f"{d.get('file')}: {d.get('said')}"
                               for d in readings))
        dos = p.get("dos") or {}
        for r, region in enumerate(DEVICE_REGIONS):
            if region in (dos.get("by_label") or {}):
                values["dos_region_per_ev"][f, b, r] = dos["by_label"][region]
        for key, name in (("lead_spectral", "lead_spectral_dos_per_ev"),
                          ("lead_bulk", "lead_bulk_dos_per_ev")):
            for k, lead in enumerate(leads):
                if lead in (dos.get(key) or {}):
                    values[name][f, b, k] = dos[key][lead]
        for s, channel in enumerate(by_spin(p)):
            copy_stated(channel, _KEYED["spin"], (f, b, s), values)
            for k, curve in enumerate(channel.get("eigenchannels") or ()):
                values["eigenchannel_transmission"][f, b, s, k] = curve
    rows = iv_rows(iv)
    for (f, v), i in zip(rows, iv.get("current_a") or ()):
        if i is not None:
            values["iv_current_a"][_cell(f, v, iv_biases)] = i
    # EACH SPIN'S I-V CURRENT, a polarized calculation's (the record's).
    for channel in TBT_SPIN_CHANNELS[SPIN_POLARIZED]:
        if channel not in spins:
            continue
        s, key = spins.index(channel), f"current_{channel}_a"
        for (f, v), i in zip(rows, iv.get(key) or ()):
            if i is not None:
                values["iv_current_by_spin_a"][_cell(f, v, iv_biases)
                                               + (s,)] = i
    _device_facts(device, device_runs, biases, parts, values,
                  versions["device"], n_frames)
    for r, name in enumerate(once):
        if (by_stage.get(name) or {}).get("engine_version"):
            versions["stage"][name] = str(by_stage[name]["engine_version"])
        _run_facts(by_stage.get(name) or {}, "stage", (r,), values)
    average = record.get("average") or {}

    # THE ROOT'S STATEMENTS -- text, a count, a flag, and the records this
    # file holds as records (:data:`RECORD_ATTRS`), as blocks until written.
    attrs: Dict[str, Any] = {
        "label": str(task.label),
        "treatment": str(record.get("treatment") or ""),
        # WHETHER THE LOW-BIAS APPROXIMATION WAS MADE, 1 or 0
        # (`engines/transport.md` § 2a.12, *The data file*).
        "low_bias_approximation": int(bool(
            record.get("low_bias_approximation"))),
        "treatment_note": str(record.get("treatment_note") or ""),
        "current_means": record.get("current_means") or {},
        "energies_relative_to_fermi_level": int(bool(
            record.get("energies_relative_to_ef"))),
        "caveat": str(record.get("caveat") or ""),
        "n_frames": int(n_frames),
        # WHERE EACH RESULT CAME FROM: each rung's run and what it was
        # gathered from, a run of points' points (`record._chain`).
        "provenance_chain": (record.get("provenance") or {}).get(
            "chain") or [],
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
    # THE FRAMES' PART -- each frame's weight and their average, and what
    # is said of them: the post-process's (`frame_average`).
    fill_data_file(values, attrs, junction, average, biases, _index)
    # HOW THE I-V WAS COMPUTED, as the record states it (§ 2a.10).
    attrs["iv_computed"] = str(iv.get("computed") or "")
    if iv.get("kt_ev") is not None:
        values["iv_kt_ev"][...] = float(iv["kt_ev"])
        values["iv_window_lowest_ev"][...], \
            values["iv_window_highest_ev"][...] = iv["window_ev"]
    # EACH NOTE BY ITS FRAME AND VOLTAGE, as the record places them -- the
    # frame by its index on this file's axis.
    attrs["iv_notes"] = [{**n, "frame": _frame_index(n["frame"])}
                         for n in iv.get("notes") or ()]
    # THE FERMI FRAMES, EACH RUN'S TOLERANCES (§ 2a.12).
    attrs["fermi_frames"] = record.get("fermi_frames") or {}
    attrs["scf_criteria"] = {
        name: by_stage[name]["scf_criteria"] for name in TRANSPORT_STAGES
        if (by_stage.get(name) or {}).get("scf_criteria")}

    # THE ENGINE THAT COMPUTED IT, as the runs' outputs state it (the user,
    # 2026-11-16: "transport can also save siesta version where appropriate
    # in the meta data").
    found = sorted({v for by in versions.values() for v in by.values()})

    # THE FILE, through the variable record's one writing loop (`ncdata`),
    # group by group, stating itself (`model/datafiles.md` § 2b).
    with NC_LOCK:
        ds = in_memory("transport.nc")
        state(ds, title=(f"the transport calculation {attrs['label']}: "
                         f"every result in groups -- the junction, each "
                         f"stage that runs once, each point and its device "
                         f"run, and a summary across the points"),
              schema=DATA_SCHEMA, molbuilder_version=__version__,
              engine=ENGINE,
              engine_version=" and ".join(found),
              computed_in=(f"the calculation's runs -- its "
                           f"{_kind('transport').engine_label(ENGINE)} and "
                           f"{_engine(ENGINE).program_words('tbtrans')} -- "
                           f"each stage's run named in provenance_chain and "
                           f"each point's in its folder; each variable's "
                           f"source names its file"))
        _write_groups(ds, dims, VARIABLES, values,
                      label=attrs["label"], once=once, biases=biases,
                      n_frames=n_frames, versions=versions,
                      group_words={STRUCTURE_GROUP: {
                          "orbitals_why": orbitals_why,
                          **structure_attrs(junction.to_dict()
                                            if junction is not None
                                            else None)},
                            SUMMARY_GROUP: json_attrs(
                                {k: attrs[k] for k in ("iv_computed",
                                                       "iv_notes")},
                                RECORD_ATTRS)})
        ds.setncatts(json_attrs(attrs, RECORD_ATTRS))
        data = as_bytes(ds)
    return data_file_path(base, attrs["label"]).name, data


def _write_groups(ds, dims: Dict[str, Any], variables: Sequence[Var],
                  values: Dict[str, Any], *, label: str, once: Sequence[str],
                  biases: Sequence[float], n_frames: int,
                  versions: Dict[str, Dict],
                  group_words: Dict[str, Dict[str, str]]) -> None:
    """THE CALCULATION AS GROUPS (`engines/transport.md` § 2a.12): every
    variable in its home (:func:`_home`) -- a stage's and a point's sliced
    from the one array at its index, the prefix its group names dropped --
    and the summary's (:data:`SUMMARY`) whole, so a value two groups hold is
    one array's.  Each group holds the coordinate of every axis it runs
    over, so it is read alone, and ``group_words`` the words a group's own
    readers need beside its values: the I-V's computation in the summary,
    why the junction holds no orbital table."""
    from dataclasses import replace
    from ..ncdata import write_group
    from ..paths import frame_token
    from .stages import Point
    axes = {v.name: v for v in variables if _is_axis(v)}
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
            # A CELL OF THE ONE ARRAY, kept an array of its own type and
            # its remaining dimensions: cut by slices of one, never indexed
            # down to a bare element -- a text the writer cannot size, a
            # record (a list) numpy would spread over a new axis.
            whole = values[v.name]
            one = [isinstance(i, (int, np.integer)) for i in idx]
            cut = whole[tuple(slice(i, i + 1) if o else i
                              for i, o in zip(idx, one))]
            out_values[name] = (cut.reshape([n for n, o in zip(cut.shape, one)
                                             if not o])
                                if idx else whole)
        for v, value in extra:
            out_vars.append(v)
            out_values[v.name] = np.asarray(value)
        used = list(dict.fromkeys(d for v in out_vars for d in v.dims))
        for d in used:
            out_vars.append(axes[d])
            out_values[d] = values[d]
        # THE LABELS OF AN AXIS'S POINTS GO WITH IT -- a frame's folder and
        # rows, an atom's element and region -- into every group that runs
        # over it, so each is read alone (CF `coordinates`).
        for v in labels_of:
            if v.dims[0] in used and v.name not in out_values:
                out_vars.append(v)
                out_values[v.name] = values[v.name]
        write_group(group, {d: dims[d] for d in used}, out_vars, out_values,
                    title=title)

    def members(home: str) -> List[Var]:
        return [v for v in variables if _HOMES.get(v.name) == home]

    structure = ds.createGroup(STRUCTURE_GROUP)
    put(structure, members("structure"), {}, "",
        f"the junction of {label} as the calculation composed it, listed "
        f"by user index: the structure whole, its cell, each atom with its "
        f"region, its basis orbitals, every frame")
    structure.setncatts(group_words[STRUCTURE_GROUP])
    stages = ds.createGroup(STAGES_GROUP)
    write_group(stages, {}, (), {}, title=(
        "each stage that runs once, a group each, by its stage name"))
    for r, name in enumerate(once):
        g = stages.createGroup(name)
        put(g, members("stage"), {"rung": r}, _PREFIX["stage"],
            f"the stage {name}: its run's state, energy, Fermi level and "
            f"SCF history")
        g.setncattr("engine_version", versions["stage"].get(name, ""))
    points = ds.createGroup(POINTS_GROUP)
    write_group(points, {}, (), {}, title=(
        "every point the description asks for, done or not, a group each, "
        "named as its folders are"))
    # A FRAME SET'S POINTS ARE GROUPED BY FRAME, each frame's group named as
    # its folder is and saying what it holds.
    if n_frames > 1:
        for f in range(n_frames):
            write_group(points.createGroup(frame_token(f)), {}, (), {},
                        title=f"the points of frame {frame_token(f)}")
    for f in range(n_frames):
        for b, volts in enumerate(biases):
            pt = Point(frame=f if n_frames > 1 else None, volts=volts,
                       of=n_frames if n_frames > 1 else None)
            said = pt.words()
            # NAMED AS ITS FOLDERS ARE: the point's own path (`Point.rel`).
            g = points.createGroup(pt.rel)
            put(g, members("point"), {"frame": f, "bias_v": b},
                _PREFIX["point"],
                f"the point {said}: its transmission, its DOS, its current "
                f"and its state",
                extra=tuple(zip(_POINT_AT, (f, volts))))
            g.setncattr("engine_version", versions["point"].get((f, b), ""))
            d = g.createGroup(DEVICE_GROUP)
            put(d, members("device"), {"frame": f, "bias_v": b},
                _PREFIX["device"],
                f"the device run of the point {said} -- the one its "
                f"transmission gathered: its NEGF figures, charges and SCF "
                f"history")
            d.setncattr("engine_version", versions["device"].get((f, b), ""))
    summary = ds.createGroup(SUMMARY_GROUP)
    put(summary, [v for v in variables if v.name in SUMMARY], {}, "",
        "what is compared across the points, gathered from the same arrays "
        "as each point's own: each point's numbers and T(E) over frame x "
        "voltage, the I-V, and the frames' average")
    summary.setncatts(group_words[SUMMARY_GROUP])


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
                  values: Dict[str, Any], versions: Dict[Tuple, str],
                  n_frames: int) -> None:
    """Each point's device facts from the record's device rung -- its run
    state, convergence, NEGF figures, contour and SCF history, and its
    engine's version into ``versions`` by point: a rung that runs per
    point, point by point; one that runs once, its one run's, at every
    point."""
    for p in runs:
        if device.get("by_point"):
            cell = _cell(p.get("frame"), p["bias_v"], biases)
            points = [cell]
        else:
            cell = (Ellipsis,)
            points = [(f, b) for f in range(n_frames)
                      for b in range(len(biases))]
        if p.get("engine_version"):
            versions.update({at: str(p["engine_version"]) for at in points})
        negf = p.get("negf") or {}
        for k, part in enumerate(parts):
            if part in (negf.get("charges") or {}):
                values["device_charge_e"][cell + (k,)] = negf["charges"][part]
        _run_facts(p, "device", cell, values)
        # EVERY NEGF ROW'S CHARGES, column by column.
        for c, row in enumerate(p.get("scf") or ()):
            for k, part in enumerate(parts):
                if part in (row.get("charges") or {}):
                    values["device_scf_charge_e"][cell + (c, k)] = (
                        row["charges"][part])


def _run_facts(run: Dict, home: str, cell: Tuple,
               values: Dict[str, Any]) -> None:
    """One run's facts into its group's variables at ``cell`` -- a stage's
    index, or a device point's (frame, voltage): each at its record key
    (:data:`_KEYED`) -- its state and sentence, its convergence, its
    figures -- and its SCF history."""
    copy_stated(run, _KEYED[home], cell, values)
    # A RUN THAT DID NOT RUN states no history: its count keeps the fill,
    # -1, never 0, which is a count (§ 2a.12).
    if "scf" not in run:
        return
    scf_history_values(values, _scf_columns(), run["scf"] or (), cell,
                       prefix=_PREFIX[home])


def read_data_file(path) -> Dict[str, Any]:
    """``<label>.transport.nc`` as data: ``{"attrs": {...}, "groups":
    {path: {"attrs": {...}, "variables": {name: {"dims", "data",
    "attrs"}}}}}`` -- every group by its path (``structure``,
    ``stages/seed``, ``points/v0.2``, ``points/v0.2/device``, ``summary``),
    each variable's values as numpy arrays (a number's fill NaN, a flag's
    -1; a record the record), its units, long name, definition and
    source; the attributes written as JSON text (:data:`RECORD_ATTRS`,
    ``/structure``'s structure) as the block each holds, where the file holds it -- the root among the
    groups under ``""``, its attributes and no variable
    (`ncdata.read_file`).  The file's door (`runfiles.WRITTEN`); refuses
    another schema by name."""
    from ..ncdata import read_file
    return read_file(path, schemas=(DATA_SCHEMA,), error=DataFileError,
                     way="jobset summarize task writes it again",
                     records=RECORD_ATTRS)


def point_groups(groups: Dict[str, Any]) -> Dict[str, Any]:
    """THE POINTS of a data file read back (:func:`read_data_file`'s
    ``groups``), each by its folder below the run (``v0.2``, ``f001/v0.2``):
    a group below ``points`` that says whether it is done -- not a frame's
    group of points, nor a point's device run."""
    below = f"{POINTS_GROUP}/"
    return {k[len(below):]: g for k, g in groups.items()
            if k.startswith(below) and "done" in g["variables"]}
