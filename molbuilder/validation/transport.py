"""THE TRANSPORT KIND'S SCIENCE -- the checks every deck of a transport
calculation runs before it is written, whatever rung and config class
(`engines/transport.md` § 6.1c).

MODULE  validation.transport (the kind's own module, `execution/
        architecture.md` § 3.3: it may choose by rung and engine)
ROLE    the part the kind table names (``validator``), run by the settings
        gate on every deck of the kind (`validation.validate`)
USED-BY `validation.validate`, through the kind table
"""
from __future__ import annotations

from typing import List

from ..issues import Issue
from ..structure import Structure


def validate_transport_kind(struct: Structure, cfg, cell, *,
                            rung=None, **_) -> List[Issue]:
    """The transport KIND's science — the part its record in the kind table
    names (``validator``), so it fires whatever config class the deck
    renders from.  A rule that only
    runs for one of two config classes is not a gate; this one runs for the
    kind.

    Its science: the pole energy against the temperature, the vacuum where
    the crystal continues (I12), the transmission window against the bias.  **The k-point sampling is
    not here** -- the transport axis's one point, a lead's own count, the
    transmission's grid are the k-point mesh's (`kmesh.py`,
    `engines/siesta.md` § 6.1), refused on every door through
    ``template.why_not`` rather than on this one alone.
    """
    out: List[Issue] = []
    # THE BIAS: no threshold voltage is invented here (`engines/transport.md`
    # § 2a.10) -- the treatment is the description's switch, and the point's
    # voltage is read below by the window rule alone.
    bias = getattr(cfg, "bias_voltage_v", None)
    # THE POLE ENERGY AND THE TEMPERATURE ARE ONE QUESTION, so neither can be
    # checked alone.  The rule is TranSIESTA's own -- SIESTA 5.4.2
    # `Src/m_ts_chem_pot.F90`, read 2026-09-16 rather than inferred -- and it
    # has one home, `transiesta.pole_count`, which the device deck also asks
    # for the count it states beside the energy.  TranSIESTA stops a run under
    # twenty poles, after the queue wait; this says so before it.
    #
    # 0 IS REFUSED TOO (`engines/transport.md` § 6.1c): TranSIESTA's own
    # choice -- about 42 poles at 300 K -- lost the charge on a real Au-BDT-Au
    # device, where 10 eV (123 poles) held it.  So the energy is
    # always written; and an energy at or below zero is not taken as one --
    # TranSIESTA keeps its 8-pole count and stops (`pole_count` says why).
    #
    # AND THE TEMPERATURE HAS ITS OWN FLOOR: TranSIESTA stops below 10 K
    # before any pole is counted (`transiesta.MIN_TS_TEMPERATURE_K`).
    #
    # JUDGED ON THE RUNG THAT READS IT -- the device's (the catalogue's
    # ``stages``, asked through `template.reads`, `template.md` § 6.4): the
    # template's value stays on a rung that never writes it, and that rung
    # is never refused for it.  With no rung named, judged as it stands.
    from ..template import engine_name, items_of, reads
    pole = getattr(cfg, "negf_eq_pole_ev", None)
    temp = getattr(cfg, "electronic_temperature", None)
    pole_item = items_of(cfg).get("negf_eq_pole_ev")
    if pole is not None and (rung is None or pole_item is None or reads(
            pole_item, engine_name(type(cfg)), "transport", rung)):
        from ..transport.transiesta import (MIN_EQ_POLES,
                                            MIN_TS_TEMPERATURE_K, pole_count,
                                            pole_energy_for)
        if temp is None or float(temp) < MIN_TS_TEMPERATURE_K:
            out.append(Issue(
                "error",
                f"the electronic temperature is "
                f"{0.0 if temp is None else float(temp):g} K, and TranSIESTA "
                f"stops a device run below {MIN_TS_TEMPERATURE_K:g} K before "
                f"it starts -- \"TranSiesta electronic temperature *must* be "
                f"larger than 10 kT\".  Set it to at least "
                f"{MIN_TS_TEMPERATURE_K:g} K; 300 K is the default "
                f"(engines/transport.md 6.1c).",
                where="config.electronic_temperature"))
        else:
            n = pole_count(pole, temp)
            if n < MIN_EQ_POLES:
                need = pole_energy_for(MIN_EQ_POLES, temp)
                said = (f"{float(pole):g} eV is not an energy TranSIESTA "
                        f"takes -- it keeps its own count of {n} poles"
                        if float(pole) <= 0 else
                        f"{float(pole):g} eV gives {n} poles at "
                        f"{float(temp):g} K")
                out.append(Issue(
                    "error",
                    f"the equilibrium contour's pole energy: {said}, and "
                    f"TranSIESTA needs at least {MIN_EQ_POLES}.  It stops "
                    f"with \"the continued fraction method requires at least "
                    f"20 poles\", after the queue wait.  The count is "
                    f"DERIVED, N = int(E / (pi kT)), so it moves with the "
                    f"temperature: at {float(temp):g} K the least energy is "
                    f"{need:.2f} eV.  The default is 10 eV "
                    f"(engines/transport.md 6.1c).",
                    where="config.negf_eq_pole_ev"))
    # THE TRANSMISSION WINDOW REACHES THE BIAS WINDOW (P2, `engines/
    # transport.md` § 2a.10, § 6.1c): on the transmission rung alone, whose
    # deck writes the window and the voltage.  One rule for the record, this
    # gate and the description's preflight (`record.window_short_of`):
    # TBtrans integrates the current over its window and says nothing when
    # the window cuts it short.
    if rung == "transmission" and bias is not None and temp is not None:
        emin = getattr(cfg, "transmission_emin_ev", None)
        emax = getattr(cfg, "transmission_emax_ev", None)
        n = getattr(cfg, "transmission_n_points", None)
        if emin is not None and emax is not None and n:
            from ..constants import BOLTZMANN_EV_K
            from ..transport.record import sampled_window, window_short_of
            short = window_short_of(*sampled_window(emin, emax, n), bias,
                                    BOLTZMANN_EV_K * float(temp))
            if short:
                out.append(Issue("error",
                                 short + " (engines/transport.md 2a.10)",
                                 where="config.transmission_emax_ev"))
    # THE K-POINT SAMPLING IS NOT HERE: it is the k-point mesh's
    # (`kmesh.py`, `engines/siesta.md` § 6.1): the third
    # component of `kgrid` and `tbt_k_grid` is one a transport calculation
    # fixes, and a lead's count is refused at 1 by its own limit and warned
    # below 50 by its range -- every door, through `template.why_not`.

    # ---- I12: no vacuum where the crystal continues (§ 5, § 6.1c) ----
    #
    # MEASURED FROM THE LEAD (TD3, user 2026-09-29): each rule reads the
    # lead's own spacing, and both are refusals -- a gap along transport
    # severs the lead, and vacuum on a periodic axis contradicts the
    # declaration.  An ISOLATED transverse axis is left alone: a wire or chain
    # lead is vacuum-surrounded across transport, and that vacuum is the one
    # the structure states.  The threshold is the seam rule's, one rule for
    # "is this boundary vacuum" (`cell.SEAM_VACUUM_FACTOR`).
    #
    # This is the reverse of the advice an isolated molecule gets, which is
    # why it is keyed on the calculation kind and not on the cell alone --
    # `cell.vacuum_thin` is right for a molecule and is gated on `axis_kind`.
    _cell = cell if cell is not None else getattr(struct, "cell", None)
    if _cell is not None and getattr(struct, "n_atoms", 0):
        from ..cell import SEAM_VACUUM_FACTOR, transverse_reach
        from ..transport.sort import ELECTRODE_LABELS
        from .sidecar import check_junction_boundary
        # THE ROOM ALONG TRANSPORT, one rule both ways (vacuum, collision)
        # -- a refusal on every transport rung; the same rule warns at a
        # labelled junction's relaxation (`sidecar.check_junction_boundary`).
        out += check_junction_boundary(struct, _cell, severity="error",
                                       whole_structure_is_the_lead=True)
        # A LEAD RUNG'S SEAM (`junction-cell.md` § 3.1, `engines/transport.md`
        # § 7.1): whether the lead tiles into the bulk crystal -- a room of
        # one spacing can still be a twin or an eclipse -- measured and
        # reported, never refused, by the rule the Transport card asks
        # (`wizard.seam_finding`); TranSIESTA checks no registry.
        from ..kind_table import lead_of
        if lead_of(rung) is not None:
            from ..transport.wizard import seam_finding
            out.append(seam_finding(struct.positions, _cell))
        # THE LEAD: the electrode-labelled atoms of a junction, one list per
        # lead; an electrode rung's structure IS the lead and carries no
        # labels, so all of it.
        leads = [list(idx) for label, idx in
                 (getattr(struct, "regions", None) or {}).items()
                 if label in ELECTRODE_LABELS and idx]
        if not leads:
            leads = [list(range(struct.n_atoms))]
        # LEAD BY LEAD, as the rule is stated: pooled, two leads answer for
        # each other -- one that tiles hides one that does not on the
        # junction's rungs, while the electrode rung that sees it alone
        # refuses the same calculation; and two one-atom leads measure the
        # distance between the leads as a "bond".
        kinds = getattr(struct, "axis_kind", None) or ("isolated",) * 3
        named = ([label for label, idx in
                  (getattr(struct, "regions", None) or {}).items()
                  if label in ELECTRODE_LABELS and idx]
                 or ["the lead"])
        for ax, name in ((0, "a"), (1, "b")):
            if kinds[ax] != "periodic":
                continue
            for label, lead in zip(named, leads):
                reach, bond = transverse_reach(struct.positions, _cell, ax,
                                               lead)
                if bond is None:
                    out.append(Issue(
                        "warn",
                        f"axis {name} is declared periodic, but {label} is "
                        f"one atom, so whether it reaches across that "
                        f"boundary cannot be measured (engines/transport.md "
                        f"6.1c).",
                        where="cell.transverse_vacuum"))
                elif reach > SEAM_VACUUM_FACTOR * bond:
                    out.append(Issue(
                        "error",
                        f"axis {name} is declared periodic, but across its "
                        f"boundary the nearest atom of {label} is "
                        f"{reach:.2f} Å away -- {reach / bond:.1f} of its own "
                        f"{bond:.2f} Å bond, so the crystal does not continue "
                        f"there: that is vacuum.  A wire or chain lead is "
                        f"isolated across the transport axis: declare the "
                        f"axis isolated on the structure (the Cell page), "
                        f"relax it, and cite that relaxation -- its deck "
                        f"records the axis kinds (engines/transport.md "
                        f"6.1c).",
                        where="cell.transverse_vacuum"))
    return out


__all__ = ["validate_transport_kind"]
