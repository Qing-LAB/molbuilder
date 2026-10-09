"""SIESTA's answers to the emission doors — `execution/script-preparation.md` § 4.2.

The same three doors PySCF answers, answered differently, which is the point:
SIESTA's deck is a keyword list a reader may scan in any order, so its sections
are conventions rather than execution order, and its notes are long enough that
each is headed by the keyword it writes.

What is here is the **layout as data** and the **syntax for one parameter**.
Everything about which values these are, and why each one is what it is, stays
in the catalogue where both engines and the form read it.
"""
from __future__ import annotations

from typing import Optional, Tuple

from ..script_emit import Parameter, Section

#: The basis and the real-space grid.
BASIS_SECTION = Section(
    "Basis & grid",
    ("mesh_cutoff", "basis_size", "pao_energy_shift"),
)

#: Exchange-correlation.  Two items, two keywords -- ``XC.functional`` names
#: the family and ``XC.authors`` the parameterisation, and SIESTA needs both.
XC_SECTION = Section(
    "Exchange-correlation",
    ("xc_functional", "xc_authors"),
)

#: The SCF settings above the free-energy pair.
SCF_SECTION = Section(
    "SCF",
    ("solution_method", "mixing_weight", "pulay_history", "dm_tolerance"),
)

#: What the run writes out.  All booleans, and **all emitted in both states**
#: -- a keyword left out hands the decision to SIESTA's own default.  Under the
#: door that is not a discipline anybody has to remember: `line` returns a
#: value for False as readily as for True.
OUTPUT_SECTION = Section(
    "Output",
    ("write_forces", "write_coor_step", "write_coor_xmol", "write_md_history",
     "write_md_xmol", "write_hs"),
)

#: The force-constant run's one parameter (the vibration kind on this
#: engine).  The run type and the atom range are structural text derived
#: from the structure -- `siesta/vibration_deck.py` -- not items.
FC_SECTION = Section(
    "Force constants",
    ("fc_displacement",),
)

#: The free-energy criterion and the switch that arms it.  **Two items, one
#: decision**: the tolerance alone does nothing -- SIESTA loads it either way
#: and installs it as a criterion only when the switch is on.  Both are written, in both states, so a person reading the
#: deck without the form sees the gate as well as the number.
FREE_ENERGY_SECTION = Section(
    "SCF free-energy convergence (a PAIR: the value + its switch)",
    ("dm_energy_tolerance", "scf_energy_converge"),
)

#: The SCF settings that follow the free-energy pair.
SCF_TAIL_SECTION = Section(
    # NAMED: a nameless section is a layout that stops saying what that part
    # of the deck IS, which is the one thing a layout is for
    # (`script-preparation.md` § 4.1).
    "SCF iteration limit and smearing",
    ("max_scf_iter", "scf_must_converge", "electronic_temperature"),
)


#: The spin treatment in SIESTA's own words (``Src/spin_subs.F90``).  The
#: item holds the engine-neutral word (`science/chemistry-correctness.md`
#: § 2a.1); spelling it is this engine's business, like `_FMT` below.
#: ``restricted-open`` has no spelling: SIESTA has no such formalism, and
#: the settings gate refuses it by name before a deck is written (ES4).
SPIN_SPELLING = {
    "restricted":    "non-polarized",
    "unrestricted":  "polarized",
    "non-collinear": "non-colinear",
    "spin-orbit":    "spin-orbit",
}

#: (Each spelling is one of the words SIESTA accepts for its treatment --
#: every one of which a READER of a SIESTA deck must know:
#: `parse/fdf.SPIN_WORDS`, which lives with the reader because it travels
#: beside a job where this package is not installed.)


def spin_section(*, fixed: bool) -> Section:
    """The spin -- always ``Spin``, and ``Spin.Fix`` + ``Spin.Total`` for a
    pinned count.  Built **per render**, because the second depends on the
    answer.

    Written in every state: `template.md` § 6.6 rules out leaving SIESTA's
    default to answer -- nothing reaches the engine by omission, and a spin
    decided from the structure is a decision the deck should state.
    """
    return Section("Spin", ("spin_treatment",)
                   + (("unpaired_electrons",) if fixed else ()))




def mpi_section(*, block_size, algorithm) -> Section:
    """How the work is split across ranks — built **per render**.

    Which items appear depends on answers this deck has already worked out, so
    the table cannot be a constant:

    * ``BlockSize`` has a third state.  ``block_size`` unset (``None``) means
      *do not emit the keyword at all* and let SIESTA choose (`tuning.md`
      § 2.11), which is not the same as emitting a number; 0 is past the
      item's limit and refused on every door (`engines/template.md` § 5.3).
    * ``Diag.Algorithm`` and its GPU switch are only meaningful for an ELPA
      solver; ScaLAPACK has no such knobs, and writing them would be the deck
      claiming a setting the solver never reads.
    """
    items = []
    if block_size is not None:
        items.append("block_size")
    items.append("parallel_over_k")
    if str(algorithm or "").upper().startswith("ELPA"):
        items += ["diag_algorithm", "use_gpu"]
    return Section("Parallel execution (MPI)", tuple(items), note=(
        "# These settings matter only with `mpirun -np N siesta`",
        "# (single-rank runs ignore them).",
        "#",
        "# BlockSize: ScaLAPACK orbital-distribution block.  Affects",
        "# cache efficiency for the diagonaliser; does NOT fix the",
        "# propor IMAX=0 crash (a sweep: BlockSize = 1, 2, 4 all",
        "# crash at the same mpi_np; propor is a matel_table",
        "# proportionality check, not a BLACS distribution check).",
        "# If your run dies at startup with ``propor: ERROR: IMAX =",
        "# 0``, follow the run script's hint: restore the stage's",
        "# checkpoint and prep it again with another --np.  Larger",
        "# BlockSize gives marginally better diag throughput on big",
        "# systems (>1000 atoms / >=16 ranks); for smaller jobs the",
        "# default is fine.  Override only for hand-tuned perf work.",
        "#",
        "# Diag.ParallelOverK: parallelise the diagonaliser over",
        "# k-points (.true.) or over orbitals (.false.).  Auto-",
        "# selected here from the kgrid above: .false. for 1x1x1",
        "# (molecule / vacuum), .true. for multi-k periodic runs.",
        "# NOTE with ELPA (CPU or GPU): SIESTA sets ParallelOverK",
        "# .false. itself (Src/diag_option.F90), so an ELPA run",
        "# always splits over orbitals whatever this line says.",
        "",
    ))




def geometry_section(*, is_md: bool, is_nose: bool) -> Section:
    """What the run DOES to the geometry — built **per render**.

    A relaxation and a molecular-dynamics run are different calculations and
    carry different keywords, so there is no one table: a relaxation caps the
    force and the step displacement, while dynamics sets temperatures and a
    timestep. Choosing the table from the answer already resolved is the same
    shape :func:`mpi_section` uses, and it keeps the ``if`` in the engine --
    which is the only place that knows what these words mean.
    """
    items = ["relax_type", "relax_steps"]
    if is_md:
        items.append("md_initial_temperature")
        if is_nose:
            items.append("md_target_temperature")
        items.append("md_length_timestep")
    else:
        items += ["relax_force_tol", "relax_max_displ"]
    return Section("Geometry optimisation / dynamics", tuple(items))




#: THE DECK'S UNIT WORD IS THE CATALOGUE'S `unit` -- one source (plan F27;
#: `engines/template.md` § 6.4): a unit is part of the SPELLING, not of the
#: value, so the deck writes the item's own unit next to the number.  Two
#: spellings differ from the catalogue's: fdf reads ASCII, so Å is `Ang`;
#: and the bias, held in volts, is written `eV` -- the energy an electron
#: gains across V volts is V electron-volts, so the number is the same and
#: only the word differs (its catalogue note says so to a reader of the
#: deck).  An item added with a `unit` renders with it, instead of bare --
#: which TranSIESTA read as rydberg, 13.6 times the value.
_FDF_SPELLING = {"Å": "Ang", "eV/Å": "eV/Ang"}
_UNIT_EXCEPTION = {"bias_voltage_v": "eV"}


def unit_word(name: str) -> Optional[str]:
    """The unit fdf reads beside ``name``'s value, or None for a bare number."""
    if name in _UNIT_EXCEPTION:
        return _UNIT_EXCEPTION[name]
    from ..template import catalogue, one
    item = one(catalogue(), name, engine="siesta")
    unit = item.unit if item is not None else None
    return _FDF_SPELLING.get(unit, unit) if unit else None

#: Items whose keyword is padded so a related pair reads as a column.
#: ``XC.functional`` names the family and ``XC.authors`` the parameterisation;
#: they are one decision written on two lines, and the alignment says so.
_PAD = {
    "xc_functional": 14, "xc_authors": 14,
    # The SCF block reads as a column.  The widths are not uniform -- they grew
    # one keyword at a time -- and they are kept exactly, because a person who
    # diffs two generations of a deck should see values change, not alignment.
    "solution_method": 18, "mixing_weight": 19, "pulay_history": 21,
    "dm_tolerance": 18,
    "write_forces": 19, "write_coor_step": 19, "write_coor_xmol": 19,
    "write_md_history": 19, "write_md_xmol": 19, "write_hs": 19,
    "dm_energy_tolerance": 19, "scf_energy_converge": 19,
    "block_size": 19, "parallel_over_k": 19, "diag_algorithm": 19,
    "use_gpu": 19,
    "md_target_temperature": 22,
    "max_scf_iter": 18, "spin_treatment": 18,
    # The NEGF and transmission settings read as one column.
    "bias_voltage_v": 23, "electrodes_bulk": 23, "negf_eq_pole_ev": 23,
    "negf_neq_eta_ev": 23, "ts_elecs_eta_ev": 23, "tbt_elecs_eta_ev": 23,
    "tbt_contours_eta_ev": 23, "tbt_spin": 23, "tbt_dos_gf": 23,
    "tbt_dos_a": 23, "tbt_dos_elecs": 23, "tbt_t_eig": 23, "tbt_t_bulk": 23,
    "tbt_t_all": 23, "tbt_verbosity": 23,
}

#: Items SIESTA wants in scientific notation.  Formatting is spelling, so it
#: is the engine's business and lives beside the rest of the spelling.
_FMT = {"dm_tolerance": ".1e", "dm_energy_tolerance": ".1e",
        "bias_voltage_v": ".4f", "negf_eq_pole_ev": ".4f",
        "negf_neq_eta_ev": ".6f", "ts_elecs_eta_ev": ".6f",
        "tbt_elecs_eta_ev": ".6f",
        "tbt_contours_eta_ev": ".6f"}

#: Items whose 0 means LEAVE IT TO THE ENGINE, so 0 writes nothing.  Each of
#: these engine defaults is a FORMULA rather than a number -- the
#: non-equilibrium and the device Green function's broadenings are the
#: leads' smallest over ten -- and a number written in its place would
#: REPLACE the rule: an explicit 0 broadening overrides the formula
#: (`engines/transport.md` § 6.1b; each item's note says what 0 leaves).
#: Where the default IS a number, the item writes it: `TBT.Spin 0` is
#: tbtrans's own default, all channels (`m_tbt_hs.F90`).
#:
#: THE POLE ENERGY IS NOT IN THIS SET (§ 6.1c).  TranSIESTA's own choice for it -- about 42 poles at 300 K -- lost the
#: charge on a real device where a stated 10 eV held it, so the energy is
#: always written, and the settings gate refuses one under 20 poles.
_ZERO_LEAVES_IT_TO_THE_ENGINE = frozenset({
    "negf_neq_eta_ev", "tbt_contours_eta_ev"})


def k_mesh_lines(mesh, *, notes: bool = True) -> list:
    """A k-point mesh as a SIESTA-family deck writes it
    (`engines/siesta.md` § 6.1): each item that answered an axis gives its
    declaration's note -- why it is what it is -- and the block is the one
    writer's (`kmesh.write`).  The SIESTA deck, the transport rungs and the
    transmission's ``TBT.k`` all write their mesh through here, so the notes
    a deck carries are the items that decided it, never a fixed pair."""
    from .. import kmesh
    from ..script_emit import parameter
    out: list = []
    if notes:
        names = []
        for axis in mesh.axes:
            if axis.source in kmesh.ITEMS and axis.source not in names:
                names.append(axis.source)
        names.append("kgrid_displacement")
        for name in names:
            out += parameter(name, "siesta").note()
    out += kmesh.write(mesh)
    return out


def note_lead(param: Parameter) -> Tuple[str, ...]:
    """Head each note with the keyword it is about.

    SIESTA's catalogue notes run to a dozen lines, so a reader meets the
    explanation well before the keyword.  Naming it first is what makes the
    block scannable -- and the name comes from the declaration, so it cannot
    drift from the line below it.
    """
    return param.writes[:1]


def line(derived: dict):
    """**Door 2 — the engine's syntax, and there is one of it.**

    Returns ``(Parameter) -> str | None`` for every item this engine lays out.

    **It takes the deck's context whole, not a keyword per derived value**:
    one argument, one channel -- what this deck derived.
    Most are the catalogue's ``anchor`` and its value; three groups are not:

    * **the MPI block's four are DERIVED** -- ``Diag.ParallelOverK`` from
      whether the k-mesh is more than Gamma, the algorithm normalised.  They reach the door through
      ``parameter(..., value=)``, so a computed number still arrives with its
      declaration, its range and its note;
    * **two geometry keywords are chosen by the run mode** -- SIESTA spells the
      step count differently per algorithm and the displacement cap exists only
      for a relaxation -- and a Nosé run with no target temperature holds it at
      the initial one, which is why that value is resolved before the
      is-it-set guard rather than after;
    * **the spin comes from the electronic state** (`_spin_facts`), not
      from the config: a blank item is decided by the class, so the field
      holds ``None`` where the deck must write the answer.  A pinned count
      expands to two keywords -- the constraint has to be switched on as well
      as given a value, and SIESTA ignores the number without ``Spin.Fix`` --
      and the pair comes from the declaration's ``expands``, so the deck
      cannot write one without the other.
    """
    # A DERIVED FACT SAID BESIDE A VALUE: the deck writer works it out and
    # this door writes it after the value as an fdf comment -- libfdf ends a
    # line's tokens at `#` outside a string or list (`parse.F90`), so the
    # engine reads the value and nothing else.  The pole energy's count at
    # the run's temperature is the one today (`engines/transport.md` § 6.1c).
    beside = derived.get("beside") or {}
    computed = {"block_size":     derived.get("block_size"),
                "parallel_over_k": derived.get("over_k"),
                "diag_algorithm": derived.get("algorithm"),
                "use_gpu":     derived.get("gpu")}
    override = {"relax_steps":     derived.get("step_kw"),
                "relax_max_displ": derived.get("displ_kw")}
    target_t = derived.get("target_t")

    def _mpi(param):
        value = computed.get(param.name)
        if value is None:
            return None
        key = param.writes[0] if param.writes else None
        if key is None:
            return None
        shown = (".true." if value else ".false.") if isinstance(value, bool) \
            else f"{value}"
        pad = _PAD.get(param.name)
        return f"{key:<{pad}}{shown}" if pad else f"{key} {shown}"

    def _geometry(param):
        if not param.known:
            return None
        value = (target_t if param.name == "md_target_temperature"
                 else param.value)
        if value is None:
            return None
        key = override.get(param.name)
        if param.name in override and key is None:
            return None                      # no such keyword in this mode
        if key is None:
            key = param.writes[0] if param.writes else None
        if key is None:
            return None
        unit = unit_word(param.name)
        pad = _PAD.get(param.name)
        shown = f"{value} {unit}" if unit else f"{value}"
        return f"{key:<{pad}}{shown}" if pad else f"{key} {shown}"

    def _pair(param):
        keys = param.writes
        return (f"{keys[0]:<18}.true.\n"
                f"{keys[1]:<18}{param.value}")

    def _plain(param):
        key = param.writes[0] if param.writes else None
        if key is None:
            return None
        # fdf spells a boolean `.true.` / `.false.`, and BOTH are written: an
        # omitted keyword is not a neutral silence, it is SIESTA's default
        # answering instead of the person who filled in the form.
        if isinstance(param.value, bool):
            shown = ".true." if param.value else ".false."
            pad = _PAD.get(param.name)
            return f"{key:<{pad}}{shown}" if pad else f"{key} {shown}"
        if isinstance(param.value, (list, tuple)):
            # AN fdf LIST IS WRITTEN IN BRACKETS, and only then is it a list:
            # libfdf's tokenizer calls a token a list when it "starts with [
            # and ends with ]" (`parse.F90`), and a reader that asks for a list
            # (`fdf_islist`) finds nothing in three bare numbers.  `TBT.k 2 2 1`
            # was skipped that way -- tbtrans fell through to the SCF's grid in
            # silence (`engines/transport.md` § 6.1b).  A list is of NUMBERS:
            # the tokenizer knows integer and real lists only, so a list of
            # words has no fdf spelling here and is refused rather than
            # written as something no reader parses.
            if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                       for v in param.value):
                raise TypeError(
                    f"{param.name}: an fdf list holds numbers only, and "
                    f"this value is {param.value!r}")
            shown = "[" + " ".join(str(v) for v in param.value) + "]"
            pad = _PAD.get(param.name)
            return f"{key:<{pad}}{shown}" if pad else f"{key} {shown}"
        fmt = _FMT.get(param.name)
        shown = format(param.value, fmt) if fmt else f"{param.value}"
        unit = unit_word(param.name)
        value = f"{shown} {unit}" if unit else shown
        # One space unless the item is half of an aligned pair.  fdf does not
        # care, but a reader diffing two generations should see the values
        # change, not the whitespace.
        pad = _PAD.get(param.name)
        return f"{key:<{pad}}{value}" if pad else f"{key} {value}"

    def _spin(param):
        # THE ELECTRONIC STATE'S VALUES, never the raw fields: a blank item is
        # decided by the class (§ 2a), so the config holds `None` where the
        # deck must write the answer -- `_spin_facts` put the answer in the
        # context.  The pair's two keywords are the declaration's first two
        # -- SIESTA's; the third is PySCF's.
        if param.name == "spin_treatment":
            treatment = derived.get("spin_treatment")
            if treatment is None:
                return None
            pad = _PAD.get(param.name)
            return f"{param.writes[0]:<{pad}}{SPIN_SPELLING[treatment]}"
        n = derived.get("spin_pinned")
        if n is None:
            return None
        keys = param.writes
        return (f"{keys[0]:<18}.true.\n"
                f"{keys[1]:<18}{float(n):.1f}")

    def _line(param: Parameter) -> Optional[str]:
        if param.name in beside and (
                param.name in ("spin_treatment", "unpaired_electrons")
                or param.name in computed or param.name in override
                or param.name == "md_target_temperature"
                or len(param.writes) == 2):
            # One line, one value, one statement: a keyword this door writes
            # as a pair or computes has no single line to say it beside, and
            # dropping the text would lose it in silence.
            raise TypeError(
                f"{param.name}: a statement beside a value is written on a "
                f"plain one-keyword line, and this item is not one")
        if param.name in ("spin_treatment", "unpaired_electrons"):
            return _spin(param)
        if param.name in computed:
            return _mpi(param)
        if param.name in override or param.name == "md_target_temperature":
            return _geometry(param)
        if not param.known or param.value is None:
            return None
        if param.name in _ZERO_LEAVES_IT_TO_THE_ENGINE and not param.value:
            return None
        if len(param.writes) == 2:
            return _pair(param)
        text = _plain(param)
        if text is not None and param.name in beside:
            text = f"{text}   # {beside[param.name]}"
        return text

    return _line


def check_rules(text: str, struct=None, cfg=None):
    """SIESTA's answer to *what must a finished deck of mine satisfy?*

    Four things a deck cannot be wrong about and still mean what it says.  All
    are read off the FILE, after it is written, which is the only way to catch
    a writer bug -- every other validator in this tree takes ``(struct, cfg)``
    and runs before emission.

    **No keyword twice with different values.**  libfdf takes the FIRST match and ignores the rest
    (``fdf_locate`` walks from the top and stops), so a duplicate does not
    conflict loudly -- it silently wins, and the later line a person edited is
    the one being ignored.  That is the worst kind of wrong: the deck reads as
    though it says what you meant.

    **The identity is the one that was stamped.**  Every warm file is keyed by
    ``SystemLabel``; a deck carrying a different one finds nothing and starts
    cold without saying so.

    **The atom count matches the coordinates**, and **every species index used
    exists** -- SIESTA reads ``NumberOfAtoms`` and the coordinate block
    separately, so a disagreement between them is a truncated or doubled block,
    and an index with no species is a startup failure after the queue wait.
    """
    from ..issues import Issue
    from ..parse.fdf import _norm as _fdf_norm

    out = []
    code = [ln.split("#", 1)[0].rstrip() for ln in text.splitlines()]

    # -- one keyword, one line ------------------------------------------
    seen: dict = {}
    in_block = False
    for ln in code:
        low = ln.strip().lower()
        if low.startswith("%block"):
            in_block = True
            continue
        if low.startswith("%endblock"):
            in_block = False
            continue
        if in_block or not ln.strip():
            continue
        key = ln.split()[0]
        # fdf's keyword rule, from the one module that owns it.  A gate
        # that enforces "one keyword, one line" must agree with the reader
        # about what ONE KEYWORD means, or it polices a different rule.
        norm = _fdf_norm(key)
        value = " ".join(ln.split()[1:])
        if norm in seen and seen[norm][1] != value:
            out.append(Issue(
                "error",
                f"{key} is written twice with different values "
                f"({seen[norm][1]!r} then {value!r}); libfdf takes the first, "
                f"so the second is silently ignored",
                where="deck.duplicate_keyword"))
        seen.setdefault(norm, (key, value))

    # -- the identity that was stamped ----------------------------------
    label = getattr(cfg, "system_label", None)
    if label and seen.get("systemlabel", (None, None))[1] != label:
        out.append(Issue(
            "error",
            f"the deck's SystemLabel is not the identity it was written for "
            f"({label!r}); every warm file is keyed by that name",
            where="deck.identity"))

    # -- the atom count against the coordinate block --------------------
    rows, inside = 0, False
    for ln in code:
        low = ln.strip().lower()
        if low.startswith("%block atomiccoordinatesandatomicspecies"):
            inside = True
            continue
        if low.startswith("%endblock") and inside:
            break
        if inside and ln.strip():
            rows += 1
    declared = seen.get("numberofatoms", (None, None))[1]
    if declared and declared.isdigit() and rows and int(declared) != rows:
        out.append(Issue(
            "error",
            f"NumberOfAtoms says {declared} but the coordinate block has "
            f"{rows} rows",
            where="deck.atom_count"))

    # -- every species index used exists --------------------------------
    # The coordinate rows name a species by its INDEX into
    # ChemicalSpeciesLabel, so the two blocks are one fact written twice and
    # nothing else in the deck relates them.  An index with no species is a
    # startup failure after the queue wait, which is the most expensive place
    # to find a writer bug.
    declared_species = set()
    inside = False
    for ln in code:
        low = ln.strip().lower()
        if low.startswith("%block chemicalspecieslabel"):
            inside = True
            continue
        if low.startswith("%endblock") and inside:
            break
        parts = ln.split()
        if inside and parts and parts[0].isdigit():
            declared_species.add(parts[0])
    used = set()
    inside = False
    for ln in code:
        low = ln.strip().lower()
        if low.startswith("%block atomiccoordinatesandatomicspecies"):
            inside = True
            continue
        if low.startswith("%endblock") and inside:
            break
        parts = ln.split()
        if inside and len(parts) >= 4:
            used.add(parts[3])
    for missing in sorted(used - declared_species):
        out.append(Issue(
            "error",
            f"the coordinate block uses species {missing}, which "
            f"ChemicalSpeciesLabel does not declare",
            where="deck.species_index"))
    return out
