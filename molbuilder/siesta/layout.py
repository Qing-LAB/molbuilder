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

#: The SCF settings above the free-energy pair.  The pair itself carries a
#: long explanation of how SIESTA ANDs its enabled convergence tests together,
#: which is method guidance rather than either value's reason, so it stays in
#: the body until it has a home in the declaration.
SCF_SECTION = Section(
    "SCF",
    ("solution_method", "mixing_weight", "pulay_history", "dm_tolerance"),
)

#: What the run writes out.  All booleans, and **all emitted in both states**
#: -- a keyword left out hands the decision to SIESTA's own default, which is
#: how decks came to have no `.ANI` while the form said otherwise.  Under the
#: door that is not a discipline anybody has to remember: `line` returns a
#: value for False as readily as for True.
OUTPUT_SECTION = Section(
    "Output",
    ("write_forces", "write_coor_step", "write_coor_xmol", "write_md_history",
     "write_md_xmol", "write_hs"),
)

#: The free-energy criterion and the switch that arms it.  **Two items, one
#: decision**: the tolerance alone does nothing -- SIESTA loads it either way
#: and installs it as a criterion only when the switch is on -- so molbuilder
#: wrote a live-looking control that could not change a result until the switch
#: was emitted too.  Both are written, in both states, so a person reading the
#: deck without the form sees the gate as well as the number.
FREE_ENERGY_SECTION = Section(
    "SCF free-energy convergence (a PAIR: the value + its switch)",
    ("dm_energy_tolerance", "scf_energy_converge"),
)

#: The SCF settings that follow the free-energy pair.
SCF_TAIL_SECTION = Section(
    # NAMED 2026-08-19.  It carried "" -- *continues under the SCF heading
    # already written* -- which was true of the deck and left these two
    # keywords under no heading at all, between the free-energy pair above
    # them and the restart group below.  A nameless section is also a layout
    # that stops saying what that part of the deck IS, which is the one thing
    # a layout is for (`script-preparation.md` § 4.1).
    "SCF iteration limit and smearing",
    ("max_scf_iter", "electronic_temperature"),
)


def spin_section(*, polarized: bool, fixed: bool) -> Section:
    """Spin, when there is any — built **per render**.

    A non-polarized calculation writes nothing: SIESTA's own default is what a
    silent deck means here, and saying so is the honest emptiness rather than
    an omission.
    """
    if not polarized:
        return Section("", ())
    return Section("", ("spin_treatment",) + (("spin_total",) if fixed else ()))




def mpi_section(*, block_size, algorithm) -> Section:
    """How the work is split across ranks — built **per render**.

    Which items appear depends on answers this deck has already worked out, so
    the table cannot be a constant:

    * ``BlockSize`` has a third state.  ``block_size = 0`` means *do not emit
      the keyword at all* and let SIESTA choose (`tuning.md` § 2.11), which is
      not the same as emitting a zero.
    * ``Diag.Algorithm`` and its GPU switch are only meaningful for an ELPA
      solver; ScaLAPACK has no such knobs, and writing them would be the deck
      claiming a setting the solver never reads.
    """
    items = []
    if block_size is not None:
        items.append("block_size")
    items.append("parallel_over_k")
    if str(algorithm or "").upper().startswith("ELPA"):
        items += ["diag_algorithm", "enable_gpu"]
    return Section("Parallel execution (MPI)", tuple(items), note=(
        "# These settings matter only with `mpirun -np N siesta`",
        "# (single-rank runs ignore them).",
        "#",
        "# BlockSize: ScaLAPACK orbital-distribution block.  Affects",
        "# cache efficiency for the diagonaliser; does NOT fix the",
        "# propor IMAX=0 crash (an earlier claim was wrong -- an",
        "# empirical sweep confirmed BlockSize = 1, 2, 4 all crash",
        "# at the same mpi_np; propor is a matel_table proportionality",
        "# check, not a BLACS distribution check).  If your run dies",
        "# at startup with ``propor: ERROR: IMAX = 0``: lower mpi_np",
        "# via the wrapper's ``-np`` flag, not BlockSize.  Larger",
        "# BlockSize gives marginally better diag throughput on big",
        "# systems (>1000 atoms / >=16 ranks); for smaller jobs the",
        "# default is fine.  Override only for hand-tuned perf work.",
        "#",
        "# Diag.ParallelOverK: parallelise the diagonaliser over",
        "# k-points (.true.) or over orbitals (.false.).  Auto-",
        "# selected here from the kgrid above: .false. for 1x1x1",
        "# (molecule / vacuum), .true. for multi-k periodic runs.",
        "# NOTE with ELPA (CPU or GPU): SIESTA's ELPA path solves per",
        "# k-point over ORBITALS; with ParallelOverK .true. each",
        "# k-group diagonalises its own k-points and the ELPA GPU",
        "# offload applies within each group.  For few-k metallic",
        "# slabs the .false. (orbital) split usually wins on GPU --",
        "# if you hand-tune one, benchmark it (jobset prep bench).",
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




#: How each value is spelled, and the unit SIESTA expects with it.  A unit is
#: part of the SPELLING, not of the value: the catalogue records ``Ry`` as the
#: item's unit and the deck has to write it next to the number.
_UNIT = {
    "mesh_cutoff":            "Ry",
    "pao_energy_shift":       "Ry",
    "dm_energy_tolerance":    "eV",
    "relax_force_tol":        "eV/Ang",
    "relax_max_displ":        "Ang",
    "md_initial_temperature": "K",
    "md_target_temperature":  "K",
    "md_length_timestep":     "fs",
    "electronic_temperature": "K",
}

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
    "enable_gpu": 19,
    "md_target_temperature": 22,
    "max_scf_iter": 18, "spin_treatment": 18,
}

#: Items SIESTA wants in scientific notation.  Formatting is spelling, so it
#: is the engine's business and lives beside the rest of the spelling.
_FMT = {"dm_tolerance": ".0e", "dm_energy_tolerance": ".0e"}


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

    **It takes the deck's context whole, not a keyword per derived value**
    *(2026-08-19)*.  Seven keyword parameters meant every caller had to
    keep a list in step with this signature, and the context dict the
    writer already keeps had to be spread into it -- so the context could
    carry nothing this door did not also accept.  One argument, one
    channel: what this deck derived.
    Most are the catalogue's ``anchor`` and its value; three groups are not,
    and they are the keyword arguments above:

    * **the MPI block's four are DERIVED** -- the block size from the atom and
      rank counts, ``Diag.ParallelOverK`` from whether the k-mesh is more than
      Gamma, the algorithm normalised.  They reach the door through
      ``parameter(..., value=)``, so a computed number still arrives with its
      declaration, its range and its note;
    * **two geometry keywords are chosen by the run mode** -- SIESTA spells the
      step count differently per algorithm and the displacement cap exists only
      for a relaxation -- and a Nosé run with no target temperature holds it at
      the initial one, which is why that value is resolved before the
      is-it-set guard rather than after;
    * **``spin_total`` expands to two keywords**: the constraint has to be
      switched on as well as given a value, and SIESTA ignores the number
      without ``Spin.Fix``.  The pair comes from the declaration's ``expands``,
      so the deck cannot write one without the other.

    **It was four functions until 2026-08-18** -- ``line``, ``spin_line``,
    ``mpi_line`` and ``geometry_line`` -- and that is why the writer built a
    separate ``DeckSpec`` per section: a spec carries ONE ``line``, so sections
    needing different syntax could not share one.  The framework was being
    worked around rather than used.
    """
    computed = {"block_size":     derived.get("block_size"),
                "parallel_over_k": derived.get("over_k"),
                "diag_algorithm": derived.get("algorithm"),
                "enable_gpu":     derived.get("gpu")}
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
        unit = _UNIT.get(param.name)
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
        fmt = _FMT.get(param.name)
        shown = format(param.value, fmt) if fmt else f"{param.value}"
        unit = _UNIT.get(param.name)
        value = f"{shown} {unit}" if unit else shown
        # One space unless the item is half of an aligned pair.  fdf does not
        # care, but the deck this replaces wrote it this way and a reader
        # diffing two generations should see the values change, not the
        # whitespace.
        pad = _PAD.get(param.name)
        return f"{key:<{pad}}{value}" if pad else f"{key} {value}"

    def _line(param: Parameter) -> Optional[str]:
        if param.name in computed:
            return _mpi(param)
        if param.name in override or param.name == "md_target_temperature":
            return _geometry(param)
        if not param.known or param.value is None:
            return None
        if len(param.writes) == 2:
            return _pair(param)
        return _plain(param)

    return _line


def check_rules(text: str, struct=None, cfg=None):
    """SIESTA's answer to *what must a finished deck of mine satisfy?*

    Four things a deck cannot be wrong about and still mean what it says.  All
    are read off the FILE, after it is written, which is the only way to catch
    a writer bug -- every other validator in this tree takes ``(struct, cfg)``
    and runs before emission.

    **No keyword twice.**  libfdf takes the FIRST match and ignores the rest
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
        norm = key.lower().replace(".", "").replace("_", "").replace("-", "")
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
