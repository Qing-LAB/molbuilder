"""3DNA backend (uses ``fiber``).

3DNA's ``fiber`` produces canonical B-form / A-form / Z-form helical
geometry from sequence, and its ``rebuild`` an explicit, possibly
mismatched duplex.

Tool limitations (and how this module compensates)
--------------------------------------------------
``fiber`` is fast, well-validated for canonical helical geometry, and
the de-facto standard for fiber-diffraction-derived starting
coordinates.  It also has quirks the user would hit:

  1. **Heavy-atom output.**  ``fiber`` writes deoxyribose, base, and
     phosphate heavy atoms only -- no H on bases, sugars, or phosphate
     oxygens.  H atoms are needed for any DFT / MD calculation: the
     electron count is wrong without them, and Watson-Crick H-bonding
     can't form.  We compensate by routing every X3DNA build through
     ``chemistry.add_hydrogens`` (OpenBabel-first, RDKit-fallback) at
     the ``nucleic.build_dna``/``build_rna`` layer.  See
     ``molbuilder/chemistry.py`` for the H-add tool comparison.

  2. **The 5'-terminal phosphate is always written, and one oxygen
     short.**  ``fiber`` and ``rebuild`` emit it whatever flag is passed,
     as P, OP1 and OP2 -- no OP3 (measured 2026-11-18: an odd electron
     count, a radical).  ``_five_prime_end`` strips it for a 5'-OH end and
     completes it otherwise, its third oxygen at the phosphate's fourth
     tetrahedral site; the bridging O5' stays, H is added later by
     chemistry.add_hydrogens.

  3. **No 3'-phosphate.**  fiber's output ends 3'-OH: a ``3P`` is built
     5'-OH / 3'-OH and a ``PP`` 5'-P / 3'-OH (``_TERMINAL_BUILT``), which
     the check after the build says (``backends.built``).

  4. **Form constraints.**  fiber's ``-z`` flag (Z-DNA) only builds the
     poly-d(GC) repeat starting with G; ``-rna`` only produces A-form RNA,
     the one form RNA is asked in (``backends.KIND_FORMS``).  ``build()``
     refuses a Z-form sequence that is not that repeat
     (:func:`_is_z_repeat`).

Detection
---------

Where 3DNA is looked for, in order, is one table (:func:`_places`: in-tree
at the repo root, ``$X3DNA``, ``fiber`` on PATH), and a root counts only
when it is complete (:func:`_missing_from`).  The ``x3dna*/`` directory at
the repo root is gitignored -- both for hygiene and to make it
structurally hard to redistribute 3DNA accidentally.

Licensing
---------
3DNA is distributed under a **non-commercial-use** license through
http://x3dna.org/ behind a registration form.  molbuilder itself is
MIT; we **do not** auto-fetch, mirror, or bundle 3DNA, and the
``BackendUnavailable`` message tells the user to download it from
x3dna.org per their instructions.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ... import repo_root as _repo_root
from ...chemistry import BackendUnavailable, BuildRefused, with_atoms
from ...structure import Structure
from . import Built
from ._common import (first_chain,
                      verify_backbone_connectivity)

#: This backend builds a duplex: canonical by fiber, explicit by rebuild.
DUPLEX = True
#: The line that installs it -- never fetched for the person: its licence
#: is theirs to accept.
INSTALL = ("download 3DNA from x3dna.org after accepting its non-commercial "
           "licence, then unpack it at the repo root or set $X3DNA")
#: No env holds it: the person installs it, as :data:`INSTALL` says.
ENV = None


# Atom names that constitute the 5'-terminal phosphate group in fiber's
# PDB output.  The bridging O5' is part of the sugar and is NOT in this
# set -- after the strip it becomes a free hydroxyl oxygen, ready to
# accept an H from chemistry.add_hydrogens().  We accept both modern
# (OP1/OP2/OP3) and legacy (O1P/O2P/O3P) PDB naming conventions to be
# robust against fiber-version drift.
_PHOSPHATE_ATOM_NAMES = {
    "P", "OP1", "OP2", "OP3", "O1P", "O2P", "O3P",
    # Sometimes phosphate H's appear in protonated outputs:
    "HOP1", "HOP2", "HOP3", "H1P", "H2P", "H3P",
}


# --------------------------------------------------------------------- #
#  Resolution                                                           #
# --------------------------------------------------------------------- #


@dataclass
class _Threedna:
    """Resolved 3DNA install: path to the fiber executable and the
    X3DNA root (which we'll inject into the env when shelling out, so
    fiber's auxiliary scripts can find their config files)."""
    fiber: str
    root:  str


def _missing_from(root: str) -> Optional[str]:
    """What a 3DNA root lacks to be one -- an executable ``bin/fiber``, and
    ``config/``, the per-base PDB templates fiber needs at run time (without
    it fiber dies cryptically) -- or ``None`` when it is complete."""
    fiber = os.path.join(root, "bin", "fiber")
    if not (os.path.isfile(fiber) and os.access(fiber, os.X_OK)):
        return "no executable bin/fiber"
    if not os.path.isdir(os.path.join(root, "config")):
        return "no config/"
    return None


def _places() -> Tuple[Tuple[str, List[str], str], ...]:
    """WHERE 3DNA IS LOOKED FOR, in order (`engines/builders.md` § 4): each
    place's words, the 3DNA roots it names, and what it is when it names
    none -- read by the detection (:func:`_resolve`) and by the account of
    a failed one (:func:`unavailable`), so the two cannot disagree.

    1. In-tree: an ``x3dna*/`` directory at the repo root, where a dev or
       editable install unpacks the tarball -- any suffix (``x3dna-v2.4``,
       ``x3dna-2.5``, a renamed ``x3dna``), a root counting only when it is
       complete (:func:`_missing_from`).
    2. ``$X3DNA``, the root 3DNA's own instructions set.
    3. ``fiber`` on ``PATH``, the root the parent of its ``bin/``."""
    fiber = shutil.which("fiber")
    env = os.environ.get("X3DNA")
    return (
        ("in-tree, at the repo root",
         [str(c) for c in sorted(_repo_root().glob("x3dna*")) if c.is_dir()],
         f"nothing named {_repo_root() / 'x3dna*'}"),
        ("$X3DNA",
         [os.path.abspath(os.path.expanduser(env))] if env else [], "unset"),
        ("fiber on PATH",
         [os.path.dirname(os.path.dirname(os.path.abspath(fiber)))]
         if fiber else [], "not on PATH"),
    )


def _resolve() -> Optional[_Threedna]:
    """The first complete 3DNA root of :func:`_places`, in order."""
    for _words, roots, _none in _places():
        for root in roots:
            if _missing_from(root) is None:
                return _Threedna(fiber=os.path.join(root, "bin", "fiber"),
                                 root=root)
    return None


def is_available() -> bool:
    """True iff some 3DNA install is reachable via the detection chain."""
    return _resolve() is not None


# --------------------------------------------------------------------- #
#  Build                                                                #
# --------------------------------------------------------------------- #


#: FIBER'S FLAG FOR EACH FORM A KIND IS BUILT IN, and the form it lays
#: (from `fiber -h`: -b B-DNA, -a A-DNA, -z Z-DNA -- poly d(GC) only --
#: -rna A-form RNA): the table the build reads and :func:`built` answers
#: from.
_FIBER_FLAGS = {
    ("dna", "B"): (["-b"], "B"),
    ("dna", "A"): (["-a"], "A"),
    ("dna", "Z"): (["-z"], "Z"),
    ("rna", "A"): (["-rna"], "A"),
}


#: THE TERMINAL EACH ASK IS BUILT WITH: fiber and rebuild write a
#: 5'-phosphate -- completed with its third oxygen (:func:`_five_prime_end`)
#: -- and never a 3'-phosphate; the 5'-P is stripped for a 5'-OH end.
_TERMINAL_BUILT = {"OH": "OH", "5P": "5P", "3P": "OH", "PP": "5P"}


def built(kind: str, form: str, terminal: str) -> Built:
    """What this backend builds of the ask (`backends.built`): the form
    fiber's flag for it lays (:data:`_FIBER_FLAGS`, the table its build
    reads), and the terminal its programs write (:data:`_TERMINAL_BUILT`)."""
    _flags, laid = _FIBER_FLAGS[(kind, form)]
    return Built(
        form=laid,
        terminal=_TERMINAL_BUILT[terminal],
        terminal_words="3DNA writes a 5'-phosphate and no 3'-phosphate")


def _is_z_repeat(seq: str) -> bool:
    """True iff ``seq`` is what fiber's ``-z`` Z-DNA model builds: d(GC)n,
    an EVEN-length strict alternation of G and C STARTING WITH G
    (`engines/builders.md` § 5).

    fiber's Z-mode takes ``-rep=N`` (number of dinucleotide GC repeats) and
    never the sequence, and lays 5'-GCGC...-3' -- so the input must be a
    positive even multiple of the GC unit (GC, GCGC, GCGCGC, ...).  A
    C-first alternation (CGCGCG) is the same repeat read one base on:
    fiber would build GCGCGC in its place, and no strand of that duplex
    reads C first.  Odd-length alternations (GCG) would imply a half-repeat
    that the helix model doesn't support.  Each is refused upfront, never
    built as something else.
    """
    return len(seq) >= 2 and seq == "GC" * (len(seq) // 2)


def build(kind: str, sequence: str, form: str, terminal: str,
          title: Optional[str] = None, double_strand: bool = False,
          strand2: Optional[str] = None) -> Structure:
    """``sequence`` -- read by the grammar already -- as fiber's canonical
    helix (a single strand, or with ``double_strand`` its Watson-Crick
    duplex), or with ``strand2`` as rebuild's explicit duplex.  What each
    cannot build is refused before 3DNA runs: rebuild's B-form and equal
    strands (:func:`_explicit_duplex_refused`), fiber's Z repeat
    (:func:`_is_z_repeat`)."""
    # An EXPLICIT second strand => an arbitrary, possibly MISMATCHED duplex:
    # fiber only makes a canonical Watson-Crick one, so rebuild lays it
    # (base-pair-step parameters, each pair naming its two bases).
    if strand2 is not None:
        _explicit_duplex_refused(form, sequence, strand2)
        return _build_arbitrary_duplex(_installed(), sequence, strand2,
                                       terminal, title)

    flags_key = (kind, form)
    if form == "Z" and not _is_z_repeat(sequence):
        raise BuildRefused(
            f"Z-DNA is built by 3DNA's fiber as the alternating poly-d(GC) "
            f"repeat starting with G (e.g. 'GCGCGC'); got {sequence!r}.  fiber lays "
            f"the repeat by count, never the sequence, so a C-first sequence "
            f"would be built as its G-first neighbour -- ask 'GCGC...' for "
            f"that helix, or another form for another sequence.")
    found = _installed()

    # fiber's output is a 5'-phosphate and a 3'-OH whatever the `-single`
    # flag: what each ask is built with is :data:`_TERMINAL_BUILT`, which the
    # check after the build says (`built`).

    # Argument shape depends on the form flag:
    # * ``-b`` / ``-a`` / ``-rna``: take ``-seq=<sequence>`` to lay
    #   bases along the canonical helix (one nucleotide per base in
    #   the input).
    # * ``-z``: pre-defined as poly d(GC); ``-seq=`` is REJECTED by
    #   fiber and it drops into an interactive 'Number of repeats'
    #   prompt instead.  Use ``-rep=N`` where N = len(sequence)/2 (the
    #   sequence is d(GC)n, :func:`_is_z_repeat` above, so len is even).
    if form == "Z":
        n_repeats = len(sequence) // 2  # 2 bases per GC repeat unit
        cmd = [found.fiber] + _FIBER_FLAGS[flags_key][0] + [f"-rep={n_repeats}"]
        if not double_strand:
            # Z-DNA is inherently a duplex (poly d(GC):poly d(GC)); -single
            # extracts one strand.  Omit it for a double strand.
            cmd.append("-single")
    else:
        cmd = [found.fiber] + _FIBER_FLAGS[flags_key][0] + [f"-seq={sequence}"]
        if not double_strand:
            # Single-stranded output (the default).  With `double_strand`,
            # omitting `-single` makes fiber lay down the given strand PLUS its
            # canonical Watson-Crick complement -> a duplex (B- or A-form via the
            # -b / -a flag in _FIBER_FLAGS).
            cmd.append("-single")

    with tempfile.TemporaryDirectory(prefix="molbuilder_3dna_") as workdir:
        pdb_path = os.path.join(workdir, "out.pdb")
        pdb_text = _pdb_made(cmd + [pdb_path], pdb_path, found, workdir)

    struct = Structure.from_pdb(
        pdb_text,
        title=title or f"{kind} {sequence} (3DNA fiber, {form}-form)",
    )

    # fiber's default output is a DUPLEX (chains 'A' + 'B'); `-single` gives one
    # chain.  For a single strand, pin to the first chain (fiber can still emit a
    # 2nd).  For a DOUBLE strand, KEEP BOTH chains -- that IS the duplex.
    if not double_strand:
        struct = first_chain(struct)

    # Each strand's 5' end as built (:func:`_five_prime_end`).  Done AFTER
    # first_chain so the 5'-terminal residue is unambiguous.
    struct = _five_prime_end(struct, terminal)

    err = verify_backbone_connectivity(struct)
    if err is not None:
        raise RuntimeError(
            f"fiber output failed connectivity check: {err}.  This is "
            f"unexpected from fiber and probably indicates a config-"
            f"file mismatch ($X3DNA={found.root})."
        )
    return struct


def _installed() -> "_Threedna":
    """The 3DNA install the detection finds, whose paths the build runs --
    the registry asked ``is_available`` before the build; an install gone in
    between, the one moment this raises, is said as the registry says it
    (:func:`unavailable`)."""
    found = _resolve()
    if found is None:
        raise BackendUnavailable(unavailable(), missing="threedna")
    return found


def _pdb_made(cmd, pdb_path: str, found, workdir: str) -> str:
    """The PDB text a 3DNA program wrote, run in ``workdir``.  A run that
    hangs, fails or writes nothing is the program's failure, raised saying
    so."""
    name = os.path.basename(cmd[0])
    # The program needs $X3DNA at run time so its auxiliary scripts and config
    # look-ups resolve.  Inject ours regardless of what the caller's shell has
    # set.
    env = os.environ.copy()
    env["X3DNA"] = found.root
    env["PATH"]  = os.path.join(found.root, "bin") + os.pathsep + env.get("PATH", "")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=workdir, env=env,
            # Pin stdin to /dev/null so an unanticipated interactive prompt
            # (e.g. an invalid flag combination that drops fiber into its
            # "Number of repeats" loop) fails FAST instead of stalling for the
            # full timeout reading from the parent's inherited stdin.
            stdin=subprocess.DEVNULL, timeout=60)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{name} did not finish within 60 s -- likely hung.  "
                           f"Command: {' '.join(cmd)}") from exc
    if result.returncode != 0 or not os.path.isfile(pdb_path):
        raise RuntimeError(f"{name} failed (exit {result.returncode}).\n"
                           f"Command: {' '.join(cmd)}\n"
                           f"--- stdout ---\n{result.stdout}\n"
                           f"--- stderr ---\n{result.stderr}")
    pdb_text = Path(pdb_path).read_text()
    if not pdb_text.strip():
        raise RuntimeError(f"{name} produced an empty PDB.")
    return pdb_text


# --------------------------------------------------------------------- #
#  The 5' end                                                           #
# --------------------------------------------------------------------- #


def _five_prime_end(struct: Structure, terminal: str) -> Structure:
    """Each strand's 5' end as the ask is built (:data:`_TERMINAL_BUILT`):
    fiber's phosphate -- on each chain's first residue, fiber and rebuild
    writing every chain 5'->3' -- stripped for a 5'-OH (the bridging O5'
    kept; :data:`_PHOSPHATE_ATOM_NAMES`), completed for a 5'-P."""
    first: dict = {}
    for chain, rid in zip(struct.chain_ids, struct.residue_ids):
        first.setdefault(chain, rid)
    ends = [[i for i, (c, r) in enumerate(zip(struct.chain_ids,
                                              struct.residue_ids))
             if c == chain and r == rid] for chain, rid in first.items()]
    if _TERMINAL_BUILT[terminal] == "OH":
        return struct.without(i for end in ends for i in end
                              if struct.atom_names[i] in _PHOSPHATE_ATOM_NAMES)
    return _complete_5prime_phosphate(struct, ends)


#: The bridging oxygen's names, modern and as older PDB files write it.
_O5_NAMES = ("O5'", "O5*")


def _complete_5prime_phosphate(struct: Structure, ends) -> Structure:
    """Each strand's 5'-phosphate with its third oxygen -- ``ends`` each 5'
    nucleotide's atoms: fiber and rebuild write it as P and two non-bridging
    oxygens, one oxygen short of a phosphate monoester (measured
    2026-11-18: ATGC 641 electrons, a radical).  The oxygen is placed at the
    phosphate's fourth tetrahedral site -- opposite the mean direction of
    its three oxygens -- at the template's own non-bridging P-O length,
    named in the template's own convention (``OP3``, or ``O3P`` beside the
    legacy ``O1P``/``O2P``), joined after its nucleotide
    (`chemistry.with_atoms`).  A phosphate of another shape -- its
    third oxygen there already among them -- is the program's output
    changed, raised."""
    pos = np.asarray(struct.positions, dtype=float)
    added = []
    for end in ends:
        mine = {struct.atom_names[i]: i for i in end}
        legacy = "O1P" in mine
        nonbridging = ("O1P", "O2P") if legacy else ("OP1", "OP2")
        third = "O3P" if legacy else "OP3"
        bridge = next((n for n in _O5_NAMES if n in mine), None)
        if ("P" not in mine or bridge is None or third in mine
                or not all(n in mine for n in nonbridging)):
            raise RuntimeError(
                f"3DNA's 5'-phosphate is not P, two non-bridging oxygens and "
                f"O5' (got {sorted(mine)}): the program's output changed")
        p = pos[mine["P"]]
        units = [(pos[mine[n]] - p) / np.linalg.norm(pos[mine[n]] - p)
                 for n in (bridge, *nonbridging)]
        away = -np.sum(units, axis=0)
        length = np.mean([np.linalg.norm(pos[mine[n]] - p)
                          for n in nonbridging])
        added.append((mine["P"], "O",
                      p + length * away / np.linalg.norm(away), third))
    return with_atoms(struct, added)


# --------------------------------------------------------------------- #
#  Arbitrary / mismatched duplex via `rebuild`                          #
# --------------------------------------------------------------------- #

# Arnott fiber B-DNA step parameters -- a straight, canonical right-handed
# B-helix (10 bp/turn).  Used uniformly for every step so an arbitrary
# (possibly mismatched) two-strand duplex lands on a standard B backbone.
_B_DNA_RISE  = 3.38    # Angstrom, rise per base-pair step
_B_DNA_TWIST = 36.0    # degrees,  twist per base-pair step


def _copy_standard_bases(root: str, workdir: str, dataset: str = "BDNA") -> None:
    """Put X3DNA's standard base templates in *workdir*, the way `rebuild` wants
    them.

    THIS IS `x3dna_utils cp_std <dataset>` DONE IN PYTHON, AND THAT MATTERS FOR
    ONE REASON: `x3dna_utils` is the ONLY ruby script in X3DNA -- `fiber`,
    `rebuild` and `analyze` are compiled binaries -- and its `cp_std`
    sub-command is a pure file copy.  Read it (`lib/miscs.rb:373`) and it clears
    `Atomic[._]?.pdb` from the working directory, copies
    `config/atomic/<dataset>_?.pdb` in under two fixed names each, and copies
    `config/Atomic.p.pdb`.  No computation, nothing version-specific, nothing
    that needs a language runtime.

    Shelling out instead would cost a whole interpreter dependency for four
    `shutil.copy` calls, and on a machine without ruby it exits 127.  Only the
    arbitrary/mismatched duplex path reaches here -- the CANONICAL duplex path
    goes through `fiber` -- so such a failure never shows up in the web UI's
    ordinary DNA build and stays invisible until someone builds a mismatch.

    The lowercase alias is X3DNA's convention for a MODIFIED base and `rebuild`
    looks for both spellings, so both are written -- dropping one would fail
    only on modified residues, which is exactly the kind of gap that hides.
    """
    cfg = Path(root) / "config"
    work = Path(workdir)
    for stale in work.glob("Atomic[._]?.pdb"):
        stale.unlink()
    templates = sorted(cfg.glob(f"atomic/{dataset}_?.pdb"))
    if not templates:
        raise RuntimeError(
            f"X3DNA install at {root} has no {dataset} base templates under "
            f"config/atomic/ -- the tree is incomplete, not merely unpacked")
    for src in templates:
        base = src.stem[-1]                     # BDNA_A.pdb -> "A"
        shutil.copy(src, work / f"Atomic_{base}.pdb")
        shutil.copy(src, work / f"Atomic.{base.lower()}.pdb")
    phos = cfg / "Atomic.p.pdb"
    if not phos.is_file():
        raise RuntimeError(
            f"X3DNA install at {root} is missing config/Atomic.p.pdb")
    shutil.copy(phos, work / "Atomic.p.pdb")


def _bp_step_par(strand1: str, strand2: str) -> str:
    """Serialize a 3DNA ``bp_step.par`` file for a two-strand duplex.

    One line per base pair: ``X-Y`` where X = ``strand1[i]`` (5'->3') is paired
    ANTIPARALLEL with Y = ``strand2[N-1-i]`` (strand2 given 5'->3', read 3'->5'
    down the duplex), followed by 12 params -- 6 intra-base-pair (Shear, Stretch,
    Stagger, Buckle, Prop-Tw, Opening) then 6 inter-base-pair STEP (Shift, Slide,
    Rise, Tilt, Roll, Twist).  Intra-bp params are 0 (ideal planar pair); step
    params are 0 for the first pair (no preceding step) and Arnott B-DNA
    (Rise 3.38 A, Twist 36 deg) thereafter.
    """
    n = len(strand1)
    lines = [
        f"{n:4d} # base-pairs",
        "   0 # ***local base-pair & step parameters***",
        "#        Shear    Stretch   Stagger   Buckle   Prop-Tw   Opening"
        "     Shift     Slide     Rise      Tilt      Roll      Twist",
    ]
    for i in range(n):
        x = strand1[i]
        y = strand2[n - 1 - i]                     # antiparallel partner
        rise  = 0.0 if i == 0 else _B_DNA_RISE
        twist = 0.0 if i == 0 else _B_DNA_TWIST
        vals = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,      # intra-bp
                0.0, 0.0, rise, 0.0, 0.0, twist]   # step
        lines.append(f"{x}-{y}  " + "".join(f"{v:10.3f}" for v in vals))
    return "\n".join(lines) + "\n"


def _explicit_duplex_refused(form: str, strand1: str, strand2: str) -> None:
    """Refuse an explicit duplex rebuild does not lay: another form than B
    (it lays the standard B-DNA templates), or strands of unequal length
    (every base paired: no overhang, no bulge)."""
    if form != "B":
        raise BuildRefused(
            f"two-strand (arbitrary / mismatched) duplexes are B-form only; got "
            f"form={form!r}.  Use \"ds,<sequence>\" for an A- or Z-form canonical "
            f"duplex (the complementary strand is generated automatically).")
    if len(strand1) != len(strand2):
        raise BuildRefused(
            f"the two strands of a duplex must be equal length (a fully "
            f"base-paired duplex; overhangs / bulges aren't supported): "
            f"strand1={len(strand1)} nt, strand2={len(strand2)} nt.")


def _build_arbitrary_duplex(found: "_Threedna", strand1: str, strand2: str,
                            terminal: str,
                            title: Optional[str]) -> Structure:
    """Build a two-strand (possibly MISMATCHED) B-form duplex via X3DNA ``rebuild``.

    ``fiber`` can only make a canonical Watson-Crick duplex (strand2 == the
    complement of strand1).  ``rebuild`` consumes a base-pair-step parameter file
    in which each pair names its two bases independently, so an arbitrary strand2
    -- including non-WC mismatches (A-G, T-T, ...) -- can be laid onto a standard
    B-DNA backbone.  Mismatched pairs are placed at the standard base-pair frame
    as a STARTING model (they may clash; relax before use).

    B-form only: ``rebuild`` uses the standard B-DNA atomic templates copied by
    ``x3dna_utils cp_std BDNA``; A/Z-form arbitrary duplexes aren't offered (use
    ``ds,<sequence>`` for an A/Z *canonical* duplex via fiber).
    """
    par_text    = _bp_step_par(strand1, strand2)
    # `x3dna_utils` is deliberately NOT used: it is 3DNA's only interpreted
    # tool (ruby, and Perl before upstream's 2.4 rewrite) and the one
    # sub-command we needed from it is a file copy.  See
    # `_copy_standard_bases`.
    rebuild     = os.path.join(found.root, "bin", "rebuild")

    with tempfile.TemporaryDirectory(prefix="molbuilder_3dna_") as workdir:
        Path(os.path.join(workdir, "bp_step.par")).write_text(par_text)
        # `rebuild -atomic` needs the standard B-DNA atomic templates in the CWD
        # first.  We COPY THEM OURSELVES rather than shelling out to
        # `x3dna_utils cp_std BDNA` -- see `_copy_standard_bases`.
        _copy_standard_bases(found.root, workdir, "BDNA")
        pdb_text = _pdb_made([rebuild, "-atomic", "bp_step.par", "out.pdb"],
                             os.path.join(workdir, "out.pdb"), found, workdir)

    struct = Structure.from_pdb(
        pdb_text,
        title=title or f"dna {strand1}/{strand2} (3DNA rebuild, B-form duplex)")
    # KEEP both chains -- that IS the duplex.  Each strand's 5' end as built
    # (:func:`_five_prime_end`); H is added downstream by chemistry.add_hydrogens.
    struct = _five_prime_end(struct, terminal)
    err = verify_backbone_connectivity(struct)
    if err is not None:
        raise RuntimeError(
            f"rebuild output failed connectivity check: {err} "
            f"($X3DNA={found.root}).")
    return struct


# --------------------------------------------------------------------- #
#  Error message                                                        #
# --------------------------------------------------------------------- #


def unavailable() -> str:
    """Why 3DNA cannot run here (`backends.unavailable_words`): each place
    the detection looks, in its order (:func:`_places`), and what it found
    there -- nothing, or a root lacking a piece (:func:`_missing_from`) --
    and how it is installed (:data:`INSTALL`)."""
    lines = ["3DNA is not installed; looked for, in order:"]
    for n, (words, roots, none) in enumerate(_places(), 1):
        held = "; ".join(f"{root}: {_missing_from(root) or 'complete'}"
                         for root in roots)
        lines.append(f"  {n}. {words}: {held or none}")
    lines.append(f"To install it: {INSTALL}.")
    return "\n".join(lines)
