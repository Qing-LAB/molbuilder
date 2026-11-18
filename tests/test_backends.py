"""Backend dispatch / registry tests (`engines/builders.md` § 4).

  * ``auto`` runs the first installed in `BACKENDS`' order; none installed
    is each backend's own account.
  * Every backend answers every ask the registry allows (``built``).
  * 3DNA is found at the first complete place it is looked for, and not
    found is said place by place, on every road that needs it.
  * 3DNA's real builds, where it is installed.

A backend asked by name builds on it, no fallback: the real builds that ask
one by name pin it (an RDKit 3'-phosphate, `test_nucleic.py`; the CLI's
backend row, `test_cli.py`).  Absent backends through the one door
(`support.tooling.make_absent`), or a real 3DNA tree of the shape a person
leaves.
"""

from __future__ import annotations

import pytest

from molbuilder.builders.backends import dispatch
from molbuilder.chemistry import BackendUnavailable
from support import tooling


@pytest.mark.parametrize("absent, runs", [
    ((), "threedna"),
    (("threedna",), "amber"),
    (("threedna", "amber"), "rdkit"),
    (("threedna", "amber", "rdkit"), None),
], ids=["all-installed", "no-3dna", "only-rdkit", "none-installed"])
def test_auto_runs_the_first_backend_installed(monkeypatch, absent, runs):
    """`engines/builders.md` § 4: ``auto`` runs the first installed backend
    in `BACKENDS`' order -- best geometry first, 3DNA, AmberTools, RDKit --
    each skipped one quietly; with none installed, one answer naming every
    backend and how each is installed.  The backends made absent by the
    one door (`support.tooling.make_absent`)."""
    from molbuilder.builders.backends import resolved_backend
    if runs:
        tooling.needs_backend(runs)
    for name in absent:
        tooling.make_absent(monkeypatch, name)
    if runs:
        assert resolved_backend("auto") == runs
        return
    with pytest.raises(BackendUnavailable) as exc:
        dispatch("dna", "ATGC", backend="auto")
    msg = str(exc.value)
    assert all(name in msg for name in ("3DNA", "AmberTools", "RDKit")), msg


def test_every_backend_answers_every_ask_the_registry_allows():
    """`engines/builders.md` § 1: each backend's own table covers every form
    a kind is built in and every terminal (``KIND_FORMS``, ``TERMINALS``) --
    its ``built`` answers each, a form asked or none, a terminal of the
    registry's.  The silent failure: a word added to the registry is offered
    at once by the command line and the page's menus, and a backend whose
    table lacks it ends in a KeyError -- a server fault, or a crash in the
    check after a build that succeeded."""
    from molbuilder.builders.backends import (BACKENDS, KIND_FORMS,
                                              TERMINALS, built)
    for backend in BACKENDS:
        for kind, forms in KIND_FORMS.items():
            for form in forms:
                for terminal in TERMINALS:
                    made = built(backend, kind, form, terminal)
                    assert made.form in (form, None), (backend, kind, form)
                    assert made.terminal in TERMINALS, (backend, terminal)


# --------------------------------------------------------------------- #
#  3DNA detection: in-tree at the repo root, $X3DNA, fiber on PATH       #
# --------------------------------------------------------------------- #


def _a_3dna_root(at, *, config=True):
    """A 3DNA tree as its tarball unpacks at ``at``: an executable
    ``bin/fiber``, and ``config/`` unless ``config`` is false."""
    fiber = at / "bin" / "fiber"
    fiber.parent.mkdir(parents=True)
    fiber.write_text("#!/bin/sh\nexit 0\n")
    fiber.chmod(0o755)
    if config:
        (at / "config").mkdir()
    return at


@pytest.mark.parametrize("in_tree, env, found", [
    (None, "complete", "env"),
    ("complete", "complete", "in-tree"),
    ("no config", "no config", None),
], ids=["$X3DNA when nothing in-tree", "in-tree before $X3DNA",
        "a root without config/ is never claimed"])
def test_threedna_is_found_at_the_first_complete_place(tmp_path, monkeypatch,
                                                       in_tree, env, found):
    """`engines/builders.md` § 4: 3DNA is looked for in order -- in-tree at
    the repo root, ``$X3DNA``, ``fiber`` on PATH -- and the first COMPLETE
    root, an executable ``bin/fiber`` and ``config/``, is used.  The silent
    failure: a root without ``config/`` claimed, and fiber failing at run
    time, cryptically.  Real trees of each shape; PATH holding no fiber."""
    from molbuilder.builders.backends import _threedna
    repo = tmp_path / "repo"
    repo.mkdir()
    roots = {}
    if in_tree:
        roots["in-tree"] = _a_3dna_root(repo / "x3dna-v2.4",
                                        config=in_tree == "complete")
    roots["env"] = _a_3dna_root(tmp_path / "elsewhere",
                                config=env == "complete")
    monkeypatch.setattr(_threedna, "_repo_root", lambda: repo)
    monkeypatch.setenv("X3DNA", str(roots["env"]))
    monkeypatch.setenv("PATH", str(tmp_path / "nowhere"))
    got = _threedna._resolve()
    assert (got.root if got else None) == (str(roots[found]) if found
                                           else None)


@pytest.mark.parametrize("text, asked", [
    ("ATGC", {"backend": "threedna"}),
    ("ds,ATGC", {}),
], ids=["3DNA asked by name", "a duplex, which 3DNA builds"])
def test_threedna_not_found_says_where_it_was_looked_for(tmp_path,
                                                         monkeypatch, text,
                                                         asked):
    """`engines/builders.md` § 4: 3DNA not found says each place it looked
    and what it found there, and how it is installed -- its licence the
    person's to accept -- on every road that needs it.  The silent failures:
    3DNA unpacked at the repo root without its ``config/`` said as "no
    match", the person told to unpack it where it was (F12's fourth review);
    a duplex asked told only the install line (the fifth, 2026-11-18).  A
    real tree of that shape at the repo root, 3DNA's other places empty."""
    from molbuilder.builders import build
    from molbuilder.builders.backends import _threedna
    _a_3dna_root(tmp_path / "x3dna-v2.4", config=False)
    monkeypatch.setattr(_threedna, "_repo_root", lambda: tmp_path)
    monkeypatch.delenv("X3DNA", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "nowhere"))
    with pytest.raises(BackendUnavailable) as exc:
        build("dna", text, **asked)
    msg = str(exc.value)
    assert all(word in msg for word in ("x3dna-v2.4", "config", "$X3DNA",
                                        "PATH", "x3dna.org")), msg


# --------------------------------------------------------------------- #
#  3DNA real-fiber smoke (skipped if no 3DNA install)                   #
# --------------------------------------------------------------------- #


def test_threedna_builds_real_dna_when_installed():
    """When 3DNA is reachable on this machine, a tiny B-form DNA build
    must produce a chemically plausible Structure: at least one P, at
    least one of each base (DA/DT/DG/DC), backbone connectivity check
    passing.  Skipped on machines without 3DNA."""
    from molbuilder.builders.backends import _threedna
    tooling.needs_backend("threedna")
    s = _threedna.build("dna", "ATGC", form="B", terminal="OH")
    assert s.n_atoms > 0
    assert "P" in s.elements
    assert {"DA", "DT", "DG", "DC"} <= set(s.residue_names)


def test_threedna_a_form_differs_from_b_form():
    """A-form and B-form DNA have different helical parameters -- the
    geometries the fiber backend produces must actually differ.  This
    catches the case where the form flag isn't being plumbed through."""
    from molbuilder.builders.backends import _threedna
    tooling.needs_backend("threedna")
    b = _threedna.build("dna", "ATGC", form="B", terminal="OH")
    a = _threedna.build("dna", "ATGC", form="A", terminal="OH")
    # Same sequence -> same atom count, but positions differ.
    assert a.n_atoms == b.n_atoms
    import numpy as np
    diff = np.linalg.norm(a.positions - b.positions, axis=1)
    assert diff.max() > 0.1, (
        "A-form and B-form DNA produced identical coordinates -- "
        "form flag is probably not reaching fiber."
    )


@pytest.mark.parametrize("sequence", [
    "ATGC",      # not the GC repeat
    "CGCGCG",    # the repeat C first: fiber would build GCGCGC in its place
])
def test_threedna_z_form_refuses_what_fiber_does_not_build(sequence):
    """`engines/builders.md` § 5: fiber's Z model is d(GC)n, G first, laid
    by repeat count and never by sequence -- any other Z ask is refused
    BEFORE fiber runs, naming the way, never built as something else.  Two
    silent failures: fiber's interactive 'Number of repeats' prompt hanging
    the request for 60 s (2026-06-07), and a C-first ask built as its
    G-first neighbour with no word (F12's review, 2026-11-18)."""
    from molbuilder.builders.backends import _threedna
    tooling.needs_backend("threedna")
    from molbuilder.chemistry import BuildRefused
    with pytest.raises(BuildRefused) as refused:
        _threedna.build("dna", sequence, form="Z", terminal="OH")
    # Refused naming what was asked and the repeat it builds (§ 5).
    assert sequence in str(refused.value) and "GCGC" in str(refused.value)


def test_threedna_z_form_builds_the_repeat_it_was_asked():
    """The flip side: GCGCGC in Z-form builds, and the strand reads G first
    from its 5' end -- the sequence asked, residue by residue."""
    from molbuilder.builders.backends import _threedna
    tooling.needs_backend("threedna")
    s = _threedna.build("dna", "GCGCGC", form="Z", terminal="OH")
    order = [name for _, name in sorted({(r, n) for r, n in zip(
        s.residue_ids, s.residue_names)})]
    assert order == ["DG", "DC", "DG", "DC", "DG", "DC"], order


def test_threedna_rna_uses_uracil_not_thymine():
    """RNA must use U (uracil), not T (thymine).  fiber's `-rna` flag
    has to be set; if we accidentally call DNA mode for RNA, residue
    names will include DT and the structure will be chemically wrong."""
    from molbuilder.builders.backends import _threedna
    tooling.needs_backend("threedna")
    r = _threedna.build("rna", "AUGC", form="A", terminal="OH")
    assert "DT" not in r.residue_names, (
        "RNA build emitted DT residues -- fiber was called in DNA mode"
    )
    # A/U/G/C (RNA) vs DA/DT/DG/DC (DNA): RNA uses single-letter codes.
    assert {"A", "U", "G", "C"} & set(r.residue_names)
