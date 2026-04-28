"""S8 regression: TER record handling in Structure.from_pdb.

Three on-disk fixtures under tests/data/:

  multi_chain_well_formed.pdb   -- explicit chain ids A, B, separated
                                   by TER (the case that already worked)
  multi_chain_ter_only.pdb      -- chain id 'A' reused on both polymers
                                   with a TER between them (the bug)
  multi_chain_blank_chain_id.pdb -- chain id column blank on every line
                                   plus a TER between polymers
"""

from __future__ import annotations

from molbuilder.structure import Structure


def test_well_formed_two_chains_unchanged(data_dir):
    """Back-compat: a well-formed PDB with explicit chain ids A and B
    must still produce chain ids exactly A and B (no segment suffixes)."""
    s = Structure.from_pdb(data_dir / "multi_chain_well_formed.pdb")
    assert s.n_atoms == 8
    assert sorted(set(s.chain_ids)) == ["A", "B"]
    # Per-chain atom counts should be 4+4
    chain_a = [i for i in range(s.n_atoms) if s.chain_ids[i] == "A"]
    chain_b = [i for i in range(s.n_atoms) if s.chain_ids[i] == "B"]
    assert len(chain_a) == 4
    assert len(chain_b) == 4


def test_ter_only_separates_reused_chain_id(data_dir):
    """The bug case: chain-id 'A' is reused on both polymers, separated
    only by TER.  Previously parsed as one chain; now must be two."""
    s = Structure.from_pdb(data_dir / "multi_chain_ter_only.pdb")
    assert s.n_atoms == 8
    chains = sorted(set(s.chain_ids))
    assert len(chains) == 2, f"expected 2 logical chains, got {chains}"
    # Convention: first segment uses the literal letter+segment-index;
    # since the letter spans multiple segments, both get suffixed.
    assert chains == ["A0", "A1"]


def test_blank_chain_id_column_disambiguated(data_dir):
    """Blank chain-id column + TER -> two distinct synthesised chain ids.

    The internal '_' placeholder makes the synthesis visible: '_0', '_1'.
    """
    s = Structure.from_pdb(data_dir / "multi_chain_blank_chain_id.pdb")
    assert s.n_atoms == 8
    chains = sorted(set(s.chain_ids))
    assert len(chains) == 2
    assert chains == ["_0", "_1"]


def test_single_chain_no_ter_is_back_compat():
    """A simple PDB with one chain ID 'A' and no TER must still parse
    to a single chain 'A' (i.e. the new logic doesn't introduce
    segment suffixes when the input was unambiguous)."""
    pdb = (
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A   1       1.460   0.000   0.000  1.00  0.00           C\n"
        "END\n"
    )
    s = Structure.from_pdb(pdb)
    assert s.n_atoms == 2
    assert s.chain_ids == ["A", "A"]


def test_blank_chain_id_no_ter_uses_legacy_a():
    """No TER + blank chain id column -> all atoms map to 'A'
    (preserves the previous parser's `or "A"` fallback)."""
    pdb = (
        "ATOM      1  N   ALA     1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  ALA     1       1.460   0.000   0.000  1.00  0.00           C\n"
        "END\n"
    )
    s = Structure.from_pdb(pdb)
    assert s.chain_ids == ["A", "A"]


def test_multiple_consecutive_ters_dont_break():
    """Some exporters emit several TERs in a row.  Each bumps the
    segment counter but only counts as separating chains if atoms
    follow.  This shouldn't crash."""
    pdb = (
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "TER       2\n"
        "TER       3\n"
        "TER       4\n"
        "ATOM      5  N   GLY A   1       5.000   0.000   0.000  1.00  0.00           N\n"
        "END\n"
    )
    s = Structure.from_pdb(pdb)
    # Two segments contain atoms (0 and 3); the chain-letter 'A' is
    # in both, so it gets disambiguated.
    chains = sorted(set(s.chain_ids))
    assert len(chains) == 2
    assert chains == ["A0", "A3"]


def test_residue_disambiguation_via_chain_ids(data_dir):
    """Cross-check: with TER separation, residue (chain_id, residue_id)
    tuples are now unique even when residue_id collides."""
    s = Structure.from_pdb(data_dir / "multi_chain_ter_only.pdb")
    keys = set(zip(s.chain_ids, s.residue_ids))
    # Before the fix, every atom had key ('A', 1) -- 8 atoms collapsed
    # into one logical residue.  Now we expect at least two unique keys.
    assert len(keys) == 2
