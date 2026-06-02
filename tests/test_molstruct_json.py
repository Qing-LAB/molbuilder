"""Tests for the ``.molstruct.json`` sidecar parser.

The sidecar travels alongside an XYZ + carries the region labels +
frozen-atom indices that XYZ can't represent.  We pin three contracts:

  1.  Round-trip identity: to_dict -> save -> load -> apply matches.
  2.  Strict validation: malformed sidecars raise MolstructJsonError
      with a message pointing at the file.
  3.  Invariant pinning: structure_hash + n_atoms_total catch the
      cases where the user edited the XYZ without re-exporting the
      sidecar (which would otherwise silently produce wrong scripts).
"""

from __future__ import annotations

import json as _json
from pathlib import Path

import numpy as np
import pytest

from molbuilder.parsers import molstruct_json as msj
from molbuilder.structure import Structure


# --------------------------------------------------------------------- #
#  Helpers                                                              #
# --------------------------------------------------------------------- #


def _toy_xyz_text(n: int = 4) -> str:
    """Tiny synthetic XYZ -- N carbons on a line.  Used to give the
    parser a real on-disk file to hash."""
    lines = [f"{n}", "synthetic"]
    for i in range(n):
        lines.append(f"C   {float(i):.3f}   0.000   0.000")
    return "\n".join(lines) + "\n"


def _write_xyz(tmp_path: Path, n: int = 4) -> Path:
    p = tmp_path / "synthetic.xyz"
    p.write_text(_toy_xyz_text(n))
    return p


# --------------------------------------------------------------------- #
#  Path helper                                                          #
# --------------------------------------------------------------------- #


class TestSidecarPathFor:
    def test_plain_xyz(self, tmp_path):
        p = tmp_path / "relaxed.xyz"
        assert msj.sidecar_path_for(p).name == "relaxed.molstruct.json"

    def test_compound_suffix(self, tmp_path):
        """``job.spectra.xyz`` -> ``job.spectra.molstruct.json``.
        ``Path.with_suffix`` chaining would have eaten ``.spectra``."""
        p = tmp_path / "job.spectra.xyz"
        assert msj.sidecar_path_for(p).name == "job.spectra.molstruct.json"

    def test_no_suffix(self, tmp_path):
        p = tmp_path / "bridge"
        assert msj.sidecar_path_for(p).name == "bridge.molstruct.json"


# --------------------------------------------------------------------- #
#  to_dict (canonical builder)                                          #
# --------------------------------------------------------------------- #


class TestToDict:
    def test_minimum_required_fields(self):
        d = msj.to_dict(n_atoms_total=4, structure_hash="a" * 64)
        assert d["schema_version"] == msj.SCHEMA_VERSION
        assert d["n_atoms_total"]  == 4
        assert d["structure_hash"] == "a" * 64
        assert d["regions"]        == {}
        assert d["frozen_atoms"]    == []
        # Auto-generated timestamp + default created_by:
        assert d["created_by"] == "molbuilder"
        assert d["created_at"].endswith("Z")

    def test_regions_sorted_and_deduped(self):
        d = msj.to_dict(
            n_atoms_total=10, structure_hash="b" * 32,
            regions={"L-electrode": [3, 1, 1, 0]},
        )
        assert d["regions"]["L-electrode"] == [0, 1, 3]

    def test_frozen_atoms_sorted_and_deduped(self):
        d = msj.to_dict(
            n_atoms_total=10, structure_hash="b" * 32,
            frozen_atoms=[5, 1, 1, 0],
        )
        assert d["frozen_atoms"] == [0, 1, 5]

    def test_region_out_of_range_raises(self):
        with pytest.raises(msj.MolstructJsonError, match="out of range"):
            msj.to_dict(
                n_atoms_total=3, structure_hash="b" * 32,
                regions={"L-electrode": [5]},
            )

    def test_region_overlap_is_allowed(self):
        """Multi-label model: an atom may carry several region tags
        at once.  Engines that need a disjoint partition validate
        that separately at engine-load time."""
        d = msj.to_dict(
            n_atoms_total=4, structure_hash="b" * 32,
            regions={"a": [0, 1], "b": [1, 2]},
        )
        assert d["regions"]["a"] == [0, 1]
        assert d["regions"]["b"] == [1, 2]

    def test_empty_label_raises(self):
        with pytest.raises(msj.MolstructJsonError, match="non-empty"):
            msj.to_dict(
                n_atoms_total=3, structure_hash="b" * 32,
                regions={"": [0]},
            )

    def test_bad_n_atoms_raises(self):
        with pytest.raises(msj.MolstructJsonError, match="non-negative"):
            msj.to_dict(n_atoms_total=-1, structure_hash="b" * 32)

    def test_bad_hash_raises(self):
        with pytest.raises(msj.MolstructJsonError, match="hex string"):
            msj.to_dict(n_atoms_total=3, structure_hash="short")


# --------------------------------------------------------------------- #
#  Save + load round-trip                                               #
# --------------------------------------------------------------------- #


class TestSaveLoadRoundTrip:
    def test_round_trip_identity(self, tmp_path):
        xyz = _write_xyz(tmp_path, n=6)
        payload = msj.to_dict(
            n_atoms_total=6,
            structure_hash=msj.sha256_of_file(xyz),
            regions={"L-electrode": [0, 1], "R-electrode": [4, 5],
                     "bridge": [2, 3]},
            frozen_atoms=[0, 5],
        )
        side = msj.sidecar_path_for(xyz)
        msj.save(side, payload)
        loaded = msj.load(side)
        # Everything except created_at (timestamp) should match:
        for key in ("schema_version", "n_atoms_total", "structure_hash",
                    "regions", "frozen_atoms", "created_by"):
            assert loaded[key] == payload[key]

    def test_atomic_write_no_partial_on_disk(self, tmp_path):
        """``save`` should leave no ``.tmp`` next to the sidecar after
        a successful write -- the atomic-replace path runs to completion."""
        xyz = _write_xyz(tmp_path)
        payload = msj.to_dict(
            n_atoms_total=4,
            structure_hash=msj.sha256_of_file(xyz),
        )
        side = msj.sidecar_path_for(xyz)
        msj.save(side, payload)
        leftover = list(tmp_path.glob("*.tmp"))
        assert leftover == [], f"left behind: {leftover}"


# --------------------------------------------------------------------- #
#  with_lock(): serialises concurrent read-modify-write cycles          #
# --------------------------------------------------------------------- #


class TestWithLockSerialisation:
    """``with_lock`` MUST hold an exclusive flock for the duration of
    the context block so two concurrent read-modify-write cycles can't
    lose one update.  Verified via threading: each thread does a
    canonical RMW (load existing -> tag in a key -> save), and after
    both threads complete the file holds the merged state."""

    def _rmw_in_thread(self, tmp_path, key, hold_time_s):
        """Worker: lock, read existing, add ``key``, sleep (to widen
        the race window), save, release.  ``hold_time_s`` is the
        between-load-and-save delay -- with the lock it doesn't matter
        how long; without it, longer delay makes the race more
        reliably reproducible."""
        import time
        side = tmp_path / "race.molstruct.json"
        with msj.with_lock(side):
            existing = msj.load(side) if side.exists() else None
            if existing is None:
                regions = {}
            else:
                regions = dict(existing.get("regions") or {})
            time.sleep(hold_time_s)
            # Each thread tags its OWN key in regions; if both reads
            # see the original (empty) state and both writes happen,
            # the second writer clobbers the first and only its tag
            # lands on disk.
            regions[key] = [0, 1, 2]
            payload = msj.to_dict(
                n_atoms_total=4,
                structure_hash="0" * 64,
                regions=regions,
            )
            msj.save(side, payload)

    def test_two_concurrent_rmw_cycles_both_land(self, tmp_path):
        """Run two RMW workers in parallel.  Each tags a different
        region key.  If ``with_lock`` serialises correctly, the
        final sidecar contains BOTH keys.  Without serialisation,
        the second writer clobbers the first and only one survives."""
        import threading
        t1 = threading.Thread(
            target=self._rmw_in_thread,
            args=(tmp_path, "alpha", 0.05),
        )
        t2 = threading.Thread(
            target=self._rmw_in_thread,
            args=(tmp_path, "beta", 0.05),
        )
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)
        assert not t1.is_alive(), "rmw thread #1 still running"
        assert not t2.is_alive(), "rmw thread #2 still running"

        side = tmp_path / "race.molstruct.json"
        final = msj.load(side)
        regions = final.get("regions") or {}
        assert "alpha" in regions, (
            f"lock failed to serialise: alpha lost from final state; "
            f"got regions={regions!r}"
        )
        assert "beta" in regions, (
            f"lock failed to serialise: beta lost from final state; "
            f"got regions={regions!r}"
        )

    def test_lock_file_created_in_sidecar_dir(self, tmp_path):
        """``with_lock`` creates a ``.lock`` sibling file in the
        sidecar's directory; left in place after the block exits so
        a subsequent save can reuse it (no cleanup overhead in the
        hot path)."""
        side = tmp_path / "lock-creation.molstruct.json"
        lock = side.with_suffix(side.suffix + ".lock")
        assert not lock.exists()
        with msj.with_lock(side):
            assert lock.exists(), "lock file should be created on entry"
        # Left behind on exit; next caller reuses it.
        assert lock.exists()

    def test_lock_does_not_create_the_sidecar_itself(self, tmp_path):
        """The lock is a SIBLING file -- entering the block on a path
        whose sidecar doesn't exist yet must NOT create the sidecar."""
        side = tmp_path / "missing-sidecar.molstruct.json"
        assert not side.exists()
        with msj.with_lock(side):
            pass
        assert not side.exists(), (
            "with_lock created the sidecar; should only create the "
            "sibling .lock file"
        )


# --------------------------------------------------------------------- #
#  load() error paths                                                   #
# --------------------------------------------------------------------- #


class TestLoadErrors:
    def test_missing_file(self, tmp_path):
        with pytest.raises(msj.MolstructJsonError, match="failed to read"):
            msj.load(tmp_path / "missing.molstruct.json")

    def test_invalid_json(self, tmp_path):
        p = tmp_path / "bad.molstruct.json"
        p.write_text("{not json,,,")
        with pytest.raises(msj.MolstructJsonError, match="not valid JSON"):
            msj.load(p)

    def test_top_level_must_be_object(self, tmp_path):
        p = tmp_path / "bad.molstruct.json"
        p.write_text('[1, 2, 3]')
        with pytest.raises(msj.MolstructJsonError, match="must be an object"):
            msj.load(p)

    def test_unknown_schema_version(self, tmp_path):
        p = tmp_path / "bad.molstruct.json"
        p.write_text(_json.dumps({
            "schema_version": 99,
            "n_atoms_total": 3,
            "structure_hash": "b" * 32,
        }))
        with pytest.raises(msj.MolstructJsonError, match="schema_version"):
            msj.load(p)

    def test_missing_required_field(self, tmp_path):
        p = tmp_path / "bad.molstruct.json"
        p.write_text(_json.dumps({
            "schema_version": 3,
            "n_atoms_total": 3,
            # structure_hash missing
        }))
        with pytest.raises(msj.MolstructJsonError, match="structure_hash"):
            msj.load(p)

    def test_load_propagates_error_with_path(self, tmp_path):
        p = tmp_path / "bad.molstruct.json"
        p.write_text(_json.dumps({
            "schema_version": 3,
            "n_atoms_total": 3,
            "structure_hash": "b" * 32,
            "regions": {"L": [99]},  # out-of-range
        }))
        with pytest.raises(msj.MolstructJsonError) as exc:
            msj.load(p)
        # Path AND the underlying invariant should both appear:
        assert str(p) in str(exc.value)
        assert "out of range" in str(exc.value)


# --------------------------------------------------------------------- #
#  Apply to a Structure                                                 #
# --------------------------------------------------------------------- #


class TestApplyToStructure:
    def _struct(self, n: int = 4) -> Structure:
        return Structure(
            elements=["C"] * n,
            positions=np.zeros((n, 3)),
        )

    def test_applies_labels_and_frozen(self, tmp_path):
        s = self._struct(n=4)
        data = msj.to_dict(
            n_atoms_total=4, structure_hash="b" * 32,
            regions={"L-electrode": [0, 1], "R-electrode": [3]},
            frozen_atoms=[0],
        )
        msj.apply_to_structure(s, data)
        assert s.regions == {"L-electrode": [0, 1], "R-electrode": [3]}
        assert s.frozen_atoms == [0]

    def test_atom_count_mismatch_raises(self, tmp_path):
        s = self._struct(n=3)
        data = msj.to_dict(n_atoms_total=4, structure_hash="b" * 32)
        with pytest.raises(msj.MolstructJsonError, match="no longer point"):
            msj.apply_to_structure(s, data)

    def test_empty_sidecar_is_a_clean_reset(self, tmp_path):
        """Loading an empty sidecar onto a Structure that already has
        labels should clear them (sidecar is the source of truth)."""
        s = Structure(
            elements=["C"] * 3, positions=np.zeros((3, 3)),
            regions={"junk": [0]}, frozen_atoms=[2],
        )
        data = msj.to_dict(n_atoms_total=3, structure_hash="b" * 32)
        msj.apply_to_structure(s, data)
        assert s.regions == {}
        assert s.frozen_atoms == []


# --------------------------------------------------------------------- #
#  Hash invariant                                                       #
# --------------------------------------------------------------------- #


class TestSchemaVersioning:
    """v3 is the only schema this build reads.  v1/v2 sidecars
    (which used the older ``fixed_atoms`` key) must be re-exported
    from /modify -- the parser refuses them rather than silently
    coercing.  The reader-version list is the single source of
    truth for which on-disk schemas this build accepts."""

    def test_v2_fails_to_load(self, tmp_path):
        """v2 had ``fixed_atoms`` key + no terminology unification.
        v3 onwards uses ``frozen_atoms``.  Strict load: refuse rather
        than silently coerce."""
        p = tmp_path / "v2.molstruct.json"
        p.write_text(_json.dumps({
            "schema_version": 2,
            "n_atoms_total":  3,
            "structure_hash": "b" * 32,
            "regions":        {"L-electrode": [0]},
            "fixed_atoms":    [],
        }))
        with pytest.raises(msj.MolstructJsonError, match="reads versions"):
            msj.load(p)

    def test_v3_writes_with_frozen_atoms_key(self, tmp_path):
        """Canonical write: the key is ``frozen_atoms``."""
        d = msj.to_dict(
            n_atoms_total=3, structure_hash="b" * 32,
            regions={"L-electrode": [0]},
            frozen_atoms=[1, 2],
        )
        assert d["schema_version"] == 3
        assert "frozen_atoms" in d
        assert "fixed_atoms" not in d   # the old key is gone

    def test_unknown_version_raises(self, tmp_path):
        p = tmp_path / "v99.molstruct.json"
        p.write_text(_json.dumps({
            "schema_version": 99,
            "n_atoms_total":  3,
            "structure_hash": "b" * 32,
        }))
        with pytest.raises(msj.MolstructJsonError, match="reads versions"):
            msj.load(p)


class TestSelectionRules:
    """Sidecar's optional ``selection_rules`` field round-trips a
    JSON rule tree from :mod:`molbuilder.selection`; validation
    re-parses via from_json so a malformed recipe surfaces at
    sidecar-build time, not at engine-load time."""

    def _hash(self, tmp_path):
        xyz = _write_xyz(tmp_path, n=6)
        return msj.sha256_of_file(xyz)

    def test_writes_and_reads_rule_tree(self, tmp_path):
        from molbuilder.selection import (
            ByElement, FirstN, to_json as rule_to_json,
        )
        rule = FirstN(ByElement(("Au",)), 4)
        payload = msj.to_dict(
            n_atoms_total=6,
            structure_hash=self._hash(tmp_path),
            regions={"L-electrode": [0, 1, 2, 3]},
            selection_rules={"L-electrode": rule_to_json(rule)},
        )
        side = tmp_path / "out.molstruct.json"
        msj.save(side, payload)
        loaded = msj.load(side)
        assert loaded["selection_rules"]["L-electrode"]["op"] == "first_n"
        assert loaded["selection_rules"]["L-electrode"]["n"] == 4

    def test_rule_for_unknown_target_raises(self, tmp_path):
        from molbuilder.selection import All, to_json as rule_to_json
        with pytest.raises(msj.MolstructJsonError, match="doesn't match"):
            msj.to_dict(
                n_atoms_total=6,
                structure_hash=self._hash(tmp_path),
                regions={"L-electrode": [0]},
                selection_rules={
                    # 'bridge' isn't in regions and isn't 'frozen_atoms':
                    "bridge": rule_to_json(All()),
                },
            )

    def test_malformed_rule_raises(self, tmp_path):
        with pytest.raises(msj.MolstructJsonError, match="invalid rule"):
            msj.to_dict(
                n_atoms_total=6,
                structure_hash=self._hash(tmp_path),
                regions={"L-electrode": [0]},
                selection_rules={"L-electrode": {"op": "not_a_real_op"}},
            )

    def test_rule_for_frozen_atoms_is_allowed(self, tmp_path):
        from molbuilder.selection import ByElement, to_json as rule_to_json
        d = msj.to_dict(
            n_atoms_total=6,
            structure_hash=self._hash(tmp_path),
            frozen_atoms=[0, 1],
            selection_rules={
                "frozen_atoms": rule_to_json(ByElement(("Au",))),
            },
        )
        assert "frozen_atoms" in d["selection_rules"]


class TestStructureHashInvariant:
    def test_hash_changes_when_xyz_changes(self, tmp_path):
        """A different XYZ file produces a different hash; this is the
        property the transport script generator relies on to detect
        stale sidecars."""
        a = tmp_path / "a.xyz"
        b = tmp_path / "b.xyz"
        a.write_text(_toy_xyz_text(4))
        # Different atom count -> different bytes -> different hash:
        b.write_text(_toy_xyz_text(5))
        assert msj.sha256_of_file(a) != msj.sha256_of_file(b)

    def test_hash_is_stable_for_same_bytes(self, tmp_path):
        a = tmp_path / "a.xyz"
        b = tmp_path / "b.xyz"
        text = _toy_xyz_text(4)
        a.write_text(text)
        b.write_text(text)
        assert msj.sha256_of_file(a) == msj.sha256_of_file(b)
