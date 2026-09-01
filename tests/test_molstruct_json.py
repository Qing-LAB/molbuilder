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

from molbuilder.sidecars import molstruct as msj
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
        assert "frozen_atoms" not in d      # one label store, one key
        # Auto-generated timestamp + default created_by:
        assert d["created_by"] == "molbuilder"
        assert d["created_at"].endswith("Z")

    def test_regions_sorted_and_deduped(self):
        d = msj.to_dict(
            {"regions": {"L-electrode": [3, 1, 1, 0]}},
            n_atoms_total=10, structure_hash="b" * 32,
        )
        assert d["regions"]["L-electrode"] == [0, 1, 3]

    def test_the_reserved_label_is_sorted_and_deduped_like_any_other(self):
        """It goes through the same normalisation as `L-electrode` above,
        because it goes through the same store."""
        d = msj.to_dict(
            {"regions": {"frozen_atoms": [5, 1, 1, 0]}},
            n_atoms_total=10, structure_hash="b" * 32,
        )
        assert d["regions"]["frozen_atoms"] == [0, 1, 5]
        assert msj.frozen_atoms(d) == [0, 1, 5]

    def test_region_out_of_range_raises(self):
        with pytest.raises(msj.MolstructJsonError, match="out of range"):
            msj.to_dict(
                {"regions": {"L-electrode": [5]}},
                n_atoms_total=3, structure_hash="b" * 32,
            )

    def test_region_overlap_is_allowed(self):
        """Multi-label model: an atom may carry several region tags
        at once.  Engines that need a disjoint partition validate
        that separately at engine-load time."""
        d = msj.to_dict(
            {"regions": {"a": [0, 1], "b": [1, 2]}},
            n_atoms_total=4, structure_hash="b" * 32,
        )
        assert d["regions"]["a"] == [0, 1]
        assert d["regions"]["b"] == [1, 2]

    def test_empty_label_raises(self):
        with pytest.raises(msj.MolstructJsonError, match="non-empty"):
            msj.to_dict(
                {"regions": {"": [0]}},
                n_atoms_total=3, structure_hash="b" * 32,
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
            {"regions": {"L-electrode": [0, 1], "R-electrode": [4, 5],
                         "bridge": [2, 3], "frozen_atoms": [0, 5]}},
            n_atoms_total=6,
            structure_hash=msj.sha256_of_file(xyz),
        )
        side = msj.sidecar_path_for(xyz)
        msj.save(side, payload)
        loaded = msj.load(side)
        # Everything except created_at (timestamp) should match:
        for key in ("schema_version", "n_atoms_total", "structure_hash",
                    "regions", "created_by"):
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

    def test_save_emits_utf8_bytes_for_unicode_region_label(self, tmp_path):
        """Regression for 2026-06-09 audit BLOCKER: ``save`` previously
        called ``open(tmp, "w")`` without ``encoding="utf-8"``, so a
        region label with non-ASCII characters mojibaked on systems with
        a non-UTF-8 platform locale (cp1252, latin-1).  Pin that the
        bytes on disk decode as UTF-8 regardless of the runner's locale.
        """
        xyz = _write_xyz(tmp_path, n=4)
        payload = msj.to_dict(
            {"regions": {"α-helix": [0, 1], "β-sheet": [2, 3]}},
            n_atoms_total=4,
            structure_hash=msj.sha256_of_file(xyz),
        )
        side = msj.sidecar_path_for(xyz)
        msj.save(side, payload)
        # Read the raw bytes and decode as UTF-8 — must succeed and
        # round-trip the non-ASCII labels verbatim.
        raw = side.read_bytes()
        text = raw.decode("utf-8")
        assert "α-helix" in text
        assert "β-sheet" in text
        # And the load() path must also see them.
        loaded = msj.load(side)
        assert "α-helix" in loaded["regions"]
        assert "β-sheet" in loaded["regions"]

    def test_load_accepts_utf8_bom(self, tmp_path):
        """Regression for the same audit: ``load`` used to call
        ``read_text()`` without ``encoding="utf-8-sig"``, so a file
        with a BOM (some Windows editors inject one) read its first
        bytes as garbage characters.  Pin BOM tolerance.
        """
        xyz = _write_xyz(tmp_path, n=4)
        side = msj.sidecar_path_for(xyz)
        payload = msj.to_dict(
            {"regions": {"L-electrode": [0, 1]}},
            n_atoms_total=4,
            structure_hash=msj.sha256_of_file(xyz),
        )
        # Write the payload with a UTF-8 BOM prepended — simulates a
        # sidecar that was hand-edited in an editor that always emits
        # a BOM.
        import json as _json
        bom = "﻿"
        side.write_text(bom + _json.dumps(payload), encoding="utf-8")
        loaded = msj.load(side)
        assert loaded["regions"] == {"L-electrode": [0, 1]}


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
                {"regions": regions},
                n_atoms_total=4,
                structure_hash="0" * 64,
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
            "schema_version": 7,
            "n_atoms_total": 3,
            # structure_hash missing
        }))
        with pytest.raises(msj.MolstructJsonError, match="structure_hash"):
            msj.load(p)

    def test_load_propagates_error_with_path(self, tmp_path):
        p = tmp_path / "bad.molstruct.json"
        p.write_text(_json.dumps({
            "schema_version": 7,
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
            {"regions": {"L-electrode": [0, 1], "R-electrode": [3],
                         "frozen_atoms": [0]}},
            n_atoms_total=4, structure_hash="b" * 32,
        )
        msj.apply_to_structure(s, data)
        assert s.regions == {"L-electrode": [0, 1], "R-electrode": [3],
                             "frozen_atoms": [0]}
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
    """``READABLE_VERSIONS`` is the single source of truth for which on-disk
    schemas this build accepts: {7, 8, 9} since 2026-08-29.  v8 only ADDED
    the optional identity columns and v9 only added the optional `info`
    block, so a v7 or v8 file reads whole (an absent addition IS its
    default, which is exactly what the older version meant); everything
    older stores the same facts in DIFFERENT places (v3's top-level frozen
    atoms) and is refused rather than silently coerced."""

    @pytest.mark.parametrize("version, why", [
        (2, "v2 kept a `fixed_atoms` key"),
        (3, "v3 kept frozen atoms in a top-level key, not in `regions`"),
        (6, "v6 was the last before the label store was unified"),
        (99, "a version from the future"),
        (None, "no version at all"),
    ])
    def test_any_version_but_the_current_one_is_refused(self, tmp_path,
                                                        version, why):
        """One gate, one answer: this build reads the current schema and refuses
        everything else, whether it is older or newer.

        RETIRED (2026-07-31) the separate `test_v2_fails_to_load` /
        `test_unknown_version_raises` pair. While several versions were readable
        those described different code paths; under a strict gate they are one
        refusal, and keeping two tests of it implied a distinction that no longer
        exists. What each version MEANT is kept above, as the reason it is here.
        """
        payload = {
            "n_atoms_total":  3,
            "structure_hash": "b" * 32,
            "regions":        {"L-electrode": [0]},
        }
        if version is not None:
            payload["schema_version"] = version
        p = tmp_path / "old.molstruct.json"
        p.write_text(_json.dumps(payload))
        with pytest.raises(msj.MolstructJsonError,
                           match=r"reads versions \[7, 8, 9\] only"):
            msj.load(p)

    def test_a_v7_file_reads_whole_under_v8(self, tmp_path):
        """The additive-bump rule (2026-08-20): v8 added only the OPTIONAL
        identity columns, so every fact a v7 file states lands in the same
        place -- and the absent identity means the synthesized defaults,
        which is what v7 always meant.  This is the case that makes the
        readable SET a set; a strict single-version gate here would have
        refused every pair on disk the day the schema learned a new word."""
        payload = {
            "schema_version": 7,
            "n_atoms_total":  3,
            "structure_hash": "b" * 32,
            "regions":        {"L-electrode": [0]},
        }
        p = tmp_path / "old.molstruct.json"
        p.write_text(_json.dumps(payload))
        loaded = msj.load(p)
        assert loaded["regions"] == {"L-electrode": [0]}
        assert "atom_names" not in loaded

    def test_writes_the_reserved_label_into_the_one_label_store(self, tmp_path):
        """Canonical write (schema 7): the reserved label is a label. Neither
        the schema-6 top-level `frozen_atoms` key nor the older `fixed_atoms`
        one is written -- both were second homes for the same fact."""
        d = msj.to_dict(
            {"regions": {"L-electrode": [0], "frozen_atoms": [1, 2]}},
            n_atoms_total=3, structure_hash="b" * 32,
        )
        assert d["schema_version"] == msj.SCHEMA_VERSION
        assert d["regions"] == {"L-electrode": [0], "frozen_atoms": [1, 2]}
        assert "frozen_atoms" not in d
        assert "fixed_atoms" not in d


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
            {"regions": {"L-electrode": [0, 1, 2, 3]}},
            n_atoms_total=6,
            structure_hash=self._hash(tmp_path),
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
                {"regions": {"L-electrode": [0]}},
                n_atoms_total=6,
                structure_hash=self._hash(tmp_path),
                selection_rules={
                    # 'bridge' isn't in regions and isn't 'frozen_atoms':
                    "bridge": rule_to_json(All()),
                },
            )

    def test_malformed_rule_raises(self, tmp_path):
        with pytest.raises(msj.MolstructJsonError, match="invalid rule"):
            msj.to_dict(
                {"regions": {"L-electrode": [0]}},
                n_atoms_total=6,
                structure_hash=self._hash(tmp_path),
                selection_rules={"L-electrode": {"op": "not_a_real_op"}},
            )

    def test_rule_for_the_reserved_label_is_allowed(self, tmp_path):
        """It needs no clause of its own: the rule targets a label, and the
        reserved one is a label in the same store as the rest."""
        from molbuilder.selection import ByElement, to_json as rule_to_json
        d = msj.to_dict(
            {"regions": {"frozen_atoms": [0, 1]}},
            n_atoms_total=6,
            structure_hash=self._hash(tmp_path),
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


class TestPbcFollowsAxisKind:
    """Review F3: pbc is the DERIVED view of axis_kind, not of cell-presence."""

    def test_isolated_axis_keeps_pbc_false_despite_cell(self):
        d = msj.to_dict({"cell": [[10, 0, 0], [0, 10, 0], [0, 0, 10]],
                         "axis_kind": ["periodic", "periodic", "isolated"]},
                        n_atoms_total=2, structure_hash="0" * 32)
        s = Structure.from_xyz("2\n\nH 0 0 0\nH 0 0 0.74\n")
        msj.apply_to_structure(s, d)
        assert tuple(s.axis_kind) == ("periodic", "periodic", "isolated")
        # pbc follows axis_kind -- NOT all-True from cell-presence (the F3 bug)
        assert tuple(bool(x) for x in s.pbc) == (True, True, False)

    def test_celled_without_axis_kind_or_pbc_is_periodic(self):
        d = msj.to_dict({"cell": [[10, 0, 0], [0, 10, 0], [0, 0, 10]]},
                        n_atoms_total=2, structure_hash="0" * 32)
        s = Structure.from_xyz("2\n\nH 0 0 0\nH 0 0 0.74\n")
        msj.apply_to_structure(s, d)
        assert tuple(s.axis_kind) == ("periodic", "periodic", "periodic")
        assert tuple(bool(x) for x in s.pbc) == (True, True, True)


class TestInfoBlock:
    """Schema 9's `info` block (archive/2026-09-01-structure-info-plan.md): free-form,
    NON-structural, additive, never hashed."""

    def _struct(self):
        import numpy as np
        from molbuilder.structure import Structure
        return Structure(elements=["C", "C"],
                         positions=np.array([[0.0, 0, 0], [0, 0, 1.3]]),
                         regions={"L-electrode": [0]},
                         cell=np.eye(3) * 10)

    def test_info_rides_the_pair_whole(self, tmp_path):
        from molbuilder.workingcopy_structure import StructureCodec
        s = self._struct()
        s.info = {"calculation": {"engine": "siesta",
                                  "contract": {"basis_size": "DZP"}}}
        StructureCodec().write(s, tmp_path / "j.xyz")
        back = StructureCodec().load(tmp_path / "j.xyz")
        assert back.info == s.info

    def test_an_empty_store_writes_no_key(self, tmp_path):
        from molbuilder.workingcopy_structure import StructureCodec
        StructureCodec().write(self._struct(), tmp_path / "k.xyz")
        raw = _json.loads((tmp_path / "k.molstruct.json").read_text())
        assert "info" not in raw, (
            "absent means 'nothing recorded' -- an empty store must "
            "leave the file shaped like a schema-8 one")

    def test_info_never_enters_the_hash(self, tmp_path):
        from molbuilder.workingcopy_structure import StructureCodec
        a, b = self._struct(), self._struct()
        b.info = {"note": "recording MORE about the same atoms"}
        StructureCodec().write(a, tmp_path / "a.xyz")
        StructureCodec().write(b, tmp_path / "b.xyz")
        ha = _json.loads((tmp_path / "a.molstruct.json").read_text())
        hb = _json.loads((tmp_path / "b.molstruct.json").read_text())
        assert ha["structure_hash"] == hb["structure_hash"], (
            "info DESCRIBES the structure; it must not read as a "
            "different one")

    def test_a_v8_file_loads_with_an_empty_store(self, tmp_path):
        from molbuilder.workingcopy_structure import StructureCodec
        s = self._struct()
        s.info = {"stale": True}
        StructureCodec().write(s, tmp_path / "v8.xyz")
        raw = _json.loads((tmp_path / "v8.molstruct.json").read_text())
        del raw["info"]
        raw["schema_version"] = 8
        (tmp_path / "v8.molstruct.json").write_text(_json.dumps(raw))
        back = StructureCodec().load(tmp_path / "v8.xyz")
        assert back.info == {}, (
            "FULL-REPLACE: a pair that no longer carries a store must "
            "not keep serving a stale one")

    def test_an_unserialisable_value_refuses_at_write(self):
        with pytest.raises(msj.MolstructJsonError, match="JSON"):
            msj.to_dict({"regions": {}}, n_atoms_total=1,
                        structure_hash="b" * 32,
                        info={"bad": object()})
