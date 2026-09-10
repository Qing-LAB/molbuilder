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
        """The sidecar's name is the structure's name with its suffix REPLACED by
        `.molstruct.json`.

        The pair is linked by filename and by nothing else (`structure-molstruct.md`
        section 6), so a namer that appended rather than replaced would write
        `relaxed.xyz.molstruct.json` -- which the loader never finds, and every region
        label and frozen-atom index on that structure is silently gone at the next
        load.
        """
        p = tmp_path / "relaxed.xyz"
        assert msj.sidecar_path_for(p).name == "relaxed.molstruct.json"

    def test_compound_suffix(self, tmp_path):
        """``job.spectra.xyz`` -> ``job.spectra.molstruct.json``.
        ``Path.with_suffix`` chaining would have eaten ``.spectra``."""
        p = tmp_path / "job.spectra.xyz"
        assert msj.sidecar_path_for(p).name == "job.spectra.molstruct.json"

    def test_no_suffix(self, tmp_path):
        """A structure file with no extension still gets `<name>.molstruct.json`.

        The suffix-replacement path has to handle "there is no suffix" without
        producing a name-less dotfile and without leaving the pair unmatched. Same
        pairing rule as `test_plain_xyz` (`structure-molstruct.md` § 6); this is
        the branch `Path.with_suffix` gets wrong in the opposite direction from the
        compound-suffix case beside it.
        """
        p = tmp_path / "bridge"
        assert msj.sidecar_path_for(p).name == "bridge.molstruct.json"


# --------------------------------------------------------------------- #
#  to_dict (canonical builder)                                          #
# --------------------------------------------------------------------- #


class TestToDict:
    def test_minimum_required_fields(self):
        """The canonical builder writes the schema version, the two integrity fields and
        an EMPTY label store -- and no second home for frozen atoms.

        `assert "frozen_atoms" not in d` is the load-bearing line: the reserved label
        lives inside `regions`, one store (`structure-annotations.md` § 2). A
        builder that also wrote a top-level key would produce files whose two copies
        can disagree -- the schema-3 shape `load` refuses outright
        (`structure-molstruct.md` § 2).
        """
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
        """Region indices are normalised to sorted, unique order at BUILD time.

        Sidecars are compared for equality (the round-trip and hash tests below) and
        walked in order by the engine emitters. Without normalisation two sidecars
        naming the same atoms in a different order are different files, and a
        duplicated index emits the same atom twice into a constraint block.
        `structure-molstruct.md` § 1.
        """
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
        """A label naming an atom the structure does not have is REFUSED at build time.

        `n_atoms_total` is the bound, and an index past it means the labels were built
        against a different structure. Caught here rather than at engine-load, where
        it would surface as a constraint on a nonexistent atom -- the same class of
        staleness `structure_hash` exists for (`structure-molstruct.md` § 3).
        """
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
        """An empty-string label is refused.

        `""` is what an unfilled name box produces. Accepted, it enters the one label
        store as a real channel (`structure-annotations.md` § 2), draws a
        nameless row in the filter panel, and rides into the script's ATOM-METADATA
        block (`job-contracts.md` § 3.4) as a region nothing can refer to.
        """
        with pytest.raises(msj.MolstructJsonError, match="non-empty"):
            msj.to_dict(
                {"regions": {"": [0]}},
                n_atoms_total=3, structure_hash="b" * 32,
            )

    def test_bad_n_atoms_raises(self):
        """A negative atom count is refused at build.

        `n_atoms_total` is one of the two integrity pins: `apply_to_structure`
        compares it against the structure it is applied to and refuses on mismatch
        (`structure-molstruct.md` § 3). A negative value can never match, so the
        pin would refuse every load instead of catching a real edit -- and a pin that
        fires always gets switched off.
        """
        with pytest.raises(msj.MolstructJsonError, match="non-negative"):
            msj.to_dict(n_atoms_total=-1, structure_hash="b" * 32)

    def test_bad_hash_raises(self):
        """`structure_hash` must look like a hash, refused at build if it does not.

        The sidecar's own reader deliberately does NOT verify the hash -- the caller
        compares it against the geometry it loaded (`structure-molstruct.md` section
        3). That makes the build-time format gate the only place a junk value is
        caught; written through, it would be compared later and never match, turning
        the staleness check into an unconditional refusal.
        """
        with pytest.raises(msj.MolstructJsonError, match="hex string"):
            msj.to_dict(n_atoms_total=3, structure_hash="short")


# --------------------------------------------------------------------- #
#  Save + load round-trip                                               #
# --------------------------------------------------------------------- #


class TestSaveLoadRoundTrip:
    def test_round_trip_identity(self, tmp_path):
        """Every field except the generated timestamp survives save then load.

        The sidecar carries the region labels and frozen atoms that XYZ cannot
        represent (`structure-molstruct.md` § 1), and there is no second copy to
        recover from: anything the codec drops on the way through is a label silently
        gone by the next load. `created_at` is excluded because it is generated per
        build rather than carried.
        """
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
        """A missing sidecar raises `MolstructJsonError`, not a bare `OSError`.

        Every caller of `load` catches this module's own error type; an `OSError`
        escaping it reaches a route as a 500 and the CLI as a traceback. The message
        names the file, which is what tells a person WHICH pair is broken when a
        project holds dozens.
        """
        with pytest.raises(msj.MolstructJsonError, match="failed to read"):
            msj.load(tmp_path / "missing.molstruct.json")

    def test_invalid_json(self, tmp_path):
        """Malformed JSON is reported as a sidecar error naming the file, not as a raw
        `JSONDecodeError`.

        Sidecars are hand-editable text, so a truncated write or a manual edit is the
        expected failure -- and the person needs to be told which file, which the
        underlying decode error does not say. `structure-molstruct.md` § 2,
        "what refused means at each surface".
        """
        p = tmp_path / "bad.molstruct.json"
        p.write_text("{not json,,,")
        with pytest.raises(msj.MolstructJsonError, match="not valid JSON"):
            msj.load(p)

    def test_top_level_must_be_object(self, tmp_path):
        """A JSON array (or any non-object) at the top level is refused at the door.

        The rest of the reader indexes the payload by key, so a list would raise a
        `TypeError` deep inside the field walk instead -- and the message would name
        a field rather than the real problem. `structure-molstruct.md` § 1 fixes
        the envelope as an object.
        """
        p = tmp_path / "bad.molstruct.json"
        p.write_text('[1, 2, 3]')
        with pytest.raises(msj.MolstructJsonError, match="must be an object"):
            msj.load(p)


    def test_missing_required_field(self, tmp_path):
        """A sidecar missing `structure_hash` is refused, and the message names the
        field.

        `structure_hash` and `n_atoms_total` are the two integrity pins
        (`structure-molstruct.md` § 3). A reader that tolerated an absent one
        would apply labels from a sidecar that cannot be checked against its
        structure -- precisely the stale-pair case the pins exist for.
        """
        p = tmp_path / "bad.molstruct.json"
        p.write_text(_json.dumps({
            "schema_version": 7,
            "n_atoms_total": 3,
            # structure_hash missing
        }))
        with pytest.raises(msj.MolstructJsonError, match="structure_hash"):
            msj.load(p)

    def test_load_propagates_error_with_path(self, tmp_path):
        """A validation failure inside `load` carries BOTH the file path and the
        underlying reason.

        The builder's own error says "region index out of range"; without the path
        wrapped around it, a person with a project full of sidecars is told a rule was
        broken and not where. The end product asserted is the raised error's message,
        which is the only place the two facts meet.
        """
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
        """Applying a sidecar puts the labels on the Structure AND makes `frozen_atoms`
        read back from that same one store.

        `frozen_atoms` is a reserved LABEL, not a second field
        (`structure-annotations.md` sections 2 and 5), and this asserts the two views
        agree after apply. A `regions` that carried the label while `s.frozen_atoms`
        stayed empty emits a deck with nothing held fixed -- the run relaxes atoms the
        person pinned.
        """
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
        """Applying a sidecar to a structure with a different atom count REFUSES.

        The integrity pin doing its job (`structure-molstruct.md` § 3): the
        indices in `regions` are positions in one specific atom list, so applying them
        to a structure that has since gained or lost atoms labels the WRONG atoms --
        silently, with every index still in range. The refusal says the labels "no
        longer point" at this structure.
        """
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
        """A `selection.py` rule tree survives the sidecar round-trip as JSON.

        `selection_rules` is a sidecar-ONLY pass-through (`structure-molstruct.md`
        section 4) -- not a `Structure` field, so nothing else in the load path would
        notice it being dropped or flattened. What is at stake is re-evaluating a
        saved selection ("the left electrode is the first 4 Au") against an edited
        structure; `op` and `n` are what `selection.from_json` needs to rebuild it.
        """
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
        """A rule aimed at a label that is not in the store is refused at BUILD time.

        A rule whose target does not exist can never be applied, so accepting it
        writes a sidecar describing a selection nothing will ever re-create --
        discovered, if at all, months later when someone tries to refresh the labels.
        `structure-molstruct.md` § 4: the rules are keyed by label, and the
        labels are the ones in `regions`.
        """
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
        """A malformed rule surfaces at sidecar-build time, not at engine-load time.

        `to_dict` re-parses each rule through `selection.from_json`, so
        `{"op": "not_a_real_op"}` is refused while the person is still looking at the
        panel that produced it. Without the re-parse the bad recipe is written to disk
        and raises when a generator later tries to re-evaluate it, far from the edit
        that caused it. `structure-molstruct.md` § 4.
        """
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
        """Identical bytes hash identically -- the half that makes the pin usable.

        Its sibling pins that DIFFERENT bytes differ; this one pins that the same
        structure written twice is not read as an edit. A hash that folded in the
        path, the mtime or a salt would make every sidecar look stale against its own
        structure (`structure-molstruct.md` § 3), and the check would be turned
        off rather than trusted.
        """
        a = tmp_path / "a.xyz"
        b = tmp_path / "b.xyz"
        text = _toy_xyz_text(4)
        a.write_text(text)
        b.write_text(text)
        assert msj.sha256_of_file(a) == msj.sha256_of_file(b)


class TestPbcFollowsAxisKind:
    """Review F3: pbc is the DERIVED view of axis_kind, not of cell-presence."""

    def test_isolated_axis_keeps_pbc_false_despite_cell(self):
        """`pbc` is DERIVED from `axis_kind`, never from the presence of a cell.

        Review F3's defect: a structure carrying a cell read back as periodic in all
        three directions, so a molecule given a vacuum box for convenience was
        computed as a 3-D crystal -- k-point sampling and periodic images along an
        axis that is physically isolated. `model/structure-periodicity.md` owns the
        rule; this is the sidecar half of it.
        """
        d = msj.to_dict({"cell": [[10, 0, 0], [0, 10, 0], [0, 0, 10]],
                         "axis_kind": ["periodic", "periodic", "isolated"]},
                        n_atoms_total=2, structure_hash="0" * 32)
        s = Structure.from_xyz("2\n\nH 0 0 0\nH 0 0 0.74\n")
        msj.apply_to_structure(s, d)
        assert tuple(s.axis_kind) == ("periodic", "periodic", "isolated")
        # pbc follows axis_kind -- NOT all-True from cell-presence (the F3 bug)
        assert tuple(bool(x) for x in s.pbc) == (True, True, False)

    def test_celled_without_axis_kind_or_pbc_is_periodic(self):
        """A cell with no `axis_kind` and no `pbc` still reads as periodic in all three
        directions.

        The other half of F3: making `pbc` follow `axis_kind` must not turn every
        sidecar written before `axis_kind` existed into an isolated molecule. A cell
        and nothing else is exactly what a genuine crystal's sidecar looks like, and
        reading it as non-periodic drops the k-point sampling from a calculation that
        needs it. `model/structure-periodicity.md`.
        """
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
        """The schema-9 `info` block survives a full `StructureCodec` write-then-load,
        nested structure and all.

        `info` is free-form and non-structural, so nothing in the load path validates
        it: a codec that flattened it, dropped a nested key, or wrote it lossily would
        lose the recorded calculation provenance with no error anywhere. Asserted
        through the shipped pair door rather than through `to_dict`.
        """
        from molbuilder.workingcopy_structure import StructureCodec
        s = self._struct()
        s.info = {"calculation": {"engine": "siesta",
                                  "contract": {"basis_size": "DZP"}}}
        StructureCodec().write(s, tmp_path / "j.xyz")
        back = StructureCodec().load(tmp_path / "j.xyz")
        assert back.info == s.info

    def test_an_empty_store_writes_no_key(self, tmp_path):
        """An empty `info` store writes NO `info` key at all.

        Absent means "nothing recorded"; `{}` would mean "recorded, and it was
        empty". Writing the empty object makes every new pair differ on disk from a
        schema-8 one for no reason, and makes "was anything recorded here?"
        unanswerable from the file.
        """
        from molbuilder.workingcopy_structure import StructureCodec
        StructureCodec().write(self._struct(), tmp_path / "k.xyz")
        raw = _json.loads((tmp_path / "k.molstruct.json").read_text())
        assert "info" not in raw, (
            "absent means 'nothing recorded' -- an empty store must "
            "leave the file shaped like a schema-8 one")

    def test_info_never_enters_the_hash(self, tmp_path):
        """Two structures with the same atoms and different `info` have the SAME
        `structure_hash`.

        `structure_hash` answers "is this sidecar still about this structure"
        (`structure-molstruct.md` § 3). `info` describes the structure rather
        than being part of it, so hashing it would make recording a note read as an
        edit -- every staleness check firing on a change that touched no atom.
        """
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
        """A pair whose sidecar no longer carries `info` loads with an EMPTY store, not
        with whatever was there before.

        Full-replace, not merge: apply is the sidecar's complete statement about the
        structure, so a v8 file -- which cannot carry `info` at all -- must not leave
        a stale block in place from a previous load. `structure-molstruct.md` section
        2's additive-bump rule is what makes reading a v8 file legal in the first
        place.
        """
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
        """A value JSON cannot carry is refused when the payload is BUILT, not when it is
        written.

        `info` is free-form, so a caller can put anything in it. Failing later, inside
        `save`, leaves the structure written and no sidecar beside it -- the unpaired
        state `structure-molstruct.md` § 6 exists to prevent -- plus a stray
        `.tmp` from the atomic-replace path. Refusing in `to_dict` means nothing has
        touched the disk yet.
        """
        with pytest.raises(msj.MolstructJsonError, match="JSON"):
            msj.to_dict({"regions": {}}, n_atoms_total=1,
                        structure_hash="b" * 32,
                        info={"bad": object()})
