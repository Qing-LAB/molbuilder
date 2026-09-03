"""P6 — the transport record (`engines/transport.md` § 6,
`transport/record.py` + `summarize run`'s transport arm + the
transmission walk).

The parse fixtures are FROZEN FROM A REAL RUN — the carbon-chain live
walk of 2026-08-29 (SIESTA/TBtrans 5.4.2): ``tests/data/
chain.TBT.AVTRANS_L-R`` is the equilibrium point's k-averaged
transmission verbatim, ``chain-tbtrans-v0.4.out`` the 0.4 V point's
output tail with the binary's own current line.  T(E) ≈ 2.0 there is
the textbook two-π-channel answer for a perfect cumulene chain, which
is what makes these fixtures also a physics pin.

Properties under guard, each named for its failure:

* the AVTRANS parse (grid + T), the current parse (Fortran floats,
  negative mantissa form included), G(E_F) by interpolation;
* `collect_record`: per-point walk, a not-yet-run point reads as
  PENDING (never a failure of the set), nothing-ran refuses naming
  what to launch;
* the record file + `summarize run`'s table;
* the transmission walk CONTINUES past a failed point, and a
  not-yet-run point reads as pending rather than broken — the two rules
  of § 6's walk callout: nothing is handed forward, so a bad point says
  nothing about the next, and `summarize` is a reader (the device
  chain's stop rule deliberately does not apply, because ITS points do
  chain a density).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from molbuilder.transport.record import (RecordError, collect_record,
                                         conductance_g0, iv_table_text,
                                         parse_avtrans, parse_current_a,
                                         write_record)
from test_transport_prep import (_conclude, _describe_transport,
                                 _junction_struct, _write_junction)

_DATA = Path(__file__).parent / "data"


class TestTheParsers:

    def test_avtrans_parses_the_real_file(self):
        e, t = parse_avtrans((_DATA / "chain.TBT.AVTRANS_L-R").read_text())
        assert len(e) == 400 and len(t) == 400
        assert e[0] == pytest.approx(-1.995)
        # the physics pin: a perfect cumulene chain carries two open
        # pi channels -- T(E_F) = 2
        assert conductance_g0(e, t) == pytest.approx(2.0, abs=0.01)

    def test_the_current_line_parses(self):
        amps = parse_current_a(
            (_DATA / "chain-tbtrans-v0.4.out").read_text())
        assert amps == pytest.approx(3.09835e-05)

    def test_fortran_negative_mantissa_parses(self):
        assert parse_current_a(
            "L -> R, V [V] / I [A]: 0.400000     V / -.619664E-05 A"
        ) == pytest.approx(-6.19664e-06)

    def test_no_current_line_is_none_not_a_crash(self):
        assert parse_current_a("nothing here") is None

    def test_conductance_needs_the_window_to_straddle_ef(self):
        assert conductance_g0([0.5, 1.0], [1.0, 1.0]) is None

    def test_garbage_refuses_by_name(self):
        with pytest.raises(RecordError):
            parse_avtrans("# only comments\n")


def _ran_transmission(calc, point, *, current_out=True):
    """A transmission point that has run: the frozen real outputs in
    its attempt."""
    att = _conclude(calc, "transmission", [], point=point)
    import shutil
    shutil.copy2(_DATA / "chain.TBT.AVTRANS_L-R",
                 att / "T.TBT.AVTRANS_L-R")
    if current_out:
        shutil.copy2(_DATA / "chain-tbtrans-v0.4.out",
                     att / "T_05_transmission-run0.out")
    return att


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path_factory):
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    # ...and the box is probed.  Moving HOME moves the machine scope
    # out from under conftest's own record, and prep refuses without
    # one (`running-a-job.md` § 3.1).
    from conftest import write_machine_record
    write_machine_record()
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))


@pytest.fixture
def calc(tmp_path):
    root = tmp_path / "projects"
    _write_junction(root, _junction_struct())
    dest = _describe_transport(root)
    from molbuilder.jobset.prep import prep_calculation
    # render the transmission stage so its v-dirs + decks exist
    prep_calculation(dest, "transmission")
    return dest


class TestTheRecord:

    def test_collects_points_and_writes_the_table(self, calc):
        from molbuilder.task import read_task
        _ran_transmission(calc, "v0")
        _ran_transmission(calc, "v0.2")
        task = read_task(calc / "task.json")
        rec = collect_record(calc, task)
        assert [p["bias_v"] for p in rec["points"]] == [0.0, 0.2]
        assert rec["points"][0]["conductance_g0"] == pytest.approx(
            2.0, abs=0.01)
        assert rec["iv"]["current_a"][1] == pytest.approx(3.09835e-05)
        assert rec["provenance"]["slot"]["citation"].endswith("run-0")
        assert rec["provenance"]["atom_permutation"] == \
            "atom-permutation.json"
        out = write_record(calc, rec)
        assert out.name == "T.transport.json"
        back = json.loads(out.read_text())
        assert back["schema"] == "molbuilder/transport-result@1"
        text = iv_table_text(rec)
        assert "1.9997" in text and "3.0983e-05" in text

    def test_a_point_not_yet_run_reads_as_pending(self, calc):
        from molbuilder.task import read_task
        _ran_transmission(calc, "v0")
        rec = collect_record(calc, read_task(calc / "task.json"))
        assert len(rec["points"]) == 1
        assert rec["pending"][0]["bias_v"] == 0.2
        assert "pending" in iv_table_text(rec)

    def test_nothing_ran_refuses_naming_the_launch(self, calc):
        from molbuilder.task import read_task
        with pytest.raises(RecordError) as e:
            collect_record(calc, read_task(calc / "task.json"))
        assert "launch run transmission" in str(e.value)

    def test_summarize_run_is_the_cli_door(self, calc, tmp_path,
                                            monkeypatch):
        from click.testing import CliRunner
        from molbuilder.jobset._cli import jobset_group
        from molbuilder.projects import PROJECTS_ROOT_ENV
        _ran_transmission(calc, "v0")
        _ran_transmission(calc, "v0.2")
        monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tmp_path / "projects"))
        r = CliRunner().invoke(jobset_group,
                               ["summarize", "run", "--bundle",
                                "J/transport/T"])
        assert r.exit_code == 0, r.output
        assert "T.transport.json" in r.output
        assert (calc / "T.transport.json").is_file()

    def test_a_single_point_record_reads_the_plain_layout(self, tmp_path):
        from molbuilder.jobset.prep import prep_calculation
        from molbuilder.task import read_task
        root = tmp_path / "projects"
        _write_junction(root, _junction_struct())
        dest = _describe_transport(root, bias=(0.0,))
        prep_calculation(dest, "transmission")
        att = _conclude(dest, "transmission", [])
        import shutil
        shutil.copy2(_DATA / "chain.TBT.AVTRANS_L-R",
                     att / "T.TBT.AVTRANS_L-R")
        rec = collect_record(dest, read_task(dest / "task.json"))
        assert [p["bias_v"] for p in rec["points"]] == [0.0]
        assert rec["points"][0]["attempt"] == "05_transmission/run-0"


class TestTheTransmissionWalk:

    def test_the_walk_continues_past_a_failed_point(self, calc,
                                                    tmp_path,
                                                    monkeypatch):
        """Independent points: unlike the device chain, a failed
        transmission point must not stop the rest -- and the exit code
        still reports it."""
        from click.testing import CliRunner
        from molbuilder.jobset._cli import jobset_group
        from molbuilder.jobset.model import JobSet
        from molbuilder.jobset.prep import prep_calculation
        from molbuilder.projects import PROJECTS_ROOT_ENV
        from molbuilder.jobset.submit import submit_transport_chain
        from molbuilder.task import read_task
        # upstream: concluded device points so the gather is satisfied
        for st in ("seed", "electrode_L", "electrode_R"):
            prep_calculation(calc, st)
        _conclude(calc, "seed", ["T.DM"])
        _conclude(calc, "electrode_L", ["T_L-electrode.TSHS"])
        _conclude(calc, "electrode_R", ["T_R-electrode.TSHS"])
        prep_calculation(calc, "device")
        _conclude(calc, "device", ["T.TS.HSX"], point="v0")
        _conclude(calc, "device", ["T.TS.HSX"], point="v0.2")
        monkeypatch.setenv(PROJECTS_ROOT_ENV, str(tmp_path / "projects"))
        r = CliRunner().invoke(jobset_group,
                               ["prep", "run", "transmission",
                                "--bundle", "J/transport/T"])
        assert r.exit_code == 0, r.output
        # stubs: v0 fails, v0.2 records that it still ran
        (calc / "05_transmission" / "v0" / "run-0"
         / "T_05_transmission.run.sh").write_text("#!/bin/bash\nexit 3\n")
        (calc / "05_transmission" / "v0.2" / "run-0"
         / "T_05_transmission.run.sh").write_text(
            "#!/bin/bash\ntouch RAN\n")
        js = JobSet.load(calc / "job-set.json")
        task = read_task(calc / "task.json")
        results = submit_transport_chain(js, calc, task, mode="direct",
                                         stage="transmission")
        assert results[0].status == "failed", (
            "the exit code must still report the failed point")
        assert (calc / "05_transmission" / "v0.2" / "run-0" / "RAN"
                ).is_file(), (
            "independent points: the walk continues past a failure")
