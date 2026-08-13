"""``molbuilder bench`` CLI group — the one config helper that remains."""

from __future__ import annotations

import click


@click.group("bench",
             context_settings={"help_option_names": ["-h", "--help"]})
def bench_group() -> None:
    """Benchmark utilities.

    \b
    THE BENCHMARK ITSELF IS A JOBSET (the 2026-08-12 fold): a benchmark
    is a described calculation's sweep, run through the jobset verbs --
      molbuilder jobset prep bench <stage>      render the trial decks
      molbuilder jobset submit bench <stage>    launch trials, ONE BY ONE
      molbuilder jobset summarize bench <stage> read outputs -> verdict
    `molbuilder jobset --help` owns that workflow; its contract is
    docs/execution/job-contracts.md section 6.

    \b
    WHAT REMAINS HERE:
      probe-scheduler  read a live SLURM cluster (sinfo/sacctmgr) into a
                       proposed `scheduler` config block for molbuilder.json

    (The legacy in-place `siesta-gpu` sweep was deleted 2026-08-13 --
    obsolete paths do not survive beside the verb that replaced them.)
    """


# ``_GENERATE_EPILOG`` stood here until U19 (2026-08-12) -- the worked
# example for the deleted `generate` verb, orphaned when u5 deleted its
# command.  The live walkthrough is running-a-job.md's jobset flow.


# --------------------------------------------------------------------- #
#  `generate`, `prep`, `summarize` and `prep-run` were DELETED 2026-08-12 #
#  (plan step 6 u5) with the shipped-bundle lifecycle they drove; the     #
#  legacy in-place `siesta-gpu` sweep followed 2026-08-13 (user: obsolete #
#  paths do not survive beside the verb that replaced them).  The loop is #
#  `jobset describe -> prep bench -> submit bench <trial> -> summarize    #
#  bench -> prep run` (job-system.md § 5.3).  Only `probe-scheduler`, a   #
#  config helper feeding that loop's routing, remains.                    #
# --------------------------------------------------------------------- #

@bench_group.command("probe-scheduler",
                     short_help="probe sinfo/sacctmgr -> proposed scheduler "
                                "config block")
@click.option("--out", default=".",
              type=click.Path(file_okay=False, resolve_path=True),
              help="bundle dir whose .molbuilder.json to update with --write "
                   "(default: current dir).")
@click.option("--write", "do_write", is_flag=True, default=False,
              help="merge the proposed scheduler block into "
                   "<out>/.molbuilder.json (shows a diff + confirms).")
@click.option("--yes", is_flag=True, default=False,
              help="skip the confirmation prompt when --write.")
def cmd_probe_scheduler(out: str, do_write: bool, yes: bool) -> None:
    """Probe this SLURM cluster (sinfo/sacctmgr) and propose a `scheduler`
    config block -- partitions, GPU type, and the routing menu derived from
    the LIVE system (job-system.md § 7).  Run on the login node;
    every name/limit comes from the cluster, none is hardcoded.
    """
    import getpass
    import json
    from pathlib import Path

    from ..runtime_config import (RuntimeConfigError, get_scheduler,
                                   write_config_scope)
    from ..environment import _run
    from ..scheduler_probe import (derive_scheduler_block, parse_allowed_qos, parse_qos,
                        parse_sinfo)

    user = getpass.getuser()
    sinfo_txt = _run(["sinfo", "-h", "-o", "%P|%30l|%D|%40G"])
    if sinfo_txt is None:
        click.echo("ERROR: could not run sinfo -- run this on a SLURM login "
                   "node (sinfo/sacctmgr must be on PATH).", err=True)
        raise SystemExit(2)
    qos_txt = _run(["sacctmgr", "-nP", "show", "qos",
                    "format=Name,MaxWall,Flags"])
    assoc_txt = _run(["sacctmgr", "-nP", "show", "assoc", f"user={user}",
                      "format=QOS"])

    parts = parse_sinfo(sinfo_txt)
    qos = parse_qos(qos_txt or "")
    allowed = parse_allowed_qos(assoc_txt or "")
    block, notes = derive_scheduler_block(parts, qos, allowed)
    if block is None:
        click.echo("Could not derive a scheduler block:", err=True)
        for n in notes:
            click.echo(f"  - {n}", err=True)
        raise SystemExit(2)

    gpu_parts = [p.name for p in parts if p.has_gpu]
    click.echo(f"Probed (user={user}): GPU partitions {gpu_parts}; "
               f"allowed QoS: {', '.join(sorted(allowed)) or '(unknown)'}; "
               f"GPU type -> {block['gpu']['default_type']}")
    click.echo("\nRouting domains (pick at run: ./run-bench --domain <name>):")
    for d in block["routing"]:
        click.echo(f"  {d['name']:<10} <= {d['max_time']:<12} "
                   f"{d['partition']}/{d['qos']}")
    click.echo("\nProposed scheduler block:\n")
    click.echo(json.dumps({"scheduler": block}, indent=2))
    click.echo("\nNotes / assumptions (read before --write):")
    for n in notes:
        click.echo(f"  - {n}")

    if not do_write:
        click.echo(f"\n(dry run -- nothing written. Re-run with --write to "
                   f"merge into {Path(out) / '.molbuilder.json'}.)")
        return

    # --write: show a before/after of the key fields, then merge.
    try:
        before = get_scheduler(project_dir=Path(out)) or {}
    except RuntimeConfigError:
        before = {}
    old_names = [d.get("name") for d in (before.get("routing") or [])]
    new_names = [d["name"] for d in block["routing"]]
    click.echo(f"\nDIFF scheduler.routing: {old_names or '(none)'} -> "
               f"{new_names}")
    click.echo(f"DIFF directives: "
               f"{before.get('directives', {}).get('partition')}/"
               f"{before.get('directives', {}).get('qos')} -> "
               f"{block['directives']['partition']}/"
               f"{block['directives']['qos']}")
    if not yes:
        click.confirm(f"Merge this scheduler block into "
                      f"{Path(out) / '.molbuilder.json'}?", abort=True)
    write_config_scope(Path(out), {"scheduler": block})
    click.echo(f"wrote scheduler block to {Path(out) / '.molbuilder.json'} "
               "(execution / script_generation preserved).")


