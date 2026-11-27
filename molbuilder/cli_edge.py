"""THE COMMAND LINE'S EDGE -- where a file molbuilder keeps that its reader
cannot read is said, once, for every command (`execution/job-contracts.md`
§ 6.1).

MODULE  cli_edge (floor 7, a surface's; click and `persist`)
ROLE    the one `click.Group` the command line is made of -- `cli`
        (`cli.py`); every group added to it, `jobset` and `pseudo` among
        them, is invoked inside its edge -- whose invoke says a refused file
        in its reader's words
USED-BY cli
"""
from __future__ import annotations

import click

from .persist import FileRefused


class GroupSayingFileRefusals(click.Group):
    """A command group whose every command, a group's beneath it included,
    says a file it cannot read as a refusal: a :class:`persist.FileRefused`
    becomes ``click.ClickException(<the words>)`` -- ``Error: <the words>``,
    exit 1, under `cli.main` and under Click's own runner alike.  So no
    command dies of a file it cannot read in a Python traceback, a command
    added later included: the mark is the reader's, said here once rather
    than caught door by door."""

    def invoke(self, ctx: click.Context):
        try:
            return super().invoke(ctx)
        except FileRefused as exc:
            raise click.ClickException(str(exc)) from None


__all__ = ["GroupSayingFileRefusals"]
