"""The two words the supervisor and the server it starts agree on.

A leaf on purpose: **it imports nothing, and it sits ABOVE ``web/``.**

The second half is not cosmetic.  It lived at ``web/reload_protocol.py`` for
about a minute, and importing it ran ``web/__init__.py`` — which does
``from .app import create_app``.  So reading a two-line protocol pulled in the
whole application and Flask with it, and the supervisor was importing the very
code it exists to survive.  Measured, not guessed: ``molbuilder.web.app`` was in
``sys.modules`` immediately afterwards.

The supervisor's whole value is that it NEVER IMPORTS APPLICATION CODE
(docs/archive/2026-08-19-server-reload-plan.md § 3.2) — that is what lets a child which fails to
import leave the parent alive, so the next reload can fix it.  A parent that had
to import the app to learn its own exit code would lose exactly the property it
exists for: a syntax error anywhere in the app would kill the supervisor too, and
the site would be down until someone reached the machine.

So both constants live here, where either side can read them at no cost and
neither has to reach into the other.

**And there is exactly one copy of each.**  Two copies of an exit code is a
reload that quietly stops respawning: the child asks with 3, the parent waits for
4, and the server simply exits — looking, from the browser, exactly like a crash.
"""
from __future__ import annotations

#: Exit code a child uses to ask for a fresh one.  Any OTHER code — Ctrl-C, a
#: crash, a clean stop — ends the supervisor too, so a broken build stops the
#: server instead of being respawned in a loop.
RELOAD_EXIT_CODE = 3

#: Set by the supervisor in the child's environment.  The reload ROUTE only
#: exists when this is present: with no supervisor there is nobody to bring the
#: server back, and an endpoint that stops it would leave a dead site with no way
#: back from the browser.
SUPERVISED_ENV = "MOLBUILDER_SUPERVISED"
