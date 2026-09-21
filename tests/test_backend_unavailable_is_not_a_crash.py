"""A backend the person did not install is not a server fault.

`web-api.md` § 1 defines the four buckets, and 5xx is *"an I/O error, an
engine that fell over, a bug"*.  A missing optional backend is none of
those: the request was well-formed and molbuilder is refusing it, which is
the contract's ADVISORY case -- 200 with ``ok: false``.

It answered **500** until 2026-09-20, for both shapes below, because
`BackendUnavailable` had no handler anywhere in the repo and fell into the
route's generic ``except Exception``.  So asking for a duplex without X3DNA
reported as a crash, naming a tool the person may never have heard of.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

import molbuilder.builders.backends._threedna as _threedna


@pytest.fixture
def no_threedna():
    """3DNA absent, however it is looked for."""
    with patch.object(_threedna, "_resolve", return_value=None), \
         patch.object(_threedna, "is_available", return_value=False):
        yield


@pytest.mark.parametrize("body,backend", [
    ({"kind": "dna", "input": "ATGC", "backend": "threedna"}, "threedna"),
    ({"kind": "dna", "input": "ds,ATGCATGC"},                 "auto"),
])
def test_a_missing_backend_answers_advisory_not_server_fault(
        web_client, no_threedna, body, backend):
    r = web_client.post("/api/build/molecule", json=body)

    assert r.status_code == 200, (
        f"a backend the person did not install answered HTTP {r.status_code}; "
        f"`web-api.md` § 1 reserves 5xx for a fault on our side")
    j = r.get_json()
    assert j["ok"] is False
    # MACHINE-READABLE, so the page can disable what this box cannot do
    # rather than letting the person find out by pressing the button.
    assert j["reason"] == "backend_unavailable"
    assert j["backend"] == backend
    # And the message still earns its place (`design.md`): it names the
    # tool, where to get it, and what still works without it.
    said = j["error"]
    assert "X3DNA" in said or "3DNA" in said
    assert "x3dna.org" in said


def test_a_build_that_can_succeed_still_does(web_client, no_threedna):
    """The guard must not turn a working fallback into a refusal."""
    r = web_client.post("/api/build/molecule",
                        json={"kind": "dna", "input": "ATGCATGC"})
    assert r.status_code == 200
    assert r.get_json()["ok"] is True
