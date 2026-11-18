"""Common-name / IUPAC-name -> 3-D Structure via PubChem + RDKit.

Looks the name up in PubChem (over the network), pulls the absolute
SMILES -- stereochemistry and isotopes included -- and then defers to
:func:`molbuilder.smiles.build_from_smiles`.  A failure is the request's
(:class:`chemistry.BuildRefused`) or PubChem's, said as advice
(:class:`chemistry.BackendUnavailable`) -- `engines/builders.md` § 6.

Requires both ``pubchempy`` and ``rdkit``.
"""

from __future__ import annotations

from typing import Optional

from .smiles import build_from_smiles
from .structure import Structure


def smiles_for_name(name: str, *, timeout: float = 30.0) -> str:
    """Look up the ABSOLUTE SMILES for a name in PubChem -- its
    stereochemistry and isotopes included (pubchempy's ``smiles``), never
    the connectivity-only one, which would build "D-alanine" as whichever
    isomer the embedder lands on (`engines/builders.md` § 6).

    Each failure said once, by whose it is:

    * no name, a name PubChem knows nothing by, a name it refuses, a record
      holding no SMILES -- the request's: :class:`chemistry.BuildRefused`;
    * PubChem not answering from here -- no network, the lookup out of
      ``timeout`` seconds, PubChem busy or failing, an answer that does not
      read -- somewhere else's: :class:`chemistry.BackendUnavailable`
      (``missing="pubchem"``), advice and never a fault of ours, each in
      words of its own.

    PubChemPy reads PubChem through ``urllib`` with no per-call timeout, so
    the global socket timeout is set for the lookup and restored after.
    """
    import socket
    from http.client import HTTPException
    from urllib.error import URLError

    from .chemistry import BackendUnavailable, BuildRefused, require
    if not name.strip():
        raise BuildRefused("Nothing to look up: the name is empty.")
    pcp = require("pubchempy", needed_for="Building by name")

    def unreachable(why: str) -> BackendUnavailable:
        return BackendUnavailable(
            f"PubChem lookup for {name!r} {why}.  Try again, or build it "
            f"from a SMILES instead.", missing="pubchem")

    prev_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout)
    try:
        try:
            compounds = pcp.get_compounds(name, "name")
            smi = compounds[0].smiles if compounds else None
        except pcp.BadRequestError as exc:
            raise BuildRefused(
                f"PubChem refused the name {name!r}: {exc}") from exc
        except (socket.timeout, pcp.TimeoutError) as exc:
            raise unreachable(f"did not complete within {timeout} s") from exc
        except URLError as exc:
            if isinstance(exc.reason, socket.timeout):
                raise unreachable(
                    f"did not complete within {timeout} s") from exc
            raise unreachable(
                f"could not reach PubChem ({exc.reason})") from exc
        except OSError as exc:
            raise unreachable(f"could not reach PubChem ({exc})") from exc
        except pcp.PubChemPyError as exc:
            raise unreachable(
                f"was answered by PubChem with an error ({exc})") from exc
        except (ValueError, KeyError, HTTPException) as exc:
            # An answer that does not read -- a proxy's or a network login's
            # page in place of PubChem's JSON, a record without its
            # compounds or its properties, a reply cut short.
            raise unreachable(
                f"was answered with something that does not read "
                f"({type(exc).__name__}: {exc})") from exc
    finally:
        socket.setdefaulttimeout(prev_timeout)
    if not compounds:
        raise BuildRefused(f"PubChem returned no compounds for name {name!r}")
    if not smi:
        raise BuildRefused(f"PubChem entry for {name!r} has no SMILES")
    return smi


def build_from_name(
    name: str,
    *,
    title: Optional[str] = None,
    **smiles_kwargs,
) -> Structure:
    """Build a 3-D molecule from a PubChem-resolvable name.

    Parameters
    ----------
    name
        Common or IUPAC name, e.g. ``"benzene"``,
        ``"1,4-benzenedithiol"``, ``"caffeine"``.
    title
        Optional title.  Defaults to the name itself.
    **smiles_kwargs
        Forwarded to :func:`build_from_smiles` (e.g. ``optimize=False``).

    Returns
    -------
    Structure
        All-atom 3-D structure.
    """
    smiles = smiles_for_name(name)
    return build_from_smiles(
        smiles,
        title=title or name,
        **smiles_kwargs,
    )
