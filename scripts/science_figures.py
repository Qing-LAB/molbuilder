"""THE FIGURES OF `docs/science/vibrational-averaging.md` § 2d -- each drawn
from the science's closed forms with numpy and scipy (the Hermite functions,
Mehler's sum, the coherent state's phase average, the one-quantum
distribution function), never from molbuilder's own numbers -- but for the
one figure that sets molbuilder's draws (`vibration.State.draw`) beside the
density they must follow.

Run from the repository root with the molbuilder env's Python:

    python scripts/science_figures.py

It writes `docs/img/vibrational-averaging/*.svg`, each white-backed so it
reads on the Documents tab's light and dark pages alike.
"""
from __future__ import annotations

import math
import sys
from fractions import Fraction
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import special, stats  # noqa: E402

_ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
OUT = Path(_ARGS[0]) if _ARGS else Path("docs/img/vibrational-averaging")

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 10,
    "legend.fontsize": 8.5, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "svg.fonttype": "none",
})
INK = "#1f2933"
C = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2",
     "#be185d"]


def psi2(n: int, x):
    """|psi_n|^2 in x = Q / Q_zp (unit spread at n = 0): the Hermite
    function's square, H_n(xi)^2 e^-xi^2 / (2^n n! sqrt(pi)), xi = x/sqrt2,
    times dxi/dx = 1/sqrt2."""
    xi = np.asarray(x, dtype=float) / math.sqrt(2.0)
    h = special.eval_hermite(n, xi)
    return h * h * np.exp(-xi * xi) / (
        2.0 ** n * math.factorial(n) * math.sqrt(math.pi)) / math.sqrt(2.0)


def gauss(x, s):
    return np.exp(-0.5 * (x / s) ** 2) / (math.sqrt(2.0 * math.pi) * s)


def thermal_p(n: int, nbar: float) -> float:
    return nbar ** n / (nbar + 1.0) ** (n + 1)


def coherent(x, ns: float):
    """The coherent state over its phase: N(Q0 cos phi, 1) averaged over a
    uniform phase, Q0 = 2 sqrt(n_s), in x = Q / Q_zp."""
    q0 = 2.0 * math.sqrt(ns)
    phi = (np.arange(4000) + 0.5) * (2.0 * math.pi / 4000)
    centres = q0 * np.cos(phi)
    return gauss(np.asarray(x)[:, None] - centres[None, :], 1.0).mean(axis=1)


def photon_added(x, nbar: float, nmax: int = 60):
    """One quantum added to a thermal state of nbar: p'_n = n nbar^(n-1) /
    (nbar+1)^(n+1)."""
    out = np.zeros_like(np.asarray(x, dtype=float))
    for n in range(1, nmax):
        out += n * nbar ** (n - 1) / (nbar + 1.0) ** (n + 1) * psi2(n, x)
    return out


def F1(x):
    """The one-quantum distribution function in x: Phi(x) - x phi(x)."""
    x = np.asarray(x, dtype=float)
    return stats.norm.cdf(x) - x * stats.norm.pdf(x)


def F1_inverse(u):
    from scipy import optimize
    return np.array([optimize.brentq(lambda x: F1(x) - v, -12, 12)
                     for v in np.atleast_1d(u)])


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, format="svg", bbox_inches="tight", dpi=150)
    if "--png" in sys.argv:                  # a look at it, for its author
        fig.savefig(OUT / name.replace(".svg", ".png"), dpi=110,
                    bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------- #
def fig_states():
    x = np.linspace(-6, 6, 1201)
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.plot(x, gauss(x, 1.0), color=C[0], lw=2,
            label=r"thermal at 0 K: ground state, $\sigma = Q_{\rm zp}$")
    ax.plot(x, gauss(x, math.sqrt(3.0)), color=C[0], lw=1.6, ls="--",
            label=r"thermal, $\bar n = 1$: $\sigma = \sqrt{3}\,Q_{\rm zp}$")
    ax.plot(x, psi2(1, x), color=C[1], lw=2,
            label=r"one quantum: $|\psi_1|^2$ (one photon, cold mode)")
    ax.plot(x, photon_added(x, 1.0), color=C[3], lw=1.6, ls="-.",
            label=r"one photon on a warm mode ($\bar n = 1$): photon-added thermal")
    ax.plot(x, coherent(x, 2.0), color=C[2], lw=1.6, ls=":",
            label=r"driven: coherent state, $n_s = 2$")
    ax.set_xlabel(r"$Q\,/\,Q_{\rm zp}$")
    ax.set_ylabel(r"$P(Q)\;\times Q_{\rm zp}$")
    ax.set_title("Where one mode sits, in each state")
    # THE LEGEND BESIDE THE AXES, never over the curves it names.
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 0.42)
    save(fig, "states-of-one-mode.svg")


def fig_mehler():
    x = np.linspace(-6, 6, 301)
    nbar = 1.0
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    total = np.zeros_like(x)
    for n in range(0, 7):
        part = thermal_p(n, nbar) * psi2(n, x)
        ax.fill_between(x, total, total + part, color=C[n % len(C)],
                        alpha=0.35, lw=0,
                        label=fr"$p_{n}\,|\psi_{n}|^2$, $p_{n}="
                              fr"{Fraction(thermal_p(n, nbar)).limit_denominator(4096)}$")
        total = total + part
    ax.plot(x, gauss(x, math.sqrt(2 * nbar + 1)), color=INK, lw=1.8,
            label=r"Gaussian, $\sigma^2 = (2\bar n+1)\,Q_{\rm zp}^2$")
    ax.set_xlabel(r"$Q\,/\,Q_{\rm zp}$")
    ax.set_ylabel(r"$P(Q)\;\times Q_{\rm zp}$")
    ax.set_title(r"A thermal mode ($\bar n = 1$): its number states, "
                 r"weighted by $p_n$, add up to a Gaussian (Mehler)",
                 loc="left")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False,
              fontsize=7.8)
    ax.set_xlim(-6, 6)
    save(fig, "thermal-is-a-sum-of-number-states.svg")


def fig_inverse_transform():
    zs = (0.31, -1.20, -0.05)
    z = np.linspace(-3.2, 3.2, 641)
    x = np.linspace(-4.2, 4.2, 841)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 3.2),
                               gridspec_kw={"wspace": 0.32})
    a.plot(z, stats.norm.cdf(z), color=C[0], lw=2)
    a.set_xlabel(r"a normal number $z$")
    a.set_ylabel(r"$u = \Phi(z)$")
    a.set_title(r"1. a normal number $\to$ a uniform $u$")
    b.plot(x, F1(x), color=C[1], lw=2)
    b.set_xlabel(r"$Q\,/\,Q_{\rm zp}$")
    b.set_ylabel(r"$F_1(Q) = \Phi(x) - x\,\phi(x)$")
    b.set_title(r"2. $u \to Q = F_1^{-1}(u)$, one quantum")
    for k, zk in enumerate(zs):
        u = stats.norm.cdf(zk)
        q = F1_inverse(u)[0]
        col = C[2 + k]
        a.plot([zk, zk], [0, u], color=col, lw=1, ls="--")
        a.plot([zk, 3.2], [u, u], color=col, lw=1, ls="--")
        a.plot(zk, u, "o", color=col, ms=4)
        b.plot([-4.2, q], [u, u], color=col, lw=1, ls="--")
        b.plot([q, q], [u, 0], color=col, lw=1, ls="--")
        b.plot(q, u, "o", color=col, ms=4)
        b.annotate(f"z = {zk:+.2f} → Q = {q:+.2f} Q$_{{zp}}$", (q, u),
                   textcoords="offset points",
                   xytext=((8, -12), (8, 4), (10, -16))[k], fontsize=7.5,
                   color=col)
    for ax in (a, b):
        ax.set_ylim(0, 1)
    save(fig, "one-quantum-inverse-transform.svg")


def fig_draws():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from molbuilder.vibration import state
    q_zp = 1.0
    one = state(q_zp, 1600.0, quanta=1).draw(40000, seed=11)
    warm = state(q_zp, 1600.0, mean_quanta=1.0).draw(40000, seed=11)
    x = np.linspace(-5.5, 5.5, 1101)
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 3.0),
                               gridspec_kw={"wspace": 0.25}, sharey=True)
    bins = np.linspace(-5.5, 5.5, 89)
    a.hist(warm, bins=bins, density=True, color=C[0], alpha=0.35,
           histtype="stepfilled", label="40 000 draws")
    a.plot(x, gauss(x, math.sqrt(3.0)), color=C[0], lw=2,
           label=r"Gaussian, $\sigma = \sqrt{3}\,Q_{\rm zp}$")
    a.set_title(r"thermal, $\bar n = 1$: $Q = \sigma z$")
    b.hist(one, bins=bins, density=True, color=C[1], alpha=0.35,
           histtype="stepfilled", label="40 000 draws")
    b.plot(x, psi2(1, x), color=C[1], lw=2, label=r"$|\psi_1|^2$")
    b.set_title(r"one quantum: $Q = F_1^{-1}(\Phi(z))$")
    for ax in (a, b):
        ax.set_xlabel(r"$Q\,/\,Q_{\rm zp}$")
        ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=8)
        ax.set_ylim(0, 0.40)
    a.set_ylabel(r"$P(Q)\;\times Q_{\rm zp}$")
    fig.suptitle("molbuilder's draws (vibration.State.draw) against the "
                 "density they must follow", fontsize=10.5, y=1.03)
    save(fig, "draws-follow-the-density.svg")


def fig_snapshot_cloud():
    """One atom's place over many snapshots: a bend (1600 cm-1) and a
    stretch (3700 cm-1), rows L1 = (0, 0, 0.30), L2 = (0.50, 0, 0); left
    every mode thermal at 300 K, right the bend after one photon."""
    rng = np.random.default_rng(7)
    qzp1, qzp2 = math.sqrt(16.8576 / 1600.0), math.sqrt(16.8576 / 3700.0)
    n = 6000
    z1, z2 = rng.standard_normal(n), rng.standard_normal(n)
    q2 = qzp2 * z2                                   # stretch: cold, Q_zp
    q1_thermal = qzp1 * z1                           # bend: cold, Q_zp
    u = stats.norm.cdf(-np.abs(z1))
    grid = np.linspace(-12, 0, 24001)
    left = np.interp(u, F1(grid), grid)
    q1_photon = -np.sign(z1) * left * qzp1
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.4), sharex=True,
                            sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, q1, title, col in (
            (axs[0], q1_thermal, "every mode thermal at 300 K", C[0]),
            (axs[1], q1_photon, "the bend after one photon (one quantum)",
             C[1])):
        # THE POINTS AS ONE PICTURE inside the vector figure: 12 000 of
        # them as vectors made a 1.4 MB file.
        ax.scatter(0.50 * q2, 0.30 * q1, s=2, color=col, alpha=0.25,
                   lw=0, rasterized=True)
        ax.plot(0, 0, "+", color=INK, ms=10, mew=1.5)
        ax.set_title(title)
        ax.set_xlabel(r"the atom's move along x (Å): $Q_2\,L_{2,x}$")
        ax.set_aspect("equal")
    axs[0].set_ylabel(r"along z (Å): $Q_1\,L_{1,z}$")
    lim = 0.12
    axs[0].set_xlim(-lim, lim)
    axs[0].set_ylim(-lim, lim)
    fig.suptitle("One atom's place over 6000 snapshots: the photon "
                 "excites one mode's factor alone", fontsize=10.5, y=1.02)
    save(fig, "one-atom-over-snapshots.svg")


def fig_mirror():
    x = np.linspace(-4.5, 4.5, 901)
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.plot(x, psi2(1, x), color=C[1], lw=2, label=r"$|\psi_1|^2$, even")
    for zk, col, height in ((0.85, C[2], 0.36), (-1.20, C[4], 0.47)):
        q = F1_inverse(stats.norm.cdf(-abs(zk)))[0] * -np.sign(zk)
        for s, mk in ((1, "o"), (-1, "s")):
            ax.plot(s * q, psi2(1, s * q), mk, color=col, ms=6)
            ax.plot([s * q, s * q], [psi2(1, q), height],
                    color=col, lw=0.8, ls=":")
        ax.annotate("", xy=(-q, height), xytext=(q, height),
                    arrowprops=dict(arrowstyle="<->", color=col, lw=1))
        ax.text(0, height + 0.015,
                f"z = {zk:+.2f} → Q = {q:+.2f};  −z → −Q",
                ha="center", fontsize=7.8, color=col)
    ax.set_xlabel(r"$Q\,/\,Q_{\rm zp}$")
    ax.set_ylabel(r"$P(Q)\;\times Q_{\rm zp}$")
    ax.set_title(r"Two mirrored pairs: $-z$ gives exactly $-Q$, a draw from the "
                 r"same even distribution")
    ax.set_ylim(0, 0.58)
    ax.legend(frameon=False, loc="upper right")
    save(fig, "mirrored-pair.svg")


if __name__ == "__main__":
    fig_states()
    fig_mehler()
    fig_inverse_transform()
    fig_draws()
    fig_snapshot_cloud()
    fig_mirror()
    print("wrote", sorted(p.name for p in OUT.glob("*.svg")))
