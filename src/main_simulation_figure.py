"""
Regenerates the main simulation figure of Section 4 (figures/sim_main.png) and its
maximum-likelihood counterpart (figures/sim_main_mle.png). Reproduces the four panels
described in the caption: MSE of the calibrated empirical Bayes estimator versus
the number of studies J=K for varying experimental noise, varying observational
noise, and varying effect size, plus the relative efficiency against the
experiment. Fully vectorized moment-matching (sim_main.png) and a vectorized
grid-search MLE (sim_main_mle.png). Run: python3 main_simulation_figure.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": ["serif"], "mathtext.fontset": "dejavuserif",
    "axes.titlesize": 15, "axes.labelsize": 14,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "lines.linewidth": 2.0, "lines.markersize": 6,
})
HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.abspath(os.environ.get("EB_ILLUSION_FIGDIR",
    os.path.join(HERE, "..", "figures")))
os.makedirs(FIGDIR, exist_ok=True)

THETA, MU, GAM2 = 1.0, 0.5, 1.0
Js = np.array([5, 10, 50, 100, 200, 500, 1000])
R = 4000
GAMMA_GRID = np.linspace(0.0, 8.0, 161)     # for MLE


def gen(rng, J, theta, se, so, sc=1.0):
    y_e = rng.normal(theta, se, size=R)
    b_o = rng.normal(MU, np.sqrt(GAM2), size=(R, J))
    y_o = rng.normal(theta + b_o, so)
    b_c = rng.normal(MU, np.sqrt(GAM2), size=(R, J))
    y_c = rng.normal(b_c, sc)
    return y_e, y_o, y_c


def theta_mom(y_e, y_o, y_c, se, so, sc=1.0):
    ybar = y_c.mean(1, keepdims=True)
    g2 = np.maximum(((y_c - ybar) ** 2).mean(1) - sc**2, 0.0)
    mu = y_c.mean(1)
    we, wo = 1.0 / se**2, 1.0 / (so**2 + g2)
    J = y_o.shape[1]
    return (we * y_e + wo * (y_o - mu[:, None]).sum(1)) / (we + J * wo)


def theta_mle(y_e, y_o, y_c, se, so, sc=1.0):
    # profile likelihood over a grid of gamma^2 using the calibration studies;
    # loop over the grid to keep memory bounded. Profiled mu is the (homoskedastic)
    # calibration mean, so the log-likelihood reduces to a variance-matching term.
    R_, K = y_c.shape
    ybar = y_c.mean(1, keepdims=True)                            # (R,1)
    ss = ((y_c - ybar) ** 2).sum(1)                             # (R,)
    ll = np.empty((R_, GAMMA_GRID.size))
    for gi, g in enumerate(GAMMA_GRID):
        v = g + sc**2
        ll[:, gi] = -0.5 * (ss / v + K * np.log(v))
    g2 = GAMMA_GRID[np.argmax(ll, axis=1)]
    mu = y_c.mean(1)
    we, wo = 1.0 / se**2, 1.0 / (so**2 + g2)
    J = y_o.shape[1]
    return (we * y_e + wo * (y_o - mu[:, None]).sum(1)) / (we + J * wo)


def mse_curve(rng, est, theta=THETA, se=1.0, so=1.0):
    return np.array([np.mean((est(*gen(rng, J, theta, se, so), se, so) - theta) ** 2)
                     for J in Js])


def make_figure(est, fname, seed):
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    # (top-left) vary sigma_e
    for se in [0.5, 1.0, 2.0]:
        ax[0, 0].plot(Js, mse_curve(rng, est, se=se, so=1.0), "o-", label=fr"$\sigma_e={se}$")
    ax[0, 0].set_title("(a) varying experimental noise")
    ax[0, 0].set_xlabel(r"$J=K$"); ax[0, 0].set_ylabel(r"MSE of $\hat\theta_{\mathrm{CEB}}$")
    ax[0, 0].set_xscale("log"); ax[0, 0].set_yscale("log"); ax[0, 0].legend(frameon=False)
    # (top-right) vary sigma_o, compare to naive (sigma_e^2 = 1)
    for so in [0.5, 1.0, 2.0]:
        ax[0, 1].plot(Js, mse_curve(rng, est, se=1.0, so=so), "o-", label=fr"$\sigma_o={so}$")
    ax[0, 1].axhline(1.0, ls="--", color="grey", label="naive (experiment)")
    ax[0, 1].set_title("(b) varying observational noise")
    ax[0, 1].set_xlabel(r"$J=K$"); ax[0, 1].set_ylabel(r"MSE")
    ax[0, 1].set_xscale("log"); ax[0, 1].set_yscale("log"); ax[0, 1].legend(frameon=False)
    # (bottom-left) vary theta*
    for th in [0.0, 1.0, 2.0]:
        ax[1, 0].plot(Js, mse_curve(rng, est, theta=th, se=1.0, so=1.0), "o-", label=fr"$\theta^\star={th}$")
    ax[1, 0].set_title("(c) varying effect size")
    ax[1, 0].set_xlabel(r"$J=K$"); ax[1, 0].set_ylabel(r"MSE of $\hat\theta_{\mathrm{CEB}}$")
    ax[1, 0].set_xscale("log"); ax[1, 0].set_yscale("log"); ax[1, 0].legend(frameon=False)
    # (bottom-right) relative efficiency
    re = mse_curve(rng, est, se=1.0, so=1.0) / 1.0
    ax[1, 1].plot(Js, re, "o-", color="#5B2C8D")
    ax[1, 1].axhline(0.5, ls=":", color="grey")
    ax[1, 1].set_title("(d) relative efficiency vs experiment")
    ax[1, 1].set_xlabel(r"$J=K$"); ax[1, 1].set_ylabel(r"$\mathrm{RE}(J)=\mathrm{MSE}_{\mathrm{CEB}}/\sigma_e^2$")
    ax[1, 1].set_xscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, fname), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return Js, re


if __name__ == "__main__":
    Jv, re = make_figure(theta_mom, "sim_main.png", 101)
    make_figure(theta_mle, "sim_main_mle.png", 202)
    print("J :", list(Jv))
    print("RE:", [round(float(x), 4) for x in re])
    below = Jv[re < 0.5]
    print("RE first < 0.5 at J =", int(below[0]) if below.size else "never")
    print("saved sim_main.png, sim_main_mle.png to", FIGDIR)
