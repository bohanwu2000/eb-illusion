"""
Regenerates the main simulation figure of Section 4 (figures/sim_main.png) and its
maximum-likelihood counterpart (figures/sim_main_mle.png). Set EB_ILLUSION_FIGDIR explicitly to the output figures directory.
New runs save replicate estimates; --from-results reuses the saved NPZ.
Reproduces the four panels
described in the caption: MSE of the calibrated empirical Bayes estimator versus
the number of studies J=K for varying experimental noise, varying observational
noise, and varying effect size, plus the relative efficiency against the
experiment. Fully vectorized moment-matching (sim_main.png) and a vectorized
grid-search MLE (sim_main_mle.png). Run: python3 main_simulation_figure.py
"""
import argparse
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from paper_plot_style import apply_style, finish_figure, PURPLE, GREEN, RED, GRAY, CEB_COLORS

apply_style()
HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.abspath(os.environ.get("EB_ILLUSION_FIGDIR", os.path.join(HERE, "..", "figures")))
os.makedirs(FIGDIR, exist_ok=True)

THETA, MU, GAM2 = 1.0, 0.5, 1.0
MAX_COUNT = 100
Js = np.array([5, 10, 50, MAX_COUNT])
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


def generate_results(est, seed):
    """Persist replicate estimates so future figure changes can reuse this run."""
    rng = np.random.default_rng(seed)
    result = dict(counts=Js.copy(), seed=np.array(seed), replicates=np.array(R))
    conditions = [('se', se, THETA, se, 1.) for se in (.5, 1., 2.)]
    conditions += [('so', so, THETA, 1., so) for so in (.5, 1., 2.)]
    conditions += [('theta', th, th, 1., 1.) for th in (0., 1., 2.)]
    conditions += [('re', 1., THETA, 1., 1.)]
    for label, value, theta, se, so in conditions:
        result[f'{label}_{value}'] = np.stack([
            est(*gen(rng, int(j), theta, se, so), se, so) for j in Js])
    return result


def make_figure(est, fname, seed, results=None):
    results = generate_results(est, seed) if results is None else results
    counts = np.asarray(results['counts'])
    keep = (counts > 0) & (counts <= MAX_COUNT)
    if not keep.any():
        raise ValueError('No saved study counts between 1 and 100.')
    plotted_counts = counts[keep]
    def curve(label, value, theta=THETA):
        return np.mean((results[f'{label}_{float(value)}'][keep]-theta)**2, axis=-1)
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))
    condition_styles = tuple(zip(CEB_COLORS[:3], ("o", "s", "^")))
    # (top-left) vary sigma_e
    for se, (color, marker) in zip([0.5, 1.0, 2.0], condition_styles):
        ax[0, 0].plot(plotted_counts, curve("se", se),
                      color=color, marker=marker, ls="-", label=fr"$\sigma_e={se}$")
    ax[0, 0].set_title("(a) varying experimental noise")
    ax[0, 0].set_xlabel(r"$J=K$"); ax[0, 0].set_ylabel(r"MSE of $\hat\theta_{\mathrm{CEB}}$")
    ax[0, 0].set_xscale("log"); ax[0, 0].set_yscale("log"); ax[0, 0].legend(frameon=False)
    # (top-right) vary sigma_o, compare to experimental (sigma_e^2 = 1)
    for so, (color, marker) in zip([0.5, 1.0, 2.0], condition_styles):
        ax[0, 1].plot(plotted_counts, curve("so", so),
                      color=color, marker=marker, ls="-", label=fr"$\sigma_o={so}$")
    ax[0, 1].axhline(1.0, ls="--", color=RED, label="experimental")
    ax[0, 1].set_title("(b) varying observational noise")
    ax[0, 1].set_xlabel(r"$J=K$"); ax[0, 1].set_ylabel(r"MSE")
    ax[0, 1].set_xscale("log"); ax[0, 1].set_yscale("log"); ax[0, 1].legend(frameon=False)
    # (bottom-left) vary theta*
    for th, (color, marker) in zip([0.0, 1.0, 2.0], condition_styles):
        ax[1, 0].plot(plotted_counts, curve("theta", th, theta=th),
                      color=color, marker=marker, ls="-", label=fr"$\theta^\star={th}$")
    ax[1, 0].set_title("(c) varying effect size")
    ax[1, 0].set_xlabel(r"$J=K$"); ax[1, 0].set_ylabel(r"MSE of $\hat\theta_{\mathrm{CEB}}$")
    ax[1, 0].set_xscale("log"); ax[1, 0].set_yscale("log"); ax[1, 0].legend(frameon=False)
    # (bottom-right) relative efficiency
    re = curve("re", 1.) / 1.0
    ax[1, 1].plot(plotted_counts, re, "o-", color=PURPLE)
    ax[1, 1].axhline(0.5, ls=":", color=GRAY)
    ax[1, 1].set_title("(d) relative efficiency")
    ax[1, 1].set_xlabel(r"$J=K$"); ax[1, 1].set_ylabel(r"$\mathrm{RE}(J)=\mathrm{MSE}_{\mathrm{CEB}}/\sigma_e^2$")
    ax[1, 1].set_xscale("log")
    for panel in ax.flat:
        panel.set_xticks(plotted_counts, labels=[str(int(j)) for j in plotted_counts])
        panel.xaxis.set_minor_locator(plt.NullLocator())
    finish_figure(fig, width_fraction=1.)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, fname), dpi=200, bbox_inches="tight")
    fig.savefig(Path(FIGDIR) / (Path(fname).stem+".pdf"), bbox_inches="tight")
    plt.close(fig)
    return plotted_counts, re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-results', type=Path)
    parser.add_argument('--results', type=Path,
                        default=Path(HERE).parent/'results'/'main_simulation_rerun.npz')
    args = parser.parse_args()
    if args.from_results:
        with np.load(args.from_results) as data:
            saved = dict(data)
    else:
        if args.results.exists():
            parser.error('Reuse --from-results or choose a new --results path.')
        saved = {}
        for name, est, seed in [('mom', theta_mom, 101), ('mle', theta_mle, 202)]:
            saved.update({name+'_'+key: value for key, value in generate_results(est, seed).items()})
        args.results.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.results, **saved)
    for name, est, seed, fname in [('mom', theta_mom, 101, 'sim_main.png'),
                                  ('mle', theta_mle, 202, 'sim_main_mle.png')]:
        data = {key[len(name)+1:]: value for key, value in saved.items() if key.startswith(name+'_')}
        jv, re = make_figure(est, fname, seed, results=data)
        print(name, 'J=K:', jv.tolist(), 'RE:', re.tolist())
        below = jv[re < .5]
        print('RE first < 0.5 at J =', int(below[0]) if below.size else 'never')
    print('Saved sim_main and sim_main_mle PDF/PNG to', FIGDIR)


if __name__ == '__main__':
    main()
