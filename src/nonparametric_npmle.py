"""Nonparametric calibrated EB under a shared bimodal bias distribution.

Fit the calibration prior by fixed-grid EM and compare posterior-mean MSE
with Gaussian moment matching. The illustration uses K=600. The comparison
uses J=K in {5,10,50,100}, with independent calibration and observational biases.
The finite grid and iteration counts give an approximation to the NPMLE.

Run normally to redraw the saved results, or use --refit to rerun the simulation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, ScalarFormatter
import numpy as np
from scipy.special import logsumexp
from scipy.integrate import trapezoid

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO / 'results/nonparametric_npmle.json'
JS = [5, 10, 50, 100]
THETA, MU, SO, SC, SE = 1., .5, 1., 1., 1.
ILLUSTRATION_K = 600


def npdf(x, m, s):
    return np.exp(-.5*((x-m)/s)**2)/(s*np.sqrt(2*np.pi))


def draw_bimodal(rng, size):
    return MU+1.3*(2*rng.integers(0, 2, size=size)-1)+.35*rng.standard_normal(size)


def npmle(y, s, grid, iters=150):
    """EM updates for the mixing weights on a fixed support grid."""
    likelihood = npdf(y[:, None], grid[None, :], s)
    weights = np.full(grid.size, 1./grid.size)
    for _ in range(iters):
        joint = likelihood*weights
        weights = (joint/(joint.sum(axis=1, keepdims=True)+1e-300)).mean(axis=0)
    return weights


def gaussian_ceb(ye, yo, yc):
    g2 = max(yc.var()-SC**2, 0.)
    wo = 1/(SO**2+g2)
    return (ye/SE**2+wo*np.sum(yo-yc.mean()))/(1/SE**2+len(yo)*wo)


def posterior_mean(ye, yo, support, weights, direct=False):
    """Integrate over every Gaussian assignment center, retaining separate modes.

    Assignment components have precision SE**-2 + J*SO**-2. Their extreme
    centers give the domain, padded by 12 component SDs at each end; spacing
    at most one quarter of that SD resolves even narrow components. The fast
    path interpolates log mixture densities; direct=True uses logsumexp at
    every node. Both paths retain the full domain and use bounded work arrays.
    """
    yo = np.asarray(yo, dtype=float)
    support = np.asarray(support, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if (yo.ndim != 1 or support.ndim != 1 or weights.shape != support.shape
            or not support.size or not np.isfinite(ye)
            or not np.isfinite(yo).all() or not np.isfinite(support).all()
            or not np.isfinite(weights).all() or (weights < 0).any()
            or not np.isclose(weights.sum(), 1., rtol=1e-8, atol=1e-12)):
        raise ValueError('Use finite observations and atoms with nonnegative weights summing to one.')
    if not (np.isfinite(SE) and SE > 0 and np.isfinite(SO) and SO > 0):
        raise ValueError('SE and SO must be finite positive standard deviations.')
    if not yo.size:
        return float(ye)
    positive = weights > 0
    support, weights = support[positive], weights[positive]
    logweights = np.log(weights)
    precision = SE**-2 + yo.size*SO**-2
    sd = precision**-.5
    observational_weight = yo.size*SO**-2/precision
    lo = ye + observational_weight*(yo.mean()-ye-support.max())-12*sd
    hi = ye + observational_weight*(yo.mean()-ye-support.min())+12*sd
    t = np.linspace(lo, hi, int(np.ceil((hi-lo)/(.25*sd)))+1)

    def log_mixture(z):
        # Bound the residual-by-atom temporary even for large direct checks.
        flat = z.ravel()
        out = np.empty_like(flat)
        block = max(1, 262144//support.size)
        for start in range(0, flat.size, block):
            values = flat[start:start+block, None]
            out[start:start+block] = logsumexp(
                logweights-.5*((values-support)/SO)**2, axis=1)
        return out.reshape(z.shape)

    if not direct:
        # Use 0.002-SO resolution in ordinary cases. Widely
        # separated atoms can make log-density transitions sharper than SO.
        step = .002*SO/max(1., np.ptp(support)/(10*SO))
        rlo, rhi = yo.min()-hi, yo.max()-lo
        count = int(np.ceil((rhi-rlo)/step))+1
        if count <= 100001:
            residual = np.linspace(rlo, rhi, count)
            logdensity = log_mixture(residual)
        else:
            direct = True

    observations, counts = np.unique(yo, return_counts=True)
    lp = np.empty_like(t)
    block = max(1, 65536//observations.size)
    for start in range(0, t.size, block):
        theta = t[start:start+block]
        z = observations[None, :]-theta[:, None]
        ll = log_mixture(z) if direct else np.interp(z, residual, logdensity)
        lp[start:start+block] = (-.5*((ye-theta)/SE)**2
                                + (ll*counts).sum(axis=1))
    p = np.exp(lp-lp.max())
    # Center the first moment to avoid cancellation when theta is far from 0.
    center = t[np.argmax(lp)]
    return float(center+trapezoid((t-center)*p, t)/trapezoid(p, t))


def evaluate(case):
    j, replicate, ye, yo, yc = case
    grid = np.linspace(yc.min()-1, yc.max()+1, 40)
    weights = npmle(yc, SC, grid, iters=80)
    estimate = posterior_mean(ye, yo, grid, weights)
    gaussian = gaussian_ceb(ye, yo, yc)
    discrepancy = 0.
    if replicate < 3:
        discrepancy = abs(estimate-posterior_mean(ye, yo, grid, weights, direct=True))
    return j, [(estimate-THETA)**2, (gaussian-THETA)**2], discrepancy


def simulate(replicates, workers):
    rng = np.random.default_rng(7)
    yc = draw_bimodal(rng, ILLUSTRATION_K)+SC*rng.standard_normal(ILLUSTRATION_K)
    grid = np.linspace(yc.min()-1, yc.max()+1, 60)
    weights = npmle(yc, SC, grid, iters=150)
    cases = []
    for j in JS:
        for replicate in range(replicates):
            ye = THETA+SE*rng.standard_normal()
            yo = THETA+draw_bimodal(rng, j)+SO*rng.standard_normal(j)
            calibration = draw_bimodal(rng, j)+SC*rng.standard_normal(j)
            cases.append((j, replicate, ye, yo, calibration))
    errors = {j: [] for j in JS}
    discrepancy = 0.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for j, error, check in pool.map(evaluate, cases, chunksize=10):
            errors[j].append(error)
            discrepancy = max(discrepancy, check)
            if len(errors[j]) == replicates:
                print(f'J=K={j}: MSE (NPMLE, Gaussian EB) = '
                      f'{np.mean(errors[j], axis=0)}', flush=True)
    if discrepancy >= 1e-4:
        raise RuntimeError(f'Posterior integration check failed: {discrepancy}')
    summary = dict(
        J=JS, replicates=replicates, seed=7,
        fitter='fixed-grid EM approximation to the NPMLE',
        method_order=['NPMLE', 'Gaussian EB'],
        model=dict(theta=THETA, bias_means=[MU-1.3, MU+1.3],
                   bias_sds=[.35, .35], bias_weights=[.5, .5],
                   experimental_sd=SE, observational_sd=SO, calibration_sd=SC),
        settings=dict(illustration_k=ILLUSTRATION_K, illustration_grid_points=60,
                      illustration_em_updates=150, comparison_grid_points=40,
                      comparison_em_updates=80, support_padding=1.),
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        max_posterior_validation_error=discrepancy,
        plot_inputs=dict(left_y=yc.tolist(), left_grid=grid.tolist(),
                         left_weights=weights.tolist()), results={})
    for j in JS:
        e = np.asarray(errors[j])
        differences = e[:, 0]-e[:, 1]
        summary['results'][str(j)] = dict(
            mse=e.mean(axis=0).tolist(),
            mcse=(e.std(axis=0, ddof=1)/np.sqrt(replicates)).tolist(),
            paired_difference=float(differences.mean()),
            paired_mcse=float(differences.std(ddof=1)/np.sqrt(replicates)))
    return summary


def plot(summary, figdir):
    plt.rcParams.update({
        'font.family': ['serif'], 'mathtext.fontset': 'dejavuserif',
        'axes.labelsize': 13, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
        'legend.fontsize': 10, 'figure.facecolor': 'white',
        'axes.facecolor': 'white', 'savefig.facecolor': 'white',
    })
    inputs = summary['plot_inputs']
    yc, grid, weights = (np.asarray(inputs[key])
                         for key in ['left_y', 'left_grid', 'left_weights'])
    xx = np.linspace(-3, 4, 400)
    truth = .5*npdf(xx, MU-1.3, .35)+.5*npdf(xx, MU+1.3, .35)
    gaussian = npdf(xx, yc.mean(), np.sqrt(max(yc.var()-SC**2, 1e-6)))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(xx, truth, color='0.35', lw=2, label='true bias density')
    ax[0].plot(xx, gaussian, color='#C1121F', lw=2, ls='--', label='Gaussian fit')
    ax[0].vlines(grid, 0, weights/(grid[1]-grid[0]), color='#5B2C8D',
                 alpha=.9, lw=1.5, label=r'NPMLE $\hat g$')
    ax[0].set_title('(a) recovering the bias distribution')
    ax[0].set_xlabel('bias $b$')
    ax[0].set_ylabel('density')
    ax[0].legend(frameon=False)
    js = summary['J']
    for index, label, color, marker in [
            (1, 'Gaussian EB', '#C1121F', 'o'), (0, 'NPMLE', '#5B2C8D', 's')]:
        mse = [summary['results'][str(j)]['mse'][index] for j in js]
        mcse = [summary['results'][str(j)]['mcse'][index] for j in js]
        ax[1].errorbar(js, mse, yerr=1.96*np.asarray(mcse), label=label,
                       color=color, marker=marker, ms=5, lw=1.8, capsize=3)
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xticks(js, labels=[str(j) for j in js])
    ax[1].xaxis.set_minor_locator(NullLocator())
    ax[1].set_yticks([.05, .1, .2, .5])
    ax[1].yaxis.set_major_formatter(ScalarFormatter())
    ax[1].yaxis.set_minor_locator(NullLocator())
    ax[1].set_xlabel('number of studies $J=K$')
    ax[1].set_ylabel(r'MSE of $\hat\theta$')
    ax[1].set_title('(b) finite-sample performance')
    ax[1].legend(frameon=False)
    fig.tight_layout()
    figdir.mkdir(parents=True, exist_ok=True)
    for extension in ['pdf', 'png']:
        fig.savefig(figdir/f'sim_nonparametric.{extension}', dpi=220,
                    bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=DEFAULT_RESULTS)
    parser.add_argument('--from-results', type=Path)
    parser.add_argument('--refit', action='store_true')
    parser.add_argument('--replicates', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--figure-dir', type=Path, default=Path(
        os.environ.get('EB_ILLUSION_FIGDIR', REPO/'figures')))
    args = parser.parse_args()
    if args.replicates < 2 or args.workers < 1:
        parser.error('Use at least two replicates and at least one worker.')
    if args.from_results and args.refit:
        parser.error('Choose --from-results or --refit.')
    source = args.from_results or args.results
    if args.from_results or (source.exists() and not args.refit):
        summary = json.loads(source.read_text())
        if summary.get('fitter') != 'fixed-grid EM approximation to the NPMLE':
            parser.error('The results must come from this fixed-grid NPMLE simulation.')
    else:
        summary = simulate(args.replicates, args.workers)
        args.results.parent.mkdir(parents=True, exist_ok=True)
        args.results.write_text(json.dumps(summary, indent=2)+'\n')
        print('Saved', args.results, flush=True)
    plot(summary, args.figure_dir)
    print('Saved sim_nonparametric.pdf and sim_nonparametric.png', flush=True)


if __name__ == '__main__':
    main()
