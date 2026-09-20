"""
Additional simulation studies for the revision of

    "The Illusion of Learning from Observational Data:
     An Empirical Bayes Perspective"

These experiments respond to the referees' requests for
  (a) finite-sample behaviour at small numbers of calibration studies K,
      including how often the moment estimator of gamma^2 truncates at zero
      and how coverage behaves when K is fixed while J grows;
  (b) misspecification of the Gaussian bias prior (skewed, heavy-tailed,
      and bimodal bias distributions);
  (c) violations of the shared-bias ("transportability") assumption, in which
      the calibration and target bias distributions have different means;
  (d) effect heterogeneity, theta_j ~ N(theta, tau^2), which the working model
      ignores, and a simple heterogeneity-corrected variant;
  (e) a comparison with the no-covariate cross-validated causal inference
      (CVCI) procedure of Yang et al. (2025), arXiv:2511.00727, Eq. (2),
      Algorithm 1 and Section 13.1. CVCI uses experimental and observational
      data; calibrated EB additionally uses the calibration studies.

The calibrated EB estimates use moment matching (eqs. for gamma^2, mu, and the
posterior mean of theta). Computations are vectorized across replicates. Run:

    python3 referee_simulations.py

To regenerate only the CVCI comparison, preserving all other results:

    python3 referee_simulations.py --experiment cvci --replicates 5000 --bootstrap-replicates 999

To redraw it from saved curves and regenerate the first-replicate forest panel:

    python3 referee_simulations.py --plot-cvci

Figures are written to the repository figures/ directory and a numerical summary to
referee_sim_summary.json.
"""

import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Plot style: match the existing figures (serif / Times, large labels).
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.family": ["serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
})

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.abspath(os.environ.get("EB_ILLUSION_FIGDIR",
    os.path.join(HERE, "..", "figures")))
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

Z = 1.959963985  # 97.5% normal quantile

# Default true parameters (match Section 4 of the paper).
THETA = 1.0
MU = 0.5
GAM2 = 1.0
SE = 1.0
SO = 1.0
SC = 1.0

summary = {}


# ----------------------------------------------------------------------
# Bias generators (all with mean `mean` and variance `var`), used to probe
# misspecification of the Gaussian prior.
# ----------------------------------------------------------------------
def draw_bias(rng, shape, mean, var, dist="gaussian"):
    sd = np.sqrt(var)
    if dist == "gaussian":
        return rng.normal(mean, sd, size=shape)
    if dist == "skewed":
        # Shifted/scaled exponential: skewness 2, standardized then rescaled.
        e = rng.exponential(1.0, size=shape) - 1.0          # mean 0, var 1
        return mean + sd * e
    if dist == "heavy":
        # Student-t with 3 df, standardized to unit variance.
        df = 3.0
        t = rng.standard_t(df, size=shape) / np.sqrt(df / (df - 2.0))
        return mean + sd * t
    if dist == "bimodal":
        # Equal mixture of two well-separated normals, mean 0 var 1.
        comp = rng.integers(0, 2, size=shape)
        centers = np.where(comp == 0, -1.0, 1.0)             # +/- 1
        z = rng.normal(0.0, np.sqrt(1e-3), size=shape)       # tiny within-mode sd
        base = centers + z                                    # var approx 1
        base = base / np.std(base)                            # enforce unit var
        return mean + sd * base
    raise ValueError(dist)


# ----------------------------------------------------------------------
# Core moment-matching calibrated empirical Bayes estimator.
# All inputs have leading axis = Monte Carlo replicate.
#   y_e   : (R,)
#   y_o   : (R, J)
#   y_c   : (R, K)
# Homoskedastic within-study variances so2, sc2, se2 are scalars.
# Returns posterior mean, posterior sd, gamma2_hat, mu_hat.
# ----------------------------------------------------------------------
def ceb_estimator(y_e, y_o, y_c, se2, so2, sc2):
    R, K = y_c.shape
    J = y_o.shape[1]
    if K >= 2:
        ybar_c = y_c.mean(axis=1, keepdims=True)
        gam2 = ((y_c - ybar_c) ** 2).mean(axis=1) - sc2       # eq. (MM-est1)
    else:
        # A single calibration study cannot identify the bias variance.
        gam2 = np.zeros(R)
    gam2 = np.maximum(gam2, 0.0)                               # truncate at zero
    # mu_hat: inverse-variance weighted mean of calibration studies (eq. MM-est2).
    wc = np.broadcast_to(1.0 / (gam2[:, None] + sc2), y_c.shape)  # (R,K)
    mu_hat = (wc * y_c).sum(axis=1) / wc.sum(axis=1)
    # Posterior mean of theta anchored on the experiment (calibrated posterior).
    we = 1.0 / se2
    wo = 1.0 / (so2 + gam2)                                    # (R,)
    num = we * y_e + wo * (y_o - mu_hat[:, None]).sum(axis=1)
    den = we + J * wo
    theta_hat = num / den
    post_sd = np.sqrt(1.0 / den)                               # plug-in interval
    # Propagate the calibration uncertainty in mu_hat into the interval:
    # theta_hat depends on mu_hat with slope c = (J*wo)/den, and
    # Var(mu_hat) ~ (gamma^2 + sc2)/K for the inverse-variance mean.
    var_mu = (gam2 + sc2) / max(K, 1)
    c = (J * wo) / den
    post_sd_corr = np.sqrt(1.0 / den + c**2 * var_mu)
    return theta_hat, post_sd, post_sd_corr, gam2, mu_hat


def direct_correction(y_e, y_o, y_c, se2, so2, sc2):
    """Direct bias correction (empirical-calibration style): pool the
    observational studies by inverse variance, subtract the estimated mean
    bias, and inflate the interval by the estimated bias dispersion. Does NOT
    use the experimental anchor. Returns point estimate and sd."""
    R, K = y_c.shape
    J = y_o.shape[1]
    if K >= 2:
        ybar_c = y_c.mean(axis=1, keepdims=True)
        gam2 = np.maximum(((y_c - ybar_c) ** 2).mean(axis=1) - sc2, 0.0)
    else:
        gam2 = np.zeros(R)
    mu_hat = y_c.mean(axis=1)
    var_mu = (gam2 + sc2) / K                                  # variance of mu_hat
    wo = 1.0 / (so2 + gam2)
    theta_hat = (wo[:, None] * (y_o - mu_hat[:, None])).sum(axis=1) / (J * wo)
    # variance: pooled observational + propagated calibration-mean uncertainty
    post_sd = np.sqrt(1.0 / (J * wo) + var_mu)
    return theta_hat, post_sd, gam2, mu_hat


def coverage(theta_hat, post_sd, truth=THETA):
    return np.mean(np.abs(theta_hat - truth) <= Z * post_sd)


def mse(theta_hat, truth=THETA):
    return np.mean((theta_hat - truth) ** 2)


def cvci_mean_estimator(x_e, y_o, lambda_grid=None):
    """No-covariate CVCI: Yang et al. (2025), Eq. (2) and Section 13.1.

    x_e: (R, n_e) independent experimental measurements, n_e >= 2.
    y_o: (R, J) observational measurements (study estimates here), J >= 1.
    Returns full-data estimates (R,), selected weights (R,), and leave-one-out
    experimental validation losses (R, L). All observational data remain in
    every training fold. There is no calibration input or oracle variance.

    The default grid has 50 equally spaced points in [0, 1], as in the paper.
    Ties select the first grid point, matching the authors' np.argmin rule.
    For experimental mean m, variance v with divisor n, and observational mean
    o, the exact LOOCV loss is v*((n-lambda)/(n-1))**2 + lambda**2*(m-o)**2.
    This evaluates the original fold losses in O(R*(n_e+J+L)) work without
    allocating an R-by-n_e-by-L prediction array. The final estimate is refit
    on the full experimental sample, not averaged from the training folds.

    Reference code: xyang23/cross_validated_causal, causal_sim.py,
    cross_validation(mode='mean', k_fold=None), commit
    d74cb7c928938fd1a2978481f9aab6ac654c7d08.
    """
    x_e = np.asarray(x_e, dtype=float)
    y_o = np.asarray(y_o, dtype=float)
    if (x_e.ndim != 2 or y_o.ndim != 2 or x_e.shape[0] == 0
            or x_e.shape[0] != y_o.shape[0] or x_e.shape[1] < 2
            or y_o.shape[1] < 1):
        raise ValueError("Expected x_e (R, n_e>=2) and y_o (R, J>=1).")
    if not (np.isfinite(x_e).all() and np.isfinite(y_o).all()):
        raise ValueError("All measurements must be finite.")
    grid = (np.linspace(0.0, 1.0, 50) if lambda_grid is None
            else np.asarray(lambda_grid, dtype=float))
    if (grid.ndim != 1 or grid.size == 0 or not np.isfinite(grid).all()
            or np.any(grid < 0) or np.any(grid > 1)
            or np.any(np.diff(grid) <= 0)):
        raise ValueError("lambda_grid must be increasing, finite, and in [0, 1].")
    n_e = x_e.shape[1]
    mean_e = x_e.mean(axis=1)
    mean_o = y_o.mean(axis=1)
    variance_e = np.mean((x_e - mean_e[:, None])**2, axis=1)
    return _cvci_from_moments(mean_e, variance_e, n_e, mean_o, grid)


def _cvci_from_moments(mean_e, variance_e, n_e, mean_o, grid=None):
    """Exact grid search from sample moments; leading axes may include bootstrap."""
    if grid is None:
        grid = np.linspace(0.0, 1.0, 50)
    cv_loss = (variance_e[..., None] * ((n_e - grid) / (n_e - 1))**2
               + (mean_e - mean_o)[..., None]**2 * grid**2)
    selected = grid[np.argmin(cv_loss, axis=-1)]
    theta_hat = (1.0 - selected) * mean_e + selected * mean_o
    return theta_hat, selected, cv_loss


def bootstrap_experimental_moments(x_e, rng, B=999, batch_size=16):
    """Resample experimental measurements with replacement within each dataset.

    Save only bootstrap means and variances (divisor n), reused across J.
    Batching bounds memory without approximating the nonparametric bootstrap.
    """
    if B < 2 or batch_size < 1:
        raise ValueError("B must be at least 2 and batch_size must be positive.")
    R, n_e = x_e.shape
    means, variances = np.empty((R, B)), np.empty((R, B))
    for start in range(0, R, batch_size):
        stop = min(start + batch_size, R)
        indices = rng.integers(n_e, size=(stop-start, B, n_e), dtype=np.int32)
        samples = np.take_along_axis(x_e[start:stop, None, :], indices, axis=-1)
        means[start:stop] = samples.mean(axis=-1)
        variances[start:stop] = samples.var(axis=-1)
    return means, variances


def cvci_bootstrap_sd(y_o, mean_e_boot, variance_e_boot, n_e, rng,
                      batch_size=16):
    """Bootstrap SD with independent within-source resampling and full retuning.

    Yang et al. report bootstrap SDs, not a prescribed 95% interval. Our
    comparison uses estimate +/- Z times this SD as an explicitly stated
    normal-bootstrap interval. Each bootstrap dataset selects its own lambda
    on the original 50-point grid, then refits using its full-sample means.
    Observational study estimates are the sampling units in this experiment.
    """
    R, J = y_o.shape
    if (mean_e_boot.ndim != 2 or mean_e_boot.shape[0] != R
            or variance_e_boot.shape != mean_e_boot.shape
            or mean_e_boot.shape[1] < 2 or n_e < 2 or batch_size < 1):
        raise ValueError("Incompatible bootstrap moments or sample sizes.")
    B = mean_e_boot.shape[1]
    sd = np.empty(R)
    for start in range(0, R, batch_size):
        stop = min(start + batch_size, R)
        indices = rng.integers(J, size=(stop-start, B, J), dtype=np.int32)
        samples = np.take_along_axis(y_o[start:stop, None, :], indices, axis=-1)
        mean_o_boot = samples.mean(axis=-1)
        estimates, _, _ = _cvci_from_moments(
            mean_e_boot[start:stop], variance_e_boot[start:stop], n_e, mean_o_boot)
        sd[start:stop] = estimates.std(axis=-1, ddof=1)
    return sd


# ======================================================================
# Experiment (a): finite-sample behaviour in K.
#   - MSE, coverage and P(gamma2_hat = 0) as a function of K (J fixed).
#   - coverage as a function of J for several fixed K (shows that K must
#     grow with J or intervals under-cover).
# ======================================================================
def experiment_finite_k(rng, R=20000):
    Ks = [1, 2, 3, 5, 10, 20, 50, 100]
    J = 50
    mse_k, cov_k, covc_k, trunc_k = [], [], [], []
    for K in Ks:
        y_e = rng.normal(THETA, SE, size=R)
        b_o = draw_bias(rng, (R, J), MU, GAM2, "gaussian")
        y_o = rng.normal(THETA + b_o, SO)
        b_c = draw_bias(rng, (R, K), MU, GAM2, "gaussian")
        y_c = rng.normal(b_c, SC)
        th, sd, sdc, g2, _ = ceb_estimator(y_e, y_o, y_c, SE**2, SO**2, SC**2)
        mse_k.append(mse(th)); cov_k.append(coverage(th, sd))
        covc_k.append(coverage(th, sdc)); trunc_k.append(np.mean(g2 <= 1e-12))

    # coverage vs J for a fixed K, plug-in vs propagated-uncertainty interval
    Js = [5, 10, 20, 50, 100, 200, 500]
    Kfix = 10
    cov_plug, cov_corr = [], []
    for Jv in Js:
        y_e = rng.normal(THETA, SE, size=R)
        b_o = draw_bias(rng, (R, Jv), MU, GAM2, "gaussian")
        y_o = rng.normal(THETA + b_o, SO)
        b_c = draw_bias(rng, (R, Kfix), MU, GAM2, "gaussian")
        y_c = rng.normal(b_c, SC)
        th, sd, sdc, _, _ = ceb_estimator(y_e, y_o, y_c, SE**2, SO**2, SC**2)
        cov_plug.append(coverage(th, sd)); cov_corr.append(coverage(th, sdc))

    summary["finite_k"] = {
        "Ks": Ks, "J": 50, "mse": mse_k, "coverage": cov_k,
        "coverage_corr": covc_k, "p_trunc": trunc_k,
        "Js": Js, "Kfix": Kfix, "cov_plug_vs_J": cov_plug, "cov_corr_vs_J": cov_corr,
    }

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(Ks, mse_k, "o-", color="#5B2C8D")
    ax[0].axhline(SE**2, ls="--", color="grey", label=r"experiment ($\sigma_e^2$)")
    ax[0].set_xscale("log"); ax[0].set_xlabel(r"number of calibration studies $K$")
    ax[0].set_ylabel(r"MSE of $\hat\theta_{\mathrm{CEB}}$"); ax[0].set_title("(a) accuracy ($J=50$)")
    ax[0].legend(frameon=False)
    ax[1].plot(Ks, cov_k, "o-", color="#C1121F", label="plug-in interval")
    ax[1].plot(Ks, covc_k, "s-", color="#5B2C8D", label="propagated uncertainty")
    ax[1].axhline(0.95, ls="--", color="grey", label="nominal 0.95")
    ax[1].set_xscale("log"); ax[1].set_xlabel(r"number of calibration studies $K$")
    ax[1].set_ylabel("coverage of 95% interval"); ax[1].set_ylim(0, 1)
    ax[1].set_title(r"(b) calibration ($J=50$)"); ax[1].legend(frameon=False)
    ax[2].plot(Js, cov_plug, "o-", color="#C1121F", label="plug-in interval")
    ax[2].plot(Js, cov_corr, "s-", color="#5B2C8D", label="propagated uncertainty")
    ax[2].axhline(0.95, ls="--", color="grey")
    ax[2].set_xscale("log"); ax[2].set_xlabel(r"number of observational studies $J$")
    ax[2].set_ylabel("coverage of 95% interval"); ax[2].set_ylim(0, 1)
    ax[2].set_title(rf"(c) coverage vs $J$ (fixed $K={Kfix}$)"); ax[2].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "sim_finite_k.pdf"), bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# Experiment (b): non-Gaussian bias distributions.
#   MSE and coverage vs J when the true bias prior is skewed/heavy/bimodal
#   but the method still assumes Gaussian.
# ======================================================================
def experiment_misspec(rng, R=20000):
    Js = [5, 10, 20, 50, 100, 200, 500]
    dists = ["gaussian", "skewed", "heavy", "bimodal"]
    labels = {"gaussian": "Gaussian (correct)", "skewed": "skewed (exp.)",
              "heavy": "heavy-tailed ($t_3$)", "bimodal": "bimodal"}
    res = {d: {"mse": [], "cov": []} for d in dists}
    for d in dists:
        for J in Js:
            K = J
            y_e = rng.normal(THETA, SE, size=R)
            b_o = draw_bias(rng, (R, J), MU, GAM2, d)
            y_o = rng.normal(THETA + b_o, SO)
            b_c = draw_bias(rng, (R, K), MU, GAM2, d)
            y_c = rng.normal(b_c, SC)
            th, sd, sdc, _, _ = ceb_estimator(y_e, y_o, y_c, SE**2, SO**2, SC**2)
            res[d]["mse"].append(mse(th)); res[d]["cov"].append(coverage(th, sdc))
    summary["misspec"] = {"Js": Js, **{d: res[d] for d in dists}}

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for d in dists:
        ax[0].plot(Js, res[d]["mse"], "o-", label=labels[d])
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel(r"number of studies $J=K$"); ax[0].set_ylabel(r"MSE of $\hat\theta_{\mathrm{CEB}}$")
    ax[0].set_title("(a) accuracy under misspecification"); ax[0].legend(frameon=False)
    for d in dists:
        ax[1].plot(Js, res[d]["cov"], "o-", label=labels[d])
    ax[1].axhline(0.95, ls="--", color="grey")
    ax[1].set_xscale("log"); ax[1].set_xlabel(r"number of studies $J=K$")
    ax[1].set_ylabel("coverage of 95% interval"); ax[1].set_ylim(0.7, 1.0)
    ax[1].set_title("(b) calibration under misspecification"); ax[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "sim_misspec.pdf"), bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# Experiment (c): sensitivity to the shared-bias (transportability)
# assumption and to effect heterogeneity.
#   Left : calibration bias mean mu_c differs from target mean mu_o by delta.
#          Report |bias| of theta_hat and coverage vs delta.
#   Right: theta_j ~ N(theta, tau^2); the working model ignores tau^2.
#          Report coverage vs tau^2 for the plug-in interval and for a
#          heterogeneity-corrected interval that adds tau^2 to the weights.
# ======================================================================
def experiment_sensitivity(rng, R=20000):
    # ---- transportability: shift calibration mean by delta ----
    deltas = np.linspace(0.0, 2.0, 9)
    J = K = 50
    bias_d, cov_d, bias_dir, cov_dir = [], [], [], []
    for dlt in deltas:
        y_e = rng.normal(THETA, SE, size=R)
        b_o = draw_bias(rng, (R, J), MU, GAM2, "gaussian")            # target mean MU
        y_o = rng.normal(THETA + b_o, SO)
        b_c = draw_bias(rng, (R, K), MU + dlt, GAM2, "gaussian")      # calib mean MU+delta
        y_c = rng.normal(b_c, SC)
        th, sd, sdc, _, _ = ceb_estimator(y_e, y_o, y_c, SE**2, SO**2, SC**2)
        bias_d.append(np.mean(th) - THETA); cov_d.append(coverage(th, sdc))
        thd, sdd, _, _ = direct_correction(y_e, y_o, y_c, SE**2, SO**2, SC**2)
        bias_dir.append(np.mean(thd) - THETA); cov_dir.append(coverage(thd, sdd))

    # ---- effect heterogeneity tau^2 ----
    taus2 = np.linspace(0.0, 1.0, 6)
    J = K = 50
    cov_naive, cov_corr = [], []
    for t2 in taus2:
        y_e = rng.normal(THETA, SE, size=R)
        theta_j = rng.normal(THETA, np.sqrt(t2), size=(R, J))         # study-specific effects
        b_o = draw_bias(rng, (R, J), MU, GAM2, "gaussian")
        y_o = rng.normal(theta_j + b_o, SO)
        b_c = draw_bias(rng, (R, K), MU, GAM2, "gaussian")
        y_c = rng.normal(b_c, SC)
        th, sd, sdc, g2, mu_hat = ceb_estimator(y_e, y_o, y_c, SE**2, SO**2, SC**2)
        cov_naive.append(coverage(th, sdc))
        # heterogeneity-corrected interval: recover tau^2 from residual dispersion
        # of the bias-corrected observational estimates minus the calibration gamma^2.
        resid = y_o - mu_hat[:, None]
        rbar = resid.mean(axis=1, keepdims=True)
        tau2_hat = np.maximum(((resid - rbar) ** 2).mean(axis=1) - SO**2 - g2, 0.0)
        we = 1.0 / SE**2
        wo = 1.0 / (SO**2 + g2 + tau2_hat)
        den = we + J * wo
        num = we * y_e + wo * (y_o - mu_hat[:, None]).sum(axis=1)
        th_c = num / den
        var_mu = (g2 + SC**2) / K
        c = (J * wo) / den
        sd_c = np.sqrt(1.0 / den + c**2 * var_mu)     # propagate mu-hat + tau^2
        cov_corr.append(coverage(th_c, sd_c))

    summary["sensitivity"] = {
        "deltas": deltas.tolist(), "bias_ceb": bias_d, "cov_ceb": cov_d,
        "bias_direct": bias_dir, "cov_direct": cov_dir,
        "taus2": taus2.tolist(), "cov_naive": cov_naive, "cov_corrected": cov_corr,
    }

    plot_sensitivity(summary["sensitivity"])


def plot_sensitivity(results):
    """Render bias and coverage separately using the saved simulation results."""
    deltas = np.asarray(results["deltas"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    ax[0].plot(deltas, np.abs(results["bias_ceb"]), "o-", color="#5B2C8D")
    ax[0].axhline(0, color="grey", lw=0.8)
    ax[0].set_ylabel(r"$|\mathrm{bias}|$ of $\hat\theta$")
    ax[0].set_title("(a) absolute bias")
    ax[1].plot(deltas, results["cov_ceb"], "s--", color="#C1121F")
    ax[1].axhline(0.95, ls=":", color="grey", label="nominal coverage (0.95)")
    ax[1].set_ylabel("coverage of 95% interval")
    ax[1].set_ylim(0, 1)
    ax[1].set_title("(b) coverage")
    ax[1].legend(frameon=False, loc="upper right")
    for panel in ax:
        panel.set_xlabel(r"mean gap $\delta=\mu_c-\mu_o$")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "sim_sensitivity.pdf"), bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# Experiment (e): faithful no-covariate CVCI versus calibrated EB.
# The experimental summary is the mean of n_e actual simulated measurements,
# each with variance n_e * SE**2. Thus y_e still has variance SE**2.
# CVCI can split these measurements; CEB uses their full-sample mean.
# ======================================================================
def draw_cvci_data(rng, R, n_e, max_j, max_k):
    """Paired data, with nested observational and calibration samples."""
    x_e = rng.normal(THETA, np.sqrt(n_e) * SE, size=(R, n_e))
    b_o = draw_bias(rng, (R, max_j), MU, GAM2)
    y_o = rng.normal(THETA + b_o, SO)
    b_c = draw_bias(rng, (R, max_k), MU, GAM2)
    y_c = rng.normal(b_c, SC)
    return x_e, y_o, y_c


def cvci_forest_estimates(x_e, y_o, y_c, Ks, B=999, bootstrap_seed=20260922, J=50):
    """First replicate, fixed before inspecting estimates; no outcome selection.

    Uses the same interval constructions as the coverage panel, with a separate
    bootstrap seed so this illustration can be regenerated without rerunning
    the Monte Carlo coverage experiment. The naive interval treats bias as zero.
    """
    if J < 1 or J > y_o.shape[1]:
        raise ValueError("Forest-panel J must be available in the simulated data.")
    x_e, y_o, y_c = x_e[:1], y_o[:1, :J], y_c[:1]
    J, n_e = y_o.shape[1], x_e.shape[1]
    y_e = x_e.mean(axis=1)
    rng = np.random.default_rng(bootstrap_seed)
    bm, bv = bootstrap_experimental_moments(x_e, rng, B)
    theta_cv, selected, _ = cvci_mean_estimator(x_e, y_o)
    sd_cv = cvci_bootstrap_sd(y_o, bm, bv, n_e, rng)

    def row(method, estimate, sd, **extra):
        estimate, sd = float(estimate), float(sd)
        return dict(method=method, estimate=estimate, sd=sd,
                    lower=estimate-Z*sd, upper=estimate+Z*sd, **extra)

    rows = [row("experiment", y_e[0], SE),
            row("naive_observational", y_o.mean(), SO / np.sqrt(J)),
            row("cvci", theta_cv[0], sd_cv[0], selected_lambda=float(selected[0]))]
    for K in Ks:
        estimate, _, sd, _, _ = ceb_estimator(y_e, y_o, y_c[:, :K], SE**2, SO**2, SC**2)
        rows.append(row("ceb", estimate[0], sd[0], K=K))
    return {
        "J": J, "replicate_index": 0, "selection": "First generated replicate, fixed before inspecting outcomes.",
        "bootstrap_replicates": B, "bootstrap_seed": bootstrap_seed,
        "naive_interval": "Observational mean +/- 1.959963985 * sigma_o / sqrt(J), ignoring bias and between-study bias dispersion.",
        "rows": rows,
    }


def experiment_cvci(rng, R=5000, n_e=100, B=999, bootstrap_seed=20260921):
    if R < 2 or n_e < 2 or B < 2:
        raise ValueError("At least two simulation and bootstrap replicates and experimental measurements are required.")
    Js = [2, 5, 10, 20, 50, 100, 200, 500]
    Ks = [5, 10, 25, 50]

    def evaluate(theta_hat, sd):
        squared_error = (theta_hat - THETA)**2
        covered = np.abs(theta_hat - THETA) <= Z * sd
        return {
            "mse": float(squared_error.mean()),
            "mse_mcse": float(squared_error.std(ddof=1) / np.sqrt(R)),
            "coverage": float(covered.mean()),
            "coverage_mcse": float(covered.std(ddof=1) / np.sqrt(R)),
            "bias": float(theta_hat.mean() - THETA),
            "mean_interval_length": float(2 * Z * sd.mean()),
            "mean_interval_length_mcse": float(2 * Z * sd.std(ddof=1) / np.sqrt(R)),
        }

    # Paired methods, nested studies across J and K, shared Gaussian bias law.
    x_e, y_o, y_c = draw_cvci_data(rng, R, n_e, max(Js), max(Ks))
    y_e = x_e.mean(axis=1)

    # Bootstrap randomness is independent of simulation draws. Resample the
    # sources separately; reuse experimental resamples across J for efficiency.
    boot_rng = np.random.default_rng(bootstrap_seed)
    mean_e_boot, variance_e_boot = bootstrap_experimental_moments(x_e, boot_rng, B)
    cvci_rows = []
    ceb_rows = {str(K): [] for K in Ks}
    for J in Js:
        obs = y_o[:, :J]
        theta_cv, lambdas, _ = cvci_mean_estimator(x_e, obs)
        sd_cv = cvci_bootstrap_sd(obs, mean_e_boot, variance_e_boot, n_e, boot_rng)
        cvci_rows.append(dict(J=J, **evaluate(theta_cv, sd_cv),
                             mean_lambda=float(lambdas.mean()),
                             fraction_lambda_zero=float(np.mean(lambdas == 0)),
                             fraction_lambda_one=float(np.mean(lambdas == 1))))
        for K in Ks:
            theta_eb, _, sd_eb, _, _ = ceb_estimator(
                y_e, obs, y_c[:, :K], SE**2, SO**2, SC**2)
            differences = (theta_eb - THETA)**2 - (theta_cv - THETA)**2
            ceb_rows[str(K)].append(dict(
                J=J, **evaluate(theta_eb, sd_eb),
                mse_difference_ceb_minus_cvci=float(differences.mean()),
                mse_difference_mcse=float(differences.std(ddof=1) / np.sqrt(R))))
        print(f"CVCI comparison: J={J}, {R} simulations, {B} bootstrap replicates complete.", flush=True)
    results = {
        "reference": "Yang et al. (2025), arXiv:2511.00727, Eq. (2), Algorithm 1, Section 13.1",
        "reference_code_commit": "d74cb7c928938fd1a2978481f9aab6ac654c7d08",
        "replicates": R, "n_exp": n_e, "cv_folds": n_e,
        "bootstrap_replicates": B, "bootstrap_seed": bootstrap_seed,
        "lambda_grid": np.linspace(0.0, 1.0, 50).tolist(),
        "theta": THETA, "mu_o": MU, "mu_c": MU, "gamma2": GAM2,
        "sigma_e2": SE**2, "sigma_o2": SO**2, "sigma_c2": SC**2,
        "experimental_measurement_variance": n_e * SE**2,
        "shared_bias_distribution": True, "bias_distribution": "Gaussian",
        "Js": Js, "Ks": Ks,
        "information": "CVCI: experimental measurements and observational estimates; CEB additionally uses calibration studies.",
        "coupling": "Paired methods; nested observational samples across J and calibration samples across K; experimental bootstrap samples reused across J.",
        "cvci_interval": "Estimate +/- 1.959963985 times nonparametric bootstrap SD (ddof=1); an explicit normal-interval extension of the authors' bootstrap SD reporting, not a coverage guarantee from their paper.",
        "bootstrap_scheme": "Independent resampling with replacement within each source at its original sample size; observational study estimates are sampling units; reselect lambda by experimental LOOCV on the full 50-point grid and refit full bootstrap data in every resample.",
        "ceb_interval": "Estimate +/- 1.959963985 times propagated-uncertainty SD from ceb_estimator (includes calibration-mean estimation variance).",
        "cvci": cvci_rows, "ceb": ceb_rows,
        "forest": cvci_forest_estimates(x_e, y_o, y_c, Ks, B, bootstrap_seed+1),
    }
    summary.pop("direct", None)
    summary["cvci"] = results
    plot_cvci(results)
    return results


def plot_cvci(results):
    """Shared-bias MSE, 95% interval coverage, and one-replicate forest plot.

    Error bars describe Monte Carlo uncertainty in the plotted performance
    estimates, distinct from the causal intervals whose coverage is measured.
    """
    fig, ax = plt.subplots(1, 3, figsize=(14.2, 5.2), layout="constrained",
                           gridspec_kw={"width_ratios": [1, 1, 1.05]})
    styles = [("CVCI", "#0A6E31", "s", results["cvci"])]
    for K, color, marker in zip(results["Ks"],
                                ("#5B2C8D", "#0072B2", "#D55E00", "#CC79A7"),
                                ("o", "^", "D", "v")):
        styles.append((rf"calibrated EB ($K={K}$)", color, marker, results["ceb"][str(K)]))
    for panel, metric in zip(ax, ("mse", "coverage")):
        for label, color, marker, rows in styles:
            panel.errorbar(results["Js"], [row[metric] for row in rows],
                           yerr=[Z * row[metric + "_mcse"] for row in rows],
                           fmt=marker + "-", color=color, label=label,
                           capsize=2.5, elinewidth=1.0)
        panel.set_xscale("log")
        panel.set_xlabel(r"observational studies $J$", fontsize=17)
        if metric == "coverage":
            # One shared method legend keeps the MSE curves unobscured.
            panel.legend(frameon=False, fontsize=12)
    ax[0].set_yscale("log")
    ax[0].set_ylabel(r"MSE of $\hat\theta$", fontsize=17)
    ax[0].set_title("(a) estimation accuracy", fontsize=17)
    ax[0].axhline(results["sigma_e2"], ls="--", color="grey", linewidth=1.3)
    ax[1].set_ylabel("coverage of 95% interval", fontsize=17)
    ax[1].set_ylim(0.7, 1.0)
    ax[1].set_title("(b) interval coverage", fontsize=17)
    ax[1].axhline(0.95, ls="--", color="grey", linewidth=1.3)
    forest = results["forest"]
    rows = forest["rows"]
    experiment = rows[0]
    ax[2].axvspan(experiment["lower"], experiment["upper"], color="#C1121F",
                  alpha=0.08, label="experimental 95% interval")
    ax[2].axvline(results["theta"], ls=":", color="0.25", linewidth=1.5,
                  label=rf"true effect $\theta^\star={results['theta']:g}$")
    colors = ["#C1121F", "#555555"] + [style[1] for style in styles]
    markers = ["o", "X"] + [style[2] for style in styles]
    labels = ["Experimental", "Naive observational\n(illusion)"] + [style[0] for style in styles]
    ys = np.arange(len(rows))[::-1]
    for row, y, color, marker in zip(rows, ys, colors, markers):
        ax[2].errorbar(row["estimate"], y,
                       xerr=[[row["estimate"]-row["lower"]], [row["upper"]-row["estimate"]]],
                       fmt=marker, color=color, capsize=3, linewidth=2, zorder=3)
    ax[2].set_yticks(ys, labels=labels, fontsize=12)
    ax[2].set_ylim(-0.6, len(rows)-0.3)
    ax[2].set_xlabel(r"effect estimate $\hat\theta$", fontsize=17)
    ax[2].set_title(rf"(c) results ($J={forest['J']}$)", fontsize=17)
    for panel in ax:
        panel.tick_params(axis="x", labelsize=13)
    for panel in ax[:2]:
        panel.tick_params(axis="y", labelsize=13)
    # The caption defines the experimental band and true-effect reference line.
    # Reuse the existing comparison artifact rather than add a project file.
    fig.savefig(os.path.join(FIGDIR, "sim_direct.pdf"), bbox_inches="tight")
    plt.close(fig)


def check_cvci():
    """Check optimized losses against literal held-out fits; no file writes."""
    rng = np.random.default_rng(981)
    grid = np.linspace(0, 1, 50)
    for n_e, J in ((2, 1), (7, 4), (100, 20)):
        x_e = rng.normal(1, 10, size=(5, n_e))
        y_o = rng.normal(1.5, np.sqrt(2), size=(5, J))
        theta, selected, losses = cvci_mean_estimator(x_e, y_o)
        for r in range(5):
            reference = []
            for weight in grid:
                fold_losses = []
                for i in range(n_e):
                    train = np.delete(x_e[r], i)
                    fitted = (1 - weight) * train.mean() + weight * y_o[r].mean()
                    fold_losses.append((x_e[r, i] - fitted)**2)
                reference.append(np.mean(fold_losses))
            np.testing.assert_allclose(losses[r], reference, rtol=2e-13, atol=2e-13)
            weight = grid[np.argmin(reference)]
            assert selected[r] == weight
            np.testing.assert_allclose(theta[r],
                (1-weight)*x_e[r].mean()+weight*y_o[r].mean(), rtol=2e-13, atol=2e-13)
    theta, selected, _ = cvci_mean_estimator(
        np.array([[0., 2.], [1., 1.], [1., 1.]]),
        np.array([[1., 1.], [3., 3.], [1., 1.]]))
    np.testing.assert_array_equal(selected, [1., 0., 0.])
    np.testing.assert_array_equal(theta, [1., 1., 1.])
    print("CVCI check passed: literal leave-one-out losses, grid selection, full-data refit, endpoints and ties.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("all", "cvci"), default="all")
    parser.add_argument("--replicates", type=int, default=None,
                        help="Monte Carlo replicates (default: 5000 for CVCI, 20000 for other experiments)")
    parser.add_argument("--seed", type=int, default=20260920, help="CVCI comparison seed")
    parser.add_argument("--bootstrap-replicates", type=int, default=999)
    parser.add_argument("--bootstrap-seed", type=int, default=20260921)
    parser.add_argument("--check-cvci", action="store_true")
    parser.add_argument("--plot-cvci", action="store_true",
                        help="Reuse saved performance curves and regenerate the first-replicate forest plot")
    args = parser.parse_args()
    if args.check_cvci:
        check_cvci()
        return
    if args.replicates is not None and args.replicates < 2:
        parser.error("--replicates must be at least 2")
    if args.bootstrap_replicates < 2:
        parser.error("--bootstrap-replicates must be at least 2")
    R = args.replicates if args.replicates is not None else 20000
    R_cvci = args.replicates if args.replicates is not None else 5000
    output = os.path.join(HERE, "referee_sim_summary.json")
    if args.plot_cvci:
        with open(output) as f:
            summary.update(json.load(f))
        results = summary["cvci"]
        expected = {"theta": THETA, "mu_o": MU, "mu_c": MU, "gamma2": GAM2,
                    "sigma_e2": SE**2, "sigma_o2": SO**2, "sigma_c2": SC**2}
        if any(results.get(key) != value for key, value in expected.items()):
            parser.error("Saved parameters differ from the current model; regenerate the CVCI experiment.")
        x_e, y_o, y_c = draw_cvci_data(np.random.default_rng(results["seed"]),
                                      results["replicates"], results["n_exp"],
                                      max(results["Js"]), max(results["Ks"]))
        results["forest"] = cvci_forest_estimates(
            x_e, y_o, y_c, results["Ks"], results["bootstrap_replicates"],
            results["bootstrap_seed"]+1)
        plot_cvci(results)
        with open(output, "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(results["forest"], indent=2))
        return
    if args.experiment == "cvci":
        if os.path.exists(output):
            with open(output) as f:
                summary.update(json.load(f))
        results = experiment_cvci(np.random.default_rng(args.seed), R=R_cvci,
                                  B=args.bootstrap_replicates,
                                  bootstrap_seed=args.bootstrap_seed)
        results["seed"] = args.seed
        with open(output, "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(results, indent=2))
        return
    rng = np.random.default_rng(20260714)
    experiment_finite_k(rng, R=R)
    experiment_misspec(rng, R=R)
    experiment_sensitivity(rng, R=R)
    experiment_cvci(np.random.default_rng(args.seed), R=R_cvci,
                    B=args.bootstrap_replicates, bootstrap_seed=args.bootstrap_seed)
    summary["cvci"]["seed"] = args.seed
    with open(os.path.join(HERE, "referee_sim_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    # concise console report
    def r(x, n=4):
        return [round(float(v), n) for v in x]
    print("=== FINITE K (J=50) ===")
    print("K:", summary["finite_k"]["Ks"])
    print("MSE:", r(summary["finite_k"]["mse"]))
    print("cov plug-in :", r(summary["finite_k"]["coverage"]))
    print("cov corrected:", r(summary["finite_k"]["coverage_corr"]))
    print("P(gamma2_hat=0):", r(summary["finite_k"]["p_trunc"]))
    print(f"cov vs J at fixed K={summary['finite_k']['Kfix']} for J={summary['finite_k']['Js']}:")
    print("  plug-in :", r(summary["finite_k"]["cov_plug_vs_J"]))
    print("  corrected:", r(summary["finite_k"]["cov_corr_vs_J"]))
    print("=== MISSPEC (coverage vs J) ===")
    for d in ["gaussian", "skewed", "heavy", "bimodal"]:
        print(f"{d:9s} cov:", r(summary["misspec"][d]["cov"]))
    print("=== SENSITIVITY: transportability (delta) ===")
    print("delta:", r(summary["sensitivity"]["deltas"], 3))
    print("bias CEB:", r(summary["sensitivity"]["bias_ceb"]))
    print("cov  CEB:", r(summary["sensitivity"]["cov_ceb"]))
    print("=== SENSITIVITY: heterogeneity (tau^2) ===")
    print("tau^2:", r(summary["sensitivity"]["taus2"], 3))
    print("cov plug-in:", r(summary["sensitivity"]["cov_naive"]))
    print("cov corrected:", r(summary["sensitivity"]["cov_corrected"]))
    print("=== CVCI vs EB (shared bias, MSE and coverage vs J) ===")
    print("J:", summary["cvci"]["Js"])
    for metric in ("mse", "coverage"):
        print(metric, "CVCI:", r([v[metric] for v in summary["cvci"]["cvci"]]))
        for K in summary["cvci"]["Ks"]:
            print(metric, f"EB K={K}:", r([v[metric] for v in summary["cvci"]["ceb"][str(K)]]))


if __name__ == "__main__":
    main()
