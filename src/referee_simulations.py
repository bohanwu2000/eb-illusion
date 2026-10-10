"""CVCI comparison and shared-bias sensitivity analyses.

Saved paper results are in results/simulations. Use --plot-only to redraw,
or --experiment cvci/sensitivity and --output PATH to run new simulations.
CVCI follows Yang et al. (2025), arXiv:2511.00727, Algorithm 1 and Section 13.1.
"""

import json
import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from paper_plot_style import (apply_style, finish_figure, PURPLE, GREEN, RED, GRAY,
                              BLUE, CEB_COLORS, METHOD_LABELS)

# ----------------------------------------------------------------------
# Plot style: match the existing figures (serif / Times, large labels).
# ----------------------------------------------------------------------
apply_style()

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.abspath(os.environ.get("EB_ILLUSION_FIGDIR",
    os.path.join(HERE, "..", "figures")))
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
# Returns posterior mean, plug-in posterior sd, propagated sd, gamma2_hat, mu_hat.
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


# ======================================================================
# Experiment (b): non-Gaussian bias distributions.
#   MSE and coverage vs J when the true bias prior is skewed/heavy/bimodal
#   but the method still assumes Gaussian.
# ======================================================================


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
    bias_d, cov_d = [], []
    for dlt in deltas:
        y_e = rng.normal(THETA, SE, size=R)
        b_o = draw_bias(rng, (R, J), MU, GAM2, "gaussian")            # target mean MU
        y_o = rng.normal(THETA + b_o, SO)
        b_c = draw_bias(rng, (R, K), MU + dlt, GAM2, "gaussian")      # calib mean MU+delta
        y_c = rng.normal(b_c, SC)
        th, sd, sdc, _, _ = ceb_estimator(y_e, y_o, y_c, SE**2, SO**2, SC**2)
        bias_d.append(np.mean(th) - THETA); cov_d.append(coverage(th, sdc))

    summary["sensitivity"] = {
        "deltas": deltas.tolist(), "bias_ceb": bias_d, "cov_ceb": cov_d,
        "replicates": R, "J": J, "K": K, "theta": THETA,
        "bias_mean": MU, "bias_variance": GAM2,
        "sigma_e": SE, "sigma_o": SO, "sigma_c": SC,
    }

    plot_sensitivity(summary["sensitivity"])


def plot_sensitivity(results):
    """Render bias and coverage separately using the saved simulation results."""
    deltas = np.asarray(results["deltas"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    ax[0].plot(deltas, np.abs(results["bias_ceb"]), "o-", color=PURPLE)
    ax[0].axhline(0, color=GRAY, lw=0.8)
    ax[0].set_ylabel(r"$|\mathrm{bias}|$ of $\hat\theta$")
    ax[0].set_title("(a) absolute bias")
    ax[1].plot(deltas, results["cov_ceb"], "s--", color=PURPLE)
    ax[1].axhline(0.95, ls=":", color=GRAY, label="nominal coverage (0.95)")
    ax[1].set_ylabel("coverage of 95% interval")
    ax[1].set_ylim(0, 1)
    ax[1].set_title("(b) coverage")
    ax[1].legend(frameon=False, loc="upper right", bbox_to_anchor=(1., 1.2))
    for panel in ax:
        panel.set_xlabel(r"mean gap $\delta=\mu_c-\mu_o$")
    finish_figure(fig)
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


def illusion_estimator(y_e, y_o, se2, so2):
    """Homoskedastic Gaussian EB with both bias parameters fit by marginal ML.

    This is the paper's eq. (eb_full), integrating out the effect with a flat
    prior. Profiling the bias mean gives mu_hat = mean(y_o) - y_e and a
    posterior centered exactly at y_e. For t = so2 + gamma2, the profiled
    negative twice log likelihood, up to constants, is

        Q/t + (J-1)*log(t) + log(t + J*se2),  Q = sum((y_o-mean(y_o))**2).

    Its nonnegative stationary root solves
        t**2 + ((J-1)*se2 - Q/J)*t - Q*se2 = 0.
    Constraining t >= so2 gives the global maximum of the marginal likelihood.
    Returns posterior mean, plug-in posterior sd, gamma2_hat, mu_hat.
    """
    y_e, y_o = np.asarray(y_e, dtype=float), np.asarray(y_o, dtype=float)
    if (y_e.ndim != 1 or y_o.ndim != 2 or y_e.size == 0
            or y_o.shape[0] != y_e.size or y_o.shape[1] < 1
            or not np.isfinite(y_e).all() or not np.isfinite(y_o).all()
            or not np.isfinite(se2) or not np.isfinite(so2)
            or se2 <= 0 or so2 <= 0):
        raise ValueError("Expected finite y_e (R,), y_o (R, J>=1), and positive variances.")
    J = y_o.shape[1]
    mean_o = y_o.mean(axis=1)
    Q = np.sum((y_o - mean_o[:, None])**2, axis=1)
    b = (J - 1)*se2 - Q/J
    discriminant = np.sqrt(b**2 + 4*Q*se2)
    # Rationalize the root for b >= 0 to avoid subtractive cancellation.
    root = np.zeros_like(Q)
    positive = Q > 0
    rational = positive & (b >= 0)
    root[rational] = 2*Q[rational]*se2 / (b[rational] + discriminant[rational])
    direct = positive & (b < 0)
    root[direct] = (discriminant[direct] - b[direct])/2
    gam2 = np.maximum(root - so2, 0.0)
    post_sd = np.sqrt(1/(1/se2 + J/(so2 + gam2)))
    return y_e.copy(), post_sd, gam2, mean_o - y_e


def cvci_forest_estimates(x_e, y_o, y_c, Ks, B=999, bootstrap_seed=20260922, J=50):
    """First replicate, fixed before inspecting estimates; no outcome selection.

    Experimental, illusion, and calibrated EB rows use 95% Gaussian posterior
    credible intervals. The two EB rows condition on fitted hyperparameters.
    CVCI uses its independently seeded bootstrap normal interval. The coverage
    curves in sim_direct retain propagated calibration-mean uncertainty for EB;
    the corresponding SD and endpoints are also retained here for auditing.
    """
    if J < 1 or J > y_o.shape[1]:
        raise ValueError("Forest-panel J must be available in the simulated data.")
    if len(Ks) == 0 or any(K < 1 or K > y_c.shape[1] for K in Ks):
        raise ValueError("Forest calibration counts must be available in the simulated data.")
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

    theta_il, sd_il, gam2_il, mu_il = illusion_estimator(y_e, y_o, SE**2, SO**2)
    rows = [row("experiment", y_e[0], SE, interval_type="Gaussian posterior credible interval"),
            row("illusion", theta_il[0], sd_il[0],
                interval_type="Gaussian plug-in posterior credible interval",
                gamma2_hat=float(gam2_il[0]), mu_hat=float(mu_il[0])),
            row("cvci", theta_cv[0], sd_cv[0],
                interval_type="bootstrap normal confidence interval",
                selected_lambda=float(selected[0]))]
    for K in Ks:
        estimate, sd, sd_corr, gam2, mu = ceb_estimator(y_e, y_o, y_c[:, :K], SE**2, SO**2, SC**2)
        rows.append(row("ceb", estimate[0], sd[0], K=int(K),
                        interval_type="Gaussian plug-in posterior credible interval",
                        gamma2_hat=float(gam2[0]), mu_hat=float(mu[0]),
                        propagated_sd=float(sd_corr[0]),
                        propagated_lower=float(estimate[0] - Z*sd_corr[0]),
                        propagated_upper=float(estimate[0] + Z*sd_corr[0])))
    return {
        "J": J, "Ks": [int(K) for K in Ks], "replicate_index": 0,
        "selection": "First generated replicate, fixed before inspecting outcomes.",
        "coupling": "All rows share the same experimental and observational data; calibrated EB uses nested prefixes of the same calibration sample.",
        "bootstrap_replicates": B, "bootstrap_seed": bootstrap_seed,
        "interval_level": 0.95, "normal_quantile": Z,
        "experimental_interval": "y_e +/- Z*sigma_e: Gaussian posterior credible interval under a flat effect prior, also the known-variance confidence interval.",
        "illusion_fit": "Global marginal maximum likelihood over bias mean and variance, integrating out theta with a flat prior; homoskedastic profile Q/t + (J-1)*log(t) + log(t + J*sigma_e2), t >= sigma_o2.",
        "illusion_interval": "y_e +/- Z*sqrt(v), v = (1/sigma_e2 + J/(sigma_o2 + gamma2_hat))^(-1); plug-in posterior credible interval, with mu_hat = mean(y_o) - y_e.",
        "ceb_interval": "theta_hat +/- Z*sqrt(v), v = (1/sigma_e2 + J/(sigma_o2 + gamma2_hat))^(-1); plug-in posterior credible interval conditional on calibration moment estimates.",
        "cvci_interval": "Estimate +/- Z times nonparametric bootstrap SD (ddof=1), with within-source resampling and full LOOCV weight reselection; a normal confidence interval.",
        "coverage_panel_ceb_interval": "The sim_direct coverage curves retain propagated SD sqrt(v + c^2*(gamma2_hat + sigma_c2)/K), c = J*v/(sigma_o2 + gamma2_hat). Forest endpoints use sqrt(v); propagated endpoints are stored separately.",
        "rows": rows,
    }


def experiment_cvci(rng, R=5000, n_e=100, B=999, bootstrap_seed=20260921):
    if R < 2 or n_e < 2 or B < 2:
        raise ValueError("At least two simulation and bootstrap replicates and experimental measurements are required.")
    Js = [2, 5, 10, 20, 50, 100]
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
    summary["cvci"] = results
    plot_cvci(results)
    plot_cvci_forest(results)
    return results


def plot_cvci(results):
    """Shared-bias MSE and 95% interval coverage in sim_direct.

    Error bars describe Monte Carlo uncertainty in the plotted performance
    estimates. Calibrated EB coverage uses propagated calibration uncertainty;
    its plug-in posterior intervals are displayed separately by plot_cvci_forest.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 4.2), layout="constrained")
    styles = [(METHOD_LABELS["cvci"] + " (Yang et al., 2025)", BLUE, "s", results["cvci"])]
    for K, color, marker in zip(results["Ks"],
                                CEB_COLORS,
                                ("o", "^", "D", "v")):
        styles.append((METHOD_LABELS["ceb"] + rf" ($K={K}$)", color, marker, results["ceb"][str(K)]))
    for panel, metric in zip(ax, ("mse", "coverage")):
        for label, color, marker, rows in styles:
            rows = [row for row in rows if row["J"] <= 100]
            panel.errorbar([row["J"] for row in rows], [row[metric] for row in rows],
                           yerr=[Z * row[metric + "_mcse"] for row in rows],
                           fmt=marker + "-", color=color, label=label,
                           capsize=3, elinewidth=2.0)
        panel.set_xscale("log")
        panel.set_xlabel(r"observational studies $J$")
        if metric == "coverage":
            # One shared method legend keeps the MSE curves unobscured.
            panel.legend(frameon=False, loc="lower left")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("MSE")
    ax[0].set_title("(a) estimation accuracy")
    experimental_line = ax[0].axhline(results["sigma_e2"], ls="--", color=RED,
                                      linewidth=1.6, label=METHOD_LABELS["experiment"])
    ax[0].legend(handles=[experimental_line], loc="lower left", frameon=False)
    ax[1].set_ylabel("coverage of 95% interval")
    ax[1].set_ylim(0.7, 1.0)
    ax[1].set_title("(b) interval coverage")
    ax[1].axhline(0.95, ls=":", color=GRAY, linewidth=1.6)
    # Explicitly distinguish these causal intervals from the forest's posterior
    # intervals. Place the note above the legend, clear of the coverage curves.
    ax[1].text(.04, .53, "calibrated EB: propagated uncertainty",
               transform=ax[1].transAxes, ha="left", va="bottom",
               fontsize=9 * fig.get_figwidth()/10.25)
    finish_figure(fig, width_fraction=1.)
    fig.savefig(os.path.join(FIGDIR, "sim_direct.pdf"))
    fig.savefig(os.path.join(FIGDIR, "sim_direct.png"), dpi=180)
    plt.close(fig)


def plot_cvci_forest(results):
    """Draw sim_finite_k_methods from the saved cvci first-replicate summary.

    This is the shared entry point for simulation_restructure.plot_comparison.
    It renders saved endpoints without rerunning simulations or writing JSON.
    """
    forest = results["forest"]
    rows = forest["rows"]
    if ([row["method"] for row in rows[:3]] != ["experiment", "illusion", "cvci"]
            or any(row["method"] != "ceb" for row in rows[3:])
            or any(row.get("interval_type") != "Gaussian plug-in posterior credible interval"
                   for row in [rows[1], *rows[3:]])):
        raise ValueError("Regenerate the posterior forest with referee_simulations.py --plot-cvci.")
    if len(rows) - 3 > len(CEB_COLORS):
        raise ValueError("Forest calibration counts exceed the shared calibrated EB palette.")
    fig, ax = plt.subplots(figsize=(6., 4.2), layout="constrained")
    ax.axvline(results["theta"], ls=":", color="0.25", linewidth=1.5,
               label=rf"true effect $\theta^\star={results['theta']:g}$")
    colors = [RED, GREEN, BLUE, *CEB_COLORS]
    markers = ["o", "X", "s", "o", "^", "D", "v"]
    # Section numbers are checked against the current main.aux. The staged
    # caption also uses symbolic LaTeX labels for the eventual manuscript edit.
    labels = ["experimental (§3.1)", "illusion (§3.4)",
              "CVCI (Yang et al., 2025)"] + [
        METHOD_LABELS["ceb"] + rf" ($K={row['K']}$, §3.5)" for row in rows[3:]]
    ys = np.arange(len(rows))[::-1]
    for row, y, color, marker in zip(rows, ys, colors, markers):
        ax.errorbar(row["estimate"], y,
                    xerr=[[row["estimate"]-row["lower"]], [row["upper"]-row["estimate"]]],
                    fmt=marker, color=color, capsize=3, linewidth=2, zorder=3)
    ax.set_yticks(ys, labels=labels)
    ax.set_ylim(-0.6, len(rows)-0.3)
    ax.set_xlabel(r"effect $\theta$")
    ax.set_title(rf"results ($J={forest['J']}$)")
    # Method names and distinct markers supplement the shared color mapping.
    finish_figure(fig, width_fraction=.72)
    fig.savefig(os.path.join(FIGDIR, "sim_finite_k_methods.pdf"))
    fig.savefig(os.path.join(FIGDIR, "sim_finite_k_methods.png"), dpi=180)
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
    parser.add_argument('--experiment', choices=('cvci', 'sensitivity'), default='cvci')
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--plot-cvci', action='store_true', help='Alias for --experiment cvci --plot-only')
    parser.add_argument('--from-results', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--replicates', type=int)
    parser.add_argument('--seed', type=int, default=20260920)
    parser.add_argument('--bootstrap-replicates', type=int, default=999)
    parser.add_argument('--bootstrap-seed', type=int, default=20260921)
    parser.add_argument('--check-cvci', action='store_true')
    args = parser.parse_args()
    if args.check_cvci:
        check_cvci()
        return
    kind = 'cvci' if args.plot_cvci else args.experiment
    saved = Path(HERE).parent / 'results/simulations' / (kind + '.json')
    if args.plot_only or args.plot_cvci or args.from_results:
        result = json.loads((args.from_results or saved).read_text())
        if kind == 'cvci':
            plot_cvci(result)
            plot_cvci_forest(result)
        else:
            plot_sensitivity(result)
        return
    output = args.output or Path(HERE).parent / 'results' / (kind + '_rerun.json')
    if output.exists():
        parser.error('Choose a new --output path to preserve existing results.')
    count = args.replicates or (5000 if kind == 'cvci' else 20000)
    if count < 2 or args.bootstrap_replicates < 2:
        parser.error('At least two simulation and bootstrap replicates are required.')
    rng = np.random.default_rng(args.seed)
    if kind == 'cvci':
        result = experiment_cvci(rng, R=count, B=args.bootstrap_replicates,
                                 bootstrap_seed=args.bootstrap_seed)
        result['seed'] = args.seed
        # Match the paper's deliberately noisy experimental illustration.
        x, o, c = draw_cvci_data(np.random.default_rng(args.seed), count, result['n_exp'],
                                 max(result['Js']), max(result['Ks']))
        x = x[:1].copy()
        original_mean = float(x.mean())
        x += THETA + 2*SE - original_mean
        result['forest'] = cvci_forest_estimates(x, o[:1], c[:1], [10],
            args.bootstrap_replicates, args.bootstrap_seed + 1, J=50)
        result['forest'].update(selection='Deliberate illustration: experimental mean theta + 2 sigma_e.',
            original_experimental_mean=original_mean, imposed_experimental_mean=THETA+2*SE,
            experimental_standardized_error=2.)
        plot_cvci_forest(result)
    else:
        experiment_sensitivity(rng, R=count)
        result = dict(summary['sensitivity'], seed=args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print('Saved', output)


if __name__ == '__main__':
    main()
