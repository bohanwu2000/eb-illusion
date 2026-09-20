"""
LEGEND-HTN example for the calibrated empirical Bayes framework.

Estimand: log hazard ratio of ACE inhibitor (ACEI) vs thiazide/thiazide-like
diuretic (THZ) as first-line monotherapy on ACUTE MYOCARDIAL INFARCTION.
Positive log-HR = ACEI worse.

Three designs (in analogy with the NSW/LaLonde example):
  * EXPERIMENTAL anchor y_e : ALLHAT randomized trial (lisinopril vs
    chlorthalidone), fatal CHD + nonfatal MI, RR 0.99 (0.91-1.08).  [REAL]
  * OBSERVATIONAL y_o,j     : per-database ACEI-vs-THZ MI estimates across the
    nine LEGEND databases.  Published pooled thiazide-vs-ACEI HR = 0.84
    (0.75-0.95), i.e. ACEI-vs-THZ ~ 1.19.  ~3.2M new-users.  [pooled REAL;
    per-database values are a REPRESENTATIVE reconstruction matched to it]
  * CALIBRATION y_c,k       : negative-control outcomes for the ACEI-vs-THZ
    contrast (true log-HR = 0), which reveal the systematic residual-confounding
    bias.  LEGEND uses a Gaussian empirical null over >50 negative controls.
    [design REAL; the 50 values here are a REPRESENTATIVE reconstruction]

To make this the exact published analysis, replace the reconstructed per-study
arrays with the extract from data.ohdsi.org/LegendBasicViewer (or the Lancet
2019 appendix): the nine per-database log-HRs and standard errors for
ACEI-vs-THZ on acute MI, and the negative-control log-HRs/SEs for that contrast.

Sources: Suchard et al., Lancet 2019 (LEGEND-HTN); ALLHAT, JAMA 2002;
Schuemie et al. (empirical calibration).
"""
import numpy as np

L = np.log
def CI_to_se(lo, hi):
    return (L(hi) - L(lo)) / (2 * 1.96)

# ---- EXPERIMENTAL anchor: ALLHAT (REAL) ----
ye = L(0.99); se_e = CI_to_se(0.91, 1.08)
n_ACEI_rct, n_THZ_rct = 9054, 15255

# ---- OBSERVATIONAL: 9 LEGEND databases (pooled REAL, per-db REPRESENTATIVE) ----
rng = np.random.default_rng(7)
db = ["MarketScan CCAE", "MarketScan MDCD", "MarketScan MDCR", "Optum DOD",
      "Optum EHR", "IQVIA Germany", "IQVIA France", "JMDC (Japan)", "IQVIA US EHR"]
n_o = np.array([905000, 215000, 330000, 520000, 300000, 240000, 205000, 150000, 135000])
se_o = np.array([0.032, 0.075, 0.055, 0.041, 0.060, 0.070, 0.078, 0.088, 0.092])
mu_bias_true, gam_between = 0.174, 0.045
yo = mu_bias_true + rng.normal(0, gam_between, 9) + rng.normal(0, se_o, 9)
wo0 = 1 / se_o**2
yo_pool = (wo0 * yo).sum() / wo0.sum(); se_pool = np.sqrt(1 / wo0.sum())

# ---- CALIBRATION: negative-control outcomes (REPRESENTATIVE), true log-HR = 0 ----
K, mu_null, gam_null = 50, 0.15, 0.09
sc = rng.uniform(0.05, 0.16, K)
yc = mu_null + rng.normal(0, gam_null, K) + rng.normal(0, sc, K)


def ceb(ye, se2, yo, so2, yc, sc2):
    """Calibrated empirical Bayes (paper's estimator), heteroskedastic, one realization."""
    ybar = yc.mean()
    gam2 = max(np.mean((yc - ybar) ** 2 - sc2), 0.0)      # eq. (MM-est1)
    wc = 1 / (gam2 + sc2)
    mu = (wc * yc).sum() / wc.sum()                        # eq. (MM-est2)
    var_mu = 1 / wc.sum()
    we = 1 / se2; wo = 1 / (so2 + gam2)
    den = we + wo.sum()
    th = (we * ye + (wo * (yo - mu)).sum()) / den
    sd = np.sqrt(1 / den)                                 # plug-in interval
    c = wo.sum() / den
    sd_c = np.sqrt(1 / den + c**2 * var_mu)               # propagated interval
    return th, sd, sd_c, gam2, mu, var_mu


if __name__ == "__main__":
    th, sd, sd_c, gam2, mu, var_mu = ceb(ye, se_e**2, yo, se_o**2, yc, sc**2)
    wo = 1 / (se_o**2 + gam2)
    yd = (wo * (yo - mu)).sum() / wo.sum(); sd_d = np.sqrt(1 / wo.sum() + var_mu)

    def show(name, m, s):
        print("  %-34s HR=%.3f  95%% CI [%.3f, %.3f]" %
              (name, np.exp(m), np.exp(m - 1.96 * s), np.exp(m + 1.96 * s)))

    print("Calibration: mu_hat=%.3f (HR bias %.3f), gamma^2_hat=%.4f" % (mu, np.exp(mu), gam2))
    print("RESULTS (HR = ACEI vs thiazide on acute MI):")
    show("Naive observational (illusion)", yo_pool, se_pool)
    show("Bias-corrected obs (no RCT)", yd, sd_d)
    show("Experimental only (ALLHAT)", ye, se_e)
    show("Calibrated EB (propagated)", th, sd_c)
    print("Benchmark (RCT truth): HR ~ 1.00")
