"""
Nonparametric calibrated empirical Bayes. The bias distribution g is left
unspecified and estimated from the calibration studies by nonparametric maximum
likelihood on a fixed grid using an EM algorithm. This is a Python grid
approximation to the Kiefer-Wolfowitz NPMLE, not a call to the ebnm R package. The causal parameter theta then has a
posterior that mixes over the fitted g.

This script (i) shows that the NPMLE recovers a bimodal bias distribution that a
Gaussian working prior cannot represent, and (ii) compares the coverage and mean
squared error of the Gaussian and nonparametric calibrated estimators under a
bimodal bias distribution. Writes figures/sim_nonparametric.pdf.

The equivalent R call is
    library(ebnm)
    fit <- ebnm(x = y_c, s = sigma_c, prior_family = "npmle")
    g   <- fit$fitted_g            # mixture of point masses (support, weights)
"""
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": ["serif"], "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 13, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 10, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
})
HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.abspath(os.environ.get("EB_ILLUSION_FIGDIR",
    os.path.join(HERE, "..", "figures")))
os.makedirs(FIGDIR, exist_ok=True)
Z = 1.959963985
rng = np.random.default_rng(7)

THETA, MU, SD_B, SO, SC, SE = 1.0, 0.5, 1.0, 1.0, 1.0, 1.0   # bias sd 1 => gamma^2 = 1


def npdf(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))


def draw_bimodal(size):
    # equal mixture of N(MU-1.3, .35^2) and N(MU+1.3, .35^2); mean MU, sd ~ 1.35
    comp = rng.integers(0, 2, size=size)
    return np.where(comp == 0, MU - 1.3, MU + 1.3) + 0.35 * rng.standard_normal(size)


def draw_gaussian(size):
    return MU + SD_B * rng.standard_normal(size)


def npmle(y, s, grid, iters=150):
    """Kiefer-Wolfowitz NPMLE weights over a fixed grid, by EM."""
    L = npdf(y[:, None], grid[None, :], s)            # (K, M)
    w = np.ones(grid.size) / grid.size
    for _ in range(iters):
        num = L * w[None, :]
        resp = num / (num.sum(1, keepdims=True) + 1e-300)
        w = resp.mean(0)
    return w


def theta_posterior(y_e, y_o, grid, w, se, so, tgrid):
    """Posterior mean and sd of theta under g = (grid, w), on a theta grid."""
    # log p(theta) = -0.5((ye-theta)/se)^2 + sum_j log sum_m w_m N(y_o,j; theta+grid_m, so)
    lp = -0.5 * ((y_e - tgrid) / se) ** 2                          # (T,)
    # mixture density of each obs at each theta: (T, J)
    dens = (w[None, None, :] *
            npdf(y_o[None, :, None], tgrid[:, None, None] + grid[None, None, :], so)).sum(2)
    lp = lp + np.log(dens + 1e-300).sum(1)
    lp -= lp.max()
    p = np.exp(lp); p /= np.trapz(p, tgrid)
    m = np.trapz(tgrid * p, tgrid)
    v = np.trapz((tgrid - m) ** 2 * p, tgrid)
    return m, np.sqrt(v)


def gaussian_ceb(y_e, y_o, y_c, se, so, sc):
    ybar = y_c.mean(); g2 = max(np.mean((y_c - ybar) ** 2) - sc ** 2, 0.0)
    mu = ybar; J = y_o.size
    we, wo = 1 / se ** 2, 1 / (so ** 2 + g2)
    den = we + J * wo
    th = (we * y_e + wo * np.sum(y_o - mu)) / den
    var_mu = (g2 + sc ** 2) / y_c.size
    c = (J * wo) / den
    return th, np.sqrt(1 / den + c ** 2 * var_mu)


# ---------- Panel (a): recover a bimodal bias distribution (single large calibration set) ----------
Kbig = 600
b = draw_bimodal(Kbig); yc = b + SC * rng.standard_normal(Kbig)
grid = np.linspace(yc.min() - 1, yc.max() + 1, 60)
w = npmle(yc, SC, grid)
xx = np.linspace(-3, 4, 400)
true_dens = 0.5 * npdf(xx, MU - 1.3, 0.35) + 0.5 * npdf(xx, MU + 1.3, 0.35)
gauss_dens = npdf(xx, yc.mean(), np.sqrt(max(yc.var() - SC ** 2, 1e-6)))

# ---------- Panel (b): point-estimation MSE, Gaussian vs NPMLE, under two bias shapes ----------
J = K = 100
R = 300
tgrid = np.linspace(THETA - 6, THETA + 6, 240)
draws = {"Gaussian bias": draw_gaussian, "bimodal bias": draw_bimodal}
summary = {}
for name, drawf in draws.items():
    eg, cg, en, cn = [], [], [], []
    for r in range(R):
        y_e = THETA + SE * rng.standard_normal()
        y_o = THETA + drawf(J) + SO * rng.standard_normal(J)
        y_c = drawf(K) + SC * rng.standard_normal(K)
        tg, sg = gaussian_ceb(y_e, y_o, y_c, SE, SO, SC)
        eg.append((tg - THETA) ** 2); cg.append(abs(tg - THETA) <= Z * sg)
        gr = np.linspace(y_c.min() - 1, y_c.max() + 1, 40)
        wv = npmle(y_c, SC, gr, iters=80)
        tn, sn = theta_posterior(y_e, y_o, gr, wv, SE, SO, tgrid)
        en.append((tn - THETA) ** 2); cn.append(abs(tn - THETA) <= Z * sn)
    summary[name] = dict(mse_g=np.mean(eg), cov_g=np.mean(cg),
                         mse_n=np.mean(en), cov_n=np.mean(cn))
    s = summary[name]
    print(f"{name}, J=K={J}, R={R}")
    print(f"  Gaussian EB : MSE={s['mse_g']:.4f}, coverage={s['cov_g']:.3f}")
    print(f"  NPMLE EB    : MSE={s['mse_n']:.4f}, coverage={s['cov_n']:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(xx, true_dens, color="0.35", lw=2, label="true bias density")
ax[0].plot(xx, gauss_dens, color="#C1121F", lw=2, ls="--", label="Gaussian fit")
ax[0].vlines(grid, 0, w / (grid[1] - grid[0]), color="#5B2C8D", alpha=0.9, lw=1.5,
             label="NPMLE $\\hat g$")
ax[0].set_title("(a) recovering the bias distribution")
ax[0].set_xlabel("bias $b$"); ax[0].set_ylabel("density"); ax[0].legend(frameon=False)
names = list(draws.keys())
xpos = np.arange(len(names)); wbar = 0.36
ax[1].bar(xpos - wbar/2, [summary[n]["mse_g"] for n in names], wbar, color="#C1121F", label="Gaussian EB")
ax[1].bar(xpos + wbar/2, [summary[n]["mse_n"] for n in names], wbar, color="#5B2C8D", label="NPMLE EB")
ax[1].set_xticks(xpos); ax[1].set_xticklabels(names)
ax[1].set_ylabel(r"MSE of $\hat\theta$"); ax[1].set_title(f"(b) accuracy ($J=K={J}$)")
ax[1].legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "sim_nonparametric.pdf"), bbox_inches="tight")
plt.close(fig)
print("saved sim_nonparametric.pdf")
