"""Plot the manuscript LEGEND illustration from rounded table summaries.
The faint per-database points are illustrative reconstructions, not an extract
of patient data or a recomputation from legend_htn_analysis.py. The ALLHAT
reference is a risk ratio for a different endpoint; empirical pooling requires
aligned estimands and verified inputs.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
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

rows = [("Calibrated empirical Bayes", 1.02, 0.98, 1.07),
        ("Naive observational (illusion)", 1.16, 1.12, 1.21),
        ("Experimental (ALLHAT)", 0.99, 0.91, 1.08)]
colors = ["#5B2C8D", "#0A6E31", "#C1121F"]
per_db = [1.17, 1.25, 1.20, 1.15, 1.10, 1.14, 1.26, 1.12, 1.12]
allhat, allhat_lo, allhat_hi = 0.99, 0.91, 1.08

fig, ax = plt.subplots(figsize=(8.2, 3.6))
ax.axvspan(allhat_lo, allhat_hi, color="#C1121F", alpha=0.12)
ax.axvline(allhat, color="#C1121F", lw=1.6, label="ALLHAT estimate (randomized benchmark)")
ax.axvline(1.0, color="0.7", lw=0.8, ls=":", label="no MI difference (null)")
ys = np.arange(len(rows))[::-1]
ax.scatter(per_db, np.full(len(per_db), ys[1]) + 0.16, s=14, color="#0A6E31", alpha=0.35, zorder=1)
for (name, m, lo, hi), yy, col in zip(rows, ys, colors):
    ax.errorbar(m, yy, xerr=[[m - lo], [hi - m]], fmt="o", color=col, capsize=3, lw=2, zorder=3)
ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows])
ax.set_xscale("log")
ax.set_xticks([0.9, 1.0, 1.1, 1.2, 1.3]); ax.set_xticklabels(["0.9", "1.0", "1.1", "1.2", "1.3"])
ax.set_xlabel("ratio, ACEI vs thiazide (illustrative comparison)")
ax.set_ylim(-0.6, len(rows) - 0.2)
ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "sim_legend.pdf"), bbox_inches="tight")
plt.close(fig)
