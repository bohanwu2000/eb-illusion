"""Forest plot for the water-consumption example (figures/sim_realdata.pdf).

Means and 95% interval endpoints are copied directly from
Table tab:illusion_vs_noillusion, retaining their reported rounding.
The layout, typography, and colors match legend_htn_figure.py; the causal
effect is shown on a linear scale because its values can be negative.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.abspath(os.environ.get("EB_ILLUSION_FIGDIR",
    os.path.join(HERE, "..", "figures")))
os.makedirs(FIGDIR, exist_ok=True)

# Point summaries and interval endpoints from tab:illusion_vs_noillusion.
rows = [("Calibrated empirical Bayes", -0.58, -0.90, -0.27),
        ("Illusion posterior", -1.40, -1.84, -0.97),
        ("Experimental (full)", -0.48, -0.86, -0.10),
        ("Experimental (reduced)", -1.40, -4.89, 2.08)]
colors = ["#5B2C8D", "#0A6E31", "#C1121F", "0.55"]
full_mean, full_lo, full_hi = rows[2][1:]

style = {
    "font.family": ["serif"], "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 13, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 10, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
}
with plt.rc_context(style):
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.axvspan(full_lo, full_hi, color="#C1121F", alpha=0.12)
    ax.axvline(full_mean, color="#C1121F", lw=1.6,
               label="full-experiment estimate")
    ax.axvline(0.0, color="0.7", lw=0.8, ls=":", label="no effect (null)")
    ys = np.arange(len(rows))[::-1]
    for (name, mean, lo, hi), yy, color in zip(rows, ys, colors):
        ax.errorbar(mean, yy, xerr=[[mean - lo], [hi - mean]],
                    fmt="o", color=color, capsize=3, lw=2, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([row[0] for row in rows])
    ax.set_xlabel(r"causal effect $\theta$")
    ax.set_ylim(-0.6, len(rows) - 0.2)
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "sim_realdata.pdf"), bbox_inches="tight")
    plt.close(fig)
print("saved sim_realdata.pdf (reported means and 95% endpoints preserved)")
