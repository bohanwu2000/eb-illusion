"""Shared manuscript plotting style, using sim_realdata as the reference.

Call apply_style() before creating a figure and finish_figure() before layout.
The latter accounts for the figure's LaTeX width so printed type sizes match
the reference (8.2-inch source, displayed at 0.8 times the text width).
This module changes presentation only; it never changes plotted data.
"""
import matplotlib as mpl
from cycler import cycler

PURPLE = "#5B2C8D"
GREEN = "#0A6E31"
RED = "#C1121F"
GRAY = "0.55"
LIGHT_GRAY = "0.7"
BLUE = "#0072B2"  # Extra categorical color when four conditions share a panel.
CEB_COLORS = (PURPLE, "#734398", "#8D5DAA", "#A275BB")
METHOD_COLORS = {"ceb": PURPLE, "gaussian_eb": PURPLE, "npmle": CEB_COLORS[2],
                 "cvci": BLUE, "illusion": GREEN, "experiment": RED}
METHOD_LABELS = {"ceb": "calibrated EB", "gaussian_eb": "calibrated EB",
                 "npmle": "calibrated EB (NPMLE)", "cvci": "CVCI",
                 "illusion": "illusion", "experiment": "experimental"}
# Shades distinguish settings of calibrated EB; method colors retain their meaning.
PRIOR_COLORS = CEB_COLORS
PRIOR_MARKERS = ("o", "s", "^", "D")

STYLE = {
    "font.family": ["serif"], "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "font.size": 11,
    "axes.labelsize": 13, "axes.titlesize": 13, "axes.titleweight": "normal",
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": "black", "axes.linewidth": .8, "axes.grid": False,
    "axes.spines.top": True, "axes.spines.right": True,
    "legend.frameon": False, "lines.linewidth": 2., "lines.markersize": 6.,
    "errorbar.capsize": 3., "xtick.direction": "out", "ytick.direction": "out",
    "axes.prop_cycle": cycler(color=[PURPLE, GREEN, RED, GRAY]),
}

def apply_style():
    mpl.rcParams.update(STYLE)

def finish_figure(fig, width_fraction=1.):
    """Match reference typography at the final manuscript size; retain axes/data."""
    scale = fig.get_figwidth() / (8.2 / .8 * width_fraction)
    for ax in fig.axes:
        ax.set_facecolor("white")
        ax.grid(False, which="both")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(.8 * scale)
            spine.set_color("black")
        for line in ax.lines:
            if not hasattr(line, "_paper_sizes"):
                line._paper_sizes = (line.get_linewidth(), line.get_markersize(),
                                     line.get_markeredgewidth())
            linewidth, markersize, markeredgewidth = line._paper_sizes
            line.set_linewidth(linewidth * scale)
            line.set_markersize(markersize * scale)
            line.set_markeredgewidth(markeredgewidth * scale)
        for collection in ax.collections:
            if not hasattr(collection, "_paper_linewidths"):
                collection._paper_linewidths = collection.get_linewidths().copy()
            collection.set_linewidths(collection._paper_linewidths * scale)
        for label in (ax.xaxis.label, ax.yaxis.label, ax.title):
            label.set_fontfamily("DejaVu Serif")
            label.set_fontsize(13 * scale)
        ax.tick_params(axis="both", which="both", labelsize=11 * scale, direction="out",
                       width=.8 * scale, length=3.5 * scale)
        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_fontfamily("DejaVu Serif")
        for axis in (ax.xaxis, ax.yaxis):
            axis.offsetText.set_fontsize(11 * scale)
        legend = ax.get_legend()
        if legend is not None:
            legend.set_frame_on(False)
            for text in legend.get_texts():
                text.set_fontfamily("DejaVu Serif")
                text.set_fontsize(10 * scale)
    for legend in fig.legends:
        legend.set_frame_on(False)
        for text in legend.get_texts():
            text.set_fontfamily("DejaVu Serif")
            text.set_fontsize(10 * scale)
