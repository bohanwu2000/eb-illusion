"""Revised simulations: python3 simulation_restructure.py [--plot-only].

Fixed in advance: seed 20260930, replicate zero for interval displays,
10,000 MC replicates, nested count grid and independent data streams.
Uses the existing moment-matching and CVCI estimators; the first-replicate
CVCI interval uses 999 bootstrap resamples with weight selection repeated.
Existing caches are read-only. Capped results and summaries are saved in the
results directory. Set EB_ILLUSION_FIGDIR explicitly before running.
"""
import argparse
import json
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
FIGDIR = Path(os.environ.setdefault("EB_ILLUSION_FIGDIR", str(HERE.parent / "figures"))).resolve()
from referee_simulations import (ceb_estimator, draw_bias, cvci_mean_estimator,
    bootstrap_experimental_moments, cvci_bootstrap_sd, THETA, MU, GAM2, Z)
from paper_plot_style import (apply_style, finish_figure, PURPLE, GREEN, RED, GRAY,
                              PRIOR_COLORS, PRIOR_MARKERS)

apply_style()

MAX_COUNT = 100
COUNTS = np.array([5, 10, 50, MAX_COUNT])
FIXED = 10
PRIORS = ("gaussian", "skewed", "heavy", "bimodal")
LABELS = ("Gaussian", "Skewed", r"Heavy-tailed ($t_3$)", "Bimodal")
COLORS = PRIOR_COLORS
CASES = ("vary_k", "vary_j", "together")
RESULT_PATH = HERE.parent / "results/simulations/finite_studies.npz"
OUTPUT_DIR = HERE.parent / "results"
SUMMARY_PATH = OUTPUT_DIR / "finite_studies_summary.json"

def capped_results(r):
    """Select displayed counts in memory; keep the complete source cache intact."""
    counts = np.asarray(r["counts"])
    keep = (counts > 0) & (counts <= MAX_COUNT)
    if not keep.any() or not 0 < int(r["fixed"]) <= MAX_COUNT:
        raise ValueError("The displayed study counts must be between 1 and 100.")
    out = dict(r)
    out["counts"] = counts[keep]
    for prior in PRIORS:
        for case in CASES:
            key = prior + "_" + case
            out[key] = r[key][keep]
    return out


def bias(rng, shape, prior):
    if prior == "bimodal":
        # Population variance 1.001; analytic scaling preserves independence.
        signs = 2*rng.integers(0, 2, size=shape)-1
        return MU + np.sqrt(GAM2/1.001)*(signs+rng.normal(0, np.sqrt(.001), size=shape))
    return draw_bias(rng, shape, MU, GAM2, prior)

def generate(replicates, seed):
    result = dict(counts=COUNTS, fixed=np.array(FIXED), seed=np.array(seed), replicates=np.array(replicates))
    for d, prior in enumerate(PRIORS):
        streams = [np.random.default_rng(np.random.SeedSequence([seed,d,c])) for c in range(5)]
        shape = (replicates, int(COUNTS.max()))
        ye = streams[0].normal(THETA, 1., size=replicates)
        yo = THETA + bias(streams[1], shape, prior)
        yo += streams[2].normal(size=shape)
        yc = bias(streams[3], shape, prior)
        yc += streams[4].normal(size=shape)
        result[prior+"_ye"] = ye
        result[prior+"_first_yo"] = yo[0].copy()
        result[prior+"_first_yc"] = yc[0].copy()
        for case in CASES:
            rows = []
            for n in COUNTS:
                j = FIXED if case=="vary_k" else int(n)
                k = FIXED if case=="vary_j" else int(n)
                rows.append(np.stack(ceb_estimator(ye, yo[:,:j], yc[:,:k], 1.,1.,1.)))
            # axes: count, (mean, plug-in SD, propagated SD, gamma2, mu), replicate
            result[prior+"_"+case] = np.stack(rows)
        print(prior, replicates, "replicates complete", flush=True)
    return result

def mse(estimates):
    return np.mean((estimates-THETA)**2, axis=-1)

def mcse(estimates):
    # MC SE of the mean squared error; used only with Gaussian inputs.
    sq = (estimates-THETA)**2
    return sq.std(axis=-1,ddof=1)/np.sqrt(sq.shape[-1])

def count_axis(ax, symbol, counts):
    ax.set_xscale("log")
    ax.set_xticks(counts)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel("number of "+("calibration" if symbol=="K" else "observational")+" studies $"+symbol+"$")

def save(fig, name):
    FIGDIR.mkdir(parents=True,exist_ok=True)
    fig.savefig(FIGDIR/(name+".pdf"),bbox_inches="tight")
    fig.savefig(FIGDIR/(name+".png"),dpi=180,bbox_inches="tight")
    plt.close(fig)

def plot_finite(r):
    r=capped_results(r)
    n=r["counts"]
    fixed=int(r["fixed"])
    ye=float(r["gaussian_ye"][0])
    fig,axes=plt.subplots(2,2,figsize=(11.6,7.8))
    for row,(case,symbol,fixed_label) in enumerate((("vary_k","K",rf"$J={fixed}$"),("vary_j","J",rf"$K={fixed}$"))):
        a=r["gaussian_"+case]
        ax=axes[row,0]
        ax.axhspan(ye-Z,ye+Z,color=RED,alpha=.12)
        ax.axhline(ye,ls="--",color=RED,label="experimental")
        ax.axhline(THETA,ls=":",color="black",label=r"true effect $\theta^\star$")
        ax.errorbar(n,a[:,0,0],yerr=Z*a[:,1,0],fmt="o-",color=COLORS[0],capsize=3,
                    label="calibrated EB")
        ax.set_ylabel(r"effect estimate $\hat\theta$")
        ax.set_title(f"({chr(97+2*row)}) {fixed_label}")
        count_axis(ax,symbol,n)
        ax=axes[row,1]
        ax.errorbar(n,mse(a[:,0]),yerr=Z*mcse(a[:,0]),fmt="o-",color=COLORS[0],capsize=3,label="calibrated EB")
        ax.axhline(1,ls="--",color=RED,label="experimental")
        ax.set_ylabel("MSE")
        ax.set_ylim(0,1.07)
        ax.set_title(f"({chr(98+2*row)}) estimation error ({fixed_label})")
        count_axis(ax,symbol,n)
    axes[0,0].legend(frameon=False,fontsize=9)
    axes[0,1].legend(frameon=False,fontsize=10)
    lo=min(axes[0,0].get_ylim()[0],axes[1,0].get_ylim()[0])
    hi=max(axes[0,0].get_ylim()[1],axes[1,0].get_ylim()[1])
    for ax in axes[:,0]:
        ax.set_ylim(lo,hi)
    finish_figure(fig)
    fig.tight_layout(pad=1.1,h_pad=2.)
    save(fig,"sim_finite_k")




def plot_priors(r):
    r=capped_results(r)
    n=r["counts"]
    fixed=int(r["fixed"])
    fixed_index=int(np.flatnonzero(n==fixed)[0])
    fig,axes=plt.subplots(1,3,figsize=(14.,4.3))
    # Literal replicate zero, as in plot_finite; these are posterior intervals,
    # not Monte Carlo confidence intervals for the MSE.
    for x,(prior,color,marker) in enumerate(zip(PRIORS,COLORS,PRIOR_MARKERS)):
        a=r[prior+"_vary_k"][fixed_index]
        axes[0].errorbar(x,a[0,0],yerr=Z*a[1,0],fmt=marker,color=color,capsize=3)
    axes[0].set_xticks(np.arange(4),["Gaussian","Skewed",r"$t_3$","Bimodal"])
    axes[0].tick_params(axis="x",labelrotation=20)
    axes[0].set_xlim(-.5,3.5)
    axes[0].set_xlabel("true bias distribution")
    axes[0].set_ylabel(r"effect estimate $\hat\theta$")
    axes[0].set_title(rf"(a) $J=K={fixed}$")
    truth=axes[0].axhline(THETA,color="black",ls=":",label=r"true effect $\theta^\star$")
    for ax,case,symbol,title in ((axes[1],"vary_j","J",rf"(b) varying $J$ ($K={fixed}$)"),
                                (axes[2],"vary_k","K",rf"(c) varying $K$ ($J={fixed}$)")):
        for d,label,color,marker in zip(PRIORS,LABELS,COLORS,PRIOR_MARKERS):
            ax.plot(n,mse(r[d+"_"+case][:,0]),marker+"-",color=color,label=label)
        count_axis(ax,symbol,n)
        ax.set_xlabel("number of "+("observational" if symbol=="J" else "calibration")+"\nstudies $"+symbol+"$")
        ax.set_title(title)
        ax.set_ylabel("MSE")
        ax.axhline(1,ls="--",color=RED,label="experimental")
        ax.set_ylim(0,1.07)
    handles,labels=axes[1].get_legend_handles_labels()
    handles.append(truth)
    labels.append(truth.get_label())
    fig.legend(handles,labels,loc="lower center",ncol=6,frameon=False,fontsize=10,bbox_to_anchor=(.5,-.005))
    finish_figure(fig)
    fig.tight_layout(rect=(0,.08,1,1))
    save(fig,"sim_misspec")

def plot_coverage(r):
    r=capped_results(r)
    n=r["counts"]
    fixed=int(r["fixed"])
    fig,axes=plt.subplots(1,2,figsize=(10.4,4.2))
    for ax,case,symbol,title in ((axes[0],"vary_k","K",rf"(a) varying $K$ ($J={fixed}$)"),
                                (axes[1],"vary_j","J",rf"(b) varying $J$ ($K={fixed}$)")):
        a=r["gaussian_"+case]
        for index,label,color,style in ((1,"posterior credible interval",PURPLE,"o-"),(2,"propagated uncertainty",PRIOR_COLORS[2],"s--")):
            ax.plot(n,np.mean(np.abs(a[:,0]-THETA)<=Z*a[:,index],axis=-1),style,color=color,label=label)
        ax.axhline(.95,ls=":",color=GRAY,label="nominal 0.95")
        count_axis(ax,symbol,n)
        ax.set_ylim(0,1)
        ax.set_ylabel("coverage of 95% interval")
        ax.set_title(title)
    axes[0].legend(frameon=False,fontsize=10)
    finish_figure(fig)
    fig.tight_layout()
    save(fig,"sim_finite_k_coverage")
    fig,ax=plt.subplots(figsize=(6.,4.))
    for prior,label,color,marker in zip(PRIORS,LABELS,COLORS,PRIOR_MARKERS):
        a=r[prior+"_together"]
        cov=np.mean(np.abs(a[:,0]-THETA)<=Z*a[:,2],axis=-1)
        ax.plot(n,cov,marker+"-",color=color,label=label)
    ax.axhline(.95,ls=":",color=GRAY)
    count_axis(ax,"J",n)
    ax.set_xlabel(r"number of studies $J=K$")
    ax.set_ylabel("coverage of 95% interval")
    ax.set_ylim(.7,1)
    ax.set_title("Propagated uncertainty under different priors")
    ax.legend(frameon=False,fontsize=10)
    finish_figure(fig,width_fraction=.72)
    fig.tight_layout()
    save(fig,"sim_misspec_coverage")

def summarize(r):
    r=capped_results(r)
    out=dict(seed=int(r["seed"]),replicates=int(r["replicates"]),counts=r["counts"].tolist(),
             fixed_count=FIXED,theta=THETA,bias_mean=MU,bias_variance=GAM2,sigma_e=1,sigma_o=1,sigma_c=1,
             interval_example="replicate 0; nested data prefixes",interval_type="95% Gaussian plug-in posterior credible interval",
             mse_error_bars="Gaussian panels only: +/-1.96 MC SE of squared errors",
             prior_error_bars="omitted: t3 has no finite fourth moment",
             bimodal_distribution="0.5 + (sign + N(0,0.001))/sqrt(1.001)")
    for prior in PRIORS:
        out[prior]={}
        for case in CASES:
            a=r[prior+"_"+case]
            out[prior][case]=dict(mse=mse(a[:,0]).tolist(),
                coverage_plugin=np.mean(np.abs(a[:,0]-THETA)<=Z*a[:,1],axis=-1).tolist(),
                coverage_propagated=np.mean(np.abs(a[:,0]-THETA)<=Z*a[:,2],axis=-1).tolist(),
                first_estimate=a[:,0,0].tolist(),first_posterior_sd=a[:,1,0].tolist())
            if prior=="gaussian":
                out[prior][case]["mse_mc_se"]=mcse(a[:,0]).tolist()
    return out

def verify(r):
    assert int(r["fixed"])==FIXED, "Saved results use an old fixed number of studies; regenerate them."
    counts=np.asarray(r["counts"])
    assert counts.ndim==1 and np.all(counts>0) and np.all(np.diff(counts)>0)
    assert np.all(counts==counts.astype(int)) and FIXED in counts
    fixed_index=int(np.flatnonzero(r["counts"]==FIXED)[0])
    for prior in PRIORS:
        for case in CASES:
            a=r[prior+"_"+case]
            assert a.shape==(len(counts),5,int(r["replicates"]))
            assert np.isfinite(a).all()
            assert (a[:,1:3]>0).all() and (a[:,3]>=0).all()
            np.testing.assert_array_equal(r[prior+"_vary_k"][fixed_index],a[fixed_index])
            for i,n in enumerate(counts):
                j=FIXED if case=="vary_k" else int(n)
                k=FIXED if case=="vary_j" else int(n)
                yc=r[prior+"_first_yc"][:k]
                yo=r[prior+"_first_yo"][:j]
                g2=max(np.mean((yc-yc.mean())**2)-1,0)
                den=1+j/(1+g2)
                th=(r[prior+"_ye"][0]+(yo-yc.mean()).sum()/(1+g2))/den
                np.testing.assert_allclose(a[i,0,0],th,rtol=1e-13)
                np.testing.assert_allclose(a[i,1,0],1/np.sqrt(den),rtol=1e-13)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates",type=int,default=10000)
    parser.add_argument("--seed",type=int,default=20260930)
    parser.add_argument("--plot-only",action="store_true")
    parser.add_argument("--input-results",type=Path,default=RESULT_PATH)
    parser.add_argument("--results",type=Path,default=OUTPUT_DIR / "simulation_restructure_count100_results.npz")
    parser.add_argument("--summary",type=Path,default=SUMMARY_PATH)
    args=parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    if args.results.resolve()==args.input_results.resolve():
        parser.error("Choose a separate output to preserve the input cache.")
    if args.plot_only:
        with np.load(args.input_results) as data: r=dict(data)
    else:
        if args.replicates<2: parser.error("at least two replicates required")
        if args.results.exists():
            parser.error("Reuse --plot-only or choose a new results path to preserve simulations.")
        r=generate(args.replicates,args.seed)
    verify(r)
    if not args.plot_only:
        np.savez_compressed(args.results,**r)
    summary=summarize(r)
    summary["source_counts"]=r["counts"].tolist()
    summary["plot_count_cap"]=MAX_COUNT
    summary["source_results"]=str(args.input_results.resolve() if args.plot_only else args.results.resolve())
    args.summary.parent.mkdir(parents=True,exist_ok=True)
    args.summary.write_text(json.dumps(summary,indent=2)+"\n")
    plot_finite(r)
    plot_priors(r)
    plot_coverage(r)
    print("Verified finite outputs, nested-count agreement, and first-replicate formulas.")
    print("Gaussian MSE vs K:",summary["gaussian"]["vary_k"]["mse"])
    print("Gaussian MSE vs J:",summary["gaussian"]["vary_j"]["mse"])
    print("Displayed counts:",summary["counts"])
    print("Saved summary:",args.summary)

if __name__=="__main__":
    main()
