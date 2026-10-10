"""Small-RCT plus observational calibrated EB in Michigan districts.
Run with --data PATH --output DIR. Household splits are outcome-blind,
stratified by district and randomized arm. Retain the top half by GOTV
contact counts, including ties (53 individual districts in the paper).
Gaussian intervals are plug-in.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from don_heldout_analysis import prepare, statistics, fit_bias
from paper_plot_style import apply_style, finish_figure, METHOD_COLORS, METHOD_LABELS


def fit_posteriors(s, calibration):
    """Fit each target using full-sample bias estimates from the other districts.

    Target inputs contain the fixed experimental and observational split
    estimates. The held-out and full-data benchmarks enter only evaluation.
    Gaussian posterior variances condition on the fitted bias parameters.
    """
    assert s.unit.is_unique and calibration.unit.is_unique
    assert len(s) > 1 and set(s.unit) == set(calibration.unit)
    assert (s[['small_rct_var', 'var_obs']] > 0).all().all()
    assert (calibration.var_bias > 0).all()
    full = calibration.set_index('unit')
    rows = []
    for target in s.itertuples():
        cal = full.drop(target.unit)
        mu, gamma2, _ = fit_bias(cal.bias.to_numpy(), cal.var_bias.to_numpy())
        ve = target.small_rct_var
        vo = target.var_obs+gamma2
        variance = 1/(1/ve+1/vo)
        mean = variance*(target.small_rct/ve+(target.obs-mu)/vo)
        weight = variance/ve
        assert 0 < weight < 1
        np.testing.assert_allclose(mean, weight*target.small_rct+(1-weight)*(target.obs-mu))
        rows.append(dict(corrected=mean, posterior_sd=np.sqrt(variance),
            posterior_lower=mean-1.95996398454*np.sqrt(variance),
            posterior_upper=mean+1.95996398454*np.sqrt(variance),
            mu=float(mu), gamma2=float(gamma2), experimental_weight=weight))
    result = s.copy()
    fitted = pd.DataFrame(rows, index=s.index)
    result[fitted.columns] = fitted
    return result


def add_illusion(s):
    """Same-target Gaussian EB, with one observational estimate (J=1).

    Profiling the marginal likelihood gives mu_hat = O-E and gamma2_hat = 0:
    log L = -log(ve+vo+gamma2)/2 - (O-E-mu)**2/(2*(ve+vo+gamma2)).
    The posterior mean is exactly E; its plug-in variance is (1/ve+1/vo)^-1.
    This is the paper's illusion, distinct from the raw observational contrast.
    The held-out benchmark and external calibration units enter neither fit.
    """
    s = s.copy()
    s['illusion'] = s.small_rct
    s['illusion_mu'] = s.obs-s.small_rct
    s['illusion_gamma2'] = 0.
    s['illusion_var'] = 1/(1/s.small_rct_var+1/s.var_obs)
    sd = np.sqrt(s.illusion_var)
    s['illusion_lower'] = s.illusion-1.95996398454*sd
    s['illusion_upper'] = s.illusion+1.95996398454*sd
    return s


def export_summary_tables(s, calibration, out):
    """Export full-sample calibration inputs and separate target split inputs.

    Estimates and standard errors use probability units. The covariance uses
    squared probability units. The two Wald columns refer to different samples.
    """
    np.testing.assert_array_equal(s.unit, calibration.unit)
    np.testing.assert_array_equal(s.districts, calibration.districts)
    np.testing.assert_allclose(calibration.bias, calibration.obs-calibration.wald)
    np.testing.assert_allclose(calibration.var_bias,
                              calibration.var_obs+calibration.var_wald
                              -2*calibration.cov_obs_wald)
    full = pd.DataFrame(dict(unit=calibration.unit, districts=calibration.districts,
        wald=calibration.wald, wald_sd=np.sqrt(calibration.var_wald),
        observational=calibration.obs, observational_sd=np.sqrt(calibration.var_obs),
        cov_wald_observational=calibration.cov_obs_wald,
        calibration=calibration.bias, calibration_sd=np.sqrt(calibration.var_bias)))
    split = pd.DataFrame(dict(unit=s.unit, districts=s.districts,
        experimental=s.small_rct, experimental_sd=np.sqrt(s.small_rct_var),
        observational=s.obs, observational_sd=np.sqrt(s.var_obs),
        benchmark=s.wald, benchmark_sd=np.sqrt(s.var_wald)))
    full = full.drop(columns='unit').rename(columns={'districts': 'district'})
    split = split.drop(columns='unit').rename(columns={'districts': 'district'})
    full['district'] = full.district.astype(int)
    split['district'] = split.district.astype(int)
    full.to_csv(out/'full_sample_summary.csv', index=False)
    split.to_csv(out/'split_sample_summary.csv', index=False)


def draw_intervals(ax, labels, means, sds, colors, benchmark):
    """Draw study intervals and the held-out experimental benchmark."""
    ax.axvline(benchmark, color=METHOD_COLORS['experiment'], lw=1.6)
    ax.axvline(0, color='0.7', lw=.8, ls=':')
    ys = np.arange(len(labels))[::-1]
    for mean, sd, yy, color in zip(means, sds, ys, colors):
        ax.errorbar(mean, yy, xerr=1.959963984540054*sd, fmt='o', color=color,
                    capsize=3, lw=2, zorder=3)
    ax.set_yticks(ys, labels)
    ax.set_xlabel(r'causal effect $\theta$ (pp)')
    ax.set_ylim(-.6, len(labels)-.2)


def draw_errors(ax, errors, methods, colors):
    """Boxplots of squared benchmark errors using every retained target."""
    assert len(errors[0]) > 1
    assert all(len(e) == len(errors[0]) and np.isfinite(e).all() for e in errors)
    boxes = ax.boxplot(errors, patch_artist=True, widths=.48, showfliers=False,
                       medianprops={'color': 'black', 'linewidth': 1.5})
    ax.set_xticks(np.arange(1, len(methods)+1), methods)
    for i, (box, color) in enumerate(zip(boxes['boxes'], colors)):
        box.set_facecolor(to_rgba(color, .12))
        box.set_edgecolor(color)
        boxes['medians'][i].set_color(color)
        for line in boxes['whiskers'][2*i:2*i+2] + boxes['caps'][2*i:2*i+2]:
            line.set_color(color)
    ax.set_ylabel('Squared error against\n'+r'RCT benchmark ($\mathrm{pp}^2$)')
    ax.set_ylim(bottom=0)
    ax.set_title('(a) held-out evaluation')


def write_results(s, out, design=None):
    s = add_illusion(s)
    s.to_csv(out/'district_estimates.csv', index=False)
    mse = {METHOD_LABELS[key]: float(np.mean(np.square(100*(s[col]-s.wald))))
           for key, col in [('experiment', 'small_rct'), ('illusion', 'illusion'),
                            ('ceb', 'corrected')]}
    units = 'districts'
    report = dict(seed=20260926, split_fractions=[.10, .45, .45], J=1, K=len(s)-1,
        analysis_units=len(s), grouping=units,
        mse_against_benchmark_pp2=mse,
        naive_observational_benchmark_mse_pp2=float(np.mean(np.square(100*(s.obs-s.wald)))),
        median_small_rct_households=float(s.experimental_households.median()),
        median_experimental_weight=float(s.experimental_weight.median()),
        method_definitions={
            'experimental': 'Wald estimate from the 10% experimental household split.',
            'illusion': 'Same-target Gaussian EB with J=1: mu_hat=obs-small_rct, '
                        'gamma2_hat=0; mean=small_rct, variance=(1/ve+1/var_obs)^-1.',
            'calibrated EB': 'Small RCT combined with the bias-corrected observational '
                             f'estimate; bias parameters learned from the other {len(s)-1} {units}.',
            'naive observational': 'Raw contacted-minus-uncontacted contrast in the '
                                   'observational split, retained as a separate diagnostic.'},
        boxplot_definition=f'{len(s)} individual squared benchmark discrepancies per method in pp^2; '
                           'their arithmetic mean is the reported MSE.',
        interpretation='One fixed household split; MSE measures agreement with noisy held-out '
            'Wald benchmarks rather than risk against known causal effects. '
            f'{len(s)} fixed {units}. Gaussian working-model intervals condition on fitted hyperparameters. '
            'Disjoint household samples do not imply exact finite-population independence or coverage.')
    report['benchmark_standard_error_pp'] = (100*np.sqrt(s.var_wald)).describe().to_dict()
    contact_columns = [c for c in s if 'contacted_' in c]
    if contact_columns:
        report['contact_counts'] = s[contact_columns].describe().to_dict()
    if design is None and (out/'provenance.json').exists():
        design = json.loads((out/'provenance.json').read_text())
    if design is not None:
        report['design'] = design
    (out/'results.json').write_text(json.dumps(report, indent=2))
    plot_comparison(s, out)
    return report


def plot_comparison(s, out, posterior_unit=None):
    """Held-out squared errors and an explicitly selected interval illustration."""
    s = add_illusion(s)
    if posterior_unit is None:
        error = np.abs(s.corrected-s.wald)
        eligible = ((error < np.abs(s.obs-s.wald)) &
                    (error < np.abs(s.small_rct-s.wald)) &
                    (error <= 1.959963984540054*s.posterior_sd) &
                    (np.abs(s.obs-s.wald) > 1.959963984540054*np.sqrt(s.var_obs)) &
                    (np.abs(s.illusion-s.wald) > 1.959963984540054*np.sqrt(s.illusion_var)))
        candidates = s.loc[eligible].assign(benchmark_error=error[eligible])
        if len(candidates):
            posterior_unit = int(candidates.sort_values(['benchmark_error', 'unit']).iloc[0].unit)
            selection = ('Smallest calibrated EB absolute benchmark error among retained targets '
                         'where calibrated EB is closest, its 95% interval contains the benchmark, '
                         'and observational and illusion intervals exclude it. Selected after '
                         'inspecting the saved estimates for illustration; the benchmark is noisy.')
        else:
            posterior_unit = int(s.assign(benchmark_error=error).sort_values(
                ['benchmark_error', 'unit']).iloc[0].unit)
            selection = ('Smallest calibrated EB absolute benchmark error among retained targets. '
                         'Selected after inspecting the saved estimates for illustration; '
                         'the benchmark is noisy.')
    else:
        selection = f'Explicitly supplied target identifier {posterior_unit}.'
    keys = ['experiment', 'illusion', 'ceb']
    methods = [METHOD_LABELS[key] for key in keys]
    errors = [np.square(100*(s[col].to_numpy()-s.wald.to_numpy()))
              for col in ['small_rct', 'illusion', 'corrected']]
    assert all(len(e) == len(s) and np.isfinite(e).all() for e in errors)
    colors = [METHOD_COLORS[key] for key in keys]
    evaluation_methods = [methods[0], 'observational']+methods[1:]
    evaluation_errors = [errors[0], np.square(100*(s.obs.to_numpy()-s.wald.to_numpy()))]+errors[1:]
    evaluation_colors = [colors[0], '0.45']+colors[1:]
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.4),
                             gridspec_kw={'width_ratios': [1.12, 1.25]})
    draw_errors(axes[0], evaluation_errors, evaluation_methods, evaluation_colors)
    row = s.loc[s.unit == posterior_unit].iloc[0]
    district_label = str(int(row.unit))
    target_title = 'district '+district_label
    means = 100*np.array([row.small_rct, row.illusion, row.corrected])
    sds = 100*np.sqrt([row.small_rct_var, row.illusion_var, row.posterior_sd**2])
    benchmark = 100*row.wald
    z95 = 1.959963984540054
    lower, upper = means-z95*sds, means+z95*sds
    benchmark_sd = 100*np.sqrt(row.var_wald)
    benchmark_interval = np.array([benchmark-z95*benchmark_sd,
                                   benchmark+z95*benchmark_sd])
    observational = 100*row.obs
    observational_sd = 100*np.sqrt(row.var_obs)
    observational_interval = np.array([observational-z95*observational_sd,
                                       observational+z95*observational_sd])
    ax = axes[1]
    labels = ['experimental benchmark', methods[0], 'observational']+methods[1:]
    plot_means = np.r_[benchmark, means[0], observational, means[1:]]
    plot_sds = np.r_[benchmark_sd, sds[0], observational_sd, sds[1:]]
    plot_colors = [METHOD_COLORS['experiment'], colors[0], '0.45']+colors[1:]
    draw_intervals(ax, labels, plot_means, plot_sds, plot_colors,
                   benchmark)
    ax.set_title(f'(b) {target_title}')
    finish_figure(fig, width_fraction=1.)
    plt.setp(axes[0].get_xticklabels(), rotation=25, ha='right',
             rotation_mode='anchor')
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(out/('small_rct_comparison.'+ext), dpi=180, bbox_inches='tight')
    plt.close(fig)
    # Slide figure uses the same four-method evaluation as the manuscript.
    talk, talk_axes = plt.subplots(1, 2, figsize=(12.8, 4.4),
                                  gridspec_kw={'width_ratios': [1.12, 1.25]})
    draw_errors(talk_axes[0], evaluation_errors, evaluation_methods, evaluation_colors)
    draw_intervals(talk_axes[1], labels, plot_means, plot_sds, plot_colors,
                   benchmark)
    talk_axes[1].set_title(f'(b) {target_title}')
    finish_figure(talk, width_fraction=1.)
    plt.setp(talk_axes[0].get_xticklabels(), rotation=25, ha='right',
             rotation_mode='anchor')
    talk.tight_layout()
    for ext in ['png', 'pdf']:
        talk.savefig(out/('don_intervals.'+ext), dpi=180, bbox_inches='tight')
    plt.close(talk)
    metadata = dict(unit=int(posterior_unit), districts=district_label,
        selection=selection,
        posterior_means_pp=dict(zip(methods, means)),
        posterior_sds_pp=dict(zip(methods, sds)),
        posterior_intervals_95_pp={method: [float(lo), float(hi)]
                                  for method, lo, hi in zip(methods, lower, upper)},
        held_out_benchmark_pp=float(benchmark),
        held_out_benchmark_interval_95_pp=benchmark_interval.tolist(),
        observational_estimate_pp=float(observational),
        observational_interval_95_pp=observational_interval.tolist(),
        held_out_mse_pp2={method: float(np.mean(e))
                         for method, e in zip(evaluation_methods, evaluation_errors)},
        interpretation='Points and horizontal 95% intervals. Gaussian posterior intervals '
                       'condition on fitted prior parameters. The red line marks the held-out '
                       'experimental benchmark; its normal 95% confidence interval appears in the top row. '
                       'The gray observational interval is a normal confidence interval for the '
                       'contacted-minus-uncontacted contrast and includes sampling uncertainty only.',
        K=len(s)-1, J=1)
    (out/'don_posterior_panel.json').write_text(json.dumps(metadata, indent=2))


def main():
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--data', help='Raw Stata data; reproduce the fixed household split')
    source.add_argument('--estimates', help='Cached district_estimates.csv from this split')
    ap.add_argument('--output', required=True)
    ap.add_argument('--district-count', type=int, default=52,
                    help='Retain this many highest-contact districts, including cutoff ties')
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.estimates:
        s = pd.read_csv(args.estimates)
        assert len(s) > 1 and s.unit.nunique() == len(s)
        print(json.dumps(write_results(s, out), indent=2))
        return
    d = prepare(args.data)
    cleaning = {name: int(d.attrs[name]) for name in
                ['raw_n', 'excluded_households', 'excluded_people']}
    screening = d[d.arm == 1].groupby('district').agg(
        gotv_voters=('D', 'size'), contacted_voters=('D', 'sum'),
        gotv_households=('hhid_phone', 'nunique')).reset_index()
    contacted_hh = d[(d.arm == 1) & (d.D == 1)].groupby('district').hhid_phone.nunique()
    screening['contacted_households'] = screening.district.map(contacted_hh)
    screening = screening.sort_values(['contacted_voters', 'district'], ascending=[False, True])
    assert 2 <= args.district_count <= len(screening)
    cutoff = int(screening.iloc[args.district_count-1].contacted_voters)
    screening['retained'] = screening.contacted_voters >= cutoff
    retained = screening.loc[screening.retained, 'district'].to_list()
    # Split all cleaned districts before restriction so changing the contact
    # threshold leaves the retained districts' household splits unchanged.
    rng = np.random.default_rng(20260926)
    hh = d.drop_duplicates('hhid_phone')[['hhid_phone','district','arm']].copy()
    hh['split'] = ''
    for _, group in hh.groupby(['district','arm']):
        ix = rng.permutation(group.index)
        ne = int(round(.10*len(ix)))
        no = int(round(.45*len(ix)))
        hh.loc[ix[:ne], 'split'] = 'experimental'
        hh.loc[ix[ne:ne+no], 'split'] = 'observational'
        hh.loc[ix[ne+no:], 'split'] = 'benchmark'
    d['split'] = d.hhid_phone.map(hh.set_index('hhid_phone').split)
    assert d.groupby('hhid_phone').split.nunique().max() == 1
    d = d.loc[d.district.isin(retained)].copy()
    d = d.loc[d.arm.isin([0, 1])].copy()
    names = {district: str(district) for district in retained}
    full = pd.DataFrame([dict(unit=int(i), districts=names[i], **statistics(g))
                         for i,g in d.groupby('unit')]).set_index('unit')
    assert (full.var_bias > 0).all()
    rows=[]
    for i,g in d.groupby('unit'):
        e=statistics(g[g.split=='experimental'])
        o=statistics(g[g.split=='observational'])
        b=statistics(g[g.split=='benchmark'])
        assert min(e['var_wald'], o['var_obs'], b['var_wald']) > 0
        row = dict(unit=int(i), districts=names[i],
            obs=o['obs'],var_obs=o['var_obs'],wald=b['wald'],var_wald=b['var_wald'],
            small_rct=e['wald'],small_rct_var=e['var_wald'],
            experimental_households=e['households'],
            observational_households=o['households'],benchmark_households=b['households'],
            experimental_contact_rate=e['contact_rate'])
        for split in ['experimental', 'observational', 'benchmark']:
            part = g[(g.split == split) & (g.arm == 1) & (g.D == 1)]
            row[split+'_contacted_voters'] = len(part)
            row[split+'_contacted_households'] = part.hhid_phone.nunique()
        row['full_contacted_voters'] = int(((g.arm == 1) & (g.D == 1)).sum())
        rows.append(row)
    s=pd.DataFrame(rows)
    full = full.reset_index()
    s = fit_posteriors(s, full)
    full.to_csv(out/'calibration_estimates.csv', index=False)
    screening.to_csv(out/'district_screening.csv', index=False)
    export_summary_tables(s, full, out)
    units = 'individual districts'
    design = dict(source='Data supplied by Donald P. Green',
        source_url='https://drive.google.com/file/d/1ie-qqv9TmPCw-kD_oLGh4QWh6DmOUU3x/view',
        source_filename=Path(args.data).name,
        sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        cleaning=cleaning, analysis_people=len(d),
        arm_counts={str(int(k)): int(v) for k,v in d.arm.value_counts().items()},
        grouping=units,
        selection=dict(rule='Rank full-sample GOTV contacted-voter counts and include all ties at the cutoff. '
                             'Selection uses no turnout outcomes, treatment-effect estimates, or benchmark errors.',
                        requested_district_count=args.district_count,
                        retained_district_count=len(retained), minimum_contacted_voters=cutoff,
                        retained_districts=sorted(map(int, retained))),
        target='Complier average effect of GOTV contact on August 2008 primary turnout within each target.',
        calibration=f'Observational minus full-sample Wald contrasts in the other {len(s)-1} retained {units}. '
                    'Household-clustered variances account for covariance and the estimated contact rate.',
        evaluation='One fixed household split within district and assignment arm: 10% experimental, '
                   '45% observational, and 45% held-out experimental benchmark. Selection is held fixed.',
        experimental_zero_gotv_turnout_districts=[int(i) for i,g in d.groupby('district')
            if g.loc[(g.arm == 1) & (g.split == 'experimental'), 'Y'].sum() == 0])
    (out/'provenance.json').write_text(json.dumps(design, indent=2))
    report = write_results(s, out, design)
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
