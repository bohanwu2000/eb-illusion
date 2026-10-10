"""Nickerson 2007 professional phone mobilization with 2000 calibration.

Use --data to analyze an externally stored official Stata file. Use
--from-results to render the included aggregate CSVs. Outputs contain site and
stratum summaries only. Input is never downloaded by this script.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize_scalar
from scipy.stats import norm

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results/nickerson"
METHODS = ("experimental", "illusion", "calibrated EB")
SEED = 20261002
Z95 = norm.ppf(.975)


def prepare(path):
    raw = pd.read_stata(path, convert_categoricals=False)
    counts = {'raw_records': len(raw), 'raw_variables': len(raw.columns)}
    d = raw.loc[(raw.onlytf == 1) | (raw.phonetrt == 0)].copy()
    counts['professional_only_or_control'] = len(d)
    d = d.loc[d.canvass == 0].copy()
    counts['also_no_door_assignment'] = len(d)
    counts['missing_age_in_no_door_subset'] = int(d.age.isna().sum())
    d = d.loc[d.age.between(23, 26)].copy()
    counts['also_age_23_to_26'] = len(d)
    required = ['voted02', 'voted00', 'onlytf', 'tfcontac', 'site']
    counts['missing_required_after_eligibility'] = int(d[required].isna().any(axis=1).sum())
    if counts['missing_required_after_eligibility']:
        raise ValueError('Unexpected outcome/design missingness requires explicit review')
    assert np.array_equal(raw.onlytf, raw.phonetrt.eq(3).astype(int))
    d['Z'] = d.onlytf.astype(int)
    d['D'] = d.tfcontac.astype(float)
    d['geo'] = d.site.str.split(',').str[0]
    assert len(d.geo.unique()) == 15 and len(d.site.unique()) == 17
    counts['contacted_controls'] = int((d.loc[d.Z == 0, 'D'] == 1).sum())
    counts['assigned_no_calls'] = int((d.Z == 0).sum())
    counts['assigned_professional'] = int((d.Z == 1).sum())
    assert d[['voted02', 'voted00', 'D']].isin([0, 1]).all().all()
    d['partition'] = ''
    rng = np.random.default_rng(SEED)
    for (site, arm), idx in d.groupby(['site', 'Z'], sort=True).groups.items():
        idx = rng.permutation(np.array(idx))
        n1, n2 = int(.25*len(idx)), int(.35*len(idx))
        d.loc[idx[:n1], 'partition'] = 'experimental'
        d.loc[idx[n1:n1+n2], 'partition'] = 'observational'
        d.loc[idx[n1+n2:], 'partition'] = 'benchmark'
    assert (d.partition != '').all()
    counts['partition_counts'] = {k: int(v) for k, v in d.partition.value_counts().items()}
    return d, counts


def stats(g, weights, outcome='voted02'):
    """Stratum-standardized [Wald, contacted-minus-control] and row IFs.
    Condition on stratum/assignment counts. IF rows sum to zero within each
    stratum/arm and include estimation of the contact-rate denominator.
    """
    n = len(g)
    estimates = np.zeros(2)
    scores = np.zeros((n, 2))
    details = []
    for stratum, a in weights.items():
        loc = np.flatnonzero(g.site.to_numpy() == stratum)
        h = g.iloc[loc]
        z, contact, y = h.Z.to_numpy(), h.D.to_numpy(), h[outcome].to_numpy()
        assert np.isfinite(y).all()
        t, c = z == 1, z == 0
        tc = t & (contact == 1)
        nt, nc, nk = int(t.sum()), int(c.sum()), int(tc.sum())
        assert min(nt, nc, nk) > 1, (stratum, nt, nc, nk)
        mt, mc, mk = y[t].mean(), y[c].mean(), y[tc].mean()
        p1, p0 = contact[t].mean(), contact[c].mean()
        p = p1-p0
        assert p > 0
        w, o = (mt-mc)/p, mk-mc
        sw = t*((y-mt)-w*(contact-p1))/(nt*p)-c*((y-mc)-w*(contact-p0))/(nc*p)
        so = tc*(y-mk)/nk-c*(y-mc)/nc
        estimates += a*np.array([w, o])
        scores[loc] = a*np.column_stack([sw, so])
        details.append({'stratum': stratum, 'n': len(h), 'assigned_call': nt,
            'assigned_no_calls': nc, 'contacted': nk, 'contact_rate': p1, 'control_contact_rate': p0, 'first_stage': p, 'weight': a})
    np.testing.assert_allclose(scores.sum(0), 0, atol=1e-12)
    return estimates, scores, details


def covariance(g, scores):
    out = np.zeros((scores.shape[1], scores.shape[1]))
    for _, indices in g.reset_index(drop=True).groupby(['site', 'Z']).groups.items():
        a = scores[np.array(indices)]
        np.testing.assert_allclose(a.sum(0), 0, atol=1e-10)
        out += a.T @ a * len(a)/(len(a)-1)
    return out


def fit_prior(y, variance):
    """Global profile likelihood over variance, with explicit zero boundary.
    Fit in percentage-point units for stable optimization. The profile score
    is negative above range(y)^2, giving a finite global-search bound.
    """
    y, variance = np.asarray(y)*100, np.asarray(variance)*10000
    assert len(y) > 1 and np.all(variance > 0)
    upper = max(float(np.ptp(y)**2), 1e-6)
    def objective(tau2):
        v = variance+tau2
        mu = np.sum(y/v)/np.sum(1/v)
        return .5*np.sum(np.log(v)+(y-mu)**2/v)
    grid = np.unique(np.r_[0., np.linspace(0, upper, 257), np.geomspace(1e-10, upper, 257)])
    loss = np.array([objective(x) for x in grid])
    candidates = [(loss[0], 0.), (loss[-1], upper)]
    for j in range(1, len(grid)-1):
        if loss[j] <= loss[j-1] and loss[j] <= loss[j+1]:
            opt = minimize_scalar(objective, bounds=(grid[j-1], grid[j+1]),
                                  method='bounded', options={'xatol': 1e-11})
            assert opt.success
            candidates.append((opt.fun, opt.x))
    value, tau2 = min(candidates)
    assert value <= loss.min()+1e-8
    v = variance+tau2
    mu = np.sum(y/v)/np.sum(1/v)
    return mu/100, tau2/10000, 1/np.sum(1/v)/10000, value


def posterior(e, ve, o, vo, mu, gamma2):
    v = 1/(1/ve+1/(vo+gamma2))
    return v*(e/ve+(o-mu)/(vo+gamma2)), v


def bootstrap_check(g, weights, repetitions=3000):
    """Multinomial bootstrap within stratum/assignment, full joint [W,O,C00]."""
    rng = np.random.default_rng(91827)
    estimates = np.zeros((repetitions, 3))
    for stratum, a in weights.items():
        h = g[g.site == stratum]
        cells = {}
        for arm in [0, 1]:
            q = h[h.Z == arm].groupby(['D', 'voted02', 'voted00']).size()
            vals = np.array(q.index.to_list())
            counts = rng.multinomial(int(q.sum()), q/q.sum(), size=repetitions)
            cells[arm] = (vals, counts)
        vc, cc = cells[0]
        vt, ct = cells[1]
        nt, nc = ct.sum(1), cc.sum(1)
        nk = ct @ vt[:, 0]
        mt, mc = ct @ vt[:, 1]/nt, cc @ vc[:, 1]/nc
        mk = ct @ (vt[:, 0]*vt[:, 1])/nk
        m00k = ct @ (vt[:, 0]*vt[:, 2])/nk
        m00c = cc @ vc[:, 2]/nc
        estimates += a*np.column_stack([(mt-mc)/(nk/nt-cc @ vc[:, 0]/nc), mk-mc, m00k-m00c])
    full, sc, _ = stats(g, weights)
    prior, pc, _ = stats(g, weights, 'voted00')
    cov = covariance(g, np.column_stack([sc, pc[:, 1]]))
    empirical = np.cov(estimates, rowvar=False)
    contrast = np.array([-1., 1., -1.])
    ratios = np.diag(empirical)/np.diag(cov)
    mismatch_ratio = (contrast@empirical@contrast)/(contrast@cov@contrast)
    # Ratio-estimator small-sample nonlinearity can increase bootstrap variance.
    assert np.all((ratios > .8) & (ratios < 1.3)), ratios
    assert .8 < mismatch_ratio < 1.3
    return {'repetitions': repetitions, 'seed': 91827,
            'variance_ratios_wald_obs_cal2000': ratios.tolist(),
            'variance_ratio_shared_bias_mismatch': float(mismatch_ratio)}


def analyze(data, output=RESULTS, figure_dir=None):
    """Save the main analysis, optionally rendering its figure."""
    data, output = Path(data), Path(output)
    d, counts = prepare(data)
    output.mkdir(parents=True, exist_ok=True)
    rows, cellrows = [], []
    site_weights = {}
    for geo, g in d.groupby('geo', sort=True):
        g = g.reset_index(drop=True)
        weights = g.site.value_counts(normalize=True).sort_index().to_dict()
        site_weights[geo] = weights
        full, fullscore, cells = stats(g, weights)
        c00, c00score, _ = stats(g, weights, 'voted00')
        joint = covariance(g, np.column_stack([fullscore, c00score[:, 1]]))
        diag_vector = np.array([-1., 1., -1.])
        diff = full[1]-full[0]-c00[1]
        vd = float(diag_vector@joint@diag_vector)
        vb = float(np.array([-1., 1., 0.])@joint@np.array([-1., 1., 0.]))
        row = dict(site=geo, n=len(g), full_wald=full[0], full_obs=full[1],
            cal2000=c00[1], var_cal2000=joint[2, 2], full_bias=full[1]-full[0],
            var_full_bias=vb, bias_minus_cal2000=diff, var_bias_minus_cal2000=vd,
            cov_full_wald_obs=joint[0, 1], cov_full_wald_cal2000=joint[0, 2],
            cov_full_obs_cal2000=joint[1, 2])
        row['full_contact_rate'] = sum(x['weight']*x['contact_rate'] for x in cells)
        for partition, name, index in [('experimental', 'e', 0), ('observational', 'o', 1), ('benchmark', 'benchmark', 0)]:
            h = g[g.partition == partition].reset_index(drop=True)
            value, score, cells = stats(h, weights)
            var = covariance(h, score)[index, index]
            assert var > 0
            row[name], row['var_'+name], row['n_'+name] = value[index], var, len(h)
            row['contact_rate_'+name] = sum(x['weight']*x['contact_rate'] for x in cells)
            cellrows.extend(dict(site=geo, partition=partition, **cell) for cell in cells)
        rows.append(row)
    s = pd.DataFrame(rows)
    celltable = pd.DataFrame(cellrows)
    celltable.to_csv(output/'split_cell_counts.csv', index=False)
    metrics = []
    for year in [2000]:
        for i, row in s.iterrows():
            c = s[(s.site != row.site) & s[f'cal{year}'].notna()]
            mu, gamma2, vmu, loss = fit_prior(c[f'cal{year}'], c[f'var_cal{year}'])
            m, v = posterior(row.e, row.var_e, row.o, row.var_o, mu, gamma2)
            for key, val in {'mu':mu, 'gamma2':gamma2, 'var_mu_given_gamma':vmu,
                             'calibrated':m, 'var_calibrated':v, 'K':len(c),
                             'experimental_weight':v/row.var_e, 'profile_nll_pp':loss}.items():
                s.loc[i, f'{key}_{year}'] = val
        for method in METHODS:
            if method == 'experimental':
                mean, var = s.e, s.var_e
            elif method == 'illusion':
                mean = s.e
                var = 1/(1/s.var_e+1/s.var_o)
                mi, vi = posterior(s.e, s.var_e, s.o, s.var_o, s.o-s.e, 0)
                np.testing.assert_allclose(mi, mean, atol=1e-12)
                np.testing.assert_allclose(vi, var, atol=1e-12)
            else:
                mean, var = s[f'calibrated_{year}'], s[f'var_calibrated_{year}']
            s[f'{method.replace(" ", "_")}_mean_{year}'] = mean
            s[f'{method.replace(" ", "_")}_var_{year}'] = var
            gap = mean-s.benchmark
            # Benchmark sampling noise is common to all methods' expected MSE.
            # The subtraction is descriptive and can be negative in finite data.
            metrics.append(dict(calibration_year=year, method=method, sites=len(s),
                benchmark_mse_pp2=float(np.mean(gap**2)*10000),
                mean_signed_gap_pp=float(gap.mean()*100),
                mean_interval_length_pp=float(np.mean(2*Z95*np.sqrt(var))*100),
                mean_posterior_sd_pp=float(np.mean(np.sqrt(var))*100),
                mean_benchmark_variance_pp2=float(s.var_benchmark.mean()*10000),
                noise_subtracted_mse_pp2=float(np.mean(gap**2-s.var_benchmark)*10000)))
    s['illusion'] = s.e
    s['var_illusion'] = 1/(1/s.var_e+1/s.var_o)
    s.to_csv(output/'site_estimates.csv', index=False)
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(output/'method_summary.csv', index=False)
    s[['site','n','full_bias','var_full_bias','cal2000','var_cal2000',
       'bias_minus_cal2000','var_bias_minus_cal2000','cov_full_wald_obs',
       'cov_full_wald_cal2000','cov_full_obs_cal2000']].to_csv(output/'shared_bias_check.csv',index=False)
    se = np.sqrt(s.var_bias_minus_cal2000.sum())/len(s)
    delta = s.bias_minus_cal2000.mean()
    summary = dict(seed=SEED, counts=counts, stratum_weights=site_weights,
        units='Raw estimates in CSV use probability units. Summary metrics use pp or pp^2.',
        primary_calibration_year=2000,
        shared_bias_check={'mean_bias2002_pp':float(100*s.full_bias.mean()),
            'mean_cal2000_pp':float(100*s.cal2000.mean()), 'mean_gap_pp':float(100*delta),
            'se_gap_pp':float(100*se), 'ci95_gap_pp':(100*(delta+Z95*se*np.array([-1,1]))).tolist(),
            'normal_p_value':float(2*norm.sf(abs(delta/se))),
            'interpretation':'Fixed-site mean mismatch, individual stratified influence variance including overlap. Model-conditional normal approximation.'},
        ranges={'site_n':[int(s.n.min()),int(s.n.max())],
            'experimental_n':[int(s.n_e.min()),int(s.n_e.max())],
            'benchmark_n':[int(s.n_benchmark.min()),int(s.n_benchmark.max())],
            'experimental_contact_rate':[float(s.contact_rate_e.min()),float(s.contact_rate_e.max())],
            'minimum_partition_stratum_contacted':int(celltable.contacted.min()),
            'primary_mu_pp':[float(100*s.mu_2000.min()),float(100*s.mu_2000.max())],
            'primary_gamma_pp':[float(100*np.sqrt(s.gamma2_2000.min())),float(100*np.sqrt(s.gamma2_2000.max()))],
            'primary_experimental_weight':[float(s.experimental_weight_2000.min()),float(s.experimental_weight_2000.max())]},
        methods=metrics.to_dict(orient='records'),
        versions={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
                  'scipy':scipy.__version__,'matplotlib':matplotlib.__version__},
        dta_sha256=hashlib.sha256(Path(data).read_bytes()).hexdigest(),
        limitations=['Convenience sites and individual-level independence working assumption.',
            'Prior zeros do not establish past registration/residence eligibility.',
            'Benchmark noisy and some site Wald estimates imprecise.',
            'EB intervals condition on fitted hyperparameters and use estimated sampling variances.',
            'Leave-one-site-out fits reuse calibration sites across targets.',
            'J=1 does not test a many-observational-study asymptotic claim.',
            'Observed shared-bias mismatch limits interpretation of shrinkage gains.'])
    (output/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
    if figure_dir is not None:
        plot(s, metrics, figure_dir)
    return summary


@plt.rc_context()
def plot(s, metrics, figure_dir=BASE / "figures"):
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    # Reuse the same style and printed type sizes as the manuscript figures.
    from paper_plot_style import apply_style, finish_figure, METHOD_COLORS
    apply_style()
    plt.rcParams['pdf.fonttype']=42
    colors={name:METHOD_COLORS[key] for name,key in
            [('experimental','experiment'),('illusion','illusion'),('calibrated EB','ceb')]}
    fig, axes = plt.subplots(1,2,figsize=(10.4,3.8),
                             gridspec_kw={'width_ratios':[1,1.35]})
    m = metrics[metrics.calibration_year==2000]
    methods=['experimental','illusion','calibrated EB']
    errors=[np.square(100*(s[col].to_numpy()-s.benchmark.to_numpy()))
            for col in ['e','illusion','calibrated_2000']]
    assert s.site.nunique()==15
    assert all(len(e)==15 and np.isfinite(e).all() for e in errors)
    assert np.allclose([e.mean() for e in errors],
                       m.set_index('method').loc[methods,'benchmark_mse_pp2'])
    ax=axes[0]
    boxes=ax.boxplot(errors,patch_artist=True,widths=.48,showfliers=False,
                     medianprops={'color':'black','linewidth':1.5})
    ax.set_xticks([1,2,3],methods)
    for i,method in enumerate(methods):
        color=colors[method]
        boxes['boxes'][i].set_facecolor(matplotlib.colors.to_rgba(color,.12))
        boxes['boxes'][i].set_edgecolor(color)
        boxes['medians'][i].set_color(color)
        for line in boxes['whiskers'][2*i:2*i+2]+boxes['caps'][2*i:2*i+2]:
            line.set_color(color)
    ax.set_ylabel('Squared error against\n'+r'RCT benchmark ($\mathrm{pp}^2$)')
    ax.set_ylim(bottom=0)
    ax.set_title('(a) held-out evaluation')

    # Illustrate the same five sites, selected by eligible sample size.
    shown=s.nlargest(5,'n')
    xs=np.arange(len(shown))
    ax=axes[1]
    ax.axhline(0,color='0.7',lw=.8,ls=':')
    for offset,method,mean,var,marker in [
            (-.24,'experimental','e','var_e','o'),
            (-.08,'illusion','illusion','var_illusion','s'),
            (.08,'calibrated EB','calibrated_2000','var_calibrated_2000','^'),
            (.24,'experimental benchmark','benchmark','var_benchmark','D')]:
        color=colors.get(method,'0.4')
        ax.errorbar(xs+offset,100*shown[mean].to_numpy(),
                    yerr=Z95*100*np.sqrt(shown[var].to_numpy()),
                    fmt=marker,color=color,capsize=3,lw=1.5,markersize=4,
                    label=method)
    ax.set_xticks(xs,shown.site.to_list())
    ax.set_xlim(-.55,len(shown)-.45)
    ax.set_ylabel('Effect on 2002 turnout (pp)')
    ax.set_title('(b) five largest sites')
    ax.legend(loc='upper right',handletextpad=.4)
    finish_figure(fig,width_fraction=1.)
    fig.tight_layout()
    for fmt in ['pdf','png']:
        fig.savefig(figure_dir/f'nickerson_results.{fmt}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument('--data', type=Path,
                        help='official pro_v_vol_2002.dta stored outside the repository')
    inputs.add_argument('--from-results', type=Path, nargs='?', const=True,
                        help='plot a site-estimates CSV (with adjacent method_summary.csv), '
                             'or use --output when no path is given')
    parser.add_argument('--output', type=Path, default=RESULTS,
                        help='numerical output directory, or source directory for --from-results')
    parser.add_argument('--figure-dir', type=Path, default=BASE/'figures',
                        help='destination for nickerson_results.pdf and .png')
    args = parser.parse_args()
    if args.from_results is not None:
        source = args.output if args.from_results is True else args.from_results
        site_csv = source/'site_estimates.csv' if source.is_dir() else source
        s = pd.read_csv(site_csv, float_precision='round_trip')
        metrics = pd.read_csv(site_csv.parent/'method_summary.csv', float_precision='round_trip')
        plot(s, metrics, args.figure_dir)
        print(f'Figures saved to {args.figure_dir.resolve()}')
    else:
        if args.data is None:
            parser.error('supply --data /path/to/pro_v_vol_2002.dta or use --from-results. '
                         'See data/nickerson/README.md for the official download.')
        if not args.data.is_file():
            parser.error(f'data file does not exist: {args.data}')
        summary = analyze(args.data, args.output, args.figure_dir)
        print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
