"""Plot Michigan squared errors and a district's posterior intervals.

Read the saved top-half district estimates with --baseline and write to --output.
The paper uses --benchmark full and --posterior-unit 89 (the defaults).
Every retained district contributes to the error summaries.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from don_small_rct import draw_errors, draw_intervals
from paper_plot_style import apply_style, finish_figure, METHOD_COLORS


METHODS = [('experimental', 'small_rct', METHOD_COLORS['experiment']),
           ('observational', 'obs', '0.45'),
           ('illusion', 'illusion', METHOD_COLORS['illusion']),
           ('calibrated EB', 'corrected', METHOD_COLORS['ceb'])]


def draw_target_intervals(ax, sample, unit, benchmark_type):
    """Keep the existing district illustration and its fitted intervals."""
    row = sample.loc[sample.unit == unit].iloc[0]
    benchmark_label = 'full RCT benchmark' if benchmark_type == 'full' else 'experimental benchmark'
    labels = [benchmark_label]+[method for method, _, _ in METHODS]
    means = 100*np.array([row.evaluation_benchmark, row.small_rct, row.obs,
                          row.illusion, row.corrected])
    sds = 100*np.array([row.evaluation_benchmark_sd, np.sqrt(row.small_rct_var),
                        np.sqrt(row.var_obs), np.sqrt(row.illusion_var), row.posterior_sd])
    colors = [METHOD_COLORS['experiment']]+[color for _, _, color in METHODS]
    assert np.isfinite(means).all() and np.isfinite(sds).all() and (sds > 0).all()
    draw_intervals(ax, labels, means, sds, colors, means[0])
    ax.set_title(f'(b) district {unit}')
    z95 = 1.959963984540054
    return dict(district=unit, J=1, K=len(sample)-1,
                selection='Explicit district identifier; retained from the previous illustration.',
                means_pp=dict(zip(labels, means.tolist())),
                intervals_95_pp={label: [float(mean-z95*sd), float(mean+z95*sd)]
                                 for label, mean, sd in zip(labels, means, sds)},
                interpretation='The benchmark and observational rows show normal 95% confidence '
                               'intervals. The experimental row also agrees with its flat-prior '
                               'Gaussian credible interval. Illusion and calibrated EB show '
                               '95% posterior credible intervals conditional on fitted parameters. '
                               'The red line marks the experimental benchmark.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--posterior-unit', type=int, default=89)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--benchmark', choices=['held-out', 'full'], default='full')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(args.baseline)
    cohort = 'top half'
    assert sample.unit.is_unique
    assert np.isfinite(sample[['wald']+[column for _, column, _ in METHODS]]).all().all()
    sample['evaluation_benchmark'] = sample.wald
    sample['evaluation_benchmark_sd'] = np.sqrt(sample.var_wald)
    if args.benchmark == 'full':
        full = pd.read_csv(args.baseline.parent/'full_sample_summary.csv').set_index('district')
        calibration = pd.read_csv(args.baseline.parent/'calibration_estimates.csv').set_index('unit')
        assert full.index.is_unique and calibration.index.is_unique
        assert set(full.index) == set(calibration.index) == set(sample.unit)
        full = full.loc[sample.unit]
        calibration = calibration.loc[sample.unit]
        np.testing.assert_allclose(full.wald, calibration.wald, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(full.wald_sd**2, calibration.var_wald,
                                   rtol=1e-12, atol=1e-14)
        assert np.isfinite(full[['wald', 'wald_sd']]).all().all()
        assert (full.wald_sd > 0).all()
        sample['evaluation_benchmark'] = full.wald.to_numpy()
        sample['evaluation_benchmark_sd'] = full.wald_sd.to_numpy()
    assert args.posterior_unit in set(sample.unit), 'Illustrated district must be retained.'
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.4),
                             gridspec_kw={'width_ratios': [1.12, 1.25]})
    errors = [np.square(100*(sample[column].to_numpy()-sample.evaluation_benchmark.to_numpy()))
              for _, column, _ in METHODS]
    draw_errors(axes[0], errors, [name for name, _, _ in METHODS],
                [color for _, _, color in METHODS])
    axes[0].set_title(f'(a) {cohort} ({len(sample)} districts)')
    if args.benchmark == 'full':
        axes[0].set_ylabel('Squared error against\nRCT benchmark (pp²)')
    summary = []
    district_errors = []
    for (method, column, _), values in zip(METHODS, errors):
        q = np.quantile(values, [.05, .25, .5, .75, .95])
        summary.append(dict(cohort=cohort, method=method, n=len(values),
                            mse_pp2=float(values.mean()), q05_pp2=float(q[0]),
                            q25_pp2=float(q[1]), median_pp2=float(q[2]),
                            q75_pp2=float(q[3]), q95_pp2=float(q[4]),
                            maximum_pp2=float(values.max()),
                            held_out_mse_pp2=float(np.mean(np.square(
                                100*(sample[column]-sample.wald))))))
        district_errors.extend(dict(cohort=cohort, district=int(district), method=method,
                                    estimate_pp=float(100*estimate),
                                    benchmark_pp=float(100*benchmark),
                                    squared_error_pp2=float(error))
                               for district, estimate, benchmark, error in zip(
                                   sample.unit, sample[column], sample.evaluation_benchmark, values))
    axes[1].set_ylabel('')
    posterior_panel = draw_target_intervals(axes[1], sample, args.posterior_unit, args.benchmark)
    finish_figure(fig)
    plt.setp(axes[0].get_xticklabels(), rotation=25, ha='right', rotation_mode='anchor')
    fig.tight_layout()
    prefix = 'full_data' if args.benchmark == 'full' else 'held_out'
    for extension in ['pdf', 'png']:
        fig.savefig(args.output/f'{prefix}_error_distribution.{extension}', dpi=180,
                    bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(summary).to_csv(args.output/'error_distribution_summary.csv', index=False)
    pd.DataFrame(district_errors).to_csv(args.output/f'{prefix}_squared_errors.csv', index=False)
    pd.DataFrame(dict(cohort=cohort, district=sample.unit,
                      benchmark=sample.evaluation_benchmark,
                      benchmark_sd=sample.evaluation_benchmark_sd,
                      held_out_benchmark=sample.wald,
                      held_out_benchmark_sd=np.sqrt(sample.var_wald))).to_csv(
                          args.output/'evaluation_benchmarks.csv', index=False)
    metadata = dict(baseline_districts=len(sample),
                    posterior_panel=posterior_panel,
                    benchmark=args.benchmark,
                    median_benchmark_sd_pp={cohort: float(100*sample.evaluation_benchmark_sd.median())},
                    median_held_out_benchmark_sd_pp={cohort: float(100*np.sqrt(sample.var_wald).median())},
                    definition=f"Squared difference from each district's {args.benchmark} Wald estimate, "
                               'in squared percentage points. Boxes show the interquartile range '
                               'and median; whiskers extend to observations within 1.5 IQR. '
                               'Outlier markers are hidden. Every district enters the MSE, '
                               'quantiles, and exported error table.',
                    design='All fitted estimates use the original household splits within each district. '
                           'Each target uses full-sample calibration estimates from the other '
                           'districts in its retained cohort. Benchmark sampling uncertainty '
                           'contributes to the squared errors.')
    if args.benchmark == 'full':
        metadata['evaluation'] = (
            'The full-data benchmark uses all eligible GOTV and control observations in the '
            'target district, including the experimental and observational subsets. '
            'All fitted method estimates and sample splits are unchanged. Shared data induce '
            'covariance between each estimate and the benchmark, which affects squared errors '
            'differently across methods. This comparison measures agreement with the full-data '
            'experimental estimate; it is not independent held-out evaluation or risk against '
            'a known causal effect. Benchmark and standard-error columns in evaluation_benchmarks.csv '
            'are in probability units. Error summaries are in squared percentage points.')
    (args.output/'error_distribution_metadata.json').write_text(json.dumps(metadata, indent=2))
    print(pd.DataFrame(summary)[['cohort', 'method', 'n', 'mse_pp2', 'median_pp2',
                                'q95_pp2']].to_string(index=False))
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
