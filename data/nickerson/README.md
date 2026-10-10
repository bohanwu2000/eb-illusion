# Nickerson (2007) professional phone mobilization

This package reproduces the paper's 2000 prior-election calibration analysis:
10,018 eligible voters across 15 sites, one observational estimate per target
site (`J = 1`), and calibration estimates from the other 14 sites (`K = 14`).
The experimental, observational, and held-out
benchmark samples contain 2,491, 3,488, and 4,039 voters.

## Official data and checksums

Download the replication archive from
[David Nickerson's Temple University website](https://sites.temple.edu/nickerson/files/2023/03/pro_v_vol_2002.zip).
The original study is available as
[Quality Is Job One: Professional and Volunteer Voter Mobilization Calls](https://sites.temple.edu/nickerson/files/2017/07/Quality.Nickerson.2007.pdf).

Keep the archive and extracted Stata file outside the repository. The following
commands download the archive when you explicitly run them:

```sh
nickerson_data_dir="$HOME/Downloads/nickerson"
mkdir -p "$nickerson_data_dir"
curl --fail --location \
  --output "$nickerson_data_dir/pro_v_vol_2002.zip" \
  https://sites.temple.edu/nickerson/files/2023/03/pro_v_vol_2002.zip
unzip -j "$nickerson_data_dir/pro_v_vol_2002.zip" \
  pro_v_vol_2002.dta -d "$nickerson_data_dir"
shasum -a 256 "$nickerson_data_dir/pro_v_vol_2002.zip"
shasum -a 256 "$nickerson_data_dir/pro_v_vol_2002.dta"
```

Expected SHA-256 values from the original retrieval on October 2, 2026:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `pro_v_vol_2002.zip` | 1,913,852 | `bcb0db4b1a9898b6d446d2e365376e1c46aa58a5e1ff31d95f66a5c88a76fc13` |
| `pro_v_vol_2002.dta` | 20,897,747 | `b8cba67fffbfdc17de4d6f70e715cc6f9b5823dae7f7d1af1852e8bcf76e067f` |

The complete original source metadata and hashes are in
[source_manifest.json](source_manifest.json). The scripts read local inputs and
perform no downloads. The repository contains aggregate results, with the
unit-level data obtained separately from the original author.

## Reproduce

From the repository root, using Python 3 and the repository's requirements:

```sh
python3 src/nickerson_analysis.py \
  --data "$HOME/Downloads/nickerson/pro_v_vol_2002.dta" \
  --output results/nickerson \
  --figure-dir figures
```

To render the same two-panel figure directly from the included aggregates:

```sh
python3 src/nickerson_analysis.py \
  --from-results results/nickerson/site_estimates.csv --figure-dir figures
```

The scripts resolve their default output locations relative to the repository,
so they also work when invoked by absolute path from another directory.
Explicit relative paths are relative to the current working directory.
Figures use `src/paper_plot_style.py` and are written as
`nickerson_results.pdf` and `nickerson_results.png`.
The plot-only command reads the site-estimates CSV and its adjacent
`method_summary.csv`, leaving numerical results intact. A bare
`--from-results` reads the directory supplied by `--output` (default:
`results/nickerson`). It also accepts an explicit results directory.

Check the included aggregate results without raw data:

```sh
python3 src/verify_nickerson.py --output results/nickerson
```

Check the official input checksum, influence derivatives, bootstrap variances,
and raw-data reproduction against those results:

```sh
python3 src/verify_nickerson.py \
  --data "$HOME/Downloads/nickerson/pro_v_vol_2002.dta" \
  --output results/nickerson
```

The full check regenerates numerical results in a temporary directory, compares
every CSV value and non-environment JSON value, then removes the temporary
outputs. Check reports are printed to the terminal.

## Analysis and included results

Eligibility uses the national professional-only and control arms, no assigned
door visit, and ages 23–26. The analysis retains both contacted controls.
Durham and Wake each have two race-specific randomization strata. Fixed
eligible-sample stratum shares weight every estimate within a site. Splitting
occurs within each stratum and assignment arm, with floor-rounded fractions
of 25% experimental and 35% observational and the remainder held out.

The experimental and benchmark estimates are stratum-weighted Wald ratios.
The observational estimate contrasts contacted assigned-treatment voters
with assigned controls. The same observational contrast on 2000 turnout,
using all eligible voters in each of the other sites, provides calibration.
Individual influence functions within stratum and assignment arm estimate
variances, including contact-rate estimation and shared-record covariance.
The Gaussian bias prior is fitted by marginal maximum likelihood, including
the zero-variance boundary. The fitted hyperparameters and sampling variances
are plugged into the Gaussian posterior.

The current paper uses 2000 calibration. The package includes this analysis
and its shared-bias diagnostic.

| File under `results/nickerson/` | Contents |
| --- | --- |
| `site_estimates.csv` | 15 site estimates, sampling variances, prior fits, and posterior summaries |
| `method_summary.csv` | Three methods' MSE, interval lengths, and related aggregate metrics |
| `shared_bias_check.csv` | 15 site-level bias/calibration contrasts and their covariance terms |
| `split_cell_counts.csv` | 51 stratum-by-partition summaries with counts and fixed stratum weights |
| `results.json` | Eligibility counts, stratum weights, summary metrics, input hash, and software versions |

Site-level estimates use probability units and their variances use squared
probability units. Method metrics use percentage points (pp) or squared
percentage points (pp²). The saved experimental and illusion MSE is
451.5398091237429 pp². Calibrated EB MSE is 118.12170387813227 pp².
Mean 95% interval lengths are 74.86188396589398 pp for experimental and
33.9119838181861 pp for both illusion and calibrated EB. All 15 fitted prior
variances are zero.

Evaluation uses one fixed split and noisy held-out experimental benchmarks.
The mean benchmark variance is 281.8905796705772 pp². The
`noise_subtracted_mse_pp2` column is a descriptive subtraction and can be
negative. The shared-bias diagnostic has mean gap 4.643739773384156 pp and
95% interval [-0.17721733216092828, 9.46469687892924] pp.
