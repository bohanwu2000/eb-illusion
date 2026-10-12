# Reproducing the paper

[Back to README](README.md)

## Reproduce the paper figures

Use Python 3.11 or later. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 src/render_paper_figures.py
```

Figures are written to `figures/`. This command uses the bundled simulation results and aggregate empirical estimates. It checks that the numerical inputs remain unchanged and requires no individual voter records. To choose another destination:

```sh
python3 src/render_paper_figures.py --figure-dir /path/to/figures
```

### From the paper to the code

| Analysis | Code | Bundled inputs / figure |
| --- | --- | --- |
| Varying observational and calibration study counts | [simulation_restructure.py](src/simulation_restructure.py) | `results/simulations/finite_studies.npz` → `sim_finite_k`, `sim_finite_k_coverage` |
| Non-Gaussian bias distributions | [simulation_restructure.py](src/simulation_restructure.py) | Same input → `sim_misspec`, `sim_misspec_coverage` |
| CVCI comparison | [referee_simulations.py](src/referee_simulations.py) | `results/simulations/cvci.json` → `sim_direct`, `sim_finite_k_methods` |
| Shared-bias sensitivity | [referee_simulations.py](src/referee_simulations.py) | `results/simulations/sensitivity.json` → `sim_sensitivity` |
| NPMLE and Gaussian EB | [nonparametric_npmle.py](src/nonparametric_npmle.py) | `results/nonparametric_npmle.json` → `sim_nonparametric` |
| Appendix asymptotic simulations | [main_simulation_figure.py](src/main_simulation_figure.py) | `results/simulations/main.npz` → `sim_main`, `sim_main_mle` |
| Water consumption | [water-analysis.ipynb](src/water-analysis.ipynb), [realdata_figure.py](src/realdata_figure.py) | Processed `src/water_data.csv` → notebook analysis and reported-summary figure `sim_realdata` |
| Michigan voter mobilization | [don_small_rct.py](src/don_small_rct.py), [don_error_distribution.py](src/don_error_distribution.py) | `results/don-top-districts/` → `don_small_rct` |
| Professional phone mobilization | [nickerson_analysis.py](src/nickerson_analysis.py) | `results/nickerson/` → `nickerson_results` |

All figures share [paper_plot_style.py](src/paper_plot_style.py). Simulation grids stop at 100 studies. The NPMLE prior illustration uses 600 calibration studies, while its MSE curves use J = K in {5, 10, 50, 100}.

## Run new simulations

Saved paper results and new runs have separate output paths. For a quick CVCI check and small run:

```sh
python3 src/referee_simulations.py --check-cvci
python3 src/referee_simulations.py --experiment cvci \
  --replicates 20 --bootstrap-replicates 19 --output results/cvci-smoke.json
```

For full simulation runs:

```sh
python3 src/simulation_restructure.py --replicates 10000 \
  --results results/finite-studies-rerun.npz --summary results/finite-studies-rerun.json
python3 src/referee_simulations.py --experiment cvci \
  --replicates 5000 --bootstrap-replicates 999 --output results/cvci-rerun.json
python3 src/referee_simulations.py --experiment sensitivity \
  --replicates 20000 --output results/sensitivity-rerun.json
python3 src/nonparametric_npmle.py --refit --workers 4 \
  --results results/npmle-rerun.json
python3 src/main_simulation_figure.py --results results/main-rerun.npz
```

The NPMLE uses a fixed-grid EM approximation. The prior illustration uses 60 support locations and 150 updates. The MSE comparison uses 40 support locations, 80 updates, and 1,000 replicates per setting. Both methods receive the same data in each replicate. Numerical tests compare posterior integration with exact small mixtures and independent adaptive quadrature.

## Empirical studies

### Michigan voter mobilization

The paper uses **53 individual districts**, the top half by GOTV contact counts including ties. Each target uses one observational study and calibration studies from the other 52 districts. The household split assigns 10% to the experimental study, 45% to the observational study, and 45% to a held-out sample. The paper evaluates the estimates against the **full-data RCT benchmark**.

- [full_sample_summary.csv](results/don-top-districts/full_sample_summary.csv): district, Wald estimate and standard error, observational estimate and standard error, their covariance, and calibration estimate and standard error.
- [district_estimates.csv](results/don-top-districts/district_estimates.csv): target estimates, fitted bias parameters, posterior intervals, and split-specific benchmark estimates.
- [calibration_estimates.csv](results/don-top-districts/calibration_estimates.csv): inputs for each leave-one-district-out calibration fit.
- [district_screening.csv](results/don-top-districts/district_screening.csv): inclusion and contact counts for all 104 districts.
- [provenance.json](results/don-top-districts/provenance.json): source, cleaning, selection, and raw-file checksum.

Estimates and standard errors in these CSVs use probability units. Figures use percentage points. Calibration is `observational - wald`, with variance `observational_sd**2 + wald_sd**2 - 2*cov_wald_observational`.

To rerun from the original data supplied by Donald P. Green:

```sh
python3 src/don_small_rct.py --data '/path/to/etov_august_2008_no PII.dta' \
  --output results/michigan-rerun
python3 src/don_error_distribution.py \
  --baseline results/michigan-rerun/district_estimates.csv \
  --benchmark full --posterior-unit 89 --output results/michigan-rerun-plots
```

The raw voter records remain outside this repository. The full-data benchmark shares observations with the target estimates, and its squared errors measure agreement with an estimated benchmark. District 89 is a selected interval illustration. Every retained district contributes to the boxplot calculations.

### Professional phone mobilization

The Nickerson (2007) analysis uses **15 sites and 10,018 voters**. Each target uses an experimental Wald estimate and an observational contacted-versus-control contrast for 2002 turnout. The other 14 sites provide calibration studies using 2000 turnout. Durham and Wake retain their race-specific randomization strata.

[Download instructions and source hashes](data/nickerson/README.md) explain how to obtain the author's public replication data. [Site estimates](results/nickerson/site_estimates.csv) and [method summaries](results/nickerson/method_summary.csv) are bundled.

```sh
python3 src/nickerson_analysis.py --data /path/to/pro_v_vol_2002.dta \
  --output results/nickerson-rerun --figure-dir figures
```

The paper's fixed split has benchmark MSE 451.54 pp² for experimental and illusion and 118.12 pp² for calibrated EB. The fitted bias variances are zero. Benchmark uncertainty and the cross-election calibration assumption are discussed in the paper.

### Water consumption

The processed water data and notebook are included. Install `requirements-dev.txt` to execute the notebook. **The figure redraws the manuscript's reported estimates.** The historical notebook gives a calibrated estimate of approximately −0.37 [−0.80, 0.06], while the manuscript figure reports −0.58 [−0.90, −0.27]. This difference remains unresolved, so the figure renderer and notebook are documented separately.

## Checks

```sh
python3 -B -m unittest discover -s tests -v
python3 src/referee_simulations.py --check-cvci
python3 src/verify_nickerson.py
node --test tests/test_playground.js
```

The tests check posterior integration and the saved Michigan calibration calculations. The figure renderer also verifies that its input files remain unchanged.

## Citation and license

Please cite *The Illusion of Learning from Observational Data: An Empirical Bayes Perspective* by Bohan Wu, Sebastian Salazar, Donald P. Green, and David M. Blei. The code is available under the [MIT license](LICENSE). Replication data retain their original sources and terms.

This package contains the analyses used in the current manuscript and appendix. Retired LEGEND, NSW, paired-district, and alternative nonparametric implementations are omitted. Earlier published code remains in Git history.
