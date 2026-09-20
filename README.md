# The Illusion of Learning from Observational Data: An Empirical Bayes Perspective

Code accompanying the manuscript by Bohan Wu, Sebastian Salazar, Donald P. Green, and David M. Blei (2026).

[Paper](https://bohanwu2000.github.io/papers/Wu_et_al_illusion2026.pdf)

This code inventory follows the active manuscript sources on September 20, 2026. The linked paper may be an earlier version. Some scripts regenerate simulations; the water and LEGEND forest plots redraw rounded manuscript summaries. The qualifications below distinguish those workflows.

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Run commands from the repository root. Plotting scripts create `figures/` automatically; set `EB_ILLUSION_FIGDIR` to choose another output directory. Use `MPLBACKEND=Agg` for headless runs. The playground is an HTML page requiring no Python packages; it loads Chart.js from jsDelivr and needs network access for that dependency.

## Manuscript-to-code map

| Manuscript analysis | Source | Output |
| --- | --- | --- |
| Main moment-matching and maximum-likelihood simulations | `src/main_simulation_figure.py` | `sim_main.png`, `sim_main_mle.png` |
| Finite calibration sample, non-Gaussian bias, shared-bias sensitivity, and CVCI comparison | `src/referee_simulations.py` | `sim_finite_k.pdf`, `sim_misspec.pdf`, `sim_sensitivity.pdf`, `sim_direct.pdf`; numerical results in `src/referee_sim_summary.json` |
| Nonparametric bias illustration | `src/nonparametric_ebnm.py` | `sim_nonparametric.pdf` |
| Water-use analysis and processed data | `src/water-analysis.ipynb`, `src/water_data.csv` | Notebook summaries; see the discrepancy below |
| Water forest plot from the manuscript table | `src/realdata_figure.py` | `sim_realdata.pdf` |
| Reconstructed LEGEND illustration | `src/legend_htn_analysis.py`, `src/legend_htn_figure.py` | Console calculation and `sim_legend.pdf` |
| Interactive simulation playground | `src/calibration_playground.html` | Open locally in a browser |

```sh
python3 src/main_simulation_figure.py
python3 src/referee_simulations.py
python3 src/nonparametric_ebnm.py
python3 src/realdata_figure.py
python3 src/legend_htn_analysis.py
python3 src/legend_htn_figure.py
```

The full simulation run can take time. A small smoke run and a deterministic CVCI implementation check are available:

```sh
python3 src/referee_simulations.py --check-cvci
python3 src/referee_simulations.py --replicates 20 --bootstrap-replicates 19
```

Simulation runs overwrite the saved summary JSON. Run smoke checks in a disposable checkout to preserve the manuscript-run results. To redraw the CVCI comparison from saved performance curves, use `python3 src/referee_simulations.py --plot-cvci`; this also recomputes its first-replicate forest panel and updates the JSON.

For the water notebook, install Jupyter separately (`python3 -m pip install jupyterlab`), launch it from the repository root or `src/`, and run all cells. The notebook reads the included processed CSV. The obsolete preprocessing step requiring an unbundled `clean.dta` and the former density-plot cell have been removed.

## Reproducibility and interpretation

- **Water analysis:** the historical notebook's calibrated estimate is approximately −0.37 with interval [−0.80, 0.06], whereas the manuscript table and forest plot report −0.58 [−0.90, −0.27]. The calibration construction and variance calculation require reconciliation before claiming end-to-end reproduction. `realdata_figure.py` preserves the manuscript's rounded values; it does not estimate them from patient or household records.
- **LEGEND:** per-database observational estimates and negative controls are reconstructed illustrations. The forest plot uses rounded manuscript summaries and illustrative database points, independently of the analysis script. ALLHAT supplies a risk ratio for a composite coronary endpoint, whereas LEGEND targets a myocardial-infarction hazard ratio. Verified inputs and aligned estimands are required for an empirical pooled analysis.
- **Simulation defaults:** the main generator uses 4,000 replicates; robustness experiments use 20,000; CVCI uses 5,000 with 999 bootstrap replicates. These differ from the manuscript's general 10,000-replicate statement. Seeds are explicit in the scripts and saved CVCI metadata. The main maximum-likelihood implementation profiles over a finite variance grid.
- **Nonparametric implementation:** this Python script fits mixture weights on a fixed grid by EM. It is a grid approximation to NPMLE, not an invocation of the R `ebnm` package mentioned in the manuscript.
- **Intervals:** forest plots retain the exact rounded interval endpoints in their source tables. A plotted interval's agreement with a benchmark is not a repeated-sampling coverage assessment.

The unused NSW, GOTV, and diabetes examples are outside the current code inventory. The duplicate simulation notebook is replaced by the script entry point; the obsolete water density PDF and duplicate source archive are omitted. Earlier public versions remain accessible in Git history.

## Validation of this update

Verified with Python 3 and NumPy 1.26.4, SciPy 1.16.3, pandas 3.0.5, and Matplotlib 3.11.1: Python compilation, deterministic CVCI checks, a 20-replicate robustness/CVCI smoke run (19 bootstrap replicates), full main and nonparametric generators, both forest plot generators, and execution of the cleaned water notebook. Smoke-run outputs were kept separate from the saved manuscript-run summary.
