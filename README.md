# The Illusion of Learning from Observational Data
### An Empirical Bayes Perspective

Bohan Wu · Sebastian Salazar · Donald P. Green · David M. Blei

[Paper](https://bohanwu2000.github.io/papers/Wu_et_al_illusion2026.pdf) · [Interactive playground](https://bohanwu2000.github.io/eb-illusion/) · [Reproduction guide](REPRODUCING.md)

Code and experiments for learning observational bias from calibration studies and combining observational and experimental evidence.

## Quick start

Use Python 3.11 or later:

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 src/render_paper_figures.py
```

Figures are saved to `figures/` using bundled results and aggregate estimates. Explore the simulations directly in the [browser playground](https://bohanwu2000.github.io/eb-illusion/).

## Code

| Analysis | Entry point |
| --- | --- |
| Study counts and non-Gaussian biases | [simulation_restructure.py](src/simulation_restructure.py) |
| CVCI comparison and sensitivity | [referee_simulations.py](src/referee_simulations.py) |
| Nonparametric empirical Bayes | [nonparametric_npmle.py](src/nonparametric_npmle.py) |
| Appendix simulations | [main_simulation_figure.py](src/main_simulation_figure.py) |
| Michigan voter mobilization | [don_small_rct.py](src/don_small_rct.py) |
| Professional phone mobilization | [nickerson_analysis.py](src/nickerson_analysis.py) |
| Water consumption | [water-analysis.ipynb](src/water-analysis.ipynb) |

See the [reproduction guide](REPRODUCING.md) for rerun commands, data sources, and checks. Raw Michigan voter records are supplied separately. The water figure uses manuscript summaries whose calibrated estimate differs from the historical notebook output. Details are recorded in the guide.

## Citation

Please cite *The Illusion of Learning from Observational Data: An Empirical Bayes Perspective* by Bohan Wu, Sebastian Salazar, Donald P. Green, and David M. Blei. Citation metadata are in [CITATION.cff](CITATION.cff).

Code: [MIT license](LICENSE). Data retain their original sources and terms.
