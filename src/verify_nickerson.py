"""Check the published Nickerson aggregates and optional raw-data reproduction.

Run without --data to verify saved aggregates. Supply the official Stata file
to also check influence derivatives, bootstrap variances, and reproduction.
Verification reports go to stdout. Temporary regenerated aggregates are removed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import norm

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results/nickerson"
TABLES = ("site_estimates.csv", "method_summary.csv",
          "shared_bias_check.csv", "split_cell_counts.csv")
METHODS = ("experimental", "illusion", "calibrated EB")


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-10)


def compare_results(actual, expected):
    """Compare every saved CSV value and all non-environment JSON values."""
    cells = 0
    for name in TABLES:
        a = pd.read_csv(actual / name, float_precision="round_trip")
        b = pd.read_csv(expected / name, float_precision="round_trip")
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-10, atol=1e-12)
        cells += a.size

    def compare_json(a, b):
        check(type(a) is type(b), "JSON value types differ")
        if isinstance(a, dict):
            check(a.keys() == b.keys(), "JSON keys differ")
            for key in a:
                if key != "versions":
                    compare_json(a[key], b[key])
        elif isinstance(a, list):
            check(len(a) == len(b), "JSON list lengths differ")
            for x, y in zip(a, b):
                compare_json(x, y)
        elif isinstance(a, float):
            close(a, b)
        else:
            check(a == b, "JSON values differ")
    compare_json(json.loads((actual / "results.json").read_text()),
                 json.loads((expected / "results.json").read_text()))
    return cells


def verify_saved(output):
    s = pd.read_csv(output / "site_estimates.csv", float_precision="round_trip")
    metrics = pd.read_csv(output / "method_summary.csv", float_precision="round_trip")
    shared = pd.read_csv(output / "shared_bias_check.csv", float_precision="round_trip")
    cells = pd.read_csv(output / "split_cell_counts.csv", float_precision="round_trip")
    summary = json.loads((output / "results.json").read_text())
    check(len(s) == s.site.nunique() == 15, "Expected 15 unique sites")
    check(s.K_2000.eq(14).all(), "Each site must use the other 14 calibration sites")
    check(metrics.calibration_year.eq(2000).all(), "Expected the 2000 analysis")
    check(len(metrics) == 3 and set(metrics.method) == set(METHODS), "Expected three methods")
    check(not any("1998" in c for c in s.columns), "Retired sensitivity columns found")
    check(np.isfinite(s.select_dtypes("number")).all().all(), "Nonfinite saved estimates")
    close(s.n, s.n_e + s.n_o + s.n_benchmark)
    counts = {"experimental": int(s.n_e.sum()), "observational": int(s.n_o.sum()),
              "benchmark": int(s.n_benchmark.sum())}
    check(counts == {"experimental": 2491, "observational": 3488, "benchmark": 4039},
          "Published split counts differ")
    check(int(s.n.sum()) == 10018, "Published eligible sample differs")
    check(summary["counts"]["partition_counts"] == counts, "JSON split counts differ")
    check(summary["counts"]["contacted_controls"] == 2, "Keep both contacted controls")
    check(summary["seed"] == 20261002 and summary["primary_calibration_year"] == 2000,
          "Published seed or calibration year differs")
    check(s.gamma2_2000.eq(0).all(), "Published prior variances should be zero")
    for column in ("var_e", "var_o", "var_benchmark", "var_cal2000"):
        check(s[column].gt(0).all(), "Sampling variances must be positive")

    # Independent closed forms: the published 15 fitted variances are on zero.
    for _, target in s.iterrows():
        other = s[s.site != target.site]
        precision = 1 / other.var_cal2000.to_numpy()
        close(target.mu_2000, np.dot(other.cal2000, precision) / precision.sum())
        close(target.var_mu_given_gamma_2000, 1 / precision.sum())
        vpp = other.var_cal2000.to_numpy() * 10000
        close(target.profile_nll_pp_2000, .5 * np.sum(
            np.log(vpp) + ((other.cal2000.to_numpy() - target.mu_2000) * 100)**2 / vpp))

    variance = 1 / (1 / s.var_e + 1 / (s.var_o + s.gamma2_2000))
    mean = variance * (s.e / s.var_e + (s.o - s.mu_2000) / (s.var_o + s.gamma2_2000))
    close(s.calibrated_2000, mean)
    close(s.var_calibrated_2000, variance)
    close(s.experimental_weight_2000, variance / s.var_e)
    close(s.illusion, s.e)
    close(s.var_illusion, 1 / (1 / s.var_e + 1 / s.var_o))
    close(s.full_bias, s.full_obs - s.full_wald)
    close(s.bias_minus_cal2000, s.full_bias - s.cal2000)
    close(s.var_bias_minus_cal2000, s.var_full_bias + s.var_cal2000
          - 2 * (s.cov_full_obs_cal2000 - s.cov_full_wald_cal2000))

    for method, mean, variance in (
        ("experimental", s.e, s.var_e), ("illusion", s.illusion, s.var_illusion),
        ("calibrated EB", s.calibrated_2000, s.var_calibrated_2000),
    ):
        key = method.replace(" ", "_")
        close(s[key + "_mean_2000"], mean)
        close(s[key + "_var_2000"], variance)
        saved = metrics.set_index("method").loc[method]
        gap = mean - s.benchmark
        expected = {
            "sites": 15, "benchmark_mse_pp2": np.mean(gap**2) * 10000,
            "mean_signed_gap_pp": gap.mean() * 100,
            "mean_interval_length_pp": np.mean(2 * norm.ppf(.975) * np.sqrt(variance)) * 100,
            "mean_posterior_sd_pp": np.mean(np.sqrt(variance)) * 100,
            "mean_benchmark_variance_pp2": s.var_benchmark.mean() * 10000,
            "noise_subtracted_mse_pp2": np.mean(gap**2 - s.var_benchmark) * 10000,
        }
        for key, value in expected.items():
            close(saved[key], value)
        record = next(x for x in summary["methods"] if x["method"] == method)
        for key in metrics.columns:
            if key == "method":
                check(record[key] == method, "JSON method differs")
            else:
                close(record[key], saved[key])
    pd.testing.assert_frame_equal(shared, s[shared.columns], check_exact=True)
    check(len(cells) == 51, "Expected 17 strata in each of three partitions")
    for partition, name in (("experimental", "e"), ("observational", "o"), ("benchmark", "benchmark")):
        subset = cells[cells.partition == partition]
        close(subset.groupby("site").n.sum().reindex(s.site), s["n_" + name])
        close(subset.groupby("site").weight.sum().reindex(s.site), np.ones(len(s)))
        close(subset.n, subset.assigned_call + subset.assigned_no_calls)
    for _, row in cells.iterrows():
        close(row.weight, summary["stratum_weights"][row.site][row.stratum])
        close(row.first_stage, row.contact_rate - row.control_contact_rate)
    diagnostic = summary["shared_bias_check"]
    delta = s.bias_minus_cal2000.mean()
    se = np.sqrt(s.var_bias_minus_cal2000.sum()) / len(s)
    for key, value in {
        "mean_bias2002_pp": 100 * s.full_bias.mean(),
        "mean_cal2000_pp": 100 * s.cal2000.mean(), "mean_gap_pp": 100 * delta,
        "se_gap_pp": 100 * se, "ci95_gap_pp": 100 * (delta + norm.ppf(.975) * se * np.array([-1, 1])),
        "normal_p_value": 2 * norm.sf(abs(delta / se)),
    }.items():
        close(diagnostic[key], value)
    return {"saved_tables": len(TABLES), "sites": 15, "J": 1, "K": 14,
            "eligible_voters": 10018, "partition_counts": counts,
            "posterior_and_prior_fits_checked": 15, "method_metric_rows_checked": 3,
            "split_stratum_rows_checked": len(cells),
            "benchmark_mse_pp2": metrics.set_index("method").benchmark_mse_pp2.to_dict(),
            "mean_interval_length_pp": metrics.set_index("method").mean_interval_length_pp.to_dict(),
            "saved_posterior_mse_and_shared_bias_identities": "passed"}


def weighted_stats(g, weights, r):
    out = np.zeros(3)
    for h, a in weights.items():
        pos = np.flatnonzero(g.site.to_numpy() == h)
        q = g.iloc[pos]
        rr = r[pos]
        t = q.Z.to_numpy() == 1
        c = ~t
        d = q.D.to_numpy()
        y = q.voted02.to_numpy()
        y0 = q.voted00.to_numpy()
        k = t & (d == 1)
        def mean(v, mask):
            return np.sum(rr[mask] * v[mask]) / rr[mask].sum()
        out += a * np.array([(mean(y, t) - mean(y, c)) / (mean(d, t) - mean(d, c)),
                              mean(y, k) - mean(y, c), mean(y0, k) - mean(y0, c)])
    return out


def verify_fit():
    from nickerson_analysis import fit_prior
    cases = [(np.array([.02, .021, .019]), np.array([.002, .003, .002])),
             (np.array([-.2, .1, .4, .25]), np.array([.001, .002, .001, .003])),
             (np.array([-.13, -.11, .06, .25, .4]), np.array([.0001, .0002, .05, .01, .04]))]
    comparisons = []
    for y, v in cases:
        mu, gamma, _, _ = fit_prior(y, v)
        def objective(z):
            return .5 * np.sum(np.log(v + z[1]) + (y - z[0])**2 / (v + z[1]))
        result = differential_evolution(
            objective, [(float(y.min()), float(y.max())), (0., float(np.ptp(y)**2))],
            seed=417, tol=1e-11, polish=True)
        check(result.success, "Independent global optimizer failed")
        own = objective([mu, gamma])
        check(own <= result.fun + 1e-8, "Profile optimizer missed the global likelihood maximum")
        comparisons.append(float(own - result.fun))
    return comparisons


def verify_raw(data, output):
    from nickerson_analysis import analyze, prepare, stats, bootstrap_check
    manifest = json.loads((BASE / "data/nickerson/source_manifest.json").read_text())
    expected = next(x["sha256"] for x in manifest["sources"] if x["file"].endswith(".dta"))
    check(hashlib.sha256(data.read_bytes()).hexdigest() == expected, "Raw data checksum differs")
    d, counts = prepare(data)
    checks = {"raw_data_sha256": expected}
    for geo in ("Wake", "Hennepin"):
        g = d[d.geo == geo].reset_index(drop=True)
        weights = g.site.value_counts(normalize=True).to_dict()
        _, scores, _ = stats(g, weights)
        _, prior_scores, _ = stats(g, weights, "voted00")
        scores = np.column_stack([scores, prior_scores[:, 1]])
        direction = np.random.default_rng(991).normal(size=len(g))
        eps = 1e-5
        derivative = (weighted_stats(g, weights, 1 + eps * direction)
                      - weighted_stats(g, weights, 1 - eps * direction)) / (2 * eps)
        np.testing.assert_allclose(derivative, direction @ scores, atol=1e-9, rtol=1e-7)
        checks["influence_derivative_" + geo] = float(np.max(np.abs(derivative - direction @ scores)))
    for geo in ("Wake", "Cambridge"):
        g = d[d.geo == geo].reset_index(drop=True)
        weights = g.site.value_counts(normalize=True).sort_index().to_dict()
        checks["bootstrap_" + geo] = bootstrap_check(g, weights)
    with tempfile.TemporaryDirectory(prefix="nickerson-verify-") as tmp:
        regenerated = Path(tmp)
        analyze(data, output=regenerated)
        checks["reproduced_csv_cells"] = compare_results(regenerated, output)
    checks["raw_data_reproduction"] = "passed"
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS,
                        help="directory of numerical results to verify")
    parser.add_argument("--data", type=Path,
                        help="optional official .dta file for complete raw-data verification")
    args = parser.parse_args()
    checks = verify_saved(args.output)
    checks["global_mle_loss_differences"] = verify_fit()
    if args.data is not None:
        checks.update(verify_raw(args.data, args.output))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
