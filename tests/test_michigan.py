"""Michigan paper regressions using public aggregate inputs and synthetic people.

Run with python3 -B -m unittest discover -s tests -p test_michigan.py -v
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / 'results' / 'don-top-districts'
BENCHMARK = ROOT / 'results' / 'don-full-data-benchmark'
sys.path.insert(0, str(ROOT / 'src'))

from don_heldout_analysis import prepare, statistics
import don_small_rct as michigan


def saved_hashes():
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for folder in (SAVED, BENCHMARK) for path in folder.rglob('*')
            if path.is_file()}


def run_script(name, *arguments):
    return subprocess.run(
        [sys.executable, '-B', str(ROOT / 'src' / name), *map(str, arguments)],
        cwd=ROOT, env={**os.environ, 'MPLBACKEND': 'Agg'},
        text=True, capture_output=True, timeout=60)


class MichiganTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_hashes = saved_hashes()
        cls.estimates = pd.read_csv(SAVED / 'district_estimates.csv')
        cls.calibration = pd.read_csv(SAVED / 'calibration_estimates.csv')
        cls.full = pd.read_csv(SAVED / 'full_sample_summary.csv')
        cls.split = pd.read_csv(SAVED / 'split_sample_summary.csv')

    @classmethod
    def tearDownClass(cls):
        if saved_hashes() != cls.original_hashes:
            raise AssertionError('Saved Michigan records changed during reconstruction.')

    def inputs(self):
        """Construct target inputs from the separately saved split summaries."""
        split = self.split
        return pd.DataFrame(dict(
            unit=split.district, districts=split.district,
            small_rct=split.experimental, small_rct_var=split.experimental_sd**2,
            obs=split.observational, var_obs=split.observational_sd**2,
            wald=split.benchmark, var_wald=split.benchmark_sd**2))

    def test_top_half_retains_all_cutoff_ties_and_53_individual_districts(self):
        screening = pd.read_csv(SAVED / 'district_screening.csv')
        ranked = screening.sort_values('contacted_voters', ascending=False)
        cutoff = ranked.iloc[51].contacted_voters
        retained = screening.loc[screening.contacted_voters >= cutoff, 'district']
        self.assertEqual(cutoff, 135)
        self.assertEqual(len(retained), 53)
        self.assertEqual(len(self.estimates), 53)
        self.assertTrue(self.estimates.unit.is_unique)
        self.assertEqual(set(retained), set(self.estimates.unit))
        np.testing.assert_array_equal(self.estimates.unit, self.estimates.districts)
        for table, key in ((self.calibration, 'unit'), (self.full, 'district'),
                           (self.split, 'district')):
            self.assertTrue(table[key].is_unique)
            np.testing.assert_array_equal(self.estimates.unit, table[key])

    def test_all_53_leave_one_out_posteriors_match_saved_records(self):
        inputs = self.inputs()
        original_inputs = inputs.copy(deep=True)
        original_calibration = self.calibration.copy(deep=True)
        actual = michigan.add_illusion(michigan.fit_posteriors(inputs, self.calibration))
        pd.testing.assert_frame_equal(inputs, original_inputs)
        pd.testing.assert_frame_equal(self.calibration, original_calibration)
        self.assertEqual(len(actual), 53)
        np.testing.assert_array_equal(actual.unit, self.estimates.unit)
        # All saved inferential columns, including intervals and mixture weights.
        np.testing.assert_allclose(actual, self.estimates[actual.columns],
                                   rtol=1e-11, atol=1e-13)
        self.assertTrue((actual[['small_rct_var', 'var_obs', 'var_wald',
                                 'posterior_sd', 'illusion_var']] > 0).all().all())
        self.assertTrue(((actual.experimental_weight > 0)
                         & (actual.experimental_weight < 1)).all())

    def test_reconstructed_posteriors_reproduce_full_data_benchmark_mse(self):
        actual = michigan.add_illusion(michigan.fit_posteriors(self.inputs(), self.calibration))
        benchmark = self.full.set_index('district').loc[actual.unit, 'wald'].to_numpy()
        expected = pd.read_csv(BENCHMARK / 'error_distribution_summary.csv').set_index('method')
        for method, column in (('experimental', 'small_rct'), ('observational', 'obs'),
                               ('illusion', 'illusion'), ('calibrated EB', 'corrected')):
            with self.subTest(method=method):
                errors = np.square(100*(actual[column].to_numpy()-benchmark))
                self.assertEqual(len(errors), 53)
                self.assertAlmostEqual(errors.mean(), expected.loc[method, 'mse_pp2'],
                                       delta=1e-10)

    def test_target_calibration_is_excluded_and_rows_align_by_district(self):
        inputs = self.inputs()
        fitted = michigan.fit_posteriors(inputs, self.calibration)
        changed = self.calibration.copy(deep=True)
        target = 89
        changed.loc[changed.unit == target, 'bias'] += 1.
        perturbed = michigan.fit_posteriors(inputs, changed)
        columns = ['mu', 'gamma2', 'corrected', 'posterior_sd']
        np.testing.assert_array_equal(
            fitted.loc[fitted.unit == target, columns],
            perturbed.loc[perturbed.unit == target, columns])
        self.assertTrue(np.any(fitted.loc[fitted.unit != target, 'mu'].to_numpy()
                               != perturbed.loc[perturbed.unit != target, 'mu'].to_numpy()))
        reordered = michigan.fit_posteriors(inputs.iloc[::-1], self.calibration.iloc[::-1])
        pd.testing.assert_frame_equal(fitted.sort_values('unit'), reordered.sort_values('unit'),
                                      rtol=1e-12, atol=1e-14)

    def test_calibration_covariance_identity_and_positive_variances(self):
        c, full = self.calibration, self.full
        np.testing.assert_allclose(c.bias, c.obs-c.wald, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(c.var_bias, c.var_obs+c.var_wald-2*c.cov_obs_wald,
                                   rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(full.calibration_sd**2,
                                   full.observational_sd**2+full.wald_sd**2
                                   -2*full.cov_wald_observational,
                                   rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(full.calibration, c.bias, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(full.calibration_sd**2, c.var_bias,
                                   rtol=1e-12, atol=1e-14)
        self.assertTrue((c[['var_obs', 'var_wald', 'var_bias']] > 0).all().all())
        self.assertTrue((c.var_obs*c.var_wald-c.cov_obs_wald**2 > 0).all())

    def test_household_influence_covariance_matches_hand_calculation(self):
        people = pd.DataFrame(dict(
            hhid_phone=np.repeat(np.arange(6), 2), arm=[0]*4+[1]*8,
            D=[0]*4+[1]*4+[0]*4, Y=[0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0]))
        actual = statistics(people)
        # Six two-person households, including the 6/5 cluster correction.
        expected = dict(n=12, households=6, contact_rate=.5, obs=.25, wald=.25,
                        bias=0., var_obs=3/80, var_wald=51/80,
                        cov_obs_wald=3/80, var_bias=3/5)
        for key, value in expected.items():
            with self.subTest(statistic=key):
                self.assertAlmostEqual(actual[key], value, delta=1e-14)

    def test_prepare_keeps_individual_districts_and_excludes_cross_district_households(self):
        # Synthetic records exercise cleaning without distributing person-level data.
        districts = list(range(1, 105)) + [13, 89]
        raw = pd.DataFrame(dict(
            treatment_check=np.ones(106, dtype=int), contact=['contacted']*106,
            p2008_final=['voted']*106, hd=[str(d) for d in districts],
            hhid_phone=list(range(104))+[999, 999]))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'synthetic.dta'
            raw.to_stata(path, write_index=False)
            cleaned = prepare(path)
        self.assertEqual(len(cleaned), 104)
        np.testing.assert_array_equal(cleaned.unit, cleaned.district)
        self.assertEqual(cleaned.attrs, dict(raw_n=106, excluded_households=1,
                                             excluded_people=2))

    def test_full_benchmark_errors_and_district_89_intervals_match_saved_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            result = run_script('don_error_distribution.py', '--baseline',
                                SAVED / 'district_estimates.csv', '--posterior-unit', 89,
                                '--benchmark', 'full', '--output', output)
            self.assertEqual(result.returncode, 0, result.stderr)
            for name in ('full_data_squared_errors.csv', 'error_distribution_summary.csv',
                         'evaluation_benchmarks.csv'):
                with self.subTest(output=name):
                    pd.testing.assert_frame_equal(pd.read_csv(output / name),
                                                  pd.read_csv(BENCHMARK / name),
                                                  rtol=1e-12, atol=1e-12)
            errors = pd.read_csv(output / 'full_data_squared_errors.csv')
            self.assertEqual(len(errors), 4*53)
            self.assertEqual(errors.groupby('method').district.nunique().to_list(), [53]*4)
            actual = json.loads((output / 'error_distribution_metadata.json').read_text())
            expected = json.loads((BENCHMARK / 'error_distribution_metadata.json').read_text())
            self.assertEqual(actual['benchmark'], 'full')
            for key in ('district', 'J', 'K', 'means_pp', 'intervals_95_pp'):
                self.assertEqual(actual['posterior_panel'][key], expected['posterior_panel'][key])
            self.assertGreater((output / 'full_data_error_distribution.pdf').stat().st_size, 0)

    def test_cached_replay_preserves_every_estimate_record(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            result = run_script('don_small_rct.py', '--estimates',
                                SAVED / 'district_estimates.csv', '--output', output)
            self.assertEqual(result.returncode, 0, result.stderr)
            pd.testing.assert_frame_equal(pd.read_csv(output / 'district_estimates.csv'),
                                          self.estimates, rtol=1e-12, atol=1e-13)
            michigan.export_summary_tables(self.estimates, self.calibration, output)
            for name in ('full_sample_summary.csv', 'split_sample_summary.csv'):
                with self.subTest(output=name):
                    pd.testing.assert_frame_equal(pd.read_csv(output / name),
                                                  pd.read_csv(SAVED / name),
                                                  rtol=1e-12, atol=1e-13)

    def test_retired_analysis_flags_are_rejected_before_reading_data(self):
        with tempfile.TemporaryDirectory() as folder:
            for script, arguments, flag in (
                ('don_small_rct.py', ['--data', 'unused.dta', '--paired'], '--paired'),
                ('don_error_distribution.py', ['--baseline', 'unused.csv',
                 '--posterior-unit', '89', '--comparison', 'unused.csv'], '--comparison')):
                with self.subTest(script=script):
                    result = run_script(script, *arguments, '--output', folder)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn('unrecognized arguments: '+flag, result.stderr)


if __name__ == '__main__':
    unittest.main()
