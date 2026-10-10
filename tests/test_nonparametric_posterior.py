"""Posterior integration checks with analytic and independent quadrature oracles.

Run with python3 -B -m unittest discover -s other/eb-illusion/tests
"""
import itertools
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln, logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import nonparametric_npmle as posterior


def enumerated_mean(ye, yo, support, weights, se, so):
    """Integrate every assignment analytically for small finite mixtures."""
    assignments = np.asarray(list(itertools.product(
        range(len(support)), repeat=len(yo))))
    centers = np.asarray(yo)[None, :] - np.asarray(support)[assignments]
    precision = se**-2 + len(yo)*so**-2
    means = (ye/se**2 + centers.sum(axis=1)/so**2)/precision
    logmass = (np.log(weights)[assignments].sum(axis=1)
               - .5*((ye-means)**2/se**2
                     + ((centers-means[:, None])**2).sum(axis=1)/so**2))
    return float(np.exp(logmass-logsumexp(logmass)) @ means)


def repeated_two_atom_mean(j, observation, weights, se=1., so=1.):
    """Binomial expansion for ye=0 and atoms [0, observation]."""
    k = np.arange(j+1)
    means = k*observation/so**2/(se**-2+j*so**-2)
    logmass = (gammaln(j+1)-gammaln(k+1)-gammaln(j-k+1)
               + k*np.log(weights[0])+(j-k)*np.log(weights[1])
               - .5*(means**2/se**2
                     + (k*(observation-means)**2+(j-k)*means**2)/so**2))
    return float(np.exp(logmass-logsumexp(logmass)) @ means)


def adaptive_mean(ye, yo, support, weights):
    """Whole-real-line adaptive quadrature, including both infinite tails.

    These fixtures use SE=SO=1. Width-one panels provide independent integration
    without the production grid, interpolation, or any likelihood floors.
    """
    def logpost(theta):
        return (-.5*(ye-theta)**2
                + logsumexp(np.log(weights)[None, :]
                            -.5*(yo[:, None]-theta-support[None, :])**2,
                            axis=1).sum())

    # This larger bracket contains all data-minus-atom centers and ye.
    lo = min(ye, yo.min()-support.max())-12.
    hi = max(ye, yo.max()-support.min())+12.
    edges = np.linspace(lo, hi, int(np.ceil(hi-lo))+1)
    shift = max(logpost(t) for t in np.linspace(lo, hi, 1001))
    edges = np.r_[-np.inf, edges, np.inf]

    def density(theta):
        return np.exp(logpost(theta)-shift)

    mass = moment = 0.
    for a, b in zip(edges[:-1], edges[1:]):
        mass += quad(density, a, b, epsabs=2e-12, epsrel=2e-12)[0]
        moment += quad(lambda t: t*density(t), a, b,
                       epsabs=2e-12, epsrel=2e-12)[0]
    return moment/mass


def sampled_cases():
    """Construct the first three replicates at each study count.

    Construct the data and 80-step grid fit independently of the production
    NPMLE solver for comparison with adaptive posterior quadrature.
    """
    rng = np.random.default_rng(7)

    def biases(size):
        return .5+1.3*(2*rng.integers(0, 2, size=size)-1) \
            + .35*rng.standard_normal(size)

    # The simulation generates 600 illustration observations first.
    biases(600)+rng.standard_normal(600)
    for j in (5, 10, 50, 100):
        for replicate in range(1000):
            ye = 1.+rng.standard_normal()
            yo = 1.+biases(j)+rng.standard_normal(j)
            yc = biases(j)+rng.standard_normal(j)
            if replicate >= 3:
                continue
            support = np.linspace(yc.min()-1., yc.max()+1., 40)
            likelihood = np.exp(-.5*(yc[:, None]-support[None, :])**2)
            weights = np.full(40, 1./40)
            for _ in range(80):
                joint = likelihood*weights
                weights = (joint/joint.sum(axis=1, keepdims=True)).mean(axis=0)
            yield j, replicate, ye, yo, support, weights


class PosteriorMeanTests(unittest.TestCase):
    def assert_paths(self, ye, yo, support, weights, expected,
                     fast_tolerance=2e-6, direct_tolerance=2e-10):
        for direct in (False, True):
            with self.subTest(direct=direct):
                actual = posterior.posterior_mean(
                    ye, np.asarray(yo), np.asarray(support),
                    np.asarray(weights), direct=direct)
                self.assertTrue(np.isfinite(actual))
                self.assertAlmostEqual(actual, expected,
                                       delta=direct_tolerance if direct
                                       else fast_tolerance)

    def test_small_mixtures_match_analytic_assignments_with_unequal_sds(self):
        ye, yo = .3, np.array([-1.1, 1.7, 2.1])
        support, weights = np.array([-1.2, .6, 2.]), np.array([.2, .3, .5])
        for se, so in ((1., 1.), (.7, 1.8), (1.9, .45)):
            with self.subTest(se=se, so=so), patch.multiple(posterior, SE=se, SO=so):
                expected = enumerated_mean(ye, yo, support, weights, se, so)
                self.assert_paths(ye, yo, support, weights, expected)

    def test_remote_dominant_mode_is_retained(self):
        expected = repeated_two_atom_mean(100, 20., [.9, .1])
        self.assertAlmostEqual(expected, 19.801980190583443, delta=1e-12)
        self.assert_paths(0., np.full(100, 20.), [0., 20.], [.9, .1], expected)

    def test_narrow_separated_modes_both_contribute(self):
        j, observation, se, so = 100, 7.93, 1., .02
        # Give the distant endpoint mode three times the near mode's mass.
        logodds = (.5*observation**2/(se**2+so**2/j)+np.log(3.))/j
        w = 1./(1.+np.exp(-logodds))
        weights = [w, 1.-w]
        expected = repeated_two_atom_mean(j, observation, weights, se, so)
        with patch.multiple(posterior, SE=se, SO=so):
            self.assert_paths(0., np.full(j, observation), [0., observation],
                              weights, expected)

    def test_far_single_atom_matches_gaussian_conjugacy(self):
        self.assert_paths(0., [50.], [0.], [1.], 25.)

    def test_tiny_likelihoods_preserve_relative_mixture_mass(self):
        ye, yo, support, weights = 0., [100.], [0., .3], [.9, .1]
        expected = enumerated_mean(ye, yo, support, weights, 1., 1.)
        self.assert_paths(ye, yo, support, weights, expected)

    def test_zero_weight_atoms_do_not_expand_the_domain(self):
        with np.errstate(divide='raise', invalid='raise'):
            self.assert_paths(0., [50.], [-1e6, 0., 1e6], [0., 1., 0.], 25.)

    def test_no_observations_returns_experimental_mean(self):
        self.assert_paths(2.3, [], [-1., 1.], [.4, .6], 2.3)

    def test_invalid_mixture_weights_raise_value_error(self):
        for weights in ([.5, -.5], [0., 0.], [.2, .3],
                        [np.nan, .5], [np.inf, .5], [1.]):
            for direct in (False, True):
                with self.subTest(weights=weights, direct=direct):
                    with self.assertRaises(ValueError):
                        posterior.posterior_mean(
                            0., np.array([1.]), np.array([0., 1.]),
                            np.asarray(weights), direct=direct)

    def test_sampled_current_cases_match_independent_quadrature(self):
        with patch.multiple(posterior, SE=1., SO=1.):
            for j, replicate, ye, yo, support, weights in sampled_cases():
                with self.subTest(j=j, replicate=replicate):
                    expected = adaptive_mean(ye, yo, support, weights)
                    self.assert_paths(ye, yo, support, weights, expected)


if __name__ == '__main__':
    unittest.main()
