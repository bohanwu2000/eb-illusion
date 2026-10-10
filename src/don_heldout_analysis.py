"""Core Michigan cleaning, household covariance, and bias-prior estimation.

The current analysis in don_small_rct.py uses individual state-house districts.
Person-level Stata inputs are supplied separately and stay outside the repository.
Calibration is observational minus Wald. Household influence functions retain
both their covariance and the estimated contact rate.
"""
import numpy as np
import pandas as pd


def prepare(path):
    """Clean household records and use individual districts as analysis units."""
    d = pd.read_stata(path)
    d['arm'] = d.treatment_check.astype(int)
    d['D'] = (d.contact == 'contacted').astype(float)
    d['Y'] = (d.p2008_final == 'voted').astype(float)
    d['district'] = d.hd.str.strip().astype(int)
    assert d[['arm', 'D', 'Y', 'district', 'hhid_phone']].notna().all().all()
    assert d.loc[d.arm == 1, 'district'].nunique() == 104
    d['unit'] = d.district
    hh = d.groupby('hhid_phone')[['unit', 'arm', 'D']].nunique()
    assert (hh[['arm', 'D']] <= 1).all().all(), 'Households cross arms/contact groups'
    # A few two-person households have inconsistent district codes. Exclude
    # both records when they cross districts, keeping validation independent.
    bad = hh.index[hh.unit > 1]
    raw_n = len(d)
    d = d.loc[~d.hhid_phone.isin(bad)].copy()
    d.attrs['raw_n'] = raw_n
    d.attrs['excluded_households'] = len(bad)
    d.attrs['excluded_people'] = raw_n-len(d)
    return d


def statistics(g):
    g = g[g.arm.isin([0, 1])].copy()
    y, contact = g.Y.to_numpy(), g.D.to_numpy()
    t, c = (g.arm == 1).to_numpy(), (g.arm == 0).to_numpy()
    tc, tu = t & (contact == 1), t & (contact == 0)
    nt, nc, nk, nu = t.sum(), c.sum(), tc.sum(), tu.sum()
    assert min(nt, nc, nk, nu) > 0
    mt, mc, mk, mu = y[t].mean(), y[c].mean(), y[tc].mean(), y[tu].mean()
    p = nk / nt
    obs, wald = mk-mu, (mt-mc)/p
    score_o = tc*(y-mk)/nk - tu*(y-mu)/nu
    score_w = t*((y-mt)-wald*(contact-p))/(nt*p) - c*(y-mc)/(nc*p)
    # Bias simplifies algebraically; verify both point and influence formulas.
    bias = obs-wald
    np.testing.assert_allclose(bias, (mc-mu)/p, atol=1e-12)
    score_b = c*(y-mc)/(nc*p)-tu*(y-mu)/(nu*p)-bias*t*(contact-p)/(nt*p)
    np.testing.assert_allclose(score_o-score_w, score_b, atol=1e-12)
    scores = pd.DataFrame({'o': score_o, 'w': score_w, 'hh': g.hhid_phone.to_numpy()})
    cluster = scores.groupby('hh')[['o', 'w']].sum().to_numpy()
    np.testing.assert_allclose(cluster.sum(0), 0, atol=1e-12)
    cov = cluster.T @ cluster * len(cluster)/(len(cluster)-1)
    vb = cov[0, 0]+cov[1, 1]-2*cov[0, 1]
    # An unused observational contrast can have zero sample variance in a
    # small experimental split. The joint covariance may be semidefinite.
    assert vb > 0 and np.linalg.eigvalsh(cov).min() >= -1e-12
    return dict(n=int(len(g)), households=len(cluster), contact_rate=p,
                obs=obs, wald=wald, bias=bias, var_obs=cov[0, 0],
                var_wald=cov[1, 1], cov_obs_wald=cov[0, 1], var_bias=vb)


def fit_bias(y, v):
    gamma2 = np.maximum(np.mean((y-y.mean(axis=-1, keepdims=True))**2-v, axis=-1), 0)
    weights = 1/(v+gamma2[..., None])
    mu = (weights*y).sum(-1)/weights.sum(-1)
    return mu, gamma2, 1/weights.sum(-1)
