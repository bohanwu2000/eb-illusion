import numpy as np
from cmdstanpy import CmdStanModel


def logistic(x):
    return 1 / (1 + np.exp(-x))


def simulate_studies(n_exp, n_obs, n_null,
                     n_studies, n_null_studies,
                     treat, no_treat):
    """
    Simulates experimental and observational studies
    """

    # simulate experimental data
    a_exp = np.random.binomial(1, 0.5, size=n_exp)
    probs = np.where(a_exp == 0, logistic(no_treat), logistic(treat))
    y_exp = np.random.binomial(1, probs)

    # simulate observational studies
    study = np.repeat(np.arange(1, n_studies + 1), n_obs // n_studies)
    select_prob = np.random.beta(1, 1, size=n_studies)
    y_0 = np.random.binomial(1, logistic(no_treat), size=n_obs)
    y_1 = np.random.binomial(1, logistic(treat), size=n_obs)
    y_obs = []
    a_obs = []
    for i in range(n_obs):
        a_prob_i = select_prob[study[i]-1] if (y_1[i] == 1) else 0.5
        a_obs_i = np.random.binomial(1, a_prob_i)
        y_obs_i = y_1[i] if (a_obs_i == 1) else y_0[i]
        y_obs.append(y_obs_i)
        a_obs.append(a_obs_i)

    # simulate null studies
    null_study = np.repeat(np.arange(1, n_null_studies + 1),
                           n_null // n_null_studies)
    select_prob = np.random.beta(1, 1, size=n_null_studies)
    y_0 = np.random.binomial(1, logistic(0), size=n_null)
    y_1 = np.random.binomial(1, logistic(0), size=n_null)
    y_null = []
    a_null = []
    for i in range(n_null):
        a_prob_i = select_prob[null_study[i]-1] if (y_1[i] == 1) else 0.5
        a_null_i = np.random.binomial(1, a_prob_i)
        y_null_i = y_1[i] if (a_null_i == 1) else y_0[i]
        y_null.append(y_null_i)
        a_null.append(a_null_i)

    return {'y_exp': y_exp,
            'a_exp': a_exp,
            'y_obs': y_obs,
            'a_obs': a_obs,
            'a_null': a_null,
            'y_null': y_null,
            'study': study,
            'null_study': null_study}


def form_stan_data(y_exp, a_exp,
                   y_obs, a_obs,
                   y_null, a_null,
                   study, null_study):
    """
    Form the Stan data structure
    """
    n_studies = len(np.unique(study))
    n_null_studies = len(np.unique(null_study))
    n_obs = len(y_obs)
    n_exp = len(y_exp)
    n_null = len(y_null)
    data = {
        'N_exp': n_exp,
        'N_obs': n_obs,
        'N_null': n_null,
        'N_studies': n_studies,
        'N_null_studies': n_null_studies,
        'y_exp': y_exp,
        'a_exp': a_exp,
        'y_obs': y_obs,
        'a_obs': a_obs,
        'y_null': y_null,
        'a_null': a_null,
        'study': study,
        'null_study': null_study
    }
    return data


def fit_model(y_exp, a_exp,
              y_obs, a_obs,
              y_null, a_null,
              study, null_study):
    """
    Fit the illusion model in stan
    """
    # load the stan model
    model = CmdStanModel(stan_file='illusion_experiment.stan')
    # set up the stan data
    stan_data = form_stan_data(y_exp, a_exp,
                               y_obs, a_obs,
                               y_null, a_null,
                               study, null_study)
    # fit the model
    fit = model.sample(data=stan_data,
                       output_dir='./stan_tmp/',
                       show_progress=False)
    # return the model fit
    return fit


def summarize_fit(fit):
    """
    Print mean and variance of the treat and no_treat parameters
    """
    treat_m = fit.stan_variables()['treat'].mean()
    treat_v = np.sqrt(fit.stan_variables()['treat'].var())
    no_treat_m = fit.stan_variables()['no_treat'].mean()
    no_treat_v = np.sqrt(fit.stan_variables()['no_treat'].var())
    print(f'treat   : mean = {treat_m:.2f} sd = {treat_v:.2f}')
    print(f'no_treat: mean = {no_treat_m:.2f} sd = {no_treat_v:.2f}')


def main():

    n_exp = 100
    n_obs = 2000
    n_studies = 20
    n_null = 2000
    n_null_studies = 20
    treat = 2
    no_treat = -2

    d = simulate_studies(n_exp, n_obs, n_null,
                         n_studies, n_null_studies,
                         treat, no_treat)

    fit_exp = fit_model(d['y_exp'], d['a_exp'], [], [], [], [], [], [])

    fit_obs = fit_model(d['y_exp'], d['a_exp'], d['y_obs'], d['a_obs'],
                        [], [], d['study'], [])

    fit_all = fit_model(d['y_exp'], d['a_exp'], d['y_obs'], d['a_obs'],
                        d['y_null'], d['a_null'], d['study'], d['null_study'])

    print('===== experiment only')
    summarize_fit(fit_exp)

    print('===== experiment + observational')
    summarize_fit(fit_obs)

    print('===== experiment + observational + null')
    summarize_fit(fit_all)
