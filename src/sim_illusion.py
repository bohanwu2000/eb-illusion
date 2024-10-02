import numpy as np
from cmdstanpy import CmdStanModel
import pdb


def logistic(x):
    return 1 / (1 + np.exp(-x))


def simulate_studies(n_exp, n_obs, n_studies, effect, bias_scale):
    """
    Simulates experimental and observational studies
    """
    y_exp = np.random.binomial(n=1, p=logistic(effect), size=n_exp)
    study = np.repeat(np.arange(1, n_studies + 1), n_obs // n_studies)
    bias = np.random.normal(loc=0, scale=bias_scale, size=n_studies)
    obs_probs = logistic(effect + bias[study - 1])
    y_obs = np.random.binomial(1, obs_probs)
    return {'y_exp': y_exp, 'y_obs': y_obs, 'study': study}


def form_stan_data(y_exp, y_obs, study):
    """
    Form the Stan data structure
    """
    n_studies = len(np.unique(study))
    n_obs = len(y_obs)
    n_exp = len(y_exp)
    data = {
        'N_exp': n_exp,
        'N_obs': n_obs,
        'N_studies': n_studies,
        'y_exp': y_exp,
        'y_obs': y_obs,
        'study': study
    }
    return data


def fit_model(y_exp, y_obs, study):
    """
    Fit the illusion model in stan
    """
    # load the stan model
    model = CmdStanModel(stan_file='illusion.stan')
    # set up the stan data
    stan_data = form_stan_data(y_exp, y_obs, study)
    # fit the model
    fit = model.sample(data=stan_data,
                       output_dir='./stan_tmp/',
                       show_progress=False)
    # return the model fit
    return fit


def print_effect(fit):
    effect = fit.stan_variables()['effect']
    mean = effect.mean()
    sd = np.sqrt(effect.var())
    print(f'mean: {mean:.2f} sd: {sd:.2f}')


def main():

    n_exp = 10
    n_obs = 1000
    n_studies = 25
    effect = 1.5
    bias_scale = 1

    data = simulate_studies(n_exp, n_obs, n_studies, effect, bias_scale)
    y_exp = data['y_exp']
    y_obs = data['y_obs']
    study = data['study']

    fit_all = fit_model(y_exp, y_obs, study)
    fit_exp = fit_model(y_exp, [1], [1])
    fit_one = fit_model(y_exp, y_obs[study == 1], study[study == 1])

    print('=== all studies')
    print_effect(fit_all)
    print('=== experiment only')
    print_effect(fit_exp)
    print('=== experiment and one observational study')
    print_effect(fit_one)

    return {'data': data,
            'fit_all': fit_all,
            'fit_exp': fit_exp,
            'fit_one': fit_one}
