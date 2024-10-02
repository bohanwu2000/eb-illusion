data {
  int<lower=0> N_exp;
  int<lower=0> N_obs;
  int<lower=0> N_studies;
  array[N_exp] int<lower=0, upper=1> y_exp;
  array[N_obs] int<lower=0, upper=1> y_obs;
  array[N_obs] int<lower=1,upper=N_studies> study;
}
parameters {
  real effect;
  array[N_studies] real bias;
  real<lower=0> bias_scale;
  real mu;
}
model {
  for (study_id in 1:N_studies) {
      // bias[study_id] ~ normal(mu, bias_scale);
      bias[study_id] ~ normal(0, bias_scale);
  }
  for (n in 1:N_exp) {
      y_exp[n] ~ bernoulli_logit(effect);
  }
  for (n in 1:N_obs) {
      y_obs[n] ~ bernoulli_logit(effect + bias[study[n]]);
  }
}
