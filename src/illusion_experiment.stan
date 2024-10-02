data {
  int<lower=0> N_exp;
  int<lower=0> N_obs;
  int<lower=0> N_studies;
  int<lower=0> N_null;
  int<lower=0> N_null_studies;
  array[N_exp] int<lower=0, upper=1> y_exp;
  array[N_exp] int<lower=0, upper=1> a_exp;
  array[N_obs] int<lower=0, upper=1> y_obs;
  array[N_obs] int<lower=0, upper=1> a_obs;
  array[N_obs] int<lower=1,upper=N_studies> study;
  array[N_null] int<lower=0, upper=1> y_null;
  array[N_null] int<lower=0, upper=1> a_null;
  array[N_null] int<lower=1,upper=N_studies> null_study;
}
parameters {
  real treat;
  real no_treat;
  array[N_studies] real bias_0;
  array[N_studies] real bias_1;
  array[N_null_studies] real null_bias_0;
  array[N_null_studies] real null_bias_1;
  real mu_0;
  real mu_1;
  real<lower=0> bias_scale_0;
  real<lower=0> bias_scale_1;
}
model {
  for (study_id in 1:N_studies) {
      bias_0[study_id] ~ normal(mu_0, bias_scale_0);
      bias_1[study_id] ~ normal(mu_1, bias_scale_1);
  }
  for (study_id in 1:N_null_studies) {
      null_bias_0[study_id] ~ normal(mu_0, bias_scale_0);
      null_bias_1[study_id] ~ normal(mu_1, bias_scale_1);
  }
  for (n in 1:N_exp) {
      y_exp[n] ~ bernoulli_logit((1 - a_exp[n]) * no_treat + a_exp[n] * treat);
  }
  for (n in 1:N_obs) {
      real no_treat_n = (1 - a_obs[n]) * (no_treat + bias_0[study[n]]);
      real treat_n = a_obs[n] * (treat + bias_1[study[n]]);
      y_obs[n] ~ bernoulli_logit(no_treat_n + treat_n);
  }
  for (n in 1:N_null) {
    real no_treat_n = (1 - a_null[n]) * (null_bias_0[null_study[n]]);
    real treat_n = a_null[n] * (null_bias_1[study[n]]);
    y_null[n] ~ bernoulli_logit(no_treat_n + treat_n);
  }

}
