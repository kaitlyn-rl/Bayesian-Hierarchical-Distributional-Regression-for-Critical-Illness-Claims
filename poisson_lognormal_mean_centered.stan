// Mean-centred Poisson log-normal model for aggregated CI rating cells.

data {
  int<lower=1> N;
  array[N] int<lower=0> y;
  vector[N] log_exposure;

  int<lower=1> P;
  matrix[N, P] X_mu;

  int<lower=1> Q;
  matrix[N, Q] Z_sigma;
}

parameters {
  vector[P] beta;
  vector[Q] delta;
  vector[N] u_std;
}

transformed parameters {
  vector[N] eta = X_mu * beta;
  vector[N] log_sigma2 = Z_sigma * delta;
  vector[N] sigma = exp(0.5 * log_sigma2);
  vector[N] log_count_mean =
    log_exposure + eta - 0.5 * square(sigma) + sigma .* u_std;
}

model {
  // Scale-aware regularising priors.  The first design-matrix column must be
  // the intercept.
  beta[1] ~ normal(-6, 2);
  if (P > 1)
    beta[2:P] ~ normal(0, 1);

  delta[1] ~ normal(-2, 1.5);
  if (Q > 1)
    delta[2:Q] ~ normal(0, 0.75);

  u_std ~ std_normal();
  y ~ poisson_log(log_count_mean);
}

generated quantities {
  vector[N] annual_mean_rate = exp(eta);
}
