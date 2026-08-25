// Mean-centred zero-inflated Poisson log-normal model

data {
  int<lower=1> N;
  array[N] int<lower=0> y;
  vector[N] log_exposure;

  int<lower=1> P;
  matrix[N, P] X_mu;

  int<lower=1> Q;
  matrix[N, Q] Z_sigma;

  int<lower=1> R;
  matrix[N, R] W_zi;
}

parameters {
  vector[P] beta;
  vector[Q] delta;
  vector[R] gamma;
  vector[N] u_std;
}

transformed parameters {
  vector[N] eta = X_mu * beta;
  vector[N] log_sigma2 = Z_sigma * delta;
  vector[N] sigma = exp(0.5 * log_sigma2);
  vector[N] log_count_mean =
    log_exposure + eta - 0.5 * square(sigma) + sigma .* u_std;
  vector[N] logit_pi = W_zi * gamma;
}

model {
  beta[1] ~ normal(-6, 2);
  if (P > 1)
    beta[2:P] ~ normal(0, 1);

  delta[1] ~ normal(-2, 1.5);
  if (Q > 1)
    delta[2:Q] ~ normal(0, 0.75);

  gamma[1] ~ normal(-3, 1.5);
  if (R > 1)
    gamma[2:R] ~ normal(0, 1);

  u_std ~ std_normal();

  for (i in 1:N) {
    if (y[i] == 0) {
      target += log_sum_exp(
        log_inv_logit(logit_pi[i]),
        log1m_inv_logit(logit_pi[i]) +
          poisson_log_lpmf(0 | log_count_mean[i])
      );
    } else {
      target += log1m_inv_logit(logit_pi[i]) +
                poisson_log_lpmf(y[i] | log_count_mean[i]);
    }
  }
}

generated quantities {
  vector[N] annual_mean_rate;
  for (i in 1:N)
    annual_mean_rate[i] = (1 - inv_logit(logit_pi[i])) *
                          exp(eta[i]);
}
