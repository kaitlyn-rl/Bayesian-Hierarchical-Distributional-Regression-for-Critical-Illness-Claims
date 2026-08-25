# Fit the six selected Bayesian count-regression specifications.

suppressPackageStartupMessages({
  library(brms)
  library(rstan)
  library(readr)
  library(dplyr)
})

rstan_options(auto_write = TRUE)

script_arg <- grep("^--file=", commandArgs(), value = TRUE)
script_path <- if (length(script_arg)) sub("^--file=", "", script_arg[[1]]) else "01_fit_models.R"
script_dir <- normalizePath(dirname(script_path), mustWork = TRUE)
project_dir <- normalizePath(file.path(script_dir, "..", ".."), mustWork = TRUE)
results_dir <- file.path(project_dir, "results")
fit_dir <- file.path(results_dir, "fits")
dir.create(fit_dir, recursive = TRUE, showWarnings = FALSE)

ci <- readRDS(file.path(results_dir, "ci_validated.rds"))

env_true <- function(name, default = FALSE) {
  value <- Sys.getenv(name, if (default) "1" else "0")
  tolower(value) %in% c("1", "true", "yes", "y")
}
env_integer <- function(name, default) {
  value <- suppressWarnings(as.integer(Sys.getenv(name, as.character(default))))
  if (is.na(value)) stop(name, " must be an integer.")
  value
}

CHAINS <- env_integer("CI_CHAINS", 4L)
ITER <- env_integer("CI_ITER", 2000L)
WARMUP <- env_integer("CI_WARMUP", 1000L)
detected_cores <- parallel::detectCores(logical = FALSE)
if (is.na(detected_cores)) detected_cores <- 1L
CORES <- env_integer("CI_CORES", min(CHAINS, detected_cores))
REFRESH <- env_integer("CI_REFRESH", 100L)
FORCE_REFIT <- env_true("CI_FORCE_REFIT")
MAX_TREEDEPTH <- 14L
stopifnot(
  ITER > WARMUP,
  WARMUP > 0L,
  CORES > 0L,
  REFRESH >= 0L
)

options(mc.cores = CORES)

mean_rhs <- paste("A.std",
                  "SmokerStatus",
  "DurationGroup",
  "DistributionChannel",
  "Year",
  "ProductCategory",
  "SumAssuredBand",
  "Gender",
  "A.std * SmokerStatus",
  sep = " + "
)
dispersion_rhs <- paste(
  "SmokerStatus", "ProductCategory", "SumAssuredBand", "Year", sep = " + "
)
zip_rhs <- paste("A.std", "SmokerStatus", "Gender", sep = " + ")

mean_formula <- as.formula(paste0(
  "TotalIncurredClaims ~ offset(log(TotalExposure)) + ", mean_rhs
))
shape_formula <- as.formula(paste0("shape ~ ", dispersion_rhs))
zip_formula <- as.formula(paste0("zi ~ ", zip_rhs))
intercept_only_zi <- as.formula("zi ~ 1")

write_lines(
  c(
    paste("Mean:", deparse(mean_formula)),
    paste("Precision/log-variance:", deparse(shape_formula)),
    paste("ZIP extra-zero:", deparse(zip_formula)),
    "ZINB and ZIPLN extra-zero: zi ~ 1"
  ),
  file.path(results_dir, "selected_model_formulas.txt")
)

prior_mean <- c(
  prior(normal(0, 100), class = "Intercept"),
  prior(normal(0, 100), class = "b")
)
prior_precision <- c(
  prior(normal(0, 100), class = "Intercept", dpar = "shape"),
  prior(normal(0, 100), class = "b", dpar = "shape")
)
prior_zi <- c(
  prior(normal(0, 100), class = "Intercept", dpar = "zi"),
  prior(normal(0, 100), class = "b", dpar = "zi")
)

fit_or_load_brms <- function(name, formula, family, prior) {
  fit_path <- file.path(fit_dir, paste0(name, ".rds"))
  if (file.exists(fit_path) && !FORCE_REFIT) {
    message("Loading existing ", fit_path)
    return(readRDS(fit_path))
  }
  message("Fitting ", name)
  fit <- brm(
    formula = formula,
    data = ci,
    family = family,
    prior = prior,
    iter = ITER,
    warmup = WARMUP,
    chains = CHAINS,
    cores = CORES,
    refresh = REFRESH,
    backend = "rstan",
    control = list(adapt_delta = 0.98, max_treedepth = MAX_TREEDEPTH),
    save_pars = save_pars(all = TRUE)
  )
  saveRDS(fit, fit_path, compress = FALSE)
  fit
}

fit_pois <- fit_or_load_brms(
  "poisson_selected",
  mean_formula,
  poisson(link = "log"),
  prior_mean
)

fit_nb <- fit_or_load_brms(
  "negative_binomial_selected",
  bf(mean_formula, shape_formula),
  negbinomial(link = "log", link_shape = "log"),
  c(prior_mean, prior_precision)
)

fit_zip <- fit_or_load_brms(
  "zero_inflated_poisson_selected",
  bf(mean_formula, zip_formula),
  zero_inflated_poisson(link = "log", link_zi = "logit"),
  c(prior_mean, prior_zi)
)

fit_zinb <- fit_or_load_brms(
  "zero_inflated_negative_binomial_selected",
  bf(mean_formula, shape_formula, intercept_only_zi),
  zero_inflated_negbinomial(
    link = "log", link_shape = "log", link_zi = "logit"
  ),
  c(
    prior_mean,
    prior_precision,
    prior(normal(0, 100), class = "Intercept", dpar = "zi")
  )
)

# Design matrices for the custom mean-centred PLN models.  
X_mu <- model.matrix(as.formula(paste0("~ ", mean_rhs)), data = ci)
Z_sigma <- model.matrix(as.formula(paste0("~ ", dispersion_rhs)), data = ci)
W_zi_intercept <- model.matrix(~ 1, data = ci)

stan_data_pln <- list(
  N = nrow(ci),
  y = as.integer(ci$TotalIncurredClaims),
  log_exposure = log(ci$TotalExposure),
  P = ncol(X_mu),
  X_mu = X_mu,
  Q = ncol(Z_sigma),
  Z_sigma = Z_sigma
)

stan_data_zipln <- c(
  stan_data_pln,
  list(R = ncol(W_zi_intercept), W_zi = W_zi_intercept)
)

fit_or_load_rstan <- function(name, stan_file, stan_data, pars) {
  fit_path <- file.path(fit_dir, paste0(name, ".rds"))
  if (file.exists(fit_path) && !FORCE_REFIT) {
    message("Loading existing ", fit_path)
    return(readRDS(fit_path))
  }
  message("Fitting ", name, "; this model has one latent effect per rating cell.")
  fit <- stan(
    file = stan_file,
    data = stan_data,
    iter = ITER,
    warmup = WARMUP,
    chains = CHAINS,
    cores = CORES,
    refresh = REFRESH,
    pars = pars,
    include = TRUE,
    control = list(adapt_delta = 0.98, max_treedepth = MAX_TREEDEPTH)
  )
  saveRDS(fit, fit_path, compress = FALSE)
  fit
}

fit_pln <- fit_or_load_rstan(
  "poisson_lognormal_selected",
  file.path(script_dir, "poisson_lognormal_mean_centered.stan"),
  stan_data_pln,
  c("beta", "delta")
)

fit_zipln <- fit_or_load_rstan(
  "zero_inflated_poisson_lognormal_selected",
  file.path(script_dir, "zero_inflated_poisson_lognormal_mean_centered.stan"),
  stan_data_zipln,
  c("beta", "delta", "gamma")
)

# Compact convergence summaries; full sampler diagnostics remain in the fit files.
write_brms_diagnostics <- function(fit, name) {
  summary_table <- as.data.frame(posterior::summarise_draws(
    posterior::as_draws(fit),
    "mean", "sd", "rhat", "ess_bulk", "ess_tail"
  ))
  write_csv(summary_table, file.path(results_dir, paste0(name, "_diagnostics.csv")))
}

write_rstan_diagnostics <- function(fit, name) {
  tab <- as.data.frame(summary(fit)$summary)
  tab$parameter <- rownames(tab)
  rownames(tab) <- NULL
  write_csv(tab, file.path(results_dir, paste0(name, "_diagnostics.csv")))
}

write_brms_diagnostics(fit_pois, "poisson")
write_brms_diagnostics(fit_nb, "negative_binomial")
write_brms_diagnostics(fit_zip, "zero_inflated_poisson")
write_brms_diagnostics(fit_zinb, "zero_inflated_negative_binomial")
write_rstan_diagnostics(fit_pln, "poisson_lognormal")
write_rstan_diagnostics(fit_zipln, "zero_inflated_poisson_lognormal")

summarise_sampler <- function(stanfit, model) {
  sampler <- rstan::get_sampler_params(stanfit, inc_warmup = FALSE)
  bind_rows(lapply(seq_along(sampler), function(chain_id) {
    chain <- sampler[[chain_id]]
    energy <- chain[, "energy__"]
    energy_variance <- stats::var(energy)
    tibble(
      model = model,
      chain = chain_id,
      post_warmup_draws = nrow(chain),
      divergences = sum(chain[, "divergent__"]),
      treedepth_hits = sum(chain[, "treedepth__"] >= MAX_TREEDEPTH),
      maximum_treedepth = max(chain[, "treedepth__"]),
      ebfmi = if (length(energy) > 1L && is.finite(energy_variance) &&
                    energy_variance > 0) {
        mean(diff(energy)^2) / energy_variance
      } else {
        NA_real_
      }
    )
  }))
}

sampler_diagnostics <- bind_rows(
  summarise_sampler(fit_pois$fit, "Poisson"),
  summarise_sampler(fit_nb$fit, "NB"),
  summarise_sampler(fit_zip$fit, "ZIP"),
  summarise_sampler(fit_zinb$fit, "ZINB"),
  summarise_sampler(fit_pln, "PLN"),
  summarise_sampler(fit_zipln, "ZIPLN")
)
write_csv(
  sampler_diagnostics,
  file.path(results_dir, "sampler_diagnostics.csv")
)

capture.output(sessionInfo(), file = file.path(results_dir, "sessionInfo.txt"))
message("Selected fits and diagnostics are complete.")
