library(MCPanel)

N <- 100 # Number of units
T <- 100 # Number of time-periods
R <- 5 # Rank of matrix
noise_sc <- 0.1 # Noise scale
delta_sc <- 0.1 # delta scale
gamma_sc <- 0.1 # gamma scale
fr_obs <- 0.8 # fraction of observed entries

## Create Matrices
A <- replicate(R,rnorm(N))
B <- replicate(T,rnorm(R))
delta <- delta_sc*rnorm(N)
gamma <- gamma_sc*rnorm(T)
noise <- noise_sc*replicate(T,rnorm(N))
true_mat <- A %*% B + replicate(T,delta) + t(replicate(N,gamma))
noisy_mat <- true_mat + noise
mask <- matrix(rbinom(N*T,1,fr_obs),N,T)
obs_mat <- noisy_mat * mask

## Estimate using mcnnm_cv (cross-validation) on lambda values
model_without_effects <- mcnnm_cv(obs_mat,mask)
model_with_delta <- mcnnm_cv(obs_mat, mask, 1, 0) ##third and fourth parameter respectively are whether
model_with_gamma <- mcnnm_cv(obs_mat, mask, 0, 1) ##to estimate delta(u) or gamma(v)
model_with_both <- mcnnm_cv(obs_mat, mask, 1, 1)

## Check criteria
sum(model_without_effects$u == 0) == N ## Checking if row-wise effects are zero
sum(model_without_effects$v == 0) == T ## Checking if column-wise effects are zero
#
sum(model_with_delta$u == 0) == N
sum(model_with_delta$v == 0) == T
#
sum(model_with_gamma$u == 0) == N
sum(model_with_gamma$v == 0) == T
#
sum(model_with_both$u == 0) == N
sum(model_with_both$v == 0) == T
#

## Comparing minimum RMSEs

model_without_effects$min_RMSE
model_with_delta$min_RMSE
model_with_gamma$min_RMSE
model_with_both$min_RMSE

## Construct estimations based on models

model_without_effects$est <- model_without_effects$L + replicate(T,model_without_effects$u) + t(replicate(N,model_without_effects$v))
model_with_delta$est <- model_with_delta$L + replicate(T,model_with_delta$u) + t(replicate(N,model_with_delta$v))
model_with_gamma$est <- model_with_gamma$L + replicate(T,model_with_gamma$u) + t(replicate(N,model_with_gamma$v))
model_with_both$est <- model_with_both$L + replicate(T,model_with_both$u) + t(replicate(N,model_with_both$v))

## Compute error matrices

model_without_effects$err <- model_without_effects$est - true_mat
model_with_delta$err <- model_with_delta$est - true_mat
model_with_gamma$err <- model_with_gamma$est - true_mat
model_with_both$err <- model_with_both$est - true_mat

## Compute masked error matrices

model_without_effects$msk_err <- model_without_effects$err*(1-mask)
model_with_delta$msk_err <- model_with_delta$err*(1-mask)
model_with_gamma$msk_err <- model_with_gamma$err*(1-mask)
model_with_both$msk_err <- model_with_both$err*(1-mask)

## Compute RMSE on test set

model_without_effects$test_RMSE <- sqrt((1/sum(1-mask)) * sum(model_without_effects$msk_err^2))
model_with_delta$test_RMSE <- sqrt((1/sum(1-mask)) * sum(model_with_delta$msk_err^2))
model_with_gamma$test_RMSE <- sqrt((1/sum(1-mask)) * sum(model_with_gamma$msk_err^2))
model_with_both$test_RMSE <- sqrt((1/sum(1-mask)) * sum(model_with_both$msk_err^2))

## Print RMSE on test set
print(model_without_effects$test_RMSE)
print(model_with_delta$test_RMSE)
print(model_with_gamma$test_RMSE)
print(model_with_both$test_RMSE)

