
mcnnm_cv <- function(M, mask, to_estimate_u = 0, to_estimate_v = 0, num_lam = 100, niter = 1000, rel_tol = 1e-5, cv_ratio = 0.8, num_folds =5, is_quiet = 1){
  k <- NNM_CV(M, mask, to_estimate_u, to_estimate_v, num_lam, niter, rel_tol, cv_ratio, num_folds, is_quiet)
  return(k)
}


mcnnm <- function(M, mask, lambda_L, to_estimate_u = 0, to_estimate_v = 0, niter = 1000, rel_tol = 1e-5, is_quiet = 1){
  k <- NNM(M, mask, to_estimate_u, to_estimate_v, lambda_L, niter, rel_tol, is_quiet)
  return(k)
}

mcnnm_fit <- function(M , mask, lambda_L, L_init = matrix(0,nrow(M),ncol(M)), u_init = matrix(0,nrow(M),1), v_init=matrix(0,ncol(M),1), to_estimate_u = 0, to_estimate_v = 0, niter = 1000, rel_tol = 1e-5, is_quiet = 1){
  k <- NNM_fit(M, mask, L_init, u_init, v_init, to_estimate_u, to_estimate_v, lambda_L, niter, rel_tol, is_quiet)
  return(k)
}
