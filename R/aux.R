simul_adapt <- function(M, N_t, T0, treat_indices=0){
  N = nrow(M)
  T = ncol(M)
  treat_mat <- matrix(1L, N, T);
  if(treat_indices[1] == 0){
    treat_indices <- sample(1:N, N_t)
  }
  for (i in 1:N_t){
    treat_mat[treat_indices[i],(T0+1):T] = 0
  }
  return(treat_mat)
}

stag_adapt <- function(M, N_t, T0, treat_indices=0){
  N = nrow(M)
  T = ncol(M)
  treat_mat <- matrix(1L, N, T);
  if(treat_indices[1] == 0){
    treat_indices <- sample(1:N, N_t)
  }
  for (i in 1:N_t){
    last_cont_time_pr = floor(T0+(T-T0)*(i-1)/N_t);
    treat_mat[treat_indices[i],(last_cont_time_pr+1):T]=0;
  }
  return(treat_mat)
}
