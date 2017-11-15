#' This function models simultaneuous adaption and produces the desired binary mask.

#' @param M Matrix of observed entries. The input should be N (number of units) by T (number of time periods).
#' @param N_t Number of treated units desired.
#' @param T0 The time just before treatment for all treated units. For instance, if T0 = 2, then first two entries of treated units are counted as control and the rest are treated.
#' @param treat_indices Optional indices for treated units. The default is sampling N_t unit from all N units randomly. However, user can manually set some units as treated.
#' @return The masked matrix which is one for control units and treated units before treatment and zero for treated units after treatment.
#' @examples 
#' simul_adapt(M = replicate(5,rnorm(5)), N_t = 3, T0 = 3)
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

#' This function models staggered adaption and produces the desired binary mask.
#' @param M Matrix of observed entries. The input should be N (number of units) by T (number of time periods).
#' @param N_t Number of treated units desired.
#' @param T0 The first treatment time. The rest of treatment times are equally spaced between T0 to T.
#' @param treat_indices Optional indices for treated units. The default is sampling N_t unit from all N units randomly. However, user can manually set some units as treated. Note that indices should be sorted increasingly based on their T0.
#' @examples 
#' stag_adapt(M = replicate(5,rnorm(5)), N_t = 3, T0 = 3)
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
