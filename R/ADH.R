#' Computing Synthetic Control Estimator when multiple units are missing.
#' The underlying algorithm is Mirror Descent with Entropy Regularizer.
#' This algorithm is also known as exponentiated gradient descent.
#' It is worth noting that this estimator was proposed by Abadie et. al and
#' the reason of new implementation here is to compare methods on some datasets.
#' 
#' @param M Matrix of observed entries. The input should be N (number of units) by T (number of time periods).
#' @param mask Binary mask with the same shape as M containing observed entries.
#' @param niter Optional parameter on the number of iterations taken in the algorithm. The default value is 10000 and if the number of treated units are large for speed consideration will be reduced to 200.
#' @param rel_tol Optional parameter on the stopping rule. Once the relative improve in objective value drops below rel_tol, execution is halted. Default value is 1e-8. 
#' @return The matrix with all missing entries filled.
#' @seealso The R package called \code{\link[Synth]{synth}}, written by Alberto Abadie, Alexis Diamond, and Jens Hainmueller.
#' @examples
#' adh_mp_rows(matrix(c(1,2,3,4),2,2), matrix(c(1,1,1,0),2,2))

adh_mp_rows <- function(M, mask, niter = 10000, rel_tol=1e-8){
  M <- M * mask
  treated_rows <- which(rowMeans(mask) < 1)
  control_rows <- setdiff(1:nrow(M), treated_rows)
  num_treated <- length(treated_rows)
  num_control <- length(control_rows)
  M_control_rows <- M[control_rows,]
  M_pred <- M
  for (l in 1:num_treated){
    tr_row_pred <- treated_rows[l]
    tr_row_miss <- which(mask[treated_rows[l],]==0)
    A <- t(M[control_rows, -tr_row_miss])
    b <- matrix(M[tr_row_pred, -tr_row_miss], , 1)
    if(num_treated > 50){
      niter <- 200
    }
    W <- my_synth(A, b, niter, rel_tol)
    M_pred_this_row <- t(M[control_rows,]) %*% W
    M_pred[treated_rows[l],]=M_pred_this_row
  }
  return(M_pred)
}

my_synth <- function(A, b, niter, rel_tol){
  row_m <- rowMeans(A)
  mean_mat <- replicate(ncol(A),row_m)
  A <- A-mean_mat
  b <- b-row_m
  max_norm <- max(abs(A))
  A <- A/max_norm
  b <- b/max_norm
  m <- nrow(A)
  n <- ncol(A)
  w <- (1/n)*matrix(1L,n,1)
  J <- t(A) %*% A
  g <- t(A) %*% b
  obj_val <- t(w) %*% J %*% w - 2* t(w) %*% g + t(b) %*% b
  alpha <- 1
  for (t in 1:niter){
    step_size = alpha
    grad = 2*(J %*% w - g)
    w_np <- mirror_dec(w,step_size,grad)
    obj_val_n <- t(w_np) %*% J %*% w_np - 2* t(w_np) %*% g + t(b) %*% b
    rel_imp = (obj_val-obj_val_n)/obj_val
    if(obj_val_n < 1e-14){
      w <- w_np
      break
    }
    if( rel_imp <= 0 ){
      alpha <- 0.95 * alpha
    } else{
      w <- w_np
      obj_val <- obj_val_n
    }
    if( (rel_imp > 0) && (rel_imp < rel_tol) ){
      w = w_np
      break
    }
  }
  return(w)
}

mirror_dec <- function(v, alpha, grad){
  h <- v * exp(-alpha*grad) ## Exponentiated Gradient Descent (or mirror-descent for entropy regularizer)
  h <- h/sum(h)
  return (h)
}