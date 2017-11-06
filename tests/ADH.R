adh_mp_rows <- function(M, mask, niter = 20000, rel_tol=1e-10){
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
      niter <- 1000
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
    if( rel_imp <= 0 ){
      alpha <- 0.95 * alpha
    } else{
      w <- w_np;
      obj_val <- obj_val_n;
    }
    if( (rel_imp > 0) && (rel_imp < rel_tol) ){
      w = w_np;
      break;
    }
  }
}

mirror_dec <- function(v, alpha, grad){
  h <- v * exp(-alpha*grad) ## Exponentiated Gradient Descent (or mirror-descent for entropy regularizer)
  h <- h/sum(h)
  return (h)
}