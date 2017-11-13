logspace <- function( d1, d2, n) exp(log(10)*seq(d1, d2, length.out=n))

en_predict <- function(M, mask, best_lam, best_alpha){
  treated_row <- which(rowMeans(mask) < 1)
  treated_cols <- which(mask[treated_row,]==0)
  control_cols <- setdiff(1:ncol(M),treated_cols)
  M_new <- M
  M_new[treated_row,] <- M[nrow(M),]
  M_new[nrow(M),] <- M[treated_row,]
  Z_train <- M_new[1:(nrow(M_new)-1),control_cols]
  to_pred <- M_new[nrow(M_new),control_cols]
  if(length(which(to_pred==0))==length(control_cols)){
    weights = matrix(0L, nrow(M_new)-1,1)
    intc = 0
  }
  else{
    A = glmnet(t(Z_train), M_new[nrow(M_new),control_cols],'gaussian',lambda=best_lam, alpha=best_alpha, thresh=1e-4)
    weights <- unname(A$beta[,1])
    intc <- unname(A$a0[1])
  }
  M_pred = t(M_new[1:nrow(M_new)-1,]) %*% weights+intc*matrix(1L,ncol(M_new),1);
  return(M_pred)
}

en_mp_rows <- function(M, mask, num_folds = 5, num_lam = 100L, num_alpha = 40L){
  M <- M * mask
  treated_rows <- which(rowMeans(mask) < 1)
  control_rows <- setdiff(1:nrow(M), treated_rows)
  num_treated <- length(treated_rows)
  num_control <- length(control_rows)
  M_control_rows <- M[control_rows,]
  M_pred <- M
  for (l in 1:num_treated){
    mask_fake <- matrix(1L,num_control+1,ncol(mask))
    tr_row_pred <- treated_rows[l]
    tr_row_miss <- which(mask[treated_rows[l],]==0)
    M_fake <- rbind(M_control_rows, M[tr_row_pred,])
    mask_fake[nrow(mask_fake),tr_row_miss] = 0
    M_pred_this_row = en_cv_single_row( M_fake, mask_fake, num_folds, num_alpha );
    M_pred[treated_rows[l],]=M_pred_this_row
  }
  return(M_pred)
}

en_cv_single_row <- function(M, mask, num_folds = 5, num_alpha = 40L){
  if(num_alpha == 1){
    alpha <- 1
  }
  else{
    alpha <- seq(1e-4,1, length.out = num_alpha)
  }

  M <- M * mask
  treated_row <- which(rowMeans(mask) < 1)
  treated_cols <- which(mask[treated_row,] ==0 )
  control_rows = setdiff(nrow(M),treated_row)
  control_cols = setdiff(1:ncol(M),treated_cols)
  num_controls <- length(control_cols)
  if(num_controls >= num_folds){
    fold_length <- floor(num_controls/num_folds)
    remin <- num_controls - fold_length*num_folds
    M_new <- M
    M_new[treated_row,] <- M[nrow(M),]
    M_new[nrow(M),] <- M[treated_row,]
    MSE <- array(0,num_alpha)
    lambda_opt <- array(0,num_alpha)
    rand_perm <- sample(1:num_controls,num_controls)
    fold_id <- matrix(0L,num_controls)
    st_ind <- 1
    for (fold in c(1:num_folds)){
      if(fold <= remin){
        fold_id[rand_perm[st_ind:(st_ind+fold_length)]] = fold
        st_ind <- st_ind+fold_length+1
      }
      else{
        fold_id[rand_perm[st_ind:(st_ind+fold_length-1)]] = fold
        st_ind <- st_ind+fold_length
      }
    }
    Z_train <- M_new[1:(nrow(M_new)-1),control_cols];
    for (j in 1:num_alpha){
      A=cv.glmnet(t(Z_train), M_new[nrow(M_new),control_cols], family = 'gaussian', alpha = alpha[j], thresh=1e-4, foldid=fold_id);
      MSE[j]=min(A$cvm)
      lambda_opt[j]=A$lambda.min
    }
    best_alpha_ind <- which(MSE == min(MSE), arr.ind=TRUE)
    best_lam <- lambda_opt[best_alpha_ind]
    best_alpha <- alpha[best_alpha_ind]
  }
  else{
    lambda <- logspace(3, -4, 100)
    best_lam <- lambda[sample(1:100,1)]
    best_alpha <- alpha[sample(1:num_alpha,1)]
  }
  return(en_predict(M, mask , best_lam, best_alpha));
}


