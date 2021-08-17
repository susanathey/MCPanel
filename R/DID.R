#' Difference-in-Difference Estimator when multiple units are missing.

#' @param M Matrix of observed entries. The input should be N (number of units) by T (number of time periods).
#' @param mask Binary mask with the same shape as M containing observed entries.
#' @return The matrix with all missing entries filled.
#' @examples
#' DID(matrix(c(1,2,3,4),2,2), matrix(c(1,1,1,0),2,2))
DID <- function( M, mask ){
  M <- M * mask
  treated_rows <- which(rowMeans(mask) < 1)
  control_rows <- setdiff(1:nrow(M), treated_rows)
  num_treated <- length(treated_rows)
  num_control <- length(control_rows)
  M_control_rows <- M[control_rows,, drop=FALSE]
  M_pred <- M
  for (l in 1:num_treated){
    tr_row_pred <- treated_rows[l]
    tr_row_miss_cols <- which(mask[treated_rows[l],]==0)
    control_cols <- setdiff(1:ncol(M),tr_row_miss_cols)
    W = (1/num_control) * matrix(1L, num_control)
    mu = mean(M[tr_row_pred, control_cols]) - mean(M[control_rows, control_cols])
    M_pred_this_row = t(M_control_rows) %*% W+mu;
    M_pred[treated_rows[l],]=M_pred_this_row
  }
  return(M_pred)
}

