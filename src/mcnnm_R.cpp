#include <iostream>
#include <cmath>
#include <Rcpp.h>
#include "Eigen/Dense"
#include "Eigen/SVD"
#include "Eigen/Core"
#include "Eigen/Sparse"
#include <RcppEigen.h>
#include <random>
#include <stdlib.h>

using namespace Eigen;
using namespace Rcpp;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////// Core Functions : All functions that have _H in the very end of their name,
////////                  consider the case where covariates exist.
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
List MySVD(NumericMatrix M){

  // This function computes the Singular Value Decomposition and it passes U,V,Sigma.
  // As SVD is one of the most time consuming part of our algorithm, this function is created to effectivelly
  // compute and compare different algorithms' speeds.

  using Eigen::Map;
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));

  JacobiSVD<MatrixXd> svd( M_.rows(), M_.cols(), ComputeThinV | ComputeThinU );
  svd.compute(M_);
  VectorXd sing = svd.singularValues();
  MatrixXd U = svd.matrixU();
  MatrixXd V = svd.matrixV();
  return List::create(Named("U") = U,
                      Named("V") = V,
                      Named("Sigma") = sing);

}

NumericVector logsp(double start_log, double end_log, int num_points){

  // This function creates logarithmically spaced numbers.

  NumericVector res(num_points);
  if(num_points == 1){
    res[0]=end_log;
    }
  else{
    double step_size = (end_log-start_log)/(num_points-1);
    for (int i = 0; i<num_points; i++){
      res[i]=pow(10.0,start_log+i*step_size);
     }
  }
  return res;
}

NumericMatrix ComputeMatrix(NumericMatrix L, NumericVector u, NumericVector v){

  // This function computes L + u1^T + 1v^T, which is our decomposition.

  using Eigen::Map;
  const Map<MatrixXd> L_(as<Map<MatrixXd> >(L));
  const Map<VectorXd> u_(as<Map<VectorXd> >(u));
  const Map<VectorXd> v_(as<Map<VectorXd> >(v));
  int num_rows = L_.rows();
  int num_cols = L_.cols();
  MatrixXd res_ = u_ * VectorXd::Constant(num_cols,1).transpose() + VectorXd::Constant(num_rows,1) * v_.transpose() + L_;
  return wrap(res_);
}

NumericMatrix ComputeMatrix_H(NumericMatrix L, NumericMatrix X, NumericMatrix Z, NumericMatrix H, NumericVector u, NumericVector v, bool to_add_ID){

  // This function computes L + X*H*Z^T + u1^T + 1v^T, which is our decomposition.

  using Eigen::Map;
  const Map<MatrixXd> L_(as<Map<MatrixXd> >(L));
  const Map<VectorXd> u_(as<Map<VectorXd> >(u));
  const Map<VectorXd> v_(as<Map<VectorXd> >(v));
  const Map<MatrixXd> X_(as<Map<MatrixXd> >(X));
  const Map<MatrixXd> Z_(as<Map<MatrixXd> >(Z));
  const Map<MatrixXd> H_(as<Map<MatrixXd> >(H));

  int H_rows_bef = X.cols();
  int H_cols_bef = Z.cols();
  int num_rows = L_.rows();
  int num_cols = L_.cols();
  if(to_add_ID == 1){
    H_rows_bef = X_.cols()-num_rows;
    H_cols_bef = Z_.cols()-num_cols;
  }

  MatrixXd res_;
  if(to_add_ID == 0){
    res_ = L_ + X_ * H_ * Z_.transpose() + u_ * VectorXd::Constant(num_cols,1).transpose() + VectorXd::Constant(num_rows,1) * v_.transpose();
  }
  else{
    res_ = L_ + u_ * VectorXd::Constant(num_cols,1).transpose() + VectorXd::Constant(num_rows,1) * v_.transpose();
    res_ += X_.topLeftCorner(num_rows, H_rows_bef) * H_.topLeftCorner(H_rows_bef, H_cols_bef) * Z_.topLeftCorner(num_cols,H_cols_bef).transpose();
    res_ += X_.topLeftCorner(num_rows, H_rows_bef) * H_.topRightCorner(H_rows_bef, num_cols);
    res_ += H_.bottomLeftCorner(num_rows, H_cols_bef) * Z_.topLeftCorner(num_cols, H_cols_bef).transpose();
  }
  return wrap(res_);
}

double Compute_objval(NumericMatrix M, NumericMatrix mask, NumericMatrix L, NumericVector u, NumericVector v, double sum_sing_vals, double lambda_L){

  // This function computes our objective value which is decomposed as the weighted combination of error plus nuclear norm.

  using Eigen::Map;
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  const Map<MatrixXd> mask_(as<Map<MatrixXd> >(mask));

  int train_size = mask_.sum();

  NumericMatrix est_mat = ComputeMatrix(L,u,v);
  const Map<MatrixXd> est_mat_(as<Map<MatrixXd> >(est_mat));

  MatrixXd err_mat_ = est_mat_ - M_;
  MatrixXd err_mask_ = (err_mat_.array()) * mask_.array();
  double obj_val = (double(1)/train_size) * (err_mask_.cwiseProduct(err_mask_)).sum() + lambda_L * sum_sing_vals;
  return obj_val;
}

double Compute_objval_H(NumericMatrix M, NumericMatrix X, NumericMatrix Z, NumericMatrix H, NumericMatrix mask, NumericMatrix L, NumericVector u, NumericVector v, double sum_sing_vals, double lambda_L, double lambda_H, bool to_add_ID){

  // This function computes our objective value which is decomposed as the weighted combination of error plus nuclear norm of L
  // and also element-wise norm 1 of H.

  using Eigen::Map;
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  const Map<MatrixXd> mask_(as<Map<MatrixXd> >(mask));
  const Map<MatrixXd> H_(as<Map<MatrixXd> >(H));

  int train_size = mask_.sum();

  double norm_H = H_.array().abs().sum();

  NumericMatrix est_mat = ComputeMatrix_H(L, X, Z, H, u, v, to_add_ID);
  const Map<MatrixXd> est_mat_(as<Map<MatrixXd> >(est_mat));

  MatrixXd err_mat_ = est_mat_ - M_;
  MatrixXd err_mask_ = (err_mat_.array()) * mask_.array();
  double obj_val = (double(1)/train_size) * (err_mask_.cwiseProduct(err_mask_)).sum() + lambda_L * sum_sing_vals + lambda_H*norm_H;
  return obj_val;
}

double Compute_RMSE(NumericMatrix M, NumericMatrix mask, NumericMatrix L, NumericVector u, NumericVector v){

  // This function computes Root Mean Squared Error of computed decomposition L,u,v.

  using Eigen::Map;
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  const Map<MatrixXd> mask_(as<Map<MatrixXd> >(mask));
  double res = 0;
  int valid_size = mask_.sum();
  NumericMatrix est_mat = ComputeMatrix(L,u,v);
  const Map<MatrixXd> est_mat_(as<Map<MatrixXd> >(est_mat));
  MatrixXd err_mat_ = est_mat_ - M_;
  MatrixXd err_mask_ = err_mat_.array() * mask_.array();
  res = std::sqrt((double(1.0)/valid_size) * (err_mask_.cwiseProduct(err_mask_)).sum());
  return res;
}

double Compute_RMSE_H(NumericMatrix M, NumericMatrix X, NumericMatrix Z, NumericMatrix H, bool to_add_ID, NumericMatrix mask, NumericMatrix L, NumericVector u, NumericVector v){

  // This function computes Root Mean Squared Error of computed decomposition L, H, u, v.

  using Eigen::Map;
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  const Map<MatrixXd> mask_(as<Map<MatrixXd> >(mask));
  const Map<MatrixXd> X_(as<Map<MatrixXd> >(X));
  const Map<MatrixXd> Z_(as<Map<MatrixXd> >(Z));

  int num_rows = M_.rows();
  int num_cols = M_.cols();
  MatrixXd Xfin = X_;
  MatrixXd Zfin = Z_;
  if(to_add_ID == 1){
    MatrixXd X_add = MatrixXd::Identity(num_rows, num_rows);
    MatrixXd Z_add = MatrixXd::Identity(num_cols, num_cols);
    MatrixXd X_conc(num_rows, X_.cols()+X_add.cols());
    X_conc << X_, X_add;
    Xfin = X_conc;
    MatrixXd Z_conc(num_cols, Z_.cols()+Z_add.cols());
    Z_conc << Z_, Z_add;
    Zfin = Z_conc;
  }
  NumericMatrix Xp = wrap(Xfin);
  NumericMatrix Zp = wrap(Zfin);

  double res = 0;
  int valid_size = mask_.sum();
  NumericMatrix est_mat = ComputeMatrix_H(L, Xp, Zp, H, u, v, to_add_ID);
  const Map<MatrixXd> est_mat_(as<Map<MatrixXd> >(est_mat));
  MatrixXd err_mat_ = est_mat_ - M_;
  MatrixXd err_mask_ = err_mat_.array() * mask_.array();
  res = std::sqrt((double(1.0)/valid_size) * (err_mask_.cwiseProduct(err_mask_)).sum());
  return res;
}

NumericMatrix SVT(NumericMatrix U, NumericMatrix V, NumericVector &sing_values, double sigma){

  // Given a singular value decomposition and a threshold sigma, this function computes Singular Value Thresholding operator.
  // Furthermore, it updates the singular values with the truncated version (new singular values of L) which would
  // then be used to compute objective value.

  using Eigen::Map;
  const Map<MatrixXd> U_(as<Map<MatrixXd> >(U));
  const Map<MatrixXd> V_(as<Map<MatrixXd> >(V));
  const Map<VectorXd> sing_values_(as<Map<VectorXd> >(sing_values));

  VectorXd trunc_sing = sing_values_ - VectorXd::Constant(sing_values_.size(),sigma);
  trunc_sing = trunc_sing.cwiseMax(0);
  MatrixXd Cp_ = U_ * trunc_sing.asDiagonal() * V_.transpose();
  sing_values = trunc_sing;
  return wrap(Cp_);
}

List update_L(NumericMatrix M, NumericMatrix mask, NumericMatrix L, NumericVector u, NumericVector v, double lambda_L){

  // This function updates L in coordinate descent algorithm. The core step of this part is
  // performing a SVT update. Furthermore, it saves the singular values (needed to compute objective value) later.
  // This would help us to only perform one SVD per iteration.

  using Eigen::Map;
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  const Map<MatrixXd> mask_(as<Map<MatrixXd> >(mask));
  const Map<MatrixXd> L_(as<Map<MatrixXd> >(L));

  int train_size = mask_.sum();
  NumericMatrix H = ComputeMatrix(L,u,v);
  const Map<MatrixXd> H_(as<Map<MatrixXd> >(H));
  MatrixXd P_omega_ = M_ - H_;
  MatrixXd masked_P_omega_ = P_omega_.cwiseProduct(mask_);
  MatrixXd proj_ = masked_P_omega_ + L_;
  NumericMatrix proj = wrap(proj_);
  List svd_dec = MySVD(proj);
  MatrixXd U_ = svd_dec["U"];
  MatrixXd V_ = svd_dec["V"];
  VectorXd sing_ = svd_dec["Sigma"];
  NumericMatrix U = wrap(U_);
  NumericMatrix V = wrap(V_);
  NumericVector sing = wrap(sing_);
  NumericMatrix L_upd = SVT(U, V, sing, lambda_L*train_size/2 );
  //L = SVT(U,V,sing,lambda_L/2);
  return List::create(Named("L") = L_upd,
                      Named("Sigma") = sing);
}

List update_L_H(NumericMatrix M, NumericMatrix X, NumericMatrix Z, NumericMatrix H, NumericMatrix mask, NumericMatrix L, NumericVector u, NumericVector v, double lambda_L, bool to_add_ID){

  // This function updates L in coordinate descent algorithm. The core step of this part is
  // performing a SVT update. Furthermore, it saves the singular values (needed to compute objective value) later.
  // This would help us to only perform one SVD per iteration. Note that this function includes covariates as well.

  using Eigen::Map;
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  const Map<MatrixXd> mask_(as<Map<MatrixXd> >(mask));
  const Map<MatrixXd> L_(as<Map<MatrixXd> >(L));

  int train_size = mask_.sum();
  NumericMatrix P = ComputeMatrix_H(L, X, Z, H, u, v, to_add_ID);
  const Map<MatrixXd> P_(as<Map<MatrixXd> >(P));
  MatrixXd P_omega_ = M_ - P_;
  MatrixXd masked_P_omega_ = P_omega_.cwiseProduct(mask_);
  MatrixXd proj_ = masked_P_omega_ + L_;
  NumericMatrix proj = wrap(proj_);
  List svd_dec = MySVD(proj);
  MatrixXd U_ = svd_dec["U"];
  MatrixXd V_ = svd_dec["V"];
  VectorXd sing_ = svd_dec["Sigma"];
  NumericMatrix U = wrap(U_);
  NumericMatrix V = wrap(V_);
  NumericVector sing = wrap(sing_);
  NumericMatrix L_upd = SVT(U, V, sing, lambda_L*train_size/2 );
  //L = SVT(U,V,sing,lambda_L/2);
  return List::create(Named("L") = L_upd,
                      Named("Sigma") = sing);
}

NumericVector update_u(NumericMatrix M, NumericMatrix mask, NumericMatrix L, NumericVector v){

  // This function updates u in coordinate descent algorithm.

  using Eigen::Map;
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  const Map<MatrixXd> mask_(as<Map<MatrixXd> >(mask));
  const Map<MatrixXd> L_(as<Map<MatrixXd> >(L));
  const Map<VectorXd> v_(as<Map<VectorXd> >(v));
  VectorXd res(M_.rows(),1);
  for (int i = 0; i<M_.rows(); i++){
    VectorXd b_ = L_.row(i)+v_.transpose()-M_.row(i);
    VectorXd h_ = mask_.row(i);
    VectorXd b_mask_ = b_.cwiseProduct(h_);
    int l = (h_.array() > 0).count();
    if (l>0){
      res(i)=-b_mask_.sum()/l;
    }
    else{
      res(i) = 0;
    }
  }
  return wrap(res);
}

NumericVector update_u_H(NumericMatrix M, NumericMatrix X, NumericMatrix Z, NumericMatrix H, NumericMatrix mask, NumericMatrix L, NumericVector v){

  // This function updates u in coordinate descent algorithm, when covariates are available.

  using Eigen::Map;
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  const Map<MatrixXd> mask_(as<Map<MatrixXd> >(mask));
  const Map<MatrixXd> L_(as<Map<MatrixXd> >(L));
  const Map<MatrixXd> X_(as<Map<MatrixXd> >(X));
  const Map<MatrixXd> Z_(as<Map<MatrixXd> >(Z));
  const Map<MatrixXd> H_(as<Map<MatrixXd> >(H));
  const Map<VectorXd> v_(as<Map<VectorXd> >(v));
  MatrixXd T_ = X_ * H_ * Z_.transpose();

  VectorXd res(M_.rows(),1);
  for (int i = 0; i<M_.rows(); i++){
    VectorXd b_ = T_.row(i)+L_.row(i)+v_.transpose()-M_.row(i);
    VectorXd h_ = mask_.row(i);
    VectorXd b_mask_ = b_.cwiseProduct(h_);
    int l = (h_.array() > 0).count();
    if (l>0){
      res(i)=-b_mask_.sum()/l;
    }
    else{
      res(i) = 0;
    }
  }
  return wrap(res);
}

NumericVector update_v(NumericMatrix M, NumericMatrix mask, NumericMatrix L, NumericVector u){

  // This function updates the matrix v in the coordinate descent algorithm.

  using Eigen::Map;
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  const Map<MatrixXd> mask_(as<Map<MatrixXd> >(mask));
  const Map<MatrixXd> L_(as<Map<MatrixXd> >(L));
  const Map<VectorXd> u_(as<Map<VectorXd> >(u));
  VectorXd res(M_.cols(),1);
  for (int i = 0; i<M_.cols(); i++)
  {
    VectorXd b_ = L_.col(i)+u_-M_.col(i);
    VectorXd h_ = mask_.col(i);
    VectorXd b_mask_ = b_.cwiseProduct(h_);
    int l = (h_.array() > 0).count();
    if (l>0){
      res(i)=-b_mask_.sum()/l;
    }
    else{
      res(i) = 0;
    }
  }
  return wrap(res);
}

NumericVector update_v_H(NumericMatrix M, NumericMatrix X, NumericMatrix Z, NumericMatrix H, NumericMatrix mask, NumericMatrix L, NumericVector u){

  // This function updates the matrix v in the coordinate descent algorithm, when covariates exist.

  using Eigen::Map;
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  const Map<MatrixXd> mask_(as<Map<MatrixXd> >(mask));
  const Map<MatrixXd> L_(as<Map<MatrixXd> >(L));
  const Map<MatrixXd> X_(as<Map<MatrixXd> >(X));
  const Map<MatrixXd> Z_(as<Map<MatrixXd> >(Z));
  const Map<MatrixXd> H_(as<Map<MatrixXd> >(H));
  const Map<VectorXd> u_(as<Map<VectorXd> >(u));
  MatrixXd T_ = X_ * H_ * Z_.transpose();
  VectorXd res(M_.cols(),1);
  for (int i = 0; i<M_.cols(); i++)
  {
    VectorXd b_ = T_.col(i)+L_.col(i)+u_-M_.col(i);
    VectorXd h_ = mask_.col(i);
    VectorXd b_mask_ = b_.cwiseProduct(h_);
    int l = (h_.array() > 0).count();
    if (l>0){
      res(i)=-b_mask_.sum()/l;
    }
    else{
      res(i) = 0;
    }
  }
  return wrap(res);
}

NumericVector Reshape_Mat(NumericMatrix M){
  // This function reshapes a matrix into a vector.
  NumericVector res(M.rows()*M.cols());
  for(int j=0; j<M.cols(); j++){
    for(int i=0; i<M.rows(); i++){
      res(j*M.rows()+i) = M(i,j);
    }
  }
  return wrap(res);
}

NumericMatrix Reshape(NumericVector M, int row, int col){
  // This function reshapes a given vector, into a matrix with given rows and columns.
  NumericMatrix res(row,col);
  for(int j=0; j<col; j++){
    for(int i=0; i<row; i++){
      res(i,j)=M(j*row+i);
    }
  }
  return wrap(res);
}

List update_H_H(NumericMatrix M, NumericMatrix X, NumericMatrix Z, NumericMatrix H, NumericMatrix T, NumericVector in_prod, NumericVector in_prod_T, bool to_add_ID, NumericMatrix mask, NumericMatrix L, NumericVector u, NumericVector v, double lambda_H){

  // This function updates the matrix H in the coordinate descent algorithm. The key idea is to use the soft-thresholding opreator.

  using Eigen::Map;
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  const Map<MatrixXd> mask_(as<Map<MatrixXd> >(mask));
  const Map<MatrixXd> X_(as<Map<MatrixXd> >(X));
  const Map<MatrixXd> Z_(as<Map<MatrixXd> >(Z));
  const Map<MatrixXd> H_(as<Map<MatrixXd> >(H));
  const Map<MatrixXd> T_(as<Map<MatrixXd> >(T));
  const Map<VectorXd> in_prod_(as<Map<VectorXd> >(in_prod));
  const Map<VectorXd> in_prod_T_(as<Map<VectorXd> >(in_prod_T));
  VectorXd inn_prod = in_prod_;
  int H_rows = X_.cols();
  int H_cols = Z_.cols();
  int num_rows = M_.rows();
  int num_cols = M_.cols();
  int num_train = mask_.sum();
  int H_cols_bef = H_rows;
  int H_rows_bef = H_cols;
  if(to_add_ID == 1){
    H_rows_bef = X_.cols()-num_rows;
    H_cols_bef = Z_.cols()-num_cols;
  }
  NumericMatrix M_hat = ComputeMatrix(L,u,v);
  const Map<MatrixXd> M_hat_(as<Map<MatrixXd> >(M_hat));
  MatrixXd b = (M_-M_hat_).cwiseProduct(mask_)/(std::sqrt(num_train));
  NumericVector b_resh = Reshape_Mat(wrap(b));
  const Map<VectorXd> b_resh_(as<Map<VectorXd> >(b_resh));

  NumericVector H_resh = Reshape_Mat(wrap(H_));
  const Map<VectorXd> H_resh_(as<Map<VectorXd> >(H_resh));
  VectorXd H__ = H_resh_;
  if(to_add_ID == 1){
    if(H_cols_bef > 0){
      for(int j = 0; j< H_cols_bef; j++){
        for (int i = 0; i<H_rows; i++){
          int cur_elem = j*H_rows + i;
          double U = in_prod_T_(cur_elem);
          if (U == 0){
            H__(cur_elem) = 0;
          }
          else{
            VectorXd b_tilde = b_resh_ - inn_prod + T_.col(cur_elem) * H__(cur_elem);
            double V = b_tilde.transpose() * T_.col(cur_elem);
            double H_new = (double(1)/2) * ( std::max( (2*V-lambda_H)/U, double(0)) - std::max( (-2*V-lambda_H)/U ,double(0)) );
            inn_prod += (H_new - H__(cur_elem))*T_.col(cur_elem);
            H__(cur_elem) = H_new;
          }
        }
      }
    }
    if(H_rows_bef>0){
      for(int j = H_cols_bef; j<H_cols; j++){
        for (int i = 0; i<H_rows_bef; i++){
          int cur_elem = j*H_rows + i;
          double U = in_prod_T_(cur_elem);
          if (U == 0){
            H__(cur_elem) = 0;
          }
          else{
            VectorXd b_tilde = b_resh_ - inn_prod + T_.col(cur_elem) * H__(cur_elem);
            double V = b_tilde.transpose() * T_.col(cur_elem);
            double H_new = (double(1)/2) * ( std::max( (2*V-lambda_H)/U, double(0)) - std::max( (-2*V-lambda_H)/U ,double(0)) );
            inn_prod += (H_new - H__(cur_elem))*T_.col(cur_elem);
            H__(cur_elem) = H_new;
          }
        }
      }
    }
  } else{
    for(int j = 0; j<H_cols; j++){
      for (int i = 0; i<H_rows; i++){
        int cur_elem = j*H_rows + i;
        double U = in_prod_T_(cur_elem);
        if (U == 0){
          H__(cur_elem) = 0;
        }
        else{
          VectorXd b_tilde = b_resh_ - inn_prod + T_.col(cur_elem) * H__(cur_elem);
          double V = b_tilde.transpose() * T_.col(cur_elem);
          double H_new = (double(1)/2) * ( std::max( (2*V-lambda_H)/U, double(0)) - std::max( (-2*V-lambda_H)/U ,double(0)) );
          inn_prod += (H_new - H__(cur_elem))*T_.col(cur_elem);
          H__(cur_elem) = H_new;
        }
      }
    }
  }
  in_prod = wrap(inn_prod);
  NumericMatrix res = Reshape(wrap(H__), H_rows, H_cols);
  return List::create(Named("H") = res,
                      Named("in_prod") = in_prod);
}

List initialize_uv(NumericMatrix M, NumericMatrix mask, bool to_estimate_u, bool to_estimate_v, int niter = 1000, double rel_tol = 1e-5){

  // This function solves finds the optimal u and v assuming that L is zero. This would be later
  // helpful when we want to perform warm start on values of lambda_L. This function also outputs
  // the smallest value of lambda_L which causes L to be zero (all singular values vanish after a SVT update)

  using Eigen::Map;
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  const Map<MatrixXd> mask_(as<Map<MatrixXd> >(mask));

  JacobiSVD<MatrixXd> svd(M_.rows(), M_.cols());
  double obj_val=0;
  double new_obj_val=0;
  int num_rows = M_.rows();
  int num_cols = M_.cols();
  VectorXd u_ = VectorXd::Zero(num_rows);
  VectorXd v_ = VectorXd::Zero(num_cols);
  MatrixXd L_ = MatrixXd::Zero(num_rows,num_cols);
  NumericVector u = wrap(u_);
  NumericVector v = wrap(v_);
  NumericMatrix L = wrap(L_);
  obj_val = Compute_objval(M, mask, L, u , v, 0, 0);
  for(int iter = 0; iter < niter; iter++){
    if(to_estimate_u == 1){
      u = update_u(M, mask, L, v);
    }
    else{
      u = wrap(VectorXd::Zero(num_rows));
    }
    if(to_estimate_v == 1){
      v = update_v(M, mask, L, u);
    }
    else{
      v = wrap(VectorXd::Zero(num_cols));
    }
    new_obj_val = Compute_objval(M, mask, L, u, v, 0, 0);
    double rel_error = (new_obj_val-obj_val)/obj_val;
    if(rel_error < rel_tol && rel_error >= 0){
      break;
    }
    obj_val = new_obj_val;
  }
  NumericMatrix E = ComputeMatrix(L, u, v);
  const Map<MatrixXd> E_(as<Map<MatrixXd> >(E));
  MatrixXd P_omega_ = (M_ - E_).array()*mask_.array();

  svd.compute(P_omega_);
  double lambda_L_max = 2.0 * svd.singularValues().maxCoeff()/mask_.sum();

  return List::create(Named("u") = u,
                      Named("v") = v,
                      Named("lambda_L_max") = lambda_L_max);
  }

List initialize_uv_H(NumericMatrix M, NumericMatrix X, NumericMatrix Z, NumericMatrix mask, bool to_estimate_u, bool to_estimate_v, bool to_add_ID, int niter = 1000, double rel_tol = 1e-5){

    // This function solves finds the optimal u and v assuming that L and H are zero. This would be later
    // helpful when we want to perform warm start on values of lambda_L and lambda_H. This function also outputs
    // the smallest value of lambda_L and lambda_H which causes L and H to be zero.

    using Eigen::Map;
    const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
    const Map<MatrixXd> mask_(as<Map<MatrixXd> >(mask));
    const Map<MatrixXd> X_(as<Map<MatrixXd> >(X));
    const Map<MatrixXd> Z_(as<Map<MatrixXd> >(Z));
    MatrixXd Xfin = X_;
    MatrixXd Zfin = Z_;
    int num_rows = M_.rows();
    int num_cols = M_.cols();
    if(to_add_ID == 1){
      MatrixXd X_add = MatrixXd::Identity(num_rows, num_rows);
      MatrixXd Z_add = MatrixXd::Identity(num_cols, num_cols);
      MatrixXd X_conc(num_rows, X_.cols()+X_add.cols());
      X_conc << X_, X_add;
      Xfin = X_conc;
      MatrixXd Z_conc(num_cols, Z_.cols()+Z_add.cols());
      Z_conc << Z_, Z_add;
      Zfin = Z_conc;
    }
    int H_rows = Xfin.cols();
    int H_cols = Zfin.cols();
    NumericMatrix Xp = wrap(Xfin);
    NumericMatrix Zp = wrap(Zfin);

    JacobiSVD<MatrixXd> svd(M_.rows(), M_.cols());
    double obj_val=0;
    double new_obj_val=0;
    VectorXd u_ = VectorXd::Zero(num_rows);
    VectorXd v_ = VectorXd::Zero(num_cols);
    MatrixXd L_ = MatrixXd::Zero(num_rows,num_cols);
    MatrixXd H_ = MatrixXd::Zero(H_rows,H_cols);
    NumericVector u = wrap(u_);
    NumericVector v = wrap(v_);
    NumericMatrix L = wrap(L_);
    NumericMatrix H = wrap(H_);
    obj_val = Compute_objval_H(M, Xp, Zp, H, mask, L, u , v, 0, 0, 0, to_add_ID);
    for(int iter = 0; iter < niter; iter++){
      if(to_estimate_u == 1){
        u = update_u_H(M, Xp, Zp, H, mask, L, v);
      }
      else{
        u = wrap(VectorXd::Zero(num_rows));
      }
      if(to_estimate_v == 1){
        v = update_v_H(M, Xp, Zp, H, mask, L, u);
      }
      else{
        v = wrap(VectorXd::Zero(num_rows));
      }
      new_obj_val = Compute_objval_H(M, Xp, Zp, H, mask, L, u, v, 0, 0, 0, to_add_ID);
      double rel_error = (new_obj_val-obj_val)/obj_val;
      if(rel_error < rel_tol && rel_error >= 0){
        break;
      }
      obj_val = new_obj_val;
    }
    NumericMatrix E = ComputeMatrix_H(L, Xp, Zp, H, u, v, to_add_ID);
    const Map<MatrixXd> E_(as<Map<MatrixXd> >(E));
    MatrixXd P_omega_ = (M_ - E_).array()*mask_.array();
    svd.compute(P_omega_);
    double lambda_L_max = 2.0 * svd.singularValues().maxCoeff()/mask_.sum();

    MatrixXd T_ = MatrixXd::Zero(num_rows*num_cols, H_rows*H_cols);
    VectorXd in_prod_T = VectorXd::Zero(H_rows*H_cols);
    for (int j = 0; j<H_cols; j++){
      for (int i = 0; i<H_rows; i++){
        MatrixXd out_prod = (Xfin.col(i) * Zfin.col(j).transpose()).cwiseProduct(mask_);
        Map<VectorXd>(out_prod.data(), out_prod.size());
        int index = j*H_rows+i;
        T_.col(index) = out_prod;
        in_prod_T(index) = T_.col(index).transpose()*T_.col(index);
      }
    }
    int num_train = mask_.sum();
    T_ = T_/std::sqrt(num_train);
    in_prod_T = in_prod_T/num_train;
    NumericVector P_omega_resh = Reshape_Mat(wrap(P_omega_));
    const Map<VectorXd> P_omega_resh_(as<Map<VectorXd> >(P_omega_resh));
    VectorXd all_Vs = (T_.transpose()*P_omega_resh_)/std::sqrt(num_train);
    double lambda_H_max = 2*(all_Vs.array()).abs().maxCoeff();

    return List::create(Named("u") = u,
                        Named("v") = v,
                        Named("lambda_L_max") = lambda_L_max,
                        Named("lambda_H_max") = lambda_H_max,
                        Named("T_mat") = T_,
                        Named("in_prod_T") = in_prod_T);
}

List create_folds(NumericMatrix M, NumericMatrix mask, bool to_estimate_u, bool to_estimate_v, int niter = 1000, double rel_tol = 1e-5, double cv_ratio = 0.8, int num_folds=5){

  // This function creates folds for cross-validation. Each fold contains a training and validation sets.
  // For each of these folds the initial solutions for fixed effects are then computed, as for large lambda_L
  // L would be equal to zero. This initialization is very helpful as it will be used later for the warm start.

  using Eigen::Map;
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  const Map<MatrixXd> mask_(as<Map<MatrixXd> >(mask));
  int num_rows = M_.rows();
  int num_cols = M_.cols();
  List out(num_folds);
  std::default_random_engine generator;
  std::bernoulli_distribution distribution(cv_ratio);
  for (int k = 0; k<num_folds; k++){
    MatrixXd ma_new(num_rows,num_cols);
    for (int i = 0; i < num_rows; i++){
      for (int j = 0; j < num_cols; j++){
        ma_new(i,j)=distribution(generator);
      }
    }
    MatrixXd fold_mask_ = mask_.array() * ma_new.array();
    MatrixXd M_tr_ = M_.array() * fold_mask_.array();
    NumericMatrix fold_mask = wrap(fold_mask_);
    NumericMatrix M_tr = wrap(M_tr_);
    List tmp_uv = initialize_uv(M_tr, fold_mask, to_estimate_u, to_estimate_v, niter = 1000, rel_tol = 1e-5);
    List fold_k = List::create(Named("u") = tmp_uv["u"],
                               Named("v") = tmp_uv["v"],
                               Named("lambda_L_max") = tmp_uv["lambda_L_max"],
                               Named("fold_mask") = fold_mask);
    out[k] = fold_k;
  }
  return out;
}

List create_folds_H(NumericMatrix M, NumericMatrix X, NumericMatrix Z, bool to_estimate_u, bool to_estimate_v, bool to_add_ID, NumericMatrix mask, int niter = 1000, double rel_tol = 1e-5, double cv_ratio = 0.8, int num_folds=5){

  // This function creates folds for cross-validation. Each fold contains a training and validation sets.
  // For each of these folds the initial solutions for fixed effects are then computed, as for large lambda_L and lambda_H,
  // L and H would be equal to zero. This initialization is very helpful as it will be used later for the warm start.

  using Eigen::Map;
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  const Map<MatrixXd> mask_(as<Map<MatrixXd> >(mask));
  int num_rows = M_.rows();
  int num_cols = M_.cols();
  List out(num_folds);
  std::default_random_engine generator;
  std::bernoulli_distribution distribution(cv_ratio);
  for (int k = 0; k<num_folds; k++){
    MatrixXd ma_new(num_rows,num_cols);
    for (int i = 0; i < num_rows; i++){
      for (int j = 0; j < num_cols; j++){
        ma_new(i,j)=distribution(generator);
      }
    }
    MatrixXd fold_mask_ = mask_.array() * ma_new.array();
    MatrixXd M_tr_ = M_.array() * fold_mask_.array();
    NumericMatrix fold_mask = wrap(fold_mask_);
    NumericMatrix M_tr = wrap(M_tr_);
    List tmp_uv = initialize_uv_H(M_tr, X, Z, fold_mask, to_estimate_u, to_estimate_v, to_add_ID, niter = 1000, rel_tol = 1e-5);
    List fold_k = List::create(Named("u") = tmp_uv["u"],
                               Named("v") = tmp_uv["v"],
                               Named("lambda_L_max") = tmp_uv["lambda_L_max"],
                               Named("lambda_H_max") = tmp_uv["lambda_H_max"],
                               Named("fold_mask") = fold_mask,
                               Named("T_mat") = tmp_uv["T_mat"],
                               Named("in_prod_T") = tmp_uv["in_prod_T"]);
    out[k] = fold_k;
  }
  return out;
}

List NNM_fit(NumericMatrix M, NumericMatrix mask, NumericMatrix L_init, NumericVector u_init, NumericVector v_init, bool to_estimate_u, bool to_estimate_v, double lambda_L, int niter = 1000, double rel_tol = 1e-5, bool is_quiet = 1){

  // This function performs coordinate descent updates.
  // For given matrices M, mask, and initial starting decomposition given by L_init, u_init, and v_init,
  // matrices L, u, and v are updated till convergence via coordinate descent.

  double obj_val;
  double new_obj_val=0;
  List svd_dec;
  svd_dec = MySVD(L_init);
  VectorXd sing = svd_dec["Sigma"];
  double sum_sigma = sing.sum();
  obj_val = Compute_objval(M, mask, L_init, u_init, v_init, sum_sigma, lambda_L);
  NumericMatrix L = L_init;
  NumericVector u = u_init;
  NumericVector v = v_init;
  int term_iter = 0;
  for(int iter = 0; iter < niter; iter++){
    // Update u
    if(to_estimate_u == 1){
      u = update_u(M, mask, L, v);
    }
    else{
      u = wrap(VectorXd::Zero(M.rows()));
    }
    // Update v
    if(to_estimate_v == 1){
      v = update_v(M, mask, L, u);
    }
    else{
      v = wrap(VectorXd::Zero(M.cols()));
    }
    // Update L
    List upd_L = update_L(M, mask, L, u, v, lambda_L);
    NumericMatrix L_upd = upd_L["L"];
    L = L_upd;
    sing = upd_L["Sigma"];
    double sum_sigma = sing.sum();
    // Check if accuracy is achieved
    new_obj_val = Compute_objval(M, mask,  L, u, v, sum_sigma, lambda_L);
    double rel_error = (obj_val-new_obj_val)/obj_val;
    if(new_obj_val < 1e-8){
      break;
    }
    if(rel_error < rel_tol && rel_error >= 0){
      break;
    }
    term_iter = iter;
    obj_val = new_obj_val;
  }
  if(is_quiet == 0){
    std::cout << "Terminated at iteration : " << term_iter << ", for lambda_L :" << lambda_L << ", with obj_val :" << new_obj_val << std::endl;
  }
  return List::create(Named("L") = L,
                      Named("u") = u,
                      Named("v") = v);
}

List NNM_fit_H(NumericMatrix M, NumericMatrix X, NumericMatrix Z, NumericMatrix H_init, NumericMatrix T, NumericVector in_prod, NumericVector in_prod_T, NumericMatrix mask, NumericMatrix L_init, NumericVector u_init, NumericVector v_init, bool to_estimate_u, bool to_estimate_v, bool to_add_ID ,double lambda_L, double lambda_H, int niter = 1000, double rel_tol = 1e-5, bool is_quiet = 1){

  // This function performs cyclic coordinate descent updates.
  // For given matrices M, mask, and initial starting decomposition given by L_init, u_init, and v_init,
  // matrices L, u, and v are updated till convergence via coordinate descent.
  using Eigen::Map;
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  const Map<MatrixXd> X_(as<Map<MatrixXd> >(X));
  const Map<MatrixXd> Z_(as<Map<MatrixXd> >(Z));
  int num_rows = M_.rows();
  int num_cols = M_.cols();
  MatrixXd Xfin = X_;
  MatrixXd Zfin = Z_;
  if(to_add_ID == 1){
    MatrixXd X_add = MatrixXd::Identity(num_rows, num_rows);
    MatrixXd Z_add = MatrixXd::Identity(num_cols, num_cols);
    MatrixXd X_conc(num_rows, X_.cols()+X_add.cols());
    X_conc << X_, X_add;
    Xfin = X_conc;
    MatrixXd Z_conc(num_cols, Z_.cols()+Z_add.cols());
    Z_conc << Z_, Z_add;
    Zfin = Z_conc;
  }
  NumericMatrix Xp = wrap(Xfin);
  NumericMatrix Zp = wrap(Zfin);
  double obj_val;
  double new_obj_val=0;
  List svd_dec;
  svd_dec = MySVD(L_init);
  VectorXd sing = svd_dec["Sigma"];
  double sum_sigma = sing.sum();
  obj_val = Compute_objval_H(M, Xp, Zp, H_init, mask, L_init, u_init, v_init, sum_sigma, lambda_L, lambda_H, to_add_ID);
  NumericMatrix H = H_init;
  NumericMatrix L = L_init;
  NumericVector u = u_init;
  NumericVector v = v_init;
  int term_iter = 0;
  for(int iter = 0; iter < niter; iter++){
    // Update u
    if(to_estimate_u == 1){
      u = update_u_H(M, Xp, Zp, H, mask, L, v);
    }
    else{
      u = wrap(VectorXd::Zero(M.rows()));
    }
    // Update v
    if(to_estimate_v == 1){
      v = update_v_H(M, Xp, Zp, H, mask, L, u);
    }
    else{
      v = wrap(VectorXd::Zero(M.cols()));
    }
    // Update H
    List upd_H = update_H_H(M, Xp, Zp, H, T, in_prod, in_prod_T, to_add_ID, mask, L, u, v, lambda_H);
    NumericMatrix H_upd = upd_H["H"];
    H = H_upd;
    NumericVector prod_in = upd_H["in_prod"];
    in_prod = prod_in;
    // Update L
    List upd_L = update_L_H(M, Xp, Zp, H, mask, L, u, v, lambda_L, to_add_ID);
    NumericMatrix L_upd = upd_L["L"];
    L = L_upd;
    sing = upd_L["Sigma"];
    double sum_sigma = sing.sum();
    // Check if accuracy is achieved
    new_obj_val = Compute_objval_H(M, Xp, Zp, H, mask,  L, u, v, sum_sigma, lambda_L, lambda_H, to_add_ID);
    double rel_error = (obj_val-new_obj_val)/obj_val;
    if(new_obj_val < 1e-8){
      break;
    }
    if(rel_error < rel_tol && rel_error >= 0){
      break;
    }
    term_iter = iter;
    obj_val = new_obj_val;
  }
  if(is_quiet == 0){
    std::cout << "Terminated at iteration : " << term_iter << ", for lambda_L :" << lambda_L << ", lambda_H : " << lambda_H << ", with obj_val :" << new_obj_val << std::endl;
  }
  return List::create(Named("H") = H,
                      Named("L") = L,
                      Named("u") = u,
                      Named("v") = v,
                      Named("in_prod") = in_prod);
}

List NNM_with_uv_init(NumericMatrix M, NumericMatrix mask, NumericVector u_init, NumericVector v_init, bool to_estimate_u, bool to_estimate_v, NumericVector lambda_L, int niter = 1000, double rel_tol = 1e-5, bool is_quiet = 1){

  // This function actually does the warm start.
  // Idea here is that we start from L_init=0, and converged u_init and v_init and then find the
  // fitted model, i.e, new L,u, and v. Then, we pass these parameters for the next value of lambda_L.
  // It is worth noting that lambda_L's are sorted in decreasing order.

  int num_lam = lambda_L.size();
  int num_rows = M.rows();
  int num_cols = M.cols();
  List res(num_lam);
  NumericMatrix L_init = wrap(MatrixXd::Zero(num_rows,num_cols));
  for (int i = 0; i<num_lam; i++){
    List fact = NNM_fit(M, mask, L_init, u_init, v_init, to_estimate_u, to_estimate_v, lambda_L[i], niter, rel_tol, is_quiet);
    res[i] = fact;
    NumericMatrix L_upd = fact["L"];
    L_init = L_upd;
    u_init = fact["u"];
    v_init = fact["v"];
  }
  return res;
}

List NNM_with_uv_init_H(NumericMatrix M, NumericMatrix X, NumericMatrix Z, NumericMatrix T, NumericVector in_prod_T, NumericMatrix mask, NumericVector u_init, NumericVector v_init, bool to_estimate_u, bool to_estimate_v, bool to_add_ID, NumericVector lambda_L, NumericVector lambda_H, int niter = 1000, double rel_tol = 1e-5, bool is_quiet = 1){

  // This function actually does the warm start.
  // Idea here is that we start from L_init=0 and H_init, and converged u_init and v_init and then find the
  // fitted model, i.e, new L, H, u, and v. Then, we pass these parameters for the next value of lambda_L and lambda_H.
  // Both lambda_L and lambda_H vectors are sorted in decreasing order. As we are cross-validating over a grid, there
  // are two options for taking as the previous model (initialization of new point on grid). Here we always use the model
  // with lambda_L just before this model, keeping lambda_H fixed. The only exception is when we are at the largest lambda_L,
  // for which we take previous lambda_H, while keep lambda_L fixed.

  int num_lam_L = lambda_L.size();
  int num_lam_H = lambda_H.size();
  int num_rows = M.rows();
  int num_cols = M.cols();
  int H_rows;
  int H_cols;
  if(to_add_ID == 1){
    H_rows = X.cols()+num_rows;
    H_cols = Z.cols()+num_cols;
  }
  else{
    H_rows = X.cols();
    H_cols = Z.cols();
  }
  List res(num_lam_L*num_lam_H);
  NumericMatrix L_init = wrap(MatrixXd::Zero(num_rows,num_cols));
  NumericMatrix H_init = wrap(MatrixXd::Zero(H_rows,H_cols));
  NumericVector in_prod = wrap(VectorXd::Zero(num_rows*num_cols));
  for (int j = 0; j<num_lam_H; j++){
    if(j > 0){
      List previous_H = res[(j-1)*num_lam_L];
      NumericMatrix L_upd = previous_H["L"];
      NumericMatrix H_upd = previous_H["H"];
      L_init = L_upd;
      NumericVector in_prod_upd = previous_H["in_prod"];
      in_prod = in_prod_upd;
      u_init = previous_H["u"];
      v_init = previous_H["v"];
      H_init = H_upd;
    }
    for (int i = 0; i<num_lam_L; i++){
      List previous_L = NNM_fit_H(M, X, Z, H_init, T, in_prod, in_prod_T, mask, L_init, u_init, v_init, to_estimate_u, to_estimate_v, to_add_ID, lambda_L[i], lambda_H[j], niter, rel_tol, is_quiet);
      res[j*num_lam_L+i] = previous_L;
      NumericMatrix L_upd = previous_L["L"];
      NumericMatrix H_upd = previous_L["H"];
      NumericVector in_prod_upd = previous_L["in_prod"];
      in_prod = in_prod_upd;
      L_init = L_upd;
      u_init = previous_L["u"];
      v_init = previous_L["v"];
      H_init = H_upd;
    }
  }
  return res;
}

List NNM_H(NumericMatrix M, NumericMatrix X, NumericMatrix Z, NumericMatrix mask, int num_lam_L = 30, int num_lam_H = 30, NumericVector lambda_L = NumericVector::create(), NumericVector lambda_H = NumericVector::create(), bool to_estimate_u = 1, bool to_estimate_v = 1, bool to_add_ID = 1, int niter = 100, double rel_tol = 1e-5, bool is_quiet = 1){

  // This function in its default format is just a wraper, which only passes vectors of all zero for u_init and v_init to NNM_with_uv_init_H.
  // The function computes the good range for lambda_L and lambda_H and fits using warm-start described in NNM_with_uv_init_H to all those values.
  // The user has the ability to set vectors of lambda_L and lambda_H manually, although it is not advisable.

  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  List tmp_uv = initialize_uv_H(M, X, Z, mask, to_estimate_u, to_estimate_v, to_add_ID, niter, rel_tol);
  if(lambda_L.size() == 0){
    NumericVector lambda_Ls(num_lam_L);
    double max_lam_L=tmp_uv["lambda_L_max"];
    NumericVector lambda_Ls_without_zero = logsp(log10(max_lam_L), log10(max_lam_L)-3, num_lam_L-1);
    for(int i=0; i<num_lam_L-1; i++){
      lambda_Ls(i)= lambda_Ls_without_zero(i);
    }
    lambda_Ls(num_lam_L-1) = 0;
    lambda_L = lambda_Ls;
  }
  else{
    num_lam_L = lambda_L.size();
  }
  if(lambda_H.size() == 0){
    NumericVector lambda_Hs(num_lam_H);
    double max_lam_H=tmp_uv["lambda_H_max"];
    NumericVector lambda_Hs_without_zero = logsp(log10(max_lam_H), log10(max_lam_H)-3, num_lam_H-1);
    for(int i=0; i<num_lam_H-1; i++){
      lambda_Hs(i)= lambda_Hs_without_zero(i);
    }
    lambda_Hs(num_lam_H-1) = 0;
    lambda_H = lambda_Hs;
  }
  else{
    num_lam_H = lambda_H.size();
  }
  List tmp_res;
  if(to_estimate_u == 1 || to_estimate_v ==1){
    tmp_res =  NNM_with_uv_init_H(M, X, Z, tmp_uv["T_mat"], tmp_uv["in_prod_T"], mask, tmp_uv["u"], tmp_uv["v"], to_estimate_u, to_estimate_v, to_add_ID, lambda_L, lambda_H, niter, rel_tol, is_quiet);
  }
  else{
    tmp_res = NNM_with_uv_init_H(M, X, Z, tmp_uv["T_mat"], tmp_uv["in_prod_T"], mask, wrap(VectorXd::Zero(M_.rows())), wrap(VectorXd::Zero(M_.cols())), to_estimate_u, to_estimate_v, to_add_ID, lambda_L, lambda_H, niter, rel_tol, is_quiet);
  }

  List out(num_lam_L*num_lam_H);
  for (int j = 0; j< num_lam_H; j++){
    for (int i = 0; i < num_lam_L; i++){
      int current_ind = j*num_lam_L+i;
      List current_config = tmp_res(j*num_lam_L+i);
      List this_config = List::create(Named("H") = current_config["H"],
                                      Named("L") = current_config["L"],
                                      Named("u") = current_config["u"],
                                      Named("v") = current_config["v"],
                                      Named("lambda_L") = lambda_L[i],
                                      Named("lambda_H") = lambda_H[j]);
      out[current_ind] = this_config;
    }
  }
  return out;
}

List NNM_with_uv_init_H_opt_path(NumericMatrix M, NumericMatrix X, NumericMatrix Z, NumericMatrix T, NumericVector in_prod_T, NumericMatrix mask, NumericVector u_init, NumericVector v_init, bool to_estimate_u, bool to_estimate_v, bool to_add_ID, NumericVector lambda_L, NumericVector lambda_H, int niter = 1000, double rel_tol = 1e-5, bool is_quiet = 1){
  // This function is similar to NNM_H, with one key difference. This function instead of fitting to all models on the grid described by lambda_Ls and lambda_Hs
  // only considers the shortest path from the point on the grid with highest lambda_L and lambda_H to the point on the grid with smallest values of lambda_L
  // and lambda_H. The key benefit of using this function is that, for chosen values of lambda_L and lambda_H, training can be much faster as the number of
  // trained models is M+N-1 compared to M*N, where M is the length of lambda_L and N is the length of lambda_H.

  int num_lam_L = lambda_L.size();
  int num_lam_H = lambda_H.size();
  int num_rows = M.rows();
  int num_cols = M.cols();
  int H_rows;
  int H_cols;
  if(to_add_ID == 1){
    H_rows = X.cols()+num_rows;
    H_cols = Z.cols()+num_cols;
  }
  else{
    H_rows = X.cols();
    H_cols = Z.cols();
  }
  List res(num_lam_L+num_lam_H-1);
  NumericMatrix L_init = wrap(MatrixXd::Zero(num_rows,num_cols));
  NumericMatrix H_init = wrap(MatrixXd::Zero(H_rows,H_cols));
  NumericVector in_prod = wrap(VectorXd::Zero(num_rows*num_cols));
  for (int j = 0; j<num_lam_H; j++){
    List previous_pt = NNM_fit_H(M, X, Z, H_init, T, in_prod, in_prod_T, mask, L_init, u_init, v_init, to_estimate_u, to_estimate_v, to_add_ID, lambda_L(0), lambda_H(j), niter, rel_tol, is_quiet);
    NumericMatrix L_upd = previous_pt["L"];
    NumericMatrix H_upd = previous_pt["H"];
    res[j] = previous_pt;
    L_init = L_upd;
    NumericVector in_prod_upd = previous_pt["in_prod"];
    in_prod = in_prod_upd;
    u_init = previous_pt["u"];
    v_init = previous_pt["v"];
    H_init = H_upd;
  }
  for (int i = 1; i<num_lam_L; i++){
    List previous_pt = NNM_fit_H(M, X, Z, H_init, T, in_prod, in_prod_T, mask, L_init, u_init, v_init, to_estimate_u, to_estimate_v, to_add_ID, lambda_L(i), lambda_H(num_lam_H-1), niter, rel_tol, is_quiet);
    res[(i-1)+num_lam_H] = previous_pt;
    NumericMatrix L_upd = previous_pt["L"];
    NumericMatrix H_upd = previous_pt["H"];
    NumericVector in_prod_upd = previous_pt["in_prod"];
    in_prod = in_prod_upd;
    L_init = L_upd;
    u_init = previous_pt["u"];
    v_init = previous_pt["v"];
    H_init = H_upd;
  }
  return res;
}

List NNM_H_opt_path(NumericMatrix M, NumericMatrix X, NumericMatrix Z, NumericMatrix mask, bool to_estimate_u, bool to_estimate_v, bool to_add_ID, NumericVector lambda_L, NumericVector lambda_H, int niter = 1000, double rel_tol = 1e-5, bool is_quiet = 1){
  // This is just a wrapper for NNM_with_uv_init_H_opt_path, which just passes the initialization to this function.
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  List tmp_uv = initialize_uv_H(M, X, Z, mask, to_estimate_u, to_estimate_v, to_add_ID, niter, rel_tol);
  if(to_estimate_u == 1 || to_estimate_v ==1){
    return NNM_with_uv_init_H_opt_path(M, X, Z, tmp_uv["T_mat"], tmp_uv["in_prod_T"], mask, tmp_uv["u"], tmp_uv["v"], to_estimate_u, to_estimate_v, to_add_ID, lambda_L, lambda_H, niter, rel_tol, is_quiet);
  }
  else{
    return NNM_with_uv_init_H_opt_path(M, X, Z, tmp_uv["T_mat"], tmp_uv["in_prod_T"], mask, wrap(VectorXd::Zero(M_.rows())), wrap(VectorXd::Zero(M_.cols())), to_estimate_u, to_estimate_v, to_add_ID, lambda_L, lambda_H, niter, rel_tol, is_quiet);
  }
}

List NNM(NumericMatrix M, NumericMatrix mask, int num_lam_L = 100, NumericVector lambda_L = NumericVector::create(), bool to_estimate_u = 1, bool to_estimate_v = 1, int niter = 1000, double rel_tol = 1e-5, bool is_quiet = 1){

  // This function is just a wraper, which only passes vectors of all zero for u_init and v_init to NNM_with_uv_init.
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  List tmp_uv = initialize_uv(M, mask, to_estimate_u, to_estimate_v, niter, rel_tol);
  if(lambda_L.size() == 0){
    NumericVector lambda_Ls(num_lam_L);
    double max_lam_L=tmp_uv["lambda_L_max"];
    NumericVector lambda_Ls_without_zero = logsp(log10(max_lam_L), log10(max_lam_L)-3, num_lam_L-1);
    for(int i=0; i<num_lam_L-1; i++){
      lambda_Ls(i)= lambda_Ls_without_zero(i);
    }
    lambda_Ls(num_lam_L-1) = 0;
    lambda_L = lambda_Ls;
  }
  else{
    num_lam_L = lambda_L.size();
  }
  List tmp_res;
  if(to_estimate_u == 1 || to_estimate_v ==1){
    tmp_res = NNM_with_uv_init(M, mask, tmp_uv["u"], tmp_uv["v"], to_estimate_u, to_estimate_v, lambda_L, niter, rel_tol, is_quiet);
  }
  else{
    tmp_res = NNM_with_uv_init(M, mask, wrap(VectorXd::Zero(M_.rows())), wrap(VectorXd::Zero(M_.cols())), to_estimate_u, to_estimate_v, lambda_L, niter, rel_tol, is_quiet);
  }
  if()
  List out(num_lam_L);
  for (int i = 0; i < num_lam_L; i++){
    int current_ind = i;
    List current_config = tmp_res(current_ind);
    List this_config = List::create(Named("L") = current_config["L"],
                                    Named("u") = current_config["u"],
                                    Named("v") = current_config["v"],
                                    Named("lambda_L") = lambda_L[i]);
    out[current_ind] = this_config;
  }
  return out;
}

List NNM_CV(NumericMatrix M, NumericMatrix mask, bool to_estimate_u, bool to_estimate_v, int num_lam, int niter = 1000, double rel_tol = 1e-5, double cv_ratio = 0.6, int num_folds = 5, bool is_quiet = 1){

  // This function is the core function of NNM. Basically, it creates num_folds number of folds and does cross-validation
  // for choosing the best value of lambda_L, using data. Then, using the best model it fits to the
  // entire training set and computes resulting L, u, and v. The output of this function basically
  // contains all the information that we want from the algorithm.

  using Eigen::Map;
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  const Map<MatrixXd> mask_(as<Map<MatrixXd> >(mask));
  int num_rows = M_.rows();
  int num_cols = M_.cols();
  List confgs = create_folds(M, mask, to_estimate_u, to_estimate_v, niter, rel_tol , cv_ratio, num_folds);
  MatrixXd MSE(num_lam, num_folds);
  double max_lam_L=-1;
  for(int k=0; k<num_folds; k++){
    List h = confgs[k];
    double lam_max = h["lambda_L_max"];
    if(lam_max > max_lam_L){
      max_lam_L = lam_max;
    }
  }
  NumericVector lambda_Ls_without_zero = logsp(log10(max_lam_L), log10(max_lam_L)-3, num_lam-1);
  NumericVector lambda_Ls(num_lam);
  for(int i=0; i<num_lam-1; i++){
    lambda_Ls(i)= lambda_Ls_without_zero(i);
  }
  lambda_Ls(num_lam-1) = 0;
  for(int k=0; k<num_folds; k++){
    if(is_quiet == 0){
      std::cout << "Fold number " << k << " started" << std::endl;
    }
    List h = confgs[k];
    NumericMatrix mask_training = h["fold_mask"];
    const Map<MatrixXd> mask_training_(as<Map<MatrixXd> >(mask_training));
    MatrixXd M_tr_ = mask_training_.array() * M_.array();
    NumericMatrix M_tr = wrap(M_tr_);
    MatrixXd mask_validation_ = mask_.array() * (MatrixXd::Constant(num_rows,num_cols,1.0)-mask_training_).array();
    NumericMatrix mask_validation = wrap(mask_validation_);
    List train_configs = NNM_with_uv_init(M_tr, mask_training, h["u"], h["v"], to_estimate_u, to_estimate_v, lambda_Ls, niter, rel_tol, is_quiet);
    for (int i = 0; i < num_lam; i++){
      List this_config = train_configs[i];
      NumericMatrix L_use = this_config["L"];
      NumericVector u_use = this_config["u"];
      NumericVector v_use = this_config["v"];
      MSE(i,k) = std::pow(Compute_RMSE(M, mask_validation, L_use, u_use, v_use) ,2);
    }
  }
  VectorXd Avg_MSE = MSE.rowwise().mean();
  VectorXd Avg_RMSE = Avg_MSE.array().sqrt();
  Index minindex;
  double minRMSE = Avg_RMSE.minCoeff(&minindex);
  if(is_quiet == 0){
    std::cout << "Minimum RMSE achieved on validation set :" << minRMSE << std::endl;
    std::cout << "Optimum value of lambda_L : " << lambda_Ls[minindex] << std::endl;
    std::cout << "Fitting to the test set using optimum lambda_L..." << std::endl;
  }
  NumericVector lambda_Ls_n = lambda_Ls[lambda_Ls >= lambda_Ls(minindex)];
  List final_config = NNM(M, mask, lambda_Ls_n.size(), lambda_Ls, to_estimate_u, to_estimate_v, niter, rel_tol, 1);
  List z = final_config[minindex];
  MatrixXd L_fin = z["L"];
  VectorXd u_fin = z["u"];
  VectorXd v_fin = z["v"];
  return List::create(Named("L") = L_fin,
                      Named("u") = u_fin,
                      Named("v") = v_fin,
                      Named("Avg_RMSE") = Avg_RMSE,
                      Named("best_lambda") = lambda_Ls[minindex],
                      Named("min_RMSE") = minRMSE,
                      Named("lambda_L") = lambda_Ls);
}

List NNM_CV_H(NumericMatrix M, NumericMatrix X, NumericMatrix Z, NumericMatrix mask, bool to_estimate_u, bool to_estimate_v, bool to_add_ID, int num_lam_L, int num_lam_H, int niter, double rel_tol, double cv_ratio, int num_folds, bool is_quiet){

  // This function is the core function of NNM. Basically, it creates num_folds number of folds and does cross-validation
  // for choosing the best value of lambda_L and lambda_H, using data. Then, using the best model it fits to the
  // entire training set and computes resulting L, H, u, and v. The output of this function basically
  // contains all the information that we want from the algorithm.

  using Eigen::Map;
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  const Map<MatrixXd> mask_(as<Map<MatrixXd> >(mask));
  int num_rows = M_.rows();
  int num_cols = M_.cols();
  List confgs = create_folds_H(M, X, Z, to_estimate_u, to_estimate_v, to_add_ID, mask, niter, rel_tol , cv_ratio, num_folds);
  double max_lam_L=-1;
  double max_lam_H=-1;
  for(int k=0; k<num_folds; k++){
    List h = confgs[k];
    double lam_L_max = h["lambda_L_max"];
    double lam_H_max = h["lambda_H_max"];
    if(lam_L_max > max_lam_L){
      max_lam_L = lam_L_max;
    }
    if(lam_H_max > max_lam_H){
      max_lam_H = lam_H_max;
    }
  }
  NumericVector lambda_Ls_without_zero = logsp(log10(max_lam_L), log10(max_lam_L)-3, num_lam_L-1);
  NumericVector lambda_Hs_without_zero = logsp(log10(max_lam_H), log10(max_lam_H)-3, num_lam_H-1);
  NumericVector lambda_Ls(num_lam_L);
  for(int i=0; i<num_lam_L-1; i++){
    lambda_Ls(i)= lambda_Ls_without_zero(i);
  }
  lambda_Ls(num_lam_L-1) = 0;
  NumericVector lambda_Hs(num_lam_H);
  for(int i=0; i<num_lam_H-1; i++){
    lambda_Hs(i)= lambda_Hs_without_zero(i);
  }
  lambda_Hs(num_lam_H-1) = 0;
  MatrixXd MSE = MatrixXd::Zero(num_lam_L,num_lam_H);
  for(int k=0; k<num_folds; k++){
    if(is_quiet == 0){
      std::cout << "Fold number " << k << " started" << std::endl;
    }
    List h = confgs[k];
    NumericMatrix mask_training = h["fold_mask"];
    const Map<MatrixXd> mask_training_(as<Map<MatrixXd> >(mask_training));
    MatrixXd M_tr_ = mask_training_.array() * M_.array();
    NumericMatrix M_tr = wrap(M_tr_);
    MatrixXd mask_validation_ = mask_.array() * (MatrixXd::Constant(num_rows,num_cols,1.0)-mask_training_).array();
    NumericMatrix mask_validation = wrap(mask_validation_);
    List train_configs = NNM_with_uv_init_H(M_tr, X, Z, h["T_mat"], h["in_prod_T"], mask_training, h["u"], h["v"], to_estimate_u, to_estimate_v, to_add_ID, lambda_Ls, lambda_Hs, niter, rel_tol, is_quiet);
    for (int i = 0; i < num_lam_L; i++){
      for (int j = 0; j < num_lam_H; j++){
        List this_config = train_configs[j*num_lam_L+i];
        NumericMatrix L_use = this_config["L"];
        NumericVector u_use = this_config["u"];
        NumericVector v_use = this_config["v"];
        NumericMatrix H_use = this_config["H"];
        MSE(i,j) += std::pow(Compute_RMSE_H(M, X, Z, H_use, to_add_ID, mask_validation, L_use, u_use, v_use) ,2);
      }
    }
  }
  MatrixXd Avg_MSE = MSE/num_folds;
  MatrixXd Avg_RMSE = Avg_MSE.array().sqrt();
  Index min_L_index;
  Index min_H_index;
  double minRMSE = Avg_RMSE.minCoeff(&min_L_index, &min_H_index);
  if(is_quiet == 0){
    std::cout << "Minimum RMSE achieved on validation set :" << minRMSE << std::endl;
    std::cout << "Optimum value of lambda_L : " << lambda_Ls(min_L_index) << std::endl;
    std::cout << "Optimum value of lambda_H : " << lambda_Hs(min_H_index) << std::endl;
    std::cout << "Fitting to the test set using optimum lambda_L and lambda_H ..." << std::endl;
  }
  NumericVector lambda_Ls_n = lambda_Ls[lambda_Ls >= lambda_Ls(min_L_index)];
  NumericVector lambda_Hs_n = lambda_Hs[lambda_Hs >= lambda_Hs(min_H_index)];
  List final_config = NNM_H_opt_path(M, X, Z, mask, to_estimate_u, to_estimate_v, to_add_ID, lambda_Ls_n, lambda_Hs_n, niter, rel_tol, 1);
  List z = final_config[min_L_index+min_H_index-1];
  MatrixXd H_fin = z["H"];
  MatrixXd L_fin = z["L"];
  VectorXd u_fin = z["u"];
  VectorXd v_fin = z["v"];
  return List::create(Named("H") = H_fin,
                      Named("L") = L_fin,
                      Named("u") = u_fin,
                      Named("v") = v_fin,
                      Named("Avg_RMSE") = Avg_RMSE,
                      Named("best_lambda_L") = lambda_Ls[min_L_index],
                      Named("best_lambda_H") = lambda_Hs[min_H_index],
                      Named("min_RMSE") = minRMSE,
                      Named("lambda_L") = lambda_Ls,
                      Named("lambda_H") = lambda_Hs);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////
/////// Input Checks
//////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////


bool mask_check(NumericMatrix mask){
  const Map<MatrixXd> mask_(as<Map<MatrixXd> >(mask));
  MatrixXd H = (MatrixXd::Constant(mask_.rows(),mask_.cols(),1.0) - mask_).cwiseProduct(mask_);
  return(H.isZero(1e-5));
}

bool X_size_check(NumericMatrix M, NumericMatrix X){
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  const Map<MatrixXd> X_(as<Map<MatrixXd> >(X));
  return (M_.rows() == X_.rows());
}

bool Z_size_check(NumericMatrix M, NumericMatrix Z){
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  const Map<MatrixXd> Z_(as<Map<MatrixXd> >(Z));
  return (M_.cols() == Z_.rows());
}

bool mask_size_check(NumericMatrix M, NumericMatrix mask){
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  const Map<MatrixXd> mask_(as<Map<MatrixXd> >(mask));
  return (M_.rows() == mask_.rows() && M_.cols() == mask_.cols());
}

List normalize(NumericMatrix mat){
  const Map<MatrixXd> mat_(as<Map<MatrixXd> >(mat));
  VectorXd col_norms = VectorXd::Zero(mat_.cols());
  MatrixXd mat_norm = MatrixXd::Zero(mat_.rows(), mat_.cols());
  if(mat_.cols()>0){
    for (int i=0; i < mat_.cols(); i++){
      col_norms(i) = mat_.col(i).norm();
      mat_norm.col(i) = mat_.col(i) / col_norms(i);
    }
  }
  return List::create(Named("mat_norm")=mat_norm,
                      Named("col_norms") = col_norms);
}

NumericMatrix normalize_back_rows(NumericMatrix H, NumericVector row_H_scales){
  const Map<MatrixXd> H_(as<Map<MatrixXd> >(H));
  MatrixXd H_new = H_;
  if(row_H_scales.size()){
    for (int i=0; i < row_H_scales.size(); i++){
      H_new.row(i) = H_new.row(i) / row_H_scales(i);
    }
  }
  return wrap(H_new);
}

NumericMatrix normalize_back_cols(NumericMatrix H, NumericVector col_H_scales){
  const Map<MatrixXd> H_(as<Map<MatrixXd> >(H));
  MatrixXd H_new = H_;
  if(col_H_scales.size()){
    for (int i=0; i < col_H_scales.size(); i++){
      H_new.col(i) = H_new.col(i) / col_H_scales(i);
    }
  }
  return wrap(H_new);
}



///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
////// Export functions to use in R
///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////
// EXPORT mcnnm_lam_range
//////////////////////////////

int mcnnm_lam_range_check(NumericMatrix M, NumericMatrix mask, bool to_estimate_u = 1, bool to_estimate_v = 1, int niter = 1000, double rel_tol = 1e-5){
  if(mask_check(mask) == 0){
    std::cerr << "Error: The mask matrix should only include 0 (for missing) and 1 (for observed entries)" << std::endl;
    return 0;
  }
  if(mask_size_check(M,mask) == 0){
    std::cerr << "Error: M matrix and mask matrix dimensions should match" << std::endl;
    return 0;
  }
  if(mask_size_check(M,mask) == 0){
    std::cerr << "Error: M matrix and mask matrix dimensions should match" << std::endl;
    return 0;
  }
  return 1;
}
// [[Rcpp::export]]
double mcnnm_lam_range(NumericMatrix M, NumericMatrix mask, bool to_estimate_u = 1, bool to_estimate_v = 1, int niter = 1000, double rel_tol = 1e-5){
  int input_checks = mcnnm_lam_range_check(M, mask, to_estimate_u, to_estimate_v, niter, rel_tol);
  if (input_checks == 0){
    throw std::invalid_argument("Invalid inputs ! Please modify");
  }
  List res= initialize_uv(M, mask, to_estimate_u, to_estimate_v, niter, rel_tol);
  return res["lambda_L_max"];
}

///////////////////////////////
// EXPORT mcnnm_lam_range
//////////////////////////////

int mcnnm_wc_lam_range_check(NumericMatrix M, NumericMatrix X, NumericMatrix Z, NumericMatrix mask, bool to_normalize = 1, bool to_estimate_u = 1, bool to_estimate_v = 1, bool to_add_ID = 1, int niter = 1000, double rel_tol = 1e-5){
  const Map<MatrixXd> X_(as<Map<MatrixXd> >(X));
  const Map<MatrixXd> Z_(as<Map<MatrixXd> >(Z));
  if(mask_check(mask) == 0){
    std::cerr << "Error: The mask matrix should only include 0 (for missing) and 1 (for observed entries)" << std::endl;
    return 0;
  }
  if(mask_size_check(M,mask) == 0){
    std::cerr << "Error: M matrix and mask matrix dimensions should match" << std::endl;
    return 0;
  }
  if(mask_size_check(M,mask) == 0){
    std::cerr << "Error: M matrix and mask matrix dimensions should match" << std::endl;
    return 0;
  }
  if(X_.rows() == 0 && to_add_ID == 0){
    std::cerr << "Error: No need for training H as X is empty and identity matrix addition is disabled. Run mcnnm_lam_range instead" << std::endl;
    return 0;
  }

  if(Z_.rows() == 0 && to_add_ID == 0){
    std::cerr << "Error: No need for training H as Z is empty and identity matrix addition is disabled. Run mcnnm_lam_range instead" << std::endl;
    return 0;
  }

  if(X_.rows() == 0 && Z_.rows() == 0){
    std::cerr << "Error: No need for training H as X and Z are both empty. Run mcnnm_lam_range instead" << std::endl;
    return 0;
  }

  if(X_.rows() > 0 && X_size_check(M,X) == 0){
    std::cerr << "Error: Number of rows of X should match with the number of rows of M" << std::endl;
    return 0;
  }
  if(Z_.rows() > 0 && Z_size_check(M,Z) == 0){
    std::cerr << "Error: Number of rows of Z should match with the number of columns of M" << std::endl;
    return 0;
  }
  return 1;
}
// [[Rcpp::export]]
List mcnnm_wc_lam_range(NumericMatrix M, NumericMatrix X, NumericMatrix Z, NumericMatrix mask, bool to_normalize = 1, bool to_estimate_u = 1, bool to_estimate_v = 1, bool to_add_ID = 1, int niter = 1000, double rel_tol = 1e-5){

  int input_checks = mcnnm_wc_lam_range_check(M, X, Z, mask, to_normalize, to_estimate_u, to_estimate_v, to_add_ID, niter, rel_tol);
  if (input_checks == 0){
    throw std::invalid_argument("Invalid inputs ! Please modify");
  }

  const Map<MatrixXd> X_(as<Map<MatrixXd> >(X));
  const Map<MatrixXd> Z_(as<Map<MatrixXd> >(Z));

  NumericMatrix X_norm = X;
  NumericVector X_col_norms;
  NumericMatrix Z_norm = Z;
  NumericVector Z_col_norms;
  if(to_normalize == 1 && X_.cols()>0){
      List X_upd = normalize(X);
      NumericMatrix X_tmp = X_upd["mat_norm"];
      X_norm = X_tmp;
      X_col_norms = X_upd["col_norms"];
  }
  if(to_normalize == 1 && Z_.cols()>0){
      List Z_upd = normalize(Z);
      NumericMatrix Z_tmp = Z_upd["mat_norm"];
      Z_norm = Z_tmp;
      Z_col_norms = Z_upd["col_norms"];
  }

  List res= initialize_uv_H(M, X_norm, Z_norm, mask, to_estimate_u, to_estimate_v, to_add_ID, niter, rel_tol);
  return List::create(Named("lambda_L_max") = res["lambda_L_max"],
                      Named("lambda_H_max") = res["lambda_H_max"]);
}

/////////////////////////////////
// EXPORT mcnnm
/////////////////////////////////

int mcnnm_check(NumericMatrix M, NumericMatrix mask, int num_lam_L = 100, NumericVector lambda_L = NumericVector::create(), bool to_estimate_u = 1, bool to_estimate_v = 1, int niter = 1000, double rel_tol = 1e-5, bool is_quiet = 1){
  if(lambda_L.size() > 0){
    num_lam_L = lambda_L.size();
  }
  if(mask_check(mask) == 0){
    std::cerr << "Error: The mask matrix should only include 0 (for missing) and 1 (for observed entries)" << std::endl;
    return 0;
  }
  if(mask_size_check(M,mask) == 0){
    std::cerr << "Error: M matrix and mask matrix dimensions should match" << std::endl;
    return 0;
  }
  if(mask_size_check(M,mask) == 0){
    std::cerr << "Error: M matrix and mask matrix dimensions should match" << std::endl;
    return 0;
  }

  if(num_lam_L > 2500){
    std::cerr << "Warning: The training might take very long. Please decrease number of lambda_Ls" << std::endl;
  }
  if(rel_tol < 1e-10){
    std::cerr << "Warning: The chosen value for relative improvement is very small. Training might take longer" << std::endl;
  }
  return 1;
}
// [[Rcpp::export]]
List mcnnm(NumericMatrix M, NumericMatrix mask, int num_lam_L = 100, NumericVector lambda_L = NumericVector::create(), bool to_estimate_u = 1, bool to_estimate_v = 1, int niter = 1000, double rel_tol = 1e-5, bool is_quiet = 1){
  List res;
  int input_checks = mcnnm_check(M, mask, num_lam_L, lambda_L, to_estimate_u, to_estimate_v, niter, rel_tol, is_quiet);
  if (input_checks == 0){
    throw std::invalid_argument("Invalid inputs ! Please modify");
  }
  return NNM(M, mask, num_lam_L, lambda_L, to_estimate_u, to_estimate_v, niter, rel_tol, is_quiet);
}


/////////////////////////////////
// EXPORT mcnnm_fit
/////////////////////////////////

int mcnnm_fit_check(NumericMatrix M, NumericMatrix mask, double lambda_L, bool to_estimate_u = 1, bool to_estimate_v = 1, int niter = 1000, double rel_tol = 1e-5, bool is_quiet = 1){
  if(mask_check(mask) == 0){
    std::cerr << "Error: The mask matrix should only include 0 (for missing) and 1 (for observed entries)" << std::endl;
    return 0;
  }
  if(mask_size_check(M,mask) == 0){
    std::cerr << "Error: M matrix and mask matrix dimensions should match" << std::endl;
    return 0;
  }
  if(mask_size_check(M,mask) == 0){
    std::cerr << "Error: M matrix and mask matrix dimensions should match" << std::endl;
    return 0;
  }
  if(rel_tol < 1e-10){
    std::cerr << "Warning: The chosen value for relative improvement is very small. Training might take longer" << std::endl;
  }
  return 1;
}
// [[Rcpp::export]]
List mcnnm_fit(NumericMatrix M, NumericMatrix mask, double lambda_L, bool to_estimate_u = 1, bool to_estimate_v = 1, int niter = 1000, double rel_tol = 1e-5, bool is_quiet = 1){
  List res;
  int input_checks = mcnnm_fit_check(M, mask, lambda_L, to_estimate_u, to_estimate_v, niter, rel_tol, is_quiet);
  if (input_checks == 0){
    throw std::invalid_argument("Invalid inputs ! Please modify");
  }
  double max_lam_L = mcnnm_lam_range(M, mask, to_estimate_u, to_estimate_v, niter, rel_tol);
  if(lambda_L >= max_lam_L){
    NumericVector lambda_Ls(1);
    lambda_Ls(0) = lambda_L;
    List Q = NNM(M, mask, 1, lambda_Ls, to_estimate_u, to_estimate_v, niter, rel_tol, is_quiet);
    return List::create(Named("L") = Q["L"],
                        Named("u") = Q["u"],
                        Named("v") = Q["v"],
                        Named("lambda_L") = lambda_L);
  }
  else{
    int num_lam_L = 100;
    NumericVector lambda_Ls(num_lam_L);
    NumericVector lambda_Ls_without_zero = logsp(log10(max_lam_L), log10(max_lam_L)-3, num_lam_L-1);
    for(int i=0; i<num_lam_L-1; i++){
      lambda_Ls(i)= lambda_Ls_without_zero(i);
    }
    lambda_Ls(num_lam_L-1) = 0;
    NumericVector lambda_Ls_n = lambda_Ls[lambda_Ls >= lambda_L];
    int num_lam_L_n = lambda_Ls_n.size();
    NumericVector lambda_Ls_fin(num_lam_L_n+1);
    for(int i=0; i<num_lam_L_n; i++){
      lambda_Ls_fin(i)= lambda_Ls_n(i);
    }
    lambda_Ls_fin(num_lam_L_n) = lambda_L;
    List Q = NNM(M, mask, num_lam_L_n+1, lambda_Ls_fin, to_estimate_u, to_estimate_v, niter, rel_tol, is_quiet);
    List final_config = Q[num_lam_L_n];
    return List::create(Named("L") = final_config["L"],
                        Named("u") = final_config["u"],
                        Named("v") = final_config["v"],
                        Named("lambda_L") = lambda_L);
  }
}

/////////////////////////////////
// EXPORT mcnnm_wc
////////////////////////////////

int mcnnm_wc_check(NumericMatrix M, NumericMatrix X, NumericMatrix Z, NumericMatrix mask, int num_lam_L = 30, int num_lam_H = 30, NumericVector lambda_L = NumericVector::create(), NumericVector lambda_H = NumericVector::create(), bool to_normalize = 1, bool to_estimate_u = 1, bool to_estimate_v = 1, bool to_add_ID = 1, int niter = 100, double rel_tol = 1e-5, bool is_quiet = 1){
  if(lambda_L.size() > 0){
    num_lam_L = lambda_L.size();
  }
  if(lambda_H.size() > 0){
    num_lam_H = lambda_H.size();
  }
  const Map<MatrixXd> X_(as<Map<MatrixXd> >(X));
  const Map<MatrixXd> Z_(as<Map<MatrixXd> >(Z));
  if(mask_check(mask) == 0){
    std::cerr << "Error: The mask matrix should only include 0 (for missing) and 1 (for observed entries)" << std::endl;
    return 0;
  }
  if(mask_size_check(M,mask) == 0){
    std::cerr << "Error: M matrix and mask matrix dimensions should match" << std::endl;
    return 0;
  }
  if(mask_size_check(M,mask) == 0){
    std::cerr << "Error: M matrix and mask matrix dimensions should match" << std::endl;
    return 0;
  }
  if(X_.rows() == 0 && to_add_ID == 0){
    std::cerr << "Error: No need for training H as X is empty and identity matrix addition is disabled. Run mcnnm_cv instead" << std::endl;
    return 0;
  }

  if(Z_.rows() == 0 && to_add_ID == 0){
    std::cerr << "Error: No need for training H as Z is empty and identity matrix addition is disabled. Run mcnnm_cv instead" << std::endl;
    return 0;
  }

  if(X_.rows() == 0 && Z_.rows() == 0){
    std::cerr << "Error: No need for training H as X and Z are both empty. Run mcnnm_cv instead" << std::endl;
    return 0;
  }

  if(X_.rows() > 0 && X_size_check(M,X) == 0){
    std::cerr << "Error: Number of rows of X should match with the number of rows of M" << std::endl;
    return 0;
  }
  if(Z_.rows() > 0 && Z_size_check(M,Z) == 0){
    std::cerr << "Error: Number of rows of Z should match with the number of columns of M" << std::endl;
    return 0;
  }
  if(num_lam_L * num_lam_H > 2500){
    std::cerr << "Warning: The training might take very long. Please decrease number of lambda_Ls or lambda_Hs" << std::endl;
  }

  if(rel_tol < 1e-10){
    std::cerr << "Warning: The chosen value for relative improvement is very small. Training might take longer" << std::endl;
  }
  return 1;
}
// [[Rcpp::export]]
List mcnnm_wc(NumericMatrix M, NumericMatrix X, NumericMatrix Z, NumericMatrix mask, int num_lam_L = 30, int num_lam_H = 30, NumericVector lambda_L = NumericVector::create(), NumericVector lambda_H = NumericVector::create(), bool to_normalize = 1, bool to_estimate_u = 1, bool to_estimate_v = 1, bool to_add_ID = 1, int niter = 100, double rel_tol = 1e-5, bool is_quiet = 1){

  int input_checks = mcnnm_wc_check(M, X, Z, mask, num_lam_L, num_lam_H, lambda_L, lambda_H, to_normalize, to_estimate_u, to_estimate_v, to_add_ID, niter, rel_tol,is_quiet);
  if (input_checks == 0){
    throw std::invalid_argument("Invalid inputs ! Please modify");
  }


  const Map<MatrixXd> X_(as<Map<MatrixXd> >(X));
  const Map<MatrixXd> Z_(as<Map<MatrixXd> >(Z));

  NumericMatrix X_norm = X;
  NumericVector X_col_norms;
  NumericMatrix Z_norm = Z;
  NumericVector Z_col_norms;
  if(to_normalize == 1 && X_.cols()>0){
      List X_upd = normalize(X);
      NumericMatrix X_tmp = X_upd["mat_norm"];
      X_norm = X_tmp;
      X_col_norms = X_upd["col_norms"];
  }
  if(to_normalize == 1 && Z_.cols()>0){
      List Z_upd = normalize(Z);
      NumericMatrix Z_tmp = Z_upd["mat_norm"];
      Z_norm = Z_tmp;
      Z_col_norms = Z_upd["col_norms"];
  }

  List res = NNM_H(M, X_norm, Z_norm, mask, num_lam_L, num_lam_H, lambda_L, lambda_H, to_estimate_u, to_estimate_v, to_add_ID, niter, rel_tol, is_quiet);

  if(to_normalize == 1 && X_.cols()>0){
    for(int i=0; i<res.size(); i++){
      List tmp = res(i);
      NumericMatrix H_renorm = normalize_back_rows(tmp["H"], X_col_norms);
      tmp["H"] = H_renorm;
      res(i) = tmp;
    }
  }
  if(to_normalize == 1 && Z_.cols()>0){
    for(int i=0; i<res.size(); i++){
      List tmp = res(i);
      NumericMatrix H_renorm = normalize_back_cols(tmp["H"], Z_col_norms);
      tmp["H"] = H_renorm;
      res(i) = tmp;
    }
  }
  return res;
}

/////////////////////////////////
// EXPORT mcnnm_wc
////////////////////////////////

int mcnnm_wc_fit_check(NumericMatrix M, NumericMatrix X, NumericMatrix Z, NumericMatrix mask, double lambda_L, double lambda_H, bool to_normalize = 1, bool to_estimate_u = 1, bool to_estimate_v = 1, bool to_add_ID = 1, int niter = 100, double rel_tol = 1e-5, bool is_quiet = 1){
  const Map<MatrixXd> X_(as<Map<MatrixXd> >(X));
  const Map<MatrixXd> Z_(as<Map<MatrixXd> >(Z));
  if(mask_check(mask) == 0){
    std::cerr << "Error: The mask matrix should only include 0 (for missing) and 1 (for observed entries)" << std::endl;
    return 0;
  }
  if(mask_size_check(M,mask) == 0){
    std::cerr << "Error: M matrix and mask matrix dimensions should match" << std::endl;
    return 0;
  }
  if(mask_size_check(M,mask) == 0){
    std::cerr << "Error: M matrix and mask matrix dimensions should match" << std::endl;
    return 0;
  }
  if(X_.rows() == 0 && to_add_ID == 0){
    std::cerr << "Error: No need for training H as X is empty and identity matrix addition is disabled. Run mcnnm_fit instead" << std::endl;
    return 0;
  }

  if(Z_.rows() == 0 && to_add_ID == 0){
    std::cerr << "Error: No need for training H as Z is empty and identity matrix addition is disabled. Run mcnnm_fit instead" << std::endl;
    return 0;
  }

  if(X_.rows() == 0 && Z_.rows() == 0){
    std::cerr << "Error: No need for training H as X and Z are both empty. Run mcnnm_fit instead" << std::endl;
    return 0;
  }

  if(X_.rows() > 0 && X_size_check(M,X) == 0){
    std::cerr << "Error: Number of rows of X should match with the number of rows of M" << std::endl;
    return 0;
  }
  if(Z_.rows() > 0 && Z_size_check(M,Z) == 0){
    std::cerr << "Error: Number of rows of Z should match with the number of columns of M" << std::endl;
    return 0;
  }

  if(rel_tol < 1e-10){
    std::cerr << "Warning: The chosen value for relative improvement is very small. Training might take longer" << std::endl;
  }
  return 1;
}
// [[Rcpp::export]]
List mcnnm_wc_fit(NumericMatrix M, NumericMatrix X, NumericMatrix Z, NumericMatrix mask, double lambda_L, double lambda_H, bool to_normalize = 1, bool to_estimate_u = 1, bool to_estimate_v = 1, bool to_add_ID = 1, int niter = 100, double rel_tol = 1e-5, bool is_quiet = 1){
  int input_checks = mcnnm_wc_fit_check(M, X, Z, mask, lambda_L, lambda_H, to_normalize, to_estimate_u, to_estimate_v, to_add_ID, niter, rel_tol, is_quiet);
  if (input_checks == 0){
    throw std::invalid_argument("Invalid inputs ! Please modify");
  }

  const Map<MatrixXd> X_(as<Map<MatrixXd> >(X));
  const Map<MatrixXd> Z_(as<Map<MatrixXd> >(Z));

  NumericMatrix X_norm = X;
  NumericVector X_col_norms;
  NumericMatrix Z_norm = Z;
  NumericVector Z_col_norms;
  if(to_normalize == 1 && X_.cols()>0){
      List X_upd = normalize(X);
      NumericMatrix X_tmp = X_upd["mat_norm"];
      X_norm = X_tmp;
      X_col_norms = X_upd["col_norms"];
  }
  if(to_normalize == 1 && Z_.cols()>0){
      List Z_upd = normalize(Z);
      NumericMatrix Z_tmp = Z_upd["mat_norm"];
      Z_norm = Z_tmp;
      Z_col_norms = Z_upd["col_norms"];
  }

  List ranges = mcnnm_wc_lam_range(M, X_norm, Z_norm, mask, to_estimate_u, to_estimate_v, to_add_ID, niter, rel_tol);
  double max_lam_L = ranges["lambda_L_max"];
  double max_lam_H = ranges["lambda_H_max"];
  NumericVector lambda_Ls_fin;
  NumericVector lambda_Hs_fin;
  if(lambda_L >= max_lam_L){
    NumericVector lambda_Ls(1);
    lambda_Ls(0) = lambda_L;
    lambda_Ls_fin = lambda_Ls;
  }
  else{
    int num_lam_L = 30;
    NumericVector lambda_Ls(num_lam_L);
    NumericVector lambda_Ls_without_zero = logsp(log10(max_lam_L), log10(max_lam_L)-3, num_lam_L-1);
    for(int i=0; i<num_lam_L-1; i++){
      lambda_Ls(i)= lambda_Ls_without_zero(i);
    }
    lambda_Ls(num_lam_L-1) = 0;
    NumericVector lambda_Ls_n = lambda_Ls[lambda_Ls >= lambda_L];
    int num_lam_L_n = lambda_Ls_n.size();
    NumericVector lambda_Ls_(num_lam_L_n+1);
    for(int i=0; i<num_lam_L_n; i++){
      lambda_Ls_(i)= lambda_Ls_n(i);
    }
    lambda_Ls_(num_lam_L_n) = lambda_L;
    lambda_Ls_fin = lambda_Ls_;
  }
  if(lambda_H >= max_lam_H){
    NumericVector lambda_Hs(1);
    lambda_Hs(0) = lambda_H;
    lambda_Hs_fin = lambda_Hs;
  } else{
    int num_lam_H = 30;
    NumericVector lambda_Hs(num_lam_H);
    NumericVector lambda_Hs_without_zero = logsp(log10(max_lam_H), log10(max_lam_H)-3, num_lam_H-1);
    for(int i=0; i<num_lam_H-1; i++){
      lambda_Hs(i)= lambda_Hs_without_zero(i);
    }
    lambda_Hs(num_lam_H-1) = 0;
    NumericVector lambda_Hs_n = lambda_Hs[lambda_Hs >= lambda_H];
    int num_lam_H_n = lambda_Hs_n.size();
    NumericVector lambda_Hs_(num_lam_H_n+1);
    for(int i=0; i<num_lam_H_n; i++){
      lambda_Hs_(i)= lambda_Hs_n(i);
    }
    lambda_Hs_(num_lam_H_n) = lambda_H;
    lambda_Hs_fin = lambda_Hs_;
  }
  List Q = NNM_H_opt_path(M, X_norm, Z_norm, mask, to_estimate_u, to_estimate_v, to_add_ID, lambda_Ls_fin, lambda_Hs_fin, niter, rel_tol, is_quiet);
  List final_config = Q[Q.size()-1];

  if(to_normalize == 1 && X_.cols()>0){
    List tmp = final_config;
    NumericMatrix H_renorm = normalize_back_rows(final_config["H"], X_col_norms);
    final_config["H"] = H_renorm;
  }
  if(to_normalize == 1 && Z_.cols()>0){
    List tmp = final_config;
    NumericMatrix H_renorm = normalize_back_cols(final_config["H"], Z_col_norms);
    final_config ["H"]= H_renorm;
  }

  return List::create(Named("H") = final_config["H"],
                      Named("L") = final_config["L"],
                      Named("u") = final_config["u"],
                      Named("v") = final_config["v"],
                      Named("lambda_L") = lambda_L,
                      Named("lambda_H") = lambda_H);
}

/////////////////////////////////
// EXPORT mcnnm_cv
/////////////////////////////////

int mcnnm_cv_check(NumericMatrix M, NumericMatrix mask, bool to_estimate_u, bool to_estimate_v, int num_lam_L , int niter, double rel_tol, double cv_ratio, int num_folds, bool is_quiet){
  if(mask_check(mask) == 0){
    std::cerr << "Error: The mask matrix should only include 0 (for missing) and 1 (for observed entries)" << std::endl;
    return 0;
  }
  if(mask_size_check(M,mask) == 0){
    std::cerr << "Error: M matrix and mask matrix dimensions should match" << std::endl;
    return 0;
  }
  if(mask_size_check(M,mask) == 0){
    std::cerr << "Error: M matrix and mask matrix dimensions should match" << std::endl;
    return 0;
  }

  if(num_lam_L> 2500){
    std::cerr << "Warning: The cross-validation might take very long. Please decrease number of lambda_Ls" << std::endl;
  }
  if(cv_ratio < 0.1 || cv_ratio > 0.9){
    std::cerr << "Error: The cross-validation ratio should be between 10 to 90 percent for getting accurate results. Please modify it" << std::endl;
    return 0;
  }

  if(num_folds > 20){
    std::cerr << "Warning: Number of random folds are chosen to be greater than 20. This process might take long" << std::endl;
  }

  if(rel_tol < 1e-10){
    std::cerr << "Warning: The chosen value for relative improvement is very small. Training might take longer" << std::endl;
  }
  return 1;
}

// [[Rcpp::export]]
List mcnnm_cv(NumericMatrix M, NumericMatrix mask, bool to_estimate_u = 1, bool to_estimate_v = 1, int num_lam_L = 100, int niter = 400, double rel_tol = 1e-5, double cv_ratio = 0.8, int num_folds = 5, bool is_quiet = 1){
  List res;
  int input_checks = mcnnm_cv_check(M, mask, to_estimate_u, to_estimate_v, num_lam_L, niter, rel_tol, cv_ratio, num_folds, is_quiet);
  if (input_checks == 0){
    throw std::invalid_argument("Invalid inputs ! Please modify");
  }
  return NNM_CV(M, mask, to_estimate_u, to_estimate_v, num_lam_L, niter, rel_tol, cv_ratio, num_folds, is_quiet);
}

//////////////////////////////////
// EXPORT mcnnm_wc_cv
/////////////////////////////////

int mcnnm_wc_cv_check(NumericMatrix M, NumericMatrix X, NumericMatrix Z, NumericMatrix mask, bool to_normalize, bool to_estimate_u, bool to_estimate_v, bool to_add_ID, int num_lam_L, int num_lam_H, int niter, double rel_tol, double cv_ratio, int num_folds, bool is_quiet){
  const Map<MatrixXd> X_(as<Map<MatrixXd> >(X));
  const Map<MatrixXd> Z_(as<Map<MatrixXd> >(Z));
  if(mask_check(mask) == 0){
    std::cerr << "Error: The mask matrix should only include 0 (for missing) and 1 (for observed entries)" << std::endl;
    return 0;
  }
  if(mask_size_check(M,mask) == 0){
    std::cerr << "Error: M matrix and mask matrix dimensions should match" << std::endl;
    return 0;
  }
  if(mask_size_check(M,mask) == 0){
    std::cerr << "Error: M matrix and mask matrix dimensions should match" << std::endl;
    return 0;
  }

  if(X_.rows() == 0 && to_add_ID == 0){
    std::cerr << "Error: No need for training H as X is empty and identity matrix addition is disabled. Run MCNNM_cv instead" << std::endl;
    return 0;
  }

  if(Z_.rows() == 0 && to_add_ID == 0){
    std::cerr << "Error: No need for training H as Z is empty and identity matrix addition is disabled. Run MCNNM_cv instead" << std::endl;
    return 0;
  }

  if(X_.rows() == 0 && Z_.rows() == 0){
    std::cerr << "Error: No need for training H as X and Z are both empty. Run MCNNM_cv instead" << std::endl;
    return 0;
  }

  if(X_.rows() > 0 && X_size_check(M,X) == 0){
    std::cerr << "Error: Number of rows of X should match with the number of rows of M" << std::endl;
    return 0;
  }
  if(Z_.rows() > 0 && Z_size_check(M,Z) == 0){
    std::cerr << "Error: Number of rows of Z should match with the number of columns of M" << std::endl;
    return 0;
  }
  if(num_lam_L * num_lam_H > 2500){
    std::cerr << "Warning: The cross-validation might take very long. Please decrease number of lambda_Ls or lambda_Hs" << std::endl;
  }
  if(cv_ratio < 0.1 || cv_ratio > 0.9){
    std::cerr << "Error: The cross-validation ratio should be between 10 to 90 percent for getting accurate results. Please modify it" << std::endl;
    return 0;
  }

  if(num_folds > 3){
    std::cerr << "Warning: Number of random folds are chosen to be greater than 3. This process might take long" << std::endl;
  }

  if(rel_tol < 1e-10){
    std::cerr << "Warning: The chosen value for relative improvement is very small. Training might take longer" << std::endl;
  }
  return 1;
}

// [[Rcpp::export]]
List mcnnm_wc_cv(NumericMatrix M, NumericMatrix X, NumericMatrix Z, NumericMatrix mask, bool to_normalize = 1, bool to_estimate_u = 1, bool to_estimate_v = 1, bool to_add_ID = 1, int num_lam_L = 30, int num_lam_H = 30, int niter = 100, double rel_tol = 1e-5, double cv_ratio = 0.8, int num_folds = 1, bool is_quiet = 1){
  List res;
  int input_checks = mcnnm_wc_cv_check(M, X, Z, mask, to_normalize, to_estimate_u, to_estimate_v, to_add_ID, num_lam_L, num_lam_H, niter, rel_tol, cv_ratio, num_folds, is_quiet);
  if (input_checks == 0){
    throw std::invalid_argument("Invalid inputs ! Please modify");
  }

  const Map<MatrixXd> X_(as<Map<MatrixXd> >(X));
  const Map<MatrixXd> Z_(as<Map<MatrixXd> >(Z));

  NumericMatrix X_norm = X;
  NumericVector X_col_norms;
  NumericMatrix Z_norm = Z;
  NumericVector Z_col_norms;
  if(to_normalize == 1 && X_.cols()>0){
      List X_upd = normalize(X);
      NumericMatrix X_tmp = X_upd["mat_norm"];
      X_norm = X_tmp;
      X_col_norms = X_upd["col_norms"];
  }
  if(to_normalize == 1 && Z_.cols()>0){
      List Z_upd = normalize(Z);
      NumericMatrix Z_tmp = Z_upd["mat_norm"];
      Z_norm = Z_tmp;
      Z_col_norms = Z_upd["col_norms"];
  }

  res = NNM_CV_H(M, X_norm, Z_norm, mask, to_estimate_u, to_estimate_v, to_add_ID, num_lam_L, num_lam_H, niter, rel_tol, cv_ratio, num_folds, is_quiet);
  if(to_normalize == 1 && X_.cols()>0){
    List tmp = res;
    NumericMatrix H_renorm = normalize_back_rows(tmp["H"], X_col_norms);
    tmp["H"] = H_renorm;
    res = tmp;
  }
  if(to_normalize == 1 && Z_.cols()>0){
    List tmp = res;
    NumericMatrix H_renorm = normalize_back_cols(tmp["H"], Z_col_norms);
    tmp["H"] = H_renorm;
    res = tmp;
  }
  return res;
}
