#include <iostream>
#include <cmath>
#include <Rcpp.h>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <RcppEigen.h>
#include <random>

using namespace Eigen;
using namespace Rcpp;

List MySVD(NumericMatrix M){

  // This function computes the Singular Value Decomposition and it passes U,V,Sigma.
  // As SVD is the most time consuming part of our algorithm, this function is created to effectivelly
  // compute and compare different algorithms' speeds.

  using Eigen::Map;
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));

  BDCSVD<MatrixXd> svd( M_.rows(), M_.cols(), ComputeThinV | ComputeThinU );
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

NumericMatrix ComputeMatrix(NumericMatrix L,NumericVector u, NumericVector v){

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

List initialize_uv(NumericMatrix M, NumericMatrix mask, int niter = 1000, double rel_tol = 1e-5){

  // This function solves finds the optimal u and v assuming that L is zero. This would be later
  // helpful when we want to perform warm start on values of lambda_L. This function also outputs
  // the smallest value of lambda_L which causes L to be zero (all singular values vanish after a SVT update)

  using Eigen::Map;
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  const Map<MatrixXd> mask_(as<Map<MatrixXd> >(mask));

  BDCSVD<MatrixXd> svd(M_.rows(), M_.cols());
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
    u = update_u(M, mask, L, v);
    v = update_v(M, mask, L, u);
    new_obj_val = Compute_objval (M, mask, L, u, v, 0, 0);
    double rel_error = (new_obj_val-obj_val)/obj_val;
    if(rel_error < rel_tol && rel_error >= 0){
      break;
    }
    obj_val = new_obj_val;
  }
  NumericMatrix H = ComputeMatrix(L,u,v);
  const Map<MatrixXd> H_(as<Map<MatrixXd> >(H));
  MatrixXd P_omega_ = (M_ - H_).array()*mask_.array();
  svd.compute(P_omega_);
  double lambda_L_max = 2.0 * svd.singularValues().maxCoeff()/mask_.sum();

  return List::create(Named("u") = u,
                      Named("v") = v,
                      Named("lambda_L_max") = lambda_L_max);
  }

List create_folds(NumericMatrix M, NumericMatrix mask, int niter = 1000, double rel_tol = 1e-5, double cv_ratio = 0.7, int num_folds=5){

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
    List tmp_uv = initialize_uv(M_tr, fold_mask,  niter = 1000, rel_tol = 1e-5);
    List fold_k = List::create(Named("u") = tmp_uv["u"],
                               Named("v") = tmp_uv["v"],
                               Named("lambda_L_max") = tmp_uv["lambda_L_max"],
                               Named("fold_mask") = fold_mask);
    out[k] = fold_k;
  }
  return out;
}

// [[Rcpp::export]]
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
    if(new_obj_val < 1e-15){
      if(is_quiet == 0){
        std::cout << "Terminated at iteration : " << iter << ", for lambda_L :" << lambda_L << ", with obj_val :" << new_obj_val << std::endl;
      }
    break;
    }
    if(rel_error < rel_tol && rel_error >= 0){
      if(is_quiet == 0){
        std::cout << "Terminated at iteration : " << iter << ", for lambda_L :" << lambda_L << ", with obj_val :" << new_obj_val << std::endl;
      }
    break;
    }
    obj_val = new_obj_val;
  }
  return List::create(Named("L") = L,
  Named("u") = u,
  Named("v") = v);
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

// [[Rcpp::export]]
List NNM(NumericMatrix M, NumericMatrix mask, bool to_estimate_u, bool to_estimate_v, NumericVector lambda_L, int niter = 1000, double rel_tol = 1e-5, bool is_quiet = 1){

  // This function is just a wraper, which only passes vectors of all zero for u_init and v_init to NNM_with_uv_init.

  using Eigen::Map;
  const Map<MatrixXd> M_(as<Map<MatrixXd> >(M));
  if(to_estimate_u == 1 || to_estimate_v ==1){
    List tmp_uv = initialize_uv(M, mask,  niter, rel_tol);
    return NNM_with_uv_init(M, mask, tmp_uv["u"], tmp_uv["v"], to_estimate_u, to_estimate_v, lambda_L, niter, rel_tol, is_quiet);
  }
  else{
    return NNM_with_uv_init(M, mask, wrap(VectorXd::Zero(M_.rows())), wrap(VectorXd::Zero(M_.cols())), to_estimate_u, to_estimate_v, lambda_L, niter, rel_tol, is_quiet);
  }
}

// [[Rcpp::export]]
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
  List confgs = create_folds(M, mask, niter, rel_tol , cv_ratio, num_folds);
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
  List final_config = NNM(M, mask, to_estimate_u, to_estimate_v, lambda_Ls, niter, rel_tol, 1);
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
                      Named("lambda") = lambda_Ls);
}
