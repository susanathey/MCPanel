rm(list=ls())

## Loading Source files
library(MCPanel)
library(glmnet)
library(ggplot2)
library(latex2exp)


## Reading data
setwd("./examples_from_paper/california/")
X <- read.csv('smok_covariates.csv',header=F)
Y <- t(read.csv('smok_outcome.csv',header=F))
treat <- t(read.csv('smok_treatment.csv',header=F))
years <- 1970:2000

## First row (treated unit)
CA_y <- Y[1,]

## Working with the rest of matrix
treat <- treat[-1,]
Y <- Y[-1,]


## Setting up the configuration
N <- nrow(treat)
T <- ncol(treat)
number_T0 = 5
T0 <- ceiling(T*((1:number_T0)*2-1)/(2*number_T0))
N_t <- 8
num_runs <- 10
is_simul <- 1 ## Whether to simulate Simultaneus Adoption or Staggered Adoption
to_save <- 1 ## Whether to save the plot or not

## Matrices for saving RMSE values

MCPanel_RMSE_test <- matrix(0L,num_runs,length(T0))
EN_RMSE_test <- matrix(0L,num_runs,length(T0))
ENT_RMSE_test <- matrix(0L,num_runs,length(T0))
DID_RMSE_test <- matrix(0L,num_runs,length(T0))
ADH_RMSE_test <- matrix(0L,num_runs,length(T0))

## Run different methods

for(i in c(1:num_runs)){
  print(paste0(paste0("Run number ", i)," started"))
  ## Fix the treated units in the whole run for a better comparison
  treat_indices <- sample(1:N, N_t)
  for (j in c(1:length(T0))){
    treat_mat <- matrix(1L, N, T);
    t0 <- T0[j]
    ## Simultaneuous (simul_adapt) or Staggered adoption (stag_adapt)
    if(is_simul == 1){
      treat_mat <- simul_adapt(Y, N_t, t0, treat_indices)
    }
    else{
      treat_mat <- stag_adapt(Y, N_t, t0, treat_indices)
    }
    Y_obs <- Y * treat_mat

    ## ------
    ## MC-NNM
    ## ------

    est_model_MCPanel <- mcnnm_cv(Y_obs, treat_mat, to_estimate_u = 1, to_estimate_v = 1)
    est_model_MCPanel$Mhat <- est_model_MCPanel$L + replicate(T,est_model_MCPanel$u) + t(replicate(N,est_model_MCPanel$v))
    est_model_MCPanel$msk_err <- (est_model_MCPanel$Mhat - Y)*(1-treat_mat)
    est_model_MCPanel$test_RMSE <- sqrt((1/sum(1-treat_mat)) * sum(est_model_MCPanel$msk_err^2))
    MCPanel_RMSE_test[i,j] <- est_model_MCPanel$test_RMSE

    ## -----
    ## EN : It does Not cross validate on alpha (only on lambda) and keep alpha = 1 (LASSO).
    ##      Change num_alpha to a larger number, if you are willing to wait a little longer.
    ## -----

    est_model_EN <- en_mp_rows(Y_obs, treat_mat, num_alpha = 1)
    est_model_EN_msk_err <- (est_model_EN - Y)*(1-treat_mat)
    est_model_EN_test_RMSE <- sqrt((1/sum(1-treat_mat)) * sum(est_model_EN_msk_err^2))
    EN_RMSE_test[i,j] <- est_model_EN_test_RMSE

    ## -----
    ## EN_T : It does Not cross validate on alpha (only on lambda) and keep alpha = 1 (LASSO).
    ##        Change num_alpha to a larger number, if you are willing to wait a little longer.
    ## -----
    est_model_ENT <- t(en_mp_rows(t(Y_obs), t(treat_mat), num_alpha = 1))
    est_model_ENT_msk_err <- (est_model_ENT - Y)*(1-treat_mat)
    est_model_ENT_test_RMSE <- sqrt((1/sum(1-treat_mat)) * sum(est_model_ENT_msk_err^2))
    ENT_RMSE_test[i,j] <- est_model_ENT_test_RMSE

    ## -----
    ## DID
    ## -----

    est_model_DID <- DID(Y_obs, treat_mat)
    est_model_DID_msk_err <- (est_model_DID - Y)*(1-treat_mat)
    est_model_DID_test_RMSE <- sqrt((1/sum(1-treat_mat)) * sum(est_model_DID_msk_err^2))
    DID_RMSE_test[i,j] <- est_model_DID_test_RMSE

    ## -----
    ## ADH
    ## -----
    est_model_ADH <- adh_mp_rows(Y_obs, treat_mat)
    est_model_ADH_msk_err <- (est_model_ADH - Y)*(1-treat_mat)
    est_model_ADH_test_RMSE <- sqrt((1/sum(1-treat_mat)) * sum(est_model_ADH_msk_err^2))
    ADH_RMSE_test[i,j] <- est_model_ADH_test_RMSE
  }
}

## Computing means and standard errors
MCPanel_avg_RMSE <- apply(MCPanel_RMSE_test,2,mean)
MCPanel_std_error <- apply(MCPanel_RMSE_test,2,sd)/sqrt(num_runs)

EN_avg_RMSE <- apply(EN_RMSE_test,2,mean)
EN_std_error <- apply(EN_RMSE_test,2,sd)/sqrt(num_runs)

ENT_avg_RMSE <- apply(ENT_RMSE_test,2,mean)
ENT_std_error <- apply(ENT_RMSE_test,2,sd)/sqrt(num_runs)

DID_avg_RMSE <- apply(DID_RMSE_test,2,mean)
DID_std_error <- apply(DID_RMSE_test,2,sd)/sqrt(num_runs)

ADH_avg_RMSE <- apply(ADH_RMSE_test,2,mean)
ADH_std_error <- apply(ADH_RMSE_test,2,sd)/sqrt(num_runs)

## Creating plots

df1 <-
    structure(
      list(
      y =  c(DID_avg_RMSE, EN_avg_RMSE, ENT_avg_RMSE, MCPanel_avg_RMSE, ADH_avg_RMSE),
      lb = c(DID_avg_RMSE - 1.96*DID_std_error, EN_avg_RMSE - 1.96*EN_std_error,
             ENT_avg_RMSE - 1.96*ENT_std_error, MCPanel_avg_RMSE - 1.96*MCPanel_std_error,
             ADH_avg_RMSE - 1.96*ADH_std_error),
      ub = c(DID_avg_RMSE + 1.96*DID_std_error, EN_avg_RMSE + 1.96*EN_std_error,
             ENT_avg_RMSE + 1.96*ENT_std_error, MCPanel_avg_RMSE + 1.96*MCPanel_std_error,
             ADH_avg_RMSE + 1.96*ADH_std_error),
      x = c(T0/T, T0/T ,T0/T, T0/T, T0/T),
      Method = c(replicate(length(T0),"DID"), replicate(length(T0),"EN"),
                 replicate(length(T0),"EN-T"), replicate(length(T0),"MC-NNM"),
                 replicate(length(T0),"SC-ADH")),
      Marker = c(replicate(length(T0),1), replicate(length(T0),2),
                 replicate(length(T0),3), replicate(length(T0),4),
                 replicate(length(T0),5))

    ),
    .Names = c("y", "lb", "ub", "x", "Method", "Marker"),
    row.names = c(NA,-25L),
    class = "data.frame"
  )

Marker = c(1,2,3,4,5)


p = ggplot(data = df1, aes(x, y, color = Method, shape = Marker)) +
  geom_point(size = 2, position=position_dodge(width=0.1)) +
  geom_errorbar(
    aes(ymin = lb, ymax = ub),
    width = 0.1,
    linetype = "solid",
    position=position_dodge(width=0.1)) +
  scale_shape_identity() +
  guides(color = guide_legend(override.aes = list(shape = Marker))) +
  theme_bw() +
  xlab(TeX('$T_0/T$')) +
  ylab("Average RMSE") +
  coord_cartesian(ylim=c(5, 50))

print(p)

##
if(to_save == 1){
  filename<-paste0(paste0(paste0(paste0(paste0(paste0("california_data_N_", N),"_T_", T),"_numruns_", num_runs), "_num_treated_", N_t), "_simultaneuous_", is_simul),".png")
  ggsave(filename, plot = last_plot(), device="png", dpi=600)
  df2<-data.frame(N,T,N_t,is_simul, DID_RMSE_test, EN_RMSE_test, ENT_RMSE_test, MCPanel_RMSE_test, ADH_RMSE_test)
  colnames(df2)<-c("N", "T", "N_t", "is_simul", replicate(length(T0), "DID"),
                   replicate(length(T0), "EN"), replicate(length(T0), "ENT"),
                   replicate(length(T0), "MC-NNM"), replicate(length(T0),"SC-ADH"))

  filename<-paste0(paste0(paste0(paste0(paste0(paste0("california_data_N_", N),"_T_", T),"_numruns_", num_runs), "_num_treated_", N_t), "_simultaneuous_", is_simul),".rds")
  save(df1, df2, file = filename)
}

