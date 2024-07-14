# -*- coding: utf-8 -*-
# @author: Dasheng Fan
# @email: fandasheng1999@163.com

import numpy as np
from NN_tool import var_diag_calculator
from NN_tool import train_batch_phi, ce_batch_predict
from tools import log_prior_pdf_list_generator, log_prior_generator, calc_Levenberg_Marquardt_hessian, \
corr_to_cov, cov_to_corr
from sample_generator import SampleGenerator

import warnings
warnings.filterwarnings('ignore')

def MPC_LM(retrieval_model, designs, N, M, priors_parameter, eps, noise_std = 0.01, noisecutrange = 0.05,
           n_layer = 2, hidden_dim = 128, intrested_index = [0,1]):

    SG = SampleGenerator(retrieval_model, designs, priors_parameter, noise_std, noisecutrange)
    para_samples_N, reflectance_samples_N = SG.generate(N)
    phis, sc_ref, sc_paras = train_batch_phi(reflectance_samples_N, para_samples_N, 
                        n_layer=n_layer, hidden_dim=hidden_dim, intrested_index = intrested_index)
    para_samples_M, reflectance_samples_M = SG.generate(M)
    ce = ce_batch_predict(reflectance_samples_M, phis=phis, sc_ref=sc_ref, sc_paras=sc_paras)
    mpv_diag = var_diag_calculator(ce, para_samples_M[:, intrested_index])
    posterior_covariance_matrices = []
    prior_pdfs = log_prior_pdf_list_generator(priors_parameter)
    prior = log_prior_generator(prior_pdfs)
    for i in range(M):
        res = para_samples_M[i, :]
        covariance_matrix = np.eye(SG.N_observations) * (noise_std**2)
        hessian_m = np.linalg.inv(
            calc_Levenberg_Marquardt_hessian(retrieval_model, 
                                             covariance_matrix, 
                                             res, eps, 
                                             SG.N_sensor, 
                                             designs, 
                                             SG.N_observations, 
                                             prior))
        if np.isnan(hessian_m).any():
            continue
        posterior_covariance_matrices.append(hessian_m)
        
    MPC_init = np.mean(np.array(posterior_covariance_matrices), axis = 0)
    vars = np.diag(MPC_init).copy()
    vars[intrested_index] = mpv_diag
    corr_matrix = cov_to_corr(MPC_init)
    MPC = corr_to_cov(corr_matrix, vars)
    return MPC