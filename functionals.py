# -*- coding: utf-8 -*-
# @author: Dasheng Fan
# @email: fandasheng1999@163.com

import numpy as np
from tools import gaussian_entropy, uniform_entropy, truncated_normal_entropy, truncated_normal_variance, \
                uniform_variance

def calc_entropy(PCM, index = [0, 1]):
    """
    This function is used for calculation of the expecation of posterior
    entropy, thus the EIG.
    """
    N_paras = len(index)
    PCM = np.array(PCM, dtype=np.float32)
    entropy = 0.5*N_paras*(np.log(2*np.pi)+1) + 0.5*np.log(np.linalg.det(PCM[index,:][:, index]))
    return entropy

def calc_prior_entropy(prior_list):
    """
    calc_entropy for independent priors.
    """
    N_prior = len(prior_list)
    entropy = 0
    for i in range(N_prior):
        cur_prior_type = prior_list[i][0]
        if cur_prior_type == 'g':
                entropy = entropy + gaussian_entropy(prior_list[i][2]**2) # entropy of normal
                # entropy = entropy + truncated_normal_entropy(*prior_list[i][1:]) # entropy of truncated normal
        elif cur_prior_type == 'u':
            entropy = entropy + uniform_entropy(prior_list[i][1], prior_list[i][2])
        else:
            raise Exception("undefined prior type")
    return entropy

def calc_prior_normal_entropy(prior_list):
    """
    calc_det for independent priors.
    """
    N_prior = len(prior_list)
    det = 1
    for i in range(N_prior):
        cur_prior_type = prior_list[i][0]
        if cur_prior_type == 'g':
                det = det * truncated_normal_variance(*prior_list[i][1:])
        elif cur_prior_type == 'u':
            det = det*uniform_variance(prior_list[i][1], prior_list[i][2])
        else:
            raise Exception("undefined prior type")
    return 0.5*np.log(det) + 0.5*N_prior*np.log(2*np.pi*np.e)

def calc_poster_entropy(PCM, index = [0, 1]):
    """
    This function is used for calculation of the expecation of posterior
    entropy, thus the EIG.
    """
    N_paras = len(index)
    PCM = np.array(PCM, dtype=np.float32)
    entropy = 0.5*N_paras*(np.log(2*np.pi)+1) + 0.5*np.log(np.linalg.det(PCM[index,:][:, index]))
    return entropy