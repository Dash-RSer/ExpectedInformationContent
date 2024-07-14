# -*- coding: utf-8 -*-
# @author: Dasheng Fan
# @email: fandasheng1999@163.com

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np

def train_phi(ref, paras, n_layer = 2, hidden_dim = 96):
    """
    Train sklearn neural network model to find CE.
    from domain of x -> domain of y.
    """
    if len(np.array(ref).shape) == 1 : ref = ref.reshape(-1, 1)
    if len(np.array(paras).shape) == 1 : paras = paras.reshape(-1, 1)
    hidden_layer = []
    for _ in range(n_layer): hidden_layer.append(hidden_dim)
    SC_ref = StandardScaler().fit(ref)
    SC_paras = StandardScaler().fit(paras)
    ref_norm = SC_ref.transform(ref)
    paras_norm = SC_paras.transform(paras)
    nn = MLPRegressor(hidden_layer, activation='relu', max_iter = 50000)
    nn.fit(ref_norm, paras_norm)
    
    return nn, SC_ref, SC_paras

def ce_predict(ref, phi, sc_ref, sc_paras):
    if len(np.array(ref).shape) == 1 : ref = ref.reshape(-1, 1)
    ref = sc_ref.transform(ref)
    paras_nn = phi.predict(ref)
    if len(np.array(paras_nn).shape) == 1 : paras_nn = paras_nn.reshape(-1, 1)
    paras_pred = sc_paras.inverse_transform(paras_nn)
    return paras_pred

def var_diag_calculator(pred, real):
    """
    Both should be [N_samples, N_paras].
    return mean posterior variance of each parameters. (a vector)
    """
    N_paras = pred.shape[1]
    var_vector = np.zeros(N_paras, dtype = np.float32)
    for i in range(N_paras):
        var_vector[i] = np.mean((pred[:, i] - real[:, i])**2)
    return var_vector

def train_batch_phi(ref, paras, n_layer = 2, hidden_dim = 96, intrested_index = [0]):
    "High accuaracy ce prediction."
    if len(np.array(ref).shape) == 1 : ref = ref.reshape(-1, 1)
    if len(np.array(paras).shape) == 1 : paras = paras.reshape(-1, 1)
    SC_ref = StandardScaler().fit(ref)
    ref_norm = SC_ref.transform(ref)
    hidden_layer = []
    for _ in range(n_layer): hidden_layer.append(hidden_dim)
    phis = []
    sc_paras = []
    for i in range(len(intrested_index)):
        cur_para = paras[:, intrested_index[i]].reshape(-1, 1)
        sc = StandardScaler().fit(cur_para)
        cur_paras_norm = sc.transform(cur_para)
        cur_nn = MLPRegressor(hidden_layer, activation='relu', max_iter = 20000, early_stopping = True)
        cur_nn.fit(ref_norm, cur_paras_norm)
        phis.append(cur_nn)
        sc_paras.append(sc)
    return phis, SC_ref, sc_paras

def ce_batch_predict(ref, phis, sc_ref, sc_paras):
    if len(np.array(ref).shape) == 1 : ref = ref.reshape(-1, 1)
    ref = sc_ref.transform(ref)
    paras_pred = np.zeros((ref.shape[0], len(phis)))
    for i in range(len(phis)):
        paras_nn = phis[i].predict(ref)
        if len(np.array(paras_nn).shape) == 1 : paras_nn = paras_nn.reshape(-1, 1)
        phi_pred = sc_paras[i].inverse_transform(paras_nn)
        paras_pred[:, i] = phi_pred.flatten()
    return paras_pred

