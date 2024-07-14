import numpy as np
from model import model_generator
from functionals import calc_entropy, calc_prior_normal_entropy
import matplotlib.pyplot as plt
from MPC_LM import MPC_LM

def band_experiment():
    retrieval_model = model_generator(['free', 'free', 8, 2, 0.01, 0.01, 60, 0.3, 0.2])
    # bands = np.arange(550, 1000, 50)
    bands = [550, 650]
    p_entropy = []
    cab_entropy = []
    lai_entropy = []
    MPCs = []
    for i in range(len(bands)):
        designs = [[20, 0, 0, [700, bands[i]]]]
        MPC= MPC_LM(
            retrieval_model = retrieval_model,
            designs = designs,
            priors_parameter = [['g', 2.5, 2, 0.5, 4.5], ['g', 30, 20, 5, 60]],
            N = 10000, M = 5000, eps = [0.01, 0.1],
            noise_std = 0.01, intrested_index = [0, 1])
        p_entropy.append(calc_prior_normal_entropy([['g', 2.5, 2, 0.5, 4.5], ['g', 30, 20, 5, 60]]) - calc_entropy(MPC, index = [0, 1]))
        cab_entropy.append(calc_prior_normal_entropy([['g', 30, 20, 5, 60]]) - calc_entropy(MPC, index = [1]))
        lai_entropy.append(calc_prior_normal_entropy([['g', 2.5, 2, 0.5, 4.5]]) - calc_entropy(MPC, index = [0]))
        MPCs.append(MPC)
        print('band = ', bands[i], 't_entropy = ', p_entropy[i], 'lai_entropy = ', lai_entropy[i], 'cab_entropy = ', cab_entropy[i])
    plt.plot(bands, p_entropy)
    plt.plot(bands, lai_entropy) 
    plt.plot(bands, cab_entropy)
    plt.show()
    print("CalcOver.")
    # 
    # t_1   array([0.87, 0.78, 0.58, 1.01, 1.89, 1.95, 1.96, 1.93, 1.86])
    # cab_1 array([0.72, 0.76, 0.47, 0.95, 0.03, 0.  , 0.01, 0.01, 0.01])
    # mse   [43.736,  41.745,  77.735,  29.277, 183.372, 191.245, 195.236,191.415, 192.836]
    # cab_2 array([1.03, 1.12, 1.06, 1.14, 1.01, 1.02, 1.02, 1.02, 1.02])
    # mse   [23.601, 21.868, 23.544, 19.56, 25.932, 25.513, 25.243, 25.432, 25.518]
    # lai_1 array([0.03, 0.02, 0.07, 0.01, 1.62, 1.95, 1.95, 1.93, 1.85])
    # mse   [1.098, 1.12 , 1.038, 1.119, 0.046, 0.024, 0.024, 0.026, 0.029]
    # lai_2 array([1.93, 1.94, 1.96, 1.96, 2.13, 2.28, 2.27, 2.25, 2.23])
    # mse   [0.024, 0.024, 0.023, 0.024, 0.017, 0.012, 0.013, 0.013, 0.014]
    # t_2   array([2.78, 2.74, 2.47, 2.95, 2.33, 2.27, 2.27, 2.22, 2.24])
    
def angle_experiment():
    retrieval_model = model_generator(['free', 30, 8, 2, 0.01, 'free', 60, 0.3, 0.2])
    angles = np.arange(-70, 80, 10)
    # angles = [-70, 70]
    p_EIGLB = []
    cm_EIGLB = []
    lai_EIGLB = []
    MPCs = []
    for i in range(len(angles)):
        if angles[i] <= 0 : tto, psi = -angles[i], 0
        else: tto, psi = angles[i], 180
        designs = [[20, 70, 180, [800]], [20, tto, psi, [800]]]
        prior_paras = [['g', 2.5, 1, 0.5, 4.5], ['g', 0.01, 0.01, 0.001, 0.02]]
        MPC= MPC_LM(
            retrieval_model = retrieval_model,
            designs = designs,  
            priors_parameter = prior_paras,
            N = 20000, M = 5000, eps = [0.1, 0.001],
            noise_std = 0.01, noisecutrange = 0.05)
        p_EIGLB.append(calc_prior_normal_entropy(prior_paras) - calc_entropy(MPC, index = [0, 1]))
        cm_EIGLB.append(calc_prior_normal_entropy([prior_paras[1]]) - calc_entropy(MPC, index = [1]))
        lai_EIGLB.append(calc_prior_normal_entropy([prior_paras[0]]) - calc_entropy(MPC, index = [0]))
        MPCs.append(MPC)
        print('tto = ', angles[i], 't_EIGLB = ', p_EIGLB[i], 'lai_EIGLB = ', lai_EIGLB[i], 'cm_EIGLB = ', cm_EIGLB[i])
    plt.plot(angles, p_EIGLB)
    plt.plot(angles, lai_EIGLB) 
    plt.plot(angles, cm_EIGLB)
    plt.show()
    print("CalcOver.")
    # TTO     [-70,   -60,    -50,   -40,   -30,   -20,   -10,   0,     10,    20,   30,    40,    50,    60,    70]
    # T_1     array([1.16, 1.23, 1.29, 1.36, 1.38, 1.42, 1.41, 1.4 , 1.39, 1.37, 1.34,1.31, 1.26, 1.22, 1.17])
    # T_2     array([2.1 , 1.8 , 1.59, 1.52, 1.43, 1.49, 1.45, 1.46, 1.47, 1.48, 1.51, 1.57, 1.68, 1.84, 2.12])
    # LAI_1   array([0.44, 0.54, 0.6 , 0.68, 0.71, 0.77, 0.73, 0.71, 0.69, 0.66, 0.64,0.61, 0.56, 0.51, 0.43])
    # MSE     [0.32 , 0.258, 0.23 , 0.205, 0.187, 0.167, 0.179, 0.185, 0.198, 0.204, 0.216, 0.229, 0.246, 0.28 , 0.335]
    # LAI_20_2array([1.122, 0.973, 0.855, 0.804, 0.751, 0.796, 0.772, 0.766, 0.769,0.788, 0.778, 0.834, 0.912, 0.992, 1.137])
    # MSE     [0.0859, 0.113, 0.143, 0.159, 0.168, 0.163, 0.167, 0.166, 0.165, 0.163, 0.161, 0.145, 0.133, 0.106, 0.08 ]
    # Cm_1    array([0.274, 0.23 , 0.192, 0.171, 0.151, 0.137, 0.146, 0.161, 0.168, 0.178, 0.184, 0.188, 0.202, 0.238, 0.296])
    # MSE     [1.497e-05, 1.672e-05, 1.786e-05, 1.905e-05, 1.933e-05, 1.993e-05,1.982e-05, 1.97e-05, 1.937e-05, 1.916e-05, 1.890e-05, 1.825e-05, 1.757e-05, 1.658e-05, 1.467e-05]
    # Cm70_2  array([0.297, 0.358, 0.443, 0.517, 0.577, 0.637, 0.61 , 0.544, 0.518, 0.492, 0.461, 0.421, 0.391, 0.336, 0.302])
    # MSE     [1.502e-05, 1.328e-05, 1.118e-05, 9.580e-06, 8.540e-06, 7.580e-06,8.260e-06, 8.850e-06, 9.490e-06, 1.000e-05, 1.076e-05, 1.155e-05,1.267e-05, 1.444e-05, 1.489e-05]

def factors(experiments = "noise"):
    retrieval_model = model_generator(['free', 30, 8, 2, 0.01, 0.01, 60, 0.3, 0.2])
    designs = [[20, 0, 0, [800]]]
    eig_lbs = []
    
    def noise_experiment():
        noises = np.arange(0.01, 0.11, 0.01)
        for i in range(len(noises)):
            MPC= MPC_LM(
                retrieval_model = retrieval_model,
                designs = designs,
                priors_parameter = [['g', 2.5, 1, 0.1, 6]],
                N = 10000, M = 5000, eps = [0.01],
                noise_std = noises[i],
                noisecutrange = 0.1, intrested_index=[0])
            eig_lbs.append(calc_prior_normal_entropy([['g', 2.5, 1, 0.1, 6]]) - calc_entropy(MPC, index = [0]))
            print('stddev of noise = ', np.round(noises[i], 3), " EIG_S = ", eig_lbs[i])
    # noise  [0.01,   0.02,  0.03,  0.04,  0.05, 0.06,   0.07,  0.08,  0.09,  0.1 ]
    # EIGLB  [1.814, 1.217, 0.865, 0.69, 0.608, 0.537, 0.518, 0.491, 0.491, 0.464]

    def sensitivity_experiment():
        means = np.arange(1, 6.5, 0.5)
        for i in range(len(means)):
            MPC= MPC_LM(
                retrieval_model = retrieval_model,
                designs = designs,
                priors_parameter = [['g', means[i], 1, 0.1, 6]],
                N = 10000, M = 5000, eps = [0.01],
                noise_std = 0.01,
                noisecutrange = 0.1, intrested_index=[0])
            eig_lbs.append(calc_prior_normal_entropy([['g', means[i], 1, 0.1, 6]]) - calc_entropy(MPC, index = [0]))
            print('mean of prior = ', means[i], " EIG_LB = ", eig_lbs[i])
    # means [1. ,   1.5,    2. ,   2.5,    3. ,  3.5,    4. ,  4.5,    5. ,  5.5,    6. ]
    # EIGLB [2.291, 2.145, 1.988, 1.842, 1.679, 1.514, 1.379, 1.274, 1.218, 1.197, 1.192]
    # array([2.012, 1.996, 1.927, 1.811, 1.673, 1.488, 1.327, 1.158, 0.986,0.837, 0.682])
    
    def priorvar_experiment():
        stds = np.arange(0.5, 2.1, 0.1)
        stds = [1.7, 1.8, 1.9]
        for i in range(len(stds)):
            MPC= MPC_LM(
                retrieval_model = retrieval_model,
                designs = designs,
                priors_parameter = [['g', 2.5, stds[i], 0.1, 6]],
                N = 10000, M = 5000, eps = [0.01],
                noise_std = 0.01,
                noisecutrange = 0.1, intrested_index=[0])
            eig_lbs.append(calc_prior_normal_entropy([['g', 2.5, stds[i], 0.1, 6]]) - calc_entropy(MPC, index = [0]))
            print('stddev of prior = ', stds[i], " EIG_LB = ", eig_lbs[i])
    
    # stds   [0.5,   0.6,   0.7,   0.8,   0.9,   1. ,   1.1,   1.2,  1.3,   1.4,   1.5,    1.6,    1.7,   1.8,   1.9,    2. ]
    # EIG_LB array([1.274, 1.461, 1.588, 1.678, 1.748, 1.849, 1.869, 1.923, 1.927, 1.95 , 1.973, 2.014, 1.991, 2.005, 2.041, 1.996])

    if experiments == 'noise':
        noise_experiment()
    elif experiments == 'sensitivity':
        sensitivity_experiment()
    elif experiments == 'priorvar':
        priorvar_experiment()
    else:
        raise NotImplementedError("No such an experiment.")
    
    print("CalcOver.")

def sensor_experiment(sensor = 'GF1'):
    """
    LB = 1.06  GF1 = 0.96  L8 = 1.15  S2 = 1.26  QB = 0.59  misr = 1.49
    """
    designs = {'landsat8' : [[30, 0, 0,  [482, 561, 655, 865, 1609, 2201]]], 
               'sentinel2' : [[30, 0, 0, [490, 560, 665, 705, 740, 783, 842, 865, 1610, 2190]]],
               'quickbird' : [[30, 0, 0, [430, 466, 590, 715]]],
               'MISR':[[30, 0, 0, [446, 558, 672, 867]],[30, 26.1, 0, [446, 558, 672, 867]],[30, 26.1, 180, [446, 558, 672, 867]],\
                        [30, 45.6, 0, [446, 558, 672, 867]],[30, 45.6, 180, [446, 558, 672, 867]],[30, 60, 0, [446, 558, 672, 867]],\
                        [30, 60, 180, [446, 558, 672, 867]],[30, 70.5, 0, [446, 558, 672, 867]],[30, 70.5, 180, [446, 558, 672, 867]]], 
               'GF1':[[30, 0, 0, [485, 555, 660, 830]]]}
    retrieval_model = model_generator(['free', 'free', 'free', 'free', 'free', 'free', 'free', 'free', 'free'])
    MPC = MPC_LM(designs = designs[sensor],
                    retrieval_model = retrieval_model,
                    priors_parameter = [['u', 0, 6], ['u', 20.1, 69.9], ['u', 0.01, 19.9], ['g', 2.27, np.sqrt(1.69), 1, 3], ['g', 0.0217, np.sqrt(0.0019), 0.001, 0.03],\
                    ['g', 0.0027, np.sqrt(0.0012), 0.001, 0.02], ['g', 75.83, np.sqrt(1.68), 30, 89.9],['g', 0.2978, np.sqrt(0.12), 0.01, 0.4], ['u', 0.01, 0.99]],
                    noise_std = 0.01, n_layer = 2, hidden_dim = 96,
                    eps = [0.01, 0.1, 0.1, 0.01, 0.0001, 0.0001, 0.1, 0.001, 0.001], 
                    N = 50000, M = 10000, 
                    noisecutrange = 0.05)
    eig_lb = calc_prior_normal_entropy([['u', 0, 6]]) - calc_entropy(MPC, index = [0])
    print('mse = ', MPC[0,0])
    print('EIG_LB = ', eig_lb)

if __name__ == '__main__' :
    sensor_experiment('quickbird')
    # band_experiment()
    # angle_experiment()
    # factors('priorvar')
    # pass
