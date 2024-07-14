import numpy as np
import scipy.stats as stats
import warnings
import math
warnings.filterwarnings('ignore')

def packed_truncated_normal(para_list, n_samples):
    """
    Pass
    """
    mean, sigma, lower, upper = para_list
    x = stats.truncnorm(
        (lower - mean) / sigma, 
        (upper - mean) / sigma,
        loc=mean, scale=sigma)
    samples = np.array(x.rvs([n_samples]), dtype = np.float32)
    return samples

def packed_uniform(para_list, n_samples):
    """
    Pass
    """
    lower, upper = para_list
    samples = np.random.uniform(lower, upper, n_samples)
    return samples

def uniform_pdf_generator(v_low, v_high):
    """
    uniform p.d.f. 1/(v_high - v_low).
    v_high should bigger than v_low.
    """
    def uniform_pdf(x):
        if x < v_low:
            return np.array(0, dtype = np.float32)
        if x >= v_high:
            return np.array(0, dtype = np.float32)
        else:
            return np.array(1/(v_high - v_low), dtype = np.float32)
    return uniform_pdf

def gaussian1d_pdf_generator(mean, variance):
    """
    Gaussian pdf for scalar.
    """
    def gaussian1d_pdf(x):
        constant = 1/np.math.sqrt(2*np.pi*variance)
        kernel = np.math.exp(-(x-mean)**2/(2*variance))
        return constant*kernel
    return gaussian1d_pdf

def concatenate_array(array, N_observation):
    N = array.shape[0]
    concatenated_array = np.zeros(N_observation, dtype= np.float32)
    index = 0
    for i in range(N):
        sub_array = array[i]
        n_elements = sub_array.shape[0]
        concatenated_array[index:index+n_elements] = sub_array
        index = index+n_elements
    return concatenated_array

def calc_hessian(func, point = [0, 0], eps = [0.001, 0.001]):
    """
    Hessian matrix of func at expendung point.
    """
    n_var = len(point)
    def grad_func_generator(func):
        def gradient_func(point):
            gradient = np.zeros(n_var, np.float32)
            # nth gradient
            for i in range(n_var):
                # 初始化左点和右点，同时不改变原来的展开点
                left_point = point.copy()
                right_point = point.copy()
                left_point[i] = point[i] - eps[i]
                right_point[i] = point[i] + eps[i]
                gradient[i] = (func(right_point) - func(left_point))/(2*eps[i])
            return gradient
        return gradient_func
    grad_func = grad_func_generator(func)
    hessian_matrix = np.zeros((n_var, n_var), np.float64)
    for i in range(n_var):
        for j in range(n_var):
            # 第一项
            left_point_j = point.copy()
            right_point_j = point.copy()
            right_point_j[j] = point[j] + eps[j]
            left_point_j[j] = point[j] - eps[j]
            diff_i = (grad_func(right_point_j)[i] - grad_func(left_point_j)[i])/(4*eps[j])
            # 第二项
            left_point_i = point.copy()
            right_point_i = point.copy()
            right_point_i[i] = point[i] + eps[i]
            left_point_i[i] = point[i] - eps[i]
            diff_j = (grad_func(right_point_i)[j] - grad_func(left_point_i)[j])/(4*eps[i])

            hessian_matrix[i, j] = diff_i + diff_j
    return hessian_matrix

def calc_Levenberg_Marquardt_hessian(RTM_model, covariance_matrix,
                        point, eps, N_sensor, designs, N_observations, prior):
    n_var = len(point)
    def func(point):
        y_real = []
        for j in range(N_sensor):
            y_real.append(RTM_model(point, designs[j]))
        return concatenate_array(np.array(y_real), N_observations)

    Jacobian = np.zeros((N_observations, n_var), dtype = np.float32)
    for i in range(n_var):
        left_point = point.copy()
        right_point = point.copy()
        left_point[i] = point[i] - eps[i]
        right_point[i] = point[i] + eps[i]
        Jacobian[:,i] = ((func(right_point) - func(left_point))/(2*eps[i])).flatten()
    hessian = Jacobian.T@np.linalg.inv(covariance_matrix)@Jacobian - calc_hessian(prior, point, eps)
    return hessian

def build_covariance(n_obversation, noise_std_dev):
    covariance_matrix = \
        np.zeros((n_obversation, n_obversation), dtype = np.float32)
    for i in range(n_obversation):
        covariance_matrix[i, i] = noise_std_dev[i]**2
    return covariance_matrix

def build_covariance_ratio(y_real, ratio = 0.1):
    n_ob = len(y_real)
    covariance_matrix = np.zeros((n_ob, n_ob), dtype = np.float32)
    for i in range(n_ob):
        covariance_matrix[i,i] = y_real[i]*ratio
    return covariance_matrix

def cov_to_corr(cov_matrix):
    d = np.sqrt(np.diag(cov_matrix))
    d_inv = np.linalg.inv(np.diag(d))
    corr_matrix = d_inv @ cov_matrix @ d_inv
    return corr_matrix

def corr_to_cov(corr_matrix, variances):
    
    std_devs = np.sqrt(variances)
    std_devs_outer = np.outer(std_devs, std_devs)
    cov_matrix = corr_matrix * std_devs_outer
    return cov_matrix

def gaussian_entropy(variance):
    g_entropy = (1/2)*(np.math.log(2*np.pi*variance)+1)
    return g_entropy

def multivar_gaussian_entropy(covariance_matrix):
    if len(covariance_matrix.shape) != 2:
        raise Exception("dimensions if covariance matrix is wrong.")
    d = covariance_matrix.shape[0]
    entropy = (d/2)*(np.log(2*np.pi)+1)+0.5*np.log(np.linalg.det(covariance_matrix))
    return entropy

def uniform_entropy(v_low, v_high):
    u_entropy = np.math.log((v_high - v_low))
    return u_entropy

def calibration_LNC(d, k, N= 500000, epsilon = 0.005):
    points = np.random.uniform(0, 1, (N, k, d))
    ratios = []
    for i in range(N):
        cur_points = points[i, :, :]
        means = np.median(cur_points, axis = 0)
        submean_points = cur_points - means
        covr = submean_points.T @ submean_points / (k)
        _, v = np.linalg.eig(covr)
        V_rect = np.abs(submean_points @ v).max(axis = 0).prod()
        alpha = V_rect*(2**d)
        ratios.append(alpha)
    ratios.sort()
    epsilon_N_alpha = ratios[np.int_(epsilon * N)]
    return epsilon_N_alpha

def posterior_pdf_generator(likelihood, priors):
    """
    likelihood: likelihood function built by likelihood generator.
    priors: a list, elements of the list is prior pdf of each parameter.
    for example: [prior_of_lai, prior_of_cab]
    """
    def posterior(parameters):
        prior_value = 1
        for i in range(len(priors)):
            # 第i个参数传入第i个先验分布pdf中计算概率
            prior_value = prior_value*priors[i](parameters[i])
        likelihood_value = likelihood(parameters)
        posterior_value = likelihood_value*prior_value
        return posterior_value
    return posterior



def log_posterior_pdf_generator(log_likelihood, priors):
    """
    likelihood: likelihood function built by likelihood generator.
    priors: a list, elements of the list is prior pdf of each parameter.
    for example: [prior_of_lai, prior_of_cab]
    """
    def posterior(parameters):
        prior_value = 1
        for i in range(len(priors)):
            # 第i个参数传入第i个先验分布pdf中计算概率
            prior_value = prior_value*priors[i](parameters[i])
        log_likelihood_value = log_likelihood(parameters)
        log_posterior_value = log_likelihood_value + np.log(prior_value)
        return log_posterior_value
    return posterior

def log_prior_generator(prior_list):
    def log_prior(parameters):
        log_prior_value = 0
        for i in range(len(prior_list)):
            log_prior_value = log_prior_value + np.log(prior_list[i](parameters[i]))
        return log_prior_value
    return log_prior

def noise_generator(dimensions, sigma, cut):
    return packed_truncated_normal([0, sigma, -cut, cut], dimensions)

def prior_generator(parameters):
    """
    parameters: [ 'g', mean, sigma, lower, upper] for gaussian density.
                ['u', lower_bound, upper_bound] for uniform density.
    """
    prior_type = parameters[0]
    if prior_type == 'tg':
        mean = parameters[1]
        sigma = parameters[2]
        lower_bound = parameters[3]
        upper_bound = parameters[4]
        return truncated_normal_density_generator(mean, sigma, lower_bound, upper_bound)
    elif prior_type == 'g':
        mean = parameters[1]
        sigma = parameters[2]
        return gaussian1d_pdf_generator(mean, sigma**2)
    elif prior_type == 'u':
        lower_bound = parameters[1]
        upper_bound = parameters[2]
        return uniform_pdf_generator(lower_bound, upper_bound)
    else:
        raise Exception("undefined prior type")

def log_prior_pdf_list_generator(prior_list):
    n_prior = len(prior_list)
    prior_pdfs = []
    for i in range(n_prior):
        prior_pdfs.append(prior_generator(prior_list[i]))
    return prior_pdfs

def samples_generator(parameters, n):
    """
    parameters: ['g', mean, sigma, lower, upper] for truncated gaussian density.
                ['u', lower_bound, upper_bound] for uniform density.
    """
    prior_type = parameters[0]
    if prior_type == 'g' or prior_type == 'tg':
        mean = parameters[1]
        sigma = parameters[2]
        lower = parameters[3]
        upper = parameters[4]
        return packed_truncated_normal([mean, sigma, lower, upper], n)
    elif prior_type == 'u':
        lower = parameters[1]
        upper = parameters[2]
        return np.random.uniform(lower, upper, n)
    else:
        raise Exception("undefined prior type")

def standard_gaussian_density(x):
    return (1/np.sqrt(2*np.pi))*np.exp(-0.5*(x**2))

def standard_gaussian_cdf(x):
    return 0.5*(1+math.erf(x/np.sqrt(2)))

def truncated_normal_density_generator(mu, sigma, a, b):
    def truncated_normal_density(x):
        if x >= a and x <= b:
            return (1/sigma)*(standard_gaussian_density((x-mu)/sigma))\
                /((standard_gaussian_cdf((b-mu)/sigma))-(standard_gaussian_cdf((a-mu)/sigma)))
        else:
            return 0
    return truncated_normal_density

def truncated_normal_entropy(mu,sigma,a,b):
    alpha = (a-mu)/sigma
    beta = (b-mu)/sigma
    Z = standard_gaussian_cdf(beta) - standard_gaussian_cdf(alpha)
    return np.log(np.sqrt(2*np.pi*np.e)*sigma*Z) +\
         (alpha*standard_gaussian_density(alpha)-beta*standard_gaussian_density(beta))/(2*Z)

def truncated_normal_variance(mu,sigma,a,b):
    alpha = (a-mu)/sigma
    beta = (b-mu)/sigma
    Z = standard_gaussian_cdf(beta) - standard_gaussian_cdf(alpha)
    p1 = 1
    p2 = (beta*standard_gaussian_density(beta)-alpha*standard_gaussian_density(alpha))/(Z)
    p3 = ((standard_gaussian_density(alpha) - standard_gaussian_density(beta))/(Z))**2
    return (sigma**2)*(p1-p2-p3)

def uniform_variance(a, b):
    return (1/12)*((b-a)**2)
def noise_generator_ratio(y, ratio, dimension, cut = 0.05):
    noise = np.zeros(dimension, dtype = np.float32)
    for i in range(dimension):
        noise[i] = packed_truncated_normal([0, y[i]*ratio, -cut, cut], 1)
    return noise

if __name__ == "__main__":
    print(calibration_LNC(5, 10, 500000, 0.005))