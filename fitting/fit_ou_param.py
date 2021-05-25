"""
Module that include algos to fit parameter for an Ornstein-Uhlenbeck process with no jump.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple

class FitOUParam:
     
    def fit_lbfgsb(self, data: np.ndarray, delta_t: float = 1/98532, **kwargs):
        """
        Estimate params of an OU process with mean at 0.
        """
        
        bnds = ((1e-3, None), (1e-3, None))  # Bounds for theta and sigma.
        # bnds = ((1e-3, None),)
        res = minimize(fun=self._lbfgsb_log_lh_neg, x0=(0.15, 0.5), args=(data, delta_t),
                       method='L-BFGS-B', bounds=bnds)
        
        return res
    
    def _lbfgsb_log_lh_neg2(self, theta: float, sigma: float, data: np.ndarray, delta_t: float):
        """
        Form the negative log-likelihood function for the L-BFGS-B fitting method.
        """
        
        N = len(data)
        sigma_tilde = sigma * np.sqrt((1 - np.exp(-2 * theta * delta_t)) / (2 * theta))
        sum_part = sum([(data[t+1] - data[t]*np.exp(-theta*delta_t)) ** 2 for t in range(1, N-1)])
        log_likelihood = -N*np.log(2*np.pi)/2 - N*np.log(sigma_tilde) - 1/(2*sigma_tilde*sigma_tilde)*sum_part

        return -log_likelihood

    def _lbfgsb_log_lh_neg(self, theta_sigma: Tuple[float], data: np.ndarray, delta_t: float):
        """
        Form the negative log-likelihood function for the L-BFGS-B fitting method.
        """
        
        theta = theta_sigma[0]
        sigma = theta_sigma[1]
        N = len(data)
        sigma_tilde = sigma * np.sqrt((1 - np.exp(-2 * theta * delta_t)) / (2 * theta))
        sum_part = sum([(data[t+1] - data[t]*np.exp(-theta*delta_t)) ** 2 for t in range(1, N-1)])
        log_likelihood = -N*np.log(2*np.pi)/2 - N*np.log(sigma_tilde) - 1/(2*sigma_tilde*sigma_tilde)*sum_part

        return -log_likelihood
    
#%% Test
if __name__ == '__main__':
    from time_series_mc.ou_sim import OUSimulator
    import matplotlib.pyplot as plt
    
    # Using synthetic data
    theta = 4
    sigma = 0.5
    n_paths = 20
    n_points = 391 * 30
    mu = 0
    dt = 1/252/10
    
    np.random.seed(10)
    
    ousim = OUSimulator()
    paths = ousim.simulate_analytical_grid(theta=theta, mu=mu, sigma=sigma, n_paths=n_paths, n_points=n_points,
                                           re_normalize = True, init_condition=0, dt=dt)
    
    # plot for sanity check
    plt.figure(dpi=200)
    for i in range(10):
        plt.plot(paths[:,i])
    plt.title("Discrete Simulated OU Process")
    plt.show()
    
    # Fit the OU params
    oufit = FitOUParam()
    params = []
    for i in range(20):
        fit_res = oufit.fit_lbfgsb(data=paths[:, i], delta_t = dt)
        params.append(fit_res.x)
        
#%%
print(np.mean(params, axis=0))
