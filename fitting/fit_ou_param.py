"""
Module that include algos to fit parameter for an Ornstein-Uhlenbeck process with no jump.
"""

import numpy as np
import statsmodels.api as sm
from scipy.optimize import minimize
from typing import Tuple

def fit_lbfgsb(data: np.ndarray, dt: float = 1/98532, **kwargs) -> Tuple[float]:
    """
    Estimate params of an de-meaned OU process with mean at 0 using L-BFGS-B.

    This is estimating the params by max likelihood. It is relatively slow. Also we assume the OU process is already
    de-meaned and it has long term average 0.

    :param data: (np.ndarray) The OU process data to be fitted.
    :param dt: (float) Optional.  Defaults to 1 min:
        1/98532 = 1/252/391.
    :param **kwarg: Other kwargs for the scipy.optimize.minimize() method.
    :return: (Tuple[float]) The fitted theta (mean reverting speed) and sigma (Brownian std).
    """

    bnds = ((1e-3, None), (1e-3, None))  # Bounds for theta and sigma.
    res = minimize(fun=_lbfgsb_log_lh_neg, x0=(0.15, 0.5), args=(data, dt),
                   method='L-BFGS-B', bounds=bnds, **kwargs)

    theta = res.x[0]
    sigma = res.x[1]

    return theta, sigma

def _lbfgsb_log_lh_neg(theta_sigma: Tuple[float], data: np.ndarray, dt: float) -> float:
    """
    Form the negative log-likelihood function for the L-BFGS-B fitting method.

    Assume the OU process has mean 0.

    :param theta_sigma: (Tuple[float]) the theta (mean-reverting speed) and sigma (Brownian std) for the OU process.
    :param data: (np.ndarray) The OU process data to be fitted.
    :param dt: (float) The time difference between each data point in the unit of year.
    :return: (float) The negative of max likelihood for the data.
    """

    theta = theta_sigma[0]
    sigma = theta_sigma[1]
    N = len(data)
    sigma_tilde = sigma * np.sqrt((1 - np.exp(-2 * theta * dt)) / (2 * theta))
    sum_part = sum([(data[t+1] - data[t]*np.exp(-theta*dt)) ** 2 for t in range(1, N-1)])
    log_likelihood = -N*np.log(2*np.pi)/2 - N*np.log(sigma_tilde) - 1/(2*sigma_tilde*sigma_tilde)*sum_part

    return -log_likelihood

def fit_ar1(data: np.ndarray, dt: float = 1/98532) -> Tuple[float]:
    """
    Estimate params of an OU process from AR(1).

    Assume the OU process has mean 0. This tends to overestimate but is very quick.

    :param data: (np.ndarray) The OU process data to be fitted.
    :param dt: (float) Optional.  Defaults to 1 min:
        1/98532 = 1/252/391.
    :return: (Tuple[float]) The fitted theta (mean reverting speed) and sigma (Brownian std).
    """

    # Create the lag for regression
    lag_0 = sm.tools.add_constant(data[1:])  # lag 0 data: not including the start
    lag_1 = data[:-1]  # lag 1 data: not including the end

    ols_results = sm.OLS(lag_1, lag_0).fit()

    theta = (1 - ols_results.params[1]) / dt  # mean-reverting speed
    sigma = np.sqrt(ols_results.mse_resid / dt)  # std for the Brownian term

    return theta, sigma

def fit_ar1_ml(data: np.ndarray, dt: float = 1/98532) -> Tuple[float]:
    """
    Estimate params of an OU process from AR(1).

    Assume the OU process has mean 0. This tends to overestimate but is very quick. It is essentially the same as
    the fit_ar1 method, however theta is estimated by the analytical solution of the maximum likelihood. The result
    is very simular to the fit_ar1 method in general.

    :param data: (np.ndarray) The OU process data to be fitted.
    :param dt: (float) Optional.  Defaults to 1 min:
        1/98532 = 1/252/391.
    :return: (Tuple[float]) The fitted theta (mean reverting speed) and sigma (Brownian std).
    """

    # Create the lag for regression
    lag_0 = sm.tools.add_constant(data[1:])  # lag 0 data: not including the start
    lag_1 = data[:-1]  # lag 1 data: not including the end

    ols_results = sm.OLS(lag_1, lag_0).fit()

    theta = -np.log(ols_results.params[1]) / dt  # mean-reverting speed
    sigma = np.sqrt(ols_results.mse_resid / dt)  # std for the Brownian term

    return theta, sigma


#%% Test for LBFGSB
if __name__ == '__main__':
    from time_series_mc.ou_sim import OUSimulator
    import matplotlib.pyplot as plt

    # Using synthetic data
    theta = 4  # True theta
    sigma = 0.5  # True sigma
    n_paths = 200
    n_points = 1000
    mu = 0
    dt = 1/252

    np.random.seed(10)

    ousim = OUSimulator()
    paths = ousim.simulate_analytical_grid(theta=theta, mu=mu, sigma=sigma, n_paths=n_paths, n_points=n_points,
                                           re_normalize = True, init_condition=0, dt=dt)

    # plot for sanity check
    # plt.figure(dpi=200)
    # for i in range(10):
    #     plt.plot(paths[:,i])
    # plt.title("Discrete Simulated OU Process")
    # plt.show()

    # Fit the OU params
    params = []
    for i in range(n_paths):
        if i % 20 == 0:
            print(i)
        fit_res = fit_lbfgsb(data=paths[:, i], dt=dt)
        params.append(fit_res)
    print(np.mean(params, axis=0))


#%% Generate the distribution of thetas
if __name__ == '__main__':
    thetas = [param[0] for param in params]
    fig, ax = plt.subplots(dpi=150)
    ax.hist(thetas, density=True, bins=20)
    plt.show()

#%% Test for AR1 fit
if __name__ == '__main__':
    from time_series_mc.ou_sim import OUSimulator
    import matplotlib.pyplot as plt
    # Using synthetic data
    theta = 5  # True theta
    sigma = 0.5  # True sigma
    n_paths = 40000
    n_points = 1000
    mu = 0
    dt = 1/252

    ousim = OUSimulator()
    paths = ousim.simulate_analytical_grid(theta=theta, mu=mu, sigma=sigma, n_paths=n_paths, n_points=n_points,
                                           re_normalize = True, init_condition=0, dt=dt)

    # Fit the OU params
    params = []
    for i in range(n_paths):
        if i % 2000 == 0:
            print(i)
        fit_res = fit_ar1(data=paths[:, i], dt=dt)
        params.append(fit_res)
    print(np.mean(params, axis=0))

#%% Generate the distribution of thetas
if __name__ == '__main__':
    thetas = [param[0] for param in params]
    fig, ax = plt.subplots(dpi=150)
    ax.hist(thetas, density=True, bins=100)
    plt.show()

#%% Test for AR1 ml fit
if __name__ == '__main__':
    from time_series_mc.ou_sim import OUSimulator
    import matplotlib.pyplot as plt
    # Using synthetic data
    theta = 5  # True theta
    sigma = 0.5  # True sigma
    n_paths = 40000
    n_points = 1000
    mu = 0
    dt = 1/252

    ousim = OUSimulator()
    paths = ousim.simulate_analytical_grid(theta=theta, mu=mu, sigma=sigma, n_paths=n_paths, n_points=n_points,
                                           re_normalize = True, init_condition=0, dt=dt)

    # Fit the OU params
    paramsml = []
    for i in range(n_paths):
        if i % 2000 == 0:
            print(i)
        fit_res = fit_ar1_ml(data=paths[:, i], dt=dt)
        paramsml.append(fit_res)
    print(np.mean(paramsml, axis=0))

#%% Generate the distribution of thetas
if __name__ == '__main__':
    thetasml = [param[0] for param in params]
    fig, ax = plt.subplots(dpi=150)
    ax.hist(thetas, density=True, bins=100, label='AR1', alpha=0.5)
    ax.hist(thetasml, density=True, bins=100, label='AR1 ML', alpha=0.5)
    ax.legend()
    plt.show()
