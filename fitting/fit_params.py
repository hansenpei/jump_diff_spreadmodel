"""
API module for fitting the params of an OU process with or without jumps.
"""

import numpy as np
from fitting import fit_ou_param, fit_jump_param
from typing import Tuple

class FitParams:
    """
    Class for various parameter fitting algos for OU or Jump-OU model.
    """

    def __init__(self, data: np.ndarray = None):

        self.data = data

    def fit_jump_param(self, data: np.ndarray = None, thresholds: Tuple[float] = None, thres_in_quantile: bool = False,
                       number_of_years: float = None) -> Tuple[float]:
        """
        Fit the jump parameters, get the jump intensity, jump mean, jump std.

        The jump is modeled as a Poission process and the parameters are estimated accordingly. For the input data,
        only those above or below a given threshold will be considered a jump. The thresholds can be put in absolute
        terms (i.e., prices of stocks) or quantiles.
        """

        if data is None:
            data = self.data
        if thresholds is None:  # Default being every data point is considered a jump.
            thresholds = (float('inf'), -float('inf'))
        if thres_in_quantile:  # Translate the quantile thresholds in absolute terms
            lower_threshold = np.quantile(data, thresholds[0])
            upper_threshold = np.quantile(data, thresholds[1])
            thresholds = (lower_threshold, upper_threshold)

        jump_rate, jump_mean, jump_std = fit_jump_param.fit_jump_params(data, thresholds, number_of_years,
                                                                        return_jumps_data=False)

        return jump_rate, jump_mean, jump_std


    def fit_ou_lbfgsb(self, data: np.ndarray = None, mu: float = None, dt: float = 1/98532, **kwargs) -> dict:
        """
        Fit the OU-process by using L-BFGS-B.

        This algorithm requires the OU-process to be de-meaned. Generally the mean-reverting speed (theta) is not very
        accurate and under estimated, however the std (sigma) is reasonably accurate. It is also relatively slow.

        :param data: (np.ndarray) Optional. The data we use to fit in 1D array. If not provided, then we use the
            default data from the class.
        :param mu: (float) Optional. The mean of the OU process. Defaults to the data average.
        :param dt: (float) Optional. The time difference between each data point in the unit of year. Defaults to
            1/98532 = 1/252/391, which is 1 min.
        :param **kwargs: Optional. Additional keyword arguments for the L-BFGS-B method from scipy.optimize.minimize.
        :return: (dict) The fitted result stored in a dictionary.
        """

        # Handle default values
        data, mu, data_demean = self._handle_default_inputs(data, mu)

        # Conduct the fitting process using L-BFGS-B
        params = fit_ou_param.fit_lbfgsb(data=data_demean, dt=dt, **kwargs)
        # Put the result in a dictionary: mu is the long term mean of the OU process, theta is the mean-reverting
        # speed, sigma is the std of the Brownian part.
        res_dict = {'mu': mu, 'theta': params[0], 'sigma': params[1]}

        return res_dict

    def fit_ou_ar1(self, data: np.ndarray = None, mu: float = None, dt: float = 1/98532) -> dict:
        """
        Fit the OU-process by using the AR1 process.

        This algorithm requires the OU-process to be de-meaned. Generally the mean-reverting speed (theta) is not very
        accurate and overestimated, however the std (sigma) is reasonably accurate. It is also relatively fast.

        :param data: (np.ndarray) Optional. The data we use to fit in 1D array. If not provided, then we use the
            default data from the class.
        :param mu: (float) Optional. The mean of the OU process. Defaults to the data average.
        :param dt: (float) Optional. The time difference between each data point in the unit of year. Defaults to
            1/98532 = 1/252/391, which is 1 min.
        :param **kwargs: Optional. Additional keyword arguments for the L-BFGS-B method from scipy.optimize.minimize.
        :return: (dict) The fitted result stored in a dictionary.
        """

        # Handle default values
        data, mu, data_demean = self._handle_default_inputs(data, mu)

        # Conduct the fitting process using L-BFGS-B
        params = fit_ou_param.fit_ar1(data=data_demean, dt=dt)
        # Put the result in a dictionary: mu is the long term mean of the OU process, theta is the mean-reverting
        # speed, sigma is the std of the Brownian part.
        res_dict = {'mu': mu, 'theta': params[0], 'sigma': params[1]}

        return res_dict

    def _handle_default_inputs(self, data: np.ndarray, mu: float) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Handle the default inputs of OU-process fitting methods.
        """

        if data is None:
            data = self.data
        if mu is None:
            mu = np.mean(data)
        data_demean = data - mu

        return data, mu, data_demean


#%% Test fit_ou_lbfgsb
if __name__ == '__main__':
    from time_series_mc.ou_sim import OUSimulator
    import matplotlib.pyplot as plt

    # Using synthetic data
    theta = 4  # True theta
    sigma = 0.5  # True sigma
    n_paths = 20
    n_points = 252
    mu = 0
    dt = 1/252

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

    fp = FitParams()
    # Fit the OU params
    params = []
    for i in range(20):
        fit_res = fp.fit_ou_lbfgsb(data=paths[:, i], dt = dt)
        params.append([fit_res['theta'], fit_res['sigma']])

#%% Test fit_ou_ar1
if __name__ == '__main__':
    from time_series_mc.ou_sim import OUSimulator
    import matplotlib.pyplot as plt

    # Using synthetic data
    theta = 10  # True theta
    sigma = 0.5  # True sigma
    n_paths = 20
    n_points = 1000
    mu = 0
    dt = 1/252

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

    fp = FitParams()
    # Fit the OU params
    params = []
    for i in range(20):
        fit_res = fp.fit_ou_ar1(data=paths[:, i], dt = dt)
        params.append([fit_res['theta'], fit_res['sigma']])

    print(np.mean(params, axis=0))
