"""
Simulator class for 1D Ornstein-Uhlenbeck process: dX = theta * (mu - X) * dt + sigma * dW.
"""

import numpy as np
import util
from typing import Callable

class OUSimulator:
    """
    Class that simulates the Ornstein-Uhlenbeck process.
    """
    def __init__(self, seed: int = None):
        
        self.seed = seed
        
    def simulate_discrete_grid(self, theta: float = 0.5, mu: float = 0, sigma: float = 0.05, dt: float = 1/252,
                               n_points: int = 252, n_paths: int = 10, re_normalize: bool = True,
                               init_condition: float = 0)-> np.ndarray:
        """
        Discretely simulate the OU paths with given time interval and total number of points per path.
        
        Use Euler advancement to simulate the paths following the drifted OU model: 
            dX = theta * (mu - X) * dt + sigma * dW,
        where theta, mu, sigma are constants.
        
        :param theta: (float) Optional. Mean-reverting speed. Defaults to 0.5.
        :param mu: (float) Optional. Mean-reverting level. Defaults to 0.
        :param sigma: (float) Optional. Standard deviation for the Brownian motion. Defalts to 0.05.
        :param dt: (float) Optional. Time advancement step. Defaults to 1/252.
        :param n_points: (int) Optional. Number of steps per simulated path, including the initial condition. Defaults
            to 252.
        :param n_paths: (float) Optional. Number of paths in the simulation. Needs to be >=2. Defaults to 10.
        :param re_normalize: (bool) Optional. Whether to renormalize the Gaussians sampled at each time advancement.
            This will only be triggered if n_paths >= 2.
        :param init_condition: (float) Optional. Initial start position for every path. Defaults to 0.
        """
        
        # Initialize the timetable. In this case it is a grid.
        timetable = np.arange(start=0, stop=dt * n_points, step=dt)
        
        # Simulated paths according to the time table.
        simulated_paths = self.simulate_discrete_timetable(
            timetable=timetable, theta=theta, mu=mu, sigma=sigma, n_paths=n_paths, re_normalize=re_normalize,
            init_condition=init_condition)
        
        # Output the array.
        return simulated_paths

    def simulate_discrete_timetable(
            self, timetable: np.ndarray, theta: float = 0.5, mu: float = 0, sigma: float = 0.05, n_paths: int = 10,
            re_normalize: bool = True, init_condition: float = 0)-> np.ndarray:
        """
        Discretely simulate the OU paths with a given time table and total number of points per path.
        
        Use Euler advancement to simulate the paths following the drifted OU model: 
            dX = theta * (mu - X) * dt + sigma * dW,
        where theta, mu, sigma are constants.
        
        :param timetable: (np.ndarray) 1D numpy array that indicate the time of interest for calculation.
        :param theta: (float) Optional. Mean-reverting speed. Defaults to 0.5.
        :param mu: (float) Optional. Mean-reverting level. Defaults to 0.
        :param sigma: (float) Optional. Standard deviation for the Brownian motion. Defalts to 0.05.
        :param n_paths: (float) Optional. Number of paths in the simulation. Needs to be >=2. Defaults to 10.
        :param re_normalize: (bool) Optional. Whether to renormalize the Gaussians sampled at each time advancement.
            This will only be triggered if n_paths >= 2.
        :param init_condition: (float) Optional. Initial start position for every path. Defaults to 0.
        """
        
        # Initialize
        n_points = len(timetable)
        simulated_paths = np.zeros(shape=(n_points, n_paths))
        simulated_paths[0, :] += init_condition
        
        # 1. Draw from the Gaussian distribution accordingly.
        gaussians = np.random.normal(loc=0, scale=1, size=(n_points-1, n_paths))

        # 2. Re-normalize the Gaussian per point or instance (normalize every row)
        if re_normalize and n_paths >= 2:
            gaussians = util.normalize_data(data_matrix=gaussians, act_on='row')

        # 3. Construct the paths via Euler advancement.
        for i in range(1, n_points):
            dt = timetable[i] - timetable[i-1]
            sqdt = np.sqrt(dt)
            increments = theta * (mu - simulated_paths[i-1,:]) * dt + sigma * gaussians[i-1, :] * sqdt
            simulated_paths[i,:] = simulated_paths[i-1,:] + increments

        # 4. Output the array.
        return simulated_paths
    
    def simulate_analytic_grid(self, theta: float = 0.5, mu: float = 0, sigma: float = 0.05, dt: float = 1/252,
                               n_points: int = 252, n_paths: int = 10, re_normalize: bool = True,
                               init_condition: float = 0)-> np.ndarray:
        """
        Analytically simulate the OU paths with given time interval and total number of points per path.
        
        Use Euler advancement to simulate the paths following the drifted OU model: 
            dX = theta * (mu - X) * dt + sigma * dW,
        where theta, mu, sigma are constants.
        
        :param theta: (float) Optional. Mean-reverting speed. Defaults to 0.5.
        :param mu: (float) Optional. Mean-reverting level. Defaults to 0.
        :param sigma: (float) Optional. Standard deviation for the Brownian motion. Defalts to 0.05.
        :param dt: (float) Optional. Time advancement step. Defaults to 1/252.
        :param n_points: (int) Optional. Number of steps per simulated path, including the initial condition. Defaults
            to 252.
        :param n_paths: (float) Optional. Number of paths in the simulation. Needs to be >=2. Defaults to 10.
        :param re_normalize: (bool) Optional. Whether to renormalize the Gaussians sampled at each time advancement.
            This will only be triggered if n_paths >= 2.
        :param init_condition: (float) Optional. Initial start position for every path. Defaults to 0.
        """

        # Initialize the timetable. In this case it is a grid.
        timetable = np.arange(start=0, stop=dt * n_points, step=dt)

        # Simulated paths according to the time table.
        simulated_paths = self.simulate_analytic_timetable(
            timetable=timetable, theta=theta, mu=mu, sigma=sigma, n_paths=n_paths, re_normalize=re_normalize,
            init_condition=init_condition)

        # Output the array.
        return simulated_paths
    
    def simulate_analytic_timetable(
            self, timetable: np.ndarray, theta: float = 0.5, mu: float = 0, sigma: float = 0.05, n_paths: int = 10,
            re_normalize: bool = True, init_condition: float = 0)-> np.ndarray:
        """
        Analytically simulate the OU paths with a given time table per path.
        
        Use Euler advancement to simulate the paths following the drifted OU model: 
            dX = theta * (mu - X) * dt + sigma * dW,
        where theta, mu, sigma are constants. Method used is the full analytical solution [Doob (1942)]:
            X_t = X_0*exp(- theta t) + mu*(1-exp(theta * t)) 
            + sigma*exp(-theta*t)*Normal(0, exp(- 2 theta t)-1) / sqrt(2*theta)
        
        :param timetable: (np.ndarray) 1D numpy array that indicate the time of interest for calculation.
        :param theta: (float) Optional. Mean-reverting speed. Defaults to 0.5.
        :param mu: (float) Optional. Mean-reverting level. Defaults to 0.
        :param sigma: (float) Optional. Standard deviation for the Brownian motion. Defalts to 0.05.
        :param n_paths: (float) Optional. Number of paths in the simulation. Needs to be >=2. Defaults to 10.
        :param re_normalize: (bool) Optional. Whether to renormalize the Gaussians sampled at each time advancement.
            This will only be triggered if n_paths >= 2.
        :param init_condition: (float) Optional. Initial start position for every path. Defaults to 0.
        """
        
        # Initialize
        n_points = len(timetable)
        simulated_paths = np.zeros(shape=(n_points, n_paths))
        simulated_paths[0, :] += init_condition
        
        # 1. Draw from Gaussians accordingly with mean = 0. Will add back the mean = r*dt later.
        gaussians = np.random.normal(loc=0, scale=1, size=(n_points-1, n_paths))
        
        # 2. Re-normalize the Gaussian per point or instance (row-wise)
        if re_normalize and n_paths >= 2:
            gaussians = util.normalize_data(data_matrix=gaussians, act_on='row')
        
        # 3. Scale time-transformed Wiener process, generate Gaussians that has mean=0, var=exp(2 theta t) - 1
        trans_gaussian = np.zeros_like(gaussians)
        trans_gaussian[0] = (np.sqrt(np.exp(2 * theta * timetable[1]) -
                                     np.exp(2 * theta * timetable[0])) * gaussians[0])
        for i in range(1, n_points-1):
            trans_gaussian[i,:] = (trans_gaussian[i-1,:] + np.sqrt(np.exp(2 * theta * timetable[i]) -
                                                                   np.exp(2 * theta * timetable[i-1])) 
                                   * gaussians[i,:])
        
        # 4. Construct the paths exactly.
        for i in range(1, n_points):
            exp_decay = np.exp(-theta * timetable[i])
            simulated_paths[i,:] = (simulated_paths[0,:] * exp_decay + mu * (1 - exp_decay)
                                    + sigma * exp_decay * trans_gaussian[i-1,:] / np.sqrt(2 * theta))
            
        # 5. Output the array.
        return simulated_paths
    
    def simulate_discrete_grid_vary_mu(self, theta: float = 0.5, mu: Callable[[float], float] = None, sigma: float = 0.05, 
                               dt: float = 1/252, n_points: int = 252, n_paths: int = 10, re_normalize: bool = True,
                               init_condition: float = 0)-> np.ndarray:
        """
        Discretely simulate the OU paths with given time interval and total number of points per path.
        
        Use Euler advancement to simulate the paths following the drifted OU model: 
            dX = theta * (mu(t) - X) * dt + sigma * dW,
        where mu(t) is a function of time, and theta, sigma are constants. Assume t=0 at the beginning.
        
        :param timetable: (np.ndarray) 1D numpy array that indicate the time of interest for calculation.
        :param theta: (float) Optional. Mean-reverting speed. Defaults to 0.5.
        :param mu: (func) Optional. Mean-reverting level as a function of time. Defalts to the constant function 0.
            Note that the time unit should be consistent with other inputs, and is t=0 at the beginning of the
            simulation.
        :param sigma: (float) Optional. Standard deviation for the Brownian motion. Defalts to 0.05.
        :param dt: (float) Optional. Time advancement step. Defaults to 1/252.
        :param n_points: (int) Optional. Number of steps per simulated path, including the initial condition. Defaults
            to 252.
        :param n_paths: (float) Optional. Number of paths in the simulation. Needs to be >=2. Defaults to 10.
        :param re_normalize: (bool) Optional. Whether to renormalize the Gaussians sampled at each time advancement.
            This will only be triggered if n_paths >= 2.
        :param init_condition: (float) Optional. Initial start position for every path. Defaults to 0.
        """
        
        # Initialize the timetable. In this case it is a grid.
        timetable = np.arange(start=0, stop=dt * n_points, step=dt)
        
        # Simulated paths according to the time table.
        simulated_paths = self.simulate_discrete_timetable_vary_mu(
            timetable=timetable, theta=theta, mu=mu, sigma=sigma, n_paths=n_paths, re_normalize=re_normalize,
            init_condition=init_condition)
        
        # Output the array.
        return simulated_paths

    def simulate_discrete_timetable_vary_mu(
            self, timetable: np.ndarray, theta: float = 0.5, mu: Callable[[float], float] = None, sigma: float = 0.05,
            n_paths: int = 10, re_normalize: bool = True, init_condition: float = 0)-> np.ndarray:
        """
        Discretely simulate the OU paths with a given time table and total number of points per path.
        
        Use Euler advancement to simulate the paths following the drifted OU model: 
            dX = theta * (mu(t) - X) * dt + sigma * dW,
        where mu(t) is a function of time, and theta, sigma are constants. Assume t=0 at the beginning.
        
        :param timetable: (np.ndarray) 1D numpy array that indicate the time of interest for calculation.
        :param theta: (float) Optional. Mean-reverting speed. Defaults to 0.5.
        :param mu: (func) Optional. Mean-reverting level as a function of time. Defalts to the constant function 0.
            Note that the time unit should be consistent with other inputs, and is t=0 at the beginning of the
            simulation.
        :param sigma: (float) Optional. Standard deviation for the Brownian motion. Defalts to 0.05.
        :param n_paths: (float) Optional. Number of paths in the simulation. Needs to be >=2. Defaults to 10.
        :param re_normalize: (bool) Optional. Whether to renormalize the Gaussians sampled at each time advancement.
            This will only be triggered if n_paths >= 2.
        :param init_condition: (float) Optional. Initial start position for every path. Defaults to 0.
        """
        
        # Initialize
        n_points = len(timetable)
        simulated_paths = np.zeros(shape=(n_points, n_paths))
        simulated_paths[0, :] += init_condition
        if mu is None:  # Default mu(t) = 0
            mu = lambda t: 0 * t
        
        # 1. Draw from the Gaussian distribution accordingly.
        gaussians = np.random.normal(loc=0, scale=1, size=(n_points-1, n_paths))

        # 2. Re-normalize the Gaussian per point or instance (normalize every row)
        if re_normalize and n_paths >= 2:
            gaussians = util.normalize_data(data_matrix=gaussians, act_on='row')

        # 3. Construct the paths via Euler advancement.
        for i in range(1, n_points):
            dt = timetable[i] - timetable[i-1]
            sqdt = np.sqrt(dt)
            increments = theta * (mu(timetable[i]) - simulated_paths[i-1,:]) * dt + sigma * gaussians[i-1, :] * sqdt
            simulated_paths[i,:] = simulated_paths[i-1,:] + increments

        # 4. Output the array.
        return simulated_paths
    
    def simulate_analytic_grid_vary_mu(
            self, theta: float = 0.5, mu: Callable[[float], float] = None, sigma: float = 0.05, dt: float = 1/252,
            n_points: int = 252, n_paths: int = 10, re_normalize: bool = True, init_condition: float = 0)-> np.ndarray:
        """
        Analytically simulate the OU paths with a given time table per path.
        
        Use Euler advancement to simulate the paths following the drifted OU model: 
            dX = theta * (mu - X) * dt + sigma * dW,
        where mu(t) is a function of time, and theta, sigma are constants. Method used is the full analytical solution
        by [Doob (1942)]:
            X_t = X_0*exp(- theta t) + mu(t)*(1-exp(theta * t)) 
            + sigma*exp(-theta*t)*Normal(0, exp(- 2 theta t)-1) / sqrt(2*theta)
        
        :param timetable: (np.ndarray) 1D numpy array that indicate the time of interest for calculation.
        :param theta: (float) Optional. Mean-reverting speed. Defaults to 0.5.
        :param mu: (func) Optional. Mean-reverting level as a function of time. Defalts to the constant function 0.
            Note that the time unit should be consistent with other inputs, and is t=0 at the beginning of the
            simulation.
        :param dt: (float) Optional. Time advancement step. Defaults to 1/252.
        :param n_points: (int) Optional. Number of steps per simulated path, including the initial condition. Defaults
            to 252.
        :param n_paths: (float) Optional. Number of paths in the simulation. Needs to be >=2. Defaults to 10.
        :param re_normalize: (bool) Optional. Whether to renormalize the Gaussians sampled at each time advancement.
            This will only be triggered if n_paths >= 2.
        :param init_condition: (float) Optional. Initial start position for every path. Defaults to 0.
        """

        # Initialize the timetable. In this case it is a grid.
        timetable = np.arange(start=0, stop=dt * n_points, step=dt)

        # Simulated paths according to the time table.
        simulated_paths = self.simulate_analytic_timetable_vary_mu(
            timetable=timetable, theta=theta, mu=mu, sigma=sigma, n_paths=n_paths, re_normalize=re_normalize,
            init_condition=init_condition)

        # Output the array.
        return simulated_paths
    
    def simulate_analytic_timetable_vary_mu(
            self, timetable: np.ndarray, theta: float = 0.5, mu: Callable[[float], float] = None, sigma: float = 0.05,
            n_paths: int = 10, re_normalize: bool = True, init_condition: float = 0)-> np.ndarray:
        """
        Analytically simulate the OU paths with a given time table per path.
        
        Use Euler advancement to simulate the paths following the drifted OU model: 
            dX = theta * (mu - X) * dt + sigma * dW,
        where mu(t) is a function of time, and theta, sigma are constants. Method used is the full analytical solution
        by [Doob (1942)]:
            X_t = X_0*exp(- theta t) + mu(t)*(1-exp(theta * t)) 
            + sigma*exp(-theta*t)*Normal(0, exp(- 2 theta t)-1) / sqrt(2*theta)
        
        :param timetable: (np.ndarray) 1D numpy array that indicate the time of interest for calculation.
        :param theta: (float) Optional. Mean-reverting speed. Defaults to 0.5.
        :param mu: (func) Optional. Mean-reverting level as a function of time. Defalts to the constant function 0.
            Note that the time unit should be consistent with other inputs, and is t=0 at the beginning of the
            simulation.
        :param sigma: (float) Optional. Standard deviation for the Brownian motion. Defalts to 0.05.
        :param n_paths: (float) Optional. Number of paths in the simulation. Needs to be >=2. Defaults to 10.
        :param re_normalize: (bool) Optional. Whether to renormalize the Gaussians sampled at each time advancement.
            This will only be triggered if n_paths >= 2.
        :param init_condition: (float) Optional. Initial start position for every path. Defaults to 0.
        """
        
        # Initialize
        n_points = len(timetable)
        simulated_paths = np.zeros(shape=(n_points, n_paths))
        simulated_paths[0, :] += init_condition
        if mu is None:  # Default mu(t) = 0
            mu = lambda t: 0 * t
        
        # 1. Draw from Gaussians accordingly with mean = 0. Will add back the mean = r*dt later.
        gaussians = np.random.normal(loc=0, scale=1, size=(n_points-1, n_paths))
        
        # 2. Re-normalize the Gaussian per point or instance (row-wise)
        if re_normalize and n_paths >= 2:
            gaussians = util.normalize_data(data_matrix=gaussians, act_on='row')
        
        # 3. Scale time-transformed Wiener process, generate Gaussians that has mean=0, var=exp(2 theta t) - 1
        trans_gaussian = np.zeros_like(gaussians)
        trans_gaussian[0] = (np.sqrt(np.exp(2 * theta * timetable[1]) -
                                     np.exp(2 * theta * timetable[0])) * gaussians[0])
        for i in range(1, n_points-1):
            trans_gaussian[i,:] = (trans_gaussian[i-1,:] + np.sqrt(np.exp(2 * theta * timetable[i]) -
                                                                   np.exp(2 * theta * timetable[i-1])) 
                                   * gaussians[i,:])
        
        # 4. Construct the paths exactly.
        for i in range(1, n_points):
            exp_decay = np.exp(-theta * timetable[i])
            simulated_paths[i,:] = (simulated_paths[0,:] * exp_decay + mu(timetable[i]) * (1 - exp_decay)
                                    + sigma * exp_decay * trans_gaussian[i-1,:] / np.sqrt(2 * theta))
            
        # 5. Output the array.
        return simulated_paths

#%% Test
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    n_paths = 10
    n_points = 252
    theta = 5
    mu = lambda t : np.sin(t * 10) / 10
    sigma = 0.05
    
    ous = OUSimulator()
    # paths_dis = ous.simulate_discrete_grid(theta=theta, mu=mu, sigma=sigma, n_paths=n_paths, n_points=n_points,
    #                                        re_normalize = True, init_condition=0.2)
    # paths_aly = ous.simulate_analytic_grid(theta=theta, mu=mu, sigma=sigma, n_paths=n_paths, n_points=n_points,
    #                                        re_normalize = True, init_condition=0.2)
    
    # Varying mu(t)
    paths_dis = ous.simulate_discrete_grid_vary_mu(theta=theta, mu=mu, sigma=sigma, n_paths=n_paths, n_points=n_points,
                                           re_normalize = True, init_condition=0)
    paths_aly = ous.simulate_analytic_grid_vary_mu(theta=theta, mu=mu, sigma=sigma, n_paths=n_paths, n_points=n_points,
                                           re_normalize = True, init_condition=0)
    
    plt.figure(dpi=200)
    for i in range(10):
        plt.plot(paths_dis[:,i])
    plt.title("Discrete Simulated OU Process")
    plt.show()
    
    plt.figure(dpi=200)
    for i in range(10):
        plt.plot(paths_aly[:,i])
    plt.title("Analytical Simulated OU Process")
    plt.show()
    