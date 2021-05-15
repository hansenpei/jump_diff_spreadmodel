"""
Simulator class for 1D Ornstein-Uhlenbeck process with drift: dX = theta * (mu - X) * dt + sigma * dW.
"""

import numpy as np
import util

class OUSimulator:
    """
    Class that simulates brownian based SDEs.
    """
    def __init__(self, seed: int = None):
        
        self.seed = seed
        
    def simulate_discrete_grid(self, r: float = 0, sigma: float = 0.05, dt: float = 1/252,
                               n_points: int = 252, n_paths: int = 10, re_normalize: bool = True,
                               init_conditions: np.ndarray = None)-> np.ndarray:
        """
        Discretely simulate the OU paths with given time interval and total number of points per path.
        
        Use Euler advancement to simulate the paths following the drifted OU model: 
            dX = theta * (mu - X) * dt + sigma * dW,
        where theta, mu, sigma are constants.
        
        :param r: (float) Optional. Drift rate. Defaults to 0.
        :param sigma: (float) Optional. Standard deviation for the Brownian motion. Defalts to 0.05.
        :param dt: (float) Optional. Time advancement step. Defaults to 1/252.
        :param n_points: (int) Optional. Number of steps per simulated path, including the initial condition. Defaults
            to 252.
        :param n_paths: (float) Optional. Number of paths in the simulation. Needs to be >=2. Defaults to 10.
        :param re_normalize: (bool) Optional. Whether to renormalize the Gaussians sampled at each time advancement.
            This will only be triggered if n_paths >= 2.
        """
        # Initialize
        simulated_paths = np.zeros(shape=(n_points, n_paths))
        
        # 1. Draw from the Gaussian distribution accordingly.
        gaussians = np.random.normal(loc=0, scale=1, size=(n_points-1, n_paths))
        
        # 2. Re-normalize the Gaussian per point or instance (normalize every row)
        if re_normalize and n_paths >= 2:
            gaussians = util.normalize_data(data_matrix=gaussians, act_on='row')
            
        # 3. Construct the path via Euler advancement.
        for i in range(1, n_points):
            simulated_paths[i,:] = simulated_paths[i-1,:] + r * dt + sigma * gaussians[i-1, :]
            
        # 4. Output the array.
        return simulated_paths