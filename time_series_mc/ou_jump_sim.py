"""
Simulator for the Ornstein-Uhlenbeck process with jump: dX = theta*(mu(t)-X)*dt + sigma*dW + lnJ*dN, or X = g(t) + Y.

Here Y is the stochastic term satisfying dY = - theta * Y * dt + sigma * dW + log(Jt) * dNt, where Jt follows
Normal(mu_Jt, sigma). Compare with the canonical formula of OU process: dX = theta * (mu(t) - X) * dt + sigma * dW,
we have mu(t) = g'(t) / theta + g(t).
"""

import numpy as np
import util
from time_series_mc.ou_sim import OUSimulator
from typing import Callable

class OUJumpSimulator(OUSimulator):
    """
    Class that simulates OU process with jump.
    """
    def __init__(self, seed: int = None):
        
        self.seed = seed

    def simulate_grid(
            self, theta: float = 0.5, g: Callable[[float], float] = None, sigma: float = 0.05, lambda_j: float = 1,
            mu_j: [[float], float] = None, sigma_j: float = 0.02, dt: float = 1/252, n_points: int = 252,
            n_paths: int = 10, re_normalize: bool = True, init_condition: float = 0)-> np.ndarray:
        """
        Simulate an OU process with jump on a grid, where the jump can happen at any time.
        
        The simulation is carried out discretely from the SDE 

        Parameters
        ----------
        theta : float, optional
            DESCRIPTION. The default is 0.5.
        mu : Callable[[float], float], optional
            DESCRIPTION. The default is None.
        sigma : float, optional
            DESCRIPTION. The default is 0.05.
        lambda_j : float, optional
            DESCRIPTION. The default is 1.
        mu_j : [[float], float], optional
            DESCRIPTION. The default is None.
        sigma_j : float, optional
            DESCRIPTION. The default is 0.02.
        dt : float, optional
            DESCRIPTION. The default is 1/252.
        n_points : int, optional
            DESCRIPTION. The default is 252.
        n_paths : int, optional
            DESCRIPTION. The default is 10.
        re_normalize : bool, optional
            DESCRIPTION. The default is True.
        init_condition : float, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        None.

        """
        
        pass

    def simulate_grid_intraday(
            self, theta: float = 0.5, mu: Callable[[float], float] = None, sigma: float = 0.05, lambda_j: float = 1,
            mu_j: [[float], float] = None, sigma_j: float = 0.02, dt: float = 1/252, n_points: int = 252,
            n_paths: int = 10, daily_dpoints: int = 361, re_normalize: bool = True,
            init_condition: float = 0)-> np.ndarray:
        """
        Simulate an OU process with jump on a grid, where the jump only happens intraday.

        Parameters
        ----------
        theta : float, optional
            DESCRIPTION. The default is 0.5.
        mu : Callable[[float], float], optional
            DESCRIPTION. The default is None.
        sigma : float, optional
            DESCRIPTION. The default is 0.05.
        lambda_j : float, optional
            DESCRIPTION. The default is 1.
        mu_j : [[float], float], optional
            DESCRIPTION. The default is None.
        sigma_j : float, optional
            DESCRIPTION. The default is 0.02.
        dt : float, optional
            DESCRIPTION. The default is 1/252.
        n_points : int, optional
            DESCRIPTION. The default is 252.
        n_paths : int, optional
            DESCRIPTION. The default is 10.
        re_normalize : bool, optional
            DESCRIPTION. The default is True.
        init_condition : float, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        None.

        """
        
        pass