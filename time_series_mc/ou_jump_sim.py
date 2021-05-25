"""
Simulator for the Ornstein-Uhlenbeck process with jump: X = g(t) + Y.

Here Y is the stochastic term satisfying dY = - theta * Y * dt + sigma * dW + log(Jt) * dNt, where Jt follows
Normal(mu_Jt, sigma). Compare with the canonical formula of OU process: dX = theta * (mu(t) - X) * dt + sigma * dW,
we have mu(t) = g'(t) / theta + g(t).
"""

import numpy as np
import util
from time_series_mc.ou_sim import OUSimulator
from typing import Callable, List

class OUJumpSimulator(OUSimulator):
    """
    Class that simulates OU process with jump.
    """
    def __init__(self, seed: int = None):
        
        self.seed = seed

    def simulate_grid(
            self, theta: float = 0.5, g: Callable[[float], float] = None, sigma: float = 0.05, lambda_j: float = 25,
            mu_j: [[float], float] = None, sigma_j: float = 0.02, dt: float = 1/98532, n_points: int = 3910,
            n_paths: int = 10, re_normalize: bool = True, init_condition: float = 0)-> np.ndarray:
        r"""
        Simulate an OU process with jump on a grid, where the jump can happen at any time.
        
        The simulation is carried out discretely from the relation:
            X_t = g(t) + Y_t
            d Y_t = - theta * Y_t * dt + sigma * dW + logJ_t * dN_t
        where g(t) is the deterministic drift, and logJ_t ~ Normal(mu_j, sigma_j**2), N_t ~ Poission(lambda_j).
        Each path is simulated according to the following:
            X_t = g(t) + (X_0 - g(0))*exp(-theta*t) + sigma*int{_0^t}{exp(-theta(t-tau)), d W_tau}
            + int{_0^t}{exp(-theta(t - tau)) logJ_t, d N_tau}
        And the two stochastic integration terms are evaluated in the Ito's sense over the grid length dt.

        :param theta: (float) Optional. Mean-reverting speed. Defaults to 0.5.
        :param g: (Callable[[float], float]). The deterministic drift function for the OU-Jump process. Defalts to
            constant 0.
        :param sigma: (float) Optional. Standard deviation for the Brownian motion. Defalts to 0.05.
        :param lambda_j: (float) Optional. Jump rate. Defaults to 25.
        :param mu_j: (Callable[[float], float]) Optional. Jump average as a function of time. Defalts to constant 0.
        :param sigma_j: (float) Optional. The jump std constant. Defaults to 0.02.
        :param dt: (float) Optional. Time advancement step. Defaults to 1/98532 = 1/252/391, which is 1 min.
        :param n_points: (int) Optional. Number of steps per simulated path, including the initial condition. Defaults
            to 3910.
        :param n_paths: (float) Optional. Number of paths in the simulation. Needs to be >=2. Defaults to 10.
        :param re_normalize: (bool) Optional. Whether to renormalize the Gaussians sampled at each time advancement.
            This will only be triggered if n_paths >= 2. Defalts to True.
        :param init_condition: (float) Optional. Initial start position for every path. Defaults to 0.
        :return: (np.ndarray) The simulated paths, dimension is (n_points, n_paths).
        """
        
        # 1. Handle the jump part
        # TimeTable where the jump occurs
        jump_timestamp = util.generate_jump_timetable_grid(
            jump_rate=lambda_j, dt=dt, n_points=n_points, n_paths=n_paths, return_just_time=True)
        # Jump size according to normal distribution, we do not renormalize due to its low frequency and changing mean
        jump_size = util.generate_jump_size(jump_timestamp=jump_timestamp, mu_j=mu_j, sigma_j=sigma_j)
        
        # 2. Simulate the paths
        simulated_paths = self._simulate_over_grid(
            theta=theta, g=g, sigma=sigma, lambda_j=lambda_j, mu_j=mu_j, sigma_j=sigma_j, dt=dt, n_points=n_points,
            n_paths=n_paths, re_normalize=re_normalize, init_condition=init_condition, jump_timestamp=jump_timestamp,
            jump_size=jump_size)
            
        return simulated_paths
        
    def simulate_grid_overnight(
            self, theta: float = 0.5, g: Callable[[float], float] = None, sigma: float = 0.05, lambda_j: float = 25,
            mu_j: [[float], float] = None, sigma_j: float = 0.02, dt: float = 1/98532, overnight_dt: float = 1/252, 
            n_points: int = 3910, n_paths: int = 10, daily_dpoints: int = 391, re_normalize: bool = True,
            init_condition: float = 0)-> np.ndarray:
        """
        Simulate an OU process with jump on a grid, where the jump only happens overnight.

        The simulation is carried out discretely from the relation:
            X_t = g(t) + Y_t
            d Y_t = - theta * Y_t * dt + sigma * dW + logJ_t * dN_t
        where g(t) is the deterministic drift, and logJ_t ~ Normal(mu_j, sigma_j**2), N_t ~ Poission(lambda_j).
        Each path is simulated according to the following:
            X_t = g(t) + (X_0 - g(0))*exp(-theta*t) + sigma*int{_0^t}{exp(-theta(t-tau)), d W_tau}
            + int{_0^t}{exp(-theta(t - tau)) logJ_t, d N_tau}
        where the jump can only occur overnight. And the two stochastic integration terms are evaluated in the Ito's
        sense (function evaluated at the beginning of each discretization interval) over the grid length dt.
        
        :param theta: (float) Optional. Mean-reverting speed. Defaults to 0.5.
        :param g: (Callable[[float], float]). The deterministic drift function for the OU-Jump process. Defalts to
            constant 0.
        :param sigma: (float) Optional. Standard deviation for the Brownian motion. Defalts to 0.05.
        :param lambda_j: (float) Optional. Jump rate. Defaults to 25.
        :param mu_j: (Callable[[float], float]) Optional. Jump average as a function of time. Defalts to constant 0.
        :param sigma_j: (float) Optional. The jump std constant. Defaults to 0.02.
        :param dt: (float) Optional. Time advancement step. Defaults to 1/98532 = 1/252/391, which is 1 min.
        :param overnight_dt: (float) Optional. Overnight time. Defaults to 1/252.
        :param n_points: (int) Optional. Number of steps per simulated path, including the initial condition. Defaults
            to 3910.
        :param n_paths: (float) Optional. Number of paths in the simulation. Needs to be >=2. Defaults to 10.
        :param daily_dpoints: (int) Optional. The number of data points per day. Defaults to 391.
        :param re_normalize: (bool) Optional. Whether to renormalize the Gaussians sampled at each time advancement.
            This will only be triggered if n_paths >= 2. Defalts to True.
        :param init_condition: (float) Optional. Initial start position for every path. Defaults to 0.
        :return: (np.ndarray) The simulated paths, dimension is (n_points, n_paths).
        """

        # 1. Handle the jumps
        # Initialize the timetable. In this case it is a grid.
        timetable = np.arange(start=0, stop=dt*n_points, step=dt)
        # TimeTable where the jump occurs
        jump_dates = util.generate_jump_timetable_overnight(
            jump_rate=lambda_j, overnight_dt=overnight_dt, n_points=(n_points // daily_dpoints), n_paths=n_paths,
            return_just_date=True)
        jump_timestamp = [timetable[daily_dpoints * jump_dates[path_idx]] for path_idx in range(n_paths)]
        # Jump size according to normal distribution, we do not renormalize due to its low frequency and changing mean
        jump_size = util.generate_jump_size(jump_timestamp=jump_timestamp, mu_j=mu_j, sigma_j=sigma_j)
        
        # 2. Simulate the paths
        simulated_paths = self._simulate_over_grid(
            theta=theta, g=g, sigma=sigma, lambda_j=lambda_j, mu_j=mu_j, sigma_j=sigma_j, dt=dt, n_points=n_points,
            n_paths=n_paths, re_normalize=re_normalize, init_condition=init_condition, jump_timestamp=jump_timestamp,
            jump_size=jump_size)
            
        return simulated_paths
    
    def _simulate_over_grid(
            self, theta: float, g: Callable[[float], float], sigma: float, lambda_j: float, mu_j: [[float], float],
            sigma_j: float, dt: float, n_points: int, n_paths: int, re_normalize: bool,
            init_condition: float, jump_timestamp: List[np.ndarray], jump_size: List[list])-> np.ndarray:
        """
        The OU-Jump simulation engine with a given timestamp and size for jumps.
        
        The simulation is carried out discretely from the relation:
            X_t = g(t) + Y_t
            d Y_t = - theta * Y_t * dt + sigma * dW + logJ_t * dN_t
        where g(t) is the deterministic drift, and logJ_t ~ Normal(mu_j, sigma_j**2), N_t ~ Poission(lambda_j).
        Each path is simulated according to the following:
            X_t = g(t) + (X_0 - g(0))*exp(-theta*t) + sigma*int{_0^t}{exp(-theta(t-tau)), d W_tau}
            + int{_0^t}{exp(-theta(t - tau)) logJ_t, d N_tau}
        the two stochastic integration terms are evaluated in the Ito's sense (function evaluated at the beginning of 
        each discretization interval) over the grid length dt.
    
        :param theta: (float) Mean-reverting speed. 
        :param g: (Callable[[float], float]). The deterministic drift function for the OU-Jump process.
        :param sigma: (float) Standard deviation for the Brownian motion.
        :param lambda_j: (float) Jump rate. Defaults to 25.
        :param mu_j: (Callable[[float], float]) Jump average as a function of time.
        :param sigma_j: (float) The jump std constant.
        :param dt: (float) Time advancement step.
        :param n_points: (int) Number of steps per simulated path, including the initial condition.
        :param n_paths: (float) Number of paths in the simulation. Needs to be >=2.
        :param daily_dpoints: (int) The number of data points per day.
        :param re_normalize: (bool) Whether to renormalize the Gaussians sampled at each time advancement.
            This will only be triggered if n_paths >= 2.
        :param init_condition: (float) Initial start position for every path.
        :param jump_timestamp: (List[np.ndarray]) The time where the jump occurs.
        :param jump_size: (List[list]) The size of each jump.
        :return: (np.ndarray) The simulated paths, dimension is (n_points, n_paths).
        """
        
        # Initialize the timetable. In this case it is a grid.
        total_time = dt * n_points
        timetable = np.arange(start=0, stop=total_time, step=dt)
        # Initializa simulated_paths
        simulated_paths = np.zeros(shape=(n_points, n_paths))
        simulated_paths[0, :] += init_condition
        
        # 1. Handle the Brownian part
        # Draw from the Gaussian distribution accordingly.
        gaussians = np.random.normal(loc=0, scale=1, size=(n_points-1, n_paths))
        # Re-normalize the Gaussian per point or instance (normalize every row)
        if re_normalize and n_paths >= 2:
            gaussians = util.normalize_data(data_matrix=gaussians, act_on='row')
        
        # 2. Generate the paths for the jump diffusion process using discretization with interval dt.
        sqdt = np.sqrt(dt)
        for path in range(n_paths):
            # Precalculate part of the Brownian and jump stoch integral for faster calculation
            cumu_brownian_precalculate = np.cumsum([
                np.exp(theta * timetable[j]) * gaussians[j][path] * sqdt for j in range(n_points-1)]) * sigma
            cumu_jump_precalculate = np.cumsum([np.exp(theta * jump_timestamp[path][j]) * jump_size[path][j]
                for j in range(len(jump_timestamp[path]))])
            # Initializing variables for the jump process
            jump_counter = 0  # Count the number of jumps. Serve as a pointer.
            cumu_jump = 0  # Initialize the cumulative jump integral value
            
            for point in range(1, n_points):
                cur_time = timetable[point]  # Current time
                pre_time = timetable[point-1]  # Time one step back
                
                # Calculate the deterministic drift part, Brownian part and jump part separately
                drift_part = g(cur_time) + (simulated_paths[0, path] - g(0)) * np.exp(-theta * cur_time)
                # Brownian increment. Note the stoch integral is taken in the Ito's sense
                cumu_brownian = np.exp(- theta * pre_time) * cumu_brownian_precalculate[point-1]
                # Jump increment. Note the stoch integral is taken in the Ito's sense
                if cur_time in jump_timestamp[path]:  # If this is the time for a jump
                    cumu_jump = cumu_jump_precalculate[jump_counter] * np.exp(- theta * pre_time)
                    jump_counter += 1
                # Assemble together for this path
                simulated_paths[point, path] = drift_part + cumu_brownian + cumu_jump
            
        return simulated_paths
    
#%% test
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    ouj = OUJumpSimulator()
    mu_j = lambda t: 0*t  # Jump avg
    g = lambda t: 0*t  # Drift
    n_paths = 20
    
    simu_paths = ouj.simulate_grid(
        theta=0.2, g=g, sigma=0.05, lambda_j=0.2, mu_j=mu_j, sigma_j=0.2, dt=1/100, n_points=4000,
        n_paths=n_paths, re_normalize=True, init_condition=0)
    
    simu_paths_overnightjump = ouj.simulate_grid_overnight(
        theta=0.2, g=g, sigma=0.05, lambda_j=0.2, mu_j=mu_j, sigma_j=0.2, dt=1/100, n_points=4000,
        n_paths=n_paths, re_normalize=True, init_condition=0, overnight_dt=1/10, daily_dpoints=10)
    
    # Plotting
    plt.figure(dpi=200)
    for i in range(n_paths):
        plt.plot(simu_paths[:,i])
    plt.title("OU with Jump Simulation")
    plt.show()
    
    plt.figure(dpi=200)
    for i in range(n_paths):
        plt.plot(simu_paths_overnightjump[:,i])
    plt.title("OU with Only Overnight Jump Simulation")
    plt.show()
    
    