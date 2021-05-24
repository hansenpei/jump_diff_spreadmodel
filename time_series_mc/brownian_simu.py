"""
Simulator class for 1D Brownian motion with drift: dX = r * dt + sigma * dW.
"""

import numpy as np
import util

class BrownianSimulator:
    """
    Class that simulates the Brownian based SDEs.
    """
    def __init__(self, seed: int = None):
        
        self.seed = seed
        
    def simulate_discrete_grid(self, r: float = 0, sigma: float = 0.05, dt: float = 1/252,
                               n_points: int = 252, n_paths: int = 10, re_normalize: bool = True,
                               init_condition: float = 0)-> np.ndarray:
        """
        Discretely simulate the brownian motions paths with given time interval and total number of points per path.
        
        Use Euler advancement to simulate the paths following the drifted Brownian model: dX = r*dt + sigma * dW,
        where r, sigma are constants. It happens that the discrete result is exact.
        
        :param r: (float) Optional. Drift rate. Defaults to 0.
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
        simulated_paths = self.simulate_discrete_timetable(timetable=timetable, r=r, sigma=sigma, n_paths=n_paths,
                                                           re_normalize=re_normalize, init_condition=init_condition)
        
        # Output the array.
        return simulated_paths
    
    def simulate_discrete_timetable(
            self, timetable: np.ndarray, r: float = 0, sigma: float = 0.05, n_paths: int = 10,
            re_normalize: bool = True, init_condition: float = 0) -> np.ndarray:
        """
        Discretely simulate the Brownian motions according to a given timetable.
        
        Use Euler advancement to simulate the paths following the drifted Brownian model: dX = r*dt + sigma * dW,
        where r, sigma are constants. It happens that the discrete result is exact.
        
        :param timetable: (np.ndarray) 1D numpy array that indicate the time of interest for calculation. 
        :param r: (float) Optional. Drift rate. Defaults to 0.
        :param sigma: (float) Optional. Standard deviation for the Brownian motion. Defalts to 0.05.
        :param dt: (float) Optional. Time advancement step. Defaults to 1/252.
        :param n_points: (int) Optional. Number of steps per simulated path, including the initial condition. Defaults
            to 252.
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
            simulated_paths[i,:] = simulated_paths[i-1,:] + r * dt + sigma * gaussians[i-1, :] * sqdt
            
        # 4. Output the array.
        return simulated_paths
        
    
    def simulate_analytical_grid(self, r: float = 0, sigma: float = 0.05, dt: float = 1/252,
                                 n_points: int = 252, n_paths: int = 10, re_normalize: bool = True,
                                 init_condition: float = 0)-> np.ndarray:
        r"""
        Analytically simulate the Brownian motions paths with given time interval and total number of points per path.
        
        Use the analytical solution to simulate the paths following the drifted Brownian model: dX = r*dt + sigma * dW,
        where r, sigma are constants. It turned out that the analytical approach is equivalent to the discrete case.
        
        dX is drawn from Gaussian(r*dt, sigma) for each advancement.
        
        :param r: (float) Optional. Drift rate. Defaults to 0.
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
        simulated_paths = self.simulate_analytical_timetable(timetable=timetable, r=r, sigma=sigma, n_paths=n_paths,
                                                             re_normalize=re_normalize, init_condition=init_condition)
        
        return simulated_paths
    
    def simulate_analytical_timetable(
            self, timetable: np.ndarray, r: float = 0, sigma: float = 0.05, n_paths: int = 10,
            re_normalize: bool = True, init_condition: float = 0) -> np.ndarray:
        """
        Analytically simulate the Brownian motions according to a given timetable.
        
        Use the analytical solution to simulate the paths following the drifted Brownian model: dX = r*dt + sigma * dW,
        where r, sigma are constants. It turned out that the analytical approach is equivalent to the discrete case.
        
        dX is drawn from Gaussian(r*dt, sigma) for each advancement.
        
        :param r: (float) Optional. Drift rate. Defaults to 0.
        :param sigma: (float) Optional. Standard deviation for the Brownian motion. Defalts to 0.05.
        :param dt: (float) Optional. Time advancement step. Defaults to 1/252.
        :param n_points: (int) Optional. Number of steps per simulated path, including the initial condition. Defaults
            to 252.
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
        
        # 3. Add back the drift term r*dt, and rescale with sigma.
        exact_advancements = np.zeros_like(gaussians)
        for i in range(n_points-1):
            dt = timetable[i+1] - timetable[i]
            sqdt = np.sqrt(dt)
            exact_advancements[i,:] = sigma * gaussians[i,:] * sqdt + r * dt
        
        # 4. Construct the paths exactly.
        for i in range(1, n_points):
            simulated_paths[i, :] = simulated_paths[i-1, :] + exact_advancements[i-1, :]
            
        # 5. Output the array.
        return simulated_paths

#%% Sanity check by plotting the paths

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    n_paths = 30
    n_points = 252
    r = 0.0
    sigma = 0.05

    bs = BrownianSimulator()
    paths_dis = bs.simulate_discrete_grid(r=r, n_paths=n_paths, n_points=n_points, sigma=sigma, re_normalize = True)
    paths_aly = bs.simulate_analytical_grid(r=r, n_paths=n_paths, n_points=n_points, sigma=sigma, re_normalize = True)

    timetable = util.generate_timetable(n_days=252, intraday_freq='30m', output_unit='y')
    paths_dis_tt = bs.simulate_discrete_timetable(timetable=timetable, r=r, sigma=sigma, n_paths=n_paths)
    paths_aly_tt = bs.simulate_analytical_timetable(timetable=timetable, r=r, sigma=sigma, n_paths=n_paths)

    plt.figure(dpi=200)
    for i in range(n_paths):
        plt.plot(paths_dis[:,i])
    plt.title("Discrete Brownian Simulation")
    plt.show()

    plt.figure(dpi=200)
    for i in range(n_paths):
        plt.plot(paths_aly[:,i])
    plt.title("Analytical Brownian Simulation")
    plt.show()

    # plt.figure(dpi=200)
    # for i in range(n_paths):
    #     plt.plot(paths_dis_tt[:,i])
    # plt.show()
    
    # plt.figure(dpi=200)
    # for i in range(n_paths):
    #     plt.plot(paths_aly_tt[:,i])
    # plt.show()
    
#%% Statistical checks
if __name__ == '__main__':
    
    # 1. Simulate and collect the path's endpoint
    n_paths = 30 * 100
    n_points = 252
    r = 0.0
    sigma = 0.05

    bs = BrownianSimulator()
    paths_dis = bs.simulate_discrete_grid(r=r, n_paths=n_paths, n_points=n_points, sigma=sigma, re_normalize = True)
    paths_aly = bs.simulate_analytical_grid(r=r, n_paths=n_paths, n_points=n_points, sigma=sigma, re_normalize = True)
    
    endpoints_dis = paths_dis[-1, :]
    endpoints_aly = paths_aly[-1, :]
    
    plt.figure(dpi=200)
    plt.hist(endpoints_dis, bins=100, density=True)
    plt.title("Discrete Sim Hist")
    plt.show()
    
    plt.figure(dpi=200)
    plt.hist(endpoints_aly, bins=100, density=True)
    plt.title("Analytical Sim Hist")
    plt.show()
    
    # 2. mean and std check
    print("discrete, mean {}, std {}".format(np.mean(endpoints_dis), np.std(endpoints_dis)))
    print("analytical, mean {}, std {}".format(np.mean(endpoints_aly), np.std(endpoints_aly)))
    
    # 3. Shaprio-Wilk test on Normality
    from scipy.stats import shapiro
    stat1, p1 = shapiro(endpoints_dis)
    stat2, p2 = shapiro(endpoints_aly)

    print(stat1, p1, stat2, p2)