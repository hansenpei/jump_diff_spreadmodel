"""
Modules to filter out the jump part from the data.
"""

import numpy as np
from typing import Tuple

class StubingerConsistentFilter:
    """
    Filter jumps that are consistent with the jump analysis part in [Stubinger (2017)].
    This method although is logically consistent, the filtered series does not have good statistical properties.
    Therefore this filter is depreciated.
    """

    def filter(self, prices: np.ndarray, thresholds: Tuple[float], thres_in_quantile: bool = False,
               calculate_on_returns: bool = False) -> np.ndarray:
        """
        Filter the prices that is consistent with the jump analysis part.

        Anything outside of a fixed threshold range will be considered a jump, and will be filtered. This method
        though is logically more consistent, but the filtered result does not have good statistical properties.

        :param prices: (np.ndarray) The price series.
        :param thresholds: (Tuple[float]) The thresholds in the form of (lower thres, upper thres). It can be in
            absolute terms or in quantile.
        :param thres_in_quantile: (bool) Optional. Whether the thresholds input are in terms of quantiles. Defaults to
            False.
        :param calculate_on_returns: (bool) Optional. Whether to conduct the filtering based on returns or diff.
            Defaults to False (i.e., use price differences).
        :return: (np.ndarray) The jump-adjusted prices.
        """

        # Calculate based on price diffs
        if not calculate_on_returns:
            diffs = np.diff(prices, prepend=0)  # Calculate diffs
            if thres_in_quantile:  # Translate the quantile thresholds in absolute terms
                lower_threshold = np.quantile(diffs, thresholds[0])
                upper_threshold = np.quantile(diffs, thresholds[1])
                thresholds = (lower_threshold, upper_threshold)

            filtered_diffs = self.filter_diffs(diffs, thresholds)
            # Reconstruct prices from filtered price differences
            filtered_prices = np.cumsum(filtered_diffs) + prices[0]

            return filtered_prices

        # Calculate based on returns
        rts = np.diff(prices) / prices[:-1]  # Calculate returns
        if thres_in_quantile:  # Translate the quantile thresholds in absolute terms
            lower_threshold = np.quantile(rts, thresholds[0])
            upper_threshold = np.quantile(rts, thresholds[1])
            thresholds = (lower_threshold, upper_threshold)

        filtered_rts = self.filter_diffs(rts, thresholds)
        # Reconstruct prices from the filtered returns
        filtered_prices = np.cumprod(filtered_rts + 1) * prices[0]
        filtered_prices = np.insert(arr=filtered_prices, obj=0, values=prices[0])

        return filtered_prices

    def filter_diffs(self, diffs: np.ndarray, thresholds: Tuple[float], jump_replacement: float = 0) -> np.ndarray:
        """
        Filter the differences for jump adjustment.

        We identify returns falling outside of the given thresholds as jumps, and we replace those by the given amount.
        This method can also handle returns as inputs.

        :param diffs: (np.ndarray) The price differences.
        :param thresholds: (Tuple[float]) The thresholds in the form of (lower thres, upper thres). It is in absolute
            terms.
        :param jump_replacement: (float) Optional. What to replace the jump with. Defaults to 0.
        :return: (np.ndarray) The jump-adjusted returns.
        """

        filtered_diffs = []
        for diff in diffs:
            if thresholds[0] <= diff <= thresholds[1]:
                filtered_diffs.append(diff)
            else:
                filtered_diffs.append(jump_replacement)

        return np.array(filtered_diffs, dtype=float)


class CarteaFigueroaFilter:
    """
    Filter jumps from a mean-reverting process described by Cartea and Figueroa (2005): Pricing in Electricity Markets:
    A Mean RevertingJump Diffusion Model with Seasonality.
    """

    def filter(self, prices: np.ndarray, k_std: float, iteration: int,
               calculate_on_returns: bool = False) -> np.ndarray:
        """
        Filter the jumps from the prices series.

        Any diff in diffs k_std standard dev away from 0 is considered a jump, and will be filtered. As a result, we
        will replace the price diff at that spot by 0. We iterate this process at a given number of times. Then we
        construct the prices series from the jump-filtered diffs series. Here we use diffs by default instead of
        returns because the prices series may come from a spread where non-positive values can be taken, and using
        returns can yield logically inconsistent results. Only if one is sure that the prices are always positive and
        then can use returns.

        :param prices: (np.ndarray) The prices series.
        :param k_std: (float) The number of std to be considered a jump.
        :param iteration: (int) The number of interations we apply for the filtering process.
        :param calculate_on_returns: (bool) Optional. Whether to conduct the filtering based on returns or diff.
            Defaults to False (i.e., use price differences).
        :return: (np.ndarray) The jump-adjusted prices.
        """

        # Calculate based on price differences
        if not calculate_on_returns:
            diffs = np.diff(prices, prepend=0)
            filtered_diffs = self.filter_diffs(diffs, k_std, iteration)
            # Reconstruct prices from filtered price differences
            filtered_prices = np.cumsum(filtered_diffs) + prices[0]

            return filtered_prices

        # Calculate based on returns
        rts = np.diff(prices) / prices[:-1]  # Calculate returns
        filtered_rts = self.filter_diffs(rts, k_std, iteration)
        # Reconstruct prices from the filtered returns
        filtered_prices = np.cumprod(filtered_rts + 1) * prices[0]
        filtered_prices = np.insert(arr=filtered_prices, obj=0, values=prices[0])

        return filtered_prices

    def filter_diffs(self, diffs: np.ndarray, k_std: float, iteration: int, jump_replacement: float = 0) -> np.ndarray:
        """
        Filter the jumps from the price differences.

        Any diff in diffs k_std standard dev away from 0 is considered a jump, and will be filtered. As a result, we
        will replace the price diff at that spot by 0. We iterate this process at a given number of times. This
        function can also handle returns as inputs.

        :param diffs: (np.ndarray) The price differences.
        :param k_std: (float) The number of std to be considered a jump.
        :param iteration: (int) The number of interations we apply for the filtering process.
        :param jump_replacement: (float) Optional. What to replace the jump by. Defaults to 0.
        :return: (np.ndarray) The filtered price diffs that has no jumps.
        """

        for i in range(iteration):
            # compute the std of the absolute difference
            sigma = np.std(np.abs(diffs))
            diffs = self._filter_diffs(diffs, sigma, k_std, jump_replacement)

        return diffs

    @staticmethod
    def _filter_diffs(diffs: np.ndarray, sigma: float, k_std: float, jump_replacement: float) -> np.ndarray:
        """
        Filter the jumps from the price differences.

        Any diff in diffs k_std standard dev away from 0 is considered a jump, and will be filtered. As a result, we
        will replace the price diff at that spot by the given amount.

        :param diffs: (np.ndarray) The price differences.
        :sigma: (float) The standard deviation.
        :param k_std: (float) The number of std to be considered a jump.
        :param jump_replacement: (float) What to replace the jump by.
        :return: (np.ndarray) The filtered price diffs that has no jumps.
        """

        abs_diffs = abs(diffs)  # The absolute values of diffs are compared.
        filtered_diffs = []
        for i, diff in enumerate(diffs):
            if abs_diffs[i] < sigma * k_std:  # Below the threshold, not a jump
                filtered_diffs.append(diff)
            else:  # Above the threshold, considered a jump
                filtered_diffs.append(jump_replacement)

        return np.array(filtered_diffs)

#%% tests, using student-t as daily increments
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    np.random.seed(1)

    diffs = np.random.standard_t(df=2, size=500) * 0.5
    prices = np.cumsum(diffs) + 50
    k_std = 2
    iteration = 3

    cfg = CarteaFigueroaFilter()
    filtered_prices_rts = cfg.filter(prices, k_std, iteration, calculate_on_returns=True)
    filtered_prices_diffs = cfg.filter(prices, k_std, iteration, calculate_on_returns=False)

    plt.figure(dpi=200)
    plt.plot(prices, label='original prices')
    plt.plot(filtered_prices_rts, label='adjusted by returns')
    plt.plot(filtered_prices_diffs, label='adjusted by diffs')
    plt.legend()
    plt.show()

#%% test, using ou-jump as daily increments
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time_series_mc

    np.random.seed(1)

    ouj = time_series_mc.ou_jump_sim.OUJumpSimulator()
    mu_j = lambda t: 0*t  # Jump avg
    g = lambda t: 0*t  # Drift
    n_paths = 2

    simu_paths_overnightjump = ouj.simulate_grid_overnight(
        theta=0.2, g=g, sigma=0.05, lambda_j=0.2, mu_j=mu_j, sigma_j=0.2, dt=1/100, n_points=10000,
        n_paths=n_paths, re_normalize=True, init_condition=0, overnight_dt=1/10, daily_dpoints=10)
    prices = simu_paths_overnightjump[:, 0]

    k_std = 3
    iteration = 1
    cfg = CarteaFigueroaFilter()
    filtered_prices_diffs = cfg.filter(prices, k_std, iteration, calculate_on_returns=False)

    plt.figure(dpi=200)
    plt.plot(prices, label='original prices')
    plt.plot(filtered_prices_diffs, label='adjusted by diffs')
    plt.legend()
    plt.show()

#%% test, using ou-jump as daily increments, SCF
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time_series_mc

    np.random.seed(1)

    ouj = time_series_mc.ou_jump_sim.OUJumpSimulator()
    mu_j = lambda t: 0*t  # Jump avg
    g = lambda t: 0*t  # Drift
    n_paths = 2

    simu_paths_overnightjump = ouj.simulate_grid_overnight(
        theta=0.2, g=g, sigma=0.05, lambda_j=0.2, mu_j=mu_j, sigma_j=0.2, dt=1/100, n_points=10000,
        n_paths=n_paths, re_normalize=True, init_condition=0, overnight_dt=1/10, daily_dpoints=10)
    prices = simu_paths_overnightjump[:, 0]

    scf = StubingerConsistentFilter()
    filtered_prices_diffs = scf.filter(prices, thresholds=(0.02, 0.98), thres_in_quantile=True,
                                       calculate_on_returns=False)

    plt.figure(dpi=200)
    plt.plot(prices, label='original prices')
    plt.plot(filtered_prices_diffs, label='adjusted by diffs')
    plt.legend()
    plt.show()
