"""
Module that fits the jump process parameter from data.
"""

import numpy as np
from typing import Tuple

def fit_jump_params(overnight_diff: np.ndarray, thresholds: Tuple[float],
                    dt: float = 1/252, return_jumps_data: bool = False) -> Tuple[float, np.ndarray]:
    """
    Fit the overnight data to find the jump rate, jump mean and std according to a threshold.

    Only above or below a given threshold will be considered a jump.

    :param overnight_diff: (np.ndarray) The overnight price difference.
    :param thresholds: (Tuple[float]) The cutoff lower and upper thresholds (in absolute terms, not in quantile). Above
        the upper threshold or below the lower threshold will be considered a jump.
    :param dt: (float) Optional. The data frequency given in the overnight_diff. Defaults to 1/252 (Daily).
    :param return_jumps_data: (bool) Optional. Whether we want to return the subcollection of data that is considered
        jumps. Defaults to False.
    :return: (Tuple[float]) The jump rate, jump mean, and jump standard estimation. Also the subcollection of data
        that are considered jumps, if 'return_jumps_data' is set True.
    """

    lower_threshold = thresholds[0]
    upper_threshold = thresholds[1]
    # The subcollections of data considered to be jumps, according to the threshold
    subcollection = overnight_diff[
        (overnight_diff >= upper_threshold) | (overnight_diff <= lower_threshold)]

    # Estimate the jump intensity
    abs_subcollection = np.abs(subcollection)
    jump_rate = len(subcollection) / len(overnight_diff) / dt  # rate = num of jumps per year
    jump_mean = np.mean(subcollection)  # np.mean(abs_subcollection)
    jump_std = np.std(subcollection)  # np.std(abs_subcollection)

    if return_jumps_data:
        return jump_rate, jump_mean, jump_std, subcollection

    return jump_rate, jump_mean, jump_std

def _get_subset_from_quantile(all_diff: np.ndarray, quantile: float = 0.9):
    """
    Get the subcollection of data from top quantiles.

    The all_diff should be 1st order difference of the whole price data, including intraday and overnight.
    """

    data = np.sort(all_diff)  # Sort data in ascending order.
    # The subcollection with the given quantile. Rightside of the cutoff point.
    subcollection = all_diff[int(quantile * len(data)):]
    cutoff = subcollection[0]

    return subcollection, cutoff

#%% Test
if __name__ == '__main__':
    # data = np.random.rand(1000)
    # subcollection = fjp._get_subset_from_quantile(data, 0.9)
    overnight_diff = np.random.default_rng().standard_t(df=4, size=20000)
    result = fit_jump_params(overnight_diff=overnight_diff, thresholds=(-2, 2))

    subcollection = result[3]

    import matplotlib.pyplot as plt

    plt.figure(dpi=150)
    plt.hist(subcollection, density=True)
    plt.title('Hist of the subcollection')
    plt.show()

    plt.figure(dpi=150)
    plt.hist(np.abs(subcollection), density=True)
    plt.title('Hist of the abs subcollection')
    plt.show()
