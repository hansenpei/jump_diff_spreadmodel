"""
Module that fits the jump process parameter from data.
"""

import numpy as np
from typing import Tuple

def fit_jump_params(overnight_diff: np.ndarray, thresholds: Tuple[float],
                    dt: float = 1/252) -> Tuple[float]:
    """
    Fit the overnight data to find the jump rate, jump mean and std according to a threshold.

    Only above or below a given threshold will be considered a jump.

    :param overnight_diff: (np.ndarray) The overnight price difference.
    :param thresholds: (Tuple[float]) The cutoff lower and upper thresholds (in absolute terms, not in quantile). Above
        the upper threshold or below the lower threshold will be considered a jump.
    :param thres_in_quantile: (bool) Optional. Whether the thresholds input are in terms of quantiles. Defaults to
        False.
    :param dt: (float) Optional. The data frequency given in the overnight_diff. Defaults to 1/252 (Daily).
    :return: (Tuple[float]) The jump rate, jump mean, and jump standard estimation.
    """

    lower_threshold = thresholds[0]
    upper_threshold = thresholds[1]
    # The subcollections of data considered to be jumps, according to the threshold
    subcollection = overnight_diff[
        (overnight_diff >= upper_threshold) | (overnight_diff <= lower_threshold)]

    # Estimate the jump intensity
    jump_rate = len(subcollection) / len(overnight_diff) / dt  # rate = num of jumps per year
    jump_mean = np.mean(subcollection)
    jump_std = np.std(subcollection)

    return jump_rate, jump_mean, jump_std

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
