"""
Module that fits the jump process parameter from data.
"""

import numpy as np
from typing import Tuple

class FitJumpParam:
    
    def fit_jump_params(self, overnight_diff: np.ndarray, thresholds: Tuple[float],
                        number_of_years: float = None):
        """
        Fit the overnight data to find the jump rate, jump mean and std according to a threshold.
        
        Only above a given threshold will be considered a jump.
        
        :param overnight_data: (np.ndarray) The overnight price difference.
        :param upper_threshold: (float) The cutoff threshold. Above this threshold will be considered a jump.
        """
        
        if number_of_years is None:
            number_of_years = len(overnight_diff) / 252
        
        lower_threshold = thresholds[0]
        upper_threshold = thresholds[1]
        
        subcollection = overnight_diff[
            (overnight_diff >= upper_threshold) | (overnight_diff <= lower_threshold)]
        
        # Estimate the jump intensity
        abs_subcollection = np.abs(subcollection)
        jump_rate = len(subcollection) / len(overnight_diff) * number_of_years
        jump_mean = np.mean(abs_subcollection)
        jump_std = np.std(abs_subcollection)
        
        return jump_rate, jump_mean, jump_std, subcollection
        
    
    @staticmethod
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
    fjp = FitJumpParam()
    # data = np.random.rand(1000)
    # subcollection = fjp._get_subset_from_quantile(data, 0.9)
    overnight_diff = np.random.default_rng().standard_t(df=4, size=20000)
    result = fjp.fit_jump_params(overnight_diff=overnight_diff, thresholds=(-2, 2))
    
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
