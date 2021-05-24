# -*- coding: utf-8 -*-
"""
Ultility functions library.
"""

from typing import Union
import numpy as np
import datetime

def normalize_data(data_matrix: np.ndarray, act_on: str = 'col') -> np.ndarray:
    """
    Normalize the data matrix acting on either the row or the column.
    """
    if act_on not in {'row', 'col'}:
        raise ValueError("act_on can only take 'row' or 'col' as inputs.")
    
    # Normalize per column.
    if act_on == 'row':
        stds_per_point = np.std(data_matrix, axis=1)
        means_per_point = np.mean(data_matrix, axis=1)
        stds_per_point = stds_per_point[:, np.newaxis]
        means_per_point = means_per_point[:, np.newaxis]
        output = (data_matrix - means_per_point) / stds_per_point

    # Normalize per row.
    if act_on == 'col':
        stds_per_point = np.std(data_matrix, axis=0)
        means_per_point = np.mean(data_matrix, axis=0)
        output = (data_matrix - means_per_point) / stds_per_point
    
    return output

def generate_timetable(
        n_days: int = 10, active_time: tuple = ('0930', '1600'), intraday_freq: Union[str, int] = '5min',
        output_unit: str = 'y') -> np.ndarray:
    """
    Generate a simple timetable with a given intraday interval, to be used for simulation.
    
    390 minutes per trading day on average.
    intraday_freq takes value from {'10s', '30s', '1min', '5min', '10min', '15min', '30m', '1h'} for string inputs.
    """
    
    # 1. Handle the intraday_freq input
    # Calculate the number of minutes per day
    market_start = datetime.time(hour=int(active_time[0][0: 2]), minute=int(active_time[0][2: 4]))
    market_close = datetime.time(hour=int(active_time[1][0: 2]), minute=int(active_time[1][2: 4]))
    delta_t = datetime.timedelta(hours = (market_close.hour - market_start.hour),
                                 minutes = (market_close.minute - market_start.minute))
    seconds_per_day = delta_t.seconds  # Number of seconds per day
    
    if isinstance(intraday_freq, str):  # If the input is string
        # Compute the number of samples per day according to input
        freq_to_n = {'10s': seconds_per_day // 10, '30s': seconds_per_day // 30, '1m': seconds_per_day // 60,
                     '5m': seconds_per_day // 300, '10m': seconds_per_day // 600, '15m': seconds_per_day // 900,
                     '30m': seconds_per_day // 1800}
        n_samples_per_day = freq_to_n[intraday_freq] + 1
        # Compute the intraday sampling interval dt in seconds
        freq_to_dt = {'10s': 10, '30s': 30, '1m': 60, '5m': 300, '10m': 600, '15m': 900, '30m': 1800}
        dt = freq_to_dt[intraday_freq]
        
    elif isinstance(intraday_freq, int):  # If the sampling frequency is given as an integer
        n_samples_per_day = intraday_freq
        dt = seconds_per_day // (n_samples_per_day - 1)
    
    # 2. Calculate the time table with unit in year, setting the origin at the initial day's trading hour
    n_samples = n_days * n_samples_per_day  # Number of total sample points throughout the time span
    time_table = np.zeros(n_samples)
    day_count = -1  # Count the number of days passed
    for i in range(n_samples):
        if (i) % n_samples_per_day == 0:
            day_count += 1  # Increment the number of days passed
        # Calculate the time table value in seconds
        time_table[i] = (i % n_samples_per_day) * dt + day_count * 86400
    
    # Translate time_table to units wanted
    unit_to_seconds = {'s': 1, 'd': 86400, 'y': 86400 * 252}  # Units in second, day, year respectively.
    time_table = time_table / unit_to_seconds[output_unit]
    
    return time_table

#%%
if __name__ == '__main__':
    timetable = generate_timetable(n_days=10, output_unit='d')