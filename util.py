"""
Ultility functions library.
"""

from typing import Union, List, Callable
import numpy as np
import datetime

def normalize_data(data_matrix: np.ndarray, act_on: str = 'row') -> np.ndarray:
    """
    Normalize the data matrix acting on either the row or the column: X <- (X - mean)/std
    
    :param data_matrix: (np.ndarray) The matrix that stores data to be normalized.
    :param act_on: (str) Optional. Taking average and std over either all rows or all columns, can only take value from
    {'col', 'row'}. For example, if choosing 'col', then we normalize each point by using the mean and std calculated
    from that column. Defaults to 'row'.
    :return: (np.ndarray) Normalized data.
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
        n_days: int = 10, active_time: tuple = ('0930', '1600'), intraday_freq: Union[str, int] = '5m',
        output_unit: str = 'y') -> np.ndarray:
    """
    Generate a 1D timetable with a given intraday interval and market active time.
    
    Do not accoynt for holidays or weekends.
    
    :param n_days: (int) Optional. The number of days for our time table. Defaults to 10.
    :param active_time: (tuple) Optinal. The active time for the market. Input should follow strictly in the military
        time form in strings. Defaults to the US stocks market time ('0930', '1600'), with 390 minutes active, or 391
        sampling points per day.
    :param intraday_freq: (Union[str, int]) Optional. Intraday frequency, can take value in string format from values
        {'10s', '30s', '1m', '5m', '10m', '15m', '30m', '1h'}, or an arbitrary integer that specifies the number of
        intraday data needed. Defaults to '5min'.
    :param output_unit: (str) Optional. Specify the output time unit. Can take value from {'s', 'd', 'y'} for second,
        day and year. Defalts to 'y'.
    :return: (np.ndarray) The 1D timetable.
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

def generate_jump_timetable_grid(
        jump_rate: float = 25, dt: float = 1/98532, n_points: int = 3910, n_paths: int = 10,
        return_just_time: bool = True) -> List[list]:
    """
    Generate a time table for jumps that occur as a Poission process with a given jump rate.
    
    The function evaluates whether a jump will occur between dt as follows (approximate by Bernoulli):
        True with probability jump_rate*dt, False with probability (1 - jump_rate*dt).

    The return format can be either just the time stamp where the jump does occur, i.e., [0.1, 0.145, 0.52], or can be
    the size of the full time table but with binary terms where 0 indicates no jump and 1 indicates a jump that does
    occur, i.e., [F, F, F, T, F, F, F, T, F].
    
    Also this function does not account for overnight time. One can treat the time table generated as a subset when
    the market is active.
    
    :param jump_rate:(float) Optional. Jump rate (and equivalently the average jump number per unit time). Defaults to
        25 (i.e., 1 jump per 10 trading days using year as unit).
    :param dt:(float) Optional. Time advancement step. Defaults to 1/98532 = 1/252/391, which is 1 minute.
    :param n_points:(int) Optional. The number of time advancement of size dt. Defaults to 3910. Combined with the
        default dt this is 10 days.
    :param n_paths:(int) Optional. The number of paths. Defaults to 10.
    :param return_just_time:(bool) Optional. Whether to return just the time where the jump occurs or the entire
        boolean table for the whole time grid. Defaults to True (just the time where the jump occurs).
    """

    # Approximate the Poission counting process as Bernoulli:
    # Success probability for each jump is jump_rate*dt
    jump_bool = np.random.rand(n_points, n_paths) < (jump_rate*dt)  # Jump cannot occur at time 0
    
    if return_just_time:
        # Initialize the timetable. In this case it is a grid.
        total_time = dt * n_points
        timetable = np.arange(start=0, stop=total_time, step=dt)
        # Fancy indexing
        jump_timestamp = []
        for path_number in range(n_paths):
            jump_timestamp.append(timetable[jump_bool[:, path_number]])
        # Return the time stamp at which the jump occurs
        return jump_timestamp
    
    # Return the boolean table at which the jump happens with the dimension as n_points * n_paths
    return jump_bool

def generate_jump_timetable_overnight(
        jump_rate: float = 25, overnight_dt: float = 1/252, n_points: int = 252, n_paths: int = 10,
        return_just_date: bool = True) -> List[list]:
    """
    Generate a time table for overnight jumps that occur as a Poission process with a given jump rate.
    
    The function evaluates whether a jump will occur between dt as follows (approximate by Bernoulli):
        True with probability jump_rate*dt, False with probability (1 - jump_rate*overnight*dt).
    The return format can be either just the time stamp where the jump does occur, i.e., [1, 5, 8], or can be
    the size of the full time table but with binary terms where 0 indicates no jump and 1 indicates a jump that does
    occur, i.e., [F, F, F, T, F, F, F, T, F].

    :param jump_rate:(float) Optional. Jump rate (and equivalently the average jump number per unit time). Defaults to
        25 (i.e., 1 jump per 10 trading days using year as unit).
    :param overnight_dt:(float) Optional. Time advancement step for overnight time. Defaults to 1/252, which is 1 day.
        Note that when accounted for all inactive hours, 1/252 is close to the average of interday time.
    :param n_points:(int) Optional. The number of time advancement of size dt. Defaults to 252. Combined with the
        default dt this is 1 year.
    :param n_paths:(int) Optional. The number of paths. Defaults to 10.
    :param return_just_date:(bool) Optional. Whether to return just the date where the jump occurs or the entire
        boolean table for the whole time grid. Defaults to True (just the time where the jump occurs).
    """
    
    jump_bool = np.random.rand(n_points, n_paths) < (jump_rate*overnight_dt)  # Jump cannot occur at time 0
    
    if return_just_date:
        jump_dates = [np.where(jump_bool[:, path_idx])[0] for path_idx in range(n_paths)]
        
        return jump_dates
    
    # Return the boolean table at which the jump happens with the dimension as n_points * n_paths
    return jump_bool

def generate_jump_size(jump_timestamp: List[np.ndarray], mu_j: Callable[[float], float], sigma_j: float) -> List[list]:
    """
    Generate jump size accoring to a timestamp for each jump.
    
    The jump size is distributed according to Normal(mu_j(t), sigma_j*sigma_j).
    
    :param jump_timestamp: (List[np.ndarray]) The time at which the jump occurs. Dimension is (n_paths, n_jumps)
    :param mu_j: (Callable[[float], float]) The moving average function of jump size.
    :param sigma_j: (float) The jump size standard deviation.
    :return: (List[list]) The list of jump size generated for each path. Dimension is (n_paths, n_jumps)
    """
    
    jump_size = []
    # Within each path
    for each_timestamp in jump_timestamp:
        each_jump_size = []
        for ith_time in each_timestamp:  # At each time where the jump occurs, draw the intensity from normal dist
            each_jump_size.append(np.random.normal(mu_j(ith_time), scale=sigma_j))

        jump_size.append(each_jump_size)

    return jump_size



#%% Tests
if __name__ == '__main__':
    # timetable = generate_timetable(n_days=10, output_unit='d')
    # jptable = generate_jump_timetable_grid(jump_rate=1, dt=1/10, n_points=200, n_paths=10, return_just_time=True)
    jump_dates = generate_jump_timetable_overnight(return_just_date=True)
    