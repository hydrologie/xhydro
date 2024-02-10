
"""
hydro_analysis_tools.py

Functions to compute objective functions and basic statistics from streamflow
time series.

"""
import spotpy as sp
import numpy as np
import pandas as pd


def calculate_all_objective_functions(Qobs, Qsim):
    """
    calculate_all_objective_functions

    INPUTS:
        Qobs -- Observed streamflow (numpy array of [n x 1])
        Qsim -- Simulated streamflow from hydrological modelling
                (numpy array of [n x 1])

    Function to compute multiple objective functions on observed and simulated
    streamflow. The objective functions are obtained from the Spotpy library.
    """

    evals = sp.objectivefunctions.calculate_all_functions(
                Qobs.tolist(), Qsim.tolist())

    # Convert to a dataframe
    obj_fun = pd.DataFrame(evals)

    return obj_fun


def get_basic_stats(Q_TEMP):  # Remove _TEMP upon deploy
    """
    get_basin_stats

    INPUTS:
        Q -- Observed or simulated streamflow time series

    Function to compute basic statistics from a streamflow timeseries.

    WPS should calculate min,mMax and average streamflow for:
        1 - The entire series;
        2 - Values per year (n_years)
        3 - Values per month (12 x n-years);
        4 - Average and standard deviation of all years from point 2 (1 x 1);
        5 - Average and standard deviation of all months from point 3 (12 x 1);
    """
    # Q is a dataframe with index being a date_time_index. Also Contains "flow"

# =============================================================================
# DELETE THIS TEST SECTION BEFORE DEPLOY vvvvvvvvvvvvv
# =============================================================================
    # Make test dataframe with datetime as index:
    times = pd.date_range('2012-10-01', periods=1000, freq='D')
    # Delete for deploy
    Q = pd.DataFrame(np.random.rand(1000, 1), index=times, columns=['flow'])
# =============================================================================
# DELETE THIS TEST SECTION BEFORE DEPLOY ^^^^^^^^^^^^
# =============================================================================

    # Add columns in the dataframe corresponding to year and month (for stats)
    Q['year'] = pd.Series(Q.index.year, index=Q.index)
    Q['month'] = pd.Series(Q.index.month, index=Q.index)

    # Map seasons -- seasons = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 1]
    seasons = [
           'winter',  # 1
           'winter',  # 1
           'spring',  # 2
           'spring',  # 2
           'spring',  # 2
           'summer',  # 3
           'summer',  # 3
           'summer',  # 3
           'fall',    # 4
           'fall',    # 4
           'fall',    # 4
           'winter'   # 1
    ]

    month_to_season = dict(zip(range(1, 13), seasons))
    Q['season'] = Q.index.month.map(month_to_season)

    # Initialize dict. Maybe another system could be use, but it should be as
    # flexible as this methodology.
    stats = {}

    # Overall statistics
    stats["overall_max"] = Q["flow"].max()
    stats["overall_min"] = Q["flow"].min()
    stats["overall_mean"] = Q["flow"].mean()
    stats["overall_STD"] = Q["flow"].std()

    # Yearly statistics
    g = Q.groupby(["year"])
    stats["year_max"] = g["flow"].max()
    stats["year_min"] = g["flow"].min()
    stats["year_mean"] = g["flow"].mean()
    stats["year_STD"] = g["flow"].std()

    # Monthly statistics
    g = Q.groupby(["year", "month"])
    stats["month_max"] = g["flow"].max()
    stats["month_min"] = g["flow"].min()
    stats["month_mean"] = g["flow"].mean()
    g = Q.groupby(["month"])
    stats["month_STD"] = g["flow"].std()
    stats["avg_per_month"] = g["flow"].mean()

    # Seasonal statistics
    g = Q.groupby(["season"])
    stats["season_max"] = g["flow"].max()
    stats["season_min"] = g["flow"].min()
    stats["season_mean"] = g["flow"].mean()
    stats["season_STD"] = g["flow"].std()

    # Write dict to textfile. Maybe this can be upgraded to NetCDF later.
    with open('basic_hydrological_statistics.txt', 'w') as f:
        print(stats, file=f)

    # Statistics could be fed to the hydrograph plotting tools
    return stats


if __name__ == '__main__':
    ggg = get_basic_stats(1)
