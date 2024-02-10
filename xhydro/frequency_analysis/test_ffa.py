# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 10:04:19 2019

@author: jlmartel
"""

# -*- coding: utf-8 -*-
"""
Library to perform flood frequency analysis of flood or low flow annual maxima
series (AMS).
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path


def test_read_raingauge_file(file_path,
                        file_type='tsf',
                        ts_format='intensity',
                        time_step=5,
                        duration=1440):
    """
    read_flow_file

    INPUTS:
        file_name -- Name with extension of the file to read
        ts_format -- Format of the time series: either 'height' or 'intensity'

    Extraction of daily streamflow data with their corresponding dates from a
    Excel spreadsheet (.xlsx) or netCDF file (.nc).

    """

    df = pd.DataFrame()
    if file_type == 'tsf':  # PCSWMM tsf file
        df = pd.read_csv(file_path,
                         sep='\t',
                         skiprows=3,
                         names=['dates', ts_format],
                         engine='python')

    elif file_type == 'xlsx':  # Excel spreadsheet
        pass
    elif file_type == 'nc':  # netCDF file
        pass

    df['dates'] = pd.to_datetime(df['dates'])  # convert to timestamps

    # Compute intensity and height of the time series
    if ts_format == 'intensity':
        df['height'] = df['intensity'] / (60 / time_step)
    elif ts_format == 'height':
        df['intensity'] = df['height'] * (60 / time_step)
    else:
        e = 'The time series format {} is invalid.'.format(ts_format)
        raise ValueError(e)

    # Compute the rolling sum that will be used to select events
    df['rolling_sum_24h'] = df['height'].rolling(int(24*60 / time_step)).sum()
    df['rolling_sum_12h'] = df['height'].rolling(int(12*60 / time_step)).sum()
    df['rolling_sum_6h'] = df['height'].rolling(int(6*60 / time_step)).sum()
    df['rolling_sum_3h'] = df['height'].rolling(int(3*60 / time_step)).sum()
    df['rolling_sum_1h'] = df['height'].rolling(int(60 / time_step)).sum()

    return df


def extract_maxima_series(df,
                          var_name,
                          time_period='year',
                          extreme_type='high'):
    """
    extract_AMS

    INPUTS:
        df -- Streamflow (pandas dataframe of [n x 2] with dates and flow)
        time_period -- Time period of to extract the maxima series. Either
                       year, season (W or winter, Sp or spring, Su or summer,
                       A or Autumn) or a selected month (1 (January) to 12
                       (December))
        extreme_type -- Either 'high' for floods or 'low' for low flows

    Extraction of the annual maxima series from a pandas dataframe.
    """

    if extreme_type == 'high':  # Flood values
        if str(time_period).lower() == 'year':
            x_max = df.resample('Y', on='dates').agg({var_name: 'max'})[var_name]

        elif str(time_period).lower() == 'winter' or \
                str(time_period).upper() == 'W':
            trimester_max = df.resample('QS-Dec', on='dates').\
                                agg({var_name: 'max'})[var_name]
            x_max = trimester_max.iloc[0::4]
        elif str(time_period).lower() == 'spring' or \
                str(time_period).upper() == 'SP':
            trimester_max = df.resample('QS-Dec', on='dates').\
                                agg({var_name: 'max'})[var_name]
            x_max = trimester_max.iloc[1::4]
        elif str(time_period).lower() == 'summer' or \
                str(time_period).upper() == 'SU':
            trimester_max = df.resample('QS-Dec', on='dates').\
                                agg({var_name: 'max'})[var_name]
            x_max = trimester_max.iloc[2::4]
        elif str(time_period).lower() == 'autumn' or \
                str(time_period).upper() == 'A':
            trimester_max = df.resample('QS-Dec', on='dates').\
                                agg({var_name: 'max'})[var_name]
            x_max = trimester_max.iloc[3::4]
        elif time_period in range(1, 13):  # Selected month
            month_max = df.resample('M', on='dates').agg({var_name: 'max'})[var_name]
            x_max = month_max.iloc[time_period-1::12]
        else:
            e = 'Time period {} is not supported for the frequency \
                analysis. See help for accepted format.' \
                .format(str(time_period))
            raise ValueError(e)

        return x_max

    elif extreme_type == 'low':  # Low flow values
        if str(time_period).lower() == 'year':
            x_min = df.resample('Y', on='dates').agg({var_name: 'min'})[var_name]

        elif str(time_period).lower() == 'winter' or \
                str(time_period).upper() == 'W':
            trimester_max = df.resample('QS-Dec', on='dates').\
                                agg({var_name: 'min'})[var_name]
            x_min = trimester_max.iloc[0::4]
        elif str(time_period).lower() == 'spring' or \
                str(time_period).upper() == 'SP':
            trimester_max = df.resample('QS-Dec', on='dates').\
                                agg({var_name: 'min'})[var_name]
            x_min = trimester_max.iloc[1::4]
        elif str(time_period).lower() == 'summer' or \
                str(time_period).upper() == 'SU':
            trimester_max = df.resample('QS-Dec', on='dates').\
                                agg({var_name: 'min'})[var_name]
            x_min = trimester_max.iloc[2::4]
        elif str(time_period).lower() == 'autumn' or \
                str(time_period).upper() == 'A':
            trimester_max = df.resample('QS-Dec', on='dates').\
                                agg({var_name: 'min'})[var_name]
            x_min = trimester_max.iloc[3::4]
        elif time_period in range(1, 13):  # Selected month
            month_max = df.resample('M', on='dates').agg({var_name: 'min'})[var_name]
            x_min = month_max.iloc[time_period-1::12]
        else:
            e = 'Time period {} is not supported for the frequency \
                analysis. See help for accepted format.'.format(time_period)
            raise ValueError(e)

        return x_min

    else:
        e = 'Extreme type {} is not supported for the frequency \
            analysis.'.format(extreme_type)
        raise ValueError(e)


def frequency_analysis_get_X(data,
                             dist='genextreme',
                             extreme_type='high',
                             T=20):
    """
    frequency_analysis_get_X

    INPUTS:
        data -- Streamflow maxima series used for the flood frequency analysis
        dist -- Statistical distribution for the ffa. Default is Gumbel
        T -- Array or single value of the desired return period to extract

    Flood frequency analysis (FFA) of the steamflow maxima time series. In this
    version, only the Gamma ('gamma'), GEV ('genextreme'), Gumbel ('gumbel_r'),
    Log-normal ('lognorm') and Pearson-III ('pearson3') distributions are made
    available for the FFA.
    """

    if extreme_type == 'high':
        p = 1 - 1 / T  # Probability of non-occurence for floods
    elif extreme_type == 'low':
        p = 1 / T  # Probability of non-occurence for low flows
    else:
        e = 'Extreme type {} is not supported for the flood frequency \
            analysis.'.format(extreme_type)
        raise ValueError(e)

    supported_dist = ['gamma',
                      'genextreme',
                      'gumbel_r',
                      'lognorm',
                      'pearson3']

    # Get the distribution object
    dist_obj = getattr(stats, dist.lower(), None)
    if dist not in supported_dist or dist_obj is None:
        e = "Statistical distribution {} is not in not supported for the \
            flood frequency analysis.".format(dist)
        raise ValueError(e)

    parmhat = dist_obj.fit(data)  # Fit the statistical distribution parameters
    if len(parmhat) == 2:  # 2-parameter distribution
        X = dist_obj.ppf(p, loc=parmhat[0], scale=parmhat[1])
    else:  # Expecting a 3-parameter distribution
        X = dist_obj.ppf(p, parmhat[0], loc=parmhat[1], scale=parmhat[2])

    return X


def frequency_analysis_get_T(data, X, dist='gumbel_r'):
    """
    frequency_analysis_get_T

    INPUTS:
        data -- Streamflow maxima series used for the flood frequency analysis
        dist -- Statistical distribution for the ffa. Default is Gumbel
        T -- Array or single value of the desired return period to extract

    Flood frequency analysis (FFA) of the steamflow maxima time series. In this
    version, only the Gamma ('gamma'), GEV ('genextreme'), Gumbel ('gumbel_r'),
    Log-normal ('lognorm') and Pearson-III ('pearson3') distributions are made
    available for the FFA.
    """

    supported_dist = ['gamma',
                      'genextreme',
                      'gumbel_r',
                      'lognorm',
                      'pearson3']

    # Get the distribution object
    dist_obj = getattr(stats, dist.lower(), None)
    if dist not in supported_dist or dist_obj is None:
        e = "Statistical distribution {} is not in not supported for the \
            flood frequency analysis.".format(dist)
        raise ValueError(e)

    parmhat = dist_obj.fit(data)  # Fit the statistical distribution parameters
    if len(parmhat) == 2:  # 2-parameter distribution
        p = dist_obj.sf(X, loc=parmhat[0], scale=parmhat[1])
    else:  # Expecting a 3-parameter distribution
        p = dist_obj.sf(X, parmhat[0], loc=parmhat[1], scale=parmhat[2])

    T = 1/p

    return T


if __name__ == '__main__':

    ts_format = 'intensity'  # either 'intensity' or 'height'
    time_step = 5  # either 5, 15 or 60 (in minutes)

    # Data folder where to find the raingauge data
    data_folder = r'S:\Etudes\!_PROJETS_ACTIFS\008002811_Mtl_GuideHydrologie'\
                  r'\6_Calculs_Analyses\Pluviomètres\3_TraitementPCSWMM'

    # File name of the raingauge data to use
    gauge_id = 'OBXL1111'
#    file_name = 'OBXL1271_sans_calib_PCSWMM_2017_2018.tsf'
    file_name = 'OBXL1111_sans_calib_PCSWMM.tsf'

    file_path = Path(data_folder, file_name)  # path to the file
    file_type = file_name.split('.')[-1]  # .tsf or .xlsx

    # Dataframe of the time series
    df_ts = read_raingauge_file(file_path=file_path,
                                file_type=file_type,
                                ts_format=ts_format,
                                time_step=time_step,
                                duration=1440)

    extreme_type = 'high'
    time_period = 'year'
    dist = 'gumbel_r'
#
    Px_24h = extract_maxima_series(df=df_ts,
                                   var_name='rolling_sum_24h',
                                   time_period=time_period,
                                   extreme_type=extreme_type)
    Px_12h = extract_maxima_series(df=df_ts,
                                   var_name='rolling_sum_12h',
                                   time_period=time_period,
                                   extreme_type=extreme_type)
    Px_6h = extract_maxima_series(df=df_ts,
                                  var_name='rolling_sum_6h',
                                  time_period=time_period,
                                  extreme_type=extreme_type)
    Px_3h = extract_maxima_series(df=df_ts,
                                  var_name='rolling_sum_3h',
                                  time_period=time_period,
                                  extreme_type=extreme_type)
    Px_1h = extract_maxima_series(df=df_ts,
                                  var_name='rolling_sum_1h',
                                  time_period=time_period,
                                  extreme_type=extreme_type)

    T = frequency_analysis_get_T(data=Px_3h, X=30, dist=dist)
    print('T = ' + str(T) + ' and X = ' + str(30))

    T = frequency_analysis_get_T(data=Px_6h, X=15, dist=dist)
    print('T = ' + str(T) + ' and X = ' + str(15))

    T = frequency_analysis_get_T(data=Px_1h, X=11, dist=dist)
    print('T = ' + str(T) + ' and X = ' + str(11))

    T = frequency_analysis_get_T(data=Px_12h, X=20, dist=dist)
    print('T = ' + str(T) + ' and X = ' + str(20))

#
#    counter = 0
#    for value in T:
#        print(f'The {value}-year return period is: {X[counter]:.2f} CMS')
#        counter += 1
