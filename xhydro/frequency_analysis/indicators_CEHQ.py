# -*- coding: utf-8 -*-
"""
28 statistical indicators in the 2018 Hydroclimatic Atlas of the Centre
d'expertise hydrique du Québec (CEHQ) that typically caracterize winter and
spring, on the one hand, and summer and fall on the other hand.

Glossary:
- Flood: Period of high flow.
- Low flow: Period of low flow.
- Hydraulicity: Average value of flows over long periods (month, season, year).
- Recurrence: Long-term statistical evaluation of the mean tiem interval
  between two hydrological events of a given intensity.
"""

import numpy as np
import pandas as pd

from ffa_tools import flood_frequency_analysis as ffa


def Q1max2Sp(df, dist='gumbel_r'):
    """
    Q1max2Sp

    INPUTS:
        df -- Streamflow (pandas dataframe of [n x 2] with dates and flow)
        dist -- Statistical distribution of extreme value. Default is Gumbel

    DESCRIPTION:
        - Indicator: Q1max2Sp
        - Hydrological phenomenon: Spring floods
        - Question: In future climate, will spring flood peaks be higher?
        - Description: Annual maximum [max] daily flow [Q1] of 2-year
          recurrence [2] in spring [Sp].
    """
    trimester_max = df.resample('QS-Dec', on='dates').agg({'Q': 'max'})['Q']
    Sp_max = trimester_max.iloc[1::4]
    Q1max2Sp = ffa(Sp_max, dist=dist, T=2)
    return Q1max2Sp


def Q1max20Sp(df, dist='gumbel_r'):
    """
    Q1max20Sp

    INPUTS:
        df -- Streamflow (pandas dataframe of [n x 2] with dates and flow)
        dist -- Statistical distribution of extreme value. Default is Gumbel

    DESCRIPTION:
        - Indicator: Q1max20Sp
        - Hydrological phenomenon: Spring floods
        - Question: In future climate, will spring flood peaks be higher?
        - Description: Annual maximum [max] daily flow [Q1] of 20-year
          recurrence [20] in spring [Sp].
    """
    trimester_max = df.resample('QS-Dec', on='dates').agg({'Q': 'max'})['Q']
    Sp_max = trimester_max.iloc[1::4]
    Q1max20Sp = ffa(Sp_max, dist=dist, T=20)
    return Q1max20Sp


def Q14max2Sp(df, dist='gumbel_r'):
    """
    Q14max2Sp

    INPUTS:
        df -- Streamflow (pandas dataframe of [n x 2] with dates and flow)
        dist -- Statistical distribution of extreme value. Default is Gumbel

    DESCRIPTION:
        - Indicator: Q14max2spr
        - Hydrological phenomenon: Spring floods
        - Question: In future climate, will spring flood volumes be higher?
        - Description: Annual 14-day flow average [Q14] of 2-year recurrence
          [2] in spring [Sp].
    """
    df['Q14'] = df.Q.rolling(14).mean()
    trimester_max = df.resample('QS-Dec', on='dates').agg({'Q14': 'max'})['Q14']
    Sp_max = trimester_max.iloc[1::4]
    Q14max2Sp = ffa(Sp_max, dist=dist, T=2)
    return Q14max2Sp


def Q14max20Sp(df, dist='gumbel_r'):
    """
    Q14max20Sp

    INPUTS:
        df -- Streamflow (pandas dataframe of [n x 2] with dates and flow)
        dist -- Statistical distribution of extreme value. Default is Gumbel

    DESCRIPTION:
        - Indicator: Q14max20Sp
        - Hydrological phenomenon: Spring floods
        - Question: In future climate, will spring flood volumes be higher?
        - Description: Annual 14-day flow average [Q14] of 20-year recurrence
          [20] in spring [Sp].
    """
    df['Q14'] = df.Q.rolling(14).mean()
    trimester_max = df.resample('QS-Dec', on='dates').agg({'Q14': 'max'})['Q14']
    Sp_max = trimester_max.iloc[1::4]
    Q14max20Sp = ffa(Sp_max, dist=dist, T=20)
    return Q14max20Sp


def D_Q1maxSp(df):
    """
    D_Q1maxSp

    INPUTS:
        df -- Streamflow (pandas dataframe of [n x 2] with dates and flow)
        dist -- Statistical distribution of extreme value. Default is Gumbel

    DESCRIPTION:
        - Indicator: D_Q1maxSp
        - Hydrological phenomenon: Spring floods
        - Question: In future climate, wll the spring floods occur earlier?
        - Description: Average day of occurence [D] of the annual maximum daily
          flow [Q1] in spring [Sp].
    """
    idxmax_Q1max = df.resample('QS-Dec', on='dates').agg({'Q': 'idxmax'})['Q']
    D_Q1maxSp = idxmax_Q1max.iloc[1::4].mod(365).mean()
    return round(D_Q1maxSp)


def Q1max2SuA(df, dist='gumbel_r'):
    """
    Q1max2SuA

    INPUTS:
        df -- Streamflow (pandas dataframe of [n x 2] with dates and flow)
        dist -- Statistical distribution of extreme value. Default is Gumbel

    DESCRIPTION:
        - Indicator: Q1max2SuA
        - Hydrological phenomenon: Summer and autumn floods
        - Question: In future climate, will the summer and autumn flood peaks
          be higher?
        - Description: Annual maximum [max] daily flow [Q1] of 2-year
          recurrence [2] in summer and autumn [SuA].
    """
    semester_max = df.resample('2QS-Dec', on='dates').agg({'Q': 'max'})['Q']
    SuA_max = semester_max.iloc[1::2]
    Q1max2SuA = ffa(SuA_max, dist='gumbel_r', T=2)
    return Q1max2SuA


def Q1max20SuA(df, dist='gumbel_r'):
    """
    Q1max20SuA

    INPUTS:
        df -- Streamflow (pandas dataframe of [n x 2] with dates and flow)
        dist -- Statistical distribution of extreme value. Default is Gumbel

    DESCRIPTION:
        - Indicator: Q1max2SuA
        - Hydrological phenomenon: Summer and autumn floods
        - Question: In future climate, will the summer and autumn flood peaks
          be higher?
        - Description: Annual maximum [max] daily flow [Q1] of 20-year
          recurrence [20] in summer and autumn [SuA].
    """
    semester_max = df.resample('2QS-Dec', on='dates').agg({'Q': 'max'})['Q']
    SuA_max = semester_max.iloc[1::2]
    Q1max20SuA = ffa(SuA_max, dist=dist, T=20)
    return Q1max20SuA


def Q7min2Su(df, dist='gumbel_r'):
    """
    Q7min2Su

    INPUTS:
        df -- Streamflow (pandas dataframe of [n x 2] with dates and flow)
        dist -- Statistical distribution of extreme value. Default is Gumbel

    DESCRIPTION:
        - Indicator: Q7min2Su
        - Hydrological phenomenon: Summer low flows
        - Question: In future climate, will summer low flows be more severe?
        - Description: Average annual minimum [min] 7-day flow [Q7] of 2-year
          recurrence [2] in summer [Su].
    """
    df['Q7'] = df.Q.rolling(7).mean()
    trimester_min = df.resample('QS-Dec', on='dates').agg({'Q7': 'min'})['Q7']
    Su_min = trimester_min.iloc[2::4]
    Q7min2Su = ffa(Su_min, dist=dist, T=2)
    return Q7min2Su

    
def Q7min10Su(df, dist='gumbel_r'):
    """
    Q7min10Su

    INPUTS:
        df -- Streamflow (pandas dataframe of [n x 2] with dates and flow)
        dist -- Statistical distribution of extreme value. Default is Gumbel

    DESCRIPTION:
        - Indicator: Q7min10Su
        - Hydrological phenomenon: Summer low flows
        - Question: In future climate, will summer low flows be more severe?
        - Description: Average annual minimum [min] 7-day flow [Q7] of 10-year
          recurrence [10] in summer [Su].
    """
    df['Q7'] = df.Q.rolling(7).mean()
    trimester_min = df.resample('QS-Dec', on='dates').agg({'Q7': 'min'})['Q7']
    Su_min = trimester_min.iloc[2::4]
    Q7min10Su = ffa(Su_min, dist='gumbel_r', T=10)
    return Q7min10Su


def Q30min5Su(df, dist='gumbel_r'):
    """
    Q30min5Su

    INPUTS:
        df -- Streamflow (pandas dataframe of [n x 2] with dates and flow)
        dist -- Statistical distribution of extreme value. Default is Gumbel

    DESCRIPTION:
        - Indicator: Q30min5Su
        - Hydrological phenomenon: Summer low flows
        - Question: In future climate, will summer low flows be more severe?
        - Description: Average annual minimum [min] 30-day flow [Q30] of 5-year
          recurrence [5] in summer [Su].
    """
    df['Q30'] = df.Q.rolling(30).mean()
    trimester_min = df.resample('QS-Dec', on='dates').agg({'Q30': 'min'})['Q30']
    Su_min = trimester_min.iloc[2::4]
    Q30min5Su = ffa(Su_min, dist=dist, T=5)
    return Q30min5Su


def Q7min2W(df, dist='gumbel_r'):
    """
    Q7min2W

    INPUTS:
        df -- Streamflow (pandas dataframe of [n x 2] with dates and flow)
        dist -- Statistical distribution of extreme value. Default is Gumbel

    DESCRIPTION:
        - Indicator: Q7min2W
        - Hydrological phenomenon: Winter low flows
        - Question: In future climate, will winter low flows be more severe?
        - Description: Average annual minimum [min] 7-day flow [Q7] of 2-year
          recurrence [2] in winter [W].
    """
    df['Q7'] = df.Q.rolling(7).mean()
    trimester_min = df.resample('QS-Dec', on='dates').agg({'Q7': 'min'})['Q7']
    W_min = trimester_min.iloc[0::4]
    Q7min2W = ffa(W_min, dist=dist, T=2)
    return Q7min2W


def Q7min10W(df, dist='gumbel_r'):
    """
    Q7min10W

    INPUTS:
        df -- Streamflow (pandas dataframe of [n x 2] with dates and flow)
        dist -- Statistical distribution of extreme value. Default is Gumbel

    DESCRIPTION:
        - Indicator: Q7min10W
        - Hydrological phenomenon: Winter low flows
        - Question: In future climate, will winter low flows be more severe?
        - Description: Average annual minimum [min] 7-day flow [Q7] of 10-year
          recurrence [10] in winter [W].
    """
    df['Q7'] = df.Q.rolling(7).mean()
    trimester_min = df.resample('QS-Dec', on='dates').agg({'Q7': 'min'})['Q7']
    W_min = trimester_min.iloc[0::4]
    Q7min10W = ffa(W_min, dist=dist, T=10)
    return Q7min10W


def Q30min5W(df, dist='gumbel_r'):
    """
    Q30min5W

    INPUTS:
        df -- Streamflow (pandas dataframe of [n x 2] with dates and flow)
        dist -- Statistical distribution of extreme value. Default is Gumbel

    DESCRIPTION:
        - Indicator: Q30min5W
        - Hydrological phenomenon: Winter low flows
        - Question: In future climate, will winter low flows be more severe?
        - Description: Average annual minimum [min] 30-day flow [Q30] of 5-year
          recurrence [5] in winter [W].
    """
    df['Q30'] = df.Q.rolling(30).mean()
    trimester_min = df.resample('QS-Dec', on='dates').agg({'Q30': 'min'})['Q30']
    W_min = trimester_min.iloc[0::4]
    Q30min5W = ffa(W_min, dist=dist, T=5)
    return Q30min5W


def Qavg(df):
    """
    Qavg

    INPUTS:
        df -- Streamflow (pandas dataframe of [n x 2] with dates and flow)

    DESCRIPTION:
    - Indicator: Qavg
    - Hydrological phenomenon: Hydraulicity
    - Question: In future climate, will hydraulicity be modified?
    - Description: Average annual flow [Qavg].
    """
    Qavg = df.resample('Y', on='dates').agg({'Q': 'mean'})['Q'].mean()
    return Qavg


def QavgWSp(df):
    """
    QavgWSp

    INPUTS:
        df -- Streamflow (pandas dataframe of [n x 2] with dates and flow)

    DESCRIPTION:
        - Indicator: QavgWSp
        - Hydrological phenomenon: Hydraulicity
        - Question: In future climate, will hydraulicity be modified?
        - Description: Average annual flow [Qavg] for the winter-spring [WSp]
          period.
    """
    semester_avg = df.resample('2QS-Dec', on='dates').agg({'Q': 'mean'})['Q']
    QavgWSp = semester_avg.iloc[0::2].mean()
    return QavgWSp


def QavgSuA(df):
    """
    QavgSuA

    INPUTS:
        df -- Streamflow (pandas dataframe of [n x 2] with dates and flow)

    DESCRIPTION:
        - Indicator: QmoySuA
        - Hydrological phenomenon: Hydraulicity
        - Question: In future climate, will hydraulicity be modified?
        - Description: Average annual flow [Qavg] for the summer-autumn [SuA]
          period.
    """
    semester_avg = df.resample('2QS-Dec', on='dates').agg({'Q': 'mean'})['Q']
    QavgSuA = semester_avg.iloc[1::2].mean()
    return QavgSuA


def Qavg_1_12(df, month):
    """
    Qavg_1_12

    INPUTS:
        df -- Streamflow (pandas dataframe of [n x 2] with dates and flow)

    DESCRIPTION:
        - Indicator: Qavg_month
        - Hydrological phenomenon: Hydraulicité
        - Question: In future climate, will hydraulicity be modified?
        - Description: Average monthly flow [Qavg] for the different months of
          the year [1-12].
    """
    month_avg = df.resample('M', on='dates').agg({'Q': 'mean'})['Q']
    Qavg_month = month_avg.iloc[month-1::12].mean()
    return Qavg_month


def extract_all_indicators(df, dist='gumbel_r'):
    """
    extract_all_indicators

    INPUTS:
        df -- Streamflow (pandas dataframe of [n x 2] with dates and flow)
        dist -- Statistical distribution of extreme value. Default is Gumbel

    DESCRIPTION:
        Extraction of all 28 statistical indicators in the 2018 Hydroclimatic
        Atlas of the Centre d'expertise hydrique du Québec (CEHQ).
    """

    s = pd.Series(np.zeros(28),
                  index=['Q1max2Sp', 'Q1max20Sp', 'Q14max2Sp', 'Q14max20Sp',
                         'D_Q1maxSp', 'Q1max2SuA', 'Q1max20SuA', 'Q7min2Su',
                         'Q7min10Su', 'Q30min5Su', 'Q7min2W', 'Q7min10W',
                         'Q30min5W', 'Qavg', 'QavgWSp', 'QavgSuA',
                         'Qavg_dec', 'Qavg_feb', 'Qavg_mar', 'Qavg_apr',
                         'Qavg_may', 'Qavg_jun', 'Qavg_jul', 'Qavg_aug',
                         'Qavg_sep', 'Qavg_oct', 'Qavg_nov', 'Qavg_dec'])

    s.Q1max2Sp = Q1max2Sp(df, dist=dist)
    s.Q1max20Sp = Q1max20Sp(df, dist=dist)
    s.Q14max2Sp = Q14max2Sp(df, dist=dist)
    s.Q14max20Sp = Q14max20Sp(df, dist=dist)
    s.D_Q1maxSp = D_Q1maxSp(df)
    s.Q1max2SuA = Q1max2SuA(df, dist=dist)
    s.Q1max20SuA = Q1max20SuA(df, dist=dist)
    s.Q7min2Su = Q7min2Su(df, dist=dist)
    s.Q7min10Su = Q7min10Su(df, dist=dist)
    s.Q30min5Su = Q30min5Su(df, dist=dist)
    s.Q7min2W = Q7min2W(df, dist=dist)
    s.Q7min10W = Q7min10W(df, dist=dist)
    s.Q30min5W = Q30min5W(df, dist=dist)
    s.Qavg = Qavg(df)
    s.QavgWSp = QavgWSp(df)
    s.QavgSuA = QavgSuA(df)
    s.Qavg_dec = Qavg_1_12(df, month=1)
    s.Qavg_feb = Qavg_1_12(df, month=2)
    s.Qavg_mar = Qavg_1_12(df, month=3)
    s.Qavg_apr = Qavg_1_12(df, month=4)
    s.Qavg_may = Qavg_1_12(df, month=5)
    s.Qavg_jun = Qavg_1_12(df, month=6)
    s.Qavg_jul = Qavg_1_12(df, month=7)
    s.Qavg_aug = Qavg_1_12(df, month=8)
    s.Qavg_sep = Qavg_1_12(df, month=8)
    s.Qavg_oct = Qavg_1_12(df, month=10)
    s.Qavg_nov = Qavg_1_12(df, month=11)
    s.Qavg_dec = Qavg_1_12(df, month=12)
    return s


if __name__ == '__main__':
    from ffa_tools import read_flow_file

    df = read_flow_file(file_name='data_test.nc')
    s = extract_all_indicators(df=df, dist='gumbel_r')
