import numpy as np
import datetime as dt
import pytz
import pandas as pd

# M2 period in hours is 12.4206012 which gives 2 tides a day, but it is more convenient 
# to think in terms of diurnal tides
M2 = 12.4206012
cycle_length = M2 * 2
# We work with 2 x M2 as the secondary tide when the Moon is close by as opposed to
# opposite is bigger and therefor varies on the  24.8 hour cycle

forecast_length = 240   # length of forecast period in hours
forecast_res = 3       # temporal resolution of forecast in hours


def get_nom_fcst_time(now_utc):
    """Takes current time and returns nominal forecast time"""

    if ((now_utc.hour >= 6) and (now_utc.hour < 18)):
        nom_fcst_time = dt.datetime(now_utc.year, now_utc.month, now_utc.day, 6, 0, 0, 0, pytz.UTC)
    elif now_utc.hour < 6:
        nom_fcst_time = dt.datetime(now_utc.year, now_utc.month, now_utc.day, 0, 0, 0, 0, pytz.UTC) - dt.timedelta(hours = 6)
    else:
        nom_fcst_time = dt.datetime(now_utc.year, now_utc.month, now_utc.day, 18, 0, 0, 0, pytz.UTC)

    return nom_fcst_time


def unmix(mixed_int, accum = False, odd = False):
    """Unmixes time series values for mixed interval parameters"""

    nt = len(mixed_int)
    unmixed = np.copy(mixed_int)
    for i in range((2 - odd), nt, 2):
        if accum:
            unmixed[i] = mixed_int[i] - mixed_int[i - 1]
        else:
            unmixed[i] = 2. * mixed_int[i] - mixed_int[i - 1]

    return unmixed


def calc_rain_rate(rain_trace):
    """Take rain trace array and return rain rate"""

# Can't convert the first data point without the previous
# but this value will be discarded anyway - the BoM datastream
# provides previous 3 days but we don't need data that far back

# Deal with missing data and convert to float
    nt = len(rain_trace)
    rain_trace[(rain_trace != '-') & (rain_trace != 'Trace')] = rain_trace[(rain_trace != '-') & (rain_trace != 'Trace')].astype(np.float64)
    rain_trace[(rain_trace == '-') | (rain_trace == 'Trace')] = np.nan
    rain_trace = rain_trace.tz_convert('Australia/Hobart')

    rain_rate = np.zeros(nt)
    for i in range(1, nt):
        if (rain_trace.index[i - 1].hour == 9) and (rain_trace.index[i - 1].minute == 0):
            if (rain_trace.index[i] - rain_trace.index[i - 1]).seconds == 0: # some sites have only daily rain accumulation so rain rate will be invalid
                rain_rate[i] = np.nan
            else:
                rain_rate[i] = rain_trace.iloc[i] / (rain_trace.index[i] - rain_trace.index[i - 1]).seconds
        else:
            rain_rate[i] = (rain_trace.iloc[i] - rain_trace.iloc[i - 1]) / (rain_trace.index[i] - rain_trace.index[i - 1]).seconds
                
    return rain_rate


def conv_wind(speed_dir):
    """Convert wind speed and direction to zonal and meridional components"""

# Wind direction conversion table - convert to degrees in 
# mathematical sense, ie degrees anti-clockwise from east.
    dirs = {'N'   :  90.0, 'NNE' :  67.5, 'NE'  :  45.0, 'ENE' :  22.5,
            'E'   :   0.0, 'ESE' : -22.5, 'SE'  : -45.0, 'SSE' : -67.5,
            'S'   : -90.0, 'SSW' :-112.5, 'SW'  :-135.0, 'WSW' :-157.5,
            'W'   : 180.0, 'WNW' : 157.5, 'NW'  : 135.0, 'NNW' : 112.5,
            'CALM':   0.0}

    nt = len(speed_dir)
    zonal = np.empty(nt)
    merid = np.empty(nt)

    for i in range(0, nt):
        if np.isnan(speed_dir['wind_spd_kmh'].iloc[i]):
            zonal[i] = np.nan
            merid[i] = np.nan
        else:
            zonal[i] = speed_dir['wind_spd_kmh'].iloc[i] * np.cos(np.pi * dirs[speed_dir['wind_dir'].iloc[i]] / 180)
            merid[i] = speed_dir['wind_spd_kmh'].iloc[i] * np.sin(np.pi * dirs[speed_dir['wind_dir'].iloc[i]] / 180)

    return {'zonal': zonal, 'merid' : merid}
