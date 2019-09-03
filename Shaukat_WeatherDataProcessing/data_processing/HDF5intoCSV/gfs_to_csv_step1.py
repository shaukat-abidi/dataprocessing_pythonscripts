import numpy as np
import netCDF4
import datetime as dt
from pylab import rcParams
import pandas as pd

import os
import pytz
import tables

# Just change the dates and file location I entered: 
def read_gfs_arch(gfs_path):
    """Retrieve all of the GFS forecasts stored locally"""

    file_names = os.listdir(gfs_path)
    gfs_data = 0

    for file_name in file_names:
        print(file_name)
        if file_name[-4:] == 'hdf5':
            store = pd.HDFStore(gfs_path + file_name)
            if type(gfs_data) == type(0):
                gfs_data = store['gfs_data']
            else:
                gfs_data = gfs_data.combine_first(store['gfs_data'])
            store.close()

    return gfs_data

def gfs_data_ts(site_id, first_index = 0, last_index = 0):
    """Read all available archived GFS data and combine analyses into time series dataframe"""
    gfs = read_gfs_arch(site_id)

    if type(first_index) == dt.datetime:
        gfs = gfs.loc[first_index:]
    if type(last_index) == dt.datetime:
        gfs = gfs.loc[:last_index]
    na = len(gfs)
    variables = gfs.keys()
    gfs_data = 0
    for i in range(0, na):
#        print gfs.index[i]
        if type(gfs['fcst_run_time'].iloc[i]) == pd.tslib.Timestamp:
            if type(gfs_data) == type(0):
                gfs_data = pd.DataFrame({'pred_temp' : gfs['pred_temp'].iloc[i],
                                         'pred_dewp' : gfs['pred_dewp'].iloc[i],
                                         'pred_rain_rate': gfs['pred_rain_rate'].iloc[i],
                                         'pred_pe':gfs['pred_pe'].iloc[i],
                                         'pred_zonal_wind':gfs['pred_zonal_wind'].iloc[i],
                                         'pred_merid_wind':gfs['pred_merid_wind'].iloc[i],
                                         'pred_surface_pressure':gfs['pred_surface_pressure'].iloc[i],
                                         'pred_sw_rad':gfs['pred_sw_rad'].iloc[i],
                                         'pred_lw_rad':gfs['pred_lw_rad'].iloc[i],
                                         'pred_sunshine':gfs['pred_sunshine'].iloc[i],
                                         'pred_cloud_cover':gfs['pred_cloud_cover'].iloc[i],
                                         'pred_soil_moisture_upper':gfs['pred_soil_moisture_upper'].iloc[i],
                                         'pred_soil_moisture_lower':gfs['pred_soil_moisture_lower'].iloc[i],
                                         'pred_surf_geo_height':gfs['pred_surf_geo_height'].iloc[i],
                                         'pred_convective_cloud':gfs['pred_convective_cloud'].iloc[i],
                                         'pred_ustorm':gfs['pred_ustorm'].iloc[i],
                                         'pred_vstorm':gfs['pred_vstorm'].iloc[i],
                                         'pred_surf_haines':gfs['pred_surf_haines'].iloc[i],
                                         'pred_total_rain':gfs['pred_total_rain'].iloc[i],
                                         'pred_surf_momentum_vflux':gfs['pred_surf_momentum_vflux'].iloc[i],
                                         'pred_low_tcc':gfs['pred_low_tcc'].iloc[i],
                                         'pred_middle_tcc':gfs['pred_middle_tcc'].iloc[i],
                                         'pred_high_tcc':gfs['pred_high_tcc'].iloc[i],
                                         'pred_cloud_cover_bound_cloud_layer':gfs['pred_cloud_cover_bound_cloud_layer'].iloc[i],
                                         'pred_rel_humidity':gfs['pred_rel_humidity'].iloc[i],
                                         'pred_surf_gnd_heat_flux':gfs['pred_surf_gnd_heat_flux'].iloc[i],
                                         'pred_wind_speed_surf':gfs['pred_wind_speed_surf'].iloc[i],
                                         'pred_surf_geowind':gfs['pred_surf_geowind'].iloc[i],
                                         'pred_max_wind_press':gfs['pred_max_wind_press'].iloc[i]},
                                        index = gfs['fcst_time'].iloc[i])
#            for var in variables:
#                gfs_data = gfs_data.combine_first(pd.DataFrame({var : gfs[var].iloc[i]}, index = gfs['fcst_time'].iloc[i]))
            print (gfs['fcst_time'].iloc[i][0])
            gfs_data = pd.DataFrame({'pred_temp' : gfs['pred_temp'].iloc[i],
                                     'pred_dewp' : gfs['pred_dewp'].iloc[i],
                                     'pred_rain_rate': gfs['pred_rain_rate'].iloc[i],
                                     'pred_pe':gfs['pred_pe'].iloc[i],
                                     'pred_zonal_wind':gfs['pred_zonal_wind'].iloc[i],
                                     'pred_merid_wind':gfs['pred_merid_wind'].iloc[i],
                                     'pred_surface_pressure':gfs['pred_surface_pressure'].iloc[i],
                                     'pred_sw_rad':gfs['pred_sw_rad'].iloc[i],
                                     'pred_lw_rad':gfs['pred_lw_rad'].iloc[i],
                                     'pred_sunshine':gfs['pred_sunshine'].iloc[i],
                                     'pred_cloud_cover':gfs['pred_cloud_cover'].iloc[i],
                                     'pred_soil_moisture_upper':gfs['pred_soil_moisture_upper'].iloc[i],
                                     'pred_soil_moisture_lower':gfs['pred_soil_moisture_lower'].iloc[i],
                                     'pred_surf_geo_height':gfs['pred_surf_geo_height'].iloc[i],
									 'pred_convective_cloud':gfs['pred_convective_cloud'].iloc[i],
									 'pred_ustorm':gfs['pred_ustorm'].iloc[i],
									 'pred_vstorm':gfs['pred_vstorm'].iloc[i],
									 'pred_surf_haines':gfs['pred_surf_haines'].iloc[i],
									 'pred_total_rain':gfs['pred_total_rain'].iloc[i],
									 'pred_surf_momentum_vflux':gfs['pred_surf_momentum_vflux'].iloc[i],
									 'pred_low_tcc':gfs['pred_low_tcc'].iloc[i],
									 'pred_middle_tcc':gfs['pred_middle_tcc'].iloc[i],
									 'pred_high_tcc':gfs['pred_high_tcc'].iloc[i],
									 'pred_cloud_cover_bound_cloud_layer':gfs['pred_cloud_cover_bound_cloud_layer'].iloc[i],
									 'pred_rel_humidity':gfs['pred_rel_humidity'].iloc[i],
									 'pred_surf_gnd_heat_flux':gfs['pred_surf_gnd_heat_flux'].iloc[i],
									 'pred_wind_speed_surf':gfs['pred_wind_speed_surf'].iloc[i],
									 'pred_surf_geowind':gfs['pred_surf_geowind'].iloc[i],
									 'pred_max_wind_press':gfs['pred_max_wind_press'].iloc[i]},
                                    index = gfs['fcst_time'].iloc[i]).combine_first(gfs_data)

    return gfs_data, gfs

# IF using Peter's historical data

if __name__ == "__main__":
	print('Script executed\n')
	data,gfs = gfs_data_ts('C:\\Shaukat\\code\\data_rep\\gfs\\orford\\sep16toJan17\\', first_index = dt.datetime(2016, 9, 1), last_index = dt.datetime(2017, 1, 31))
	data = data.resample('3H').ffill()
	data.to_csv("C:\\Shaukat\\code\\data_rep\\gfs\\orford\\csv\\orford.csv")
	gfs.to_csv("C:\\Shaukat\\code\\data_rep\\gfs\\orford\\csv\\full_gfs_orford.csv")
