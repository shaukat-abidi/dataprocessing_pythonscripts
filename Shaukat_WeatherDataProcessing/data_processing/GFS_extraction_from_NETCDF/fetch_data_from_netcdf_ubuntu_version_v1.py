# !/usr/bin/env python
import numpy as np
import netCDF4
import datetime as dt
import pandas as pd
import os

def prep_dataframe(data, date_time, idx_lat, idx_lon):
	try:
		
		df_ret = pd.DataFrame({'utc_datetime' : date_time,
		'apparent_temperature' : data.variables['APTMP_P0_L103_GLL0'][0:1, idx_lat, idx_lon],
		'frozen_precipt' : data.variables['CPOFP_P0_L1_GLL0'][0:1, idx_lat, idx_lon],
		'cloud_water' : data.variables['CWAT_P0_L200_GLL0'][0:1, idx_lat, idx_lon],
		'dew_point_temp' : data.variables['DPT_P0_L103_GLL0'][0:1, idx_lat, idx_lon],
		'wind_speed_gust' : data.variables['GUST_P0_L1_GLL0'][0:1, idx_lat, idx_lon],
		'haines_index' : data.variables['HINDEX_P0_L1_GLL0'][0:1, idx_lat, idx_lon],
		'surface_lifted_ind' : data.variables['LFTX_P0_L1_GLL0'][0:1, idx_lat, idx_lon],
		'potential_temp' : data.variables['POT_P0_L104_GLL0'][0:1, idx_lat, idx_lon],
		'press_l103' : data.variables['PRES_P0_L103_GLL0'][0:1, idx_lat, idx_lon],
		'press_l1' : data.variables['PRES_P0_L1_GLL0'][0:1, idx_lat, idx_lon],
		'press_l6' : data.variables['PRES_P0_L6_GLL0'][0:1, idx_lat, idx_lon],
		'press_l7' : data.variables['PRES_P0_L7_GLL0'][0:1, idx_lat, idx_lon],
		'precipitable_water' : data.variables['PWAT_P0_L200_GLL0'][0:1, idx_lat, idx_lon],
		'rel_humidity_level_a' : data.variables['RH_P0_2L108_GLL0'][0:1, idx_lat, idx_lon],
		'rel_humidity_level_b' : data.variables['RH_P0_L103_GLL0'][0:1, idx_lat, idx_lon],
		'rel_humidity_level_c' : data.variables['RH_P0_L104_GLL0'][0:1, idx_lat, idx_lon],
		'rel_humidity_level_d' : data.variables['RH_P0_L200_GLL0'][0:1, idx_lat, idx_lon],
		'rel_humidity_level_e' : data.variables['RH_P0_L204_GLL0'][0:1, idx_lat, idx_lon],
		'rel_humidity_level_f' : data.variables['RH_P0_L4_GLL0'][0:1, idx_lat, idx_lon],
		'sunshine_duration' : data.variables['SUNSD_P0_L1_GLL0'][0:1, idx_lat, idx_lon],
		'temp_level_a' : data.variables['TMP_P0_2L108_GLL0'][0:1, idx_lat, idx_lon],
		'temp_level_b' : data.variables['TMP_P0_L104_GLL0'][0:1, idx_lat, idx_lon],
		'temp_level_c' : data.variables['TMP_P0_L1_GLL0'][0:1, idx_lat, idx_lon],
		'temp_level_d' : data.variables['TMP_P0_L6_GLL0'][0:1, idx_lat, idx_lon],
		'temp_level_e' : data.variables['TMP_P0_L7_GLL0'][0:1, idx_lat, idx_lon],
		'u_wind_level_a' : data.variables['UGRD_P0_2L108_GLL0'][0:1, idx_lat, idx_lon],
		'u_wind_level_b' : data.variables['UGRD_P0_L104_GLL0'][0:1, idx_lat, idx_lon],
		'u_wind_level_c' : data.variables['UGRD_P0_L220_GLL0'][0:1, idx_lat, idx_lon],
		'u_wind_level_d' : data.variables['UGRD_P0_L6_GLL0'][0:1, idx_lat, idx_lon],
		'u_wind_level_e' : data.variables['UGRD_P0_L7_GLL0'][0:1, idx_lat, idx_lon],
		'u_comp_storm' : data.variables['USTM_P0_2L103_GLL0'][0:1, idx_lat, idx_lon],
		'v_comp_wind_a' : data.variables['VGRD_P0_2L108_GLL0'][0:1, idx_lat, idx_lon],
		'v_comp_wind_b' : data.variables['VGRD_P0_L104_GLL0'][0:1, idx_lat, idx_lon],
		'v_comp_wind_c' : data.variables['VGRD_P0_L220_GLL0'][0:1, idx_lat, idx_lon],
		'v_comp_wind_d' : data.variables['VGRD_P0_L6_GLL0'][0:1, idx_lat, idx_lon],
		'v_comp_wind_3' : data.variables['VGRD_P0_L7_GLL0'][0:1, idx_lat, idx_lon],
		'v_comp_storm' : data.variables['VSTM_P0_2L103_GLL0'][0:1, idx_lat, idx_lon],
		'vert_vel' : data.variables['VVEL_P0_L104_GLL0'][0:1, idx_lat, idx_lon],
		'encoded_time' : data.variables['initial_time0_encoded'][:],
		'lat' : data.variables['lat_0'][idx_lat],
		'lon' : data.variables['lon_0'][idx_lon],
		'wilting_point' : data.variables['WILT_P0_L1_GLL0'][0:1, idx_lat, idx_lon]})
		
		return df_ret
	
	except KeyError:
		print 'WARNING: Key missing. Skipping this file \n'

def get_lat_lon(lat_0_array, lon_0_array, target_lat, target_lon):	
	idx_lat = 55555
	idx_lon = 55555

	# Making sure we are passing correct ranges of lat and lon
	if ( (target_lat <= -10.0 and target_lat >= -44.0) and (target_lon <= 154.0 and target_lon >= 112.0) ):
		idx_lat = np.argmin(np.abs(lat_0_array - target_lat))
		idx_lon = np.argmin(np.abs(lon_0_array - target_lon))
		print 'idx_lat: ', idx_lat, 'value in NetCDF: ', lat_0_array[idx_lat], 'given lat: ', target_lat
		print 'idx_lon: ', idx_lon, 'value in NetCDF: ', lon_0_array[idx_lon], 'given lon: ', target_lon

	return idx_lat,idx_lon


if __name__ == "__main__":
	# Target lat/lon (CHANGE THIS)
	target_lat = -41.26101779
	target_lon =  148.166736

	# Generate filenames and read it to formulate dataframes (CHANGE THIS)
	path_to_file='//home//thorweather//gfs_files//nc//'

	start_date = pd.datetime(2015,1,15) #YYYY,month,day
	end_date = pd.datetime(2017,3,25) #YYYY,month,day
	utc_datetime_range = pd.date_range(start=start_date, end=end_date, freq='6H')
	start_of_loop=1

	for date_time in utc_datetime_range:
		
		# Generate filename (We can do it from reading netcdf directly)
		str_year = str(date_time.year)
		str_month = str('%02d' %(date_time.month))
		str_day = str('%02d' %(date_time.day))
		str_fcst_hour = str('%02d' %(date_time.hour))
		netcdf_filename='gfs.0p25.'+str_year+str_month+str_day+str_fcst_hour+'.f000.grib2.abrie233580.nc'
		file_path=path_to_file+netcdf_filename
		# print file_path, os.path.isfile(file_path)
		
		

		if (os.path.isfile(file_path)):
			
			print 'processing: ', file_path

			if (start_of_loop == 1):
				start_of_loop = 0

				# Read NETCDF File
				data = netCDF4.Dataset(file_path)

				# Get lat/lon
				lat_0_array = data.variables['lat_0'][:]
				lon_0_array = data.variables['lon_0'][:]
				idx_lat,idx_lon = get_lat_lon(lat_0_array, lon_0_array, target_lat, target_lon)

				# Generate dataframe
				df_accum = prep_dataframe(data, date_time, idx_lat, idx_lon)

				# delete data
				del data
			
			

			else:
				
				# Read NETCDF File
				data = netCDF4.Dataset(file_path)

				# Generate dataframe
				df_ret = prep_dataframe(data, date_time, idx_lat, idx_lon)

				# Append it to the old dataframe
				df_accum = df_accum.append(df_ret, ignore_index=True)

				# Delete the following
				del df_ret
				del data


		else:
			print 'File does not exist.'
			pass

	# Storing GFS file
	store_gfs_file = '//home//thorweather//gfs_files//csv//' + 'gfs_lat_' +str(lat_0_array[idx_lat]) + '_lon_' + str(lon_0_array[idx_lon]) + '.csv'
	print 'storing GFS file as csv at: ', store_gfs_file
	df_accum.to_csv(store_gfs_file)
	print 'Done.'
