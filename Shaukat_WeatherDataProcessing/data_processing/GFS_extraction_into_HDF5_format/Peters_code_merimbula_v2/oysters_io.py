#from siphon.catalog import get_latest_access_url
#from siphon.catalog import TDSCatalog
#from siphon.ncss import NCSS
import datetime as dt
import calendar
import pandas as pd
import numpy as np
import os
import logging
import lxml
import html5lib
import bs4
import requests
import oysters_data as odat
#import oysters_analysis as oana
import netCDF4
import pytz

log_global = logging.getLogger('glob')
log_gfs = logging.getLogger('gfs')
log_obs = logging.getLogger('obs')
log_ty = logging.getLogger('ty')


def update_gfs(site_cfg, nom_fcst_time, archive = 0.):
	"""Download most recent GFS forecast and update site data structures"""

	oysters_root = os.environ['OYSTERS_ROOT']
	site_gfs_file = oysters_root + '/oysters/sensors/' + site_cfg['ident'] + '/gfs/' + nom_fcst_time.strftime('%Y%m') + '_gfs.hdf5'
# If there is no GFS file for this month then create a new file
	if not os.access(site_gfs_file, os.F_OK):
		new_gfs_file(site_cfg['ident'], nom_fcst_time)

# Get latest GFS forecast data for current site
	try:
		if archive:
			site_gfs_data = get_gfs_arch(site_cfg, nom_fcst_time)
			site_gfs_data_prev = get_gfs_arch(site_cfg, nom_fcst_time - dt.timedelta(0.25))
		else:
			site_gfs_data = get_gfs(site_cfg, nom_fcst_time)
			site_gfs_data_prev = get_gfs(site_cfg, nom_fcst_time - dt.timedelta(0.25))
	except Exception as err:
		# log_gfs.exception('Could not get GFS data for site %s', site_cfg['ident'])
		print('Could not get GFS data for site %s' %(site_cfg['ident']))
		site_gfs_data = 0
	else:

# Store GFS forecast data in site GFS HDF file
		store_gfs(site_gfs_data, site_cfg['ident'], nom_fcst_time)
		store_gfs(site_gfs_data_prev, site_cfg['ident'], nom_fcst_time - dt.timedelta(0.25))

	return site_gfs_data


def update_gfs_modified_for_rda_only(site_cfg, nom_fcst_time, archive = 0.):
	"""Download most recent GFS forecast and update site data structures"""

	oysters_root = os.environ['OYSTERS_ROOT']
	# archive = 0
	site_gfs_file = oysters_root + '/oysters/sensors/' + site_cfg['ident'] + '/gfs/' + nom_fcst_time.strftime('%Y%m') + '_gfs.hdf5'
# If there is no GFS file for this month then create a new file
	if not os.access(site_gfs_file, os.F_OK):
		new_gfs_file(site_cfg['ident'], nom_fcst_time)

# Get latest GFS forecast data for current site
	site_gfs_data = get_gfs_arch(site_cfg, nom_fcst_time)
	site_gfs_data_prev = get_gfs_arch(site_cfg, nom_fcst_time - dt.timedelta(0.25))

	# Store GFS forecast data in site GFS HDF file
	store_gfs(site_gfs_data, site_cfg['ident'], nom_fcst_time)
	store_gfs(site_gfs_data_prev, site_cfg['ident'], nom_fcst_time - dt.timedelta(0.25))

	return site_gfs_data


def get_gfs(cfg_data, nom_fcst_time):
	"""Get GFS forecast from NOMADS"""

	# Define date/hour strings for NOMADS URL
	fcst_date_str = nom_fcst_time.strftime('%Y%m%d')
	fcst_hour_str = nom_fcst_time.strftime('%H')
	# Define URL string for OPeNDAP/DODS access to NOMADS GrADS data server for GFS 3-hour res forecast (10 days)
	url = 'http://nomads.ncep.noaa.gov:9090/dods/gfs_0p25/gfs' + fcst_date_str + '/gfs_0p25_' + fcst_hour_str + 'z'
#    print url

	# log_gfs.info('Accessing NOMADS for latest GFS 0.25 deg forecast for site %s at %s', cfg_data['ident'], nom_fcst_time.strftime('%Y-%m-%dT%H:%M:%S'))
	print('Accessing NOMADS for latest GFS 0.25 deg forecast for site %s at %s' %(cfg_data['ident'], nom_fcst_time.strftime('%Y-%m-%dT%H:%M:%S')) )


	# Open NetCDF4 file
	file = netCDF4.Dataset(url)

#    file.set_auto_mask(False)

	# Read in complete time vector and convert to datetime
	time = file.variables['time'][:]
	index = []
	for i in (np.arange(0, len(time))):
		index.append(dt.datetime(1, 1, 1, 0, 0, 0, 0, pytz.UTC) + dt.timedelta(time[i] - 2))

	lat_ind = np.around((cfg_data['lat'] + 90) * 4)
#    print lat_ind, file.variables['lat'][lat_ind], cfg_data['lat']
	lon_ind = np.around(cfg_data['lon'] * 4)
#    print lon_ind, file.variables['lon'][lon_ind], cfg_data['lon']

# Extract data into a dataframe

	gfs_data = pd.DataFrame({'pred_temp' : file.variables['tmp2m'][:, lat_ind, lon_ind],
							 'pred_dewp' : file.variables['dpt2m'][:, lat_ind, lon_ind],
							 'pred_surface_pressure' : file.variables['pressfc'][:, lat_ind, lon_ind],
							 'pred_zonal_wind' : file.variables['ugrd10m'][:, lat_ind, lon_ind],
							 'pred_merid_wind' : file.variables['vgrd10m'][:, lat_ind, lon_ind],
							 'pred_rain_rate' : odat.unmix(file.variables['pratesfc'][:, lat_ind, lon_ind]),
							 'pred_total_rain' : odat.unmix(file.variables['apcpsfc'][:, lat_ind, lon_ind], accum = True),
							 'pred_sw_rad' : odat.unmix(file.variables['dswrfsfc'][:, lat_ind, lon_ind]),
							 'pred_lw_rad' : odat.unmix(file.variables['dlwrfsfc'][:, lat_ind, lon_ind]),
							 'pred_pe' : file.variables['pevprsfc'][:, lat_ind, lon_ind],
							 'pred_soil_moisture_upper' : file.variables['soilw0_10cm'][:, lat_ind, lon_ind],
							 'pred_soil_moisture_lower' : file.variables['soilw10_40cm'][:, lat_ind, lon_ind],
							 'pred_sunshine' : file.variables['sunsdsfc'][:, lat_ind, lon_ind],
							 'pred_cloud_cover' : odat.unmix(file.variables['tcdcclm'][:, lat_ind, lon_ind]),
							 'pred_rel_humidity' : file.variables['rh2m'][:, lat_ind, lon_ind],
							 'pred_surf_gnd_heat_flux' : odat.unmix(file.variables['gfluxsfc'][:, lat_ind, lon_ind]),
							 'pred_wind_speed_surf' : file.variables['gustsfc'][:, lat_ind, lon_ind],
							 'pred_surf_geowind' : file.variables['hgtmwl'][:, lat_ind, lon_ind],
							 'pred_surf_haines' : file.variables['hindexsfc'][:, lat_ind, lon_ind],
							 'pred_max_wind_press' : file.variables['presmwl'][:, lat_ind, lon_ind],
							 'pred_cloud_cover_bound_cloud_layer' : odat.unmix(file.variables['tcdcblcll'][:, lat_ind, lon_ind]),
							 'pred_low_tcc' : odat.unmix(file.variables['tcdclcll'][:, lat_ind, lon_ind]),
							 'pred_middle_tcc' : odat.unmix(file.variables['tcdcmcll'][:, lat_ind, lon_ind]),
							 'pred_high_tcc' : odat.unmix(file.variables['tcdchcll'][:, lat_ind, lon_ind]),
							 'pred_convective_cloud' : file.variables['tcdcccll'][:, lat_ind, lon_ind],
							 'pred_surf_geo_height' : file.variables['hgtsfc'][:, lat_ind, lon_ind],
							 'pred_ustorm' : file.variables['ustm6000_0m'][:, lat_ind, lon_ind],
							 'pred_vstorm' : file.variables['vstm6000_0m'][:, lat_ind, lon_ind],
							 'pred_surf_momentum_vflux' : odat.unmix(file.variables['vflxsfc'][:, lat_ind, lon_ind])},
							 index = index)

	gfs_data[gfs_data > 1e+20] = np.nan
	gfs_data['pred_rain_rate'][gfs_data['pred_rain_rate'] < 0] = 0
	gfs_data['pred_total_rain'][gfs_data['pred_total_rain'] < 0] = 0
	gfs_data['pred_sw_rad'][gfs_data['pred_sw_rad'] < 0] = 0
	gfs_data['pred_lw_rad'][gfs_data['pred_lw_rad'] < 0] = 0
	gfs_data['pred_cloud_cover'][gfs_data['pred_cloud_cover'] < 0] = 0
	gfs_data['pred_surf_gnd_heat_flux'][gfs_data['pred_surf_gnd_heat_flux'] < 0] = 0
	gfs_data['pred_cloud_cover_bound_cloud_layer'][gfs_data['pred_cloud_cover_bound_cloud_layer'] < 0] = 0
	gfs_data['pred_low_tcc'][gfs_data['pred_low_tcc'] < 0] = 0
	gfs_data['pred_middle_tcc'][gfs_data['pred_middle_tcc'] < 0] = 0
	gfs_data['pred_high_tcc'][gfs_data['pred_high_tcc'] < 0] = 0
	gfs_data['pred_surf_momentum_vflux'][gfs_data['pred_surf_momentum_vflux'] < 0] = 0

	return gfs_data


def get_gfs_arch(cfg_data, nom_fcst_time):
	"""Retrieve archived GFS data from RDA THREDDS"""

	# Define URL string for OPeNDAP/DODS access to CISL RDA THREDDS data server
	url = 'http://rda.ucar.edu/thredds/dodsC/aggregations/g/ds084.1/1/ds084.1-2016/TwoD'

	# log_gfs.info('Accessing RDA for GFS 0.25 deg forecast for site %s at %s', cfg_data['ident'], nom_fcst_time.strftime('%Y-%m-%dT%H:%M:%S'))
	print('Accessing RDA for GFS 0.25 deg forecast for site %s -%s-lat %f lon %f at %s' %(cfg_data['ident'], cfg_data['name'], cfg_data['lat'], cfg_data['lon'], nom_fcst_time.strftime('%Y-%m-%dT%H:%M:%S')))
	# print('Lat: %f Lon:%f ' %(cfg_data['lat'], cfg_data('lon')))

	# Open NetCDF data file
	file = netCDF4.Dataset(url)


	# Latitude is specified from 90 degrees to -90 degrees - OPPOSITE TO NOMADS!
	lat_ind = np.around(((-1) * cfg_data['lat'] + 90) * 4)
	lon_ind = np.around(cfg_data['lon'] * 4)


# Fixed interval variables

	coordsa = file.variables['Temperature_height_above_ground'].coordinates.rsplit() # tuple of coordinate (variable) names
#    print 'Temperature_height_above_ground', coordsa[0], coordsa[1]
	coordsa = file.variables['Dewpoint_temperature_height_above_ground'].coordinates.rsplit() # tuple of coordinate (variable) names
#    print 'Dewpoint_temperature_height_above_ground', coordsa[0], coordsa[1]
	coordsa = file.variables['Pressure_surface'].coordinates.rsplit() # tuple of coordinate (variable) names
#    print 'Pressure_surface', coordsa[0], coordsa[1]
	coordsa = file.variables['Sunshine_Duration_surface'].coordinates.rsplit() # tuple of coordinate (variable) names
#    print 'Sunshine_Duration_surface', coordsa[0], coordsa[1]
	coordsa = file.variables['u-component_of_wind_height_above_ground'].coordinates.rsplit() # tuple of coordinate (variable) names
#    print 'u-component_of_wind_height_above_ground', coordsa[0], coordsa[1]
	coordsa = file.variables['v-component_of_wind_height_above_ground'].coordinates.rsplit() # tuple of coordinate (variable) names
#    print 'v-component_of_wind_height_above_ground', coordsa[0], coordsa[1]
	coordsa = file.variables['Volumetric_Soil_Moisture_Content_depth_below_surface_layer'].coordinates.rsplit() # tuple of coordinate (variable) names
#    print 'Volumetric_Soil_Moisture_Content_depth_below_surface_layer', coordsa[0], coordsa[1]

#	Shaukat Adding vars here
	coordsa = file.variables['Geopotential_height_surface'].coordinates.rsplit() # tuple of coordinate (variable) names
#    print 'Geopotential_height_surface', coordsa[0], coordsa[1]
	coordsa = file.variables['Total_cloud_cover_convective_cloud'].coordinates.rsplit() # tuple of coordinate (variable) names
#    print 'Total_cloud_cover_convective_cloud', coordsa[0], coordsa[1]
	coordsa = file.variables['U-Component_Storm_Motion_height_above_ground_layer'].coordinates.rsplit() # tuple of coordinate (variable) names
#    print 'U-Component_Storm_Motion_height_above_ground_layer', coordsa[0], coordsa[1]
	coordsa = file.variables['V-Component_Storm_Motion_height_above_ground_layer'].coordinates.rsplit() # tuple of coordinate (variable) names
#    print 'V-Component_Storm_Motion_height_above_ground_layer', coordsa[0], coordsa[1]
	coordsa = file.variables['Haines_Index_surface'].coordinates.rsplit() # tuple of coordinate (variable) names
#    print 'Haines_Index_surface', coordsa[0], coordsa[1]
	coordsa = file.variables['Relative_humidity_height_above_ground'].coordinates.rsplit() # tuple of coordinate (variable) names
#    print 'Relative_humidity_height_above_ground', coordsa[0], coordsa[1]
	coordsa = file.variables['Wind_speed_gust_surface'].coordinates.rsplit() # tuple of coordinate (variable) names
#    print 'Wind_speed_gust_surface', coordsa[0], coordsa[1]
	coordsa = file.variables['Geopotential_height_maximum_wind'].coordinates.rsplit() # tuple of coordinate (variable) names
#    print 'Geopotential_height_maximum_wind', coordsa[0], coordsa[1]
	coordsa = file.variables['Pressure_maximum_wind'].coordinates.rsplit() # tuple of coordinate (variable) names
#    print 'Pressure_maximum_wind', coordsa[0], coordsa[1]


	# Determine forecast cycle offset (hours since 2015-01-15) for desired
	# forecast.
	ref_timea = dt.datetime.strptime(file.variables[coordsa[1]].units[11:], '%Y-%m-%dT%H:%M:%SZ')
	ref_timea = ref_timea.replace(tzinfo = pytz.UTC)
	fcst_cycle_offseta = nom_fcst_time - ref_timea                                        # offset as dt.timedelta
	fcst_cycle_offseta = fcst_cycle_offseta.days * 24 + fcst_cycle_offseta.seconds / 3600 # offset in hours
#    print fcst_cycle_offseta

	# Determine indices of reftimeX corresponding to forecast cycle offset
	ref_timea_ind = np.where(file.variables[coordsa[0]][:] == fcst_cycle_offseta)[0]
#    print ref_timea_ind
#    print file.variables[coordsa[0]][ref_timea_ind]
#    print file.variables[coordsa[0]][ref_timea_ind[0] - 1], file.variables[coordsa[0]][ref_timea_ind[-1] + 1] # check that the limits of this forecast cycle are being identified correctly

	# Extract timeX index for specified forecast cycle and convert to datetime.
	timea = file.variables[coordsa[1]][ref_timea_ind[0:81]] # Take only 3-hourly forecasts to 10 days (starting from 0 hours)
	indexa = []
	for i in np.arange(0, len(timea)):
		indexa.append(ref_timea + dt.timedelta(0, timea[i] * 3600))
#    print nom_fcst_time
#    print indexa[0], indexa[-1]

	# height index, will select 2m temperature, 2m dewpt, 10m wind
	h_ind = 0
	# depth indices for soil moisture
	du_ind = 0
	dl_ind = 1

	gfs_data_a = pd.DataFrame({'pred_temp' : file.variables['Temperature_height_above_ground'][ref_timea_ind[0:81], h_ind, lat_ind, lon_ind],
							   'pred_dewp' : file.variables['Dewpoint_temperature_height_above_ground'][ref_timea_ind[0:81], h_ind, lat_ind, lon_ind],
							   'pred_rel_humidity' : file.variables['Relative_humidity_height_above_ground'][ref_timea_ind[0:81], h_ind, lat_ind, lon_ind],
							   'pred_surface_pressure' : file.variables['Pressure_surface'][ref_timea_ind[0:81], lat_ind, lon_ind],
							   'pred_zonal_wind' : file.variables['u-component_of_wind_height_above_ground'][ref_timea_ind[0:81], h_ind, lat_ind, lon_ind],
							   'pred_merid_wind' : file.variables['v-component_of_wind_height_above_ground'][ref_timea_ind[0:81], h_ind, lat_ind, lon_ind],
							   'pred_soil_moisture_upper' : file.variables['Volumetric_Soil_Moisture_Content_depth_below_surface_layer'][ref_timea_ind[0:81], du_ind, lat_ind, lon_ind],
							   'pred_soil_moisture_lower' : file.variables['Volumetric_Soil_Moisture_Content_depth_below_surface_layer'][ref_timea_ind[0:81], dl_ind, lat_ind, lon_ind],
							   'pred_sunshine' : file.variables['Sunshine_Duration_surface'][ref_timea_ind[0:81], lat_ind, lon_ind],
							   'pred_wind_speed_surf' : file.variables['Wind_speed_gust_surface'][ref_timea_ind[0:81], lat_ind, lon_ind],
							   'pred_surf_geo_height' : file.variables['Geopotential_height_surface'][ref_timea_ind[0:81], lat_ind, lon_ind],
							   'pred_surf_geowind' : file.variables['Geopotential_height_maximum_wind'][ref_timea_ind[0:81], lat_ind, lon_ind],
							   'pred_max_wind_press' : file.variables['Pressure_maximum_wind'][ref_timea_ind[0:81], lat_ind, lon_ind],
							   'pred_convective_cloud' : file.variables['Total_cloud_cover_convective_cloud'][ref_timea_ind[0:81], lat_ind, lon_ind],
							   'pred_ustorm' : file.variables['U-Component_Storm_Motion_height_above_ground_layer'][ref_timea_ind[0:81], h_ind, lat_ind, lon_ind],
							   'pred_vstorm' : file.variables['V-Component_Storm_Motion_height_above_ground_layer'][ref_timea_ind[0:81], h_ind, lat_ind, lon_ind],
							   'pred_surf_haines' : file.variables['Haines_Index_surface'][ref_timea_ind[0:81], lat_ind, lon_ind]},
							   index = indexa)


# Fixed interval rate variables

	coordsb = file.variables['Potential_Evaporation_Rate_surface'].coordinates.rsplit() # tuple of coordinate (variable) names
#    print 'Potential_Evaporation_Rate_surface', coordsb[0], coordsb[1]

	# Determine forecast cycle offset (hours since 2015-01-15) for desired
	# forecast.
	ref_timeb = dt.datetime.strptime(file.variables[coordsb[1]].units[11:], '%Y-%m-%dT%H:%M:%SZ')
	ref_timeb = ref_timeb.replace(tzinfo = pytz.UTC)
	fcst_cycle_offsetb = nom_fcst_time - ref_timeb                                        # offset as dt.timedelta
	fcst_cycle_offsetb = fcst_cycle_offsetb.days * 24 + fcst_cycle_offsetb.seconds / 3600 # offset in hours
#    print fcst_cycle_offsetb

	# Determine indices of reftimeX corresponding to forecast cycle offset
	ref_timeb_ind = np.where(file.variables[coordsb[0]][:] == fcst_cycle_offsetb)[0]
#    print ref_timeb_ind
#    print file.variables[coordsb[0]][ref_timeb_ind]
#    print file.variables[coordsb[0]][ref_timeb_ind[0] - 1], file.variables[coordsb[0]][ref_timeb_ind[-1] + 1] # check that the limits of this forecast cycle are being identified correctly

	# Extract timeX index for specified forecast cycle and convert to datetime.
	timeb = file.variables[coordsb[1]][ref_timeb_ind[0:80]] # Take only 3-hourly forecasts to 10 days (starting from 3 hours)
	indexb = []
	for i in np.arange(0, len(timeb)):
		indexb.append(ref_timeb + dt.timedelta(0, timeb[i] * 3600))
#    print nom_fcst_time
#    print indexb[0], indexb[-1]


	gfs_data_b = pd.DataFrame({'pred_pe' : file.variables['Potential_Evaporation_Rate_surface'][ref_timeb_ind[0:80], lat_ind, lon_ind],
							   }, index = indexb)


# Mixed interval variables

	coordsc = file.variables['Precipitation_rate_surface_Mixed_intervals_Average'].coordinates.rsplit() # tuple of coordinate (variable) names
#    print 'Precipitation_rate_surface_Mixed_intervals_Average', coordsc[0], coordsc[1]
	coordsc = file.variables['Total_precipitation_surface_Mixed_intervals_Accumulation'].coordinates.rsplit() # tuple of coordinate (variable) names
#    print 'Total_precipitation_surface_Mixed_intervals_Accumulation', coordsc[0], coordsc[1]
	coordsc = file.variables['Downward_Short-Wave_Radiation_Flux_surface_Mixed_intervals_Average'].coordinates.rsplit() # tuple of coordinate (variable) names
#    print 'Downward_Short-Wave_Radiation_Flux_surface_Mixed_intervals_Average', coordsc[0], coordsc[1]
	coordsc = file.variables['Downward_Long-Wave_Radp_Flux_surface_Mixed_intervals_Average'].coordinates.rsplit() # tuple of coordinate (variable) names
#    print 'Downward_Long-Wave_Radp_Flux_surface_Mixed_intervals_Average', coordsc[0], coordsc[1]
	coordsc = file.variables['Total_cloud_cover_entire_atmosphere_Mixed_intervals_Average'].coordinates.rsplit() # tuple of coordinate (variable) names
#    print 'Total_cloud_cover_entire_atmosphere_Mixed_intervals_Average', coordsc[0], coordsc[1]

	# Shaukat adding vars here

	coordsc = file.variables['Momentum_flux_v-component_surface_Mixed_intervals_Average'].coordinates.rsplit() # tuple of coordinate (variable) names
#    print 'Momentum_flux_v-component_surface_Mixed_intervals_Average', coordsc[0], coordsc[1]
	coordsc = file.variables['Total_cloud_cover_low_cloud_Mixed_intervals_Average'].coordinates.rsplit() # tuple of coordinate (variable) names
#    print 'Total_cloud_cover_low_cloud_Mixed_intervals_Average', coordsc[0], coordsc[1]
	coordsc = file.variables['Total_cloud_cover_middle_cloud_Mixed_intervals_Average'].coordinates.rsplit() # tuple of coordinate (variable) names
#    print 'Total_cloud_cover_middle_cloud_Mixed_intervals_Average', coordsc[0], coordsc[1]
	coordsc = file.variables['Total_cloud_cover_high_cloud_Mixed_intervals_Average'].coordinates.rsplit() # tuple of coordinate (variable) names
#    print 'Total_cloud_cover_high_cloud_Mixed_intervals_Average', coordsc[0], coordsc[1]
	coordsc = file.variables['Total_cloud_cover_boundary_layer_cloud_Mixed_intervals_Average'].coordinates.rsplit() # tuple of coordinate (variable) names
#    print 'Total_cloud_cover_boundary_layer_cloud_Mixed_intervals_Average', coordsc[0], coordsc[1]
	coordsc = file.variables['Ground_Heat_Flux_surface_Mixed_intervals_Average'].coordinates.rsplit() # tuple of coordinate (variable) names
#    print 'Ground_Heat_Flux_surface_Mixed_intervals_Average', coordsc[0], coordsc[1]

	# Determine forecast cycle offset (hours since 2015-01-15) for desired
	# forecast.
	ref_timec = dt.datetime.strptime(file.variables[coordsc[1]].units[11:], '%Y-%m-%dT%H:%M:%SZ')
	ref_timec = ref_timec.replace(tzinfo = pytz.UTC)
	fcst_cycle_offsetc = nom_fcst_time - ref_timec                                        # offset as dt.timedelta
	fcst_cycle_offsetc = fcst_cycle_offsetc.days * 24 + fcst_cycle_offsetc.seconds / 3600 # offset in hours
#    print fcst_cycle_offsetc

	# Determine indices of reftimeX corresponding to forecast cycle offset
	ref_timec_ind = np.where(file.variables[coordsc[0]][:] == fcst_cycle_offsetc)[0]
#    print ref_timec_ind
#    print file.variables[coordsc[0]][ref_timec_ind]
#    print file.variables[coordsc[0]][ref_timec_ind[0] - 1], file.variables[coordsc[0]][ref_timec_ind[-1] + 1] # check that the limits of this forecast cycle are being identified correctly

	# Extract timeX index for specified forecast cycle and convert to datetime.
	timec = file.variables[coordsc[1]][ref_timec_ind[0:80]] # Take only 3-hourly forecasts to 10 days (starting from 3 hours)
	indexc = []
	for i in np.arange(0, len(timec)):
		indexc.append(ref_timec + dt.timedelta(0, timec[i] * 3600))
#    print nom_fcst_time
#    print indexc[0], indexc[-1]

	# Extract time_bounds for mixed intervals variables
	time_bounds = file.variables[file.variables[coordsc[1]].bounds][ref_timec_ind[0:80]]

	gfs_data_c = pd.DataFrame({'pred_rain_rate' : odat.unmix(file.variables['Precipitation_rate_surface_Mixed_intervals_Average'][ref_timec_ind[0:80], lat_ind, lon_ind], odd = True),
							   'pred_total_rain' : odat.unmix(file.variables['Total_precipitation_surface_Mixed_intervals_Accumulation'][ref_timec_ind[0:80], lat_ind, lon_ind], accum = True, odd = True),
							   'pred_sw_rad' : odat.unmix(file.variables['Downward_Short-Wave_Radiation_Flux_surface_Mixed_intervals_Average'][ref_timec_ind[0:80], lat_ind, lon_ind], odd = True),
							   'pred_lw_rad' : odat.unmix(file.variables['Downward_Long-Wave_Radp_Flux_surface_Mixed_intervals_Average'][ref_timec_ind[0:80], lat_ind, lon_ind], odd = True),
							   'pred_cloud_cover' : odat.unmix(file.variables['Total_cloud_cover_entire_atmosphere_Mixed_intervals_Average'][ref_timec_ind[0:80], lat_ind, lon_ind], odd = True),
							   'pred_surf_momentum_vflux' : odat.unmix(file.variables['Momentum_flux_v-component_surface_Mixed_intervals_Average'][ref_timec_ind[0:80], lat_ind, lon_ind], odd = True),
							   'pred_low_tcc' : odat.unmix(file.variables['Total_cloud_cover_low_cloud_Mixed_intervals_Average'][ref_timec_ind[0:80], lat_ind, lon_ind], odd = True),
							   'pred_middle_tcc' : odat.unmix(file.variables['Total_cloud_cover_middle_cloud_Mixed_intervals_Average'][ref_timec_ind[0:80], lat_ind, lon_ind], odd = True),
							   'pred_high_tcc' : odat.unmix(file.variables['Total_cloud_cover_high_cloud_Mixed_intervals_Average'][ref_timec_ind[0:80], lat_ind, lon_ind], odd = True),
							   'pred_surf_gnd_heat_flux' : odat.unmix(file.variables['Ground_Heat_Flux_surface_Mixed_intervals_Average'][ref_timec_ind[0:80], lat_ind, lon_ind], odd = True),
							   'pred_cloud_cover_bound_cloud_layer' : odat.unmix(file.variables['Total_cloud_cover_boundary_layer_cloud_Mixed_intervals_Average'][ref_timec_ind[0:80], lat_ind, lon_ind], odd = True)},
							   index = indexc)

	gfs_data_c['pred_rain_rate'][gfs_data_c['pred_rain_rate'] < 0] = 0
	gfs_data_c['pred_total_rain'][gfs_data_c['pred_total_rain'] < 0] = 0
	gfs_data_c['pred_sw_rad'][gfs_data_c['pred_sw_rad'] < 0] = 0
	gfs_data_c['pred_lw_rad'][gfs_data_c['pred_lw_rad'] < 0] = 0
	gfs_data_c['pred_cloud_cover'][gfs_data_c['pred_cloud_cover'] < 0] = 0
	gfs_data_c['pred_surf_momentum_vflux'][gfs_data_c['pred_surf_momentum_vflux'] < 0] = 0
	gfs_data_c['pred_low_tcc'][gfs_data_c['pred_low_tcc'] < 0] = 0
	gfs_data_c['pred_middle_tcc'][gfs_data_c['pred_middle_tcc'] < 0] = 0
	gfs_data_c['pred_high_tcc'][gfs_data_c['pred_high_tcc'] < 0] = 0
	gfs_data_c['pred_cloud_cover_bound_cloud_layer'][gfs_data_c['pred_cloud_cover_bound_cloud_layer'] < 0] = 0

	gfs_data = pd.concat([gfs_data_a, gfs_data_b, gfs_data_c], axis = 1)

	return gfs_data


def new_gfs_file(site_id, nom_fcst_time):
	"""Initialise a new GFS dataframe for the current month and save in HDF5"""

	nf = calendar.monthrange(nom_fcst_time.year, nom_fcst_time.month)[1] * 4       # number of forecasts in the current month
	start_time = dt.datetime(nom_fcst_time.year, nom_fcst_time.month, 1, 0, 0, 0)  # nominal time of first forecast
	index = pd.date_range(start_time, periods = nf, freq = '6H', tz = 'utc')       # define index at 12 hour resolution
	fcst_arr = np.empty(odat.forecast_length / odat.forecast_res)                  # define empty array
	fcst_arr[:] = np.nan                                                           # fill empty array with nans
	fcst_arr_list = [fcst_arr] * nf                                 # repeat empty array for number of forecasts in current month
	gfs_data = pd.DataFrame({'pred_surface_pressure':fcst_arr_list,
							 'pred_sw_rad':fcst_arr_list,
							 'pred_lw_rad':fcst_arr_list,
							 'pred_sunshine':fcst_arr_list,
							 'pred_rain_rate':fcst_arr_list,
							 'pred_temp':fcst_arr_list,
							 'pred_dewp':fcst_arr_list,
							 'pred_cloud_cover':fcst_arr_list,
							 'pred_zonal_wind':fcst_arr_list,
							 'pred_merid_wind':fcst_arr_list,
							 'pred_pe':fcst_arr_list,
							 'pred_soil_moisture_upper':fcst_arr_list,
							 'pred_soil_moisture_lower':fcst_arr_list,
							 'pred_surf_geo_height':fcst_arr_list,
							 'pred_convective_cloud':fcst_arr_list,
							 'pred_ustorm':fcst_arr_list,
							 'pred_vstorm':fcst_arr_list,
							 'pred_surf_haines':fcst_arr_list,
							 'pred_total_rain':fcst_arr_list,
							 'pred_surf_momentum_vflux':fcst_arr_list,
							 'pred_low_tcc':fcst_arr_list,
							 'pred_middle_tcc':fcst_arr_list,
							 'pred_high_tcc':fcst_arr_list,
							 'pred_cloud_cover_bound_cloud_layer':fcst_arr_list,
							 'pred_rel_humidity':fcst_arr_list,
							 'pred_surf_gnd_heat_flux':fcst_arr_list,
							 'pred_wind_speed_surf':fcst_arr_list,
							 'pred_surf_geowind':fcst_arr_list,
							 'pred_max_wind_press':fcst_arr_list,
							 'fcst_time':fcst_arr_list,
							 'fcst_hour':fcst_arr_list,
							 'fcst_run_time':np.nan},
							index = index)



	oysters_root = os.environ['OYSTERS_ROOT']
	site_gfs_file = oysters_root + '/oysters/sensors/' + site_id + '/gfs/' + nom_fcst_time.strftime('%Y%m') + '_gfs.hdf5'
#    gfs_data.to_hdf(site_gfs_file, 'table')
# following line previously raised error, now fixed - check
	# log_gfs.info('Creating new GFS file for site %s: %s', site_id, nom_fcst_time.strftime('%Y%m') + '_gfs.hdf5')
	print('Creating new GFS file for site %s: %s' %(site_id, nom_fcst_time.strftime('%Y%m') + '_gfs.hdf5'))
	store = pd.HDFStore(site_gfs_file)
	store['gfs_data'] = gfs_data
	store.close()

	return 0



def store_gfs(site_gfs_data, site_id, nom_fcst_time):
	"""Store GFS forecast data in site GFS HDF file"""

	oysters_root = os.environ['OYSTERS_ROOT']
	file_name = oysters_root + '/oysters/sensors/' + site_id + '/gfs/' + nom_fcst_time.strftime('%Y%m') + '_gfs.hdf5'
	store = pd.HDFStore(file_name)
	gfs_data = store['gfs_data']

# The following line sometimes raises a SetWithCopyWarning but is actually ok
	gfs_data['pred_surface_pressure'].loc[nom_fcst_time] = np.asarray(site_gfs_data['pred_surface_pressure'])
#    print site_gfs_data['pred_surface_pressure']
#    print gfs_data['pred_surface_pressure'].loc[nom_fcst_time]
#    print file_name, nom_fcst_time
	gfs_data['pred_sw_rad'].loc[nom_fcst_time] = np.asarray(site_gfs_data['pred_sw_rad'])
	gfs_data['pred_lw_rad'].loc[nom_fcst_time] = np.asarray(site_gfs_data['pred_lw_rad'])
	gfs_data['pred_rain_rate'].loc[nom_fcst_time] = np.asarray(site_gfs_data['pred_rain_rate'])
	gfs_data['pred_temp'].loc[nom_fcst_time] = np.asarray(site_gfs_data['pred_temp'])
	gfs_data['pred_dewp'].loc[nom_fcst_time] = np.asarray(site_gfs_data['pred_dewp'])
	gfs_data['pred_cloud_cover'].loc[nom_fcst_time] = np.asarray(site_gfs_data['pred_cloud_cover'])
	gfs_data['pred_zonal_wind'].loc[nom_fcst_time] = np.asarray(site_gfs_data['pred_zonal_wind'])
	gfs_data['pred_merid_wind'].loc[nom_fcst_time] = np.asarray(site_gfs_data['pred_merid_wind'])
	gfs_data['pred_pe'].loc[nom_fcst_time] = np.asarray(site_gfs_data['pred_pe'])
	gfs_data['pred_sunshine'].loc[nom_fcst_time] = np.asarray(site_gfs_data['pred_sunshine'])
	gfs_data['pred_soil_moisture_upper'].loc[nom_fcst_time] = np.asarray(site_gfs_data['pred_soil_moisture_upper'])
	gfs_data['pred_soil_moisture_lower'].loc[nom_fcst_time] = np.asarray(site_gfs_data['pred_soil_moisture_lower'])
	gfs_data['pred_surf_geo_height'].loc[nom_fcst_time] = np.asarray(site_gfs_data['pred_surf_geo_height'])
	gfs_data['pred_convective_cloud'].loc[nom_fcst_time] = np.asarray(site_gfs_data['pred_convective_cloud'])
	gfs_data['pred_ustorm'].loc[nom_fcst_time] = np.asarray(site_gfs_data['pred_ustorm'])
	gfs_data['pred_vstorm'].loc[nom_fcst_time] = np.asarray(site_gfs_data['pred_vstorm'])
	gfs_data['pred_surf_haines'].loc[nom_fcst_time] = np.asarray(site_gfs_data['pred_surf_haines'])
	gfs_data['pred_total_rain'].loc[nom_fcst_time] = np.asarray(site_gfs_data['pred_total_rain'])
	gfs_data['pred_surf_momentum_vflux'].loc[nom_fcst_time] = np.asarray(site_gfs_data['pred_surf_momentum_vflux'])
	gfs_data['pred_low_tcc'].loc[nom_fcst_time] = np.asarray(site_gfs_data['pred_low_tcc'])
	gfs_data['pred_middle_tcc'].loc[nom_fcst_time] = np.asarray(site_gfs_data['pred_middle_tcc'])
	gfs_data['pred_high_tcc'].loc[nom_fcst_time] = np.asarray(site_gfs_data['pred_high_tcc'])
	gfs_data['pred_cloud_cover_bound_cloud_layer'].loc[nom_fcst_time] = np.asarray(site_gfs_data['pred_cloud_cover_bound_cloud_layer'])
	gfs_data['pred_rel_humidity'].loc[nom_fcst_time] = np.asarray(site_gfs_data['pred_rel_humidity'])
	gfs_data['pred_surf_gnd_heat_flux'].loc[nom_fcst_time] = np.asarray(site_gfs_data['pred_surf_gnd_heat_flux'])
	gfs_data['pred_wind_speed_surf'].loc[nom_fcst_time] = np.asarray(site_gfs_data['pred_wind_speed_surf'])
	gfs_data['pred_surf_geowind'].loc[nom_fcst_time] = np.asarray(site_gfs_data['pred_surf_geowind'])
	gfs_data['pred_max_wind_press'].loc[nom_fcst_time] = np.asarray(site_gfs_data['pred_max_wind_press'])
	gfs_data['fcst_time'].loc[nom_fcst_time] = site_gfs_data.index # site_gfs_data['fcst_time']
	gfs_data['fcst_hour'].loc[nom_fcst_time] = np.arange(0, 241, 3) # site_gfs_data['fcst_hour']
	gfs_data['fcst_run_time'].loc[nom_fcst_time] = site_gfs_data.index[0] # site_gfs_data['fcst_run_time']

	store['gfs_data'] = gfs_data
	store.close()

	return 0

def read_gfs_arch(site_id):
	"""Retrieve all of the GFS forecasts stored locally"""

	oysters_root = os.environ['OYSTERS_ROOT']
	gfs_path = oysters_root + '/oysters/sensors/' + site_id + '/gfs/'
	file_names = os.listdir(gfs_path)
	gfs_data = 0

	for file_name in file_names:
		if file_name[-4:] == 'hdf5':
			store = pd.HDFStore(gfs_path + file_name)
#            print file_name
			if type(gfs_data) == type(0):
				gfs_data = store['gfs_data']
			else:
				gfs_data = gfs_data.combine_first(store['gfs_data'])
			store.close()

	return gfs_data




def update_obs(site_cfg, nom_fcst_time, archive = False, sensor_only = False):
	"""Download most recent observations from site sensors"""

	oysters_root = os.environ['OYSTERS_ROOT']
	site_obs_file = oysters_root + '/oysters/sensors/' + site_cfg['ident'] + '/obs/' + nom_fcst_time.strftime('%Y%m') + '_obs.hdf5'
# If there is no obs file for this month then create a new file
	if not os.access(site_obs_file, os.F_OK):
		new_obs_file(site_cfg['ident'], nom_fcst_time)

# Get site obs data from one full tidal cycle prior to nominal forecast time
	try:
		if archive:
			site_obs_data = get_obs_arch(site_cfg, nom_fcst_time, sensor_only)
		else:
			site_obs_data = get_obs(site_cfg, nom_fcst_time)
	except Exception as err:
		# log_obs.exception('Could not get obs data')
		print('Could not get obs data')
		print(err)
		site_obs_data = 0
	else:
		store_obs(site_obs_data, site_cfg['ident'], nom_fcst_time)

	return site_obs_data




def get_obs(site_cfg, nom_fcst_time):
	"""Get latest sensor obs from specified site"""

	obs_window_start = nom_fcst_time - dt.timedelta((24 + 40 / 60.) / 24.)

	# log_obs.info('Accessing BoM data stream for site %s for obs to %s', site_cfg['ident'], nom_fcst_time.strftime('%Y-%m-%dT%H:%M:%S'))
	print('Accessing BoM data stream for site %s for obs to %s', site_cfg['ident'], nom_fcst_time.strftime('%Y-%m-%dT%H:%M:%S'))

	response_bom = requests.get(site_cfg['bom_stn'])

	bom = pd.DataFrame(response_bom.json()['observations']['data'], columns = ['aifstime_utc', 'air_temp', 'wind_dir', 'wind_spd_kmh', 'press_msl', 'cloud_oktas', 'rain_trace', 'rel_hum', 'dewpt'])
	bom.rename(columns={'aifstime_utc' : 'timestamp', 'press_msl' : 'at_pressure', 'air_temp' : 'at_temp', 'rain_trace' : 'at_rain_rate', 'rel_hum' : 'at_humidity', 'dewpt' : 'at_dew_point', }, inplace=True)
	bom['timestamp'] = pd.to_datetime(bom['timestamp'], format = '%Y%m%d%H%M%S', utc = True)
	bom.set_index('timestamp', inplace = True)
	bom.sort_index(inplace = True)
	bom = bom.tz_localize('utc')

	bom['at_rain_rate'] = odat.calc_rain_rate(bom['at_rain_rate'])
	wind = odat.conv_wind(bom[['wind_spd_kmh', 'wind_dir']])
	bom['at_zonal_wind'] = wind['zonal']
	bom['at_merid_wind'] = wind['merid']

	bom = bom.resample('10T').mean().interpolate(limit = 3)

	if site_cfg['sensor_data'] != 999:
		bosch_url = 'http://yield-agri.bosch-si.com/rest/the-yield/oyster/v2/feeds/'
		ss = "start="+obs_window_start.strftime("%Y-%m-%dT%H:%M:%S")
#    es = "&finish="+nom_fcst_time.strftime("%Y-%m-%dT%H:%M:%S")

		# log_obs.info('Accessing Bosch data stream for site %s', site_cfg['ident'])
		print('Accessing Bosch data stream for site %s', site_cfg['ident'])

		response_temp = requests.get(bosch_url + str(site_cfg['sensor_data']) + '/datastreams/1?' + ss, auth=('OysterAdmin', '1234'))
		response_cond = requests.get(bosch_url + str(site_cfg['sensor_data']) + '/datastreams/2?' + ss, auth=('OysterAdmin', '1234'))
		response_pres = requests.get(bosch_url + str(site_cfg['sensor_data']) + '/datastreams/3?' + ss, auth=('OysterAdmin', '1234'))
		response_sali = requests.get(bosch_url + str(site_cfg['sensor_data']) + '/datastreams/4?' + ss, auth=('OysterAdmin', '1234'))

		temp = pd.DataFrame(response_temp.json()[u'data'])
		cond = pd.DataFrame(response_cond.json()[u'data'])
		pres = pd.DataFrame(response_pres.json()[u'data'])
		sali = pd.DataFrame(response_sali.json()[u'data'])

# Convert the timestamp from an object into a datetime object
# rename the column of data from a generic term to an explicit term.
		temp['timestamp']= pd.to_datetime(temp['timestamp'])
		temp.rename(columns={'value' : 'water_temp',}, inplace=True)
		cond['timestamp']= pd.to_datetime(cond['timestamp'])
		cond.rename(columns={'value' : 'water_conductivity',}, inplace=True)
		pres['timestamp']= pd.to_datetime(pres['timestamp'])
		pres.rename(columns={'value' : 'water_pressure',}, inplace=True)
		sali['timestamp']= pd.to_datetime(sali['timestamp'])
		sali.rename(columns={'value' : 'water_salinity',}, inplace=True)

		temp_utc = temp.set_index('timestamp').tz_localize('utc')
		cond_utc = cond.set_index('timestamp').tz_localize('utc')
		pres_utc = pres.set_index('timestamp').tz_localize('utc')
		sali_utc = sali.set_index('timestamp').tz_localize('utc')
	else:
		temp_utc = pd.DataFrame(index = bom.index)
		cond_utc = pd.DataFrame(index = bom.index)
		pres_utc = pd.DataFrame(index = bom.index)
		sali_utc = pd.DataFrame(index = bom.index)


	site_obs_data = pd.concat([temp_utc,
							   cond_utc,
							   pres_utc,
							   sali_utc,
							   bom], axis = 1)
	site_obs_data = site_obs_data.resample('10T').mean()
	site_obs_data.sort_index(inplace = True)
	return site_obs_data.loc[obs_window_start:nom_fcst_time]

def get_obs_arch(site_cfg, nom_fcst_time, sensor_only = False):
	"""Get sensor obs from specified site for specified time window"""

	obs_window_start = nom_fcst_time - dt.timedelta((24 + 40 / 60.) / 24.)
# Window end time is later than nominal hindcast time so that hindcast
# time is returned from Bosch, time series will be truncated appropriately
# at end of routine.
	obs_window_end = nom_fcst_time + dt.timedelta(0, 3600)

	bosch_url = 'http://yield-agri.bosch-si.com/rest/the-yield/oyster/v2/feeds/'
	ss = "start="+obs_window_start.strftime("%Y-%m-%dT%H:%M:%S")
	es = "&finish="+obs_window_end.strftime("%Y-%m-%dT%H:%M:%S")

	# log_obs.info('Accessing Bosch data stream for site %s', site_cfg['ident'])
	print('Accessing Bosch data stream for site %s', site_cfg['ident'])

	response_temp = requests.get(bosch_url + str(site_cfg['sensor_data']) + '/datastreams/1?' + ss + es, auth=('OysterAdmin', '1234'))
	response_cond = requests.get(bosch_url + str(site_cfg['sensor_data']) + '/datastreams/2?' + ss + es, auth=('OysterAdmin', '1234'))
	response_pres = requests.get(bosch_url + str(site_cfg['sensor_data']) + '/datastreams/3?' + ss + es, auth=('OysterAdmin', '1234'))
	response_sali = requests.get(bosch_url + str(site_cfg['sensor_data']) + '/datastreams/4?' + ss + es, auth=('OysterAdmin', '1234'))

#    response_wind = requests.get(bosch_url + str(site_cfg['wz_data']) + '/datastreams/6?' + ss + es, auth=('OysterAdmin', '1234'))
#    response_airp = requests.get(bosch_url + str(site_cfg['wz_data']) + '/datastreams/7?' + ss + es, auth=('OysterAdmin', '1234'))
#    response_rain = requests.get(bosch_url + str(site_cfg['wz_data']) + '/datastreams/8?' + ss + es, auth=('OysterAdmin', '1234'))


	temp = pd.DataFrame(response_temp.json()[u'data'])
	cond = pd.DataFrame(response_cond.json()[u'data'])
	pres = pd.DataFrame(response_pres.json()[u'data'])
	sali = pd.DataFrame(response_sali.json()[u'data'])

#    wind = pd.DataFrame(response_wind.json()[u'data'])
#    airp = pd.DataFrame(response_airp.json()[u'data'])
#    rain = pd.DataFrame(response_rain.json()[u'data'])

	if sensor_only == False:
		response_wzdh = requests.get(bosch_url + str(site_cfg['wz_data']) + '/datastreams/9?' + ss + es, auth=('OysterAdmin', '1234'))

		rl = len(response_wzdh.json()['data'])
		at_pressure = []
		for i in range(0, rl):
			at_pressure.append(response_wzdh.json()['data'][i]['value']['countries'][0]['locations'][0]['historical_observation'][0]['pressure'])

		wind_speed = []
		for i in range(0, rl):
			wind_speed.append(response_wzdh.json()['data'][i]['value']['countries'][0]['locations'][0]['historical_observation'][0]['wind_speed'])

		wind_dir = []
		for i in range(0, rl):
			wind_dir.append(response_wzdh.json()['data'][i]['value']['countries'][0]['locations'][0]['historical_observation'][0]['wind_direction_compass'])

		at_temp = []
		for i in range(0, rl):
			at_temp.append(response_wzdh.json()['data'][i]['value']['countries'][0]['locations'][0]['historical_observation'][0]['temperature'])

		dew_point = []
		for i in range(0, rl):
			dew_point.append(response_wzdh.json()['data'][i]['value']['countries'][0]['locations'][0]['historical_observation'][0]['dew_point'])

		rain_trace = []
		for i in range(0, rl):
			rain_trace.append(response_wzdh.json()['data'][i]['value']['countries'][0]['locations'][0]['historical_observation'][0]['rainfall_since_9am'])

		timestamp = []
		for i in range(0, rl):
			timestamp.append(response_wzdh.json()['data'][i]['value']['countries'][0]['locations'][0]['historical_observation'][0]['utc_time'])
			timestamp[i] = np.datetime64(timestamp[i])



#    rain['timestamp']= pd.to_datetime(rain['timestamp'])
#    rain.rename(columns={'rainfall' : 'at_rain_rate',}, inplace=True)
#    airp['timestamp']= pd.to_datetime(airp['timestamp'])
#    airp.rename(columns={'pressure' : 'at_pressure',}, inplace=True)

		wzdh = pd.DataFrame({'at_pressure' : at_pressure,
							 'wind_spd_kmh' : wind_speed,
							 'wind_dir' : wind_dir,
							 'at_temp' : at_temp,
							 'at_dew_point' : dew_point,
							 'at_rain_rate' : rain_trace,
							 'timestamp' : timestamp})
		wzdh_utc = wzdh.set_index('timestamp').tz_localize('utc')
		wzdh_utc.sort_index(inplace = True)

		wzdh_utc['at_rain_rate'] = odat.calc_rain_rate(wzdh_utc['at_rain_rate'])
		wind = odat.conv_wind(wzdh_utc[['wind_spd_kmh', 'wind_dir']])
		wzdh_utc['at_zonal_wind'] = wind['zonal']
		wzdh_utc['at_merid_wind'] = wind['merid']

		wzdh_utc = wzdh_utc.resample('10T').mean().interpolate(limit = 5)

# Convert the timestamp from an object into a datetime object
# rename the column of data from a generic term to an explicit term.
	temp['timestamp']= pd.to_datetime(temp['timestamp'])
	temp.rename(columns={'value' : 'water_temp',}, inplace=True)
	cond['timestamp']= pd.to_datetime(cond['timestamp'])
	cond.rename(columns={'value' : 'water_conductivity',}, inplace=True)
	pres['timestamp']= pd.to_datetime(pres['timestamp'])
	pres.rename(columns={'value' : 'water_pressure',}, inplace=True)
	sali['timestamp']= pd.to_datetime(sali['timestamp'])
	sali.rename(columns={'value' : 'water_salinity',}, inplace=True)

	temp_utc = temp.set_index('timestamp').tz_localize('utc')
	cond_utc = cond.set_index('timestamp').tz_localize('utc')
	pres_utc = pres.set_index('timestamp').tz_localize('utc')
	sali_utc = sali.set_index('timestamp').tz_localize('utc')
#    rain_utc = rain.set_index('timestamp').tz_localize('utc')
#    airp_utc = airp.set_index('timestamp').tz_localize('utc')


# Resample rainfall separately as different padding is required
#    rain_utc = rain_utc.resample('10T').pad(limit = 5)
	if sensor_only == False:
		site_obs_data = pd.concat([temp_utc,
								   cond_utc,
								   pres_utc,
								   sali_utc,
								   wzdh_utc],
								  axis = 1)
	else:
		site_obs_data = pd.concat([temp_utc,
								   cond_utc,
								   pres_utc,
								   sali_utc],
								  axis = 1)

	site_obs_data = site_obs_data.resample('10T').mean()
	site_obs_data.sort_index(inplace = True)
	return site_obs_data.loc[obs_window_start:nom_fcst_time]




def new_obs_file(site_id, nom_fcst_time):
	"""Initialise a new obs dataframe for the current month and save in HDF5"""

	nf = calendar.monthrange(nom_fcst_time.year, nom_fcst_time.month)[1] * 24 * 6
	start_time = dt.datetime(nom_fcst_time.year, nom_fcst_time.month, 1, 0, 0, 0)
	index = pd.date_range(start_time, periods = nf, freq = '10T', tz = 'utc')
	obs_data = pd.DataFrame({'at_pressure':np.nan,
							'at_temp':np.nan,
							'at_rain_rate':np.nan,
							'at_zonal_wind':np.nan,
							'at_merid_wind':np.nan,
							'at_humidity':np.nan,
							'at_dew_point':np.nan,
							'at_cloud_oktas':np.nan,
							'water_temp':np.nan,
							'water_level':np.nan,
							'water_pressure':np.nan,
							'water_conductivity':np.nan,
							'water_salinity':np.nan},
							index = index)

	oysters_root = os.environ['OYSTERS_ROOT']
	site_obs_file = oysters_root + '/oysters/sensors/' + site_id + '/obs/' + nom_fcst_time.strftime('%Y%m') + '_obs.hdf5'
#    obs_data.to_hdf(site_obs_file, 'table')
	# log_obs.info('Creating new obs data file for site %s: %s', site_id, nom_fcst_time.strftime('%Y%m') + '_obs.hdf5')
	print('Creating new obs data file for site %s: %s', site_id, nom_fcst_time.strftime('%Y%m') + '_obs.hdf5')
	store = pd.HDFStore(site_obs_file)
	store['obs_data'] = obs_data
	store.close()

	return 0



def store_obs(site_obs_data, site_id, nom_fcst_time):
	"""Store observation data in obs HDF data file"""

	oysters_root = os.environ['OYSTERS_ROOT']
	site_obs_file = oysters_root + '/oysters/sensors/' + site_id + '/obs/' + nom_fcst_time.strftime('%Y%m') + '_obs.hdf5'
	store = pd.HDFStore(site_obs_file)
	obs_data = store['obs_data']

	obs_data.update(site_obs_data)

	store['obs_data'] = obs_data
	store.close()

	if nom_fcst_time.day == 1:
		prev_site_obs_file = oysters_root + '/oysters/sensors/' + site_id + '/obs/' + (nom_fcst_time - dt.timedelta(1)).strftime('%Y%m') + '_obs.hdf5'
		if os.access(prev_site_obs_file, os.F_OK):
			prev_store = pd.HDFStore(prev_site_obs_file)
			prev_obs_data = prev_store['obs_data']
			prev_obs_data.update(site_obs_data)
			prev_store['obs_data'] = prev_obs_data
			prev_store.close()

	return 0


def read_obs_arch(site_id):
	"""Retrieve all of the observation data stored locally"""

	oysters_root = os.environ['OYSTERS_ROOT']
	obs_path = oysters_root + '/oysters/sensors/' + site_id + '/obs/'
	file_names = os.listdir(obs_path)
	obs_data = 0

	for file_name in file_names:
		if file_name[-4:] == 'hdf5':
			store = pd.HDFStore(obs_path + file_name)
			if type(obs_data) == type(0):
				obs_data = store['obs_data']
			else:
				obs_data = obs_data.combine_first(store['obs_data'])
			store.close()

	return obs_data



def get_tide(site_cfg, nom_fcst_time):
	"""Get tidal prediction data as output from Tappy"""

	oysters_root = os.environ['OYSTERS_ROOT']
	site_tide_file = oysters_root + '/oysters/sensors/' + site_cfg['ident'] + '/tide/' + str(nom_fcst_time.year) + '_' + site_cfg['ident'] + '.hdf5'
	if not os.access(site_tide_file, os.F_OK):
		new_tide_file(site_cfg, nom_fcst_time)

#    start_time = (nom_fcst_time - dt.timedelta(2)).strftime()
#    end_time = (nom_fcst_time + dt.timedelta(4)).strftime()
#    wheres = ['index > "' + start_time + '"', 'index < "' + end_time + '"']
	start_time = nom_fcst_time - dt.timedelta(2)
	end_time = nom_fcst_time + dt.timedelta(oana.fcst_num_cycles * oana.fcst_cycle_length)
	wheres = ['index > start_time', 'index < end_time']
	site_tide_data = pd.read_hdf(site_tide_file, 'tide_table', where = wheres)

	return site_tide_data


def read_tide(site_id):
	"""Get whole year tidal prediction data as output from Tappy"""

	oysters_root = os.environ['OYSTERS_ROOT']
	site_tide_file = oysters_root + '/oysters/sensors/' + site_cfg['ident'] + '/tide/' + str(nom_fcst_time.year) + '_' + site_cfg['ident'] + '.hdf5'
	site_tide_data = pd.read_hdf(site_tide_file)

	return site_tide_file


def new_tide_file(site_cfg, nom_fcst_time):
	"""Convert tide csv data file to pandas data frame and store in HDF"""

	oysters_root = os.environ['OYSTERS_ROOT']
	site_tide_csv = oysters_root + '/oysters/sensors/' + site_cfg['ident'] + '/tide/' + str(nom_fcst_time.year) + '_' + site_cfg['ident'] + '.tid'
	site_tide_data = pd.read_csv(site_tide_csv, sep=' ', names=['', 'astro_tide'], parse_dates=[0], index_col=[0])
	site_tide_data = site_tide_data.tz_localize('utc').resample('10T').pad(limit=0)
# Interpolate AstroTide to same temporal resolution as Pressure observations, do this
# in chunks so that calculations don't bog down
	for i in np.arange(0, len(site_tide_data), 200):
		site_tide_data.iloc[i:i + 220] = site_tide_data.iloc[i:i + 220].interpolate(method='quadratic')

# Time of the first high tide of the year
	start_time = site_tide_data.head(150).idxmax()

# site_tide_data.index is the tide times. StartTime[0] is reference start time for comparison
# once the calc is done the number needs to be converted to a timedelta in the format of minutes
# This can then be divided by a the number of minutes in a tide cycle (odat.cycle_length) to
# give a float that represents the tidal cycle number rather than day.
# This value is used for tide comparison and comparison of things related to tide.

	site_tide_data['tide_cycle'] = (site_tide_data.index - start_time[0]).astype('timedelta64[m]') / 60.0 / odat.cycle_length
	site_tide_data['int_tide_cycle'] = np.floor(site_tide_data['tide_cycle']).astype('int')
	site_tide_data['part_tide_cycle'] = site_tide_data['tide_cycle'] - site_tide_data['int_tide_cycle']

	site_tide_file = oysters_root + '/oysters/sensors/' + site_cfg['ident'] + '/tide/' + str(nom_fcst_time.year) + '_' + site_cfg['ident'] + '.hdf5'
	site_tide_data.to_hdf(site_tide_file, 'tide_table', format = 'table')

	return 0


def new_ty_file(site_id, nom_fcst_time):
	"""Initialise a new TY dataframe for the current month and save in HDF5"""

	nf = calendar.monthrange(nom_fcst_time.year, nom_fcst_time.month)[1] * 2       # number of forecasts in the current month
	start_time = dt.datetime(nom_fcst_time.year, nom_fcst_time.month, 1, 6, 0, 0)  # nominal time of first forecast
	index = pd.date_range(start_time, periods = nf, freq = '12H', tz = 'utc')      # define index at 12 hour resolution
	fcst_arr = np.empty(odat.forecast_length)                                      # define empty array
	fcst_arr[:] = np.nan                                                           # fill empty array with nans
	fcst_arr_list = [fcst_arr] * nf                                 # repeat empty array for number of forecasts in current month
	ty_data = pd.DataFrame({'pred_tide':fcst_arr_list,
							'pred_salinity':fcst_arr_list,
							'pred_temp':fcst_arr_list},
							index = index)



	oysters_root = os.environ['OYSTERS_ROOT']
	site_ty_file = oysters_root + '/oysters/sensors/' + site_id + '/ty/' + nom_fcst_time.strftime('%Y%m') + '_ty.hdf5'
#    ty_data.to_hdf(site_ty_file, 'table')
# following line previously raised error, now fixed - check
	# log_ty.info('Creating new TY file for site %s: %s', site_id, nom_fcst_time.strftime('%Y%m') + '_ty.hdf5')
	print('Creating new TY file for site %s: %s', site_id, nom_fcst_time.strftime('%Y%m') + '_ty.hdf5')
	store = pd.HDFStore(site_ty_file)
	store['ty_data'] = ty_data
	store.close()

	return 0



def store_ty(forecast, parameter, site_id, nom_fcst_time):
	"""Store TY forecast data in site TY HDF file"""

	oysters_root = os.environ['OYSTERS_ROOT']
	file_name = oysters_root + '/oysters/sensors/' + site_id + '/ty/' + nom_fcst_time.strftime('%Y%m') + '_ty.hdf5'
	if not os.access(file_name, os.F_OK):
		new_ty_file(site_id, nom_fcst_time)

	store = pd.HDFStore(file_name)
	ty_data = store['ty_data']

# The following line sometimes raises a SetWithCopyWarning but is actually ok
	if type(forecast) != int:
		# log_global.info('Writing %s prediction to archive for site %s: %s', parameter, site_id, nom_fcst_time)
		print('Writing %s prediction to archive for site %s: %s', parameter, site_id, nom_fcst_time)
#        ty_data['pred_tide'].loc[nom_fcst_time] = np.asarray(pred_tide.resample('H').mean())
		ty_data['pred_' + parameter].loc[nom_fcst_time] = np.asarray(forecast)
	else:
		#log_global.critical('No %s prediction to archive for site %s: %s', parameter, site_id, nom_fcst_time)
		print('No %s prediction to archive for site %s: %s', parameter, site_id, nom_fcst_time)

	store['ty_data'] = ty_data
	store.close()

	return 0


def read_ty_arch(site_id):
	"""Retrieve all of the observation data stored locally"""

	oysters_root = os.environ['OYSTERS_ROOT']
	ty_path = oysters_root + '/oysters/sensors/' + site_id + '/ty/'
	file_names = os.listdir(ty_path)
	ty_data = 0

	for file_name in file_names:
#        print file_name
#        print file_name[-8:]
		if file_name[-8:] == '_ty.hdf5':
			store = pd.HDFStore(ty_path + file_name)
			if type(ty_data) == type(0):
				ty_data = store['ty_data']
			else:
				ty_data = ty_data.combine_first(store['ty_data'])
			store.close()

	return ty_data


def store_ty_latest(forecast, parameter, site_id, nom_fcst_time):
	"""Store TY forecast data in site TY HDF file"""

	oysters_root = os.environ['OYSTERS_ROOT']
	path = oysters_root + '/oysters/sensors/' + site_id + '/ty/'

	if type(forecast) != int:
		# log_global.info('Writing %s prediction to latest record for site %s: %s', parameter, site_id, nom_fcst_time)
		print('Writing %s prediction to latest record for site %s: %s', parameter, site_id, nom_fcst_time)
		if os.access(path + 'latest_' + parameter + '.hdf5', os.F_OK):
			os.remove(path + 'latest_' + parameter + '.hdf5')
		store = pd.HDFStore(path + 'latest_' + parameter + '.hdf5')
#        print forecast.keys()
#        store['pred_' + parameter] = forecast.resample('H').mean()
		store['pred_' + parameter] = forecast
		store.close()

	return 0


def read_ty_latest(parameter, site_id):

	oysters_root = os.environ['OYSTERS_ROOT']
	path = oysters_root + '/oysters/sensors/' + site_id + '/ty/'
	store = pd.HDFStore(path + 'latest_' + parameter + '.hdf5')
	forecast = store['pred_' + parameter]
	store.close()

	return forecast
