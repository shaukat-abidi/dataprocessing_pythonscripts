#import logging
#import logging.config
import csv
import os

def get_site_cfg(site_id):
    """Return dictionary of site specific parameters"""
    oysters_root = os.environ['OYSTERS_ROOT']

    site_cfg_data = {'ident':str,
                     'name':str,
                     'lat':float,
                     'lon':float,
                     'msl_offset':float,
                     'sensor_data':int,
                     'wz_data':int,
                     'bom_stn':str,
                     'rain_lag':int,
                     'rain_rise':int,
                     'rain_relax':int,
                     'rain_scale':float,
                     'temp_lag_d':int,
                     'temp_lag_f':int,
                     'temp_rad_scale':float,
                     'temp_rad_offset':float,
                     'temp_ls_ratio':float,
                     'temp_taper':int,
                     'temp_scale':float,
                     'temp_scale_prof_a':float,
                     'temp_scale_prof_b':float,
                     'temp_scale_prof_c':float,
                     'temp_scale_prof_max':float}

    with open(oysters_root + '/oysters/sensors/' + site_id + '/' + site_id + '.cfg') as site_cfg_file:
        site_cfg = csv.reader(site_cfg_file)
        for row in site_cfg:
            site_cfg_data[row[0]] = site_cfg_data[row[0]](row[1])

    return site_cfg_data


def get_global_cfg():
    """Return list of site IDs"""
    oysters_root = os.environ['OYSTERS_ROOT']
    global_cfg_data = []

    with open(oysters_root + '/oysters/oysters.cfg') as global_cfg_file:
        global_cfg = csv.reader(global_cfg_file)
        for row in global_cfg:
            global_cfg_data.append(row[0])

    return global_cfg_data


def set_logging():
    """Set up log files and handlers"""
    oysters_root = os.environ['OYSTERS_ROOT']
    log_cfg_file = oysters_root + '/oysters/log.cfg'
    log_files = {'globlogfile':oysters_root + '/oysters/analysis/log/global.log',
                 'gfslogfile':oysters_root + '/oysters/analysis/log/gfs.log',
                 'obslogfile':oysters_root + '/oysters/analysis/log/obs.log',
                 'tylogfile':oysters_root + '/oysters/analysis/log/ty.log'}
    logging.config.fileConfig(log_cfg_file, defaults=log_files)

    return 0
