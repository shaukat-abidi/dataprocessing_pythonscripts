# !/usr/bin/env python
import os
import sys
import numpy as np
import datetime as dt
#import logging
import pytz
import oysters_io as oio
import oysters_config as ocfg
import oysters_data as odat
#import oysters_analysis as oana

def update_gfs(site_id, start_date, end_date, rda):
    """Retrieves historical GFS data from NOMADS GRaDS or RDA THREDDS"""
    print('I am here\n')
    rda = int(rda)

    # log_global = logging.getLogger('glob')
    # log_global.info(start_date)
    # log_global(end_date)

    print(start_date)
    print(start_date)

    start_date = dt.datetime.strptime(start_date, '%Y-%m-%dT%H:%M:%S')
    start_date = start_date.replace(tzinfo = pytz.UTC)
    print(start_date)
    end_date = dt.datetime.strptime(end_date, '%Y-%m-%dT%H:%M:%S')
    end_date = end_date.replace(tzinfo = pytz.UTC)
    nom_fcst_time = odat.get_nom_fcst_time(start_date)
    print(end_date)


    # log_global.info('Initiating GFS update sequence for %s: %s - %s', site_id, start_date, end_date)
    print('Initiating GFS update sequence for %s: %s - %s' %(site_id, start_date, end_date) )

    site_cfg_data = ocfg.get_site_cfg(site_id)

    while nom_fcst_time < end_date:
        oio.update_gfs(site_cfg_data, nom_fcst_time, rda)
        nom_fcst_time += dt.timedelta(0.5)



def main(site_id, start_date, end_date, rda):
    # ocfg.set_logging()
    # log_global = logging.getLogger('glob')
    update_gfs(site_id, start_date, end_date, rda)

#    try:
#        update_gfs(site_id, start_date, end_date, rda)
#    except Exception as err:
        # log_global.exception('Unknown error in update_gfs utility for site %s', site_id)
#        print('Unknown error in update_gfs utility for site %s' %(site_id) )
#        return 1

if __name__ == "__main__":
    print('I am here before calling main\n')
    print(sys.argv[1])
    print(sys.argv[2])
    print(sys.argv[3])
    print(sys.argv[4])

    # set os.environ['OYSTERS_ROOT']
    os.environ['OYSTERS_ROOT'] = '/home/thorweather/shaukat/Peters_code_merimbula_v2'
    print(os.environ['OYSTERS_ROOT'])
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]))
