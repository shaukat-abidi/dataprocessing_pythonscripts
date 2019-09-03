Change two things for running it over VM
1) Line 65 of update_gfs.py
2) name, lat and lon of tas_pw_02.cfg

To record data:
python update_gfs.py tas_pw_02 2016-01-01T00:00:00 2017-03-01T00:00:00 1

GFS files will be stored in the folder named gfs/

