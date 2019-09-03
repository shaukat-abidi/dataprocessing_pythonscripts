import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
import scipy.stats as stats
from datetime import datetime, timedelta

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from keras.layers.recurrent import LSTM

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from copy import deepcopy

from keras.models import load_model

df_gfs = pd.read_csv("C://Shaukat//code//data_rep//gfs//csv_files//gfs_orford_clipped.csv")
df_bom = pd.read_csv("C://Shaukat//code//data_rep//bom//Orford.csv")

# Setting up GFS
df_gfs['Unnamed: 0'] = pd.to_datetime(df_gfs['Unnamed: 0'])
df_gfs.set_index('Unnamed: 0', inplace = True)

# Setting up BOM
# df_bom['date_only'] = df_bom.UtcTime.dt.date
df_bom.UtcTime = pd.to_datetime(df_bom.UtcTime)
df_bom.set_index('UtcTime', inplace=True)

# Trim df_bom and df_gfs from november 2016
df_bom = df_bom.loc['2016-09-08 00:00:00':]
df_gfs = df_gfs.loc['2016-09-08 00:00:00':]

# Prepare groundtruth dataframe from BOM 
tot_gfs_entries = len(df_gfs.index)
df_gt = pd.DataFrame(index=df_gfs.index, columns=['rainfall_t+1','rainfall_t+2','rainfall_t+3'])

for dt_gfs in range(0,tot_gfs_entries):
    # Iterate each date in GFS and get the groundtruth rainfall from BOM
    
    date_a = df_gfs.index[dt_gfs] # Forecast time in gfs (made every three hours)
    date_t1 = date_a + timedelta(hours=1) # rainfall_t+1
    date_t2 = date_a + timedelta(hours=2) # rainfall_t+2
    date_t3 = date_a + timedelta(hours=3) # rainfall_t+3
    print 'processing: ', date_a 

    # df_patch for t+1 (Gather groundtruth rainfall from BOM)
    df_patch = df_bom.loc[date_a:date_t1].copy()
    rainfall_t1 = df_patch.RainfallLast10Minutes.sum()
    print 'date_t1: ', date_t1 

    
    # df_patch for t+2 (Gather groundtruth rainfall from BOM)
    df_patch = df_bom.loc[date_t1:date_t2].copy()
    rainfall_t2 = df_patch.RainfallLast10Minutes.sum()
    print 'date_t2: ', date_t2 

    
    # df_patch for t+3 (Gather groundtruth rainfall from BOM)
    df_patch = df_bom.loc[date_t2:date_t3].copy()
    rainfall_t3 = df_patch.RainfallLast10Minutes.sum()
    print 'date_t3: ', date_t3, '\n' 

    
    # Add it to the groundtruth df
    df_gt.loc[date_a] = [rainfall_t1, rainfall_t2, rainfall_t3]
    
    # Delete variables
    del date_a
    del date_t1
    del date_t2
    del date_t3
    del rainfall_t1
    del rainfall_t2
    del rainfall_t3
    
col_list = ['pred_cloud_cover', 'pred_cloud_cover_bound_cloud_layer',
       'pred_convective_cloud', 'pred_dewp', 'pred_high_tcc',
       'pred_low_tcc', 'pred_lw_rad', 'pred_max_wind_press',
       'pred_merid_wind', 'pred_middle_tcc', 'pred_rain_rate',
       'pred_rel_humidity', 'pred_sunshine', 'pred_surf_geowind', 'pred_surf_momentum_vflux', 'pred_surface_pressure', 'pred_sw_rad',
       'pred_temp', 'pred_total_rain', 'pred_ustorm', 'pred_vstorm',
       'pred_wind_speed_surf', 'pred_zonal_wind']
df_X = df_gfs[col_list].copy()
df_Y = df_gt.copy()

# Make sure that df_X is not null
df_X_bool = df_X.isnull().any().any()
print df_X_bool, 'THIS MUST BE FALSE' # This must be FALSE

# df_X.isnull().any()

df_data_matrix = df_X.as_matrix()
df_gt_rainfall_forecast_matrix = df_Y.as_matrix()
print 'Shape of data matrix: ',df_data_matrix.shape
print 'Shape of label matrix: ',df_gt_rainfall_forecast_matrix.shape

# Normalize data matrix with zero mean and unit covariance
# http://scikit-learn.org/stable/modules/preprocessing.html
df_data_matrix = preprocessing.scale(df_data_matrix)

data_matrix = np.copy(df_data_matrix)
label_matrix = np.copy(df_gt_rainfall_forecast_matrix)

sliding_window = 3
tot_samples = data_matrix.shape[0]
sample_dimension = data_matrix.shape[1]
print "Total Sample: ",tot_samples
print "Dimension of each sample: ",sample_dimension
    
# Accumulate examples here in this list
input_lstm = []
label_lstm = []
label_entry = sliding_window

    
for sequence in range(0,tot_samples-sliding_window):
    input_lstm.append(data_matrix[sequence:sequence+sliding_window, :])
    label_lstm.append(label_matrix[label_entry,0:3])
    print "sequence, sequence+sliding_window, label_entry: ",sequence,sequence+sliding_window,label_entry
    label_entry = label_entry + 1


return_data_matrix_lstm = np.array(input_lstm)
return_label_matrix_lstm = np.array(label_lstm)

print "LSTM input: ",return_data_matrix_lstm.shape
print "LSTM output: ",return_label_matrix_lstm.shape

# Prepare test and train points
tot_points = return_data_matrix_lstm.shape[0]
train_points = int(np.floor(0.7*tot_points)) 
test_points = tot_points - train_points
print 'tot_points: ', tot_points, ' train_points: ', train_points, ' test_points: ', test_points

# Generate Train Sequence
x_train = np.copy(return_data_matrix_lstm[0:train_points + 1,:])
y_train = np.copy(return_label_matrix_lstm[0:train_points + 1,:])
# Generate Test Sequence
x_test = np.copy(return_data_matrix_lstm[train_points + 1:tot_points + 1 ,:])
y_test = np.copy(return_label_matrix_lstm[train_points + 1:tot_points + 1 ,:])

print 'x_train_shape: ', x_train.shape, ' y_train_shape: ', y_train.shape
print 'x_test_shape: ', x_test.shape, ' y_test_shape: ', y_test.shape

input_dim = x_train.shape[2] # Input vector dimension for LSTM (N x timesteps x dim of seq)
output_dim = y_train.shape[1] # Output dimension for LSTM (rainfall for t+1,t+2,t+3)

print 'Input Dim: ', input_dim, ' Output Dim: ', output_dim

# create model
model = Sequential()
layers = [input_dim, 50, 50, 100, output_dim]

model.add(LSTM(input_dim=layers[0],output_dim=layers[1],return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(layers[2], return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(layers[3], return_sequences=False))
model.add(Dropout(0.3))

model.add(Dense(output_dim=layers[4]))
model.add(Activation("linear"))

    
model.compile(loss="mse", optimizer="rmsprop")

filepath_savemodel = 'C://Shaukat//code//rainfall//model_stored/orford//model_lstm_orford_v2.h5'

checkpointer = ModelCheckpoint(filepath_savemodel, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0.001)

history = model.fit(x_train, y_train, nb_epoch=1000, verbose=1, validation_data=(x_test, y_test), batch_size = train_points, callbacks=[checkpointer, reduce_lr])

    
