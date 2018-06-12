#!/usr/bin/env python

# Import basic python modules
from __future__ import division
from __future__ import print_function

import datetime as datetime
import numpy as np

import matplotlib.pylab as plt

# Add root STEPS directory to system PATH
import sys
sys.path.append('../')

# Import STEPS modules
from iotools import archive, importers, utils
from motion import optflow, advection
from datatools.conversion import mmhr2dBR, dBR2mmhr, dBZ2mmhr
from stepsnwc.nowcast_generators import simple_advection
from visualization.precipfields import plot_precip_field
from visualization.motionfields import plot_motion_field_quiver

####################
## Input parameters
root_path_data  = "/users/lforesti/steps-data"
path_figs  = "/users/lforesti/steps-out"

n_prev_images = 5 # Number of previous images to load (to estimate the motion field)
n_next_images = 6 # Number of futures images to forecast. This also loads the corresponding observed radar fields for verification.
time_step_min = 5 # Time step between two radar images

start_time = "201701121200" # Start time fo the nowcast

####################
###### READ-IN DATA
# Filename pattern and format
fn_pattern = "%Y%m%d%H%M_fmi.radar.composite.lowest_FIN_SUOMI1"
fn_ext     = "pgm.gz"

# Read-in previous rainfall fields
startdate  = datetime.datetime.strptime(start_time, "%Y%m%d%H%M")
input_files_prev = archive.find_by_date(startdate, root_path_data, "", fn_pattern, fn_ext, time_step_min, n_prev_images)
dBZ, geodata, metadata = utils.read_timeseries(input_files_prev, importers.read_pgm, gzipped=True)

R = dBZ2mmhr(dBZ, R_threshold=0.1)
print(input_files_prev[1])

# Read-in future rainfall fields (for verification)
enddate = startdate + datetime.timedelta(minutes=time_step_min + time_step_min*n_next_images)
input_files_next = archive.find_by_date(enddate, root_path_data, "", fn_pattern, fn_ext, time_step_min, n_next_images-1)
dBZ_obs,_,_ = utils.read_timeseries(input_files_next, importers.read_pgm, gzipped=True)

R_obs = dBZ2mmhr(dBZ_obs, R_threshold=0.1)

# Check the structure of the data
print("Size of data array (n_times, n_rows, n_cols):", R.shape)
n_rows = R.shape[1]
n_cols = R.shape[2]
field_dim = [n_rows, n_cols]

# Read-in a single file to get the geodata
xmin = geodata['x1']
xmax = geodata['x2']
ymin = geodata['y1']
ymax = geodata['y2']
extent = np.array([xmin, xmax, ymin, ymax])/1000

###### PLOT THE RADAR RAINFALL FIELDS
# Plot the sequence of radar fields
plt.figure()
for i in range(0, R.shape[0]):
    R_field = R[i,:,:]
    time = input_files_prev[1][i]
    
    # Plot fig
    plt.clf()
    plot_precip_field(R_field, units='mmhr', colorscale='STEPS-BE', extent=extent, title=time, colorbar=True)
    
    plt.pause(0.1)
    plt.draw()
    
    # Save fig
    time_str = time.strftime('%Y%m%d%H%M')
    figName = path_figs + '/radar_obs_' + time_str + '.png'
    plt.savefig(figName)
    print('Saved: ', figName)

# Convert linear rainrates to logarithimc dBR units
R_threshold = 0.1
dBR, dBRmin = mmhr2dBR(R, R_threshold)
dBR[~np.isfinite(dBR)] = dBRmin

###### COMPUTE MOTION FIELD
print('Computing motion vectors...')
oflow_method = optflow.get_method("lucaskanade") # Choose optical flow method to define a function handle
n_prev_images = 2 # Number of previous images to use (minimum = 2)
V = oflow_method(dBR[-n_prev_images:, :, :]) # Use the function handle to start the motion estimation
print('done.')

## Plot motion field
# plt.figure()
# R_field = R[0,:,:]
# plot_precip_field(R_field, units='mmhr', colorscale='STEPS-BE', colorbar=True)
# plot_motion_field_quiver(V, step=100)
# plt.show()

###### EXTRAPOLATE LAST FIELD BASED ON MOTION
print('Computing extrapolation...')
adv_method = advection.get_method("semilagrangian") # Choose advection method
dBR_forecast = adv_method(dBR[-1,:,:], V, n_next_images) # Extrapolate the last radar image for a certain number of time steps
print('done.')

# Convert the forecasted dBR to mmhr
R_forecast = dBR2mmhr(dBR_forecast, R_threshold=0.1)

# Plot nowcasts
plt.figure()
for i in range(0, R_forecast.shape[0]):
    R_field = R_forecast[i,:,:]
    time = input_files_next[1][i]
    
    # Plot fig
    plt.clf()
    plot_precip_field(R_field, units='mmhr', colorscale='STEPS-BE', extent=extent, title=time, colorbar=True)
    
    plt.pause(0.1)
    plt.draw()
    
    # Save fig
    time_str = time.strftime('%Y%m%d%H%M')
    figName = path_figs + '/radar_fx_' + time_str + '.png'
    plt.savefig(figName)
    print('Saved: ', figName)

###### SIMPLE FORECAST VERIFICATION 
# Compute RMSE between the forecast and the observations. The average is done over the n_rows and n_cols for each lead time.
RMSE_array = np.sqrt(np.nanmean((R_forecast - R_obs)**2, axis=(1,2)))

# Plot verification results
plt.figure()
plt.plot(RMSE_array)
plt.show()

