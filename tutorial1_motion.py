"""Tutorial 1: Motion field estimation and extrapolation forecast

The tutorial guides you into the basic notions and techniques for extrapolation 
nowcasting. 

There are two possible ways to run the tutorials:
    1. You make the python script executable ("chmod + x tutorial1_motion.py") 
        and run it (./tutorial1_motion.py).
    2. You open a python session and copy the script step by step so that you 
        have more time to understand how the code works.
"""

from __future__ import division
from __future__ import print_function

import sys
import datetime as datetime
import numpy as np
import matplotlib.pylab as plt

sys.path.append('../') # add root pySTEPS dir to system path

from iotools import archive, importers, utils
from motion import optflow, advection
from datatools.conversion import mmhr2dBR, dBR2mmhr, dBZ2mmhr
from stepsnwc.nowcast_generators import simple_advection
from visualization.precipfields import plot_precip_field

# List of case studies that can be used in this tutorial

#+-------+--------------+-------------+----------------------------+
#| event |  start_time  | data_source | description                |
#+=======+==============+=============+============================+
#|   1   | 201701121200 |     mch     | orographic precipitation   |
#+-------+--------------+-------------+----------------------------+
#|   2   | 201701121200 |     mch     | orographic precipitation   |
#+-------+--------------+-------------+----------------------------+

# Set parameters for this tutorial

## input data (copy/paste values from table above)
start_time     = "201701121200"
data_source    = "mch"

## data paths
path_inputs  = "/scratch/ned/tmp/tutorial_data/in"
path_outputs = "/scratch/ned/tmp/tutorial_data/out"

if path_inputs is None or path_outputs is None:
    raise ValueError("Please define paths to data folders")

## forecast parameters
n_lead_times = 12
 
# Read-in the data

## data specifications
if data_source=="fmi":
    fn_pattern = "%Y%m%d%H%M_fmi.radar.composite.lowest_FIN_SUOMI1"
    fn_ext     = "pgm.gz"
    time_step_min = 5 # timestep between two radar images

if data_source=="mch":
    fn_pattern = "AQC%y%j%H%M?_00005.801"
    fn_ext     = "gif"
    time_step_min = 5

## read-in previous rainfall fields
startdate  = datetime.datetime.strptime(start_time, "%Y%m%d%H%M")
input_files = archive.find_by_date(startdate, path_inputs, "", fn_pattern, fn_ext, time_step_min, 3)
if all(fpath is None for fpath in input_files[0]):
    raise ValueError("Input data not found")

R, geodata, metadata = utils.read_timeseries(input_files_prev, importers.read_pgm, gzipped=True)
if data_source=="fmi":
    R = dBZ2mmhr(dBZ, R_threshold=0.1)
    print(input_files_prev[1])

# Read-in future rainfall fields (for verification)
enddate = startdate + datetime.timedelta(minutes=time_step_min + time_step_min*n_next_images)
input_files_next = archive.find_by_date(enddate, path_inputs, "", fn_pattern, fn_ext, time_step_min, n_next_images-1)
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

