"""Tutorial 1: Motion field estimation and extrapolation forecast

The tutorial guides you into the basic notions and techniques for extrapolation 
nowcasting. 
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
from datatools import conversion, dimension
from stepsnwc.nowcast_generators import simple_advection
from visualization.precipfields import plot_precip_field
from visualization.motionfields import plot_motion_field_quiver, plot_motion_field_streamplot

# List of case studies that can be used in this tutorial

#+-------+--------------+-------------+----------------------------+
#| event |  start_time  | data_source | description                |
#+=======+==============+=============+============================+
#|   1   | 201701311000 |     mch     | orographic precipitation   |
#+-------+--------------+-------------+----------------------------+
#|   2   | 201505151600 |     mch     | non-stationary field       |
#+-------+--------------+-------------+----------------------------+

# Set parameters for this tutorial

## input data (copy/paste values from table above)
startdate_str   = "201701311000"
data_source     = "mch"

## data paths
path_inputs     = "/scratch/ned/tmp/tutorial_data/in"
path_outputs    = "/scratch/ned/tmp/tutorial_data/out"

## methods
oflow_method    = "lucaskanade"
adv_method      = "semilagrangian"

## forecast parameters
n_lead_times    = 12
R_threshold     = 0.1 # [mmhr]

## optical flow parameters
oflow_kwargs    = {
                # to control the smoothness of the motion field
                "decl_grid"         : 3,
                "min_nr_samples"    : 1,
                "kernel_bandwidth"  : None
                }

## visualization parameters
colorscale      = 'STEPS-BE'
motion_plot     = 'streamplot' # streamplot or quiver
 
# Read-in the data
print('Read the data...')

## data specifications
if data_source=="fmi":
    fn_pattern      = "%Y%m%d%H%M_fmi.radar.composite.lowest_FIN_SUOMI1"
    fn_ext          = "pgm.gz"
    time_step_min   = 5 # timestep between two radar images
    data_units      = 'dBZ'
    importer        = importers.read_pgm
    importer_kwargs = {"gzipped":True}

if data_source=="mch":
    fn_pattern      = "AQC%y%j%H%M?_00005.801"
    fn_ext          = "gif"
    time_step_min   = 5
    data_units      = 'mmhr'
    importer        = importers.read_aqc
    importer_kwargs = {}
    
startdate  = datetime.datetime.strptime(startdate_str, "%Y%m%d%H%M")

## find radar field filenames
input_files = archive.find_by_date(startdate, path_inputs, "", fn_pattern, fn_ext, 
                                   time_step_min, 2)
if all(fpath is None for fpath in input_files[0]):
    raise ValueError("Input data not found")

## read radar field files
R, geodata, metadata = utils.read_timeseries(input_files, importer, **importer_kwargs)
print("Size of data array (n_times, n_rows, n_cols):", R.shape)
orig_field_dim = R.shape

# Prepare input files
print('Prepare the data...')

## convert units
if data_units is 'dBZ':
    R = conversion.dBZ2mmhr(R, R_threshold)

## make sure we work with a square domain
R_ = dimension.square_domain(R, 'crop')

## visualize the input radar fields
print('Plot the data...')
doanimation     = False
savefig         = False
loop = 0
nloops = 5
while loop < nloops:
    
    for i in xrange(R.shape[0]):
        plt.clf()
        if doanimation:
            plot_precip_field(R[i,:,:], geodata, units='mmhr', colorscale=colorscale, 
                              title=input_files[1][i], colorbar=True)
                    
            plt.pause(.5)
        
        if loop==0 and savefig:
            time_str = input_files[1][i].strftime('%Y%m%d%H%M')
            figName = path_outputs + '/radar_obs_' + time_str + '.png'
            plt.savefig(figName)
            print('Saved: ', figName)
    
    if doanimation:
        plt.pause(1)
    loop += 1
    
# YOUR TURN:
# Is the radar animation OK? Do the data look correct and in the right order?
# If yes, then delete the command below to continue this tutorial.

sys.exit()

## convert linear rainrates to logarithimc dBR units
dBR, dBRmin = conversion.mmhr2dBR(R_, R_threshold)
dBR[~np.isfinite(dBR)] = dBRmin

# Compute motion field
print('Computing motion vectors...')

oflow_method = optflow.get_method(oflow_method)
UV = oflow_method(dBR, **oflow_kwargs) 

print('Plot the motion field...')
doanimation     = True
savefig         = True
loop = 0
nloops = 5
while loop < nloops:
    
    for i in xrange(R.shape[0]):
        plt.clf()
        if doanimation:
            plot_precip_field(R_[i,:,:], None, units='mmhr', colorscale=colorscale, 
                              title='Motion field', colorbar=True)
            if motion_plot=='quiver':
                plot_motion_field_quiver(UV, None, 20)
            if motion_plot=='streamplot':    
                plot_motion_field_streamplot(UV, None)        
            plt.pause(.5)
        
        if loop==0 and savefig:
            time_str = input_files[1][i].strftime('%Y%m%d%H%M')
            figName = path_outputs + '/radar_motionfield_' + time_str + '.png'
            plt.savefig(figName)
            print('Saved: ', figName)
    
    if doanimation:
        plt.pause(1)
    loop += 1

# YOUR TURN:
# Try to modify some of the optical flow parameters to see what changes in the
# estimation of the motion field. 
# Do the precipitation patterns move along the estimated motion field?
# If yes, then delete the command below to continue this tutorial.

sys.exit()

###### EXTRAPOLATE LAST FIELD BASED ON MOTION
print('Computing extrapolation...')
adv_method = advection.get_method(adv_method) 
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

