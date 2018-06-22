#!/bin/env python

"""Tutorial 1: Motion field estimation and extrapolation forecast

The tutorial guides you into the basic notions and techniques for extrapolation 
nowcasting. 

More info: https://pysteps.github.io/
"""

from __future__ import division
from __future__ import print_function

import sys
import os
import datetime as datetime
import numpy as np
import matplotlib.pylab as plt
import pickle

sys.path.append('../')
from iotools   import archive, importers, utils
from motion    import optflow, advection
from datatools import conversion, dimension
from stepsnwc.nowcast_generators import simple_advection
from visualization.precipfields  import plot_precip_field
from visualization.motionfields  import plot_motion_field_quiver, plot_motion_field_streamplot
from verification.detcatscores   import scores_det_cat_fcst

# List of case studies that can be used in this tutorial

#+-------+--------------+-------------+----------------------------------------+
#| event |  start_time  | data_source | description                            |
#+=======+==============+=============+========================================+
#|  01   | 201701311000 |     mch     | orographic precipitation               |
#+-------+--------------+-------------+----------------------------------------+
#|  02   | 201505151600 |     mch     | non-stationary field, apparent rotation|
#+-------+--------------+------------------------------------------------------+
#|  03   | 201609281500 |     fmi     | stratiform rain band                   |
#+-------+--------------+-------------+----------------------------------------+
#|  04   | 201705091100 |     fmi     | widespread convective activity         |
#+-------+--------------+-------------+----------------------------------------+

# Set parameters for this tutorial

## input data (copy/paste values from table above)
startdate_str = "201701311000"
data_source   = "mch"

## Read-in data and output paths (to be set beforehand in file data_paths.py)
from data_paths import path_inputs, path_outputs

## methods
oflow_method    = "lucaskanade"
adv_method      = "semilagrangian"

## forecast parameters
n_lead_times    = 12
R_threshold     = 0.1 # [mmhr]

## optical flow parameters
oflow_kwargs    = {
                # to control the number of tracking objects
                "quality_level_ST"  : 0.05,
                "min_distance_ST"   : 5,
                "block_size_ST"     : 15,
                # to control the tracking of those objects
                "winsize_LK"        : (50, 50),
                "nr_levels_LK"      : 10,
                # to control the removal of outliers
                "max_speed"         : 10,
                "nr_IQR_outlier"    : 3,
                # to control the smoothness of the interpolation
                "decl_grid"         : 10,
                "min_nr_samples"    : 2,
                "kernel_bandwidth"  : None
                }

## visualization parameters
colorscale      = 'MeteoSwiss' # MeteoSwiss or STEPS-BE
motion_plot     = 'streamplot' # streamplot or quiver

## verification parameters
skill_score     = 'CSI'
verif_thr       = 1 # [mmhr]
 
# Read-in the data
print('Read the data...')

## data specifications
if data_source == "fmi":
    fn_pattern      = "%Y%m%d%H%M_fmi.radar.composite.lowest_FIN_SUOMI1"
    path_fmt        = "fmi/%Y%m%d"
    fn_ext          = "pgm.gz"
    data_units      = "dBZ"
    importer        = importers.read_pgm
    importer_kwargs = {"gzipped":True}
    grid_res_km     = 1.0
    time_step_min   = 5.0
elif data_source == "mch":
    fn_pattern      = "AQC%y%j%H%M?_00005.801"
    path_fmt        = "mch/%Y%m%d"
    fn_ext          = "gif"
    data_units      = "mmhr"
    importer        = importers.read_aqc
    importer_kwargs = {}
    grid_res_km     = 1.0
    time_step_min   = 5.0
    
startdate  = datetime.datetime.strptime(startdate_str, "%Y%m%d%H%M")

## find radar field filenames
input_files = archive.find_by_date(startdate, path_inputs, path_fmt, fn_pattern, fn_ext, time_step_min, 2)
if all(fpath is None for fpath in input_files[0]):
    raise ValueError("Input data not found in", path_inputs)

## read radar field files
R, _, _ = utils.read_timeseries(input_files, importer, **importer_kwargs)
print("Size of data array (n_times, n_rows, n_cols):", R.shape)
orig_field_dim = R.shape

# Prepare input files
print('Prepare the data...')

## convert units
if data_units is 'dBZ':
    R = conversion.dBZ2mmhr(R, R_threshold)

## make sure we work with a square domain
R = dimension.square_domain(R, 'crop')

## convert linear rainrates to logarithimc dBR units
dBR, dBRmin = conversion.mmhr2dBR(R, R_threshold)
dBR[~np.isfinite(dBR)] = dBRmin

## visualize the input radar fields
doanimation = True
nloops = 2 # how many times to loop

loop = 0
while loop < nloops:
    for i in xrange(R.shape[0]):
        plt.clf()
        if doanimation:
            plot_precip_field(R[i,:,:], None, units='mmhr', colorscale=colorscale, title=input_files[1][i].strftime('%Y-%m-%d %H:%M'), colorbar=True)
            plt.pause(.5)
    if doanimation:
        plt.pause(1)
    loop += 1

if doanimation == True:
    plt.close()
    
# YOUR TURN:
# Is the radar animation OK? Do the data look correct and in the right order?
# If yes, then comment out the "print(...)" and "sys.exit()" command below to continue this tutorial.
# Set the above doanimation = False to avoid the above animation.
print("\n***** To continue, open the file at line number", sys._getframe().f_lineno-3, 'and read the instructions. *****')
sys.exit()

# Compute motion field
print('Computing motion vectors...')

oflow_method = optflow.get_method(oflow_method) # This provides a callable function "oflow_method" for any type of optical flow function used.
UV = oflow_method(dBR, **oflow_kwargs) 

## plot the motion field
doanimation = True
nloops = 2

loop = 0
while loop < nloops:
    
    for i in xrange(R.shape[0]):
        plt.clf()
        if doanimation:
            plot_precip_field(R[i,:,:], None, units='mmhr', colorscale=colorscale, title='Motion field', colorbar=True)
            if motion_plot=='quiver':
                plot_motion_field_quiver(UV, None, 20)
            if motion_plot=='streamplot':    
                plot_motion_field_streamplot(UV, None)        
            plt.pause(.5)
        
    if doanimation:
        plt.pause(1)
    loop += 1

if doanimation == True:
    plt.close()

# YOUR TURN:
# Try to modify some of the optical flow parameters in oflow_kwargs to see what 
# changes in the estimation of the motion field. Which are the most sensitive 
# parameters? Check the quality of your motion field: do the precipitation patterns 
# move along the estimated motion field?
# If yes, then comment out the "print(...)" and "sys.exit()" command below to continue this tutorial.
# Set the above doanimation = False to avoid the above animation.
print("\n***** To continue, open the file at line number", sys._getframe().f_lineno-6, 'and read the instructions. *****')
sys.exit()

# Perform the advection of the radar field
print('Computing extrapolation...')

adv_method = advection.get_method(adv_method) 
dBR_forecast = adv_method(dBR[-1,:,:], UV, n_lead_times) 

## convert the forecasted dBR to mmhr
R_forecast = conversion.dBR2mmhr(dBR_forecast, R_threshold)
print('The forecast array has size [nleadtimes,nrows,ncols] =', R_forecast.shape)

## plot the nowcast...
doanimation     = True
savefig         = True
nloops = 2

loop = 0
while loop < nloops:
    
    for i in xrange(R.shape[0] + n_lead_times):
        plt.clf()
        if doanimation:
            if i < R.shape[0]:
                # Plot last observed rainfields
                plot_precip_field(R[i,:,:], None, units='mmhr', colorscale=colorscale, 
                              title=input_files[1][i].strftime('%Y-%m-%d %H:%M'), 
                              colorbar=True)
                if savefig & (loop == 0):
                    figname = "%s/%s_%s_simple_advection_%02d_obs.png" % (path_outputs, startdate_str, data_source, i)
                    plt.savefig(figname)
                    print(figname, 'saved.')
            else:
                # Plot nowcast
                plot_precip_field(R_forecast[i - R.shape[0],:,:], None, units='mmhr', 
                                  title='%s +%02d min' % 
                                  (input_files[1][-1].strftime('%Y-%m-%d %H:%M'),
                                  (1 + i - R.shape[0])*time_step_min),
                                  colorscale=colorscale, colorbar=True)
                if savefig & (loop == 0):
                    figname = "%s/%s_%s_simple_advection_%02d_nwc.png" % (path_outputs, startdate_str, data_source, i)
                    plt.savefig(figname)
                    print(figname, 'saved.')
            plt.pause(.5)
    if doanimation:
        plt.pause(1)
    loop += 1

if doanimation == True:
    plt.close()

# Forecast verification
print('Forecast verification...')

## find the verifying observations
input_files_verif = archive.find_by_date(startdate + datetime.timedelta(minutes=n_lead_times*time_step_min), 
                                   path_inputs, path_fmt, fn_pattern, fn_ext, 
                                   time_step_min, n_lead_times - 1)
if all(fpath is None for fpath in input_files_verif[0]):
    raise ValueError("Verification data not found")

## read observations
Robs, _, _ = utils.read_timeseries(input_files_verif, importer, **importer_kwargs)

## convert units
if data_units is 'dBZ':
    Robs = conversion.dBZ2mmhr(Robs, R_threshold)

## and square domain
Robs_ = dimension.square_domain(Robs, 'crop')

## compute verification scores
scores = np.zeros(n_lead_times)*np.nan
for i in xrange(n_lead_times):
    scores[i] = scores_det_cat_fcst(R_forecast[i,:,:], Robs_[i,:,:], verif_thr, 
                                   [skill_score])[0]

## if already exists, load the figure object to append the new verification results
filename = "%s/%s" % (path_outputs, "tutorial1_fig_verif")
if os.path.exists("%s.dat" % filename):
    ax = pickle.load(open("%s.dat" % filename, 'rb'))
    print("Figure object loaded: %s.dat" % filename) 
else:
    fig, ax = plt.subplots()
    
## plot the scores
nplots = len(ax.lines)
x = (np.arange(n_lead_times) + 1)*time_step_min
ax.plot(x, scores, color='C%i'%(nplots + 1), label = 'run %02d' % (nplots + 1))
ax.set_xlabel('Lead-time [min]')
ax.set_ylabel('%s' % skill_score)
plt.legend()

## dump the figure object
pickle.dump(plt.gca(), open("%s.dat" % filename, 'wb'))
print("Figure object saved: %s.dat" % filename)
# remove the pickle object to plot a new figure

plt.show()

# YOUR TURN:
# Now you can run the whole code multiple times with different combinations of 
# parameters and input files and you will be able to compare their skills in the
# verification plot. To start with a new clean plot, delete the tutorial1_fig_verif.dat
# file in your output folder.
print("\n*****", os.path.basename(__file__), "run successfully! *****")
print("\n***** To continue, open the file at line number", sys._getframe().f_lineno-5, 'and read the instructions. *****')
