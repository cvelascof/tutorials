"""Tutorial 1: Motion field estimation and extrapolation forecast

The tutorial guides you into the basic notions and techniques for extrapolation 
nowcasting. 
"""

from __future__ import division
from __future__ import print_function

import sys
import os
import datetime as datetime
import numpy as np
import matplotlib.pylab as plt
import pickle

sys.path.append('../') # add root pySTEPS dir to system path

from iotools   import archive, importers, utils
from motion    import optflow, advection
from datatools import conversion, dimension
from stepsnwc.nowcast_generators import simple_advection
from visualization.precipfields  import plot_precip_field
from visualization.motionfields  import plot_motion_field_quiver, plot_motion_field_streamplot
from verification.detcatscores   import scores_det_cat_fcst

# List of case studies that can be used in this tutorial

#+-------+--------------+-------------+----------------------------+
#| event |  start_time  | data_source | description                |
#+=======+==============+=============+============================+
#|  01   | 201701311000 |     mch     | orographic precipitation   |
#+-------+--------------+-------------+----------------------------+
#|  02   | 201505151600 |     mch     | non-stationary field       |
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
                "decl_grid"         : 10,
                "min_nr_samples"    : 1,
                "kernel_bandwidth"  : None
                }

## visualization parameters
colorscale      = 'STEPS-BE'
motion_plot     = 'streamplot' # streamplot or quiver

## verification parameters
skill_score     = 'CSI'
verif_thr       = 1 # [mmhr]
 
# Read-in the data
print('Read the data...')

## data specifications
if data_source == "fmi":
    fn_pattern      = "%Y%m%d%H%M_fmi.radar.composite.lowest_FIN_SUOMI1"
    fn_ext          = "pgm.gz"
    time_step_min   = 5 # timestep between two radar images
    data_units      = 'dBZ'
    importer        = importers.read_pgm
    importer_kwargs = {"gzipped":True}

if data_source == "mch":
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
R, _, _ = utils.read_timeseries(input_files, importer, **importer_kwargs)
print("Size of data array (n_times, n_rows, n_cols):", R.shape)
orig_field_dim = R.shape

# Prepare input files
print('Prepare the data...')

## convert units
if data_units is 'dBZ':
    R = conversion.dBZ2mmhr(R, R_threshold)

## make sure we work with a square domain
R_ = dimension.square_domain(R, 'crop')

## convert linear rainrates to logarithimc dBR units
dBR, dBRmin = conversion.mmhr2dBR(R_, R_threshold)
dBR[~np.isfinite(dBR)] = dBRmin

## visualize the input radar fields
doanimation     = False
savefig         = False
loop = 0
nloops = 2
while loop < nloops:
    
    for i in xrange(R.shape[0]):
        plt.clf()
        if doanimation:
            plot_precip_field(R_[i,:,:], None, units='mmhr', colorscale=colorscale, 
                              title=input_files[1][i].strftime('%Y-%m-%d %H:%M'), 
                              colorbar=True)
                    
            plt.pause(.5)
        
        if loop==0 and savefig:
            time_str = input_files[1][i].strftime('%Y%m%d%H%M')
            figName = path_outputs + '/radar_obs_' + time_str + '.png'
            plt.savefig(figName)
            print('Saved: ', figName)
    
    if doanimation:
        plt.pause(1)
    loop += 1
plt.close()
    
# YOUR TURN:
# Is the radar animation OK? Do the data look correct and in the right order?
# If yes, then delete the command below to continue this tutorial.

# sys.exit()

# Compute motion field
print('Computing motion vectors...')

oflow_method = optflow.get_method(oflow_method)
UV = oflow_method(dBR, **oflow_kwargs) 

## plot the motion field
doanimation     = False
savefig         = False
loop = 0
nloops = 3
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
plt.close()

# YOUR TURN:
# Try to modify some of the optical flow parameters to see what changes in the
# estimation of the motion field. 
# Do the precipitation patterns move along the estimated motion field?
# If yes, then delete the command below to continue this tutorial.

# sys.exit()

# Perform the advection of the radar field
print('Computing extrapolation...')

adv_method = advection.get_method(adv_method) 
dBR_forecast = adv_method(dBR[-1,:,:], UV, n_lead_times) 

## convert the forecasted dBR to mmhr
R_forecast = conversion.dBR2mmhr(dBR_forecast, R_threshold)

## plot the nowcast...
doanimation     = False
savefig         = False
loop = 0
nloops = 5
while loop < nloops:
    
    for i in xrange(R_.shape[0] + n_lead_times):
        plt.clf()
        if doanimation:
            if i < R_.shape[0]:
                plot_precip_field(R_[i,:,:], None, units='mmhr', colorscale=colorscale, 
                              title=input_files[1][i].strftime('%Y-%m-%d %H:%M'), 
                              colorbar=True)
                
            else:
                plot_precip_field(R_forecast[i - R_.shape[0],:,:], None, units='mmhr', 
                                  title='%s +%02d min' % 
                                  (input_files[1][-1].strftime('%Y-%m-%d %H:%M'),
                                  (1 + i - R_.shape[0])*time_step_min),
                                  colorscale='STEPS-BE', colorbar=True)
            plt.pause(.5)
        
    if doanimation:
        plt.pause(1)
    loop += 1
plt.close()

# Forecast verification

## find the verifying observations
input_files = archive.find_by_date(startdate + datetime.timedelta(minutes=n_lead_times*time_step_min), 
                                   path_inputs, "", fn_pattern, fn_ext, 
                                   time_step_min, n_lead_times - 1)
if all(fpath is None for fpath in input_files[0]):
    raise ValueError("Verification data not found")

## read observations
Robs, _, _ = utils.read_timeseries(input_files, importer, **importer_kwargs)

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

## if already exists, load the figure object
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