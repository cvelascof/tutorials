#!/bin/env python

"""Tutorial 4: Stochastic ensemble precipitation nowcasting

This tutorial brings all the material from the previous tutorials together in
order to generate a stochastic ensemble of precipitation nowcasts.

More info: https://pysteps.github.io/
"""

from __future__ import division
from __future__ import print_function

import datetime
import matplotlib.pylab as plt
import numpy as np
import sys
import os
import pickle

sys.path.append("../")
from datatools import conversion, dimension
from iotools import archive, importers, utils
from motion import optflow
from stepsnwc.nowcast_generators import steps
from visualization.precipfields import plot_precip_field
from verification.probscores import CRPS

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
oflow_method             = "lucaskanade"
extrap_method            = "semilagrangian"
bandpass_filter_method   = "gaussian" # gaussian or uniform
decomp_method            = "fft"
noise_method             = "nonparametric" # nonparametric or nested

## forecast parameters
n_lead_times        = 12
n_ens_members       = 3
n_cascade_levels    = 6
ar_order            = 2
R_threshold         = 0.1 # [mmhr]
prob_matching       = True

## visualization parameters
colorscale      = "MeteoSwiss" # MeteoSwiss or STEPS-BE

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
elif data_source == "bom":
    fn_pattern      = "2_%Y%m%d_%H%M00.prcp-cscn"
    path_fmt        = "bom/prcp-cscn/2/%Y/%m/%d"
    fn_ext          = "nc"
    data_units      = "mmhr"
    importer        = importers.read_bom_rf3
    importer_kwargs = {}
    grid_res_km     = 1.0
    time_step_min   = 6.0

startdate = datetime.datetime.strptime(startdate_str, "%Y%m%d%H%M")

## find radar field files
input_files = archive.find_by_date(startdate, path_inputs, path_fmt, fn_pattern, 
                                   fn_ext, time_step_min, ar_order)
if all(fpath is None for fpath in input_files[0]):
    raise ValueError("Input data not found")
    
## read radar field files
R, _, _ = utils.read_timeseries(input_files, importer, **importer_kwargs)

## make sure we work with a square domain
R = dimension.square_domain(R, "crop")

## convert units
if data_units is "dBZ":
    R = conversion.dBZ2mmhr(R, R_threshold)

## convert precipitation intensity (mm/hr) to dBR for the cascade decomposition
dBR, dBRmin = conversion.mmhr2dBR(R, R_threshold)
dBR[~np.isfinite(R)] = dBRmin

# Compute motion field
oflow_method = optflow.get_method(oflow_method)
UV = oflow_method(dBR)

# Generate the nowcast
dBR_forecast = steps(dBR, UV, n_lead_times, n_ens_members, 
                    n_cascade_levels, R_threshold, extrap_method, decomp_method, 
                    bandpass_filter_method, noise_method, grid_res_km, time_step_min, ar_order, 
                    None, None, False, False, prob_matching)

## convert to mmhr                    
R_forecast = conversion.dBR2mmhr(dBR_forecast, R_threshold)

# visualize the forecast
doanimation     = True
savefig         = True
nloops = 1

loop = 0
while loop < nloops:
    for n in range(n_ens_members):
        for i in range(R.shape[0] + n_lead_times):
            plt.clf()
            if doanimation:
                if i < R.shape[0]:
                    # Plot last observed rainfields
                    plot_precip_field(R[i,:,:], None, units="mmhr", colorscale=colorscale, 
                                  title=input_files[1][i].strftime("%Y-%m-%d %H:%M"), 
                                  colorbar=True)
                    if savefig and loop == 0 and n == 0:
                        figname = "%s/%s_%s_stochastic_extrapolation_%02d_obs.png" % (path_outputs, startdate_str, data_source, i)
                        plt.savefig(figname)
                        print("%s saved." % figname)
                else:
                    # Plot nowcast
                    plot_precip_field(R_forecast[n, i - R.shape[0], :, :], None, units="mmhr", 
                                      title="%s +%02d min (member %02d)" % 
                                      (input_files[1][-1].strftime("%Y-%m-%d %H:%M"),
                                      (1 + i - R.shape[0])*time_step_min, n),
                                      colorscale=colorscale, colorbar=True)
                    if savefig and loop == 0:
                        figname = "%s/%s_%s_stochastic_extrapolation_%02d_%02d_nwc.png" % (path_outputs, startdate_str, data_source, i, n)
                        plt.savefig(figname)
                        print("%s saved." % figname)
                plt.pause(.5)
        if doanimation:
            plt.pause(.5)
    if doanimation:
        plt.pause(1)
    loop += 1
plt.close()
    
# Forecast verification

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
Robs = dimension.square_domain(Robs, "crop")

## compute the average continuous ranked probability score (CRPS)
scores = np.zeros(n_lead_times)*np.nan
for i in range(n_lead_times):
    scores[i] = CRPS(R_forecast[:,i,:,:].reshape((n_ens_members, -1)).transpose(), 
                     Robs[i,:,:].flatten())

## if already exists, load the figure object to append the new verification results
filename = "%s/%s" % (path_outputs, "tutorial4_fig_verif")
if os.path.exists("%s.dat" % filename):
    ax = pickle.load(open("%s.dat" % filename, "rb"))
    print("Figure object loaded: %s.dat" % filename) 
else:
    fig, ax = plt.subplots()
    
## plot the scores
nplots = len(ax.lines)
x = (np.arange(n_lead_times) + 1)*time_step_min
ax.plot(x, scores, color="C%i"%(nplots + 1), label = "run %02d" % (nplots + 1))
ax.set_xlabel("Lead-time [min]")
ax.set_ylabel("CRPS")
plt.legend()

## dump the figure object
pickle.dump(plt.gca(), open("%s.dat" % filename, "wb"))
print("Figure object saved: %s.dat" % filename)
# remove the pickle object to plot a new figure

plt.show()

# YOUR TURN:
# Now you can run the whole code multiple times with different combinations of 
# parameters and input files and you will be able to compare their skills in the
# verification plot. To start with a new clean plot, delete the tutorial4_fig_verif.dat
# file in your output folder.

print("\n*****", os.path.basename(__file__), "run successfully! *****")
print("\n***** To continue, open the file at line number", sys._getframe().f_lineno-6, "and read the instructions. *****")
