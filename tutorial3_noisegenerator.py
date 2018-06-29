#!/bin/env python

"""Tutorial 3: Generation of stochastic noise

This tutorial demonstrates the stochastic noise generators.

More info: https://pysteps.github.io/
"""

from __future__ import division
from __future__ import print_function

import datetime
from matplotlib import cm
import matplotlib.pylab as plt
import numpy as np
import sys
import os

sys.path.append("../") 
from datatools import conversion, dimension
from iotools import archive, importers
from perturbation import precip_generators
from visualization.precipfields import plot_precip_field

# List of case studies that can be used in this tutorial

#+-------+--------------+-------------+----------------------------+
#| event |  start_time  | data_source | description                |
#+=======+==============+=============+============================+
#|  01   | 201701311000 |     mch     | orographic precipitation   |
#+-------+--------------+-------------+----------------------------+
#|  02   | 201505151600 |     mch     | non-stationary field       |
#+-------+--------------+-------------+----------------------------+
#|  03   | 201609281500 |     fmi     | stratiform rain band       |
#+-------+--------------+-------------+----------------------------+
#|  04   | 201705091100 |     fmi     | large convective system    |
#+-------+--------------+-------------+----------------------------+

# Set parameters for this tutorial

## input data (copy/paste values from table above)
startdate_str = "201701311000"
data_source   = "mch"

## Read-in data and output paths (to be set beforehand in file data_paths.py)
from data_paths import path_inputs, path_outputs

## parameters
R_threshold = 0.1 # [mmhr]
perturbation_method = "nonparametric" # nonparametric, nested
num_realizations = 7

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

fn = archive.find_by_date(startdate, path_inputs, path_fmt, fn_pattern, fn_ext, time_step_min)[0]
if fn is None:
    raise ValueError("Input data not found in %s" % path_inputs)
R = importer(fn, **importer_kwargs)[0]

## make sure we work with a square domain
R = dimension.square_domain(R, "crop")

## convert units
if data_units is "dBZ":
    R = conversion.dBZ2mmhr(R, R_threshold)

# plot the input field
fig = plt.figure()
plot_precip_field(R, units="mmhr", title="Input field")
plt.show()

# convert precipitation intensity (mm/hr) to dBR
dBR, dBRmin = conversion.mmhr2dBR(R, R_threshold)
dBR[~np.isfinite(R)] = dBRmin
dBR[dBR < dBRmin] = dBRmin

# initialize the filter for generating the noise
# the Fourier spectrum of the input field in dBR is used as a filter (i.e. the 
# "nonparametric" method)
# this produces a noise field having spatial correlation structure similar to 
# the input field
init_noise, generate_noise = precip_generators.get_method(perturbation_method)
F = init_noise(dBR)

# plot four realizations of the stochastic noise
nrows = int(np.ceil((1+num_realizations)/4.))
plt.subplot(nrows,4,1)
for k in range(num_realizations+1):
    if k==0:
        plt.subplot(nrows,4,k+1)
        plot_precip_field(R, units="mmhr", title="Rainfall field", colorbar=False)
    else:
        N = generate_noise(F)
        
        plt.subplot(nrows,4,k+1)
        plt.imshow(N, cmap=cm.jet)
        plt.xticks([])
        plt.yticks([])
        plt.title("Noise field %d" % (k+1))

plt.show()

print("\n*****", os.path.basename(__file__), "run successfully! *****")
