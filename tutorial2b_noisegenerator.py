"""Tutorial 2a: Cascade decomposition and generation of stochastic noise

This tutorial demonstrates the stochastic noise generators.
"""

import datetime
from matplotlib import cm
import matplotlib.pylab as plt
import numpy as np
import sys

sys.path.append("../") # add root pySTEPS dir to system path

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
startdate_str = "201609281500"
data_source   = "fmi"

## data paths
path_inputs  = ""
path_outputs = ""

R_threshold = 0.1 # [mmhr]

## data specifications
if data_source == "fmi":
    fn_pattern      = "%Y%m%d%H%M_fmi.radar.composite.lowest_FIN_SUOMI1"
    fn_ext          = "pgm.gz"
    time_step_min   = 5 # timestep between two radar images
    data_units      = "dBZ"
    importer        = importers.read_pgm
    importer_kwargs = {"gzipped":True}
elif data_source == "mch":
    fn_pattern      = "AQC%y%j%H%M?_00005.801"
    fn_ext          = "gif"
    time_step_min   = 5
    data_units      = "mmhr"
    importer        = importers.read_aqc
    importer_kwargs = {}

startdate = datetime.datetime.strptime(startdate_str, "%Y%m%d%H%M")

fn = archive.find_by_date(startdate, path_inputs, "", fn_pattern, fn_ext, time_step_min)[0]
R = importers.read_pgm(fn, gzipped=True)[0]

## make sure we work with a square domain
R = dimension.square_domain(R, "crop")

## convert units
if data_units is "dBZ":
    R = conversion.dBZ2mmhr(R, R_threshold)

# plot the input field
fig = plt.figure()
plot_precip_field(R, units="mmhr", title="Input field")

# convert precipitation intensity (mm/hr) to dBR
dBR, dBRmin = conversion.mmhr2dBR(R, R_threshold)
dBR[~np.isfinite(R)] = dBRmin
dBR[dBR < dBRmin] = dBRmin

# initialize the filter for generating the noise
# the Fourier spectrum of the input field in dBR is used as a filter (i.e. the 
# "nonparametric" method)
# this produces a noise field having spatial correlation structure similar to 
# the input field
F = precip_generators.initialize_nonparam_2d_fft_filter(dBR)

# plot four realizations of the stochastic noise
for i in xrange(4):
    N = precip_generators.generate_noise_2d_fft_filter(F)

    plt.figure()
    plt.imshow(N, cmap=cm.jet)
    plt.xticks([])
    plt.yticks([])
    plt.title("Noise field %d" % (i+1))

plt.show()
