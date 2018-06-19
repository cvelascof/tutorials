"""Tutorial 2: Cascade decomposition and generation of stochastic noise

This tutorial demonstrates 
"""

import datetime
from matplotlib import cm
import matplotlib.pylab as plt
import numpy as np
import sys

sys.path.append("../") # add root pySTEPS dir to system path

from datatools import conversion, dimension
from iotools import archive, importers
from stepsnwc.cascade.bandpass_filters import filter_gaussian
from stepsnwc.cascade.decomposition import decomposition_fft
from visualization.precipfields  import plot_precip_field

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
path_inputs     = ""
path_outputs    = ""

num_cascade_levels = 6
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

# convert precipitation intensity (mm/hr) to dBR for the cascade decomposition
dBR, dBRmin = conversion.mmhr2dBR(R, R_threshold)
dBR[~np.isfinite(R)] = dBRmin
dBR[dBR < dBRmin] = dBRmin

# plot the Fourier transform of the input field
F = abs(np.fft.fftshift(np.fft.fft2(dBR)))
fig = plt.figure()
L = F.shape[0]
im = plt.imshow(np.log(F**2), vmin=4, vmax=24, cmap=cm.jet, 
                extent=(-L/2, L/2, -L/2, L/2))
cb = fig.colorbar(im)
plt.xlabel("Wavenumber $k_x$")
plt.ylabel("Wavenumber $k_y$")
plt.title("Log-power spectrum of dBR")

# construct the bandpass filter
filter = filter_gaussian(dBR.shape[0], num_cascade_levels)

# compute the cascade decomposition
decomp = decomposition_fft(dBR, filter)

# plot the normalized cascade levels (mean zero and standard deviation one)
mu,sigma = decomp["means"],decomp["stds"]
for k in xrange(num_cascade_levels):
  dBR_k = decomp["cascade_levels"][k, :, :]
  dBR_k = (dBR_k - mu[k]) / sigma[k]
  fig = plt.figure()
  im = plt.imshow(dBR_k, cmap=cm.jet, vmin=-6, vmax=6)
  cb = fig.colorbar(im)
  cb.set_label("Rainfall rate (dBR)")
  plt.xticks([])
  plt.yticks([])
  plt.title("Normalized cascade level %d" % (k+1))

plt.show()
