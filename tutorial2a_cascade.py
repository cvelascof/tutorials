"""Tutorial 2a: Cascade decomposition and generation of stochastic noise

This tutorial demonstrates the cascade decomposition.
"""

import datetime
from matplotlib import cm, ticker
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
gridres = 1.0

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
R = importer(fn, **importer_kwargs)[0]

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
filter = filter_gaussian(dBR.shape[0], num_cascade_levels, gauss_scale=0.15, 
                         gauss_scale_0=0.2)

# plot the bandpass filter weights
fig = plt.figure()
ax = fig.gca()

for k in xrange(num_cascade_levels):
    ax.semilogx(np.linspace(0, L/2, len(filter["weights_1d"][k, :])), 
                filter["weights_1d"][k, :], "k-", 
                basex=pow(0.5*L/3, 1.0/(num_cascade_levels-2)))

ax.set_xlim(1, L/2)
ax.set_ylim(0, 1)
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
xt = np.hstack([[1.0], filter["central_freqs"][1:]])
ax.set_xticks(xt)
ax.set_xticklabels(["%.2f" % cf for cf in filter["central_freqs"]])
ax.set_xlabel("Radial wavenumber $|\mathbf{k}|$")
ax.set_ylabel("Normalized weight")
ax.set_title("Bandpass filter weights")

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
