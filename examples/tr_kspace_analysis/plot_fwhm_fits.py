# coding=utf-8

"""
This file provides a script that plots fitted FWHM with the time resolved data for a specific Datapoint.
The script can only be used with a DataSets containing the the fit parameters from a nD fitobject from snomtools.
The fitted AC is plotted together with the measured Datapoints to assess the quality of the fit.
The methods work on time resolved nD DataSet imported with snomtools.
"""

import os.path
import snomtools.data.datasets as ds
import matplotlib.pyplot as plt
from snomtools.data.fits import Gauss_Fit_nD, Lorentz_Fit_nD, Sech2_Fit_nD
import numpy as np

# HDF5 Dataset to analyze quality of Fit
# Change name to fitted and binned dataset respectifely
fith5 = os.path.abspath("Fitted HDF5")  # example : "example_data_set_tr_Gaussfit.hdf5"
binnedh5 = os.path.abspath("Data HDF5")  # example : "example_data_set_tr.hdf5"

# Define AC of Interest
energy = "energy channel"  # example : 16
x = "x-Pixel"  # example: 33
y = "y-Pixel"  # example : 34

# Read in Dataset from binned and fitted HDF5
binned_data = ds.DataSet.from_h5(binnedh5)
fit_data = ds.DataSet.from_h5(fith5)

# Load time points and count rate
delay_axis = binned_data.get_axis('delay')
counts = binned_data.get_datafield(0)

# Chose the fitting type you used, possible: 'gaussian', 'lorentzian', 'sech2'
mode = "Fit Mode"

# Trigger for saving the plot
save = False
figname = "sample_FWHMfit_energy_pixel"  # example : 'Au788_FWHMfit_2,62eV_x31y23_SU_SHG'

# Plot settings:
# Colormap
cmap = plt.get_cmap('PuBu_r')
colors = [cmap(i) for i in np.linspace(0, 1, 10)]
# Plot
plt.figure(figsize=(7, 4.8))
ax = plt.subplot(111)
# Axis
ax.plot(delay_axis.data, counts[:, energy, y, x].data, '-o', fillstyle='none', markersize=10, label="data point",
        color=colors[3])
ax.tick_params(axis='both', labelsize=12)
plt.xlabel("time delay / $(um)$", fontsize=14)
plt.ylabel("countrate / $(arb. u.)$", fontsize=14)

# Plot Gaussian:
if mode == 'gaussian':
    # Load gauss fit parameters from DataSet into newly created Object
    fitparam = Gauss_Fit_nD()
    fitparam.import_parameters(fit_data)
    # Plot gaussian fit curve
    ax.plot(delay_axis.data, fitparam.gaussian(delay_axis, (energy, y, x)), '-', markersize=10,
            label="gaussian fit curve", color='Red')

# Plot Lorentzian:
elif mode == 'lorentzian':
    # Load lorentz fit parameters from DataSet into newly created Object
    fitparam = Lorentz_Fit_nD()
    fitparam.import_parameters(fit_data)
    # Plot lorentzian fit curve
    ax.plot(delay_axis.data, fitparam.lorentzian(delay_axis, (energy, y, x)), '-', markersize=10,
            label="lorentzian fit curve", color='Red')

# Plot Sech2:
elif mode == 'sech2':
    # Load sech2 fit parameters from DataSet into newly created Object
    fitparam = Sech2_Fit_nD()
    fitparam.import_parameters(fit_data)
    # Plot sech2 fit curve
    ax.plot(delay_axis.data, fitparam.sech2(delay_axis, (energy, y, x)), '-', markersize=10,
            label="sech2 fit curve", color='Red')

# Legend
ax.legend(facecolor='white', framealpha=1, edgecolor='black', loc='upper right', ncol=1, fontsize=12)

if save:
    plt.savefig(figname, bbox_inches='tight', dpi=200)
plt.show()
