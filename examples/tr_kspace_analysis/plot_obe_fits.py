"""
This file provides a script that plots the simulated AC with the time resolved data for a specific Datapoint.
The script can only be used with a DataSets containing the the fit parameters from a cuda lifetime simulation.
The simulated AC is plotted together with the measured Datapoints to assess the quality of the fit.
The methods work on time resolved nD DataSet imported with snomtools.
"""

import os.path
import snomtools.data.datasets as ds
import matplotlib.pyplot as plt
import snomtools.calcs.units as u
import numpy as np

# Define DataSet containing the fit parameters from cuda simulations
fith5 = os.path.abspath("fitparameter HDF5 file")
# Define DataSet containing the time resolved data
datah5 = os.path.abspath("tr HDF5 file")

# Read in data
fitdata = ds.DataSet.from_h5(fith5)
data = ds.DataSet.from_h5(datah5)

# Change unit of delay axes to be the same for both DataSets
fitdata_um = u.to_ureg(fitdata.delay.data, 'fs').to('um', 'light')
data_fs = u.to_ureg(data.delay.data, 'um').to('fs', 'light')

# Define time zero of the measurement
time_zero_um = u.to_ureg("Time Zero", 'um')  # example : 3
time_zero_fs = u.to_ureg(time_zero_um).to('fs', 'light')

# Parameters to select the datapoint of interest
energy = "energy channel"  # example : 14
x = "x-coordinate"  # example : 8
y = "y-coordinate"  # example : 19

# Shift
# Chose in the axis which scale you want to use
delay_shifted = u.to_ureg(data.delay.data - time_zero_um, 'um')
fit_shifted = u.to_ureg(fitdata.delay.data + time_zero_fs, 'fs')

# Trigger for saving the plot
save = False
figname = "sample_ACfit_energy_pixel"  # example : 'Au788_ACfit_2,62eV_x34y31_SSmiddle_SHG'

# Plot settings:
# Colormap
cmap = plt.get_cmap('PuBu_r')
colors = [cmap(i) for i in np.linspace(0, 1, 10)]
# Plot
plt.figure(figsize=(7, 4.8))
ax = plt.subplot(111)
# Axis
# Chose data_fs + fit_shifted for fs scale or delay_shifted + fitdata_um for um scale
ax.plot(data_fs, data.binned_counts.data[:, energy, y, x], '-o', fillstyle='none', markersize=10, label="data point",
        color=colors[3])
ax.plot(fit_shifted, fitdata.obefit.data[:, 0, y, x], '-', markersize=10, label="fit curve", color='Red')
ax.tick_params(axis='both', labelsize=12)
# If using um scale change unit accordingly
plt.xlabel("time delay / $(fs)$", fontsize=14)
plt.ylabel("countrate / $(arb. u.)$", fontsize=14)
# Legend
ax.legend(facecolor='white', framealpha=1, edgecolor='black', loc='upper right', ncol=1, fontsize=12)

if save:
    plt.savefig(figname, bbox_inches='tight', dpi=200)
plt.show()
