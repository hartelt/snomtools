"""
This file provides a script that plots an adjusted electron parabola to a DataSet measured in k-space.
The script can be used for both (k_x-, k_y-axis) directions, by adjusting the parameters.
The DataSet is projected onto (energy,k||)-plane (energy distribution map) and plotted with an electron parabola.
For adjusting the parabola to a specific band the plots are shown and can be saved.
The methods work on a nD DataSet imported with snomtools.
"""

import snomtools.evaluation.kscalecalibration
import os.path
import snomtools.data.datasets as ds
import snomtools.data.fits
import snomtools.calcs.units as u
import snomtools.calcs.constants as c
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

# Load experimental data, copy to new target and project dispersion data:
# Define DataSet you want to scale
k_h5 = os.path.abspath("Kscaled HDF5")  # example : "example_data_set_Kscaled.hdf5"
data = ds.DataSet.from_h5(k_h5)

# Parameters for fitting the Parabola to your data
# Example parameter for occ surface state
bandmass = None  # example: u.to_ureg(0.28 * c.m_e , 'kg')
e_offset = None  # example: u.to_ureg(33.5, 'eV')
zero = None  # example: u.to_ureg(0.01, 'angstrom**-1')

# Set axes labels, typically y_axisid is the k|| axis for the plot
x_axisid = "k||-axis"  # example : 'k_x'
y_axisid = "k||-axis"  # example : 'k_y'
e_axisid = "energy-axis"  # example : 'energy'

# Projects dataset on energy, y-pixel plane; x-pixel plane summed over for higher statistic
# Parameter : x_window represents amount of pixels summed together; d_axisid for tr data
dispersion_data = snomtools.evaluation.kscalecalibration.load_dispersion_data(data, y_axisid, x_axisid, e_axisid,
    x_window="Amount", d_axisid=False)  # example : x_window=10

# Example for dispersion along k_y direction

# Trigger for saving Image, with figname as name of saved file
save = False
figname = "Figure Name"  # example : 'sample_band_ky'

# Show parabola for intermediate state by plotting parabola along data:
(k_parallel, parab_data), bandmass = snomtools.evaluation.kscalecalibration.show_state_parabola(
    dispersion_data, figname, zero, bandmass, e_offset, y_axisid)

# Plot dispersion and ParabolaFit
plt.figure(figsize=(7, 4.8))
ax = plt.subplot(111)
# example for plot : cmap='BuPu_r', vmin=0, vmax=25000
snomtools.plots.datasets.project_2d(dispersion_data, ax, e_axisid, y_axisid, cmap="Colormap", vmin="Min", vmax="Max")
ax.plot(k_parallel, parab_data, 'r-', label="fit parabola")
# project_2d flips the y axis as it assumes standard matrix orientation, so flip it back.
ax.invert_yaxis()
ax.legend(loc='lower left')
ax.tick_params(axis='both', labelsize=12)
plt.xlabel("$k_{||,y}$ / $(\AA^{-1})$", fontsize=14)
# plt.xticks(np.arange(-1, 1.25, 0.25))
plt.ylabel("$E_{Interm.}$ / (eV)", fontsize=14)
cb = plt.colorbar(cm.ScalarMappable(cmap='BuPu_r'))
cb.set_label("normalized countrate", labelpad=6, size=14)
cb.ax.tick_params(labelsize=12)
if save:
    plt.savefig(figname, bbox_inches='tight', dpi=200)
plt.show()

# Returns effective bandmass, determined with adjusted parabola
print(bandmass)
