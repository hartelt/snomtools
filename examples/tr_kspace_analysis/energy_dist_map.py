"""
This file provides a script that plots a energy distribution map for a DataSet measured in k-space.
The script can be used for both k_x and k_y direction.
The DataSet gets projected onto the energy, k_|| plane.
The methods work on a nD DataSet imported with snomtools.
"""

import snomtools.evaluation.kscalecalibration
import os.path
import snomtools.data.datasets as ds
import matplotlib.pyplot as plt
import snomtools.plots.datasets

# Load experimental data, copy to new target and project dispersion data:
# Define run you want to scale
k_h5 = os.path.abspath("Kscaled HDF5")  # example : "example_data_set_Kscaled.hdf5"
data = ds.DataSet.from_h5(k_h5)

# Projects dataset on energy, k_|| plane
dispersion_data = snomtools.evaluation.kscalecalibration.load_dispersion_data(data, y_axisid="k_|| axis",
                                                                              # example : 'k_y'
                                                                              x_axisid="k_|| axis",
                                                                              # example : 'k_x'
                                                                              e_axisid="energy-axis",
                                                                              # example : 'energy'
                                                                              x_window="Amount",
                                                                              # example : 10
                                                                              d_axisid=False
                                                                              )
# Example for k_y direction, for k_x switch k_y and k_x and vise versa

# Trigger for saving Image, with figname as name of saved file
save = False
figname = "Energy Distribution Map" # example : 'sample_edm_k_y'

# Plot settings :
plt.figure(figsize=(6.9,4.8))
ax = plt.subplot(111)
lifetime_plot = snomtools.plots.datasets.project_2d(dispersion_data, ax, 'energy', 'k_y', cmap='BuPu_r',vmin=0)
ax.invert_yaxis()  # project_2d flips the y axis as it assumes standard matrix orientation, so flip it back.)
# Axes
ax.tick_params(axis='both', labelsize=12)
plt.xlabel("$k_{||,y}$ / $(\AA^{-1})$", fontsize=14)
plt.ylabel("$E_{Interm.}$ / (eV)", fontsize=14)
# Colorbar
cb = plt.colorbar(lifetime_plot, ax=ax)
cb.set_label("countrate / $(arb. u.)$", labelpad=6, size=14)
cb.ax.tick_params(labelsize=12)
if save:
    plt.savefig(figname, bbox_inches='tight', dpi=200)
plt.show()