"""
This file provides a script that plots a constant energy map for specific E-channels of a DataSet measured in k-space.
The script can be used for both static and time resolved DataSets.
The chosen energy channels (multiple channels are possible by setting the RoI) get projected onto the k_|| plane.
The methods work on a nD DataSet imported with snomtools.
"""

import os.path
import snomtools.data.datasets as ds
import matplotlib.pyplot as plt
import snomtools.data.transformation.project
import snomtools.plots.datasets
import numpy as np

# Define DataSet to display
k_h5 = os.path.abspath("Kscaled HDF5")  # example : "example_data_set_Kscaled.hdf5"
data = ds.DataSet.from_h5(k_h5)

# Define energy slice to plot
sum_boundaries_index = {}
sum_boundaries_index["Energy Axis"] = ["Channel", "Channel"]  # example : ['energy']  = [16,16]

# Initialize RoI:
sumroi = ds.ROI(data, sum_boundaries_index, by_index=True)

# Project RoI to k_||-plane and return data:
kmap = snomtools.data.transformation.project.project_2d(sumroi, 'k_y', 'k_x')

# Trigger for saving Image, with figname as name of saved file
save = False
figname = "k_map_name"  # example : 'sample_kspace_map_energy_channel'

# Plot settings :
plt.figure(figsize=(7, 4.8))
ax = plt.subplot(111)
lifetime_plot = snomtools.plots.datasets.project_2d(kmap, ax, 'k_y', 'k_x', cmap='BuPu_r')
ax.set_aspect('equal')
# Axes
ax.tick_params(axis='both', labelsize=12)
plt.xlabel("$k_x$ / $(\AA^{-1})$", fontsize=14)
plt.ylabel("$k_y$ / $(\AA^{-1})$", fontsize=14)
# Colorbar
cb = plt.colorbar(lifetime_plot, ax=ax)
cb.set_label("countrate / $(arb. u.)$", labelpad=6, size=14)
cb.ax.tick_params(labelsize=12)
if save:
    plt.savefig(figname, bbox_inches='tight', dpi=200)
plt.show()
