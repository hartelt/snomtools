"""
This file provides a script that plots a lifetime map for specific E-channels of a DataSet measured in k-space.
The script can only be used on DataSets containing the lifetimes, usually simulated with cuda.
The chosen energy channels are projected onto the k_|| plane.
The methods work on time resolved nD DataSet imported with snomtools.
"""

import os.path
import snomtools.data.datasets as ds
import matplotlib.pyplot as plt
import snomtools.data.transformation.project
import snomtools.plots.datasets
import numpy as np

# Define simulated DataSet containing the extracted lifetimes
tr_h5 = os.path.abspath("Lifetime HDF5")  # example : "1. Durchlauf_e4_binned_int_kscaled_lifetimes_eslice17.hdf5"
data = ds.DataSet.from_h5(tr_h5)

sum_boundaries_index = {}
sum_boundaries_index["Energy Axis"] = [0, 0]  # example : 'energy binned x4'

# Initialize RoI:
sumroi = ds.ROI(data, sum_boundaries_index, by_index=True)

# Project RoI to k_||-Plane and return data:
kmap = snomtools.data.transformation.project.project_2d(sumroi, 'k_y', 'k_x')

# Trigger for saving Image, with figname as name of saved file
save = False
figname = "lifetime_map_name"  # example : 'sample_lifetime_map_energy_channel'

# Plot settings:
plt.figure(figsize=(7, 4.8))
ax = plt.subplot(111)
lifetime_plot = snomtools.plots.datasets.project_2d(kmap, ax, 'k_y', 'k_x', cmap='PuBu_r')
ax.set_aspect('equal')
# Axes
ax.tick_params(axis='both', labelsize=12)
plt.xlabel("$k_x$ / $(\AA^{-1})$", fontsize=14)
plt.ylabel("$k_y$ / $(\AA^{-1})$", fontsize=14)
# Colorbar
cb = plt.colorbar(lifetime_plot, ax=ax)
cb.set_label("lifetime / $(fs)$", labelpad=6, size=14)
cb.ax.tick_params(labelsize=12)
if save:
    plt.savefig(figname, bbox_inches='tight', dpi=200)
plt.show()
