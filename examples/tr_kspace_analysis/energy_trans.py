"""
This file provides a script to shift the energy-axis in respect to different states.
Usually either final-, intermediate- or binding-energy are looked at.
To apply the transformation, the kinetic energy at the fermi edge has to be extracted from the data.
This is done by using a Fermi fit from snomtools.evaluation.pes.FermiEdge.
To analyze the quality of the fit, the spectrum is plotted with the fit and saved.
The methods work on a nD DataSet imported with snomtools.
"""

import snomtools.data.datasets as ds
import snomtools.calcs.units as u
from snomtools.evaluation.pes import FermiEdge
import matplotlib.pyplot as plt
import numpy as np

# Define DataSet and rename target file
file = "HDF5 File"  # example : "example_data_set.hdf5"
full_data = ds.DataSet.from_h5(file, file.replace('.hdf5', '_Int.hdf5'))

energy_id = "energy-axis"  # example : 'energy'
E_laser = "photon energy"  # example : u.to_ureg(4.6, 'eV')

# Define a RoI in which the FermiEdge is located and excludes other peaks to perform a more accurate fit
fermi_boundaries = {}
fermi_boundaries[energy_id] = ["Channel1", "Channel2"]  # example : [0, 20]
# Initialize RoI:
fermi_area = ds.ROI(full_data, fermi_boundaries, by_index=True)

# Fermi fit used to extract the fermi edge
fermifit = FermiEdge(fermi_area, guess=None, normalize=False)
E_kin_fermi = fermifit.E_f

# Transforming energyscale to Final energy
final_axis = full_data.get_axis(energy_id) - E_kin_fermi + 2 * E_laser
# Transforming energyscale to Intermediate energy
interm_axis = full_data.get_axis(energy_id) - E_kin_fermi + E_laser
# Transforming energyscale to Binding energy
binding_axis = full_data.get_axis(energy_id) - E_kin_fermi

# Choose axis to which you want to scale # example : interm_axis
shifted_axis = interm_axis

# Plot settings:
# Colormap
cmap = plt.get_cmap('PuBu_r')
colors = [cmap(i) for i in np.linspace(0, 1, 10)]
# Plot
fig = plt.figure()
ax = plt.subplot(111)
# Plot the data as projected 1D spectrum
fermiplot = FermiEdge.extract_data(full_data)
# Use in case of normalized Fermi fit
# normalized_data = fermiplot.get_datafield('intensity').data / fermiplot.get_datafield('intensity')[
#     fermi_area.get_limits('energy', by_index=True)[1]]
ax.plot(fermiplot.get_axis('energy').data, fermiplot.get_datafield('intensity').data, '-o', fillstyle='none',
        markersize=10, label="projected 1D spectrum", color=colors[2])
# Plot the Fermi fit for the selected ROI
boundaries = fermi_area.get_limits('energy')
fit_energies = u.to_ureg(np.linspace(boundaries[1], boundaries[0], 1000))
fit = fermifit.fermi_edge(fit_energies)
ax.plot(fit_energies, fit, '-', fillstyle='none', markersize=10, label="fermi fit", color='Red')
# Axis
ax.set_yscale('log')
ax.tick_params(axis='both', labelsize=12)
plt.xlabel("kin. energy / $(eV)$", fontsize=14)
plt.ylabel("countrate / $(arb. u.)$", fontsize=14)
# Legend
ax.legend(facecolor='white', framealpha=1, edgecolor='black', loc='upper right', ncol=1, fontsize=12)
plt.savefig("spectrum_fitted_fremiedge", bbox_inches='tight', dpi=200)
plt.show()

# Changes axis of DataSet
full_data.replace_axis(energy_id, shifted_axis)
# Saves transformed DataSet with target name
full_data.saveh5()
