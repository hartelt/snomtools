"""
This file provides a script to shift the energy-axis in respect to different states.
Usually either final-, intermediate- or binding-energy are looked at.
To apply the transformation, the kinetic energy at the fermi edge has to be extracted from the data.
The methods work on a nD DataSet imported with snomtools.
"""

import snomtools.data.datasets as ds
import snomtools.calcs.units as u
from snomtools.evaluation.pes import FermiEdge

# Define DataSet and rename target file
file = "HDF5 File"  # example : "example_data_set.hdf5"
full_data = ds.DataSet.from_h5(file, file.replace('.hdf5', '_Int.hdf5'))

energy_id = "energy-axis"  # example : 'energy'
E_kin_fermi = "kinetic energy at fermi edge"  # example : u.to_ureg(34.0798, 'eV')
E_laser = "photon energy"  # example : u.to_ureg(4.6, 'eV')

# TIP: A Fermi fit to data can be done easily with the class snomtools.evaluation.pes.FermiEdge! Example:
# fermifit = FermiEdge(full_data,
#                      # instead of the full data, a ROI can be used, to not use the full projected spectrum
#                      guess=(u.to_ureg(34.0798, 'eV'), u.to_ureg(0.1, 'eV'), u.to_ureg(1), u.to_ureg(0.01)),
#                      # guess = (E_f, width of the edge, height, lower level)
#                      normalize=True
#                      # Fit to normalized data so we can guess height 1, since we don't need the height and offset.
#                      )
# E_kin_fermi = fermifit.E_f

# Transforming energyscale to Final energy
final_axis = full_data.get_axis(energy_id) - E_kin_fermi + 2 * E_laser
# Transforming energyscale to Intermediate energy
interm_axis = full_data.get_axis(energy_id) - E_kin_fermi + E_laser
# Transforming energyscale to Binding energy
binding_axis = full_data.get_axis(energy_id) - E_kin_fermi

# Choose axis to which you want to scale # example : interm_axis
shifted_axis = interm_axis
# Changes axis of DataSet
full_data.replace_axis(energy_id, shifted_axis)
# Saves transformed DataSet with target name
full_data.saveh5()
