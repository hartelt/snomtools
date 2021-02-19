"""
This file provides a script to rotate a 3D DataSet.
The plane spanned by the chosen axes is rotated around a specific angle theta.
Usually used to project measure data to high symmetry directions and a workaround,
since snomtools.transformation.project can't project onto arbitrary angles (instead whole DataSet is rotated).
The methods work on a 3D DataSet imported with snomtools.
"""

import os.path
import snomtools.data.datasets as ds
import snomtools.data.transformation.rotate

# Define HDF5 file you want to rotate and name of target HDF5
file = os.path.abspath("HDF5 file")  # example: "example_data_set.hdf5"

# Load DataSet
full_data = ds.DataSet.from_h5(file)

# Set rotation angle, label of rotated angles
angle = "theta"  # example : 45
x_axisid = "x-axis"  # example : 'x'
y_axisid = "y-axis"  # example : 'y'

# Change target name depending on angle
file_end = '_' + str(angle) + 'deg.hdf5'
rot_data = file.replace('.hdf5', file_end)

print("Rotate...")

# Rotate DataSet and save in HDF5 file
# Uses the preliminary version on the rotations branch!
# ToDo: Make rotation work on large Datasets
# ToDo: Merge rotations to master

# Define the plane by the axes you want to rotate
rot_obj = snomtools.data.transformation.rotate.Rotation(full_data, angle, rot_plane_axes=(x_axisid, y_axisid))
rotated_data = rot_obj.rotate_data()

# Save rotated DataSet with target name
rotated_data.saveh5(rot_data)

print("Done.")
