# coding=utf-8

"""
This file provides a script to bin a nD DataSet in hdf5 format.
Usually uses hdf5 DataSets imported with import_static.py or import_tr.py measured e.g. with PEEM in k-space.
The methods work on a nD DataSet imported with snomtools.
"""

import os.path
import snomtools.data.datasets as ds
import snomtools.data.transformation.binning
import snomtools.data.fits
import sys

if '-v' in sys.argv:
    verbose = True
else:
    verbose = False

# Define DataSet to bin
data_h5 = os.path.abspath("Data HDF5")  # example : "example_data_set.hdf5"

# Binning:
# Choose in binAxisID what you want to bin: 'x','y','energy' and set factor in binFactor

# Load DataSet from file and rename target
binned_h5 = data_h5.replace('.hdf5', '_Binned.hdf5')
dataset = ds.DataSet.in_h5(data_h5)

# Default setting: Bin x&y by factor 5, energy by factor 2
binned_data = snomtools.data.transformation.binning.Binning(dataset, binAxisID=('x', 'y', 'energy'),
                                                            binFactor=(5, 5, 2))
# Change labels
binned_dataset = binned_data.bin(h5target=binned_h5)

# Save DataSet in new file with target name
binned_dataset.saveh5()

print("done.")
