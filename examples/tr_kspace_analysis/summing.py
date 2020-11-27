# coding=utf-8

"""
This file provides a script to sum up multiple time resolved HDF5 DataSets measured with PEEM in k-space.
The methods works with 4D DataSets imported with snomtools.
"""


import snomtools.data.datasets as ds
import sys

if '-v' in sys.argv:
    verbose = True
else:
    verbose = False

# Define runs to sum over
runs = ["{0}. Durchlauf.hdf5".format(n) for n in range(1, 3)]

# Reads in 'runs' from the HDF5 files
run_datasets = []
for run in runs:
    print("Reading in " + run)
    data = ds.DataSet.in_h5(run)
    run_datasets.append(data)

# Sum up runs
# Change name of end product file
print("Summing runs...")
h5_target_all_runs = "Destination"  # example "runs1_2"
summed_data = ds.DataSet.add(run_datasets, "Destination", h5target=h5_target_all_runs)
summed_data.saveh5()
