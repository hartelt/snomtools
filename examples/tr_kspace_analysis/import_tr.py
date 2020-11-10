"""
This file provides a script to import time resolved data folder measured with PEEM in k-space.
The chosen runs are each imported in a separate hdf5 file and an energycalibration is applied.
The methods work on a 4D DataSet imported with snomtools.
"""

import snomtools.data.imports.tiff
import snomtools.evaluation.peem_dld
import os

# Define folders that should be imported
rawdatafolder = os.path.abspath("Add folderpath")  # example : "E:\\Messdaten\\20200117_Au788_Zheng"
ecal = os.path.join(rawdatafolder, "ecal_folder", "ecal_kalfit_txt")
trfolder = os.path.join(rawdatafolder, "Add tr_datafolder")

# Define runs with PEEM Ch3 specific folder structure that should be imported
runs = ["{0}. Durchlauf".format(n) for n in range(x, y)]  # example : range(1, 2)


def runfolder(run):
    return os.path.join(trfolder, run)


# Renames target file ending
def h5_target(run):
    return run + ".hdf5"


# Define Dataset for desired runs and import them
print("Importing all runs...")
for run in runs:
    # Imports run
    print("Reading in " + run)
    data = snomtools.data.imports.tiff.tr_normal_folder_peem_dld_terra(runfolder(run), h5_target(run))
    # Applys energycalibration
    snomtools.evaluation.peem_dld.energy_apply_calibration(data, ecal)
    data.saveh5()
