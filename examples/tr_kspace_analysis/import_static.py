"""
This file provides a script to import a static tif file measured with PEEM in k-space.
The imported file is saved in hdf5 format and an energycalibration is applied.
The methods work on a 3D DataSet imported with snomtools.
"""

import snomtools.data.imports.tiff
import snomtools.evaluation.peem_dld
import os

# Define folder structure that should be imported
# Example is for a typical folder structure
rawdatafolder = os.path.abspath("Add folderpath")  # example: "E:\\Messdaten\\20200117_Au788_Zheng"
ecal = os.path.join(rawdatafolder, "ecal_folder", "ecal_kalfit_txt")
staticfolder = os.path.join(rawdatafolder, "Add static datafolder")  # example "02_kspace"
tiffname = "tiff data_name"  # example "01_kspace_THG_GI_Texp45m3s.tif"
tiff_path = os.path.join(staticfolder, tiffname)

# Renames target file ending
data_h5 = tiffname.replace('.tif', '.hdf5')

# Define Dataset and imports it
print("Importing static tiff-stack...")
# Imports run
data = snomtools.data.imports.tiff.peem_dld_read_terra(tiff_path)
# Applys energycalibration
snomtools.evaluation.peem_dld.energy_apply_calibration(data, ecal)
# Saves Dataset in a HDF5 file
data.saveh5(h5dest=data_h5)
