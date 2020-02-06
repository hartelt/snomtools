# -*- coding:utf-8 -*-

"""
This module contains evaluation scripts that run on Nvidia Cuda, using the pycuda module.

The Cuda C++ Source codes are in the .cu files, assembled with their absolute paths in the cuda_sources dict.

For using this module, a working pycuda installation on the system is necessary.
"""
import os

__author__ = "Lukas Hellbrück"


def is_cuda_sourcefile(filename):
	"""
	Checks if a filename is a cuda source file. (.cu)

	:param str filename: The filename.

	:rtype: bool
	"""
	return os.path.splitext(filename)[1] == ".cu"


module_folder = os.path.abspath(os.path.dirname(__file__))
cuda_sources = {f: os.path.join(module_folder, f) for f in filter(is_cuda_sourcefile, os.listdir(module_folder))}
