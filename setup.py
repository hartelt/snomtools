#!/usr/bin/env python
"""
This is an example for the possible setup of a library
located by default within the src/ folder.
All packages will be installed to python.site-packages
simply run:

    >>> python setup.py install

For a local installation or if you like to develop further

    >>> python setup.py develop --user


The test_suite located within the test/ folder
will be executed automatically.

Regarding tifffile on Windows x64-systems:
In case of problems while installing the package for x64 Python, use Anaconda Distribution
"""
from setuptools import setup, find_packages
from pip.req import parse_requirements

# Default version information
source_path = 'src'
__version__ = '1.0'
# install_requirements = ['numpy>=1.10.0', 'pint', 'h5py', 'scipy',
# 						'tifffile', 'h5py_cache', 'six', 'psutil', 'opencv-python']

# Parse requirements from requirements.txt
install_reqs = parse_requirements('requirements.txt')
reqs = [str(ir.req) for ir in install_reqs]

packages = find_packages(source_path)

# Define your setup
# version should be considered using git's short or better the full hash
def get_version_from_git():
	"""
    Get the short version string of a git repository
    :return: (str) version information
    """
	import subprocess
	try:
		v = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
		return __version__ + v
	except Exception as ex:

		print("Could not retrieve git version information")
		print("#" * 30)
		print(ex)

	return __version__  # default


setup(name='snomtools',
	  version=get_version_from_git(),
	  packages=packages,
	  package_dir={'': source_path},
	  install_requires=reqs
	  )
