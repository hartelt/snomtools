#!/usr/bin/env python
"""
This is a setup script, adapted from an example for the possible setup of a 
library located by default within the src/ folder.
All packages will be installed to python.site-packages
simply run:

    >>> python setup.py install

NOTE: Due to a bug in the version detection system, this is not supported yet.
Use local installation as described below.

For a local installation or if you like to develop further

    >>> python setup.py develop --user

Make sure to install all requirements first, typically with pip.

Regarding tifffile on Windows x64-systems:
In case of problems while installing the package for x64 Python, 
use Anaconda Distribution.

As soon as we implemented proper unit testing, the test_suite located within 
the test/ folder will be executed automatically.
"""
from setuptools import setup, find_packages


# from pip.req import parse_requirements #obsolete since pip version x

def parse_requirements(filename):
	""" load requirements from a pip requirements file """
	lineiter = (line.strip() for line in open(filename))
	return [line for line in lineiter if line and not line.startswith("#")]


# Default version information
source_path = 'src'
__version__ = '1.0'

# Parse requirements from requirements.txt
install_reqs = parse_requirements('requirements.txt')
reqs = install_reqs

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
		v = v.decode().strip()
		return __version__ + '.' + v
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
