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
"""
from setuptools import setup, find_packages

# Default version information
source_path = 'src'
__version__ = '1.0'
install_requirements = ['']

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
        print("#"*30)
        print(ex)

    return __version__  # default

setup(name='snomtools',
      version=get_version_from_git(),
      packages=packages,
      package_dir={'': source_path},
      install_requirements=install_requirements
)
