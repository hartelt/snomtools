#! /bin/python
# coding: utf-8
"""
This Package contains various tools written in Python and oriented to analysis and presentation of data.

"""
import os.path

__author__ = 'Michael Hartelt, Cristian Gonzalez'

module_source = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def get_version_from_file():
	"""
	Get the named version from the version.txt file in the package root.

	:return: version information
	:rtype: str
	"""
	version_file = os.path.join(module_source, "version.txt")
	with open(version_file) as f:
		read_data = f.read()
	return str(read_data)


__version__ = get_version_from_file()


def get_subversion_from_git(version_base=None):
	"""
    Get the short version string of a git repository

    :return: version information
    :rtype: str
    """
	import subprocess
	try:
		v = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=module_source)
		v = v.decode().strip()
		if version_base is None:
			return str(v)
		else:
			return version_base + '.' + str(v)
	except Exception as ex:

		print("Could not retrieve git version information")
		print("#" * 30)
		print(ex)

	return version_base  # default


__version__ = get_subversion_from_git(__version__)