SNOMTOOLS Package

Copyright (C) 2017 M. Hartelt and others, AG Aeschlimann, FB Physik, TU Kaiserslautern

In case of problems, bugs or questions, contact
    Michael Hartelt
    hartelt@physik.uni-kl.de

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

For optimal performance, use Python 3, 64 bit.
The package was originally written in Python 2 and was adapted to Python 3.
Python 2 is supported through the use of the future module, but full support is not prioritized in the further development, especially since the End-of-Life of Python 2 in 2020.
32 bit versions of Python work, but especially for large data and on Windows, out-of-memory problems may occur.

To install, run:
    >>> pip install -r requirements.txt
    >>> python setup.py develop --user
If python 2 is the standard python version of your system, and you want to install under python 3:
    >>> pip3 install -r requirements.txt
    >>> python3 setup.py develop --user

To build documentation with sphinx, you have to install sphinx on your computer:
e.g. 
Debian based Linux:
    >>> sudo apt-get install python-sphinx
Redhat based Linux(as root):
    >>> yum install install python-sphinx

Than you can execute the makefile in ./doc, for example in html output format:
    >>> cd doc
    >>> make html
The result will be in ./doc/build/
