SNOMTOOLS Package

Copyright (C) 2017

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


To install, run:
>>> python setup.py develop --user

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
