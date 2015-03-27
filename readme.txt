SNOMTOOLS Package

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
