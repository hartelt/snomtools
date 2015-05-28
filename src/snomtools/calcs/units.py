__author__ = 'hartelt'
'''
This file provides the central unit registry that should be used in all scripts that use snomtools.
This avoids errors between quantities of different unit registries that occur when using multiple imports.

'''

#Import pint and initialize a standard unit registry:
import pint
ureg = pint.UnitRegistry()

# Custom units that we use frequently can be defined here:
#ureg.define('dog_year = 52 * day = dy')

# Custom prefixes we use frequently can be defined here:
#ureg.define('myprefix- = 30 = my-')