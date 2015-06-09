__author__ = 'hartelt'
'''
This file provides the central unit registry that should be used in all scripts that use snomtools.
This avoids errors between quantities of different unit registries that occur when using multiple imports.
Custom units and prefixes that we use frequently should be defined here to get consistency.
'''

#Import pint and initialize a standard unit registry:
import pint
ureg = pint.UnitRegistry()

# Custom units that we use frequently can be defined here:
#ureg.define('dog_year = 52 * day = dy')

# Custom prefixes we use frequently can be defined here:
#ureg.define('myprefix- = 30 = my-')

def to_ureg(input_,unit=None):
	'''
	This method is an import function to import alien quantities (of different unit registries) or numeric formats into the ureg.
	:param input_: The input quantity or numeric format (e.g. float, int, numpy array)
	:param unit: If a numeric format is used, this specifies the unit of it. Given as a valid unit string.
	:return: The imported quantity.
	'''

	#Check if input is quantity:
	if hasattr(input_,'_REGISTRY'):
		#Check if input is already of our ureg. If so, nothing to do:
		if input_._REGISTRY is ureg:
			return input_
		else: #Use inputs magnitude, but our corresponding ureg unit.
			return input_.magnitude * ureg(str(input_.units))
	else: #we are dealing with numerial data
		return input_ * ureg(unit)