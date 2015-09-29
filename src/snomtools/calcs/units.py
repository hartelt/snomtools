__author__ = 'hartelt'
'''
This file provides the central unit registry that should be used in all scripts that use snomtools.
This avoids errors between quantities of different unit registries that occur when using multiple imports.
Custom units and prefixes that we use frequently should be defined here to get consistency.
'''

# Import pint and initialize a standard unit registry:
import pint
import pint.quantity

ureg = pint.UnitRegistry()

# Custom units that we use frequently can be defined here:
# ureg.define('dog_year = 52 * day = dy')

# Custom prefixes we use frequently can be defined here:
# ureg.define('myprefix- = 30 = my-')


def is_valid_unit(tocheck):
	"""
	Checks if a string can be interpreted as a unit.
	:param tocheck: String, The input to check.
	:return: Boolean if the input can be interpreted as a unit.
	"""
	try:
		ureg(tocheck)
		return True
	except pint.UndefinedUnitError:
		return False


def is_quantity(tocheck):
	"""
	Tries if the given object is a pint quantity.
	:param tocheck: The object to check.
	:return: Bool.
	"""
	return isinstance(tocheck, pint.quantity._Quantity)


def to_ureg(input_, unit=None):
	"""
	This method is an import function to import alien quantities (of different unit registries) or numeric formats into
	the ureg.
	:param input_: The input quantity or numeric format (e.g. float, int, numpy array)
	:param unit: Given as a valid unit string. If a numeric format is used, this specifies the unit of it. If a quantity
	is used, the output quantity will be converted to it.
	:return: The imported quantity.
	"""

	#Check if input is quantity:
	if is_quantity(input_):
		# If output unit is specified, make sure it has a compatible dimension. Else a DimensionalityError will be
		# raised by trying to convert:
		if unit:
			input_.to(unit)
		# Check if input is already of our ureg. If so, nothing to do other then convert if unit is specified:
		if input_._REGISTRY is ureg:
			if unit:
				return input_.to(unit)
			else:
				return input_
		else:  # Use inputs magnitude, but our corresponding ureg unit.
			importedquantity = input_.magnitude * ureg(str(input_.units))
			if unit:
				return importedquantity.to(unit)
			else:
				return importedquantity
	else:  # we are dealing with numerial data
		return input_ * ureg(unit)