__author__ = 'hartelt'
"""
This file provides the central unit registry that should be used in all scripts that use snomtools.
This avoids errors between quantities of different unit registries that occur when using multiple imports.
Custom units and prefixes that we use frequently should be defined here to get consistency.
"""

# Import pint and initialize a standard unit registry:
import pint
import pint.quantity
import numpy

ureg = pint.UnitRegistry()
Quantity = ureg.Quantity

# Custom units that we use frequently can be defined here:
# ureg.define('dog_year = 52 * day = dy')
ureg.define('pixel = []')
ureg.define('count = []')


# Custom prefixes we use frequently can be defined here:
# ureg.define('myprefix- = 30 = my-')

def same_dimension(*args):
	"""
	Checks if all of the given arguments are of the same (physical) dimension.

	:param args:

	:return: Boolean.
	"""
	uregargs = as_ureg_quantities(args)
	#with as_ureg_quantities(args) as a:
	arg0 = next(uregargs)
	for arg in uregargs:
		if arg0.dimensionality != arg.dimensionality:
			return False
	return True


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


def unit_from_str(tocheck):
	"""
	Checks if a string can be interpreted as a unit and returns the corresponding unit.

	:param tocheck: String, The input to check.

	:return: Quantity with magnitude 1 and the corresponding unit, or None if not castable.
	"""
	try:
		unit = ureg(tocheck)
		return unit
	except pint.UndefinedUnitError:
		return None


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

	:param input_: The input quantity or numeric format (e.g. float, int, numpy array) or string (e.g. "5 nm")

	:param unit: Given as a valid unit string. If a numeric format is used, this specifies the unit of it. If a quantity
	is used, the output quantity will be converted to it.

	:return: The imported quantity.
	"""

	# Check if input is quantity:
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
			if unit:
				return Quantity(input_.magnitude, str(input_.units)).to(unit)
			else:
				return Quantity(input_.magnitude, str(input_.units))
	elif (isinstance(input_, str) or isinstance(input_, unicode)):
		if unit:
			return ureg(input_).to(unit)
		else:
			return ureg(input_)
	else:  # we are dealing with numerial data
		return Quantity(input_,unit)


def as_ureg_quantities(stream):
	for e in stream:
		yield to_ureg(e)


def meshgrid(*args):
	"""
	Does the same as numpy.meshgrid, but preserves units.

	:param *xi:

	:return:
	"""
	unitbuffer = []
	for x in args:
		unitbuffer.append(str(x.units))
	gridtup = numpy.meshgrid(*args)
	outlist = []
	for i in range(len(gridtup)):
		outlist.append(to_ureg(gridtup[i], unitbuffer[i]))
	return tuple(outlist)
