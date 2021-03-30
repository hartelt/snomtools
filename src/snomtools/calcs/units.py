"""
This file provides the central unit registry that should be used in all scripts that use snomtools.
This avoids errors between quantities of different unit registries that occur when using multiple imports.
Custom units and prefixes that we use frequently should be defined here to get consistency.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pint
import pint.quantity
import numpy

__author__ = 'hartelt'

# Initialize Standard UnitRegistry:
ureg = pint.UnitRegistry()
Quantity = ureg.Quantity

# Custom units that we use frequently can be defined here:
# ureg.define('dog_year = 52 * day = dy')
ureg.define('pixel = []')
ureg.define('count = []')

# Custom prefixes we use frequently can be defined here:
# ureg.define('myprefix- = 30 = my-')

# Custom contexts we use frequently can be defined here:
c = pint.Context('light')
c.add_transformation('[length]', '[time]', lambda ureg, x: x / ureg.speed_of_light)
c.add_transformation('[time]', '[length]', lambda ureg, x: x * ureg.speed_of_light)
ureg.add_context(c)


def same_dimension(*args):
    """
    Checks if all of the given arguments are of the same (physical) dimension.

    :param args:

    :return: Boolean.
    """
    uregargs = as_ureg_quantities(args)
    # with as_ureg_quantities(args) as a:
    arg0 = next(uregargs)
    for arg in uregargs:
        if arg0.dimensionality != arg.dimensionality:
            return False
    return True


def same_unit(*args):
    """
    Checks if all of the given arguments are of the same unit.

    :param args:

    :return: Boolean.
    """
    uregargs = as_ureg_quantities(args)
    # with as_ureg_quantities(args) as a:
    arg0 = next(uregargs)
    for arg in uregargs:
        if arg0.units != arg.units:
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


def normalize_unitstr(unitstr):
    """
    Returns the normalized unit string, meaning the string conversion of the casted unit.

    :param unitstr: string: to convert

    :return: string: converted
    """
    return str(ureg(str(unitstr)).units)


def is_quantity(tocheck):
    """
    Tries if the given object is a pint quantity.

    :param tocheck: The object to check.

    :return: Bool.
    """
    return isinstance(tocheck, pint.quantity._Quantity)


def to_ureg(input_, unit=None, convert_quantities=True):
    """
    This method is an import function to import alien quantities (of different unit registries) or numeric formats into
        the ureg.

    :param input_: The input quantity or numeric format (e.g. float, int, numpy array) or string (e.g. "5 nm")

    :param unit: Given as a valid unit string. If a numeric format is used, this specifies the unit of it. If a quantity
        is used, the output quantity will be converted to it.

    :param convert_quantities: Boolean: Specifies what should be done if input_ is a quantity.
        True: convert the quantity to the unit if specified by unit parameter. (The default)
        False: only assure the dimension is correct and import to ureg, but leave unit of input_ as is is.

    :return: The imported quantity.
    """
    # Python2/3 compatibility: If unit is a byte string, decode it for usage in pint.Quantity.
    if isinstance(unit, bytes):
        unit = unit.decode()

    # Check if input is quantity:
    if is_quantity(input_):
        # If output unit is specified, make sure it has a compatible dimension. Else a DimensionalityError will be
        # raised:
        if unit and not same_dimension(input_, to_ureg(1, unit)):
            raise pint.DimensionalityError(input_.units, to_ureg(1, unit).units,
                                           extra_msg="Incompatible units given to to_ureg!")
        # Check if input is already of our ureg. If so, nothing to do other then convert if unit is specified:
        if input_._REGISTRY is ureg:
            if unit and convert_quantities and input_.units != to_ureg(1, unit).units:
                return input_.to(unit)
            else:
                return input_
        else:  # Use inputs magnitude, but our corresponding ureg unit.
            if unit and convert_quantities:
                return Quantity(input_.magnitude, str(input_.units)).to(unit)
            else:
                return Quantity(input_.magnitude, str(input_.units))
    elif (isinstance(input_, str) or isinstance(input_, str)):
        if unit:
            return ureg(input_).to(unit)
        else:
            return ureg(input_)
    else:  # we are dealing with numerical data
        return Quantity(input_, unit)


def as_ureg_quantities(stream, unit=None):
    for e in stream:
        yield to_ureg(e, unit)


def magnitudes(stream):
    for e in stream:
        yield e.magnitude


def units(stream):
    for e in stream:
        yield e.units


def meshgrid(*args):
    """
    Does the same as numpy.meshgrid, but preserves units.

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


def fix_siunitx(unitstr):
    siunitx_mapping = {
        "degC": "degreeCelsius",
        "metric_ton": "tonne",
        "unified_atomic_mass_unit": "atomicmassunit",
        "speed_of_light": "clight",
        "dirac_constant": "planckbar",
        "millimeter_Hg": "mmHg",
        "\\milli\\meter_Hg": "\\mmHg"
    }
    for k in siunitx_mapping:
        unitstr = unitstr.replace(k, siunitx_mapping[k])
    unitstr = unitstr.replace("delta_", "Delta \\")
    unitstr = unitstr.replace("_", "")
    return unitstr


def latex_si(input_):
    """
    Returns a LaTeX \si command for a correct LaTeX representation of the unit.

    :param input_: Must be castable as a quantity.

    :return: str: The command string, e.g. '\\si[]{\\femto\\second}'.
    """
    input_ = to_ureg(input_)
    # noinspection PyStringFormat
    return fix_siunitx("{:Lx}".format(input_.units))


def latex_SI(input_):
    """
    Returns a LaTeX \SI command for a correct LaTeX representation of the quantity.

    :param input_: Must be castable as a quantity.

    :return: str: The command string, e.g. '\\SI[]{1}{\\femto\\second}'.
    """
    input_ = to_ureg(input_)
    # noinspection PyStringFormat
    return fix_siunitx("{:Lx}".format(input_))
