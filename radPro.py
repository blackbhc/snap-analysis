# Author: Bin-Hui, Chen
# Purpose: functions to calculate the radial profiles of a galaxy
# Return values convention: all functions return the radial bin centers and the corresponding values
import numpy as np


# calculate the 1st order derivative of an array in evenly spaced bins
def deriv1_u(array, xmin, xmax):
    """
    Function to calculate the first order derivative of an array: The array is assumed to be evenly spaced.
    ----------------
    Parameters:
    array: 1D numpy.array like, the array of which the 1st order derivative to be calculated.

    xmin: float, the minimum value of the argument of the function.

    xmax: float, the maximum value of the argument of the function.

    ----------------
    Returns:
    binCenters: 1D numpy.array like, the bin centers of the array.

    der1: 1D numpy.array like, the 1st order derivative of the given array.

    ----------------
    Notes: the 1st order derivative is calculated by the formula: (f(x+dx/2) - f(x-dx/2)) / (dx).
    """
    array = np.array(array)
    num = len(array) - 1  # number of bins
    xbin = (xmax - xmin) / num  # bin width
    return (
        np.linspace(xmin + xbin / 2, xmax - xbin / 2, num),
        (array[1:] - array[:-1]) / xbin,
    )


# calculate the 2nd order derivative of an array in evenly spaced bins
def deriv2_u(array, xmin, xmax):
    """
    Function to calculate the second order derivative of an array: The array is assumed to be evenly spaced.
    ----------------
    Parameters:
    array: 1D numpy.array like, the array of which the 2nd order derivative to be calculated.

    xmin: float, the minimum value of the argument of the function.

    xmax: float, the maximum value of the argument of the function.

    ----------------
    Returns:
    binCenters: 1D numpy.array like, the bin centers of the array.

    der2: 1D numpy.array like, the 2nd order derivative of the given array.

    ----------------
    Notes: the 2nd order derivative is calculated by the formula: (f(x+dx) + f(x-dx) - 2f(x)) / dx^2
    """
    array = np.array(array)
    num = len(array) - 1  # number of bins
    xbin = (xmax - xmin) / num  # bin width
    return (
        np.linspace(xmin, xmax, num + 1)[1:-1],
        (array[2:] + array[:-2] - 2 * array[1:-1]) / xbin**2,
    )


# calculate the 1st order derivative of an array in random spacing
def deriv1_r(x, array):
    """
    Function to calculate the first order derivative of an data points' array at the given argument values.
    ----------------
    Parameters:
    x: 1D numpy.array like, the argument values at whether the array is evaluated.

    array: 1D numpy.array like, the array of which the 1st order derivative to be calculated.

    ----------------
    Returns:
    binCenters: 1D numpy.array like, the bin centers of calculated derivative values.

    der1: 1D numpy.array like, the 1st order derivative of the given array.

    ----------------
    Notes: the 1st order derivative is calculated by the formula: (f(x_j) - f(x_j-1)) / (x_j - x_j-1).
    """
    array = np.array(array)[np.argsort(x)]
    x = np.array(x)[np.argsort(x)]
    binCenters = (x[1:] + x[:-1]) / 2
    return binCenters, (array[1:] - array[:-1]) / (x[1:] - x[:-1])


# calculate the 2nd order derivative of an array in random spacing
def deriv2_r(x, array):
    """
    Function to calculate the second order derivative of an data points' array at the given argument values.
    ----------------
    Parameters:
    x: 1D numpy.array like, the argument values at whether the array is evaluated.

    array: 1D numpy.array like, the array of which the 2nd order derivative to be calculated.

    ----------------
    Returns:
    binCenters: 1D numpy.array like, the bin centers of calculated derivative values.

    der2: 1D numpy.array like, the 2nd order derivative of the given array.

    ----------------
    Notes: the 2nd order derivative is calculated by the formula:
    ( (f_xi+1 - f_xi) / (xi+1 - xi) - (f_xi - f_xi-1) / (xi - xi-1) ) / ((xi+1 - xi-1) / 2).
    """
    array = np.array(array)[np.argsort(x)]
    x = np.array(x)[np.argsort(x)]
    bin1, der1 = deriv1_r(x, array)
    return deriv1_r(bin1, der1)


# function to calculate the rotation curve of a galaxy from the potential at some azimuthal direction
def rotCurve(radii, potentials):
    """
    Function to calculate the rotation curve of a galaxy from the potential at some azimuthal direction
    ----------------
    Parameters:
    radii: 1D numpy.array like, the radii of the potential values.

    potentials: 1D numpy.array like, the potential values at the given radii.

    ----------------
    Returns:
    binCenters: 1D numpy.array like, the bin centers of the rotation curve.

    rotCurve: 1D numpy.array like, the rotation velocities at binCenters.
    """
    radii, potentials = np.array(radii), np.array(potentials)
    binCenters, der1 = deriv1_r(
        radii, potentials
    )  # calculate the 1st order derivative of the potential values
    return binCenters, np.sqrt(binCenters * der1)  # calculate the rotation curve


# function to calculate the Kappa (radial epicycle frequency) profile of a galaxy from the potential at some azimuthal direction
def kappa(radii, potentials):
    """
    Function to calculate the Kappa (radial epicycle frequency) profile of a galaxy from the potential at some azimuthal direction
    ----------------
    Parameters:
    radii: 1D numpy.array like, the radii of the potential values.

    potentials: 1D numpy.array like, the potential values at the given radii.

    ----------------
    Returns:
    binCenters: 1D numpy.array like, the bin centers of the Kappa profile.

    Kappa: 1D numpy.array like, the Kappa values at binCenters.
    """
    radii, potentials = np.array(radii), np.array(potentials)
    rs1, der1 = deriv1_r(
        radii, potentials
    )  # calculate the 1st order derivative of the potential values
    Omega = np.sqrt(der1 / rs1)  # calculate the angular velocity
    rs2, dOmega_dR = deriv1_r(
        rs1, Omega
    )  # calculate the 1st order derivative of the angular velocity
    B = -(Omega[1:] + Omega[:-1]) / 2 - rs2 * dOmega_dR / 2
    # calculate the B Oort constant
    midOmega = np.array((Omega[1:] + Omega[:-1])) / 2
    # calculate the mid angular velocity
    Kappa2 = -4 * B * midOmega  # calculate the Kappa profile
    return rs2, np.sqrt(Kappa2)


# function to calculate the Toomre Q profile of a galaxy from the potential profile at some azimuthal direction
def toomreQ(radii, potentials, disR, surDen, G=43007.1):
    """
    Function to calculate the Toomre Q profile of a galaxy from the potential at some azimuthal direction
    ----------------
    Parameters:
    radii: 1D numpy.array like, the radii of the potential values.

    potentials: 1D numpy.array like, the potential values at the given radii.

    disR: 1D numpy.array like, the radial velocity dispersion profile of the galaxy. len= len(radii) - 2.

    surDen: 1D numpy.array like, the surface density profile of the galaxy. len= len(radii) - 2.

    G: float, the gravitational constant.

    (Note: the radii, potentials should have more data points at the beginning and the end than disR and
    surDen, to make sure the calculation is accurate.)

    ----------------
    Returns:
    binCenters: 1D numpy.array like, the bin centers of the Toomre Q profile. len = len(radii)-2.

    Q: 1D numpy.array like, the Toomre Q values at binCenters.
    """
    radii, potentials, disR, surDen = (
        np.array(radii),
        np.array(potentials),
        np.array(disR),
        np.array(surDen),
    )
    surDen = polishNaN(surDen)  # remove the nan values in the surface density profile
    rs, kappas = kappa(radii, potentials)  # calculate the Kappa profile
    return rs, disR * kappas / (3.36 * G * surDen)  # calculate the Toomre Q profile


def polishNaN(data):
    # This function is used to polish the array/matrix to set its nan values to 0
    data = np.array(data)
    if len(data.shape) == 1:
        index = np.where(np.isnan(data))[0]
    else:
        index = np.where(np.isnan(data))
    data[index] = 0
    return data
