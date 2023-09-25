import numpy as np


def barLength1(m2_angles, threshold=3, majorAxis=0):
    """
    Function to calculate the bar length of a galaxy: method 1 of Soumavo Ghosh & Di Matteo 2023,
    (m2_angles - majorAxis) >= threshold to locate the bar ends, see paper for details.
    ----------
    Parameters:
    m2_angles: array like, shape (N,), the m=2 Fourier phase angles of the galaxy.
    threshold: float, the threshold in degrees to be used to calculate the bar length.
    majorAxis: float, the major axis of the galaxy in radians.
    ----------
    Returns:
    barEndLoc: int, the index of the bar end in the m2_angles array.
    """
    m2_angles = np.array(m2_angles)  # ensure m2_angles is an array
    return np.where(np.abs(m2_angles - majorAxis) >= np.deg2rad(threshold))[0][0]


def barLength3(bar_strengths, threshold=70):
    """
    Function to calculate the bar length of a galaxy: method 3 of Soumavo Ghosh & Di Matteo 2023,
    see paper for details.
    ----------
    Parameters:
    sbars: array like, shape (N,), the m=2 Fourier amplitude of the galaxy.
    threshold: float, the threshold in percent to be used to calculate the bar length.
    ----------
    Returns:
    barEndLoc: int, the index of the bar end in the m2_angles array.
    """
    bar_strengths = np.array(bar_strengths)  # ensure bar_strengths is an array
    id_max = np.argmax(bar_strengths)
    critical_sbar = np.percentile(bar_strengths[id_max:], threshold)
    index = np.where(bar_strengths[id_max:] <= critical_sbar)[0]
    barEndLoc = index[0] + id_max
    return barEndLoc
