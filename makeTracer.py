import numpy as np


def potGridTracer(anglebinNum=64, rbinNum=50, rMax=20.0, rbinForm="log"):
    """
    Functions to create a tracer grid for the potential field.
    ------------
    Parameters:
    anglebinNum: int, the number of bins in the polar angle direction.
    rbinNum: int, the number of bins in the radial direction.
    rMax: double, the maximum radius of the tracer grid, in unit of length (dimensionless in the function itself).
    rbinForm: string, the form of the radial bins, can be "log" or "linear".

    ------------
    Returns:
    tracerGrid: 2D numpy array, the coordinates of the tracer grid, in size N x 3, where N is the number of tracer points (depends on the resolution).
    """
    rbinForm = rbinForm.lower()  # make sure the input is lower case
    if rbinForm == "linear":
        radii = np.linspace(0.0, rMax, rbinNum + 1)[1:]
    elif rbinForm == "log":
        radii = np.linspace(
            0.0, np.log(rMax + 1), rbinNum + 1
        )  # linearly spaced in log space
        radii = np.exp(radii)[1:] - 1.0  # in linear space
    else:
        raise ValueError(
            "rbinForm must be either 'linear' or 'log'!"
        )  # raise error if the input is not valid
    thetas = np.linspace(0.0, 2.0 * np.pi, anglebinNum + 1)[:-1]
    tracerGrid = [
        [0.0, 0.0, 0.0]  # the center of the tracer grid
    ]  # 2D array, each row is a tracer point, in the form of [x, y, z]
    for theta in thetas:
        for radius in radii:
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            tracerGrid.append([x, y, 0.0])
    tracerGrid = np.array(tracerGrid)
    return tracerGrid
