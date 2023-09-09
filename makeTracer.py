import numpy as np


def potGridTracer(resolution, rMax=20.0):
    """
    Functions to create a tracer grid for the potential field.
    ------------
    Parameters:
    resolution: double, the resolution of the tracer grid, in unit of length (dimensionless in the function itself).
    rMax: double, the maximum radius of the tracer grid, in unit of length (dimensionless in the function itself).

    ------------
    Returns:
    tracerGrid: 2D numpy array, the coordinates of the tracer grid, in size N x 3, where N is the number of tracer points (depends on the resolution).
    """
    nBinR = int(rMax / resolution)
    radiuses = np.linspace(0.0, rMax, nBinR + 1)
    tracerGrid = [
        [0.0, 0.0, 0.0]
    ]  # 2D array, each row is a tracer point, in the form of [x, y, z]
    for i in range(nBinR):
        radius = radiuses[i + 1]
        perimeter = 2.0 * np.pi * radius
        nBinTheta = int(perimeter / resolution)
        thetas = np.linspace(0.0, 2.0 * np.pi, nBinTheta + 1)
        for j in range(len(thetas)):
            theta = thetas[j]
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            tracerGrid.append([x, y, 0.0])
    tracerGrid = np.array(tracerGrid)
    return tracerGrid
