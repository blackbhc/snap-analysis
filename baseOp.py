#! /usr/bin/env python3
import numpy as np

# from scipy.stats import binned_statistic as bin1d
# from scipy.stats import binned_statistic_2d as bin2d


def A0(masses):  # the function to calculate the A0 component
    return np.sum(masses)


def A2(masses, phis):  # the function to calculate the A2 component
    return np.sum(masses * np.exp(2 * phis * 1j))


def BarStrength(masses, phis):  # the function to calculate the bar strength
    return np.abs(A2(masses, phis) / A0(masses))


def BucklingStrength(
    masses, phis, zs
):  # the function to calculate the buckling strength
    return np.abs(np.sum(masses * zs * np.exp(2 * phis * 1j))) / A0(masses)


def GetA2phase(masses, phis):  # the function to calculate the bar angle
    return np.angle(A2(masses, phis)) / 2


def GetCoM(
    coordinates, mass, size, maxLoop=500
):  # the function to calculate the center of masses
    index = np.where(np.linalg.norm(coordinates, axis=1) < 10000)[
        0
    ]  # the condition to select the particles
    cenOfMass = np.sum(coordinates[index] * mass[index, None], axis=0) / np.sum(
        mass[index]
    )
    older = cenOfMass  # initialize the center of masses
    for i in range(maxLoop):  # the iteration to find the center of masses
        index = np.where(np.linalg.norm(coordinates - older, axis=1) < size)[0]
        # the condition to select the particles: inside the sphere with radius = size
        cenOfMass = np.sum(coordinates[index] * mass[index, None], axis=0) / np.sum(
            mass[index]
        )
        if np.allclose(
            older, cenOfMass, atol=0.01
        ):  # the condition to stop the iteration: < 0.01kpc
            break
        older = cenOfMass

        if i == maxLoop - 1:  # the condition to stop the iteration: > maxLoop times
            print(
                "The center of masses is not converged after {} times iterations.".format(
                    i
                )
            )

    return cenOfMass


def Car2Cylin6D(coordinates, velocities, cenOfMass):
    """
    Function to convert the coordinates and velocities from cartesian to cylindrical.
    -----------
    Parameters:
        coordinates: the coordinates of particles
        velocities: the velocities of particles
        cenOfMass: the center of masses of particles

    Return:
        (RphiZ, VrPhiZ)
        RphiZ: the coordinates in cylindrical
        VrPhiZ: the velocities in cylindrical
    """
    x = coordinates[:, 0] - cenOfMass[0]
    y = coordinates[:, 1] - cenOfMass[1]
    z = coordinates[:, 2] - cenOfMass[2]
    R = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    Vx = velocities[:, 0]
    Vy = velocities[:, 1]
    Vz = velocities[:, 2]
    Vr = (x * Vx + y * Vy) / R
    Vphi = (x * Vy - y * Vx) / R
    return np.column_stack((R, phi, z)), np.column_stack((Vr, Vphi, Vz))


def Car2Cylin3D(coordinates, cenOfMass):
    """
    Function to convert the coordinates from cartesian to cylindrical.
    -----------
    Parameters:
        coordinates: the coordinates of particles
        cenOfMass: the center of masses of particles

    Return:
        (R, phi, z): the coordinates in cylindrical
    """

    x = coordinates[:, 0] - cenOfMass[0]
    y = coordinates[:, 1] - cenOfMass[1]
    z = coordinates[:, 2] - cenOfMass[2]
    R = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return R, phi, z


def GetA2max(coordinates, masses, range=[0, 10], bins=50):
    """
    Function to calculate the maximum A2 absolute value in some radial bins.
    -------------------------
    Parameters:
        coordinates: the coordinates of particles
        masses: the masses of particles
        range: the range of radial bins
        bins: the number of radial bins

    Return:
        A2max: the maximum A2 absolute value in radial bins
    """
    Rs, phis, _ = Car2Cylin3D(
        coordinates, GetCoM(coordinates, masses, size=10)
    )  # convert the coordinates to cylindrical
    RbinEdges = np.linspace(range[0], range[1], bins + 1)  # the edges of radial bins
    A2s = np.zeros(bins)  # initialize the A2 values in radial bins
    for i in range(bins):
        index = np.where((Rs >= RbinEdges[i]) & (Rs < RbinEdges[i + 1]))[0]
        # the condition to select the particles: inside the radial bin
        A2s[i] = np.abs(
            A2(masses[index], phis[index])
        )  # calculate the A2 value in radial bin
    return np.max(A2s)


def GetA2phases(coordinates, masses, range=[0, 10], bins=50):
    """
    Function to calculate the phase angles of m=2 Fourier mode in some radial bins.
    -------------------------
    Parameters:
        coordinates: the coordinates of particles
        masses: the masses of particles
        range: the range of radial bins
        bins: the number of radial bins

    Return:
        A2phases: the phase angles of m=2 Fourier mode in radial bins
    """
    Rs, phis, _ = Car2Cylin3D(
        coordinates, GetCoM(coordinates, masses, size=10)
    )  # convert the coordinates to cylindrical
    RbinEdges = np.linspace(range[0], range[1], bins + 1)  # the edges of radial bins
    A2phases = np.zeros(bins)  # initialize the A2 values in radial bins
    for i in range(bins):
        index = np.where((Rs >= RbinEdges[i]) & (Rs < RbinEdges[i + 1]))[0]
        # the condition to select the particles: inside the radial bin
        A2phases[i] = GetA2phase(
            masses[index], phis[index]
        )  # calculate the A2 value in radial bin
    return A2phases