#! /usr/bin/env python3
import numpy as np
from scipy.stats import binned_statistic_2d as bin2d
import h5py


def A0(masses):  # the function to calculate the A0 component
    return np.sum(masses)


def A2(masses, phis):  # the function to calculate the A2 component
    return np.sum(masses * np.exp(2 * phis * 1j))


def BarStrength(masses, phis):  # the function to calculate the bar strength
    return np.abs(A2(masses, phis) / A0(masses))

def BucklingStrength(masses, phis, zs):  # the function to calculate the buckling strength
    return np.abs(np.sum(masses * zs * np.exp(2 * phis * 1j))) / A0(masses)

def BarAngle(masses, phis):  # the function to calculate the bar angle
    return np.angle(A2(masses, phis)) / 2


def GetCoM(coordinates, mass):  # the function to calculate the center of masses
    index = np.where(np.linalg.norm(coordinates, axis=1) < 1000)[
        0
    ]  # the condition to select the particles
    coordinates = coordinates[index]
    cenOfMass = np.sum(coordinates * mass[:, None], axis=0) / np.sum(mass)
    older = cenOfMass
    for i in range(100):  # the iteration to find the center of masses
        coordinates = coordinates - cenOfMass
        cenOfMass = np.sum(coordinates * mass[:, None], axis=0) / np.sum(mass)
        if np.allclose(
            older, cenOfMass, atol=0.01
        ):  # the condition to stop the iteration: < 0.01kpc
            break
        older = cenOfMass

    return cenOfMass


def Car2cylin(
    coordinates, velocities, cenOfMass
):  # the function to convert the coordinates and velocities from cartesian to cylindrical
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


def SigmaStdProfile(
    R, phis, range=[0, 10], bins=25
):  # the function to calculate the standard deviation of the surface density profile
    sigma, rEdges, _, _ = bin2d(
        x=R,
        y=phis,
        values=None,
        statistic="count",
        bins=bins,
        range=[[range[0], range[1]], [-np.pi, np.pi]],
    )
    index = np.where(sigma == 0)
    sigma[index] = 1  # avoid the zero value for the log10 function
    sigma = np.log10(sigma) / np.max(
        np.log10(sigma)
    )  # normalize the surface density profile
    sigma = np.std(sigma, axis=1)  # calculate the standard deviation w.r.t. the radius
    return sigma, (rEdges[1:] + rEdges[:-1]) / 2
    # return the standard deviation of different radial bins and the radius of the bins


def CriticalRadius(
    sigma, radiuses, threshold=0.1
):  # the function to calculate the critical radius such that the standard deviation is the first minimum
    """
    sigma: 1D array, the standard deviation of the surface density profile.
    radiuses: 1D array, the radius of the bins' center.
    threshold: the fraction of decreasing, the first r<threshold is the critical radius.
    """
    maxLoc = np.where(sigma == np.max(sigma))[0][0]  # the location of the maximum
    max = np.max(sigma[maxLoc:])  # the maximum of the standard deviation
    min = np.min(sigma[maxLoc:])  # the minimum of the standard deviation
    sigma = sigma[maxLoc:]  # the standard deviation of the region of the maximum
    region = max - min  # the region of the standard deviation
    index = np.where(sigma < min + region * threshold)[
        0
    ]  # the condition to find the critical radius
    try:
        return radiuses[maxLoc + index[0]]  # return the critical radius if it exists
    except:
        return -1  # return -1 if the critical radius does not exist
