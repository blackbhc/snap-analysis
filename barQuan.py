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


timeDuration = 10  # the duration of the simulation
startTime = 0  # the start time of the simulation
allSteps = 100  # the maximum step of the simulation
startStep = 0  # the start step of the simulation
endStep = 100  # the end step of the simulation
times = np.arange(startStep, allSteps + 1, 1) / allSteps * timeDuration + startTime
# the time of the snapshots

files = ["snapshot_" + str(i).zfill(3) + ".hdf5" for i in range(startStep, endStep, 1)]
# the name of the snapshots
size = 10 # the size of the cylinder
binnum = 50 # the number of the radial bins

model = "Model4"
dir = "/home/bhchen/BarFormation/timeScale/Simulations/{}/output/".format(model) # the directory of the snapshots
logDir = "/home/bhchen/BarFormation/timeScale/Analysis/Quantities/{}/".format(model) # the directory of the log files
logFile = logDir + "log_{}.txt".format(model) # the log file
txtLog = open(logFile, "w") # open the log file
txtLog.write("#time\tbarStrength\tbarAngle\tbucklingStrength\tcriticalRadius\n") # write the header of the log file

for file in files:
    file = h5py.File(dir + file, "r")
    coordinates = np.array(file["PartType2"]["Coordinates"]) # the coordinates of the particles
    velocities = np.array(file["PartType2"]["Velocities"]) # the velocities of the particles
    mass = np.array(file["PartType2"]["Masses"]) # the mass of the particles
    # calculate the center of masses
    cenOfMass = GetCoM(coordinates, mass) # the center of masses
    coordinates = coordinates - cenOfMass # the coordinates of the particles w.r.t. the center of masses
    cylinder, _ = Car2cylin(coordinates, velocities, cenOfMass) # the coordinates of the particles in the cylindrical coordinate
    R = cylinder[:, 0] # the radius of the particles
    phi = cylinder[:, 1] # the angle of the particles
    z = cylinder[:, 2] # the z coordinate of the particles
    index = np.where(R < size)[0] # the particles in the cylinder
    # calculate the bar strength
    barStrength = BarStrength(mass[index], phi[index]) # the bar strength
    # calculate the buckling strength
    bucklingStrength = BucklingStrength(mass[index], phi[index], z[index]) # the buckling strength
    # calculate the bar angle
    barAngle = BarAngle(mass[index], phi[index]) # the bar angle
    # calculate the standard deviation of the surface density profile
    sigma, radiuses = SigmaStdProfile(R[index], phi[index], range=[0, size], bins=binnum)
    # calculate the critical radius
    criticalRadius = CriticalRadius(sigma, radiuses, threshold=0.3)
    # save the results
    txtLog.write( # write the results to the log file
        "{}\t{}\t{}\t{}\t{}\n".format(
            file["Header"].attrs["Time"],
            barStrength,
            barAngle,
            bucklingStrength,
            criticalRadius,
        )
    )
    txtLog.flush() # flush the buffer
    file.close() # close the snapshot file

txtLog.close() # close the log file
