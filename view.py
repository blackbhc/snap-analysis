import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import binned_statistic_2d as bin2d
import h5py

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 12,
        "mathtext.fontset": "cm",
    }
)

size = 10
binnum = 100
figsize = 8
ratio = 0.4
barWidthRatio = 0.05
margin = 0.15 * figsize
W = figsize + 2 * margin
H = figsize * (1 + ratio) + 2 * margin
maxStep = 100
timeDuration = 10 
startStep = 0 
endStep = 100
startTime = 0
dir = "/home/bhchen/BarFormation/MajorMerger/Simulations/main/output/"
files = ["snapshot_" + f"{i}".zfill(3) + ".hdf5" for i in range(startStep, endStep + 1)]
times = np.arange(startStep, maxStep + 1, 1) / maxStep * timeDuration + startTime


def logNorm(mat):
    index = np.where(mat <= 0)
    mat[index] = 1
    mat = np.log10(mat)
    return mat / np.max(mat)


for i, file in enumerate(files):
    data = h5py.File(dir + file, "r")
    coordinates = data["PartType2"]["Coordinates"][...]
    index = np.where(np.linalg.norm(coordinates, axis=1) < size * 1000)[0]
    CoM = np.mean(coordinates, axis=0)
    coordinates = coordinates - CoM
    plt.figure(i)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(W, H))
    axes[0].set_position(
        [margin / W, (margin + ratio * figsize) / H, figsize / W, figsize / H]
    )
    axes[1].set_position([margin / W, margin / H, figsize / W, ratio * figsize / H])
    cax = fig.add_axes(
        [
            (margin + figsize) / W,
            margin / H,
            barWidthRatio * figsize / W,
            figsize * (1 + ratio) / H,
        ]
    )
    mat_xy, _, _, _ = bin2d(
        x=coordinates[:, 1],
        y=coordinates[:, 0],
        values=None,
        statistic="count",
        bins=binnum,
        range=[[-size, size], [-size, size]],
    )
    mat_xz, _, _, _ = bin2d(
        x=coordinates[:, 2],
        y=coordinates[:, 0],
        values=None,
        statistic="count",
        bins=[binnum * ratio, binnum],
        range=[[-size * ratio, size * ratio], [-size, size]],
    )
    im = axes[0].imshow(logNorm(mat_xy), cmap="jet", origin="lower")
    axes[1].imshow(logNorm(mat_xz), cmap="jet", origin="lower")
    phy2pixel1 = lambda x: (x + size) / (2 * size) * (binnum - 1)
    phy2pixel2 = (
        lambda x: (x + size * ratio) / (2 * size * ratio) * (binnum * ratio - 1)
    )
    ticks1 = np.linspace(-size, size, 5)
    ticks2 = np.linspace(-size * ratio, size * ratio, 5)
    axes[0].set_xticks([])
    axes[0].set_yticks(phy2pixel1(ticks1))
    axes[1].set_xticks(phy2pixel1(ticks1))
    axes[1].set_yticks(phy2pixel2(ticks2))

    axes[0].set_yticklabels(ticks1)
    axes[1].set_xticklabels(ticks1)
    axes[1].set_yticklabels(ticks2)

    axes[0].set_ylabel(r"$Y$ [kpc]")
    axes[1].set_xlabel(r"$X$ [kpc]")
    axes[1].set_ylabel(r"$Z$ [kpc]")
    axes[0].text(
        phy2pixel1(-.75 * size),
        phy2pixel1(.75 * size),
        r"$T=$" + f"{times[i]:.1f}" + " Gyr",
        fontsize=36,
        color="red",
    )
    plt.colorbar(im, cax=cax)
    plt.savefig("./Figs/" + file[:-5] + ".png")
    plt.close(i)
