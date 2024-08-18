from matplotlib.ticker import NullFormatter
import numpy as np
import h5py
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import os
from pylab import *
from scipy.special import erf

# List of HDF5 files representing different redshifts
hdf5_files = ['Hist_2d_027.hdf5', 'Hist_2d_043.hdf5', 'Hist_2d_080.hdf5']

# Redshift values corresponding to the HDF5 files
redshifts = [10, 8, 5.5]

fig, axes_non_ionized = plt.subplots(1, 3, figsize=(12, 8))
fig.subplots_adjust(hspace=0.3, wspace=0.3)

fig2, axes_ionized = plt.subplots(1, 3, figsize=(12, 8))
fig2.subplots_adjust(hspace=0.3, wspace=0.3)

for i, hdf5_file in enumerate(hdf5_files):
    filename = '/Users/kaan/Downloads/python/MIT/' + hdf5_file

    with h5py.File(filename, 'r') as f:
        # print(f.keys())
        counts = f['hist_ne_B_1'][:]
        counts2 = f['hist_ne_B_2'][:]
        xedges, yedges = f['edges_ne'][:], f['edges_B'][:]
        yedges *= 1e6  # convert to uGauss
        x_centers = np.sqrt(xedges[1:] * xedges[:-1])
        y_centers = np.sqrt(yedges[1:] * yedges[:-1])

    def weighted_percentile(Z, W, q):
        isort = np.argsort(Z)
        Z_sorted = Z[isort]
        W_sorted = W[isort]
        IW_sorted = np.cumsum(W_sorted)
        IW_sorted /= IW_sorted[-1]
        wp = np.zeros_like(q)
        for i_q in range(len(q)):
            q_frac = q[i_q] / 100.
            i = np.searchsorted(IW_sorted, q_frac)
            assert IW_sorted[i] > q_frac
            wp[i_q] = (Z_sorted[i]-Z_sorted[i-1]) * (q_frac-IW_sorted[i-1]) / (IW_sorted[i]-IW_sorted[i-1]) + Z_sorted[i-1]
        return wp
    
    sigma_68 = erf(1. / np.sqrt(2.))
    percentiles = [50., 50. * (1. - sigma_68), 50. * (1. + sigma_68)]
    n_percentiles = len(percentiles)

    B_flat = np.sum(counts, axis=1)
    mask = B_flat > 0
    log_x_centers = np.log10(x_centers)[mask]
    inds = np.array(range(len(y_centers)))[mask]
    B_p = np.array([weighted_percentile(y_centers, counts[i, :], percentiles) for i in inds]).T
    log_B_p = np.log10(B_p)
    i_med = np.argmin(np.abs(log_x_centers + 3.5))  # where the data is lots (3.5)
    b = log_B_p[0][i_med] - 2. / 3. * log_x_centers[i_med]
    log_B_a = b + 2 / 3 * log_x_centers

    B_flat2 = np.sum(counts2, axis=1)
    mask2 = B_flat2 > 0
    log_x_centers2 = np.log10(x_centers)[mask2]
    inds2 = np.array(range(len(y_centers)))[mask2]
    B_p2 = np.array([weighted_percentile(y_centers, counts2[i, :], percentiles) for i in inds2]).T
    log_B_p2 = np.log10(B_p2)
    i_med2 = np.argmin(np.abs(log_x_centers2 + 3.5))  # where the data is lots (3.5)
    b2 = log_B_p2[0][i_med2] - 2. / 3. * log_x_centers2[i_med2]
    log_B_a2 = b2 + 2 / 3 * log_x_centers2

    extent1 = [np.log10(xedges[0]), np.log10(xedges[-1]), np.log10(yedges[0]), np.log10(yedges[-1])]

    ax_non_ionized = axes_non_ionized[i]
    ax_non_ionized.set_xlabel(r"$\log_{10}(n_{e}),\ [cm^{-3}]$", fontsize=20)
    ax_non_ionized.set_ylabel(r"$\log_{10}(B_{mag}),\ [\mu G])$", fontsize=20)
    ax_non_ionized.set_title("z = {:.1f}".format(redshifts[i]), fontsize=12)

    ax_ionized = axes_ionized[i]
    ax_ionized.set_xlabel(r"$\log_{10}(n_{e}),\ [cm^{-3}]$", fontsize=20)
    ax_ionized.set_ylabel(r"$\log_{10}(B_{mag}),\ [\mu G])$", fontsize=20)
    ax_ionized.set_title("z = {:.1f}".format(redshifts[i]), fontsize=12)

    fig.text(0.02, 0.5, 'non-ionized', fontsize=25, va='center', rotation='vertical', fontstyle='italic')
    fig2.text(0.02, 0.5, 'ionized', fontsize=25, va='center', rotation='vertical', fontstyle='italic')

    # Plot non-ionized (counts) on the top row
    image_non_ionized = ax_non_ionized.imshow(counts.T, interpolation='nearest', origin='lower', aspect='auto', cmap='BuPu', extent=extent1, norm=matplotlib.colors.LogNorm())
    ax_non_ionized.plot(log_x_centers, log_B_a, lw=2, c='red', label=r'$\propto n_e^{2/3}$')
    ax_non_ionized.plot(log_x_centers, log_B_p[0], lw=2., c='grey', label=r'${\rm Median}$')
    ax_non_ionized.plot(log_x_centers, log_B_p[1], lw=1., ls='--', c='grey', label=r'$\pm 1\sigma$')
    ax_non_ionized.plot(log_x_centers, log_B_p[2], lw=1., ls='--', c='grey')
    ax_non_ionized.legend(loc='upper left', frameon=False, borderaxespad=1, handlelength=2., ncol=1, fontsize=10)
    ax_non_ionized.set_xlim([-7.5, -1])
    ax_non_ionized.set_ylim([-7.5, -1])

    # Plot ionized weighting (counts2) on the bottom row
    image_ionized = ax_ionized.imshow(counts2.T, interpolation='nearest', origin='lower', aspect='auto', cmap='BuPu', extent=extent1, norm=matplotlib.colors.LogNorm())
    ax_ionized.plot(log_x_centers2, log_B_a2, lw=2, c='red', label=r'$\propto n_e^{2/3}$')
    ax_ionized.plot(log_x_centers2, log_B_p2[0], lw=2., c='grey', label=r'${\rm Median}$')
    ax_ionized.plot(log_x_centers2, log_B_p2[1], lw=1., ls='--', c='grey', label=r'$\pm 1\sigma$')
    ax_ionized.plot(log_x_centers2, log_B_p2[2], lw=1., ls='--', c='grey')
    ax_ionized.legend(loc='upper left', frameon=False, borderaxespad=1, handlelength=2., ncol=1, fontsize=10)
    ax_ionized.set_xlim([-7.5, -1])
    ax_ionized.set_ylim([-7.5, -1])

    colorbar_non_ionized = plt.colorbar(image_non_ionized, ax=ax_non_ionized)
    colorbar_ionized = plt.colorbar(image_ionized, ax=ax_ionized)

plt.show()