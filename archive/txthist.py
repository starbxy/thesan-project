from matplotlib.ticker import NullFormatter
import numpy as np
import h5py
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

hist1 = np.loadtxt('B_mag_histogram_mass.txt')
hist2 = np.loadtxt('B_mag_histogram_volume.txt')
hist3 = np.loadtxt('B_mag_histogram_RM.txt')

hist1_normalized = np.cumsum(hist1) / np.sum(hist1)
hist2_normalized = np.cumsum(hist2) / np.sum(hist2)
hist3_normalized = np.cumsum(hist3) / np.sum(hist3)

n_bins=100
loglim1 = -15
loglim2 = 6
edges = np.logspace(loglim1, loglim2, n_bins+1)

plt.plot(edges[:-1], hist1_normalized, label='Mass weighting')
plt.plot(edges[:-1], hist2_normalized, label='Volume weighting')
plt.plot(edges[:-1], hist3_normalized, label='RM weighting')

plt.xlabel('B-field magnitude (log), [Gauss]', fontsize=15)
plt.ylabel('CDF weighting', fontsize=15)
plt.title('B-field magnitude [Gauss] histogram weighted by mass, volume, and rotation measure', fontsize=18)
plt.xscale('log')
# plt.yscale('log')
plt.legend()
plt.ylim(0, 1)
plt.xlim(1e-15, 1e-5)
plt.grid(True)
plt.show()