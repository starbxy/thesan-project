from matplotlib.ticker import NullFormatter
import numpy as np
import h5py
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

with h5py.File('Density.hdf5', 'r') as file_x, h5py.File('dz.hdf5', 'r') as file_y, h5py.File('RM2.hdf5', 'r') as file_z, h5py.File('dRMbydl.hdf5', 'r') as file_a:
    # Read the data and weighting from the HDF5 files
    data = file_a['dRMbydl'][:]
    density = file_x['Density'][:]
    weighting = file_y['dz'][:]
    weighting2 = weighting * density
    weighting3 = file_z['RM2'][:]

bins = np.logspace(np.log10(data.min()), np.log10(data.max()), 30)
# symlog 
# bin_width = 3.5 * np.std(data) / np.power(len(data), 1/3)
# bins = int((data.max() - data.min()) / bin_width)
H_all, bin_edges_all = np.histogram(data, bins=bins, weights=weighting)
H2_all, bin_edges2_all = np.histogram(data, bins=bins, weights=weighting2)
# H3_all, bin_edges3_all = np.histogram(data, bins=bins, weights=weighting3)
fig, ax = plt.subplots()
ax.plot(bin_edges_all[:-1], H_all, drawstyle='steps-mid', color='black', label='Volume weighting')
ax.plot(bin_edges2_all[:-1], H2_all, drawstyle='steps-mid', color='violet', label='Mass weighting')
# ax.plot(bin_edges3_all[:-1], H3_all, drawstyle='steps-mid', color='blue', label='RM weighting')
ax.set_xlabel('dRM/dl', fontsize=15)
ax.set_ylabel('Weighted count', fontsize =15)
ax.set_xscale('log')
ax.set_yscale('log')
plt.title("dRM/dl logspace histogram, showing mass and volume weightings", fontsize=18)
ax.legend()
plt.show()