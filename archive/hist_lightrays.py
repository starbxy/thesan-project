import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl

# Read data from the HDF5 file
with h5py.File('histogram_data.h5', 'r') as hf:
    histograms = hf['histograms'][:]
    z_vals = hf['z_vals'][:]
    x_HI_vals = hf['x_HI_vals'][:]
    bin_edges = hf['bin_edges'][:]

# Create a colormap for plotting
n_colors = len(z_vals) - 1
colors = pl.cm.jet(np.linspace(0, 1, n_colors))

# Plot the histograms with corresponding z and x_HI ranges
plt.figure(figsize=(10, 6))

#for i in range(n_colors):
    #plt.plot(bin_edges[:-1], histograms[i], color=colors[i], label=f'z={z_vals[i]:.2f}-{z_vals[i+1]:.2f}, x_HI={x_HI_vals[i]:.1f}-{x_HI_vals[i+1]:.1f}')

hist_all = np.sum(histograms, axis=0)
plt.plot(bin_edges[:-1], hist_all)

plt.title('B-field magnitude histogram summed for all redshifts across all light-rays, weighted by RM', fontsize=18, y=1.05)
plt.xlabel(r'$\log_{10}(B_{mag}), [G]$', fontsize=18)
plt.xscale('log')
plt.xlim(1e-13, 1e-8)
#plt.legend()
plt.show()