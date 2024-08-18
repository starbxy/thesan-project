import h5py
import numpy as np
import matplotlib.pyplot as plt

# List of snapshots corresponding to redshifts
snapshots = [80, 70, 54, 43, 34, 27]
redshifts = [5.5, 6, 7, 8, 9, 10]

# Create a figure and axis for the plot
plt.figure(figsize=(10, 6))
plt.title(r'$\ dRM/dl\, [rad \ m^{-3}]$ histograms weighted by length', fontsize=18, y=1.05)
plt.xlabel(r'$\ dRM/dl\, [rad \ m^{-3}]$', fontsize=18)
plt.ylabel('CDF', fontsize=15)
plt.xscale('log')
plt.xlim(1e-32, 1e-21)

# Create a colormap for plotting
n_colors = len(snapshots)
colors = plt.cm.jet(np.linspace(0, 1, n_colors))

# Loop over snapshots and redshifts
for snap, redshift, color in zip(snapshots, redshifts, colors):
    filename = f'hist_{snap:03d}.hdf5'
    with h5py.File(filename, 'r') as f:
        hist_total = f['hist_total'][:] * 100 # convert from cm^-1 to m^-1
        n_bins = f['n_bins'][()]
        log_min = f['log_min'][()]
        log_max = f['log_max'][()]

    log_edges = np.logspace(log_min, log_max, n_bins + 1)
    hist_normalized = np.cumsum(hist_total) / np.sum(hist_total)

    plt.plot(log_edges[:-1], hist_normalized, color=color, label=f'z={redshift}')

plt.legend()
plt.show()
