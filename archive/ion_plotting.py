import h5py
import numpy as np
import matplotlib.pyplot as plt

variable = 'Bmag'

# List of HDF5 files representing different redshifts
hdf5_files = [f'{variable}_hist_027_ion.hdf5', f'{variable}_hist_034_ion.hdf5', f'{variable}_hist_043_ion.hdf5', f'{variable}_hist_054_ion.hdf5', f'{variable}_hist_070_ion.hdf5', f'{variable}_hist_080_ion.hdf5']
redshifts = [10, 9, 8, 7, 6, 5.5]

weighting_type_names = {
    'hist_xHI': 'Non-ionised',
    'hist_xHII': 'Ionised',
}

for hdf5_file, redshift in zip(hdf5_files, redshifts):
    with h5py.File(hdf5_file, 'r') as file:
        n_bins = file['n_bins'][()]
        logmin = file['logmin'][()]
        logmax = file['logmax'][()]

edges = np.logspace(logmin, logmax, n_bins+1)

# Arrays to store histogram data for each redshift and weighting type
histograms_xHI = []
histograms_xHII = []

# Iterate over each HDF5 file
for hdf5_file, redshift in zip(hdf5_files, redshifts):
    with h5py.File(hdf5_file, 'r') as file:
        data_xHI = file['hist_xHI'][:]
        data_xHII = file['hist_xHII'][:]

    # Normalize the histograms
    hist_normalized_xHI = np.cumsum(data_xHI) / np.sum(data_xHI)
    hist_normalized_xHII = np.cumsum(data_xHII) / np.sum(data_xHII)

    # Append the histograms to the respective lists with their corresponding labels
    histograms_xHI.append((hist_normalized_xHI, redshift))
    histograms_xHII.append((hist_normalized_xHII, redshift))

fig, axs = plt.subplots(1, 2, figsize=(16, 4))
colors = ['b', 'g', 'r', 'c', 'm', 'y']

plt.subplot(1, 2, 1)
for hist_normalized, redshift in histograms_xHI:
    plt.plot(edges[1:], hist_normalized, color=colors[redshifts.index(redshift)], label=f'z={redshift}')  
plt.title('Non-Ionised')
plt.xscale('log')
plt.xlim(1e-14, 1e-10)
plt.legend()

plt.subplot(1, 2, 2)
for hist_normalized, redshift in histograms_xHII:
    plt.plot(edges[1:], hist_normalized, color=colors[redshifts.index(redshift)], label=f'z={redshift}')  
plt.title('Ionised')
plt.xscale('log')
plt.xlim(1e-14, 1e-10)
plt.legend()

fig.text(0.5, 0.04, r"$\log_{10}(B_{mag}),\ [G])$", ha='center', fontsize=25)  # Universal x-label
fig.text(0.04, 0.5, 'CDF weighting', va='center', rotation='vertical', fontsize=25)  # Universal y-label
plt.show()