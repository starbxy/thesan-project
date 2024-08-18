import h5py
import numpy as np
import matplotlib.pyplot as plt

# List of HDF5 files representing different redshifts
hdf5_files = ['Temperature_hist_027.hdf5', 'Temperature_hist_034.hdf5', 'Temperature_hist_043.hdf5', 'Temperature_hist_054.hdf5', 'Temperature_hist_070.hdf5', 'Temperature_hist_080.hdf5']

# Redshift values corresponding to the HDF5 files
redshifts = [10, 9, 8, 7, 6, 5.5]

weighting_type_names = {
    'hist_V': 'volume',
    'hist_m': 'mass',
    'hist_RM': 'rotation measure'
}

save_pdf = input("Do you want to save PDF files of the plots? (y/n): ")

# Iterate over each weighting type
for weighting_type in ['hist_V', 'hist_m', 'hist_RM']:
    # Arrays to store histogram data for each redshift
    histograms = []

    # Iterate over each HDF5 file
    for hdf5_file, redshift in zip(hdf5_files, redshifts):
        with h5py.File(hdf5_file, 'r') as file:
            data = file[weighting_type][:]

        # Normalize the histogram
        hist_normalized = np.cumsum(data) / np.sum(data)

        # Append the histogram to the list with its corresponding label
        histograms.append((hist_normalized, redshift))


    # Reading hdf5 file to obtain data for edges
    for hdf5_file, redshift in zip(hdf5_files, redshifts):
        with h5py.File(hdf5_file, 'r') as file:
            n_bins = file['n_bins'][()]
            logmin = file['logmin'][()]
            logmax = file['logmax'][()]

    # Plotting
    edges = np.logspace(logmin, logmax, n_bins+1)

    plt.figure(figsize=(10, 6))
    colors = ['b', 'g', 'r', 'c', 'm', 'y']

    for hist_normalized, redshift in histograms:
        plt.plot(edges[:-1], hist_normalized, color=colors[redshifts.index(redshift)], label=f'z={redshift}')

    plt.xlabel('Temperature (log), [K]', fontsize=15)
    plt.ylabel('CDF weighting', fontsize=15)
    plt.title(f'Temperature histograms weighted by {weighting_type_names[weighting_type]}', fontsize=18)
    plt.xscale('log')
    # plt.yscale('log')
    plt.legend()
    plt.ylim(0, 1)
    plt.xlim(1e-1, 1e6)
    plt.grid(True)

    if save_pdf.lower() == 'y':
        filename = f'Temperature_{weighting_type}.pdf'
        plt.savefig(filename, format='pdf')

    plt.show()