import h5py
import numpy as np
from scipy.stats import pearsonr

with h5py.File('Density.hdf5', 'r') as file1, h5py.File('Bmag.hdf5', 'r') as file2:
    data1 = file1['Density'][:]
    data2 = file2['Bmag'][:]

correlation_coefficient, p_value = pearsonr(data1.flatten(), data2.flatten())

print("Pearson correlation coefficient:", correlation_coefficient)
print("P-value:", p_value)