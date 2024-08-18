import h5py
import matplotlib.pyplot as plt

# Open the HDF5 file for reading
filename = 'data.h5'
with h5py.File(filename, 'r') as f:
    redshift = f['redshift'][:]
    variable = f['variable'][:] # * 100 # convert from cm^-1 to m^-1 for RM

plt.figure(figsize=(8, 6))
plt.plot(redshift, variable, marker='x', color='black')
plt.xlabel('z', fontsize=20)
#plt.ylabel(r'$\ log((<B_{mag}>, \ [G])$', fontsize=20)
plt.ylabel(r'$\ n_{e} \ [cm^{-3}]$', fontsize=20)
plt.title(r'Volume-average electron density plotted as a function of redshift', fontsize=20, y=1.05)
#plt.yscale('log')
plt.xlim(5.5, 20)
plt.show()