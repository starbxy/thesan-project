from matplotlib.ticker import NullFormatter
import numpy as np
import h5py
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

# Arepo constants
BOLTZMANN = 1.38065e-16     # Boltzmann's constant [g cm^2/sec^2/k]
PROTONMASS = 1.67262178e-24 # Mass of hydrogen atom [g]
GAMMA = 5. / 3.             # Adiabatic index of simulated gas
GAMMA_MINUS1 = GAMMA - 1.   # For convenience

# General constants
Msun = 1.988435e33         # Solar mass [g]
c = 2.99792458e10          # Speed of light [cm/s]
km = 1e5                   # Units: 1 km  = 1e5  cm
pc = 3.085677581467192e18  # Units: 1 pc  = 3e18 cm
kpc = 1e3 * pc             # Units: 1 kpc = 3e21 cm
Mpc = 1e6 * pc             # Units: 1 Mpc = 3e24 cm
kB = 1.380648813e-16       # Boltzmann's constant [g cm^2/s^2/K]
mH = 1.6735327e-24         # Mass of hydrogen atom (g)
me = 9.109382917e-28       # Electron mass [g]
ee = 4.80320451e-10        # Electron charge [g^(1/2) cm^(3/2) / s]
X  = 0.76                  # Primordial hydrogen mass fraction
f12 = 0.4162               # Oscillator strength
nu0 = 2.466e15             # Lya frequency [Hz]
epsilon0 = 8.854187e-12    # Vacuum permittivity [F m^-1]
lambda0 = 1e8 * c / nu0    # Lya wavelength [Angstroms]
DnuL = 9.936e7             # Natural line width [Hz]
kappa_dust = 7.177e4       # Lya dust opacity [cm^2/g dust]

def read_3d_2(snap=80, out_dir='.'):
    file_dir = f'/nfs/mvogelsblab001/Thesan/HighRes/L4_N256/output/snapdir_{snap:03d}'
    # /nfs/mvogelsblab001/Thesan/Thesan-1/output/snapdir_080
    filename = f'{file_dir}/snap_{snap:03d}.0.hdf5'
    with h5py.File(filename, 'r') as f:
        g = f['Parameters']
        Omega0 = g.attrs['Omega0'] # Cosmic matter density (~0.3)
        OmegaBaryon = g.attrs['OmegaBaryon'] # Cosmic baryon density
        UnitLength_in_cm = g.attrs['UnitLength_in_cm'] # Code length units (no cosmology)
        UnitMass_in_g = g.attrs['UnitMass_in_g'] # Code mass units (no cosmology)
        UnitVelocity_in_cm_per_s = g.attrs['UnitVelocity_in_cm_per_s'] # Code velocity units (no cosmology)
        UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s # Code time units (no cosmology)
        n_files = g.attrs['NumFilesPerSnapshot'] # Number of files per snapshot
        h = g.attrs['HubbleParam'] # Hubble constant [100 km/s/Mpc]
        BoxSize = g.attrs['BoxSize'] # ckpc [Comoving]

        g = f['Header']
        z = g.attrs['Redshift'] # Cosmological redshift
        a = 1. / (1. + z) # Cosmological scale factor
        length_to_cgs = a * UnitLength_in_cm / h # Code length to cm - proper units conv
        cm_to_cMpc = (1. + z) / Mpc # cm to cMpc
        volume_to_cgs = length_to_cgs**3 # Code volume to cm^3
        mass_to_cgs = UnitMass_in_g / h # Code mass to g
        density_to_cgs = mass_to_cgs / volume_to_cgs # Code density to g/cm^3
        X_mH = X / mH # X_mH = X / mH
        velocity_to_cgs = np.sqrt(a) * UnitVelocity_in_cm_per_s # Code velocity to cm/s
        magnetic_to_cgs = h/a**2 * np.sqrt(UnitMass_in_g/UnitLength_in_cm) / UnitTime_in_s # Code magnetic field to Gauss
        T_div_emu = GAMMA_MINUS1 * UnitVelocity_in_cm_per_s**2 * PROTONMASS / BOLTZMANN # T / (e * mu)
        Hz = 100. * h * np.sqrt(1. - Omega0 + Omega0/a**3) # Hubble parameter [km/s/Mpc]
        Hz_cgs = Hz * km / Mpc # Hubble parameter [1/s]

    n_bins = 100

    hist_1 = np.zeros((n_bins, n_bins))
    hist_2 = np.zeros((n_bins, n_bins))

    log_ne_min = -9
    log_ne_max = 6
    log_B_min = -14
    log_B_max = -6
    log_T_min = -1
    log_T_max = 6

    ne_min_clip = 1.000001*10.**log_ne_min
    ne_max_clip = 0.999999*10.**log_ne_max
    B_min_clip = 1.000001*10.**log_B_min
    B_max_clip = 0.999999*10.**log_B_max
    T_min_clip = 1.000001*10.**log_T_min
    T_max_clip = 0.999999*10.**log_T_max

    log_edges_ne = np.logspace(log_ne_min, log_ne_max, n_bins+1)
    log_edges_B = np.logspace(log_B_min, log_B_max, n_bins+1)
    log_edges_T = np.logspace(log_T_min, log_T_max, n_bins+1)

    for i in range(n_files):
        filename = f'{file_dir}/snap_{snap:03d}.{i}.hdf5'
        with h5py.File(filename, 'r') as f:
            p = f['PartType0']
            
            x_HI = p['HI_Fraction'][:].astype(np.float64) # Neutral hydrogen fraction (n_HI / n_H)
            x_HII = 1 - x_HI
            x_e = p['ElectronAbundance'][:].astype(np.float64) # Electron abundance (n_e / n_H)
            mu = 4. / (1. + 3.*X + 4.*X * x_e) # Mean molecular mass [mH] units of proton mass
            T = T_div_emu * p['InternalEnergy'][:].astype(np.float64) * mu # Gas temperature [K]
            v = velocity_to_cgs * p['Velocities'][:].astype(np.float64) # Line of sight velocity [cm/s]
            rho = density_to_cgs * p['Density'][:].astype(np.float64) # Density [g/cm^3]
            m = mass_to_cgs * p['Masses'][:].astype(np.float64) # Mass of gas [g]
            V = m / rho # Volume [cm^3]
            D = p['GFM_DustMetallicity'][:].astype(np.float64) # Dust-to-gas ratio
            Z = p['GFM_Metallicity'][:].astype(np.float64) # Metallicity [mass fraction]
            n_H = X_mH * rho * (1. - Z) # Hydrogen number density [cm^-3]s
            n_phot = p['PhotonDensity'][:].astype(np.float64) # Radiation photon density [HI, HeI, HeII] [code units]
            SFR = p['StarFormationRate'][:].astype(np.float64) # Star formation rate [M_sun / Yr]
            B = magnetic_to_cgs * p['MagneticField'][:].astype(np.float64) # Magnetic field vector (x,y,z) [Gauss]
            B_mag = np.sqrt(np.sum(B**2, axis=1)) # Magnitude of B field [Gauss]
            n_e = n_H * x_e # Electron number density [cm^-3]

            n_e[n_e<ne_min_clip] = ne_min_clip
            n_e[n_e>ne_max_clip] = ne_max_clip
            B_mag[B_mag<B_min_clip] = B_min_clip
            B_mag[B_mag>B_max_clip] = B_max_clip
            T[T<T_min_clip] = T_min_clip
            T[T>T_max_clip] = T_max_clip

            # Creating the 2d histograms
            counts, xedges, yedges = np.histogram2d(n_e, B_mag, weights=m, normed=False, bins=[log_edges_ne, log_edges_B]) # normed <-> density
            counts2, xedges2, yedges2 = np.histogram2d(n_e, T, weights=m, normed=False, bins=[log_edges_ne, log_edges_T])

            hist_1 += counts
            hist_2 += counts2

        filename = f'Hist_2d_{snap:03d}.hdf5'
        with h5py.File(filename, 'w') as f:
            f.create_dataset('hist_ne_B', data=hist_1)
            f.create_dataset('hist_ne_T', data=hist_2)
            f.create_dataset('edges_ne', data=xedges)
            f.create_dataset('edges_B', data=yedges)
            f.create_dataset('edges_T', data=yedges2)

for snap in [80, 70, 54, 43, 34, 27]: # change number depending on whatever redshift you want to plot
    read_3d_2(snap=snap)