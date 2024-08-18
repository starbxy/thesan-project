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

def read_rays(snap=80, rays_dir='.'):
    filename = f'/Users/kaan/Downloads/python/MIT/rays_{snap:03d}.hdf5'
    with h5py.File(filename, 'r') as f:
        h = f.attrs['HubbleParam'] # Hubble constant [100 km/s/Mpc]
        n_rays = f.attrs['NumRays'] # Number of rays
        n_pixels = int(np.sqrt(n_rays))
        Omega0 = f.attrs['Omega0'] # Cosmic matter density (~0.3)
        OmegaBaryon = f.attrs['OmegaBaryon'] # Cosmic baryon density
        z = f.attrs['Redshift'] # Cosmological redshift
        UnitLength_in_cm = f.attrs['UnitLength_in_cm'] # Code length units (no cosmology)
        UnitMass_in_g = f.attrs['UnitMass_in_g'] # Code mass units (no cosmology)
        UnitVelocity_in_cm_per_s = f.attrs['UnitVelocity_in_cm_per_s'] # Code velocity units (no cosmology)
        UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s # Code time units (no cosmology)
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

        RM_grid = np.zeros(n_rays)

        n_bins = 100
        logmin = -15
        logmax = 6
        min_clip = 1.000001*10.**logmin
        max_clip = 0.999999*10.**logmax
        edges = np.logspace(logmin, logmax, n_bins+1)
        hist_V = np.zeros(n_bins)
        hist_m = np.zeros(n_bins)
        hist_RM = np.zeros(n_bins)

        for i in range(n_rays):
            s = str(i)
            dz = length_to_cgs * f['RaySegments'][s][:].astype(np.float64) # Segment lengths [cm], delta(l)
            zr = np.cumsum(dz) # Right segment positions [cm] dz_0, dz_0+dz_1, +...
            zl = zr - dz # Left segment positions [cm]
            zc = (zl + zr)/2 # Midpoint of each segment (L, cm)
            x_HI = f['HI_Fraction'][s][:].astype(np.float64) # Neutral hydrogen fraction (n_HI / n_H)
            x_e = f['ElectronAbundance'][s][:].astype(np.float64) # Electron abundance (n_e / n_H)
            mu = 4. / (1. + 3.*X + 4.*X * x_e) # Mean molecular mass [mH] units of proton mass
            T = T_div_emu * f['InternalEnergy'][s][:].astype(np.float64) * mu # Gas temperature [K]
            v = velocity_to_cgs * f['Velocity'][s][:].astype(np.float64) # Line of sight velocity [cm/s]
            rho = density_to_cgs * f['Density'][s][:].astype(np.float64) # Density [g/cm^3]
            D = f['GFM_DustMetallicity'][s][:].astype(np.float64) # Dust-to-gas ratio
            Z = f['GFM_Metallicity'][s][:].astype(np.float64) # Metallicity [mass fraction]
            n_H = X_mH * rho * (1. - Z) # Hydrogen number density [cm^-3]
            n_phot = f['PhotonDensity'][s][:].astype(np.float64) # Radiation photon density [HI, HeI, HeII] [code units]
            SFR = f['StarFormationRate'][s][:].astype(np.float64) # Star formation rate [M_sun / Yr]
            B = magnetic_to_cgs * f['MagneticField'][s][:].astype(np.float64) # Magnetic field vector (x,y,z) [Gauss]
            B_mag = np.sqrt(np.sum(B**2, axis=1)) # Magnitude of B field [Gauss]
            B_los = B[:,2] # Line of sight (z), use this in RM calculations
            # RM = (0.812*1e12/pc)*np.sum(n_H * x_e * B_los * dz / ((1+z)**2))
            RM = (0.812*1e12/pc)*(n_H * x_e * B_los * dz / ((1+z)**2))
            dRMbydl = RM / dz
            n_e = n_H * x_e # Electron number density [cm^-3]
            # RM_grid[i] = RM
            
            B_mag[B_mag<min_clip] = min_clip
            B_mag[B_mag>max_clip] = max_clip

            hist_local_V, _ = np.histogram(B_mag, weights=dz, density=False, bins=edges)
            hist_local_m, _ = np.histogram(B_mag, weights=rho * dz, density=False, bins=edges)
            hist_local_RM, _ = np.histogram(B_mag, weights=RM, density=False, bins=edges)

            hist_V += hist_local_V
            hist_m += hist_local_m
            hist_RM += hist_local_RM

        np.savetxt('B_mag_histogram_volume.txt', hist_V)
        np.savetxt('B_mag_histogram_mass.txt', hist_m)
        np.savetxt('B_mag_histogram_RM.txt', hist_RM)
        RM_grid = RM_grid.reshape([n_pixels, n_pixels])
 
read_rays()