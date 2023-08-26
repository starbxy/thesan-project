import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
epsilon0 = 8.854187e-12    # Vacuum permittivity [F m^-1]

def read_lightrays():
    filename = f'/nfs/mvogelsblab001/Thesan/Thesan-1/postprocessing/lightrays.hdf5'
    with h5py.File(filename, 'r') as f:
        h = f.attrs['HubbleParam'] # Hubble constant [100 km/s/Mpc]
        n_rays = f.attrs['NumRays'] # Number of rays
        n_pixels = int(np.sqrt(n_rays))
        Omega0 = f.attrs['Omega0'] # Cosmic matter density (~0.3)
        OmegaBaryon = f.attrs['OmegaBaryon'] # Cosmic baryon density
        UnitLength_in_cm = f.attrs['UnitLength_in_cm'] # Code length units (no cosmology)
        UnitMass_in_g = f.attrs['UnitMass_in_g'] # Code mass units (no cosmology)
        UnitVelocity_in_cm_per_s = f.attrs['UnitVelocity_in_cm_per_s'] # Code velocity units (no cosmology)
        UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s # Code time units (no cosmology)
        mass_to_cgs = UnitMass_in_g / h # Code mass to g
        X_mH = X / mH # X_mH = X / mH
        T_div_emu = GAMMA_MINUS1 * UnitVelocity_in_cm_per_s**2 * PROTONMASS / BOLTZMANN # T / (e * mu)

        dz_tot = 0
        dz1_tot = 0
        dz2_tot = 0

        bins = 100
        logmin = -32
        logmax = -23
        min_clip = 1.00000000001*10.**logmin
        max_clip = 0.99999999999*10.**logmax
        min_clip1 = 1.00000000001*10.**logmin
        max_clip1 = 0.99999999999*10.**logmax
        min_clip2 = 1.00000000001*10.**logmin
        max_clip2 = 0.99999999999*10.**logmax
        edges = np.logspace(logmin, logmax, bins+1)
        H_final = np.zeros(bins)
        H1_final = np.zeros(bins)
        H2_final = np.zeros(bins)

        RM_sums = np.zeros(n_rays) # Ray RM sums
        for i in range(n_rays):
            s = str(i)
            z_edges = f['Redshifts'][s][:] # Redshift edges
            z = 0.5 * (z_edges[:-1] + z_edges[1:]) # Cosmological redshift
            a = 1. / (1. + z) # Cosmological scale factor
            length_to_cgs = a * UnitLength_in_cm / h # Code length to cm - proper units conv
            volume_to_cgs = length_to_cgs**3 # Code volume to cm^3
            density_to_cgs = mass_to_cgs / volume_to_cgs # Code density to g/cm^3
            velocity_to_cgs = np.sqrt(a) * UnitVelocity_in_cm_per_s # Code velocity to cm/s
            magnetic_to_cgs = h/a**2 * np.sqrt(UnitMass_in_g/UnitLength_in_cm) / UnitTime_in_s # Code magnetic field to Gauss
            Hz = 100. * h * np.sqrt(1. - Omega0 + Omega0/a**3) # Hubble parameter [km/s/Mpc]
            Hz_cgs = Hz * km / Mpc # Hubble parameter [1/s]
            dz = length_to_cgs * f['RaySegments'][s][:].astype(np.float64) # Segment lengths [cm], delta(l)
            zr = np.cumsum(dz) # Right segment positions [cm] dz_0, dz_0+dz_1, +...
            zl = zr - dz # Left segment positions [cm]
            zc = (zl + zr)/2 # Midpoint of each segment (L, cm)
            # x_HI = f['HI_Fraction'][s][:].astype(np.float64) # Neutral hydrogen fraction (n_HI / n_H)
            x_e = f['ElectronAbundance'][s][:].astype(np.float64) # Electron abundance (n_e / n_H)
            mu = 4. / (1. + 3.*X + 4.*X * x_e) # Mean molecular mass [mH] units of proton mass
            T = T_div_emu * f['InternalEnergy'][s][:].astype(np.float64) * mu # Gas temperature [K]
            # v = velocity_to_cgs * f['Velocity'][s][:].astype(np.float64) # Line of sight velocity [cm/s]
            rho = density_to_cgs * f['Density'][s][:].astype(np.float64) # Density [g/cm^3]
            # D = f['GFM_DustMetallicity'][s][:].astype(np.float64) # Dust-to-gas ratio
            Z = f['GFM_Metallicity'][s][:].astype(np.float64) # Metallicity [mass fraction]
            n_H = X_mH * rho * (1. - Z) # Hydrogen number density [cm^-3]
            n_e = x_e * n_H # Electron number density [cm^-3]
            # n_phot = f['PhotonDensity'][s][:].astype(np.float64) # Radiation photon density [HI, HeI, HeII] [code units]
            # SFR = f['StarFormationRate'][s][:].astype(np.float64) # Star formation rate [M_sun / Yr]
            B = f['MagneticField'][s][:].astype(np.float64) # Magnetic field vector (x,y,z) [code units]
            for j in range(3):
                B[:,j] *= magnetic_to_cgs # Convert magnetic field vector (x,y,z) to Gauss
            B_mag = np.sqrt(np.sum(B**2, axis=1)) # Magnitude of B field [Gauss]
            B_los = B[:,2] # Line of sight (z), use this in RM calculations
            dRM_dl = (0.812*1e12/pc) * n_e * B_los / (1.+z)**2 # Rotation measure integrand [rad/m^2/cm]
            RM = dRM_dl * dz # Rotation measure [rad/m^2]
            RM_sum = np.sum(RM) # Sum of RM along line of sight [rad/m^2]
            RM_sums[i] = RM_sum # Save RM sum
            # print(f'RM = {RM_sum:.3f} rad/m^2')

            x_HI_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
            z_vals = np.array([6.08892, 6.38506, 6.67, 6.96592, 7.3157, 7.73876, 8.28728, 9.05884, 10.3148])

            z_50 = 7.3157
            mask = z>z_50

            dRM_dl[dRM_dl<min_clip] = min_clip
            dRM_dl[dRM_dl>max_clip] = max_clip
            dRM_dl[mask][dRM_dl[mask]<min_clip] = min_clip1
            dRM_dl[mask][dRM_dl[mask]>max_clip] = max_clip1
            dRM_dl[~mask][dRM_dl[~mask]<min_clip] = min_clip2
            dRM_dl[~mask][dRM_dl[~mask]>max_clip] = max_clip2

            dz_tot += np.sum(dz)
            dz1_tot += np.sum(dz[mask])
            dz2_tot += np.sum(dz[~mask])

            # creating 1-dimensional histograms
            x = np.abs(dRM_dl) # variable
            H_local, _ = np.histogram(x, weights=dz, density=False, bins=edges)
            H_final += H_local
            x1 = np.abs(dRM_dl) # variable
            H1_local, _ = np.histogram(x1[mask], weights=dz[mask], density=False, bins=edges)
            H1_final += H1_local
            x2 = np.abs(dRM_dl) # variable
            H2_local, _ = np.histogram(x2[~mask], weights=dz[~mask], density=False, bins=edges)
            H2_final += H2_local
            print(n_rays)
            
    RM_avg = np.mean(RM_sums) # Average RM [rad/m^2]
    RM_std = np.std(RM_sums) # Standard deviation of RM [rad/m^2]
    RM_min = np.min(RM_sums) # Minimum RM [rad/m^2]
    RM_max = np.max(RM_sums) # Maximum RM [rad/m^2]
    RM_med = np.median(RM_sums) # Median RM [rad/m^2]
    print(f'RM_avg = {RM_avg:.3f} rad/m^2')
    print(f'RM_std = {RM_std:.3f} rad/m^2')
    print(f'RM_min = {RM_min:.3f} rad/m^2')
    print(f'RM_max = {RM_max:.3f} rad/m^2')
    print(f'RM_med = {RM_med:.3f} rad/m^2')

    fig, ax = plt.subplots()
    plt.xscale('log')
    ax.set_xlabel(r"$\ log_{10}(d(RM)/dl)$", fontsize=15)
    H0 = H_final/dz_tot
    H1 = H1_final/dz1_tot
    H2 = H2_final/dz2_tot
    ax.set_ylabel('Normalized by segment length (dz)', fontsize=15)
    ax.plot(edges[:-1], H0, color='black', linestyle='-', linewidth=2, alpha=1)
    ax.plot(edges[:-1], H1, color='green', linestyle='-', linewidth=1, alpha=0.7, label='x_HI > 0.5 (z>7.3)')
    ax.plot(edges[:-1], H2, color='red', linestyle='-', linewidth=1, alpha=0.7, label='x_HI < 0.5 (z<7.3)')
    plt.legend()
    plt.title("Rotation measure histogram [lightrays] weighted by segement length", fontsize=9, y=1.05)
    plt.savefig('1.pdf', dpi=500)

read_lightrays()