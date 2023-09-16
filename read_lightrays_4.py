import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import erf

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
G = 6.6725985e-8 # Gravitational constant [cm^3/g/s^2]

def read_lightrays():
    filename = f'/nfs/mvogelsblab001/Thesan/Thesan-1/postprocessing/lightrays.hdf5'
    with h5py.File(filename, 'r') as f:
        h = f.attrs['HubbleParam'] # Hubble constant [100 km/s/Mpc]
        n_rays = f.attrs['NumRays'] # Number of rays
        n_pixels = int(np.sqrt(n_rays))
        Omega0 = f.attrs['Omega0'] # Cosmic matter density (~0.3)
        OmegaB = f.attrs['OmegaBaryon']
        H0 = h * 100. * km / Mpc
        OmegaBaryon = f.attrs['OmegaBaryon'] # Cosmic baryon density
        UnitLength_in_cm = f.attrs['UnitLength_in_cm'] # Code length units (no cosmology)
        UnitMass_in_g = f.attrs['UnitMass_in_g'] # Code mass units (no cosmology)
        UnitVelocity_in_cm_per_s = f.attrs['UnitVelocity_in_cm_per_s'] # Code velocity units (no cosmology)
        UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s # Code time units (no cosmology)
        mass_to_cgs = UnitMass_in_g / h # Code mass to g
        X_mH = X / mH # X_mH = X / mH
        T_div_emu = GAMMA_MINUS1 * UnitVelocity_in_cm_per_s**2 * PROTONMASS / BOLTZMANN # T / (e * mu)

        z_arrays = []
        cumulative_arrays = []
        cumulative_array_3 = []
        cumulative_array_2 = []
        cumulative_array_1 = []

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
            SFR = f['StarFormationRate'][s][:].astype(np.float64) # Star formation rate [M_sun / Yr]
            B = f['MagneticField'][s][:].astype(np.float64) # Magnetic field vector (x,y,z) [code units]
            for j in range(3):
                B[:,j] *= magnetic_to_cgs # Convert magnetic field vector (x,y,z) to Gauss
            B_mag = np.sqrt(np.sum(B**2, axis=1)) # Magnitude of B field [Gauss]
            B_los = B[:,2] # Line of sight (z), use this in RM calculations
            dRM_dl = (0.812*1e12/pc) * n_e * B_los / (1.+z)**2 # Rotation measure integrand [rad/m^2/cm]
            RM = dRM_dl * dz # Rotation measure [rad/m^2]
            rho_crit_0 = 3. * H0**2 / (8. * np.pi * G) # Present critical density = 3H0^2/8piG [g/cm^3]
            rho_z = OmegaB * rho_crit_0 * (1. + z)**3 # Physical density [g/cm^3]
            RM[SFR>0]=0 # we ignore cells from the equation of state (EoS)
            cumulative_arrays.append(np.cumsum(RM))
            RM[rho/rho_z > 1e3] = 0. # Ignore gas that is overdense by a factor of 10^3
            cumulative_array_3.append(np.cumsum(RM))
            RM[rho/rho_z > 1e2] = 0. # Ignore gas that is overdense by a factor of 10^2
            cumulative_array_2.append(np.cumsum(RM))
            RM[rho/rho_z > 1e1] = 0. # Ignore gas that is overdense by a factor of 10
            cumulative_array_1.append(np.cumsum(RM))
            z_arrays.append(z)

    sigma_68 = erf(1. / np.sqrt(2.))
    percentiles = [50., 50. * (1. - sigma_68), 50. * (1. + sigma_68)]
    n_percentiles = len(percentiles)

    # Calcualting median and +-1 sigma
    n_meds = 120
    z_med = np.linspace(5.5, 20, n_meds)
    RM_p = np.zeros([3, n_meds])
    RM_p_1 = np.zeros([3, n_meds])
    RM_p_2 = np.zeros([3, n_meds])
    RM_p_3 = np.zeros([3, n_meds])
    for i_med in range(n_meds):
        i_mins = np.array([np.argmin(np.abs(z_med[i_med]-z_arrays[i_ray])) for i_ray in range(n_rays)]) 
        RM_vals = np.array([cumulative_arrays[i_ray][i_mins[i_ray]] for i_ray in range(n_rays)])
        RM_p[:, i_med] = np.percentile(RM_vals, percentiles)
    for i_med in range(n_meds):
        i_mins = np.array([np.argmin(np.abs(z_med[i_med]-z_arrays[i_ray])) for i_ray in range(n_rays)]) 
        RM_vals = np.array([cumulative_array_3[i_ray][i_mins[i_ray]] for i_ray in range(n_rays)])
        RM_p_3[:, i_med] = np.percentile(RM_vals, percentiles)
    for i_med in range(n_meds):
        i_mins = np.array([np.argmin(np.abs(z_med[i_med]-z_arrays[i_ray])) for i_ray in range(n_rays)]) 
        RM_vals = np.array([cumulative_array_2[i_ray][i_mins[i_ray]] for i_ray in range(n_rays)])
        RM_p_2[:, i_med] = np.percentile(RM_vals, percentiles)
    for i_med in range(n_meds):
        i_mins = np.array([np.argmin(np.abs(z_med[i_med]-z_arrays[i_ray])) for i_ray in range(n_rays)]) 
        RM_vals = np.array([cumulative_array_1[i_ray][i_mins[i_ray]] for i_ray in range(n_rays)])
        RM_p_1[:, i_med] = np.percentile(RM_vals, percentiles)

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.plot(z_med, RM_p[0], lw=1.75, c='C0', label='Median')
    plt.fill_between(z_med, RM_p[1], RM_p[2], lw=0., color='C0', alpha=.25, label=r'$\pm 1\sigma$')  
    plt.title('All gas')
    plt.xlim(5.5, 15)
    plt.legend()

    plt.subplot(1, 4, 2)
    plt.plot(z_med, RM_p_3[0], lw=1.75, c='C0', label='Median')
    plt.fill_between(z_med, RM_p_3[1], RM_p_3[2], lw=0., color='C0', alpha=.25, label=r'$\pm 1\sigma$')  
    plt.title('Overdense by 10^3')
    plt.xlim(5.5, 15)

    plt.subplot(1, 4, 3)
    plt.plot(z_med, RM_p_2[0], lw=1.75, c='C0', label='Median')
    plt.fill_between(z_med, RM_p_2[1], RM_p_2[2], lw=0., color='C0', alpha=.25, label=r'$\pm 1\sigma$')  
    plt.title('Overdense by 10^2')
    plt.xlim(5.5, 15)

    plt.subplot(1, 4, 4)
    plt.plot(z_med, RM_p_1[0], lw=1.75, c='C0', label='Median')
    plt.fill_between(z_med, RM_p_1[1], RM_p_1[2], lw=0., color='C0', alpha=.25, label=r'$\pm 1\sigma$')  
    plt.title('Overdense by 10')
    plt.xlim(5.5, 15)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    plt.subplots_adjust(left=0.1)
    fig.text(0.5, 0.04, 'z', ha='center', fontsize=25)  # Universal x-label
    fig.text(0.04, 0.5, r"$\ RM_{cum},\ [rad\ m^{-2}]$", va='center', rotation='vertical', fontsize=25)  # Universal y-label
    plt.savefig('1.pdf', dpi=1000)

read_lightrays()