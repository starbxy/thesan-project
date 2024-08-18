# 2-dimensional histograms
x = n_H  # variable 1
y = T   # variable 2
weights = dz  # weighting, np.ones_like(T)
xbins = np.logspace(np.log10(x.min()), np.log10(x.max()), 50)    # number of bins for variable 1. Remove np.log10s to get out of logspace
ybins = np.logspace(np.log10(y.min()), np.log10(y.max()), 50)   # number of bins for variable 2
H, xedges, yedges = np.histogram2d(x, y, bins=[xbins, ybins], weights=weights)
fig, ax = plt.subplots()
ax.set_xlabel('x')
ax.set_ylabel('y')
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
pcm = ax.pcolormesh(xedges, yedges, H.T, cmap='viridis', shading='auto')    
cbar = fig.colorbar(pcm)
cbar.set_label('Mass')
plt.xscale('log')
plt.yscale('log')
plt.show()

# 1-dimensional histograms
x = rho  # variable
weights = dz  # volume weighting
bins = np.logspace(np.log10(x.min()), np.log10(x.max()), 50)
H, bin_edges = np.histogram(x, bins=bins, weights=weights)

fig, ax = plt.subplots()
ax.set_xlabel('Density [g/cm^3]', fontsize=15)
ax.set_ylabel('Weighted count', fontsize =15)
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(bin_edges[:-1], H, drawstyle='steps-mid', color='black', label='Volume weighting')

weights2 = dz * rho # mass weighting
H2, bin_edges2 = np.histogram(x, bins=bins, weights=weights2)
ax.plot(bin_edges[:-1], H2, drawstyle='steps-mid', color='violet', label='Mass weighting')

plt.title("Density logspace histogram, showing both mass and volume weightings", fontsize=18)
ax.legend()
plt.show()

# 1-dimensional histograms [with all rays]
# outside for loop
x_all = []
weights_all = []
weights2_all = []

# inside for loop
x_all.append(rho)
weights_all.append(dz)
weights2_all.append(dz*rho)

# outside for loop
x_all = np.concatenate(x_all)
weights_all=np.concatenate(weights_all)
weights2_all=np.concatenate(weights2_all)

bins = np.logspace(np.log10(x_all.min()), np.log10(x_all.max()), 30)

H_all, bin_edges_all = np.histogram(x_all, bins=bins, weights=weights_all)
H2_all, bin_edges2_all = np.histogram(x_all, bins=bins, weights=weights2_all)
fig, ax = plt.subplots()
ax.plot(bin_edges_all[:-1], H_all, drawstyle='steps-mid', color='black', label='Volume weighting')
ax.plot(bin_edges_all[:-1], H2_all, drawstyle='steps-mid', color='violet', label='Mass weighting')
ax.set_xlabel('Density [g/cm^3]', fontsize=15)
ax.set_ylabel('Weighted count', fontsize =15)
ax.set_xscale('log')
ax.set_yscale('log')
plt.title("Density logspace histogram, showing both mass and volume weightings", fontsize=18)
ax.legend()
plt.show()

# standard graphing
fig,ax = plt.subplots(figsize=(7,7))
ax.set_xlabel("L [Mpc]", fontsize=15)
ax.set_ylabel(r"$\ log_{10}(Gas \ temperature \ [K])$", fontsize=15)
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(zc/Mpc,T)
plt.title("r -- 1: Plotting midpoint against different things", fontsize=18)

# rotation measure 2D plot code
L = 95.5 # simulation size
im = ax.imshow(RM_grid.T, cmap="Reds", origin='lower', aspect='equal', extent=[0,L,0,L], norm=LogNorm()) #.T transpose, gives "2D regular raster"
cbar = fig.colorbar(im)
cbar.set_label('Rotation measure', fontsize=13)
plt.show()