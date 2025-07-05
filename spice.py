import spiceypy as spice
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Current question:
# Can I plot the motion of Mars relative to the Sun (and the Earth?)

# Print toolkit version
print(spice.tkvrsn("TOOLKIT"))


def load_meta_kernel(path):
    # Change into the meta kernel's dir so that we can correctly load kernels
    # from the paths defined within
    os.chdir(os.path.dirname(path))
    # Load spatiotemporal data downlinked from rover
    spice.furnsh(os.path.basename(path))

#load_meta_kernel("./exomarsrsp/kernels/mk/emrsp_test_tlm_0003_v010.tm")
load_meta_kernel("./rocc-spice-kernels/kernels/mk/emrsp_test_rec_0382_v005.tm")


step = 50000
# Give times in UTC to save our brain a little bit, but we actually want to convert to
# ephemeris times for data to do with the mission and local solar times for anything to
# do with the Sun.
utc = ['Dec 15, 2023', 'Dec 16, 2025']
# ephemeris times
etOne = spice.str2et(utc[0])
etTwo = spice.str2et(utc[1])
times = [x*(etTwo-etOne)/step+etOne for x in range(step)]

# Oxia Planum (landing site) position in MCMF
oxia_planum="RM_SITE_000"
oxia_planum_mcmf, light_times = spice.spkpos(oxia_planum, times[0], "IAU_MARS", "NONE", "MARS")
# Confirm this is Oxia Planum by converting to lat/lon
oxia_planum_lonlat = spice.reclat(oxia_planum_mcmf)

mars_naif_id = 499

# Having generated these times, convert them to LST to find our first midnight
days = []
current_day = []

for i in range(step-1):
    et0 = times[i]
    lst0 = spice.et2lst(et0, mars_naif_id, oxia_planum_lonlat[1], "PLANETOCENTRIC")

    et1 = times[i+1]
    lst1 = spice.et2lst(et1, mars_naif_id, oxia_planum_lonlat[1], "PLANETOCENTRIC")

    # Check if the two times are either side of midnight
    h0 = (lst0[0] + 12) % 24
    h1 = (lst1[0] + 12) % 24

    if h0 < 12 and h1 >= 12:
        days.append(current_day)
        current_day = []

    current_day.append(et0)
    # TODO: the last data element will never be added! but that might not be a problem

print(f"{len(days)} days")

# We now have binned days of ET timestamps. We now collect the data for each day and find the
# min and max dot product to find the range of zenith angle over that day.
sols = []
noon_zenith_angles = []
midnight_zenith_angles = []

for i in range(len(days)):
    times = days[i]

    site_positions, light_times = spice.spkpos(oxia_planum, times, "J2000", "NONE", "RM_ROVER")
    site_positions_n = [p / np.linalg.norm(p) for p in site_positions]
    site_positions_trans = np.asarray(site_positions).T

    sun_positions, light_times = spice.spkpos("SUN", times, "J2000", "NONE", "RM_ROVER")
    sun_positions_trans = np.asarray(sun_positions).T

    to_sun_vectors = [sp / np.linalg.norm(sp) for sp in sun_positions]
    to_sun = to_sun_vectors[0]

    dot_products = [np.dot(s, p) for (s, p) in zip(to_sun_vectors, site_positions_n)]

    sols.append(i)

    midnight_zenith_angles.append(dot_products[0])
    noon_zenith_angles.append(dot_products[int(len(times)/2)])

spice.kclear()




fig = plt.figure()
#ax = fig.add_subplot(121, projection='3d')
#ax.set_proj_type('ortho')
#ax.set_xlabel("X-axis")
#ax.set_ylabel("Y-axis")
#ax.set_zlabel("Z-axis")
#
#ax.quiver(-3000,-3000,3000,to_sun[0],to_sun[1],to_sun[2],length=1000,color="blue",  pivot="middle")
#
#ax.plot(site_positions_trans[0],site_positions_trans[1],site_positions_trans[2])
#
## Plot mars as a sphere
#r = 3.3895e3 # radius of Mars in km
#u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
#x = np.cos(u)*np.sin(v)*r
#y = np.sin(u)*np.sin(v)*r
#z = np.cos(v)*r
#ax.plot_surface(x,y,z, color="r", alpha=0.3)

#ax = fig.add_subplot(122)
ax = fig.add_subplot(111)

#ax.plot(times, dot_products)
ax.plot(sols, midnight_zenith_angles)
ax.plot(sols, noon_zenith_angles)

#plt.title(f"Position of Oxia Planum from {utc[0]} to {utc[1]}")
plt.show()


#help(spice.spkpos)

# target body name                              SUN
# observer epoch                                J2000? How can we get times associated with a SPICE kernel?
# reference frame of output vector
# aberration correction flag
# observing body of target
#
# outputs position of target,
# one way light time from observer to target
