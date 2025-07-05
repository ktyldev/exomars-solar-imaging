import os
import sys
from datetime import date, time, datetime, timezone, timedelta
import spiceypy as spice
import tabulate
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math
import time


def load_meta_kernel(path):
    initial_dir = os.getcwd()
    # Change into the meta kernel's dir so that we can correctly load kernels
    # from the paths defined within
    os.chdir(os.path.dirname(path))
    # Load spatiotemporal data downlinked from rover
    spice.furnsh(os.path.basename(path))
    # Change back to the original directory
    os.chdir(initial_dir)


def get_oxia_planum_longitude(ephemeris_time):
    # Oxia Planum (landing site) position in MCMF
    oxia_planum="RM_SITE_000"
    oxia_planum_mcmf, light_times = spice.spkpos(oxia_planum, ephemeris_time, "IAU_MARS", "NONE", "MARS")
    oxia_planum_lonlat = spice.reclat(oxia_planum_mcmf)
    return oxia_planum_lonlat[1]


def get_mars_sol_length():
    # Prime meridian data contains rotation rate in degrees/day
    pm1 = spice.gdpool("BODY499_PM", 1, 1)[0]
    # Convert rotation rate to radians/second
    rotation_rate = spice.rpd() * pm1 / 86400
    rotation_period = 2 * spice.pi() / rotation_rate
    return rotation_period


# Local solar time is given in hours and minutes to represent an angle. Mars' day is longer
# than Earth's, so a solar second on Mars is longer than an SI second. This function converts
# from Martian solar seconds to SI seconds.
def solar_to_si_seconds(solar_seconds):
    # Multiply by the ratio of the length of a sol to the length of an SI day
    sol_length = get_mars_sol_length()
    return solar_seconds * sol_length / 86400


# Convert hours, minutes, seconds to seconds
def hms_to_seconds(hours, minutes, seconds):
    return seconds + 60*minutes + 3600*hours


# Take a time in ET and return an ET at which the next local midnight occurs to within one
# second, such that inputting the returned ET to spice.et2lst() returns 00:00:00
# TODO: qualify the accuracy of these midnight calculations
def get_next_solar_midnight_et(et):
    # Determine the amount of solar time between the given et and midnight
    lst = spice.et2lst(et, 499, get_oxia_planum_longitude(et), "PLANETOCENTRIC")

    # Time to solar midnight
    seconds = 60-lst[2]
    minutes = 59-lst[1]
    hours = 23-lst[0]

    si_seconds = solar_to_si_seconds(hms_to_seconds(hours, minutes, seconds))

    return et + si_seconds


def get_prev_solar_midnight_et(et):
    # Determine the amount of solar time between the given et and midnight
    lst = spice.et2lst(et, 499, get_oxia_planum_longitude(et), "PLANETOCENTRIC")

    seconds = lst[2]
    minutes = lst[1]
    hours = lst[0]

    si_seconds = solar_to_si_seconds(hms_to_seconds(hours, minutes, seconds))

    return et - si_seconds


def get_solar_noon(et):
    prev_midnight = get_prev_solar_midnight_et(et)
    next_midnight = get_next_solar_midnight_et(et)
    return (prev_midnight + next_midnight) / 2


# Convert an ephemeris time to a Python datetime object in UTC
def et_to_datetime(et):
    utc = spice.et2utc(et, "ISOC", 0)
    return datetime.strptime(utc, "%Y-%m-%dT%H:%M:%S")


def get_solar_elevation_base(et):
    oxia_planum_mcmf, light_times = spice.spkpos("RM_SITE_000", et, "IAU_MARS", "NONE", "MARS")

    # https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/cspice/azlcpo_c.html
    return spice.azlcpo(
            "ELLIPSOID",        # Only currently supported method
            "SUN",
            et,
            "NONE",             # Type of aberration correction. TODO: what is the appropriate type of aberration correction to do?
            False,              # Azimuth increases CCW if true. TODO: Does this increase over time?
            True,               # Elevation increases towards +Z
            oxia_planum_mcmf,
            "MARS",
            "IAU_MARS")


# Get the solar elevation in degrees at Oxia Planum at a specific ephemeris time
def get_solar_elevation(et):
    return get_solar_elevation_base(et)[0][2] * spice.dpr()


# Get the solar azimuth in degrees at Oxia Planum at a specific ephemeris time
def get_solar_azimuth(et):
    return get_solar_elevation_base(et)[0][1] * spice.dpr()


def get_solar_elevation_rate(et):
    return get_solar_elevation_base(et)[0][5]


def seconds_to_hms(seconds):
    hours = int(seconds / 3600)
    seconds -= hours * 3600
    minutes = int(seconds / 60)
    seconds -= minutes * 60
    return (hours, minutes, seconds)


def lerp(a, b, t):
    x = a + (b - a) * t
    return x


def inverse_lerp(a, b, x):
    t = (x - a) / (b - a)
    return t


def get_air_mass_plane_parallel(solar_zenith_angle, h_atmo):
    return 1.0 / (math.cos(math.radians(solar_zenith_angle)) / h_atmo)

def get_air_mass_circle_intersect(solar_zenith_angle, h_atmo):
    # TODO: get this from SPICE
    R_mars = 3389.5
    R_atmo = R_mars + h_atmo
    sza = math.radians(solar_zenith_angle)

    _2ab = (-2*R_mars*math.cos(sza))
    _4ac = 4*(1+math.cos(sza)**2)*(R_mars**2-R_atmo**2)
    sqrt = math.sqrt(_2ab**2 - _4ac)

    x1 = (-2*R_mars*math.cos(sza) + sqrt) / (2 * (1+math.cos(sza)**2))

    A = (0, R_mars)
    B = (x1, math.cos(sza)*x1+R_mars)
    dist = math.sqrt(x1**2 + (B[1]-A[1])**2)

    return dist

def get_air_mass(solar_zenith_angle, h_atmo):
    return get_air_mass_circle_intersect(solar_zenith_angle, h_atmo)

# Given a start and end time, determine when the Sun reaches a particular elevation after solar noon.
def get_solar_afternoon_elevation_transit_time(et, elevation):
    start_et = get_solar_noon(et)
    end_et = get_next_solar_midnight_et(et)

    # Generate some number of samples 
    step = 100
    times = [x*(end_et-start_et)/step+start_et for x in range(step)]
    elevation_angles = [get_solar_elevation(t) for t in times]

    before = None
    after = None

    for idx in range(len(times)):
        time = times[idx]
        if elevation_angles[idx] < elevation:
            before = (times[idx-1], elevation_angles[idx-1])
            after = (times[idx], elevation_angles[idx])
            break

    # Linearly interpolate to get time sun crosses the elevation angle
    t = inverse_lerp(before[1], after[1], elevation)

    return lerp(before[0], after[0], t)


def get_solar_disc_angular_width():
    # angular width = 2 * arcsin(diameter / 2*distance)
    # angular diameter of a sphere https://en.wikipedia.org/wiki/Angular_diameter

    # TODO: it would be more accurate to determine these from SPICE data
    diameter = 1.3914e9 # radius of the Sun in metres
    distance = 2.28e11  # distance from Mars to the Sun in metres

    angle_radians = 2*math.asin(diameter/(2*distance))
    angle_degrees = np.rad2deg(angle_radians)

    return angle_degrees


# TODO: this is a more generic range sampling function, is it the same as np linspace?
def get_times(from_et, to_et, step = 1000):
    return [x*(to_et-from_et)/step+from_et for x in range(step)]


# Get a number of ephemeris times within the sol specified by ET. The number of times corresponds to
# the optional step parameter, with a default value of 1000.
def get_sol_times(et, step = 1000):
    next_solar_midnight_et = get_next_solar_midnight_et(et)
    prev_solar_midnight_et = get_prev_solar_midnight_et(et)
    #noon_et = get_solar_noon(et)

    # Generate a bunch of times between prev and next midnight
    return get_times(prev_solar_midnight_et, next_solar_midnight_et, step)


# Get ET, elevation and azimuth data for the sunset period of the sol containing the specified ET.
def get_sunset_data(et):

    sunset_start_et = get_solar_afternoon_elevation_transit_time(et, 10)
    sunset_end_et = get_solar_afternoon_elevation_transit_time(et, 0)
    times = get_times(sunset_start_et, sunset_end_et)
    elevation = [get_solar_elevation(t) for t in times]
    azimuth = [get_solar_azimuth(t) for t in times]

    return times, azimuth, elevation


def plot_sun_position_base(ax, times, azimuths, elevations, et):
    ax.set_title(f"Path of sun as from {et_to_datetime(times[0])} to {et_to_datetime(times[-1])}")
    ax.set_ylabel(f"Elevation angle [$\\degree$]")
    ax.set_xlabel(f"Azimuth angle [$\\degree$]")
    ax.plot(azimuths, elevations, label="Sun path")

    ax.legend()


def get_sun_position_at_time(et):
    times = get_sol_times(et)
    elevations = [get_solar_elevation(t) for t in times]
    azimuths = [get_solar_azimuth(t) for t in times]

    return (np.interp(et, times, azimuths), np.interp(et, times, elevations))


def get_solar_disc_position(et, sunset_data = None):

    #start = time.time()
    if sunset_data == None:
        times, azimuths, elevations = get_sunset_data(et)
    else:
        times, azimuths, elevations = sunset_data
    #end = time.time()
    #print(f"get_sunset_data(): {end - start}")

    azimuth = get_solar_azimuth(et)
    elevation = get_solar_elevation(et)

    return (azimuth, elevation)


def plot_sunset_sun_position(ax, disc_pos, disc_diameter, et):
    times, azimuths, elevations = get_sunset_data(et)

    # Plot horizon
    margin = 2
    ax.plot([min(azimuths)-margin, max(azimuths)+margin], [0, 0], color="red", label="Horizon")

    # TODO: Plot notable points, eg sunset start and end
    # TODO: Demarcate elevation angles along the Sun's path 

    disc = patches.Circle(disc_pos, disc_diameter/2)
    text_pos = (disc_pos[0]+.5, disc_pos[1]+.5)
    ax.annotate(f"({disc_pos[0]:.2f}, {disc_pos[1]:.2f})", disc_pos, xytext=text_pos, arrowprops={
        "arrowstyle":"-",
        "relpos":(0,0)})
    ax.add_patch(disc)

    plot_sun_position_base(ax, times, azimuths, elevations, et)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(linestyle="--")


def plot_sol_sun_position(ax, et):
    times = get_sol_times(et)
    elevations = [get_solar_elevation(t) for t in times]
    azimuths = [get_solar_azimuth(t) for t in times]

    ax.set_ylim(-90, 90)

    # Show compass directions
    for hdg in [("N",0), ("E",90), ("S",180), ("W",270), ("N", 360)]:
        ax.plot([hdg[1], hdg[1]], [-5, 5], color="black")
        ax.text(hdg[1]-1.5, 6.5, hdg[0], color="black")

    ax.plot([0, 360], [0, 0], color="red", label="Horizon")

    sunset_times, sunset_azimuths, sunset_elevations = get_sunset_data(et)

    sunset_start_et = sunset_times[0]
    sunset_start_pos = get_sun_position_at_time(sunset_start_et)
    text_pos = (sunset_start_pos[0]+15, sunset_start_pos[1]+30)
    ax.annotate(et_to_datetime(sunset_start_et), sunset_start_pos, xytext=text_pos, arrowprops={
        "arrowstyle":"-",
        "relpos":(0,0)})

    sunset_end_et = sunset_times[-1]
    sunset_end_pos = get_sun_position_at_time(sunset_end_et)
    text_pos=(sunset_end_pos[0]+15, sunset_end_pos[1]+30)
    ax.annotate(et_to_datetime(sunset_end_et), sunset_end_pos, xytext=text_pos, arrowprops={
        "arrowstyle":"-",
        "relpos":(0,0)})

    rect_width = np.absolute(sunset_end_pos[0] - sunset_start_pos[0])
    rect_height = max(sunset_elevations) - min(sunset_elevations)
    rect_color = ("pink", .5)
    rect = patches.Rectangle((sunset_start_pos[0], max(sunset_elevations)), rect_width, -rect_height, facecolor=rect_color, label="Sunset")
    ax.add_patch(rect)

    plot_sun_position_base(ax, times, azimuths, elevations, et)


def sunset_graph():

    load_meta_kernel("./rocc-spice-kernels/kernels/mk/emrsp_test_rec_0382_v005.tm")
    #load_meta_kernel("./rocc-spice-kernels/kernels/mk/emrsp_test_rec_0003_v009.tm")
    #load_meta_kernel("./rocc-spice-kernels/kernels/mk/emrsp_test_tlm_0003_v009.tm")

    # Get the current date
    #dt = datetime.now()

    # An arbitrary date in the future (in the middle of a sunset maybe?)
    #dt = datetime(2026, 2, 25, hour=9, minute=10, second=0, microsecond=0, tzinfo=timezone.utc)

    # expected landing date
    landing_dt = datetime(2030, 11, 30, tzinfo=timezone.utc)
    # Subtract integer number of mars years
    landing_dt -= timedelta(days = 687 * 2)
    dt = landing_dt

    # Get primary mission end date by adding 218 sols to landing date
    #primary_mission_end_dt = landing_dt + timedelta(days=218)
    #print(primary_mission_end_dt)
    #sys.exit()

    utc = dt.strftime("%b %d, %Y %H:%M:%S")
    # Get the ephemeris time of the mission
    base_et = spice.str2et(utc)

    end_dt = dt + timedelta(days = 687)
    end_utc = end_dt.strftime("%b %d, %Y %H:%M:%S") 
    end_et = spice.str2et(end_utc)

    # Get the last and next midnight in LST at Oxia Planum of the current Martian sol
    next_midnight_et = get_next_solar_midnight_et(base_et)
    next_midnight = et_to_datetime(next_midnight_et)
    prev_midnight_et = get_prev_solar_midnight_et(base_et)
    prev_midnight = et_to_datetime(prev_midnight_et)
    noon_et = (prev_midnight_et+next_midnight_et)/2
    noon = et_to_datetime(noon_et)
    sunset_start_et = get_solar_afternoon_elevation_transit_time(base_et, 10)
    sunset_start = et_to_datetime(sunset_start_et)
    sunset_end_et = get_solar_afternoon_elevation_transit_time(base_et, 0)
    sunset_end = et_to_datetime(sunset_end_et)

    results = []

    results.append(("SPICE toolkit version", spice.tkvrsn("TOOLKIT")))
    #results.append(("Ephemeris UTC", et_to_datetime(frame_et)))
    #results.append(("Ephemeris ET", frame_et))

    # Get the current local solar time at Oxia Planum
    #lst = spice.et2lst(frame_et, 499, get_oxia_planum_longitude(frame_et), "PLANETOCENTRIC")
    #results.append(("Local Solar Time (LST) at Oxia Planum", lst[3]))

    #results.append(("Prev midnight", f"{prev_midnight} UTC"))
    #results.append(("Noon", f"{noon} UTC"))
    #results.append(("Sunset start", f"{sunset_start} UTC"))
    #results.append(("Sunset end", f"{sunset_end} UTC"))
    #results.append(("Next midnight", f"{next_midnight} UTC"))

    #print(tabulate.tabulate(results))


    fps = 4
    runtime = 6
    #fps = 1
    #runtime = 1
    frames = runtime * fps

    #os.makedirs("frames", exist_ok=True)

    for frame_idx in range(frames):

        fig = plt.figure()

        # Interpolate datetime
        #frame_et = lerp(sunset_start_et, sunset_end_et, frame_idx/frames)
        frame_et = lerp(base_et, end_et, frame_idx/frames)

        print(f"rendering frame {frame_idx}/{frames}")

        # TODO: show dates in L_s

        #plot_sol_elevation_angle(fig.add_subplot(221), et)
        #plot_sunset_elevation_angle(fig.add_subplot(222), et)
        # Plot elevation and azimuth on the same graph - the Sun will go all the way around over the course 
        # of one planetary rotation. This is like the first graph, but the X-axis is azimuth, rather than 
        # time.
        plot_sol_sun_position(fig.add_subplot(121), frame_et)

        sun_position = get_sun_position_at_time(frame_et)
        sun_diameter = 0.35
        plot_sunset_sun_position(fig.add_subplot(122), sun_position, sun_diameter, frame_et)

        # TODO: Identify key points (midnight, sunrise, noon, sunset, midnight) and label them with the
        # appropriate time.

        fig.suptitle(f"Solar disc position as seen from Oxia Planum at {et_to_datetime(frame_et)} UTC")

        plt.gcf().set_size_inches(20, 10)
        #plt.tight_layout()
        path = f"frames/frame{frame_idx:03}.png"
        plt.savefig(path, dpi=96)

        plt.close()

        # TODO: Calculate date in terms of M-year and sol number
        # TODO: Plot the min/max elevation angle of the Sun from Oxia Planum over a Martian year


def air_mass_graph():
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_title(f"Plane-parallel vs. circle intersect air mass approximation for 1-3km Mars planetary boundary layer")
    ax.margins(x=0.01, y=0.01)
    ax.set_ylim(0, 150)
    min_angle = 85
    max_angle = 90
    ax.set_xlim(min_angle, max_angle)

    # x axis zenith angle 0-90
    ax.set_xlabel(f"Solar zenith angle [deg]")

    # y axis air mass
    ax.set_ylabel(f"Air mass $\\eta$")

    # generate x points
    res = 10
    angles = [x/res for x in range(min_angle*res,max_angle*res)]
    xticks = [x for x in range(min_angle, max_angle+1)]
    ax.set_xticks(xticks)

    styles = ["-","--","-."]

    for h_atmo in range(1,4):
        style = styles[h_atmo-1]
        plane_parallel_y = [get_air_mass_plane_parallel(x, h_atmo) for x in angles]
        ax.plot(angles, plane_parallel_y, f"{style}c", label=f"plane-parallel {h_atmo}km")

    for h_atmo in range(1,4):
        style = styles[h_atmo-1]
        circle_intersect_y = [get_air_mass_circle_intersect(x, h_atmo) for x in angles]
        ax.plot(angles, circle_intersect_y, f"{style}m", label=f"circle intersect {h_atmo}km")

    ax.legend()
    plt.show()


if __name__ == "__main__":
    #sunset_graph()
    air_mass_graph()
