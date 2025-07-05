# geometry and optics components
import math
import geometry
import optics
import wac
import matplotlib.pyplot as plt
from datetime import date, time, datetime, timezone, timedelta
import spiceypy as spice
import numpy as np
import time
import sys

geometry.load_meta_kernel("./rocc-spice-kernels/kernels/mk/emrsp_test_rec_0382_v005.tm")

# expected landing date
landing_dt = datetime(2030, 11, 30, tzinfo=timezone.utc)
# Subtract integer number of mars years
landing_dt -= timedelta(days = 687 * 2)
dt = landing_dt

utc = dt.strftime("%b %d, %Y %H:%M:%S")
# Get the ephemeris time of the mission
base_et = spice.str2et(utc)

# observation params
sunset_start_angle = 7
sunset_end_angle = 0
# env params
h_atmo = 1
mixing_ratio = 8

sunset_start_et = geometry.get_solar_afternoon_elevation_transit_time(base_et, sunset_start_angle)
sunset_end_et = geometry.get_solar_afternoon_elevation_transit_time(base_et, sunset_end_angle)
sunset_duration = sunset_end_et - sunset_start_et

# Work out solar elevation and time halfway through sunset
mean_angle = (sunset_start_angle + sunset_end_angle)/2
mean_et = geometry.get_solar_afternoon_elevation_transit_time(base_et, mean_angle)

num_observations_per_trial = 25


def get_observation_times(sigma, interval):

    offset = np.random.normal(scale=sigma)
    # move sunset back one second to allow for measurement at exactly sunset. bit of a fudge
    # to avoid artifact in the graph
    observation_start_et = sunset_end_et-1 - (num_observations_per_trial+1) * (interval)

    times = []
    
    for i in range(num_observations_per_trial):
        times.append(observation_start_et + offset + interval * i)

    return times


def run_timing_trials(sigma, interval, trials, sunset_data):

    #print(f"sigma: {sigma} seconds")
    # constant
    mixing_ratio = 8
    h_atmo = 1

    good_observations = 0
    # TODO: Calcuate % of observations within an arbitrary "good" radius
    total_observations = trials * num_observations_per_trial

    for trial_idx in range(trials):

        times = get_observation_times(sigma, interval)

        for et in times:

            # Aim at centre of sun path
            aim_position = geometry.get_solar_disc_position((sunset_start_et + sunset_end_et) / 2, sunset_data = sunset_data)
            aim_position_x = aim_position[0]
            aim_position = (aim_position_x, aim_position[1])

            # Get solar position at time
            solar_disc_position = geometry.get_solar_disc_position(et, sunset_data)

            # Calculate angle of incidence (pythagorean distance between sun and aim position)
            angle_of_incidence = math.dist(solar_disc_position, aim_position)
    
            # calculate air mass from solar elevation angle
            solar_zenith_angle = 90-solar_disc_position[1]

            # can't see the sun below the horizon
            if solar_zenith_angle > 90:
                break

            air_mass = geometry.get_air_mass(solar_zenith_angle, h_atmo)
            # calculate radius from air mass and mixing ratio
            good_radius = optics.estimate_good_radius(air_mass, mixing_ratio)

            #print(angle_of_incidence, good_radius)
            if angle_of_incidence < good_radius:
                good_observations += 1

    return good_observations / total_observations


def run_timing_uncertainty():

    trials = 10
    good_radius = 5

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(f"Fraction of observations within {good_radius}$\\degree$ of boresight [{trials} trials]")
    ax.set_ylabel(f"Fraction")
    ax.set_xlabel(f"Timing error [minutes]")

    # Level of timing uncertainty, in minutes
    sigmas = [10, 15, 20, 25, 30]
    x = list(sigmas)
    good_observations = []

    # Convert to seconds
    sigmas = [60 * x for x in sigmas]

    #for sigma in sigmas:

    #    # Each trial has a random time offset so use many trials to get representative results
    #    trials = 1
    #    trial_data = []

    #    num_observations_per_trial = 10

    #    for trial_idx in range(trials):


    y = [run_timing_trials(xi, trials, good_radius) for xi in sigmas]

    print(x, y)
    ax.bar(x, y)

    #print(sigma, total_observations, good_observations[trial_idx] / total_observations)

    plt.show()
    plt.close()


# TODO: probably broken
def run_animation():

    #fps = 30
    #runtime = 6
    fps = 1
    runtime = 1
    frames = runtime * fps

    for frame_idx in range(frames):

        fig = plt.figure()

        # TODO: extract lerp (or better use an existing lerp from math or numpy modules)
        frame_et = geometry.lerp(sunset_start_et, sunset_end_et, frame_idx/frames)


        # Detemine sun positions in sky
        solar_disc_position = geometry.get_solar_disc_position(frame_et)

        # Plot sunset with Sun at ET
        # TODO: Get form WAC or move calculation out of WAC
        disc_diameter = 0.35

        #geometry.plot_sunset_sun_position(fig.add_subplot(121), solar_disc_position, disc_diameter, frame_et)

        # TODO: extract time experiment to its own function
        num_observations = 10
        sigma = 20
        time_between_observations = sunset_duration / num_observations

        aim_position = geometry.get_solar_disc_position((sunset_start_et + sunset_end_et) / 2)

        #for observation_idx in range(num_observations):

        # Get solar position at time

        # aim at centre of sun path
        aim_position_x = aim_position[0]
        aim_position = (aim_position_x, aim_position[1])

        # calculate angle of incidence (pythagorean distance between sun and aim position)
        angle_of_incidence = math.dist(solar_disc_position, aim_position)
        #print(angle_of_incidence)

        times = get_observation_times(sigma=sigma*60)
        solar_disc_positions = [geometry.get_solar_disc_position(t) for t in times]

        # Aim WAC at mean solar position
        cam = wac.WAC("L", 0, (0,0), aim_position)
        ##cam.render_image(fig.add_subplot(222), solar_disc_position)
        ax = fig.add_subplot(111)
        cam.render_image(ax, solar_disc_positions)
        ax.set_title(f"Simulated solar disc from PanCam WAC_L [$\\sigma={sigma} minutes$]")

        ## TODO: show water vapour signal change as sun moves
        ##optics.plot_effective_spectral_transmission(fig.add_subplot(224), angle_of_incidence)

        #plt.gcf().set_size_inches(20, 10)
        plt.gcf().set_size_inches(10, 10)
        #path = f"frames/frame{frame_idx:03}.png"
        path = f"frames/sigma{sigma}.png"
        plt.savefig(path, dpi=96)

        #plt.show()
        plt.close()

def run_metric_vis():
    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.set_facecolor("#000")
    ax.set_aspect(1)


    ax.set_title(f"Variation of useful observation radius over sunset at with water vapour layer thickness {h_atmo}km\n and water column depth of {mixing_ratio} pr. µm.")
    ax.set_xlabel("Heading [deg]")
    ax.set_ylabel("Solar elevation angle [deg]")

    padding = 10
    ax.set_ylim(sunset_end_angle - padding, sunset_start_angle + padding)
    avg_position = geometry.get_solar_disc_position((sunset_start_et + sunset_end_et) / 2)
    avg_x = avg_position[0]
    width = math.fabs(sunset_end_angle - sunset_start_angle)
    ax.set_xlim(avg_x - width/2 - padding, avg_x + width/2 + padding)

    # linearly interpolate observation times between start and end ETs
    observation_times = []
    res = 10
    for i in [x/res for x in range(0, res)]:
        observation_times.append(geometry.lerp(sunset_start_et, sunset_end_et, i))

    # determine aim positions
    observation_positions = [geometry.get_solar_disc_position(et) for et in observation_times]

    # TODO: plot horizon
    ground = plt.Rectangle((0, -10), 360, 10, color="orange", label="martian surface")
    ax.add_patch(ground)

    for i in range(len(observation_positions)):
        pos = observation_positions[i]

        # add sun to image
        solar_disc = plt.Circle(pos, geometry.get_solar_disc_angular_width(), color="#fff", label="sun" if i == 0 else "")
        ax.add_patch(solar_disc)

        # add radius around sun to image
        sza = 90-pos[1]

        air_mass = geometry.get_air_mass(sza, h_atmo)
        # TODO: determine radius from geometry? optical? component
        a = optics.estimate_good_radius(air_mass, mixing_ratio)
        green="#00ff00"
        ring = plt.Circle(pos, a, color=green, fill=False, zorder=2, label="1% max signal" if i == 0 else "")

        ax.add_patch(ring)

        # maximum possible radius
        a_max = 10
        red="#ff000088"
        ring_max = plt.Circle(pos, a_max, color=red, fill=False, zorder=1, label="10$\\degree$ limit" if i == 0 else "")
        ax.add_patch(ring_max)

        # label radius
        ax.text(math.sin(math.pi/4)*a+.5, -math.sin(math.pi/4)*a-.5, str(a), color=green)

    ax.legend()
    plt.show()

    # make a graph of good radius against solar elevation (we expect this to increase more than linearly)

def run_timing_and_spacing_uncertainty():
    print(f"run_timing_and_spacing_uncertainty")
    print(f"sunset: {sunset_start_angle}->{sunset_end_angle}", f"h_atmo: {h_atmo}km", f"mixing_ratio: {mixing_ratio} pr. µm")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)

    # ideal observation ENDS at exactly sunset
    # so to get the start time of an observation we multiply N image pairs * d interval back from end time
    # TODO: what are all the parameters for a single image pair?
    # TODO: dear god let there be a way to multithread this

    res = 32
    rows, cols = res, res

    # timing uncertainty range in minutes
    timing_uncertainty_min = 0
    timing_uncertainty_max = 15
    timing_uncertainties = np.linspace(timing_uncertainty_min, timing_uncertainty_max, cols).reshape(1, cols)

    # TODO: how to we make sure this is angles 
    interval_min = .0
    interval_max = .4
    # here we go max->min so as to start from the top left in the graph
    intervals = np.linspace(interval_max, interval_min, rows).reshape(1, rows)
    
    gradient_data = np.zeros(res).reshape(1, cols)
    gradient_data = np.tile(gradient_data, (rows, 1))

    # Preload data for sunset to speed up trials
    sunset_data = geometry.get_sunset_data(sunset_start_et)
    rate = (sunset_end_angle - sunset_start_angle) / (sunset_end_et - sunset_start_et)

    trials = 200
    total_trials = rows * cols * trials
    completed_trials = 0

    for i in range(len(timing_uncertainties[0])):

        timing_uncertainty = timing_uncertainties[0][i] * 60

        for j in range(len(intervals[0])):

            interval = math.fabs(intervals[0][j] * (1 / rate))

            result = run_timing_trials(timing_uncertainty, interval, trials, sunset_data)
            gradient_data[j, i] = result

            completed_trials += trials

            print(completed_trials, total_trials, completed_trials/total_trials)

    # then show something with the colours

    im = ax.imshow(gradient_data, cmap="viridis", aspect="auto", extent=[timing_uncertainty_min, timing_uncertainty_max, interval_min, interval_max])

    cs = ax.contour(gradient_data, levels=[.7, .8, .9], colors=["#000", "#000", "#000"], aspect="auto", extent=[timing_uncertainty_min, timing_uncertainty_max, interval_max, interval_min])
    ax.clabel(cs, cs.levels, fontsize=15)

    # color: % images with sun in target area
    cbar = fig.colorbar(im, ax=ax, label="Fraction of successful observations")

    #ax.grid(True, linestyle="--", alpha=0.7, color="black")


    # grid x: timing uncertainty y: interval
    ax.set_xlabel("Timing uncertainty $\\sigma$ [minutes]")
    ax.set_ylabel("Interval between images [deg]")

    fig.suptitle(f"Variation of observation success rate with degree interval between consecutive images and timing uncertainty [{trials} trials]")

    plt.show()

#run_timing_uncertainty()
run_timing_and_spacing_uncertainty()
#run_animation()
#run_metric_vis()
