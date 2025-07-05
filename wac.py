import matplotlib.pyplot as plt
import math
import numpy as np
import random

def get_rover_orientation_uncertainty():
    # Gaussian distribution to model uncertainty in rover orientation
    # TODO: is this a good model?
    x_mu = 0
    x_sigma = 10

    y_mu = 0
    y_sigma = 3

    x = random.gauss(x_mu, x_sigma)
    y = random.gauss(y_mu, y_sigma)

    return [x, y]


class WAC:
    # PanCam has two Wide-Angle Cameras (WAC) positioned at each end of the optical bench.

    # name
    # TODO: specify relation of toe in to aim position, some kind of diagram would be good
    # toe_in                how much the camera is rotated in the horizontal direction
    #                       relative to the PanCam assembly +Z (?) direction
    # pointing_uncertainty
    # aim_position
    def __init__(self, name, toe_in, pointing_uncertainty, aim_position):
        self.name = name
        self.toe_in = toe_in
        self.fov = 52.3 # degrees
        self.pointing_uncertainty = pointing_uncertainty
        self.aim_position = aim_position


    def get_fov_range(self):
        # TODO: reference technical drawing
        wac_fov = 52.3  # degrees
        return [-.5*wac_fov,.5*wac_fov]


    def get_solar_disc_position(self):
        # Begin with the assumption that the whole PanCam assembly is pointed at the Sun. In
        # this configuration the image from one WAC will be offset in the horizontal direction
        # due to the camera's toe-in. Value is negated since the position of the solar disc
        # will move in the opposite direction to the camera's toe-in.
        position = [-self.toe_in,0]
        position[0] = position[0] + self.pointing_uncertainty[0]
        position[1] = position[1] + self.pointing_uncertainty[1]
        return position




    def render_image(self, ax, solar_disc_positions):
        # Set background colour of figure
        ax.set_facecolor("#000")
        # WAC output is square
        ax.set_aspect(1)
        ax.set_title(f"Simulated solar disc from PanCam WAC-{self.name}")

        # Label the axes based on WAC field of view
        ax.set_xlim(self.get_fov_range())
        ax.set_xlabel("Horizontal degrees from optical axis")
        ax.set_ylim(self.get_fov_range())
        ax.set_ylabel("Vertical degrees from optical axis")

        horizon_y = -self.aim_position[1]
        ax.plot([-100, 100], [horizon_y, horizon_y], color="red", label="Horizon")

        # Produce an overlay concentric rings from 8 to 40 degrees in increments of 8
        #for a in range(5, 40, 10):
        a = 5
        green="#00ff00"
        ring = plt.Circle((0,0), a, color=green, fill=False, zorder=2)
        ax.add_patch(ring)
        ax.text(math.sin(math.pi/4)*a+.5, -math.sin(math.pi/4)*a-.5, str(a), color=green)

        for solar_disc_position in solar_disc_positions:

            #pos = self.get_solar_disc_position()
            pos_x = solar_disc_position[0] - self.aim_position[0]
            pos_y = solar_disc_position[1] - self.aim_position[1]
            pos = (pos_x, pos_y)
            #pos = (0, pos[1])

            solar_disc = plt.Circle(pos, self.get_solar_disc_angular_width(), color="#fff")
            ax.add_patch(solar_disc)



if __name__ == "__main__":
    fig, (ax_l, ax_r) = plt.subplots(1, 2)

    # Since the two WACs are rigidly connected to each other we should use the same pointing
    # uncertainty for a single observation from each camera
    pointing_uncertainty = get_rover_orientation_uncertainty()

    # Let's make a model of PanCam's left Wide-Angle Camera (WAC)
    toe_in = 2.8
    wac_l = WAC("L", toe_in, pointing_uncertainty)
    wac_l.render_image(ax_l)

    wac_r = WAC("R", -toe_in, pointing_uncertainty)
    wac_r.render_image(ax_r)

    # TODO: Stochastic simulation of many attempts at imaging
    #n_observations = 1000
    #solar_disc_positions = [get_solar_disc_position() for _ in range(n_observations)]

    #for pos in solar_disc_positions:
    #    solar_disc = plt.Circle(pos, get_solar_disc_angular_width(), color="#fff")
    #    ax.add_patch(solar_disc)
    print(f"solar disc is {wac_l.get_solar_disc_angular_width()} degrees wide")


    #plt.title(f"{n_observations} simulated solar discs from PanCam WAC-L")
    plt.show()

    fig.suptitle("TITLE")
    fig.savefig("wac.pdf")
