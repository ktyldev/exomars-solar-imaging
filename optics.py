import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import numpy as np


def gaussian(x, mu, sig):
    return np.exp(-((x-mu)**2)/(2*sig**2))


class SolarFilter:
    def __init__(self, label, central_wavelength_against_aoi, plot_color):
        # meta
        self.label = label
        self.plot_color = plot_color

        # data
        self.aoi = [0, 8, 16, 24, 32]
        self.central_wavelength_against_aoi = central_wavelength_against_aoi
        # TODO: get real data for amplitude vs. AOI. This is eyeballed from the graphs by Patel (2024)
        # TODO: extract sample_amplitude function
        self.amplitudes = [0.8,0.775,0.75,0.725,0.7]

    def sample_cw(self, angle_of_incidence):
        return np.interp(angle_of_incidence, self.aoi, self.central_wavelength_against_aoi)

    # Return a function of wavelength
    def get_transmission_profile(self, angle_of_incidence):
        amplitude = np.interp(angle_of_incidence, self.aoi, self.amplitudes)
        cw = self.sample_cw(angle_of_incidence)
        sigma=1
        # TODO: in the data these are actually not simple Gaussians curves, but without actual
        # data points we model as being approximately gaussian curves.
        return (lambda wavelength : amplitude*gaussian(wavelength, cw, sigma))

        
# Filter CW shift data from Gunn et al.
# S01 = 925mm
# S02 = 935mm
# TODO: rename array since we would much like to use aoi in local scopes
aoi = [0, 8, 16, 24, 32]
s01_925_cw = [926.04,923.62,916.04,902.93,888.49]
# the amplitude drops as the wavelength is shifted
s01_925_a = [0.8,0.775,0.75,0.725,0.7]
s02_935_cw = [935.98,932.97,924.53,912.09,893.11]
s02_935_a = [0.8,0.775,0.75,0.725,0.7]

s01 = SolarFilter("S01 (925nm)", s01_925_cw, "red")
s02 = SolarFilter("S02 (935nm)", s02_935_cw, "blue")
# TODO: dust filters

#-----------------------------------------------------------------------------------------------
#Angle   Effective Transmission S01     Effective Transmission S02  Difference $\Delta\phi$ [%]
#        $\phi_{925}$                   $\phi_{935}$
#------  ----------------------------   --------------------------- ----------------------------
#0       0.99944                        0.99034                     0.911
#8       0.99962                        0.9928                      0.681
#16      0.999                          0.99957                     -0.057
#24      0.9987                         0.99881                     -0.0111
#32      0.99998                        0.99984                     0.014
#40      1                              1                           1.10e-5
delta_phis = [0.911,0.681,-0.057,-0.0111,0.014]


# TODO: Include current AOI
def plot_cw_shift_vs_aoi(ax, angle_of_incidence):

    # Plot base data
    ax.plot(aoi, [s01.sample_cw(a) for a in aoi], color=s01.plot_color)
    ax.plot(aoi ,[s02.sample_cw(a) for a in aoi], color=s02.plot_color)

    # TODO: Draw vertical line indicating AOI
    x_points = [angle_of_incidence, angle_of_incidence]
    y_points = [880, 940]
    ax.plot(x_points, y_points, linestyle="dashed", color="green")
    ax.text(angle_of_incidence, 880, f"{angle_of_incidence:.2f}$\\degree$", color="green")

    # TODO: draw horizontal lines indicating shifted CWs

    ax.set_title("CW vs Angle of Incidence for both Solar Filters")
    ax.set_ylabel("Central wavelength [nm]")
    ax.set_xlabel("Angle of incidence [deg]")


def plot_transmission_profile(ax, solar_filter, aoi):
    x_values = np.linspace(870, 950, 1000)

    transmission_profile = solar_filter.get_transmission_profile(aoi)
    ax.plot(x_values, transmission_profile(x_values), color=solar_filter.plot_color, label=solar_filter.label)


def plot_transmission_profiles(ax, solar_filters, aoi):
    for f in solar_filters:
        plot_transmission_profile(ax, f, aoi)
        # TODO: Draw lines indicating the apparent CWL

    ax.legend()


def plot_filter_transmission_profile_aoi_comparison(ax, solar_filter, base_color):
    for idx in range(len(aoi)):
        opacity = (len(aoi)-idx)/(len(aoi))-0.01
        plot_color = (base_color, opacity)
        plot_transmission_profile(ax, solar_filter, aoi[idx], plot_color)

    ax.legend()


def get_h2o_spectrum(min_wavelength, max_wavelength):
    # We generated a water vapour spectrum as in accordance Patel (2024)
    # for use in our simulation.

    # Plot H2O spectrum from 900-950nm based on PSG data (2030/10/10)
    # TODO: get spectra for other dates in Patel (2024)
    # * 2031/03/31 (Ls 120)
    # * 2031/10/31 (Ls 210)
    spectrum = []
    with open("../dat/psg-spectrum/psg_transmittance_203010100819_400-1000nm.txt") as f:
        for l in f.readlines():
            if l[0] == "#":
                continue

            parts = l.split()
            wavelength = float(parts[0])
            if wavelength < min_wavelength or wavelength > max_wavelength:
                continue

            h2o = float(parts[6])
            spectrum.append((wavelength, h2o))

    return spectrum

h2o_spectrum = get_h2o_spectrum(870, 950)

def draw_filter_region(ax, central_wavelength, filter_width, base_color):
    graph_base = 0.96
    # draw dashed line
    x_points = [central_wavelength, central_wavelength]
    y_points = [graph_base, 1]
    ax.plot(x_points, y_points, linestyle="dashed", color=base_color)
    ax.text(central_wavelength, graph_base, f"{central_wavelength:.2f}nm", color=base_color)

    rect_height = 1-graph_base
    rect_color = (base_color, .125)
    rect = patches.Rectangle((central_wavelength-filter_width/2, graph_base), filter_width, rect_height, facecolor=rect_color)
    ax.add_patch(rect)


# TODO: define transmittance
# Water vapour spectrum showing positions of narrow-band filters

def plot_h2o_spectrum_with_filter_positions(ax, aoi):
    ax.set_xlabel("Wavelength $\\lambda$ (nm)")
    ax.set_ylabel("Transmittance")
    ax.set_title("Transmittance of water spectrum between 900nm and 950nm")
    ax.plot([s[0] for s in h2o_spectrum], [s[1] for s in h2o_spectrum])

    draw_filter_region(ax, s01.sample_cw(aoi), 10, "blue")
    draw_filter_region(ax, s02.sample_cw(aoi), 10, "red")

def plot_aoi_vs_weighted_transmission_difference(ax):
    ax.set_title("Effect of incidence angle $\\theta$ on weighted transmission difference $\\Delta\\phi$ between 925nm and 935nm solar filters")
    ax.set_ylabel("Weighted transmission difference $\\Delta\\phi$ (%)")
    ax.set_xlabel("Incidence angle $\\theta$ ($\\degree$)")
    ax.plot(aoi, delta_phis)


def compute_spectral_product(spectrum_sample, filter_transmission_profile_sample):
    # As per equation 5.1 by Patel (2024):
    #
    #   Spectral Product (SP) = p(wl) x t(wl)
    #
    # where p(wl) is the water vapour spectrum from PSG and t(wl) is the modelled filter
    # characteristic.
    spectral_product = spectrum_sample*filter_transmission_profile_sample
    # TODO: question for Andrew/Priya: when multiplying as per eq 5.1 to get the spectral
    # product we get a curve that is almost the same as the transmission profile. This is
    # expected, since the water absorbtion feature is a tiny difference in transmission,
    # from ~1.0 to ~0.98, so almost all the value comes from filter profile. We were able
    # to get a similar shape only by subsequently subtracting the filter profile and
    # negating the result, whereby a spectral peak was obtained of the right order of
    # magnitude, but this approach has no scientific basis.
    return -(spectral_product-filter_transmission_profile_sample)*10


def estimate_good_radius(air_mass, mixing_ratio):
    # determined 
    base_radius = 10
    ideal_air_mass = 10
    ideal_mixing_ratio = 25 # pr. µm

    return base_radius * np.min([(air_mass/ideal_air_mass) * (mixing_ratio/ideal_mixing_ratio), 1])


def plot_effective_spectral_transmission(ax, angle_of_incidence):

    ax.set_title("Effective spectral transmission")
    ax.set_xlabel("Wavelength $\\lambda$ [nm]")
    ax.set_ylabel("Transmission")

    # Keep the scale of this graph constant regardless of output size. At 0deg the signal
    # is large, but it looks just as large when the y-axis is scaled, so we should not scale it.
    ax.set_ylim(0, 0.2)

    # Sample the water vapour spectrum at a given wavelength
    spectrum = get_h2o_spectrum(870, 950)
    wavelengths = [sample[0] for sample in spectrum]

    for solar_filter in [s01, s02]:
        spectral_transmissions = []
        for i in range(len(wavelengths)):
            wl = wavelengths[i]

            # TODO: Is the effect of the neutral density filter, whatever it may be
            # already present in the data?

            # Sample the transmission profile of a filter at a given wavelength
            filter_profile = solar_filter.get_transmission_profile(angle_of_incidence)

            spectral_product = compute_spectral_product(spectrum[i][1], filter_profile(wl))
            spectral_transmissions.append(spectral_product)

        #alpha=(len(aoi)-angle_idx+1)/(len(aoi)+1)
        ax.plot(wavelengths, spectral_transmissions, solar_filter.plot_color, label=solar_filter.label)

    #draw_filter_region(ax, 925, 10, "blue")
    #draw_filter_region(ax, 935, 10, "red")
    
    ax.legend()

if __name__ == "__main__":

    angle_of_incidence = 0


    fps = 30
    runtime = 6
    frames = runtime * fps
    max_angle_of_incidence = 16

    print()
    for frame_idx in range(frames):

        fig = plt.figure()
        print(f"rendering frame {frame_idx}/{frames}")
        
        angle_of_incidence = 1-(np.cos(frame_idx/frames*2*np.pi)+1)/2
        angle_of_incidence = angle_of_incidence * max_angle_of_incidence

        # Top-left
        plot_cw_shift_vs_aoi(fig.add_subplot(221), angle_of_incidence)

        # Top-right
        #plot_filter_transmission_profile_aoi_comparison(fig.add_subplot(443, ylabel="Transmittance"), s01, "blue")
        #plot_transmission_profile(fig.add_subplot(243, ylabel=""), s01, angle_of_incidence, "blue")
        #plot_filter_transmission_profile_aoi_comparison(fig.add_subplot(447, xlabel="Wavelength $\\lambda$ [nm]", ylabel="Transmittance"), s02, "red")
        #plot_transmission_profile(fig.add_subplot(244, yticklabels=[], xlabel="Wavelength $\\lambda$ [nm]"), s02, angle_of_incidence, "red")
        ax = fig.add_subplot(222, 
                             xlabel="Wavelength $\\lambda$[nm]",
                             ylabel="Transmittance",
                             title=f"Solar filter transmission profiles at {angle_of_incidence:.2f}$\\degree$ AOI")
        plot_transmission_profiles(ax, [s01, s02], angle_of_incidence)
        # Bottom-left
        plot_h2o_spectrum_with_filter_positions(fig.add_subplot(223), angle_of_incidence)

        # Bottom-right
        #plot_aoi_vs_weighted_transmission_difference(fig.add_subplot(224))
        plot_effective_spectral_transmission(fig.add_subplot(224), angle_of_incidence)

        fig.suptitle(f"Performance of S01 and S02 solar filters at {angle_of_incidence:.2f}$\\degree$ AOI using PSG spectrum generated for Oxia Planum at 2030/10/10 08:19")

        plt.gcf().set_size_inches(20, 10)
        plt.tight_layout()
        plt.savefig(f"frames/frame{frame_idx:03}.png", dpi=96)

        plt.close()
        #plt.show()
