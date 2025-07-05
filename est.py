import matplotlib.pyplot as plt

def lerp(a, b, t):
    x = a + (b - a) * t
    return x


def inverse_lerp(a, b, x):
    t = (x - a) / (b - a)
    return t

angles = [0, 8, 16, 24, 32, 40]
delta_phi = [0.911, 0.681, -0.057, -0.0111, 0.014, 1.10e-05]
ylim = [-0.1,1.0]

fig = plt.figure()
fig.suptitle("Effective spectral transmission (EST) $\\Delta\\phi$ between $\\phi_{925}$ and $\\phi_{935}$ solar filters")
ax = fig.add_subplot(111)

t = inverse_lerp(delta_phi[1], delta_phi[2], .5)
a = lerp(angles[1], angles[2], t)

ax.plot([a, a], ylim, "g--", label="$\\Delta\\phi = 0.5$")
ax.grid()
ax.margins(x=0.00, y=0.01)
ax.set_ylim(ylim)

ax.plot(angles, delta_phi, label="$\\Delta\\phi$")
ax.set_xlabel("Incidence angle $i$ [deg]")
ax.set_ylabel("$\\Delta\\phi$ [%]")

plt.legend()
plt.show()
