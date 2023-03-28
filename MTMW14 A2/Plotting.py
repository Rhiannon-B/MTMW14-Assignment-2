from matplotlib import pyplot as plt
from Functions import *
import numpy as np
import math as m
from mpl_toolkits.axes_grid1 import make_axes_locatable


def analytic_plot(param):

    u_an, v_an, eta_an, X, Y = analytic(param)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ax1.contourf(X, Y, u_an)
    ax1.set_title("Zonal velocity profile")
    ax1.set_xlabel("Longitude (metres)")
    ax1.set_ylabel("Latitude (metres)")
    ax1.tick_params(labelsize=9)
    # Colourbar for U plot
    divider1 = make_axes_locatable(ax1)
    cbar1 = ax1.pcolormesh(u_an)
    cbar1_loc = divider1.new_vertical(size="5%", pad=0.6, pack_start=True)
    cbar1_loc.tick_params(labelsize=9)
    fig.add_axes(cbar1_loc)
    fig.colorbar(cbar1, cbar1_loc, orientation="horizontal",
                 label="Velocity (metres/second)")

    ax2.contourf(X, Y, v_an)
    ax2.set_title("Meridional velocity profile")
    ax2.set_xlabel("Longitude (metres)")
    ax2.set_ylabel("Latitude (metres)")
    ax2.tick_params(labelsize=9)
    # Colourbar for V plot
    divider2 = make_axes_locatable(ax2)
    cbar2 = ax2.pcolormesh(v_an)
    cbar2_loc = divider2.new_vertical(size="5%", pad=0.6, pack_start=True)
    cbar2_loc.tick_params(labelsize=9)
    fig.add_axes(cbar2_loc)
    fig.colorbar(cbar2, cbar2_loc, orientation="horizontal",
                 label="Velocity (metres/second)")

    ax3.contourf(X, Y, eta_an)
    ax3.set_title("Surface elevation")
    ax3.set_xlabel("Longitude (metres)")
    ax3.set_ylabel("Latitude (metres)")
    ax3.tick_params(labelsize=9)
    # Colourbar for eta plot
    divider3 = make_axes_locatable(ax3)
    cbar3 = ax3.pcolormesh(eta_an)
    cbar3_loc = divider3.new_vertical(size="5%", pad=0.6, pack_start=True)
    cbar3_loc.tick_params(labelsize=9)
    fig.add_axes(cbar3_loc)
    fig.colorbar(cbar3, cbar3_loc, orientation="horizontal",
                 label="Elevation (metres)")

    fig.set_size_inches(13, 5)
    fig.suptitle("Analytical Model", fontsize="16", va="center")
    plt.show()


def task_d1_plot(param):

    L = param["L"]
    dx = param["dx"]
    dy = param["dy"]
    days = param["days"]

    nx = m.ceil(L/dx)
    ny = m.ceil(L/dy)

    u, v, eta, X, Y, energy = scheme(param)

    # Isolating the edges of the gyre boundaries
    x_1d = np.linspace(0, L-dx, nx)
    y_1d = np.linspace(0, L-dy, ny)

    fig, axs = plt.subplots(2, 2)

    fig.suptitle("Numerical model run for %s days" %days, va="center")
    fig.set_size_inches(7, 8.2)
    fig.tight_layout(h_pad=5, w_pad=3.2)
    fig.subplots_adjust(left=0.13, top=0.90, bottom=0.08)

    # U velocity at southern boundary
    axs[0, 0].plot(x_1d, u[0, :])
    axs[0, 0].set_title("Zonal velocity along the \nsouthern edge of the basin",
                        fontsize="9")
    axs[0, 0].set_xlabel("Longitude (metres)", fontsize="9")
    axs[0, 0].set_ylabel("Zonal velocity (metres/second)", fontsize="9")
    axs[0, 0].tick_params(labelsize=9)

    # V velocity at western boundary
    axs[0, 1].plot(y_1d, v[:, 0])
    axs[0, 1].set_title("Meridional velocity along the \nwestern edge of the "
                        "basin", fontsize="9")
    axs[0, 1].set_xlabel("Latitude (metres)", fontsize="9")
    axs[0, 1].set_ylabel("Meridional velocity (metres/second)", fontsize="9")
    axs[0, 1].tick_params(labelsize=9)

    # Eta (surface height) through the longitudinal middle of the gyre
    axs[1, 0].plot(x_1d, eta[m.ceil(nx/2):(m.ceil(nx/2)+1), :].ravel())
    axs[1, 0].set_title("Surface elevation perturbation along \nthe zonal "
                        "center of the basin", fontsize="9")
    axs[1, 0].set_xlabel("Longitude (metres)", fontsize="9")
    axs[1, 0].set_ylabel("Surface elevation perturbation (metres)",
                         fontsize="9")
    axs[1, 0].tick_params(labelsize=9)

    axs[1, 1].contourf(X, Y, eta)
    axs[1, 1].set_title("Surface elevation perturbation", fontsize="9")
    axs[1, 1].set_xlabel("Longitude (metres)", fontsize="9")
    axs[1, 1].set_ylabel("Latitude (metres)", fontsize="9")
    axs[1, 1].tick_params(labelsize=9)

    # Colourbar for the eta plot
    divider = make_axes_locatable(axs[1, 1])
    cbar = axs[1, 1].pcolormesh(eta)
    cbar_loc = divider.new_vertical(size="5%", pad=0.6, pack_start="True")
    cbar_loc.tick_params(labelsize=9)
    fig.add_axes(cbar_loc)
    fig.colorbar(cbar, cbar_loc, orientation="horizontal",
                 label="Elevation (metres)")

    plt.show()


def numerical_plot(param):

    u, v, eta, X, Y, energy = scheme(param)
    days = param["days"]
    t = param["t"] * days
    dt = param["dt"]

    nt = m.ceil(0.5*t/dt)

    fig, axs = plt.subplots(2, 2)

    fig.suptitle("Numerical model: %s days" %days, va="center")
    fig.set_size_inches(7, 8.2)
    fig.tight_layout(h_pad=3.8, w_pad=2.8)
    fig.subplots_adjust(left=0.13, top=0.94, bottom=0.06)

    axs[0, 0].contourf(X, Y, u)
    axs[0, 0].set_title("Zonal velocity profile", fontsize="9")
    axs[0, 0].set_xlabel("Longitude (metres)", fontsize="9")
    axs[0, 0].set_ylabel("Latitude (metres)", fontsize="9")
    axs[0, 0].tick_params(labelsize=9)
    # Colourbar for u plot
    divider1 = make_axes_locatable(axs[0, 0])
    cbar1 = axs[0, 0].pcolormesh(u)
    cbar1_loc = divider1.new_vertical(size="5%", pad=0.6, pack_start="True")
    cbar1_loc.tick_params(labelsize=9)
    fig.add_axes(cbar1_loc)
    fig.colorbar(cbar1, cbar1_loc, orientation="horizontal",
                 label="Velocity (metres/second)")

    axs[0, 1].contourf(X, Y, v)
    axs[0, 1].set_title("Meridional velocity profile", fontsize="9")
    axs[0, 1].set_xlabel("Longitude (metres)", fontsize="9")
    axs[0, 1].set_ylabel("Latitude (metres)", fontsize="9")
    axs[0, 1].tick_params(labelsize=9)
    # Colourbar for the v plot
    divider2 = make_axes_locatable(axs[0, 1])
    cbar2 = axs[0, 1].pcolormesh(v)
    cbar2_loc = divider2.new_vertical(size="5%", pad=0.6, pack_start="True")
    cbar2_loc.tick_params(labelsize=9)
    fig.add_axes(cbar2_loc)
    fig.colorbar(cbar2, cbar2_loc, orientation="horizontal",
                 label="Velocity (metres/second)")

    axs[1, 0].contourf(X, Y, eta)
    axs[1, 0].set_title("Surface elevation", fontsize="9")
    axs[1, 0].set_xlabel("Longitude (metres)", fontsize="9")
    axs[1, 0].set_ylabel("Latitude (metres)", fontsize="9")
    axs[1, 0].tick_params(labelsize=9)
    # Colourbar for the eta plot
    divider3 = make_axes_locatable(axs[1, 0])
    cbar3 = axs[1, 0].pcolormesh(eta)
    cbar3_loc = divider3.new_vertical(size="5%", pad=0.6, pack_start="True")
    cbar3_loc.tick_params(labelsize=9)
    fig.add_axes(cbar3_loc)
    fig.colorbar(cbar3, cbar3_loc, orientation="horizontal",
                 label="Elevation (metres)")

    energy_arr = np.linspace(0, days, nt*2)

    axs[1, 1].plot(energy_arr, energy)
    axs[1, 1].set_title("Energy total in the numerical solution",
                        fontsize="9", pad=15)
    axs[1, 1].set_xlabel("Time (days)", fontsize="9")
    axs[1, 1].set_ylabel("Energy (Joules)", fontsize="9")
    axs[1, 1].tick_params(labelsize=9)

    print("Final energy difference (N - A) = %s Joules" %
          "{:e}".format(energy[-1]))

    plt.show()


def varying_IC_plot(param1, param2, param3):

    days1 = param1["days"]
    days2 = param2["days"]
    days3 = param3["days"]
    t = param3["t"] * days3
    dt = param3["dt"]
    eta0_sin = param1["eta0_sin"]

    u1, v1, eta1, X1, Y1, energy1 = scheme(param1)
    u2, v2, eta2, X2, Y2, energy2 = scheme(param2)
    u3, v3, eta3, X3, Y3, energy3 = scheme(param3)

    fig, axs = plt.subplots(2, 2)

    fig.suptitle("Numerical model: Initial sinusoidal surface \nelevation "
                 "perturbation with %s metre amplitude "
                 "\nMaxima in the east" %eta0_sin)
    fig.set_size_inches(7, 8.2)
    fig.tight_layout(h_pad=4, w_pad=3)
    fig.subplots_adjust(left=0.13, top=0.88, bottom=0.06)

    axs[0, 0].contourf(X1, Y1, eta1)
    axs[0, 0].set_title("Surface elevation at %s days" % days1, fontsize="9")
    axs[0, 0].set_xlabel("Longitude (metres)", fontsize="9")
    axs[0, 0].set_ylabel("Latitude (metres)", fontsize="9")
    axs[0, 0].tick_params(labelsize=9)
    # Colourbar for first plot
    divider1 = make_axes_locatable(axs[0, 0])
    cbar1 = axs[0, 0].pcolormesh(eta1)
    cbar1_loc = divider1.new_vertical(size="5%", pad=0.6, pack_start="True")
    cbar1_loc.tick_params(labelsize=9)
    fig.add_axes(cbar1_loc)
    fig.colorbar(cbar1, cbar1_loc, orientation="horizontal",
                 label="Velocity (metres/second)")

    axs[0, 1].contourf(X2, Y2, eta2)
    axs[0, 1].set_title("Surface elevation at %s days" % days2, fontsize="9")
    axs[0, 1].set_xlabel("Longitude (metres)", fontsize="9")
    axs[0, 1].set_ylabel("Latitude (metres)", fontsize="9")
    axs[0, 1].tick_params(labelsize=9)
    # Colourbar for second plot
    divider2 = make_axes_locatable(axs[0, 1])
    cbar2 = axs[0, 1].pcolormesh(eta1)
    cbar2_loc = divider2.new_vertical(size="5%", pad=0.6, pack_start="True")
    cbar2_loc.tick_params(labelsize=9)
    fig.add_axes(cbar2_loc)
    fig.colorbar(cbar2, cbar2_loc, orientation="horizontal",
                 label="Velocity (metres/second)")

    axs[1, 0].contourf(X3, Y3, eta3)
    axs[1, 0].set_title("Surface elevation at %s days" % days3, fontsize="9")
    axs[1, 0].set_xlabel("Longitude (metres)", fontsize="9")
    axs[1, 0].set_ylabel("Latitude (metres)", fontsize="9")
    axs[1, 0].tick_params(labelsize=9)
    # Colourbar for first plot
    divider3 = make_axes_locatable(axs[1, 0])
    cbar3 = axs[1, 0].pcolormesh(eta1)
    cbar3_loc = divider3.new_vertical(size="5%", pad=0.6, pack_start="True")
    cbar3_loc.tick_params(labelsize=9)
    fig.add_axes(cbar3_loc)
    fig.colorbar(cbar3, cbar3_loc, orientation="horizontal", label="Elevation")

    nt = m.ceil(0.5*t/dt)
    energy_arr = np.linspace(0, days3, nt*2)

    axs[1, 1].plot(energy_arr, energy3)
    axs[1, 1].set_title("Energy difference between numerical\n and analytical "
                        "solutions", fontsize="9")
    axs[1, 1].set_xlabel("Time (days)", fontsize="9")
    axs[1, 1].set_ylabel("Energy difference (Joules)", fontsize="9")

    plt.show()


# Difference plots

def diff_plot(param):

    u, v, eta, X, Y, energy = scheme(param)
    u_an, v_an, eta_an, X_an, Y_an = analytic(param)

    days = param["days"]
    t = param["t"] * days
    dt = param["dt"]

    nt = m.ceil(0.5*t/dt)

    fig, axs = plt.subplots(2, 2)

    fig.suptitle("Difference between Numerical and Analytical (N - A): %s days"
                 %days, va="center")
    fig.set_size_inches(7, 8.2)
    fig.tight_layout(h_pad=3.8, w_pad=2.8)
    fig.subplots_adjust(left=0.13, top=0.94, bottom=0.06)

    axs[0, 0].contourf(X, Y, u-u_an)
    axs[0, 0].set_title("Zonal velocity difference", fontsize="9")
    axs[0, 0].set_xlabel("Longitude (metres)", fontsize="9")
    axs[0, 0].set_ylabel("Latitude (metres)", fontsize="9")
    axs[0, 0].tick_params(labelsize=9)
    # Colourbar for u plot
    divider1 = make_axes_locatable(axs[0, 0])
    cbar1 = axs[0, 0].pcolormesh(u)
    cbar1_loc = divider1.new_vertical(size="5%", pad=0.6, pack_start="True")
    cbar1_loc.tick_params(labelsize=9)
    fig.add_axes(cbar1_loc)
    fig.colorbar(cbar1, cbar1_loc, orientation="horizontal",
                 label="Velocity (metres/second)")

    axs[0, 1].contourf(X, Y, v-v_an)
    axs[0, 1].set_title("Meridional velocity difference", fontsize="9")
    axs[0, 1].set_xlabel("Longitude (metres)", fontsize="9")
    axs[0, 1].set_ylabel("Latitude (metres)", fontsize="9")
    axs[0, 1].tick_params(labelsize=9)
    # Colourbar for the v plot
    divider2 = make_axes_locatable(axs[0, 1])
    cbar2 = axs[0, 1].pcolormesh(v)
    cbar2_loc = divider2.new_vertical(size="5%", pad=0.6, pack_start="True")
    cbar2_loc.tick_params(labelsize=9)
    fig.add_axes(cbar2_loc)
    fig.colorbar(cbar2, cbar2_loc, orientation="horizontal",
                 label="Velocity (metres/second)")

    axs[1, 0].contourf(X, Y, eta-eta_an)
    axs[1, 0].set_title("Surface elevation difference", fontsize="9")
    axs[1, 0].set_xlabel("Longitude (metres)", fontsize="9")
    axs[1, 0].set_ylabel("Latitude (metres)", fontsize="9")
    axs[1, 0].tick_params(labelsize=9)
    # Colourbar for the eta plot
    divider3 = make_axes_locatable(axs[1, 0])
    cbar3 = axs[1, 0].pcolormesh(eta)
    cbar3_loc = divider3.new_vertical(size="5%", pad=0.6, pack_start="True")
    cbar3_loc.tick_params(labelsize=9)
    fig.add_axes(cbar3_loc)
    fig.colorbar(cbar3, cbar3_loc, orientation="horizontal",
                 label="Elevation (metres)")

    energy_arr = np.linspace(0, days, nt*2)

    axs[1, 1].plot(energy_arr, energy)
    axs[1, 1].set_title("Energy difference between numerical\n and analytical "
                        "solutions", fontsize="9")
    axs[1, 1].set_xlabel("Time (days)", fontsize="9")
    axs[1, 1].set_ylabel("Energy difference (Joules)", fontsize="9")

    plt.show()
