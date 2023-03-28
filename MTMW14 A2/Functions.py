from Constants import *
import numpy as np
import math as m


# Functions for the analytic solution
def f1(x, a, b):
    """
    Computes the f1 function required in the analytic solution

    :param x: zonal location along the grid
    :param a: quadratic root of one part of the analytic solution
    :param b: quadratic root of one part of the analytic solution
    :return: the result of f1 to be used in the analytic solution
    """

    return np.pi * (1 + (((np.exp(a) - 1) * np.exp(b * x) + (1 - np.exp(b)) *
                          np.exp(a * x)) / (np.exp(b) - np.exp(a))))


def f2(x, a, b):
    """
    Computes the f2 function required in the analytic solution

    :param x: zonal location along the grid
    :param a: quadratic root of one part of the analytic solution
    :param b: quadratic root of one part of the analytic solution
    :return: the result of f2 to be used in the analytic solution
    """
    return (((np.exp(a) - 1) * b * np.exp(b * x)) +
            ((1 - np.exp(b)) * a * np.exp(a * x))) / (np.exp(b) - np.exp(a))


# Analytic solution
def analytic(param):
    """
    Computes the analytic solution for the ocean gyre problem

    :param param: Set of variables given in Constants.py
    :return: u: zonal velocity grid
             v: meridional velocity grid
             eta: surface perturbation grid
             x: x-grid necessary for plotting the solutions
             y: y-grid necessary for plotting the solutions
    """
    L = param["L"]
    f0 = param["f0"]
    beta = param["beta"]
    g = param["g"]
    gamma = param["gamma"]
    rho = param["rho"]
    H = param["H"]
    tau0 = param["tau0"]
    tau_v = param["tauv"]
    eta0 = param["eta0"]
    dx = param["dx"]
    dy = param["dy"]
    dt = param["dt"]
    days = param["days"]
    t = param["t"] * days

    nx = m.ceil(L / dx)

    x_arr = np.linspace(0, L, nx)
    y_arr = np.linspace(0, L, nx)

    x, y = np.meshgrid(x_arr, y_arr)

    eps = gamma/(L*beta)
    a = (-1 - np.sqrt(1+((2*np.pi*eps)**2))/(2*eps))
    b = (-1 + np.sqrt(1+((2*np.pi*eps)**2)))/(2*eps)

    u = -(tau0/(np.pi*gamma*rho*H)) * f1(x/L, a, b)*np.cos(np.pi*y/L)
    v = (tau0/(np.pi*gamma*rho*H)) * f2(x/L, a, b)*np.sin(np.pi*y/L)
    for i in range(2):
        eta = eta0 + (tau0/(np.pi*gamma*rho*H) * (f0*L/g) *
                      gamma/(f0*np.pi) * f2(x/L, a, b)*np.cos(np.pi*y/L) +
                      f1(x/L, a, b)/np.pi * np.sin(np.pi*y/L) *
                      (1 + (beta*y/f0)) + (beta*L/(f0*np.pi))*np.cos(np.pi*y/L))

        # Resetting eta0 to be equal to eta(0, L/2)
        eta0 = eta[0, m.ceil(nx/2):(m.ceil(nx/2)+1)].ravel()

    return u, v, eta, x, y


# Functions for the numerical solution
def coriolis_f(y, nx, f0, beta, u=False):
    """
    Creates a 2D grid with values of the coriolis force as a function of
    latitude

    :param y: latitudinal through the ocean basin
    :param nx: number of x gridpoints
    :param f0: coriolis parameter independent of beta
    :param beta: change in the coriolis force per latitude unit
    :param u: zonal velocity
    :return: 2D grid of the coriolis force
    """

    if u == True:
        nx += 1

    cor_1d = f0 + (beta*y)
    cor_2d = np.tile(cor_1d, (nx, 1))
    cor_2d = cor_2d.transpose()
    return cor_2d


def tau_u_f(y, nx, L, tau0):
    """
    Creates a 2D grid with values of the wind stress force as a function of
    latitude

    :param y: latitudinal distance through the ocean basin
    :param nx: number of x gridpoints
    :param L: length of the ocean basin
    :param tau0: maximum wind stress vector
    :return: 2D grid of the wind stress force
    """

    tau_1d = -tau0*np.cos(np.pi*y/L)
    tau_2d = np.tile(tau_1d, (nx+1, 1))
    tau = np.transpose(tau_2d)
    return tau


def dudx_f(u, dx):
    """
    Computes the change in zonal velocity over a single gridspace in x

    :param u: zonal velocity grid
    :param dx: zonal gridsize
    :return: change in zonal velocity over a gridspace in x
    """

    dudx = np.diff(u, axis=1)/dx
    return dudx


def dvdy_f(v, dy):
    """
    Computes the change in meridional velocity over a single gridspace in y

    :param v: meridional velocity grid
    :param dy: meridional gridsize
    :return: change in meridional velocity over a gridspace in y
    """

    dvdy = np.diff(v, axis=0)/dy
    return dvdy


def detadx_f(eta, dx):
    """
    Computes the change in ocean surface height perturbation over a single
    gridspace in x

    :param eta: ocean surface height perturbation grid
    :param dx: zonal gridsize
    :return: change in ocean surface height perturbation over a gridspace in x
    """

    x_length = len(eta)

    eta_static = np.hstack((eta, np.zeros((x_length, 1))))/dx
    eta_shifted = np.hstack((np.zeros((x_length, 1)), eta))/dx

    detadx = eta_static - eta_shifted

    return detadx


def detady_f(eta, dy):
    """
    Computes the change in ocean surface height perturbation over a single
    gridspace in y

    :param eta: ocean surface height perturbation grid
    :param dy: zonal gridsize
    :return: change in ocean surface height perturbation over a gridspace in y
    """

    y_length = len(eta.transpose())

    eta_static = np.vstack((eta, np.zeros((1, y_length))))/dy
    eta_shifted = np.vstack((np.zeros((1, y_length)), eta))/dy

    detadx = eta_static - eta_shifted

    return detadx


def v_interpolation(v_grid):
    """
    Maps the meridional velocity grid onto the gridpoints of the zonal velocity
    grid

    :param v_grid: meridional velocity grid
    :return: meridional velocity grid averaged onto the zonal velocity grid
    """

    grid_length = len(v_grid)-1
    zero = np.zeros((grid_length, 1))

    y0x0 = np.hstack((zero, v_grid[:-1, :]))/4
    y1x0 = np.hstack((zero, v_grid[1:, :]))/4
    y0x1 = np.hstack((v_grid[:-1, :], zero))/4
    y1x1 = np.hstack((v_grid[1:, :], zero))/4

    v_interpolated = y0x0 + y0x1 + y1x0 + y1x1

    return v_interpolated


def u_interpolation(u_grid):
    """
    Maps the zonal velocity grid onto the gridpoints of the meridional velocity
    grid

    :param u_grid: zonal velocity grid
    :return: zonal velocity grid averaged onto the meridional velocity grid
    """

    grid_length = len(u_grid.transpose())-1
    zero = np.zeros((1, grid_length))

    y0x0 = np.vstack((zero, u_grid[:, :-1]))/4
    y0x1 = np.vstack((zero, u_grid[:, 1:]))/4
    y1x0 = np.vstack((u_grid[:, :-1], zero))/4
    y1x1 = np.vstack((u_grid[:, 1:], zero))/4

    u_interpolated = y0x0 + y0x1 + y1x0 + y1x1

    return u_interpolated


def eta_v_interp(v_grid):
    """
    Maps the meridional velocity grid onto the gridpoints of the ocean surface
    height perturbation grid

    :param v_grid: meridional velocity grid
    :return: meridional velocity grid averaged onto the ocean surface height
    perturbation grid
    """

    v_forward = v_grid[1:, :]/2
    v_backward = v_grid[:-1, :]/2

    eta_v_interpolated = v_forward + v_backward

    return eta_v_interpolated


def eta_u_interp(u_grid):
    """
    Maps the zonal velocity grid onto the gridpoints of the ocean surface height
    perturbation grid

    :param u_grid: zonal velocity grid
    :return: zonal velocity grid averaged onto the ocean surface height
    perturbation grid
    """

    u_forward = u_grid[:, 1]/2
    u_backward = u_grid[:, -1]/2

    eta_u_interpolated = u_forward + u_backward

    return eta_u_interpolated


def energy_per_step(u, v, eta, rho, H, g, dx, dy):
    """
    Computes the energy anomaly associated with the ocean gyre

    :param u: zonal velocity grid
    :param v: meridional velocity grid
    :param eta: ocean surface height perturbation grid
    :param rho: water density
    :param H: ocean basin depth
    :param g: vertical acceleration due to gravity
    :param dx: gridspace size in x
    :param dy: gridspace size in y
    :return: total energy anomaly in Joules for the gyre
    """

    energy = np.sum(0.5*rho*(H*(u**2 + v**2) + g*eta**2)*dx*dy)

    return energy


# Numerical solution
def scheme(param):

    L = param["L"]
    f0 = param["f0"]
    beta = param["beta"]
    g = param["g"]
    gamma = param["gamma"]
    rho = param["rho"]
    H = param["H"]
    tau0 = param["tau0"]
    tau_v = param["tauv"]
    eta0 = param["eta0"]
    eta0_sin = param["eta0_sin"]
    dx = param["dx"]
    dy = param["dy"]
    dt = param["dt"]
    days = param["days"]
    t = param["t"] * days

    # nt is halved because the time loop iterates two timesteps
    nt = m.ceil(0.5*t/dt)
    nx = m.ceil(L/dx)
    ny = m.ceil(L/dy)

    # Latitude grids used for the wind stress and coriolis effect
    y_u = np.linspace(0, L-dy, ny)
    y_v = np.linspace(0, L, ny+1)

    cor_u = coriolis_f(y_u, nx, f0, beta, u=True)
    cor_v = coriolis_f(y_v, nx, f0, beta, u=False)
    tau_u = tau_u_f(y_u, nx, L, tau0)

    # Used in plotting
    x_arr = np.linspace(0, L-dy, nx)
    y_arr = np.linspace(0, L, nx)
    x, y = np.meshgrid(x_arr, y_arr)

    # Analytical model used for energy calculations
    u_an, v_an, eta_an, X_an, Y_an = analytic(param)
    u_diff = np.zeros(nt*2)
    v_diff = np.zeros(nt*2)
    eta_diff = np.zeros(nt*2)
    timestep_energy_total = np.zeros(nt*2)
    timestep_energy_diff = np.zeros(nt*2)
    energy_total_an = energy_per_step(u_an, v_an, eta_an, rho, H, g, dx, dy)

    # Parameter grids
    eta0 = np.zeros((ny, nx))
    v0 = np.zeros((ny + 1, nx))
    u0 = np.zeros((ny, nx + 1))

    if eta0_sin > 0:
        eta0 = eta0_sin*np.tile(-np.sin(np.linspace(0, 2 * np.pi, nx)), (nx, 1))

    for t in range(nt):

        # Eta, first timestep
        eta1 = eta0 - H*dt*(dudx_f(u0, dx) + dvdy_f(v0, dy))

        # U, first timestep
        u1 = u0 + cor_u*dt*v_interpolation(v0) - g*dt*detadx_f(eta1, dx) - \
             gamma*dt*u0 + (tau_u/(rho*H))*dt
        # Applying boundary conditions at the edges
        u1[:, 0] = 0
        u1[:, -1] = 0

        # V, first timestep
        v1 = v0 - cor_v*dt*u_interpolation(u1) - g*dt*detady_f(eta1, dy) - \
             gamma*dt*v0 + (tau_v/(rho*H))*dt
        # Applying boundary conditions at the edges
        v1[0, :] = 0
        v1[-1, :] = 0

        # Calculating total energy in the system and energy difference from the
        # analytical model for the first timestep
        timestep_energy_total[2*t] = \
            energy_per_step(eta_u_interp(u1), eta_v_interp(v1),
                            eta0, rho, H, g, dx, dy)

        # Eta, second timestep
        eta2 = eta1 - H*dt*(dudx_f(u1, dx) + dvdy_f(v1, dy))

        # V, second timestep
        v2 = v1 - cor_v*dt*u_interpolation(u1) - g*dt*detady_f(eta2, dy) - \
             gamma*dt*v1 + (tau_v/(rho*H))*dt
        # Applying boundary conditions at the edges
        v2[0, :] = 0
        v2[-1, :] = 0

        # U, second timestep
        u2 = u1 + cor_u*dt*v_interpolation(v2) - g*dt*detadx_f(eta2, dx) - \
             gamma*dt*u1 + (tau_u/(rho*H))*dt
        # Applying boundary conditions at the edges
        u2[:, 0] = 0
        u2[:, -1] = 0

        # Calculating total energy in the system and energy difference from the
        # second analytical model
        timestep_energy_total[2*t + 1] = \
            energy_per_step(eta_u_interp(u2), eta_v_interp(v2),
                            eta2, rho, H, g, dx, dy)

        # Resetting initial grids for the next timestep
        eta0 = eta2
        u0 = u2
        v0 = v2

    return u0[:, :-1], v0[:-1, :], eta0, x, y, \
           energy_total_an - timestep_energy_total
