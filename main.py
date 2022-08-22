#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 14:58:58 2022

@author: lars
"""


# Project: Sod shock tube
# by Lars Maiwald

import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
import os

from get_f_prime import get_f_prime


# Functions
def state_vec_decomp(u):
    D = u[0]
    S = u[1]
    tau = u[2]
    return D, S, tau


def flux_vec(p, v, u):
    D, S, tau = state_vec_decomp(u)
    F = np.array([D*v, S*v + p, S - D*v])
    return F


def state_vec(D, S, tau):
    u = np.array([D, S, tau])
    return u


def ideal_gas_EoS_eps(rho, gamma, p):
    eps = p/(rho*(gamma - 1))
    return eps


def ideal_gas_EoS_p(rho, gamma, eps):
    p = eps*(rho*(gamma - 1))
    return p


def lorentz_factor(v):
    W = 1/np.sqrt(1 - v**2)
    return W


def specific_enthalpy(p, rho, eps):
    h = 1 + eps + p/rho
    return h


def conservatives(p, rho, v, eps):
    h = specific_enthalpy(p, rho, eps)
    W = lorentz_factor(v)
    D = rho*W
    S = rho*h*W**2*v
    tau = rho*h*W**2 - p - D
    return D, S, tau


def cfl_condition(v, c_s, dx, c_cfl):
    V = (np.abs(v) + c_s)/(1 + np.abs(v)*c_s)
    dt = c_cfl*dx/np.max(V)
    return dt


def relativistic_sound_velocity(p, rho, eps, gamma):
    e = rho*(eps + 1)
    # is it okay to take abs()?
    c_s = np.sqrt(
        np.abs((gamma*(gamma - 1)*(e - gamma))/(rho + gamma*(e - rho))))
    return c_s


def updating(p, u, dt, dx, tol, gamma):
    F_diff = flux_hlle(p, u, dx, tol)
    u_a = u - dt/dx*F_diff
    # maybe we dont need that line and use p in the next
    p_a, rho_a, v_a, eps_a = primitives(p, u_a, tol)
    F_diff_a = flux_hlle(p_a, u_a, dx, tol)
    u_aa = u_a - dt/dx*F_diff_a
    u = (u + u_aa)/2
    return u


def flux_hlle(p, u, dx, tol):
    sigma_l = minmod_slope(roll_l(u), dx)
    u_l_m, u_l_p = intercell_states(roll_l(u), sigma_l, dx)
    p_l_p, rho_l_p, v_l_p, eps_l_p = primitives(p, u_l_p, tol)
    sigma = minmod_slope(u, dx)
    u_m, u_p = intercell_states(u, sigma, dx)
    p_m, rho_m, v_m, eps_m = primitives(p, u_m, tol)
    p_p, rho_p, v_p, eps_p = primitives(p, u_p, tol)
    sigma_r = minmod_slope(roll_r(u), dx)
    u_r_m, u_r_p = intercell_states(roll_r(u), sigma_r, dx)
    p_r_m, rho_r_m, v_r_m, eps_r_m = primitives(p, u_r_m, tol)
    F_m_l = flux_vec(p_l_p, v_l_p, u_l_p)
    F_m_r = flux_vec(p_m, v_m, u_m)
    F_p_l = flux_vec(p_p, v_p, u_p)
    F_p_r = flux_vec(p_r_m, v_r_m, u_r_m)
    c_s_m = relativistic_sound_velocity((p_l_p + p_m)/2, (rho_l_p + rho_m)/2,
                                        (eps_l_p + eps_m)/2, gamma)
    c_s_p = relativistic_sound_velocity((p_p + p_r_m)/2, (rho_p + rho_r_m)/2,
                                        (eps_p + eps_r_m)/2, gamma)
    a_m_l, a_m_r = signal_velocities(v_l_p, v_m, c_s_m)
    a_p_l, a_p_r = signal_velocities(v_p, v_r_m, c_s_p)
    b_m_l, b_m_r = min_insert(a_m_l), max_insert(a_m_r)
    b_p_l, b_p_r = min_insert(a_p_l), max_insert(a_p_r)
    F_m = (b_m_r*F_m_l - b_m_l*F_m_r + b_m_l *
           b_m_r*(u_m - u_l_p))/(b_m_r - b_m_l)
    F_p = (b_p_r*F_p_l - b_p_l*F_p_r + b_p_l *
           b_p_r*(u_r_m - u_p))/(b_p_r - b_p_l)
    F_diff = F_p - F_m
    return F_diff


def signal_velocities(v_l, v_r, c_s):
    v_bar = (v_l + v_r)/2
    c_s_bar = (c_s[0] + c_s[-1])/2
    a_l = (v_bar - c_s_bar)/(1 - v_bar*c_s_bar)
    a_r = (v_bar + c_s_bar)/(1 + v_bar*c_s_bar)
    return a_l, a_r


# the argument p is here just the initial guess for the Newton solver
def primitives(p, u, tol):
    p = newton_solver(p, u, tol, newton_max_it)
    rho, v, eps = primitives_helper(p, u)
    return p, rho, v, eps


def primitives_helper(p, u):
    D, S, tau = state_vec_decomp(u)
    v = S/(tau + D + p)
    if (v >= 1).any():
        print('Error: Velocity \'v\' exceeding speed of light.')
        sys.exit(1)
    W = lorentz_factor(v)
    rho = D/W
    if (rho < 0).any():
        print('Error: Negative mass density \'rho\' encountered.')
        sys.exit(1)
    eps = (tau - D*W + p*(1 - W**2) + D)/(D*W)
    if (eps < 0).any():
        print('Error: Negative specific internal energy \'eps\' encountered.')
        sys.exit(1)
    return rho, v, eps


def newton_solver(p, u, tol, newton_max_it):
    # print(p)
    counter = 0
    check = False
    p_new = np.zeros(np.shape(p))
    p_old = p.copy()
    while not check:
        counter += 1
        p_new = p_old - f(p_old, u)/f_prime(p_old, u, gamma, f_prime_command)
        max_norm = np.max(np.abs(p_new - p_old))
        check = (max_norm < tol)
        p_old = p_new.copy()
        if counter > newton_max_it:
            print(
                f'Newton solver did not converge in {newton_max_it} iterations.')
            sys.exit(1)
        if (p_new < 0).any():
            print('Error: Negative pressure \'p\' encountered.')
            sys.exit(1)
        # print(f'Newton solver: iteration = {counter}, max_norm = {max_norm}')
        # if check:
        #     print('Newton solver done.')
    return p_new


def f(p, u):
    rho, v, eps = primitives_helper(p, u)
    result = ideal_gas_EoS_p(rho, gamma, eps) - p
    return result


# could be computed more easily using f_prime = v**2*c_s**2 - 1
def f_prime(p, u, gamma, f_prime_command):
    D, S, tau = state_vec_decomp(u)
    result = eval(f_prime_command)
    return result


def roll_r(arr):
    if np.ndim(arr) == 2:
        axis = 1
    elif np.ndim(arr) == 1:
        axis = None
    arr_roll = np.roll(arr, -1, axis=axis)
    if axis == 1:
        arr_roll[:, -1] = arr_roll[:, -2]
    else:
        arr_roll[-1] = arr_roll[-2]
    return arr_roll


def roll_l(arr):
    if np.ndim(arr) == 2:
        axis = 1
    elif np.ndim(arr) == 1:
        axis = None
    arr_roll = np.roll(arr, 1, axis=axis)
    if axis == 1:
        arr_roll[:, 0] = arr_roll[:, 1]
    else:
        arr_roll[0] = arr_roll[1]
    return arr_roll


def minmod_basic(a, b):
    result = 0
    if a*b > 0:
        if np.abs(a) <= np.abs(b):
            result = a
        elif np.abs(b) < np.abs(a):
            result = b
    return result


minmod = np.vectorize(minmod_basic)


def mc_limiter(u, dx):
    sigma = minmod((roll_r(u) - roll_l(u))/(2*dx),
                   minmod(2*(u - roll_l(u)/dx),
                          2*(roll_r(u) - u/dx)))
    return sigma


def intercell_states(u, sigma, dx):
    u_m = u - dx*sigma/2
    u_p = u + dx*sigma/2
    return u_m, u_p


def min_insert(arr):
    mask = (arr > 0)
    arr[mask] = 0
    return arr


def max_insert(arr):
    mask = (arr < 0)
    arr[mask] = 0
    return arr


def isnan_checker(arr, name):
    mask = np.isnan(arr)
    if mask.any():
        result = True
    else:
        result = False
    print(f'{name} has nan element: {result}')


def minmod_slope(u, dx):
    sigma = minmod((u - roll_l(u))/dx, (roll_r(u) - u)/dx)
    return sigma


# Clearing output directory 'plots'
os.system('rm -r plots/*')

# Setup
# Loading the configuration
cfg = yaml.safe_load(open('config.yml'))
N_xcells, t_final, c_cfl = cfg['N_xcells'], cfg['t_final'], cfg['c_cfl']
gamma, L, tol, save_step = cfg['gamma'], cfg['L'], cfg['tol'], cfg['save_step']
newton_max_it = cfg['newton_max_it']
v0_l, v0_r = cfg['v0_l'], cfg['v0_r']
p0_l, p0_r = cfg['p0_l'], cfg['p0_r']
rho0_l, rho0_r = cfg['rho0_l'], cfg['rho0_r']

# Computing the analytical derivative f_prime
f_prime_command = get_f_prime()

# Computing constants
dx = L/N_xcells
half = int(N_xcells/2)

# Initializing arrays
x = np.arange(-L/2, L/2, dx)
rho = np.ones(np.shape(x))
rho[:half] *= rho0_l
rho[half:] *= rho0_r
v = np.ones(np.shape(x))
v[:half] *= v0_l
v[half:] *= v0_r
p = np.ones(np.shape(x))
p[:half] *= p0_l
p[half:] *= p0_r

eps = ideal_gas_EoS_eps(rho, gamma, p)
D, S, tau = conservatives(p, rho, v, eps)
u = state_vec(D, S, tau)

# Preparing time evolution
c_s = relativistic_sound_velocity(p, rho, eps, gamma)
dt = cfl_condition(v, c_s, dx, c_cfl)

# Time evolution
t = 0
k = 0
l = 0
replace_list = [['$', ''], ['\\', '']]
print('Starting simulation…')
while t < t_final:
    if k == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
    if k % save_step == 0:
        l += 1
        for var, name in [[p, r'$p$'], [rho, r'$\rho$'], [v, r'$v$'],
                          [eps, r'$\epsilon$']]:
            ax.clear()
            ax.plot(x, var)
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(name)
            ax.set_xlim(-L/2, L/2)
            ax.text(0.8, 1.05, f't = {t:6f}', horizontalalignment='left',
                    verticalalignment='center', transform=ax.transAxes)
            fig.tight_layout()
            filename = name
            for i, o in replace_list:
                filename = filename.replace(i, o)
            fig.savefig(f'plots/{filename}_{l}.png', dpi=200)

    print(f'Time evolution: iteration = {k}, time = {t:6f} of {t_final}')
    u = updating(p, u, dt, dx, tol, gamma)
    D, S, tau = state_vec_decomp(u)     # line not really neccessary
    p, rho, v, eps = primitives(p, u, tol)
    t += dt
    k += 1
print('Simulation done.')
print('Rendering animations…')
for name in ['p', 'rho', 'v', 'epsilon']:
    os.system(
        f'ffmpeg -r {int(1/(20*dt*save_step))} -f image2 -s 1200x800 -i plots/{name}_%d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p animations/{name}.mp4')
print('Rendering done.')
plt.show()
