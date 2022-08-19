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


def ideal_gas_EoS_e(rho, gamma, p):
    eps = p/(rho*(gamma - 1))
    return eps


def ideal_gas_EoS_p(rho, gamma, eps):
    p = eps*(rho*(gamma - 1))
    return p


# def specific_internal_energy(rho, u, E):
#     eps = E/rho - 0.5*u**2
#     return eps


# def U_decomp(U):
#     rho = U[0]
#     u = U[1]/rho
#     E = U[2]
#     return rho, u, E


# def total_energy_per_unit_volume(rho, u, e):
#     E = rho*(0.5*u**2 - e)
#     return E


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


def relativistic_sound_velocity(p, rho, gamma):
    c_s = np.sqrt(gamma*p/rho)        # that is not relativistic
    return c_s


def updating(p, v, u, a_l, a_r, dt, dx):
    F_diff = flux_hlle(p, v, u, a_l, a_r)
    u -= dt/dx*F_diff
    return u


def flux_hlle(p, v, u, a_l, a_r):
    F_m_l = flux_vec(np.roll(p, 1), np.roll(v, 1), np.roll(u, 1, axis=1))
    F_m_r = flux_vec(p, v, u)
    F_p_l = flux_vec(p, v, u)
    F_p_r = flux_vec(np.roll(p, -1), np.roll(v, -1), np.roll(u, -1, axis=1))
    if a_l <= 0 <= a_r:
        F_m = (a_r*F_m_l - a_l*F_m_r + a_l*a_r *
               (u - np.roll(u, 1, axis=1)))/(a_r - a_l)
        F_p = (a_r*F_p_l - a_l*F_p_r + a_l*a_r *
               (np.roll(u, -1, axis=1) - u))/(a_r - a_l)
    elif a_l >= 0:
        F_m = F_m_l
        F_p = F_p_l
    elif a_r <= 0:
        F_m = F_m_r
        F_p = F_p_r
    F_diff = F_p - F_m
    return F_diff


def signal_velocities(v0_l, v0_r, c_s0):
    v_bar = (v0_l + v0_r)/2
    c_s_bar = (c_s0[0] + c_s0[-1])/2
    a_l = (v_bar - c_s_bar)/(1 - v_bar*c_s_bar)
    a_r = (v_bar + c_s_bar)/(1 + v_bar*c_s_bar)
    return a_l, a_r


def primitives(p, u, tol):
    p = newton_solver(p, u, tol)
    rho, v, eps = primitives_helper(p, u)
    return p, rho, v, eps


def primitives_helper(p, u):
    D, S, tau = state_vec_decomp(u)
    v = S/(tau + D + p)
    W = lorentz_factor(v)
    rho = D/W
    eps = (tau - D*W + p*(1 - W**2) + D)/(D*W)
    return rho, v, eps


def newton_solver(p, u, tol):
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
        # print(f'Newton solver: iteration = {counter}, max_norm = {max_norm}')
    return p_new


def f(p, u):
    rho, v, eps = primitives_helper(p, u)
    result = ideal_gas_EoS_p(rho, gamma, eps) - p
    return result


def f_prime(p, u, gamma, f_prime_command):
    D, S, tau = state_vec_decomp(u)
    result = eval(f_prime_command)
    return result


# Setup
# Loading the configuration
cfg = yaml.safe_load(open('config.yml'))
N_xcells, t_final, c_cfl = cfg['N_xcells'], cfg['t_final'], cfg['c_cfl']
gamma, L, tol, save_step = cfg['gamma'], cfg['L'], cfg['tol'], cfg['save_step']
v0_l, v0_r = cfg['v0_l'], cfg['v0_r']
p0_l, p0_r = cfg['p0_l'], cfg['p0_r']
rho0_l, rho0_r = cfg['rho0_l'], cfg['rho0_r']

# Computing the analytical derivative f_prime
f_prime_command = get_f_prime()

# Computing constants
dx = L/N_xcells
half = int(N_xcells/2)

# Initializing arrays
x = np.arange(-L/2, L/2, dx)    # Is that correct?
rho0 = np.ones(np.shape(x))
rho0[:half] *= rho0_l       # There seems to be problems…
rho0[half:] *= rho0_r
v0 = np.ones(np.shape(x))
v0[:half] *= v0_l
v0[half:] *= v0_r
p0 = np.ones(np.shape(x))
p0[:half] *= p0_l
p0[half:] *= p0_r

eps0 = ideal_gas_EoS_e(rho0, gamma, p0)
D0, S0, tau0 = conservatives(p0, rho0, v0, eps0)
u = state_vec(D0, S0, tau0)
p = p0.copy()       # I could directly name p0 as p above
rho = rho0.copy()
v = v0.copy()
eps = eps0.copy()

# Preparing time evolution
c_s0 = relativistic_sound_velocity(p0, rho0, gamma)
dt = cfl_condition(v0, c_s0, dx, c_cfl)
a_l, a_r = signal_velocities(v0_l, v0_r, c_s0)

# Time evolution
t = 0
k = 0
replace_list = [['$', ''], ['\\', '']]
while t < t_final:
    if k == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
    if k % save_step == 0:
        for var, name in [[p, r'$p$'], [rho, r'$\rho$'], [v, r'$v$'], [eps, r'$\epsilon$']]:
            ax.clear()
            ax.plot(x, var)
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(name)
            ax.set_xlim(-L/2, L/2)
            # fig.legend(f't = {t}')
            ax.text(0.05, 1.05, f't = {t}', horizontalalignment='left',
                    verticalalignment='center', transform=ax.transAxes)
            fig.tight_layout()
            # plt.draw()
            filename = name
            for i, o in replace_list:
                filename = filename.replace(i, o)
            fig.savefig(f'plots/{filename}_{k}.png', dpi=200)
            plt.pause(0.001)        # do I really need that?

    # print(f'Time evolution: iteration = {k}, time = {t} of {t_final}')
    u = updating(p, v, u, a_l, a_r, dt, dx)
    p, rho, v, eps = primitives(p, u, tol)
    t += dt
    k += 1
plt.show()

# The np.roll() leads to periodic boundary condition which we do not want.
