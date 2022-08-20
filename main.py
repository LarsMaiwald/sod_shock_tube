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
from matplotlib.animation import FuncAnimation

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
    c_s = np.sqrt((gamma*(gamma - 1)*(e - gamma))/(rho + gamma*(e - rho)))
    return c_s


def updating(p, v, u, a_l, a_r, dt, dx):
    F_diff = flux_hlle(p, v, u, a_l, a_r)
    u -= dt/dx*F_diff
    return u


def flux_hlle(p, v, u, a_l, a_r):
    F_m_l = flux_vec(roll_mod_m(p), roll_mod_m(v), roll_mod_m(u, axis=1))
    F_m_r = flux_vec(p, v, u)
    F_p_l = flux_vec(p, v, u)
    F_p_r = flux_vec(roll_mod_p(p), roll_mod_p(v), roll_mod_p(u, axis=1))
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
    p = newton_solver(p, u, tol, newton_max_it)
    rho, v, eps = primitives_helper(p, u)
    return p, rho, v, eps


def primitives_helper(p, u):
    D, S, tau = state_vec_decomp(u)
    v = S/(tau + D + p)
    W = lorentz_factor(v)
    rho = D/W
    eps = (tau - D*W + p*(1 - W**2) + D)/(D*W)
    return rho, v, eps


def newton_solver(p, u, tol, newton_max_it):
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
        if counter == newton_max_it:
            sys.exit(1)
        # print(f'Newton solver: iteration = {counter}, max_norm = {max_norm}')
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


def roll_mod_p(arr, axis=None):
    arr_roll = np.roll(arr, -1, axis=axis)
    arr_roll[-1] = arr_roll[-2]
    return arr_roll


def roll_mod_m(arr, axis=None):
    arr_roll = np.roll(arr, 1, axis=axis)
    arr_roll[0] = arr_roll[1]
    return arr_roll


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

eps = ideal_gas_EoS_e(rho, gamma, p)
D, S, tau = conservatives(p, rho, v, eps)
u = state_vec(D, S, tau)

# Preparing time evolution
c_s = relativistic_sound_velocity(p, rho, eps, gamma)
dt = cfl_condition(v, c_s, dx, c_cfl)
a_l, a_r = signal_velocities(np.max(v[:half]), np.max(v[half:]), c_s)

# Time evolution
t = 0
k = 0
replace_list = [['$', ''], ['\\', '']]
while t < t_final:
    if k == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
    if k % save_step == 0:
        for var, name in [[p, r'$p$'], [rho, r'$\rho$'], [v, r'$v$'],
                          [eps, r'$\epsilon$']]:
            ax.clear()
            ax.plot(x, var)
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(name)
            ax.set_xlim(-L/2, L/2)
            ax.text(0.05, 1.05, f't = {t:6f}', horizontalalignment='left',
                    verticalalignment='center', transform=ax.transAxes)
            fig.tight_layout()
            filename = name
            for i, o in replace_list:
                filename = filename.replace(i, o)
            fig.savefig(f'plots/{filename}_{k}.png', dpi=200)

    print(f'Time evolution: iteration = {k}, time = {t:6f} of {t_final}',
          end='\r')
    u = updating(p, v, u, a_l, a_r, dt, dx)
    p, rho, v, eps = primitives(p, u, tol)

    c_s = relativistic_sound_velocity(p, rho, eps, gamma)
    dt = cfl_condition(v, c_s, dx, c_cfl)
    a_l, a_r = signal_velocities(np.max(v[:half]), np.max(v[half:]), c_s)

    t += dt
    k += 1
print('\n')
plt.show()


# fig, ax = plt.subplots(figsize=(6, 4))
# plot, = ax.plot(C[1:-1], Q[1:-1])
# text = ax.text(0.1, 0.9, f'{t=:.4f}')
# ax.set_xlabel(r'$C$')
# ax.set_ylabel(r'$Q$')
# ax.set_xlim(0,L)
# ax.set_ylim(0,1)
# fig.tight_layout()
# def animation_frame(frame, f, dx, dt, inter):
#     print(frame, end='\\r')
#     global t, Q, Q_new
#     if frame == 0:
#         plot.set_ydata(Q[1:-1])
#         return plot,
#     elif frame != 0:
#         for i in range(inter):
#             dt = 0.1*np.abs(dx/np.max(Q))
#             t += dt
#             setting_boundaries(Q)
#             sigma = limited_slope(Q, dx, mode)
#             Q_L_m_t, Q_R_m_t, Q_L_p_t, Q_R_p_t = half_way_states(Q, sigma, f, dx, dt)
#             F_m, F_p = godunov_flux(Q_L_m_t, Q_R_m_t, Q_L_p_t, Q_R_p_t, q_s, f)
#             Q = time_step(Q_new, Q, F_m, F_p, dx, dt)
#         plot.set_ydata(Q[1:-1])
#         text.set_text(f'{t=:.4f}')
#     return plot,
# animation = FuncAnimation(fig, func=animation_frame, frames=np.arange(0,100,1), interval=dt*0.6*1e5, fargs=(f_bu, dx, dt, inter)) # interval=2*dt*1000
# animation.save('anim1.mp4', dpi=200)
# plt.show()
