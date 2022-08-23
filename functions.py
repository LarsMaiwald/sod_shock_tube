#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 14:59:35 2022

@author: lars
"""

# Importing libraries
import numpy as np
import sympy as sp
import sys
# from math import log10, floor

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
    dt = 0.1*c_cfl*dx/np.max(V)
    return dt


def relativistic_sound_velocity(p, rho, eps, gamma):
    e = rho*(eps + 1)
    c_s = np.sqrt(
        np.abs((gamma*(gamma - 1)*(e - gamma))/(rho + gamma*(e - rho))))
    return c_s


def updating(p, u, dt, dx, gamma, f_prime_command, tol, newton_max_it):
    F_diff = flux_hlle(p, u, dx, gamma, f_prime_command, tol, newton_max_it)
    u_a = u - dt/dx*F_diff
    # maybe we dont need that line and use p in the next
    p_a, rho_a, v_a, eps_a = primitives(
        p, u_a, gamma, f_prime_command, tol, newton_max_it)
    F_diff_a = flux_hlle(p_a, u_a, dx, gamma,
                         f_prime_command, tol, newton_max_it)
    u_aa = u_a - dt/dx*F_diff_a
    u = (u + u_aa)/2
    return u


def flux_hlle(p, u, dx, gamma, f_prime_command, tol, newton_max_it):
    sigma_l = minmod_slope(roll_l(u), dx)
    u_l_m, u_l_p = intercell_states(roll_l(u), sigma_l, dx)
    p_l_p, rho_l_p, v_l_p, eps_l_p = primitives(
        p, u_l_p, gamma, f_prime_command, tol, newton_max_it)
    sigma = minmod_slope(u, dx)
    u_m, u_p = intercell_states(u, sigma, dx)
    p_m, rho_m, v_m, eps_m = primitives(
        p, u_m, gamma, f_prime_command, tol, newton_max_it)
    p_p, rho_p, v_p, eps_p = primitives(
        p, u_p, gamma, f_prime_command, tol, newton_max_it)
    sigma_r = minmod_slope(roll_r(u), dx)
    u_r_m, u_r_p = intercell_states(roll_r(u), sigma_r, dx)
    p_r_m, rho_r_m, v_r_m, eps_r_m = primitives(
        p, u_r_m, gamma, f_prime_command, tol, newton_max_it)
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
def primitives(p, u, gamma, f_prime_command, tol, newton_max_it):
    p = newton_solver(p, u, gamma, f_prime_command, tol, newton_max_it)
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


def newton_solver(p, u, gamma, f_prime_command, tol, newton_max_it):
    counter = 0
    check = False
    p_new = np.zeros(np.shape(p))
    p_old = p.copy()
    while not check:
        counter += 1
        p_new = p_old - f(p_old, u, gamma) / \
            f_prime(p_old, u, gamma, f_prime_command)
        max_norm = np.max(np.abs(p_new - p_old))
        check = (max_norm < tol)
        p_old = p_new.copy()
        if counter > newton_max_it:
            print(f'Newton solver did not converge in {newton_max_it}' +
                  ' iterations.')
            sys.exit(1)
        if (p_new < 0).any():
            print('Error: Negative pressure \'p\' encountered.')
            sys.exit(1)
    return p_new


def f(p, u, gamma):
    rho, v, eps = primitives_helper(p, u)
    result = ideal_gas_EoS_p(rho, gamma, eps) - p
    return result


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


def minmod_slope(u, dx):
    sigma = minmod((u - roll_l(u))/dx, (roll_r(u) - u)/dx)
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


def get_f_prime():
    p, rho, v, eps = sp.symbols('p, rho, v, eps')
    D, S, tau = sp.symbols('D, S, tau')
    W = sp.Symbol('W')
    gamma = sp.Symbol('gamma')
    f = sp.Function('f')(p)
    v = S/(tau + D + p)
    W = 1/sp.sqrt(1 - v**2)
    rho = D/W
    eps = (tau - D*W + p*(1 - W**2) + D)/(D*W)
    f = rho*eps*(gamma - 1) - p
    f_prime = f.diff(p)
    return str(f_prime).replace('sqrt', 'np.sqrt')


def round_significant(x, sig):
    result = np.round(x, sig - int(np.floor(np.log10(np.abs(x)))) - 1)
    return result


def significant_digits(x):
    result = - int(np.floor(np.log10(np.abs(x)))) - 1
    return result
