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


# Functions
# Create state vector from conservative variables
def state_vec(D, S, tau):
    u = np.array([D, S, tau])
    return u


# Decompose a state vector into the conservative variables
def state_vec_decomp(u):
    D = u[0]
    S = u[1]
    tau = u[2]
    return D, S, tau


# Compute flux vector
def flux_vec(p, v, u):
    D, S, tau = state_vec_decomp(u)
    F = np.array([D*v, S*v + p, S - D*v])
    return F


# Computing the flux vector from the primitive variables
def flux_vec_prim(U):
    p, rho, v, eps = U[0], U[1], U[2], U[3]
    D, S, tau = conservatives(p, rho, v, eps)
    u = state_vec(D, S, tau)
    F = flux_vec(p, v, u)
    return F


# Using EoS for ideal gas to compute the specific internal energy
def ideal_gas_EoS_eps(rho, gamma, p):
    eps = p/(rho*(gamma - 1))
    return eps


# Using EoS for ideal gas to compute the pressure
def ideal_gas_EoS_p(rho, gamma, eps):
    p = eps*(rho*(gamma - 1))
    return p


# Compute the specific enthalpy
def specific_enthalpy(p, rho, eps):
    h = 1 + eps + p/rho
    return h


# Compute the Lorentz factor
def lorentz_factor(v):
    W = 1/np.sqrt(1 - v**2)
    return W


# Compute the conservative variables from the primitive variables
def conservatives(p, rho, v, eps):
    h = specific_enthalpy(p, rho, eps)
    W = lorentz_factor(v)
    D = rho*W
    S = rho*h*W**2*v
    tau = rho*h*W**2 - p - D
    return D, S, tau


# Compute the primitive variables from the state vector / conservatives
def primitives(p, u, gamma, f_prime_command, tol, newton_max_it):
    p = newton_solver(p, u, gamma, f_prime_command, tol, newton_max_it)
    rho, v, eps = primitives_helper(p, u)
    return p, rho, v, eps


# Compute density, velocity and specific internal energy from the state vector
# and the pressure
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


# Newton solver for the pressure
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


# Function to minimize for pressure Newton solver
def f(p, u, gamma):
    rho, v, eps = primitives_helper(p, u)
    result = ideal_gas_EoS_p(rho, gamma, eps) - p
    return result


# Derivative of f(p)
def f_prime(p, u, gamma, f_prime_command):
    D, S, tau = state_vec_decomp(u)
    result = eval(f_prime_command)
    return result


# Derive analytical equation for derivative of f(p)
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


# Compute the time-step according to the CFL condition
def cfl_condition(v, c_s, dx, c_cfl):
    V = (np.abs(v) + c_s)/(1 + np.abs(v)*c_s)
    dt = 0.1*c_cfl*dx/np.max(V)
    return dt


# Compute the relativistic sound velocity from the primitive variables
def relativistic_sound_velocity(p, rho, eps, gamma):
    e = rho*(eps + 1)
    c_s = np.sqrt(
        np.abs((gamma*(gamma - 1)*(e - gamma))/(rho + gamma*(e - rho))))
    return c_s


# Compute the slowest and fastest signal velocity
def signal_velocities(v_l, v_r, c_s):
    v_bar = (v_l + v_r)/2
    c_s_bar = (c_s[0] + c_s[-1])/2
    a_l = (v_bar - c_s_bar)/(1 - v_bar*c_s_bar)
    a_r = (v_bar + c_s_bar)/(1 + v_bar*c_s_bar)
    return a_l, a_r


# Compute intercell states using state vector and slope limiter
def intercell_states(u, sigma, dx):
    u_m = u - dx*sigma/2
    u_p = u + dx*sigma/2
    return u_m, u_p


# Compute minmod for scalars
def minmod_basic(a, b):
    result = 0
    if a*b > 0:
        if np.abs(a) <= np.abs(b):
            result = a
        elif np.abs(b) < np.abs(a):
            result = b
    return result


# Compute minmod for arrrays
minmod = np.vectorize(minmod_basic)


# Compute minmod slope as slope limiter
def minmod_slope(u, dx):
    sigma = minmod((u - roll_l(u))/dx, (roll_r(u) - u)/dx)
    return sigma


# Update the state vector one time-step
def updating(p, u, dt, dx, gamma, f_prime_command, tol, newton_max_it, hor):
    F_diff = flux_hlle(p, u, dx, gamma, f_prime_command,
                       tol, newton_max_it, hor)
    u_a = u - dt/dx*F_diff
    p_a, rho_a, v_a, eps_a = primitives(
        p, u_a, gamma, f_prime_command, tol, newton_max_it)
    F_diff_a = flux_hlle(p_a, u_a, dx, gamma,
                         f_prime_command, tol, newton_max_it, hor)
    u_aa = u_a - dt/dx*F_diff_a
    u = (u + u_aa)/2
    return u


# Compute the flux vector difference according to the RHLLE scheme
def flux_hlle(p, u, dx, gamma, f_prime_command, tol, newton_max_it, hor):
    if hor == 0:
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
        c_s_m = relativistic_sound_velocity((p_l_p + p_m)/2,
                                            (rho_l_p + rho_m)/2,
                                            (eps_l_p + eps_m)/2, gamma)
        c_s_p = relativistic_sound_velocity((p_p + p_r_m)/2,
                                            (rho_p + rho_r_m)/2,
                                            (eps_p + eps_r_m)/2, gamma)
        a_m_l, a_m_r = signal_velocities(v_l_p, v_m, c_s_m)
        a_p_l, a_p_r = signal_velocities(v_p, v_r_m, c_s_p)
    elif hor == 1:
        p, rho, v, eps = primitives(
            p, u, gamma, f_prime_command, tol, newton_max_it)
        U = np.array([p, rho, v, eps])
        sigma_l = minmod_slope(roll_l(U), dx)
        U_l_m, U_l_p = intercell_states(roll_l(U), sigma_l, dx)
        u_l_p = state_vec(
            *conservatives(U_l_p[0], U_l_p[1], U_l_p[2], U_l_p[3]))
        sigma = minmod_slope(U, dx)
        U_m, U_p = intercell_states(U, sigma, dx)
        u_m = state_vec(*conservatives(U_m[0], U_m[1], U_m[2], U_m[3]))
        u_p = state_vec(*conservatives(U_p[0], U_p[1], U_p[2], U_p[3]))
        sigma_r = minmod_slope(roll_r(U), dx)
        U_r_m, U_r_p = intercell_states(roll_r(U), sigma_r, dx)
        u_r_m = state_vec(
            *conservatives(U_r_m[0], U_r_m[1], U_r_m[2], U_r_m[3]))
        F_m_l = flux_vec_prim(U_l_p)
        F_m_r = flux_vec_prim(U_m)
        F_p_l = flux_vec_prim(U_p)
        F_p_r = flux_vec_prim(U_r_m)
        c_s_m = relativistic_sound_velocity((U_l_p + U_m)[0]/2,
                                            (U_l_p + U_m)[1]/2,
                                            (U_l_p + U_m)[3]/2, gamma)
        c_s_p = relativistic_sound_velocity((U_p + U_r_m)[0]/2,
                                            (U_p + U_r_m)[1]/2,
                                            (U_p + U_r_m)[3]/2, gamma)
        a_m_l, a_m_r = signal_velocities(U_l_p[2], U_m[2], c_s_m)
        a_p_l, a_p_r = signal_velocities(U_p[2], U_r_m[2], c_s_p)
    b_m_l, b_m_r = min_insert(a_m_l), max_insert(a_m_r)
    b_p_l, b_p_r = min_insert(a_p_l), max_insert(a_p_r)
    F_m = (b_m_r*F_m_l - b_m_l*F_m_r + b_m_l *
           b_m_r*(u_m - u_l_p))/(b_m_r - b_m_l)
    F_p = (b_p_r*F_p_l - b_p_l*F_p_r + b_p_l *
           b_p_r*(u_r_m - u_p))/(b_p_r - b_p_l)
    F_diff = F_p - F_m
    return F_diff


# Roll array by -1 so that i+1 (the right value) is now at i
# and apply zero-gradient boundary condition
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


# Roll array by +1 so that i-1 (the left value) is now at i
# and apply zero-gradient boundary condition
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


# Set all values of an array greater 0 to 0
def min_insert(arr):
    mask = (arr > 0)
    arr[mask] = 0
    return arr


# Set all values of an array smaller 0 to 0
def max_insert(arr):
    mask = (arr < 0)
    arr[mask] = 0
    return arr


# Round to given number of significant digits
def round_significant(x, sig):
    result = np.round(x, sig - int(np.floor(np.log10(np.abs(x)))) - 1)
    return result


# Get number of significant digits
def significant_digits(x):
    result = - int(np.floor(np.log10(np.abs(x)))) - 1
    return result
