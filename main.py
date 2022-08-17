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


# Functions
def flux_vec(r, u, p, E):
    F = np.array([r*u, r*u**2 + p, u*(E + p)])
    return F


def ideal_gas_EoS_e(r, g, p):
    e = p/(r*(g - 1))
    return e


def ideal_gas_EoS_p(r, g, e):
    p = e*(r*(g - 1))
    return p


def specific_internal_energy(r, u, E):
    e = E/r - 0.5*u**2
    return e


def U_decomp(U):
    r = U[0]
    u = U[1]/r
    E = U[2]
    return r, u, E


def total_energy_per_unit_volume(r, u, p, e, g):
    E = r*(0.5*u**2 - e)
    return E


def updating(U, dt, dx):
    U_old = U.copy()
    F_diff = hlle_flux(U_old)
    U_new = U_old - dt/dx*F_diff
    U_old.delete()
    return U_new

# def hlle_flux(U):


# Setup
# Loading the configuration
cfg = yaml.safe_load(open('config.yml'))
N_cells, t_final, CFL, g = cfg['N_cells'], cfg['t_final'], cfg['CFL'], cfg['g']
x_min, x_max = cfg['x_min'], cfg['x_max']
u0_l, u0_r = cfg['u0_l'], cfg['u0_r']
p0_l, p0_r = cfg['p0_l'], cfg['p0_r']
r0_l, r0_r = cfg['r0_l'], cfg['r0_r']

# Computing constants
dx = (x_max - x_min)/N_cells
N_points = N_cells + 1
half = int(N_points/2)

# Initializing arrays
x = np.arange(x_min, x_max + dx, dx)    # Is that correct?
r0 = np.ones(np.shape(x))
r0[:half] *= r0_l       # There seems to be problems…
r0[half:] *= r0_r
u0 = np.ones(np.shape(x))
u0[:half] *= u0_l
u0[half:] *= u0_r
p0 = np.ones(np.shape(x))
p0[:half] *= p0_l
p0[half:] *= p0_r
E0 = total_energy_per_unit_volume(r0, u0, p0)
U = np.array([r0, r0*u0, E0])       # could be U0 be we want to update it

# Computing the time-step size
c0 = np.sqrt(g*p0/r0)
dt = CFL*dx/np.max(c0 + np.abs(u0))
