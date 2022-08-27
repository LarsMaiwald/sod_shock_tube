#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 14:58:58 2022

@author: lars
"""

# Project: Sod shock tube
# by Lars Maiwald

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os

# Importing own functions
from functions import (ideal_gas_EoS_eps, conservatives, state_vec,
                       relativistic_sound_velocity, cfl_condition, updating,
                       state_vec_decomp, primitives, get_f_prime,
                       round_significant, significant_digits)

# Creating output directories if not yet existent
os.system('mkdir -p plots')
os.system('mkdir -p animations')

# Clearing output directories 'plots' and 'animations'
os.system('rm plots/*.png')
os.system('rm animations/*.mp4')

# Setup
# Loading the configuration
cfg = yaml.safe_load(open('config.yml'))
N_xcells, t_final, c_cfl = cfg['N_xcells'], cfg['t_final'], cfg['c_cfl']
gamma, L, tol, save_step = cfg['gamma'], cfg['L'], cfg['tol'], cfg['save_step']
newton_max_it, hor = cfg['newton_max_it'], cfg['hor']
v0_l, v0_r = cfg['v0_l'], cfg['v0_r']
p0_l, p0_r = cfg['p0_l'], cfg['p0_r']
rho0_l, rho0_r = cfg['rho0_l'], cfg['rho0_r']

# Computing the analytical derivative f_prime
f_prime_command = get_f_prime()

# Computing constants
dx = L/N_xcells
half = int(N_xcells/2)
sig_digits = significant_digits(dx)

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
dt = round_significant(cfl_condition(v, c_s, dx, c_cfl), sig_digits)
t_len = int(1 - np.floor(np.log10(dt)))

# Time evolution
t = 0
k = 0
k_in = 0
replace_list = [['$', ''], ['\\', '']]
print('Starting simulation…')
while t < t_final:
    # Plotting
    if k == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
    if k % save_step == 0:
        k_in += 1
        for var, name in [[p, r'$p$'], [rho, r'$\rho$'], [v, r'$v$'],
                          [eps, r'$\epsilon$']]:
            ax.clear()
            ax.plot(x, var)
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(name)
            ax.set_xlim(-L/2, L/2)
            ax.text(1.0, 1.05, f't = {t:.{t_len}f}',
                    horizontalalignment='right', verticalalignment='center',
                    transform=ax.transAxes)
            fig.tight_layout()
            filename = name
            for i, o in replace_list:
                filename = filename.replace(i, o)
            fig.savefig(f'plots/{filename}_{k_in}.png', dpi=200)
        ax.clear()
        ax.set_xlabel(r'$x$')
        ax.set_xlim(-L/2, L/2)
        for var, name in [[p, r'$p$'], [rho, r'$\rho$'], [v, r'$v$'],
                          [eps, r'$\epsilon$']]:
            ax.plot(x, var, label=name)
        ax.text(1.0, 1.05, f't = {t:.{t_len}f}',
                horizontalalignment='right', verticalalignment='center',
                transform=ax.transAxes)
        fig.tight_layout()
        ax.legend(loc=(0.86, 0.665))
        fig.savefig(f'plots/all_{k_in}.png', dpi=200)
    # Simulation
    print(
        f'Time evolution: iteration = {k}, time = {t:.{t_len}f} of {t_final}')
    u = updating(p, u, dt, dx, gamma, f_prime_command, tol, newton_max_it, hor)
    D, S, tau = state_vec_decomp(u)
    p, rho, v, eps = primitives(
        p, u, gamma, f_prime_command, tol, newton_max_it)
    t += dt
    k += 1
print('Simulation done.')
print('Plots saved to folder \'plots\'.')

# Exporting
for var, name in [[p, 'p'], [rho, 'rho'], [v, 'v'],
                  [eps, 'epsilon']]:
    np.savetxt('comparison/' + name + '.csv', var, delimiter=',')
with open('comparison/t.txt', 'w') as f:
    f.write(f'{t:.{t_len}f}')
print('Final state saved to folder \'comparison\'.')

# Animations
print('Rendering animations…')
for name in ['p', 'rho', 'v', 'epsilon', 'all']:
    command_string = (f'ffmpeg -r {int(1/(20*dt*save_step))} -f image2 ' +
                      f'-s 1200x800 -i plots/{name}_%d.png -vcodec libx264 ' +
                      f'-crf 25 -pix_fmt yuv420p animations/{name}.mp4')
    os.system(command_string)
print('Rendering done.')
print('Animations saved to folder \'animations\'.')
