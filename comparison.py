#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 13:08:05 2022

@author: lars
"""

# Imorting libraries
import numpy as np
import matplotlib.pyplot as plt
import sys

# Importing own functions
from functions import L2_norm

# Time of data
t = np.genfromtxt('comparison/t.txt')

# Loading data from our RHLLE implementation
p = np.genfromtxt('comparison/p.csv', delimiter=',')
rho = np.genfromtxt('comparison/rho.csv', delimiter=',')
v = np.genfromtxt('comparison/v.csv', delimiter=',')
eps = np.genfromtxt('comparison/epsilon.csv', delimiter=',')

# Loading data of exact solution from RIEMANN.f
solution = np.genfromtxt('comparison/solution.dat',
                         delimiter='  ', skip_header=1)
x = solution[:, 0] - 0.5
p_exact = solution[:, 1]
rho_exact = solution[:, 2]
v_exact = solution[:, 3]
eps_exact = solution[:, 4]

# Check if times match
solution_header = np.genfromtxt('comparison/solution.dat',
                                delimiter='    ', skip_footer=400)
t_solution = solution_header[1]
if np.round(t, 5) != np.round(t_solution, 5):
    print(f'Time of numerical solution (t = {t}) and time of exact solution ' +
          f'(t = {t_solution}) do not match.')
    print('Please generate new file \'solution.dat\' using RIEMANN.f ' +
          f'at time t = {t}.')
    sys.exit(1)

# Plotting and error calculation
replace_list = [['$', ''], ['\\', '']]
fig, ax = plt.subplots(figsize=(6, 4))
for var1, var2, name in [[p_exact, p, r'$p$'],
                         [rho_exact, rho, r'$\rho$'], [v_exact, v, r'$v$'],
                         [eps_exact, eps, r'$\epsilon$']]:
    ax.clear()
    ax.plot(x, var1, label='exact', linestyle='dotted', color='black')
    ax.plot(x, var2, label='RHLLE')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(name)
    ax.set_xlim(-0.5, 0.5)
    ax.text(1.0, 1.05, f't = {t}', horizontalalignment='right',
            verticalalignment='center', transform=ax.transAxes)
    fig.legend(loc=(0, 0), frameon=False)
    fig.tight_layout()
    filename = name
    for i, o in replace_list:
        filename = filename.replace(i, o)
    fig.savefig(f'comparison/{filename}.png', dpi=200)
    print(f'{filename}: L2_norm: {L2_norm(var1 - var2)}')
print('Plots saved to folder \'comparison\'.')
