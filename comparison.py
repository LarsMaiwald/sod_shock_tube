#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 13:08:05 2022

@author: lars
"""

# Imorting libraries
import numpy as np
import matplotlib.pyplot as plt

# Time of data
t = 0.55013

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

# Plotting
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
print('Plots saved to folder \'comparison\'.')
