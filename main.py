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


# Setup
# Loading the configuration
cfg = yaml.safe_load(open('config.yml'))
N_x, t_final, CFL, g = cfg['N_x'], cfg['t_final'], cfg['CFL'], cfg['g']
x_min, x_max = cfg['x_min'], cfg['x_max']
u0_l, u0_r = cfg['u0_l'], cfg['u0_r']
p0_l, p0_r = cfg['p0_l'], cfg['p0_r']
rho0_l, rho0_r = cfg['r0_l'], cfg['r0_r']
