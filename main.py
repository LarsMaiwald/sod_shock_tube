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
N_x, x_min, x_max, t_final = (cfg['N_x'], cfg['x_min'], cfg['x_max'],
                              cfg['t_final'])
