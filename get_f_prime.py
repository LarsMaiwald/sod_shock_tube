#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 17:17:58 2022

@author: lars
"""
from sympy import *
import mpmath


def get_f_prime():
    p, rho, v, eps = symbols('p, rho, v, eps')
    D, S, tau = symbols('D, S, tau')
    W = Symbol('W')
    gamma = Symbol('gamma')
    f = Function('f')(p)
    v = S/(tau + D + p)
    W = 1/sqrt(1 - v**2)
    rho = D/W
    eps = (tau - D*W + p*(1 - W**2) + D)/(D*W)
    f = rho*eps*(gamma - 1) - p
    f_prime = f.diff(p)
    return str(f_prime).replace('sqrt', 'np.sqrt')
