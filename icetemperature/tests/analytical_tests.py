#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2021 bhills <benjaminhhills@gmail.com>
# Distributed under terms of the GNU GPL3.0 license.

"""
Author:
Benjamin Hills
University of Washington
Earth and Space Sciences
July 15, 2021
"""

import numpy as np
import unittest

from constants import constants
const = constants()
from analytical_functions import *

from scipy.optimize import fsolve

class TestAnalyticalFunctions(unittest.TestCase):

    def test_conductivity(self):
        T = -50.
        rho = 800.
        krho = conductivity(T,rho,const)
        self.assertTrue(krho>.8)
        self.assertTrue(krho<3.5)

    def test_heatcapacity(self):
        T = -50.
        cp = heat_capacity(T,const)
        self.assertTrue(cp>1500.)
        self.assertTrue(cp<2100.)

    def test_viscosity(self):
        Ts = -50.
        qgeo = 0.05
        H = 2000.
        adot = .1
        z,T = Robin_T(Ts,qgeo,H,adot,verbose=True)

        A = viscosity(T,z,const)
        self.assertTrue(A[0]>1e-28)
        self.assertTrue(A[0]<1e-23)

        tau_xz = const.rho*const.g*(H-z)*abs(0.03)
        A = viscosity(T,z,const,tau_xz=tau_xz,v_surf=10.)
        self.assertTrue(A[0]>1e-28)
        self.assertTrue(A[0]<1e-23)

    def test_rezvan(self):
        Ts = -50.
        qgeo = 0.05
        H = 2000.
        adot = .1
        z,T = Rezvan_T(Ts,qgeo,H,adot,verbose=True)
        self.assertTrue(np.all(T>-51.))
        self.assertTrue(np.all(T<0.))

    def test_robin(self):
        Ts = -50.
        qgeo = 0.05
        H = 2000.
        adot = .1
        z,T = Robin_T(Ts,qgeo,H,adot,verbose=True)
        self.assertTrue(np.all(T>-51.))
        self.assertTrue(np.all(T<0.))

        qgeo = 0.2
        z,T = Robin_T(Ts,qgeo,H,adot)
        self.assertTrue(np.all(T>-51.))
        self.assertTrue(np.all(T<1.))

if __name__ == '__main__':
    unittest.main()
