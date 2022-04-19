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
from ice_properties import *
from analytical_solutions import Robin_T
from numerical_model import ice_temperature

class TestIceProperties(unittest.TestCase):

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

    def test_rate_factor(self):
        Ts = -50.
        qgeo = 0.05
        H = 2000.
        adot = .1
        v_surf = 10./const.spy
        m = ice_temperature(Ts=Ts,qgeo=qgeo,H=H,adot=adot)
        T,M = Robin_T(m)

        A = rate_factor(T,const=const)
        self.assertTrue(A[0]>1e-28)
        self.assertTrue(A[0]<1e-23)

        tau_xz = const.rho*const.g*(H-m.z)*abs(0.03)
        A = rate_factor(T,z=m.z,H=H,const=const,tau_xz=tau_xz,v_surf=v_surf)
        eps_xz = A*tau_xz**const.n
        Q = 2.*(eps_xz*tau_xz)
        Q_opt = np.trapz(Q,m.z)
        print(abs((Q_opt-v_surf*tau_xz[0])/const.Astar))
        self.assertTrue(abs((Q_opt-v_surf*tau_xz[0])) < 1e-5)

if __name__ == '__main__':
    unittest.main()
