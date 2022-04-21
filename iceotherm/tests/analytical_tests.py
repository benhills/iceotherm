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

from iceotherm.lib.constants import constants
const = constants()
from iceotherm.lib.analytical_solutions import *
from iceotherm.lib.numerical_model import ice_temperature

class TestAnalyticalSolutions(unittest.TestCase):

    def test_robin(self):
        Ts = -50.
        qgeo = 0.05
        H = 2000.
        adot = .1
        m = ice_temperature(Ts=Ts,qgeo=qgeo,H=H,adot=adot)
        T,M = Robin_T(m)
        self.assertTrue(np.all(T>-51.))
        self.assertTrue(np.all(T<0.))

        m.qgeo = 0.2
        T,M = Robin_T(m)
        self.assertTrue(np.all(T>-51.))
        self.assertTrue(np.all(T<1.))

    def test_rezvan(self):
        Ts = -50.
        qgeo = 0.05
        H = 2000.
        adot = .1
        m = ice_temperature(Ts=Ts,qgeo=qgeo,H=H,adot=adot)
        T = Rezvan_T(m)
        self.assertTrue(np.all(T>-51.))
        self.assertTrue(np.all(T<0.))

    def test_meyer(self):
        Ts = -50.
        H = 1000.
        adot = .1
        eps_xy = 0.01
        m = ice_temperature(Ts=Ts,H=H,adot=adot,eps_xy=eps_xy)
        T = Meyer_T(m)
        self.assertTrue(np.all(T>-51.))
        self.assertTrue(np.all(T<=0.))

        m.eps_xy = 0.
        T = Meyer_T(m)
        self.assertTrue(np.all(T>-51.))
        self.assertTrue(np.all(T<=0.))

    def test_perol(self):
        Ts = -50.
        H = 1000.
        adot = .1
        eps_xy = 0.01
        m = ice_temperature(Ts=Ts,H=H,adot=adot,eps_xy=eps_xy)
        T = Perol_T(m)
        self.assertTrue(np.all(T>-51.))
        self.assertTrue(np.all(T<=50.))

        m.eps_xy = 0.
        T = Perol_T(m)
        self.assertTrue(np.all(T>-51.))
        self.assertTrue(np.all(T<=0.))

if __name__ == '__main__':
    unittest.main()
