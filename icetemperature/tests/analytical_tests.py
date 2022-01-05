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
from analytical_solutions import *

class TestAnalyticalSolutions(unittest.TestCase):

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

    def test_rezvan(self):
        Ts = -50.
        qgeo = 0.05
        H = 2000.
        adot = .1
        z,T = Rezvan_T(Ts,qgeo,H,adot,verbose=True)
        self.assertTrue(np.all(T>-51.))
        self.assertTrue(np.all(T<0.))

    def test_meyer(self):
        Ts = -50.
        H = 1000.
        adot = .1
        eps_xy = 3e-09
        z,T = Meyer_T(Ts,H,adot,eps_xy,verbose=True)
        self.assertTrue(np.all(T>-51.))
        self.assertTrue(np.all(T<=0.))

        z,T = Meyer_T(Ts,H,adot,0.,verbose=True)
        self.assertTrue(np.all(T>-51.))
        self.assertTrue(np.all(T<=0.))

    def test_perol(self):
        Ts = -50.
        H = 1000.
        adot = .1
        eps_xy = 3e-09
        z,T = Perol_T(Ts,H,adot,eps_xy,verbose=True)
        self.assertTrue(np.all(T>-51.))
        self.assertTrue(np.all(T<=50.))

        z,T = Perol_T(Ts,H,adot,0.,verbose=True)
        self.assertTrue(np.all(T>-51.))
        self.assertTrue(np.all(T<=0.))

if __name__ == '__main__':
    unittest.main()
