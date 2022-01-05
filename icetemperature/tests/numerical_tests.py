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
from model import ice_temperature

class TestNumericalFunctions(unittest.TestCase):

    def test_instantiation(self):
        m = ice_temperature()

    def test_initialization(self):
        m = ice_temperature()
        m.initial_conditions()

    def test_sourceterms(self):
        m = ice_temperature()
        m.initial_conditions()
        m.source_terms()

    def test_stencil(self):
        m = ice_temperature()
        m.initial_conditions()
        m.source_terms()
        m.stencil(dt=1.*const.spy)

    def test_steadystate(self):
        m = ice_temperature()
        m.initial_conditions()
        m.source_terms()
        m.stencil(dt=1.*const.spy)
        m.run_to_steady_state()

    def test_run(self):
        m = ice_temperature()
        ntsteps = 100
        adot = .1
        Ts = -25.
        H = 1000.
        m.adot = np.array([adot/const.spy]*ntsteps)
        m.Ts = np.array([Ts]*ntsteps)
        m.H = H
        m.ts = np.arange(ntsteps)*const.spy
        m.initial_conditions()
        m.source_terms()
        m.stencil()
        m.run()

    def test_temp_dependency(self):
        m = ice_temperature()
        m.flags.append('temp-dependent')
        m.initial_conditions()
        m.k = conductivity(m.T.copy(),m.rho)
        m.Cp = heat_capacity(m.T.copy())

if __name__ == '__main__':
    unittest.main()
