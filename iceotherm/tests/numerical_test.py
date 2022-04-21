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
from iceotherm.lib.ice_properties import *
from iceotherm.lib.numerical_model import ice_temperature


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
        m.numerical_to_steady_state()

    def test_run(self):
        ntsteps = 1000
        adot = .1
        Ts = -25.
        H = 1000.
        m = ice_temperature(Ts=Ts,adot=adot,H=H)
        m.ts = np.arange(ntsteps)*const.spy
        m.Ts_s = [m.Ts]*ntsteps
        m.adot_s = [m.adot]*ntsteps
        m.Hs = np.linspace(H,H+10,ntsteps)
        m.initial_conditions()
        m.source_terms()
        m.stencil()
        m.flags.append('verbose')
        m.flags.append('save_all')
        m.flags.append('temp-dependent')
        m.k = conductivity(m.T.copy(),m.rho)
        m.Cp = heat_capacity(m.T.copy())
        m.numerical_transient()

    def test_melt(self):
        m = ice_temperature(Ts=-25.,adot=.05,H=500.,qgeo=.1)
        ntsteps = 100
        m.ts = np.arange(ntsteps)*const.spy
        m.Ts_s = [m.Ts]*ntsteps
        m.adot_s = [m.adot]*ntsteps
        m.flags.append('water_cum')
        m.Mcum_max = 1e-6
        m.initial_conditions()
        m.source_terms()
        m.stencil()
        m.numerical_transient()

    def test_freeze(self):
        m = ice_temperature(Ts=-25.,adot=.2,H=1000.,qgeo=.01)
        ntsteps = 100
        m.ts = np.arange(ntsteps)*const.spy
        m.Ts_s = [m.Ts]*ntsteps
        m.adot_s = [m.adot]*ntsteps
        m.flags.append('water_cum')
        m.Mcum = 1e-4
        m.initial_conditions()
        m.source_terms()
        m.stencil()
        m.numerical_transient()

if __name__ == '__main__':
    unittest.main()
