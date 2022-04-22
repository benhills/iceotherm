#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2022 bhills <benjaminhhills@gmail.com>
# Distributed under terms of the GNU GPL3.0 license.

"""
Author:
Benjamin Hills
University of Washington
Earth and Space Sciences
April 22, 2022
"""

import numpy as np
import unittest

from iceotherm.lib.constants import constants
const = constants()
from iceotherm.lib.simple_conduction import *

class TestConductionSolutions(unittest.TestCase):

    def test_erf(self):
        xs = np.linspace(0,1,100)
        Ts = erf_solution(1.,xs,1.,const)
        self.assertTrue(np.all(Ts>=0.))
        self.assertTrue(np.all(Ts<=1.))

    def test_harmonic(self):
        xs = np.linspace(0,1,100)
        Ts = harmonic_surface(0.,1.,0.,xs)
        self.assertTrue(np.all(Ts>=-1.))
        self.assertTrue(np.all(Ts<=1.))

        Ts = harmonic_advection(0.,1.,0.,xs,vel=.1/const.spy)
        self.assertTrue(np.all(Ts>=-1.))
        self.assertTrue(np.all(Ts<=1.))

    def test_plates(self):
        l = 1.
        Ts = parallel_plates(1.,1.,.5*l,l)
        self.assertTrue(np.all(Ts>=0.))
        self.assertTrue(np.all(Ts<=1.))

    def test_inst_source(self):
        Q,t,x,y,z = 1.,1.,1.,1.,1.
        T = inst_source(Q,t,x,y,z,dim=3)
        self.assertTrue(T>=0.)
        T = inst_source(Q,t,x,y,z,dim=2)
        self.assertTrue(T>=0.)
        T = inst_source(Q,t,x,y,z,dim=1)
        self.assertTrue(T>=0.)

if __name__ == '__main__':
    unittest.main()
