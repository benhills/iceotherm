#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright © 2020 bhills <benjaminhhills@gmail.com>
# Distributed under terms of the GNU GPL3.0 license.

"""
Author:
Benjamin Hills
University of Washington
Earth and Space Sciences
September 9, 2019
"""

import numpy as np
import unittest

from iceotherm.lib.cylindricalstefan import instantaneous_mixing_model as inst
from iceotherm.lib.cylindricalstefan import thermal_2d_model as th2d

class TestNumericalModels(unittest.TestCase):

    def test_instantaneous_model(self):
        m = inst.model()

    @unittest.skipIf(not inst.fe_enabled, 'No dolfin')
    def test_instantaneous_melt(self):
        # Import model
        m = inst.model()

        # Model setup
        m.log_transform()
        m.get_domain()
        m.get_initial_conditions()
        m.save_times = mod.ts[::5]
        m.get_boundary_conditions()

        # Model run
        m.run()

    @unittest.skipIf(not inst.fe_enabled, 'No dolfin')
    def test_instantaneous_freeze(self):
        # Import the model
        m = inst.model()

        # Model setup
        m.log_transform()
        m.get_domain()
        m.get_initial_conditions()
        m.u0_i.vector()[:] = -1.
        m.save_times = mod.ts[::50]
        m.get_boundary_conditions()

        # Model run
        m.run()

    def test_2d_instantiation(self):
        m = th2d.model()

    @unittest.skipIf(not th2d.fe_enabled, 'No dolfin')
    def test_2d(self):
        # Import model
        m = th2d.model()

        # Model setup
        m.log_transform()
        m.get_domain()
        m.get_initial_conditions(melt_velocity=0.)
        m.get_boundary_conditions(no_bottom_bc=True)
        ts_out = mod.ts[::10]
        z_out = 4.
        ws_out = np.linspace(mod.w0,mod.wf,100)

        # Model run
        m.run(ts_out,ws_out,z_out,initialize_array=True)

if __name__ == '__main__':
    unittest.main()
