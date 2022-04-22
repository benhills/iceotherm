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

from iceotherm.lib.cylindricalstefan.instantaneous_mixing_model import instantaneous_mixing_model
from iceotherm.lib.cylindricalstefan.thermal_model_2d import thermal_model_2d

class TestNumericalModels(unittest.TestCase):

    def test_instantaneous_melt(self):
        # Import model
        mod = instantaneous_mixing_model()

        # Model setup
        mod.log_transform()
        mod.get_domain()
        mod.get_initial_conditions()
        mod.save_times = mod.ts[::5]
        mod.get_boundary_conditions()

        # Model run
        mod.run()

    def test_instantaneous_freeze(self):
        # Import the model
        mod = instantaneous_mixing_model()

        # Model setup
        mod.log_transform()
        mod.get_domain()
        mod.get_initial_conditions()
        mod.u0_i.vector()[:] = -1.
        mod.save_times = mod.ts[::50]
        mod.get_boundary_conditions()

        # Model run
        mod.run()

    def test_2d(self):
        # Import model
        mod = thermal_model_2d()

        # Model setup
        mod.log_transform()
        mod.get_domain()
        mod.get_initial_conditions(melt_velocity=0.)
        mod.get_boundary_conditions(no_bottom_bc=True)
        ts_out = mod.ts[::10]
        z_out = 4.
        ws_out = np.linspace(mod.w0,mod.wf,100)

        # Model run
        mod.run(ts_out,ws_out,z_out,initialize_array=True)

if __name__ == '__main__':
    unittest.main()
