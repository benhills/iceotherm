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

import unittest

from iceotherm.lib.cylindricalstefan.double_diffusion_model import double_diffusion_model

class TestDoubleDiffusion(unittest.TestCase):

    def test_double_diffusion(self):
        # Import model
        mod = double_diffusion_model()

        # Source timing
        mod.source_timing = 3000.
        mod.source_duration = 500.

        # Flags for solution solves
        mod.flags.append('solve_sol_temp')
        mod.flags.append('solve_sol_mol')

        # Model setup
        mod.log_transform()
        mod.get_domain()
        mod.get_initial_conditions()
        mod.save_times = mod.ts[::25]
        mod.get_boundary_conditions()

        # Model run
        mod.run()

if __name__ == '__main__':
    unittest.main()
