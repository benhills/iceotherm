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
from iceotherm.lib.cylindricalstefan import double_diffusion_model as ddiff

class TestDoubleDiffusion(unittest.TestCase):

    def test_instantiation(self):
        m = ddiff.model()

    def test_log_transform(self):
        m = ddiff.model()
        m.log_transform()

    @unittest.skipIf(not ddiff.fe_enabled, 'No dolfin')
    def test_domain(self):
        m = ddiff.model()
        m.flags.append('solve_sol_temp')
        m.flags.append('solve_sol_mol')
        m.get_domain()

    @unittest.skipIf(not ddiff.fe_enabled, 'No dolfin')
    def test_initialization(self):
        m = ddiff.model()
        m.source_timing = 3000.
        m.source_duration = 500.
        m.get_initial_conditions()

    @unittest.skipIf(not ddiff.fe_enabled, 'No dolfin')
    def test_bcs(self):
        m = ddiff.model()
        m.get_boundary_conditions()

    @unittest.skipIf(not ddiff.fe_enabled, 'No dolfin')
    def test_run(self):
        m = ddiff.model()
        m.save_times = m.ts[::25]
        m.run()

if __name__ == '__main__':
    unittest.main()
