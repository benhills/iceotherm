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

from cylindricalstefan.lib.constants import constantsHotPointDrill
const = constantsHotPointDrill()
from cylindricalstefan.lib.concentration_functions import *

class TestConcentrationFunctions(unittest.TestCase):

    def test_run_conversions(self):
        C = 400.
        pbm = .5
        molality = 10.
        C_pbv(C,const.rhoe)
        C_Molality(C,const.rhoe,const.mmass_e)
        C_MoleFrac(C,const.rhoe,const.mmass_e)
        C_pbm(C,const.rhoe)
        pbm_Molality(molality,const.mmass_e)
        pbm_C(pbm,const.rhoe)

    def test_pbm(self):
        pbm = .5
        C = pbm_C(pbm,const.rhoe)
        pbm_ = C_pbm(C,const.rhoe)

        self.assertAlmostEqual(pbm,pbm_)

    def test_molality(self):
        C = 400.
        molality = C_Molality(C,const.rhoe,const.mmass_e)
        pbm = C_pbm(C,const.rhoe)
        molality_ = pbm_Molality(pbm,const.mmass_e)

        self.assertAlmostEqual(molality,molality_)

    def test_molecular_diffusivity(self):
        T = -10.
        C = 400.
        Xe = C_MoleFrac(C,const.rhoe,const.mmass_e)
        eta_s = etaKhattab(Xe,T)
        D = molDiff(T,eta_s)

        self.assertLess(D,1e-7)
        self.assertGreater(D,1e-15)

    def test_freezing_depression(self):
        C = 400.

        Tf_e = Tf_depression(C,solute='ethanol')
        self.assertLess(Tf_e,0.)

        Tf_m = Tf_depression(C)
        self.assertLess(Tf_m,0.)

        self.assertNotEqual(Tf_e,Tf_m)

    def test_enthalpy_of_mixing(self):
        C = 400.

        H_e,phi_e = Hmix(C,solute='ethanol')
        self.assertLess(H_e,0.)

        H_m,phi_m = Hmix(C)
        self.assertLess(H_m,0.)

        self.assertNotEqual(H_e,H_m)


if __name__ == '__main__':
    unittest.main()
