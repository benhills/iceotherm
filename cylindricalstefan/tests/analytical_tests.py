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

from cylindricalstefan.lib.analytical_pure_solution import *
from cylindricalstefan.lib.analytical_binary_solution import *

class TestAnalyticalFunctions(unittest.TestCase):

    def test_analytical_melt_center(self):
        rs = np.linspace(0.001,1,100)
        T_inf = -1.
        Q_melt = 1000.
        Tf = 0.
        R_target = 0.1
        t_target = 100.
        
        T,lam,R_melt,t_melt = analyticalMelt(rs,T_inf,Q_melt,Tf,
                                             R_target=R_target,target='Dist',
                                             fluxLoc='Center')
        self.assertAlmostEqual(R_target,R_melt)
        self.assertTrue(np.all(T>=T_inf))
        self.assertTrue(np.all(T[rs>R_melt]<=Tf))

        T,lam,R_melt,t_melt = analyticalMelt(rs,T_inf,Q_melt,Tf,
                                             t_target=t_target,target='Time',
                                             fluxLoc='Center')
        self.assertAlmostEqual(t_target,t_melt)
        self.assertTrue(np.all(T>=T_inf))
        self.assertTrue(np.all(T[rs>R_melt]<=Tf))
        
    def test_analytical_melt_wall(self):
        rs = np.linspace(0.001,1,100)
        T_inf = -1.
        Q_melt = 1000.
        Tf = 0.
        R_target = 0.1
        t_target = 100.
        
        T,lam,R_melt,t_melt = analyticalMelt(rs,T_inf,Q_melt,Tf,
                                             R_target=R_target,target='Dist',
                                             fluxLoc='Wall')
        self.assertAlmostEqual(R_target,R_melt)
        self.assertTrue(np.all(T>=T_inf))
        self.assertTrue(np.all(T[rs>R_melt]<=Tf))

        T,lam,R_melt,t_melt = analyticalMelt(rs,T_inf,Q_melt,Tf,
                                             t_target=t_target,target='Time',
                                             fluxLoc='Wall')
        self.assertAlmostEqual(t_target,t_melt)
        self.assertTrue(np.all(T>=T_inf))
        self.assertTrue(np.all(T[rs>R_melt]<=Tf))
        
    def test_analytical_freeze(self):
        r0 = .1
        T_inf = -1.
        Q_sol = 0.
        Tf = 0.
        
        T,R,rs,ts = analyticalFreeze(r0,T_inf,Q_sol,Tf=Tf)
        self.assertTrue(np.all(T>=T_inf))
        self.assertTrue(np.all(T<=Tf))
        self.assertTrue(np.all(np.diff(R)<0.))
        

    #def test_binary_worster(self):
        
    #def test_binary_cylindrical(self):
        
if __name__ == '__main__':
    unittest.main()