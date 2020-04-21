#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 17:15:22 2018

@author: benhills
"""

class constantsIceDiver(object):
    """
    Temperature Constants

    Used for Ice Diver modeling
    """
    def __init__(self):
        # general
        self.spy  = 60.*60.*24.*365.24          # sec yr-1
        self.g = 9.81                           # Gravity m s-2
        self.Tf0 = 273.15                        # Reference Tempearature, triple point for water, K
        self.kBoltz = 1.38064852e-23            # m2 kg s-2 K-1
        self.tol = 1e-5                              # tolerance for numerics
        # ice
        self.rhoi = 917.                         # Ice Density kg m-3; Cuffey and Paterson (2010) pg. 12
        self.ki = 2.1                            # Thermal Conductivity J m-1 K-1 s-1; Cuffey and Paterson (2010) pg. 12
        self.ci = 2097.                         # Specific Heat Capacity J kg-1 K-1; Cuffey and Paterson (2010) pg. 12
        self.L = 3.335e5                        # Latent Heat of Fusion J kg-1; Cuffey and Paterson (2010) pg. 12
        self.alphai = self.ki/(self.rhoi*self.ci)
        # ethanol
        self.rhoe = 800                         # approximate density of ethanol at 0degC (Engineering Toolbox)
        self.ke = 0.167                         # thermal conductivity of ethanol at 25degC (Engineerign Toolbox)
        self.ce = 2270.                         # heat capacity of ethanol at 0degC (Engineering Toolbox)
        self.eta_e = 1.786e-3                      # dynamic viscosity of ethanol (Pa s) at 0degC (Engineering Toolbox)
        self.mmass_e = 46.07                    # molar mass of ethanol (g/mol)
        self.mol_diff_e = .84e-9                  # molecular diffusivity of aqueous ethanol at 25degC (Cussler (1997))
        self.rad_e = .22-9                  # radius of ethanol molecule
        # methanol
        self.rhom = 810                         # approximate density of ethanol at 0degC (Engineering Toolbox)
        self.km = 0.203                         # thermal conductivity of ethanol at 25degC (Engineering Toolbox)
        self.cm = 2400.                         # heat capacity of ethanol at 0degC (Engineering Toolbox)
        self.eta_m = .796e-3                      # dynamic viscosity of ethanol at 0degC (Pa s) (Engineering Toolbox)
        self.mmass_m = 32.04                    # molar mass of ethanol (g/mol)
        self.mol_diff_m = .84e-9                  # molecular diffusivity of aqueous ethanol at 25degC (Cussler (1997))
        self.rad_m = .21-9                  # radius of methanol molecule
        # water
        self.rhow = 1000.                       # Density of water kg m-3
        self.kw = 0.555                         # thermal conductivity of water
        self.cw = 4212.                   # heat capacity of water
        self.etaw = 1.5e-3                      # dynamic viscosity of water (Pa s) (Engineering Toolbox)
        self.mmass_w = 18.02                      # molar mass of water
        self.alphaw = self.kw/(self.rhow*self.cw)
