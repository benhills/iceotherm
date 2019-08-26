#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 17:15:22 2018

@author: benhills
"""

class constantsUniversal(object):
    """
    Universal Constants
    """
    def __init__(self):
        self.g = 9.81                           # Gravity m s-2
        self.spy  = 60.*60.*24.*365.24          # sec yr-1
        self.spd  = 60.*60.*24.                 # sec day-1
        self.rhow = 1000.                       # Density of Water kg m-3
        self.T0 = 273.15                        # Reference Tempearature, triple point for water, K
        self.R = 8.321                          # Gas Constant J mol-1 K-1
        self.kBoltz = 1.38064852e-23            # m2 kg s-2 K-1
        self.c = 3e8                            # Speed of Light in Free Space m s-1

class constantsTempCuffPat(object):
    """
    Temperature Constants

    Cuffey and Paterson (2010)
    """
    def __init__(self):
        # general
        self.spy  = 60.*60.*24.*365.24          # sec yr-1
        self.g = 9.81                           # Gravity m s-2
        self.T0 = 273.15                        # Reference Tempearature, triple point for water, K
        self.R = 8.321                          # Gas Constant J mol-1 K-1
        self.rhow = 1000.                       # Density of water kg m-3
        # CP (2010) pg. 72
        self.n = 3.                             # Creep Exponent
        self.Tstar = 263.                       # Reference Temperature K
        self.Qminus = 6.0e4                     # Activation Energy <10C J mol-1
        self.Qplus = 11.5e4                     # Activation Energy >10C J mol-1
        self.Astar = 3.5e-25                    # Creep Parameter Pa-3 s-1
        # CP (2010) pg. 12
        self.rho = 917.                         # Ice Density kg m-3
        # CP (2010) pg. 400
        self.Cp = 2097.                         # Specific Heat Capacity J kg-1 K-1
        self.L = 3.335e5                        # Latent Heat of Fusion J kg-1
        self.k = 2.1                            # Thermal Conductivity J m-1 K-1 s-1
        self.K = 1.09e-6                        # Thermal Diffusivity m2 s-1
        # CP (2010) pg. 406
        self.beta = -7.42e-8                    # Clausius-Clapeyron K Pa-1

class constantsTempVanDerVeen(object):
    """
    Temperature Constants

    van der Veen (2013)
    """
    def __init__(self):
        # VDV (2013) pg. 33
        self.A0 = 9.302e7                      # kPa-3 yr-1
        self.Q = 7.88e4                         # J mol-1
        self.C = 0.16612                        # K^k
        self.Tr = 273.39                        # Reference Temperature, K
        self.Kk = 1.17
        self.B0 = 2.207e-3                      # kPa yr1/3
        self.T0 = 3155.                         # K
        # VDV (2013) pg 144
        self.Cp = 2097.                         # Specific Heat Capacity J kg-1 K-1
        self.L = 3.335e5                        # Latent Heat of Fusion J kg-1
        self.k = 2.1                            # Thermal Conductivity J m-1 K-1 s-1
        self.K = 1.09e-6                        # Thermal Diffusivity m2 s-1
        self.rho = 917.                         # Bulk Density kg m-3
        # VDV (2013) pg 209
        self.Cw = 4.217                         # Specific Heat Capacity of Water J kg-1 K-1
        self.Ct = -7.4e-8                       # Pressure-Dependence of Melting Temp K Pa-1
        # General
        self.R = 8.321                          # Gas Constant J mol-1 K-1


### Constants
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
        self.rhow = 1000.                       # Density of water kg m-3
        self.kBoltz = 1.38064852e-23            # m2 kg s-2 K-1
        # CP (2010) pg. 12
        self.rhoi = 917.                         # Ice Density kg m-3
        # CP (2010) pg. 400
        self.ki = 2.1                            # Thermal Conductivity J m-1 K-1 s-1
        self.ci = 2097.                         # Specific Heat Capacity J kg-1 K-1
        self.alphai = self.ki/(self.rhoi*self.ci)
        self.L = 3.335e5                        # Latent Heat of Fusion J kg-1
        # others
        self.rhoe = 800                         # approximate density of ethanol at 0degC (Engineering Toolbox)
        self.mmass_e = 46.07                    # molar mass of ethanol (g/mol)
        self.ke = 0.19                         # thermal conductivity of ethanol
        self.etae = 2.5e-3                      # dynamic viscosity of ethanol (Pa s) (Engineering Toolbox)
        self.mmass_w = 18.02                      # molar mass of water
        self.etaw = 1.5e-3                      # dynamic viscosity of water (Pa s) (Engineering Toolbox)
        self.Kf = -1.99
        self.ce = 2175.                         # heat capacity of ethanol
        self.cw = 4212.                   # heat capacity of water
        self.kw = 0.555                         # thermal conductivity of water
        self.alphaw = self.kw/(self.rhow*self.cw)
        self.mol_diff = 1.24e-9                  # molecular diffusivity of aqueous ethanol at 25degC (Cussler (1997))
        # random
        self.tol = 1e-5                              # tolerance for numerics

### Constants
class constantsWorster(object):
    """
    Temperature Constants

    From Worster Solidification of Fluids
    """
    def __init__(self):
        # general
        self.rhow = 1000.                       # Density of water kg m-3
        """
        # CP (2010) pg. 12
        self.rhoi = 917.                         # Ice Density kg m-3
        # CP (2010) pg. 400
        self.ki = 2.1                            # Thermal Conductivity J m-1 K-1 s-1
        self.ci = 2097.                         # Specific Heat Capacity J kg-1 K-1
        self.alphai = self.ki/(self.rhoi*self.ci)
        self.L = 3.335e5                        # Latent Heat of Fusion J kg-1
        """
        # worster
        self.Tinf = 20.
        self.eps = 0.05
        self.LCp = 80.
        # others
        self.rhoe = 800                         # approximate density of ethanol at 0degC (Engineering Toolbox)
        self.mmass_e = 46.07                    # molar mass of ethanol
        self.mmass_w = 18.02                      # molar mass of water
        """
        self.ke = 0.167                         # thermal conductivity of ethanol
        self.etae = 2.5e-3                      # dynamic viscosity of ethanol (Pa s) (Engineering Toolbox)
        self.etaw = 1.5e-3                      # dynamic viscosity of water (Pa s) (Engineering Toolbox)
        self.ce = 2460.                         # heat capacity of ethanol
        self.cw = 4212.                   # heat capacity of water
        self.kw = 0.555                         # thermal conductivity of water
        self.alphaw = self.kw/(self.rhow*self.cw)
        self.mol_diff = 1.24e-9                  # molecular diffusivity of aqueous ethanol at 25degC (Cussler (1997))
        """


        self.Kf = -.5

