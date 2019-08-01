#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:39:42 2019

@author: benhills
"""

import numpy as np
from constants_stefan import constantsIceDiver
const = constantsIceDiver()

# industrial solvents handbook, percent by mass
#Tfd = np.load('ethanol_freezingdepression_PBM.npy')
# linear interpolation between points
#from scipy.interpolate import interp1d
#Tfd_interp = interp1d(Tfd[0], Tfd[1])

def rhoe(T,A=99.39,B=0.31,C=513.18,D=0.305):
    T = T+const.Tf0
    return A/(B**(1+(1-T/C)**D))

def pbmPBV(pbv,T):
    """
    Percent by Volume to Percent by Mass
    """
    # calculate the density of the solution
    rho_s = const.rhow*(1.-pbv)+rhoe(T)*pbv
    # calculate the percent by mass of the solution
    pbm = pbv*(rhoe(T)/rho_s)
    return pbm


def Tf_depression(C,const=const):
    """
    Freezing point depression
    """
    # calculate the density of the solution
    rhos = C + const.rhow*(1.-C/const.rhoe)
    # calculate the molality of the solution (mole/kg)
    molality = 1000.*C/(const.mmass_e*(rhos-C))
    # return the freezing point depression
    Tf = molality*const.Kf
    return Tf

def Hmix(C,const=const):
    """
    # Enthalpy of mixing
    """
    # calculate the density of the solution
    rhos = C + const.rhow*(1.-C/const.rhoe)
    # mole fraction
    hold = C*const.mmass_w/(const.mmass_e*rhos)
    Xe = hold/(1-C/rhos+hold)
    Xw = 1.-Xe
    return 1000.*(-10.6*Xw**6.*Xe-1.2*Xw*Xe+.1*Xw*Xe**2.)

def thermalSink(C_inject,C_init,dt,R):
    """
    # Energy source based on enthalpy of mixing
    # TODO: more robust checks on this
    """
    # enthalpy of mixing
    dHmix = Hmix(C_inject)-Hmix(C_init)
    # energy source (J m-3 s-1)
    phi = dHmix*1000.*C_inject/(dt*const.mmass_e)
    # density and heat capacity of the solution
    rhos = C_inject + const.rhow*(1.-C_inject/const.rhoe)
    cs = C_inject + const.cw*(1.-C_inject/const.ce)
    # convert energy source to temperature change
    dTmix = phi/(2.*np.pi*R**2.*rhos*cs)
    return dTmix

def molDiff(T,b=6,const=const):
    """ Stokes-Einstein relation
    Viscosity approximated from  Khattab et al. 2012,
    but this is still for warm temperatures (~293 K)"""
    #eta_w =
    #eta_e =
    #eta = eta_w*(1.-C) + eta_e*C
    eta = 3e-3
    r = .22e-9
    return const.kBoltz*T/(b*r*np.pi*eta)


