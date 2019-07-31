#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 08:58:51 2019

@author: benhills
"""

import numpy as np
from constants_stefan import constantsIceDiver
const = constantsIceDiver()

# -------------------------------------------------------------------------------------------------------------------------------

from scipy.special import expi
from scipy.optimize import fsolve

def analyticalMelt(rs,Tinf,Qmelt,Tm=0,t_target=0,R_target=0,target='Dist',const=const,fluxLoc='Wall'):
    """
    Analytical Solution for borehole melting in cylindrical coordinates.

    1) Problem setup
        Calculate the Stefan number and nondimensionalize the heat flux

    2) Get the phase boundary location, R
        transcendendal equation
        Carslaw and Jaeger (1959) Sec. 11.6, eq. 6
        phase boundary location
        Carslaw and Jaeger (1959) Sec. 11.6 eq. 5

    3) Solve for the temperature in the liquid, 0 < r < R
        Carslaw and Jaeger (1959) Sec. 11.6, eq. 3

    4) Solve for the temperature in the solid, r > R
        Carslaw and Jaeger (1959) Sec. 11.6, eq. 4

    Parameters
    ----------
    rs: array
        radial distance profile (m)
    Tinf: float
        far-field temperature, away from hole (degC)
    Qmelt: float
        heat flux for melting (W)
    Tm: float; optional
        melting temperature
    t_target: float; optional
        total time of melting (seconds)
    R_target: float; optional
        point to melt out to (m)
    target: string; optional
        whether to solve with target distance or target time
    const: class; optional
        constants class.
    fluxLoc: string; optional
        location of the heat flux for melting ['Wall' or 'Center']

    Output
    ----------
    T: array
        solution temperatures at the points of the radial distance profile (degC)
    lam:
        solved in the transcendental equation (used to get the phase boundary)
    R_melt:
        phase boundary location (m)
    t_melt:
        time to melt to current phase boundary location (sec)

    """

    # --- Setup --- #

    # Solve for the Stefan number
    St = const.ci*(Tm-Tinf)/const.L
    # Nondimensionalize the heat flux
    Qbar = Qmelt/(4.*np.pi*const.alphai*const.rhoi*const.L)

    # --- Phase Boundary --- #

    # solve the transcendental equation
    lam = fsolve(transcendental,1.,args=(St,Qbar,const.alphai,const.alphaw,fluxLoc))[0]
    # Solve for the location of the phase boundary, R
    if target=='Dist':
        R_melt = R_target
        t_melt = ((R_melt/(2.*lam))**2.)/(const.alphai)
    elif target=='Time':
        t_melt = t_target
        R_melt = 2.*lam*(const.alphai*t_melt)**.5

    # --- Water Temperature --- #

    # Solve for the water temperature
    if fluxLoc == 'Wall':
        Tw = np.zeros_like(rs)
    else:
        Tw = Tm + (-Qmelt/(4.*np.pi*const.kw))*\
            (expi(-rs**2./(4.*const.alphaw*t_melt))-\
            expi(-(const.alphai/const.alphaw)*lam**2.))
    # nan where there is ice instead of water
    Tw[rs>R_melt] = np.nan

    # --- Ice Temperature --- #

    # Solve for the ice temperature
    Ti = Tinf - ((Tinf-Tm)/expi(-lam**2.))*\
            expi(-rs**2./(4.*const.alphai*t_melt))
    # nan where there is water instead of ice
    Ti[rs<=R_melt] = np.nan

    # --- Output Temperature --- #
    T = Tw
    T[rs>R_melt] = Ti[rs>R_melt]

    return T,lam,R_melt,t_melt

def transcendental(lam,St,Qbar,alphai,alphaw,fluxLoc):
    """
    Transcendental equation
    Solve for the phase boundary by solving this equation.

    Different equations for two options:
        Constant flux at the hole wall
            based on the heat flux and the Stefan number solve for lambda
        Constant flux at the center of the hole, r=0
            Carslaw and Jaeger (1959) Sec. 11.6
            equation to solve for lambda, eq. 6, set this == 0

    Parameters
    ----------
    lam: float

    St: float
        Stefan Number
    Qbar: float
        Nondimensionalized heat flux
    alphai: flaot
        ice diffusivity
    alphaw: float
        water diffusivity
    fluxLoc: string
        location of the heat source, 'Wall' for borehole wall and 'Center' for borehole center.

    Output
    ----------
    equation to solve in scipy.fsolve.

    """
    if fluxLoc == 'Wall':
        rhs = lam**2.
        lhs = St/expi(-lam**2.)*np.exp(-lam**2.)+Qbar
        return rhs - lhs
    elif fluxLoc == 'Center':
        rhs = lam**2.
        lhs = Qbar*np.exp(-(alphai/alphaw)*lam**2.)+\
                St/expi(-lam**2.)*np.exp(-lam**2.)
        return rhs - lhs

# -------------------------------------------------------------------------------------------------------------------------------

from scipy.integrate import ode

def analyticalFreeze(self):
    self.r0 = 0.1
    self.rs = np.linspace(0,self.r0,1000)
    self.qdot = self.Q*(self.Tm-self.Tinf)*self.c.ki/(self.r0**2.)
    self.t0 = 0.
    self.tf = 10.*self.r0**2./(self.c.ki/(self.c.rhoi*self.c.ci))
    self.dt = self.tf/100.
    self.ts = np.arange(self.t0,self.tf,self.dt)

    S = ode(self.f,self.jac).set_integrator('vode', method='bdf', with_jacobian=True)
    S.set_initial_value(.9*self.r0, self.t0)

    self.R = S.y[0]
    Tw = self.qdot*self.R**2./(4.*const.kw)*(1.-self.rs**2./self.R**2.) + self.Tm
    Ti = iceTemperature(self)

    self.T[0,self.rs<self.R] = self.Tw[self.rs<self.R]
    self.T[0,self.rs==self.R] = self.Tm
    self.T[0,self.rs>self.R] = self.Ti[self.rs>self.R]

    i = 1
    while self.S.successful() and self.S.t < self.tf:
        self.S.integrate(self.S.t+self.dt)
        print("%s: %g %g" % (i, self.S.t*(self.c.ki/(self.c.rhoi*self.c.ci))/(self.r0**2.), self.S.y/self.r0))

        self.R = S.y[0]
        Tw = self.qdot*self.R**2./(4.*const.kw)*(1.-self.rs**2./self.R**2.) + self.Tm
        Ti = iceTemperature(self)

        self.T[i,self.rs<self.R] = Tw[self.rs<self.R]
        self.T[i,self.rs==self.R] = self.Tm
        self.T[i,self.rs>self.R] = Ti[self.rs>self.R]
        i+=1


def f(self,t,R,const=const):
    """
    Function to integrate
    """
    top = (R**2.-self.r0**2.)*self.qdot+4.*const.ki*(self.Tm-self.Tinf)
    bottom = const.rhoi*const.L*4.*R*(np.log(R)-np.log(self.r0))
    return top/bottom

def jac(self,t,R,const=const):
    """
    Jacobian
    """
    bottom = np.log(R)-np.log(self.r0)
    return self.qdot/(4.*const.rhoi*const.L)*(1/bottom-1./bottom**2.)

def iceTemperature(self,const=const):
    """
    Calculate the ice temperature in the freezing case.
    """
    term1 = -self.qdot*self.rs**2./(4*const.ki)
    term2 = (self.R**2.*np.log(self.r0)-self.r0**2.*np.log(self.R)-\
            (self.R**2.-self.r0**2.)*np.log(self.rs))*self.qdot
    term3 = 4.*const.ki*(self.Tm*np.log(self.r0)-\
            self.Tinf*np.log(self.R)-(self.Tm-self.Tinf)*np.log(self.rs))
    term4 = 4.*const.ki*(np.log(self.R)-np.log(self.r0))
    return term1-(term2+term3)/term4
