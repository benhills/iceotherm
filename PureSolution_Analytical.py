#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 08:58:51 2019

@author: benhills
"""

import numpy as np
from Constants import *
const = constantsIceDiver()

from scipy.special import expi
from scipy.optimize import fsolve
from scipy.integrate import ode

# -------------------------------------------------------------------------------------------------------------------------------

class Melt():
    def __init__(self):
        self.c = const
        self.fluxLoc = 'Wall'
        self.Tm = 0.
        self.Q = 2500.
        self.T0 = -15.
        self.t = 1.
        self.rs = np.array([0.])

    def waterTemp(self):
        """
        Solve for the temperature in the liquid, 0 < r < R
        Carslaw and Jaeger (1959) Sec. 11.6 eq. 3

        Parameters
        ----------
        Tm
        Q
        rs
        ts
        lam

        Output
        ----------
        Tw

        """
        if self.fluxLoc == 'Wall':
            self.Tw = np.zeros(len(self.rs))
        else:
            self.Tw = self.Tm + (-self.Q/(4.*np.pi*self.c.kw))*\
                (expi(-self.rs**2./(4.*self.c.alphaw*self.t))-\
                expi(-(self.c.alphai/self.c.alphaw)*self.lam**2.))
        # nan where there is ice instead of water
        self.Tw[self.rs>self.R] = np.nan
        return


    def iceTemp(self):
        """
        Solve for the temperature in the solid, r > R
        Carslaw and Jaeger (1959) Sec. 11.6 eq. 4

        Parameters
        ----------
        T0
        Tm
        lam
        rs
        ts

        Output
        ----------
        Ti

        """
        self.Ti = self.T0 - ((self.T0-self.Tm)/expi(-self.lam**2.))*\
                expi(-self.rs**2./(4.*self.c.alphai*self.t))
        # nan where there is water instead of ice
        self.Ti[self.rs<=self.R] = np.nan
        return

    def meltLoc(self):
        """
        Solve for the location of the phase boundary, R
        Carslaw and Jaeger (1959) Sec. 11.6 eq. 5

        Parameters
        ----------
        lam
        ts

        Output
        ----------
        R

        """
        self.R = 2.*self.lam*(self.c.alphai*self.t)**.5
        return

    def transcendentalHoleCenter(self, lam):
        """
        Constant flux at the center of the hole, r=0
        Carslaw and Jaeger (1959) Sec. 11.6
        equation to solve for lambda, eq. 6, set this == 0
        """
        rhs = lam**2.
        lhs = self.Qbar*np.exp(-(self.c.alphai/self.c.alphaw)*lam**2.)+\
                self.St/expi(-lam**2.)*np.exp(-lam**2.)
        return lhs-rhs

    def transcendentalHoleWall(self, lam):
        """
        # Constant flux at the hole wall
        # based on the heat flux and the Stefan number solve for lambda
        """
        rhs = lam**2.
        lhs = self.St/expi(-lam**2.)*np.exp(-lam**2.)+self.Qbar
        return rhs - lhs

    def solveLam(self):
        """
        # Find lambda
        """

        if self.fluxLoc == 'Wall':
            transcendental = self.transcendentalHoleWall
        elif self.fluxLoc == 'Center':
            transcendental = self.transcendentalHoleCenter

        self.lam = fsolve(transcendental,1.)[0]
        return

    def solveStefan(self):
        self.St = self.c.ci*(self.Tm-self.T0)/self.c.L
        return

    def solveDimensionlessFlux(self):
        self.Qbar = self.Q/(4.*np.pi*self.c.alphai*self.c.rhoi*self.c.L)
        return

    def main(self):
        self.solveStefan()
        self.solveDimensionlessFlux()
        self.solveLam()
        self.meltLoc()
        self.iceTemp()
        self.waterTemp()
        self.T = self.Tw
        self.T[self.rs>self.R] = self.Ti[self.rs>self.R]


# -------------------------------------------------------------------------------------------------------------------------------

class Freeze():
    def __init__(self):
        self.c = const
        self.r0 = 0.1
        self.rs = np.linspace(0,self.r0,1000)
        self.Tm = 0.
        self.T0 = -15.
        self.Q = 0.
        self.qdot = self.Q*(self.Tm-self.T0)*self.c.ki/(self.r0**2.)
        self.t0 = 0.
        self.tf = 10.*self.r0**2./(self.c.ki/(self.c.rhoi*self.c.ci))
        self.dt = self.tf/100.
        self.ts = np.arange(self.t0,self.tf,self.dt)
        self.T = np.zeros((len(self.ts),len(self.rs)))

    def waterTemp(self):
        self.Tw = self.qdot*self.R**2./(4.*self.c.kw)*(1-self.rs**2./self.R**2.) + self.Tm
        return

    def iceTemp(self):
        term1 = -self.qdot*self.rs**2./(4*self.c.ki)
        term2 = (self.R**2.*np.log(self.r0)-self.r0**2.*np.log(self.R)-\
                (self.R**2.-self.r0**2.)*np.log(self.rs))*self.qdot
        term3 = 4.*self.c.ki*(self.Tm*np.log(self.r0)-\
                self.T0*np.log(self.R)-(self.Tm-self.T0)*np.log(self.rs))
        term4 = 4.*self.c.ki*(np.log(self.R)-np.log(self.r0))
        self.Ti = term1-(term2+term3)/term4
        return

    def f(self,t,R):
        top = (R**2.-self.r0**2.)*self.qdot+4.*self.c.ki*(self.Tm-self.T0)
        bottom = self.c.rhoi*self.c.L*4.*R*(np.log(R)-np.log(self.r0))
        return top/bottom

    def jac(self,t,R):
        bottom = np.log(R)-np.log(self.r0)
        return self.qdot/(4.*self.c.rhoi*self.c.L)*(1/bottom-1./bottom**2.)

    def main(self):
        self.S = ode(self.f,self.jac).set_integrator('vode', method='bdf', with_jacobian=True)
        self.S.set_initial_value(.9*self.r0, self.t0)
        self.R = self.S.y[0]
        self.waterTemp()
        self.iceTemp()

        self.T[0,self.rs<self.R] = self.Tw[self.rs<self.R]
        self.T[0,self.rs==self.R] = self.Tm
        self.T[0,self.rs>self.R] = self.Ti[self.rs>self.R]

        i = 1
        while self.S.successful() and self.S.t < self.tf:
            self.S.integrate(self.S.t+self.dt)
            print("%s: %g %g" % (i, self.S.t*(self.c.ki/(self.c.rhoi*self.c.ci))/(self.r0**2.), self.S.y/self.r0))

            self.R = self.S.y[0]
            self.waterTemp()
            self.iceTemp()

            self.T[i,self.rs<self.R] = self.Tw[self.rs<self.R]
            self.T[i,self.rs==self.R] = self.Tm
            self.T[i,self.rs>self.R] = self.Ti[self.rs>self.R]
            i+=1
