#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 08:58:51 2019

@author: benhills
"""

import numpy as np
import matplotlib.pyplot as plt

from Constants import *
const = constantsTempCuffPat()





##################################################################################################################################

### Functions ###
from scipy.special import expi
from scipy.optimize import fsolve
from scipy.integrate import ode

class HoleWallFlux():
    # Constant flux at the hole wall
    # based on the heat flux and the Stefan number solve for lambda
    def transcendental(lam,St,Qbar):
        rhs = lam**2.
        lhs = St/expi(-lam**2.)*np.exp(-lam**2.)+Qbar
        return rhs - lhs
    # Use lambda to solve for the melt location and ice temperature profiles through time
    def meltLoc(lam,t):
        return 2.*lam*(alpha_i*t)**.5
    def iceTemp(lam,t,T0,Tm,r):
        return T0 - (T0-Tm)/expi(-lam**2.)*expi(-r**2./(4.*alpha_i*t))
    # Find lambda
    lam = fsolve(transcendental,1.,args=(St,Qbar))[0]



class HoleCenterFlux():
    # Constant flux at the center of the hole, r=0
    # Carslaw and Jaeger (1959) Sec. 11.6
    # equation to solve for lambda, eq. 6, set this == 0
    def transcendental(lam,St,Qbar):
        rhs = lam**2.
        lhs = Qbar*np.exp(-(alpha_i/alpha_w)*lam**2.)+St/expi(-lam**2.)*np.exp(-lam**2.)
        return lhs-rhs
    # location of the phase boundary eq. 5
    def meltLoc(lam,t):
        return 2.*lam*(alpha_i*t)**.5
    # Temperature in the liquid eq. 3, 0 < r < R
    def waterTemp(lam,t,Q,Tm,r):
        return Tm + (-Q/(4.*np.pi*kw))*(expi(-r**2./(4.*alpha_w*t))-expi(-(alpha_i/alpha_w)*lam**2.))
    # Temperature in the solid eq. 4, r > R
    def iceTemp(lam,t,T0,Tm,r):
        return T0 - ((T0-Tm)/expi(-lam**2.))*expi(-r**2./(4.*alpha_i*t))
    # Find lambda
    lam = fsolve(transcendental,1.,args=(St,Qbar))[0]



class FreezeApproximate():
    def Tliq(qdot,rs,s):
        return qdot*s**2./(4.*kliq)*(1-rs**2./s**2.) + Tm
    
    def Tsol(qdot,rs,s,r0,T0):
        term1 = -qdot*rs**2./(4*ksol)
        term2 = (s**2.*np.log(r0)-r0**2.*np.log(s)-(s**2.-r0**2.)*np.log(rs))*qdot
        term3 = 4.*ksol*(Tm*np.log(r0)-T0*np.log(s)-(Tm-T0)*np.log(rs))
        term4 = 4.*ksol*(np.log(s)-np.log(r0))
        return term1-(term2+term3)/term4
    
    def f(t,s):
        top = (s**2.-r0**2.)*qdot+4.*ksol*(Tm-T0)
        bottom = rhoi*L*4.*s*(np.log(s)-np.log(r0))
        return top/bottom
    
    def jac(t,s):
        bottom = np.log(s)-np.log(r0)
        return qdot/(4.*rhoi*L)*(1/bottom-1./bottom**2.)

    S = ode(f,jac).set_integrator('vode', method='bdf', with_jacobian=True)
    S.set_initial_value(.9*r0, t0)
    
    rs = np.linspace(0,r0,1000)
    Ts = np.empty_like(rs)
    Ts[rs<S.y] = Tliq(qdot,rs[rs<S.y],S.y)
    Ts[rs==S.y] = Tm
    Ts[rs>S.y] = Tsol(qdot,rs[rs>S.y],S.y,r0,T0)
    
    tf = 10.*r0**2./(ksol/(rhoi*ci))
    dt = tf/100.
    while S.successful() and S.t < tf:
        S.integrate(S.t+dt)
        print("%g %g" % (S.t*(ksol/(rhoi*ci))/(r0**2.), S.y/r0))
        ax2.plot(S.t/3600., S.y,'k.')
    
        Ts[rs<S.y] = Tliq(qdot,rs[rs<S.y],S.y)
        Ts[rs==S.y] = Tm
        Ts[rs>S.y] = Tsol(qdot,rs[rs>S.y],S.y,r0,T0)

