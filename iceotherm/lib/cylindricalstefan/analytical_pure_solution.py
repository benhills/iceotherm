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
June 12, 2019
"""

import numpy as np

from ..constants import constantsHotPointDrill
const = constantsHotPointDrill()

# -------------------------------------------------------------------------------------------------------------------------------

from scipy.special import expi
from scipy.optimize import fsolve

def analyticalMelt(rs,T_inf,Q_melt,Tf=0,t_target=0,R_target=0,target='Dist',const=const,fluxLoc='Wall'):
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
    T_inf: float
        far-field temperature, away from hole (degC)
    Q_melt: float
        heat flux for melting (W)
    Tf: float; optional
        freezing temperature
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
    St = const.ci*(Tf-T_inf)/const.L
    # Nondimensionalize the heat flux
    Qbar = Q_melt/(4.*np.pi*const.alphai*const.rhoi*const.L)

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
        Tw = Tf + (-Q_melt/(4.*np.pi*const.kw))*\
            (expi(-rs**2./(4.*const.alphaw*t_melt))-\
            expi(-(const.alphai/const.alphaw)*lam**2.))
    # nan where there is ice instead of water
    Tw[rs>R_melt] = np.nan

    # --- Ice Temperature --- #

    # Solve for the ice temperature
    Ti = T_inf - ((T_inf-Tf)/expi(-lam**2.))*\
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

def analyticalFreeze(r0,T_inf,Q_sol,tf=10.,n=100,Tf=0,const=const,verbose=False):
    """
    A quasi-static approximation for freezing in a cylinder
    Crepeau and Siahpush 2008

    The domain starts as a liquid and slowly freezes in from the edges.
    They apply a source throughout the domain (solid and liquid) because they
    are using this for nuclear reactor design, but one can simply turn
    off the source and the approximation holds.

    Parameters
    ----------
    r0: float
        outer wall radius
    T_inf: float
        far-field temperature
    Q_sol: float
        heat source inside the hole
    tf: float
        time for end of simulation
    n: int, optional
        number of points
    Tf: float, optional
        freezing temperature
    const: class, optional
        constants
    verbose: boolean, optional
        print output in ODE loop

    Output
    ----------
    T_out: 2-D array
        resulting temperatures at each time step
    R_out: 1-D array
        resulting hole wall radius through time
    rs: 1-D array
        radial distances at which temperatures are output
    ts: 1-D array
        times

    """

    # --- Setup --- #

    rs = np.linspace(0,r0,n)
    qdot = Q_sol*(Tf-T_inf)*const.ki/(r0**2.)
    t0 = 0.
    tf *= r0**2./(const.ki/(const.rhoi*const.ci))
    dt = tf/1000.
    ts = np.arange(t0,tf,dt)

    # --- Define ODE --- #

    S = ode(f,jac).set_integrator('vode', method='bdf', with_jacobian=True)
    S.set_f_params(r0,qdot,Tf,T_inf,const)
    S.set_jac_params(r0,qdot,const)
    S.set_initial_value(r0*0.99, t0)

    # --- Initial Output --- #

    R_out = np.nan*np.ones(len(ts))
    T_out = np.nan*np.ones((len(ts),len(rs)))

    R = S.y[0]
    R_out[0] = R
    T_out[0] = freezingTemperature(r0,R,rs,T_inf,qdot,Tf,const)

    # --- Iterate on the ODE --- #

    i = 1
    while S.successful() and S.t+dt < tf:
        S.integrate(S.t+dt)
        if verbose:
            print("%s: %g %g" % (i, S.t*(const.alphai)/(r0**2.), S.y/r0))
            print(S.t,tf)
        R = S.y[0]
        R_out[i] = R
        T_out[i] = freezingTemperature(r0,R,rs,T_inf,qdot,Tf,const)
        i+=1

    return T_out,R_out,rs,ts


def f(t,R,r0,qdot,Tf,T_inf,const=const):
    """
    Function to integrate
    Crepeau and Siahpush eq. 9
    """
    top = (R**2.-r0**2.)*qdot+4.*const.ki*(Tf-T_inf)
    bottom = const.rhoi*const.L*4.*R*(np.log(R)-np.log(r0))
    return top/bottom

def jac(t,R,r0,qdot,const=const):
    """
    Jacobian for ODE integration
    Derivative of Crepeau and Siahpush eq. 9
    """
    bottom = np.log(R)-np.log(r0)
    return qdot/(4.*const.rhoi*const.L)*(1./bottom-1./bottom**2.)

def freezingTemperature(r0,R,rs,T_inf,qdot,Tf=0,const=const):
    """
    Calculate the ice temperature in the freezing case.
    This is to be done at each time step after the integration.
    Crepeau and Siahpush (2008)

    Parameters
    -----------
    r0: float
        outer radius of domain
    R: float
        phase-boundary radius
    rs: array
        radial distance for output array
    T_inf: float
        far-field temperature
    qdot: float
        internal heat generation
    Tf: float, optional
        freezing temperature
    const: const, optional
        constants class

    Output
    -----------
    T: array
        Full temperature profile in solid and liquid regions

    """

    # --- Ice Temperature --- #
    # Crepeau and Siahpush eq. 7

    term1 = -qdot*rs**2./(4*const.ki)
    term2 = (R**2.*np.log(r0)-r0**2.*np.log(R)-\
            (R**2.-r0**2.)*np.log(rs))*qdot
    term3 = 4.*const.ki*(Tf*np.log(r0)-\
            T_inf*np.log(R)-(Tf-T_inf)*np.log(rs))
    term4 = 4.*const.ki*(np.log(R)-np.log(r0))

    T_ice = term1-(term2+term3)/term4

    # --- Solution Temperature --- #
    # Creqeau and Siahpush eq. 3

    T_sol = qdot*R**2./(4.*const.kw)*(1.-rs**2./R**2.) + Tf

    # --- Total Temerature --- #

    T = T_ice
    T[rs==R] = Tf
    T[rs<R] = T_sol[rs<R]

    return T
