#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author:
Benjamin Hills
University of Washington
Earth and Space Sciences
April 28, 2020
"""

import numpy as np
from scipy.optimize import minimize
from constants import constants

# ---------------------------------------------------

def conductivity(T,rho,const=constants()):
    """
    Temperature- and density-dependent conductivity

    Parameters
    --------
    T:      array
        Ice Temperature (C)
    rho:    array
        Firn density (kg m-3)
    const:  class,  Constants

    Output
    krho:   array, conductivity
    --------
    """
    if np.any(T<150):
        T += const.T0
    ki = 9.828*np.exp(-5.7e-3*T)
    krho = 2.*ki*rho/(3.*const.rho-rho)
    return krho

# ---------------------------------------------------

def heat_capacity(T,const=constants()):
    """
    Temperature--dependent heat capacity

    Parameters
    --------
    T:      array
        Ice Temperature (C)
    const:  class,  Constants

    Output
    --------
    Cp:     array, heat capacity
    """
    if np.any(T<150):
        T += const.T0
    Cp = 152.5 + 7.122*T
    return Cp

# ---------------------------------------------------

def rate_factor(temp,const,P=0.):
    """
    Rate Facor function for ice viscosity, A(T)
    Cuffey and Paterson (2010), equation 3.35

    Parameters
    --------
    temp:   float,  Temperature
    const:  class,  Constants
    P:      float,  Pressure

    Output
    --------
    A:      float,  Rate Factor, viscosity = A^(-1/n)/2
    """
    # create an array for activation energies
    Q = const.Qminus*np.ones_like(temp)
    Q[temp>-10.] = const.Qplus
    # Convert to K
    T = temp + const.T0
    # equation 3.35
    A = const.Astar*np.exp(-(Q/const.R)*((1./(T+const.beta*P))-(1/const.Tstar)))
    return A

# ---------------------------------------------------

def viscosity(T,z,const=constants(),
        tau_xz=None,v_surf=None):
    """
    Rate Facor function for ice viscosity, A(T)
    Cuffey and Paterson (2010), equation 3.35

    Optional case for optimization to the surface velocity using function
    surf_vel_opt()

    Parameters
    ----------
    T:      array
        Ice Temperature (C)
    z:      array
        Depth (m)
    const:  class
        Constants
    tau_xz: array, optional
        Shear stress profile, only needed if optimizing the strain rate to match surface
    v_surf: float, optional
        Surface velocity to be matched in optimization

    Output
    ----------
    A:      array,  Rate Factor, viscosity = A^(-1/n)/2
    """

    # create an array for activation energies
    Q = const.Qminus*np.ones_like(T)
    Q[T>-10.] = const.Qplus
    # Overburden pressure
    P = const.rho*const.g*z

    if v_surf is None:
        # rate factor Cuffey and Paterson (2010) equation 3.35
        A = const.Astar*np.exp(-(Q/const.R)*((1./(T+const.T0+const.beta*P))-(1./const.Tstar)))
    else:
        # Get the final coefficient value
        res = minimize(surf_vel_opt, 1, args=(Q,P,tau_xz,T,z,v_surf))
        # C was scaled for appropriate stepping of the minimization function, scale back
        C_fin = res['x']*1e-13
        # rate factor Cuffey and Paterson (2010) equation 3.35
        A = C_fin*np.exp(-(Q/const.R)*((1./(T+const.T0+const.beta*P))-(1./const.Tstar)))
        # A is optimized to a m/yr velocity so bring the dimensions back to seconds
        A /= const.spy
    return A

# ---------------------------------------------------

def surf_vel_opt(C,Q,P,tau_xz,T,z,v_surf,const=constants()):
    """
    Optimize the viscosity profile using the known surface velocity
    TODO: has not been fully tested
    """
    # Change the coefficient so that the minimization function takes appropriate steps
    C_opt = C*1e-13
    # rate factor Cuffey and Paterson (2010) equation 3.35
    A = C_opt*np.exp(-(Q/const.R)*((1./(T+const.T0+const.beta*P))-(1/const.Tstar)))
    # Shear Strain Rate, Weertman (1968) eq. 7
    eps_xz = A*tau_xz**const.n
    Q = 2.*(eps_xz*tau_xz)
    # Integrate the strain rate to get the surface velocity
    vx_opt = np.trapz(eps_xz,z)
    Q_opt = np.trapz(Q,z)
    # Optimize to conserve energy
    return abs(Q_opt-v_surf*tau_xz[0])*const.spy
