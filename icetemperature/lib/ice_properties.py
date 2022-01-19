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

def rate_factor(T,const=constants(),
             d=None,pmp=0.,
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
    const:  class, optional
        Constants
    d:      array, optional
        Depth (m)
    pmp:      float, optional
        Pressure-Melting Point (C)
    tau_xz: array, optional
        Shear stress profile, only needed if optimizing the strain rate to match surface
    v_surf: float, optional
        Surface velocity to be matched in optimization

    Output
    ----------
    A:      array,  Rate Factor, viscosity = A^(-1/n)/2
    """

    # create an array for activation energies
    Qact = const.Qminus*np.ones_like(T)
    if hasattr(T,'__len__'):
        Qact[T>-10.] = const.Qplus
    elif T>-10.:
        Qact = const.Qplus
    # Overburden pressure
    if d is not None:
        pmp = const.rho*const.g*d*const.beta

    if v_surf is None:
        # rate factor Cuffey and Paterson (2010) equation 3.35
        A = const.Astar*np.exp(-(Qact/const.R)*((1./(T+const.T0-pmp))-(1./(const.Tstar-pmp))))
    else:
        # Get the final coefficient value
        res = minimize(surf_vel_opt, 1, args=(T,d,pmp,Qact,tau_xz,v_surf))
        # C was scaled for appropriate stepping of the minimization function, scale back
        C_fin = res['x']*1e-13
        # rate factor Cuffey and Paterson (2010) equation 3.35
        A = C_fin*np.exp(-(Qact/const.R)*((1./(T+const.T0-pmp))-(1./(const.Tstar-pmp))))
        # A is optimized to a m/yr velocity so bring the dimensions back to seconds
        A /= const.spy
    return A

# ---------------------------------------------------

def surf_vel_opt(C,T,d,pmp,Qact,tau_xz,v_surf,const=constants()):
    """
    Optimize the viscosity profile using the known surface velocity

    Parameters
    ----------
    C:      float
        Rate factor coefficient (multiplier)
    T:      array
        Ice Temperature (C)
    d:      array
        Depth (m)
    pmp:      float, optional
        Pressure-Melting Point (C)
    Qact:   float
        Activation Energy
    tau_xz: array
        Shear stress profile, only needed if optimizing the strain rate to match surface
    v_surf: float
        Surface velocity to be matched in optimization
    const:  class, optional
        Constants

    Output
    ----------
    energy_optimizer    float
        cost to be minimized
    """
    # Change the coefficient so that the minimization function takes appropriate steps
    C_opt = C*1e-13
    # rate factor Cuffey and Paterson (2010) equation 3.35
    A = C_opt*np.exp(-(Qact/const.R)*((1./(T+const.T0-pmp))-(1/(const.Tstar-pmp))))
    # Shear Strain Rate, Weertman (1968) eq. 7
    eps_xz = A*tau_xz**const.n/(const.spy**const.n)
    Q = 2.*(eps_xz*tau_xz)
    # Integrate the strain rate to get the surface velocity
    Q_opt = np.trapz(Q,d)
    # Optimize to conserve energy
    energy_optimizer = abs(Q_opt-v_surf*tau_xz[0])*const.spy
    return energy_optimizer
