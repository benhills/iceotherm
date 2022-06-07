#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author:
Benjamin Hills
University of Washington
Earth and Space Sciences
Jan 5, 2022
"""

import numpy as np
from scipy.special import gamma as γ
from scipy.special import gammaincc as γincc
from scipy.special import erf
from scipy.integrate import quad
from scipy.special import lambertw
from iceotherm.lib.constants import constants

from iceotherm.lib.ice_properties import *

def Robin_T(m,T_bulk=None,const=constants(),melt=True,verbose=False):
    """
    Analytic ice temperature model from Robin (1955)

    Assumptions:
        1) no horizontal advection
        2) vertical advection is linear with depth
        3) firn column is treated as equivalent thickness of ice
        4) If base is warmer than the melting temperature recalculate with new basal gradient
        5) no strain heating

    Parameters
    ----------
    m:          class, Model
    T_bulk      float, Temperature input to the rate factor function, A(T)
    const:      class,  Constants
    melt:       bool,   Choice to allow melting, when true the bed temperature
                        is locked at the pressure melting point and melt rates
                        are calculated
    verbose:    bool, option to print all output

    Output
    ----------
    T:          1-D array,  Analytic solution for ice temperature
    M:          float, Melt rate
    """

    # Thermal constants
    if T_bulk is None:
        k = const.k
        Cp = const.Cp
    else:
        if T_bulk == 'average':
            T_bulk = np.mean([m.Ts,m.pmp[0]])
        k = conductivity(T_bulk,const.rho)
        Cp = heat_capacity(T_bulk)
    alpha = k/(const.rho*Cp)

    q2 = m.adot/(2*alpha*m.H)
    Tb_grad = -m.qgeo/k
    f = lambda z : np.exp(-(z**2.)*q2)
    TTb = Tb_grad*np.array([quad(f,0,zi)[0] for zi in m.z])
    dTs = m.Ts - TTb[-1]
    T = TTb + dTs
    # recalculate if basal temperature is above melting (see van der Veen pg 148)
    if melt and T[0] > m.pmp[0]:
        Tb_grad = -2.*np.sqrt(q2)*(m.pmp[0]-m.Ts)/np.sqrt(np.pi)*(np.sqrt(erf(m.adot*m.H/(2.*alpha)))**(-1))
        TTb = Tb_grad*np.array([quad(f,0,zi)[0] for zi in m.z])
        dTs = m.Ts - TTb[-1]
        T = TTb + dTs
        M = (Tb_grad + m.qgeo/k)*k/const.L
        if verbose:
            print('Melting at the bed: ', np.round(M*const.spy/const.rho*1000.,2), 'mm/year')
    else:
        M = 0.
    if verbose:
        print('Finished Robin Solution for analytic temperature profile.\n')
    return T,M

# ---------------------------------------------------

def Rezvan_T(m,const=constants(),rate_factor=rate_factor,T_bulk=-10.,tau_dx=0.,verbose=False):
    """
    1-D Analytical temperature profile from Rezvanbehbahani et al. (2019)
    Main improvement from the Robin (1955) solution is the nonlinear vertical velocity profile

    Assumptions:
        1) no horizontal advection
        2) vertical advection takes the form v=(z/H)**(gamma)
        3) firn column is treated as equivalent thickness of ice
        TODO: 4) If base is warmer than the melting temperature recalculate with new basal gradient
        5) strain heating is added to the geothermal flux

    Parameters
    ----------
    m:              class,      Model
    const:          class,      Constants
    rate_factor:    function,   Calculate the rate factor from Glen's Flow Law
    T_bulk:         float,      Temperature input to rate factor function (C)
    tau_dx:         float,      Driving stress input directly (Pa)
    gamma_plus:     bool,       Optional, Determine gama_plus from the logarithmic regression with Pe Number
    verbose:        bool,       Print all output

    Output
    ----------
    T:              1-D array,  Analytic solution for ice temperature
    """

    # Thermal constants
    if T_bulk is None:
        k = const.k
        Cp = const.Cp
        A = const.Astar
    else:
        if T_bulk == 'average':
            T_bulk = np.mean([m.Ts,m.pmp[0]])
        k = conductivity(T_bulk,const.rho)
        Cp = heat_capacity(T_bulk)
        A = rate_factor(T_bulk,const)
    alpha = k/(const.rho*Cp)

    if m.gamma is None:
        # Solve for gamma using the logarithmic regression with the Pe number
        Pe = m.adot*m.H/alpha
        if Pe < 5. and verbose:
            print('Pe:',Pe)
            print('The gamma_plus fit is not well-adjusted for low Pe numbers.')
        # Rezvanbehbahani (2019) eq. (19)
        m.gamma = 1.39+.044*np.log(Pe)
    if tau_dx == 0.:
        # driving stress Nye (1952)
        tau_dx = const.rho*const.g*m.H*np.sin(m.dS)
    # Rezvanbehbahani (2019) eq. (22)
    qgeo_s = (2./5.)*A*m.H*tau_dx**4.
    qgeo = m.qgeo + qgeo_s
    # Rezvanbehbahani (2019) eq. (19)
    lamb = m.adot/(alpha*m.H**m.gamma)
    phi = -lamb/(m.gamma+1)

    # Rezvanbehbahani (2019) eq. (17)
    Γ_1 = γincc(1/(1+m.gamma),-phi*m.z**(m.gamma+1))*γ(1/(1+m.gamma))
    Γ_2 = γincc(1/(1+m.gamma),-phi*m.H**(m.gamma+1))*γ(1/(1+m.gamma))
    term2 = Γ_1-Γ_2
    T = m.Ts + m.qgeo*(-phi)**(-1./(m.gamma+1.))/(k*(m.gamma+1))*term2
    return T

# ---------------------------------------------------

def Meyer_T(m,const=constants(),
            rate_factor=rate_factor,
            T_bulk='average',
            Tb=0.,lam=0.,
            verbose=False):
    """
    Meyer and Minchew (2018)
    A 1-D analytical model of temperate ice in shear margins
    Uses the contact problem in applied mathematics

    Assumptions:
        1) Horizontal advection is treated as an energy sink by subtracting from the shear heat source
        2) Vertical advection is constant in depth (they do some linear analysis in their supplement)
        3) All ice properties, including rate factor A, are constant
        4) Ice base is at the melting temperature
        5) Melting temperature is 0 through the column

    Parameters
    ----------
    m:              class,  Model
    const:          class,  Constants
    rateFactor:     func,   function for the rate factor, A in Glen's Law
    T_bulk:         float, Temperature input to rate factor function (C)
    Tb:             float,  Basal temperature, at the pressure melting point
    lam:            float,  Paramaterized horizontal advection term
                            Meyer and Minchew (2018) eq. 11
    verbose:        bool, option to print all output

    Output
    ----------
    T:              1-D array,  Analytic solution for ice temperature
    """

    # Thermal constants
    if T_bulk is None:
        k = const.k
        Cp = const.Cp
        # rate factor (Meyer uses 2.4e-24; Table 1)
        A = 2.4e-24
    else:
        if T_bulk == 'average':
            T_bulk = np.mean([m.Ts,m.pmp[0]])
        k = conductivity(T_bulk,const.rho)
        Cp = heat_capacity(T_bulk)
        A = rate_factor(T_bulk,const)
    alpha = k/(const.rho*Cp)

    # Brinkman Number
    S = 2.*A**(-1./const.n)*(m.eps_xy)**((const.n+1.)/const.n)
    dT = Tb - m.Ts
    Br = (S*m.H**2.)/(k*dT)
    # Peclet Number
    Pe = (const.rho*Cp*m.adot*m.H)/(k)
    LAM = lam*m.H**2./(k*dT)
    if verbose:
        print('Meyer; Pe:', Pe,'Br:',Br)
    # temperature solution is different for diffusion only vs. advection-diffusion
    if abs(Pe) < 1e-3:
        # Critical Shear Strain
        eps_bar = (k*dT/(A**(-1/const.n)*m.H**(2.)))**(const.n/(const.n+1.))
        # Find the temperate thickness
        if m.eps_xy > eps_bar:
            hbar = 1.-np.sqrt(2./Br)
        else:
            hbar = 0.
        # Solve for the temperature profile
        T = m.Ts + dT*(Br/2.)*(1.-((m.z/m.H)**2.)-2.*hbar*(1.-m.z/m.H))
        T[m.z/m.H<hbar] = 0.
    else:
        # Critical Shear Strain
        eps_1 = (((0.5*Pe**2.)/(Pe-1.+np.exp(-Pe))+0.5*LAM)**(const.n/(const.n+1.)))
        eps_bar = eps_1 * ((k*dT/(A**(-1./const.n)*m.H**(2.)))**(const.n/(const.n+1.)))
        # Find the temperate thickness
        if m.eps_xy > eps_bar:
            h_1 = 1.-(Pe/(Br-LAM))
            h_2 = -(1./Pe)*(1.+np.real(lambertw(-np.exp(-(Pe**2./(Br-LAM))-1.))))
            hbar = h_1 + h_2
        else:
            hbar = 0.
        T = m.Ts + dT*((Br-LAM)/Pe)*(1.-m.z/m.H+(1./Pe)*np.exp(Pe*(hbar-1.))-(1./Pe)*np.exp(Pe*((hbar-m.z/m.H))))
        T[m.z/m.H<hbar] = 0.
    return T

# ---------------------------------------------------

def Perol_T(m,const=constants(),
                rate_factor=rate_factor,
                T_bulk='average',
                verbose=False):
    """
    Perol and Rice (2015)
    Analytic Solution for temperate ice in shear margins (equation #5)

    Assumptions:
        1) Bed is at the melting point
        2) All constants are temperature independent

    Parameters
    ----------
    m:          class,  Model
    const:      class,  Constants
    rateFactor: func,   function for the rate factor, A in Glen's Law
    T_bulk      float, Temperature input to the rate factor function, A(T)
    verbose:    bool, option to print all output

    Output
    ----------
    T:          1-D array,  Analytic solution for ice temperature
    """

    # Thermal constants
    if T_bulk is None:
        k = const.k
        Cp = const.Cp
        # rate factor (Meyer uses 2.4e-24; Table 1)
        A = 2.4e-24
    else:
        if T_bulk == 'average':
            T_bulk = np.mean([m.Ts,m.pmp[0]])
        k = conductivity(T_bulk,const.rho)
        Cp = heat_capacity(T_bulk)
        A = rate_factor(T_bulk,const)
    alpha = k/(const.rho*Cp)

    # Peclet Number
    Pe = m.adot*m.H/(k/(const.rho*Cp))
    # Strain Heating
    S = 2.*A**(-1./const.n)*(m.eps_xy)**((const.n+1.)/const.n)
    if verbose:
        print('Perol; A:',A, 'S:',S)
    # Empty Array for Temperatures, then loop through all z's
    T = np.empty_like(m.z)
    for i in range(len(m.z)):
        # Two functions to be integrated
        def f1(lam):
            return (1.-np.exp(-lam*Pe*m.z[i]**2./(2.*m.H**2.)))/(2.*lam*np.sqrt(1.-lam))
        def f2(lam):
            return (1.-np.exp(-lam*Pe/2.))/(2.*lam*np.sqrt(1.-lam))
        # Calculate temperature profile
        T[i] = m.pmp[0] + (m.Ts-m.pmp[0])*erf(np.sqrt(Pe/2.)*(m.z[i]/m.H))/erf(np.sqrt(Pe/2.)) - \
            S*m.H**2./(k*Pe) * (quad(f1,0.,1.)[0] - \
            (erf(np.sqrt(Pe/2.)*(m.z[i]/m.H))/erf(np.sqrt(Pe/2.))) * quad(f2,0.,1.)[0])
    return T
