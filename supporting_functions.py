#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Author:
Benjamin Hills
University of Washington
Earth and Space Sciences
April 28, 2020
"""

import numpy as np
from scipy.special import gamma as γ
from scipy.special import gammaincc as γincc
from scipy.special import erf
from scipy.integrate import quad
from scipy.optimize import minimize
from constants import constants, rateFactor

# ---------------------------------------------------

def conductivity(T,rho,const=constants()):
    if np.any(T<150):
        T += const.T0
    ki = 9.828*np.exp(-5.7e-3*T)
    krho = 2.*ki*rho/(3.*const.rho-rho)
    return krho

def heat_capacity(T,const=constants()):
    if np.any(T<150):
        T += const.T0
    return 152.5 + 7.122*T

# ---------------------------------------------------

def viscosity(T,z,const=constants(),
        tau_xz=None,v_surf=None):
    """
    Rate Facor function for ice viscosity, A(T)
    Cuffey and Paterson (2010), equation 3.35

    Optional case for optimization to the surface velocity using function
    surfVelOpt()

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
        res = minimize(surfVelOpt, 1, args=(Q,P,tau_xz,T,z,v_surf))
        # C was scaled for appropriate stepping of the minimization function, scale back
        C_fin = res['x']*1e-13
        # rate factor Cuffey and Paterson (2010) equation 3.35
        A = C_fin*np.exp(-(Q/const.R)*((1./(T+const.T0+const.beta*P))-(1/const.Tstar)))
    return A

def surfVelOpt(C,Q,P,tau_xz,T,z,v_surf,const=constants()):
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
    #return abs(vx_opt-v_surf)*const.spy
    return abs(Q_opt-v_surf*tau_xz[0])*const.spy

# ---------------------------------------------------

def analytical_model(Ts,qgeo,H,adot,nz=101,
             const=constants(),
             rateFactor=rateFactor,
             T_ratefactor=-10.,
             dHdx=0.,tau_dx=0.,
             gamma=1.397,gamma_plus=True,
             verbose=False):
    """
    1-D Analytical temperature profile from Rezvanbehbahani et al. (2019)
    Main improvement from the Robin (1955) solution is the nonlinear velocity profile

    Assumptions:
        1) no horizontal advection
        2) vertical advection takes the form v=(z/H)**(gamma)
        3) firn column is treated as equivalent thickness of ice
        TODO: 4) If base is warmer than the melting temperature recalculate with new basal gradient
        5) strain heating is added to the geothermal flux

    Parameters
    ----------
    Ts:     float,  Surface Temperature (C)
    qgeo:   float,  Geothermal flux (W/m2)
    H:      float,  Ice thickness (m)
    adot:   float,  Accumulation rate (m/yr)
    nz:     int,    Number of layers in the ice column
    const:  class,  Constants
    rateFactor:     function, to calculate the rate factor from Glen's Flow Law
    T_ratefactor:   float, Temperature input to rate factor function (C)
    dHdx:       float, Surface slope to calculate tau_dx
    tau_dx:     float, driving stress input directly (Pa)
    gamma:      float, exponent on the vertical velocity
    gamma_plus: bool, optional to determine gama_plus from the logarithmic regression with Pe Number

    Output
    ----------
    z:      1-D array,  Discretized height above bed through the ice column
    T:      1-D array,  Analytic solution for ice temperature
    """

    # if the surface accumulation is input in m/yr convert to m/s
    if adot>1e-5:
        adot/=const.spy
    # Thermal diffusivity
    K = const.k/(const.rho*const.Cp)
    if gamma_plus:
        # Solve for gamma using the logarithmic regression with the Pe number
        Pe = adot*H/K
        if Pe < 5. and verbose:
            print('Pe:',Pe)
            print('The gamma_plus fit is not well-adjusted for low Pe numbers.')
        # Rezvanbehbahani (2019) eq. (19)
        gamma = 1.39+.044*np.log(Pe)
    if dHdx != 0. and tau_dx == 0.:
        # driving stress Nye (1952)
        tau_dx = const.rho*const.g*H*abs(dHdx)
    if tau_dx != 0:
        # Energy from strain heating is added to the geothermal flux
        A = rateFactor(np.array([T_ratefactor]),const)[0]
        # Rezvanbehbahani (2019) eq. (22)
        qgeo_s = (2./5.)*A*H*tau_dx**4.
        qgeo += qgeo_s
    # Rezvanbehbahani (2019) eq. (19)
    lamb = adot/(K*H**gamma)
    phi = -lamb/(gamma+1)
    z = np.linspace(0,H,nz)

    # Rezvanbehbahani (2019) eq. (17)
    Γ_1 = γincc(1/(1+gamma),-phi*z**(gamma+1))*γ(1/(1+gamma))
    Γ_2 = γincc(1/(1+gamma),-phi*H**(gamma+1))*γ(1/(1+gamma))
    term2 = Γ_1-Γ_2
    T = Ts + qgeo*(-phi)**(-1./(gamma+1.))/(const.k*(gamma+1))*term2
    return z,T


def Robin_T(Ts,qgeo,H,adot,nz=101,
        const=constants(),melt=True,verbose=False):
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
    Ts:     float,  Surface Temperature (C)
    qgeo:   float,  Geothermal flux (W/m2)
    H:      float,  Ice thickness (m)
    adot:   float,  Accumulation rate (m/yr)
    nz:     int,    Number of layers in the ice column
    const:  class,  Constants
    melt:   bool,   Choice to allow melting, when true the bed temperature
                    is locked at the pressure melting point and melt rates
                    are calculated

    Output
    ----------
    z:      1-D array,  Discretized height above bed through the ice column
    T:      1-D array,  Analytic solution for ice temperature
    """

    if verbose:
        print('Solving Robin Solution for analytic temperature profile')
        print('Surface Temperature:',Ts)
        print('Geothermal Flux:',qgeo)
        print('Ice Thickness:',H)
        print('Accumulation Rate',adot)

    z = np.linspace(0,H,nz)
    q2 = adot/(2*(const.k/(const.rho*const.Cp))*H)
    Tb_grad = -qgeo/const.k
    f = lambda z : np.exp(-(z**2.)*q2)
    TTb = Tb_grad*np.array([quad(f,0,zi)[0] for zi in z])
    dTs = Ts - TTb[-1]
    T = TTb + dTs
    # recalculate if basal temperature is above melting (see van der Veen pg 148)
    Tm = const.beta*const.rho*const.g*H
    if melt and T[0] > Tm:
        Tb_grad = -2.*np.sqrt(q2)*(Tm-Ts)/np.sqrt(np.pi)*(np.sqrt(erf(adot*H*const.rho*const.Cp/(2.*const.k)))**(-1))
        TTb = Tb_grad*np.array([quad(f,0,zi)[0] for zi in z])
        dTs = Ts - TTb[-1]
        T = TTb + dTs
        M = (Tb_grad + qgeo/const.k)*const.k/const.L
        if verbose:
            print('Melting at the bed: ', np.round(M*const.spy/const.rho*1000.,2), 'mm/year')
    if verbose:
        print('Finished Robin Solution for analytic temperature profile.\n')
    return z,T


