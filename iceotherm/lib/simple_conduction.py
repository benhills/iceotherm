#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author:
Benjamin Hills
University of Washington
Earth and Space Sciences
Aug 1, 2018
"""

import numpy as np
from scipy.special import erfc
from iceotherm.lib.constants import constants
const = constants()

# Analytic solutions to conduction problems
# All are from Carslaw and Jaeger (1959)

# ---------------------------------------------------

### Temperature near a constant boundary ###
# The infinite and semi-infinite solid
# Carslaw and Jaeger Ch. 2

def erf_solution(dT,t,x,const=const):
    """
    Carslaw and Jaeger Section 2.4
    Boundary step change in temperature, held constant at boundary

    Parameters
    --------
    dT: float
        Temperature change
    t:  float
        Time since temperature change (seconds)
    x:  array
        Distance from temperature change (meters)

    Output
    --------
    T:  array
        Temperature profile
    """
    # diffusivity
    alpha = const.k/(const.rho*const.Cp)
    # equation 2.4-7
    T = dT*erfc(abs(x)/(2.*np.sqrt(alpha*t)))
    return T

def harmonic_surface(Tmaat,Tamp,t,x,w=2.*np.pi/const.spy,const=const):
    """
    Carslaw and Jaeger Section 2.6
    Surface Boundary Temperature Harmonic Function

    Parameters
    --------
    Tmaat:  float
        Mean annual air temperature
    Tamp:   float
        Seasonal amplitude
    t:      float
        Time in period (seconds)
    x:      array
        Distance from surface boundary (meters)
    w:      float
        Period of the cycle

    Output
    --------
    T:  array
        Temperature profile
    """
    # diffusivity
    alpha = const.k/(const.rho*const.Cp)
    # equation 2.6-8
    T = Tmaat + Tamp * np.exp(-x*np.sqrt(w/(2.*alpha))) * np.cos((w*t)-x*np.sqrt(w/(2*alpha)))
    return T

# ---------------------------------------------------

def harmonic_advection(Tmaat,Tamp,t,x,w=2.*np.pi/const.spy,vel=0./const.spy,const=const):
    """
    Survace boundary temperature, harmonic function AND advection ###
    Logan and Zlotnic (1995)

    Parameters
    --------
    Tmaat:  float
        Mean annual air temperature
    Tamp:   float
        Seasonal amplitude
    t:      float
        Time in period (seconds)
    x:      array
        Distance from surface boundary (meters)
    w:      float
        Period of the cycle
    vel:    float
        downward velocity

    Output
    --------
    T:  array
        Temperature profile
    """

    # diffusivity
    alpha = const.k/(const.rho*const.Cp)
    # set up with variables from LZ (1995) eq. 4.4
    phi = w/alpha
    psi = vel**2/(4*alpha**2)
    mu = np.sqrt((np.sqrt(phi**2+psi**2)+psi)/2.)
    rho = np.sqrt((np.sqrt(phi**2+psi**2)-psi)/2.)
    # LZ (1995) eq. 4.6
    T = Tmaat + Tamp * np.exp((vel/(2*alpha)-mu)*x) * np.cos(w*t-rho*x)
    return T

# ---------------------------------------------------

def parallel_plates(dT,t,x,l,N=1000,const=const):
    """
    Temperature between two plates

    Carslaw and Jaeger Ch. 3
    Linear flow of heat in the solid bounded by two parallel planes

    Parameters
    --------
    dT:     float,  Temperature difference between boundaries and internal material
    t:      float,  Time after initiation
    x:      float,  Distance from left hand boundary
    l:      float,  Distance between plates
    N:      int,    Number of iterations to sum over

    Output
    --------
    T:      float,  Temperature at output time and location
    """

    # diffusivity
    alpha = const.k/(const.rho*const.Cp)
    # infinite sum for equation 3.4-2
    infsum = 0.
    for n in range(N):
        infsum += (((-1.)**n)/(2*n+1))*np.exp(-alpha*(2.*n+1)**2.*np.pi**2.*t/(4.*l**2.))*np.cos((2*n+1)*np.pi*x/(2*l))
    # equation 3.4-2
    T = dT - (4.*dT/np.pi)*infsum
    return T

# ---------------------------------------------------

def inst_source(Q,t,x,y,z,x_=0.,y_=0.,z_=0.,dim=3,const=const):
    """
    Instantaneous Source

    Carslaw and Jaeger Ch. 10, pg 256-259
    The use of sources and sinks in cases of variable temperature

    Parameters
    ----------
    Q:          float,  Source magnitude
    t:          float,  Time after source input or 1-d array for times
    x,y,z:      float,  3-d space
    x_,y_,z_:   float,  Centerpoint for source
    dim:        int,    # of dimensions

    Output
    -------
    T:          float,  Resulting temperature

    """
    # Define the diffusivity
    alpha = const.k/(const.rho*const.Cp)
    # Point source, CJ (1959) pg. 256
    if dim==3:
        return Q/(8.*(np.pi*alpha*t)**(3/2.))*np.exp(-((x-x_)**2.+(y-y_)**2.+(z-z_)**2.)/(4.*alpha*t))
    # Linear source, CJ (1959) pg. 258
    elif dim==2:
        return Q/(4.*np.pi*alpha*t)*np.exp(-((x-x_)**2.+(y-y_)**2.)/(4*alpha*t))
    # Planar source, CJ (1959) pg. 259
    elif dim==1:
        return Q/(2.*(np.pi*alpha*t)**(1/2.))*np.exp(-((x-x_)**2.)/(4*alpha*t))

