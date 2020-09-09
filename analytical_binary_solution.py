#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author:
Benjamin Hills
University of Washington
Earth and Space Sciences
June 12, 2019
"""

import numpy as np

from constants import constantsIceDiver
const = constantsIceDiver()
from concentration_functions import Tf_depression,molDiff

# ---------------------------------------------------------------------

# --- Dimensionless Diffusivity --- #

def diffusivityEps(Tinf,C0,const=const):
    """
    Get dimensionless diffusivity (i.e. ratio between molecular and thermal diffusion)

    Parameters
    ----------
    Tinf: float
        bulk temperature far from phase boundary
    C0: float
        initial concentration in the solution that the solid is moving into
    const: class
        constants

    Output
    ----------
    eps_i: float
        solid diffusivity ratio
    eps_s: float
        solution diffusivity ratio
    """

    # Ice
    D = molDiff(C0,Tinf)
    Lewis_i = const.ki/(const.rhoi*const.ci*D)
    eps_i = np.sqrt(1./Lewis_i)

    # Solution
    ks = C0/const.rhoe*const.ke+(1.-C0/const.rhoe)*const.kw
    rhos = C0+(1.-C0/const.rhoe)*const.rhow
    cs = (C0/const.rhoe)*const.ce+(1.-(C0/const.rhoe))*const.cw
    Lewis_s = ks/(rhos*cs*D)
    eps_s = np.sqrt(1./Lewis_s)

    return eps_i,eps_s

# ---------------------------------------------------------------------

# --- Classical Worster Inequality for 1-D Freezing --- #

from scipy.optimize import fsolve
from scipy.special import erf,erfc

def F(lam):
    """
    Similarity solution in the solid, Worster eq. 2.14

    Parameters
    ----------
    lam: float
        solidification coefficient

    Output
    ----------
    F: float
        similarity solution in the solid
    """

    F = np.sqrt(np.pi)*lam*np.exp(lam**2.)*erfc(lam)

    return F

def G(lam):
    """
    Similarity solution in the liquid Worster eq. 2.8

    Parameters
    ----------
    lam: float
        solidification coefficient

    Output
    ----------
    G: flaot
        similarity solution in the liquid
    """

    G = np.sqrt(np.pi)*lam*np.exp(lam**2.)*erf(lam)

    return G

def worsterLam(lam,Tb,Tinf,C0,const=const,worsterCheck=False):
    """
    Optimize for lambda, Worster 4.12b

    Parameters
    ----------
    lam: float
        solidification coefficient
    Tb: float
        fixed temperature at the inner boundary condition
    Tinf: float
        bulk temperature far from phase boundary
    C0: float
        initial concentration in the solution that the solid is moving into
    const: class
        constants
    worsterCheck: bool
        either use the constants from Worster (2000)
        or get diffusivity from function diffusivityEps()

    Output
    ----------
    lhs-rhs: float
        output to minimize
    """

    if worsterCheck:
        # Get dimensionless diffusivity
        eps_i = const.eps_i
        eps_s = const.eps_s
    else:
        # Get dimensionless diffusivity
        eps_i,eps_s = diffusivityEps(Tinf,C0,const=const)

    # Concentration from Worster 4.12a
    Ci = C0/(1.-F(lam))
    # Freezing point depression, Worster 4.6
    Ti = -const.Kf*Ci

    # Setup the optimization
    if worsterCheck:
        lhs = const.LCp
    else:
        lhs = const.L/const.ci
    rhs1 = (Ti-Tb)/G(eps_i*lam)
    rhs2 = (Tinf-Ti)/F(eps_s*lam)

    return lhs - rhs1 + rhs2

def worsterInequality(Tb,Tinf,C0,const=const,worsterCheck=False):
    """
    The inequality which describes the conditions under which we have
    constitutional supercooling (i.e. slush), Worster 4.15

    Parameters
    ----------
    Tb: float
        fixed temperature at the inner boundary condition
    Tinf: float
        bulk temperature far from phase boundary
    C0: float
        initial concentration in the solution that the solid is moving into
    const: class
        constants
    worsterCheck: bool
        either use the constants from Worster (2000)
        or get diffusivity from function diffusivityEps()

    Output
    ----------
    constitutional_supercooling: float
        solution to the inequality
        where this value is < 0 there is constitutional supercooling.
    """

    if worsterCheck:
        # Get dimensionless diffusivity
        eps_s = const.eps_s
        eps_i = const.eps_i
        # Solve optimization for lambda
        lam = fsolve(worsterLam,0.01,args=(Tb,Tinf,C0,const,worsterCheck))[0]
    else:
        # Get dimensionless diffusivity
        eps_i,eps_s = diffusivityEps(Tinf,C0,const=const)
        # Solve optimization for lambda
        lam = fsolve(worsterLam,0.01,args=(Tb,Tinf,C0,const,worsterCheck))

    # Concentration from Worster 4.12a
    Ci = C0/(1.-F(lam))
    # Freezing point depression, Worster 4.6
    Ti = -const.Kf*Ci

    constitutional_supercooling = abs(eps_s**2.*(Tinf-Ti)/F(eps_s*lam) - const.Kf*(Ci-C0)/F(lam))

    return constitutional_supercooling

def worsterProfiles(lam,eta,Tb,Tinf,C0,const=const,worsterCheck=False):
    """
    Temperature profiles in the solid and solution, Worster 4.8 and 4.9
    Concentration profile in solution, Worster 4.10

    Parameters
    ----------
    lam: float
        solidification coefficient
    eta: float
        simlarity variable, Worster 4.11
    Tb: float
        fixed temperature at the inner boundary condition
    Tinf: float
        bulk temperature far from phase boundary
    C0: float
        initial concentration in the solution that the solid is moving into
    const: class
        constants
    worsterCheck: bool
        either use the constants from Worster (2000)
        or get diffusivity from function diffusivityEps()

    Output
    ----------
    Tsol: float
        temperature in the solid
    Tliq: float
        temmperature in the liquid
    Cliq: float
        concentration in the liquid
    """

    # Get dimensionless diffusivity
    if worsterCheck:
        eps_i = const.eps_i
        eps_s = const.eps_s
    else:
        eps_i,eps_s = diffusivityEps(Tinf,C0,const=const)

    # Concentration from Worster 4.12a
    Ci = C0/(1.-F(lam))
    # Freezing point depression, Worster 4.6
    Ti = -const.Kf*Ci

    # Temperature profile in the solid, Worster 4.8
    Tsol = Tb + (Ti-Tb)*erf(eps_i*eta)/erf(eps_i*lam)
    Tsol[eta>lam] = np.nan
    # Concentration in the liquid, Worster 4.10
    Cliq = C0 + (Ci-C0)*erfc(eta)/erfc(lam)
    Cliq[eta<lam] = np.nan
    # Temperature profile in the liquid, Worster 4.9
    Tliq = Tinf + (Ti-Tinf)*erfc(eps_s*eta)/erfc(eps_s*lam)
    Tliq[eta<lam] = np.nan

    return Tsol,Tliq,Cliq

# ---------------------------------------------------------------------

# --- Adaptation of Worster Inequality for Outward Cylindrical Freezing --- #

""" Note: These functions were used to build intuition but do not apply to the
borehole case. These are for outward freezing from some linear heat sink at r=0."""

from scipy.special import expi

def cylindricalLam(lam,Tb,Tinf,C0,Q,const=const):
    """
    Optimize for lambda

    Parameters
    ----------
    lam: float
        solidification coefficient
    Tb: float
        fixed temperature at the inner boundary condition
    Tinf: float
        bulk temperature far from phase boundary
    C0: float
        initial concentration in the solution that the solid is moving into
    Q: float
    const: class
        constants

    Output
    ----------
    lhs-rhs: float
        output to minimize
    """

    # Get dimensionless diffusivity
    eps_i,eps_s = diffusivityEps(Tinf,C0,const)

    # Concentration from Worster 4.12a
    Ci = C0/(1.+lam**2.*np.exp(lam**2.)*expi(-lam**2.))
    # Freezing point depression, Worster 4.6
    Ti = Tf_depression(Ci,const=const)

    # Setup the optimization
    lhs = -const.L/const.Cp
    rhs1 = Q/(4.*np.pi*const.ki*eps_i**2.*lam**2.)
    rhs2 = (Ti-Tinf)/(eps_s**2.*lam**2.*np.exp(eps_s**2.*lam**2.)*expi(-eps_s**2.*lam**2.))

    return lhs - rhs1 + rhs2

def cylindricalInequality(Tb,Tinf,C0,const=const):
    """
    The inequality which describes the conditions under which we have
    constitutional supercooling (i.e. slush)

    Parameters
    ----------
    Tb: float
        fixed temperature at the inner boundary condition
    Tinf: float
        bulk temperature far from phase boundary
    C0: float
        initial concentration in the solution that the solid is moving into
    const: class
        constants

    Output
    ----------
    constitutional_supercooling: float
        solution to the inequality
        where this value is < 0 there is constitutional supercooling.
    """

    # Get dimensionless diffusivity
    eps_i,eps_s = diffusivityEps(Tinf,C0,const=const)
    lam = fsolve(cylindricalLam,0.01,args=(Tb,Tinf,C0,const))

    # Concentration from Worster 4.12a
    Ci = C0/(1.-np.sqrt(np.pi)*lam*np.exp(lam**2.)*erfc(lam))
    # Freezing point depression, Worster 4.6
    Ti = Tf_depression(Ci,const=const)

    # transcendental equation from Worster 4.14
    F = np.sqrt(np.pi)*(eps_s*lam)*np.exp((eps_s*lam)**2.)*erfc(eps_s*lam)

    constitutional_supercooling = abs(eps_s**2.*(Tinf-Ti)/F + const.Kf*(Ci-C0)/F)

    return constitutional_supercooling

def cylindricalProfiles(eta,Tb,Tinf,C0,Q,const=const):
    """
    Temperature profiles in the solid and solution, based on Worster 4.8 and 4.9 for cylindrical coordinates
    Concentration profile in solution, based on Worster 4.10 for cylindrical coordinates

    Parameters
    ----------
    lam: float
        solidification coefficient
    eta: float
        simlarity variable, Worster 4.11
    Tb: float
        fixed temperature at the inner boundary condition
    Tinf: float
        bulk temperature far from phase boundary
    C0: float
        initial concentration in the solution that the solid is moving into
    const: class
        constants

    Output
    ----------
    Tsol: float
        temperature in the solid
    Tliq: float
        temperature in the liquid
    Cliq: float
        concentration in the liquid
    """

    # Get dimensionless diffusivity
    eps_i,eps_s = diffusivityEps(Tinf,C0,const=const)
    lam = fsolve(cylindricalLam,0.01,args=(Tb,Tinf,C0,const))

    # Concentration, based on Worster 4.12a for cylindrical coordinates
    Ci = C0/(1.+lam**2.*np.exp(lam**2.)*expi(-lam**2.))
    # Freezing point depression, Worster 4.6
    Ti = Tf_depression(Ci,const=const)

    # Concentration in the liquid
    Cliq = C0 + (Ci-C0)*expi(-eta**2.)/expi(-lam**2.)
    # Temperature profile in the solid
    flux = (Q/2.*np.pi*const.k)
    integral = np.log(eta/lam)
    Tsol = flux*integral+Tinf + (Ti-Tinf)*expi(-eps_s**2.*lam**2.)/expi(-eps_s**2.*lam**2.)
    # Temperature profile in the liquid
    Tliq = Tinf + (Ti-Tinf)*expi(-eps_s**2.*eta**2.)/expi(-eps_s**2.*lam**2.)

    return Tsol,Tliq,Cliq

