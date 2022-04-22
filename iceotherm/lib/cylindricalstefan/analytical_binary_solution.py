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
from iceotherm.lib.cylindricalstefan.concentration_functions import molDiff

# ---------------------------------------------------------------------

class constantsWorster(object):
    def __init__(self):
        self.Kf = -0.5       # Freezing depression constant (degC wt.%-1)
        self.eps = 0.05     # Dimensionless diffusivity (Le/alpha)
        self.Tinf = 20.     # Far-field temperature (degC)
        self.LCp = 80.      # Latent heat over specific heat capacity (degC)

const_worst = constantsWorster()

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

def worsterLam(lam,Tb,Tinf,C0,const=const,const_worst=const_worst,worsterCheck=False):
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
    const_worst: class
        constants from Worster (2000)
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
        eps_sol = const_worst.eps
        eps_liq = const_worst.eps
        # Freezing point depression
        m = const_worst.Kf
    else:
        # Get dimensionless diffusivity
        eps_sol,eps_liq = diffusivityEps(Tinf,C0,const=const)
        # Freezing point depression
        m = const.Kf

    # Concentration from Worster 4.12a
    Ci = C0/(1.-F(lam))
    # Freezing point depression, Worster 4.6
    Ti = m*Ci

    # Setup the optimization
    if worsterCheck:
        lhs = const_worst.LCp
    else:
        lhs = const.L/const.ci
    rhs1 = (Ti-Tb)/G(eps_sol*lam)
    rhs2 = (Tinf-Ti)/F(eps_liq*lam)

    return lhs - rhs1 + rhs2

def worsterInequality(Tb,Tinf,C0,const=const,const_worst=const_worst,worsterCheck=False):
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
    const_worst: class
        constants from Worster (2000)
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
        eps_sol = const_worst.eps_sol
        eps_liq = const_worst.eps_liq
        # Freezing point depression
        m = const_worst.Kf
    else:
        # Get dimensionless diffusivity
        eps_sol,eps_liq = diffusivityEps(Tinf,C0,const=const)
        # Freezing point depression
        m = const.Kf

    # Solve optimization for lambda
    lam = fsolve(worsterLam,0.01,args=(Tb,Tinf,C0,const,worsterCheck))[0]

    # Concentration from Worster 4.12a
    Ci = C0/(1.-F(lam))
    # Freezing point depression, Worster 4.6
    Ti = m*Ci

    # Inequality for constitutional supercooling, Worster 4.15
    constitutional_supercooling = abs(eps_liq**2.*(Tinf-Ti)/F(eps_liq*lam) - const.Kf*(Ci-C0)/F(lam))

    return constitutional_supercooling

def worsterProfiles(lam,eta,Tb,Tinf,C0,const=const,const_worst=const_worst,worsterCheck=False):
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
    const_worst: class
        constants from Worster (2000)
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

    if worsterCheck:
        # Get dimensionless diffusivity
        eps_sol = const_worst.eps
        eps_liq = const_worst.eps
        # Freezing point depression
        m = const_worst.Kf
    else:
        # Get dimensionless diffusivity
        eps_sol,eps_liq = diffusivityEps(Tinf,C0,const=const)
        # Freezing point depression
        m = const.Kf

    # Concentration from Worster 4.12a
    Ci = C0/(1.-F(lam))
    # Freezing point depression, Worster 4.6
    Ti = m*Ci

    # Temperature profile in the solid, Worster 4.8
    Tsol = Tb + (Ti-Tb)*erf(eps_sol*eta)/erf(eps_sol*lam)
    Tsol[eta>lam] = np.nan
    # Concentration in the liquid, Worster 4.10
    Cliq = C0 + (Ci-C0)*erfc(eta)/erfc(lam)
    Cliq[eta<lam] = np.nan
    # Temperature profile in the liquid, Worster 4.9
    Tliq = Tinf + (Ti-Tinf)*erfc(eps_liq*eta)/erfc(eps_liq*lam)
    Tliq[eta<lam] = np.nan

    return Tsol,Tliq,Cliq

# ---------------------------------------------------------------------

# --- Adaptation of Worster Inequality for Outward Cylindrical Freezing --- #

""" Note: These functions were used to build intuition but do not apply to the
borehole case. These are for outward freezing from some linear heat sink within
the solid near r=0.
These functions are still in development, so they should be used with caution."""

from scipy.special import expi

def cylindricalLam(lam,Q,Tinf,C0,const=const,worsterCheck=False):
    """
    Optimize for lambda

    Parameters
    ----------
    lam: float
        solidification coefficient
    Q: float
        heat flux in the solid
    Tinf: float
        bulk temperature far from phase boundary
    C0: float
        initial concentration in the solution that the solid is moving into
    const: class
        constants

    Output
    ----------
    lhs-rhs: float
        output to minimize
    """

    if worsterCheck:
        # Get dimensionless diffusivity
        eps_sol = const_worst.eps
        eps_liq = const_worst.eps
        # Freezing point depression
        m = const_worst.Kf
    else:
        # Get dimensionless diffusivity
        eps_sol,eps_liq = diffusivityEps(Tinf,C0,const=const)
        # Freezing point depression
        m = const.Kf

    # Concentration from Worster 4.12a
    Ci = C0/(1.+lam**2.*np.exp(lam**2.)*expi(-lam**2.))
    # Freezing point depression, Worster 4.6
    Ti = m*Ci

    # Setup the optimization
    if worsterCheck:
        lhs = const_worst.LCp
    else:
        lhs = const.L/const.ci
    rhs1 = Q/(4.*np.pi*const.ki*eps_sol**2.*lam**2.)
    rhs2 = (Ti-Tinf)/(eps_liq**2.*lam**2.*np.exp(eps_liq**2.*lam**2.)*expi(-eps_liq**2.*lam**2.))

    return lhs - rhs1 + rhs2

def cylindricalInequality(Q,Tinf,C0,const=const,worsterCheck=False):
    """
    The inequality which describes the conditions under which we have
    constitutional supercooling (i.e. slush)

    Parameters
    ----------
    Q: float
        heat flux in the solid
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

    if worsterCheck:
        # Get dimensionless diffusivity
        eps_sol = const_worst.eps
        eps_liq = const_worst.eps
        # Freezing point depression
        m = const_worst.Kf
    else:
        # Get dimensionless diffusivity
        eps_sol,eps_liq = diffusivityEps(Tinf,C0,const=const)
        # Freezing point depression
        m = const.Kf

    # Solve optimization for lambda
    lam = fsolve(cylindricalLam,0.01,args=(Q,Tinf,C0,const))

    # Concentration from Worster 4.12a
    Ci = C0/(1.-np.sqrt(np.pi)*lam*np.exp(lam**2.)*erfc(lam))
    # Freezing point depression, Worster 4.6
    Ti = m*Ci

    # transcendental equation from Worster 4.14
    F = np.sqrt(np.pi)*(eps_liq*lam)*np.exp((eps_liq*lam)**2.)*erfc(eps_liq*lam)

    # Inequality for constitutional supercooling, Worster 4.15
    constitutional_supercooling = abs(eps_liq**2.*(Tinf-Ti)/F + const.Kf*(Ci-C0)/F)

    return constitutional_supercooling

def cylindricalProfiles(lam,eta,Q,Tinf,C0,const=const,worsterCheck=False):
    """
    Temperature profiles in the solid and solution, based on Worster 4.8 and 4.9 for cylindrical coordinates
    Concentration profile in solution, based on Worster 4.10 for cylindrical coordinates

    Parameters
    ----------
    lam: float
        solidification coefficient
    eta: float
        simlarity variable, Worster 4.11
    Q: float
        heat flux in the solid
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

    if worsterCheck:
        # Get dimensionless diffusivity
        eps_sol = const_worst.eps
        eps_liq = const_worst.eps
        # Freezing point depression
        m = const_worst.Kf
    else:
        # Get dimensionless diffusivity
        eps_sol,eps_liq = diffusivityEps(Tinf,C0,const=const)
        # Freezing point depression
        m = const.Kf

    # Concentration, based on Worster 4.12a for cylindrical coordinates
    Ci = C0/(1.+lam**2.*np.exp(lam**2.)*expi(-lam**2.))
    # Freezing point depression, Worster 4.6
    Ti = m*Ci

    # Concentration in the liquid
    Cliq = C0 + (Ci-C0)*expi(-eta**2.)/expi(-lam**2.)
    # Temperature profile in the solid
    flux = (Q/2.*np.pi*const.ki)
    integral = np.log(eta/lam)
    Tsol = flux*integral + Tinf + (Ti-Tinf)*expi(-eps_liq**2.*lam**2.)/expi(-eps_liq**2.*lam**2.)
    # Temperature profile in the liquid
    Tliq = Tinf + (Ti-Tinf)*expi(-eps_liq**2.*eta**2.)/expi(-eps_liq**2.*lam**2.)

    return Tsol,Tliq,Cliq

