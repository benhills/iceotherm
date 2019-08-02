#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 09:12:02 2019

@author: benhills
"""

import numpy as np

from constants_stefan import *
const = constantsIceDiver()
from concentration_functions import Tf_depression,molDiff
from scipy.optimize import fsolve
from scipy.special import erf,erfc,expi

def diffusivityEps(Tinf,C0,const=const):
    # get dimensionless diffusivity
    D = molDiff(C0,Tinf)
    Lewis_i = const.ki/(const.rhoi*const.ci*D)
    eps_i = np.sqrt(1./Lewis_i)
    ks = C0/const.rhoe*const.ke+(1.-C0/const.rhoe)*const.kw
    rhos = C0+(1.-C0/const.rhoe)*const.rhow
    cs = (C0/const.rhoe)*const.ce+(1.-(C0/const.rhoe))*const.cw
    Lewis_s = ks/(rhos*cs*D)
    eps_s = np.sqrt(1./Lewis_s)
    return eps_i,eps_s

# ---------------------------------------------------------------------

# --- Classical Worster Inequality for 1-D Freezing --- #

def F(lam):
    """
    Similarity solution from Worster eq. 2.14
    """
    return np.sqrt(np.pi)*lam*np.exp(lam**2.)*erfc(lam)

def G(lam):
    """
    Similarity solution from Worster eq. 2.8
    """
    return np.sqrt(np.pi)*lam*np.exp(lam**2.)*erf(lam)

def worsterLam(lam,Tb,Tinf,C0,const=const,worsterCheck=False):
    """
    Optimize for lambda, Worster 4.12b
    """
    # get dimensionless diffusivity
    if worsterCheck:
        eps_i = const.eps_i
        eps_s = const.eps_s
    else:
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



def approxLam(lam,Tb,Tinf,C0,const=const,worsterCheck=False):
    """
    Optimize for lambda, Worster 4.14
    """
    # Concentration
    Cb = Tb/const.Kf
    scriptC = -Cb/(C0-Cb)
    # Setup the optimization
    lhs = F(lam)
    rhs1 = 1./scriptC
    return lhs - rhs1


def worsterInequality(Tb,Tinf,C0,const=const,worsterCheck=False):
    """
    The inequality which describes the conditions under which we have
    constitutional supercooling (i.e. slush), Worster 4.15
    """
    # get dimensionless diffusivity
    if worsterCheck:
        eps_s = const.eps_s
        eps_i = const.eps_i
        lam = fsolve(approxLam,0.01,args=(Tb,Tinf,C0,const,worsterCheck))[0]
        #lam = fsolve(worsterLam,0.01,args=(Tb,Tinf,C0,const,worsterCheck))[0]
    else:
        eps_i,eps_s = diffusivityEps(Tinf,C0,const=const)
        # solve optimization for lambda
        lam = fsolve(worsterLam,0.01,args=(Tb,Tinf,C0,const,worsterCheck))
    # Concentration from Worster 4.12a
    Ci = C0/(1.-F(lam))
    # Freezing point depression, Worster 4.6
    Ti = -const.Kf*Ci
    return abs(eps_s**2.*(Tinf-Ti)/F(eps_s*lam) - const.Kf*(Ci-C0)/F(lam))

def worsterProfiles(lam,eta,Tb,Tinf,C0,const=const,worsterCheck=False):

    # get dimensionless diffusivity
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

# --- Adaptation of Worster Inequality for Cylindrical Freezing --- #

def cylindricalLam(lam,Tb,Tinf,C0,Q,const=const):
    """
    Optimize for lambda
    """
    # get dimensionless diffusivity
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
    """
    # get dimensionless diffusivity
    eps_i,eps_s = diffusivityEps(Tinf,C0,const=const)

    lam = fsolve(worsterLam,0.01,args=(Tb,Tinf,C0,const))
    # Concentration from Worster 4.12a
    Ci = C0/(1.-np.sqrt(np.pi)*lam*np.exp(lam**2.)*erfc(lam))
    # Freezing point depression, Worster 4.6
    Ti = Tf_depression(Ci,const=const)
    # transcendental equation from Worster 4.14
    F = np.sqrt(np.pi)*(eps_s*lam)*np.exp((eps_s*lam)**2.)*erfc(eps_s*lam)
    return abs(eps_s**2.*(Tinf-Ti)/F + const.Kf*(Ci-C0)/F)

def cylindricalProfiles(eta,Tb,Tinf,C0,Q,const=const):

    # get dimensionless diffusivity
    eps_i,eps_s = diffusivityEps(Tinf,C0,const=const)

    lam = fsolve(worsterLam,0.01,args=(Tb,Tinf,C0,const))
    # Concentration from Worster 4.12a
    Ci = C0/(1.-np.sqrt(np.pi)*lam*np.exp(lam**2.)*erfc(lam))
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

