#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 09:12:02 2019

@author: benhills
"""

import numpy as np

from Constants import *
const = constantsTempCuffPat()
from scipy.optimize import fsolve

class binaryCylindrical():

    def Tsol(eta,lam):
        flux = (Q/2.*np.pi*const.k)
        integral = np.log(eta/lam)
        return flux*integral+Tliq(lam,lam)

    def Tliq(eta,lam):
        return Tinf + (fTi(lam)-Tinf)*expi(-epsliq**2.*eta**2.)/expi(-epsliq**2.*lam**2.)

    def Cliq(eta,lam):
        return C0 + (fCi(lam)-C0)*expi(-eta**2.)/expi(-lam**2.)

    def fCi(lam):
        return C0/(1.+lam**2.*np.exp(lam**2.)*expi(-lam**2.))

    def fTi(lam):
        return Tm - m*fCi(lam)

    def f2(lam):
        Ti = fTi(lam)
        lhs = -const.L/const.Cp
        rhs1 = Q/(4.*np.pi*const.k*epssol**2.*lam**2.)
        rhs2 = (Ti-Tinf)/(epsliq**2.*lam**2.*np.exp(epsliq**2.*lam**2.)*expi(-epsliq**2.*lam**2.))
        return lhs - rhs1 - rhs2

    lam = fsolve(f2,1.)[0]



class binaryWorster():

    def molDiff(T,b=6):
        """ Stokes-Einstein relation
        Viscosity approximated from  Khattab et al. 2012,
        but this is still for warm temperatures (~293 K)"""
        #eta_w =
        #eta_e =
        #eta = eta_w*(1.-pbv) + eta_e*pbv
        eta = 3e-3
        r = .22e-9
        return const.kBoltz*T/(b*r*np.pi*eta)

    # Freezing point depression
    def pbvMolality(pbv):
        """ Dimensional conversion from percent by volume to molality """
        # calculate the density of the solution
        rho_s = const.rhow*(1.-pbv)+const.rhoe*pbv
        # calculate the percent by mass of the solution
        pbm = pbv*(const.rhoe/rho_s)
        # calculate the molality of the solution (mole/kg)
        molality = 1000./(const.mmass_e)*pbm/(1-pbm)
        # return teh freezing point depression
        return molality

    def fTi(Tm,m,Ci):
        """ Freezing point depression
        Worster 4.6"""
        return Tm - m*Ci

    def Tsol(eta,lam,m,C0,Tb,Tm,epssol):
        """ Temperature profile in the solid,
        Worster 4.8 """
        return Tb + (fTi(Tm,m,fCi(lam,C0))-Tb)*erf(epssol*eta)/erf(epssol*lam)

    def Tliq(eta,lam,m,C0,Tinf,Tm,epsliq):
        """ Temperature profile in the liquid,
        Worster 4.9 """
        return Tinf + (fTi(Tm,m,fCi(lam,C0))-Tinf)*erfc(epsliq*eta)/erfc(epsliq*lam)

    def Cliq(eta,lam,C0):
        """ Concentration in the liquid,
        Worster 4.10 """
        return C0 + (fCi(lam,C0)-C0)*erfc(eta)/erfc(lam)

    def fCi(lam,C0):
        """ Concentration from Worster 4.12a """
        return C0/(1.-np.sqrt(np.pi)*lam*np.exp(lam**2.)*erfc(lam))

    def fTrue(lam,Tb,Tinf,m,C0,Tm,epssol,epsliq):
        """ Optimize for lambda, Worster 4.12b """
        Ti = fTi(Tm,m,fCi(lam,C0))
        lhs = const.L/const.ci
        rhs1 = (Ti-Tb)/(np.sqrt(np.pi)*epssol*lam*np.exp(epssol**2.*lam**2.)*erf(epssol*lam))
        rhs2 = (Tinf-Ti)/(np.sqrt(np.pi)*epsliq*lam*np.exp(epsliq**2.*lam**2.)*erfc(epsliq*lam))
        return lhs - rhs1 + rhs2

    def F(lam):
        """ transcendental equation from Worster 4.14 """
        return np.sqrt(np.pi)*lam*np.exp(lam**2.)*erfc(lam)

    def fApprox(lam,scriptC):
        """ Optimize for lambda, Worster 4.14 approximation """
        return F(lam)-scriptC

    def inequality(Tb,Tinf,Tm,eps,m,C0,epssol,epsliq):
        """ The inequality which describes the conditions under which we have
        constitutional supercooling (i.e. slush)
        Worster 4.15 """
        lam = fsolve(fTrue,0.01,args=(Tb,Tinf,m,C0,Tm,epssol,epsliq))
        return abs(eps**2.*(Tinf-fTi(Tm,m,fCi(lam,C0)))/F(eps*lam) - m*(fCi(lam,C0)-C0)/F(lam))

    def kbar(phi):
        return phi*const.ki + (1.-phi)*const.kw

    def lamM(Tm,m,D,C0,Tb,phi):
        S = const.L/(const.ci*(fTi(Tm,m,C0)-Tb))
        alphai = const.ki/(const.rhoi*const.ci)
        return np.sqrt((kbar(phi)*alphai)/(2.*S*const.kw*phi*D))
