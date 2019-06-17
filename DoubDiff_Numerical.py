#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 17:10:04 2018

@author: benhills
"""

import numpy as np
from dolfin import *
import matplotlib.pyplot as plt

#ax1 = plt.subplot(211)
#plt.xlim(0,0.1)
#ax2 = plt.subplot(212)

### Constants
class constants(object):
    def __init__(self):
        # general
        self.spy  = 60.*60.*24.*365.24          # sec yr-1
        self.g = 9.81                           # Gravity m s-2
        self.Tf0 = 273.15                        # Reference Tempearature, triple point for water, K
        self.rhow = 1000.                       # Density of water kg m-3
        # CP (2010) pg. 12
        self.rhoi = 917.                         # Ice Density kg m-3
        # CP (2010) pg. 400
        self.ci = 2097.                         # Specific Heat Capacity J kg-1 K-1
        self.L = 3.335e5                        # Latent Heat of Fusion J kg-1
        self.ki = 2.1                            # Thermal Conductivity J m-1 K-1 s-1
        # others
        self.rhoe = 789.
        self.mmass_e = 46.07
        self.mmass_w=18.02
        self.Kf = -1.99
        self.ce = 2460.
        self.cw = 4212.                   # heat capacity for ethanol and water
        self.kw = 0.555
        self.mol_diff = 3e-8                  # TODO: figure out what value to use
        # random
        self.tol = 1e-5                              # tolerance for numerics

const = constants()

parameters['allow_extrapolation'] = True

##################################################################################################################################
 
# Problem variables
T0 = -15.                             # Far Field Temperature
R0 = 0.01                               # Initial Radius
Rf = 0.04
Q, Qmelt = 0., 2500.                 # Heat flux for problem and for melting (Qmelt is used in the analytical solution)
Qwater = 2500.
Qsource = 0.*(-T0)*const.ki/(.1**2.)
C_init,C_inject = 0.0, 0.2*const.rhoe
inject_mass = C_inject

# Nondimensionalize (Humphrey and Echelmeyer, 1990)
Tstar = T0/abs(T0)
Rstar = R0/R0
Qstar = Q/(2.*np.pi*const.ki*abs(T0))
astar_i = const.L*const.rhoi/(const.rhoi*const.ci*abs(T0))
t0 = const.rhoi*const.ci/const.ki*astar_i*R0**2.

##################################################################################################################################

### Define supporting functions to call ###

# Freezing point depression    
def Tf_depression(C):
    # calculate the density of the solution
    rhos = C + const.rhow*(1.-C/const.rhoe)
    # calculate the molality of the solution (mole/kg)
    molality = 1000.*C/(const.mmass_e*(rhos-C))
    # return the freezing point depression
    return molality*const.Kf

# Enthalpy of mixing
def Hmix(C):
    # calculate the density of the solution
    rhos = C + const.rhow*(1.-C/const.rhoe)
    # mole fraction
    hold = C*const.mmass_w/(const.mmass_e*rhos)
    Xe = hold/(1-C/rhos+hold)
    Xw = 1.-Xe
    return 1000.*(-10.6*Xw**6.*Xe-1.2*Xw*Xe+.1*Xw*Xe**2.)

# Energy source based on enthalpy of mixing
# TODO: more robust checks on this
def thermalSource(C_inject,C_init,dt,R):
    # enthalpy of mixing
    dHmix = Hmix(C_inject)-Hmix(C_init)
    # energy source (J m-3 s-1)
    phi = dHmix*1000.*C_inject/(dt*const.mmass_e)
    # density and heat capacity of the solution
    rhos = C_inject + const.rhow*(1.-C_inject/const.rhoe)
    cs = C_inject + const.cw*(1.-C_inject/const.ce)
    # convert energy source to temperature change
    dTmix = phi/(2.*np.pi*R**2.*rhos*cs)
    return dTmix

# Melting/Freezing at the hole wall
def moveWall(iMesh,sMesh,Rstar,dt,u0_i,u0_s,Qstar):
    # melting/freezing at the hole wall from prescribed flux and temperature gradient 
    # Humphrey and Echelmeyer (1990) eq. 13
    # TODO: change kw to ks
    dR = dt*(-(const.kw/const.ki)*project(Expression('exp(-x[0])')*u0_s.dx(0),sV).vector()[len(sCoords)-1] + \
             project(Expression('exp(-x[0])')*u0_i.dx(0),iV).vector()[0] + \
             Qstar/Rstar)[0]
    # Is the hole completely frozen? If so, exit
    Frozen = np.exp(iCoords[:,0])[0]+dR < 0.
    if Frozen:
        return iMesh,sMesh,Rstar,dR,Frozen

    #######################################################
    # stretch mesh rather than uniform displacement
    dRsi = dR/(Rinf-Rstar)*(Rinf-np.exp(iCoords[:,0]))
    # Interpolate the points onto what will be the new mesh (ice)
    u0_i.vector()[:] = np.array([u0_i(xi) for xi in np.log(np.exp(iCoords[:,0])+dRsi)])
    # advect the mesh according to the movement of teh hole wall
    ALE.move(iMesh,Expression('log(exp(x[0])+dRi*(Rinf-exp(x[0])))-x[0]',dRi=dR/(Rinf-Rstar),Rinf=Rinf))
    iMesh.bounding_box_tree().build(iMesh)
    #######################################################
    # stretch mesh rather than uniform displacement
    dRss = dR/(Rstar-Rcenter)*(np.exp(sCoords[:,0])-Rcenter)
    # Interpolate the points onto what will be the new mesh (solution)
    u0_s.vector()[:] = np.array([u0_s(xi) for xi in np.log(np.exp(sCoords[:,0])+dRss)])
    u0_c.vector()[:] = np.array([u0_c(xi) for xi in np.log(np.exp(sCoords[:,0])+dRss)])
    # advect the mesh according to the movement of teh hole wall
    ALE.move(sMesh,Expression('log(exp(x[0])+dRs*(exp(x[0])-Rcenter))-x[0]',dRs=dR/(Rstar-Rcenter),Rcenter=Rcenter))
    sMesh.bounding_box_tree().build(sMesh)

    #######################################################
    Rstar = np.exp(iCoords[0][0])
    return iMesh,sMesh,Rstar,dR,Frozen

##################################################################################################################################

### Define the domain for the problem ###

Rinf = Rstar*100.
Rcenter = Rstar/5.
w0,wf,n = np.log(Rstar),np.log(Rinf), 100
# Finite Element Mesh in solid
iMesh = IntervalMesh(n,w0,wf)
iCoords = iMesh.coordinates()
iV = FunctionSpace(iMesh,'CG',1)
# Finite Element Mesh in solution
sMesh = IntervalMesh(n,Rcenter,Rstar)
sMesh.coordinates()[:] = np.log(sMesh.coordinates())
sCoords = sMesh.coordinates()
sV = FunctionSpace(sMesh,'CG',1)

##################################################################################################################################

### Define Initial Condition ###

from scipy.special import expi
from scipy.optimize import fsolve
alpha_i = const.ki/(const.ci*const.rhoi)
alpha_w = const.kw/(const.cw*const.rhow)
"""
# Constant flux at the hole wall
# based on the heat flux and the Stefan number solve for lambda
def transcendental(lam,St,Qbar):
    rhs = lam**2.
    lhs = St/expi(-lam**2.)*np.exp(-lam**2.)+Qbar
    return rhs - lhs
# calculate lambda using the Stefan number and the nondimensional heat flux
St = const.ci*(Tf_depression(0.)-T0)/const.L
Qbar = Qmelt/(4.*np.pi*alpha_i*const.rhoi*const.L)    
lam = fsolve(transcendental,1.,args=(St,Qbar))[0]
# use lambda to find the melt location
def meltLoc(lam,t):
    return 2.*lam*(alpha_i*t)**.5
# invert for the time to melt to the desired location
def optLoc(t,lam,R):
    return abs(meltLoc(lam,t)-R)
t_init = fsolve(optLoc,100.,args=(lam,R0))[0]
t_drill = fsolve(optLoc,100.,args=(lam,Rf))[0]
# finally calculate the temperature profile at time t_melt
def iceTemp(lam,t,T0,Tm,r):
    return T0 - (T0-Tm)/expi(-lam**2.)*expi(-r**2./(4.*alpha_i*t))
Tinit = iceTemp(lam,t_init,Tstar,Tf_depression(0.),np.exp(iCoords)*R0)
"""




St = const.ci*(Tf_depression(0.)-T0)/const.L
Qbar = Qmelt/(4.*np.pi*alpha_i*const.rhoi*const.L)  

def transcendental(lam,St,Qbar):
    rhs = lam**2.
    lhs = Qbar*np.exp(-(alpha_i/alpha_w)*lam**2.)+St/expi(-lam**2.)*np.exp(-lam**2.)
    return lhs-rhs
lam = fsolve(transcendental,1.,args=(St,Qbar))[0]

# location of the phase boundary eq. 5
def meltLoc(lam,t):
    return 2.*lam*(alpha_i*t)**.5
# Temperature in the liquid eq. 3, 0 < r < R
def waterTemp(lam,t,Q,Tm,r):
    return Tm + (-Q/(4.*np.pi*const.kw))*(expi(-r**2./(4.*alpha_w*t))-expi(-(alpha_i/alpha_w)*lam**2.))
# Temperature in the solid eq. 4, r > R
def iceTemp(lam,t,T0,Tm,r):
    return T0 - ((T0-Tm)/expi(-lam**2.))*expi(-r**2./(4.*alpha_i*t))
# invert for the time to melt to the desired location
def optLoc(t,lam,R):
    return abs(meltLoc(lam,t)-R)
t_init = fsolve(optLoc,100.,args=(lam,R0))[0]
t_drill = fsolve(optLoc,100.,args=(lam,Rf))[0]
Tinit_s = waterTemp(lam,t_init,Qmelt,Tf_depression(0.),np.exp(sCoords)*R0)/abs(T0)
Tinit_i = iceTemp(lam,t_init,Tstar,Tf_depression(0.),np.exp(iCoords)*R0)






### Set the initial Condition ###
#u0_i = interpolate(Constant(Tstar),iV)        
#u0_s = interpolate(Constant(Tf_depression(C_init)),sV)        
u0_c = interpolate(Constant(C_init),sV)        

u0_s = Function(sV)
u0_s.vector()[:] = Tinit_s

u0_i = Function(iV)
u0_i.vector()[:] = Tinit_i

En_init = Tstar*(np.exp(Rinf)**2.-np.exp(Rstar)**2.)

### Times ###
dt = 60.
t_final = 5.*3600.
ts = np.arange(t_drill,t_final+dt,dt)/t0
dt/=t0
#ts = np.linspace(0,100,15)/t0

t_inject=np.inf#ts[1]*t0
#t_inject = np.inf#ts[5]*t0
#t_final = t_drill + 1.*3600.
#ts = np.append(ts,np.arange(t_drill/t0+dt,t_final/t0+dt,dt))
#t_inject = ts[np.argmin(abs(ts*t0-t_drill-.5*3600.))]*t0     # injection time

ax1.plot(np.exp(sCoords)*R0,u0_s.vector()*abs(T0),'b:')#,alpha=0.25)
ax1.plot(np.exp(iCoords)*R0,u0_i.vector()*abs(T0),'r:')#,alpha=0.25)

##################################################################################################################################

### Define Boundary Conditions ###

# Left boundary is the center of the borehole, so it is at the melting temperature
class iWall(SubDomain):
    def inside(self, x, on_boundary):    
        return on_boundary and x[0] < iCoords[0] + const.tol
# Right boundary is the far-field temperature
class Inf(SubDomain):
    def inside(self, x, on_boundary):  
            return on_boundary and x[0] > wf - const.tol
bc_inf = DirichletBC(iV, Tstar, Inf())

# Liquid boundary condition at hole wall (same temperature as ice)
class sWall(SubDomain):
    def inside(self, x, on_boundary):    
        return on_boundary and x[0] > sCoords[-1] - const.tol
# center flux
class center(SubDomain):
    def inside(self, x, on_boundary):    
        return on_boundary and x[0] < sCoords[0] + const.tol
    
# Identify boundary for hole wall
# This will be used in the boundary condition for mass diffusion
boundaries = FacetFunction("size_t", sMesh, 0 ) # this index 0 is an alternative to the command boundaries.set_all(0)
sWall().mark(boundaries, 1)
center().mark(boundaries, 2)
sds = Measure("ds")(subdomain_data=boundaries)

# Update the thermal boundary condition at the wall based on the current concentration
def updateBCs(u0_c):
    # update the melting temperature
    Tf = Tf_depression(u0_c.vector().array())/abs(T0)
    Tf_wall = Tf[len(sCoords)-1]
    # Reset boundary condition
    bc_iWall = DirichletBC(iV, Tf_wall, iWall())
    # Reset boundary condition
    bc_sWall = DirichletBC(sV, Tf_wall, sWall())
    return bc_iWall,bc_sWall

##################################################################################################################################

### Define the test and trial functions ###

u_i = TrialFunction(iV)
v_i = TestFunction(iV)

u_s = TrialFunction(sV)
v_s = TestFunction(sV)

T_i = Function(iV)
T_s = Function(sV)
C = Function(sV)

##################################################################################################################################

### Iterate ###


iplot = 0

ts = np.linspace(t_init,t_drill,101)/t0
dt = np.mean(np.gradient(ts))
R_out = np.array(R0)
t_out = np.array(ts[0]*t0)
for t in ts[1:]:   
    print t*t0/3600.
    
    ###########################################################################

    ### Move the mesh
    if abs(t*t0-t_drill) < const.tol:
        print 'Drilling Finished.'
        Qstar = 0.
    iMesh,sMesh,Rstar,dR,Frozen = moveWall(iMesh,sMesh,Rstar,dt,u0_i,u0_s,Qstar)
    if Frozen:
        print 'Frozen Hole!'
        break

    ###########################################################################

    ### Inject ethanol
    if abs(t*t0-t_inject) < const.tol:
        u0_c.vector()[:] = C_inject
        inject_mass = C_inject*(np.exp(Rstar)**2.-np.exp(Rcenter)**2.)
        print 'Inject!'  
        # add thermal sink from mixing
        #dTmix = thermalSource(C_inject,C_init)
        #u0_s.vector()[:] = dTmix + u0_s.vector()[:]

    if (t*t0-const.tol) >= t_inject:  
        # Solve solution concentration
        # Diffusivity
        Lewis = const.ki/(const.rhoi*const.ci*const.mol_diff)
        Dlog_c = project(Expression('mol_diff*exp(-2.*x[0])',mol_diff=astar_i/Lewis),sV)
        # calculate solute flux
        Cwall = u0_c(sCoords[-1])
        Dwall = Dlog_c(sCoords[-1])
        solFlux = Constant(np.exp(sCoords[-1][0])*(Cwall/Dwall)*(dR/dt))
        # Variational Problem
        F_c = (u_s-u0_c)*v_s*dx + dt*inner(grad(u_s), grad(Dlog_c*v_s))*dx + dt*solFlux*v_s*sds(1)
        a_c = lhs(F_c)
        L_c = rhs(F_c)
        solve(a_c==L_c,C)
        u0_c.assign(C)       

    ###########################################################################

    ### Update boundary conditions
    bc_iWall,bc_sWall = updateBCs(u0_c)

    ### Diffusivities
    #rhos = Expression('u0_c + rhow*(1.-u0_c/rhoe)',u0_c=u0_c,rhow=const.rhow,rhoe=const.rhoe)
    #cs = Expression('u0_c + cw*(1.-u0_c/ce)',u0_c=u0_c,cw=const.cw,ce=const.ce)
    # TODO: change to solution constants
    diff_ratio = (const.kw*const.rhoi*const.ci)/(const.ki*const.rhow*const.cw)
    alphalog_s = project(Expression('astar*exp(-2.*x[0])',astar=astar_i*diff_ratio),sV)
    alphalog_i = project(Expression('astar*exp(-2.*x[0])',astar=astar_i),iV)

    ### Solve heat equation
    # Set up the variational form for the current mesh location
    F_i = (u_i-u0_i)*v_i*dx + dt*inner(grad(u_i), grad(alphalog_i*v_i))*dx
    #F_i -= dt*(Qsource*t0/abs(T0)/(const.rhoi*const.ci))*v_i*dx
    a_i = lhs(F_i)
    L_i = rhs(F_i)
    F_s = (u_s-u0_s)*v_s*dx + dt*inner(grad(u_s), grad(alphalog_s*v_s))*dx 
    # Source in solution
    #F_s -= dt*(Qsource*t0/abs(T0)/(const.rhow*const.cw))*v_s*dx
    # Center heat flux
    F_s -= (Qwater/(const.kw*diff_ratio*2.*np.pi*abs(T0)))*v_s*sds(2)
    a_s = lhs(F_s)
    L_s = rhs(F_s)
    # Solve solution temperature
    solve(a_s==L_s,T_s,bc_sWall)
    u0_s.assign(T_s)    
    # Solve ice temperature
    solve(a_i==L_i,T_i,[bc_inf,bc_iWall])
    
    ### TODO: Freezing in hole
    # Hard reset on temps below Tf
    # Ice concentration
    
    u0_i.assign(T_i)  

    ###########################################################################

    ### Plot
    if iplot!=0 and iplot%10==0:#t*t0 > t_inject:
        ax1.plot(np.exp(sCoords)*R0,u0_s.vector()*abs(T0),'b:')#,alpha=0.25+0.75*t/max(ts))
        #ax1.plot(np.exp(sCoords)*R0,Tf_depression(u0_c.vector().array()),'r:',alpha=0.25+0.75*t/max(ts))
        ax1.plot(np.exp(iCoords)*R0,u0_i.vector()*abs(T0),'r:')#,alpha=0.25+0.75*t/max(ts))
    iplot += 1
    
    x_exp = Expression('pow(exp(x[0]),2)')
    #print "Conservation: ", const.cw*const.rhow*assemble(x_exp*u0_s*dx)+\
                            #const.ci*const.rhoi*assemble(x_exp*u0_i*dx)# - En_init
    if (t*t0-const.tol) >= t_inject:  
        print "Mass conserveation: ", assemble(x_exp*u0_c*dx)# - inject_mass
        
    ### Export   
    t_out = np.append(t_out,t*t0)
    R_out = np.append(R_out,Rstar*R0)
        
    
plt.plot(t_out/3600.,R_out,'r:')
#"""