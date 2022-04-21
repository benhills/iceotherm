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
from constants import constants
from scipy.interpolate import interp1d

# Instantiate the constants class
const = constants()

# ------------------------------------------------------------------------------------------

def print_and_save(self,i,print_increment=1000):
    """
    Print and save output every 1000th time step
    """
    if i%print_increment== 0 or self.ts[i] == self.ts[-1]:
        if 'verbose' in self.flags:
            print('t =',int(self.ts[i]/const.spy),'; dt =',self.dt/const.spy,'; melt rate =',np.round(self.Mrate*1000.,2),'; melt cum = ',np.round(self.Mcum,2),'; q_b = ',self.q_b)
        if 'save_all' in self.flags:
            self.Mrate_all = np.append(self.Mrate_all,self.Mrate)
            self.Mcum_all = np.append(self.Mcum_all,self.Mcum)
            self.Ts_out = np.append(self.Ts_out,[self.T],axis=0)
            if self.Hs is not None:
                self.zs_out = np.append(self.zs_out,[self.z],axis=0)

# ------------------------------------------------------------------------------------------

def update_time(self,i):
    """
    Update variables to current time
    """
    self.Udef,self.Uslide = self.Udefs[i],self.Uslides[i]   # Update the velocity terms from input
    self.Ts,self.adot = self.Ts_s[i],self.adot_s[i]   # Update the velocity terms from input
    self.T[-1] = self.Ts                                 # Set surface temperature condition from input
    if self.Hs is not None:
        thickness_update(self,self.Hs[i]) # thickness update
    v_z_surf = self.adot      # set vertical velocity
    # add extra term from Weertman if desired
    if 'weertman_vel' in self.flags:
        v_z_surf += self.Udef*self.dH
    if self.p == 0.: # by exponent, gamma
        self.v_z = self.Mrate/const.spy + v_z_surf*(self.z/self.H)**self.gamma
    else: # by shape factor, p
        zeta = (1.-(self.z/self.H))
        self.v_z = self.Mrate/const.spy + v_z_surf*(1.-((self.p+2.)/(self.p+1.))*zeta+(1./(self.p+1.))*zeta**(self.p+2.))
    for i in range(len(self.z)):
        adv = (-self.v_z[i]*self.dt/self.dz)
        self.B[i,i] = adv
        self.B[i,i-1] = -adv
    # Ice Properties
    if 'temp-dependent' in self.flags:
        diffusivity_update(self)
    # Boundary Conditions
    self.B[0,:] = 0.  # Neumann at bed
    self.B[-1,:] = 0. # Dirichlet at surface
    # Source
    self.source_terms()
    self.Tgrad = -(self.qgeo+self.q_b)/self.k[0]             # Temperature gradient at bed
    self.Sdot[0] += -2*self.dz*self.Tgrad*self.diff[0]/self.dt # update boundaries on heat source vector
    self.Sdot[-1] = 0.

# ------------------------------------------------------------------------------------------

def melt_rate(self):
    """
    Calculate the melt/freeze rate and save the cumulative melt to an output field
    """
    ### Calculate the volume melted/frozen during the time step, then hard reset to pmp.
    if np.any(self.T>self.pmp): # If Melting
        Tplus = (self.T[self.T>self.pmp]-self.pmp[self.T>self.pmp])*self.int_stencil[self.T>self.pmp]*self.dz # Integrate the temperature above PMP (units- degCm)
        rho = self.rho[self.T>self.pmp]
        Cp = self.Cp[self.T>self.pmp]
        self.Mrate = np.sum(Tplus*rho*Cp*const.spy/(const.rhow*const.L*self.dt)) # calculate the melt rate in m/yr
        self.T[self.T>self.pmp] = self.pmp[self.T>self.pmp] # reset temp to PMP
        self.Mcum += self.Mrate*self.dt/const.spy # Update the cumulative melt by the melt rate
    elif self.Mcum > 0 and 'water_cum' in self.flags: # If freezing
        Tminus = (self.T[0]-self.pmp[0])*0.5*self.dz # temperature below the PMP; this is only for the point at the bed because we assume water drains
        rho = self.rho[0]
        Cp = self.Cp[0]
        self.Mrate = Tminus*rho*Cp*const.spy/(const.rhow*const.L*self.dt) # melt rate should be negative now.
        if self.Mrate*self.dt/const.spy < self.Mcum: # If the amount frozen this time step is less than water available
            self.T[0] = self.pmp[0] # reset to PMP
            self.Mcum += self.Mrate*self.dt/const.spy # Update the cumulative melt by the melt rate
        else: # If the amount frozen this time step is more than water available
            M_ = (self.Mrate*self.dt/const.spy-self.Mcum) # calculate the 'extra' amount frozen in m
            Tminus = M_*(const.rhow*const.L)/(self.rho*self.Cp) # What is the equivalent temperature to cool bottom node (units - degCm)
            self.T[0] = Tminus/(0.5*self.dz) + self.pmp[0] # update the temperature at the bed
            self.Mcum = 0. # cumulative melt to zero because everything is frozen
    else:
        self.Mrate = 0.
    # Cap the lake level
    if self.Mcum_max is not None:
        if self.Mcum > self.Mcum_max:
            self.Mcum = self.Mcum_max

def thickness_update(self,H_new,T_upper=None):
    """
    Stretch/shrink the depth array to match a new thickness at each timestep.
    Interpolate temperatures to their new position.
    """

    # If no upper fill value is provided for the interpolation, use the current surface temperature
    if T_upper is None:
        T_upper = self.T[-1]
    # Build an interpolator from the prior state
    T_interp = interp1d(self.z,self.T,fill_value=(np.nan,T_upper),bounds_error=False)
    # Interpolate for new temperatures
    self.z = np.linspace(0.,H_new,self.nz)
    self.T = T_interp(self.z)

    # Assign the new thickness value
    self.H = H_new

    # Update variables that are thickness dependent
    self.dz = np.mean(np.gradient(self.z))      # Vertical step
    self.P = const.rho*const.g*(self.H-self.z)  # Pressure
    self.pmp = self.P*self.beta                # Pressure melting

    # Stability, check the CFL
    if np.max(self.v_z)*self.dt/self.dz > 1.:
        print('CFL = ',max(self.v_z)*self.dt/self.dz,'; cannot be > 1.')
        print('dt = ',self.dt/const.spy,' years')
        print('dz = ',self.dz,' meters')
        raise ValueError("Numerically unstable, choose a smaller time step or a larger spatial step.")

    # Update diffusion stencil (advection gets updated with velocity profile)
    self.diff = (self.k/(self.rho*self.Cp))*(self.dt/(self.dz**2.))
    self.A.setdiag((1.-2.*self.diff)*np.ones(self.nz))       # Set the diagonal
    self.A.setdiag((1.*self.diff[1:])*np.ones(self.nz-1),k=-1)     # Set the diagonal
    self.A.setdiag((1.*self.diff[:-1])*np.ones(self.nz-1),k=1)      # Set the diagonal
    # Boundary Conditions
    # Neumann at bed
    self.A[0,1] = 2.*self.diff[0]
    # Dirichlet at surface
    self.A[-1,:] = 0.
    self.A[-1,-1] = 1.

# ------------------------------------------------------------------------------------------

def diffusivity_update(self):
    """
    Calculate the thermal diffusivity (k/rho/Cp) based on the updated temperature and density profile.
    Reset the stencils accordingly.
    """

    if 'conductivity' in self.flags:
        self.k = conductivity(self.T.copy(),self.rho)
    if 'heat_capacity' in self.flags:
        self.Cp = heat_capacity(self.T.copy(),self.rho)

    # Update diffusion stencil (advection gets updated with velocity profile)
    self.diff = (self.k/(self.rho*self.Cp))*(self.dt/(self.dz**2.))
    self.A.setdiag((1.-2.*self.diff)*np.ones(self.nz))       # Set the diagonal
    self.A.setdiag((1.*self.diff[1:])*np.ones(self.nz-1),k=-1)     # Set the diagonal
    self.A.setdiag((1.*self.diff[:-1])*np.ones(self.nz-1),k=1)      # Set the diagonal
    # Boundary Conditions
    # Neumann at bed
    self.A[0,1] = 2.*self.diff[0]
    # Dirichlet at surface
    self.A[-1,:] = 0.
    self.A[-1,-1] = 1.

