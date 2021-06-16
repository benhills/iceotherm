#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author:
Benjamin Hills
University of Washington
Earth and Space Sciences
April 28, 2020
"""

# Import necessary libraries
import numpy as np
from scipy import sparse
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from supporting_functions import analytical_model, viscosity, Robin_T, conductivity, heat_capacity
from constants import constants
const = constants()

class ice_temperature():
    """
    1-D finite-difference model for ice temperature based on
    Weertman (1968)

    Assumptions:
        1) Initialize to 1-D Analytical temperature profile from either Robin (1955) or Rezvanbehbahani et al. (2019)
            then spin up the numbercal model to steady state until all points are changing by less then 'tol' every time step.
        2) Horizontal velocity
            Shear stress for lamellar flow then optimize the viscosity to match the surface velocity.
        3) Vertical velocity
            Two options:    first is to use an exponential form as in Rezvanbehbahani et al. (2019)
                            second is to use the shape factor, p, as from Lliboutry
    """

    def __init__(self,const=const):
        """
        Initialize the model with constant terms
        """

        ### Numerical Inputs ###
        self.nz=101         # Number of layers in the ice column
        self.tol=1e-5       # Convergence criteria

        ### Boundary Constraints ###
        self.Ts = -50.                  # Surface Temperature   [C]
        self.qgeo = .050                # Geothermal flux       [W/m2]
        self.H = 2850.                  # Ice thickness         [m]
        self.adot = .1/const.spy        # Accumulation rate     [m/s]
        self.gamma = 1.532              # Exponent for vertical velocity
        self.p = 0.      # Lliboutry shape factor for vertical velocity (large p is ~linear)

        ### Ice Properties ###
        self.k = const.k*np.ones(self.nz)
        self.Cp = const.Cp*np.ones(self.nz)
        self.rho = const.rho*np.ones(self.nz)

        ### Gradients ###
        self.dTs = 0.                       # Change in air temperature over distance x/y [C/m]
        self.dH = np.sin(.2*np.pi/180.)     # Thickness gradient in x/y directions, used for deformational flow calculation        [m/m]
        self.da = 0.                        # Accumulation gradient in x/y directions     [m/yr/m]

        ### Velocity Terms ###
        self.Udef = 0.          # Deformational velocity [m/s]
        self.Uslide = 0.        # Sliding velocity [m/s]

        ### Thickness over time (default to None) ###
        self.Hs = None      # Array of ice thicknesses

        ### Melting Conditions ###
        self.Mrate = 0.     # Melt rate [m/s]
        self.Mcum = 0.      # Cumulative melt [m]
        self.Mcum_max = None  # Max Cumulative melt for a capped lake [m]

        ### Empty Time Array as Default ###
        self.ts=[]

        ### Flags ###
        self.flags = ['verbose']

    # ---------------------------------------------

    def initial_conditions(self,const=const,analytical='Robin'):
        """
        Define the initial ice column properties using an analytical solution
        with paramaters from the beginning of the time series.
        """

        # get the initial surface temperature and downward velocity for input to analytical solution
        if hasattr(self.adot,"__len__"):
            v_z_surf = self.adot[0]
            T_surf = self.Ts[0]
        else:
            v_z_surf = self.adot
            T_surf = self.Ts

        # Weertman (1968) has this extra term to add to the vertical velocity
        if 'weertman_vel' in self.flags:
            v_z_surf += self.Udef*self.dH

        # initial temperature from analytical solution
        if analytical == 'Robin':
            self.z,self.T = Robin_T(T_surf,self.qgeo,self.H,
                    v_z_surf,const=const,nz=self.nz)
        elif analytical == 'Rezvan':
            self.z,self.T = analytical_model(T_surf,self.qgeo,self.H,
                    v_z_surf,const=const,nz=self.nz,gamma=self.gamma,gamma_plus=False)

        # vertical velocity
        if self.p == 0.:
            # by exponent, gamma
            self.v_z = v_z_surf*(self.z/self.H)**self.gamma
        else:
            # by shape factor, p
            zeta = (1.-(self.z/self.H))
            self.v_z = v_z_surf*(1.-((self.p+2.)/(self.p+1.))*zeta+(1./(self.p+1.))*zeta**(self.p+2.))

        ### Discretize the vertical coordinate ###
        self.dz = np.mean(np.gradient(self.z))      # Vertical step
        self.P = const.rho*const.g*(self.H-self.z)  # Pressure
        self.pmp = self.P*const.beta                # Pressure melting


    def source_terms(self,const=const):
        """
        Heat sources from strain heating and downstream advection
        """

        # Shear Stress by Lamellar Flow (van der Veen section 4.2)
        tau_xz = const.rho*const.g*(self.H-self.z)*abs(self.dH)

        ### Strain Heat Production ###
        if self.Udef == 0.:
            Q = np.zeros_like(tau_xz)
        else:
            # Calculate the viscosity
            A = viscosity(self.T,self.z,const=const,tau_xz=tau_xz,v_surf=self.Udef*const.spy)
            # Strain rate, Weertman (1968) eq. 7
            eps_xz = (A*tau_xz**const.n)/const.spy
            # strain heat term (K s-1)
            Q = 2.*(eps_xz*tau_xz)/(self.rho*self.Cp)

        # Sliding friction heat production
        self.tau_b = tau_xz[0]
        self.q_b = self.tau_b*self.Uslide

        ### Advection Term ###
        if 'weertman_vel' in self.flags:
            v_x = self.Uslide + np.insert(cumtrapz(eps_xz,self.z),0,0)    # Horizontal velocity
            # Horizontal Temperature Gradients, Weertman (1968) eq. 6b
            dTdx = self.dTs + (self.T-np.mean(self.Ts))/2.*(1./self.H*self.dH-(1./np.mean(self.adot))*self.da)
            ### Final Source Term ###
            self.Sdot = Q - v_x*dTdx
        else:
            self.Sdot = Q

    # ---------------------------------------------

    def stencil(self,dt=None,const=const):
        """
        Finite Difference Scheme for 1-d advection diffusion
        Surface boundary is fixed (air temperature)
        Bed boundary is gradient (geothermal flux)
        """

        # Choose time step
        if dt is None:
            # Check if the time series is monotonically increasing
            if len(self.ts) == 0:
                raise ValueError("If not steady, must input a time array.")
            if not np.all(np.gradient(np.gradient(self.ts))<self.tol):
                raise ValueError("Time series must monotonically increase.")
            self.dt = np.mean(np.gradient(self.ts))
        elif dt == 'CFL':
            # set time step with CFL
            self.dt = 0.5*self.dz/np.max(self.v_z)
        else:
            self.dt = dt
        # Stability, check the CFL
        if np.max(self.v_z)*self.dt/self.dz > 1.:
            print('CFL = ',max(self.v_z)*self.dt/self.dz,'; cannot be > 1.')
            print('dt = ',self.dt/const.spy,' years')
            print('dz = ',self.dz,' meters')
            raise ValueError("Numerically unstable, choose a smaller time step or a larger spatial step.")

        # Stencils
        self.diff = (self.k/(self.rho*self.Cp))*(self.dt/(self.dz**2.))
        self.A = sparse.lil_matrix((self.nz, self.nz))           # Create a sparse Matrix
        self.A.setdiag((1.-2.*self.diff)*np.ones(self.nz))       # Set the diagonal
        self.A.setdiag((1.*self.diff[1:])*np.ones(self.nz-1),k=-1)     # Set the diagonal
        self.A.setdiag((1.*self.diff[:-1])*np.ones(self.nz-1),k=1)      # Set the diagonal
        self.B = sparse.lil_matrix((self.nz, self.nz))           # Create a sparse Matrix
        for i in range(len(self.z)):
            adv = (-self.v_z[i]*self.dt/self.dz)
            self.B[i,i] = adv
            self.B[i,i-1] = -adv

        # Boundary Conditions
        # Neumann at bed
        self.A[0,1] = 2.*self.diff[0]
        self.B[0,:] = 0.
        # Dirichlet at surface
        self.A[-1,:] = 0.
        self.A[-1,-1] = 1.
        self.B[-1,:] = 0.

        # Source Term
        if 'Sdot' not in vars(self):
            raise ValueError('Must run the source_terms function before defining the stencil.')
        self.Tgrad = -(self.qgeo+self.q_b)/self.k[0]             # Temperature gradient at bed
        self.Sdot[0] += -2.*self.dz*self.Tgrad*self.diff[0]/self.dt
        self.Sdot[-1] = 0.

        # Integration stencil to calculate melt volume near the bottom of the profile
        self.int_stencil = np.ones_like(self.z)
        self.int_stencil[[0,-1]] = 0.5

    # ---------------------------------------------

    def thickness_update(self,H_new,T_upper=None):
        """
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
        self.pmp = self.P*const.beta                # Pressure melting

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

    def diffusivity_update(self):
        """
        """
        self.k = conductivity(self.T.copy(),self.rho)
        self.Cp = heat_capacity(self.T.copy())

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

    # ---------------------------------------------

    def run_to_steady_state(self,const=const):
        """
        """

        # Run the initial conditions until stable
        T_new = self.A*self.T - self.B*self.T + self.dt*self.Sdot
        steady_iter = 0
        if 'verbose' in self.flags:
            print('Running model to steady state')
        while any(abs(self.T[1:]-T_new[1:])>self.tol):
            if 'verbose' in self.flags and steady_iter%1000==0:
                print('.',end='')
            self.T = T_new.copy()
            if 'temp-dependent' in self.flags:
                self.diffusivity_update()
            T_new = self.A*self.T - self.B*self.T + self.dt*self.Sdot
            T_new[T_new>self.pmp] = self.pmp[T_new>self.pmp]
            steady_iter += 1
        self.T = T_new.copy()
        if 'verbose' in self.flags:
            print('')

        # Run one more time to see how much things are changing still
        self.T_steady = self.A*self.T - self.B*self.T + self.dt*self.Sdot
        self.T_steady[self.T_steady>self.pmp] = self.pmp[self.T_steady>self.pmp]

    # ---------------------------------------------

    def run(self,const=const):
        """
        Non-Steady Model
        Run the finite-difference model as it has been set up through the other functions.
        """

        # Set up the output arrays
        if 'save_all' in self.flags:
            self.Ts_out = np.empty((0,len(self.T)))
            self.Mrate_all = np.empty((0))
            self.Mcum_all = np.array([0])
            if self.Hs is not None:
                self.zs_out = np.empty((0,len(self.z)))

        # Expand the velocity terms into an array if that has not been added manually yet
        if len(self.ts)>0 and 'Udefs' not in vars(self):
            if 'verbose' in self.flags:
                print('No velocity arrays set, setting to constant value.')
            self.Udefs, self.Uslides = self.Udef*np.ones_like(self.ts), self.Uslide*np.ones_like(self.ts)

        # Iterate through all times
        for i in range(len(self.ts)):

            ### Print and output
            if i%1000 == 0 or self.ts[i] == self.ts[-1]:
                if 'verbose' in self.flags:
                    print('t =',int(self.ts[i]/const.spy),'; dt =',self.dt/const.spy,'; melt rate =',np.round(self.Mrate*1000.,2),'; melt cum = ',np.round(self.Mcum,2),'; q_b = ',self.q_b)
                if 'save_all' in self.flags:
                    self.Mrate_all = np.append(self.Mrate_all,self.Mrate)
                    self.Mcum_all = np.append(self.Mcum_all,self.Mcum)
                    self.Ts_out = np.append(self.Ts_out,[self.T],axis=0)
                    if self.Hs is not None:
                        self.zs_out = np.append(self.zs_out,[self.z],axis=0)

            ### Update to current time
            self.Udef,self.Uslide = self.Udefs[i],self.Uslides[i]   # update the velocity terms from input
            self.T[-1] = self.Ts[i]                                 # set surface temperature condition from input
            if self.Hs is not None:
                self.thickness_update(self.Hs[i]) # thickness update
            v_z_surf = self.adot[i]      # set vertical velocity
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
                self.diffusivity_update()
            # Boundary Conditions
            self.B[0,:] = 0.  # Neumann at bed
            self.B[-1,:] = 0. # Dirichlet at surface
            if i%1000 == 0: # Only update the deformational heat source periodically because it is computationally expensive
                self.source_terms()
            self.q_b = self.tau_b*self.Uslide # Update sliding heat flux
            self.Tgrad = -(self.qgeo+self.q_b)/self.k[0]  # Temperature gradient at bed updated from sliding heat flux
            self.Sdot[0] += -2*self.dz*self.Tgrad*self.diff[0]/self.dt # update boundaries on heat source vector
            self.Sdot[-1] = 0.

            ### Solve
            T_new = self.A*self.T - self.B*self.T + self.dt*self.Sdot
            self.T = T_new

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
                self.Mrate = Tminus*self.rho*self.Cp*const.spy/(const.rhow*const.L*self.dt) # melt rate should be negative now.
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
