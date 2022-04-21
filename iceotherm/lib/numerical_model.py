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
from analytical_solutions import *
from ice_properties import *
from supporting_functions import *
from constants import constants

# Instantiate the constants class
const = constants()

class ice_temperature():
    """
    1-D finite-difference model for ice temperature based on
    Weertman (1968)

    Features:
        - Initialization
            1-D Analytical temperature profile from either Robin (1955) or Rezvanbehbahani et al. (2019)
            then spin up the numbercal model to steady state until all points are changing by less then 'self.tol' every time step.
        - Vertical Advection
            Two options:
                1) Use an exponential form as in Rezvanbehbahani et al. (2019)
                2) Use the shape factor, p, as from Lliboutry
        - Horizontal Advection
            Longitudinal temperature, accumulation, and thickness gradients must be input manually.
            Shear stress for lamellar flow then optimize the rate_factor to match the surface velocity.
        - Strain Heating
            Option for additional xy plane shear component (i.e. Meyer/Perol models)
        - Melt
    """

    def __init__(self,Ts=-50.,H=2850.,adot=0.1,qgeo=0.050,p=1000.,
                 dS=0.,dTs=0.,dH=0.,da=0.,
                 eps_xy=0.,A_xy='full',
                 nz=101,tol=1e-5,const=const):
        """
        Initialize the model with constant terms

        Parameters
        ----------
        Ts:         float,  Surface Temperature (C)
        H:          float,  Ice thickness (m)
        adot:       float,  Accumulation rate (m/yr)
        qgeo:       float,  Geothermal Flux (W/m2)
        p:          float,  Lliboutry Shape Factor
        dS:         float,  Surface slope (degrees)
        eps_xy:     float,  Plane Strain Rate (yr-1)
        nz:         int,    Number of layers in the ice column
        const:      class,  Constants
        """

        ### Numerical Inputs ###
        self.tol = tol                  # Convergence criteria
        self.nz=nz                      # Number of layers in the ice column

        ### Boundary Constraints ###
        self.Ts = Ts                    # Surface Temperature   [C]
        self.qgeo = qgeo                # Geothermal flux       [W/m2]
        self.H = H                      # Ice thickness         [m]
        self.adot = adot/const.spy      # Accumulation rate     [m/s]
        self.gamma = None               # Exponent for vertical velocity; if None calculate using Rezvanbehbahani (2019) eq. 19
        self.p = p                      # Lliboutry shape factor for vertical velocity (large p is ~linear)

        ### Internal Constraints ###
        self.eps_xy = eps_xy/const.spy  # Plane Strain rate [s-1]
        self.A_xy_init = A_xy           # Rate factor for plane strain heat source

        ### Ice Properties ###
        self.beta = const.beta                  # Melting point depression (default to const.beta)  [K/Pa]
        self.k = const.k*np.ones(self.nz)       # Thermal conductivity (default to const.k)         [W/m/K]
        self.Cp = const.Cp*np.ones(self.nz)     # Heat capacity (default to const.Cp)               [J/kg/K]
        self.rho = const.rho*np.ones(self.nz)   # Density (default to const.rho)                    [kg/m3]

        ### Gradients ###
        self.dS = np.sin(dS*np.pi/180.) # Surface gradient in x/y directions, used for deformational flow calculation [m/m]
        self.dTs = dTs                  # Change in air temperature over distance x/y   [C/m]
        self.dH = dH                    # Thickness gradient in x/y directions          [m/m]
        self.da = da/const.spy          # Accumulation gradient in x/y directions       [m/yr/m]

        ### Velocity Terms ###
        self.Udef = 0.                  # Deformational velocity    [m/s]
        self.Uslide = 0.                # Sliding velocity          [m/s]

        ### Thickness over time (default to None) ###
        self.Hs = None                  # Array of ice thicknesses  [m]

        ### Melting Conditions ###
        self.Mrate = 0.                 # Melt rate                             [m/s]
        self.Mcum = 0.                  # Cumulative melt                       [m]
        self.Mcum_max = None            # Max Cumulative melt for a capped lake [m]

        ### Discretize the vertical coordinate ###
        self.z = np.linspace(0,self.H,self.nz)
        self.dz = np.mean(np.gradient(self.z))      # Vertical step
        self.P = const.rho*const.g*(self.H-self.z)  # Pressure
        self.pmp = self.P*self.beta                # Pressure melting

        ### Empty Time Array as Default ###
        self.ts=[]

        ### Flags ###
        self.flags = ['verbose']

    # ------------------------------------------------------------------------------------------

    def initial_conditions(self,const=const,analytical=Robin_T):
        """
        Define the initial ice column properties using an analytical solution
        with paramaters from the beginning of the time series.
        """

        # initial temperature from analytical solution
        if analytical==Robin_T:
            self.T,M = analytical(self)
        else:
            self.T = analytical(self)

        # vertical velocity
        if self.p is None:
            # by exponent, gamma
            self.v_z = self.adot*(self.z/self.H)**self.gamma
        else:
            # by shape factor, p
            zeta = (1.-(self.z/self.H))
            self.v_z = self.adot*(1.-((self.p+2.)/(self.p+1.))*zeta+(1./(self.p+1.))*zeta**(self.p+2.))


    def source_terms(self,const=const):
        """
        Heat sources from strain heating and downstream advection (this is typically a heat sink)
        """

        # Shear Stress by Lamellar Flow (van der Veen section 4.2)
        tau_xz = const.rho*const.g*(self.H-self.z)*abs(self.dS)     # [Pa]
        # Sliding friction heat production
        self.tau_b = tau_xz[0]                          # [Pa]
        self.q_b = self.tau_b*self.Uslide               # [W/m2]

        # Initialize source term at zeros
        eps_xz = np.zeros_like(tau_xz)
        self.Sdot = np.zeros_like(tau_xz)

        ### Vertical Shear Heat Production ###
        if 'vertical_shear' in self.flags:
            # Calculate the rate_factor
            A = rate_factor(self.T,z=self.z,H=self.H,const=const,tau_xz=tau_xz,v_surf=self.Udef)   # [/s/Pa3]
            # Strain rate, Weertman (1968) eq. 7
            eps_xz = (A*tau_xz**const.n)                # [/s]
            # strain heat term
            Q = 2.*(eps_xz*tau_xz)/(self.rho*self.Cp)   # [K/s]
            self.Sdot += Q

        ### Plane Strain Heat Production ###
        if 'plane_strain' in self.flags:
            # Calculate the rate_factor
            if self.A_xy_init == 'full':
                self.A_xy = rate_factor(self.T,z=self.z,H=self.H,const=const)
            elif self.A_xy_init == 'temperate':
                self.A_xy = rate_factor(self.pmp,z=self.z,H=self.H,const=const)
            else:
                self.A_xy = self.A_xy_init
            tau_xy = (self.eps_xy/self.A_xy)**(1./const.n)
            Q_xy = 2.*(self.eps_xy*tau_xy)/(self.rho*self.Cp)
            # Add to the source term
            self.Sdot += Q_xy

        ### Advective Heat Sink ###
        if 'long_advection' in self.flags:
            v_x = self.Uslide + np.insert(cumtrapz(eps_xz,self.z),0,0)    # Horizontal velocity
            # Horizontal Temperature Gradients, Weertman (1968) eq. 6b
            dTdx = self.dTs + (self.T-self.Ts)/2. * (self.dH/self.H - self.da/self.adot)
            # Final Source Term
            self.Sdot -= v_x*dTdx

    # ------------------------------------------------------------------------------------------

    def stencil(self,dt=None,const=const):
        """
        Finite Difference Scheme for 1-d advection diffusion
        Surface boundary is fixed (air temperature)
        Bed boundary is gradient (geothermal flux plus sliding heat source)
        """

        # Choose time step
        if dt is None:
            # Check if the time series is monotonically increasing
            if len(self.ts) == 0:
                raise ValueError("If no time array, must input a time step or use 'CFL'.")
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


        ### Stencils ###

        # Diffusion Matrix
        self.diff = (self.k/(self.rho*self.Cp))*(self.dt/(self.dz**2.)) # Thermal diffusivity (adjusted for time and spatial step)
        self.A = sparse.lil_matrix((self.nz, self.nz))                  # Create a sparse matrix
        self.A.setdiag((1.-2.*self.diff)*np.ones(self.nz))              # Set the center diagonal (centered difference)
        self.A.setdiag((1.*self.diff[1:])*np.ones(self.nz-1),k=-1)      # Set the -1 diagonal
        self.A.setdiag((1.*self.diff[:-1])*np.ones(self.nz-1),k=1)      # Set the +1 diagonal
        self.B = sparse.lil_matrix((self.nz, self.nz))                  # Create a sparse matrix

        # Advection Matrix
        for i in range(len(self.z)):
            adv = (-self.v_z[i]*self.dt/self.dz)    # Advection (adjusted for time and spatial step)
            self.B[i,i] = adv                       # Set the center diagonal
            self.B[i,i-1] = -adv                    # Set the -1 diagonal (upwind scheme)

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
        self.Tgrad = -(self.qgeo+self.q_b)/self.k[0]                    # Temperature gradient at bed
        self.Sdot[0] += -2.*self.dz*self.Tgrad*self.diff[0]/self.dt     # Set the source term at the bed to match the appropriate Temperature gradient
        self.Sdot[-1] = 0.                                              # Set the source term at the surface to 0.

        # Integration stencil to calculate melt volume near the bottom of the profile
        self.int_stencil = np.ones_like(self.z)
        self.int_stencil[[0,-1]] = 0.5              # Lowest point only represents half of a full spatial step (dz/2)

    # ------------------------------------------------------------------------------------------

    def numerical_to_steady_state(self,const=const):
        """
        Run the initial conditions until stable within self.tol
        """

        if 'Sdot' not in vars(self):
            raise ValueError('Must calculate the source_terms before running to steady state.')
        if 'A' not in vars(self):
            raise ValueError('Must calculate the stencil before running to steady state.')

        # Calculate the first updated temperature profile using the stencils and heat sources calculated above.
        T_new = self.A*self.T - self.B*self.T + self.dt*self.Sdot
        # Iteration counter
        steady_iter = 0
        if 'verbose' in self.flags:
            print('Running model to steady state')
        # Continue to iterate until the updated temperature profile is all within self.tol
        while steady_iter < 1000 or any(abs(self.T[1:]-T_new[1:])>self.tol):
            # Print output
            if 'verbose' in self.flags and steady_iter%1000==0:
                print('.',end='')
            self.T = T_new.copy()
            # Update the thermal diffusivity based on the new temperature profile
            if 'temp-dependent' in self.flags:
                diffusivity_update(self)
            if 'long_advection' in self.flags and steady_iter%1000==0:
                self.source_terms()
                self.stencil(self.dt)
            # Calculate the updated temperature profile using the stencils and heat sources
            T_new = self.A*self.T - self.B*self.T + self.dt*self.Sdot
            # Reset anything above the pressure melting point
            T_new[T_new>self.pmp] = self.pmp[T_new>self.pmp]
            # Update the iteration counter
            steady_iter += 1
        self.T = T_new.copy()
        # Print one more line to break the '...'
        if 'verbose' in self.flags:
            print('')

        # Run one final iteration to see how much things are changing still
        self.T = self.A*self.T - self.B*self.T + self.dt*self.Sdot
        melt_rate(self)
        self.T_steady = self.T.copy()

    # ------------------------------------------------------------------------------------------

    def numerical_transient(self,const=const,*args,**kwargs):
        """
        Non-Steady Model
        Run the finite-difference model as it has been set up through the other functions.
        """

        # Set up the output arrays
        if 'save_all' in self.flags:
            # Output temperature profiles
            self.Ts_out = np.empty((0,len(self.T)))
            # Output melt rates
            self.Mrate_all = np.empty((0))
            # Output cumulative melt
            self.Mcum_all = np.array([0])
            # Output the depth arrays
            if self.Hs is not None:
                self.zs_out = np.empty((0,len(self.z)))

        # Expand the deformational and sliding velocity terms into an array if that has not been added manually yet
        if len(self.ts)>0 and 'Udefs' not in vars(self):
            if 'verbose' in self.flags:
                print('No velocity arrays set, setting to constant value.')
            self.Udefs, self.Uslides = self.Udef*np.ones_like(self.ts), self.Uslide*np.ones_like(self.ts)

        # Iterate through all times
        for i in range(len(self.ts)):

            ### Print and output
            print_and_save(self,i,*args,**kwargs)

            ### Update to current time
            update_time(self,i)

            ### Solve (forward difference in time)
            T_new = self.A*self.T - self.B*self.T + self.dt*self.Sdot
            self.T = T_new

            ### Calculate melting/freezing
            melt_rate(self)
