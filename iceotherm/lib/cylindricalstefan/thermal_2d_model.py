#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2020 bhills <benjaminhhills@gmail.com>
# Distributed under terms of the GNU GPL3.0 license.

"""
Author:
Benjamin Hills
University of Washington
Earth and Space Sciences
April 22, 2020
"""

import numpy as np

from iceotherm.lib.constants import constantsHotPointDrill
const = constantsHotPointDrill()

try:
    import dolfin
    dolfin.parameters['allow_extrapolation'] = True
    fe_enabled = True
except ImportError:
    fe_enabled = False

# ----------------------------------------------------------------------------------------------------------------------------------------

class thermal_2d_model():
    """
    This is a 2-dimensional model for heat transfer in cylindrical coordinates.
    The purpose of the model is to test whether we can justify a reduction to
    one-dimension for other model runs.

    The problem is solved in cylindrical coordinates with a logarithmic transform,
    following Humphrey and Echelmeyer (1990).
    An upward velocity is imposed to simulate the downward motion of the drill
    through ice. That is, the mesh is in a Lagrangian framework and cold ice moves
    upward past it.
    """

    def __init__(self,const=const):
        """
        Initial Variables
        """

        # Temperature Variables
        self.T_inf = -20.                       # Far Field Temperature
        self.Tf = 0.                            # Pure Melting Temperature

        # Domain Specifications
        self.R_inf = 3.                         # Outer Domain Edge (m)
        self.R_melt = 0.04                      # Melt-Out Radius (m)
        self.zmin = 0                           # Lower depth boundary (m)
        self.zmax = 40                          # Upper depth boundary (m)
        self.n = 200                            # Mesh resolution
        self.dt = .1*3600                       # Time step (s)
        self.t_final = 10.*3600                 # End simulation time (s)

        # Flags to keep operations in order
        self.flags = []

    # ----------------------------------------------------------------------------------------------------------------------------------------

    def log_transform(self,const=const):
        """
        Nondimensionalize and transform to logarithmic coordinates.
        This puts most of the points near the borehole wall and actually makes the
        math more like the cartesian diffusion problem.
        """

        # Nondimensionalize (Humphrey and Echelmeyer, 1990)
        self.Tstar = self.T_inf/abs(self.T_inf)                                     # Dimensionless temperature
        self.Rstar = self.R_melt/self.R_melt                                        # Dimensionless melt-out radius
        self.Rstar_inf = self.R_inf/self.R_melt                                     # Dimensionless outer domain edge
        Lv = const.L*const.rhoi     # Latent heat of fusion per unit volume         # Dimensionless heat source for meltin
        self.astar_i = Lv/(const.rhoi*const.ci*abs(self.T_inf)) # thermal diffusivit# Latent heat of fusion per unit volumy in ice
        self.t0 = const.rhoi*const.ci/const.ki*self.astar_i*self.R_melt**2.         # Thermal diffusivity of ice
                                                                                    # Characteristic time (~freeze time)
        # Dimensionless Constants
        self.St = const.ci*(self.Tf-self.T_inf)/const.L                             # Stefan number

        # Tranform to a logarithmic coordinate system so that there are more points near the borehole wall
        self.w0 = np.log(self.Rstar)                                                # Log dimensionless melt-out radius
        self.wf = np.log(self.Rstar_inf)                                            # Log dimensionless outer domain edge

        self.flags.append('log_transform')

    def get_domain(self):
        """
        Define the Finite Element domain for the problem
        """

        # Finite Element Mesh in solid
        self.ice_mesh = dolfin.RectangleMesh(dolfin.Point(self.w0,self.zmin),dolfin.Point(self.wf,self.zmax),self.n,self.n)
        self.ice_V = dolfin.FunctionSpace(self.ice_mesh,'CG',1)

        self.flags.append('get_domain')

    # ----------------------------------------------------------------------------------------------------------------------------------------

    def get_initial_conditions(self,melt_velocity=-1./3600.):
        """
        Set the initial condition to no melted ice and all at the bulk ice temperature

        Parameters
        ----------
        melt_velocity: float
            velocity of the downgoing drill (m/s)
        """

        # --- Initial states --- #
        # ice temperature
        self.u0_i = dolfin.interpolate(dolfin.Constant(self.Tstar),self.ice_V)
        # the upward velocity is equal to negative the melt rate (the mesh is Lagrangian following the drill)
        self.velocity = dolfin.Expression('melt',melt=melt_velocity*self.t0,degree=1)

        # --- Time Array --- #
        # Now that we have the melt-out time, we can define the time array
        self.ts = np.arange(0.,self.t_final+self.dt,self.dt)/self.t0
        self.dt /= self.t0

        # --- Define the test and trial functions --- #
        self.u_i = dolfin.TrialFunction(self.ice_V)
        self.v_i = dolfin.TestFunction(self.ice_V)
        self.T_i = dolfin.Function(self.ice_V)

        self.flags.append('get_ic')

    def get_boundary_conditions(mod,no_bottom_bc=False):
        """
        Define Boundary Conditions

        Parameters
        ----------
        no_bottom_bc: bool
            set the bottom boundary?
            either Dirichlet condition to the bulk ice temperature
            or free
        """

        # Left boundary is the center of the borehole, so it is at the melting temperature
        class iWall(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[0] < mod.w0 + const.tol
        # Right boundary is the bulk-ice temperature
        class Inf(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                    return on_boundary and x[0] > mod.wf - const.tol
        # Bottom boundary is the bulk-ice temperature
        class Bottom(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[1] < mod.zmin + const.tol

        # Set the Dirichlet Boundary condition at
        mod.bc_iWall = dolfin.DirichletBC(mod.ice_V, mod.Tf, iWall())
        mod.bc_Inf = dolfin.DirichletBC(mod.ice_V, mod.Tstar, Inf())
        mod.bc_Bottom = dolfin.DirichletBC(mod.ice_V, mod.Tstar, Bottom())

        # Decision on bottom boundary condition
        if no_bottom_bc:
            mod.bcs = [mod.bc_iWall,mod.bc_Inf]
        else:
            mod.bcs = [mod.bc_iWall,mod.bc_Inf,mod.bc_Bottom]

        mod.flags.append('get_bc')

    # ----------------------------------------------------------------------------------------------------------------------------------------

    def run(self,ts_out,ws_out,z_out,verbose=False,initialize_array=True):
        """
        Iterate the model through the given time array.

        Parameters
        ----------
        ts_out: array
        ws_out: array
        z_out: array
        verbose: bool; default=False
        initialized_array: bool; default=True
        """

        # --- Set up the variational form --- #

        # Diffusivity needs to be transformed into log coordinates
        alphalog_i = dolfin.project(dolfin.Expression('astar*exp(-2.*x[0])',degree=1,astar=self.astar_i),self.ice_V)
        # Functional form
        F_i = (self.u_i-self.u0_i)*self.v_i*dolfin.dx + \
               self.dt*dolfin.inner(dolfin.grad(self.u_i), dolfin.grad(alphalog_i*self.v_i))*dolfin.dx - \
               self.dt*dolfin.inner(self.velocity,self.u_i.dx(1))*self.v_i*dolfin.dx
        a_i = dolfin.lhs(F_i)
        L_i = dolfin.rhs(F_i)

        # --- Iterate over all time steps --- #

        # Initialize output array
        self.T_ice_result=np.empty((0,len(ws_out)))
        if initialize_array:
            self.T_ice_result = np.array([[self.u0_i(wi,z_out) for wi in ws_out]])*abs(self.T_inf)
        # All times
        for t in self.ts:
            if verbose:
                print(round(t*self.t0/60.),end=' min, ')
            # Solve
            dolfin.solve(a_i==L_i,self.T_i,self.bcs)
            self.u0_i.assign(self.T_i)
            # Export
            if t in ts_out:
                self.T_ice_result = np.append(self.T_ice_result,np.array([[self.T_i(wi,z_out) for wi in ws_out]])*abs(self.T_inf),axis=0)
