#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 11:31:28 2016

@author: ben
"""

import numpy as np
from constants import constantsIceDiver
import dolfin
const = constantsIceDiver()

dolfin.parameters['allow_extrapolation'] = True

# ----------------------------------------------------------------------------------------------------------------------------------------

class thermal_model_2d():
    def __init__(self,const=const):
        """
        Initial Variables
        """

        # Temperature Variables
        self.T_inf = -15.                       # Far Field Temperature
        self.Tf = 0.                            # Pure Melting Temperature

        # Domain Specifications
        self.R_inf = 3.                          # Outer Domain Edge
        self.R_melt = 0.04                          # Melt-Out Radius
        self.ymin = 0
        self.ymax = 20
        self.n = 200
        self.dt = .1
        self.t_final = 10.

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
        self.Tstar = self.T_inf/abs(self.T_inf)
        self.Rstar = self.R_melt/self.R_melt
        self.Rstar_center = self.R_center/self.R_melt
        self.Rstar_inf = self.R_inf/self.R_melt
        self.Qstar = self.Q_wall/(2.*np.pi*const.ki*abs(self.T_inf))
        Lv = const.L*const.rhoi     # Latent heat of fusion per unit volume
        self.astar_i = Lv/(const.rhoi*const.ci*abs(self.T_inf))
        self.t0 = const.rhoi*const.ci/const.ki*self.astar_i*self.R_melt**2.

        # Dimensionless Constants
        self.St = const.ci*(self.Tf-self.T_inf)/const.L
        self.Lewis = const.ki/(const.rhoi*const.ci*self.mol_diff)

        # Tranform to a logarithmic coordinate system so that there are more points near the borehole wall.
        self.w0 = np.log(self.Rstar)
        self.wf = np.log(self.Rstar_inf)
        self.ws = np.log(self.rs)

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

    def get_initial_conditions(self):
        """
        Set the initial condition at the end of melting (melting can be solved analytically
        """

        # --- Initial states --- #
        # (ice temperature)
        self.u0_i = dolfin.interpolate(dolfin.Constant(self.T_inf),self.ice_V)
        self.velocity = dolfin.as_vector([dolfin.Constant(0.),dolfin.Constant(2.)])

        # --- Time Array --- #
        # Now that we have the melt-out time, we can define the time array
        self.ts = np.arange(0.,self.t_final+self.dt,self.dt)/self.t0
        self.dt /= self.t0

        # --- Define the test and trial functions --- #
        self.u_i = dolfin.TrialFunction(self.ice_V)
        self.v_i = dolfin.TestFunction(self.ice_V)
        self.T_i = dolfin.Function(self.ice_V)

        self.flags.append('get_ic')

    def get_boundary_conditions(mod):
        """
        Define Boundary Conditions
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
        mod.bc_iWall = dolfin.DirichletBC(mod.ice_V, mod.Tf, mod.iWall())
        mod.bc_Inf = dolfin.DirichletBC(mod.ice_V, mod.Tstar, mod.Inf())
        mod.bc_Bottom = dolfin.DirichletBC(mod.ice_V, mod.Tstar, mod.Bottom())
        mod.bcs = [mod.bc_iWall,mod.bc_Inf,mod.bc_Bottom]

        mod.flags.append('get_bc')

    # ----------------------------------------------------------------------------------------------------------------------------------------

    def run(self,xs_out,z_out,verbose=False,initialize_array=True):
        """
        Iterate the model through the given time array.
        """

        # Set up the variational form
        alphalog_i = dolfin.project(dolfin.Expression('astar*exp(-2.*x[0])',degree=1,astar=self.astar_i),self.ice_V)
        F_1 = (self.u_i-self.u0_i)*self.v_i*dolfin.dx + \
               self.dt*dolfin.inner(dolfin.grad(self.u_i), dolfin.grad(alphalog_i*self.v_i))*dolfin.dx - \
               self.dt*dolfin.inner(self.velocity,self.u_i.dx(1))*self.v_i*dolfin.dx
        F = dolfin.action(F_1,self.T_i)
        # Compute Jacobian of F
        J = dolfin.derivative(F, self.T_i, self.u_i)
        # Set up the non-linear problem
        problem = dolfin.NonlinearVariationalProblem(F, self.T_i, self.bcs, J=J)
        # Set up the non-linear solver
        solver = dolfin.NonlinearVariationalSolver(problem)
        dolfin.info(solver.parameters, True)

        if initialize_array:
            self.T_ice_result = np.array([[self.u0_i(xi,z_out) for xi in xs_out]]*abs(self.T_inf))
        for t in self.ts:
            if verbose:
                print(round(t*self.t0/60.),end=' min, ')
            (iter, converged) = solver.solve()
            self.u0_i.assign(self.T_i)
            if t%1==0:
                self.T_ice_result = np.append(self.T_ice_result,[[self.T_i(xi,z_out) for xi in xs_out]],axis=0)
            # --- Export --- #
            if t in self.save_times:
                self.T_ice_result = np.append(self.T_ice_result,[self.u0_i.vector()[:]*abs(self.T_inf)],axis=0)

