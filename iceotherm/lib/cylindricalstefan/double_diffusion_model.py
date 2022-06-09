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
April 22, 2020
"""

import numpy as np

from ..constants import constantsHotPointDrill
const = constantsHotPointDrill()
from iceotherm.lib.cylindricalstefan.concentration_functions import Tf_depression,Hmix
from iceotherm.lib.cylindricalstefan.analytical_pure_solution import analyticalMelt

try:
    import dolfin
    fe_enabled = True
except ImportError:
    fe_enabled = False

# -----------------------------------------------------------------------------------------------------

class double_diffusion_model():
    """

    *** This model is still in development.
    *** It has issues with numerical stability.

    This is a 1-dimensional thermal model for borehole evolution.
    The hole melts and freezes according to the Stefan condition.
    As opposed to instantaneous_mixing_model, in which the antifreeze
    concentration is always uniform throughout the solution, this model
    tries to diagnose slush formation more directly by directly modeling
    both thermal and molecular diffusion within the solution. Locations
    where the solution temperature drops below the liquidus line are
    what Worster (2000) calls 'constitutional supercooling' (i.e. slush).

    The problem is solved in cylindrical coordinates with a logarithmic transform,
    following Humphrey and Echelmeyer (1990).

    The finite element mesh is over the domain of ice, from the borehole wall out
    to some distance where the temperature can safely be assumed constant.
    The mesh stretches (shrinks) to maintain its coverage of the ice domain as the
    borehole wall freezes (melts).

    """

    def __init__(self,const=const):
        """
        Initial Variables
        """

        # Temperature Variables
        self.T_inf = -20.                       # Far Field Temperature
        self.Tf = 0.                            # Pure Melting Temperature
        self.Tf_reg = 1.0                       # Regularization for thermal boundary condition at hole wall
        self.Q_wall = 0.0                       # Heat Source after Melting (W)
        self.Q_initialize = 2500.               # Heat Source for Melting (W)
        self.Q_center = 0.0                     # Line source at borehole center (W/m?? TODO: think about how this is implemented)
        self.Q_sol = 0.0                        # Solution source term (W/m?? TODO: think about how this is implemented)

        # Concentration Variables
        self.source_timing = np.inf             # Time at which antifreeze is added to the solution
        self.source_duration = np.inf           # Half-width of the gaussian source
        self.C_init = 0.0                       # Initial Solute Concentration (before injection) (kg m-3)
        self.source_mass_final = 1.             # Total Injection Mass (kg)
        self.mol_diff = 0.84e-9                 # Molecular diffusivity

        # Domain Specifications
        self.R_center = 0.001                   # Innner Domain Edge (m)
        self.R_inf = 1.                         # Outer Domain Edge (m)
        self.R_melt = 0.04                      # Melt-Out Radius (m)
        self.n = 100                            # Mesh resolution
        self.dt = 10.                           # Time step (s)
        self.t_final = 2.*3600.                 # End simulation time (s)

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
        self.Tstar = self.T_inf/abs(self.T_inf)                                 # Dimensionless temperature
        self.Rstar = self.R_melt/self.R_melt                                    # Dimensionless melt-out radius
        self.Rstar_center = self.R_center/self.R_melt                           # Dimensionless center radius
        self.Rstar_inf = self.R_inf/self.R_melt                                 # Dimensionless outer domain edge
        self.Qstar = self.Q_wall/(2.*np.pi*const.ki*abs(self.T_inf))            # Dimensionless heat source for melting
        Lv = const.L*const.rhoi                                                 # Latent heat of fusion per unit volume
        self.astar_i = Lv/(const.rhoi*const.ci*abs(self.T_inf))                 # Thermal diffusivity of ice
        self.t0 = const.rhoi*const.ci/const.ki*self.astar_i*self.R_melt**2.     # Characteristic time (~freeze time)

        # Dimensionless Constants
        self.St = const.ci*(self.Tf-self.T_inf)/const.L                         # Stefan number
        self.Lewis = const.ki/(const.rhoi*const.ci*self.mol_diff)               # Lewis number

        # Tranform to a logarithmic coordinate system so that there are more points near the borehole wall.
        self.w0 = np.log(self.Rstar)                                            # Log dimensionless melt-out radius
        self.wf = np.log(self.Rstar_inf)                                        # Log dimensionless outer domain edge
        self.w_center = np.log(self.Rstar_center)                               # Log dimensionless domain center

        self.flags.append('log_transform')

    def get_domain(self):
        """
        Define the Finite Element domain for the problem
        """

        # Finite Element Mesh in solid
        self.ice_mesh = dolfin.IntervalMesh(self.n,self.w0,self.wf)
        self.ice_V = dolfin.FunctionSpace(self.ice_mesh,'CG',1)
        self.ice_coords = self.ice_V.tabulate_dof_coordinates().copy()
        self.ice_idx_wall = np.argmin(self.ice_coords)
        # Finite Element Mesh in solution
        self.sol_mesh = dolfin.IntervalMesh(self.n,self.Rstar_center,self.Rstar)
        self.sol_V = dolfin.FunctionSpace(self.sol_mesh,'CG',1)
        self.sol_mesh.coordinates()[:] = np.log(self.sol_mesh.coordinates())
        self.sol_coords = self.sol_V.tabulate_dof_coordinates().copy()
        self.sol_idx_wall = np.argmax(self.sol_coords)

        self.flags.append('get_domain')

    # ----------------------------------------------------------------------------------------------------------------------------------------

    def get_initial_conditions(self,rho_solute=const.rhom):
        """
        Set the initial condition at the end of melting (melting can be solved analytically
        """

        # --- Initial states --- #
        # ice temperature
        self.u0_i = dolfin.Function(self.ice_V)
        T,lam,self.R_melt,self.t_melt = analyticalMelt(np.exp(self.ice_coords[:,0])*self.R_melt,self.T_inf,self.Q_initialize,R_target=self.R_melt)
        self.u0_i.vector()[:] = T/abs(self.T_inf)
        # solution temperature
        self.u0_s = dolfin.Function(self.sol_V)
        T,lam,self.R_melt,self.t_melt = analyticalMelt(np.exp(self.sol_coords[:,0])*self.R_melt,self.T_inf,self.Q_initialize,R_target=self.R_melt)
        self.u0_s.vector()[:] = T/abs(self.T_inf)
        # solution concentration
        self.u0_c = dolfin.interpolate(dolfin.Constant(self.C_init),self.sol_V)

        # --- Time Array --- #
        # Now that we have the melt-out time, we can define the time array
        self.ts = np.arange(self.t_melt,self.t_final+self.dt,self.dt)/self.t0
        self.dt /= self.t0
        # Define the ethanol source
        self.source_timing = self.ts[np.argmin(abs(self.ts-self.source_timing/self.t0))]
        if 'gaussian_source' in self.flags:
            self.source_duration /= self.t0
            self.source = self.source_mass_final/(self.source_duration*np.sqrt(np.pi))*np.exp(-((self.ts-self.source_timing)/self.source_duration)**2.)
        else:
            self.source = np.zeros_like(self.ts)
            self.source[np.argmin(abs(self.ts-self.source_timing))] = self.source_mass_final/self.dt

        # --- Define the test and trial functions --- #
        self.u_i = dolfin.TrialFunction(self.ice_V)
        self.v_i = dolfin.TestFunction(self.ice_V)
        self.T_i = dolfin.Function(self.ice_V)
        self.u_s = dolfin.TrialFunction(self.sol_V)
        self.v_s = dolfin.TestFunction(self.sol_V)
        self.T_s = dolfin.Function(self.sol_V)
        self.C = dolfin.Function(self.sol_V)
        self.Tf = Tf_depression(self.C,linear=True)/abs(self.T_inf)
        self.Tf_wall = dolfin.project(self.Tf,self.sol_V).vector()[self.sol_idx_wall]
        # Get the updated solution properties
        self.rhos = dolfin.project(dolfin.Expression('C + rhow*(1.-C/rho)',degree=1,C=self.C,rhow=const.rhow,rho=rho_solute),self.sol_V)
        self.cs = dolfin.project(dolfin.Expression('ce*(C/rho) + cw*(1.-C/rho)',degree=1,C=self.C,cw=const.cw,ce=const.ce,rho=rho_solute),self.sol_V)
        self.ks = dolfin.project(dolfin.Expression('ke*(C/rho) + kw*(1.-C/rho)',degree=1,C=self.C,kw=const.kw,ke=const.ke,rho=rho_solute),self.sol_V)
        self.rhos_wall = self.rhos.vector()[self.sol_idx_wall]
        self.cs_wall = self.cs.vector()[self.sol_idx_wall]
        self.ks_wall = self.ks.vector()[self.sol_idx_wall]

        self.flags.append('get_ic')

    def get_boundary_conditions(mod):
        """
        Define Boundary Conditions
        """

        # Left boundary is the center of the borehole, so it is at the melting temperature
        class iWall(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[0] < mod.ice_coords[mod.ice_idx_wall] + const.tol
        # Right boundary is the far-field temperature
        class Inf(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                    return on_boundary and x[0] > mod.wf - const.tol

        # Initialize boundary classes
        mod.iWall = iWall()
        mod.Inf = Inf()
        # Set the Dirichlet Boundary condition at
        mod.bc_inf = dolfin.DirichletBC(mod.ice_V, mod.Tstar, mod.Inf)

        # Liquid boundary condition at hole wall (same temperature as ice)
        class sWall(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[0] > mod.sol_coords[mod.sol_idx_wall] - const.tol
        # center flux
        class center(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[0] < mod.w_center + const.tol

        # Initialize boundary classes
        mod.sWall = sWall()
        mod.center = center()

        # This will be used in the boundary condition for mass diffusion
        mod.boundaries = dolfin.MeshFunction("size_t", mod.sol_mesh, 0) # this index 0 is an alternative to the command boundaries.set_all(0)
        mod.sWall.mark(mod.boundaries, 1)
        mod.center.mark(mod.boundaries, 2)
        mod.sds = dolfin.Measure("ds")(subdomain_data=mod.boundaries)

        mod.flags.append('get_bc')


    def initiate_solution_diffusion(mod,rho_solute=const.rhom):
        """
        Before the equations can be solved for the liquid solution: initial conditions and solution properties need to be setup
            update domain
            time array and ethanol source timing
            initial conditions
            variational problem
            solution properties
            boundary conditions
        To be run once after get_initial_conditions() and get_boundary_conditions()
        """

        # --- Get new domain --- #
        mod.sol_mesh = dolfin.IntervalMesh(mod.n,mod.Rstar_center,mod.Rstar)
        mod.sol_V = dolfin.FunctionSpace(mod.sol_mesh,'CG',1)
        mod.sol_mesh.coordinates()[:] = np.log(mod.sol_mesh.coordinates())
        mod.sol_coords = mod.sol_V.tabulate_dof_coordinates().copy()
        mod.sol_idx_wall = np.argmax(mod.sol_coords)

        # --- Time array --- #
        # Now that we have the melt-out time, we can define the time array
        mod.ts = np.arange(mod.t_init,mod.t_final+mod.dt,mod.dt)/mod.t0
        mod.dt /= mod.t0
        # Define the ethanol source
        mod.source_timing = mod.source_timing/mod.t0
        mod.source_duration /= mod.t0
        mod.source = mod.source_mass_final/(mod.source_duration*np.sqrt(np.pi))*np.exp(-((mod.ts-mod.source_timing)/mod.source_duration)**2.)

        # --- Set initial conditions --- #
        # solution temperature
        mod.u0_s = dolfin.Function(mod.sol_V)
        mod.u0_s.vector()[:] = mod.Tf_wall
        # solution concentration
        mod.u0_c = dolfin.project(dolfin.Constant(mod.C),mod.sol_V)

        # --- Set up the variational Problem --- #
        mod.u_s = dolfin.TrialFunction(mod.sol_V)
        mod.v_s = dolfin.TestFunction(mod.sol_V)
        mod.T_s = dolfin.Function(mod.sol_V)
        mod.C = dolfin.project(dolfin.Constant(mod.C),mod.sol_V)
        mod.Tf = Tf_depression(mod.C,linear=True)/abs(mod.T_inf)
        mod.Tf_wall = dolfin.project(mod.Tf,mod.sol_V).vector()[mod.sol_idx_wall]

        # --- Get the solution properties --- #
        mod.rhos = dolfin.project(dolfin.Expression('C + rhow*(1.-C/rho_solute)',degree=1,C=mod.C,rhow=const.rhow,rho_solute=rho_solute),mod.sol_V)
        mod.cs = dolfin.project(dolfin.Expression('ce*(C/rho_solute) + cw*(1.-C/rho_solute)',degree=1,C=mod.C,cw=const.cw,ce=const.ce,rho_solute=rho_solute),mod.sol_V)
        mod.ks = dolfin.project(dolfin.Expression('ke*(C/rho_solute) + kw*(1.-C/rho_solute)',degree=1,C=mod.C,kw=const.kw,ke=const.ke,rho_solute=rho_solute),mod.sol_V)
        mod.rhos_wall = mod.rhos.vector()[mod.sol_idx_wall]
        mod.cs_wall = mod.cs.vector()[mod.sol_idx_wall]
        mod.ks_wall = mod.ks.vector()[mod.sol_idx_wall]

        # --- Boundary Conditions --- #
        # Liquid boundary condition at hole wall (same temperature as ice)
        class sWall(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[0] > mod.sol_coords[mod.sol_idx_wall] - const.tol
        # center flux
        class center(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[0] < mod.w_center + const.tol

        # Initialize boundary classes
        mod.sWall = sWall()
        mod.center = center()
        # This will be used in the boundary condition for mass diffusion
        mod.boundaries = dolfin.MeshFunction("size_t", mod.sol_mesh, 0) # this index 0 is an alternative to the command boundaries.set_all(0)
        mod.sWall.mark(mod.boundaries, 1)
        mod.center.mark(mod.boundaries, 2)
        mod.sds = dolfin.Measure("ds")(subdomain_data=mod.boundaries)

    # ----------------------------------------------------------------------------------------------------------------------------------------

    def update_boundary_conditions(self):
        """
        Update the boundary conditions for new wall location and new wall temperature.
        """

        # Update the thermal boundary condition at the wall based on the current concentration
        Tf_wall_last = self.Tf_wall
        self.Tf = Tf_depression(self.C,linear=True)/abs(self.T_inf)
        self.C_wall = dolfin.project(self.C,self.sol_V).vector()[self.sol_idx_wall]
        Tf_wall_hold = Tf_depression(self.C_wall,linear=True)/abs(self.T_inf)
        self.Tf_wall = Tf_wall_last + (Tf_wall_hold-Tf_wall_last)*self.Tf_reg
        # Reset ice boundary condition
        self.bc_iWall = dolfin.DirichletBC(self.ice_V, self.Tf_wall, self.iWall)
        # Reset solution boundary condition
        self.bc_sWall = dolfin.DirichletBC(self.sol_V, self.Tf_wall, self.sWall)

    def solve_molecular(self,rho_solute=const.rhom):
        """
        Solve the molecular diffusion problem
        if 'solve_sol_mol' is not in the flags list this will be an instantaneous mixing problem.
        """

        # Solve solution concentration
        # Diffusivity
        ramp = dolfin.Expression('x[0] > minR ? 100. : 1.',minR=np.log(self.Rstar)-0.5,degree=1)
        self.Dstar = dolfin.project(dolfin.Expression('ramp*diff_ratio*exp(-2.*x[0])',degree=1,ramp=ramp,diff_ratio=self.astar_i/self.Lewis),self.sol_V)
        # calculate solute flux (this is like the Stefan condition for the molecular diffusion problem
        Dwall = self.Dstar(self.sol_coords[self.sol_idx_wall])
        self.solFlux = -(self.C_wall/Dwall)*(self.dR/self.dt)
        # Set the concentration for points that moved out of the grid to match the solute flux
        u0_c_hold = self.u0_c.vector()[:].copy()
        u0_c_hold[~self.sol_idx_extrapolate] = self.C_wall+self.solFlux*(np.exp(self.sol_coords[~self.sol_idx_extrapolate,0])-self.Rstar)
        u0_c_hold[u0_c_hold<0.]=0.
        self.u0_c.vector()[:] = u0_c_hold[:]
        # Variational Problem
        F_c = (self.u_s-self.u0_c)*self.v_s*dolfin.dx + \
                self.dt*dolfin.inner(dolfin.grad(self.u_s), dolfin.grad(self.Dstar*self.v_s))*dolfin.dx - \
                self.dt*self.solFlux*self.v_s*self.sds(1)
        F_c = dolfin.action(F_c,self.C)
        # First derivative
        J = dolfin.derivative(F_c, self.C, self.u_s)
        # handle the bounds
        lower = dolfin.project(dolfin.Constant(0.0),self.sol_V)
        upper = dolfin.project(dolfin.Constant(rho_solute),self.sol_V)
        # set bounds and solve
        snes_solver_parameters = {"nonlinear_solver": "snes",
                                  "snes_solver": {"linear_solver": "lu",
                                                  "maximum_iterations": 20,
                                                  "report": True,
                                                  "error_on_nonconvergence": False}}
        problem = dolfin.NonlinearVariationalProblem(F_c, self.C, J=J)
        problem.set_bounds(lower, upper)
        solver = dolfin.NonlinearVariationalSolver(problem)
        solver.parameters.update(snes_solver_parameters)
        dolfin.info(solver.parameters, True)
        (iter, converged) = solver.solve()
        self.u0_c.assign(self.C)

        # Recalculate the freezing temperature
        self.Tf_last = self.Tf
        self.Tf = Tf_depression(self.C.vector()[self.sol_idx_wall,0],linear=True)
        # Get the updated solution properties
        self.rhos = dolfin.project(dolfin.Expression('C + rhow*(1.-C/rho_solute)',
            degree=1,C=self.C,rhow=const.rhow,rho_solute=rho_solute),self.sol_V)
        self.cs = dolfin.project(dolfin.Expression('ce*(C/rho_solute) + cw*(1.-C/rho_solute)',
            degree=1,C=self.C,cw=const.cw,ce=const.ce,rho_solute=rho_solute),self.sol_V)
        self.ks = dolfin.project(dolfin.Expression('ke*(C/rho_solute) + kw*(1.-C/rho_solute)',
            degree=1,C=self.C,kw=const.kw,ke=const.ke,rho_solute=rho_solute),self.sol_V)
        self.rhos_wall = self.rhos.vector()[self.sol_idx_wall]
        self.cs_wall = self.cs.vector()[self.sol_idx_wall]
        self.ks_wall = self.ks.vector()[self.sol_idx_wall]


    def solve_thermal(self):
        """
        Solve the thermal diffusion problem.
        Both ice and solution.
        """

        ### Solve heat equation
        alphalog_i = dolfin.project(dolfin.Expression('astar*exp(-2.*x[0])',degree=1,astar=self.astar_i),self.ice_V)
        # Set up the variational form for the current mesh location
        F_i = (self.u_i-self.u0_i)*self.v_i*dolfin.dx + self.dt*dolfin.inner(dolfin.grad(self.u_i), dolfin.grad(alphalog_i*self.v_i))*dolfin.dx
        a_i = dolfin.lhs(F_i)
        L_i = dolfin.rhs(F_i)
        # Solve ice temperature
        dolfin.solve(a_i==L_i,self.T_i,[self.bc_inf,self.bc_iWall])
        # Update previous profile to current
        self.u0_i.assign(self.T_i)

        diff_ratio = (self.ks*const.rhoi*const.ci)/(const.ki*self.rhos*self.cs)
        alphalog_s = dolfin.project(dolfin.Expression('astar*exp(-2.*x[0])',degree=1,astar=self.astar_i),self.sol_V)
        # Set up the variational form for the current mesh location
        F_s = (self.u_s-self.u0_s)*self.v_s*dolfin.dx + self.dt*dolfin.inner(dolfin.grad(self.u_s), dolfin.grad(alphalog_s*diff_ratio*self.v_s))*dolfin.dx
                #- self.dt*(self.Q_sol*self.t0/abs(self.T_inf)/(const.rhoi*const.ci))*self.v_s*dolfin.dx #TODO: check this solution source term
        # TODO: Center heat flux
        #F_s -= (self.Q_center/(self.ks*diff_ratio*2.*np.pi*abs(self.T_inf)))*self.v_s*self.sds(2)
        a_s = dolfin.lhs(F_s)
        L_s = dolfin.rhs(F_s)
        # Solve solution temperature
        dolfin.solve(a_s==L_s,self.T_s,self.bc_sWall)
        # Update previous profile to current
        self.u0_s.assign(self.T_s)

    def injection_energy_balance(self,source):
        """
        Update the concentration and temperature at the time of injection.
        """

        # Calculate the injection source actually adds to total concentration
        C_inject = source*self.dt/(np.pi*(self.Rstar*self.R_melt)**2.)

        # Hard set on concentration (assume that it mixes quickly)
        self.C.vector()[:] += C_inject
        self.u0_c.vector()[:] += C_inject

        # enthalpy of mixing, always exothermic so gives off energy (J m-3)
        H,phi = Hmix(C_inject)
        # put this added energy toward uniformly warming the solution
        #phi_dT = -phi/(self.rhos*self.cs)/abs(self.T_inf)
        phi_dT = -phi/(const.rhow*const.cw)/abs(self.T_inf)

        # Hard set on solution temperature (assume that the mixing energy spreads evenly)
        self.u0_s.vector()[:] += phi_dT#dolfin.project(phi_dT,self.sol_V).vector()[:]

    def move_wall(self,const=const):
        """
        Calculate the amount of melting/freezing at the hole wall
        This is the Stefan condition.
        """

        # --- Calculate Distance Wall Moves --- #

        # Melting/freezing at the hole wall from prescribed flux and temperature gradient
        # Humphrey and Echelmeyer (1990) eq. 13
        dRdt = dolfin.project(dolfin.Expression('exp(-x[0])',degree=1)*self.u0_i.dx(0),self.ice_V).vector()[self.ice_idx_wall] + \
                    self.Qstar/self.Rstar
        # second half of the Stefan condition
        dRdt -= (self.ks_wall/const.ki)*dolfin.project(dolfin.Expression('exp(-x[0])',degree=1)*self.u0_s.dx(0),self.sol_V).vector()[self.sol_idx_wall]
        self.dR = dRdt*self.dt

        # Is the hole completely frozen? If so, exit
        Frozen = np.exp(self.ice_coords[self.ice_idx_wall,0])+self.dR < 0.
        if Frozen:
            self.flags.append('Frozen')
            return

        # --- Move the Mesh --- ###

        # stretch mesh rather than uniform displacement
        dRsi = self.dR/(self.Rstar_inf-self.Rstar)*(self.Rstar_inf-np.exp(self.ice_coords[:,0]))
        # Interpolate the points onto what will be the new mesh (ice)
        self.ice_idx_extrapolate = np.exp(self.ice_coords[:,0]) + self.dR >= self.Rstar
        u0_i_hold = self.u0_i.vector()[:].copy()
        u0_i_hold[self.ice_idx_extrapolate] = np.array([self.u0_i(xi) for xi in np.log(np.exp(self.ice_coords[self.ice_idx_extrapolate,0])+dRsi[self.ice_idx_extrapolate])])
        u0_i_hold[~self.ice_idx_extrapolate] = self.Tf_wall
        self.u0_i.vector()[:] = u0_i_hold[:]
        # advect the mesh according to the movement of the hole wall
        dolfin.ALE.move(self.ice_mesh,dolfin.Expression('std::log(exp(x[0])+dRsi*(Rstar_inf-exp(x[0])))-x[0]',degree=1,dRsi=self.dR/(self.Rstar_inf-self.Rstar),Rstar_inf=self.Rstar_inf))
        self.ice_mesh.bounding_box_tree().build(self.ice_mesh)
        self.ice_coords = self.ice_V.tabulate_dof_coordinates().copy()

        # stretch mesh rather than uniform displacement
        dRss = self.dR/(self.Rstar-self.Rstar_center)*(np.exp(self.sol_coords[:,0])-self.Rstar_center)
        # Interpolate the points onto what will be the new mesh (solution)
        self.sol_idx_extrapolate = np.exp(self.sol_coords[:,0]) + self.dR <= self.Rstar
        u0_s_hold = self.u0_s.vector()[:].copy()
        u0_s_hold[self.sol_idx_extrapolate] = np.array([self.u0_s(xi) for xi in np.log(np.exp(self.sol_coords[self.sol_idx_extrapolate,0])+dRss[self.sol_idx_extrapolate])])
        u0_s_hold[~self.sol_idx_extrapolate] = self.Tf_wall
        self.u0_s.vector()[:] = u0_s_hold[:]
        u0_c_hold = self.u0_c.vector()[:].copy()
        u0_c_hold[self.sol_idx_extrapolate] = np.array([self.u0_c(xi) for xi in np.log(np.exp(self.sol_coords[self.sol_idx_extrapolate,0])+dRss[self.sol_idx_extrapolate])])
        u0_c_hold[~self.sol_idx_extrapolate] = np.nan
        self.u0_c.vector()[:] = u0_c_hold[:]
        # advect the mesh according to the movement of teh hole wall
        dolfin.ALE.move(self.sol_mesh,dolfin.Expression('std::log(exp(x[0])+dRs*(exp(x[0])-Rstar_center))-x[0]',
            degree=1,dRs=self.dR/(self.Rstar-self.Rstar_center),Rstar_center=self.Rstar_center))
        self.sol_mesh.bounding_box_tree().build(self.sol_mesh)
        self.sol_coords = self.sol_V.tabulate_dof_coordinates().copy()

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def conservation(self,const=const):
        """
        Integrate the temeprature profile for checking that the model conserves energy
        Integrate the concentration profile for checking that the model conserves mass
        """

        x_exp = dolfin.Expression('pow(exp(x[0]),2)',degree=1)
        self.EnCon = dolfin.assemble(self.cs*self.rhos*x_exp*self.u0_s*dolfin.dx)+\
                        const.ci*const.rhoi*dolfin.assemble(x_exp*self.u0_i*dolfin.dx)
        self.MassCon = dolfin.assemble(x_exp*self.u0_c*dolfin.dx)

    def run(self,verbose=False,initialize_array=True):
        """
        Iterate the model through the given time array.
        """

        if initialize_array:
            self.r_ice_result = [np.exp(self.ice_coords[:,0])*self.R_melt]
            self.T_ice_result = [np.array(self.u0_i.vector()[:]*abs(self.T_inf))]
            self.r_sol_result = [np.exp(self.sol_coords[:,0])*self.R_melt]
            self.T_sol_result = [np.array(self.u0_s.vector()[:]*abs(self.T_inf))]
            self.Tf_result = [Tf_depression(np.array(self.u0_c.vector()[:]),linear=True)]
        for i,t in enumerate(self.ts[1:]):
            if verbose:
                print(round(t*self.t0/60.),end=' min, ')

            # --- Ethanol Injection --- #
            self.injection_energy_balance(self.source[i+1])

            # --- Thermal Diffusion --- #
            self.update_boundary_conditions()
            self.solve_thermal()

            # --- Move the Mesh --- #
            self.move_wall()
            # break the loop if the hole is completely frozen
            if 'Frozen' in self.flags:
                print('Frozen Hole!')
                break

            # --- Molecular Diffusion --- #
            self.solve_molecular()

            # Save the new wall location
            self.Rstar = np.exp(self.ice_coords[self.ice_idx_wall,0])

            # --- Export --- #
            if t in self.save_times:
                self.r_ice_result = np.append(self.r_ice_result,[np.exp(self.ice_coords[:,0])*self.R_melt],axis=0)
                self.T_ice_result = np.append(self.T_ice_result,[self.u0_i.vector()[:]*abs(self.T_inf)],axis=0)
                self.r_sol_result = np.append(self.r_sol_result,[np.exp(self.sol_coords[:,0])*self.R_melt],axis=0)
                self.T_sol_result = np.append(self.T_sol_result,[self.u0_s.vector()[:]*abs(self.T_inf)],axis=0)
                self.Tf_result = np.append(self.Tf_result,[Tf_depression(self.u0_c.vector()[:],linear=True)],axis=0)
