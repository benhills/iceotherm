#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:39:42 2019

@author: benhills
"""

import numpy as np
from analytical_pure_solution import analyticalMelt
from constants_stefan import constantsIceDiver
from concentration_functions import Tf_depression
const = constantsIceDiver()
import dolfin
dolfin.parameters['allow_extrapolation']=True

class cylindrical_stefan():
    def __init__(self,const=const):
        # Temperature and Concentration Variables
        self.T_inf = -15.                        # Far Field Temperature
        self.Tf = 0.                            # Pure Melting Temperature
        self.Q = 0.0                            # Heat Source after Melting (W)
        self.Q_melt = 2500.                      # Heat Source for Melting (W)
        self.C_init = 0.0                        # Initial Solute Concentration (before injection)
        self.C_inject = 0.2*const.rhoe           # Injection Solute Concentration

        # Domain Specifications
        self.R_center = 0.01                     # Innner Domain Edge
        self.R_inf = 1.                          # Outer Domain Edge
        self.R_melt = 0.04                          # Melt-Out Radius
        self.n = 100
        self.rs = np.linspace(self.R_center,self.R_inf,self.n)
        self.dt = 10.
        self.t_final = 2.*3600.

        # Flags to keep operations in order
        self.flags = []

    # ----------------------------------------------------------------------------------------------------------------------------------------

    def log_transform(self,const=const):
        # Nondimensionalize (Humphrey and Echelmeyer, 1990)
        self.Tstar = self.T_inf/abs(self.T_inf)
        self.Rstar = self.R_melt/self.R_melt
        self.Rstar_center = self.R_center/self.R_melt
        self.Rstar_inf = self.R_inf/self.R_melt
        self.Qstar = self.Q/(2.*np.pi*const.ki*abs(self.T_inf))
        self.astar_i = const.L*const.rhoi/(const.rhoi*const.ci*abs(self.T_inf))
        self.t0 = const.rhoi*const.ci/const.ki*self.astar_i*self.R_center**2.

        # Tranform to a logarithmic coordinate system so that there are more points near the borehole wall.
        self.w0 = np.log(self.Rstar)
        self.wf = np.log(self.Rstar_inf)
        self.w_center = np.log(self.Rstar_center)
        self.ws = np.log(self.rs)

        self.flags.append('log_transform')

    def get_domain(self):
        """
        Define the Finite Element domain for the problem
        """
        # Finite Element Mesh in solid
        self.ice_mesh = dolfin.IntervalMesh(self.n,self.w0,self.wf)
        self.ice_V = dolfin.FunctionSpace(self.ice_mesh,'CG',1)
        self.ice_coords = self.ice_V.tabulate_dof_coordinates()
        self.ice_idx_wall = np.argmin(self.ice_coords)
        # Finite Element Mesh in solution
        if 'solve_sol_temp' in self.flags or 'solve_sol_mol' in self.flags:
            self.sol_mesh = dolfin.IntervalMesh(self.n,self.Rstar_center,self.Rstar)
            self.sol_V = dolfin.FunctionSpace(self.sol_mesh,'CG',1)
            self.sol_V.tabulate_dof_coordinates()[:] = np.log(self.sol_V.tabulate_dof_coordinates())
            self.sol_coords = self.sol_V.tabulate_dof_coordinates()
            self.sol_idx_wall = np.argmax(self.sol_coords)

        self.flags.append('get_domain')

    # ----------------------------------------------------------------------------------------------------------------------------------------

    def get_initial_conditions(self):
        """
        Set the initial condition at the end of melting (melting can be solved analytically
        """

        # --- Initial states --- #
        # (ice temperature)
        self.u0_i = dolfin.Function(self.ice_V)
        T,lam,self.R_melt,self.t_melt = analyticalMelt(np.exp(self.ice_coords[:,0])*self.R_melt,self.T_inf,self.Q_melt,R_target=self.R_melt)
        self.u0_i.vector()[:] = T[:]/abs(self.T_inf)
        if 'solve_sol_temp' in self.flags:
            # (solution temperature),
            self.u0_s = dolfin.Function(self.sol_V)
            T,lam,self.R_melt,self.t_melt = analyticalMelt(np.exp(self.sol_coords[:,0])*self.R_melt,self.T_inf,self.Q_melt,R_target=self.R_melt)
            self.u0_s.vector()[:] = T[:]/abs(self.T_inf)
        if 'solve_sol_mol' in self.flags:
            # (solution concentration),
            self.u0_c = dolfin.interpolate(dolfin.Constant(self.C_init),self.sol_V)

        # --- Time Array --- #
        # Now that we have the melt-out time, we can define the time array
        self.ts = np.arange(self.t_melt,self.t_final+self.dt,self.dt)/self.t0
        self.dt /= self.t0
        self.t_inject = self.ts[np.argmin(abs(self.ts-self.t_inject/self.t0))]
        self.save_times = self.ts[::10]

        # --- Define the test and trial functions --- #
        self.u_i = dolfin.TrialFunction(self.ice_V)
        self.v_i = dolfin.TestFunction(self.ice_V)
        self.T_i = dolfin.Function(self.ice_V)
        if 'solve_sol_temp' in self.flags:
            self.u_s = dolfin.TrialFunction(self.sol_V)
            self.v_s = dolfin.TestFunction(self.sol_V)
            self.T_s = dolfin.Function(self.sol_V)
        if 'solve_sol_mol' in self.flags:
            self.C = dolfin.Function(self.sol_V)
        else:
            self.C = self.C_init
            # Get the updated solution properties
            self.rhos = self.C + const.rhow*(1.-self.C/const.rhoe)
            self.cs = (self.C/const.rhoe)*const.ce+(1.-(self.C/const.rhoe))*const.cw

        self.flags.append('get_ic')

    def get_boundary_conditions(mod):
        """
        ### Define Boundary Conditions ###
        """
        # Left boundary is the center of the borehole, so it is at the melting temperature
        class iWall(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and x[0] < mod.ice_coords[mod.ice_idx_wall] + const.tol
        # Right boundary is the far-field temperature
        class Inf(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                    return on_boundary and x[0] > mod.wf - const.tol

        if 'solve_sol_temp' in mod.flags:
            # Liquid boundary condition at hole wall (same temperature as ice)
            class sWall(dolfin.SubDomain):
                def inside(self, x, on_boundary):
                    return on_boundary and x[0] > mod.sol_coords[mod.sol_idx_wall] - const.tol
            # center flux
            class center(dolfin.SubDomain):
                def inside(self, x, on_boundary):
                    return on_boundary and x[0] < mod.w_center + const.tol

        # Initialize boundary classes
        mod.iWall = iWall()
        mod.Inf = Inf()
        if 'solve_sol_temp' in mod.flags:
            mod.sWall = sWall()
            mod.center = center()

        # Set the Dirichlet Boundary condition at
        mod.bc_inf = dolfin.DirichletBC(mod.ice_V, mod.Tstar, mod.Inf)

        # Identify flux boundary for hole wall
        if 'solve_sol_mol' in mod.flags:
            # This will be used in the boundary condition for mass diffusion
            mod.boundaries = dolfin.MeshFunction("size_t", mod.sol_mesh, 0) # this index 0 is an alternative to the command boundaries.set_all(0)
            mod.sWall.mark(mod.boundaries, 1)
            mod.center.mark(mod.boundaries, 2)
            mod.sds = dolfin.Measure("ds")(subdomain_data=mod.boundaries)

        mod.flags.append('get_bc')

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def update_boundary_conditions(self):
        # Update the thermal boundary condition at the wall based on the current concentration
        # update the melting temperature
        self.Tf = Tf_depression(self.C)
        self.Tf /= abs(self.T_inf)
        if 'solve_sol_mol' in self.flags:
            Tf_wall = dolfin.project(self.Tf,self.sol_V).vector()[self.sol_idx_wall]
        else:
            Tf_wall = self.Tf
        # Reset ice boundary condition
        self.bc_iWall = dolfin.DirichletBC(self.ice_V, Tf_wall, self.iWall)
        # Reset solution boundary condition
        if 'solve_sol_temp' in self.flags:
            self.bc_sWall = dolfin.DirichletBC(self.sol_V, Tf_wall, self.sWall)


    """

    def solve_molecular(self,t):
        ### Inject ethanol
        if abs(t*self.t0-self.t_inject) < const.tol:
            self.u0_c.vector()[:] = self.C_inject
            self.inject_mass = self.C_inject*(np.exp(self.Rstar)**2.-np.exp(self.R_center)**2.)
            print('Inject!')
            # add thermal sink from mixing
            self.thermalSink()

        if (t*self.t0-const.tol) >= self.t_inject:
            # Solve solution concentration
            # Diffusivity
            self.Lewis = const.ki/(const.rhoi*const.ci*const.mol_diff)
            Dlog_c = dolfin.project(dolfin.Expression('mol_diff*exp(-2.*x[0])',mol_diff=self.astar_i/self.Lewis),self.sol_V)
            # calculate solute flux
            Cwall = self.u0_c(self.sol_coords[-1])
            Dwall = Dlog_c(self.sol_coords[-1])
            solFlux = dolfin.Constant(np.exp(self.sol_coords[-1][0])*(Cwall/Dwall)*(self.dR/self.dt))
            # Variational Problem
            F_c = (self.u_s-self.u0_c)*self.v_s*dolfin.dx + \
                    self.dt*dolfin.inner(dolfin.grad(self.u_s), dolfin.grad(Dlog_c*self.v_s))*dolfin.dx + \
                    self.dt*solFlux*self.v_s*self.sds(1)
            a_c = dolfin.lhs(F_c)
            L_c = dolfin.rhs(F_c)
            dolfin.solve(a_c==L_c,self.C)
            self.u0_c.assign(self.C)
    """

    def solve_thermal(self):

        ### Solve heat equation
        alphalog_i = dolfin.project(dolfin.Expression('astar*exp(-2.*x[0])',degree=1,astar=self.astar_i),self.ice_V)
        # Set up the variational form for the current mesh location
        F_i = (self.u_i-self.u0_i)*self.v_i*dolfin.dx + self.dt*dolfin.inner(dolfin.grad(self.u_i), dolfin.grad(alphalog_i*self.v_i))*dolfin.dx
        #F_i -= dt*(Qsource*t0/abs(T_inf)/(const.rhoi*const.ci))*v_i*dx
        a_i = dolfin.lhs(F_i)
        L_i = dolfin.rhs(F_i)
        # Solve ice temperature
        dolfin.solve(a_i==L_i,self.T_i,[self.bc_inf,self.bc_iWall])

        self.u0_i.assign(self.T_i)

        """

        # Source in solution
        # TODO: change to solution constants
        #diff_ratio = (const.kw*const.rhoi*const.ci)/(const.ki*const.rhow*const.cw)
        #alphalog_s = dolfin.project(dolfin.Expression('astar*exp(-2.*x[0])',astar=self.astar_i*diff_ratio),self.sol_V)

        F_s = (self.u_s-self.u0_s)*self.v_s*dolfin.dx + self.dt*dolfin.inner(dolfin.grad(self.u_s), dolfin.grad(alphalog_s*self.v_s))*dolfin.dx
        #F_s -= dt*(Qsource*t0/abs(T_inf)/(const.rhow*const.cw))*v_s*dx
        # Center heat flux
        F_s -= (self.Qwater/(const.kw*diff_ratio*2.*np.pi*abs(self.T_inf)))*self.v_s*self.sds(2)
        a_s = dolfin.lhs(F_s)
        L_s = dolfin.rhs(F_s)

        # Solve solution temperature
        dolfin.solve(a_s==L_s,self.T_s,self.bc_sWall)
        self.u0_s.assign(self.T_s)

        """


    def move_wall(self,const=const):
        """
        # Melting/Freezing at the hole wall
        """

        # --- Calculate Distance Wall Moves --- #
        if 'solve_sol_mol' in self.flags:
            return
        else:
            # calculate sensible heat constibution toward wall melting associated with change in the freezing temp
            self.Tf = Tf_depression(self.C)
            self.Tf_last = Tf_depression(self.C_last)
            self.dR = np.sqrt(((self.rhos*self.cs*(self.Tf_last-self.Tf)*self.Rstar**2.)/\
                    (const.rhow*const.L))+self.Rstar**2.)-self.Rstar

        # melting/freezing at the hole wall from prescribed flux and temperature gradient
        # Humphrey and Echelmeyer (1990) eq. 13
        self.dR += dolfin.project(dolfin.Expression('exp(-x[0])',degree=1)*self.u0_i.dx(0),self.ice_V).vector()[self.ice_idx_wall] + \
                    self.Qstar/self.Rstar
        if 'solve_sol_temp' in self.flags:
            # TODO: change kw to ks
            self.dR += -(const.kw/const.ki)*dolfin.project(dolfin.Expression('exp(-x[0])',degree=1)*self.u0_s.dx(0),self.sol_V).vector()[self.sol_idx_wall]
        self.dR *= self.dt

        # Is the hole completely frozen? If so, exit
        Frozen = np.exp(self.ice_coords[self.ice_idx_wall,0])+self.dR < 0.
        if Frozen:
            self.flags.append('Frozen')
            return

        # --- Move the Mesh --- ###

        # stretch mesh rather than uniform displacement
        dRsi = self.dR/(self.Rstar_inf-self.Rstar)*(self.Rstar_inf-np.exp(self.ice_coords[:,0]))
        # Interpolate the points onto what will be the new mesh (ice)
        self.u0_i.vector()[:] = np.array([self.u0_i(xi) for xi in np.log(np.exp(self.ice_coords[:,0])+dRsi)])
        # advect the mesh according to the movement of the hole wall
        dolfin.ALE.move(self.ice_mesh,dolfin.Expression('std::log(exp(x[0])+dRsi*(Rstar_inf-exp(x[0])))-x[0]',degree=1,dRsi=self.dR/(self.Rstar_inf-self.Rstar),Rstar_inf=self.Rstar_inf))
        self.ice_mesh.bounding_box_tree().build(self.ice_mesh)
        self.ice_coords = self.ice_V.tabulate_dof_coordinates()

        if 'solve_sol_temp' in self.flags:
            # stretch mesh rather than uniform displacement
            dRss = self.dR/(self.Rstar-self.Rstar_center)*(np.exp(self.sol_coords[:,0])-self.Rstar_center)
            # Interpolate the points onto what will be the new mesh (solution)
            self.u0_s.vector()[:] = np.array([self.u0_s(xi) for xi in np.log(np.exp(self.sol_coords[:,0])+dRss)])
            self.u0_c.vector()[:] = np.array([self.u0_c(xi) for xi in np.log(np.exp(self.sol_coords[:,0])+dRss)])
            # advect the mesh according to the movement of teh hole wall
            dolfin.ALE.move(self.sol_mesh,dolfin.Expression('std::log(exp(x[0])+dRs*(exp(x[0])-Rstar_center))-x[0]',
                dRs=self.dR/(self.Rstar-self.Rstar_center),Rstar_center=self.Rstar_center))
            self.sol_mesh.bounding_box_tree().build(self.sol_mesh)

        # --- Recalculate Solution Concentration and Properties --- #

        if 'solve_sol_mol' in self.flags:
            ### Diffusivities
            #rhos = Expression('u0_c + rhow*(1.-u0_c/rhoe)',u0_c=u0_c,rhow=const.rhow,rhoe=const.rhoe)
            #cs = Expression('u0_c + cw*(1.-u0_c/ce)',u0_c=u0_c,cw=const.cw,ce=const.ce)
            return
        else:
            # Instantaneous Mixing
            # Recalculate the solution concentration after wall moves
            self.C_last = self.C
            #TODO: figure out the units here
            self.C *= (self.Rstar/np.exp(self.ice_coords[self.ice_idx_wall,0]))**2.
            # Get the updated solution properties
            self.rhos = self.C + const.rhow*(1-self.C/const.rhoe)
            self.cs = (self.C/self.rhos)*const.ce+(1.-(self.C/self.rhos))*const.cw

        # Save the new wall location
        self.Rstar = np.exp(self.ice_coords[self.ice_idx_wall,0])

    # ----------------------------------------------------------------------------------------------------------------------------------------
    """
    def conservation(self,const=const):
        x_exp = dolfin.Expression('pow(exp(x[0]),2)')
        #print "Conservation: ", const.cw*const.rhow*assemble(x_exp*u0_s*dx)+\
                                #const.ci*const.rhoi*assemble(x_exp*u0_i*dx)# - En_init
        if (self.t*self.t0-const.tol) >= self.t_inject:
            print("Mass conserveation: ", dolfin.assemble(x_exp*u0_c*dx))# - inject_mass


        self.En = self.Tstar*(np.exp(self.Rstar_inf)**2.-np.exp(self.Rstar)**2.)
    """

    def run(self):
        ### Iterate ###

        self.r_ice_result = [np.exp(self.ice_coords[:,0])*self.R_melt]
        self.T_result = [np.array(self.u0_i.vector()[:]*abs(self.T_inf))]
        if 'solve_sol_temp' in self.flags or 'solve_sol_mol' in self.flags:
            self.r_ice_result = [np.exp(self.ice_coords[:,0])*self.R_melt]
        for t in self.ts[1:]:
            print(t*self.t0/60.)

            # --- Injection --- #
            if t == self.t_inject:
                self.C = self.C_inject

            # --- Move the Mesh --- #
            self.move_wall()
            # break the loop if the hole is completely frozen
            if 'Frozen' in self.flags:
                print('Frozen Hole!')
                break

            # --- Boundary Conditions --- #
            self.update_boundary_conditions()

            # --- Molecular Diffusion --- #
            #if t >= (self.t_inject - self.tol):
            #    self.solve_molecular(t)

            # --- Thermal Diffusion --- #
            self.solve_thermal()

            ### TODO: Freezing in hole
            # Hard reset on temps below Tf
            # Ice concentration

            # --- Export --- #
            if t in self.save_times:
                self.r_ice_result = np.append(self.r_ice_result,[np.exp(self.ice_coords[:,0])*self.R_melt],axis=0)
                self.T_result = np.append(self.T_result,[self.u0_i.vector()[:]*abs(self.T_inf)],axis=0)
                if 'solve_sol_temp' in self.flags or 'solve_sol_mol' in self.flags:
                    self.r_sol_result = np.append(self.r_ice_result,[np.exp(self.sol_coords[:,0])*self.R_melt],axis=0)
