#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:39:42 2019

@author: benhills
"""

import numpy as np
from analytical_pure_solution import analyticalMelt
from constants_stefan import constantsIceDiver
from concentration_functions import Tf_depression,Hmix
const = constantsIceDiver()
import dolfin

class cylindrical_stefan():
    def __init__(self,const=const):
        # Temperature and Concentration Variables
        self.T_inf = -15.                        # Far Field Temperature
        self.Tf = 0.                            # Pure Melting Temperature
        self.Q_wall = 0.0                            # Heat Source after Melting (W)
        self.Q_initialize = 2500.                      # Heat Source for Melting (W)
        self.Q_center = 0.0                     # line source at borehole center (W/m?? TODO)
        self.Q_sol = 0.0
        self.C_init = 0.0                        # Initial Solute Concentration (before injection)
        self.C_inject = 0.2*const.rhoe           # Injection Solute Concentration
        self.mol_diff = 1.24e-9

        # Domain Specifications
        self.R_center = 0.001                     # Innner Domain Edge
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
        self.Qstar = self.Q_wall/(2.*np.pi*const.ki*abs(self.T_inf))
        self.astar_i = const.L*const.rhoi/(const.rhoi*const.ci*abs(self.T_inf))
        self.t0 = const.rhoi*const.ci/const.ki*self.astar_i*self.R_melt**2.

        # Dimensionless Constants
        self.St = const.ci*(self.Tf-self.T_inf)/const.L
        self.Lewis = const.ki/(const.rhoi*const.ci*self.mol_diff)

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
        self.ice_coords = self.ice_V.tabulate_dof_coordinates().copy()
        self.ice_idx_wall = np.argmin(self.ice_coords)
        # Finite Element Mesh in solution
        if 'solve_sol_temp' in self.flags or 'solve_sol_mol' in self.flags:
            self.sol_mesh = dolfin.IntervalMesh(self.n,self.Rstar_center,self.Rstar)
            self.sol_V = dolfin.FunctionSpace(self.sol_mesh,'CG',1)
            self.sol_mesh.coordinates()[:] = np.log(self.sol_mesh.coordinates())
            self.sol_coords = self.sol_V.tabulate_dof_coordinates().copy()
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
        T,lam,self.R_melt,self.t_melt = analyticalMelt(np.exp(self.ice_coords[:,0])*self.R_melt,self.T_inf,self.Q_initialize,R_target=self.R_melt)
        self.u0_i.vector()[:] = T/abs(self.T_inf)
        if 'solve_sol_temp' in self.flags:
            # (solution temperature),
            self.u0_s = dolfin.Function(self.sol_V)
            T,lam,self.R_melt,self.t_melt = analyticalMelt(np.exp(self.sol_coords[:,0])*self.R_melt,self.T_inf,self.Q_initialize,R_target=self.R_melt)
            self.u0_s.vector()[:] = T/abs(self.T_inf)
        if 'solve_sol_mol' in self.flags:
            # (solution concentration),
            self.u0_c = dolfin.interpolate(dolfin.Constant(self.C_init),self.sol_V)

        # --- Time Array --- #
        # Now that we have the melt-out time, we can define the time array
        self.ts = np.arange(self.t_melt,self.t_final+self.dt,self.dt)/self.t0
        self.dt /= self.t0
        self.t_inject = self.ts[np.argmin(abs(self.ts-self.t_inject/self.t0))]

        # --- Define the test and trial functions --- #
        self.u_i = dolfin.TrialFunction(self.ice_V)
        self.v_i = dolfin.TestFunction(self.ice_V)
        self.T_i = dolfin.Function(self.ice_V)
        if 'solve_sol_temp' in self.flags or 'solve_sol_mol' in self.flags:
            self.u_s = dolfin.TrialFunction(self.sol_V)
            self.v_s = dolfin.TestFunction(self.sol_V)
        if 'solve_sol_temp' in self.flags:
            self.T_s = dolfin.Function(self.sol_V)
        if 'solve_sol_mol' in self.flags:
            self.C = dolfin.Function(self.sol_V)
            self.Tf = Tf_depression(self.C)
            self.Tf /= abs(self.T_inf)
            self.Tf_wall = dolfin.project(self.Tf,self.sol_V).vector()[self.sol_idx_wall]
            # Get the updated solution properties
            self.rhos = dolfin.project(dolfin.Expression('C + rhow*(1.-C/rhoe)',degree=1,C=self.C,rhow=const.rhow,rhoe=const.rhoe),self.sol_V)
            self.cs = dolfin.project(dolfin.Expression('ce*(C/rhoe) + cw*(1.-C/rhoe)',degree=1,C=self.C,cw=const.cw,ce=const.ce,rhoe=const.rhoe),self.sol_V)
            self.ks = dolfin.project(dolfin.Expression('ke*(C/rhoe) + kw*(1.-C/rhoe)',degree=1,C=self.C,kw=const.kw,ke=const.ke,rhoe=const.rhoe),self.sol_V)
            self.rhos_wall = self.rhos.vector()[self.sol_idx_wall]
            self.cs_wall = self.cs.vector()[self.sol_idx_wall]
            self.ks_wall = self.ks.vector()[self.sol_idx_wall]
        else:
            self.C = self.C_init
            self.Tf_wall = Tf_depression(self.C)
            # Get the updated solution properties
            self.rhos = self.C + const.rhow*(1.-self.C/const.rhoe)
            self.cs = (self.C/const.rhoe)*const.ce+(1.-(self.C/const.rhoe))*const.cw
            self.ks = (self.C/const.rhoe)*const.ke+(1.-(self.C/const.rhoe))*const.kw
            self.rhos_wall,self.cs_wall,self.ks_wall = self.rhos,self.cs,self.ks

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

        # Initialize boundary classes
        mod.iWall = iWall()
        mod.Inf = Inf()
        # Set the Dirichlet Boundary condition at
        mod.bc_inf = dolfin.DirichletBC(mod.ice_V, mod.Tstar, mod.Inf)

        if 'solve_sol_temp' in mod.flags or 'solve_sol_mol' in mod.flags:
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

    # ----------------------------------------------------------------------------------------------------------------------------------------
    def update_boundary_conditions(self):
        # Update the thermal boundary condition at the wall based on the current concentration
        # update the melting temperature
        self.Tf = Tf_depression(self.C)
        self.Tf /= abs(self.T_inf)
        if 'solve_sol_mol' in self.flags:
            self.Tf_wall = dolfin.project(self.Tf,self.sol_V).vector()[self.sol_idx_wall]
        else:
            self.Tf_wall = self.Tf
        # Reset ice boundary condition
        self.bc_iWall = dolfin.DirichletBC(self.ice_V, self.Tf_wall, self.iWall)
        # Reset solution boundary condition
        if 'solve_sol_temp' in self.flags:
            self.bc_sWall = dolfin.DirichletBC(self.sol_V, self.Tf_wall, self.sWall)

    def solve_molecular(self):
        """

        """
        if 'solve_sol_mol' in self.flags:
            # Solve solution concentration
            # Diffusivity
            self.D = dolfin.project(dolfin.Expression('diff_ratio*exp(-2.*x[0])',degree=1,diff_ratio=self.astar_i/self.Lewis),self.sol_V)
            #L = dolfin.Expression('Lewis*(1-.005*pow(max(maxx-x[0],0.01),-1))',degree=1,
            #                      Lewis=self.Lewis,maxx=np.nanmax(self.sol_coords))
            #self.D = dolfin.project(dolfin.Expression('astar_i/L*exp(-2.*x[0])',degree=1,
            #                                     astar_i=self.astar_i,L=L),self.sol_V)
            # calculate solute flux
            Cwall = self.u0_c(self.sol_coords[self.sol_idx_wall])
            Dwall = self.D(self.sol_coords[self.sol_idx_wall])
            solFlux = -dolfin.Constant(np.exp(self.sol_coords[self.sol_idx_wall,0])*(Cwall/Dwall)*(self.dR/self.dt))
            # Variational Problem
            F_c = (self.u_s-self.u0_c)*self.v_s*dolfin.dx + \
                    self.dt*dolfin.inner(dolfin.grad(self.u_s), dolfin.grad(self.D*self.v_s))*dolfin.dx - \
                    self.dt*solFlux*self.v_s*self.sds(1)
            a_c = dolfin.lhs(F_c)
            L_c = dolfin.rhs(F_c)
            dolfin.solve(a_c==L_c,self.C)
            self.u0_c.assign(self.C)
            # Recalculate the freezing temperature
            self.Tf_last = self.Tf
            self.Tf = Tf_depression(self.C.vector()[self.sol_idx_wall,0])
            # Get the updated solution properties
            self.rhos = dolfin.project(dolfin.Expression('C + rhow*(1.-C/rhoe)',degree=1,C=self.C,rhow=const.rhow,rhoe=const.rhoe),self.sol_V)
            self.cs = dolfin.project(dolfin.Expression('ce*(C/rhoe) + cw*(1.-C/rhoe)',degree=1,C=self.C,cw=const.cw,ce=const.ce,rhoe=const.rhoe),self.sol_V)
            self.ks = dolfin.project(dolfin.Expression('ke*(C/rhoe) + kw*(1.-C/rhoe)',degree=1,C=self.C,kw=const.kw,ke=const.ke,rhoe=const.rhoe),self.sol_V)
            self.rhos_wall = self.rhos.vector()[self.sol_idx_wall]
            self.cs_wall = self.cs.vector()[self.sol_idx_wall]
            self.ks_wall = self.ks.vector()[self.sol_idx_wall]
        else:
            # Instantaneous Mixing
            # Recalculate the solution concentration after wall moves
            self.C_last = self.C
            self.C *= (self.Rstar/np.exp(self.ice_coords[self.ice_idx_wall,0]))**2.
            # Recalculate the freezing temperature
            self.Tf_last = self.Tf
            self.Tf = Tf_depression(self.C)
            # Get the updated solution properties
            self.rhos = self.C + const.rhow*(1-self.C/const.rhoe)
            self.cs = (self.C/const.rhoe)*const.ce+(1.-(self.C/const.rhoe))*const.cw
            self.ks = (self.C/const.rhoe)*const.ke+(1.-(self.C/const.rhoe))*const.kw
            self.rhos_wall,self.cs_wall,self.ks_wall = self.rhos,self.cs,self.ks


    def solve_thermal(self):

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

        if 'solve_sol_temp' in self.flags:
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

    def thermalSink(self):
        """
        # Energy source based on enthalpy of mixing
        # TODO: more robust checks on this
        """
        # enthalpy of mixing
        dHmix = Hmix(self.C_inject)-Hmix(self.C_init)
        # energy source (J m-3 s-1)
        phi = dHmix*1000.*self.C_inject/(self.dt*const.mmass_e)
        # convert energy source to temperature change
        dTmix = phi/(2.*np.pi*self.Rstar**2.*self.rhos*self.cs)
        return dTmix

    def injection(self):
        self.Tf_last = Tf_depression(self.C)
        if 'solve_sol_mol' in self.flags:
            self.u0_c.vector()[:] = self.C_inject
            self.u0_s.vector()[:] = Tf_depression(self.C_inject)/abs(self.T_inf)
            # TODO: add thermal sink from mixing
            #self.inject_mass = self.C_inject*(np.exp(self.Rstar)**2.-np.exp(self.R_center)**2.)
        else:
            self.C = self.C_inject
            # TODO: add thermal sink from mixing
            #self.thermalSink()


    def move_wall(self,const=const):
        """
        # Melting/Freezing at the hole wall
        """

        # --- Calculate Distance Wall Moves --- #
        # calculate sensible heat contribution toward wall melting associated with change in the freezing temp
        if 'solve_sol_mol' in self.flags:
            # TODO: fix this movement, it shouldn't be 0.
            self.dR = 0.
        else:
            self.dR = np.sqrt(((self.rhos_wall*self.cs_wall*(self.Tf_last-self.Tf)*self.Rstar**2.)/(const.rhow*const.L))+self.Rstar**2.)-self.Rstar

        # melting/freezing at the hole wall from prescribed flux and temperature gradient
        # Humphrey and Echelmeyer (1990) eq. 13
        self.dR += dolfin.project(dolfin.Expression('exp(-x[0])',degree=1)*self.u0_i.dx(0),self.ice_V).vector()[self.ice_idx_wall] + \
                    self.Qstar/self.Rstar
        if 'solve_sol_temp' in self.flags:
            self.dR -= (self.ks_wall/const.ki)*dolfin.project(dolfin.Expression('exp(-x[0])',degree=1)*self.u0_s.dx(0),self.sol_V).vector()[self.sol_idx_wall]
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
        self.idx_ice = np.exp(self.ice_coords[:,0]) + self.dR >= self.Rstar
        u0_i_hold = self.u0_i.vector()[:].copy()
        u0_i_hold[self.idx_ice] = np.array([self.u0_i(xi) for xi in np.log(np.exp(self.ice_coords[self.idx_ice,0])+dRsi[self.idx_ice])])
        u0_i_hold[~self.idx_ice] = self.Tf_wall
        self.u0_i.vector()[:] = u0_i_hold[:]
        # advect the mesh according to the movement of the hole wall
        dolfin.ALE.move(self.ice_mesh,dolfin.Expression('std::log(exp(x[0])+dRsi*(Rstar_inf-exp(x[0])))-x[0]',degree=1,dRsi=self.dR/(self.Rstar_inf-self.Rstar),Rstar_inf=self.Rstar_inf))
        self.ice_mesh.bounding_box_tree().build(self.ice_mesh)
        self.ice_coords = self.ice_V.tabulate_dof_coordinates().copy()

        if 'solve_sol_temp' in self.flags or 'solve_sol_mol' in self.flags:
            # stretch mesh rather than uniform displacement
            dRss = self.dR/(self.Rstar-self.Rstar_center)*(np.exp(self.sol_coords[:,0])-self.Rstar_center)
            # Interpolate the points onto what will be the new mesh (solution)
            self.idx_sol = np.exp(self.sol_coords[:,0]) + self.dR <= self.Rstar
            if 'solve_sol_temp' in self.flags:
                u0_s_hold = self.u0_s.vector()[:].copy()
                u0_s_hold[self.idx_sol] = np.array([self.u0_s(xi) for xi in np.log(np.exp(self.sol_coords[self.idx_sol,0])+dRss[self.idx_sol])])
                u0_s_hold[~self.idx_sol] = self.Tf_wall
                self.u0_s.vector()[:] = u0_s_hold[:]
            if 'solve_sol_mol' in self.flags:
                u0_c_hold = self.u0_c.vector()[:].copy()
                u0_c_hold[self.idx_sol] = np.array([self.u0_c(xi) for xi in np.log(np.exp(self.sol_coords[self.idx_sol,0])+dRss[self.idx_sol])])
                u0_c_hold[~self.idx_sol] = 0. #TODO: test this
                self.u0_c.vector()[:] = u0_c_hold[:]
            # advect the mesh according to the movement of teh hole wall
            dolfin.ALE.move(self.sol_mesh,dolfin.Expression('std::log(exp(x[0])+dRs*(exp(x[0])-Rstar_center))-x[0]',
                degree=1,dRs=self.dR/(self.Rstar-self.Rstar_center),Rstar_center=self.Rstar_center))
            self.sol_mesh.bounding_box_tree().build(self.sol_mesh)
            self.sol_coords = self.sol_V.tabulate_dof_coordinates().copy()

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

    def run(self,verbose=False):
        ### Iterate ###

        self.r_ice_result = [np.exp(self.ice_coords[:,0])*self.R_melt]
        self.T_ice_result = [np.array(self.u0_i.vector()[:]*abs(self.T_inf))]
        if 'solve_sol_temp' in self.flags or 'solve_sol_mol' in self.flags:
            self.r_sol_result = [np.exp(self.sol_coords[:,0])*self.R_melt]
        if 'solve_sol_temp' in self.flags:
            self.T_sol_result = [np.array(self.u0_s.vector()[:]*abs(self.T_inf))]
        if 'solve_sol_mol' in self.flags:
            self.Tf_result = [Tf_depression(np.array(self.u0_c.vector()[:]))]
        for t in self.ts[1:]:
            if verbose:
                print(round(t*self.t0/60.),end='min , ')

            # --- Ethanol Injection --- #
            if t == self.t_inject:
                self.injection()

            # --- Move the Mesh --- #
            self.move_wall()
            # break the loop if the hole is completely frozen
            if 'Frozen' in self.flags:
                print('Frozen Hole!')
                break

            # --- Boundary Conditions --- #
            self.update_boundary_conditions()

            # --- Molecular Diffusion --- #
            if t >= (self.t_inject):
                self.solve_molecular()

            # --- Thermal Diffusion --- #
            self.solve_thermal()

            ### TODO: Freezing in hole
            # Hard reset on temps below Tf
            # Ice concentration

            # Save the new wall location
            self.Rstar = np.exp(self.ice_coords[self.ice_idx_wall,0])

            # --- Export --- #
            if t in self.save_times:
                self.r_ice_result = np.append(self.r_ice_result,[np.exp(self.ice_coords[:,0])*self.R_melt],axis=0)
                self.T_ice_result = np.append(self.T_ice_result,[self.u0_i.vector()[:]*abs(self.T_inf)],axis=0)
                if 'solve_sol_temp' in self.flags or 'solve_sol_mol' in self.flags:
                    self.r_sol_result = np.append(self.r_sol_result,[np.exp(self.sol_coords[:,0])*self.R_melt],axis=0)
                if 'solve_sol_temp' in self.flags:
                    self.T_sol_result = np.append(self.T_sol_result,[self.u0_s.vector()[:]*abs(self.T_inf)],axis=0)
                if 'solve_sol_mol' in self.flags:
                    self.Tf_result = np.append(self.Tf_result,[Tf_depression(self.u0_c.vector()[:])],axis=0)
