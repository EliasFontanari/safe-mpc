import numpy as np
import re
from copy import deepcopy
import scipy.linalg as lin
from casadi import MX, vertcat, norm_2, Function, Opti, integrator
from acados_template import AcadosModel, AcadosSim, AcadosSimSolver, AcadosOcp, AcadosOcpSolver, AcadosMultiphaseOcp
import torch
import torch.nn as nn
import l4casadi as l4c


class NeuralNetDIR(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetDIR, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.linear_relu_stack(x)
        return out


class AbstractModel:
    def __init__(self, params):
        self.params = params
        self.amodel = AcadosModel()
        # Dummy dynamics (double integrator)
        self.amodel.name = "double_integrator"
        self.x = MX.sym("x")
        self.x_dot = MX.sym("x_dot")
        self.u = MX.sym("u")
        self.f_expl = self.u
        self.p = MX.sym("p")
        self.addDynamicsModel(params)
        self.amodel.x = self.x
        self.amodel.xdot = self.x_dot
        self.amodel.u = self.u
        self.amodel.f_expl_expr = self.f_expl
        self.amodel.p = self.p

        self.nx = self.amodel.x.size()[0]
        self.nu = self.amodel.u.size()[0]
        self.ny = self.nx + self.nu
        self.nq = int(self.nx / 2)
        self.nv = self.nx - self.nq

        # Joint limits
        self.u_min = -params.u_max * np.ones(self.nu)
        self.u_max = params.u_max * np.ones(self.nu)
        self.x_min = np.hstack([params.q_min * np.ones(self.nq), -params.dq_max * np.ones(self.nq)])
        self.x_max = np.hstack([params.q_max * np.ones(self.nq), params.dq_max * np.ones(self.nq)])

        # Target
        self.x_ref = np.zeros(self.nx)
        self.x_ref[:self.nq] = np.pi
        self.x_ref[params.joint_target] = params.q_max - params.ubound_gap

        # NN model (viability constraint)
        self.l4c_model = None
        self.nn_model = None
        self.nn_func = None

        self.bounds_tol =params.bounds_tol

    def addDynamicsModel(self, params):
        pass

    def checkStateConstraints(self, x):
        return np.all(np.logical_and(x >= self.x_min-self.bounds_tol, x <= self.x_max+self.bounds_tol))

    def checkControlConstraints(self, u):
        return np.all(np.logical_and(u >= self.u_min-1e-4, u <= self.u_max+1e-4))

    def checkRunningConstraints(self, x, u):
        return self.checkStateConstraints(x) and self.checkControlConstraints(u)

    def checkSafeConstraints(self, x):
        return self.nn_func(x, self.params.alpha) >= 0. 

    def setNNmodel(self):
        device = torch.device('cpu')
        model = NeuralNetDIR(self.nx, (self.nx - 1) * 100, 1).to(device)
        model.load_state_dict(torch.load(self.params.NN_DIR + 'model.zip', map_location=device))
        mean = torch.load(self.params.NN_DIR + 'mean.zip')
        std = torch.load(self.params.NN_DIR + 'std.zip')

        x_cp = deepcopy(self.x)
        x_cp[self.nq] += self.params.eps
        vel_norm = norm_2(x_cp[self.nq:])
        pos = (x_cp[:self.nq] - mean) / std
        vel_dir = x_cp[self.nq:] / vel_norm
        state = vertcat(pos, vel_dir)

        # i = 0
        # out = state
        # weights = list(model.parameters())
        # for weight in weights:
        #     weight = MX(np.array(weight.tolist()))
        #     if i % 2 == 0:
        #         out = weight @ out
        #     else:
        #         out = weight + out
        #         if i == 1 or i == 3:
        #             out = fmax(0., out)
        #     i += 1
        # self.nn_model = out * (100 - self.p) / 100 - vel_norm

        self.l4c_model = l4c.L4CasADi(model,
                                      device='cpu',
                                      name=self.amodel.name + '_model',
                                      build_dir=self.params.GEN_DIR + 'nn_' + self.amodel.name)
        self.nn_model = self.l4c_model(state) * (100 - self.p) / 100 - vel_norm
        self.nn_func = Function('nn_func', [self.x, self.p], [self.nn_model])


class SimDynamics:
    def __init__(self, model):
        self.model = model
        self.params = model.params
        sim = AcadosSim()
        sim.model = model.amodel
        sim.solver_options.T = self.params.dt_s
        sim.solver_options.integrator_type = self.params.integrator_type
        sim.solver_options.num_stages = self.params.num_stages
        sim.parameter_values = np.array([0.])
        gen_name = self.params.GEN_DIR + '/sim_' + sim.model.name
        sim.code_export_directory = gen_name
        self.sim=sim
        self.integrator = AcadosSimSolver(sim, build=self.params.regenerate, json_file=gen_name + '.json')
        
    def simulate(self, x, u):
        self.integrator.set("x", x)
        self.integrator.set("u", u)
        self.integrator.solve()
        x_next = self.integrator.get("x")
        return x_next

    def checkDynamicsConstraints(self, x, u):
        # Rollout the control sequence
        n = np.shape(u)[0]
        x_sim = np.zeros((n + 1, self.model.nx))
        x_sim[0] = np.copy(x[0])
        for i in range(n):
            x_sim[i + 1] = self.simulate(x_sim[i], u[i])
        # Check if the rollout state trajectory is almost equal to the optimal one
        return np.linalg.norm(x - x_sim) < self.params.state_tol * np.sqrt(n+1) 
    
    def checkSafeIntegrate(self, x, u, n_safe):
        x_sim = x[0]
        for i in range(n_safe):
            x_sim = self.simulate(x_sim,u[i])
            if not(self.model.checkStateConstraints(x_sim)):
                return False, None
        return self.model.nn_func(x_sim, self.params.alpha) >= 0. , x_sim 
    
class AbstractController:
    def __init__(self, simulator):
        self.ocp_name = "".join(re.findall('[A-Z][^A-Z]*', self.__class__.__name__)[:-1]).lower()
        self.simulator = simulator
        self.params = simulator.params
        self.model = simulator.model
        self.err_thr=self.params.err_thr

        self.N = int(self.params.T / self.params.dt)
        self.ocp = AcadosOcp()

        # Dimensions
        self.ocp.solver_options.tf = self.params.T
        self.ocp.dims.N = self.N

        # Model
        self.ocp.model = self.model.amodel

        # Cost
        self.Q = 1e-4 * np.eye(self.model.nx)
        self.Q[0, 0] = 5e2
        self.R = 1e-4 * np.eye(self.model.nu)

        self.ocp.cost.W = lin.block_diag(self.Q, self.R)
        self.ocp.cost.W_e = self.Q

        self.ocp.cost.cost_type = "LINEAR_LS"
        self.ocp.cost.cost_type_e = "LINEAR_LS"

        self.ocp.cost.Vx = np.zeros((self.model.ny, self.model.nx))
        self.ocp.cost.Vx[:self.model.nx, :self.model.nx] = np.eye(self.model.nx)
        self.ocp.cost.Vu = np.zeros((self.model.ny, self.model.nu))
        self.ocp.cost.Vu[self.model.nx:, :self.model.nu] = np.eye(self.model.nu)
        self.ocp.cost.Vx_e = np.eye(self.model.nx)

        self.ocp.cost.yref = np.zeros(self.model.ny)
        self.ocp.cost.yref_e = np.zeros(self.model.nx)
        # Set alpha to zero as default
        self.ocp.parameter_values = np.array([0.])

        # Constraints
        self.ocp.constraints.lbx_0 = self.model.x_min
        self.ocp.constraints.ubx_0 = self.model.x_max
        self.ocp.constraints.idxbx_0 = np.arange(self.model.nx)

        self.ocp.constraints.lbu = self.model.u_min
        self.ocp.constraints.ubu = self.model.u_max
        self.ocp.constraints.idxbu = np.arange(self.model.nu)
        self.ocp.constraints.lbx = self.model.x_min
        self.ocp.constraints.ubx = self.model.x_max
        self.ocp.constraints.idxbx = np.arange(self.model.nx)

        self.ocp.constraints.lbx_e = self.model.x_min
        self.ocp.constraints.ubx_e = self.model.x_max
        self.ocp.constraints.idxbx_e = np.arange(self.model.nx)

        # Solver options
        self.ocp.solver_options.nlp_solver_type = self.params.solver_type
        self.ocp.solver_options.hpipm_mode = self.params.solver_mode
        self.ocp.solver_options.nlp_solver_max_iter = self.params.nlp_max_iter
        self.ocp.solver_options.qp_solver_iter_max = self.params.qp_max_iter
        self.ocp.solver_options.globalization = self.params.globalization
        #self.ocp.solver_options.tol=self.params.tol

        
        # if self.params.cont_type== 'abort':
        #     self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        #     self.ocp.solver_options.nlp_solver_tol_eq = 5e-4
        #     self.ocp.solver_options.nlp_solver_tol_ineq = 5e-4
        
        if self.params.rti:
            self.ocp.solver_options.levenberg_marquardt = self.params.lm_reg

        # Additional settings, in general is an empty method
        self.additionalSetting()

        self.gen_name = self.params.GEN_DIR + 'ocp_' + self.ocp_name + '_' + self.model.amodel.name
        self.ocp.code_export_directory = self.gen_name
        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file=self.gen_name + '.json', build=self.params.regenerate) 

        # Initialize guess
        self.fails = 0
        self.x_ref = np.zeros(self.model.nx)

        # Empty initial guess and temp vectors
        self.x_guess = np.zeros((self.N + 1, self.model.nx))
        self.u_guess = np.zeros((self.N, self.model.nu))
        self.x_temp, self.u_temp = np.copy(self.x_guess), np.copy(self.u_guess)

        # Viable state (None for Naive and ST controllers)
        self.x_viable = None

        self.solver_integrator = AcadosSimSolver(self.ocp,json_file=self.gen_name +'2.json',build=self.params.regenerate,verbose=False)

        
        # Time stats
        self.time_fields = ['time_lin', 'time_sim', 'time_qp', 'time_qp_solver_call',
                            'time_glob', 'time_reg', 'time_tot']
        
        
    def reinit_solver(self):
        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file=self.gen_name +'.json', build=True,verbose=False)

    def simulate_solver(self, x, u):
        self.solver_integrator.set("x", x)
        self.solver_integrator.set("u", u)
        self.solver_integrator.solve()
        x_next = self.solver_integrator.get("x")
        return x_next
    
    def additionalSetting(self):
        pass

    def terminalConstraint(self, soft=True):
        self.model.setNNmodel()
        self.model.amodel.con_h_expr_e = self.model.nn_model

        self.ocp.solver_options.model_external_shared_lib_dir = self.model.l4c_model.shared_lib_dir
        self.ocp.solver_options.model_external_shared_lib_name = self.model.l4c_model.name

        self.ocp.constraints.lh_e = np.array([0.])
        self.ocp.constraints.uh_e = np.array([1e6])

        if soft:
            self.ocp.constraints.idxsh_e = np.array([0])

            self.ocp.cost.zl_e = np.ones((1,)) * self.params.ws_t
            self.ocp.cost.zu_e = np.zeros((1,))
            self.ocp.cost.Zl_e = np.zeros((1,))
            self.ocp.cost.Zu_e = np.zeros((1,))

    def runningConstraint(self, soft=True):
        # Suppose that the NN model is already set (same for external model shared lib)
        self.model.amodel.con_h_expr = self.model.nn_model

        self.ocp.constraints.lh = np.array([0.])
        self.ocp.constraints.uh = np.array([1e6])

        if soft:
            self.ocp.constraints.idxsh = np.array([0])

            # Set zl initially to zero, then apply receding constraint in the step method
            self.ocp.cost.zl = np.zeros((1,))
            self.ocp.cost.zu = np.zeros((1,))
            self.ocp.cost.Zl = np.zeros((1,))
            self.ocp.cost.Zu = np.zeros((1,))

    def solve(self, x0):
        # Reset current iterate
        self.ocp_solver.reset()

        # Constrain initial state
        self.ocp_solver.constraints_set(0, "lbx", x0)
        self.ocp_solver.constraints_set(0, "ubx", x0)

        y_ref = np.zeros(self.model.ny)
        y_ref[:self.model.nx] = self.x_ref
        W = lin.block_diag(self.Q, self.R)

        for i in range(self.N):
            self.ocp_solver.set(i, 'x', self.x_guess[i])
            self.ocp_solver.set(i, 'u', self.u_guess[i])
            self.ocp_solver.cost_set(i, 'yref', y_ref, api='new')
            self.ocp_solver.cost_set(i, 'W', W, api='new')
        self.ocp_solver.set(0,'x',x0)

        self.ocp_solver.set(self.N, 'x', self.x_guess[-1])
        self.ocp_solver.cost_set(self.N, 'yref', y_ref[:self.model.nx], api='new')
        self.ocp_solver.cost_set(self.N, 'W', self.Q, api='new')

        # Solve the OCP
        status = self.ocp_solver.solve()

        # Save the temporary solution, independently of the status
        for i in range(self.N):
            self.x_temp[i] = self.ocp_solver.get(i, "x")
            self.u_temp[i] = self.ocp_solver.get(i, "u")
        self.x_temp[-1] = self.ocp_solver.get(self.N, "x")

        return status

    def provideControl(self):
        """ Save the guess for the next MPC step and return u_opt[0] """
        if self.fails > 0:
            u = self.u_guess[0]
            # Rollback the previous guess
            #self.guessCorrection()
            self.x_guess = np.roll(self.x_guess, -1, axis=0)
            self.u_guess = np.roll(self.u_guess, -1, axis=0)
            if 'receding' in self.params.cont_type or self.params.cont_type== 'parallel2' or self.params.cont_type== 'parallel_limited':
                self.x_guess[-1] = self.simulator.simulate(self.x_guess[-2],self.u_guess[-2])
        else:
            u = self.u_temp[0]
            #self.x_guess = np.copy(self.x_temp)
            #self.guessCorrection()
            # Save the current temporary solution
            self.x_guess = np.roll(self.x_temp, -1, axis=0)
            self.u_guess = np.roll(self.u_temp, -1, axis=0)
            if 'receding' in self.params.cont_type or self.params.cont_type== 'parallel2' or self.params.cont_type== 'parallel_limited':
                self.guessCorrection()
                      
        # Copy the last values        
        if not('receding' in self.params.cont_type)  and self.params.cont_type != 'parallel2' and self.params.cont_type != 'parallel_limited':
            self.x_guess[-1] = np.copy(self.x_guess[-2])
        self.u_guess[-1] = np.copy(self.u_guess[-2])
        return u

    def step(self, x0):
        pass

    def setQRWeights(self, Q, R):
        self.Q = Q
        self.R = R

    def setReference(self, x_ref):
        self.x_ref = x_ref

    def getTime(self):
        return np.array([self.ocp_solver.get_stats(field) for field in self.time_fields])

    def setGuess(self, x_guess, u_guess):
        self.x_guess = x_guess
        self.u_guess = u_guess

    def getGuess(self):
        return np.copy(self.x_guess), np.copy(self.u_guess)

    def getLastViableState(self):
        return np.copy(self.x_viable)
    

    def guessCorrection(self):
        error = 0
        x_corrected = np.copy(self.x_guess) 
        for i in range(len(self.u_guess)):
            x_corrected[i+1]=self.simulator.simulate(x_corrected[i],self.u_guess[i])
            error += np.linalg.norm(np.abs(self.x_guess[i+1]-x_corrected[i+1]))
        if np.linalg.norm(self.x_guess - x_corrected) > self.err_thr* np.sqrt(self.N+1):
            #print(error)
            self.x_guess = x_corrected
    
class AbstractControllerSingle:
    def __init__(self, simulator):
        self.ocp_name = "".join(re.findall('[A-Z][^A-Z]*', self.__class__.__name__)[:-1]).lower()
        self.simulator = simulator
        self.model = simulator.model ###
        self.params = simulator.params
        self.err_thr=self.params.err_thr
        if 'parallel' in self.params.cont_type:
            self.parallel = True
        else: 
            self.parallel = None

        

        self.N = int(self.params.T / self.params.dt)
        self.solvers = []
        self.create_solvers()

    def create_solvers(self):
        # First solver:
        self.ocp = AcadosOcp()

        # Dimensions
        self.ocp.solver_options.tf = self.params.T
        self.ocp.dims.N = self.N
        
        self.model = deepcopy(self.simulator.model)
        # Model
        self.ocp.model = deepcopy(self.simulator.model.amodel)

        # Cost
        self.Q = 1e-4 * np.eye(self.model.nx)
        self.Q[0, 0] = 5e2
        self.R = 1e-4 * np.eye(self.model.nu)

        self.ocp.cost.W = lin.block_diag(self.Q, self.R)
        self.ocp.cost.W_e = self.Q

        self.ocp.cost.cost_type = "LINEAR_LS"
        self.ocp.cost.cost_type_e = "LINEAR_LS"

        self.ocp.cost.Vx = np.zeros((self.model.ny, self.model.nx))
        self.ocp.cost.Vx[:self.model.nx, :self.model.nx] = np.eye(self.model.nx)
        self.ocp.cost.Vu = np.zeros((self.model.ny, self.model.nu))
        self.ocp.cost.Vu[self.model.nx:, :self.model.nu] = np.eye(self.model.nu)
        self.ocp.cost.Vx_e = np.eye(self.model.nx)

        self.ocp.cost.yref = np.zeros(self.model.ny)
        self.ocp.cost.yref_e = np.zeros(self.model.nx)
        # Set alpha to zero as default
        self.ocp.parameter_values = np.array([0.])

        # Constraints
        self.ocp.constraints.lbx_0 = self.model.x_min
        self.ocp.constraints.ubx_0 = self.model.x_max
        self.ocp.constraints.idxbx_0 = np.arange(self.model.nx)

        self.ocp.constraints.lbu = self.model.u_min
        self.ocp.constraints.ubu = self.model.u_max
        self.ocp.constraints.idxbu = np.arange(self.model.nu)
        self.ocp.constraints.lbx = self.model.x_min
        self.ocp.constraints.ubx = self.model.x_max
        self.ocp.constraints.idxbx = np.arange(self.model.nx)

        self.ocp.constraints.lbx_e = self.model.x_min
        self.ocp.constraints.ubx_e = self.model.x_max
        self.ocp.constraints.idxbx_e = np.arange(self.model.nx)

        # Solver options
        self.ocp.solver_options.tf = self.params.T
        self.ocp.solver_options.nlp_solver_type = self.params.solver_type
        self.ocp.solver_options.hpipm_mode = self.params.solver_mode
        self.ocp.solver_options.nlp_solver_max_iter = self.params.nlp_max_iter
        self.ocp.solver_options.qp_solver_iter_max = self.params.qp_max_iter
        self.ocp.solver_options.globalization = self.params.globalization
        self.ocp.solver_options.levenberg_marquardt = self.params.lm_reg
        #self.ocp.solver_options.tol=self.params.tol

        self.gen_name = self.params.GEN_DIR + 'ocp_' + self.ocp_name + str(0) + '_' + self.model.amodel.name
        self.ocp.code_export_directory = self.gen_name
        solver = AcadosOcpSolver(self.ocp, json_file=self.gen_name + '.json', generate=self.params.regenerate,build=self.params.regenerate)
        self.solvers.append(solver)
        for j in range(1,self.N-1):        
            N_list = [j,1,self.N-j-1] 
            self.model = deepcopy(self.simulator.model)
            self.ocp = AcadosMultiphaseOcp(N_list=N_list)
            phase_0 = AcadosOcp()
            phase_1 = AcadosOcp()
            phase_2 = AcadosOcp()
            phases= [phase_0,phase_1,phase_2]
            
            for i in range(len(N_list)):
                self.assign_problem(phases[i],i,N_list[i])
                self.ocp.set_phase(phases[i],i)
            self.ocp.solver_options.model_external_shared_lib_dir = self.model.l4c_model.shared_lib_dir
            self.ocp.solver_options.model_external_shared_lib_name = self.model.l4c_model.name
            # Solver options
            self.ocp.solver_options.tf = self.params.T
            self.ocp.solver_options.nlp_solver_type = self.params.solver_type
            self.ocp.solver_options.hpipm_mode = self.params.solver_mode
            self.ocp.solver_options.nlp_solver_max_iter = self.params.nlp_max_iter
            self.ocp.solver_options.qp_solver_iter_max = self.params.qp_max_iter
            self.ocp.solver_options.globalization = self.params.globalization
            self.ocp.solver_options.levenberg_marquardt = self.params.lm_reg
            #self.ocp.solver_options.tol=self.params.tol


            
            self.gen_name = self.params.GEN_DIR + 'ocp_' + self.ocp_name + str(j) + '_' + self.model.amodel.name
            self.ocp.code_export_directory = self.gen_name
            solver = AcadosOcpSolver(self.ocp, json_file=self.gen_name + '.json',generate=self.params.regenerate, build=self.params.regenerate)
            self.solvers.append(solver)
        j=j+1
        N_list = [j,1] 
        self.model = deepcopy(self.simulator.model)
        self.ocp = AcadosMultiphaseOcp(N_list=N_list)
        phase_0 = AcadosOcp()
        phase_1 = AcadosOcp()
        phases= [phase_0,phase_1]
        k=0
        for phase_ocp in phases: 
            phase_ocp.dims.N = self.N

            # Model
            phase_ocp.model = deepcopy(self.model.amodel)
            #phase_ocp.p_global=0

            # Cost
            self.Q = 1e-4 * np.eye(self.model.nx)
            self.Q[0, 0] = 5e2
            self.R = 1e-4 * np.eye(self.model.nu)

            phase_ocp.cost.W = lin.block_diag(self.Q, self.R)

            phase_ocp.cost.cost_type = "LINEAR_LS"

            phase_ocp.cost.Vx = np.zeros((self.model.ny, self.model.nx))
            phase_ocp.cost.Vx[:self.model.nx, :self.model.nx] = np.eye(self.model.nx)
            phase_ocp.cost.Vu = np.zeros((self.model.ny, self.model.nu))
            phase_ocp.cost.Vu[self.model.nx:, :self.model.nu] = np.eye(self.model.nu)

            phase_ocp.cost.yref = np.zeros(self.model.ny)
            # Set alpha to zero as default
            phase_ocp.parameter_values = np.array([0.])
            #phase_ocp.p_global_values = np.array([0.])

            # Constraints
            
            phase_ocp.constraints.lbu = self.model.u_min
            phase_ocp.constraints.ubu = self.model.u_max
            phase_ocp.constraints.idxbu = np.arange(self.model.nu)
            phase_ocp.constraints.lbx = self.model.x_min
            phase_ocp.constraints.ubx = self.model.x_max
            phase_ocp.constraints.idxbx = np.arange(self.model.nx)
            if k==0:
                phase_ocp.constraints.lbx_0 = self.model.x_min
                phase_ocp.constraints.ubx_0 = self.model.x_max
                phase_ocp.constraints.idxbx_0 = np.arange(self.model.nx)
            elif k==1:
                self.model.setNNmodel()
                self.model.amodel.con_h_expr = self.model.nn_model
                phase_ocp.model.con_h_expr = self.model.amodel.con_h_expr 
                phase_ocp.constraints.lh = np.array([0.])
                phase_ocp.constraints.uh = np.array([1e6])
                
                phase_ocp.constraints.idxsh = np.array([0])
                phase_ocp.cost.zl = np.ones((1,))*self.params.ws_r
                phase_ocp.cost.zu = np.zeros((1,))
                phase_ocp.cost.Zl = np.zeros((1,))
                phase_ocp.cost.Zu = np.zeros((1,))

       
                phase_ocp.cost.W_e = self.Q
                phase_ocp.cost.cost_type_e = "LINEAR_LS"
                phase_ocp.cost.Vx_e = np.eye(self.model.nx)
                phase_ocp.cost.yref_e = np.zeros(self.model.nx)

                phase_ocp.constraints.lbx_e = self.model.x_min
                phase_ocp.constraints.ubx_e = self.model.x_max
                phase_ocp.constraints.idxbx_e = np.arange(self.model.nx)
                #phase_ocp.model.con_h_expr = None

                if self.parallel == None:
                    self.model.amodel.con_h_expr_e = self.model.nn_model
                    phase_ocp.model.con_h_expr_e = self.model.amodel.con_h_expr_e 
                    phase_ocp.constraints.lh_e = np.array([0.])
                    phase_ocp.constraints.uh_e = np.array([1e6])
                
                    phase_ocp.constraints.idxsh_e = np.array([0])
                    phase_ocp.cost.zl_e = np.ones((1,))*self.params.ws_t
                    phase_ocp.cost.zu_e = np.zeros((1,))
                    phase_ocp.cost.Zl_e = np.zeros((1,))
                    phase_ocp.cost.Zu_e = np.zeros((1,))

            
            phase_ocp.model.name = f'phase_{k}'
            k+=1

        for i in range(len(N_list)):
            self.ocp.set_phase(phases[i],i)
        self.ocp.solver_options.model_external_shared_lib_dir = self.model.l4c_model.shared_lib_dir
        self.ocp.solver_options.model_external_shared_lib_name = self.model.l4c_model.name
        # Solver options
        self.ocp.solver_options.tf = self.params.T
        self.ocp.solver_options.nlp_solver_type = self.params.solver_type
        self.ocp.solver_options.hpipm_mode = self.params.solver_mode
        self.ocp.solver_options.nlp_solver_max_iter = self.params.nlp_max_iter
        self.ocp.solver_options.qp_solver_iter_max = self.params.qp_max_iter
        self.ocp.solver_options.globalization = self.params.globalization
        self.ocp.solver_options.levenberg_marquardt = self.params.lm_reg
        #self.ocp.solver_options.tol=self.params.tol


        
        self.gen_name = self.params.GEN_DIR + 'ocp_' + self.ocp_name + str(j) + '_' + self.model.amodel.name
        self.ocp.code_export_directory = self.gen_name
        solver = AcadosOcpSolver(self.ocp, json_file=self.gen_name + '.json',generate=self.params.regenerate, build=self.params.regenerate)
        self.solvers.append(solver)

        self.model = self.simulator.model
        # Last solver: common terminal constraint
        self.ocp = AcadosOcp()

        # Dimensions
        self.ocp.solver_options.tf = self.params.T
        self.ocp.dims.N = self.N

        # Model
        self.ocp.model = self.model.amodel

        # Cost
        self.Q = 1e-4 * np.eye(self.model.nx)
        self.Q[0, 0] = 5e2
        self.R = 1e-4 * np.eye(self.model.nu)

        self.ocp.cost.W = lin.block_diag(self.Q, self.R)
        self.ocp.cost.W_e = self.Q

        self.ocp.cost.cost_type = "LINEAR_LS"
        self.ocp.cost.cost_type_e = "LINEAR_LS"

        self.ocp.cost.Vx = np.zeros((self.model.ny, self.model.nx))
        self.ocp.cost.Vx[:self.model.nx, :self.model.nx] = np.eye(self.model.nx)
        self.ocp.cost.Vu = np.zeros((self.model.ny, self.model.nu))
        self.ocp.cost.Vu[self.model.nx:, :self.model.nu] = np.eye(self.model.nu)
        self.ocp.cost.Vx_e = np.eye(self.model.nx)

        self.ocp.cost.yref = np.zeros(self.model.ny)
        self.ocp.cost.yref_e = np.zeros(self.model.nx)
        # Set alpha to zero as default
        self.ocp.parameter_values = np.array([0.])

        # Constraints
        self.ocp.constraints.lbx_0 = self.model.x_min
        self.ocp.constraints.ubx_0 = self.model.x_max
        self.ocp.constraints.idxbx_0 = np.arange(self.model.nx)

        self.ocp.constraints.lbu = self.model.u_min
        self.ocp.constraints.ubu = self.model.u_max
        self.ocp.constraints.idxbu = np.arange(self.model.nu)
        self.ocp.constraints.lbx = self.model.x_min
        self.ocp.constraints.ubx = self.model.x_max
        self.ocp.constraints.idxbx = np.arange(self.model.nx)

        self.ocp.constraints.lbx_e = self.model.x_min
        self.ocp.constraints.ubx_e = self.model.x_max
        self.ocp.constraints.idxbx_e = np.arange(self.model.nx)

        # Solver options
        self.ocp.solver_options.tf = self.params.T
        self.ocp.solver_options.nlp_solver_type = self.params.solver_type
        self.ocp.solver_options.hpipm_mode = self.params.solver_mode
        self.ocp.solver_options.nlp_solver_max_iter = self.params.nlp_max_iter
        self.ocp.solver_options.qp_solver_iter_max = self.params.qp_max_iter
        self.ocp.solver_options.globalization = self.params.globalization
        self.ocp.solver_options.levenberg_marquardt = self.params.lm_reg
        #self.ocp.solver_options.tol=self.params.tol
        # Terminal constraint
        self.model.setNNmodel()
        self.model.amodel.con_h_expr_e = self.model.nn_model

        self.ocp.solver_options.model_external_shared_lib_dir = self.model.l4c_model.shared_lib_dir
        self.ocp.solver_options.model_external_shared_lib_name = self.model.l4c_model.name

        self.ocp.constraints.lh_e = np.array([0.])
        self.ocp.constraints.uh_e = np.array([1e6])
        if self.params.soft == True:
            self.ocp.constraints.idxsh_e = np.array([0])
            self.ocp.cost.zl_e = np.ones((1,)) * self.params.ws_t
            self.ocp.cost.zu_e = np.zeros((1,))
            self.ocp.cost.Zl_e = np.zeros((1,))
            self.ocp.cost.Zu_e = np.zeros((1,))

        self.gen_name = self.params.GEN_DIR + 'ocp_' + self.ocp_name + str(j+1) + '_' + self.model.amodel.name
        self.ocp.code_export_directory = self.gen_name
        solver = AcadosOcpSolver(self.ocp, json_file=self.gen_name + '.json', generate=self.params.regenerate,build=self.params.regenerate)
        self.solvers.append(solver)

        # Initialize guess
        self.fails = 0
        self.x_ref = np.zeros(self.model.nx)

        # Empty initial guess and temp vectors
        self.x_guess = np.zeros((self.N + 1, self.model.nx))
        self.u_guess = np.zeros((self.N, self.model.nu))
        self.x_temp, self.u_temp = np.copy(self.x_guess), np.copy(self.u_guess)

        # Viable state (None for Naive and ST controllers)
        self.x_viable = None

        # Time stats
        self.time_fields = ['time_lin', 'time_sim', 'time_qp', 'time_qp_solver_call',
                            'time_glob', 'time_reg', 'time_tot']

    def assign_problem(self,phase_ocp,pos,N):
        # Dimensions
        phase_ocp.dims.N = N

        # Model
        phase_ocp.model = deepcopy(self.model.amodel)
        #phase_ocp.p_global=0

        # Cost
        self.Q = 1e-4 * np.eye(self.model.nx)
        self.Q[0, 0] = 5e2
        self.R = 1e-4 * np.eye(self.model.nu)

        phase_ocp.cost.W = lin.block_diag(self.Q, self.R)

        phase_ocp.cost.cost_type = "LINEAR_LS"

        phase_ocp.cost.Vx = np.zeros((self.model.ny, self.model.nx))
        phase_ocp.cost.Vx[:self.model.nx, :self.model.nx] = np.eye(self.model.nx)
        phase_ocp.cost.Vu = np.zeros((self.model.ny, self.model.nu))
        phase_ocp.cost.Vu[self.model.nx:, :self.model.nu] = np.eye(self.model.nu)

        phase_ocp.cost.yref = np.zeros(self.model.ny)
        # Set alpha to zero as default
        phase_ocp.parameter_values = np.array([0.])
        #phase_ocp.p_global_values = np.array([0.])

        # Constraints
        
        phase_ocp.constraints.lbu = self.model.u_min
        phase_ocp.constraints.ubu = self.model.u_max
        phase_ocp.constraints.idxbu = np.arange(self.model.nu)
        phase_ocp.constraints.lbx = self.model.x_min
        phase_ocp.constraints.ubx = self.model.x_max
        phase_ocp.constraints.idxbx = np.arange(self.model.nx)

        if pos==0:
            phase_ocp.constraints.lbx_0 = self.model.x_min
            phase_ocp.constraints.ubx_0 = self.model.x_max
            phase_ocp.constraints.idxbx_0 = np.arange(self.model.nx)

        if pos ==1:
            self.model.setNNmodel()
            self.model.amodel.con_h_expr = self.model.nn_model
            phase_ocp.model.con_h_expr = self.model.amodel.con_h_expr 
            phase_ocp.constraints.lh = np.array([0.])
            phase_ocp.constraints.uh = np.array([1e6])
            if self.params.soft == True:
                phase_ocp.constraints.idxsh = np.array([0])
                phase_ocp.cost.zl = np.ones((1,))*self.params.ws_r
                phase_ocp.cost.zu = np.zeros((1,))
                phase_ocp.cost.Zl = np.zeros((1,))
                phase_ocp.cost.Zu = np.zeros((1,))

        if pos==2:
            phase_ocp.cost.W_e = self.Q
            phase_ocp.cost.cost_type_e = "LINEAR_LS"
            phase_ocp.cost.Vx_e = np.eye(self.model.nx)
            phase_ocp.cost.yref_e = np.zeros(self.model.nx)

            phase_ocp.constraints.lbx_e = self.model.x_min
            phase_ocp.constraints.ubx_e = self.model.x_max
            phase_ocp.constraints.idxbx_e = np.arange(self.model.nx)
            phase_ocp.model.con_h_expr = None

            if self.parallel == None:
                self.model.amodel.con_h_expr_e = self.model.nn_model
                phase_ocp.model.con_h_expr_e = self.model.amodel.con_h_expr_e 
                phase_ocp.constraints.lh_e = np.array([0.])
                phase_ocp.constraints.uh_e = np.array([1e6])
                
                phase_ocp.constraints.idxsh_e = np.array([0])
                phase_ocp.cost.zl_e = np.ones((1,))*self.params.ws_t
                phase_ocp.cost.zu_e = np.zeros((1,))
                phase_ocp.cost.Zl_e = np.zeros((1,))
                phase_ocp.cost.Zu_e = np.zeros((1,))
            
        phase_ocp.model.name = f'phase_{pos}'

        

        # Additional settings, in general is an empty method
        #self.additionalSetting()
    def reinit_solver(self):
        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file=self.gen_name +'.json', build=True,verbose=False)

    def simulate_solver(self, x, u):
        self.solver_integrator.set("x", x)
        self.solver_integrator.set("u", u)
        self.solver_integrator.solve()
        x_next = self.solver_integrator.get("x")
        return x_next
    
    def additionalSetting(self):
        pass

    def solve(self, x0):
        # Reset current iterate
        self.ocp_solver.reset()

        # Constrain initial state
        self.ocp_solver.constraints_set(0, "lbx", x0)
        self.ocp_solver.constraints_set(0, "ubx", x0)

        y_ref = np.zeros(self.model.ny)
        y_ref[:self.model.nx] = self.x_ref
        W = lin.block_diag(self.Q, self.R)

        for i in range(self.N):
            self.ocp_solver.set(i, 'x', self.x_guess[i])
            self.ocp_solver.set(i, 'u', self.u_guess[i])
            self.ocp_solver.cost_set(i, 'yref', y_ref, api='new')
            self.ocp_solver.cost_set(i, 'W', W, api='new')
        self.ocp_solver.set(0,'x',x0)

        self.ocp_solver.set(self.N, 'x', self.x_guess[-1])
        self.ocp_solver.cost_set(self.N, 'yref', y_ref[:self.model.nx], api='new')
        self.ocp_solver.cost_set(self.N, 'W', self.Q, api='new')

        # Solve the OCP
        status = self.ocp_solver.solve()

        # Save the temporary solution, independently of the status
        for i in range(self.N):
            self.x_temp[i] = self.ocp_solver.get(i, "x")
            self.u_temp[i] = self.ocp_solver.get(i, "u")
        self.x_temp[-1] = self.ocp_solver.get(self.N, "x")

        return status

    def provideControl(self):
        """ Save the guess for the next MPC step and return u_opt[0] """
        if self.fails > 0:
            u = self.u_guess[0]
            # Rollback the previous guess
            #self.guessCorrection()
            self.x_guess = np.roll(self.x_guess, -1, axis=0)
            self.u_guess = np.roll(self.u_guess, -1, axis=0)
            if 'receding' in self.params.cont_type or self.params.cont_type== 'parallel2' or self.params.cont_type== 'parallel_limited':
                self.x_guess[-1] = self.simulator.simulate(self.x_guess[-2],self.u_guess[-2])
        else:
            u = self.u_temp[0]
            #self.x_guess = np.copy(self.x_temp)
            #self.guessCorrection()
            # Save the current temporary solution
            self.x_guess = np.roll(self.x_temp, -1, axis=0)
            self.u_guess = np.roll(self.u_temp, -1, axis=0)
            if 'receding' in self.params.cont_type or self.params.cont_type== 'parallel2' or self.params.cont_type== 'parallel_limited':
                self.guessCorrection()
                      
        # Copy the last values        
        if not('receding' in self.params.cont_type)  and self.params.cont_type != 'parallel2' and self.params.cont_type != 'parallel_limited':
            self.x_guess[-1] = np.copy(self.x_guess[-2])
        self.u_guess[-1] = np.copy(self.u_guess[-2])
        return u

    def step(self, x0):
        pass

    def setQRWeights(self, Q, R):
        self.Q = Q
        self.R = R

    def setReference(self, x_ref):
        self.x_ref = x_ref

    def getTime(self,node):
        pass

    def setGuess(self, x_guess, u_guess):
        self.x_guess = x_guess
        self.u_guess = u_guess

    def getGuess(self):
        return np.copy(self.x_guess), np.copy(self.u_guess)

    def getLastViableState(self):
        return np.copy(self.x_viable)
    

    def guessCorrection(self):
        error = 0
        x_corrected = np.copy(self.x_guess) 
        for i in range(len(self.u_guess)):
            x_corrected[i+1]=self.simulator.simulate(x_corrected[i],self.u_guess[i])
            error += np.linalg.norm(np.abs(self.x_guess[i+1]-x_corrected[i+1]))
        if np.linalg.norm(self.x_guess - x_corrected) > self.err_thr* np.sqrt(self.N+1):
            #print(error)
            self.x_guess = x_corrected



class IpoptController:
    def __init__(self, params, model, x_ref, terminal=False):
        self.params = params
        self.model = model
        self.N = params.N
        self.x = model.x
        self.u = model.u
        self.x_ref = x_ref

        self.x_guess = np.zeros((self.N + 1, model.nx))
        self.u_guess = np.zeros((self.N, model.nu))

        # Define the integrator
        self.f_expl = integrator('f_expl', 'cvodes', {'x': self.x, 'p': self.u, 'ode': model.f_expl},
                                 0, self.params.dt)

        # QR regularization
        self.Q = 1e-4 * np.eye(self.model.nx)
        self.Q[0, 0] = 5e2
        self.R = 1e-4 * np.eye(self.model.nu)

        # Define the OCP
        self.opti = Opti()
        self.xs = self.opti.variable(self.model.nx, self.N + 1)
        self.us = self.opti.variable(self.model.nu, self.N)

        self.cost = Function('cost', [self.x, self.u], [self.runningCost(self.x, self.u)])

        total_cost = 0
        for i in range(self.N):
            total_cost += self.cost(self.xs[:, i], self.us[:, i])
            self.opti.subject_to(self.xs[:, i + 1] == self.f_expl(x0=self.xs[:, i], p=self.us[:, i])['xf'])
            self.opti.subject_to(self.opti.bounded(self.model.x_min, self.xs[:, i], self.model.x_max))
            self.opti.subject_to(self.opti.bounded(self.model.u_min, self.us[:, i], self.model.u_max))
        t_cost = self.cost(self.xs[:, -1], np.zeros(self.model.nu))
        total_cost += t_cost
        self.opti.subject_to(self.opti.bounded(self.model.x_min, self.xs[:, -1], self.model.x_max))

        if terminal:
            self.opti.subject_to(self.model.nn_func(self.xs[:, -1], params.alpha) >= 0.)

        self.opti.minimize(total_cost)

        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes', 'ipopt.linear_solver': 'ma57'}
        self.opti.solver('ipopt', opts)

    def solve(self, x0):

        for i in range(self.N):
            self.opti.set_initial(self.xs[:, i], x0)
            self.opti.set_initial(self.us[:, i], np.zeros(self.model.nu))
        self.opti.set_initial(self.xs[:, -1], x0)

        try:
            sol = self.opti.solve()
            self.x_guess = np.copy(sol.value(self.xs))
            self.u_guess = np.copy(sol.value(self.us))
            return 1
        except RuntimeError:
            return 0

    def setReference(self, x_ref):
        self.x_ref = x_ref

    def runningCost(self, x, u):
        return (x - self.x_ref).T @ self.Q @ (x - self.x_ref) + u.T @ self.R @ u

    def getGuess(self):
        return np.copy(self.x_guess), np.copy(self.u_guess)