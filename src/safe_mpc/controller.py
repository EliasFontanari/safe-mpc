import numpy as np
import scipy.linalg as lin
from .abstract_multiphase import AbstractController, AbstractControllerSingle
#from .abstract import AbstractController

class NaiveController(AbstractController):
    def __init__(self, simulator):
        super().__init__(simulator)

    def checkGuess(self):
        return self.model.checkRunningConstraints(self.x_temp, self.u_temp) and \
               self.simulator.checkDynamicsConstraints(self.x_temp, self.u_temp)

    def initialize(self, x0, u0=None):
        # Trivial guess
        self.x_guess = np.full((self.N + 1, self.model.nx), x0)
        if u0 is None:
            u0 = np.zeros(self.model.nu)
        self.u_guess = np.full((self.N, self.model.nu), u0)
        # Solve the OCP
        status = self.solve(x0)
        if (status == 0 or status==2) and self.checkGuess():
            self.x_guess = np.copy(self.x_temp)
            self.u_guess = np.copy(self.u_temp)
            return 1
        return 0

    def step(self, x):
        status = self.solve(x)
        if status == 0 and self.model.checkControlConstraints(self.u_temp[0]):# and \
           #self.simulator.checkDynamicsConstraints(self.x_temp[:2], np.array([self.u_temp[0]])):
            self.fails = 0
        else:
            self.fails += 1
        #self.guessCorrection()
        return self.provideControl()


class STController(NaiveController):
    def __init__(self, simulator):
        super().__init__(simulator)

    def additionalSetting(self):
        self.terminalConstraint(soft=False)
        #self.runningConstraint()


class STWAController(STController):
    def __init__(self, simulator):
        super().__init__(simulator)
        self.x_viable = None

    def checkGuess(self):
        return self.model.checkRunningConstraints(self.x_temp, self.u_temp) and \
               self.simulator.checkDynamicsConstraints(self.x_temp, self.u_temp) and \
               self.model.checkSafeConstraints(self.x_temp[-1])

    def step(self, x):
        status = self.solve(x)
        if status == 0 and self.model.checkRunningConstraints(self.x_temp, self.u_temp) and \
                self.model.checkSafeConstraints(self.x_temp[-1]):
            self.fails = 0
        else:
            if self.fails == 0:
                self.x_viable = np.copy(self.x_guess[-2])
            if self.fails >= self.N:
                return None
            self.fails += 1
        return self.provideControl()

    def setGuess(self, x_guess, u_guess):
        self.x_guess = x_guess
        self.u_guess = u_guess
        self.x_viable = x_guess[-1]


class HTWAController(STWAController):
    def __init__(self, simulator):
        super().__init__(simulator)

    def additionalSetting(self):
        self.terminalConstraint(soft=False)


class RecedingController(STWAController):
    def __init__(self, simulator):
        super().__init__(simulator)
        self.r = self.N
        self.min_negative_jump = self.params.min_negative_jump
        self.step_old_solution = 0

    def additionalSetting(self):
        # Terminal constraint before, since it construct the nn model
        self.terminalConstraint()
        self.runningConstraint()

    def solve(self, x0,constr_nodes=None):
        #self.ocp_solver.load_iterate('/home/utente/Documents/Optim/safe-mpc/triple_pendulum_iterate.json',verbose=False)
        # Reset current iterate
        self.ocp_solver.reset()

        for i in range(self.N+1):
            self.ocp_solver.set(i,'p',self.params.alpha)
            
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
        self.ocp_solver.set(self.N, 'x', self.x_guess[-1])
        self.ocp_solver.set(0,'x',x0)
        self.ocp_solver.cost_set(self.N, 'yref', y_ref[:self.model.nx], api='new')
        self.ocp_solver.cost_set(self.N, 'W', self.Q, api='new')

        if constr_nodes != None:
            for i in range(1,self.N+1):
                if np.all(np.array(constr_nodes) != i):
                    self.ocp_solver.set(i, 'sl', 1e4)
                    self.ocp_solver.set(i, 'su', 1e4)

        # Solve the OCP
        status = self.ocp_solver.solve()

        # Save the temporary solution, independently of the status
        for i in range(self.N):
            self.x_temp[i] = self.ocp_solver.get(i, "x")
            self.u_temp[i] = self.ocp_solver.get(i, "u")
        self.x_temp[-1] = self.ocp_solver.get(self.N, "x")

        return status

    def step(self, x):
        # Terminal constraint
        self.ocp_solver.cost_set(self.N, "zl", self.params.ws_t * np.ones((1,)))
        # Receding constraint
        self.ocp_solver.cost_set(self.r, "zl", self.params.ws_r * np.ones((1,)))
        for i in range(1, self.N):
            if i != self.r:
                # No constraints on other running states
                self.ocp_solver.cost_set(i, "zl", np.zeros((1,)))
        # Solve the OCP
        status = self.solve(x,[self.r,self.N])
        #print(f'status = {status}')
        #print(f'last solution: {np.round(self.solvers[self.r].get(self.N, "x"),4)}, node: {self.r}')



        r_new = -1
        for i in range(1, self.N + 1):
            if self.model.checkSafeConstraints(self.x_temp[i]) and (i - self.r) >= self.min_negative_jump:
                r_new = i 
        #or(status==2 and self.simulator.checkDynamicsConstraints(self.x_temp, self.u_temp))
        if (status == 0 ) and self.model.checkRunningConstraints(self.x_temp, self.u_temp) \
            and r_new > 1 and self.simulator.checkSafeIntegrate([x],self.u_temp,r_new)[0]:
            
            self.fails = 0
            self.r = r_new
            self.step_old_solution = 0
        else:
            if self.r == 1:
                #self.x_viable = np.copy(self.x_guess[self.r])
                _,self.x_viable = self.simulator.checkSafeIntegrate([x],self.u_guess,self.r)
                self.step_old_solution += 1

                print("NOT SOLVED")
                print(f'x viable:{self.x_viable}')
                print(f'is x viable:{self.model.nn_func(self.x_viable,self.model.params.alpha)}')
                
                return None
            self.fails += 1
        self.r -= 1
        return self.provideControl()

class RecedingSingleConstraint(AbstractControllerSingle):
    def __init__(self, simulator):
        super().__init__(simulator)
        self.r = self.N
        self.min_negative_jump = self.params.min_negative_jump
        self.step_old_solution = 0

    def setGuess(self, x_guess, u_guess):
        self.x_guess = x_guess
        self.u_guess = u_guess
        self.x_viable = x_guess[-1]

    def checkGuess(self):
        return self.model.checkRunningConstraints(self.x_temp, self.u_temp) and \
               self.simulator.checkDynamicsConstraints(self.x_temp, self.u_temp)
    
    def getTime(self):
        return np.array([self.solvers[self.r+1].get_stats(field) for field in self.time_fields])


    def initialize(self, x0, u0=None):
        # Trivial guess
        self.x_guess = np.full((self.N + 1, self.model.nx), x0)
        if u0 is None:
            u0 = np.zeros(self.model.nu)
        self.u_guess = np.full((self.N, self.model.nu), u0)
        # Solve the OCP
        status = self.solve(x0)
        if (status == 0 or status==2) and self.checkGuess():
            self.x_guess = np.copy(self.x_temp)
            self.u_guess = np.copy(self.u_temp)
            return 1
        return 0
    
    def additionalSetting(self):
        pass

    def solve(self, x0,solver_n=None):
        if solver_n == None:
            solver_n = self.r

        #self.ocp_solver.load_iterate('/home/utente/Documents/Optim/safe-mpc/triple_pendulum_iterate.json',verbose=False)
        # Reset current iterate
        self.solvers[solver_n].reset()
        for i in range(self.N+1):
            self.solvers[solver_n].set(i,'p',self.params.alpha)
        # Constrain initial state
        self.solvers[solver_n].constraints_set(0, "lbx", x0)
        self.solvers[solver_n].constraints_set(0, "ubx", x0)

        y_ref = np.zeros(self.model.ny)
        y_ref[:self.model.nx] = self.x_ref
        W = lin.block_diag(self.Q, self.R)

        
        for i in range(self.N):
            self.solvers[solver_n].set(i, 'x', self.x_guess[i])
            self.solvers[solver_n].set(i, 'u', self.u_guess[i])
            self.solvers[solver_n].cost_set(i, 'yref', y_ref, api='new')
            self.solvers[solver_n].cost_set(i, 'W', W, api='new')
        self.solvers[solver_n].set(self.N, 'x', self.x_guess[-1])
        self.solvers[solver_n].set(0,'x',x0)
        self.solvers[solver_n].cost_set(self.N, 'yref', y_ref[:self.model.nx], api='new')
        self.solvers[solver_n].cost_set(self.N, 'W', self.Q, api='new')

        # Solve the OCP
        status = self.solvers[solver_n].solve()

        # Save the temporary solution, independently of the status
        for i in range(self.N):
            self.x_temp[i] = self.solvers[solver_n].get(i, "x")
            self.u_temp[i] = self.solvers[solver_n].get(i, "u")
        self.x_temp[-1] = self.solvers[solver_n].get(self.N, "x")

        return status

    def step(self, x):
        if self.r==10:
            pass
        status = self.solve(x)
        #rint(f'status = {status}')
        #print(f'last solution: {np.round(self.solvers[self.r].get(self.N, "x"),4)}, node: {self.r}')
        
        r_new = -1
        for i in range(1, self.N + 1):
            if self.model.checkSafeConstraints(self.x_temp[i]) and (i - self.r) >= self.min_negative_jump:
                r_new = i 
        #or(status==2 and self.simulator.checkDynamicsConstraints(self.x_temp, self.u_temp))
        if (status == 0 ) and self.model.checkRunningConstraints(self.x_temp, self.u_temp) \
            and r_new > 1 and self.simulator.checkSafeIntegrate([x],self.u_temp,r_new)[0]:
            
            self.fails = 0
            self.r = r_new
            self.step_old_solution = 0
        else:
            if self.r == 1:
                #self.x_viable = np.copy(self.x_guess[self.r])
                _,self.x_viable = self.simulator.checkSafeIntegrate([x],self.u_guess,self.r)
                self.step_old_solution += 1

                print("NOT SOLVED")
                print(f'is x viable:{self.model.nn_func(self.x_viable,self.model.params.alpha)}')
                
                return None
            self.fails += 1
        self.r -= 1
        return self.provideControl()


class ParallelWithCheck(RecedingController):    
    def __init__(self, simulator):
        super().__init__(simulator)
        self.r = self.N
        self.core_solution=0
        self.constraints =  np.linspace(1,self.N,self.N).round().astype(int).tolist() 
        self.times=[]
          

    def checkGuess(self):
        return self.model.checkRunningConstraints(self.x_temp, self.u_temp) and \
               self.simulator.checkDynamicsConstraints(self.x_temp, self.u_temp) and \
               self.model.checkSafeConstraints(self.x_temp[-1])


    def constrain_n(self,n_constr):
        for i in range(1, self.N+1):
            self.ocp_solver.cost_set(i, "zl", np.zeros((1,)))
        if 0 < n_constr <= self.N:
            self.ocp_solver.cost_set(n_constr, "zl", self.params.ws_r * np.ones((1,)))

    def check_safe_n(self):
        r=0
        for i in range(1, self.N + 1):
            if self.model.checkSafeConstraints(self.x_temp[i]):
                r = i
        return r


    def sing_step(self, x, n_constr):
        success = False
        self.constrain_n(n_constr)
        status = self.solve(x,[n_constr])
        checked_r = self.check_safe_n()
        if ((check:=self.model.checkSafeConstraints(self.x_temp[n_constr])) or checked_r > 1) \
            and (status==0):
            constr_ver = n_constr if check else 0
            n_step_safe = max(checked_r,constr_ver)
            safe, x_safe = self.simulator.checkSafeIntegrate([x],self.u_temp,n_step_safe)
            if safe:
                success = True


        if success and self.model.checkRunningConstraints(self.x_temp, self.u_temp) and \
           (n_step_safe-self.r)>= self.min_negative_jump:
            
            success = True
            


        else:
            success = False

        return n_step_safe if success else 0

    def step(self,x):
        self.times=[]
        node_success = 0
        core = None
        for i in reversed(self.constraints):
            result = self.sing_step(x,i)
            self.times.append(np.array([self.ocp_solver.get_stats(field) for field in self.time_fields]))
            if result > node_success:
                core = i
                node_success = result
                tmp_x = np.copy(self.x_temp)
                tmp_u = np.copy(self.u_temp)
                if result==self.N:
                    break
        if node_success > 1:
            self.core_solution = core
            self.step_old_solution = 0
            self.r = node_success
            self.x_temp = tmp_x
            self.u_temp = tmp_u
            self.fails = 0
        else: 
            self.core_solution = None 
            self.step_old_solution +=1
            self.fails = 1
            if self.r ==1:
                print("NOT SOLVED")
                _,self.x_viable = self.simulator.checkSafeIntegrate([x],self.u_guess,self.r)
                print(f'x viable:{self.x_viable}')
                print(f'is x viable:{self.model.nn_func(self.x_viable,self.model.params.alpha)}')
                return None
        self.r -= 1

        return self.provideControl()
    
class ParallelSingleConstraint(RecedingSingleConstraint):
    def __init__(self, simulator):
        super().__init__(simulator)
        self.constraints =  np.linspace(1,self.N,self.N).round().astype(int).tolist()   
        self.times = []

    def checkGuess(self):
        return self.model.checkRunningConstraints(self.x_temp, self.u_temp) and \
               self.simulator.checkDynamicsConstraints(self.x_temp, self.u_temp) and \
               self.model.checkSafeConstraints(self.x_temp[-1])

    def check_safe_n(self):
        r=0
        for i in range(1, self.N + 1):
            if self.model.checkSafeConstraints(self.x_temp[i]):
                r = i
        return r


    def sing_step(self, x, n_constr):
        success = False
        status = self.solve(x,n_constr)
        checked_r = self.check_safe_n()
        if ((check:=self.model.checkSafeConstraints(self.x_temp[n_constr])) or checked_r > 1) \
            and (status==0 ):
            constr_ver = n_constr if check else 0
            n_step_safe = max(checked_r,constr_ver)
            safe, x_safe = self.simulator.checkSafeIntegrate([x],self.u_temp,n_step_safe)
            if safe:
                success = True


        if success and self.model.checkRunningConstraints(self.x_temp, self.u_temp) and \
           (n_step_safe-self.r)>= self.min_negative_jump:
            
            success = True
            


        else:
            success = False

        return n_step_safe if success else 0

    def step(self,x):
        self.times=[]
        node_success = 0
        core = None
        for i in reversed(self.constraints):
            result = self.sing_step(x,i)
            self.times.append(np.array([self.solvers[i].get_stats(field) for field in self.time_fields]))
            if result > node_success:
                core = i
                node_success = result
                tmp_x = np.copy(self.x_temp)
                tmp_u = np.copy(self.u_temp)
                if result==self.N:
                    break
        if node_success > 1:
            self.core_solution = core
            self.step_old_solution = 0
            self.r = node_success
            self.x_temp = tmp_x
            self.u_temp = tmp_u
            self.fails = 0
        else: 
            self.core_solution = None 
            self.step_old_solution +=1
            self.fails = 1
            if self.r ==1:
                print("NOT SOLVED")
                _,self.x_viable = self.simulator.checkSafeIntegrate([x],self.u_guess,self.r)
                print(self.x_viable)
                print(f'is x viable:{self.model.nn_func(self.x_viable,self.model.params.alpha)}')
                return None
        self.r -= 1

        return self.provideControl()
    
    def getTime(self):
        pass
        #return self.times

class ParallelLimited(ParallelWithCheck):
    def __init__(self,simulator):
        super().__init__(simulator)
        self.cores = self.params.n_cores
        self.constraints = []
        self.constraint_mode = self.CIS_distance_constraint                # choose mode

    def high_nodes_constraint(self):
        self.constraints = []
        self.constraints.append(self.r)
        i,j=0,0
        while i < self.cores - 1:
            if self.N -j != self.r:
                self.constraints.append(self.N - j)
                i +=1
            j +=1
        self.constraints = sorted(self.constraints)

    def uniform_constraint(self):
        if self.cores == self.N:
            self.constraints = np.linspace(1,self.N,self.cores).round().astype(int).tolist()    
        else: 
            self.constraints = []
            step = (self.N-1)/self.cores
            if self.r == 1 or self.r == self.N:
                self.constraints = np.linspace(1,self.N,self.cores).round().astype(int).tolist()
            else:
                if self.r < 1 + step:
                    i=1
                    self.constraints.append(self.r)
                    while i < self.cores:
                        self.constraints.append(int(min(self.N,round(self.r + i*step))))
                        i+=1
                elif self.r >  self.N - step:
                    i=1
                    self.constraints.append(self.r)
                    while i < self.cores:                        
                        self.constraints.insert(0,int(max(1,round(self.r - i*step))))
                        i+=1
                else:
                    self.constraints=[] 
                    self.constraints.append(self.r)
                    portion_h = (self.cores-1)*((self.N - self.r)/(self.N-1))
                    portion_h = int(portion_h) if portion_h-int(portion_h)<=0.5 else int(portion_h+1)
                    portion_l = (self.cores-1)*((self.r-1)/(self.N-1))
                    portion_l = int(portion_l) if portion_l-int(portion_l)<=0.5 else int(portion_l+1)
                    if portion_h == portion_l and self.cores%2==0 or portion_h +portion_l < self.cores -1:
                        self.constraints = np.linspace(int(max(1,round(self.r-portion_l*step))),round(self.r-step),portion_l).round().astype(int).tolist() + self.constraints \
                            + np.linspace(round(self.r+step),self.N,portion_h+1).round().astype(int).tolist()
                    else: 
                        self.constraints_l = []
                        self.constraints_h = []
                        i,j=1,1
                        while i < portion_l+1:
                            self.constraints_l.insert(0,int(max(1,round(self.r - i*step))))
                            i+=1
                        while j < portion_h+1:
                            self.constraints_h.append(int(min(self.N,round(self.r + j*step)))) 
                            j+=1
                        self.constraints = self.constraints_l+self.constraints+self.constraints_h


    def CIS_distance_constraint(self):
        self.constraints=[]
        self.constraints.append(self.r)
        distances=[]
        for i in range(1,self.N+1):
            distances.append(self.model.nn_func(self.x_guess[i], self.params.alpha))
        indx_sorted = np.argsort(np.array(distances).squeeze())[::-1]
        i,j = 0,0
        while i < self.cores - 1:
            if (indx_sorted[j]+1)!=self.r:
                self.constraints.append(indx_sorted[j]+1)
                i+=1 
            j+=1
        self.constraints = sorted(self.constraints)
    
    def step(self,x):
        node_success = 0
        core = None
        self.constraint_mode()
        if len(self.constraints) != self.cores or len(self.constraints) != len(set(self.constraints)):
            print(self.r)
            print('ERROR\n\n\n\n\n\n')
            print(self.constraints)
        if not(self.r in self.constraints):
            print('ERROR NOT PRESENT R\n\n\n\n\n\n')
        k=1
        for i in reversed(self.constraints):
            #print(self.constraints)
            result = self.sing_step(x,int(i))
            if result > node_success:
                #print(self.constraints)
                
                core = k
                node_success = result
                tmp_x = np.copy(self.x_temp)
                tmp_u = np.copy(self.u_temp)
                if result==self.N:
                    break
            k+=1
        if node_success > 1:
            self.core_solution = core
            self.step_old_solution = 0
            self.r = node_success
            self.x_temp = tmp_x
            self.u_temp = tmp_u
            self.fails = 0
        else: 
            self.core_solution = None 
            self.step_old_solution +=1
            self.fails = 1
            if self.r ==1:
                print("NOT SOLVED")
                _,self.x_viable = self.simulator.checkSafeIntegrate([x],self.u_guess,self.r)
                print(f'x viable:{self.x_viable}')
                print(f'is x viable:{self.model.nn_func(self.x_viable,self.model.params.alpha)}')
                return None
        self.r -= 1


        return self.provideControl()

class SafeBackupController(AbstractController):
    def __init__(self, simulator):
        super().__init__(simulator)

    def additionalSetting(self):
        self.Q = np.zeros((self.model.nx, self.model.nx))
        self.Q[self.model.nq:, self.model.nq:] = np.eye(self.model.nv) * self.params.q_dot_gain
        #self.R = 0e-4 * np.eye(self.model.nu)

        self.ocp.cost.W = lin.block_diag(self.Q, self.R)
        self.ocp.cost.W_e = self.Q

        q_fin_lb = np.hstack([self.model.x_min[:self.model.nq], np.zeros(self.model.nv)])
        q_fin_ub = np.hstack([self.model.x_max[:self.model.nq], np.zeros(self.model.nv)])

        self.ocp.constraints.lbx_e = q_fin_lb
        self.ocp.constraints.ubx_e = q_fin_ub
        self.ocp.constraints.idxbx_e = np.arange(self.model.nx)
        # self.ocp.cost.zl_e = np.ones(self.model.nv) * 5e2
        # self.ocp.cost.Zl_e = np.zeros(self.model.nv)
        # self.ocp.cost.zu_e = np.ones(self.model.nv) * 5e2
        # self.ocp.cost.Zu_e = np.zeros(self.model.nv)
        # self.ocp.constraints.idxsbx_e = np.array([3,4,5])



# class ParallelLimited(ParallelController):
#     def __init__(self,simulator):
#         super().__init__(simulator)
#         self.cores = self.params.n_cores
#         self.constraints = []
#         self.constraint_mode = self.high_nodes_constraint

#     def high_nodes_constraint(self):
#         self.constraints = []
#         self.constraints.append(self.r)
#         i,j=0,0
#         while i < self.cores - 1:
#             if self.N -j != self.r:
#                 self.constraints.append(self.N - j)
#                 i +=1
#             j +=1
#         self.constraints = sorted(self.constraints,reverse=True)

#     def uniform_constraint(self):
#         self.constraints=[]
#         self.constraints.append(self.r)
#         low_alloc = int((self.cores)*((self.r-1)/(self.N-1)))
#         high_alloc = int((self.cores)*((self.N-self.r)/(self.N-1)))
#         if low_alloc - high_alloc == 0 and low_alloc%2!=0:
#             #low_alloc -= 1
#             low_constr = np.linspace(1,self.r,low_alloc+1).round().astype(int)
#             high_constr = np.linspace(self.r,self.N,high_alloc+1).round().astype(int)
#             for i in range(low_alloc):
#                 self.constraints.append(int(low_constr[i]))
#             for i in range(high_alloc):
#                 self.constraints.append(int(high_constr[-1-i]))
#         elif low_alloc - high_alloc == 0 and low_alloc%2==0:
#             low_alloc -= 1
#             low_constr = np.linspace(1,self.r,low_alloc+1).round().astype(int)
#             high_constr = np.linspace(self.r,self.N,high_alloc+1).round().astype(int)
#             for i in range(low_alloc):
#                 self.constraints.append(int(low_constr[i]))
#             for i in range(high_alloc):
#                 self.constraints.append(int(high_constr[-1-i]))
#         else:
#             if high_alloc + low_alloc == self.cores and low_alloc == 0:
#                 high_alloc -=1
#             elif high_alloc + low_alloc == self.cores and high_alloc == 0:
#                 low_alloc -=1
#             low_constr = np.linspace(1,self.r,low_alloc+1).round().astype(int)
#             high_constr = np.linspace(self.r,self.N,high_alloc+1).round().astype(int)
#             for i in range(low_alloc):
#                 self.constraints.append(int(low_constr[i]))
#             for i in range(high_alloc):
#                 self.constraints.append(int(high_constr[-1-i]))
#         self.constraints = sorted(self.constraints,reverse=True)


#     def CSI_distance_constraint(self):
#         self.constraints=[]
#         self.constraints.append(self.r)
#         distances=[]
#         for i in range(1,self.N+1):
#             distances.append(self.model.nn_func(self.x_guess[i], self.params.alpha))
#         indx_sorted = np.argsort(np.array(distances).squeeze())[::-1]
#         i,j = 0,0
#         while i < self.cores - 1:
#             if (indx_sorted[j]+1)!=self.r:
#                 # if indx_sorted[j]==0:
#                 #     pass
#                 self.constraints.append(indx_sorted[j]+1)
#                 i+=1 
#             j+=1
#         self.constraints = sorted(self.constraints,reverse=True)
    
#     def step(self,x):
#         self.constraint_mode()
#         if len(self.constraints) != self.cores or len(self.constraints) != len(set(self.constraints)):
#             print(self.r)
#             print('ERROR')
#             print(self.constraints)
#         # print(self.r)
#         #print(self.constraints)
#         i = 0
#         failed = True
#         while (i < self.cores and (failed:=not(self.sing_step(x,int(self.constraints[i]))))):
#             #print(self.constraints[i])
#             i+=1
#         if not(failed):
#             self.core_solution = self.cores-i+1
#             self.step_old_solution = 0
#         else:
#             self.core_solution = None 
#             self.step_old_solution +=1
#         if self.r ==1:
#             print("NOT SOLVED")
#             _,self.x_viable = self.simulator.checkSafeIntegrate([x],self.u_guess,self.r)
#             print(self.x_viable)
#             print(f'is x viable:{self.model.nn_func(self.x_viable,self.model.params.alpha)}')
#             return None
#         self.r -= 1
#         #self.guessCorrection()
#         return self.provideControl()



# def unif_alloc(cores,fix,hor):
#     step = (hor-1)/cores
#     if fix == 1 or fix == hor or (fix-1)% step == 0:
#         l = np.linspace(1,hor,cores).round().astype(int).tolist()
#     else:
#         if fix < 1 + step:
#             l = np.linspace(fix,hor,cores).round().astype(int).tolist()
#         elif fix >  hor - step:
#             l = np.linspace(1,fix,cores).round().astype(int).tolist()
#         else:
#             l=[] 
#             l.append(fix)
#             portion_h = (cores-1)*((hor - fix)/(hor-1))
#             portion_h = int(portion_h) if portion_h-int(portion_h)<=0.5 else int(portion_h+1)
#             portion_l = (cores-1)*((fix-1)/(hor-1))
#             portion_l = int(portion_l) if portion_l-int(portion_l)<=0.5 else int(portion_l+1)
#             if portion_h == portion_l and cores%2==0 or portion_h +portion_l < cores -1:
#                 # print(portion_h)
#                 # print(portion_l)
#                 l = np.linspace(1,round(fix-step),portion_l).round().astype(int).tolist() + l \
#                       + np.linspace(round(fix+step),hor,portion_h+1).round().astype(int).tolist()
#             else: 
#                 l = np.linspace(1,round(fix-step),portion_l).round().astype(int).tolist() + l \
#                       + np.linspace(round(fix+step),hor,portion_h).round().astype(int).tolist()
#         if not(len(l)==cores):
#             print(f'length = cores ? {len(l)==cores}')
#             print(f'cores = {cores}, fix = {fix}')
#             print(l)
#         if len(l) != len(set(l)):
#             print(f'repeated arguments ? {len(l) != len(set(l))}')
        
            
            
#     return l
    