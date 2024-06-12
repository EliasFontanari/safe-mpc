import numpy as np
import scipy.linalg as lin
from .abstract import AbstractController


class NaiveController(AbstractController):
    def __init__(self, simulator):
        super().__init__(simulator)

    def checkStateConstraintsController(self, x):
        return np.all(np.logical_and(x >= self.model.x_min, x <= self.model.x_max))
    
    def checkControlConstraintsController(self, u):
        return np.all(np.logical_and(u >= self.model.u_min, u <= self.model.u_max))

    def checkRunningConstraintsController(self, x, u):
        return self.checkStateConstraintsController(x) and self.checkControlConstraintsController(u)

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
        if (status == 0 or status == 2) and self.checkGuess():
            self.x_guess = np.copy(self.x_temp)
            self.u_guess = np.copy(self.u_temp)
            return 1
        return 0

    def step(self, x):
        status = self.solve(x)
        if status == 0 and self.model.checkControlConstraints(self.u_temp[0]): #and \
        #    self.simulator.checkDynamicsConstraints(self.x_temp[:2], np.array([self.u_temp[0]])):
            self.fails = 0
        else:
            self.fails += 1
        return self.provideControl()


class STController(NaiveController):
    def __init__(self, simulator):
        super().__init__(simulator)

    def additionalSetting(self):
        self.terminalConstraint()
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
        self.alternative_x_guess = self.x_guess
        self.alternative_u_guess = self.u_guess 

    def additionalSetting(self):
        # Terminal constraint before, since it construct the nn model
        self.terminalConstraint()
        self.runningConstraint()

    def solve(self, x0,constr_nodes=None,alternative_guess=None):
        # Reset current iterate
        self.ocp_solver.reset()

        # Constrain initial state
        self.ocp_solver.constraints_set(0, "lbx", x0)
        self.ocp_solver.constraints_set(0, "ubx", x0)

        y_ref = np.zeros(self.model.ny)
        y_ref[:self.model.nx] = self.x_ref
        W = lin.block_diag(self.Q, self.R)

        if alternative_guess == None:
            for i in range(self.N):
                self.ocp_solver.set(i, 'x', self.x_guess[i])
                self.ocp_solver.set(i, 'u', self.u_guess[i])
                self.ocp_solver.cost_set(i, 'yref', y_ref, api='new')
                self.ocp_solver.cost_set(i, 'W', W, api='new')
            self.ocp_solver.set(self.N, 'x', self.x_guess[-1])
            self.ocp_solver.cost_set(self.N, 'yref', y_ref[:self.model.nx], api='new')
            self.ocp_solver.cost_set(self.N, 'W', self.Q, api='new')
        else:
            for i in range(self.N):
                self.ocp_solver.set(i, 'x', self.alternative_x_guess[i])
                self.ocp_solver.set(i, 'u', self.alternative_u_guess[i])
                self.ocp_solver.cost_set(i, 'yref', y_ref, api='new')
                self.ocp_solver.cost_set(i, 'W', W, api='new')
            self.ocp_solver.set(self.N, 'x', self.alternative_x_guess[-1])
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

        r_new = -1
        for i in range(1, self.N + 1):
            if self.model.checkSafeConstraints(self.x_temp[i]):
                r_new = i - 1

        if status == 0 and self.checkRunningConstraintsController(self.x_temp, self.u_temp) and r_new > 0:
            self.fails = 0
            self.r = r_new
        else:
            if self.r == 1:
                self.x_viable = np.copy(self.x_guess[self.r])
                return None
            self.fails += 1
            self.r -= 1
        return self.provideControl()


class ParallelController(RecedingController):
    def __init__(self, simulator):
        super().__init__(simulator)
        self.n_prob =  self.N - 1
        self.safe_hor = self.N
        self.alternative_x_guess = self.x_guess
        self.alternative_u_guess = self.u_guess
        #self.ocp.cost.zl_e = np.zeros((1,))
        #self.ocp_solver.cost_set(self.N, "zl", np.zeros((1,)))

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
    
    def constrain_n(self,n_constr):
        for i in range(1, self.N+1):
            self.ocp_solver.cost_set(i, "zl", np.zeros((1,)))
        #self.ocp_solver.acados_ocp.cost.zl_e = np.array([0.])
        if 0 < n_constr <= self.N:
            self.ocp_solver.cost_set(n_constr, "zl", self.params.ws_r * np.ones((1,)))


    def sing_step(self, x, n_constr):
        success = False

        self.constrain_n(n_constr)
        status = self.solve(x,[n_constr],alternative_guess=None)
        checked_r= self.check_safe_n()

        if (check:=self.model.checkSafeConstraints(self.x_temp[n_constr])) or checked_r>1:
            constr_ver = n_constr if check else 0
            n_step_safe = max(constr_ver,checked_r)
            # Comment this line to use parallel with check
            n_step_safe = constr_ver
            success = True

        if success and status == 0 and self.checkRunningConstraintsController(self.x_temp, self.u_temp) and n_step_safe>=self.safe_hor:
            self.fails = 0 
            self.safe_hor = n_step_safe
            
            for i in range(self.N):
                self.alternative_x_guess[i] = self.ocp_solver.get(i, "x")
                self.alternative_u_guess[i] = self.ocp_solver.get(i, "u")
            self.alternative_x_guess[-1] = self.ocp_solver.get(self.N, "x")

            # if  n_step_safe>=self.safe_hor:
            #     self.fails = 0 
            #     self.safe_hor = n_step_safe
            # else:
            #     self.fails +=1
        
        else:
            self.fails +=1
            success = False

        return success
    
    def step(self,x):
        i = self.N
        failed = True
        while (i > (self.N-self.n_prob)) and (failed:=not(self.sing_step(x,i))):
            i-=1
            #print(i)
        # if not(solved) and i == (self.N-self.n_prob):
        #     #self.x_viable = np.copy(self.x_guess[-1])
        #     #if self.x_guess
        #     #return None
        solved=not(failed)
        if not(solved) and self.safe_hor < 2:
            print("NOT SOLVED")
            return None
        self.safe_hor -= 1
        return self.provideControl()

class ParallelWithCheck(RecedingController):
    def __init__(self, simulator):
        super().__init__(simulator)
        #self.n_prob =  self.N - 1
        self.safe_hor = self.N
        #self.ocp.cost.zl_e = np.zeros((1,))
        #self.ocp_solver.cost_set(self.N, "zl", np.zeros((1,)))

    def checkGuess(self):
        return self.model.checkRunningConstraints(self.x_temp, self.u_temp) and \
               self.simulator.checkDynamicsConstraints(self.x_temp, self.u_temp) and \
               self.model.checkSafeConstraints(self.x_temp[-1])


    def constrain_n(self,n_constr):
        for i in range(1, self.N+1):
            self.ocp_solver.cost_set(i, "zl", np.zeros((1,)))
        #self.ocp_solver.acados_ocp.cost.zl_e = np.array([0.])
        if 0 < n_constr <= self.N:
            self.ocp_solver.cost_set(n_constr, "zl", self.params.ws_r * np.ones((1,)))

    def check_r(self):
        r=0
        for i in range(1, self.N + 1):
            if self.model.checkSafeConstraints(self.x_temp[i]):
                r = i
        return r


    def sing_step(self, x, n_constr):
        r=0
        success = False

        # for i in range(0, self.N):
        #     self.ocp_solver.cost_set(i, "zl", np.zeros((1,)))
        # self.ocp_solver.cost_set(n_constr, "zl", self.params.ws_r * np.ones((1,)))
        # Solve the OCP

        self.constrain_n(n_constr)
        status = self.solve(x)
        r = self.check_r()

        if (check:=self.model.checkSafeConstraints(self.x_temp[n_constr])) or r > 0:
            success = True
            constr_ver = n_constr if check else 0
            r = max(r,constr_ver)
            # if r> n_constr:
            #     print(r)


        if status == 0 and success and self.model.checkRunningConstraints(self.x_temp, self.u_temp)\
              and self.model.checkSafeConstraints(self.x_temp[r]):# and r>=self.safe_hor: #and success:
            #self.fails = 0
            success = True

        else:
            #self.fails +=1
            success = False

        return r if success else 0

    def step(self,x):
        results = []
        for i in range(1,self.N+1):
            results.append(self.sing_step(x,i))
        r = results[-1]
        r_indx = len(results)-1
        for i in range(self.N-1,-1,-1):
            if results[i] > r:
                r=results[i]
                r_indx = i
        #r = int(np.argmax(results))+1
        if 1 < r  and r > self.safe_hor:
            self.constrain_n(r_indx+1)
            status=self.solve(x) 
            status+=not(self.model.checkSafeConstraints(self.x_temp[r]))
            if status==1:
                pass
            self.fails = 0 + status
            self.safe_hor = r
            #print(r)
        elif r<=1 and self.safe_hor <= 1:
            print("NOT SOLVED")
            return None
        else:
            self.fails += 1  
        self.safe_hor -= 1
        

        return self.provideControl()


class SafeBackupController(AbstractController):
    def __init__(self, simulator):
        super().__init__(simulator)

    def additionalSetting(self):
        self.Q = np.zeros((self.model.nx, self.model.nx))
        self.Q[self.model.nq:, self.model.nq:] = np.eye(self.model.nv) * self.params.q_dot_gain

        self.ocp.cost.W = lin.block_diag(self.Q, self.R)
        self.ocp.cost.W_e = self.Q

        # q_fin_lb = np.hstack([self.model.x_min[:self.model.nq], np.zeros(self.model.nv)])
        # q_fin_ub = np.hstack([self.model.x_max[:self.model.nq], np.zeros(self.model.nv)])

        # self.ocp.constraints.lbx_e = q_fin_lb
        # self.ocp.constraints.ubx_e = q_fin_ub
        # self.ocp.constraints.idxbx_e = np.arange(self.model.nx)
