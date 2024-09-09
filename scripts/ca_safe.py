import casadi as ca
import scipy.linalg
import safe_mpc.parser as parameters
import safe_mpc.model as models
import safe_mpc.abstract as abstract
from scipy.stats import qmc
import numpy as np
import scipy
import os
import pickle

def load_data(control,alpha,min_negative_jump,err_thr,mode=None,cores=None):
    folder = os.path.join(os.getcwd(),'DATI_PARALLELIZED')
    files = os.listdir(folder)
    for i in files:
        if 'Thr'+str(err_thr) in i and control in i and str(alpha) in i \
            and str(min_negative_jump) in i:
                if control  == 'ParallelLimited':
                    if 'cores'+str(cores) in i and mode in i:
                        path = os.path.join(folder,i +'/'+i+'x_viable.npy')
                        data_loaded = np.load(path)
                        break
                else:
                    path = os.path.join(folder,i +'/'+i+'x_viable.npy')
                    data_loaded = np.load(path)
                    break
    path = os.path.join(folder,i +'/'+i+'x_viable.npy')
    print(path)
    return data_loaded

system = 'triple_pendulum'

control = 'abort'
abort = 'parallel2'
if abort == 'parallel_limited':
    # mode CIS, uni or high
    mode = 'uni'
    cores = 4



q_dot_gain=8e7

conf = parameters.Parameters('triple_pendulum', control,rti=False)
model = getattr(models,'TriplePendulumModel')(conf)
sampler = qmc.Halton(model.nq, scramble=False)
l_bounds = model.x_min[:model.nq] + conf.state_tol
u_bounds = model.x_max[:model.nq] - conf.state_tol
x_viable = load_data('ParallelLimited',15,0,1e-3,'high',16)
f_expl = model.f_expl
opti = ca.Opti()
model.setNNmodel()
nn_safe = model.nn_func

dynamics = ca.Function('dyn', [model.x, model.u], [model.f_expl])

# # Objective term
Q = 1e-4 * np.eye(model.nx)
Q[model.nq:, model.nq:] = np.eye(model.nv) * q_dot_gain

R = 1e-4 * np.eye(model.nu)

W = scipy.linalg.block_diag(Q,R)
W_e = Q

x_ref = model.x_ref
x=model.x
u=model.u
L = (x).T@Q@(x) + u.T@R@u 
L_end = (x).T@Q@(x)
L_end=ca.Function('L_end', [model.x], [L_end])

# Formulate discrete time dynamics
if False:
   # CVODES from the SUNDIALS suite
   dae = {'x':x, 'p':u, 'ode':model.f_expl, 'quad':L}
   F = ca.integrator('F', 'cvodes', dae, 0, 0.005)
else:
   # Fixed step Runge-Kutta 4 integrator
   M = 4 # RK4 steps per interval
   DT = conf.dt
   #f = Function('f', [x, u], [xdot, L])
   f=ca.Function('dyn', [model.x, model.u], [model.f_expl,L])
   X0 = ca.MX.sym('X0', model.nx)
   U = ca.MX.sym('U',model.nu)
   X = X0
   Q = 0
   for j in range(M):
       k1, k1_q = f(X, U)
       k2, k2_q = f(X + DT/2 * k1, U)
       k3, k3_q = f(X + DT/2 * k2, U)
       k4, k4_q = f(X + DT * k3, U)
       X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
       Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)
   F = ca.Function('F', [X0, U], [X, Q],['x0','u'],['xf','qf'])

# Evaluate at a test point
Fk = F(x0=[0.2,0.3,0.1,0,0,0],u=[0,0,0])
print(Fk['xf'])
print(Fk['qf'])

if nn_safe(x_viable[0],conf.alpha) >=0:
    print('SAFE')
    print(len(x_viable))
successes=0
for m in range(x_viable.shape[0]):
    # Start with an empty NLP
    w=[]
    w0 = []
    lbw = []
    ubw = []
    J = 0
    g=[]
    lbg = []
    ubg = []

    # initial conditions
    x0 = np.zeros((model.nx,))
    x0 = x_viable[m]
    print(x0)
    Xk = ca.MX.sym('X0', 6)
    w += [Xk]
    lbw += x0.tolist()
    ubw += x0.tolist()
    w0  += x0.tolist()

    # lbw += [-ca.inf]*6
    # ubw += [ca.inf]*6
    # w0  += [0]*6
    N=100
    # Formulate the NLP
    for k in range(N):
        # New NLP variable for the control
        Uk = ca.MX.sym('U_' + str(k),3)
        w   += [Uk]
        lbw += [-10]*3
        ubw += [10]*3
        w0  += [0]*3

        # Integrate till the end of the interval
        Fk = F(x0=Xk, u=Uk)
        Xk_end = Fk['xf']
        J=J+Fk['qf']

        # New NLP variable for state at end of interval
        Xk = ca.MX.sym('X_' + str(k+1), 6)
        w   += [Xk]
        lbw += model.x_min.tolist()
        ubw += model.x_max.tolist()
        w0  += [0]*model.nx

        # Add equality constraint
        g   += [Xk_end-Xk]
        lbg += [0]*model.nx
        ubg += [0]*model.nx

        if k == N-1:
            g+=[Xk_end]
            lbg += [-ca.inf,-ca.inf,-ca.inf,-1e-2,-1e-2,-1e-2]
            ubg += [ca.inf,ca.inf,ca.inf,1e-2,1e-2,1e-2]
    J=J+L_end(Xk)

    # Create an NLP solver
    prob = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g)}
    solver = ca.nlpsol('solver', 'ipopt', prob,{'ipopt':{'max_iter':1000, 'tol':1e-5}})

    # Solve the NLP
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    w_opt = sol['x'].full().flatten()
    if solver.stats()['success']: successes +=1
    if False:
        # Plot the solution
        x1_opt = w_opt[0::9]
        x2_opt = w_opt[1::9]
        x3_opt = w_opt[2::9]
        x4_opt = w_opt[3::9]
        x5_opt = w_opt[4::9]
        x6_opt = w_opt[5::9]
        u1_opt = w_opt[6::9]
        u2_opt = w_opt[7::9]
        u3_opt = w_opt[8::9]



        tgrid = [(N*DT)/N*k for k in range(N+1)]
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.clf()
        plt.plot(tgrid, x4_opt, '-')
        plt.plot(tgrid, x5_opt, '-')
        plt.plot(tgrid, x6_opt, '-')
        plt.xlabel('t')
        plt.legend(['dx1','dx2','dx3'])
        plt.grid()

        plt.figure(2)
        plt.clf()
        plt.plot(tgrid, x1_opt, '-')
        plt.plot(tgrid, x2_opt, '-')
        plt.plot(tgrid, x3_opt, '-')
        plt.xlabel('t')
        plt.legend(['x1','x2','x3'])
        plt.grid()

        plt.show()



        integrated_sol=np.array(x0) 
        for i in range(len(u1_opt)):
            if i == 0:
                integrated_sol= np.vstack([integrated_sol,np.array(F(x0=integrated_sol, u=[u1_opt[i],u2_opt[i],u3_opt[i]])['xf']).squeeze()])
            else:
                integrated_sol= np.vstack([integrated_sol,np.array(F(x0=integrated_sol[i,:], u=[u1_opt[i],u2_opt[i],u3_opt[i]])['xf']).squeeze()])
        plt.figure(2)
        plt.clf()
        plt.plot(tgrid, integrated_sol[:,3], '--')
        plt.plot(tgrid, integrated_sol[:,4], '-')
        plt.plot(tgrid, integrated_sol[:,5], '-')
        plt.xlabel('t')
        plt.legend(['dx1','dx2','dx3'])
        plt.grid()
        plt.show()
    print(f'{successes} over {m+1}')