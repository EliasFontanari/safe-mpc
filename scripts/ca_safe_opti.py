import casadi as ca
import scipy.linalg
import safe_mpc.casadiParam as param
import safe_mpc.model as models
import safe_mpc.abstract as abstract
from scipy.stats import qmc
import numpy as np
import scipy

system = 'triple_pendulum'

q_dot_gain=1e7

conf = param.Parameters(system)
model = getattr(models, 'TriplePendulumModel')(conf)
cont_type = 'parallel'
sampler = qmc.Halton(model.nq, scramble=False)
l_bounds = model.x_min[:model.nq] + conf.state_tol
u_bounds = model.x_max[:model.nq] - conf.state_tol
x_viable = np.load(conf.DATA_DIR + cont_type + '_' + 'x_viable.npy')
f_expl = model.f_expl
model.setNNmodel()
nn_safe = model.nn_func

dynamics = ca.Function('dyn', [model.x, model.u], [model.f_expl])

N = 100 # number of control intervals

opti = ca.Opti() # Optimization problem

# ---- decision variables ---------
X = opti.variable(6,N+1) # state trajectory
pos   = X[0,:]
speed = X[1,:]
U = opti.variable(3,N)   # control trajectory (throttle)

# ---- objective          ---------
# # Objective term
Q = 1e-4 * np.eye(model.nx)
Q[model.nq:, model.nq:] = np.eye(model.nv) * q_dot_gain

R = 1e-4 * np.eye(model.nu)

W = scipy.linalg.block_diag(Q,R)
W_e = Q

L = lambda x,u :(x).T@Q@(x) + u.T@R@u 

cost = 0
for i in range(N-1):
    cost+= L(X[:,i],U[:,i])
    opti.subject_to(X[:,i]<=model.x_max.tolist())
    opti.subject_to(X[:,i]>=model.x_min.tolist())
    opti.subject_to(U[:,i]<=model.u_max.tolist())
    opti.subject_to(U[:,i]>=model.u_min.tolist())    
cost += (X[:,N-1]).T@Q@(X[:,N-1])
opti.subject_to(X[:3,N]<=model.x_max[:3].tolist())
opti.subject_to(X[:3,N]<=model.x_min[:3].tolist())
opti.subject_to(X[3:,N]<=1e-2)
opti.subject_to(X[3:,N]>=-1e-2)

opti.minimize(cost) 
            
# ---- dynamic constraints --------
#f = lambda x,u: vertcat(x[1],u-x[1]) # dx/dt = f(x,u)
f = dynamics

dt = conf.dt # length of a control interval
for k in range(N): # loop over control intervals
   # Runge-Kutta 4 integration
   k1 = f(X[:,k],         U[:,k])
   k2 = f(X[:,k]+dt/2*k1, U[:,k])
   k3 = f(X[:,k]+dt/2*k2, U[:,k])
   k4 = f(X[:,k]+dt*k3,   U[:,k])
   x_next = X[:,k] + dt/6*(k1+2*k2+2*k3+k4) 
   opti.subject_to(X[:,k+1]==x_next) # close the gaps

# ---- path constraints -----------
#limit = lambda pos: 1-sin(2*pi*pos)/2
# opti.subject_to(X<=model.x_max)   # track speed limit
# opti.subject_to(X>=model.x_min) 
# opti.subject_to(U<=model.u_max) # control is limited
# opti.subject_to(U>=model.u_min) # control is limited


# ---- boundary conditions --------
x0=x_viable[1]
opti.subject_to(X[:,0]==x0)   # start at position 0 ...
#opti.subject_to(pos[-1]==1)  # finish line at position 1


# # ---- initial values for solver ---
# opti.set_initial(speed, 1)
# opti.set_initial(T, 1)

# ---- solve NLP              ------
opti.solver("ipopt") # set numerical backend
sol = opti.solve()   # actual solve
