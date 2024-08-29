import casadi as ca
import scipy.linalg
import safe_mpc.casadiParam as param
import safe_mpc.model as models
import safe_mpc.abstract as abstract
from scipy.stats import qmc
import numpy as np
import scipy

def solve(x_i):
    opti.set_value(x, x_i)
    sol = opti.solve()
    return sol.value(u)

system = 'triple_pendulum'

conf = param.Parameters(system)
model = getattr(models, 'TriplePendulumModel')(conf)

sampler = qmc.Halton(model.nq, scramble=False)
l_bounds = model.x_min[:model.nq] + conf.state_tol
u_bounds = model.x_max[:model.nq] - conf.state_tol
x_g0 = qmc.scale(sampler.random(), l_bounds, u_bounds)
f_expl = model.f_expl
opti = ca.Opti()
model.setNNmodel()
nn_safe = model.nn_func
# x=opti.variable(model.nx)
# u = opti.variable(model.nu)
dynamics = ca.Function('dyn', [model.x, model.u], [model.f_expl[model.nq:]])

# opti.minimize(-nn_safe(x,conf.alpha))
# opti.subject_to(nn_safe(x,conf.alpha) >= 0)    
# #self.opti.subject_to(self.opti.bounded(self.model.u_min, self.u, self.model.u_max))
# opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes','ipopt.max_iter' : 200}
# opti.solver('ipopt', opts)
# x0 = np.zeros((model.nx,))
# x0[:model.nq] = x_g0
# opti.set_initial(x, x0)
# sol = opti.solve()
# print(sol.value(x))


# # Objective term
Q = 1e-4 * np.eye(model.nx)
Q[0, 0] = 5e2
Q[1,1] = 1e-4#0.65e2  #0.65
Q[2,2] = 1e-4#0.65e2  #0.65
R = 1e-4 * np.eye(model.nu)

W = scipy.linalg.block_diag(Q,R)
W_e = Q

x_ref = model.x_ref
x=model.x
u=model.u
L = (x-x_ref).T@Q@(x-x_ref) + u.T@R@u 

# Formulate discrete time dynamics
if False:
   # CVODES from the SUNDIALS suite
   dae = {'x':x, 'p':u, 'ode':xdot, 'quad':L}
   F = integrator('F', 'cvodes', dae, 0, T/N)
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
x0[:model.nq] = x_g0
Xk = ca.MX.sym('X0', 6)
w += [Xk]
# lbw += x0.tolist()
# ubw += x0.tolist()
# w0  += x0.tolist()

lbw += [-ca.inf]*6
ubw += [ca.inf]*6
w0  += [0]*6
N=35
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
        g+=[nn_safe(Xk_end,conf.alpha)]
        lbg += [0]
        ubg += [ca.inf]

# Create an NLP solver
prob = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g)}
solver = ca.nlpsol('solver', 'ipopt', prob)
solver = ca.nlpsol('solver', 'ipopt', prob,{'ipopt':{'max_iter':100, 'tol':1e-5}})


# Solve the NLP
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
w_opt = sol['x'].full().flatten()

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

if nn_safe(w_opt[-6:],conf.alpha) >=0:
    print('SAFE')

tgrid = [(N*DT)/N*k for k in range(N+1)]
import matplotlib.pyplot as plt
plt.figure(1)
plt.clf()
plt.plot(tgrid, x1_opt, '--')
plt.plot(tgrid, x2_opt, '-')
plt.step(tgrid, ca.vertcat(ca.DM.nan(1), u1_opt), '-.')
plt.xlabel('t')
plt.legend(['x1','x2','u'])
plt.grid()
plt.show()