import os
import pickle
import numpy as np
from tqdm import tqdm
import safe_mpc.model as models
import safe_mpc.controller as controllers
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.abstract_multiphase import SimDynamics
from safe_mpc.gravity_compensation import GravityCompensation
from acados_template import AcadosOcpSolver
from datetime import datetime
import matplotlib.pyplot as plt
from copy import deepcopy


def convergenceCriteria(x, mask=None):
    if mask is None:
        mask = np.ones(model.nx)
    return np.linalg.norm(np.multiply(mask, x - model.x_ref)) < conf.conv_tol

def simulate_mpc(x0_g,x_guess,u_guess,controller):
    x0 = np.zeros((model.nx,))
    x0[:model.nq] = x0_g 

    x_sim = np.empty((conf.n_steps + 1, model.nx)) * np.nan
    u = np.empty((conf.n_steps, model.nu)) * np.nan
    x_sim[0] = x0

    controller.setGuess(x_guess, u_guess)
    if 'parallel' in controller.params.cont_type:
        controller.safe_hor = controller.N
    elif 'receding' in controller.params.cont_type:
        controller.r=controller.N
    controller.fails = 0
    stats = []
    x_simu = []
    u_simu= []
    jumps = []
    safe_hor_hist = []
    core_sol = []
    convergence = 0
    k = 0
    
    x_simu.append(x0)
    if 'parallel' in controller.params.cont_type:
        safe = controller.safe_hor
        safe_hor_hist.append(safe)
    elif controller.params.cont_type == 'receding':
        safe = controller.r
        safe_hor_hist.append(safe)
        
    for k in range(conf.n_steps):
        u[k] = controller.step(x_sim[k])
        #stats.append(controller.getTime())
        x_sim[k+1]=simulator.simulate(x_sim[k], u[k])
        
        x_simu.append(x_sim[k + 1])
        u_simu.append(u[k])
        
        if 'parallel' in controller.params.cont_type:
            jump = controller.safe_hor - safe
            safe = controller.safe_hor
            safe_hor_hist.append(safe)
            jumps.append(jump)
            core_sol.append(controller.core_solution)
        elif controller.params.cont_type == 'receding':
            jump = controller.r - safe
            safe = controller.r
            safe_hor_hist.append(safe)
            jumps.append(jump)
        if not model.checkStateConstraints(x_sim[k + 1]):
            print(f"{x0}=> FAILED")
            break
        # Check convergence --> norm of diff btw x_sim and x_ref (only for first joint)
        if convergenceCriteria(x_sim[k + 1], np.array([1, 0, 0, 1, 0, 0])):
            print(f"{x0}=> SUCCESS")
            convergence = 1
            break
        if k == conf.n_steps-1:
            print(f'{x0}=>Not converged\n')
            convergence = None
    x_v = controller.getLastViableState()
    return k, convergence, x_sim, stats, x_v, u,x_simu,u_simu,jumps,safe_hor_hist,core_sol

if __name__ == '__main__':
    available_controllers = {'naive': 'NaiveController',
                             'st': 'STController',
                             'stwa': 'STWAController',
                             'htwa': 'HTWAController',
                             'receding': 'RecedingController',
                             'receding_single':'RecedingSingleConstraint',
                             'parallel': 'ParallelWithCheck',
                             'parallel_limited':'ParallelLimited',
                             'abort': 'SafeBackupController'}
    
    # Define the configuration object, model, simulator and controller
    conf = Parameters('triple_pendulum', 'receding',rti=False)
    #conf_m = Parameters('triple_pendulum', 'receding_single',rti=False) 
    model = getattr(models,'TriplePendulumModel')(conf)
    #gc = GravityCompensation(conf, model)
    simulator = SimDynamics(model)
    simulator_m = SimDynamics(model)
    controller_m = getattr(controllers, available_controllers['receding_single'])(simulator_m)
    
    controller_r = getattr(controllers, available_controllers['receding'])(simulator)
    controller_r.setReference(model.x_ref)
    controller_m.setReference(model.x_ref)

    data_name = conf.DATA_DIR + 'receding' + '_'

    x0_vec = np.load(conf.DATA_DIR + f'x_init_{conf.alpha}.npy')
    x_guess_vec = np.load(data_name + f'x_guess_{conf.alpha}.npy')
    u_guess_vec = np.load(data_name + f'u_guess_{conf.alpha}.npy')

    
    for i in range(30):

        prob = np.random.randint(0,len(x0_vec))
        #prob = 294
        #controller.setGuess(x_guess_vec[prob],u_guess_vec[prob])
        #controller_m.setGuess(x_guess_vec[prob],u_guess_vec[prob])

        rr = np.random.randint(1,controller_r.N+1)
        #rr=34
        controller_r.r=rr
        print(controller_r.step(x_guess_vec[prob][0]))
        print(controller_r.model.checkSafeConstraints(controller_r.x_temp[10]))

        controller_m.r=rr
        print(controller_m.step(x_guess_vec[prob][0]))
        print(controller_m.model.checkSafeConstraints(controller_m.x_temp[10]))

        plt.figure(f'Control difference, constrained node {rr}, problem {prob}')
        plt.clf()
        plt.plot(controller_r.u_temp[:,0]-controller_m.u_temp[:,0], '--')
        plt.plot(controller_r.u_temp[:,1]-controller_m.u_temp[:,1], '-')
        plt.plot(controller_r.u_temp[:,2]-controller_m.u_temp[:,2], '-')
        plt.xlabel('t')
        plt.legend(['u1','u2','u3'])
        plt.grid()
        plt.axhline(y=controller_r.model.u_min[0], color='r', linestyle='-')
        plt.axhline(y=controller_r.model.u_max[0], color='r', linestyle='-')

        plt.show()

    # plt.figure(f'Control receding_single')
    # plt.clf()
    # plt.plot(controller_m.u_temp[:,0], '--')
    # plt.plot(controller_m.u_temp[:,1], '-')
    # plt.plot(controller_m.u_temp[:,2], '-')
    # plt.xlabel('t')
    # plt.legend(['u1','u2','u3'])
    # plt.grid()
    # plt.axhline(y=controller.model.u_min[0], color='r', linestyle='-')
    # plt.axhline(y=controller.model.u_max[0], color='r', linestyle='-')


    #    plt.show()    
    

    pass
    
    