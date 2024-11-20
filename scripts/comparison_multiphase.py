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
    elif 'receding' in controller.params.cont_type:
        safe = controller.r
        safe_hor_hist.append(safe)
        
    for k in range(conf.n_steps):
        u[k] = controller.step(x_sim[k])
        stats.append(controller.getTime())
        x_sim[k+1]=simulator.simulate(x_sim[k], u[k])
        
        x_simu.append(x_sim[k + 1])
        u_simu.append(u[k])
        
        if 'parallel' in controller.params.cont_type:
            jump = controller.safe_hor - safe
            safe = controller.safe_hor
            safe_hor_hist.append(safe)
            jumps.append(jump)
            core_sol.append(controller.core_solution)
        elif 'receding' in controller.params.cont_type:
            jump = controller.r - safe
            safe = controller.r
            safe_hor_hist.append(safe)
            jumps.append(jump)
        if not model.checkStateConstraints(x_sim[k + 1]):
            print(f"{x0}=> FAILED at step {k}")
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
    #conf = Parameters('triple_pendulum', 'receding',rti=True)
    conf = Parameters('triple_pendulum', 'receding_single',rti=True) 
    #conf.cont_type = 'receding2'
    model = getattr(models,'TriplePendulumModel')(conf)
    #gc = GravityCompensation(conf, model)
    simulator = SimDynamics(model)
    #simulator_m = SimDynamics(model)
    controller_m = getattr(controllers, available_controllers['receding_single'])(simulator)
    
    controller_r = getattr(controllers, available_controllers['receding'])(simulator)
    controller_r.setReference(model.x_ref)
    controller_m.setReference(model.x_ref)

    data_name = conf.DATA_DIR + 'receding' + '_'

    x0_vec = np.load(conf.DATA_DIR + f'x_init_{conf.alpha}.npy')
    x_guess_vec = np.load(data_name + f'x_guess_{conf.alpha}.npy')
    u_guess_vec = np.load(data_name + f'u_guess_{conf.alpha}.npy')

    data_name = conf.DATA_DIR + 'receding_single' + '_'
    x0_vec_m = np.load(conf.DATA_DIR + f'x_init_{conf.alpha}.npy')
    x_guess_vec_m = np.load(data_name + f'x_guess_{conf.alpha}.npy')
    u_guess_vec_m = np.load(data_name + f'u_guess_{conf.alpha}.npy')

    res_r, res_m = [],[]
    r_succ, m_succ=0,0
    for i in range(0,len(x0_vec)):
        print(i)
        prob = np.random.randint(0,len(x0_vec))
        prob=i
        #prob = 294
        # controller_r.setGuess(x_guess_vec[prob],u_guess_vec[prob])
        res_r.append(simulate_mpc(x0_vec[prob],x_guess_vec[prob],u_guess_vec[prob],controller_r))
        #controller_m.setGuess(x_guess_vec[prob],u_guess_vec[prob])
        res_m.append(simulate_mpc(x0_vec_m[prob],x_guess_vec_m[prob],u_guess_vec_m[prob],controller_m))
        if res_r[i][1]==1:
            r_succ+=1
        if res_m[i][1]==1:
            m_succ+=1
        if False:
            plt.figure(f'Receding trajectory {prob}')
            plt.clf()
            plt.plot(np.array(res_r[i][2])[:,0], '-')
            plt.plot(np.array(res_r[i][2])[:,1], '-')
            plt.plot(np.array(res_r[i][2])[:,2], '-')
            plt.plot(np.array(res_r[i][2])[:,3], '-')
            plt.plot(np.array(res_r[i][2])[:,4], '-')
            plt.plot(np.array(res_r[i][2])[:,5], '-')
            plt.plot(np.array(res_r[i][9]))
            plt.xlabel('t')
            plt.legend(['x1','x2','x3','dx1','dx2','dx3'])
            plt.grid()
            # plt.axhline(y=controller.model.u_min[0], color='r', linestyle='-')
            # plt.axhline(y=controller.model.u_max[0], color='r', linestyle='-')
            plt.figure(f'Receding_single trajectory {prob}')
            plt.clf()
            plt.plot(np.array(res_m[i][2])[:,0], '-')
            plt.plot(np.array(res_m[i][2])[:,1], '-')
            plt.plot(np.array(res_m[i][2])[:,2], '-')
            plt.plot(np.array(res_m[i][2])[:,3], '-')
            plt.plot(np.array(res_m[i][2])[:,4], '-')
            plt.plot(np.array(res_m[i][2])[:,5], '-')
            plt.plot(np.array(res_m[i][9]))
            plt.xlabel('t')
            plt.legend(['x1','x2','x3','dx1','dx2','dx3'])
            plt.grid()

            plt.show()
    print(f'receding: {r_succ}/{len(x0_vec)}, receding_single: {m_succ}/{len(x0_vec)}')
    
    _, _, _, t_stats, _, _,_,_,_,_,_ = zip(*res_r)
    print('99% quantile computation time receding:')
    times = np.array([t for arr in t_stats for t in arr])
    for field, t in zip(controller_r.time_fields, np.quantile(times, 0.97, axis=0)):
        print(f"{field:<20} -> {t}")
    
    _, _, _, t_stats, _, _,_,_,_,_,_ = zip(*res_m)
    print('99% quantile computation time receding_single:')
    times = np.array([t for arr in t_stats for t in arr])
    for field, t in zip(controller_m.time_fields, np.quantile(times, 0.97, axis=0)):
        print(f"{field:<20} -> {t}")
    #rr = np.random.randint(1,controller.N+1)
    #rr=34
    #controller.r=rr
    #print(controller.step(x_guess_vec[prob][0]))
    #print(controller.model.checkSafeConstraints(controller.x_temp[10]))

    #controller_m.r=rr
    #print(controller_m.step(x_guess_vec[prob][0]))
    #print(controller_m.model.checkSafeConstraints(controller_m.x_temp[10]))
    
    # for i in range(30):

    #     prob = np.random.randint(0,len(x0_vec))
    #     prob = 294
    #     controller.setGuess(x_guess_vec[prob],u_guess_vec[prob])
    #     controller_m.setGuess(x_guess_vec[prob],u_guess_vec[prob])

    #     rr = np.random.randint(1,controller.N+1)
    #     rr=34
    #     controller.r=rr
    #     print(controller.step(x_guess_vec[prob][0]))
    #     print(controller.model.checkSafeConstraints(controller.x_temp[10]))

    #     controller_m.r=rr
    #     print(controller_m.step(x_guess_vec[prob][0]))
    #     print(controller_m.model.checkSafeConstraints(controller_m.x_temp[10]))

    #     plt.figure(f'Control difference, constrained node {rr}, problem {prob}')
    #     plt.clf()
    #     plt.plot(controller.u_temp[:,0]-controller_m.u_temp[:,0], '--')
    #     plt.plot(controller.u_temp[:,1]-controller_m.u_temp[:,1], '-')
    #     plt.plot(controller.u_temp[:,2]-controller_m.u_temp[:,2], '-')
    #     plt.xlabel('t')
    #     plt.legend(['u1','u2','u3'])
    #     plt.grid()
    #     plt.axhline(y=controller.model.u_min[0], color='r', linestyle='-')
    #     plt.axhline(y=controller.model.u_max[0], color='r', linestyle='-')

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
    
    