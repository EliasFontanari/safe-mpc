import multiprocessing
import os
import pickle
import numpy as np
from tqdm import tqdm
import safe_mpc.model as models
import safe_mpc.controller as controllers
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.abstract import SimDynamics
from safe_mpc.gravity_compensation import GravityCompensation
from acados_template import AcadosOcpSolver
from datetime import datetime
from scipy.stats import qmc

def convergenceCriteria(x, mask=None):
    if mask is None:
        mask = np.ones(model.nx)
    return np.linalg.norm(np.multiply(mask, x - model.x_ref)) < conf.conv_tol

# def simulate_mpc(x0_g,x_guess,u_guess):
#     x0 = np.zeros((model.nx,))
#     x0[:model.nq] = x0_g 

#     x_sim = np.empty((conf.n_steps + 1, model.nx)) * np.nan
#     u = np.empty((conf.n_steps, model.nu)) * np.nan
#     x_sim[0] = x0

#     controller.setGuess(x_guess, u_guess)
#     if 'parallel' in control:
#         controller.safe_hor = controller.N
#     elif control == 'receding':
#         controller.r=controller.N
#     controller.fails = 0
#     stats = []
#     x_simu = []
#     u_simu= []
#     jumps = []
#     safe_hor_hist = []
#     core_sol = []
#     convergence = 0
#     k = 0
    
#     x_simu.append(x0)
#     if 'parallel' in control:
#         safe = controller.safe_hor
#         safe_hor_hist.append(safe)
#     elif control == 'receding':
#         safe = controller.r
#         safe_hor_hist.append(safe)
        
#     for k in range(conf.n_steps):
#         u[k] = controller.step(x_sim[k])
#         stats.append(controller.getTime())
#         x_sim[k+1]=simulator.simulate(x_sim[k], u[k])
        
#         x_simu.append(x_sim[k + 1])
#         u_simu.append(u[k])
        
#         if 'parallel' in control:
#             jump = controller.safe_hor - safe
#             safe = controller.safe_hor
#             safe_hor_hist.append(safe)
#             jumps.append(jump)
#             core_sol.append(controller.core_solution)
#         elif control == 'receding':
#             jump = controller.r - safe
#             safe = controller.r
#             safe_hor_hist.append(safe)
#             jumps.append(jump)
#         if not model.checkStateConstraints(x_sim[k + 1]):
#             with counter.get_lock():
#                 counter.value += 1
#                 print(f"{counter.value}:{x0}=> FAILED")
#             break
#         # Check convergence --> norm of diff btw x_sim and x_ref (only for first joint)
#         if convergenceCriteria(x_sim[k + 1], np.array([1, 0, 0, 1, 0, 0])):
#             with counter.get_lock():
#                 counter.value += 1
#                 print(f"{counter.value}:{x0}=> SUCCESS")
#                 with counter_success.get_lock():
#                     counter_success.value += 1
#             convergence = 1
#             break
#         if k == conf.n_steps-1:
#             with counter.get_lock():
#                 counter.value += 1
#                 print(f'{counter.value}:{x0}=>Not converged\n')
#                 convergence = None
#     x_v = controller.getLastViableState()
#     return k, convergence, x_sim, stats, x_v, u,x_simu,u_simu,jumps,safe_hor_hist,core_sol

def simulate_mpc(x0_g,x_guess,u_guess):
    x0 = np.zeros((model.nx,))
    x0[:model.nq] = x0_g 

    x_sim = np.empty((conf.n_steps + 1, model.nx)) * np.nan
    u = np.empty((conf.n_steps, model.nu)) * np.nan
    x_sim[0] = x0

    controller.setGuess(x_guess, u_guess)
    if 'parallel' in controller.params.cont_type:
        controller.r = controller.N
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
        safe = controller.r
        safe_hor_hist.append(safe)
    elif 'receding' in controller.params.cont_type:
        safe = controller.r
        safe_hor_hist.append(safe)
        
    for k in range(conf.n_steps):
        #print(k)
        u[k] = controller.step(x_sim[k])
        if k==72:
            pass     
        if not('parallel' in  controller.params.cont_type):   
            stats.append(controller.getTime())
        else:
            for l in range(len(controller.times)):
                stats.append(controller.times[l])
        x_sim[k+1]=simulator.simulate(x_sim[k], u[k])
        
        x_simu.append(x_sim[k + 1])
        u_simu.append(u[k])
        
        if 'parallel' in controller.params.cont_type:
            jump = controller.r - safe
            safe = controller.r
            safe_hor_hist.append(safe)
            jumps.append(jump)
            core_sol.append(controller.core_solution)
        elif 'receding' in controller.params.cont_type:
            jump = controller.r - safe
            safe = controller.r
            safe_hor_hist.append(safe)
            jumps.append(jump)
        if not model.checkStateConstraints(x_sim[k + 1]):
            with counter.get_lock():
                counter.value += 1
                print(f"{counter.value}:{x0}=> FAILED")
                with counter_fails.get_lock():
                    counter_fails.value += 1
                    print(f"Number of failures: {counter_fails.value}")
            break
        # Check convergence --> norm of diff btw x_sim and x_ref (only for first joint)
        if convergenceCriteria(x_sim[k + 1], np.array([1, 0, 0, 1, 0, 0])):
            with counter.get_lock():
                counter.value += 1
                print(f"{counter.value}:{x0}=> SUCCESS")
                with counter_success.get_lock():
                    counter_success.value += 1
            convergence = 1
            break
        if k == conf.n_steps-1:
            with counter.get_lock():
                counter.value += 1
                print(f'{counter.value}:{x0}=>Not converged\n')
                with counter_fails.get_lock():
                    counter_fails.value += 1
                    print(f"Number of failures: {counter_fails.value}")
                convergence = None
    x_v = controller.getLastViableState()
    return k, convergence, x_sim, stats, x_v, u,x_simu,u_simu,jumps,safe_hor_hist,core_sol

counter = multiprocessing.Value('i', 0)
counter_success = multiprocessing.Value('i', 0)
counter_fails = multiprocessing.Value('i',0)


if __name__ == '__main__':
    print(f'Available cores: {os.cpu_count()}')
    available_controllers = {'naive': 'NaiveController',
                             'st': 'STController',
                             'stwa': 'STWAController',
                             'htwa': 'HTWAController',
                             'receding': 'RecedingController',
                             'receding_single':'RecedingSingleConstraint',
                             'parallel': 'ParallelWithCheck',
                             'parallel_limited':'ParallelLimited',
                             'parallel_single':'ParallelSingleConstraint',
                             'abort': 'SafeBackupController'}
    
    
    # Set in variable control the controller to test, between the keys in the above dictionary. 
    
    
    control = 'parallel_limited'
    # Define the configuration object, model, simulator
    conf = Parameters('triple_pendulum', control,rti=True)
    model = getattr(models,'TriplePendulumModel')(conf)
    simulator = SimDynamics(model)
    controller = getattr(controllers, available_controllers[control])(simulator)
    controller.setReference(model.x_ref)
    

    x0_success = []
    violations=[]
    if (control=='receding' or 'parallel' in control):
        results_name = f'/{available_controllers[control]}_minJump{controller.min_negative_jump}_errThr{controller.err_thr}_alpha{controller.params.alpha}'
        if 'limited' in control:
            results_name = results_name + f'_cores{controller.cores}_method_{controller.constraint_mode.__name__}'
    else: 
        results_name = f'/{control}_alpha{controller.params.alpha}'
    DATA_DIR = os.getcwd()+'/DATI_PARALLELIZED'+results_name
    # Check if data folder exists, if not create it
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    data_name = conf.DATA_DIR + control + '_'

    x0_vec = np.load(conf.DATA_DIR + f'x_init_{conf.alpha}.npy')#[:100]
    x_guess_vec = np.load(data_name + f'x_guess_{conf.alpha}.npy')#[:100]
    u_guess_vec = np.load(data_name + f'u_guess_{conf.alpha}.npy')#[:100]
    
    
        
    
    guess_args = list(zip(x0_vec, x_guess_vec, u_guess_vec))
    length = x0_vec.shape[0]
    
    #simulate_mpc(*guess_args[7])
    if not('single' in controller.params.cont_type):
        controller.solve(np.zeros(6))
    else:
        for i in range(controller.N):
            controller.solve(np.zeros(6),i)
    if control != 'naive':
        for i in range(x_guess_vec.shape[0]):
            if not(controller.model.checkSafeConstraints(x_guess_vec[i][-1])):
                print(controller.model.checkSafeConstraints(x_guess_vec[i][-1]))
            
    print('\n\n\n Starting simulation \n\n\n')
    
    
    n_processes = 27

    with multiprocessing.Pool(processes=n_processes) as pool:
        # Use starmap to apply the process_data function to the list of tuples
        results = pool.starmap(simulate_mpc, guess_args)


    print(f'{counter_success.value} successes over {length}')
    with open(DATA_DIR+results_name+'.pkl', 'wb') as file:
        pickle.dump(results,file)
    
    x_viable=[]
    x_suspended=[]
    for j in range(len(results)):
            if results[j][1] == 0:
                x_viable.append(results[j][4])
            if results[j][1] == None:
                x_suspended.append(results[j][6][-1])
    np.save(DATA_DIR + results_name + 'x_viable.npy', np.asarray(x_viable))
    np.save(DATA_DIR + results_name + 'x_suspended.npy', np.asarray(x_suspended))
    print(len(x_viable))

    _, _, _, t_stats, _, _,_,_,_,_,_ = zip(*results)
    print('99% quantile computation time:')
    times = np.array([t for arr in t_stats for t in arr])
    for field, t in zip(controller.time_fields, np.quantile(times, 0.99, axis=0)):
        print(f"{field:<20} -> {t}")