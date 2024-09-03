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



# def init_guess(q0):
#     x0 = np.zeros((model.nx,))
#     x0[:model.nq] = q0
#     #u0 = gc.solve(x0)
#     flag = controller.initialize(x0)
#     return controller.getGuess(), flag

def init_search():
    controller = getattr(controllers, 'RecedingController')(simulator)
    controller.setReference(model.x_ref)
    # Soft constraints on all the trajectory
    for i in range(1, controller.N):
        controller.ocp_solver.cost_set(i, "zl", conf.ws_r * np.ones((1,)))
    controller.ocp_solver.cost_set(controller.N, "zl", conf.ws_t * np.ones((1,)))

    while len(x_init_vec)<length:
        while j.value < bins:
            np.random.seed()
            x_g0 = (model.x_max-model.x_min)[0]*np.random.random_sample((3,)) + model.x_min[0]*np.ones((3,))
            x_g0 = np.hstack((x_g0,[0,0,0]))
            if x0_0_intervals[j.value]<=x_g0[0]<x0_0_intervals[j.value+1]:
                status = controller.initialize(x_g0)       
                (x_g, u_g) = controller.getGuess()
                if status:
                    with lock:
                        if x0_0_intervals[j.value]<=x_g0[0]<x0_0_intervals[j.value+1]:
                            x_init_vec.append(x_g[0, :model.nq])
                            x_guess_vec.append(x_g)
                            u_guess_vec.append(u_g)
                            print(len(x_init_vec))
                            found.value += 1
                            if found.value == length/bins:
                                found.value =0
                                j.value+=1
                                print('ADVANCE')
               

if __name__ == '__main__':
    print(f'Available cores: {os.cpu_count()}')
    length = 500
    bins = 10
     
    lock =  multiprocessing.Lock() 
    manager = multiprocessing.Manager()
    final_list = manager.list()
    j = manager.Value('i',0)
    found = manager.Value('i',0)
    processes = 10
    jobs = []

    # Define the configuration object, model, simulator and controller
    conf = Parameters('triple_pendulum', 'receding',rti=False)
    model = getattr(models,'TriplePendulumModel')(conf)
    #gc = GravityCompensation(conf, model)
    simulator = SimDynamics(model)
    
    print(conf.alpha)

    sampler = qmc.Halton(model.nq, scramble=False)
    l_bounds = model.x_min[:model.nq] 
    u_bounds = model.x_max[:model.nq] 

    # Check if data folder exists, if not create it
    if not os.path.exists(conf.DATA_DIR):
        os.makedirs(conf.DATA_DIR)
    #data_name = conf.DATA_DIR + args['controller'] + '_'
    x0_0_intervals = np.linspace(model.x_min[0],model.x_max[0],bins+1)

    x_init_vec, x_guess_vec, u_guess_vec = manager.list(), manager.list(), manager.list()
    for i in range(processes):
        p = multiprocessing.Process(target=init_search,args=())
        p.start()
        jobs.append(p)

    while True:
        if len(x_guess_vec)>=length:
            for i in jobs:
                i.terminate()
            break

    #print(final_list)
    print(len(x_init_vec))
    np.save(conf.DATA_DIR + f'Parallelized_x_init_{conf.alpha}.npy', np.asarray(x_init_vec))
    np.save(conf.DATA_DIR + f'Parallelized_x_guess_vec_{conf.alpha}.npy', np.asarray(x_guess_vec))
    np.save(conf.DATA_DIR + f'Parallelized_u_guess_vec_{conf.alpha}.npy', np.asarray(u_guess_vec))