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
import matplotlib.pyplot as plt


if __name__ == '__main__':
    available_controllers = {'naive': 'NaiveController',
                             'st': 'STController',
                             'stwa': 'STWAController',
                             'htwa': 'HTWAController',
                             'receding': 'RecedingController',
                             'parallel': 'ParallelController',
                             'parallel_limited':'ParallelLimited',
                             'parallel2': 'ParallelWithCheck',
                             'parallel_receding':'RecedingParallel',
                             'abort': 'SafeBackupController'}
    
    
    control = 'abort'
    abort = 'parallel2'
    err_thr = 1e-3
    if abort == 'parallel_limited':
        # mode CIS, uni or high
        mode = 'uni'
        cores = 4
    
    # Define the configuration object, model, simulator and controller
    conf = Parameters('triple_pendulum', control,rti=False)
    model = getattr(models,'TriplePendulumModel')(conf)
    #gc = GravityCompensation(conf, model)
    simulator = SimDynamics(model)
    
    folder = os.path.join(conf.ROOT_DIR,'DATI_PARALLELIZED')
    folder_list = os.listdir(folder)
    viable = ''
    for i in folder_list:
        if 'Thr'+str(err_thr) in i and available_controllers[abort] in i and str(conf.alpha) in i \
            and str(conf.min_negative_jump) in i:
                if abort == 'parallel_limited':
                    if 'cores'+str(cores) in i and mode in i:
                        viable = i +'/'+i+'x_viable.npy'
                        break
                else:
                    viable = i +'/'+i+'x_viable.npy'
                    break
    print(i)
    
    x_viable = np.load(os.path.join(folder,viable)) 
    n_a = np.shape(x_viable)[0]
    
    mark_size = 1
    
    plt.figure('Position viable, parallel')
    plt.clf()
    for j in range(n_a):
        if not(model.checkStateConstraints(x_viable[j])):
            print(j)
    plt.plot(np.arange(n_a), x_viable[:,0],  marker='o', linestyle='', color='b', markersize=mark_size)
    plt.plot(np.arange(n_a), x_viable[:,1],  marker='o', linestyle='', color='r', markersize=mark_size)
    plt.plot(np.arange(n_a), x_viable[:,2],  marker='o', linestyle='', color='g', markersize=mark_size)
    plt.xlabel('t')
    plt.legend(['x1','x2','x3'])
    plt.grid()
    plt.axhline(y=model.x_min[0], color='r', linestyle='-')
    plt.axhline(y=model.x_max[0], color='r', linestyle='-')
    
    
    abort = 'receding'
    err_thr = 1e-3
    if abort == 'parallel_limited':
        # mode CIS, uni or high
        mode = 'uni'
        cores = 4
        
    folder = os.path.join(conf.ROOT_DIR,'DATI_PARALLELIZED')
    folder_list = os.listdir(folder)
    viable = ''
    for i in folder_list:
        if 'Thr'+str(err_thr) in i and available_controllers[abort] in i and str(conf.alpha) in i \
            and str(conf.min_negative_jump) in i:
                if abort == 'parallel_limited':
                    if 'cores'+str(cores) in i and mode in i:
                        viable = i +'/'+i+'x_viable.npy'
                        break
                else:
                    viable = i +'/'+i+'x_viable.npy'
                    break
    print(i)
    
    x_viable = np.load(os.path.join(folder,viable)) 
    n_a = np.shape(x_viable)[0]
    
    
    plt.figure('Position viable, receding')
    plt.clf()
    for j in range(n_a):
        if not(model.checkStateConstraints(x_viable[j])):
            print(j)
    plt.plot(np.arange(n_a), x_viable[:,0],  marker='o', linestyle='', color='b', markersize=mark_size)
    plt.plot(np.arange(n_a), x_viable[:,1],  marker='o', linestyle='', color='r', markersize=mark_size)
    plt.plot(np.arange(n_a), x_viable[:,2],  marker='o', linestyle='', color='g', markersize=mark_size)
    plt.xlabel('t')
    plt.legend(['x1','x2','x3'])
    plt.grid()
    plt.axhline(y=model.x_min[0], color='r', linestyle='-')
    plt.axhline(y=model.x_max[0], color='r', linestyle='-')
    plt.show()