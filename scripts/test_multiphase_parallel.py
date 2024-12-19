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


if __name__ == '__main__':
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
    
    # Define the configuration object, model, simulator and controller
    #conf = Parameters('triple_pendulum', 'receding',rti=True)
    controller1='parallel_single'
    controller2 = 'parallel'
    
    conf = Parameters('triple_pendulum',controller1,rti=False) 
    #conf.cont_type = 'receding2'
    model = getattr(models,'TriplePendulumModel')(conf)
    #gc = GravityCompensation(conf, model)
    simulator = SimDynamics(model)
    #simulator_m = SimDynamics(model)
    controller_m = getattr(controllers, available_controllers[controller1])(simulator)
    
    controller_r = getattr(controllers, available_controllers[controller2])(simulator)
    controller_r.setReference(model.x_ref)
    controller_m.setReference(model.x_ref)
    x0_vec = np.load(conf.DATA_DIR + f'x_init_{conf.alpha}.npy')
    x0=np.zeros(model.nx)
    while(True):
        #x0[:model.nq] =  (model.x_max-model.x_min)[0]*np.random.random_sample((3,)) + model.x_min[0]*np.ones((3,))
        #x0[model.nq:] =  (model.x_max-model.x_min)[-1]*np.random.random_sample((3,)) + model.x_min[-1]*np.ones((3,))
        x0[:controller_m.model.nq] = x0_vec[np.random.randint(0,len(x0_vec))]
        n_solver = np.random.randint(1,controller_m.N+1)
        status1=controller_m.solve(x0,n_solver)
        controller_r.constrain_n(n_solver)
        status2=controller_r.solve(x0,[n_solver,controller_r.N])
        print(f'success1 = {status1} & success2 = {status2}')
        print(f'Safe?: multiphase {controller_m.model.nn_func(controller_m.x_temp[n_solver], controller_m.params.alpha)} \
               and single phase {controller_r.model.nn_func(controller_r.x_temp[n_solver], controller_r.params.alpha)}  ')
        if True and (status1 == 2 or status1 == 0) or  (status2 == 2 or status2 == 0):
            print(f'x0={x0}')
            plt.figure(f'Multiphase solution,  n_solver = {n_solver}')
            plt.clf()
            plt.plot(np.abs(controller_r.u_temp[:,0] - controller_m.u_temp[:,0]), '-')
            plt.plot(np.abs(controller_r.u_temp[:,1] - controller_m.u_temp[:,1]), '-')
            plt.plot(np.abs(controller_r.u_temp[:,2] - controller_m.u_temp[:,2]), '-')
            plt.xlabel('t')
            plt.legend(['u1','u2','u3'])
            plt.grid()
            
            # plt.figure(f'Single phase solution,n_solver = {n_solver}')
            # plt.clf()
            # plt.plot(controller_r.u_temp[:,0], '-')
            # plt.plot(controller_r.u_temp[:,1], '-')
            # plt.plot(controller_r.u_temp[:,2], '-')
            
            # plt.xlabel('t')
            # plt.legend(['u1','u2','u3'])
            # plt.grid()

            plt.show()

    