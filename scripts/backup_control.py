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
    abort = 'parallel_limited'
    if abort == 'parallel_limited':
        # mode CIS, uni or high
        mode = 'high'
        cores = 16
    
    # Define the configuration object, model, simulator and controller
    conf = Parameters('triple_pendulum', control,rti=False)
    model = getattr(models,'TriplePendulumModel')(conf)
    #gc = GravityCompensation(conf, model)
    simulator = SimDynamics(model)
    
    controller = getattr(controllers, available_controllers[control])(simulator)
    controller.setReference(model.x_ref)

    folder = os.path.join(conf.ROOT_DIR,'DATI_PARALLELIZED')
    folder_list = os.listdir(folder)
    viable = ''
    for i in folder_list:
        if available_controllers[abort] in i and str(controller.params.alpha) in i \
            and str(conf.min_negative_jump) in i:
                if abort == 'parallel_limited':
                    if 'cores'+str(cores) in i and mode in i:
                        viable = i +'/'+i+'.npy'
                        break
                else:
                    viable = i +'/'+i+'.npy'
                    break
    print(i)
    


    tgrid = np.linspace(0,conf.T,int(conf.T/conf.dt)+1)
    
    x_viable = np.load(os.path.join(folder,viable)) 
    n_a = np.shape(x_viable)[0]
    rep = 1
    t_rep = np.empty((n_a, rep)) * np.nan
    controller.model.setNNmodel()
    for i in range(n_a):
        print(f'{i} x_viable={x_viable[i]} \n\n\n\n\n')
        #controller.reinit_solver()
        #controller.ocp_solver.reset()
        # Repeat each test rep times
        for j in range(rep):
            #u0 = gc.solve(x_viable[i])
            controller.setGuess(np.full((controller.N + 1, model.nx), x_viable[i]),
                                np.zeros((controller.N,model.nu)))
            status = controller.solve(x_viable[i])
            if controller.model.nn_func(x_viable[i], controller.model.params.alpha)<0:
                print('SOMETHING WENT WRONG')
                break
            print(f'is x viable:{controller.model.nn_func(x_viable[i], controller.model.params.alpha)}')
            print(f"status {status} nlp iter:{controller.ocp_solver.get_stats('nlp_iter')} qp iter:{controller.ocp_solver.get_stats('nlp_iter')} qp_stat:\
                {controller.ocp_solver.get_stats('qp_stat')}")
            if status == 0 or status==2:
                print(status)
                if (np.abs(controller.x_temp[-1][3:]) < 1e-2).all():
                    
                    integrated_sol=x_viable[i].reshape(1,-1) 
                    for k in range(controller.u_temp.shape[0]):
                        integrated_sol= np.vstack([integrated_sol,simulator.simulate(integrated_sol[k,:],controller.u_temp[k])])

                    if controller.model.checkStateConstraints(integrated_sol) and (np.abs(integrated_sol[-1,3:])<1e-3).all():
                        t_rep[i, j] = controller.ocp_solver.get_stats('time_tot')
                        print('SUCCESS')
                    if False:
                        plt.figure(f'Position,problem {i}, status {status}')
                        plt.title(f'Position,problem {i}, status {status}')
                        plt.clf()
                        plt.plot(tgrid, controller.x_temp[:,0], '--')
                        plt.plot(tgrid, controller.x_temp[:,1], '-')
                        plt.plot(tgrid, controller.x_temp[:,2], '-')
                        plt.plot(tgrid, integrated_sol[:,0], '--')
                        plt.plot(tgrid, integrated_sol[:,1], '-')
                        plt.plot(tgrid, integrated_sol[:,2], '-')
                        plt.xlabel('t')
                        plt.legend(['x1','x2','x3','x1_sim','x2_sim','x3_sim'])
                        plt.grid()
                        plt.axhline(y=controller.model.x_min[0], color='r', linestyle='-')
                        plt.axhline(y=controller.model.x_max[0], color='r', linestyle='-')
                        plt.show()

                        plt.figure(f'Velocity, problem {i}')
                        plt.title(f'Velocity, problem {i}')
                        plt.clf()
                        plt.plot(tgrid, controller.x_temp[:,3], '--')
                        plt.plot(tgrid, controller.x_temp[:,4], '-')
                        plt.plot(tgrid, controller.x_temp[:,5], '-')
                        plt.plot(tgrid, integrated_sol[:,3], '--')
                        plt.plot(tgrid, integrated_sol[:,4], '-')
                        plt.plot(tgrid, integrated_sol[:,5], '-')
                        plt.axhline(y=controller.model.x_min[3], color='r', linestyle='-')
                        plt.axhline(y=controller.model.x_max[3], color='r', linestyle='-')
                        plt.xlabel('t')
                        plt.legend(['x1','x2','x3','x1_sim','x2_sim','x3_sim'])
                        plt.grid()
                        plt.show()
                    print(integrated_sol[-1][3:])

                

    # Compute the minimum time for each initial condition
    t_min = np.min(t_rep, axis=1)
    # Remove the nan values (i.e. the initial conditions for which the controller failed)
    t_min = t_min[~np.isnan(t_min)]
    print('Controller: %s\nAbort: %d over %d\nQuantile (99) time: %.3f'
            % (abort, len(t_min), n_a, 0))