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

def bounds_dist(x):
    # x0_b = [model.x_min[0],1] if np.abs(x[0]-model.x_min[0]) < np.abs(x[0]-model.x_max[0]) else [model.x_max[0],-1] 
    # x1_b = [model.x_min[1],1] if np.abs(x[1]-model.x_min[1]) < np.abs(x[1]-model.x_max[1]) else [model.x_max[1],-1]
    # x2_b = [model.x_min[2],1] if np.abs(x[2]-model.x_min[2]) < np.abs(x[2]-model.x_max[2]) else [model.x_max[2],-1] 
    
    x0_b = model.x_min[0] if np.abs(x[0]-model.x_min[0]) < np.abs(x[0]-model.x_max[0]) else model.x_max[0] 
    x1_b = model.x_min[1] if np.abs(x[1]-model.x_min[1]) < np.abs(x[1]-model.x_max[1]) else model.x_max[1]
    x2_b = model.x_min[2] if np.abs(x[2]-model.x_min[2]) < np.abs(x[2]-model.x_max[2]) else model.x_max[2]
    #return min(np.abs(x[0]-x0_b),np.abs(x[1]-x1_b),np.abs(x[2]-x2_b))
    # return np.linalg.norm(np.array([np.abs(x[0]-x0_b[0])+x[3]/(10*(np.abs(x[0]-model.x_max[0])))\
    #     ,np.abs(x[1]-x1_b[1])+x[4]/(10*(np.abs(x[1]-model.x_max[1]))),\
    #     np.abs(x[2]-x2_b[0])+x[5]/(10*(np.abs(x[2]-model.x_max[2])))]))
    #return np.abs(x[0]-model.x_max[0]) - x[3]/ 10*np.abs(x[0]-model.x_max[0]) 
    return np.linalg.norm(np.array([x[0]-x0_b]))
    
def convergenceCriteria(x):
    mask = np.array([1,0,0,1,0,0])
    return np.linalg.norm(np.multiply(mask, x - model.x_ref))

if __name__ == '__main__':
    available_controllers = {'naive': 'NaiveController',
                             'st': 'STController',
                             'stwa': 'STWAController',
                             'htwa': 'HTWAController',
                             'receding': 'RecedingController',
                             'parallel': 'ParallelWithCheck',
                             'parallel_limited':'ParallelLimited',
                             'abort': 'SafeBackupController'}
    
    
    control = 'abort'
    
    # Set in variable abort the controller's states on which execute the backup control, between the keys in the above dictionary.
    
    abort = 'parallel'
    if abort == 'parallel_limited':
        # mode CIS, uni or high
        mode = 'uni'
        cores = 16
    min_jump = 0
    
    concluded = True    
    # Define the configuration object, model, simulator and controller
    conf = Parameters('triple_pendulum', control,rti=False)
    model = getattr(models,'TriplePendulumModel')(conf)
    #gc = GravityCompensation(conf, model)
    simulator = SimDynamics(model)
    
    controller = getattr(controllers, available_controllers[control])(simulator)
    #controller.setReference(model.x_ref)

    folder = os.path.join(conf.ROOT_DIR,'DATI_PARALLELIZED')
    folder_list = os.listdir(folder)
    viable = ''
    for i in folder_list:
        if abort == 'naive':
            if not(concluded) and 'naive' in i and str(controller.params.alpha) in i:
                viable = i +'/'+i+'x_suspended.npy'
                break
        elif 'Thr'+str(controller.err_thr)+'_' in i and available_controllers[abort] in i and 'alpha'+str(controller.params.alpha) in i \
            and 'Jump'+str(min_jump) in i:
                if abort == 'parallel_limited':
                    if 'cores'+str(cores) in i and mode in i:
                        if concluded:
                            viable = i +'/'+i+'x_viable.npy'
                        else:
                            viable = i +'/'+i+'x_suspended.npy'
                            
                        break
                else:
                    if concluded:
                            viable = i +'/'+i+'x_viable.npy'
                    else:
                        viable = i +'/'+i+'x_suspended.npy'
                    break
    print(i)
    


    tgrid = np.linspace(0,conf.T,int(conf.T/conf.dt)+1)
    failed =[]
    
    x_viable = np.load(os.path.join(folder,viable)) 
    n_a = np.shape(x_viable)[0]
    rep = 1
    t_rep = np.empty((n_a, rep)) * np.nan
    controller.model.setNNmodel()
    closeness_success,closeness_failed = [],[]
    convergence_index = []
    for i in range(n_a):
        print(f'{i} x_viable={x_viable[i]} \n')
        print(f'{i} convergence_criteria={convergenceCriteria(x_viable[i])} \n')
        convergence_index.append(convergenceCriteria(x_viable[i]))
        
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
                print(f'closeness :{bounds_dist(x_viable[i])}')
                if (np.abs(controller.x_temp[-1][3:]) < 1e-2).all():
                    
                    integrated_sol=x_viable[i].reshape(1,-1) 
                    for k in range(controller.u_temp.shape[0]):
                        integrated_sol= np.vstack([integrated_sol,simulator.simulate(integrated_sol[k,:],controller.u_temp[k])])

                    if controller.model.checkStateConstraints(integrated_sol) and (np.abs(integrated_sol[-1,3:])<1e-3).all():
                        t_rep[i, j] = controller.ocp_solver.get_stats('time_tot')
                        print('SUCCESS')
                        print(f'closeness :{bounds_dist(x_viable[i])}')
                        closeness_success.append(bounds_dist(x_viable[i]))

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
                    
                    
            else: 
                failed.append(i)
                print(f'closeness :{bounds_dist(x_viable[i])}')
                closeness_failed.append(bounds_dist(x_viable[i]))

                

    # Compute the minimum time for each initial condition
    t_min = np.min(t_rep, axis=1)
    # Remove the nan values (i.e. the initial conditions for which the controller failed)
    t_min = t_min[~np.isnan(t_min)]
    print('Controller: %s\nAbort: %d over %d\nQuantile (99) time: %.3f'
            % (abort, len(t_min), n_a, np.quantile(t_min, 0.99)))
    print(failed)
    
    
    closeness_failed=np.array(closeness_failed)
    closeness_success=np.array(closeness_success)
    
    if False:
        plt.figure()
        plt.hist(closeness_success[closeness_success<100],density=False,bins=20,color='blue', edgecolor='black', alpha=.5)
        plt.hist(closeness_failed[closeness_failed<100] ,density=False,bins=20,color='yellow', edgecolor='black', alpha=.5)
        
    
        # Adding labels and title
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.title('Successes vs fails closeness to bounds viable states')
        
        plt.show()
    print(f'Mean of closeness : {np.mean(np.hstack((closeness_failed,closeness_success)))}')
    
    if not(concluded):
        print(f'max distance form convergence={max(convergence_index)}')