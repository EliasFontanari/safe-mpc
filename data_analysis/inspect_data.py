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
import matplotlib.pyplot as plt
import matplotlib as mpl
import safe_mpc.plut as plut



def load_data(control,alpha,min_negative_jump,err_thr,mode=None,cores=None):
    control = available_controllers[control]
    folder = os.path.join(os.getcwd(),'DATI_PARALLELIZED')
    files = os.listdir(folder)
    for i in files:
        if 'Thr'+str(err_thr)+'_' in i and control in i and 'alpha'+str(alpha) in i \
            and 'Jump'+str(min_negative_jump) in i:
                if control  == 'ParallelLimited':
                    if 'cores'+str(cores) in i and mode in i:
                        path = os.path.join(folder,i +'/'+i+'.pkl')
                        with open(path,'rb') as f:
                            data_loaded = pickle.load(f)
                        break
                else:
                    path = os.path.join(folder,i +'/'+i+'.pkl')
                    with open(path,'rb') as f:
                        data_loaded = pickle.load(f)
                    break
    path = os.path.join(folder,i +'/'+i+'.pkl')
    print(f'loaded dataset :{path}')
    return data_loaded

def convergenceValue(x):
    return np.linalg.norm(np.multiply(np.array([1, 0, 0, 1, 0, 0]), x - model.x_ref),axis=1)

def bounds_dist(x):
    x0_b = model.x_min[0] if np.abs(x[0]-model.x_min[0]) < np.abs(x[0]-model.x_max[0]) else model.x_max[0] 
    x1_b = model.x_min[1] if np.abs(x[1]-model.x_min[1]) < np.abs(x[1]-model.x_max[1]) else model.x_max[1]
    x2_b = model.x_min[2] if np.abs(x[2]-model.x_min[2]) < np.abs(x[2]-model.x_max[2]) else model.x_max[2] 
    
    #return np.linalg.norm(np.array([x[0]-x0_b,(x[1]-x1_b),(x[2]-x2_b)]))
    return np.abs(x[0]-x0_b)


if __name__ == '__main__':
    
    
    available_controllers = {'naive': 'NaiveController',
                             'st': 'STController',
                             'stwa': 'STWAController',
                             'htwa': 'HTWAController',
                             'receding': 'RecedingController',
                             'parallel': 'ParallelWithCheck',
                             'parallel_limited':'ParallelLimited',
                             'abort': 'SafeBackupController'}
    
    conf = Parameters('triple_pendulum', 'receding',rti=True)
    model = getattr(models,'TriplePendulumModel')(conf)
    simulator = SimDynamics(model)
    model.setNNmodel()
    # Data content :  k, convergence, x_sim, stats, x_v, u,x_simu,u_simu,jumps,safe_hor_hist,core_sol
    dataset1 = 'parallel'
    dataset2 = 'receding'
    
    # To load a dataset obtained by mpc_parallelized, arguments to be passed are
    # Controller, as a key of the above dictionary, safety factor alpha, min_jump, threshold guess correction. In addition,
    # if you want to load a dataset of a parallel limited, insert the type ( 'uni', 'high' or 'CIS' for closest) and number of 
    # computatioanl units. 
    
    data_par = load_data(dataset1,2,0,1e-3,'uni',16) 
    data_rec = load_data(dataset2,2,0,1e-3,'uni',8)

    x_v_par,x_v_rec=[],[]
    x_v_par_val,x_v_rec_val = [],[] 
    x_v_par_bounds_dist, x_v_rec_bounds_dist = [],[]
    for i in range(len(data_par)):
        if data_par[i][1]==0: 
            x_v_par.append(data_par[i][4])
            x_v_par_val.append(model.nn_func(x_v_par[-1],conf.alpha))
            x_v_par_bounds_dist.append(bounds_dist(data_par[i][4]))
        if data_rec[i][1]==0: 
            x_v_rec.append(data_rec[i][4])
            x_v_rec_val.append(model.nn_func(x_v_rec[-1],conf.alpha))
            x_v_rec_bounds_dist.append(bounds_dist(data_rec[i][4]))
            
    x_v_par_val = np.array(x_v_par_val).squeeze()
    x_v_rec_val = np.array(x_v_rec_val).squeeze()
    
    par_mean, par_std = np.mean(x_v_par_val),np.std(x_v_par_val)
    rec_mean, rec_std = np.mean(x_v_rec_val),np.std(x_v_rec_val) 
     
    print(f'Viable parallel states:{len(x_v_par)}, mean:{par_mean}, std:{par_std}')
    print(f'Viable receding states:{len(x_v_rec)}, mean:{rec_mean}, std:{rec_std}')
    
    density = False  
    plt.figure()
    plt.hist(x_v_par_val,density=density,bins=10 ,color='blue', edgecolor='black', alpha=.5)
    plt.hist(x_v_rec_val,density=density,bins=10 ,color='yellow', edgecolor='black', alpha=.5)
    
 
    # Adding labels and title
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title(f'{dataset1} vs  {dataset2} viable distribution')
    plt.legend([dataset1, dataset2])


    # Display the plot
    plt.show()
    
    plt.figure()
    plt.hist(x_v_par_bounds_dist,density=False,bins=30,color='blue', edgecolor='black', alpha=.5)
    plt.hist(x_v_rec_bounds_dist ,density=False,bins=30,color='yellow', edgecolor='black', alpha=.5)
    plt.legend([dataset1, dataset2])
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    
    
 
    # Adding labels and title
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    #plt.title(f'{dataset1} vs  {dataset2} closeness to bounds viable states')
    plt.savefig('receding_parallell_closenesess',dpi=300)
    # Display the plot
    plt.show()
    x_v_par_bounds_dist = np.array(x_v_par_bounds_dist)
    x_v_rec_bounds_dist = np.array(x_v_rec_bounds_dist)
    print(f'mean closeness index for parallell: {np.mean(x_v_par_bounds_dist)}')
    print(f'mean closeness index for receding: {np.mean(x_v_rec_bounds_dist)}')
    print(f'percentage of problems under 0.23 for parallell: {x_v_par_bounds_dist[x_v_par_bounds_dist<0.23].size/len(x_v_par_bounds_dist)}')
    print(f'percentage of problems under 0.23 for receding: {x_v_rec_bounds_dist[x_v_rec_bounds_dist<0.23].size/len(x_v_rec_bounds_dist)}')
    
    
    # convergence values for the trajectory and the viable states
    data_par = load_data(dataset1,2,0,1e-3,'uni',16) 
    data_high4 = load_data(dataset2,2,0,1e-3,'high',16)
    x_v_par,x_v_high=[],[]
    x_val_par_min_traj,x_val_high_min_traj = [],[] 
    x_val_par_end_traj,x_val_high_end_traj = [],[] 
    for i in range(len(data_par)):
        if data_par[i][1]==0: 
            x_sim = data_par[i][2]
            x_val_par_min_traj.append(np.min(convergenceValue(x_sim[~np.all(np.isnan(x_sim),axis=1)])))
            x_val_par_end_traj.append(convergenceValue([data_par[i][4]])[0])
        if data_high4[i][1]==0: 
            x_sim = data_high4[i][2] 
            x_val_high_min_traj.append(np.min(convergenceValue(x_sim[~np.all(np.isnan(x_sim),axis=1)])))
            x_val_high_end_traj.append(convergenceValue([data_high4[i][4]])[0])
    
    
    plt.figure()
    plt.hist(x_val_par_min_traj,density=False,color='blue', edgecolor='black', alpha=.5)
    plt.hist(x_val_high_min_traj,density=False,color='yellow', edgecolor='black', alpha=.5)
    
 
    # Adding labels and title
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title(f'{dataset1} vs  {dataset2} minimum convergence value')
    plt.legend([dataset1, dataset2])
    
    # Display the plot
    plt.show()
    
    
    plt.figure()
    plt.hist(x_val_par_end_traj,density=False,color='blue', edgecolor='black', alpha=.5)
    plt.hist(x_val_high_end_traj,density=False,color='yellow', edgecolor='black', alpha=.5)
    
 
    # Adding labels and title
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title(f'{dataset1} vs  {dataset2} end convergence value')
    plt.legend([dataset1, dataset2])
    
    # Display the plot
    plt.show()
    
    



