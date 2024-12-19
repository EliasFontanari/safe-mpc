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
import safe_mpc.plut as plut


def state_traj_cost(x,u):
    x=np.array(x)
    u=np.array(u)
    Q = 1e-4 * np.eye(6)
    Q[0, 0] = 5e2
    R = 1 * np.eye(3)
    x_ref = np.array([conf.q_max - 0.05, np.pi, np.pi, 0, 0, 0])
    n = x.shape[0]
    cost = 0
    for i in range(n-1):
        cost += (x[i] - x_ref).T @ Q @ (x[i] - x_ref) + u[i].T @ R @ u[i]
    cost += (x[-1] - x_ref).T @ Q @ (x[-1] - x_ref)
    return cost

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


if __name__ == '__main__':
    available_controllers = {'naive': 'NaiveController',
                             'st': 'STController',
                             'stwa': 'STWAController',
                             'htwa': 'HTWAController',
                             'receding': 'RecedingController',
                             'parallel': 'ParallelWithCheck',
                             'parallel_limited':'ParallelLimited',
                             'abort': 'SafeBackupController'}
    
    conf = Parameters('triple_pendulum', 'naive')
    ALPHA = [10,15,20]
    length_1 = [[] for _ in range(len(ALPHA))]
    length_2 = [[] for _ in range(len(ALPHA))]
    costs_1 = [[] for _ in range(len(ALPHA))]
    costs_2 = [[] for _ in range(len(ALPHA))]
    
    k=0
    for a in ALPHA:
        
        # To load a dataset obtained by mpc_parallelized, arguments to be passed are
        # Controller, as a key of the above dictionary, safety factor alpha, min_jump, threshold guess correction. In addition,
        # if you want to load a dataset of a parallel limited, insert the type ( 'uni', 'high' or 'CIS' for closest) and number of 
        # computational units. 
        data_1 = load_data('parallel',a,0,1e-3) 
        data_2 = load_data('receding',a,0,1e-3,'uni',8)
        

        hor_par,hor_high=[],[]
        jumps_par, jumps_high = [],[]
        x0s=[]
        for i in range(len(data_1)):
            for j in range(len(data_1)):
                if (data_1[i][2][0]==data_2[j][2][0]).all(): 
                    if data_1[i][1]==1 and data_2[j][1]==1: 
                        # print(f'x0:{data_1[i][2][0]}, par {data_1[i][1]}, high {data_2[j][1]}')
                        # hor_par.append(data_1[i][9])
                        # hor_high.append(data_2[j][9])  
                        # jumps_par.append(data_1[i][8])
                        # jumps_high.append(data_2[j][8])
                        x0s.append(i)
                        # print(i)
                        # print(j) 
        print(len(x0s))
        # comparison beetween completed tasks
        for i in range(len(x0s)):
            length_1[k].append(data_1[x0s[i]][0])
            length_2[k].append(data_2[x0s[i]][0])
            costs_1[k].append(state_traj_cost(data_1[x0s[i]][6],data_1[x0s[i]][7]))
            costs_2[k].append(state_traj_cost(data_2[x0s[i]][6],data_2[x0s[i]][7]))
            
        
        plt.figure()
        bin_edges = np.linspace(min(min(length_1[k]),min(length_2[k])), max(max(length_1[k]),max(length_2[k])), num=10)
        
        plt.hist(length_1[k],bins=bin_edges,color='blue', edgecolor='black', alpha=.5)
        plt.hist(length_2[k],bins=bin_edges,color='yellow', edgecolor='black', alpha=.5)
        

        # Adding labels and title
        plt.xlabel('Time steps to convergence')
        plt.ylabel('Frequency')
        plt.legend(['Parallel', 'Receding'])
        plt.grid(True)
        
        plt.savefig(f'histogram_alpha{a}',dpi=300)


        # Display the plot
        plt.show()
        
        plt.figure()
        bin_edges = np.linspace(min(min(costs_1[k]),min(costs_2[k])), max(max(costs_1[k]),max(costs_2[k])), num=10)
        
        plt.hist(costs_1[k],bins=bin_edges,color='blue', edgecolor='black', alpha=.5)
        plt.hist(costs_2[k],bins=bin_edges,color='yellow', edgecolor='black', alpha=.5)
        

        # Adding labels and title
        plt.xlabel('Cost')
        plt.ylabel('Frequency')
        plt.legend(['Parallel', 'Receding'])
        plt.grid(True)
        
        plt.savefig(f'histogram_costs{a}',dpi=300)


        # Display the plot
        plt.show()
        
        print(f'total cost parallel {np.sum(costs_1[k])}')
        print(f'total cost receding {np.sum(costs_2[k])}')
        
        k+=1
    
    length_1 = np.concatenate(length_1)
    length_2= np.concatenate(length_2)
    costs_1 = np.concatenate(costs_1)
    costs_2= np.concatenate(costs_2)
    
    mean_1 = np.sum(length_1)/len(length_1)
    mean_2 = np.sum(length_2)/len(length_2)
    plt.figure()
    bin_edges = np.linspace(min(min(length_1),min(length_2)), max(max(length_1),max(length_2)), num=10)
   
    plt.hist(length_1,bins=bin_edges,color='blue', edgecolor='black', alpha=.5,label='Parallel')
    plt.hist(length_2,bins=bin_edges,color='yellow', edgecolor='black', alpha=.5,label='Receding')
    
    plt.axvline(x=mean_1,ymin=0,ymax=250, color='blue', linestyle='--', label='Parallel mean')
    plt.axvline(x=mean_2,ymin=0,ymax=255, color='orange', linestyle='--', label='Receding mean')
    
    plt.xlim(0,600)
    

    # Adding labels and title
    plt.xlabel('Time steps to convergence')
    plt.ylabel('Frequency')
    plt.legend(loc='lower right',bbox_to_anchor=(1.025, 0.7))
    
    plt.grid(True)
    
    plt.savefig(f'histogram_all_alpha',dpi=300)


    # Display the plot
    plt.show()
    
    
    plt.figure()
    bin_edges = np.linspace(min(min(costs_1),min(costs_2)), max(max(costs_1),max(costs_2)), num=10)
    
    plt.hist(costs_1,bins=bin_edges,color='blue', edgecolor='black', alpha=.5)
    plt.hist(costs_2,bins=bin_edges,color='yellow', edgecolor='black', alpha=.5)
    
    
    # Adding labels and title
    plt.xlabel('Cost')
    plt.ylabel('Frequency')
    plt.legend(['Parallel', 'Receding'])
    plt.grid(True)
    
    plt.savefig(f'histogram_costs_all_alpha',dpi=300)


    # Display the plot
    plt.show()
    
    print(f'total cost parallel {np.sum(costs_1)}')
    print(f'total cost receding {np.sum(costs_2)}')
    
    print(f'total step parallel {np.sum(length_1)}')
    print(f'total step receding {np.sum(length_2)}')
    print(mean_1)
    print(mean_2)