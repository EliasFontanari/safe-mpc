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

def load_data(control,alpha,min_negative_jump,err_thr,mode=None,cores=None):
    folder = os.path.join(os.getcwd(),'DATI_PARALLELIZED')
    files = os.listdir(folder)
    for i in files:
        if 'Thr'+str(err_thr) in i and control in i and str(alpha) in i \
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
    alpha = 2
    data_1 = load_data('ParallelWithCheck',alpha,0,1e-3) 
    data_2 = load_data('ParallelLimited',alpha,0,1e-3,'uni',8)
    

    hor_par,hor_high=[],[]
    jumps_par, jumps_high = [],[]
    x0s=[]
    for i in range(len(data_1)):
        for j in range(len(data_1)):
            if (data_1[i][2][0]==data_2[j][2][0]).all(): 
                if data_1[i][1]==0 and data_2[j][1]==1: 
                    print(f'x0:{data_1[i][2][0]}, par {data_1[i][1]}, high {data_2[j][1]}')
                    hor_par.append(data_1[i][9])
                    hor_high.append(data_2[j][9])  
                    jumps_par.append(data_1[i][8])
                    jumps_high.append(data_2[j][8])
                    x0s.append(i)
                    print(i)
                    print(j) 
    print(len(hor_par))
    n_plot = len(hor_par)    
    
    if True:
        for i in range(0,n_plot):
            print(x0s[i])
            plt.figure(f'{i}')
            
            plt.clf()
            plt.plot(hor_par[i],  linestyle='-', color='b')
            
            #plt.figure('High 16')
            plt.plot(hor_high[i],  linestyle='-', color='orange')
            plt.legend(['Parallel','Receding'])
            plt.xlabel('Time [s]')
            plt.ylabel('r safe node')
            plt.ylim(0,37)
            plt.grid(True)
            plt.savefig('r_parallel_receding',dpi=300)
            
            #plt.show()
            
            plt.figure('u0')
            plt.clf()
            plt.plot(np.array(data_1[x0s[i]][5])[:,0] -np.array(data_2[x0s[i]][5])[:,0],  linestyle='-', color='b')
            #plt.plot(np.array(data_2[x0s[i]][7])[:,0],  linestyle='-', color='r')
            plt.figure('u1')
            plt.clf()
            plt.plot(np.array(data_1[x0s[i]][5])[:,1] -np.array(data_2[x0s[i]][5])[:,1],  linestyle='-', color='b')
            #plt.plot(np.array(data_2[x0s[i]][7])[:,1],  linestyle='-', color='r')
            plt.figure('u2')
            plt.clf()
            plt.plot(np.array(data_1[x0s[i]][5])[:,2] -np.array(data_2[x0s[i]][5])[:,2],  linestyle='-', color='b')
            #plt.plot(np.array(data_2[x0s[i]][7])[:,2],  linestyle='-', color='r')
            # plt.plot(np.array(data_1[x0s[i]][6])[:,3],  linestyle='-', color='b')
            # plt.plot(np.array(data_2[x0s[i]][6])[:,3],  linestyle='-', color='r')
            
            #plt.legend(['x1','x2','x3'])
            plt.grid(True)
            plt.show()      
    if False:
        for i in range(n_plot):
            plt.figure('Parallel')
            plt.clf()
            plt.plot(jumps_par[i],  linestyle='-', color='b')
            #plt.legend(['x1','x2','x3'])
            plt.grid()
            
            plt.figure('High 16')
            plt.clf()
            plt.plot(jumps_high[i],  linestyle='-', color='b')
            #plt.legend(['x1','x2','x3'])
            plt.grid()
            plt.show()
    if False:
        plt.figure()
        plt.hist(np.hstack(jumps_par).flatten(),density=True,color='blue', edgecolor='black', alpha=.5)
        plt.hist(np.hstack(jumps_high).flatten(),density=True,color='yellow', edgecolor='black', alpha=.5)
        
    
        # Adding labels and title
        plt.xlabel('Values of r')
        plt.ylabel('Relative frequency')
        plt.legend(['par', 'high'])
        plt.grid(True)


        # Display the plot
        plt.show()
    
    hor_tot_par = []
    hor_tot_rec = []
    
    for i in range(len(data_1)):
        for j in range(len(data_1[i][-2])):
            hor_tot_par.append(data_1[i][-2][j])
        for j in range(len(data_2[i][-2])):
            hor_tot_rec.append(data_2[i][-2][j])
    
    if True:
        plt.figure()
        plt.hist(hor_tot_par,density=True,bins=20,color='blue', edgecolor='black', alpha=.5)
        plt.hist(hor_tot_rec,density=True,bins=20,color='yellow', edgecolor='black', alpha=.5)
        
    
        # Adding labels and title
        plt.xlabel('Values of r')
        plt.ylabel('Relative frequency')
        plt.legend(['Parallel', 'Receding'])
        plt.grid(True)
        
        plt.savefig(f'histogram_r{alpha}',dpi=300)


        # Display the plot
        plt.show()