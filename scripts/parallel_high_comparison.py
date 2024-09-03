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

def load_data(control,alpha,min_negative_jump,err_thr,mode=None,cores=None):
    folder = os.path.join(os.getcwd(),'DATI_PARALLELIZED')
    files = os.listdir(folder)
    for i in files:
        if 'Thr'+str(err_thr) in i and control in i and str(alpha) in i \
            and str(min_negative_jump) in i:
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
    print(path)
    return data_loaded


if __name__ == '__main__':
    data_par = load_data('ParallelWithCheck',2,-1,1e-3) 
    data_high = load_data('ParallelLimited',2,-1,1e-3,'high',16)

    hor_par,hor_high=[],[]
    for i in range(len(data_par)):
        for j in range(len(data_par)):
            if (data_par[i][2][0]==data_high[j][2][0]).all(): 
                if data_par[i][1]==1 and data_high[j][1]==0: 
                    print(f'x0:{data_par[i][2][0]}, par {data_par[i][1]}, high {data_high[j][1]}')
                    hor_par.append(data_par[i][9])
                    hor_high.append(data_high[j][9])    
    print(len(hor_par))
    
    for i in range(len(hor_par)):
        plt.figure('Parallel')
        plt.clf()
        plt.plot(hor_par[i],  linestyle='-', color='b')
        #plt.legend(['x1','x2','x3'])
        plt.grid()
        
        plt.figure('High 16')
        plt.clf()
        plt.plot(hor_high[i],  linestyle='-', color='b')
        #plt.legend(['x1','x2','x3'])
        plt.grid()
        plt.show()