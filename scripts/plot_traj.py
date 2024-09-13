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

def convergenceCriteria(x):
    mask = np.array([1,0,0,1,0,0])
    return np.linalg.norm(np.multiply(mask, x - model.x_ref))

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
    data = load_data('ParallelLimited',2,0,1e-3,'high',16) 
    
    
    conf = Parameters('triple_pendulum', 'naive',rti=False)
    model = getattr(models,'TriplePendulumModel')(conf)
    successes = 0
    for i in range(0,len(data)):
        if data[i][1]==1:
            successes +=1
    print(successes)
    for i in range(0,len(data)): 
        
        
            if data[i][1] !=1:
                print("FAILED")  
                print(convergenceCriteria(data[i][6][-2])) 
            else:
                print("SUCCESS")        
                plt.figure('q')
                plt.clf()
                plt.plot(np.array(data[i][2])[:,0],  linestyle='-', color='b')
                plt.plot(np.array(data[i][2])[:,1],  linestyle='-', color='g')
                plt.plot(np.array(data[i][2])[:,2],  linestyle='-', color='orange')
                plt.axhline(y=model.x_min[0], color='r', linestyle='-')
                plt.axhline(y=model.x_max[0], color='r', linestyle='-')
                
                plt.plot()
                plt.legend(['q1','q2','q3'])
                plt.grid(True)
                #plt.show()     
                
                plt.figure('dq')
                plt.clf()
                plt.plot(np.array(data[i][2])[:,3],  linestyle='-', color='b')
                plt.plot(np.array(data[i][2])[:,4],  linestyle='-', color='g')
                plt.plot(np.array(data[i][2])[:,5],  linestyle='-', color='orange')
                plt.axhline(y=model.x_min[3], color='r', linestyle='-')
                plt.axhline(y=model.x_max[3], color='r', linestyle='-')
                
                
                plt.plot()
                plt.legend(['dq1','dq2','dq3'])
                plt.grid(True)
                #plt.show()  
                
                plt.figure('u')
                plt.clf()
                plt.plot(np.array(data[i][5])[:,0],  linestyle='-', color='b')
                plt.plot(np.array(data[i][5])[:,1],  linestyle='-', color='g')
                plt.plot(np.array(data[i][5])[:,2],  linestyle='-', color='orange')
                plt.axhline(y=model.u_min[0], color='r', linestyle='-')
                plt.axhline(y=model.u_max[0], color='r', linestyle='-')
                
                
                plt.plot()
                plt.legend(['u1','u2','u3'])
                plt.grid(True)
                plt.show()  