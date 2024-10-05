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


if __name__ == '__main__':
    available_controllers = {'naive': 'NaiveController',
                             'st': 'STController',
                             'stwa': 'STWAController',
                             'htwa': 'HTWAController',
                             'receding': 'RecedingController',
                             'parallel': 'ParallelWithCheck',
                             'parallel_limited':'ParallelLimited',
                             'abort': 'SafeBackupController'}
    
    control = 'naive'
    # Define the configuration object, model, simulator
    conf = Parameters('triple_pendulum', control,rti=True)
    model = getattr(models,'TriplePendulumModel')(conf)
    simulator = SimDynamics(model)
    
    x0 = np.array([0,np.pi,0,0,0,0])
    x_sim = []
    x_sim.append(x0) 
    for i in range(100000):
        x_sim.append(simulator.simulate(x_sim[i], np.array([0,0,0])))
    print(x_sim[-1])