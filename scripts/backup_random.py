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
    x0_b = model.x_min[0] if np.abs(x[0]-model.x_min[0]) < np.abs(x[0]-model.x_max[0]) else model.x_max[0] 
    x1_b = model.x_min[1] if np.abs(x[1]-model.x_min[1]) < np.abs(x[1]-model.x_max[1]) else model.x_max[1]
    x2_b = model.x_min[2] if np.abs(x[2]-model.x_min[2]) < np.abs(x[2]-model.x_max[2]) else model.x_max[2] 
    
     
    #return min(np.abs(x[0]-x0_b),np.abs(x[1]-x1_b),np.abs(x[2]-x2_b))
    return np.linalg.norm(np.array([x[0]-x0_b,x[1]-x1_b,x[2]-x2_b]))
if __name__ == '__main__':
    # Define the configuration object, model, simulator and controller
    conf = Parameters('triple_pendulum', 'abort',rti=False)
    conf.alpha = 2
    model = getattr(models,'TriplePendulumModel')(conf)
    #gc = GravityCompensation(conf, model)
    simulator = SimDynamics(model)
    
    controller = getattr(controllers, 'SafeBackupController')(simulator)
    controller.setReference(model.x_ref)
    controller.model.setNNmodel()
    x0 = np.ones(model.nx)*1000
    print(controller.model.params.alpha)
    tests=100
    successes=0
    for i in range(tests):
        while True:
            x0[:model.nq] =  (model.x_max-model.x_min)[0]*np.random.random_sample((3,)) + model.x_min[0]*np.ones((3,))
            x0[model.nq:] =  (model.x_max-model.x_min)[-1]*np.random.random_sample((3,)) + model.x_min[-1]*np.ones((3,))
            if 0.5>controller.model.nn_func(x0, controller.model.params.alpha) >0:     
                if bounds_dist(x0)<0.1:
                    break
        print(x0)
        print(bounds_dist(x0))
        print(controller.model.nn_func(x0, controller.model.params.alpha) )
        controller.setGuess(np.full((controller.N + 1, model.nx), x0),
                                np.zeros((controller.N,model.nu)))
        status = controller.solve(x0)
        if status == 0 or status==2:
            if (np.abs(controller.x_temp[-1][3:]) < 1e-2).all():
                    
                    integrated_sol=x0.reshape(1,-1) 
                    for k in range(controller.u_temp.shape[0]):
                        integrated_sol= np.vstack([integrated_sol,simulator.simulate(integrated_sol[k,:],controller.u_temp[k])])

                    if controller.model.checkStateConstraints(integrated_sol) and (np.abs(integrated_sol[-1,3:])<1e-3).all():
                        print('SUCCESS')
                        print(f'closeness :{bounds_dist(x0)}')
                        successes+=1
        else: print('Failed')
        x0=1000*np.zeros(model.nx)
    print(successes/tests)