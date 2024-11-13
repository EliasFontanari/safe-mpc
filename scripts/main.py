import os
import pickle
import numpy as np
from tqdm import tqdm
import safe_mpc.model as models
import safe_mpc.controller as controllers
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.abstract_multiphase import SimDynamics
#from safe_mpc.abstract import SimDynamics
from safe_mpc.gravity_compensation import GravityCompensation
from acados_template import AcadosOcpSolver
from datetime import datetime


def convergenceCriteria(x, mask=None):
    if mask is None:
        mask = np.ones(model.nx)
    return np.linalg.norm(np.multiply(mask, x - model.x_ref)) < conf.conv_tol 


def init_guess(q0):
    x0 = np.zeros((model.nx,))
    x0[:model.nq] = q0
    #u0 = gc.solve(x0)
    flag = controller.initialize(x0)
    return controller.getGuess(), flag

def simulate_mpc(p):
    #controller.reinit_solver()
    #controller.ocp_solver = AcadosOcpSolver(controller.ocp,verbose=False)
    x0 = np.zeros((model.nx,))
    x0[:model.nq] = x0_vec[p]

    x_sim = np.empty((conf.n_steps + 1, model.nx)) * np.nan
    u = np.empty((conf.n_steps, model.nu)) * np.nan
    x_sim[0] = x0
    simulator_state = x0

    controller.setGuess(x_guess_vec[p], u_guess_vec[p])
    if 'parallel' in args['controller']:
        controller.safe_hor = controller.N 
        controller.alternative_x_guess = controller.x_guess
        controller.alternative_u_guess = controller.u_guess
    if args['controller'] == 'receding':
        controller.r=controller.N
    controller.fails = 0
    stats = []
    convergence = 0
    k = 0
    x_simu[p].append(x0)

    if 'parallel' in args['controller']:
        safe = controller.safe_hor
        safe_hor_hist[p].append(safe)
    elif args['controller'] == 'receding':
        safe = controller.r
        safe_hor_hist[p].append(safe)
    
    
    for k in range(conf.n_steps):        
        if k ==54:
            pass
        u[k] = controller.step(x_sim[k])
        #stats.append(controller.getTime())
        #simulator_state = controller.simulate_solver(x_sim[k], u[k])
        x_sim[k+1]=simulator.simulate(x_sim[k], u[k])
        #print(controller.x_viable-controller.simulator.checkSafeIntegrate([x_sim[k]],controller.u_guess,controller.safe_hor)[1])
        x_simu[p].append(x_sim[k + 1])
        u_simu[p].append(u[k])

        with open(folder_name+'integration.txt', 'a') as file:
            np.set_printoptions(precision=6)
            file.write(f'solver state: {controller.x_guess[0]}\n')
            file.write(f'simulator solver state: {x_sim[k+1]}\n')
            file.write(f'simulator state: {simulator_state}\n')
            file.write(f'difference: {np.abs(x_sim[k+1]-controller.x_guess[0])}\n')
            file.write(f'control: {u[k]}\n')
            if 'parallel' in args['controller']:
                file.write(f'problem :{p} step: {k} safe_hor:{controller.safe_hor}\n')
            if args['controller'] == 'receding':
                file.write(f'problem :{p} step: {k} safe_hor:{controller.r}\n')

        err_intgr = np.abs(x_sim[k+1]-controller.x_guess[0])
        if 'parallel' in args['controller']:
            jump = controller.safe_hor - safe
            # if controller.safe_hor - safe > 6:
            #     print(f'new hor at problem :{p} step: {k}\n')
            safe = controller.safe_hor
            safe_hor_hist[p].append(safe)
            jumps.append(jump)
            error_jumps[jump+1].append(err_intgr)
            if controller.core_solution != None:
                core_sol.append(controller.core_solution)

        elif args['controller'] == 'receding':
            jump = controller.r - safe
            safe = controller.r
            safe_hor_hist[p].append(safe)
            jumps.append(jump)
            error_jumps[jump+1].append(np.abs(err_intgr))
        errors[p].append(err_intgr)
        # if controller.step_old_solution > 0:
        #     controller.guessCorrection()
        #     #controller.step_old_solution = 0
        #controller.guessCorrection()

        
        # Check if the next state is inside the state bounds

        # if (np.abs(x_sim[k+1]-controller.simulate_solver(x_sim[k], u[k]))>1e-6).any():
        #     print(np.abs(x_sim[k+1]-controller.simulate_solver(x_sim[k], u[k])))
        # with open(conf.DATA_DIR+ 'constraint_violations/'+'integration.txt', 'a') as file:
        #     file.write(f'residuals: {controller.ocp_solver.get_residuals()}\n')
        #     file.write(f'sqp_iter:{controller.ocp_solver.get_stats("sqp_iter")}\n')
        #     file.write(f'qp_iter:{controller.ocp_solver.get_stats("qp_iter")}\n')
        #     file.write(f'qp_stat:{controller.ocp_solver.get_stats("qp_stat")}\n')

        # if not controller.checkStateConstraintsController(x_sim[k + 1]):
        #     violation_max = (x_sim[k+1]>controller.model.x_max)
        #     violation_min = (x_sim[k+1]<controller.model.x_min)
        #     # for i in range(violation_max.shape[0]):
        #     #     if args['controller'] == 'parallel' or args['controller'] == 'parallel2':
        #     #         if violation_max[i]: print(f'no tol upper bounds violated of : {x_sim[k+1][i]-controller.model.x_max[i]}, safe_hor = {controller.safe_hor}, state={i}\n')
        #     #         if violation_min[i]: print(f'no tol lower bounds violated of : {-x_sim[k+1][i]+controller.model.x_min[i]}, safe_hor = {controller.safe_hor}, state ={i}\n')
        if not model.checkStateConstraints(x_sim[k + 1]):
            if args['controller'] == 'receding':
                print(controller.r)
            print(f"{p}:{x0}=> Violated constraint")
            with open(folder_name+'integration.txt', 'a') as file:
                if (u[k][0]==None):
                    file.write('NOT SOLVED')
                file.write(f"{p}:{x0}=> Violated constraint\n")
            if not(np.isnan(u[k][0])):
                violation_max = (x_sim[k+1]>controller.model.x_max)
                violation_min = (x_sim[k+1]<controller.model.x_min)
                for i in range(violation_max.shape[0]):
                    if violation_max[i]: violations.append([x_sim[k+1][i]-controller.model.x_max[i],p,k])#,controller.safe_hor])
                    if violation_min[i]: violations.append([-x_sim[k+1][i]+controller.model.x_min[i],p,k])#,controller.safe_hor])
                # print(f'control={u[k]}, state={x_sim[k+1]}\n')
                # print(k)
            break
        # Check convergence --> norm of diff btw x_sim and x_ref (only for first joint)
        if convergenceCriteria(x_sim[k + 1], np.array([1, 0, 0, 1, 0, 0])):
            convergence = 1
            print(f"{p}:{x0}=> SUCCESS")
            with open(folder_name+'integration.txt', 'a') as file:
                file.write(f"{p}:{x0}=> SUCCESS")
            x0_success.append(x0)
            break
        if k == conf.n_steps-1:
            print(f'{p}:{x0}=>Not converged\n')
            with open(folder_name+'integration.txt', 'a') as file:
                file.write(f'{p}:{x0}=>Not converged\n')
    x_v = controller.getLastViableState()
    with open(folder_name+'integration.txt', 'a') as file:
        file.write('\n\n\n')
    return k, convergence, x_sim, stats, x_v, u


if __name__ == '__main__':
    args = parse_args()
    # Define the available systems and controllers
    available_systems = {'double_pendulum': 'DoublePendulumModel',
                         'triple_pendulum': 'TriplePendulumModel'}
    available_controllers = {'naive': 'NaiveController',
                             'st': 'STController',
                             'stwa': 'STWAController',
                             'htwa': 'HTWAController',
                             'receding': 'RecedingController',
                             'receding_single':'RecedingSingleConstraint',
                             'parallel': 'ParallelWithCheck',
                             'parallel_limited':'ParallelLimited',
                             'abort': 'SafeBackupController'}
    if args['init_conditions']:
        args['controller'] = 'receding'
    # Check if the system and controller selected are available
    if args['system'] not in available_systems.keys():
        raise ValueError('Unknown system. Available: ' + str(available_systems))
    if args['controller'] not in available_controllers.keys():
        raise ValueError('Unknown controller. Available: ' + str(available_controllers))

    # Define the configuration object, model, simulator and controller
    conf = Parameters(args['system'], args['controller'], args['rti'])
    # Set the safety margin
    #conf.alpha = args['alpha']
    model = getattr(models, available_systems[args['system']])(conf)
    gc = GravityCompensation(conf, model)
    simulator = SimDynamics(model)
    controller = getattr(controllers, available_controllers[args['controller']])(simulator)
    controller.setReference(model.x_ref)

    # Check if data folder exists, if not create it
    if not os.path.exists(conf.DATA_DIR):
        os.makedirs(conf.DATA_DIR)
    data_name = conf.DATA_DIR + args['controller'] + '_'

    print(f'Running {available_controllers[args["controller"]]} with alpha = {conf.alpha}...')
    print(f'x_min = {controller.model.x_min}\n')
    print(f'x_max = {controller.model.x_max}\n')
    reset_step=20
    #x_simu = []
    x_simu = [[] for _ in range(conf.test_num)]
    u_simu = [[] for _ in range(conf.test_num)]
    safe_hor_hist = [[] for _ in range(conf.test_num)]
    jumps = []
    error_jumps = [[] for _ in range(controller.N+1)]
    errors = [[] for _ in range(conf.test_num)]
    core_sol = []
    
    now = datetime.now()
    folder_name = conf.DATA_DIR+now.strftime("%Y-%m-%d_%H-%M-%S") + str(controller.ocp.solver_options.qp_solver) \
        + str(controller.ocp.solver_options.qp_solver_iter_max) \
        + str(conf.n_steps)+'/'
    os.makedirs(folder_name)
    with open(folder_name+'integration.txt', 'w') as file:
        pass
    # If ICs is active, compute the initial conditions for all the controller
    if args['init_conditions']:
        from scipy.stats import qmc

        sampler = qmc.Halton(model.nq, scramble=False)
        l_bounds = model.x_min[:model.nq] + conf.state_tol
        u_bounds = model.x_max[:model.nq] - conf.state_tol

        # Soft constraints on all the trajectory
        for i in range(1, controller.N):
            controller.ocp_solver.cost_set(i, "zl", conf.ws_r * np.ones((1,)))
        controller.ocp_solver.cost_set(controller.N, "zl", conf.ws_t * np.ones((1,)))

        bins = 10
        x0_0_intervals = np.linspace(controller.model.x_min[0],controller.model.x_max[0],bins+1)
        h = 0
        j = 0
        failures = 0
        x_init_vec, x_guess_vec, u_guess_vec = [], [], []
        progress_bar = tqdm(total=conf.test_num, desc='Init guess processing')
        while h < conf.test_num:
            found = 0
            while j < bins:
                x_g0 = qmc.scale(sampler.random(), l_bounds, u_bounds)
                if x0_0_intervals[j]<=x_g0[0][0]<x0_0_intervals[j+1]:
                    (x_g, u_g), status = init_guess(x_g0)
                    if status:
                        x_init_vec.append(x_g[0, :model.nq])
                        x_guess_vec.append(x_g)
                        u_guess_vec.append(u_g)
                        h += 1
                        progress_bar.update(1)
                        found +=1
                    else:
                        failures += 1
                    if controller.ocp_solver.get_stats('qp_stat')[-1]==3: # failures % reset_step == 0:
                        print('RESET')
                        controller.reinit_solver()
                        for i in range(1, controller.N):
                            controller.ocp_solver.cost_set(i, "zl", conf.ws_r * np.ones((1,)))
                        controller.ocp_solver.cost_set(controller.N, "zl", conf.ws_t * np.ones((1,)))
                    if found == conf.test_num/bins:
                        found =0
                        j+=1
                        with open(folder_name+'integration.txt', 'a') as file:
                            file.write('!!!!!!!!!!!!!!!!!!!!\n!!!!!!!!!!!!!!!\n\n\n')
                

        progress_bar.close()
        np.save(conf.DATA_DIR + f'x_init_{conf.alpha}.npy', np.asarray(x_init_vec))
        np.save(conf.DATA_DIR + f'x_guess_vec_{conf.alpha}.npy', np.asarray(x_guess_vec))
        np.save(conf.DATA_DIR + f'u_guess_vec_{conf.alpha}.npy', np.asarray(u_guess_vec))
        print(f'Found {conf.test_num} initial conditions after {failures} failures.')

    elif args['guess']:
        x_init_vec = np.load(conf.DATA_DIR + f'x_init_{conf.alpha}.npy')
        x_guess_vec, u_guess_vec, successes = [], [], []
        progress_bar = tqdm(total=conf.test_num, desc='Init guess processing')
        if args['controller'] in ['naive', 'st']:
            for x_init in x_init_vec:
                (x_g, u_g), status = init_guess(x_init)
                x_guess_vec.append(x_g)
                u_guess_vec.append(u_g)
                successes.append(status)
                progress_bar.update(1)
        else:
            x_feasible = np.load(conf.DATA_DIR + f'x_guess_vec_{conf.alpha}.npy')
            u_feasible = np.load(conf.DATA_DIR + f'u_guess_vec_{conf.alpha}.npy')
            # Try to refine the guess with respect to the controller used
            for i in range(conf.test_num):
                controller.setGuess(x_feasible[i], u_feasible[i])
                x_init = np.zeros((model.nx,))
                x_init[:model.nq] = x_init_vec[i]
                if 'receding_single' in args['controller']:
                    controller.r = controller.N-1  
                status = controller.solve(x_init)
                if (status == 0 or status == 2) and controller.checkGuess():
                    # Refinement successful
                    x_g, u_g = controller.getGuess()
                    x_guess_vec.append(x_g)
                    u_guess_vec.append(u_g)
                    successes.append(1)
                else:
                    # Refinement failed, use the feasible guess
                    x_guess_vec.append(x_feasible[i])
                    u_guess_vec.append(u_feasible[i])
                progress_bar.update(1)
        progress_bar.close()
        np.save(data_name + f'x_guess_{controller.params.alpha}.npy', np.asarray(x_guess_vec))
        np.save(data_name + f'u_guess_{controller.params.alpha}.npy', np.asarray(u_guess_vec))
        print('Init guess success: %d over %d' % (sum(successes), conf.test_num))

    elif args['rti'] and args['controller']!= 'abort':
        x0_success = []
        violations=[]
        x0_vec = np.load(conf.DATA_DIR + f'x_init_{conf.alpha}.npy')
        x_guess_vec = np.load(data_name + f'x_guess_{conf.alpha}.npy')
        u_guess_vec = np.load(data_name + f'u_guess_{conf.alpha}.npy')
        res = []
        progress_bar = tqdm(total=conf.test_num, desc='Running on %d' %(conf.test_num))
        for i in range(0,conf.test_num):
            res.append(simulate_mpc(i))
            progress_bar.update(1)
        progress_bar.close()
        steps, conv_vec, x_sim_vec, t_stats, x_viable, u_sim_vec = zip(*res)
        steps = np.array(steps)
        conv_vec = np.array(conv_vec)
        idx = np.where(conv_vec == 1)[0]
        idx_abort = np.where(conv_vec == 0)[0]
        print('Total convergence: %d over %d' % (np.sum(conv_vec), conf.test_num))

        print('99% quantile computation time:')
        times = np.array([t for arr in t_stats for t in arr])
        for field, t in zip(controller.time_fields, np.quantile(times, 0.99, axis=0)):
            print(f"{field:<20} -> {t}")
        
        with open(folder_name+'integration.txt', 'a') as file:
            file.write('Total convergence: %d over %d\n' % (np.sum(conv_vec), conf.test_num))
            for field, t in zip(controller.time_fields, np.quantile(times, 0.99, axis=0)):
                file.write(f"{field:<20} -> {t}\n")

        # Save last viable states (useful only for terminal/receding controllers)
        np.save(data_name + 'x_viable.npy', np.asarray(x_viable)[idx_abort])

        # Save all the results
        with open(data_name + 'results.pkl', 'wb') as f:
            pickle.dump({'x_sim': np.asarray(x_sim_vec),
                         'times': times,
                         'steps': steps,
                         'idx_abort': idx_abort,
                         'x_viable': np.asarray(x_viable)}, f)
        # Save constraint violations
        # viol_dir=conf.DATA_DIR+ 'constraint_violations/'
        # if not os.path.exists(viol_dir):
        #     os.makedirs(viol_dir)
        # np.savetxt(viol_dir+args['controller']+f'{controller.model.state_tol}.txt', violations)
        with open(data_name + 'x0succes.pkl', 'wb') as f:
                pickle.dump(x0_success, f)
        with open(folder_name + 'x_u.pkl', 'wb') as f:
                pickle.dump({'x_sim': x_simu,
                             'u_sim': u_simu},f)
        with open(folder_name + 'jumps.pkl', 'wb') as f:
                pickle.dump(jumps,f)
        with open(folder_name + 'safehor_hist.pkl', 'wb') as f:
                pickle.dump(safe_hor_hist,f)
        with open(folder_name + 'error_jump.pkl', 'wb') as f:
                pickle.dump(error_jumps,f)
        with open(folder_name + 'errors.pkl', 'wb') as f:
                pickle.dump(errors,f)
        with open(folder_name + 'coreused.pkl', 'wb') as f:
                pickle.dump(core_sol,f)
        
    

    elif args['controller'] == 'abort' and args['abort'] in ['stwa', 'htwa', 'receding','parallel','parallel_limited','parallel2']:
        import matplotlib.pyplot as plt
        tgrid = np.linspace(0,conf.T,int(conf.T/conf.dt)+1)
        # Increase time horizon
        x_viable = np.load(conf.DATA_DIR + args['abort'] + '_' + 'x_viable.npy')
        n_a = np.shape(x_viable)[0]
        rep = args['repetition']
        t_rep = np.empty((n_a, rep)) * np.nan
        controller.model.setNNmodel()
        for i in range(n_a):
            print(f'{i} x_viable={x_viable[i]} \n\n\n\n\n')
            controller.reinit_solver()
            #controller.ocp_solver.reset()
            # Repeat each test rep times
            for j in range(rep):
                #u0 = gc.solve(x_viable[i])
                controller.setGuess(np.full((controller.N + 1, model.nx), x_viable[i]),
                                    np.zeros((controller.N,model.nu)))
                status = controller.solve(x_viable[i])
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
              % (args['abort'], len(t_min), n_a, 0))
    else:
        pass


    # elif args['controller'] == 'abort' and args['abort'] in ['stwa', 'htwa', 'receding','parallel']:
    #     # Increase time horizon
    #     x_viable = np.load(conf.DATA_DIR + args['abort'] + '_' + 'x_viable.npy')
    #     n_a = np.shape(x_viable)[0]
    #     rep = args['repetition']
    #     t_rep = np.empty((n_a, rep)) * np.nan
    #     for i in range(n_a):
    #         # Repeat each test rep times
    #         for j in range(rep):
    #             print(i)
    #             controller.setGuess(np.full((controller.N + 1, model.nx), x_viable[i]),
    #                                 np.zeros((controller.N, model.nu)))
    #             status = controller.solve(x_viable[i])
    #             if status == 0:
    #                 t_rep[i, j] = controller.ocp_solver.get_stats('time_tot')
    #                 print(controller.x_temp[-1])
                    
    #     # Compute the minimum time for each initial condition
    #     t_min = np.min(t_rep, axis=1)
    #     # Remove the nan values (i.e. the initial conditions for which the controller failed)
    #     t_min = t_min[~np.isnan(t_min)]
    #     print('Controller: %s\nAbort: %d over %d\nQuantile (99) time: %.3f'
    #           % (args['abort'], len(t_min), n_a, 0))
    # else:
    #     pass

