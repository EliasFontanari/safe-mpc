import time
import numpy as np
import matplotlib.pyplot as plt
import meshcat
from casadi import Function
from safe_mpc.parser import Parameters
from safe_mpc.abstract import AdamModel
from safe_mpc.utils import obstacles
from safe_mpc.controller import NaiveController 
import pinocchio as pin
import pickle

model_name = 'fr3'
cont_name = 'naive'
rti = True
N = 30
params = Parameters(model_name, rti)
params.build = False
model = AdamModel(params, n_dofs=7)

ee = np.empty((params.n_steps, 3)) * np.nan
ee_radius = 0.075
ee_ref = np.array([0.6, 0.28, 0.078])

controller = NaiveController(model, obstacles)
controller.resetHorizon(N)
controller.setReference(ee_ref)

# q0 = np.array([-0.430, 2.469, -1.838, -0.781])
# x0 = np.zeros((model.nx,))
# x0[:model.nq] = q0
# u0 = np.zeros(model.nq)        
flag_mpc = False

if rti:
    data = pickle.load(open(f'{params.DATA_DIR}{model_name}_{cont_name}_guess.pkl', 'rb'))
    x_guess = data['xg']
    u_guess = data['ug']

    init_x = np.random.randint(0,x_guess.shape[0])


    q0 = x_guess[init_x,0,:model.nq]
    x0 = np.zeros((model.nx,))
    x0[:model.nq] = q0

    fk = Function('fk', [model.x], [model.fk(np.eye(4), model.x[:model.nq])]) 

    x_sim = np.empty((params.n_steps + 1, model.nx)) * np.nan
    u = np.empty((params.n_steps, model.nu)) * np.nan
    x_sim[0] = x_guess[init_x,0]
    
    controller.setGuess(x_guess[init_x], u_guess[init_x])
    for k in range(params.n_steps):
        u[k] = controller.step(x_sim[k])[0]
        # print(controller.ocp_solver.get_stats('residuals'))
        x_sim[k + 1] = model.integrate(x_sim[k], u[k])[0]
        # Check if the next state is inside the state bounds
        if not model.checkStateConstraints(x_sim[k + 1]):
            print('Violation (state constraint) at: ', x_sim[k + 1])
            break
        if not controller.checkCollision(x_sim[k + 1]):
            print('Collision at: ', x_sim[k + 1])
            break

        T_ee = fk(x_sim[k])
        T_ee[:3, 3] += T_ee[:3, :3] @ model.t_loc
        ee[k,:] = T_ee[:3, 3].T

    flag_mpc = True

else:
    flag = controller.initialize(x0, u0)    
    controller.ocp_solver.print_statistics()
    if flag:
        print('\nSuccess!\n')
        x_guess, u_guess = controller.getGuess()
        np.save('data/x_guess.npy', x_guess)
        np.save('data/u_guess.npy', u_guess)

    else:
        print('\nFailed!\n')

if flag_mpc:
    description_dir = params.ROBOTS_DIR + f'{model_name}_description'
    rmodel, collision, visual = pin.buildModelsFromUrdf(description_dir + f'/urdf/{model_name}.urdf',
                                                        package_dirs=params.ROOT_DIR)
    geom = [collision, visual]

    lockIDs = []
    joint_names = [joint.name for joint in model.robot.joints if joint.type != 'fixed']
    lockNames =  [x for x in joint_names if x not in model.joint_names]
    for name in lockNames:
        lockIDs.append(rmodel.getJointId(name))

    rmodel_red, geom_red = pin.buildReducedModel(rmodel, geom, lockIDs, np.zeros(rmodel.nq))

    viz = pin.visualize.MeshcatVisualizer(rmodel_red, geom_red[0], geom_red[1])
    viz.initViewer(loadModel=True, open=True)
    # viz.setCameraPosition(np.array([1.2, 0., 0.4]))
    #viz.setCameraPosition(np.array([0., 1.2, 0.4]))
    viz.display(q0)

    box = meshcat.geometry.Box([2, 2, 1e-3])
    viz.viewer['world/obstacle/floor'].set_object(box)
    viz.viewer['world/obstacle/floor'].set_property('color', [0, 0, 1, 0.5])
    viz.viewer['world/obstacle/floor'].set_property('visible', True)
    T_floor = np.eye(4)
    viz.viewer['world/obstacle/floor'].set_transform(T_floor)

    shpere = meshcat.geometry.Sphere(ee_radius)
    viz.viewer['world/robot/target'].set_object(shpere)
    viz.viewer['world/robot/target'].set_property('color', [0, 1, 0, 0.5])
    viz.viewer['world/robot/target'].set_property('visible', True)
    T_target = np.eye(4)
    T_target[:3, 3] = ee_ref
    viz.viewer['world/robot/target'].set_transform(T_target)

    viz.viewer['world/robot/ee'].set_object(shpere)
    viz.viewer['world/robot/ee'].set_property('color', [1, 1, 0, 0.5])
    viz.viewer['world/robot/ee'].set_property('visible', True)
    T_ee = np.eye(4)
    T_ee[:3, 3] = ee[0]
    viz.viewer['world/robot/ee'].set_transform(T_ee)

    ball = meshcat.geometry.Sphere(0.12)
    viz.viewer['world/obstacle/ball'].set_object(ball)
    viz.viewer['world/obstacle/ball'].set_property('color', [0, 1, 1, 0.5])
    viz.viewer['world/obstacle/ball'].set_property('visible', True)
    T_ball = np.eye(4)
    T_ball[:3, 3] = np.array([0.5, 0., 0.12])
    viz.viewer['world/obstacle/ball'].set_transform(T_ball)

    time.sleep(5)
    for i in range(1, params.n_steps):
        viz.display(x_sim[i, :model.nq])
        T_ee[:3, 3] = ee[i]
        viz.viewer['world/robot/ee'].set_transform(T_ee)
        time.sleep(params.dt*10)

    fig, ax = plt.subplots(3, 1, sharex='col')
    for i in range(3):
        ax[i].plot(ee[:, i], label='ee', lw=2)
        ax[i].axhline(ee_ref[i], color='r', linestyle='--', label='ref', lw=1.5)
        ax[i].legend()    
        ax[i].grid(True)    
        ax[i].set_ylabel('ee' + str(i + 1))
    ax[-1].set_xlabel('time')

    plt.show()