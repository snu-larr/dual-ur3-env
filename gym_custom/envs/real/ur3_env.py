import numpy as np
import time
import sys
import warnings
import pickle
import os
import gym_custom
from gym_custom.envs.real.ur.interface import URScriptInterface, convert_action_to_space, convert_observation_to_space, COMMAND_LIMITS
from gym_custom.envs.real.utils import ROSRate, prompt_yes_or_no

class UR3RealEnv(gym_custom.Env):
    
    def __init__(self, host_ip, rate):
        self.host_ip = host_ip
        self.rate = ROSRate(rate)
        self.interface = URScriptInterface(host_ip)
        
        # dscho mod
        self._define_class_variables()
        self.dt = 1/self.rate._freq

        # UR3 (6DOF), 2F-85 gripper (1DOF)
        # self._init_qpos = np.zeros([6])
        # self._init_qvel = np.zeros([6])
        # self._init_gripperpos = np.zeros([1])
        # self._init_grippervel = np.zeros([1])
        
        # self.action_space =  self._set_action_space()
        # obs = self._get_obs()
        # self.observation_space = self._set_observation_space(obs)

        # self._episode_step = None

    def close(self):
        self.interface.comm.close()

    def set_initial_joint_pos(self, q=None):
        if q is None: pass
        elif q == 'current': self._init_qpos = self.interface.get_joint_positions()
        else:
            assert q.shape[0] == 6
            self._init_qpos = q
        print('Initial joint position is set to %s'%(q))
    
    def set_initial_joint_vel(self, qd=None):
        if qd is None: pass
        elif qd == 'current': self._init_qvel = self.interface.get_joint_speeds()
        else:
            assert qd.shape[0] == 6
            self._init_qvel = qd
        print('Initial joint velocity is set to %s'%(qd))

    def set_initial_gripper_pos(self, g=None):
        if g is None: pass
        elif g == 'current': self._init_gripperpos = self.interface.get_gripper_position()
        else:
            assert g.shape[0] == 1
            self._init_gripperpos = g
        print('Initial gripper position is set to %s'%(g))

    def set_initial_gripper_vel(self, gd=None):
        if gd is None: pass
        elif gd == 'current': self._init_grippervel = self.interface.get_gripper_position()
        else:
            assert gd.shape[0] == 1
            self._init_grippervel = gd
        print('Initial gripper velocity is set to %s'%(gd))

    def step(self, action):
        start = time.time()
        assert self._episode_step is not None, 'Must reset before step!'
        for command_type, command_val in action.items():
            getattr(self.interface, command_type)(**command_val)
        self._episode_step += 1
        self.rate.sleep()
        ob = self._get_obs()
        reward = 1.0
        done = False
        finish = time.time()
        if finish - start > 1.5/self.rate._freq:
            warnings.warn('Desired rate of %dHz is not satisfied! (current rate: %dHz, current time interval: %.4f)'%(self.rate._freq, 1/(finish-start), finish-start))
        return ob, reward, done, {}

    def reset(self):
        self.interface.reset_controller()
        ob = self.reset_model()
        self.rate.reset()
        return ob

    def render(self, mode='human'):
        warnings.warn('Real environment. "Render" with your own two eyes!')

    def close(self):
        self.interface.close()

    def reset_model(self):
        self.interface.movej(q=self._init_qpos)
        self.interface.move_gripper(g=self._init_gripperpos)
        self._episode_step = 0
        return self._get_obs()

    def get_obs_dict(self):
        return {
            'qpos': self.interface.get_joint_positions(),
            'qvel': self.interface.get_joint_speeds(),
            'gripperpos': self.interface.get_gripper_position(),
            'grippervel': self.interface.get_gripper_speed()
        }

    def _get_obs(self):
        return self._dict_to_nparray(self.get_obs_dict())

    @staticmethod
    def _dict_to_nparray(obs_dict):
        return np.concatenate([obs_dict['qpos'], obs_dict['gripperpos'], obs_dict['qvel'], obs_dict['grippervel']]).ravel()

    @staticmethod
    def _nparray_to_dict(obs_nparray):
        return {
            'qpos': obs_nparray[0:6],
            'qvel': obs_nparray[7:13],
            'gripperpos': obs_nparray[6:7],
            'grippervel': obs_nparray[13:14]
        }

    @staticmethod
    def _set_action_space():
        return convert_action_to_space(COMMAND_LIMITS)

    @staticmethod
    def _set_observation_space(observation):
        return convert_observation_to_space(observation)

    # from here, dscho add
    
    def _define_class_variables(self):
        '''overridable method'''
        # Initial position/velocity
        self._init_qpos = np.zeros([6])
        self._init_qvel = np.zeros([6])
        self._init_gripperpos = np.zeros([1])
        self._init_grippervel = np.zeros([1])

        # Variables for forward/inverse kinematics
        # https://www.universal-robots.com/articles/ur-articles/parameters-for-calculations-of-kinematics-and-dynamics/
        self.kinematics_params = {}

        # 1. Last frame aligns with (right/left)_ee_link body frame
        # self.kinematics_params['d'] = np.array([0.1519, 0, 0, 0.11235, 0.08535, 0.0819]) # in m
        # 2. Last frame aligns with (right/left)_gripper:hand body frame
        self.kinematics_params['d'] = np.array([0.1519, 0, 0, 0.11235, 0.08535, 0.0819+0.12]) # in m
        self.kinematics_params['a'] = np.array([0, -0.24365, -0.21325, 0, 0, 0]) # in m
        self.kinematics_params['alpha'] =np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]) # in rad
        self.kinematics_params['offset'] = np.array([0, 0, 0, 0, 0, 0])
        self.kinematics_params['ub'] = np.array([2*np.pi for _ in range(6)])
        self.kinematics_params['lb'] = np.array([-2*np.pi for _ in range(6)])
        
        if self.init_qpos_type=='upright':
            path_to_pkl = os.path.join(os.path.dirname(__file__), 'ur/ur3_kinematics_params.pkl')
        else: # dual arm posture params       
            path_to_pkl = os.path.join(os.path.dirname(__file__), 'ur/dual_ur3_kinematics_params.pkl')

        if os.path.isfile(path_to_pkl):
            kinematics_params_from_pkl = pickle.load(open(path_to_pkl, 'rb'))
            self.kinematics_params['T_wb_right'] = kinematics_params_from_pkl['T_wb_right']
            self.kinematics_params['T_wb_left'] = kinematics_params_from_pkl['T_wb_left']
        else:
            raise FileNotFoundError('No such file: %s. Run MuJoCo-based simulated environment to generate file.'%(path_to_pkl))
        
        # Define spaces
        self.action_space = self._set_action_space()
        obs = self._get_obs()
        self.observation_space = self._set_observation_space(obs)

        # Misc
        self._episode_step = None

    

    def forward_kinematics_DH(self, q, arm):
        assert len(q) == self.ur3_nqpos

        if arm == 'right':
            T_0_i = self.kinematics_params['T_wb_right']
        elif arm == 'left':
            T_0_i = self.kinematics_params['T_wb_left']
        else:
            raise ValueError('Invalid arm type!')
        T = np.zeros([self.ur3_nqpos+1, 4, 4])
        R = np.zeros([self.ur3_nqpos+1, 3, 3])
        p = np.zeros([self.ur3_nqpos+1, 3])
        # Base frame
        T[0,:,:] = T_0_i
        R[0,:,:] = T_0_i[0:3,0:3]
        p[0,:] = T_0_i[0:3,3]

        for i in range(self.ur3_nqpos):
            ct = np.cos(q[i] + self.kinematics_params['offset'][i])
            st = np.sin(q[i] + self.kinematics_params['offset'][i])
            ca = np.cos(self.kinematics_params['alpha'][i])
            sa = np.sin(self.kinematics_params['alpha'][i])

            T_i_iplus1 = np.array([[ct, -st*ca, st*sa, self.kinematics_params['a'][i]*ct],
                                   [st, ct*ca, -ct*sa, self.kinematics_params['a'][i]*st],
                                   [0, sa, ca, self.kinematics_params['d'][i]],
                                   [0, 0, 0, 1]])
            T_0_i = np.matmul(T_0_i, T_i_iplus1)
            # cf. base frame at i=0
            T[i+1, :, :] = T_0_i
            R[i+1, :, :] = T_0_i[0:3,0:3]
            p[i+1, :] = T_0_i[0:3,3]

        return R, p, T

    def forward_kinematics_ee(self, q, arm):
        R, p, T = self.forward_kinematics_DH(q, arm)
        return R[-1,:,:], p[-1,:], T[-1,:,:]

    def _jacobian_DH(self, q, arm):
        assert len(q) == self.ur3_nqpos
        epsilon = 1e-6
        epsilon_inv = 1/epsilon
        _, ps, _ = self.forward_kinematics_DH(q, arm)
        p = ps[-1,:] # unperturbed position

        jac = np.zeros([3, self.ur3_nqpos])
        for i in range(self.ur3_nqpos):
            q_ = q.copy()
            q_[i] = q_[i] + epsilon
            _, ps_, _ = self.forward_kinematics_DH(q_, arm)
            p_ = ps_[-1,:] # perturbed position
            jac[:, i] = (p_ - p)*epsilon_inv

        return jac

    def inverse_kinematics_ee(self, ee_pos, null_obj_func, arm,
            q_init='current', threshold=0.01, threshold_null=0.001, max_iter=100, epsilon=1e-6
        ):
        '''
        inverse kinematics with forward_kinematics_DH() and _jacobian_DH()
        '''
        # Set initial guess
        if arm == 'right':
            if type(q_init).__name__ == 'ndarray': q = q_init.copy()
            elif q_init == 'current': q = self.interface_right.get_joint_positions()
            elif q_init == 'zero': q = np.zeros([self.ur3_nqpos])
            else: raise ValueError("q_init must be one of the following: ['current', 'zero', numpy.ndarray]")
        elif arm == 'left':
            if type(q_init).__name__ == 'ndarray': q = q_init.copy()
            elif q_init == 'current': q = self.interface_left.get_joint_positions()
            elif q_init == 'zero': q = np.zeros([self.ur3_nqpos])
            else: raise ValueError("q_init must be one of the following: ['current', 'zero', numpy.ndarray]")
        else:
            raise ValueError('Invalid arm type!')
        
        SO3, x, _ = self.forward_kinematics_ee(q, arm)
        jac = self._jacobian_DH(q, arm)
        delta_x = ee_pos - x
        err = np.linalg.norm(delta_x)
        null_obj_val = null_obj_func.evaluate(SO3)
        iter_taken = 0

        while True:
            if (err < threshold and null_obj_val < threshold_null) or iter_taken >= max_iter: break
            else: iter_taken += 1

            # pseudo-inverse + null-space approach
            jac_dagger = np.linalg.pinv(jac)
            jac_null = np.eye(self.ur3_nqpos) - np.matmul(jac_dagger, jac) # null space of Jacobian
            phi = np.zeros(self.ur3_nqpos) # find phi (null objective derivative)
            for i in range(self.ur3_nqpos):
                q_perturb = q.copy()
                q_perturb[i] += epsilon
                SO3_perturb, _, _ = self.forward_kinematics_ee(q_perturb, arm)
                null_obj_val_perturb = null_obj_func.evaluate(SO3_perturb)
                phi[i] = (null_obj_val_perturb - null_obj_val)/epsilon
            # update
            delta_x = ee_pos - x
            delta_q = np.matmul(jac_dagger, delta_x) - np.matmul(jac_null, phi)
            q += delta_q
            q = np.minimum(self.kinematics_params['ub'], np.maximum(q, self.kinematics_params['lb'])) # clip within theta bounds
            SO3, x, _ = self.forward_kinematics_ee(q, arm)
            jac = self._jacobian_DH(q, arm)
            null_obj_val = null_obj_func.evaluate(SO3)
            # evaluate
            err = np.linalg.norm(delta_x)
        
        if iter_taken == max_iter:
            warnings.warn('Max iteration limit reached! err: %f (threshold: %f), null_obj_err: %f (threshold: %f)'%(err, threshold, null_obj_val, threshold_null),
                RuntimeWarning)
        
        return q, iter_taken, err, null_obj_val


## Examples
def servoj_speedj_example(host_ip, rate):
    real_env = UR3RealEnv(host_ip=host_ip, rate=rate)
    real_env.set_initial_joint_pos('current')
    real_env.set_initial_gripper_pos('current')
    if prompt_yes_or_no('current qpos is %s deg?'%(np.rad2deg(real_env._init_qpos))) is False:
        print('exiting program!')
        sys.exit()
    obs = real_env.reset()
    init_qpos = real_env._nparray_to_dict(obs)['qpos']
    goal_qpos = init_qpos.copy()
    goal_qpos[-1] += np.pi/2*1.0
    waypoints_qpos = np.linspace(init_qpos, goal_qpos, rate*2, axis=0)
    waypoints_qvel = np.diff(waypoints_qpos, axis=0)*real_env.rate._freq
    
    # close-open-close gripper
    # print('close')
    # real_env.step({'close_gripper': {}})
    # time.sleep(3.0)
    # print('open')
    # real_env.step({'open_gripper': {}})
    # time.sleep(3.0)
    # print('close')
    # real_env.step({'close_gripper': {}})
    # time.sleep(5.0)
    

    wait = True
    
    print('test open gripper')
    real_env.interface.move_gripper(g=10, wait=wait)
    time.sleep(3)
    grip_pos = real_env.interface.get_gripper_position()
    print('grip pos : ', grip_pos)

    print('test close gripper')
    real_env.interface.move_gripper(g=150, wait=wait)
    time.sleep(3)
    grip_pos = real_env.interface.get_gripper_position()
    print('grip pos : ', grip_pos)


    if prompt_yes_or_no('servoj to %s deg?'%(np.rad2deg(goal_qpos))) is False:
        print('exiting program!')
        sys.exit()
    # servoj example
    print('Testing servoj')
    start = time.time()
    for n, waypoint in enumerate(waypoints_qpos[1:,:]):
        real_env.step({
            'servoj': {'q': waypoint, 't': 2/real_env.rate._freq, 'wait': False},
            'move_gripper_position': {'g' : 0, 'wait' : False} # open
        })
        print('action %d sent!'%(n))
    real_env.step({'stopj': {'a': 5}})
    finish = time.time()
    print('done! (elapsed time: %.3f [s])'%(finish - start))
    time.sleep(5)
    curr_qpos = real_env._nparray_to_dict(real_env._get_obs())['qpos']
    print('current - goal qpos is %s deg'%(np.rad2deg(curr_qpos - goal_qpos)))
    time.sleep(5)
    print('Moving to initial position...')
    real_env.step({'movej': {'q': init_qpos}})
    print('done!')

    if prompt_yes_or_no('speedj to %s deg?'%(np.rad2deg(goal_qpos))) is False:
        print('exiting program!')
        sys.exit()
    # speedj example
    print('Testing speedj')
    start = time.time()
    for n, waypoint in enumerate(waypoints_qvel):
        real_env.step({
            'speedj': {'qd': waypoint, 'a': 5, 't': 2/real_env.rate._freq, 'wait': False},
            'move_gripper_position': {'g' : 255, 'wait' : False} # close
        })
        print('action %d sent!'%(n))
    real_env.step({'stopj': {'a': 5}})
    finish = time.time()
    print('done! (elapsed time: %.3f [s])'%(finish - start))
    time.sleep(5)
    curr_qpos = real_env._nparray_to_dict(real_env._get_obs())['qpos']
    print('current - goal qpos is %s deg'%(np.rad2deg(curr_qpos - goal_qpos)))
    time.sleep(5)
    print('Moving to initial position...')
    real_env.step({'movej': {'q': init_qpos}})
    print('done!')
    
    # open-close-open gripper
    # print('open')
    # real_env.step({'open_gripper': {}})
    # time.sleep(3.0)
    # print('close')
    # real_env.step({'close_gripper': {}})
    # time.sleep(3.0)
    # print('open')
    # real_env.step({'open_gripper': {}})
    # time.sleep(5.0)

    

def sanity_check(host_ip):
    from gym_custom.envs.real.ur.drivers import URBasic

    gripper_kwargs = {
        'robot' : None,
        'payload' : 0.85,
        'speed' : 255, # 0~255
        'force' : 255,  # 0~255
        'socket_host' : host_ip,
        'socket_name' : 'gripper_socket'
    }
    robotModel = URBasic.robotModel.RobotModel()
    robot = URBasic.urScriptExt.UrScriptExt(host=host_ip, robotModel=robotModel, **gripper_kwargs)
    robot.reset_error()

    qpos = robot.get_actual_joint_positions()
    init_theta = qpos
    qpos = robot.get_actual_joint_positions()
    qpos[-1] += np.pi/2
    goal_theta = qpos
    
    # init_theta = [-91.13*np.pi/180, -92.48*np.pi/180, -89.77*np.pi/180, -12.91*np.pi/180, 83.09*np.pi/180, 318.61*np.pi/180]
    # goal_theta = [-85.33*np.pi/180, -149.59*np.pi/180, -22.44*np.pi/180, -18.6*np.pi/180, 83.4*np.pi/180, 318.61*np.pi/180]
    
    print('initial joint position (deg):')
    print(goal_theta)
    query = 'movej to goal joint position?'
    response = prompt_yes_or_no(query)
    if response is False:
        robot.close()
        time.sleep(1)
        print('exiting program!')
        sys.exit()
    # robot.movej(q=goal_theta, a=0.3, v=0.3)
    robot.speedj(qd=[0, 0, 0, 0, 0, 0.25], a=1.5, t=4, wait=False)
    time.sleep(2.0)
    print('done!')
    curr_pos = np.array(robot.get_actual_joint_positions())
    print('curr_pos = get_actual_joint_positions() # %s deg' %(str(np.rad2deg(curr_pos))) )
    time.sleep(1)

    robot.operate_gripper(255) #close
    time.sleep(1)
    # gripper_pos = robot.get_gripper_position()
    # print('Gripper position : {}'.format(gripper_pos))
    
    query = 'movej to initial joint position?'
    response = prompt_yes_or_no(query)
    print(init_theta)
    if response is False:
        robot.close()
        time.sleep(1)
        print('exiting program!')
        sys.exit()
    robot.movej(q=init_theta, a=0.3, v=1)
    print('done!')
    curr_pos = np.array(robot.get_actual_joint_positions())
    print('curr_pos = get_actual_joint_positions() # %s deg' %(str(np.rad2deg(curr_pos))) )
    time.sleep(1)
    
    robot.operate_gripper(0) #open
    time.sleep(1)

    robot.close()

def gripper_check(host_ip):
    from gym_custom.envs.real.ur.drivers import URBasic

    gripper_kwargs = {
        'robot' : None,
        'payload' : 0.85,
        'speed' : 255, # 0~255
        'force' : 255,  # 0~255
        'socket_host' : host_ip,
        'socket_name' : 'gripper_socket'
    }
    robotModel = URBasic.robotModel.RobotModel()
    robot = URBasic.urScriptExt.UrScriptExt(host=host_ip, robotModel=robotModel, **gripper_kwargs)
    robot.reset_error()
    
    output_bit_register = robotModel.OutputBitRegister()
    output_int_register = robotModel.OutputIntRegister()
    output_double_register = robotModel.OutputDoubleRegister()
    
    
    def print_register():
        print('output bit register : ', robotModel.OutputBitRegister())
        print('output int register : ', robotModel.OutputIntRegister())
        print('output double register : ', robotModel.OutputDoubleRegister())
        print('tool analog input0 : ', robotModel.ToolAnalogInput0())
        print('tool analog input1 : ', robotModel.ToolAnalogInput1())
        

    option=2
    if option==1: #jgkim ver
        # close-open-close-open gripper
        print('closing gripper')
        robot.operate_gripper(255)
        print_register()
        time.sleep(3)
        print('opening gripper')
        robot.operate_gripper(0)
        print_register()
        time.sleep(3)
        print('closing gripper')
        robot.operate_gripper(255)
        print_register()
        time.sleep(3)
        print('opening gripper')
        robot.operate_gripper(0)
        print_register()
        time.sleep(3)

        print('done')
        robot.close()
    elif option==2:
        wait = True
        
        # print('test gripper reset')
        # robot.test_gripper_reset()
        # time.sleep(3)

        print('test open gripper')
        robot.move_gripper(10, wait=wait)
        
        time.sleep(3)
        grip_pos = robot.get_gripper_position()
        print('grip pos : ', grip_pos)

        print('test close gripper')
        robot.move_gripper(150, wait=wait)
        
        time.sleep(3)
        grip_pos = robot.get_gripper_position()
        print('grip pos : ', grip_pos)

        robot.close()

if __name__ == "__main__":
    # sanity_check(host_ip='192.168.5.101')
    # gripper_check(host_ip='192.168.5.101')
    servoj_speedj_example(host_ip='192.168.5.101', rate=20)
    pass