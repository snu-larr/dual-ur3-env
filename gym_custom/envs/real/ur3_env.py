import numpy as np
import time
import sys
import warnings

import gym_custom
from gym_custom.envs.real.ur.interface import URScriptInterface, convert_action_to_space, convert_observation_to_space, COMMAND_LIMITS
from gym_custom.envs.real.utils import ROSRate, prompt_yes_or_no

class UR3RealEnv(gym_custom.Env):
    
    def __init__(self, host_ip, rate):
        self.host_ip = host_ip
        self.rate = ROSRate(rate)
        self.interface = URScriptInterface(host_ip)

        # UR3 (6DOF), 2F-85 gripper (1DOF)
        self._init_qpos = np.zeros([6])
        self._init_qvel = np.zeros([6])
        self._init_gripperpos = np.zeros([1])
        self._init_grippervel = np.zeros([1])

        self.action_space =  self._set_action_space()
        obs = self._get_obs()
        self.observation_space = self._set_observation_space(obs)

        self._episode_step = None

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
    goal_qpos[-1] += np.pi/2*1.5
    waypoints_qpos = np.linspace(init_qpos, goal_qpos, rate*2, axis=0)
    waypoints_qvel = np.diff(waypoints_qpos, axis=0)*real_env.rate._freq
    
    # close-open-close gripper
    print('close')
    real_env.step({'close_gripper': {}})
    time.sleep(3.0)
    print('open')
    real_env.step({'open_gripper': {}})
    time.sleep(3.0)
    print('close')
    real_env.step({'close_gripper': {}})
    time.sleep(5.0)
    
    if prompt_yes_or_no('servoj to %s deg?'%(np.rad2deg(goal_qpos))) is False:
        print('exiting program!')
        sys.exit()
    # servoj example
    print('Testing servoj')
    start = time.time()
    for n, waypoint in enumerate(waypoints_qpos[1:,:]):
        real_env.step({
            'servoj': {'q': waypoint, 't': 2/real_env.rate._freq, 'wait': False},
            # 'close_gripper': {}
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
            # 'close_gripper': {}
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
    print('open')
    real_env.step({'open_gripper': {}})
    time.sleep(3.0)
    print('close')
    real_env.step({'close_gripper': {}})
    time.sleep(3.0)
    print('open')
    real_env.step({'open_gripper': {}})
    time.sleep(5.0)

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

    # close-open-close-open gripper
    print('closing gripper')
    robot.operate_gripper(255)
    time.sleep(3)
    print('opening gripper')
    robot.operate_gripper(0)
    time.sleep(3)
    print('closing gripper')
    robot.operate_gripper(255)
    time.sleep(3)
    print('opening gripper')
    robot.operate_gripper(0)
    time.sleep(3)

    print('done')
    robot.close()

if __name__ == "__main__":
    # sanity_check(host_ip='192.168.5.101')
    # gripper_check(host_ip='192.168.5.101')
    # servoj_speedj_example(host_ip='192.168.5.101', rate=25)
    pass