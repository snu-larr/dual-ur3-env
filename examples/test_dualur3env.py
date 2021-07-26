import argparse
import gym
import numpy as np
import time
import sys

import gym_custom

from gym_custom.envs.custom.ur_utils import NullObjectiveBase
from gym_custom.envs.custom.ur_utils import URScriptWrapper_DualUR3_deprecated as URScriptWrapper_deprecated
from gym_custom.envs.custom.ur_utils import URScriptWrapper_DualUR3 as URScriptWrapper

from gym_custom.envs.real.utils import ROSRate, prompt_yes_or_no

class NoConstraint(NullObjectiveBase):

    def __init__(self):
        pass

    def _evaluate(self, SO3):
        return 0.0

class UprightConstraint(NullObjectiveBase):
    
    def __init__(self):
        pass

    def _evaluate(self, SO3):
        axis_des = np.array([0, 0, -1])
        axis_curr = SO3[:,2]
        return 1.0 - np.dot(axis_curr, axis_des)

def show_dual_ur3():
    env = gym_custom.make('dual-ur3-larr-v0')
    obs = env.reset()
    dt = env.dt

    while True:
        env.render()
        time.sleep(dt)

def run_dual_ur3():
    env = gym_custom.make('dual-ur3-larr-v0')
    obs = env.reset()
    dt = env.dt

    for t in range(int(60/dt)):
        action = env.action_space.sample()
        action = np.zeros_like(action)
        obs, _, _, _ = env.step(action)
        # env.render()
        print('time: %.2f'%(t*dt))
        print('  joint_pos: %s (rad)'%(env._get_ur3_qpos()*180/np.pi))
        print('  joint_vel: %s (rad/s)'%(env._get_ur3_qvel()*180/np.pi))
        print('  joint_bias: %s (Nm)'%(env._get_ur3_bias()*180/np.pi))
        print('  gripper_pos: %s (m)'%(env._get_gripper_qpos()[[2,7,12,17]]))
        print('  gripper_vel: %s (m/s)'%(env._get_gripper_qvel()[[2,7,12,17]]))
        print('  gripper_bias: %s (N)'%(env._get_gripper_bias()[[2,7,12,17]]))
        time.sleep(dt)

def test_fkine_ikine():
    env = gym_custom.make('dual-ur3-larr-v0')
    obs = env.reset()
    dt = env.dt

    # test forward kinematics
    q = env._get_ur3_qpos()[:env.ur3_nqpos] # right
    Rs, ps, Ts = env.forward_kinematics_DH(q, arm='right')
    R_base, p_base, T_base = env.get_body_se3(body_name='right_arm_rotz')
    # R_hand, p_hand, T_hand = env.get_body_se3(body_name='right_gripper:hand')
    R_hand, p_hand, T_hand = env.get_body_se3(body_name='right_gripper:hand')
    print('base:')
    print('  pos: (DH) %s vs. (MjData) %s'%(ps[0,:], p_base))
    print('  rotMat: (DH) \n%s \nvs. \n  rotMat: (MjData) \n%s'%(Rs[0,:,:], R_base))
    print('hand:')
    print('  pos: (DH) %s vs. (MjData) %s'%(ps[-1,:], p_hand))
    print('  rotMat: (DH) \n%s \nvs. \n  rotMat: (MjData) \n%s'%(Rs[-1,:,:], R_hand))

    null_obj_func = UprightConstraint()

    ee_pos_right = np.array([0.1, -0.5, 0.9])
    ee_pos_left = np.array([-0.1, -0.5, 0.9])
    q_right_des, iter_taken_right, err_right, null_obj_right = env.inverse_kinematics_ee(ee_pos_right, null_obj_func, arm='right')
    q_left_des, iter_taken_left, err_left, null_obj_left = env.inverse_kinematics_ee(ee_pos_left, null_obj_func, arm='left')

    qpos_des = env.init_qpos.copy()
    qpos_des[0:env.ur3_nqpos] = q_right_des
    qpos_des[env.ur3_nqpos+env.gripper_nqpos:2*env.ur3_nqpos+env.gripper_nqpos] = q_left_des
    env.render()
    time.sleep(3.0)
    while True:
        env.set_state(qpos_des, env.init_qvel)
        env.render()

def servoj_and_forceg_deprecated():

    env = gym_custom.make('dual-ur3-larr-v0')
    obs = env.reset()
    dt = env.dt

    null_obj_func = UprightConstraint()

    ee_pos_right = np.array([0.1, -0.5, 0.9])
    ee_pos_left = np.array([-0.1, -0.5, 0.9])
    q_right_des, iter_taken_right, err_right, null_obj_right = env.inverse_kinematics_ee(ee_pos_right, null_obj_func, arm='right')
    q_left_des, iter_taken_left, err_left, null_obj_left = env.inverse_kinematics_ee(ee_pos_left, null_obj_func, arm='left')

    PID_gains = {'P': 1.0, 'I': 0.5, 'D': 0.2}
    ur3_scale_factor = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0])*np.array([1.0, 1.0, 1.0, 4.0, 4.0, 1.0])
    gripper_scale_factor = np.array([1.0, 1.0])
    env = URScriptWrapper_deprecated(env, PID_gains, ur3_scale_factor, gripper_scale_factor)
    
    t = 0
    qpos_err, qvel = np.inf, np.inf
    while qpos_err > 1e-1*np.pi/180.0 or qvel > 1e-1*np.pi/180.0:
        command = {
            'ur3': {'type': 'servoj', 'command': np.concatenate([q_right_des, q_left_des])},
            'gripper': {'type': 'forceg', 'command': np.array([1.0, 1.0])}
        }
        obs, _, _, _ = env.step(command)
        env.render()
        joint_angles = env.env._get_ur3_qpos()
        right_err = np.linalg.norm(joint_angles[:env.ur3_nqpos] - q_right_des)
        left_err = np.linalg.norm(joint_angles[-env.ur3_nqpos:] - q_left_des)
        print('time: %f [s]'%(t*dt))
        print('right arm joint error [deg]: %f'%(right_err*180.0/np.pi))
        print('left arm joint error [deg]: %f'%(left_err*180.0/np.pi))
        qpos_err = max(right_err, left_err)
        qvel = np.linalg.norm(env.env._get_ur3_qvel())
        t += 1
    time.sleep(100)

def servoj_and_forceg(env_type='sim', render=False):
    list_of_env_types = ['sim', 'real']
    
    if env_type == list_of_env_types[0]:
        env = gym_custom.make('dual-ur3-larr-v0')
        servoj_args = {'t': None, 'wait': None}
    elif env_type == list_of_env_types[1]:
        env = gym_custom.make('dual-ur3-larr-real-v0',
            host_ip_right='192.168.5.102',
            host_ip_left='192.168.5.101',
            rate=25
        )
        servoj_args = {'t': 2/env.rate._freq, 'wait': False}
        # 1. Set initial as current configuration
        env.set_initial_joint_pos('current')
        env.set_initial_gripper_pos('current')
        # 2. Set inital as default configuration
        env.set_initial_joint_pos(np.deg2rad([90, -45, 135, -180, 45, 0, -90, -135, -135, 0, -45, 0]))
        env.set_initial_gripper_pos(np.array([0.0, 0.0]))
        assert render is False
    else: raise ValueError('Invalid env_type! Availiable options are %s'%(list_of_env_types))
    obs = env.reset()
    dt = env.dt

    null_obj_func = UprightConstraint()

    ee_pos_right = np.array([0.1, -0.5, 0.9])
    ee_pos_left = np.array([-0.1, -0.5, 0.9])
    q_right_des, iter_taken_right, err_right, null_obj_right = env.inverse_kinematics_ee(ee_pos_right, null_obj_func, arm='right')
    q_left_des, iter_taken_left, err_left, null_obj_left = env.inverse_kinematics_ee(ee_pos_left, null_obj_func, arm='left')

    if env_type == list_of_env_types[0]:
        PID_gains = {'servoj': {'P': 1.0, 'I': 0.5, 'D': 0.2}}
        ur3_scale_factor = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0])*np.array([1.0, 1.0, 1.0, 4.0, 4.0, 1.0])
        gripper_scale_factor = np.array([1.0])
        env = URScriptWrapper(env, PID_gains, ur3_scale_factor, gripper_scale_factor)
    elif env_type == list_of_env_types[1]:
        env.env = env

    if env_type == list_of_env_types[1]:
        if prompt_yes_or_no('current qpos is \r\n right: %s deg\r\n left: %s deg\r\n?'
            %(np.rad2deg(env.env._init_qpos[:6]), np.rad2deg(env.env._init_qpos[6:]))) is False:
            print('exiting program!')
            env.close()
            sys.exit()

    t = 0
    qpos_err, qvel = np.inf, np.inf
    while qpos_err > np.deg2rad(1e-1) or qvel > np.deg2rad(1e0):
        ob, _, _, _ = env.step({
            'right': {
                'servoj': {'q': q_right_des, 't': servoj_args['t'], 'wait': servoj_args['wait']},
                'move_gripper_force': {'gf': np.array([1.0])}
            },
            'left': {
                'servoj': {'q': q_left_des, 't': servoj_args['t'], 'wait': servoj_args['wait']},
                'move_gripper_force': {'gf': np.array([1.0])}
            }
        })
        if render: env.render()
        obs_dict = env.env.get_obs_dict()
        right_err = np.linalg.norm(obs_dict['right']['qpos'] - q_right_des)
        left_err = np.linalg.norm(obs_dict['left']['qpos'] - q_left_des)
        print('time: %f [s]'%(t*dt))
        print('right arm joint error [deg]: %f'%(np.rad2deg(right_err)))
        print('left arm joint error [deg]: %f'%(np.rad2deg(left_err)))
        qpos_err = max(right_err, left_err)
        qvel = np.linalg.norm(np.concatenate([obs_dict['right']['qvel'], obs_dict['left']['qvel']]))
        t += 1
    
    if env_type == list_of_env_types[0]:
        time.sleep(100)
    else:
        env.close()
        sys.exit()

def speedj_and_forceg_deprecated():

    env = gym_custom.make('dual-ur3-larr-v0')
    obs = env.reset()
    dt = env.dt

    null_obj_func = UprightConstraint()

    ee_pos_right = np.array([0.1, -0.5, 0.9])
    ee_pos_left = np.array([-0.1, -0.5, 0.9])
    q_right_des, iter_taken_right, err_right, null_obj_right = env.inverse_kinematics_ee(ee_pos_right, null_obj_func, arm='right')
    q_left_des, iter_taken_left, err_left, null_obj_left = env.inverse_kinematics_ee(ee_pos_left, null_obj_func, arm='left')

    PI_gains = {'P': 0.20, 'I': 10.0}
    ur3_scale_factor = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0])*np.array([1.0, 1.0, 1.0, 2.5, 2.5, 2.5])
    gripper_scale_factor = np.array([1.0, 1.0])
    env = URScriptWrapper_deprecated(env, PI_gains, ur3_scale_factor, gripper_scale_factor)

    q_init = env.env._get_ur3_qpos()
    q_vel_des = (np.concatenate([q_right_des, q_left_des]) - q_init)/3.0
    for t in range(int(3.0/dt)):
        command = {
            'ur3': {'type': 'speedj', 'command': q_vel_des},
            'gripper': {'type': 'forceg', 'command': np.array([1.0, 1.0])}
        }
        obs, _, _, _ = env.step(command)
        env.render()
        joint_angles = env.env._get_ur3_qpos()
        right_pos_err = np.linalg.norm(joint_angles[:env.ur3_nqpos] - q_right_des)
        left_pos_err = np.linalg.norm(joint_angles[-env.ur3_nqpos:] - q_left_des)
        right_vel_err = np.linalg.norm(env.env._get_ur3_qvel()[:env.ur3_nqpos] - q_vel_des[:env.ur3_nqpos])
        left_vel_err = np.linalg.norm(env.env._get_ur3_qvel()[-env.ur3_nqpos:] - q_vel_des[-env.ur3_nqpos:])
        print('time: %f [s]'%(t*dt))
        print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(right_pos_err*180.0/np.pi, right_vel_err*180.0/np.pi))
        print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(left_pos_err*180.0/np.pi, left_vel_err*180.0/np.pi))
    
    t = 0.0
    qvel = np.inf
    q_vel_des = np.zeros_like(env.env._get_ur3_qvel())
    while qvel > 1e-1*np.pi/180.0:
        command = {
            'ur3': {'type': 'speedj', 'command': q_vel_des},
            'gripper': {'type': 'forceg', 'command': np.array([-1.0, -1.0])}
        }
        obs, _, _, _ = env.step(command)
        env.render()
        joint_angles = env.env._get_ur3_qpos()
        right_pos_err = np.linalg.norm(joint_angles[:env.ur3_nqpos] - q_right_des)
        left_pos_err = np.linalg.norm(joint_angles[-env.ur3_nqpos:] - q_left_des)
        right_vel_err = np.linalg.norm(env.env._get_ur3_qvel()[:env.ur3_nqpos] - q_vel_des[:env.ur3_nqpos])
        left_vel_err = np.linalg.norm(env.env._get_ur3_qvel()[-env.ur3_nqpos:] - q_vel_des[-env.ur3_nqpos:])
        print('time: %f [s]'%(t*dt))
        print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(right_pos_err*180.0/np.pi, right_vel_err*180.0/np.pi))
        print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(left_pos_err*180.0/np.pi, left_vel_err*180.0/np.pi))
        qvel = np.linalg.norm(env.env._get_ur3_qvel() - q_vel_des)
        t += 1
    time.sleep(100)

def speedj_and_forceg(env_type='sim', render=False):
    list_of_env_types = ['sim', 'real']
    
    if env_type == list_of_env_types[0]:
        env = gym_custom.make('dual-ur3-larr-v0')
        speedj_args = {'a': 5, 't': None, 'wait': None}
    elif env_type == list_of_env_types[1]:
        env = gym_custom.make('dual-ur3-larr-real-v0',
            host_ip_right='192.168.5.102',
            host_ip_left='192.168.5.101',
            rate=25
        )
        speedj_args = {'a': 5, 't': 2/env.rate._freq, 'wait': False}
        env.set_initial_joint_pos('current')
        env.set_initial_gripper_pos('current')
        # 2. Set inital as default configuration
        env.set_initial_joint_pos(np.deg2rad([90, -45, 135, -180, 45, 0, -90, -135, -135, 0, -45, 0]))
        env.set_initial_gripper_pos(np.array([0.0, 0.0]))
        assert render is False
    else: raise ValueError('Invalid env_type! Availiable options are %s'%(list_of_env_types))
    obs = env.reset()
    dt = env.dt

    null_obj_func = UprightConstraint()

    ee_pos_right = np.array([0.1, -0.5, 0.9])
    ee_pos_left = np.array([-0.1, -0.5, 0.9])
    q_right_des, iter_taken_right, err_right, null_obj_right = env.inverse_kinematics_ee(ee_pos_right, null_obj_func, arm='right')
    q_left_des, iter_taken_left, err_left, null_obj_left = env.inverse_kinematics_ee(ee_pos_left, null_obj_func, arm='left')

    if env_type == list_of_env_types[0]:
        PI_gains = {'speedj': {'P': 0.2, 'I': 10.0}} # was 0.2, 10.0
        ur3_scale_factor = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0])*np.array([1.0, 1.0, 1.0, 2.5, 2.5, 2.5])
        gripper_scale_factor = np.array([1.0])
        env = URScriptWrapper(env, PI_gains, ur3_scale_factor, gripper_scale_factor)
    elif env_type == list_of_env_types[1]:
        env.env = env

    if env_type == list_of_env_types[1]:
        if prompt_yes_or_no('current qpos is \r\n right: %s deg\r\n left: %s deg\r\n?'
            %(np.rad2deg(env.env._init_qpos[:6]), np.rad2deg(env.env._init_qpos[6:]))) is False:
            print('exiting program!')
            env.close()
            sys.exit()

    # Move to goal
    duration = 3.0 # in seconds
    obs_dict_current = env.env.get_obs_dict()
    q_right_des_vel = (q_right_des - obs_dict_current['right']['qpos'])/duration
    q_left_des_vel = (q_left_des - obs_dict_current['left']['qpos'])/duration
    start = time.time()
    for t in range(int(duration/dt)):
        obs, _, _, _ = env.step({
            'right': {
                'speedj': {'qd': q_right_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([1.0])}
            },
            'left': {
                'speedj': {'qd': q_left_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([1.0])}
            }
        })
        if render: env.render()
        # TODO: get_obs_dict() takes a long time causing timing issues.
        #   Is it due to Upboard's lackluster performance or some deeper
        #   issues within UR Script wrppaer?
        # obs_dict = env.env.get_obs_dict()
        # right_pos_err = np.linalg.norm(obs_dict['right']['qpos'] - q_right_des)
        # left_pos_err = np.linalg.norm(obs_dict['left']['qpos'] - q_left_des)
        # right_vel_err = np.linalg.norm(obs_dict['right']['qvel'] - q_right_des_vel)
        # left_vel_err = np.linalg.norm(obs_dict['left']['qvel'] - q_left_des_vel)
        print('time: %f [s]'%(t*dt))
        # print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(right_pos_err), np.rad2deg(right_vel_err)))
        # print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(left_pos_err), np.rad2deg(left_vel_err)))
    finish = time.time()

    # Stop
    t = 0
    qvel_err = np.inf
    q_right_des_vel, q_left_des_vel = np.zeros([env.ur3_nqpos]), np.zeros([env.ur3_nqpos])
    while qvel_err > np.deg2rad(1e0):
        obs, _, _, _ = env.step({
            'right': {
                'speedj': {'qd': q_right_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([-1.0])}
            },
            'left': {
                'speedj': {'qd': q_left_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([-1.0])}
            }
        })
        if render: env.render()
        obs_dict = env.env.get_obs_dict()
        right_pos_err = np.linalg.norm(obs_dict['right']['qpos'] - q_right_des)
        left_pos_err = np.linalg.norm(obs_dict['left']['qpos'] - q_left_des)
        right_vel_err = np.linalg.norm(obs_dict['right']['qvel'] - q_right_des_vel)
        left_vel_err = np.linalg.norm(obs_dict['left']['qvel'] - q_left_des_vel)
        print('time: %f [s]'%(t*dt))
        print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(right_pos_err), np.rad2deg(right_vel_err)))
        print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(left_pos_err), np.rad2deg(left_vel_err)))
        qvel_err = np.linalg.norm(np.concatenate([obs_dict['right']['qvel'], obs_dict['left']['qvel']]) - np.concatenate([q_right_des_vel, q_left_des_vel]))
        t += 1
    
    if env_type == list_of_env_types[0]:
        time.sleep(100)
    else:
        env.close()
        print('%.3f seconds'%(finish-start))
        sys.exit()

def pick_and_place_deprecated():

    env = gym_custom.make('dual-ur3-larr-v0')
    obs = env.reset()
    dt = env.dt

    null_obj_func = UprightConstraint()

    PI_gains = {'P': 0.20, 'I': 5.0}
    ur3_scale_factor = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0])*np.array([1.0, 1.0, 1.0, 2.5, 2.5, 2.5])
    gripper_scale_factor = np.array([1.0, 1.0])
    env = URScriptWrapper_deprecated(env, PI_gains, ur3_scale_factor, gripper_scale_factor)

    # Move to position
    q_init = env.env._get_ur3_qpos()
    ee_pos_right = np.array([0.0, -0.4, 0.9])
    ee_pos_left = np.array([-0.3, -0.5, 0.9])
    q_right_des, iter_taken_right, err_right, null_obj_right = env.env.inverse_kinematics_ee(ee_pos_right, null_obj_func, arm='right')
    q_left_des, iter_taken_left, err_left, null_obj_left = env.env.inverse_kinematics_ee(ee_pos_left, null_obj_func, arm='left')
    q_vel_des = (np.concatenate([q_right_des, q_left_des]) - q_init)/5.0
    for t in range(int(5.0/dt)):
        command = {
            'ur3': {'type': 'speedj', 'command': q_vel_des},
            'gripper': {'type': 'forceg', 'command': np.array([10.0, 10.0])}
        }
        obs, _, _, _ = env.step(command)
        env.render()
        joint_angles = env.env._get_ur3_qpos()
        right_pos_err = np.linalg.norm(joint_angles[:env.ur3_nqpos] - q_right_des)
        left_pos_err = np.linalg.norm(joint_angles[-env.ur3_nqpos:] - q_left_des)
        right_vel_err = np.linalg.norm(env.env._get_ur3_qvel()[:env.ur3_nqpos] - q_vel_des[:env.ur3_nqpos])
        left_vel_err = np.linalg.norm(env.env._get_ur3_qvel()[-env.ur3_nqpos:] - q_vel_des[-env.ur3_nqpos:])
        print('time: %f [s]'%(t*dt))
        print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(right_pos_err*180.0/np.pi, right_vel_err*180.0/np.pi))
        print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(left_pos_err*180.0/np.pi, left_vel_err*180.0/np.pi))

    # Open right gripper
    q_vel_des = np.zeros_like(q_vel_des)
    for t in range(int(5.0/dt)):
        command = {
            'ur3': {'type': 'speedj', 'command': q_vel_des},
            'gripper': {'type': 'forceg', 'command': np.array([-10.0, 10.0])}
        }
        obs, _, _, _ = env.step(command)
        env.render()
        joint_angles = env.env._get_ur3_qpos()
        right_pos_err = np.linalg.norm(joint_angles[:env.ur3_nqpos] - q_right_des)
        left_pos_err = np.linalg.norm(joint_angles[-env.ur3_nqpos:] - q_left_des)
        right_vel_err = np.linalg.norm(env.env._get_ur3_qvel()[:env.ur3_nqpos] - q_vel_des[:env.ur3_nqpos])
        left_vel_err = np.linalg.norm(env.env._get_ur3_qvel()[-env.ur3_nqpos:] - q_vel_des[-env.ur3_nqpos:])
        print('time: %f [s]'%(t*dt))
        print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(right_pos_err*180.0/np.pi, right_vel_err*180.0/np.pi))
        print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(left_pos_err*180.0/np.pi, left_vel_err*180.0/np.pi))

    # Place right gripper
    q_init = env.env._get_ur3_qpos()
    ee_pos_right = np.array([0.0, -0.4, 0.78])
    ee_pos_left = np.array([-0.3, -0.5, 0.9])
    q_right_des, iter_taken_right, err_right, null_obj_right = env.env.inverse_kinematics_ee(ee_pos_right, null_obj_func, arm='right')
    q_left_des, iter_taken_left, err_left, null_obj_left = env.env.inverse_kinematics_ee(ee_pos_left, null_obj_func, arm='left')
    q_vel_des = (np.concatenate([q_right_des, q_left_des]) - q_init)/5.0
    for t in range(int(5.0/dt)):
        command = {
            'ur3': {'type': 'speedj', 'command': q_vel_des},
            'gripper': {'type': 'forceg', 'command': np.array([-10.0, 10.0])}
        }
        obs, _, _, _ = env.step(command)
        env.render()
        joint_angles = env.env._get_ur3_qpos()
        right_pos_err = np.linalg.norm(joint_angles[:env.ur3_nqpos] - q_right_des)
        left_pos_err = np.linalg.norm(joint_angles[-env.ur3_nqpos:] - q_left_des)
        right_vel_err = np.linalg.norm(env.env._get_ur3_qvel()[:env.ur3_nqpos] - q_vel_des[:env.ur3_nqpos])
        left_vel_err = np.linalg.norm(env.env._get_ur3_qvel()[-env.ur3_nqpos:] - q_vel_des[-env.ur3_nqpos:])
        print('time: %f [s]'%(t*dt))
        print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(right_pos_err*180.0/np.pi, right_vel_err*180.0/np.pi))
        print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(left_pos_err*180.0/np.pi, left_vel_err*180.0/np.pi))
    env.PID_gains = {'P': 1.0, 'I': 0.5, 'D': 0.2}
    qpos_err, qvel = np.inf, np.inf
    while qpos_err > 1e-1*np.pi/180.0 or qvel > 1e-1*np.pi/180.0:
        command = {
            'ur3': {'type': 'servoj', 'command': np.concatenate([q_right_des, q_left_des])},
            'gripper': {'type': 'forceg', 'command': np.array([-10.0, 10.0])}
        }
        obs, _, _, _ = env.step(command)
        env.render()
        joint_angles = env.env._get_ur3_qpos()
        right_err = np.linalg.norm(joint_angles[:env.ur3_nqpos] - q_right_des)
        left_err = np.linalg.norm(joint_angles[-env.ur3_nqpos:] - q_left_des)
        print('time: %f [s]'%(t*dt))
        print('right arm joint error [deg]: %f'%(right_err*180.0/np.pi))
        print('left arm joint error [deg]: %f'%(left_err*180.0/np.pi))
        qpos_err = max(right_err, left_err)
        qvel = np.linalg.norm(env.env._get_ur3_qvel())
        t += 1

    # Grip
    for t in range(int(5.0/dt)):
        command = {
            'ur3': {'type': 'servoj', 'command': np.concatenate([q_right_des, q_left_des])},
            'gripper': {'type': 'forceg', 'command': np.array([10.0, 10.0])}
        }
        obs, _, _, _ = env.step(command)
        env.render()
        joint_angles = env.env._get_ur3_qpos()
        right_err = np.linalg.norm(joint_angles[:env.ur3_nqpos] - q_right_des)
        left_err = np.linalg.norm(joint_angles[-env.ur3_nqpos:] - q_left_des)
        print('time: %f [s]'%(t*dt))
        print('right arm joint error [deg]: %f'%(right_err*180.0/np.pi))
        print('left arm joint error [deg]: %f'%(left_err*180.0/np.pi))

    # Lift right gripper
    env.PID_gains = {'P': 0.20, 'I': 5.0}
    q_init = env.env._get_ur3_qpos()
    ee_pos_right = np.array([0.3, -0.5, 0.9])
    ee_pos_left = np.array([-0.3, -0.5, 0.9])
    q_right_des, iter_taken_right, err_right, null_obj_right = env.env.inverse_kinematics_ee(ee_pos_right, null_obj_func, arm='right')
    q_left_des, iter_taken_left, err_left, null_obj_left = env.env.inverse_kinematics_ee(ee_pos_left, null_obj_func, arm='left')
    q_vel_des = (np.concatenate([q_right_des, q_left_des]) - q_init)/5.0
    for t in range(int(5.0/dt)):
        command = {
            'ur3': {'type': 'speedj', 'command': q_vel_des},
            'gripper': {'type': 'forceg', 'command': np.array([10.0, 10.0])}
        }
        obs, _, _, _ = env.step(command)
        env.render()
        joint_angles = env.env._get_ur3_qpos()
        right_pos_err = np.linalg.norm(joint_angles[:env.ur3_nqpos] - q_right_des)
        left_pos_err = np.linalg.norm(joint_angles[-env.ur3_nqpos:] - q_left_des)
        right_vel_err = np.linalg.norm(env.env._get_ur3_qvel()[:env.ur3_nqpos] - q_vel_des[:env.ur3_nqpos])
        left_vel_err = np.linalg.norm(env.env._get_ur3_qvel()[-env.ur3_nqpos:] - q_vel_des[-env.ur3_nqpos:])
        right_actuator_torque = np.linalg.norm(env.env._get_ur3_actuator()[:env.ur3_nqpos])
        left_actuator_torque = np.linalg.norm(env.env._get_ur3_actuator()[-env.ur3_nqpos:])
        right_bias_torque = np.linalg.norm(env.env._get_ur3_bias()[:env.ur3_nqpos])
        left_bias_torque = np.linalg.norm(env.env._get_ur3_bias()[-env.ur3_nqpos:])
        right_constraint_torque = np.linalg.norm(env.env._get_ur3_constraint()[:env.ur3_nqpos])
        left_constraint_torque = np.linalg.norm(env.env._get_ur3_constraint()[-env.ur3_nqpos:])
        print('time: %f [s]'%(t*dt))
        print('right arm joint pos error [deg]: %f vel error [dps]: %f actuator torque [Nm]: %f bias torque [Nm]: %f constraint torque [Nm]: %f'
            %(right_pos_err*180.0/np.pi, right_vel_err*180.0/np.pi, right_actuator_torque, right_bias_torque, right_constraint_torque))
        print('left arm joint pos error [deg]: %f vel error [dps]: %f actuator torque [Nm]: %f bias torque [Nm]: %f constraint torque [Nm]: %f'
            %(left_pos_err*180.0/np.pi, left_vel_err*180.0/np.pi, left_actuator_torque, left_bias_torque, left_constraint_torque))
    env.PID_gains = {'P': 1.0, 'I': 2.5, 'D': 0.2}
    qpos_err, qvel = np.inf, np.inf
    while qpos_err > 1e-1*np.pi/180.0 or qvel > 1e-1*np.pi/180.0:
        command = {
            'ur3': {'type': 'servoj', 'command': np.concatenate([q_right_des, q_left_des])},
            'gripper': {'type': 'forceg', 'command': np.array([10.0, 10.0])}
        }
        obs, _, _, _ = env.step(command)
        env.render()
        joint_angles = env.env._get_ur3_qpos()
        right_err = np.linalg.norm(joint_angles[:env.ur3_nqpos] - q_right_des)
        left_err = np.linalg.norm(joint_angles[-env.ur3_nqpos:] - q_left_des)
        qpos_err = max(right_err, left_err)
        qvel = np.linalg.norm(env.env._get_ur3_qvel())
        print('time: %f [s]'%(t*dt))
        print('right arm joint error [deg]: %f'%(right_err*180.0/np.pi))
        print('left arm joint error [deg]: %f'%(left_err*180.0/np.pi))
        print('joint velocity [dps]: %f'%(qvel*180.0/np.pi))
        t += 1
    
    time.sleep(3.0)

    # Open gripper
    for t in range(int(5.0/dt)):
        command = {
            'ur3': {'type': 'servoj', 'command': np.concatenate([q_right_des, q_left_des])},
            'gripper': {'type': 'forceg', 'command': np.array([-25.0, 10.0])}
        }
        obs, _, _, _ = env.step(command)
        env.render()
        joint_angles = env.env._get_ur3_qpos()
        right_err = np.linalg.norm(joint_angles[:env.ur3_nqpos] - q_right_des)
        left_err = np.linalg.norm(joint_angles[-env.ur3_nqpos:] - q_left_des)
        right_actuator_torque = np.linalg.norm(env.env._get_ur3_actuator()[:env.ur3_nqpos])
        left_actuator_torque = np.linalg.norm(env.env._get_ur3_actuator()[-env.ur3_nqpos:])
        right_bias_torque = np.linalg.norm(env.env._get_ur3_bias()[:env.ur3_nqpos])
        left_bias_torque = np.linalg.norm(env.env._get_ur3_bias()[-env.ur3_nqpos:])
        right_constraint_torque = np.linalg.norm(env.env._get_ur3_constraint()[:env.ur3_nqpos])
        left_constraint_torque = np.linalg.norm(env.env._get_ur3_constraint()[-env.ur3_nqpos:])
        print('time: %f [s]'%(t*dt))
        print('right arm joint error [deg]: %f actuator torque [Nm]: %f bias torque [Nm]: %f constraint torque [Nm]: %f'
            %(right_err*180.0/np.pi, right_actuator_torque, right_bias_torque, right_constraint_torque))
        print('left arm joint error [deg]: %f actuator torque [Nm]: %f bias torque [Nm]: %f constraint torque [Nm]: %f'
            %(left_err*180.0/np.pi, left_actuator_torque, left_bias_torque, left_constraint_torque))
    
    while True:
        env.render()

def pick_and_place(env_type='sim', render=False):
    list_of_env_types = ['sim', 'real']
    
    if env_type == list_of_env_types[0]:
        env = gym_custom.make('dual-ur3-larr-v0')
        servoj_args, speedj_args = {'t': None, 'wait': None}, {'a': 5, 't': None, 'wait': None}
    elif env_type == list_of_env_types[1]:
        env = gym_custom.make('dual-ur3-larr-real-v0',
            host_ip_right='192.168.5.102',
            host_ip_left='192.168.5.101',
            rate=25
        )
        servoj_args, speedj_args = {'t': 2/env.rate._freq, 'wait': False}, {'a': 5, 't': 2/env.rate._freq, 'wait': False}
        # 1. Set initial as current configuration
        env.set_initial_joint_pos('current')
        env.set_initial_gripper_pos('current')
        # 2. Set inital as default configuration
        env.set_initial_joint_pos(np.deg2rad([90, -45, 135, -180, 45, 0, -90, -135, -135, 0, -45, 0]))
        env.set_initial_gripper_pos(np.array([0.0, 0.0]))
        assert render is False
    else: raise ValueError('Invalid env_type! Availiable options are %s'%(list_of_env_types))
    obs = env.reset()
    dt = env.dt

    if env_type == list_of_env_types[0]:
        PID_gains = {'servoj': {'P': 1.0, 'I': 0.5, 'D': 0.2}, 'speedj': {'P': 0.20, 'I':5.0}}
        ur3_scale_factor = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0])*np.array([1.0, 1.0, 1.0, 2.5, 2.5, 2.5])
        gripper_scale_factor = np.array([1.0])
        env = URScriptWrapper(env, PID_gains, ur3_scale_factor, gripper_scale_factor)
    elif env_type == list_of_env_types[1]:
        env.env = env

    null_obj_func = UprightConstraint()

    if env_type == list_of_env_types[1]:
        if prompt_yes_or_no('current qpos is \r\n right: %s deg\r\n left: %s deg\r\n?'
            %(np.rad2deg(env.env._init_qpos[:6]), np.rad2deg(env.env._init_qpos[6:]))) is False:
            print('exiting program!')
            env.close()
            sys.exit()

    print('Moving to position... (step 1 of 6)')
    time.sleep(1.0)
    # 1. Move to initial position
    duration = 5.0
    ee_pos_right = np.array([0.0, -0.4, 0.9])
    ee_pos_left = np.array([-0.3, -0.5, 0.9])
    q_right_des, iter_taken_right, err_right, null_obj_right = env.env.inverse_kinematics_ee(ee_pos_right, null_obj_func, arm='right')
    q_left_des, iter_taken_left, err_left, null_obj_left = env.env.inverse_kinematics_ee(ee_pos_left, null_obj_func, arm='left')
    obs_dict_current = env.env.get_obs_dict()
    q_right_des_vel = (q_right_des - obs_dict_current['right']['qpos'])/duration
    q_left_des_vel = (q_left_des - obs_dict_current['left']['qpos'])/duration
    start = time.time()
    for t in range(int(duration/dt)):
        obs, _, _, _ = env.step({
            'right': {
                'speedj': {'qd': q_right_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([10.0])}
            },
            'left': {
                'speedj': {'qd': q_left_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([10.0])}
            }
        })
        if render: env.render()
        # obs_dict = env.env.get_obs_dict()
        # right_pos_err = np.linalg.norm(obs_dict['right']['qpos'] - q_right_des)
        # left_pos_err = np.linalg.norm(obs_dict['left']['qpos'] - q_left_des)
        # right_vel_err = np.linalg.norm(obs_dict['right']['qvel'] - q_right_des_vel)
        # left_vel_err = np.linalg.norm(obs_dict['left']['qvel'] - q_left_des_vel)
        print('time: %f [s]'%(t*dt))
        # print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(right_pos_err), np.rad2deg(right_vel_err)))
        # print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(left_pos_err), np.rad2deg(left_vel_err)))
    finish = time.time()
    env.step({'right': {'stopj': {'a': speedj_args['a']}}, 'left': {'stopj': {'a': speedj_args['a']}}})
    print('speedj duration: %.3f (actual) vs. %.3f (desired)'%(finish-start, duration))

    print('Opening right gripper... (step 2 of 6)')
    time.sleep(1.0)
    env.step({'right': {'open_gripper': {}}, 'left': {}})
    time.sleep(3.0)
    # 2. Open right gripper
    duration = 1.0
    q_right_des_vel, q_left_des_vel = np.zeros_like(q_right_des_vel), np.zeros_like(q_left_des_vel)
    start = time.time()
    for t in range(int(duration/dt)):
        obs, _, _, _ = env.step({
            'right': {
                'speedj': {'qd': q_right_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([-10.0])}
            },
            'left': {
                'speedj': {'qd': q_left_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([10.0])}
            }
        })
        if render: env.render()
        # obs_dict = env.env.get_obs_dict()
        # right_pos_err = np.linalg.norm(obs_dict['right']['qpos'] - q_right_des)
        # left_pos_err = np.linalg.norm(obs_dict['left']['qpos'] - q_left_des)
        # right_vel_err = np.linalg.norm(obs_dict['right']['qvel'] - q_right_des_vel)
        # left_vel_err = np.linalg.norm(obs_dict['left']['qvel'] - q_left_des_vel)
        print('time: %f [s]'%(t*dt))
        # print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(right_pos_err), np.rad2deg(right_vel_err)))
        # print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(left_pos_err), np.rad2deg(left_vel_err)))
    finish = time.time()
    print('speedj duration: %.3f (actual) vs. %.3f (desired)'%(finish-start, duration))

    print('Placing right gripper... (step 3 of 6)')
    time.sleep(1.0)
    # 3. Place right gripper
    duration = 5.0
    ee_pos_right = np.array([0.0, -0.4, 0.78])
    ee_pos_left = np.array([-0.3, -0.5, 0.9])
    q_right_des, iter_taken_right, err_right, null_obj_right = env.env.inverse_kinematics_ee(ee_pos_right, null_obj_func, arm='right')
    q_left_des, iter_taken_left, err_left, null_obj_left = env.env.inverse_kinematics_ee(ee_pos_left, null_obj_func, arm='left')
    obs_dict_current = env.env.get_obs_dict()
    q_right_des_vel = (q_right_des - obs_dict_current['right']['qpos'])/duration
    q_left_des_vel = (q_left_des - obs_dict_current['left']['qpos'])/duration
    start = time.time()
    for t in range(int(duration/dt)):
        obs, _, _, _ = env.step({
            'right': {
                'speedj': {'qd': q_right_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([-10.0])}
            },
            'left': {
                'speedj': {'qd': q_left_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([10.0])}
            }
        })
        if render: env.render()
        # obs_dict = env.env.get_obs_dict()
        # right_pos_err = np.linalg.norm(obs_dict['right']['qpos'] - q_right_des)
        # left_pos_err = np.linalg.norm(obs_dict['left']['qpos'] - q_left_des)
        # right_vel_err = np.linalg.norm(obs_dict['right']['qvel'] - q_right_des_vel)
        # left_vel_err = np.linalg.norm(obs_dict['left']['qvel'] - q_left_des_vel)
        print('time: %f [s]'%(t*dt))
        # print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(right_pos_err), np.rad2deg(right_vel_err)))
        # print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(left_pos_err), np.rad2deg(left_vel_err)))
    finish = time.time()
    print('speedj duration: %.3f (actual) vs. %.3f (desired)'%(finish-start, duration))
    qpos_err, qvel = np.inf, np.inf
    while qpos_err > np.deg2rad(1e-1) or qvel > np.deg2rad(3e0):
        ob, _, _, _ = env.step({
            'right': {
                'servoj': {'q': q_right_des, 't': servoj_args['t'], 'wait': servoj_args['wait']},
                'move_gripper_force': {'gf': np.array([-10.0])}
            },
            'left': {
                'servoj': {'q': q_left_des, 't': servoj_args['t'], 'wait': servoj_args['wait']},
                'move_gripper_force': {'gf': np.array([10.0])}
            }
        })
        if render: env.render()
        obs_dict = env.env.get_obs_dict()
        right_err = np.linalg.norm(obs_dict['right']['qpos'] - q_right_des)
        left_err = np.linalg.norm(obs_dict['left']['qpos'] - q_left_des)
        print('time: %f [s]'%(t*dt))
        print('right arm joint error [deg]: %f'%(np.rad2deg(right_err)))
        print('left arm joint error [deg]: %f'%(np.rad2deg(left_err)))
        qpos_err = max(right_err, left_err)
        qvel = np.linalg.norm(np.concatenate([obs_dict['right']['qvel'], obs_dict['left']['qvel']]))
        t += 1

    if env_type == list_of_env_types[1]:
        if prompt_yes_or_no('grasp object?') is False:
            print('exiting program!')
            env.close()
            sys.exit()

    print('Gripping object... (step 4 of 6)')
    time.sleep(1.0)
    env.step({'right': {'close_gripper': {}}, 'left': {}})
    time.sleep(3.0)
    # 4. Grip object
    duration = 1.0
    start = time.time()
    for t in range(int(duration/dt)):
        ob, _, _, _ = env.step({
            'right': {
                'servoj': {'q': q_right_des, 't': servoj_args['t'], 'wait': servoj_args['wait']},
                'move_gripper_force': {'gf': np.array([10.0])}
            },
            'left': {
                'servoj': {'q': q_left_des, 't': servoj_args['t'], 'wait': servoj_args['wait']},
                'move_gripper_force': {'gf': np.array([10.0])}
            }
        })
        if render: env.render()
        # obs_dict = env.env.get_obs_dict()
        # right_err = np.linalg.norm(obs_dict['right']['qpos'] - q_right_des)
        # left_err = np.linalg.norm(obs_dict['left']['qpos'] - q_left_des)
        print('time: %f [s]'%(t*dt))
        # print('right arm joint error [deg]: %f'%(np.rad2deg(right_err)))
        # print('left arm joint error [deg]: %f'%(np.rad2deg(left_err)))
    finish = time.time()
    print('speedj duration: %.3f (actual) vs. %.3f (desired)'%(finish-start, duration))

    print('Lifting object... (step 5 of 6)')
    time.sleep(1.0)
    # 5. Lift object
    duration = 5.0
    duration = 5.0
    ee_pos_right = np.array([0.3, -0.5, 0.9])
    ee_pos_left = np.array([-0.3, -0.5, 0.9])
    q_right_des, iter_taken_right, err_right, null_obj_right = env.env.inverse_kinematics_ee(ee_pos_right, null_obj_func, arm='right')
    q_left_des, iter_taken_left, err_left, null_obj_left = env.env.inverse_kinematics_ee(ee_pos_left, null_obj_func, arm='left')
    obs_dict_current = env.env.get_obs_dict()
    q_right_des_vel = (q_right_des - obs_dict_current['right']['qpos'])/duration
    q_left_des_vel = (q_left_des - obs_dict_current['left']['qpos'])/duration
    start = time.time()
    for t in range(int(duration/dt)):
        obs, _, _, _ = env.step({
            'right': {
                'speedj': {'qd': q_right_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([10.0])}
            },
            'left': {
                'speedj': {'qd': q_left_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([10.0])}
            }
        })
        if render: env.render()
        # obs_dict = env.env.get_obs_dict()
        # right_pos_err = np.linalg.norm(obs_dict['right']['qpos'] - q_right_des)
        # left_pos_err = np.linalg.norm(obs_dict['left']['qpos'] - q_left_des)
        # right_vel_err = np.linalg.norm(obs_dict['right']['qvel'] - q_right_des_vel)
        # left_vel_err = np.linalg.norm(obs_dict['left']['qvel'] - q_left_des_vel)
        print('time: %f [s]'%(t*dt))
        # print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(right_pos_err), np.rad2deg(right_vel_err)))
        # print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(left_pos_err), np.rad2deg(left_vel_err)))
        if env_type == list_of_env_types[0]: # sim only attributes
            right_actuator_torque = np.linalg.norm(env.env._get_ur3_actuator()[:env.ur3_nqvel])
            left_actuator_torque = np.linalg.norm(env.env._get_ur3_actuator()[-env.ur3_nqvel:])
            right_bias_torque = np.linalg.norm(env.env._get_ur3_bias()[:env.ur3_nqvel])
            left_bias_torque = np.linalg.norm(env.env._get_ur3_bias()[-env.ur3_nqvel:])
            right_constraint_torque = np.linalg.norm(env.env._get_ur3_constraint()[:env.ur3_nqvel])
            left_constraint_torque = np.linalg.norm(env.env._get_ur3_constraint()[-env.ur3_nqvel:])
            print('right arm actuator torque [Nm]: %f bias torque [Nm]: %f constraint torque [Nm]: %f'
                %(right_actuator_torque, right_bias_torque, right_constraint_torque))
            print('left arm actuator torque [Nm]: %f bias torque [Nm]: %f constraint torque [Nm]: %f'
                %(left_actuator_torque, left_bias_torque, left_constraint_torque))
    finish = time.time()
    print('speedj duration: %.3f (actual) vs. %.3f (desired)'%(finish-start, duration))
    qpos_err, qvel = np.inf, np.inf
    if env_type == list_of_env_types[0]:
        env.wrapper_right.servoj_gains, env.wrapper_left.servoj_gains = {'P': 1.0, 'I': 2.5, 'D': 0.2}, {'P': 1.0, 'I': 2.5, 'D': 0.2}
    while qpos_err > np.deg2rad(1e-1) or qvel > np.deg2rad(3e0):
        ob, _, _, _ = env.step({
            'right': {
                'servoj': {'q': q_right_des, 't': servoj_args['t'], 'wait': servoj_args['wait']},
                'move_gripper_force': {'gf': np.array([10.0])}
            },
            'left': {
                'servoj': {'q': q_left_des, 't': servoj_args['t'], 'wait': servoj_args['wait']},
                'move_gripper_force': {'gf': np.array([10.0])}
            }
        })
        if render: env.render()
        obs_dict = env.env.get_obs_dict()
        right_err = np.linalg.norm(obs_dict['right']['qpos'] - q_right_des)
        left_err = np.linalg.norm(obs_dict['left']['qpos'] - q_left_des)
        print('time: %f [s]'%(t*dt))
        print('right arm joint error [deg]: %f'%(np.rad2deg(right_err)))
        print('left arm joint error [deg]: %f'%(np.rad2deg(left_err)))
        qpos_err = max(right_err, left_err)
        qvel = np.linalg.norm(np.concatenate([obs_dict['right']['qvel'], obs_dict['left']['qvel']]))
        print('joint velocity [dps]: %f'%(np.rad2deg(qvel)))
        t += 1

    time.sleep(3.0)

    print('Opening gripper... (step 6 of 6)')
    time.sleep(1.0)
    # 6. Open gripper
    duration = 1.0
    env.step({'right': {'open_gripper': {}}, 'left': {}})
    time.sleep(3.0)
    for t in range(int(duration/dt)):
        obs, _, _, _ = env.step({
            'right': {
                'servoj': {'q': q_right_des, 't': servoj_args['t'], 'wait': servoj_args['wait']},
                'move_gripper_force': {'gf': np.array([-25.0])}
            },
            'left': {
                'servoj': {'q': q_left_des, 't': servoj_args['t'], 'wait': servoj_args['wait']},
                'move_gripper_force': {'gf': np.array([10.0])}
            }
        })
        if render: env.render()
        obs_dict = env.env.get_obs_dict()
        right_err = np.linalg.norm(obs_dict['right']['qpos'] - q_right_des)
        left_err = np.linalg.norm(obs_dict['left']['qpos'] - q_left_des)
        print('time: %f [s]'%(t*dt))
        print('right arm joint error [deg]: %f'%(np.rad2deg(right_err)))
        print('left arm joint error [deg]: %f'%(np.rad2deg(left_err)))
        if env_type == list_of_env_types[0]: # sim only attributes
            right_actuator_torque = np.linalg.norm(env.env._get_ur3_actuator()[:env.ur3_nqvel])
            left_actuator_torque = np.linalg.norm(env.env._get_ur3_actuator()[-env.ur3_nqvel:])
            right_bias_torque = np.linalg.norm(env.env._get_ur3_bias()[:env.ur3_nqvel])
            left_bias_torque = np.linalg.norm(env.env._get_ur3_bias()[-env.ur3_nqvel:])
            right_constraint_torque = np.linalg.norm(env.env._get_ur3_constraint()[:env.ur3_nqvel])
            left_constraint_torque = np.linalg.norm(env.env._get_ur3_constraint()[-env.ur3_nqvel:])
            print('right arm actuator torque [Nm]: %f bias torque [Nm]: %f constraint torque [Nm]: %f'
                %(right_actuator_torque, right_bias_torque, right_constraint_torque))
            print('left arm actuator torque [Nm]: %f bias torque [Nm]: %f constraint torque [Nm]: %f'
                %(left_actuator_torque, left_bias_torque, left_constraint_torque))
    print('done!')

def collide_deprecated():

    env = gym_custom.make('dual-ur3-larr-v0')
    obs = env.reset()
    dt = env.dt

    null_obj_func = UprightConstraint()

    PI_gains = {'P': 0.25, 'I': 10.0}
    ur3_scale_factor = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0])*np.array([1.0, 1.0, 1.0, 2.5, 2.5, 2.5])
    gripper_scale_factor = np.array([1.0, 1.0])
    env = URScriptWrapper_deprecated(env, PI_gains, ur3_scale_factor, gripper_scale_factor)

    # Move to position
    q_init = env.env._get_ur3_qpos()
    ee_pos_right = np.array([0.15, -0.4, 0.9])
    ee_pos_left = np.array([-0.3, -0.4, 0.9])
    q_right_des, iter_taken_right, err_right, null_obj_right = env.env.inverse_kinematics_ee(ee_pos_right, null_obj_func, arm='right')
    q_left_des, iter_taken_left, err_left, null_obj_left = env.env.inverse_kinematics_ee(ee_pos_left, null_obj_func, arm='left')
    q_vel_des = (np.concatenate([q_right_des, q_left_des]) - q_init)/5.0
    for t in range(int(5.0/dt)):
        command = {
            'ur3': {'type': 'speedj', 'command': q_vel_des},
            'gripper': {'type': 'forceg', 'command': np.array([-10.0, -10.0])}
        }
        obs, _, _, _ = env.step(command)
        env.render()
        joint_angles = env.env._get_ur3_qpos()
        right_pos_err = np.linalg.norm(joint_angles[:env.ur3_nqpos] - q_right_des)
        left_pos_err = np.linalg.norm(joint_angles[-env.ur3_nqpos:] - q_left_des)
        right_vel_err = np.linalg.norm(env.env._get_ur3_qvel()[:env.ur3_nqpos] - q_vel_des[:env.ur3_nqpos])
        left_vel_err = np.linalg.norm(env.env._get_ur3_qvel()[-env.ur3_nqpos:] - q_vel_des[-env.ur3_nqpos:])
        print('time: %f [s]'%(t*dt))
        print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(right_pos_err*180.0/np.pi, right_vel_err*180.0/np.pi))
        print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(left_pos_err*180.0/np.pi, left_vel_err*180.0/np.pi))

    # Collide with surface
    q_init = env.env._get_ur3_qpos()
    ee_pos_right = np.array([0.15, -0.4, 0.69])
    ee_pos_left = np.array([-0.3, -0.4, 0.9])
    q_right_des, iter_taken_right, err_right, null_obj_right = env.env.inverse_kinematics_ee(ee_pos_right, null_obj_func, arm='right')
    q_left_des, iter_taken_left, err_left, null_obj_left = env.env.inverse_kinematics_ee(ee_pos_left, null_obj_func, arm='left')
    q_vel_des = (np.concatenate([q_right_des, q_left_des]) - q_init)/5.0
    for t in range(int(60.0/dt)):
        command = {
            'ur3': {'type': 'speedj', 'command': q_vel_des},
            'gripper': {'type': 'forceg', 'command': np.array([-10.0, -10.0])}
        }
        obs, _, _, _ = env.step(command)
        env.render()
        joint_angles = env.env._get_ur3_qpos()
        right_pos_err = np.linalg.norm(joint_angles[:env.ur3_nqpos] - q_right_des)
        left_pos_err = np.linalg.norm(joint_angles[-env.ur3_nqpos:] - q_left_des)
        right_vel_err = np.linalg.norm(env.env._get_ur3_qvel()[:env.ur3_nqpos] - q_vel_des[:env.ur3_nqpos])
        left_vel_err = np.linalg.norm(env.env._get_ur3_qvel()[-env.ur3_nqpos:] - q_vel_des[-env.ur3_nqpos:])
        right_actuator_torque = np.linalg.norm(env.env._get_ur3_actuator()[:env.ur3_nqpos])
        left_actuator_torque = np.linalg.norm(env.env._get_ur3_actuator()[-env.ur3_nqpos:])
        right_bias_torque = np.linalg.norm(env.env._get_ur3_bias()[:env.ur3_nqpos])
        left_bias_torque = np.linalg.norm(env.env._get_ur3_bias()[-env.ur3_nqpos:])
        right_constraint_torque = np.linalg.norm(env.env._get_ur3_constraint()[:env.ur3_nqpos])
        left_constraint_torque = np.linalg.norm(env.env._get_ur3_constraint()[-env.ur3_nqpos:])
        print('time: %f [s]'%(t*dt))
        print('right arm joint pos error [deg]: %f vel error [dps]: %f actuator torque [Nm]: %f bias torque [Nm]: %f constraint torque [Nm]: %f'
            %(right_pos_err*180.0/np.pi, right_vel_err*180.0/np.pi, right_actuator_torque, right_bias_torque, right_constraint_torque))
        print('    err_integ: %s'%(env.ur3_err_integ[:env.ur3_nqpos]))
        print('left arm joint pos error [deg]: %f vel error [dps]: %f actuator torque [Nm]: %f bias torque [Nm]: %f constraint torque [Nm]: %f'
            %(left_pos_err*180.0/np.pi, left_vel_err*180.0/np.pi, left_actuator_torque, left_bias_torque, left_constraint_torque))
        print('    err_integ: %s'%(env.ur3_err_integ[-env.ur3_nqpos:]))

def collide(env_type='sim', render=False):
    list_of_env_types = ['sim']

    if env_type == list_of_env_types[0]:
        env = gym_custom.make('dual-ur3-larr-v0')
        speedj_args = {'a': 5, 't': None, 'wait': None}
    # elif env_type == list_of_env_types[1]:
    #     env = gym_custom.make('dual-ur3-larr-real-v0')
    #     speedj_args = {'a': 5, 't': 2/env.rate._freq, 'wait': False}
    else: raise ValueError('Invalid env_type! Availiable options are %s'%(list_of_env_types))
    obs = env.reset()
    dt = env.dt

    null_obj_func = UprightConstraint()

    if env_type == list_of_env_types[0]:
        PI_gains = {'speedj': {'P': 0.25, 'I': 10.0}}
        ur3_scale_factor = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0])*np.array([1.0, 1.0, 1.0, 2.5, 2.5, 2.5])
        gripper_scale_factor = np.array([1.0])
        env = URScriptWrapper(env, PI_gains, ur3_scale_factor, gripper_scale_factor)
    elif env_type == list_of_env_types[1]:
        env.env = env

    # Move to goal
    duration = 5.0 # in seconds
    ee_pos_right = np.array([0.15, -0.4, 0.9])
    ee_pos_left = np.array([-0.3, -0.4, 0.9])
    q_right_des, iter_taken_right, err_right, null_obj_right = env.inverse_kinematics_ee(ee_pos_right, null_obj_func, arm='right')
    q_left_des, iter_taken_left, err_left, null_obj_left = env.inverse_kinematics_ee(ee_pos_left, null_obj_func, arm='left')
    obs_dict_current = env.env.get_obs_dict()
    q_right_des_vel = (q_right_des - obs_dict_current['right']['qpos'])/duration
    q_left_des_vel = (q_left_des - obs_dict_current['left']['qpos'])/duration
    for t in range(int(duration/dt)):
        obs, _, _, _ = env.step({
            'right': {
                'speedj': {'qd': q_right_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([-10.0])}
            },
            'left': {
                'speedj': {'qd': q_left_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([-10.0])}
            }
        })
        if render: env.render()
        obs_dict = env.env.get_obs_dict()
        right_pos_err = np.linalg.norm(obs_dict['right']['qpos'] - q_right_des)
        left_pos_err = np.linalg.norm(obs_dict['left']['qpos'] - q_left_des)
        right_vel_err = np.linalg.norm(obs_dict['right']['qvel'] - q_right_des_vel)
        left_vel_err = np.linalg.norm(obs_dict['left']['qvel'] - q_left_des_vel)
        print('time: %f [s]'%(t*dt))
        print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(right_pos_err), np.rad2deg(right_vel_err)))
        print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(left_pos_err), np.rad2deg(left_vel_err)))

    # Collide with surface
    ee_pos_right = np.array([0.15, -0.4, 0.69])
    ee_pos_left = np.array([-0.3, -0.4, 0.9])
    q_right_des, iter_taken_right, err_right, null_obj_right = env.inverse_kinematics_ee(ee_pos_right, null_obj_func, arm='right')
    q_left_des, iter_taken_left, err_left, null_obj_left = env.inverse_kinematics_ee(ee_pos_left, null_obj_func, arm='left')
    obs_dict_current = env.env.get_obs_dict()
    q_right_des_vel = (q_right_des - obs_dict_current['right']['qpos'])/duration
    q_left_des_vel = (q_left_des - obs_dict_current['left']['qpos'])/duration
    for t in range(int(10*duration/dt)):
        obs, _, _, _ = env.step({
            'right': {
                'speedj': {'qd': q_right_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([-10.0])}
            },
            'left': {
                'speedj': {'qd': q_left_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([-10.0])}
            }
        })
        if render: env.render()
        obs_dict = env.env.get_obs_dict()
        right_pos_err = np.linalg.norm(obs_dict['right']['qpos'] - q_right_des)
        left_pos_err = np.linalg.norm(obs_dict['left']['qpos'] - q_left_des)
        right_vel_err = np.linalg.norm(obs_dict['right']['qvel'] - q_right_des_vel)
        left_vel_err = np.linalg.norm(obs_dict['left']['qvel'] - q_left_des_vel)
        print('time: %f [s]'%(t*dt))
        print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(right_pos_err), np.rad2deg(right_vel_err)))
        print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(left_pos_err), np.rad2deg(left_vel_err)))
        if env_type == list_of_env_types[0]: # sim only attributes
            right_actuator_torque = np.linalg.norm(env.env._get_ur3_actuator()[:env.ur3_nqvel])
            left_actuator_torque = np.linalg.norm(env.env._get_ur3_actuator()[-env.ur3_nqvel:])
            right_bias_torque = np.linalg.norm(env.env._get_ur3_bias()[:env.ur3_nqvel])
            left_bias_torque = np.linalg.norm(env.env._get_ur3_bias()[-env.ur3_nqvel:])
            right_constraint_torque = np.linalg.norm(env.env._get_ur3_constraint()[:env.ur3_nqvel])
            left_constraint_torque = np.linalg.norm(env.env._get_ur3_constraint()[-env.ur3_nqvel:])
            print('right arm actuator torque [Nm]: %f bias torque [Nm]: %f constraint torque [Nm]: %f'
                %(right_actuator_torque, right_bias_torque, right_constraint_torque))
            print('    err_integ: %s'%(env.wrapper_right.ur3_err_integ))
            print('left arm actuator torque [Nm]: %f bias torque [Nm]: %f constraint torque [Nm]: %f'
                %(left_actuator_torque, left_bias_torque, left_constraint_torque))
            print('    err_integ: %s'%(env.wrapper_left.ur3_err_integ))
    
def real_env_get_obs_rate_test(wait=True):
    env = gym_custom.make('dual-ur3-larr-real-v0',
            host_ip_right='192.168.5.102',
            host_ip_left='192.168.5.101',
            rate=25
        )
    stime = time.time()
    [env._get_obs(wait=wait) for _ in range(100)]
    ftime = time.time()
    # stats
    # single call: 8ms (wait=True, default), <1ms (wait=False)
    # 2 calls: 16ms (wait=True, default), <1ms (wait=False)
    # 3 calls: 17ms (wait=True, default), <1ms (wait=False)
    # 4 calls: 24ms (wait=True, default), <1ms (wait=False)
    print('\r\ntime per call: %f ms'%((ftime-stime)/100*1000))
    print('done!\r\n')

def real_env_command_send_rate_test(wait=True):
    # ~25ms (wait=True, default), ~11ms (wait=False)
    env = gym_custom.make('dual-ur3-larr-real-v0',
            host_ip_right='192.168.5.102',
            host_ip_left='192.168.5.101',
            rate=100
        )
    env.set_initial_joint_pos(np.deg2rad([90, -45, 135, -180, 45, 0, -90, -135, -135, 0, -45, 0]))
    env.set_initial_gripper_pos(np.array([0.0, 0.0]))
    env.reset()
    command = {
        'right': {'speedj': {'qd': np.zeros([6]), 'a': 1.0, 't': 1.0, 'wait': False}},
        'left': {'speedj': {'qd': np.zeros([6]), 'a': 1.0, 't': 1.0, 'wait': False}}
    }
    stime = time.time()
    [env.step(command, wait=wait) for _ in range(100)]
    ftime = time.time()
    command = {
        'right': {'stopj': {'a': 1.0}},
        'left': {'stopj': {'a': 1.0}}
    }
    env.step(command)
    print('\r\ntime per call: %f ms'%((ftime-stime)/100*1000))
    print('done!\r\n')


def dscho_pick_and_place(env_type='sim', render=False):
    list_of_env_types = ['sim', 'real']
    
    if env_type == list_of_env_types[0]:
        env = gym_custom.make('dscho-dual-ur3-pickandplace-v0')
        servoj_args, speedj_args = {'t': None, 'wait': None}, {'a': 5, 't': None, 'wait': None}
    elif env_type == list_of_env_types[1]:
        env = gym_custom.make('dual-ur3-larr-real-v0',
            host_ip_right='192.168.5.102',
            host_ip_left='192.168.5.101',
            rate=25
        )
        servoj_args, speedj_args = {'t': 2/env.rate._freq, 'wait': False}, {'a': 5, 't': 2/env.rate._freq, 'wait': False}
        # 1. Set initial as current configuration
        env.set_initial_joint_pos('current')
        env.set_initial_gripper_pos('current')
        # 2. Set inital as default configuration
        env.set_initial_joint_pos(np.deg2rad([90, -45, 135, -180, 45, 0, -90, -135, -135, 0, -45, 0]))
        env.set_initial_gripper_pos(np.array([0.0, 0.0]))
        assert render is False
    else: raise ValueError('Invalid env_type! Availiable options are %s'%(list_of_env_types))
    obs = env.reset()
    dt = env.dt

    if env_type == list_of_env_types[0]:
        PID_gains = {'servoj': {'P': 1.0, 'I': 0.5, 'D': 0.2}, 'speedj': {'P': 0.20, 'I':5.0}}
        ur3_scale_factor = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0])*np.array([1.0, 1.0, 1.0, 2.5, 2.5, 2.5])
        gripper_scale_factor = np.array([1.0])
        env = URScriptWrapper(env, PID_gains, ur3_scale_factor, gripper_scale_factor)
    elif env_type == list_of_env_types[1]:
        env.env = env

    # null_obj_func = UprightConstraint()

    from gym_custom.envs.custom.ur_utils import SO3Constraint
    SO3_r = 'vertical_side' # 'horizontal_right_side'
    null_obj_func_r = SO3Constraint(SO3_r)
    SO3_l = 'vertical_side' #'horizontal_left_front'
    null_obj_func_l = SO3Constraint(SO3_l)

    if env_type == list_of_env_types[1]:
        if prompt_yes_or_no('current qpos is \r\n right: %s deg\r\n left: %s deg\r\n?'
            %(np.rad2deg(env.env._init_qpos[:6]), np.rad2deg(env.env._init_qpos[6:]))) is False:
            print('exiting program!')
            env.close()
            sys.exit()

    print('Moving to position... (step 1 of 6)')
    time.sleep(1.0)
    # 1. Move to initial position
    duration = 5.0
    # ee_pos_right = np.array([0.0, -0.4, 0.9])
    # ee_pos_left = np.array([-0.3, -0.5, 0.9])
    ee_pos_right = env.env.get_site_pos('objSite') + np.array([0,0,0.1])
    ee_pos_left = env.env.get_site_pos('second_objSite') + np.array([0,0,0.1])

    q_right_des, iter_taken_right, err_right, null_obj_right = env.env.inverse_kinematics_ee(ee_pos_right, null_obj_func_r, arm='right')
    q_left_des, iter_taken_left, err_left, null_obj_left = env.env.inverse_kinematics_ee(ee_pos_left, null_obj_func_l, arm='left')
    obs_dict_current = env.env.get_obs_dict()
    q_right_des_vel = (q_right_des - obs_dict_current['right']['qpos'])/duration
    q_left_des_vel = (q_left_des - obs_dict_current['left']['qpos'])/duration
    start = time.time()
    #dscho mod
    multi_step =20
    dt = dt*multi_step
    for t in range(int(duration/dt)):
        for  _ in range(multi_step):
            obs, _, _, _ = env.step({
                'right': {
                    'speedj': {'qd': q_right_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                    'move_gripper_force': {'gf': np.array([0.0])}
                },
                'left': {
                    'speedj': {'qd': q_left_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                    'move_gripper_force': {'gf': np.array([0.0])}
                }
            })

        if render: env.render()
        obs_dict = env.env.get_obs_dict()
        right_pos_err = np.linalg.norm(obs_dict['right']['qpos'] - q_right_des)
        left_pos_err = np.linalg.norm(obs_dict['left']['qpos'] - q_left_des)
        right_vel_err = np.linalg.norm(obs_dict['right']['qvel'] - q_right_des_vel)
        left_vel_err = np.linalg.norm(obs_dict['left']['qvel'] - q_left_des_vel)
        print('time: %f [s]'%(t*dt))
        print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(right_pos_err), np.rad2deg(right_vel_err)))
        print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(left_pos_err), np.rad2deg(left_vel_err)))
    finish = time.time()
    # env.step({'right': {'stopj': {'a': speedj_args['a']}}, 'left': {'stopj': {'a': speedj_args['a']}}})
    print('speedj duration: %.3f (actual) vs. %.3f (desired)'%(finish-start, duration))

    print('Opening right gripper... (step 2 of 6)')
    time.sleep(1.0)
    # env.step({'right': {'open_gripper': {}}})
    # 2. Open right gripper
    pass
    

    print('Placing right gripper... (step 3 of 6)')
    time.sleep(1.0)
    # 3. Place right gripper
    duration = 5.0
    ee_pos_right = env.env.get_site_pos('objSite') + np.array([0.0, 0.0, 0.05])
    ee_pos_left = env.env.get_site_pos('second_objSite')+ np.array([0.0, 0.0, 0.05])
    q_right_des, iter_taken_right, err_right, null_obj_right = env.env.inverse_kinematics_ee(ee_pos_right, null_obj_func_r, arm='right')
    q_left_des, iter_taken_left, err_left, null_obj_left = env.env.inverse_kinematics_ee(ee_pos_left, null_obj_func_l, arm='left')
    obs_dict_current = env.env.get_obs_dict()
    q_right_des_vel = (q_right_des - obs_dict_current['right']['qpos'])/duration
    q_left_des_vel = (q_left_des - obs_dict_current['left']['qpos'])/duration
    start = time.time()
    for t in range(int(duration/dt)):
        for  _ in range(multi_step):
            obs, _, _, _ = env.step({
                'right': {
                    'speedj': {'qd': q_right_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                    'move_gripper_force': {'gf': np.array([-10.0])}
                },
                'left': {
                    'speedj': {'qd': q_left_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                    'move_gripper_force': {'gf': np.array([-10.0])}
                }
            })
        if render: env.render()
        # obs_dict = env.env.get_obs_dict()
        # right_pos_err = np.linalg.norm(obs_dict['right']['qpos'] - q_right_des)
        # left_pos_err = np.linalg.norm(obs_dict['left']['qpos'] - q_left_des)
        # right_vel_err = np.linalg.norm(obs_dict['right']['qvel'] - q_right_des_vel)
        # left_vel_err = np.linalg.norm(obs_dict['left']['qvel'] - q_left_des_vel)
        print('time: %f [s]'%(t*dt))
        # print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(right_pos_err), np.rad2deg(right_vel_err)))
        # print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(left_pos_err), np.rad2deg(left_vel_err)))
    finish = time.time()
    print('speedj duration: %.3f (actual) vs. %.3f (desired)'%(finish-start, duration))
    qpos_err, qvel = np.inf, np.inf
    while qpos_err > np.deg2rad(1e-1) or qvel > np.deg2rad(1e0):
        for  _ in range(multi_step):
            ob, _, _, _ = env.step({
                'right': {
                    'servoj': {'q': q_right_des, 't': servoj_args['t'], 'wait': servoj_args['wait']},
                    'move_gripper_force': {'gf': np.array([-10.0])}
                },
                'left': {
                    'servoj': {'q': q_left_des, 't': servoj_args['t'], 'wait': servoj_args['wait']},
                    'move_gripper_force': {'gf': np.array([10.0])}
                }
            })
        if render: env.render()
        obs_dict = env.env.get_obs_dict()
        right_err = np.linalg.norm(obs_dict['right']['qpos'] - q_right_des)
        left_err = np.linalg.norm(obs_dict['left']['qpos'] - q_left_des)
        print('time: %f [s]'%(t*dt))
        print('right arm joint error [deg]: %f'%(np.rad2deg(right_err)))
        print('left arm joint error [deg]: %f'%(np.rad2deg(left_err)))
        qpos_err = max(right_err, left_err)
        qvel = np.linalg.norm(np.concatenate([obs_dict['right']['qvel'], obs_dict['left']['qvel']]))
        t += 1

    if env_type == list_of_env_types[1]:
        if prompt_yes_or_no('grasp object?') is False:
            print('exiting program!')
            env.close()
            sys.exit()

    print('Gripping object... (step 4 of 6)')
    time.sleep(1.0)
    # env.step({'right': {'close_gripper': {}}})
    # 4. Grip object
    duration = 5.0
    start = time.time()
    for t in range(int(duration/dt)):
        for  _ in range(multi_step):
            ob, _, _, _ = env.step({
                'right': {
                    'servoj': {'q': q_right_des, 't': servoj_args['t'], 'wait': servoj_args['wait']},
                    'move_gripper_force': {'gf': np.array([10.0])}
                },
                'left': {
                    'servoj': {'q': q_left_des, 't': servoj_args['t'], 'wait': servoj_args['wait']},
                    'move_gripper_force': {'gf': np.array([10.0])}
                }
            })
        if render: env.render()
        # obs_dict = env.env.get_obs_dict()
        # right_err = np.linalg.norm(obs_dict['right']['qpos'] - q_right_des)
        # left_err = np.linalg.norm(obs_dict['left']['qpos'] - q_left_des)
        print('time: %f [s]'%(t*dt))
        # print('right arm joint error [deg]: %f'%(np.rad2deg(right_err)))
        # print('left arm joint error [deg]: %f'%(np.rad2deg(left_err)))
    finish = time.time()
    print('speedj duration: %.3f (actual) vs. %.3f (desired)'%(finish-start, duration))

    print('Lifting object... (step 5 of 6)')
    time.sleep(1.0)
    # 5. Lift object
    duration = 5.0
    # ee_pos_right = np.array([0.3, -0.5, 0.9])
    # ee_pos_left = np.array([-0.3, -0.5, 0.9])
    ee_pos_right = env.env.get_site_pos('objSite') + np.array([0.0, 0.0, 0.15])
    ee_pos_left = env.env.get_site_pos('second_objSite') + np.array([0.0, 0.0, 0.15])
    q_right_des, iter_taken_right, err_right, null_obj_right = env.env.inverse_kinematics_ee(ee_pos_right, null_obj_func_r, arm='right')
    q_left_des, iter_taken_left, err_left, null_obj_left = env.env.inverse_kinematics_ee(ee_pos_left, null_obj_func_l, arm='left')
    obs_dict_current = env.env.get_obs_dict()
    q_right_des_vel = (q_right_des - obs_dict_current['right']['qpos'])/duration
    q_left_des_vel = (q_left_des - obs_dict_current['left']['qpos'])/duration
    start = time.time()
    for t in range(int(duration/dt)):
        for  _ in range(multi_step):
            obs, _, _, _ = env.step({
                'right': {
                    'speedj': {'qd': q_right_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                    'move_gripper_force': {'gf': np.array([10.0])}
                },
                'left': {
                    'speedj': {'qd': q_left_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                    'move_gripper_force': {'gf': np.array([10.0])}
                }
            })
        if render: env.render()
        # obs_dict = env.env.get_obs_dict()
        # right_pos_err = np.linalg.norm(obs_dict['right']['qpos'] - q_right_des)
        # left_pos_err = np.linalg.norm(obs_dict['left']['qpos'] - q_left_des)
        # right_vel_err = np.linalg.norm(obs_dict['right']['qvel'] - q_right_des_vel)
        # left_vel_err = np.linalg.norm(obs_dict['left']['qvel'] - q_left_des_vel)
        print('time: %f [s]'%(t*dt))
        # print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(right_pos_err), np.rad2deg(right_vel_err)))
        # print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(left_pos_err), np.rad2deg(left_vel_err)))
        if env_type == list_of_env_types[0]: # sim only attributes
            right_actuator_torque = np.linalg.norm(env.env._get_ur3_actuator()[:env.ur3_nqvel])
            left_actuator_torque = np.linalg.norm(env.env._get_ur3_actuator()[-env.ur3_nqvel:])
            right_bias_torque = np.linalg.norm(env.env._get_ur3_bias()[:env.ur3_nqvel])
            left_bias_torque = np.linalg.norm(env.env._get_ur3_bias()[-env.ur3_nqvel:])
            right_constraint_torque = np.linalg.norm(env.env._get_ur3_constraint()[:env.ur3_nqvel])
            left_constraint_torque = np.linalg.norm(env.env._get_ur3_constraint()[-env.ur3_nqvel:])
            print('right arm actuator torque [Nm]: %f bias torque [Nm]: %f constraint torque [Nm]: %f'
                %(right_actuator_torque, right_bias_torque, right_constraint_torque))
            print('left arm actuator torque [Nm]: %f bias torque [Nm]: %f constraint torque [Nm]: %f'
                %(left_actuator_torque, left_bias_torque, left_constraint_torque))
    finish = time.time()
    print('speedj duration: %.3f (actual) vs. %.3f (desired)'%(finish-start, duration))
    qpos_err, qvel = np.inf, np.inf
    if env_type == list_of_env_types[0]:
        env.wrapper_right.servoj_gains, env.wrapper_left.servoj_gains = {'P': 1.0, 'I': 2.5, 'D': 0.2}, {'P': 1.0, 'I': 2.5, 'D': 0.2}
    while qpos_err > np.deg2rad(1e-1) or qvel > np.deg2rad(1e0):
        for  _ in range(multi_step):
            ob, _, _, _ = env.step({
                'right': {
                    'servoj': {'q': q_right_des, 't': servoj_args['t'], 'wait': servoj_args['wait']},
                    'move_gripper_force': {'gf': np.array([10.0])}
                },
                'left': {
                    'servoj': {'q': q_left_des, 't': servoj_args['t'], 'wait': servoj_args['wait']},
                    'move_gripper_force': {'gf': np.array([10.0])}
                }
            })
        if render: env.render()
        obs_dict = env.env.get_obs_dict()
        right_err = np.linalg.norm(obs_dict['right']['qpos'] - q_right_des)
        left_err = np.linalg.norm(obs_dict['left']['qpos'] - q_left_des)
        print('time: %f [s]'%(t*dt))
        print('right arm joint error [deg]: %f'%(np.rad2deg(right_err)))
        print('left arm joint error [deg]: %f'%(np.rad2deg(left_err)))
        qpos_err = max(right_err, left_err)
        qvel = np.linalg.norm(np.concatenate([obs_dict['right']['qvel'], obs_dict['left']['qvel']]))
        print('joint velocity [dps]: %f'%(np.rad2deg(qvel)))
        t += 1

    time.sleep(3.0)

    print('Opening gripper... (step 6 of 6)')
    time.sleep(1.0)
    # 6. Open gripper
    duration = 5.0
    # env.step({'right': {'open_gripper': {}}})
    for t in range(int(duration/dt)):
        for  _ in range(multi_step):
            obs, _, _, _ = env.step({
                'right': {
                    'servoj': {'q': q_right_des, 't': servoj_args['t'], 'wait': servoj_args['wait']},
                    'move_gripper_force': {'gf': np.array([-25.0])}
                },
                'left': {
                    'servoj': {'q': q_left_des, 't': servoj_args['t'], 'wait': servoj_args['wait']},
                    'move_gripper_force': {'gf': np.array([-25.0])}
                }
            })
        if render: env.render()
        obs_dict = env.env.get_obs_dict()
        right_err = np.linalg.norm(obs_dict['right']['qpos'] - q_right_des)
        left_err = np.linalg.norm(obs_dict['left']['qpos'] - q_left_des)
        print('time: %f [s]'%(t*dt))
        print('right arm joint error [deg]: %f'%(np.rad2deg(right_err)))
        print('left arm joint error [deg]: %f'%(np.rad2deg(left_err)))
        if env_type == list_of_env_types[0]: # sim only attributes
            right_actuator_torque = np.linalg.norm(env.env._get_ur3_actuator()[:env.ur3_nqvel])
            left_actuator_torque = np.linalg.norm(env.env._get_ur3_actuator()[-env.ur3_nqvel:])
            right_bias_torque = np.linalg.norm(env.env._get_ur3_bias()[:env.ur3_nqvel])
            left_bias_torque = np.linalg.norm(env.env._get_ur3_bias()[-env.ur3_nqvel:])
            right_constraint_torque = np.linalg.norm(env.env._get_ur3_constraint()[:env.ur3_nqvel])
            left_constraint_torque = np.linalg.norm(env.env._get_ur3_constraint()[-env.ur3_nqvel:])
            print('right arm actuator torque [Nm]: %f bias torque [Nm]: %f constraint torque [Nm]: %f'
                %(right_actuator_torque, right_bias_torque, right_constraint_torque))
            print('left arm actuator torque [Nm]: %f bias torque [Nm]: %f constraint torque [Nm]: %f'
                %(left_actuator_torque, left_bias_torque, left_constraint_torque))


def dscho_single_arm_speedj_and_forceg(env_type='sim', render=False):
    list_of_env_types = ['sim', 'real']
    
    if env_type == list_of_env_types[0]:
        env_kwargs = {'which_hand' : 'left'}
        env = gym_custom.make('dscho-single-ur3-reach-v0', **env_kwargs)
        speedj_args = {'a': 5, 't': None, 'wait': None}
    elif env_type == list_of_env_types[1]:
        raise NotImplementedError
        env = gym_custom.make('dual-ur3-larr-real-v0',
            host_ip_right='192.168.5.102',
            host_ip_left='192.168.5.101',
            rate=25
        )
        speedj_args = {'a': 5, 't': 2/env.rate._freq, 'wait': False}
        env.set_initial_joint_pos('current')
        env.set_initial_gripper_pos('current')
        # 2. Set inital as default configuration
        env.set_initial_joint_pos(np.deg2rad([90, -45, 135, -180, 45, 0, -90, -135, -135, 0, -45, 0]))
        env.set_initial_gripper_pos(np.array([0.0, 0.0]))
        assert render is False
    else: raise ValueError('Invalid env_type! Availiable options are %s'%(list_of_env_types))
    obs = env.reset()
    dt = env.dt

    null_obj_func = UprightConstraint()

    # ee_pos_right = np.array([0.1, -0.5, 0.9])
    ee_pos_right = np.array([0.1 ,-0.5, 0.9])
    ee_pos_left = np.array([-0.1, -0.5, 0.9])
    q_right_des, iter_taken_right, err_right, null_obj_right = env.inverse_kinematics_ee(ee_pos_right, null_obj_func, arm='right')
    q_left_des, iter_taken_left, err_left, null_obj_left = env.inverse_kinematics_ee(ee_pos_left, null_obj_func, arm='left')
    print('q_right_des : {}, iter_taken : {}, err : {}, null_obj : {}'.format(q_right_des, iter_taken_right, err_right, null_obj_right))
    print('q_left_des : {}, iter_taken : {}, err : {}, null_obj : {}'.format(q_left_des, iter_taken_left, err_left, null_obj_left))
    if env_type == list_of_env_types[0]:
        PI_gains = {'speedj': {'P': 0.2, 'I': 10.0}} # was 0.2, 10.0
        ur3_scale_factor = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0])*np.array([1.0, 1.0, 1.0, 2.5, 2.5, 2.5])
        gripper_scale_factor = np.array([1.0])
        # env = URScriptWrapper(env, PI_gains, ur3_scale_factor, gripper_scale_factor)
        from gym_custom.envs.custom.dscho_dual_ur3_goal_env import SingleWrapper
        q_control_type='speedj'
        g_control_type='move_gripper_force'
        #multi_step=5 # 1step : 0.01s -> 100Hz
        multi_step=20 # 1step : 0.04s -> 25Hz 
        gripper_action = False
        env = SingleWrapper(q_control_type, g_control_type, multi_step, gripper_action, env, PI_gains, ur3_scale_factor, gripper_scale_factor)
        dt = env.dt
    elif env_type == list_of_env_types[1]:
        env.env = env

    if env_type == list_of_env_types[1]:
        if prompt_yes_or_no('current qpos is \r\n right: %s deg\r\n left: %s deg\r\n?'
            %(np.rad2deg(env.env._init_qpos[:6]), np.rad2deg(env.env._init_qpos[6:]))) is False:
            print('exiting program!')
            env.close()
            sys.exit()
    
    # Move to goal
    duration = 3.0 # in seconds
    obs_dict_current = env.env.get_obs_dict()
    q_right_des_vel = (q_right_des - obs_dict_current['right']['qpos'])/duration
    q_left_des_vel = (q_left_des - obs_dict_current['left']['qpos'])/duration
    start = time.time()
    for t in range(int(duration/dt)):
        # obs, _, _, _ = env.step({
        #     'right': {
        #         'speedj': {'qd': q_right_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
        #         'move_gripper_force': {'gf': np.array([1.0])}
        #     },
        #     'left': {
        #         'speedj': {'qd': q_left_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
        #         'move_gripper_force': {'gf': np.array([1.0])}
        #     }
        # })
        if env.which_hand =='right':
            action = q_right_des_vel
        elif env.which_hand =='left':
            action = q_left_des_vel
        obs, _, _, _ = env.step(action)
        if render: env.render()
        # TODO: get_obs_dict() takes a long time causing timing issues.
        #   Is it due to Upboard's lackluster performance or some deeper
        #   issues within UR Script wrppaer?
        obs_dict = env.env.get_obs_dict()
        right_pos_err = np.linalg.norm(obs_dict['right']['qpos'] - q_right_des)
        left_pos_err = np.linalg.norm(obs_dict['left']['qpos'] - q_left_des)
        right_vel_err = np.linalg.norm(obs_dict['right']['qvel'] - q_right_des_vel)
        left_vel_err = np.linalg.norm(obs_dict['left']['qvel'] - q_left_des_vel)
        print('time: %f [s]'%(t*dt))
        print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(right_pos_err), np.rad2deg(right_vel_err)))
        print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(left_pos_err), np.rad2deg(left_vel_err)))
    finish = time.time()

    # Stop
    t = 0
    qvel_err = np.inf
    q_right_des_vel, q_left_des_vel = np.zeros([env.ur3_nqpos]), np.zeros([env.ur3_nqpos])
    while qvel_err > np.deg2rad(1e0):
        obs, _, _, _ = env.step({
            'right': {
                'speedj': {'qd': q_right_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([-1.0])}
            },
            'left': {
                'speedj': {'qd': q_left_des_vel, 'a': speedj_args['a'], 't': speedj_args['t'], 'wait': speedj_args['wait']},
                'move_gripper_force': {'gf': np.array([-1.0])}
            }
        })
        if render: env.render()
        obs_dict = env.env.get_obs_dict()
        right_pos_err = np.linalg.norm(obs_dict['right']['qpos'] - q_right_des)
        left_pos_err = np.linalg.norm(obs_dict['left']['qpos'] - q_left_des)
        right_vel_err = np.linalg.norm(obs_dict['right']['qvel'] - q_right_des_vel)
        left_vel_err = np.linalg.norm(obs_dict['left']['qvel'] - q_left_des_vel)
        print('time: %f [s]'%(t*dt))
        print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(right_pos_err), np.rad2deg(right_vel_err)))
        print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(left_pos_err), np.rad2deg(left_vel_err)))
        qvel_err = np.linalg.norm(np.concatenate([obs_dict['right']['qvel'], obs_dict['left']['qvel']]) - np.concatenate([q_right_des_vel, q_left_des_vel]))
        t += 1
    
    if env_type == list_of_env_types[0]:
        time.sleep(100)
    else:
        env.close()
        print('%.3f seconds'%(finish-start))
        sys.exit()

def dscho_single_arm_goal_sample_debug(env_type='sim', render=False):
    list_of_env_types = ['sim', 'real']
    
    if env_type == list_of_env_types[0]:
        reduced_observation= False 
        initMode = None
        env_kwargs = {
            'sparse_reward' : False,
            'reduced_observation' : reduced_observation,
            'trigonometry_observation' : True,
            'reward_by_ee' : True,
            'ur3_random_init' : False,
            'goal_random_init' : True, 
            'automatically_set_spaces' : False,
            'reward_success_criterion' : 'ee_pos',
            'distance_threshold' : 0.05,
            'initMode' : initMode,
        }
        
        if reduced_observation:
            env_kwargs.update({'full_state_goal' : False})
        else :
            env_kwargs.update({'full_state_goal' : True})

        env = gym_custom.make('dscho-single-ur3-reach-v0', which_hand = 'left', **env_kwargs)
        speedj_args = {'a': 5, 't': None, 'wait': None}
    
    else: raise ValueError('Invalid env_type! Availiable options are %s'%(list_of_env_types))
    

    if env_type == list_of_env_types[0]:
        q_control_type = 'speedj'
        if q_control_type == 'servoj':
            PID_gains = {'servoj': {'P': 1.0, 'I': 0.5, 'D': 0.2}}
        elif q_control_type == 'speedj':
            PID_gains = {'speedj': {'P': 0.2, 'I': 10.0}} # was 0.2, 5.0
        ur3_scale_factor = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0])*np.array([1.0, 1.0, 1.0, 2.5, 2.5, 2.5])
        gripper_scale_factor = np.array([1.0])
        g_control_type='move_gripper_force'
        #multi_step=5 # 1step : 0.01s -> 100Hz
        multi_step=20 # 1step : 0.04s -> 25Hz 
        gripper_action = False
        from gym_custom.envs.custom.dscho_dual_ur3_goal_env import SingleWrapper
        env = SingleWrapper(q_control_type, g_control_type, multi_step, gripper_action, env, PID_gains, ur3_scale_factor, gripper_scale_factor)
        dt = env.dt
        from envs.wrapper import TDMGoalEnvWrapper
        env = TDMGoalEnvWrapper(reward_scale=1.0,
                                    norm_order=1,  
                                    vectorized = True,
                                    weighted_reward=False,
                                    env=env
                                    )

    print('before reset')
    while True:
        obs = env.reset()
        goal = env.get_current_goal()
        goal_ee_pos = goal[-3:]
        print(goal_ee_pos)
        assert (goal_ee_pos >=env.goal_ee_pos_space.low).all() and (goal_ee_pos <= env.goal_ee_pos_space.high).all()
    


def dscho_single_arm_IK_debug(env_type='sim', render=False):
    list_of_env_types = ['sim', 'real']
    
    if env_type == list_of_env_types[0]:
        reduced_observation= False 
        initMode = None
        env_kwargs = {
            'sparse_reward' : False,
            'reduced_observation' : reduced_observation,
            'trigonometry_observation' : True,
            'reward_by_ee' : True,
            'ur3_random_init' : False,
            'goal_random_init' : True, 
            'automatically_set_spaces' : False,
            'reward_success_criterion' : 'ee_pos',
            'distance_threshold' : 0.05,
            'initMode' : initMode,
        }
        
        if reduced_observation:
            env_kwargs.update({'full_state_goal' : False})
        else :
            env_kwargs.update({'full_state_goal' : True})

        env = gym_custom.make('dscho-single-ur3-reach-v0', which_hand = 'right', **env_kwargs)
        speedj_args = {'a': 5, 't': None, 'wait': None}
    
    else: raise ValueError('Invalid env_type! Availiable options are %s'%(list_of_env_types))
    obs = env.reset()
    dt = env.dt

    # while True:
    #     env.render()
    
    

    if env_type == list_of_env_types[0]:
        q_control_type = 'speedj'
        if q_control_type == 'servoj':
            PID_gains = {'servoj': {'P': 1.0, 'I': 0.5, 'D': 0.2}}
        elif q_control_type == 'speedj':
            PID_gains = {'speedj': {'P': 0.2, 'I': 10.0}} # was 0.2, 5.0
        ur3_scale_factor = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0])*np.array([1.0, 1.0, 1.0, 2.5, 2.5, 2.5])
        gripper_scale_factor = np.array([1.0])
        g_control_type='move_gripper_force'
        #multi_step=5 # 1step : 0.01s -> 100Hz
        multi_step=20 # 1step : 0.04s -> 25Hz 
        gripper_action = False
        from gym_custom.envs.custom.dscho_dual_ur3_goal_env import SingleWrapper
        env = SingleWrapper(q_control_type, g_control_type, multi_step, gripper_action, env, PID_gains, ur3_scale_factor, gripper_scale_factor)
        dt = env.dt
        from envs.wrapper import TDMGoalEnvWrapper
        env = TDMGoalEnvWrapper(reward_scale=1.0,
                                    norm_order=1,  
                                    vectorized = True,
                                    weighted_reward=False,
                                    env=env
                                    )

    # for IK debug
    obs = env.reset()
    obs_dict = env.convert_obs_to_dict(obs)
    pure_obs = obs_dict['observation']
    achieved_goal = obs_dict['achieved_goal']
    current_ee_pos = env.get_endeff_pos(arm=env.which_hand)


def dscho_dual_arm_test(env_type='sim', render=False):
    list_of_env_types = ['sim', 'real']
    env_id = 'dscho-dual-ur3-bar-pickandplace-obstacle-v3'
    # env = gym_custom.make('dscho-dual-ur3-cylinder-pickandplace-v0', ur3_random_init=True) # ur3_random_init =True 하면, initMode 설정하든말든 무의미
    env = gym_custom.make(env_id, initMode = None)
    # env = gym_custom.make('dscho-dual-ur3-stick-pull-obstacle-v0', initMode = None)    
    # env = gym_custom.make('dscho-dual-ur3-cylinder-pickandplace-obstacle-v0', initMode = None)    
                
    q_control_type = 'speedj'
    if q_control_type == 'servoj':
        PID_gains = {'servoj': {'P': 1.0, 'I': 0.5, 'D': 0.2}}
    elif q_control_type == 'speedj':
        PID_gains = {'speedj': {'P': 0.2, 'I': 10.0}} # was 0.2, 5.0
    ur3_scale_factor = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0])*np.array([1.0, 1.0, 1.0, 2.5, 2.5, 2.5])
    gripper_scale_factor = np.array([1.0])
    g_control_type='move_gripper_force'
    #multi_step=5 # 1step : 0.01s -> 100Hz
    # multi_step=20 # 1step : 0.04s -> 25Hz
    multi_step=1 # 1step : 0.002s -> 500Hz
    gripper_action = True
    from gym_custom.envs.custom.dscho_dual_ur3_goal_env import DualWrapper
    env = DualWrapper(env=env,
                        q_control_type=q_control_type,
                        g_control_type=g_control_type,
                        multi_step=multi_step,
                        gripper_action=gripper_action,
                        PID_gains=PID_gains,
                        ur3_scale_factor=ur3_scale_factor,
                        gripper_scale_factor=gripper_scale_factor
                        )
    dt = env.dt
    print('dt : ', dt)

    from gym_custom.envs.custom.ur_utils import SO3Constraint
    null_obj_func = SO3Constraint(SO3='vertical_side')
    
    obs = env.reset()
    
    # for i in range(5000):
    #     action = np.zeros(14)
    #     action[6] = 10
    #     action[-1] = 10
    #     obs, _, _, _ = env.step(action)
    #     env.render()
    while True:
        # action = np.ones(14)*1
        action = np.zeros(14)
        obs, _, _, _ = env.step(action)
        if render: env.render()


    # ee_pos_right = np.array([0.1, -0.5, 0.9])
    # ee_pos_right = env.get_endeff_pos('right') + np.array([0. , 0., -0.1])
    # ee_pos_left = env.get_endeff_pos('left') + np.array([0. , 0., -0.1])
    ee_pos_right = env.get_site_pos('objSite') + np.array([0,0,0.1])
    ee_pos_left = env.get_site_pos('second_objSite') + np.array([0,0,0.1])
    
    q_right_des, iter_taken_right, err_right, null_obj_right = env.inverse_kinematics_ee(ee_pos_right, null_obj_func, arm='right')
    q_left_des, iter_taken_left, err_left, null_obj_left = env.inverse_kinematics_ee(ee_pos_left, null_obj_func, arm='left')
    print('q_right_des : {}, iter_taken : {}, err : {}, null_obj : {}'.format(q_right_des, iter_taken_right, err_right, null_obj_right))
    print('q_left_des : {}, iter_taken : {}, err : {}, null_obj : {}'.format(q_left_des, iter_taken_left, err_left, null_obj_left))

    # Move to goal
    duration = 3.0 # in seconds
    q_right_init = env._get_ur3_qpos()[:env.ur3_nqpos]
    q_left_init = env._get_ur3_qpos()[env.ur3_nqpos:]

 
    q_right_des_vel = (q_right_des - q_right_init)/duration
    q_left_des_vel = (q_left_des - q_left_init)/duration
    right_action = np.concatenate([q_right_des_vel, np.array([-10])])
    left_action = np.concatenate([q_left_des_vel, np.array([-10])])
    start = time.time()
    for t in range(int(duration/dt)):
        # if env.which_hand =='right':
        #     action = q_right_des_vel
        # elif env.which_hand =='left':
        #     action = q_left_des_vel
        action = np.concatenate([right_action, left_action])
        obs, _, _, _ = env.step(action)
        if render: env.render()
        # TODO: get_obs_dict() takes a long time causing timing issues.
        #   Is it due to Upboard's lackluster performance or some deeper
        #   issues within UR Script wrppaer?
        qpos_right = env._get_ur3_qpos()[:env.ur3_nqpos]
        qpos_left = env._get_ur3_qpos()[env.ur3_nqpos:]
        qvel_right = env._get_ur3_qvel()[:env.ur3_nqvel]
        qvel_left = env._get_ur3_qvel()[env.ur3_nqvel:]
        right_pos_err = np.linalg.norm(qpos_right - q_right_des)
        left_pos_err = np.linalg.norm(qpos_left - q_left_des)
        right_vel_err = np.linalg.norm(qvel_right - q_right_des_vel)
        left_vel_err = np.linalg.norm(qvel_left - q_left_des_vel)
        _, right_ee_pos, _ = env.forward_kinematics_ee(qpos_right, 'right')
        _, left_ee_pos, _ = env.forward_kinematics_ee(qpos_left, 'left')
        print('time: %f [s]'%(t*dt))
        print('right arm ee_pos : {} left arm ee_pos : {}'.format(right_ee_pos, left_ee_pos))
        print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(right_pos_err), np.rad2deg(right_vel_err)))
        print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(left_pos_err), np.rad2deg(left_vel_err)))
        print('objSite pos : {}, second_objSite pos : {}'.format(env.get_site_pos('objSite'),env.get_site_pos('second_objSite')))
        

    # Move to grasp
    ee_pos_right = env.get_site_pos('objSite') 
    ee_pos_left = env.get_site_pos('second_objSite')
    
    q_right_des, iter_taken_right, err_right, null_obj_right = env.inverse_kinematics_ee(ee_pos_right, null_obj_func, arm='right')
    q_left_des, iter_taken_left, err_left, null_obj_left = env.inverse_kinematics_ee(ee_pos_left, null_obj_func, arm='left')
    print('q_right_des : {}, iter_taken : {}, err : {}, null_obj : {}'.format(q_right_des, iter_taken_right, err_right, null_obj_right))
    print('q_left_des : {}, iter_taken : {}, err : {}, null_obj : {}'.format(q_left_des, iter_taken_left, err_left, null_obj_left))


    
    q_right_init = env._get_ur3_qpos()[:env.ur3_nqpos]
    q_left_init = env._get_ur3_qpos()[env.ur3_nqpos:]

 
    q_right_des_vel = (q_right_des - q_right_init)/duration
    q_left_des_vel = (q_left_des - q_left_init)/duration
    right_action = np.concatenate([q_right_des_vel, np.array([-10])])
    left_action = np.concatenate([q_left_des_vel, np.array([-10])])
    start = time.time()
    for t in range(int(duration/dt)):
        # if env.which_hand =='right':
        #     action = q_right_des_vel
        # elif env.which_hand =='left':
        #     action = q_left_des_vel
        action = np.concatenate([right_action, left_action])
        obs, _, _, _ = env.step(action)
        if render: env.render()
        # TODO: get_obs_dict() takes a long time causing timing issues.
        #   Is it due to Upboard's lackluster performance or some deeper
        #   issues within UR Script wrppaer?
        qpos_right = env._get_ur3_qpos()[:env.ur3_nqpos]
        qpos_left = env._get_ur3_qpos()[env.ur3_nqpos:]
        qvel_right = env._get_ur3_qvel()[:env.ur3_nqvel]
        qvel_left = env._get_ur3_qvel()[env.ur3_nqvel:]
        right_pos_err = np.linalg.norm(qpos_right - q_right_des)
        left_pos_err = np.linalg.norm(qpos_left - q_left_des)
        right_vel_err = np.linalg.norm(qvel_right - q_right_des_vel)
        left_vel_err = np.linalg.norm(qvel_left - q_left_des_vel)
        _, right_ee_pos, _ = env.forward_kinematics_ee(qpos_right, 'right')
        _, left_ee_pos, _ = env.forward_kinematics_ee(qpos_left, 'left')
        print('time: %f [s]'%(t*dt))
        print('right arm ee_pos : {} left arm ee_pos : {}'.format(right_ee_pos, left_ee_pos))
        print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(right_pos_err), np.rad2deg(right_vel_err)))
        print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(left_pos_err), np.rad2deg(left_vel_err)))


    # Grasp
    q_right_des_vel = np.zeros(6)
    q_left_des_vel = np.zeros(6)
    right_action = np.concatenate([q_right_des_vel, np.array([20])])
    left_action = np.concatenate([q_left_des_vel, np.array([20])])
    start = time.time()
    for t in range(int(duration/dt)):
        # if env.which_hand =='right':
        #     action = q_right_des_vel
        # elif env.which_hand =='left':
        #     action = q_left_des_vel
        action = np.concatenate([right_action, left_action])
        obs, _, _, _ = env.step(action)
        if render: env.render()
        # TODO: get_obs_dict() takes a long time causing timing issues.
        #   Is it due to Upboard's lackluster performance or some deeper
        #   issues within UR Script wrppaer?
        # qpos_right = env._get_ur3_qpos()[:env.ur3_nqpos]
        # qpos_left = env._get_ur3_qpos()[env.ur3_nqpos:]
        # qvel_right = env._get_ur3_qvel()[:env.ur3_nqvel]
        # qvel_left = env._get_ur3_qvel()[env.ur3_nqvel:]
        # right_pos_err = np.linalg.norm(qpos_right - q_right_des)
        # left_pos_err = np.linalg.norm(qpos_left - q_left_des)
        # right_vel_err = np.linalg.norm(qvel_right - q_right_des_vel)
        # left_vel_err = np.linalg.norm(qvel_left - q_left_des_vel)
        # _, right_ee_pos, _ = env.forward_kinematics_ee(qpos_right, 'right')
        # _, left_ee_pos, _ = env.forward_kinematics_ee(qpos_left, 'left')
        # print('time: %f [s]'%(t*dt))
        # print('right arm ee_pos : {} left arm ee_pos : {}'.format(right_ee_pos, left_ee_pos))
        # print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(right_pos_err), np.rad2deg(right_vel_err)))
        # print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(left_pos_err), np.rad2deg(left_vel_err)))

    # Pull
    right_qpos = env._get_ur3_qpos()[:env.ur3_nqpos]
    left_qpos = env._get_ur3_qpos()[env.ur3_nqpos:]
    ee_pos_right = env.get_endeff_pos('right')  + np.array([0, 0. ,0.15])
    ee_pos_left = env.get_endeff_pos('left')  + np.array([0, 0. ,0.15])
    
    q_right_des, iter_taken_right, err_right, null_obj_right = env.inverse_kinematics_ee(ee_pos_right, null_obj_func, arm='right')
    q_left_des, iter_taken_left, err_left, null_obj_left = env.inverse_kinematics_ee(ee_pos_left, null_obj_func, arm='left')
    print('q_right_des : {}, iter_taken : {}, err : {}, null_obj : {}'.format(q_right_des, iter_taken_right, err_right, null_obj_right))
    print('q_left_des : {}, iter_taken : {}, err : {}, null_obj : {}'.format(q_left_des, iter_taken_left, err_left, null_obj_left))

    q_right_des_vel = (q_right_des - right_qpos)/duration
    q_left_des_vel = (q_left_des - left_qpos)/duration
    right_action = np.concatenate([q_right_des_vel, np.array([10])])
    left_action = np.concatenate([q_left_des_vel, np.array([10])])
    start = time.time()
    for t in range(int(duration/dt)):
        # if env.which_hand =='right':
        #     action = q_right_des_vel
        # elif env.which_hand =='left':
        #     action = q_left_des_vel
        action = np.concatenate([right_action, left_action])
        obs, _, _, _ = env.step(action)
        if render: env.render()
        # TODO: get_obs_dict() takes a long time causing timing issues.
        #   Is it due to Upboard's lackluster performance or some deeper
        #   issues within UR Script wrppaer?
        qpos_right = env._get_ur3_qpos()[:env.ur3_nqpos]
        qpos_left = env._get_ur3_qpos()[env.ur3_nqpos:]
        qvel_right = env._get_ur3_qvel()[:env.ur3_nqvel]
        qvel_left = env._get_ur3_qvel()[env.ur3_nqvel:]
        right_pos_err = np.linalg.norm(qpos_right - q_right_des)
        left_pos_err = np.linalg.norm(qpos_left - q_left_des)
        right_vel_err = np.linalg.norm(qvel_right - q_right_des_vel)
        left_vel_err = np.linalg.norm(qvel_left - q_left_des_vel)
        _, right_ee_pos, _ = env.forward_kinematics_ee(qpos_right, 'right')
        _, left_ee_pos, _ = env.forward_kinematics_ee(qpos_left, 'left')
        print('time: %f [s]'%(t*dt))
        print('right arm ee_pos : {} left arm ee_pos : {}'.format(right_ee_pos, left_ee_pos))
        print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(right_pos_err), np.rad2deg(right_vel_err)))
        print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(left_pos_err), np.rad2deg(left_vel_err)))


    while True:
        env.render()


def dscho_single_arm_test(env_type='sim', render=False):
    list_of_env_types = ['sim', 'real']

    # env = gym_custom.make('dscho-dual-ur3-pickandplace-v0', initMode = None)
    
    q_control_type = 'speedj'
    if q_control_type == 'servoj':
        PID_gains = {'servoj': {'P': 1.0, 'I': 0.5, 'D': 0.2}}
    elif q_control_type == 'speedj':
        PID_gains = {'speedj': {'P': 0.2, 'I': 10.0}} # was 0.2, 5.0
    ur3_scale_factor = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0])*np.array([1.0, 1.0, 1.0, 2.5, 2.5, 2.5])
    gripper_scale_factor = np.array([1.0])
    g_control_type='move_gripper_force'
    gripper_action = False

    env = gym_custom.make('dscho-single-ur3-reach-obstacle-v0', initMode = None, which_hand='left')
    from gym_custom.envs.custom.dscho_dual_ur3_goal_env import SingleWrapper
    multi_step=20
    env = SingleWrapper(env=env,
                        q_control_type=q_control_type,
                        g_control_type=g_control_type,
                        multi_step=multi_step,
                        gripper_action=gripper_action,
                        PID_gains=PID_gains,
                        ur3_scale_factor=ur3_scale_factor,
                        gripper_scale_factor=gripper_scale_factor,
                        )
    
    dt = env.dt
    print('dt : ', dt)
    obs = env.reset()
    from gym_custom.envs.custom.ur_utils import SO3Constraint
    null_obj_func = SO3Constraint(SO3='vertical_side')

    # ee_pos_right = np.array([0.1, -0.5, 0.9])
    ee_pos_right = env.get_endeff_pos('right') + np.array([0. , 0., -0.1])
    ee_pos_left = env.get_endeff_pos('left') + np.array([0. , 0., -0.1])
    # ee_pos_right = env.get_site_pos('objSite')
    # ee_pos_left = env.get_site_pos('second_objSite')
    
    q_right_des, iter_taken_right, err_right, null_obj_right = env.inverse_kinematics_ee(ee_pos_right, null_obj_func, arm='right')
    q_left_des, iter_taken_left, err_left, null_obj_left = env.inverse_kinematics_ee(ee_pos_left, null_obj_func, arm='left')
    print('q_right_des : {}, iter_taken : {}, err : {}, null_obj : {}'.format(q_right_des, iter_taken_right, err_right, null_obj_right))
    print('q_left_des : {}, iter_taken : {}, err : {}, null_obj : {}'.format(q_left_des, iter_taken_left, err_left, null_obj_left))

    # # Move to goal
    duration = 6 # in seconds
    q_right_init = env._get_ur3_qpos()[:env.ur3_nqpos]
    q_left_init = env._get_ur3_qpos()[env.ur3_nqpos:]

 
    q_right_des_vel = (q_right_des - q_right_init)/duration
    q_left_des_vel = (q_left_des - q_left_init)/duration
    right_action = np.concatenate([q_right_des_vel, np.array([10])])
    left_action = np.concatenate([q_left_des_vel, np.array([10])])
    start = time.time()
    
        
    for t in range(int(duration/dt)):
        action = np.concatenate([right_action, left_action])
        action = left_action
        obs, _, _, _ = env.step(action)
        print('step : ',  t)
        obs, _, _, _ = env.step(action)
        if render: env.render()
        # TODO: get_obs_dict() takes a long time causing timing issues.
        #   Is it due to Upboard's lackluster performance or some deeper
        #   issues within UR Script wrppaer?
        qpos_right = env._get_ur3_qpos()[:env.ur3_nqpos]
        qpos_left = env._get_ur3_qpos()[env.ur3_nqpos:]
        qvel_right = env._get_ur3_qvel()[:env.ur3_nqvel]
        qvel_left = env._get_ur3_qvel()[env.ur3_nqvel:]
        right_pos_err = np.linalg.norm(qpos_right - q_right_des)
        left_pos_err = np.linalg.norm(qpos_left - q_left_des)
        right_vel_err = np.linalg.norm(qvel_right - q_right_des_vel)
        left_vel_err = np.linalg.norm(qvel_left - q_left_des_vel)
        _, right_ee_pos, _ = env.forward_kinematics_ee(qpos_right, 'right')
        _, left_ee_pos, _ = env.forward_kinematics_ee(qpos_left, 'left')
        print('time: %f [s]'%(t*dt))
        print('right arm ee_pos : {} left arm ee_pos : {}'.format(right_ee_pos, left_ee_pos))
        print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(right_pos_err), np.rad2deg(right_vel_err)))
        print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(left_pos_err), np.rad2deg(left_vel_err)))
    finish = time.time()

    while True:
        env.render()

def dscho_posxyz_v4_test(env_type='sim', render=False):
    list_of_env_types = ['sim', 'real']

    q_control_type = 'speedj'
    if q_control_type == 'servoj':
        PID_gains = {'servoj': {'P': 1.0, 'I': 0.5, 'D': 0.2}}
    elif q_control_type == 'speedj':
        PID_gains = {'speedj': {'P': 0.2, 'I': 10.0}} # was 0.2, 5.0
    ur3_scale_factor = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0])*np.array([1.0, 1.0, 1.0, 2.5, 2.5, 2.5])
    gripper_scale_factor = np.array([1.0])
    g_control_type='move_gripper_force'
    gripper_action = True
    
    from gym_custom.envs.custom.dscho_dual_ur3_goal_env import EndEffectorPositionControlSingleWrapper, EndEffectorPositionControlDualWrapper

    env_id = 'dscho-dual-ur3-bar-pickandplace-obstacle-v4'
    # env_id = 'dscho-dual-ur3-stick-pull-obstacle-v4'
    env = gym_custom.make(env_id, initMode = None, full_state_goal = False)
    multi_step=50 # 1step : 0.1s -> 10Hz
    env = EndEffectorPositionControlDualWrapper(env=env,
                        q_control_type=q_control_type,
                        g_control_type=g_control_type,
                        multi_step=multi_step,
                        gripper_action=gripper_action,
                        PID_gains=PID_gains,
                        ur3_scale_factor=ur3_scale_factor,
                        gripper_scale_factor=gripper_scale_factor,
                        so3_constraint='vertical_side',
                        action_downscale=0.01,
                        )
    dt = env.dt
    print('dt : ', dt)
    
    obs = env.reset()
    while True:
        env.render()
    if 'obstacle-v4' in env_id:
        pkl_name = 'init_qpos_obstacle-v4.pkl'
        right_obstacle_pos, left_obstacle_pos = env.get_obstacle_positions()
        
        # obstacle-v3 : "+-0.25 -0.4 0.65"
        # 8개 
        right_ee_pos_candidate = [
                                right_obstacle_pos + np.array([0.15, 0.0, -0.05]),
                                right_obstacle_pos + np.array([0.15, 0.0, 0.1]),
                                right_obstacle_pos + np.array([0.0, 0.0, 0.2]),
                                # right_obstacle_pos + np.array([-0.05, 0.1, 0.35]),
                                right_obstacle_pos + np.array([-0.2, 0.0, 0.1]),
                                right_obstacle_pos + np.array([-0.2, 0.0, -0.05]), #ee pos [0.15, 0.5, 0.7]쯤이 한계
                                
                                ]
        left_ee_pos_candidate = [
                                left_obstacle_pos + np.array([-0.15, 0.0, -0.05]),
                                left_obstacle_pos + np.array([-0.15, 0.0, 0.1]),                                
                                left_obstacle_pos + np.array([-0.0, 0.0, 0.2]),
                                # left_obstacle_pos + np.array([0.05, 0.1, 0.35]),
                                left_obstacle_pos + np.array([0.2, 0.0, 0.1]),
                                left_obstacle_pos + np.array([0.2, 0.0, -0.05]), 
                                ]
    else :
        raise NotImplementedError

    from gym_custom.envs.custom.ur_utils import SO3Constraint, NoConstraint
    null_obj_func = SO3Constraint(SO3='vertical_side')
    play = False
    if play :
        # if self.which_hand =='right':
        #     ee_low = np.array([-0.2, -0.6, 0.65])
        #     ee_high = np.array([0.5, -0.18, 1.05])
        
        # elif self.which_hand =='left':
        #     ee_low = np.array([-0.5, -0.6, 0.65])
        #     ee_high = np.array([0.2, -0.18, 1.05])
        
        idx = 4
        right_ee_pos_candidate.append(np.array([0.1,-0.35, 0.75]))
        left_ee_pos_candidate.append(np.array([-0.1,-0.35, 0.75]))

        q_right_des, iter_taken_right, err_right, null_obj_right = env.inverse_kinematics_ee(right_ee_pos_candidate[idx], null_obj_func, arm='right')
        q_left_des, iter_taken_left, err_left, null_obj_left = env.inverse_kinematics_ee(left_ee_pos_candidate[idx], null_obj_func, arm='left')
        print('q_right_des : {}, iter_taken : {}, err : {}, null_obj : {}'.format(q_right_des, iter_taken_right, err_right, null_obj_right))
        print('q_left_des : {}, iter_taken : {}, err : {}, null_obj : {}'.format(q_left_des, iter_taken_left, err_left, null_obj_left))
        print('goal : {}'.format(env.get_site_pos('goal')))
        init_qpos = env.init_qpos.copy()
        init_qvel = env.init_qvel.copy()
        q_right_des_candidates =q_right_des
        q_left_des_candidates = q_left_des
        init_qpos[0:env.ur3_nqpos] = q_right_des_candidates
        init_qpos[env.ur3_nqpos+env.gripper_nqpos:2*env.ur3_nqpos+env.gripper_nqpos] = q_left_des_candidates
        env.set_state(init_qpos, init_qvel)
        # for i in range(30):
        #     env.step(env.action_space.sample())
        while True:
            env.render()
            # env.reset()
            # env.step(env.action_space.sample())
            # print('goal : {}'.format(env.get_site_pos('goal')))

    else:
        q_right_des_list, q_left_des_list = [], []
        for idx in range(len(right_ee_pos_candidate)):
            q_right_des, iter_taken_right, err_right, null_obj_right = env.inverse_kinematics_ee(right_ee_pos_candidate[idx], null_obj_func, arm='right')
            q_left_des, iter_taken_left, err_left, null_obj_left = env.inverse_kinematics_ee(left_ee_pos_candidate[idx], null_obj_func, arm='left')
            print('q_right_des : {}, iter_taken : {}, err : {}, null_obj : {}'.format(q_right_des, iter_taken_right, err_right, null_obj_right))
            print('q_left_des : {}, iter_taken : {}, err : {}, null_obj : {}'.format(q_left_des, iter_taken_left, err_left, null_obj_left))
            q_right_des_list.append(q_right_des)
            q_left_des_list.append(q_left_des)
        candidates = dict(q_right_des = np.stack(q_right_des_list, axis =0), #[num_candidate, dim]
                        q_left_des = np.stack(q_left_des_list, axis =0), #[num_candidate, dim]
                        )
        import pickle
        pickle.dump(candidates, open(pkl_name, 'wb'))
        sys.exit()

    duration = 1.5 # in seconds
    start = time.time()
    # reach

    for i in range(5):
        
        for t in range(int(duration/dt)):
            if i>=0 and i<=1 :
                if single:
                    action = np.array([0.0, 0.0, 1.0, -5.0])
                else :
                    action = np.array([-0.0, 0.0, 1.0, -5.0, 0.0, 0.0, 1.0, -5.0])
            elif i>=2 and i<=3 :
                if single:
                    action = np.array([1.0, 0.0, -0.1, -5.0])
                else :
                    action = np.array([-1.0, 0.0, -0.3, -5.0, 1.0, 0.0, -0.3,-5.0])
            elif i>=3 and i<=4:

                right_grasp_point = env.get_site_pos('objSite')
                left_grasp_point = env.get_site_pos('second_objSite')
                right_ee_pos = env.get_endeff_pos('right')
                left_ee_pos = env.get_endeff_pos('left')
                right_from_ee_to_gp = right_grasp_point - right_ee_pos
                left_from_ee_to_gp = left_grasp_point - left_ee_pos
                
                if single:
                    # action = np.array([0.0, 0.1, -1.0, -5.0])
                    action = np.concatenate([right_from_ee_to_gp*10, np.array([-5.0])])
                else :
                    # action = np.array([0.0, 0.1, -1.0, -5.0, 0.0, 0.0, -1.0, -5.0])
                    action = np.concatenate([right_from_ee_to_gp*20, np.array([-5.0]),left_from_ee_to_gp*20, np.array([-5.0])])
            obs, _, _, _ = env.step(action)
            if render: env.render()
            # TODO: get_obs_dict() takes a long time causing timing issues.
            #   Is it due to Upboard's lackluster performance or some deeper
            #   issues within UR Script wrppaer?
            qpos_right = env._get_ur3_qpos()[:env.ur3_nqpos]
            qpos_left = env._get_ur3_qpos()[env.ur3_nqpos:]
            qvel_right = env._get_ur3_qvel()[:env.ur3_nqvel]
            qvel_left = env._get_ur3_qvel()[env.ur3_nqvel:]
            _, right_ee_pos, _ = env.forward_kinematics_ee(qpos_right, 'right')
            _, left_ee_pos, _ = env.forward_kinematics_ee(qpos_left, 'left')
            # print('time: %f [s]'%(t*dt))
            print('step : {}, right arm ee_pos : {} left arm ee_pos : {}'.format(t, right_ee_pos, left_ee_pos))
            # print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(right_pos_err), np.rad2deg(right_vel_err)))
            # print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(left_pos_err), np.rad2deg(left_vel_err)))
    
    # pick and lift
    for i in range(3):
        
        for t in range(int(duration/dt)):
            if i==0 or i==1 :
                if single:
                    action = np.array([0.0, 0.0, 0.0, 15.0])
                else :
                    action = np.array([0.0, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 15.0])
            elif i==2 :
                if single:
                    action = np.array([0.0, 0.0, 1.0, 15.0])
                else :
                    action = np.array([0.0, 0.0, 1.0, 15.0, 0.0, 0.0, 1.0, 15.0])
            
            obs, _, _, _ = env.step(action)
            if render: env.render()
            # TODO: get_obs_dict() takes a long time causing timing issues.
            #   Is it due to Upboard's lackluster performance or some deeper
            #   issues within UR Script wrppaer?
            qpos_right = env._get_ur3_qpos()[:env.ur3_nqpos]
            qpos_left = env._get_ur3_qpos()[env.ur3_nqpos:]
            qvel_right = env._get_ur3_qvel()[:env.ur3_nqvel]
            qvel_left = env._get_ur3_qvel()[env.ur3_nqvel:]
            _, right_ee_pos, _ = env.forward_kinematics_ee(qpos_right, 'right')
            _, left_ee_pos, _ = env.forward_kinematics_ee(qpos_left, 'left')
            # print('time: %f [s]'%(t*dt))
            print('step : {}, right arm ee_pos : {} left arm ee_pos : {}'.format(t, right_ee_pos, left_ee_pos))
            # print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(right_pos_err), np.rad2deg(right_vel_err)))
            # print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(left_pos_err), np.rad2deg(left_vel_err)))
    finish = time.time()

    while True:
        env.render()
def dscho_posxyz_v5_test(env_type='sim', render=False):
    list_of_env_types = ['sim', 'real']

    q_control_type = 'speedj'
    if q_control_type == 'servoj':
        PID_gains = {'servoj': {'P': 1.0, 'I': 0.5, 'D': 0.2}}
    elif q_control_type == 'speedj':
        PID_gains = {'speedj': {'P': 0.2, 'I': 10.0}} # was 0.2, 5.0
    ur3_scale_factor = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0])*np.array([1.0, 1.0, 1.0, 2.5, 2.5, 2.5])
    gripper_scale_factor = np.array([1.0])
    g_control_type='move_gripper_force'
    gripper_action = True
    
    from gym_custom.envs.custom.dscho_dual_ur3_goal_env import EndEffectorPositionControlSingleWrapper, EndEffectorPositionControlDualWrapper
    
    env_id = 'dscho-dual-ur3-bar-pickandplace-obstacle-v5'
    env = gym_custom.make(env_id, initMode = None)
    multi_step=50 # 1step : 0.1s -> 10Hz
    env = EndEffectorPositionControlDualWrapper(env=env,
                        q_control_type=q_control_type,
                        g_control_type=g_control_type,
                        multi_step=multi_step,
                        gripper_action=gripper_action,
                        PID_gains=PID_gains,
                        ur3_scale_factor=ur3_scale_factor,
                        gripper_scale_factor=gripper_scale_factor,
                        so3_constraint='vertical_side',
                        action_downscale=0.01,
                        )
    dt = env.dt
    print('dt : ', dt)
    
    obs = env.reset()
    
    if 'obstacle-v5' in env_id:
        pkl_name = 'init_qpos_obstacle-v5.pkl'
        right_obstacle_pos, left_obstacle_pos = env.get_obstacle_positions()
        
        # obstacle-v3 : "+-0.25 -0.4 0.65"
        # 8개 
        right_ee_pos_candidate = [right_obstacle_pos + np.array([0.15, 0.05, 0.1]),
                                right_obstacle_pos + np.array([0.15, 0.3, 0.1]),
                                right_obstacle_pos + np.array([-0.1, 0.3, 0.1]),
                                right_obstacle_pos + np.array([-0.2, 0.15, 0.1]),
                                right_obstacle_pos + np.array([-0.2, 0.05, 0.1]), #ee pos [0.15, 0.5, 0.7]쯤이 한계
                                ]
        left_ee_pos_candidate = [left_obstacle_pos + np.array([-0.15, 0.05, 0.1]),
                                left_obstacle_pos + np.array([-0.15, 0.3, 0.1]),
                                left_obstacle_pos + np.array([0.1, 0.3, 0.1]),
                                left_obstacle_pos + np.array([0.2, 0.15, 0.1]),
                                left_obstacle_pos + np.array([0.2, 0.05, 0.1]), 
                                ]
    else :
        raise NotImplementedError

    from gym_custom.envs.custom.ur_utils import SO3Constraint, NoConstraint
    null_obj_func = SO3Constraint(SO3='vertical_side')
    play = True
    if play:
        
        idx = -1
        # import joblib
        # data = joblib.load('init_qpos_obstacle-v5.pkl')
        # q_right_des = data['q_right_des'][idx]
        # q_left_des = data['q_left_des'][idx]
        #[0.5 , -0.5, 0.82]
        #[0.5 , -0.25, 0.82]
        right_ee_pos_candidate.append(np.array([0.5, -0.25, 0.82]))
        left_ee_pos_candidate.append(np.array([-0.5, -0.25, 0.82]))

        q_right_des, iter_taken_right, err_right, null_obj_right = env.inverse_kinematics_ee(right_ee_pos_candidate[idx], null_obj_func, arm='right')
        q_left_des, iter_taken_left, err_left, null_obj_left = env.inverse_kinematics_ee(left_ee_pos_candidate[idx], null_obj_func, arm='left')
        print('q_right_des : {}, iter_taken : {}, err : {}, null_obj : {}'.format(q_right_des, iter_taken_right, err_right, null_obj_right))
        print('q_left_des : {}, iter_taken : {}, err : {}, null_obj : {}'.format(q_left_des, iter_taken_left, err_left, null_obj_left))
        init_qpos = env.init_qpos.copy()
        init_qvel = env.init_qvel.copy()
        q_right_des_candidates =q_right_des
        q_left_des_candidates = q_left_des
        init_qpos[0:env.ur3_nqpos] = q_right_des_candidates
        init_qpos[env.ur3_nqpos+env.gripper_nqpos:2*env.ur3_nqpos+env.gripper_nqpos] = q_left_des_candidates
        env.set_state(init_qpos, init_qvel)
        print('right ee pos : {}, left ee pos : {}'.format(env.get_endeff_pos('right'), env.get_endeff_pos('left')))
        # for i in range(30):
        #     env.step(env.action_space.sample())
        while True:
            env.render()
    else :

        q_right_des_list, q_left_des_list = [], []
        for idx in range(len(right_ee_pos_candidate)):
            q_right_des, iter_taken_right, err_right, null_obj_right = env.inverse_kinematics_ee(right_ee_pos_candidate[idx], null_obj_func, arm='right')
            q_left_des, iter_taken_left, err_left, null_obj_left = env.inverse_kinematics_ee(left_ee_pos_candidate[idx], null_obj_func, arm='left')
            print('q_right_des : {}, iter_taken : {}, err : {}, null_obj : {}'.format(q_right_des, iter_taken_right, err_right, null_obj_right))
            print('q_left_des : {}, iter_taken : {}, err : {}, null_obj : {}'.format(q_left_des, iter_taken_left, err_left, null_obj_left))
            q_right_des_list.append(q_right_des)
            q_left_des_list.append(q_left_des)
        candidates = dict(q_right_des = np.stack(q_right_des_list, axis =0), #[num_candidate, dim]
                        q_left_des = np.stack(q_left_des_list, axis =0), #[num_candidate, dim]
                        )
        import pickle
        pickle.dump(candidates, open(pkl_name, 'wb'))
        sys.exit()



    duration = 1.5 # in seconds
    start = time.time()
    # reach

    for i in range(6):
        
        for t in range(int(duration/dt)):
            if i>=0 and i<=1:
                if single:
                    action = np.array([0.0, 0.0, 1.0, -5.0])
                else :
                    action = np.array([0.0, 1.0, 0.0, -5.0, 0.0, 1.0, 0.0, -5.0])
            elif i>=2 and i<=2 :
                if single:
                    action = np.array([1.0, 0.0, -0.5, -5.0])
                else :
                    action = np.array([-1.0, -0., 0.5,-5.0, 1.0, -0., 0.5,-5.0])
            elif i>=3 and i<=3 :
                if single:
                    action = np.array([1.0, 0.0, -0.5, -5.0])
                else :
                    action = np.array([-1.0, -1.0, 0.5,-5.0, 1.0, -1.0, 0.5,-5.0])
            elif i>=4 and i<=5:
                right_grasp_point = env.get_site_pos('objSite')
                left_grasp_point = env.get_site_pos('second_objSite')
                right_ee_pos = env.get_endeff_pos('right')
                left_ee_pos = env.get_endeff_pos('left')
                right_from_ee_to_gp = right_grasp_point - right_ee_pos
                left_from_ee_to_gp = left_grasp_point - left_ee_pos
                
                if single:
                    # action = np.array([0.0, 0.1, -1.0, -5.0])
                    action = np.concatenate([right_from_ee_to_gp*8, np.array([-5.0])])
                else :
                    # action = np.array([0.0, 0.1, -1.0, -5.0, 0.0, 0.0, -1.0, -5.0])
                    action = np.concatenate([right_from_ee_to_gp*8, np.array([-5.0]),left_from_ee_to_gp*8, np.array([-5.0])])
            obs, _, _, _ = env.step(action)
            if render: env.render()
            # TODO: get_obs_dict() takes a long time causing timing issues.
            #   Is it due to Upboard's lackluster performance or some deeper
            #   issues within UR Script wrppaer?
            qpos_right = env._get_ur3_qpos()[:env.ur3_nqpos]
            qpos_left = env._get_ur3_qpos()[env.ur3_nqpos:]
            qvel_right = env._get_ur3_qvel()[:env.ur3_nqvel]
            qvel_left = env._get_ur3_qvel()[env.ur3_nqvel:]
            _, right_ee_pos, _ = env.forward_kinematics_ee(qpos_right, 'right')
            _, left_ee_pos, _ = env.forward_kinematics_ee(qpos_left, 'left')
            # print('time: %f [s]'%(t*dt))
            print('step : {}, right arm ee_pos : {} left arm ee_pos : {}'.format(t, right_ee_pos, left_ee_pos))
            # print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(right_pos_err), np.rad2deg(right_vel_err)))
            # print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(left_pos_err), np.rad2deg(left_vel_err)))
    
    # pick and lift
    for i in range(3):
        
        for t in range(int(duration/dt)):
            if i==0 or i==1 :
                if single:
                    action = np.array([0.0, 0.0, 0.0, 15.0])
                else :
                    action = np.array([0.0, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 15.0])
            elif i==2 :
                if single:
                    action = np.array([0.0, 0.0, 1.0, 15.0])
                else :
                    action = np.array([0.0, 0.0, 1.0, 15.0, 0.0, 0.0, 1.0, 15.0])
            
            obs, _, _, _ = env.step(action)
            if render: env.render()
            # TODO: get_obs_dict() takes a long time causing timing issues.
            #   Is it due to Upboard's lackluster performance or some deeper
            #   issues within UR Script wrppaer?
            qpos_right = env._get_ur3_qpos()[:env.ur3_nqpos]
            qpos_left = env._get_ur3_qpos()[env.ur3_nqpos:]
            qvel_right = env._get_ur3_qvel()[:env.ur3_nqvel]
            qvel_left = env._get_ur3_qvel()[env.ur3_nqvel:]
            _, right_ee_pos, _ = env.forward_kinematics_ee(qpos_right, 'right')
            _, left_ee_pos, _ = env.forward_kinematics_ee(qpos_left, 'left')
            # print('time: %f [s]'%(t*dt))
            print('step : {}, right arm ee_pos : {} left arm ee_pos : {}'.format(t, right_ee_pos, left_ee_pos))
            # print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(right_pos_err), np.rad2deg(right_vel_err)))
            # print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(left_pos_err), np.rad2deg(left_vel_err)))
    finish = time.time()

    while True:
        env.render()


def dscho_posxyz_single_v4_v5_test(env_type='sim', render=False):
    list_of_env_types = ['sim', 'real']

    q_control_type = 'speedj'
    if q_control_type == 'servoj':
        PID_gains = {'servoj': {'P': 1.0, 'I': 0.5, 'D': 0.2}}
    elif q_control_type == 'speedj':
        PID_gains = {'speedj': {'P': 0.2, 'I': 10.0}} # was 0.2, 5.0
    ur3_scale_factor = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0])*np.array([1.0, 1.0, 1.0, 2.5, 2.5, 2.5])
    gripper_scale_factor = np.array([1.0])
    g_control_type='move_gripper_force'
    gripper_action = True
    
    from gym_custom.envs.custom.dscho_dual_ur3_goal_env import EndEffectorPositionControlSingleWrapper, EndEffectorPositionControlDualWrapper

    env_id = 'dscho-single-ur3-reach-obstacle-v5'
    env = gym_custom.make(env_id, initMode = None, which_hand='left', ur3_random_init=True, full_state_goal = False)
    multi_step=50 # 1step : 0.1s -> 10Hz
    env = EndEffectorPositionControlSingleWrapper(env=env,
                                                q_control_type=q_control_type,
                                                g_control_type=g_control_type,
                                                multi_step=multi_step,
                                                gripper_action=gripper_action,
                                                PID_gains=PID_gains,
                                                ur3_scale_factor=ur3_scale_factor,
                                                gripper_scale_factor=gripper_scale_factor,
                                                so3_constraint='vertical_side',
                                                action_downscale=0.01,
                                                )
    dt = env.dt
    print('dt : ', dt)
    
    obs = env.reset()
            

    from gym_custom.envs.custom.ur_utils import SO3Constraint, NoConstraint
    null_obj_func = SO3Constraint(SO3='vertical_side')
    play = True
    if play :
        # if self.which_hand =='right':
        #     ee_low = np.array([-0.2, -0.6, 0.65])
        #     ee_high = np.array([0.5, -0.18, 1.05])
        
        # elif self.which_hand =='left':
        #     ee_low = np.array([-0.5, -0.6, 0.65])
        #     ee_high = np.array([0.2, -0.18, 1.05])
        
        idx = 1
        # right_ee_pos_candidate.append(np.array([0.1,-0.45, 0.75]))
        # left_ee_pos_candidate.append(np.array([-0.1,-0.45, 0.75]))

        # q_right_des, iter_taken_right, err_right, null_obj_right = env.inverse_kinematics_ee(right_ee_pos_candidate[idx], null_obj_func, arm='right')
        # q_left_des, iter_taken_left, err_left, null_obj_left = env.inverse_kinematics_ee(left_ee_pos_candidate[idx], null_obj_func, arm='left')
        # print('q_right_des : {}, iter_taken : {}, err : {}, null_obj : {}'.format(q_right_des, iter_taken_right, err_right, null_obj_right))
        # print('q_left_des : {}, iter_taken : {}, err : {}, null_obj : {}'.format(q_left_des, iter_taken_left, err_left, null_obj_left))
        # print('goal : {}'.format(env.get_site_pos('goal')))
        # init_qpos = env.init_qpos.copy()
        # init_qvel = env.init_qvel.copy()
        # q_right_des_candidates =q_right_des
        # q_left_des_candidates = q_left_des
        # init_qpos[0:env.ur3_nqpos] = q_right_des_candidates
        # init_qpos[env.ur3_nqpos+env.gripper_nqpos:2*env.ur3_nqpos+env.gripper_nqpos] = q_left_des_candidates
        # env.set_state(init_qpos, init_qvel)
        while True:
            env.render()
            env.reset()
            # env.step(env.action_space.sample())
            print('goal : {}'.format(env.get_site_pos('goal')))

    
    duration = 1.5 # in seconds
    start = time.time()
    # reach

    for i in range(5):
        
        for t in range(int(duration/dt)):
            if i>=0 and i<=1 :
                if single:
                    action = np.array([0.0, 0.0, 1.0, -5.0])
                else :
                    action = np.array([-0.0, 0.0, 1.0, -5.0, 0.0, 0.0, 1.0, -5.0])
            elif i>=2 and i<=3 :
                if single:
                    action = np.array([1.0, 0.0, -0.1, -5.0])
                else :
                    action = np.array([-1.0, 0.0, -0.3, -5.0, 1.0, 0.0, -0.3,-5.0])
            elif i>=3 and i<=4:

                right_grasp_point = env.get_site_pos('objSite')
                left_grasp_point = env.get_site_pos('second_objSite')
                right_ee_pos = env.get_endeff_pos('right')
                left_ee_pos = env.get_endeff_pos('left')
                right_from_ee_to_gp = right_grasp_point - right_ee_pos
                left_from_ee_to_gp = left_grasp_point - left_ee_pos
                
                if single:
                    # action = np.array([0.0, 0.1, -1.0, -5.0])
                    action = np.concatenate([right_from_ee_to_gp*10, np.array([-5.0])])
                else :
                    # action = np.array([0.0, 0.1, -1.0, -5.0, 0.0, 0.0, -1.0, -5.0])
                    action = np.concatenate([right_from_ee_to_gp*20, np.array([-5.0]),left_from_ee_to_gp*20, np.array([-5.0])])
            obs, _, _, _ = env.step(action)
            if render: env.render()
            # TODO: get_obs_dict() takes a long time causing timing issues.
            #   Is it due to Upboard's lackluster performance or some deeper
            #   issues within UR Script wrppaer?
            qpos_right = env._get_ur3_qpos()[:env.ur3_nqpos]
            qpos_left = env._get_ur3_qpos()[env.ur3_nqpos:]
            qvel_right = env._get_ur3_qvel()[:env.ur3_nqvel]
            qvel_left = env._get_ur3_qvel()[env.ur3_nqvel:]
            _, right_ee_pos, _ = env.forward_kinematics_ee(qpos_right, 'right')
            _, left_ee_pos, _ = env.forward_kinematics_ee(qpos_left, 'left')
            # print('time: %f [s]'%(t*dt))
            print('step : {}, right arm ee_pos : {} left arm ee_pos : {}'.format(t, right_ee_pos, left_ee_pos))
            # print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(right_pos_err), np.rad2deg(right_vel_err)))
            # print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(left_pos_err), np.rad2deg(left_vel_err)))
    
    # pick and lift
    for i in range(3):
        
        for t in range(int(duration/dt)):
            if i==0 or i==1 :
                if single:
                    action = np.array([0.0, 0.0, 0.0, 15.0])
                else :
                    action = np.array([0.0, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 15.0])
            elif i==2 :
                if single:
                    action = np.array([0.0, 0.0, 1.0, 15.0])
                else :
                    action = np.array([0.0, 0.0, 1.0, 15.0, 0.0, 0.0, 1.0, 15.0])
            
            obs, _, _, _ = env.step(action)
            if render: env.render()
            # TODO: get_obs_dict() takes a long time causing timing issues.
            #   Is it due to Upboard's lackluster performance or some deeper
            #   issues within UR Script wrppaer?
            qpos_right = env._get_ur3_qpos()[:env.ur3_nqpos]
            qpos_left = env._get_ur3_qpos()[env.ur3_nqpos:]
            qvel_right = env._get_ur3_qvel()[:env.ur3_nqvel]
            qvel_left = env._get_ur3_qvel()[env.ur3_nqvel:]
            _, right_ee_pos, _ = env.forward_kinematics_ee(qpos_right, 'right')
            _, left_ee_pos, _ = env.forward_kinematics_ee(qpos_left, 'left')
            # print('time: %f [s]'%(t*dt))
            print('step : {}, right arm ee_pos : {} left arm ee_pos : {}'.format(t, right_ee_pos, left_ee_pos))
            # print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(right_pos_err), np.rad2deg(right_vel_err)))
            # print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(left_pos_err), np.rad2deg(left_vel_err)))
    finish = time.time()

    while True:
        env.render()

def dscho_posxyz_v1_test(env_type='sim', render=False):
    list_of_env_types = ['sim', 'real']

    q_control_type = 'speedj'
    if q_control_type == 'servoj':
        PID_gains = {'servoj': {'P': 1.0, 'I': 0.5, 'D': 0.2}}
    elif q_control_type == 'speedj':
        PID_gains = {'speedj': {'P': 0.2, 'I': 10.0}} # was 0.2, 5.0
    ur3_scale_factor = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0])*np.array([1.0, 1.0, 1.0, 2.5, 2.5, 2.5])
    gripper_scale_factor = np.array([1.0])
    g_control_type='move_gripper_force'
    gripper_action = True
    
    from gym_custom.envs.custom.dscho_dual_ur3_goal_env import EndEffectorPositionControlSingleWrapper, EndEffectorPositionControlDualWrapper

    env_id = 'dscho-dual-ur3-bar-pickandplace-obstacle-v1'
    # env_id = 'dscho-dual-ur3-stick-pull-obstacle-v4'
    env = gym_custom.make(env_id, initMode = None, full_state_goal = False)
    multi_step=50 # 1step : 0.1s -> 10Hz
    env = EndEffectorPositionControlDualWrapper(env=env,
                        q_control_type=q_control_type,
                        g_control_type=g_control_type,
                        multi_step=multi_step,
                        gripper_action=gripper_action,
                        PID_gains=PID_gains,
                        ur3_scale_factor=ur3_scale_factor,
                        gripper_scale_factor=gripper_scale_factor,
                        so3_constraint='vertical_side',
                        action_downscale=0.01,
                        )
    dt = env.dt
    print('dt : ', dt)
    
    obs = env.reset()
    while True:
        env.render()
    if 'obstacle-v1' in env_id:
        pkl_name = 'init_qpos_obstacle-v1.pkl'
        obstacle_pos = env.get_obstacle_positions()[0]
        
        # obstacle-v3 : "+-0.25 -0.4 0.65"
        # 8개 
        right_ee_pos_candidate = [
                                obstacle_pos + np.array([0.1, 0.1, -0.05]),
                                obstacle_pos + np.array([0.0, 0.1, -0.05]),
                                obstacle_pos + np.array([0.0, 0.0, 0.2]),
                                obstacle_pos + np.array([0.05, -0.1, 0.05]),
                                obstacle_pos + np.array([0.1, -0.1, 0.05]), #ee pos [0.15, 0.5, 0.7]쯤이 한계
                                
                                ]
        left_ee_pos_candidate = [
                                obstacle_pos + np.array([-0.1, 0.1, -0.05]),
                                obstacle_pos + np.array([-0.0, 0.1, -0.05]),                                
                                obstacle_pos + np.array([-0.0, 0.0, 0.2]),
                                obstacle_pos + np.array([-0.05, -0.1, 0.05]),
                                obstacle_pos + np.array([-0.1, -0.1, 0.05]), 
                                ]
    else :
        raise NotImplementedError
    while True:
        env.render()
    from gym_custom.envs.custom.ur_utils import SO3Constraint, NoConstraint
    null_obj_func = SO3Constraint(SO3='vertical_side')
    play = True
    if play :
        # if self.which_hand =='right':
        #     ee_low = np.array([-0.2, -0.6, 0.65])
        #     ee_high = np.array([0.5, -0.18, 1.05])
        
        # elif self.which_hand =='left':
        #     ee_low = np.array([-0.5, -0.6, 0.65])
        #     ee_high = np.array([0.2, -0.18, 1.05])
        
        idx = 0
        import joblib
        data = joblib.load('init_qpos_obstacle-v1.pkl')
        q_right_des = data['q_right_des'][idx]
        q_left_des = data['q_left_des'][idx]
        
        # right_ee_pos_candidate.append(np.array([0.1,-0.35, 0.75]))
        # left_ee_pos_candidate.append(np.array([-0.1,-0.35, 0.75]))

        # q_right_des, iter_taken_right, err_right, null_obj_right = env.inverse_kinematics_ee(right_ee_pos_candidate[idx], null_obj_func, arm='right')
        # q_left_des, iter_taken_left, err_left, null_obj_left = env.inverse_kinematics_ee(left_ee_pos_candidate[idx], null_obj_func, arm='left')
        # print('q_right_des : {}, iter_taken : {}, err : {}, null_obj : {}'.format(q_right_des, iter_taken_right, err_right, null_obj_right))
        # print('q_left_des : {}, iter_taken : {}, err : {}, null_obj : {}'.format(q_left_des, iter_taken_left, err_left, null_obj_left))
        # print('goal : {}'.format(env.get_site_pos('goal')))
        init_qpos = env.init_qpos.copy()
        init_qvel = env.init_qvel.copy()
        q_right_des_candidates =q_right_des
        q_left_des_candidates = q_left_des
        init_qpos[0:env.ur3_nqpos] = q_right_des_candidates
        init_qpos[env.ur3_nqpos+env.gripper_nqpos:2*env.ur3_nqpos+env.gripper_nqpos] = q_left_des_candidates
        env.set_state(init_qpos, init_qvel)
        # for i in range(30):
        #     env.step(env.action_space.sample())
        while True:
            env.render()
            # env.reset()
            # env.step(env.action_space.sample())
            # print('goal : {}'.format(env.get_site_pos('goal')))

    else:
        q_right_des_list, q_left_des_list = [], []
        for idx in range(len(right_ee_pos_candidate)):
            q_right_des, iter_taken_right, err_right, null_obj_right = env.inverse_kinematics_ee(right_ee_pos_candidate[idx], null_obj_func, arm='right')
            q_left_des, iter_taken_left, err_left, null_obj_left = env.inverse_kinematics_ee(left_ee_pos_candidate[idx], null_obj_func, arm='left')
            print('q_right_des : {}, iter_taken : {}, err : {}, null_obj : {}'.format(q_right_des, iter_taken_right, err_right, null_obj_right))
            print('q_left_des : {}, iter_taken : {}, err : {}, null_obj : {}'.format(q_left_des, iter_taken_left, err_left, null_obj_left))
            q_right_des_list.append(q_right_des)
            q_left_des_list.append(q_left_des)
        candidates = dict(q_right_des = np.stack(q_right_des_list, axis =0), #[num_candidate, dim]
                        q_left_des = np.stack(q_left_des_list, axis =0), #[num_candidate, dim]
                        )
        import pickle
        pickle.dump(candidates, open(pkl_name, 'wb'))
        sys.exit()

    duration = 1.5 # in seconds
    start = time.time()
    # reach

    for i in range(5):
        
        for t in range(int(duration/dt)):
            if i>=0 and i<=1 :
                if single:
                    action = np.array([0.0, 0.0, 1.0, -5.0])
                else :
                    action = np.array([-0.0, 0.0, 1.0, -5.0, 0.0, 0.0, 1.0, -5.0])
            elif i>=2 and i<=3 :
                if single:
                    action = np.array([1.0, 0.0, -0.1, -5.0])
                else :
                    action = np.array([-1.0, 0.0, -0.3, -5.0, 1.0, 0.0, -0.3,-5.0])
            elif i>=3 and i<=4:

                right_grasp_point = env.get_site_pos('objSite')
                left_grasp_point = env.get_site_pos('second_objSite')
                right_ee_pos = env.get_endeff_pos('right')
                left_ee_pos = env.get_endeff_pos('left')
                right_from_ee_to_gp = right_grasp_point - right_ee_pos
                left_from_ee_to_gp = left_grasp_point - left_ee_pos
                
                if single:
                    # action = np.array([0.0, 0.1, -1.0, -5.0])
                    action = np.concatenate([right_from_ee_to_gp*10, np.array([-5.0])])
                else :
                    # action = np.array([0.0, 0.1, -1.0, -5.0, 0.0, 0.0, -1.0, -5.0])
                    action = np.concatenate([right_from_ee_to_gp*20, np.array([-5.0]),left_from_ee_to_gp*20, np.array([-5.0])])
            obs, _, _, _ = env.step(action)
            if render: env.render()
            # TODO: get_obs_dict() takes a long time causing timing issues.
            #   Is it due to Upboard's lackluster performance or some deeper
            #   issues within UR Script wrppaer?
            qpos_right = env._get_ur3_qpos()[:env.ur3_nqpos]
            qpos_left = env._get_ur3_qpos()[env.ur3_nqpos:]
            qvel_right = env._get_ur3_qvel()[:env.ur3_nqvel]
            qvel_left = env._get_ur3_qvel()[env.ur3_nqvel:]
            _, right_ee_pos, _ = env.forward_kinematics_ee(qpos_right, 'right')
            _, left_ee_pos, _ = env.forward_kinematics_ee(qpos_left, 'left')
            # print('time: %f [s]'%(t*dt))
            print('step : {}, right arm ee_pos : {} left arm ee_pos : {}'.format(t, right_ee_pos, left_ee_pos))
            # print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(right_pos_err), np.rad2deg(right_vel_err)))
            # print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(left_pos_err), np.rad2deg(left_vel_err)))
    
    # pick and lift
    for i in range(3):
        
        for t in range(int(duration/dt)):
            if i==0 or i==1 :
                if single:
                    action = np.array([0.0, 0.0, 0.0, 15.0])
                else :
                    action = np.array([0.0, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 15.0])
            elif i==2 :
                if single:
                    action = np.array([0.0, 0.0, 1.0, 15.0])
                else :
                    action = np.array([0.0, 0.0, 1.0, 15.0, 0.0, 0.0, 1.0, 15.0])
            
            obs, _, _, _ = env.step(action)
            if render: env.render()
            # TODO: get_obs_dict() takes a long time causing timing issues.
            #   Is it due to Upboard's lackluster performance or some deeper
            #   issues within UR Script wrppaer?
            qpos_right = env._get_ur3_qpos()[:env.ur3_nqpos]
            qpos_left = env._get_ur3_qpos()[env.ur3_nqpos:]
            qvel_right = env._get_ur3_qvel()[:env.ur3_nqvel]
            qvel_left = env._get_ur3_qvel()[env.ur3_nqvel:]
            _, right_ee_pos, _ = env.forward_kinematics_ee(qpos_right, 'right')
            _, left_ee_pos, _ = env.forward_kinematics_ee(qpos_left, 'left')
            # print('time: %f [s]'%(t*dt))
            print('step : {}, right arm ee_pos : {} left arm ee_pos : {}'.format(t, right_ee_pos, left_ee_pos))
            # print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(right_pos_err), np.rad2deg(right_vel_err)))
            # print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(left_pos_err), np.rad2deg(left_vel_err)))
    finish = time.time()

    while True:
        env.render()

def dscho_posxyz_v2_test(env_type='sim', render=False):
    list_of_env_types = ['sim', 'real']

    q_control_type = 'speedj'
    if q_control_type == 'servoj':
        PID_gains = {'servoj': {'P': 1.0, 'I': 0.5, 'D': 0.2}}
    elif q_control_type == 'speedj':
        PID_gains = {'speedj': {'P': 0.2, 'I': 10.0}} # was 0.2, 5.0
    ur3_scale_factor = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0])*np.array([1.0, 1.0, 1.0, 2.5, 2.5, 2.5])
    gripper_scale_factor = np.array([1.0])
    g_control_type='move_gripper_force'
    gripper_action = True
    
    from gym_custom.envs.custom.dscho_dual_ur3_goal_env import EndEffectorPositionControlSingleWrapper, EndEffectorPositionControlDualWrapper

    env_id = 'dscho-single-ur3-reach-obstacle-v2'
    # env_id = 'dscho-dual-ur3-stick-pull-obstacle-v4'
    env = gym_custom.make(env_id, initMode = None, full_state_goal = False)
    multi_step=50 # 1step : 0.1s -> 10Hz
    env = EndEffectorPositionControlDualWrapper(env=env,
                        q_control_type=q_control_type,
                        g_control_type=g_control_type,
                        multi_step=multi_step,
                        gripper_action=gripper_action,
                        PID_gains=PID_gains,
                        ur3_scale_factor=ur3_scale_factor,
                        gripper_scale_factor=gripper_scale_factor,
                        so3_constraint='vertical_side',
                        action_downscale=0.005,
                        )
    dt = env.dt
    print('dt : ', dt)
    
    obs = env.reset()
    if 'obstacle-v2' in env_id:
        pkl_name = 'init_qpos_obstacle-v2.pkl'
        right_obstacle_pos, left_obstacle_pos = env.get_obstacle_positions()
        
        # obstacle-v3 : "+-0.25 -0.4 0.65"
        # 8개 
        right_ee_pos_candidate = [
                                right_obstacle_pos + np.array([0.1, 0.1, -0.05]),
                                right_obstacle_pos + np.array([0.0, 0.15, -0.05]),
                                right_obstacle_pos + np.array([-0.15, 0.0, -0.05]),
                                right_obstacle_pos + np.array([-0.2, -0.1, -0.05]),
                                right_obstacle_pos + np.array([0.15, -0.1, 0.05]), #ee pos [0.15, 0.5, 0.7]쯤이 한계
                                
                                ]
        left_ee_pos_candidate = [
                                left_obstacle_pos + np.array([-0.1, 0.1, -0.05]),
                                left_obstacle_pos + np.array([-0.0, 0.15, -0.05]),                                
                                left_obstacle_pos + np.array([0.15, 0.0, -0.05]),
                                left_obstacle_pos + np.array([0.2, -0.1, -0.05]),
                                left_obstacle_pos + np.array([-0.15, -0.1, 0.05]), 
                                ]
    else :
        raise NotImplementedError

    from gym_custom.envs.custom.ur_utils import SO3Constraint, NoConstraint
    null_obj_func = SO3Constraint(SO3='vertical_side')
    play = True
    if play :
        # if self.which_hand =='right':
        #     ee_low = np.array([-0.2, -0.6, 0.65])
        #     ee_high = np.array([0.5, -0.18, 1.05])
        
        # elif self.which_hand =='left':
        #     ee_low = np.array([-0.5, -0.6, 0.65])
        #     ee_high = np.array([0.2, -0.18, 1.05])
        
        idx = 4
        # right_ee_pos_candidate.append(np.array([0.1,-0.35, 0.75]))
        # left_ee_pos_candidate.append(np.array([-0.1,-0.35, 0.75]))

        q_right_des, iter_taken_right, err_right, null_obj_right = env.inverse_kinematics_ee(right_ee_pos_candidate[idx], null_obj_func, arm='right')
        q_left_des, iter_taken_left, err_left, null_obj_left = env.inverse_kinematics_ee(left_ee_pos_candidate[idx], null_obj_func, arm='left')
        print('q_right_des : {}, iter_taken : {}, err : {}, null_obj : {}'.format(q_right_des, iter_taken_right, err_right, null_obj_right))
        print('q_left_des : {}, iter_taken : {}, err : {}, null_obj : {}'.format(q_left_des, iter_taken_left, err_left, null_obj_left))
        print('goal : {}'.format(env.get_site_pos('goal')))
        init_qpos = env.init_qpos.copy()
        init_qvel = env.init_qvel.copy()
        q_right_des_candidates =q_right_des
        q_left_des_candidates = q_left_des
        init_qpos[0:env.ur3_nqpos] = q_right_des_candidates
        init_qpos[env.ur3_nqpos+env.gripper_nqpos:2*env.ur3_nqpos+env.gripper_nqpos] = q_left_des_candidates
        env.set_state(init_qpos, init_qvel)
        print('right ee pos : {}, left ee pos : {}'.format(env.get_endeff_pos('right'), env.get_endeff_pos('left')))
        # for i in range(30):
        #     env.step(env.action_space.sample())
        while True:
            env.render()
            # env.reset()
            # env.step(env.action_space.sample())
            # print('goal : {}'.format(env.get_site_pos('goal')))

    else:
        q_right_des_list, q_left_des_list = [], []
        for idx in range(len(right_ee_pos_candidate)):
            q_right_des, iter_taken_right, err_right, null_obj_right = env.inverse_kinematics_ee(right_ee_pos_candidate[idx], null_obj_func, arm='right')
            q_left_des, iter_taken_left, err_left, null_obj_left = env.inverse_kinematics_ee(left_ee_pos_candidate[idx], null_obj_func, arm='left')
            print('q_right_des : {}, iter_taken : {}, err : {}, null_obj : {}'.format(q_right_des, iter_taken_right, err_right, null_obj_right))
            print('q_left_des : {}, iter_taken : {}, err : {}, null_obj : {}'.format(q_left_des, iter_taken_left, err_left, null_obj_left))
            q_right_des_list.append(q_right_des)
            q_left_des_list.append(q_left_des)
        candidates = dict(q_right_des = np.stack(q_right_des_list, axis =0), #[num_candidate, dim]
                        q_left_des = np.stack(q_left_des_list, axis =0), #[num_candidate, dim]
                        )
        import pickle
        pickle.dump(candidates, open(pkl_name, 'wb'))
        sys.exit()

    duration = 1.5 # in seconds
    start = time.time()
    # reach

    for i in range(5):
        
        for t in range(int(duration/dt)):
            if i>=0 and i<=1 :
                if single:
                    action = np.array([0.0, 0.0, 1.0, -5.0])
                else :
                    action = np.array([-0.0, 0.0, 1.0, -5.0, 0.0, 0.0, 1.0, -5.0])
            elif i>=2 and i<=3 :
                if single:
                    action = np.array([1.0, 0.0, -0.1, -5.0])
                else :
                    action = np.array([-1.0, 0.0, -0.3, -5.0, 1.0, 0.0, -0.3,-5.0])
            elif i>=3 and i<=4:

                right_grasp_point = env.get_site_pos('objSite')
                left_grasp_point = env.get_site_pos('second_objSite')
                right_ee_pos = env.get_endeff_pos('right')
                left_ee_pos = env.get_endeff_pos('left')
                right_from_ee_to_gp = right_grasp_point - right_ee_pos
                left_from_ee_to_gp = left_grasp_point - left_ee_pos
                
                if single:
                    # action = np.array([0.0, 0.1, -1.0, -5.0])
                    action = np.concatenate([right_from_ee_to_gp*10, np.array([-5.0])])
                else :
                    # action = np.array([0.0, 0.1, -1.0, -5.0, 0.0, 0.0, -1.0, -5.0])
                    action = np.concatenate([right_from_ee_to_gp*20, np.array([-5.0]),left_from_ee_to_gp*20, np.array([-5.0])])
            obs, _, _, _ = env.step(action)
            if render: env.render()
            # TODO: get_obs_dict() takes a long time causing timing issues.
            #   Is it due to Upboard's lackluster performance or some deeper
            #   issues within UR Script wrppaer?
            qpos_right = env._get_ur3_qpos()[:env.ur3_nqpos]
            qpos_left = env._get_ur3_qpos()[env.ur3_nqpos:]
            qvel_right = env._get_ur3_qvel()[:env.ur3_nqvel]
            qvel_left = env._get_ur3_qvel()[env.ur3_nqvel:]
            _, right_ee_pos, _ = env.forward_kinematics_ee(qpos_right, 'right')
            _, left_ee_pos, _ = env.forward_kinematics_ee(qpos_left, 'left')
            # print('time: %f [s]'%(t*dt))
            print('step : {}, right arm ee_pos : {} left arm ee_pos : {}'.format(t, right_ee_pos, left_ee_pos))
            # print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(right_pos_err), np.rad2deg(right_vel_err)))
            # print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(left_pos_err), np.rad2deg(left_vel_err)))
    
    # pick and lift
    for i in range(3):
        
        for t in range(int(duration/dt)):
            if i==0 or i==1 :
                if single:
                    action = np.array([0.0, 0.0, 0.0, 15.0])
                else :
                    action = np.array([0.0, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 15.0])
            elif i==2 :
                if single:
                    action = np.array([0.0, 0.0, 1.0, 15.0])
                else :
                    action = np.array([0.0, 0.0, 1.0, 15.0, 0.0, 0.0, 1.0, 15.0])
            
            obs, _, _, _ = env.step(action)
            if render: env.render()
            # TODO: get_obs_dict() takes a long time causing timing issues.
            #   Is it due to Upboard's lackluster performance or some deeper
            #   issues within UR Script wrppaer?
            qpos_right = env._get_ur3_qpos()[:env.ur3_nqpos]
            qpos_left = env._get_ur3_qpos()[env.ur3_nqpos:]
            qvel_right = env._get_ur3_qvel()[:env.ur3_nqvel]
            qvel_left = env._get_ur3_qvel()[env.ur3_nqvel:]
            _, right_ee_pos, _ = env.forward_kinematics_ee(qpos_right, 'right')
            _, left_ee_pos, _ = env.forward_kinematics_ee(qpos_left, 'left')
            # print('time: %f [s]'%(t*dt))
            print('step : {}, right arm ee_pos : {} left arm ee_pos : {}'.format(t, right_ee_pos, left_ee_pos))
            # print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(right_pos_err), np.rad2deg(right_vel_err)))
            # print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(left_pos_err), np.rad2deg(left_vel_err)))
    finish = time.time()

    while True:
        env.render()

def dscho_init_qpos_candidate_pickling(env_type='sim', render=False):
    list_of_env_types = ['sim', 'real']
    env_id = 'dscho-dual-ur3-bar-obstacle-v3'
    # env = gym_custom.make('dscho-dual-ur3-cylinder-pickandplace-v0', ur3_random_init=True) # ur3_random_init =True 하면, initMode 설정하든말든 무의미
    env = gym_custom.make(env_id, initMode = None)
    # env = gym_custom.make('dscho-dual-ur3-stick-pull-obstacle-v0', initMode = None)    
    # env = gym_custom.make('dscho-dual-ur3-cylinder-pickandplace-obstacle-v0', initMode = None)    
                
    q_control_type = 'speedj'
    if q_control_type == 'servoj':
        PID_gains = {'servoj': {'P': 1.0, 'I': 0.5, 'D': 0.2}}
    elif q_control_type == 'speedj':
        PID_gains = {'speedj': {'P': 0.2, 'I': 10.0}} # was 0.2, 5.0
    ur3_scale_factor = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0])*np.array([1.0, 1.0, 1.0, 2.5, 2.5, 2.5])
    gripper_scale_factor = np.array([1.0])
    g_control_type='move_gripper_force'
    #multi_step=5 # 1step : 0.01s -> 100Hz
    # multi_step=20 # 1step : 0.04s -> 25Hz
    multi_step=1 # 1step : 0.002s -> 500Hz
    gripper_action = True
    from gym_custom.envs.custom.dscho_dual_ur3_goal_env import DualWrapper
    env = DualWrapper(env=env,
                        q_control_type=q_control_type,
                        g_control_type=g_control_type,
                        multi_step=multi_step,
                        gripper_action=gripper_action,
                        PID_gains=PID_gains,
                        ur3_scale_factor=ur3_scale_factor,
                        gripper_scale_factor=gripper_scale_factor
                        )
    dt = env.dt
    print('dt : ', dt)

    from gym_custom.envs.custom.ur_utils import SO3Constraint
    null_obj_func = SO3Constraint(SO3='vertical_side')
    
    obs = env.reset()
    if 'obstacle-v0' in env_id or 'obstacle-v2' in env_id :
        if 'obstacle-v0' in env_id :
            pkl_name = 'init_qpos_obstacle-v0.pkl'
        elif 'obstacle-v2' in env_id :
            pkl_name = 'init_qpos_obstacle-v2.pkl'
        right_obstacle_pos, left_obstacle_pos = env.get_obstacle_positions()
        
        # obstacle-v0, v2 : "+-0.25 -0.4 0.65"
        # 8개 
        right_ee_pos_candidate = [right_obstacle_pos + np.array([0.1, 0, 0.1]),
                                right_obstacle_pos + np.array([0.1, 0.1, 0.1]),
                                right_obstacle_pos + np.array([0.0, 0.1, 0.1]),
                                right_obstacle_pos + np.array([-0.1, 0.1, 0.1]),
                                right_obstacle_pos + np.array([-0.1, 0., 0.1]),
                                right_obstacle_pos + np.array([-0.1, -0.1, 0.1]),
                                right_obstacle_pos + np.array([0., -0.15, 0.1]),
                                right_obstacle_pos + np.array([0.1, -0.1, 0.1]),
                                ]
        left_ee_pos_candidate = [left_obstacle_pos + np.array([0.1, 0, 0.1]),
                                left_obstacle_pos + np.array([0.1, 0.1, 0.1]),
                                left_obstacle_pos + np.array([0.0, 0.1, 0.1]),
                                left_obstacle_pos + np.array([-0.1, 0.1, 0.1]),
                                left_obstacle_pos + np.array([-0.1, 0., 0.1]),
                                left_obstacle_pos + np.array([-0.1, -0.1, 0.1]),
                                left_obstacle_pos + np.array([0., -0.15, 0.1]),
                                left_obstacle_pos + np.array([0.1, -0.1, 0.1]),
                                ]
    elif 'obstacle-v1' in env_id:
        
        pkl_name = 'init_qpos_obstacle-v1.pkl'
        obstacle_pos = env.get_obstacle_positions()[0] #list
        
        # obstacle-v1 : "0.0 -0.4 0.65"
        # 7개 
        right_ee_pos_candidate = [obstacle_pos + np.array([0.3, 0, 0.1]),
                                obstacle_pos + np.array([0.3, -0.1, 0.1]),
                                obstacle_pos + np.array([0.2, -0.1, 0.1]),
                                obstacle_pos + np.array([0.1, -0.1, 0.1]),
                                obstacle_pos + np.array([0.3, 0.1, 0.1]),
                                obstacle_pos + np.array([0.2, 0.1, 0.1]),
                                obstacle_pos + np.array([0.1, 0.1, 0.1]),
                                # obstacle_pos + np.array([0.1, -0.1, 0.1]),
                                ]
        left_ee_pos_candidate = [obstacle_pos + np.array([-0.3, 0, 0.1]),
                                 obstacle_pos + np.array([-0.3, -0.1, 0.1]),
                                 obstacle_pos + np.array([-0.2, -0.1, 0.1]),
                                 obstacle_pos + np.array([-0.1, -0.1, 0.1]),
                                 obstacle_pos + np.array([-0.3, 0.1, 0.1]),
                                 obstacle_pos + np.array([-0.2, 0.1, 0.1]),
                                 obstacle_pos + np.array([-0.1, 0.1, 0.1]),
                                #  obstacle_pos + np.array([0.1, -0.1, 0.1]),
                                ]
    elif 'obstacle-v3' in env_id:
        pkl_name = 'init_qpos_obstacle-v3.pkl'
        right_obstacle_pos, left_obstacle_pos = env.get_obstacle_positions()
        
        # obstacle-v3 : "+-0.25 -0.4 0.65"
        # 8개 
        right_ee_pos_candidate = [right_obstacle_pos + np.array([0.1, 0, 0.1]),
                                right_obstacle_pos + np.array([0.1, 0.1, 0.1]),
                                right_obstacle_pos + np.array([0.0, 0.2, 0.1]),
                                right_obstacle_pos + np.array([-0.1, 0.1, 0.1]),
                                right_obstacle_pos + np.array([-0.1, 0., 0.1]),
                                right_obstacle_pos + np.array([-0.1, -0.1, 0.1]),
                                right_obstacle_pos + np.array([0., -0.2, 0.15]),
                                right_obstacle_pos + np.array([0.1, -0.1, 0.1]),
                                ]
        left_ee_pos_candidate = [left_obstacle_pos + np.array([0.1, 0, 0.1]),
                                left_obstacle_pos + np.array([0.1, 0.1, 0.1]),
                                left_obstacle_pos + np.array([0.0, 0.2, 0.1]),
                                left_obstacle_pos + np.array([-0.1, 0.1, 0.1]),
                                left_obstacle_pos + np.array([-0.1, 0., 0.1]),
                                left_obstacle_pos + np.array([-0.1, -0.1, 0.1]),
                                left_obstacle_pos + np.array([0., -0.2, 0.15]),
                                left_obstacle_pos + np.array([0.1, -0.1, 0.1]),
                                ]
    else :
        raise NotImplementedError

    q_right_des_list, q_left_des_list = [], []
    for idx in range(len(right_ee_pos_candidate)):
        q_right_des, iter_taken_right, err_right, null_obj_right = env.inverse_kinematics_ee(right_ee_pos_candidate[idx], null_obj_func, arm='right')
        q_left_des, iter_taken_left, err_left, null_obj_left = env.inverse_kinematics_ee(left_ee_pos_candidate[idx], null_obj_func, arm='left')
        print('q_right_des : {}, iter_taken : {}, err : {}, null_obj : {}'.format(q_right_des, iter_taken_right, err_right, null_obj_right))
        print('q_left_des : {}, iter_taken : {}, err : {}, null_obj : {}'.format(q_left_des, iter_taken_left, err_left, null_obj_left))
        q_right_des_list.append(q_right_des)
        q_left_des_list.append(q_left_des)
    candidates = dict(q_right_des = np.stack(q_right_des_list, axis =0), #[num_candidate, dim]
                    q_left_des = np.stack(q_left_des_list, axis =0), #[num_candidate, dim]
                    )
    import pickle
    pickle.dump(candidates, open(pkl_name, 'wb'))
    sys.exit()

    init_qpos = env.init_qpos.copy()
    init_qvel = env.init_qvel.copy()
    
    init_qpos[0:env.ur3_nqpos] = q_right_des
    init_qpos[env.ur3_nqpos+env.gripper_nqpos:2*env.ur3_nqpos+env.gripper_nqpos] = q_left_des
    
    env.set_state(init_qpos, init_qvel)
    
    while True:
        # action = np.ones(14)*1
        action = np.zeros(14)
        obs, _, _, _ = env.step(action)
        if render: env.render()

def dscho_single_ur3_object_test(env_type='sim', render=False, make_video = False):
    list_of_env_types = ['sim', 'real']

    q_control_type = 'speedj'
    if q_control_type == 'servoj':
        PID_gains = {'servoj': {'P': 1.0, 'I': 0.5, 'D': 0.2}}
    elif q_control_type == 'speedj':
        PID_gains = {'speedj': {'P': 0.2, 'I': 10.0}} # was 0.2, 5.0
    ur3_scale_factor = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0])*np.array([1.0, 1.0, 1.0, 2.5, 2.5, 2.5])
    gripper_scale_factor = np.array([1.0])
    g_control_type='move_gripper_force'
    gripper_action = True
    
    from gym_custom.envs.custom.dscho_dual_ur3_goal_env_without_obstacle import EndEffectorPositionControlSingleWrapper, EndEffectorPositionControlDualWrapper
    env_id = 'dscho-single-ur3-pickandplace-v1'
    # env_id = 'dscho-single-ur3-push-v1'
    # env_id = 'dscho-single-ur3-reach-v1'
    
    which_hand = 'right'
    flat_gripper = True
    upright_ver = True
    if flat_gripper:
        if upright_ver:
            xml_filename= 'dscho_dual_ur3_upright_object_flat_gripper.xml' 
        else:
            xml_filename= 'dscho_dual_ur3_object_flat_gripper.xml' 
    else:
        if upright_ver:
            raise NotImplementedError
        else:
            xml_filename= 'dscho_dual_ur3_object_gripper.xml' 
    
    env = gym_custom.make(env_id, 
                        initMode = None, 
                        full_state_goal = False, 
                        observation_type='ee_object_all', 
                        trigonometry_observation= False, 
                        ur3_random_init = False,                        
                        warm_start=False, 
                        # ur3_random_init_so3_constraint = 'vertical_side',
                        so3_constraint='vertical_side',
                        sparse_reward = True, 
                        which_hand=which_hand,
                        flat_gripper = flat_gripper, 
                        xml_filename = xml_filename,
                        )
    multi_step=10  # xml dt : 0.005 -> dt 0.05 -> 20Hz
    gripper_force_scale = 1 if flat_gripper else 250
    env = EndEffectorPositionControlSingleWrapper(env=env,
                                                q_control_type=q_control_type,
                                                g_control_type=g_control_type,
                                                multi_step=multi_step,
                                                gripper_action=gripper_action,
                                                PID_gains=PID_gains,
                                                ur3_scale_factor=ur3_scale_factor,
                                                gripper_scale_factor=gripper_scale_factor,
                                                so3_constraint='vertical_side',
                                                flat_gripper=flat_gripper,
                                                action_downscale=0.015, # Assuming tanh action, step당 최대 0.015m-> 20Hz이면 1초당 30cm
                                                gripper_force_scale=gripper_force_scale,#250,
                                                )
    dt = env.dt
    print('dt : ', dt)
    
    
    if make_video:
        import os
        import tensorflow as tf
        assert not render
        cur_vid_dir = os.path.join('./', 'example_video')
        tf.io.gfile.makedirs(cur_vid_dir)
        from dscho_util.video_wrapper import VideoWrapper        
        full_vid_name = 'rollout_'+env_id
        ur3_cam = True
        custom_env = True
        env = VideoWrapper(env, base_path=cur_vid_dir, base_name=full_vid_name, ur3_cam=ur3_cam, custom_env = custom_env)

    obs = env.reset()
    right_gripper_right_state = env.data.get_site_xpos('right_gripper:rightEndEffector')
    right_gripper_left_state = env.data.get_site_xpos('right_gripper:leftEndEffector')
    distance = np.linalg.norm(right_gripper_right_state - right_gripper_left_state, axis =-1)
    print('distance after reset : ', distance)
    
    # while True:
    #     env.render()
    # sys.exit()

    # import gym
    # fetch_env = gym.make('FetchPickAndPlace-v1')
    # fetch_env.reset()

    # start = time.time()
    # for i in range(100):
    #     next_obs, reward, done, info = env.step(env.action_space.sample())
    #     print('obs : {}, rew : {}, done : {}'.format(next_obs, reward, done))
    # print('UR3 step time : ', (time.time() - start)/100)
    # sys.exit()

    # start = time.time()
    # for i in range(100):
    #     fetch_env.step(fetch_env.action_space.sample())
    # print('Fetch step time : ', (time.time() - start)/100)
    # sys.exit()




    # for checking position controller is working
    qpos_right = env._get_ur3_qpos()[:env.ur3_nqpos]
    qpos_left = env._get_ur3_qpos()[env.ur3_nqpos:]
    qvel_right = env._get_ur3_qvel()[:env.ur3_nqvel]
    qvel_left = env._get_ur3_qvel()[env.ur3_nqvel:]
    _, right_ee_pos, _ = env.forward_kinematics_ee(qpos_right, 'right')
    _, left_ee_pos, _ = env.forward_kinematics_ee(qpos_left, 'left')
    current_right_ee_pos = env.get_endeff_pos('right')
    object_pos = env.get_site_pos('objSite')
    
    gripper_qpos_right = env._get_gripper_qpos()[:env.gripper_nqpos]
    gripper_qpos_left = env._get_gripper_qpos()[env.gripper_nqpos:]
    print('reset :  right ee : {} obj : {}'.format(right_ee_pos, object_pos))
    
    for t in range(10):
        action = np.array([1.0, -0.5, -0.0, 0.0]) # x,y,z,방향으로 1*act_scale(0.02) -> step당 (0.1초(dt)) 2cm씩
        next_obs, _, _, _ = env.step(action)
        # env.render()
        qpos_right = env._get_ur3_qpos()[:env.ur3_nqpos]
        qpos_left = env._get_ur3_qpos()[env.ur3_nqpos:]
        qvel_right = env._get_ur3_qvel()[:env.ur3_nqvel]
        qvel_left = env._get_ur3_qvel()[env.ur3_nqvel:]
        _, right_ee_pos, _ = env.forward_kinematics_ee(qpos_right, 'right')
        _, left_ee_pos, _ = env.forward_kinematics_ee(qpos_left, 'left')
        current_right_ee_pos = env.get_endeff_pos('right')
        object_pos = env.get_site_pos('objSite')
        
        gripper_qpos_right = env._get_gripper_qpos()[:env.gripper_nqpos]
        gripper_qpos_left = env._get_gripper_qpos()[env.gripper_nqpos:]
        print('step : {}, right ee : {} obj : {} act : {}'.format(t, right_ee_pos, object_pos, action))
    while True:
        env.render()
    sys.exit()
    



    right_ee_pos_candidate = [np.array([0.15, -0.35, 0.8])]
    from gym_custom.envs.custom.ur_utils import SO3Constraint, NoConstraint
    null_obj_func = SO3Constraint(SO3='vertical_side')
    
    idx = 0
    q_right_des, iter_taken_right, err_right, null_obj_right = env.inverse_kinematics_ee(right_ee_pos_candidate[idx], null_obj_func, arm=which_hand)
    # q_left_des, iter_taken_left, err_left, null_obj_left = env.inverse_kinematics_ee(left_ee_pos_candidate[idx], null_obj_func, arm='left')
    print('q_right_des : {}, iter_taken : {}, err : {}, null_obj : {}'.format(q_right_des, iter_taken_right, err_right, null_obj_right))
    # print('q_left_des : {}, iter_taken : {}, err : {}, null_obj : {}'.format(q_left_des, iter_taken_left, err_left, null_obj_left))
    print('goal : {}'.format(env.get_site_pos('goal')))
    # init_qpos = env.data.qpos.copy()
    # init_qvel = env.init_qvel.copy()
    # q_right_des_candidates =q_right_des
    # # q_left_des_candidates = q_left_des
    # init_qpos[0:env.ur3_nqpos] = q_right_des_candidates
    # # init_qpos[env.ur3_nqpos+env.gripper_nqpos:2*env.ur3_nqpos+env.gripper_nqpos] = q_left_des_candidates
    # env.set_state(init_qpos, init_qvel)
    current_right_ee_pos = env.get_endeff_pos('right')
    object_pos = env.get_site_pos('objSite')
    print('right ee pos : {}, left ee pos : {} object pos : {}'.format(env.get_endeff_pos('right'), env.get_endeff_pos('left'), object_pos))
    
    # env.step(env.action_space.sample())
    # while True:
    #     env.render()

    desired_goal =obs['desired_goal']

    duration = 1 # in seconds
    start = time.time()
    
    # reach
    single = True
    action_scale = 30 # 빠르게 움직이고 싶으면 action downscale or action scale조절(NOTE : action scale은 원래 학습의 영역임)
    grip_scale = 1
    for i in range(2):
        
        for t in range(int(duration/dt)):
            if i==0  :
                if single:
                    action_xyz = object_pos+ np.array([0,0,0.1])-current_right_ee_pos
                    action_xyz = np.tanh(action_scale*action_xyz)
                    action_grip = grip_scale*np.array([-1])
                    action = np.concatenate([action_xyz, action_grip], axis =-1) # open
                    
                else :
                    action = np.array([-0.0, 0.0, 1.0, -5.0, 0.0, 0.0, 1.0, -5.0])
            elif i==1:
                if single:
                    action_xyz = object_pos-np.array([0,0,0.0])-current_right_ee_pos
                    action_xyz = np.tanh(action_scale*action_xyz)
                    action_grip = grip_scale*np.array([0.0]) # obj쪽으로 가면서 gripper 조금씩 close하려했는데 force이다보니 아무리 작은 value여도 0이상이면 닫히는 속도는 same
                    action = np.concatenate([action_xyz, action_grip]) # open
                else :
                    action = np.array([-1.0, 0.0, -0.3, -5.0, 1.0, 0.0, -0.3,-5.0])
            
            
            
            obs, _, _, _ = env.step(action.copy())
            if render: env.render()
            # TODO: get_obs_dict() takes a long time causing timing issues.
            #   Is it due to Upboard's lackluster performance or some deeper
            #   issues within UR Script wrppaer?
            qpos_right = env._get_ur3_qpos()[:env.ur3_nqpos]
            qpos_left = env._get_ur3_qpos()[env.ur3_nqpos:]
            qvel_right = env._get_ur3_qvel()[:env.ur3_nqvel]
            qvel_left = env._get_ur3_qvel()[env.ur3_nqvel:]
            _, right_ee_pos, _ = env.forward_kinematics_ee(qpos_right, 'right')
            _, left_ee_pos, _ = env.forward_kinematics_ee(qpos_left, 'left')
            current_right_ee_pos = env.get_endeff_pos('right')
            object_pos = env.get_site_pos('objSite')
            
            gripper_qpos_right = env._get_gripper_qpos()[:env.gripper_nqpos]
            gripper_qpos_left = env._get_gripper_qpos()[env.gripper_nqpos:]

            # print('time: %f [s]'%(t*dt))
            print('step : {}, right ee : {} obj : {} act : {}'.format(t, right_ee_pos, object_pos, action))
                
            # print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(right_pos_err), np.rad2deg(right_vel_err)))
            # print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(left_pos_err), np.rad2deg(left_vel_err)))
    
    # pick and lift
    for i in range(3):
        
        for t in range(int(duration/dt)):
            if i==0:
                if single:                    
                    action_xyz = np.tanh(np.zeros(3))
                    action_grip = grip_scale*np.array([1])
                    action = np.concatenate([action_xyz, action_grip]) # close
                else :
                    action = np.array([0.0, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 15.0])
            elif i>=1 :
                if single:                    
                    action_xyz = desired_goal-current_right_ee_pos
                    action_xyz = np.tanh(action_scale*action_xyz)
                    action_grip = grip_scale*np.array([1])
                    action = np.concatenate([action_xyz, action_grip]) # open
                    
                else :
                    action = np.array([0.0, 0.0, 1.0, 15.0, 0.0, 0.0, 1.0, 15.0])
            
            

            obs, reward, done, info = env.step(action.copy())

            if render: env.render()
            # TODO: get_obs_dict() takes a long time causing timing issues.
            #   Is it due to Upboard's lackluster performance or some deeper
            #   issues within UR Script wrppaer?
            qpos_right = env._get_ur3_qpos()[:env.ur3_nqpos]
            qpos_left = env._get_ur3_qpos()[env.ur3_nqpos:]
            qvel_right = env._get_ur3_qvel()[:env.ur3_nqvel]
            qvel_left = env._get_ur3_qvel()[env.ur3_nqvel:]
            _, right_ee_pos, _ = env.forward_kinematics_ee(qpos_right, 'right')
            _, left_ee_pos, _ = env.forward_kinematics_ee(qpos_left, 'left')
            current_right_ee_pos = env.get_endeff_pos('right')
            object_pos = env.get_site_pos('objSite')
            # print('time: %f [s]'%(t*dt))
            print('step : {}, right ee : {} obj : {} act : {} reward : {}'.format(t, right_ee_pos, object_pos, action, reward))
            # print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(right_pos_err), np.rad2deg(right_vel_err)))
            # print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(left_pos_err), np.rad2deg(left_vel_err)))
    finish = time.time()

    if make_video:
        env.close()
    else:
        while True:
            env.render()

def dscho_mocap_single_ur3_object_test(env_type='sim', render=False, make_video = False):
    list_of_env_types = ['sim', 'real']

    q_control_type = 'speedj'
    if q_control_type == 'servoj':
        PID_gains = {'servoj': {'P': 1.0, 'I': 0.5, 'D': 0.2}}
    elif q_control_type == 'speedj':
        PID_gains = {'speedj': {'P': 0.2, 'I': 10.0}} # was 0.2, 5.0
    ur3_scale_factor = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0])*np.array([1.0, 1.0, 1.0, 2.5, 2.5, 2.5])
    gripper_scale_factor = np.array([1.0])
    g_control_type='move_gripper_force'
    gripper_action = True
    
    #env_id = 'dscho-single-ur3-mocap-pickandplace-v1'
    # env_id = 'dscho-single-ur3-mocap-pickandplace-multiobject-v1'
    env_id = 'dscho-single-ur3-mocap-door-v1'
    # env_id = 'dscho-single-ur3-mocap-button-v1'
    # env_id = 'dscho-single-ur3-mocap-drawer-v1'
    # env_id = 'dscho-single-ur3-mocap-reach-v1'
    # make_video = False
    which_hand = 'right'
    from gym_custom.envs.custom.dscho_dual_ur3_goal_mocap_env_without_obstacle import DSCHOSingleUR3PickAndPlaceEnv, MocapSingleWrapper
    
    upright_ver = True
    multi_objects = False
    if multi_objects:
        assert upright_ver
        num_objects = 6
        xml_filename= 'dscho_dual_ur3_upright_mocap_'+str(num_objects)+'object_flat_gripper.xml'
        env_kwargs= dict(initMode = None, 
                        sparse_reward = True, 
                        which_hand=which_hand,
                        observation_type = 'ee_object_all',
                        trigonometry_observation = False,
                        so3_constraint='vertical_side', #사실상 의미x. so3 error calculation에만 사용
                        flat_gripper = True, 
                        xml_filename = xml_filename,
                        custom_frame_skip = 10, # 0.005 * 10 =0.05s per step
                        num_objects=num_objects,
                        )
    else:
        if 'door' in env_id:
            env_kwargs = dict(xml_filename= 'dscho_dual_ur3_upright_mocap_door_flat_gripper_ver2.xml' if upright_ver else None,
                             task='door_close',
                            )
        elif 'button' in env_id:
            env_kwargs = dict(xml_filename= 'dscho_dual_ur3_upright_mocap_button_flat_gripper.xml' if upright_ver else None,
                            task='button_press',
                            )
        elif 'drawer' in env_id:
            env_kwargs = dict(xml_filename= 'dscho_dual_ur3_upright_mocap_drawer_flat_gripper_ver2.xml' if upright_ver else None,
                            task='drawer_close',
                            )
        else:
            env_kwargs = dict(xml_filename= 'dscho_dual_ur3_upright_mocap_object_flat_gripper.xml' if upright_ver else 'dscho_dual_ur3_mocap_object_flat_gripper.xml',
                            )
        
        env_kwargs.update(dict(initMode = None, 
                            sparse_reward = True, 
                            which_hand=which_hand,
                            observation_type = 'ee_object_all',
                            trigonometry_observation = False,
                            so3_constraint='vertical_side', #사실상 의미x. so3 error calculation에만 사용
                            flat_gripper = True,                             
                            custom_frame_skip = 10, # 0.005 * 10 =0.05s per step
                            ))
    env = gym_custom.make(env_id , **env_kwargs)
    
    # multi_step=50 # 1step : 0.1s -> 10Hz
    # multi_step=50 # 1step : 0.002s -> 500Hz
    multi_step=1 # 1step : 0.005s -> framsskip 10 곱하면 20Hz (실제로 multi step만큼 밟는건 아니지만 dt를 위해서?(dt곱해진 게 state에 들어가니?))
    env = MocapSingleWrapper(env=env,
                            # q_control_type=q_control_type,
                            # g_control_type=g_control_type,
                            multi_step=multi_step,
                            gripper_action=gripper_action,
                            PID_gains=PID_gains,
                            ur3_scale_factor=ur3_scale_factor,
                            gripper_scale_factor=gripper_scale_factor,
                            # so3_constraint='vertical_side',
                            action_downscale=0.02, # Assuming tanh action,
                            gripper_force_scale=1,
                            )
    # 1 * 0.02 = maximum 0.02m per 1step

    dt = env.dt
    # dt = 0.1
    print('dt : ', dt)
    print('Mocap env는 어처피 한 스텝안에 그만큼 움직이기만 하면 되는거라 굳이 dt가 의미없음. 즉 action scale이 커도 실제 움직일떄 긴 타임스텝동안 움직이면 되니까 문제 x!')
    if multi_objects:
        weight = np.array([0,1,0,0], dtype=np.float32)
        env.set_goal_weight(weight) 
        
        indices =[]
        num_elements = [3, 3*num_objects, 3*num_objects, 2, 3*num_objects, 3*num_objects, 3*num_objects, 3, 2]
        elements_sum = 0
        indices = []
        for idx, element in enumerate(num_elements):
            elements_sum+=element
            indices.append(elements_sum)   
    else:
        pass
    obs = env.reset()
    r_joint_qpos = env.sim.data.get_joint_qpos('right_gripper:r_gripper_finger_joint')
    l_joint_qpos = env.sim.data.get_joint_qpos('right_gripper:l_gripper_finger_joint')
    print('r joint q : {} l joint q : {}'.format(r_joint_qpos, l_joint_qpos))
    print('dummy right body for weld : {}'.format(env.sim.data.get_body_xpos('right_gripper:right_dummy_body_for_weld')))
    if multi_objects:
        grip_pos, object_pos, object_rel_pos, gripper_state, object_rot, object_velp, object_velr, grip_velp, gripper_vel, _ = np.split(obs['observation'], indices, axis=-1)
    else:
        grip_pos, object_pos, object_rel_pos, gripper_state, object_rot, object_velp, object_velr, grip_velp, gripper_vel, _ = np.split(obs['observation'], [3, 6, 9, 11, 14, 17, 20, 23, 25], axis=-1)
    print('g pos : {} o pos : {} o relpos : {} g state : {} o rot : {} o velp : {} o velr : {} g velp : {} g vel : {}'.format(grip_pos, object_pos, object_rel_pos, gripper_state, object_rot, object_velp, object_velr, grip_velp, gripper_vel))
    
    # env.render()
    # time.sleep(2)
    # for i in range(1):
    #     env.step(np.array([0,0,0,0]))
    # while True:
    #     env.render()
    # sys.exit()
    # for i in range(100):
    #     env.render()
    #     env.step(np.array([0.0, 0, 0, 1.0]))
    # sys.exit()
    

    # start = time.time()
    # for i in range(100):
    #     next_obs, reward, done, info = env.step(env.action_space.sample())
    #     print('obs : {}, rew : {}, done : {}'.format(next_obs, reward, done))
    # print('UR3 step time : ', (time.time() - start)/100)
    # sys.exit()

    # To test action is delta pos
    # print('close')    
    # for i in range(10):        
    #     next_obs, reward, done, info = env.step(np.array([1.0, -0.5, 0.0, -1]))
    #     env.render()
    #     grip_pos, object_pos, object_rel_pos, gripper_state, object_rot, object_velp, object_velr, grip_velp, gripper_vel, _ = np.split(next_obs['observation'], [3, 6, 9, 11, 14, 17, 20, 23, 25], axis=-1)
    #     r_joint_qpos = env.sim.data.get_joint_qpos('right_gripper:r_gripper_finger_joint')
    #     l_joint_qpos = env.sim.data.get_joint_qpos('right_gripper:l_gripper_finger_joint')
    #     # print('ee pos : {} '.format(env.get_endeff_pos('right')))
    #     # print('o pos : {} o relpos : {} o rot : {} o velp : {} o velr : {} g velp : {} '.format(object_pos, object_rel_pos, object_rot, object_velp, object_velr, grip_velp))
    #     # print('g pos : {} o pos : {} o relpos : {} g state : {} o rot : {} o velp : {} o velr : {} g velp : {} g vel : {}'.format(grip_pos, object_pos, object_rel_pos, gripper_state, object_rot, object_velp, object_velr, grip_velp, gripper_vel))
    #     print('g pos : {} g state : {} g velp : {} g vel : {}'.format(grip_pos, gripper_state, grip_velp, gripper_vel))
    #     # print('r joint q : {} l joint q : {}'.format(r_joint_qpos, l_joint_qpos))
    #     # print('dummy right body for weld : {}'.format(env.sim.data.get_body_xpos('right_gripper:right_dummy_body_for_weld')))
    #     obs = next_obs
    # while True:
    #     env.render()
    # sys.exit()


    # To test robotiq is well welded to flat gripper
    for i in range(100):
        if i % 10 <5:
            action = np.array([0.0, 0, 0, 1])
        else:
            action = np.array([0.0, 0, 0, -1])
        next_obs, reward, done, info = env.step(action)
        # env.render()
        if multi_objects:
            grip_pos, object_pos, object_rel_pos, gripper_state, object_rot, object_velp, object_velr, grip_velp, gripper_vel, _ = np.split(next_obs['observation'], indices, axis=-1)
        else:
            grip_pos, object_pos, object_rel_pos, gripper_state, object_rot, object_velp, object_velr, grip_velp, gripper_vel, _ = np.split(next_obs['observation'], [3, 6, 9, 11, 14, 17, 20, 23, 25], axis=-1)
        r_joint_qpos = env.sim.data.get_joint_qpos('right_gripper:r_gripper_finger_joint')
        l_joint_qpos = env.sim.data.get_joint_qpos('right_gripper:l_gripper_finger_joint')
        # print('ee pos : {} '.format(env.get_endeff_pos('right')))
        # print('o pos : {} o relpos : {} o rot : {} o velp : {} o velr : {} g velp : {} '.format(object_pos, object_rel_pos, object_rot, object_velp, object_velr, grip_velp))
        # print('g pos : {} o pos : {} o relpos : {} g state : {} o rot : {} o velp : {} o velr : {} g velp : {} g vel : {}'.format(grip_pos, object_pos, object_rel_pos, gripper_state, object_rot, object_velp, object_velr, grip_velp, gripper_vel))
        print('g pos : {} g state : {} g velp : {} g vel : {}'.format(grip_pos, gripper_state, grip_velp, gripper_vel))
        # print('r joint q : {} l joint q : {}'.format(r_joint_qpos, l_joint_qpos))
        # print('dummy right body for weld : {}'.format(env.sim.data.get_body_xpos('right_gripper:right_dummy_body_for_weld')))
        obs = next_obs
    # while True:
    #     env.render()
    # sys.exit()



    # pick and place test
    
    if make_video:
        import os
        import tensorflow as tf
        assert not render
        cur_vid_dir = os.path.join('./', 'example_video')
        tf.io.gfile.makedirs(cur_vid_dir)
        from dscho_util.video_wrapper import VideoWrapper        
        full_vid_name = 'rollout_'+env_id
        ur3_cam = True
        custom_env = True
        env = VideoWrapper(env, base_path=cur_vid_dir, base_name=full_vid_name, ur3_cam=ur3_cam, custom_env = custom_env)


    duration = 1 # in seconds
    single = True
    action_scale = 30 # 빠르게 움직이고 싶으면 action downscale or action scale조절(NOTE : action scale은 원래 학습의 영역임)
    grip_scale = 1
    n_episodes=1
    wait_per_for_loop = False
    
    observations =[]

    for episode in range(n_episodes):
        # obs = env.reset()
        desired_goal =obs['desired_goal']
        current_right_ee_pos = env.get_endeff_pos('right')
        if multi_objects:
            grip_pos, object_pos, object_rel_pos, gripper_state, object_rot, object_velp, object_velr, grip_velp, gripper_vel, _ = np.split(obs['observation'], indices, axis=-1)
            object_pos = object_pos[env.goal_object_idx*3:(env.goal_object_idx+1)*3]
        else:
            grip_pos, object_pos, object_rel_pos, gripper_state, object_rot, object_velp, object_velr, grip_velp, gripper_vel, _ = np.split(obs['observation'], [3, 6, 9, 11, 14, 17, 20, 23, 25], axis=-1)
        
        # print('reset, g pos : {} o pos : {} o relpos : {} g state : {} o rot : {} o velp : {} o velr : {} g velp : {} g vel : {}'.format(grip_pos, object_pos, object_rel_pos, gripper_state, object_rot, object_velp, object_velr, grip_velp, gripper_vel))
        print('reset, o pos : {} o rot : {} '.format(object_pos, object_rot))
        # print('reset, o velp : {} o velr : {} g velp : {} '.format(object_velp, object_velr, grip_velp))
        # for i in range(100): # 일부러 gripper 부시고 시작
        #     if i % 10 <5:
        #         action = np.array([0.0, 0, 0, 1])
        #     else:
        #         action = np.array([0.0, 0, 0, -1])
        #     next_obs, reward, done, info = env.step(action)
                
        obs_list =[]
        obs_list.append(np.concatenate([obs['observation'], obs['desired_goal']], axis =-1))
        for i in range(2):
            
            for t in range(int(duration/dt)):
                if i==0  :
                    if single:
                        if env.task in ['door_open', 'door_close','drawer_open', 'drawer_close', 'button_press']:
                            action_xyz = object_pos+ np.array([0.1,-0.1,0.2])-current_right_ee_pos
                            # action_xyz = np.zeros(4)
                        else:
                            action_xyz = object_pos+ np.array([0,0,0.1])-current_right_ee_pos
                        
                        action_xyz = np.tanh(action_scale*action_xyz)
                        action_grip = grip_scale*np.array([-1])
                        action = np.concatenate([action_xyz, action_grip], axis =-1) # open
                        
                    else :
                        action = np.array([-0.0, 0.0, 1.0, -5.0, 0.0, 0.0, 1.0, -5.0])
                elif i==1:
                    if single:
                        if env.task in ['door_open', 'door_close','drawer_open', 'drawer_close', 'button_press']:
                            action_xyz = object_pos+ np.array([0.0,-0.05,0.1])-current_right_ee_pos
                        else:
                            action_xyz = object_pos-np.array([0,0,0.0])-current_right_ee_pos
                        action_xyz = np.tanh(action_scale*action_xyz)
                        action_grip = grip_scale*np.array([0.0]) # obj쪽으로 가면서 gripper 조금씩 close하려했는데 force이다보니 아무리 작은 value여도 0이상이면 닫히는 속도는 same
                        action = np.concatenate([action_xyz, action_grip]) # open
                    else :
                        action = np.array([-1.0, 0.0, -0.3, -5.0, 1.0, 0.0, -0.3,-5.0])
                # elif i==2:
                #     action = np.array([0.1, 0, -0.05, 0.1])
                
                
                obs, reward, _, _ = env.step(action.copy())
                if render: env.render()
                # TODO: get_obs_dict() takes a long time causing timing issues.
                #   Is it due to Upboard's lackluster performance or some deeper
                #   issues within UR Script wrppaer?
                qpos_right = env._get_ur3_qpos()[:env.ur3_nqpos]
                qpos_left = env._get_ur3_qpos()[env.ur3_nqpos:]
                qvel_right = env._get_ur3_qvel()[:env.ur3_nqvel]
                qvel_left = env._get_ur3_qvel()[env.ur3_nqvel:]
                _, right_ee_pos, _ = env.forward_kinematics_ee(qpos_right, 'right')
                _, left_ee_pos, _ = env.forward_kinematics_ee(qpos_left, 'left')
                current_right_ee_pos = env.get_endeff_pos('right')
                object_pos = env.get_site_pos('objSite')
                
                gripper_qpos_right = env._get_gripper_qpos()[:env.gripper_nqpos]
                gripper_qpos_left = env._get_gripper_qpos()[env.gripper_nqpos:]

                # print('time: %f [s]'%(t*dt))
                # print('step : {}, right ee : {} obj : {} act : {}'.format(t, right_ee_pos, object_pos, action))
                if multi_objects:
                    grip_pos, object_pos, object_rel_pos, gripper_state, object_rot, object_velp, object_velr, grip_velp, gripper_vel, _ = np.split(obs['observation'], indices, axis=-1)
                    object_pos = object_pos[env.goal_object_idx*3:(env.goal_object_idx+1)*3]

                else:
                    grip_pos, object_pos, object_rel_pos, gripper_state, object_rot, object_velp, object_velr, grip_velp, gripper_vel, _ = np.split(obs['observation'], [3, 6, 9, 11, 14, 17, 20, 23, 25], axis=-1)
                # print('step : {}, g pos : {} o pos : {} o relpos : {} g state : {} o rot : {} o velp : {} o velr : {} g velp : {} g vel : {}'.format(t, grip_pos, object_pos, object_rel_pos, gripper_state, object_rot, object_velp, object_velr, grip_velp, gripper_vel))
                print('step : {} o pos : {}  g_state : {}  rew : {}'.format(t, object_pos, gripper_state, reward))
                # print('step : {} o velp : {} o velr : {} g velp : {} '.format(t, object_velp, object_velr, grip_velp))
                # print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(right_pos_err), np.rad2deg(right_vel_err)))
                # print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(left_pos_err), np.rad2deg(left_vel_err)))
                obs_list.append(np.concatenate([obs['observation'], obs['desired_goal']], axis =-1))
            
            if wait_per_for_loop:
                wait = input("Press Enter to continue.")
        # pick and lift & open gripper
        for i in range(4):
            
            for t in range(int(duration/dt)):
                if i==0:
                    if single:                    
                        action_xyz = np.tanh(np.zeros(3))
                        action_grip = grip_scale*np.array([1])
                        action = np.concatenate([action_xyz, action_grip]) # close
                    else :
                        action = np.array([0.0, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0, 15.0])
                elif i==1 or i==2 :
                    if single:                    
                        action_xyz = desired_goal-current_right_ee_pos
                        action_xyz = np.tanh(action_scale*action_xyz)
                        action_grip = grip_scale*np.array([1])
                        action = np.concatenate([action_xyz, action_grip]) # open
                        
                    else :
                        action = np.array([0.0, 0.0, 1.0, 15.0, 0.0, 0.0, 1.0, 15.0])
                elif i==3: #open gripper
                    action = np.array([0.0, 0.0, 0.0, -1.0])
                # elif i==4:
                #     action = np.array([-0.1, 0.0, 0.0, -1.0])

                obs, reward, done, info = env.step(action.copy())

                if render: env.render()
                # TODO: get_obs_dict() takes a long time causing timing issues.
                #   Is it due to Upboard's lackluster performance or some deeper
                #   issues within UR Script wrppaer?
                qpos_right = env._get_ur3_qpos()[:env.ur3_nqpos]
                qpos_left = env._get_ur3_qpos()[env.ur3_nqpos:]
                qvel_right = env._get_ur3_qvel()[:env.ur3_nqvel]
                qvel_left = env._get_ur3_qvel()[env.ur3_nqvel:]
                _, right_ee_pos, _ = env.forward_kinematics_ee(qpos_right, 'right')
                _, left_ee_pos, _ = env.forward_kinematics_ee(qpos_left, 'left')
                current_right_ee_pos = env.get_endeff_pos('right')
                object_pos = env.get_site_pos('objSite')
                # print('time: %f [s]'%(t*dt))
                # print('step : {}, right ee : {} obj : {} act : {} reward : {}'.format(t, right_ee_pos, object_pos, action, reward))
                if multi_objects:
                    grip_pos, object_pos, object_rel_pos, gripper_state, object_rot, object_velp, object_velr, grip_velp, gripper_vel, _ = np.split(obs['observation'], indices, axis=-1)
                    object_pos = object_pos[env.goal_object_idx*3:(env.goal_object_idx+1)*3]

                else:
                    grip_pos, object_pos, object_rel_pos, gripper_state, object_rot, object_velp, object_velr, grip_velp, gripper_vel, _ = np.split(obs['observation'], [3, 6, 9, 11, 14, 17, 20, 23, 25], axis=-1)
                # print('step : {}, g pos : {} o pos : {} o relpos : {} g state : {} o rot : {} o velp : {} o velr : {} g velp : {} g vel : {}'.format(t, grip_pos, object_pos, object_rel_pos, gripper_state, object_rot, object_velp, object_velr, grip_velp, gripper_vel))
                print('step : {} o pos : {}  g_state : {} reward : {}'.format(t, object_pos, gripper_state, reward))
                # print('step : {} o velp : {} o velr : {} g velp : {} '.format(t, object_velp, object_velr, grip_velp))                
                # print('right arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(right_pos_err), np.rad2deg(right_vel_err)))
                # print('left arm joint pos error [deg]: %f vel error [dps]: %f'%(np.rad2deg(left_pos_err), np.rad2deg(left_vel_err)))
                obs_list.append(np.concatenate([obs['observation'], obs['desired_goal']], axis =-1))
            
            if wait_per_for_loop:
                wait = input("Press Enter to continue.")

        observations.append(np.stack(obs_list, axis =0)) #[ts, dim])
    
    observations = np.stack(observations,axis=0) #[bs, ts, dim]
    
    plot = True
    if plot:
        import matplotlib.pyplot as plt
        import os
        import tensorflow as tf
        cur_vid_dir = os.path.join('./', 'example_video')
        tf.io.gfile.makedirs(cur_vid_dir)
        for i in range(n_episodes):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for j in range(observations.shape[-1]):
                ax.plot(observations[i, :, j], label='obs_'+str(j))
            plt.legend(loc='best')
            plt.savefig(cur_vid_dir+'/obs_traj_'+str(i))
            plt.close()


    if make_video:
        env.close()
    else:
        while True:
            env.render()

if __name__ == '__main__':
    # 1. MuJoCo model verification
    # show_dual_ur3()
    # run_dual_ur3()
    # test_fkine_ikine()

    # 2.1 Updated UR wrapper examples
    # servoj_and_forceg(env_type='real', render=False)
    # speedj_and_forceg(env_type='sim', render=True)
    # pick_and_place(env_type='real', render=False)
    # collide(env_type='sim', render=True)

    # 2.2 Deprecated UR wrapper examples 
    # servoj_and_forceg_deprecated()
    # speedj_and_forceg_deprecated()
    # pick_and_place_deprecated()
    # collide_deprecated()

    #dscho mod
    #dscho_pick_and_place(env_type='sim', render=True)
    # dscho_single_arm_speedj_and_forceg(env_type='sim', render=True)
    # dscho_single_arm_IK_debug(env_type='sim', render=True)
    # dscho_single_arm_goal_sample_debug(env_type='sim', render=True)
    # dscho_dual_arm_test(render=True)
    # dscho_single_arm_test(render=True)
    # dscho_posxyz_v1_test(render=True)
    # dscho_posxyz_v2_test(render=True)
    # dscho_posxyz_v4_test(render=True)
    # dscho_posxyz_v5_test(render=True)
    # dscho_posxyz_single_v4_v5_test(render=True)
    # dscho_init_qpos_candidate_pickling(render=True)
    # dscho_single_ur3_object_test(render=False, make_video = False)
    dscho_mocap_single_ur3_object_test(render=False, make_video = True)


    # 3. Misc. tests
    # real_env_get_obs_rate_test(wait=False)
    # real_env_command_send_rate_test(wait=False)
    # pass
