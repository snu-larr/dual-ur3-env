import argparse
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
    env.step({'right': {'open_gripper': {}}})
    # 2. Open right gripper
    duration = 5.0
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
    while qpos_err > np.deg2rad(1e-1) or qvel > np.deg2rad(1e0):
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
    env.step({'right': {'close_gripper': {}}})
    # 4. Grip object
    duration = 5.0
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
    while qpos_err > np.deg2rad(1e-1) or qvel > np.deg2rad(1e0):
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
    env.step({'right': {'open_gripper': {}}})
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
    dscho_posxyz_v1_test(render=True)
    # dscho_posxyz_v2_test(render=True)
    # dscho_posxyz_v4_test(render=True)
    # dscho_posxyz_v5_test(render=True)
    # dscho_posxyz_single_v4_v5_test(render=True)
    # dscho_init_qpos_candidate_pickling(render=True)